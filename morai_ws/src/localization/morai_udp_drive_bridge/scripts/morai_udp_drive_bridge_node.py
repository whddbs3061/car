#!/usr/bin/env python3
"""Competition Vehicle Status input and the single EgoCtrlCmd UDP output."""

from __future__ import annotations

import math
import json
import socket
import threading
import time
from typing import Optional

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import String
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus

from morai_udp_drive_bridge.protocol import (
    ProtocolError,
    build_ego_ctrl_cmd,
    parse_competition_vehicle_status,
)


class MoraiUdpDriveBridge:
    def __init__(self) -> None:
        rospy.init_node("morai_udp_drive_bridge", anonymous=False)

        self.status_bind_ip = rospy.get_param("~status_bind_ip", "0.0.0.0")
        self.status_port = int(rospy.get_param("~status_port", 9099))
        self.status_source_port = int(rospy.get_param("~status_source_port", 9088))
        if not 1000 <= self.status_port <= 65535:
            raise ValueError("Competition status must use a four-or-more-digit UDP port (1000..65535)")
        if not 0 <= self.status_source_port <= 65535:
            raise ValueError("Competition status source port must be 0..65535")
        self.status_protocol = str(rospy.get_param("~status_protocol", "competition"))
        if self.status_protocol != "competition":
            raise ValueError("new_final_ws accepts Competition Vehicle Status only")
        self.status_topic = rospy.get_param("~status_topic", "/Ego_topic")
        self.status_frame_id = rospy.get_param("~status_frame_id", "map")
        # Read once at startup: dry-run output cannot be enabled at runtime.
        self.control_output_enabled = bool(rospy.get_param("~control_output_enabled", True))
        self.control_remote_ip = rospy.get_param("~control_remote_ip", "192.168.0.147")
        self.control_remote_port = int(rospy.get_param("~control_remote_port", 9093))
        self.control_bind_ip = rospy.get_param("~control_bind_ip", "0.0.0.0")
        self.control_bind_port = int(rospy.get_param("~control_bind_port", 0))
        self.command_topic = rospy.get_param("~command_topic", "/ctrl_cmd")
        self.send_rate_hz = float(rospy.get_param("~send_rate_hz", 20.0))
        self.command_timeout_sec = float(rospy.get_param("~command_timeout_sec", 0.5))
        self.max_wheel_angle_rad = float(
            rospy.get_param("~max_wheel_angle_rad", math.radians(40.0))
        )
        self.status_use_packet_time = bool(rospy.get_param("~status_use_packet_time", False))
        self.longl_cmd_type = int(rospy.get_param("~longl_cmd_type", 1))

        self.last_command: Optional[CtrlCmd] = None
        self.last_command_time = 0.0
        self.stop_event = threading.Event()
        self.last_packet_time = None
        self.send_socket = None
        self.send_timer = None

        self.status_pub = rospy.Publisher(self.status_topic, EgoVehicleStatus, queue_size=20)
        self.metadata_pub = rospy.Publisher("~status_metadata", String, queue_size=1)
        rospy.Subscriber(self.command_topic, CtrlCmd, self.command_callback, queue_size=20)

        if self.control_output_enabled:
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.control_bind_port > 0:
                self.send_socket.bind((self.control_bind_ip, self.control_bind_port))
            self.send_timer = rospy.Timer(
                rospy.Duration(1.0 / max(self.send_rate_hz, 1.0)), self.send_timer_callback
            )

        self.status_thread = threading.Thread(
            target=self.status_receive_loop,
            name="morai-competition-status-receiver",
            daemon=True,
        )
        self.status_thread.start()
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo(
            "MORAI UDP drive bridge: status bind=%s:%d topic=%s, control remote=%s:%d, output enabled=%s",
            self.status_bind_ip,
            self.status_port,
            self.status_topic,
            self.control_remote_ip,
            self.control_remote_port,
            self.control_output_enabled,
        )

    def status_receive_loop(self) -> None:
        receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        try:
            receive_socket.bind((self.status_bind_ip, self.status_port))
        except OSError as exc:
            rospy.logfatal(
                "Vehicle status UDP bind 실패 %s:%d: %s",
                self.status_bind_ip,
                self.status_port,
                exc,
            )
            receive_socket.close()
            return
        receive_socket.settimeout(0.5)
        rospy.loginfo(
            "Vehicle status UDP bind %s:%d protocol=%s (Competition: 181/229 bytes)",
            self.status_bind_ip,
            self.status_port,
            self.status_protocol,
        )

        try:
            while not rospy.is_shutdown() and not self.stop_event.is_set():
                try:
                    # Read the entire datagram. recvfrom(181) silently truncates
                    # a 229-byte packet before its real tail can be validated.
                    packet, sender = receive_socket.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self.stop_event.is_set():
                        rospy.logerr_throttle(5.0, "Vehicle status UDP 수신 오류: %s", exc)
                    break

                if self.status_source_port and sender[1] != self.status_source_port:
                    rospy.logwarn_throttle(
                        5.0,
                        "Vehicle status source port mismatch sender=%s:%d expected=%d",
                        sender[0], sender[1], self.status_source_port,
                    )
                    continue

                try:
                    measurement = parse_competition_vehicle_status(packet)
                except (ProtocolError, ValueError) as exc:
                    rospy.logwarn_throttle(
                        5.0,
                        "Vehicle status 패킷 폐기 sender=%s:%d len=%d error=%s",
                        sender[0],
                        sender[1],
                        len(packet),
                        exc,
                    )
                    continue

                source_time = (measurement.sec, measurement.nsec)
                if self.status_protocol == 'competition':
                    if self.last_packet_time is not None and source_time <= self.last_packet_time:
                        rospy.logwarn_throttle(5.0, 'Competition status duplicate/out-of-order source time; '
                                              'restart the launch after a scene reset')
                        continue
                    self.last_packet_time = source_time
                self.publish_status(measurement)
        finally:
            receive_socket.close()

    def publish_status(self, measurement) -> None:
        received = rospy.Time.now()
        message = EgoVehicleStatus()
        if self.status_use_packet_time and measurement.sec > 0:
            message.header.stamp = rospy.Time(measurement.sec, measurement.nsec)
        else:
            message.header.stamp = received
        message.header.frame_id = self.status_frame_id
        message.unique_id = 0
        # Keep the existing ROS type for consumers, but never expose reserved
        # competition slots as measured zeros/ground truth. Consumers explicitly
        # read longitudinal velocity.x; pose remains GPS/IMU EKF only.
        unavailable = float('nan')
        message.acceleration = Vector3(unavailable, unavailable, unavailable)
        message.position = Vector3(unavailable, unavailable, unavailable)
        message.velocity = Vector3(
            measurement.velocity_x_kmh / 3.6,
            unavailable,
            unavailable,
        )
        message.heading = measurement.heading_deg
        message.accel = measurement.accel_pedal
        message.brake = measurement.brake_pedal
        message.wheel_angle = measurement.steer_deg
        self.status_pub.publish(message)
        self.metadata_pub.publish(String(data=json.dumps({
            'timestamp': received.to_sec(), 'packet_sec': measurement.sec,
            'packet_nsec': measurement.nsec, 'packet_length': measurement.packet_length,
            'source': self.status_protocol, 'listen_port': self.status_port,
            'expected_source_port': self.status_source_port,
            'velocity_unit': 'm/s', 'velocity_x_mps': message.velocity.x,
            'signed_velocity_mps': measurement.signed_velocity_kmh/3.6,
            'heading_deg': measurement.heading_deg, 'wheel_angle_deg': measurement.steer_deg,
            'ctrl_mode': measurement.ctrl_mode, 'gear': measurement.gear,
            'position_available': False, 'acceleration_available': False,
            'velocity_yz_available': False,
        }, allow_nan=False, separators=(',', ':'))))
        rospy.loginfo_throttle(
            2.0,
            "Vehicle UDP source=%s port=%d bytes=%d vx=%.2fm/s heading=%.2fdeg ctrl=%d gear=%d",
            self.status_protocol, self.status_port, measurement.packet_length,
            measurement.velocity_x_kmh / 3.6,
            measurement.heading_deg,
            measurement.ctrl_mode, measurement.gear,
        )

    def command_callback(self, message: CtrlCmd) -> None:
        self.last_command = message
        self.last_command_time = time.monotonic()

    def send_timer_callback(self, _event) -> None:
        if not self.control_output_enabled or self.send_socket is None or self.stop_event.is_set():
            return

        message = self.last_command
        is_fresh = message is not None and time.monotonic() - self.last_command_time <= self.command_timeout_sec

        if not is_fresh:
            packet = build_ego_ctrl_cmd(
                cmd_type=self.longl_cmd_type, velocity_kmh=0.0, brake=1.0
            )
        else:
            cmd_type = int(getattr(message, "longlCmdType", self.longl_cmd_type))
            steering_rad = float(getattr(message, "steering", 0.0))
            steer_normalized = steering_rad / max(self.max_wheel_angle_rad, 1e-6)
            velocity_kmh = (
                max(0.0, float(getattr(message, "velocity", 0.0)) * 3.6)
                if cmd_type == 2
                else 0.0
            )
            packet = build_ego_ctrl_cmd(
                cmd_type=cmd_type,
                velocity_kmh=velocity_kmh,
                acceleration_mps2=float(getattr(message, "acceleration", 0.0)),
                accel=float(getattr(message, "accel", 0.0)),
                brake=float(getattr(message, "brake", 0.0)),
                steer_normalized=steer_normalized,
                ctrl_mode=2,
                gear=4,
            )

        try:
            self.send_socket.sendto(packet, (self.control_remote_ip, self.control_remote_port))
        except OSError as exc:
            rospy.logerr_throttle(5.0, "EgoCtrlCmd UDP 송신 오류: %s", exc)

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.send_timer is not None:
            self.send_timer.shutdown()
        if not self.control_output_enabled or self.send_socket is None:
            return

        try:
            stop_packet = build_ego_ctrl_cmd(
                cmd_type=self.longl_cmd_type, velocity_kmh=0.0, brake=1.0
            )
            self.send_socket.sendto(stop_packet, (self.control_remote_ip, self.control_remote_port))
        except OSError:
            pass
        finally:
            self.send_socket.close()


if __name__ == "__main__":
    try:
        MoraiUdpDriveBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
