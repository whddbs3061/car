"""Exercise the real bridge with mocked ROS, sockets, threads and clock.

Run directly with: python -B test/test_transport_safety.py
No ROS installation, network traffic or background threads are required.
"""

import importlib.util
from pathlib import Path
import socket
import struct
import math
import json
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import xml.etree.ElementTree as ET


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE.parents[1]


class Message:
    def __init__(self, **values):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.__dict__.update(values)


def vector3(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


class TransportSafetyTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.parameters = {
            "~control_bind_ip": "127.0.0.1",
            "~control_bind_port": 9094,
            "~control_remote_ip": "192.0.2.1",
        }
        self.ros = Mock()
        self.ros.get_param.side_effect = lambda key, default: self.parameters.get(key, default)
        self.ros.Publisher.side_effect = lambda *a, **k: Mock()
        self.ros.is_shutdown.return_value = False
        self.ros.Time.now.return_value = SimpleNamespace(to_sec=lambda: self.now)
        dependencies = {
            "rospy": self.ros,
            "geometry_msgs.msg": SimpleNamespace(Vector3=vector3),
            "std_msgs.msg": SimpleNamespace(String=Message),
            "morai_msgs.msg": SimpleNamespace(CtrlCmd=Message, EgoVehicleStatus=Message),
        }
        spec = importlib.util.spec_from_file_location(
            "_transport_safety_bridge", PACKAGE / "scripts/morai_udp_drive_bridge_node.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies), patch.object(
            sys, "path", [str(PACKAGE / "src")] + sys.path
        ):
            spec.loader.exec_module(self.module)

        self.sockets = []
        self.socket_factory = Mock(side_effect=self.make_socket)
        self.module.socket = SimpleNamespace(
            socket=self.socket_factory,
            AF_INET=socket.AF_INET,
            SOCK_DGRAM=socket.SOCK_DGRAM,
            SOL_SOCKET=socket.SOL_SOCKET,
            SO_REUSEADDR=socket.SO_REUSEADDR,
            SO_RCVBUF=socket.SO_RCVBUF,
            timeout=socket.timeout,
        )
        self.thread_factory = Mock()
        self.module.threading = SimpleNamespace(Thread=self.thread_factory, Event=threading.Event)
        self.module.time = SimpleNamespace(monotonic=lambda: self.now)

    def make_socket(self, *_args):
        transport = Mock(spec=["bind", "setsockopt", "settimeout", "recvfrom", "sendto", "close"])
        self.sockets.append(transport)
        return transport

    def bridge(self, enabled=None):
        if enabled is not None:
            self.parameters["~control_output_enabled"] = enabled
        return self.module.MoraiUdpDriveBridge()

    def assert_no_sends(self):
        for transport in self.sockets:
            transport.sendto.assert_not_called()

    def assert_startup_setting_read_once(self):
        reads = [entry for entry in self.ros.get_param.call_args_list
                 if entry.args[0] == "~control_output_enabled"]
        self.assertEqual(reads, [call("~control_output_enabled", True)])

    def assert_full_brake(self, sent, cmd_type=1):
        packet, destination = sent.args
        self.assertEqual(destination, ("192.0.2.1", 9093))
        self.assertEqual(len(packet), 55)
        fields = struct.unpack("<14s i 3i 3b 5f 2s", packet)
        self.assertEqual(fields[:8], (b"#MoraiCtrlCmd$", 23, 0, 0, 0, 2, 4, cmd_type))
        self.assertEqual(fields[8:13], (0.0, 0.0, 0.0, 1.0, 0.0))
        self.assertEqual(fields[13], b"\r\n")

    def test_disabled_startup_has_no_send_socket_bind_or_timer(self):
        node = self.bridge(False)
        self.assertFalse(node.control_output_enabled)
        self.assertIsNone(node.send_socket)
        self.assertIsNone(node.send_timer)
        self.socket_factory.assert_not_called()
        self.ros.Timer.assert_not_called()
        self.ros.Publisher.assert_any_call("/Ego_topic", Message, queue_size=20)
        self.thread_factory.assert_called_once_with(
            target=node.status_receive_loop, name="morai-competition-status-receiver", daemon=True
        )
        self.thread_factory.return_value.start.assert_called_once_with()
        self.ros.on_shutdown.assert_called_once_with(node.shutdown)
        self.assert_no_sends()

    def test_disabled_ticks_with_missing_fresh_and_stale_commands_never_send(self):
        node = self.bridge(False)
        node.send_timer_callback(None)
        for cmd_type in (1, 2):
            node.command_callback(Message(longlCmdType=cmd_type, accel=0.8, velocity=5.0))
            node.send_timer_callback(None)
            self.now += node.command_timeout_sec + 0.1
            node.send_timer_callback(None)
        self.socket_factory.assert_not_called()
        self.ros.Timer.assert_not_called()
        self.assert_no_sends()

    def test_disabled_shutdown_never_sends_even_after_receiving_a_command(self):
        node = self.bridge(False)
        node.command_callback(Message(longlCmdType=1, accel=0.8))
        self.ros.on_shutdown.call_args.args[0]()
        node.send_timer_callback(None)
        self.assertTrue(node.stop_event.is_set())
        self.socket_factory.assert_not_called()
        self.ros.Timer.assert_not_called()
        self.assert_no_sends()

    def test_disabled_output_cannot_be_enabled_by_runtime_parameter_change(self):
        node = self.bridge(False)
        self.parameters["~control_output_enabled"] = True
        node.command_callback(Message(longlCmdType=1, accel=0.8))
        node.send_timer_callback(None)
        node.shutdown()
        self.assertFalse(node.control_output_enabled)
        self.assert_startup_setting_read_once()
        self.socket_factory.assert_not_called()
        self.ros.Timer.assert_not_called()
        self.assert_no_sends()

    def test_disabled_status_receiver_survives_timeout_and_publishes_without_sending(self):
        node = self.bridge(False)
        receive_socket = self.make_socket()
        self.socket_factory.side_effect = None
        self.socket_factory.return_value = receive_socket
        packet = struct.pack(
            "<11s i 3i i i b b f i 24f 38s 2s",
            b"#MoraiInfo$", 152, 0, 0, 0, 10, 20, 2, 4, 7.2, 10000,
            *[float(index) for index in range(24)], b"LINK_1", b"\r\n",
        )
        second_packet=bytearray(packet)
        struct.pack_into("<I",second_packet,31,21)
        receive_socket.recvfrom.side_effect = [
            socket.timeout(), (packet, ("192.0.2.1", 9088)),
            (bytes(second_packet), ("192.0.2.1", 9088))
        ]
        published = []

        def publish(message):
            published.append(message)
            if len(published) == 2:
                node.stop_event.set()

        node.status_pub.publish.side_effect = publish
        self.thread_factory.call_args.kwargs["target"]()
        self.socket_factory.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        receive_socket.bind.assert_called_once_with(("0.0.0.0", 9099))
        receive_socket.settimeout.assert_called_once_with(0.5)
        self.assertEqual(receive_socket.recvfrom.call_count, 3)
        self.assertEqual(len(published), 2)
        self.assertEqual(published[0].header.frame_id, "map")
        self.assertTrue(math.isnan(published[0].position.x))
        self.assertAlmostEqual(published[0].velocity.x, 14.0 / 3.6)
        receive_socket.close.assert_called_once_with()
        node.send_timer_callback(None)
        node.shutdown()
        self.ros.Timer.assert_not_called()
        self.assert_no_sends()

    def test_standalone_default_keeps_control_socket_bind_and_20_hz_timer(self):
        node = self.bridge()
        self.assertTrue(node.control_output_enabled)
        self.socket_factory.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        node.send_socket.bind.assert_called_once_with(("127.0.0.1", 9094))
        self.ros.Duration.assert_called_once_with(0.05)
        self.ros.Timer.assert_called_once_with(self.ros.Duration.return_value, node.send_timer_callback)
        self.assert_startup_setting_read_once()

    def test_enabled_missing_command_sends_full_brake(self):
        node = self.bridge(True)
        self.ros.Timer.call_args.args[1](None)
        node.send_socket.sendto.assert_called_once()
        self.assert_full_brake(node.send_socket.sendto.call_args)

    def test_enabled_fresh_command_then_stale_ticks_keep_full_brake(self):
        node = self.bridge(True)
        node.command_callback(Message(longlCmdType=1, accel=0.8, steering=0.2))
        tick = self.ros.Timer.call_args.args[1]
        tick(None)
        fields = struct.unpack("<14s i 3i 3b 5f 2s", node.send_socket.sendto.call_args.args[0])
        self.assertAlmostEqual(fields[10], 0.8)
        self.assertEqual(fields[11], 0.0)
        self.assertAlmostEqual(fields[12], 0.2 / node.max_wheel_angle_rad)
        self.now += node.command_timeout_sec + 0.1
        tick(None)
        self.now += 1.0
        tick(None)
        self.assertEqual(node.send_socket.sendto.call_count, 3)
        for sent in node.send_socket.sendto.call_args_list[1:]:
            self.assert_full_brake(sent)

    def test_enabled_stale_and_shutdown_brake_keep_configured_command_type(self):
        self.parameters["~longl_cmd_type"] = 2
        node = self.bridge(True)
        node.command_callback(Message(longlCmdType=2, velocity=5.0))
        self.now += node.command_timeout_sec + 0.1
        node.send_timer_callback(None)
        node.shutdown()
        self.assertEqual(node.send_socket.sendto.call_count, 2)
        for sent in node.send_socket.sendto.call_args_list:
            self.assert_full_brake(sent, cmd_type=2)

    def test_enabled_shutdown_sends_full_brake_and_stops_later_ticks(self):
        node = self.bridge(True)
        node.command_callback(Message(longlCmdType=1, accel=0.8))
        self.parameters["~control_output_enabled"] = False
        self.ros.on_shutdown.call_args.args[0]()
        self.assertTrue(node.stop_event.is_set())
        node.send_timer.shutdown.assert_called_once_with()
        node.send_socket.sendto.assert_called_once()
        self.assert_full_brake(node.send_socket.sendto.call_args)
        node.send_socket.close.assert_called_once_with()
        node.send_timer_callback(None)
        self.assertEqual(node.send_socket.sendto.call_count, 1)
        self.assert_startup_setting_read_once()

    def test_enabled_shutdown_closes_socket_if_brake_send_fails(self):
        node = self.bridge(True)
        node.send_socket.sendto.side_effect = OSError("mock send failure")
        node.shutdown()
        node.send_socket.sendto.assert_called_once()
        self.assert_full_brake(node.send_socket.sendto.call_args)
        node.send_socket.close.assert_called_once_with()


class TransportLaunchTest(unittest.TestCase):
    def test_control_flag_reaches_typed_bridge_parameter_and_preserves_defaults(self):
        bridge = ET.parse(PACKAGE / "launch/morai_udp_drive_bridge.launch").getroot()
        self.assertEqual(bridge.find("arg[@name='control_output_enabled']").get("default"), "true")
        parameter = bridge.find("node/param[@name='control_output_enabled']")
        self.assertEqual(parameter.get("type"), "bool")
        self.assertEqual(parameter.get("value"), "$(arg control_output_enabled)")
        bringup = ET.parse(
            SOURCE / "bringup/morai_bringup/launch/morai_udp_ekf_purepursuit.launch"
        ).getroot()
        self.assertEqual(bringup.find("arg[@name='enable_control']").get("default"), "false")
        include = bringup.find(
            "include[@file='$(find morai_udp_drive_bridge)/launch/morai_udp_drive_bridge.launch']"
        )
        self.assertIsNotNone(include)
        self.assertIsNone(include.get("if"))
        self.assertIsNone(include.get("unless"))
        self.assertEqual(include.find("arg[@name='control_output_enabled']").get("value"),
                         "$(arg enable_control)")


if __name__ == "__main__":
    unittest.main()
