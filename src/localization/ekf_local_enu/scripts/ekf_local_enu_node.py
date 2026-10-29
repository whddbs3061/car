#!/usr/bin/env python3
"""MGeo local ENU GPS·IMU 2D EKF 노드.

상태:
  [x_map, y_map, yaw_map, forward_velocity, gyro_bias_z, accel_bias_x]

GPS는 /localization/gps(nav_msgs/Odometry), IMU는 /Imu(sensor_msgs/Imu)를
입력으로 사용한다. map->base_link는 이 노드가 발행한다.
"""

from __future__ import annotations

import math
import json
import time
import threading
from functools import wraps
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from morai_perception_msgs.msg import SensorQuality
from morai_msgs.msg import EgoVehicleStatus
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ekf_local_enu.math_utils import quaternion_to_yaw, wrap_angle, yaw_to_quaternion
from ekf_local_enu.distance_scale import DistanceScale

def stamp_seconds(stamp: rospy.Time) -> float:
    value = stamp.to_sec()
    return value if value > 0.0 else rospy.Time.now().to_sec()


def imu_planar_inputs(msg, includes_gravity=True):
    q = msg.orientation
    values = (q.x, q.y, q.z, q.w, msg.linear_acceleration.x,
              msg.linear_acceleration.y, msg.linear_acceleration.z,
              msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
    if not all(math.isfinite(v) for v in values):
        return None
    norm = math.sqrt(sum(v*v for v in values[:4]))
    if norm < 1e-9:
        return None
    x, y, z, w = [v/norm for v in values[:4]]
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    acceleration = np.array(values[4:7],dtype=float)
    if includes_gravity:
        acceleration -= rotation.T @ np.array([0.,0.,9.80665])
    world = rotation @ acceleration
    yaw = math.atan2(rotation[1,0],rotation[0,0])
    forward = world[0]*math.cos(yaw)+world[1]*math.sin(yaw)
    roll = math.atan2(rotation[2,1],rotation[2,2])
    pitch = math.asin(max(-1.,min(1.,-rotation[2,0])))
    if abs(math.cos(pitch)) < 0.1:
        return None
    yaw_rate = (msg.angular_velocity.y*math.sin(roll)+msg.angular_velocity.z*math.cos(roll))/math.cos(pitch)
    return float(forward), float(yaw_rate)


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class LocalEnuEkf:
    def __init__(self) -> None:
        rospy.init_node("ekf_local_enu", anonymous=False)

        self.lock = threading.RLock()
        self.distance_scale = DistanceScale(rospy.get_param("~distance_scale_enabled", True))
        self.trust_imu = bool(rospy.get_param("~trust_imu", True))
        self.use_vehicle_speed_aiding = bool(rospy.get_param("~use_vehicle_speed_aiding", True))
        self.ego_diagnostic_sample = None
        self.imu_acceleration_includes_gravity = bool(rospy.get_param("~imu_acceleration_includes_gravity", True))
        self.latest_gyro_z = 0.0
        self.last_planar_accel = 0.0
        self.prediction_history = []
        self.vehicle_speed_topic = rospy.get_param("~vehicle_speed_topic", "/Ego_topic")
        self.vehicle_speed_timeout = float(rospy.get_param("~vehicle_speed_timeout_sec", 0.5))
        self.vehicle_speed_std = float(rospy.get_param("~vehicle_speed_std_mps", 0.2))
        self.max_vehicle_speed = float(rospy.get_param("~max_vehicle_speed_mps", 60.0))
        self.gps_timeout = float(rospy.get_param("~gps_timeout_sec", 1.0))
        self.max_gps_local_coordinate = float(rospy.get_param("~max_gps_local_coordinate_m", 100000.0))
        self.gps_recovery_max_residual = float(rospy.get_param("~gps_recovery_max_residual_m", 50.0))
        self.gps_correction_step = float(rospy.get_param("~gps_correction_step_m", 1.0))
        self.accel_bias_limit = float(rospy.get_param("~accel_bias_limit_mps2", 3.0))
        if not all(math.isfinite(v) and v > 0 for v in (
                self.vehicle_speed_timeout, self.vehicle_speed_std, self.max_vehicle_speed,
                self.gps_timeout, self.max_gps_local_coordinate,
                self.gps_recovery_max_residual, self.gps_correction_step, self.accel_bias_limit)):
            raise ValueError("Invalid tunnel speed parameters")
        self.vehicle_sample = None
        self.last_fused_speed_stamp = None
        self.gps_recovering = False
        self.gps_recovery_sample = None
        self.gps_recovery_count = 0
        self.last_speed_diag_wall = float('-inf')
        self.gps_topic = rospy.get_param("~gps_topic", "/localization/gps")
        self.imu_topic = rospy.get_param("~imu_topic", "/Imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/localization/odometry")
        self.pose_topic = rospy.get_param("~pose_topic", "/localization/pose")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.use_imu_orientation = bool(rospy.get_param("~use_imu_orientation", True))
        self.yaw_offset_rad = math.radians(float(rospy.get_param("~yaw_offset_deg", 0.0)))
        self.default_gps_variance = float(rospy.get_param("~default_gps_variance_m2", 1.0))
        self.default_yaw_variance = math.radians(
            float(rospy.get_param("~default_yaw_std_deg", 8.0))
        ) ** 2
        self.gyro_noise_std = float(rospy.get_param("~gyro_noise_std_rad_s", 0.08))
        self.accel_noise_std = float(rospy.get_param("~accel_noise_std_m_s2", 0.8))
        self.bias_random_walk = float(rospy.get_param("~bias_random_walk", 0.001))
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.use_sensor_quality_gate = bool(
            rospy.get_param("~use_sensor_quality_gate", False)
        )
        self.quality_topic = rospy.get_param(
            "~quality_topic", "/localization/sensor_quality"
        )
        self.quality_timeout_sec = max(
            0.05, float(rospy.get_param("~quality_timeout_sec", 0.5))
        )
        self.enable_gps_jump_gate = bool(
            rospy.get_param("~enable_gps_jump_gate", False)
        )
        self.gps_jump_gate_m = max(
            0.1, float(rospy.get_param("~gps_jump_gate_m", 3.0))
        )
        self.gps_speed_gate_mps = max(
            0.5, float(rospy.get_param("~gps_speed_gate_mps", 10.0))
        )
        self.enable_gps_innovation_gate = bool(
            rospy.get_param("~enable_gps_innovation_gate", False)
        )
        self.gps_innovation_gate_m = max(
            0.1, float(rospy.get_param("~gps_innovation_gate_m", 5.0))
        )

        # [x, y, yaw, v_forward, gyro_bias_z, accel_bias_x]
        self.state = np.zeros((6, 1), dtype=float)
        self.covariance = np.diag([4.0, 4.0, 0.5, 4.0, 0.05, 0.5]).astype(float)
        self.initialized = False
        self.last_imu_stamp: Optional[float] = None
        self.last_gps_stamp: Optional[float] = None
        self.latest_z = 0.0
        self.latest_imu_yaw: Optional[float] = None
        self.latest_quality: Optional[SensorQuality] = None
        self.latest_quality_time = 0.0
        self.last_accepted_gps_position: Optional[tuple] = None
        self.last_accepted_gps_stamp: Optional[float] = None

        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=20)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseWithCovarianceStamped, queue_size=20)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.speed_diag_pub = rospy.Publisher("/localization/tunnel_speed_status", String, queue_size=1)
        if self.vehicle_speed_topic:
            self.speed_sub = rospy.Subscriber(self.vehicle_speed_topic, EgoVehicleStatus,
                                             self.vehicle_speed_callback, queue_size=1)
        self.gps_sub = rospy.Subscriber(self.gps_topic, Odometry, self.gps_callback, queue_size=20)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=50)
        self.quality_sub = None
        if self.use_sensor_quality_gate:
            self.quality_sub = rospy.Subscriber(
                self.quality_topic,
                SensorQuality,
                self.quality_callback,
                queue_size=20,
            )

        rospy.loginfo(
            "ENU EKF: gps=%s imu=%s output=%s frame=%s->%s imu_orientation=%s",
            self.gps_topic,
            self.imu_topic,
            self.odom_topic,
            self.map_frame,
            self.base_frame,
            self.use_imu_orientation,
        )

    @locked
    def quality_callback(self, message: SensorQuality) -> None:
        self.latest_quality = message
        self.latest_quality_time = rospy.get_time()

    def quality_fresh(self) -> bool:
        return (
            self.use_sensor_quality_gate
            and self.latest_quality is not None
            and rospy.get_time() - self.latest_quality_time <= self.quality_timeout_sec
        )

    def gps_update_allowed(self) -> bool:
        if not self.quality_fresh():
            return True
        return not bool(self.latest_quality.gps_blackout)

    @locked
    def imu_callback(self, msg: Imu) -> None:
        stamp = stamp_seconds(msg.header.stamp)
        planar = imu_planar_inputs(msg, self.imu_acceleration_includes_gravity)
        if planar is None:
            return
        accel_forward, yaw_rate = planar
        measured_yaw = quaternion_to_yaw(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        if measured_yaw is not None and not math.isfinite(measured_yaw):
            return
        if measured_yaw is not None:
            self.latest_imu_yaw = wrap_angle(measured_yaw + self.yaw_offset_rad)

        if not self.initialized:
            return

        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp
            return

        dt = stamp - self.last_imu_stamp
        if not math.isfinite(stamp) or dt <= 0.0001:
            return  # accumulate short intervals; do not rewind integration time
        self.last_imu_stamp = stamp
        if dt > 0.25:
            self.distance_scale.pause("imu_gap_frozen")
            self.distance_scale.motion.clear()
            rospy.logwarn_throttle(2.0, "IMU gap %.6f s: unknown motion is not reconstructed", dt)
            return
        self.latest_gyro_z = yaw_rate
        self.last_planar_accel = accel_forward
        if self.trust_imu:
            self.state[4:6, 0] = 0.0
            # Use the orientation at both ends of this time interval.
            if self.use_imu_orientation and self.latest_imu_yaw is not None:
                yaw_rate = wrap_angle(self.latest_imu_yaw-self.state[2, 0])/dt
        # A stale velocity at tunnel entry otherwise persists even with exact
        # zero acceleration. Apply NEW speed observations before integrating XY.
        self.fuse_vehicle_speed(stamp-dt)
        self.predict(gyro_z=yaw_rate, accel_x=accel_forward, dt=dt)
        if self.use_imu_orientation and self.latest_imu_yaw is not None:
            self.update_yaw(self.latest_imu_yaw)

        self.remember_prediction(stamp)
        self.publish(stamp)
        self.publish_speed_diagnostics()

    @locked
    def gps_callback(self, msg: Odometry) -> None:
        stamp = stamp_seconds(msg.header.stamp)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        z = float(msg.pose.pose.position.z)
        if (msg.header.frame_id != self.map_frame
                or not all(math.isfinite(v) for v in (stamp, x, y, z))
                or max(abs(x), abs(y)) > self.max_gps_local_coordinate
                or not -0.05 <= rospy.get_time()-stamp <= self.gps_timeout
                or not self.gps_update_allowed()):
            self.distance_scale.pause("invalid_gps_frozen")
            return

        gps_vx = float(msg.twist.twist.linear.x)
        gps_vy = float(msg.twist.twist.linear.y)
        gps_speed = math.hypot(gps_vx, gps_vy)
        if not math.isfinite(gps_speed):
            return

        if not self.initialized:
            self.state[0, 0] = x
            self.state[1, 0] = y
            if self.latest_imu_yaw is not None:
                self.state[2, 0] = self.latest_imu_yaw
            elif gps_speed > 0.3:
                self.state[2, 0] = math.atan2(gps_vy, gps_vx)
            else:
                self.state[2, 0] = 0.0
            self.state[3, 0] = gps_vx * math.cos(self.state[2, 0]) + gps_vy * math.sin(self.state[2, 0])
            vehicle_speed = self.fresh_vehicle_speed()
            if vehicle_speed is not None:
                self.state[3, 0] = vehicle_speed
            self.latest_z = z
            self.initialized = True
            self.last_gps_stamp = stamp
            self.last_accepted_gps_position = (x, y)
            self.last_accepted_gps_stamp = stamp
            if self.last_imu_stamp is None:
                self.last_imu_stamp = stamp
            rospy.loginfo(
                "ENU EKF 초기화: x=%.3f y=%.3f yaw=%.2f deg v=%.3f",
                self.state[0, 0],
                self.state[1, 0],
                math.degrees(self.state[2, 0]),
                self.state[3, 0],
            )
            self.remember_prediction(stamp)
            self.publish(stamp)
            return

        if self.last_gps_stamp is not None and stamp <= self.last_gps_stamp:
            return
        self.last_gps_stamp = stamp
        # Recovery must see three mutually consistent GPS samples; invalid
        # samples never move the accepted-position or freshness baseline.
        gps_gap = (self.last_accepted_gps_stamp is not None
                   and stamp-self.last_accepted_gps_stamp > self.gps_timeout)
        if gps_gap or self.gps_recovering:
            self.distance_scale.pause("gps_recovery_frozen")
            self.gps_recovering = True
            previous = self.gps_recovery_sample
            consistent = (previous is not None and 0 < stamp-previous[2] <= self.gps_timeout
                          and math.hypot(x-previous[0], y-previous[1]) <=
                          3.0+self.max_vehicle_speed*(stamp-previous[2]))
            self.gps_recovery_count = self.gps_recovery_count+1 if consistent else 1
            self.gps_recovery_sample = (x, y, stamp)
            if self.gps_recovery_count < 3:
                return
            if math.hypot(x-self.state[0, 0], y-self.state[1, 0]) > self.gps_recovery_max_residual:
                return

        if not self.gps_recovering and not self.gps_motion_gate_allowed(x, y, stamp):
            rospy.logwarn_throttle(
                2.0,
                "GPS jump gate가 비정상 위치 변화를 거부하여 GPS update를 건너뛴다.",
            )
            return

        if not self.gps_update_allowed():
            rospy.logwarn_throttle(
                2.0,
                "sensor_quality가 GPS blackout을 판정하여 GPS update를 건너뛴다.",
            )
            return

        if (
            self.enable_gps_innovation_gate and not self.gps_recovering
            and math.hypot(x - self.state[0, 0], y - self.state[1, 0])
            > self.gps_innovation_gate_m
        ):
            rospy.logwarn_throttle(
                2.0,
                "GPS innovation gate가 큰 위치 잔차를 거부하여 GPS update를 건너뛴다.",
            )
            return

        variance_x = float(msg.pose.covariance[0])
        variance_y = float(msg.pose.covariance[7])
        if not math.isfinite(variance_x) or variance_x <= 0.0:
            variance_x = self.default_gps_variance
        if not math.isfinite(variance_y) or variance_y <= 0.0:
            variance_y = self.default_gps_variance
        raw_x, raw_y = x, y
        aligned = self.align_gps_to_prediction(x, y, stamp)
        if aligned is None:
            return  # no time-matched prediction, never apply an old fix as current
        x, y = aligned
        # Limit the immediate correction in XY and in the correlated speed/
        # bias states when GPS resumes. Never teleport the inertial trajectory.
        before_state = self.state.copy()
        before_covariance = self.covariance.copy()
        self.update_position(x, y, variance_x, variance_y)
        correction = self.state-before_state
        correction[2, 0] = wrap_angle(correction[2, 0])
        length = math.hypot(correction[0, 0], correction[1, 0])
        if length > self.gps_correction_step:
            alpha = self.gps_correction_step/length
            self.state = before_state+alpha*correction
            self.state[2, 0] = wrap_angle(self.state[2, 0])
            # Conservative covariance mixture: do not claim a full update.
            self.covariance = (1-alpha)*before_covariance+alpha*self.covariance
        self.latest_z = z
        if math.hypot(x-self.state[0, 0], y-self.state[1, 0]) <= self.gps_correction_step:
            self.gps_recovering = False
            self.gps_recovery_sample = None
            self.gps_recovery_count = 0
        if not self.gps_recovering and not gps_gap:
            self.distance_scale.observe_gps(stamp, raw_x, raw_y)
        self.last_accepted_gps_position = (raw_x, raw_y)
        self.last_accepted_gps_stamp = stamp
        shift_x = self.state[0, 0]-before_state[0, 0]
        shift_y = self.state[1, 0]-before_state[1, 0]
        self.prediction_history = [(t, px+shift_x, py+shift_y) for t, px, py in self.prediction_history]
        output_stamp = max(stamp, self.last_imu_stamp or stamp)
        self.remember_prediction(output_stamp)
        self.publish(output_stamp)

    def remember_prediction(self, stamp):
        sample = (stamp, float(self.state[0, 0]), float(self.state[1, 0]))
        if self.prediction_history and stamp == self.prediction_history[-1][0]:
            self.prediction_history[-1] = sample
        elif not self.prediction_history or stamp > self.prediction_history[-1][0]:
            self.prediction_history.append(sample)
        cutoff = stamp-2.0
        self.prediction_history = [s for s in self.prediction_history if s[0] >= cutoff]

    def align_gps_to_prediction(self, x, y, stamp):
        if self.last_imu_stamp is None or stamp >= self.last_imu_stamp:
            return x, y
        h = self.prediction_history
        if not h or stamp < h[0][0]:
            return None
        for first, second in zip(h, h[1:]):
            if first[0] <= stamp <= second[0]:
                ratio = (stamp-first[0])/max(second[0]-first[0], 1e-12)
                px = first[1]+ratio*(second[1]-first[1])
                py = first[2]+ratio*(second[2]-first[2])
                return x+self.state[0, 0]-px, y+self.state[1, 0]-py
        return None

    @locked
    def vehicle_speed_callback(self, msg):
        # The existing UDP bridge already converted km/h to ROS m/s.
        now, wall = rospy.get_time(), time.monotonic()
        stamp = msg.header.stamp.to_sec()
        speed = math.hypot(float(msg.velocity.x), float(msg.velocity.y))
        if (not all(math.isfinite(v) for v in (stamp, speed)) or stamp <= 0
                or not 0 <= speed <= self.max_vehicle_speed
                or not -0.05 <= now-stamp <= self.vehicle_speed_timeout):
            return
        if self.vehicle_sample is not None and stamp <= self.vehicle_sample[1]:
            return
        self.vehicle_sample = (speed, stamp, wall)
        # Simulator truth is diagnostic only; it never updates filter state.
        position = getattr(msg, "position", None)
        if position is not None:
            xy = (float(position.x), float(position.y))
            if all(math.isfinite(v) for v in xy):
                self.ego_diagnostic_sample = (stamp, xy[0], xy[1])

    def fresh_vehicle_speed(self):
        sample = self.vehicle_sample
        if sample is None:
            return None
        speed, stamp, wall = sample
        if (-0.05 <= rospy.get_time()-stamp <= self.vehicle_speed_timeout
                and 0 <= time.monotonic()-wall <= self.vehicle_speed_timeout):
            return speed
        return None

    def fuse_vehicle_speed(self, imu_stamp):
        if not self.use_vehicle_speed_aiding:
            return
        speed = self.fresh_vehicle_speed()
        if speed is None or self.vehicle_sample[1] == self.last_fused_speed_stamp:
            return
        if self.vehicle_sample[1]-imu_stamp > 0.05:
            return
        if self.trust_imu:
            # The bridge already supplies m/s, independently of GPS. Correct
            # only velocity, never XY/yaw or an invented IMU bias. Extrapolate
            # a delayed speed sample with the current exact IMU acceleration.
            lag = imu_stamp-self.vehicle_sample[1]
            self.state[3, 0] = speed+self.last_planar_accel*lag
            self.covariance[3, :] = 0.0
            self.covariance[:, 3] = 0.0
            self.covariance[3, 3] = self.vehicle_speed_std**2
            self.last_fused_speed_stamp = self.vehicle_sample[1]
            return
        h = np.zeros((1, 6)); h[0, 3] = 1.0
        variance = self.vehicle_speed_std**2
        gain = self.covariance @ h.T / float((h @ self.covariance @ h.T)[0, 0]+variance)
        if self.trust_imu:
            # Optional speed aiding must not reinterpret exact IMU as a bias,
            # rotate heading or retroactively shift the integrated position.
            gain[[0, 1, 2, 4, 5], 0] = 0.0
        self.state += gain*(speed-self.state[3, 0])
        self.state[2, 0] = wrap_angle(self.state[2, 0])
        self.state[5, 0] = max(-self.accel_bias_limit, min(self.accel_bias_limit, self.state[5, 0]))
        identity = np.eye(6)-gain @ h
        self.covariance = identity @ self.covariance @ identity.T+variance*(gain @ gain.T)
        self.last_fused_speed_stamp = self.vehicle_sample[1]

    def publish_speed_diagnostics(self):
        wall, now = time.monotonic(), rospy.get_time()
        if wall-self.last_speed_diag_wall < 0.5:
            return
        self.last_speed_diag_wall = wall
        speed = self.fresh_vehicle_speed()
        gps_age = None if self.last_accepted_gps_stamp is None else now-self.last_accepted_gps_stamp
        truth = self.ego_diagnostic_sample
        position_error = None
        if truth is not None and -0.05 <= now-truth[0] <= self.vehicle_speed_timeout:
            # Interpolate the prediction to the truth timestamp, not receipt time.
            for a, b in zip(self.prediction_history, self.prediction_history[1:]):
                if a[0] <= truth[0] <= b[0]:
                    fraction = (truth[0]-a[0])/max(b[0]-a[0], 1e-12)
                    px, py = a[1]+fraction*(b[1]-a[1]), a[2]+fraction*(b[2]-a[2])
                    position_error = math.hypot(px-truth[1], py-truth[2])
                    break
        self.speed_diag_pub.publish(String(data=json.dumps(dict(
            version="TUNNEL_MOTION_V3", timestamp=now, trust_imu=self.trust_imu,
            vehicle_speed_aiding=self.use_vehicle_speed_aiding, accel_forward_mps2=self.last_planar_accel,
            gps_mode="RECOVERING" if self.gps_recovering else
                     "IMU_PREDICTION" if gps_age is None or gps_age > self.gps_timeout else "GPS_IMU",
            speed_source="ego_speed_fused" if speed is not None and self.use_vehicle_speed_aiding else "imu_prediction",
            vehicle_speed_mps=speed, ekf_forward_speed_mps=float(self.state[3, 0]),
            accel_bias_mps2=float(self.state[5, 0]), gps_age_s=gps_age,
            ekf_x_m=float(self.state[0, 0]), ekf_y_m=float(self.state[1, 0]),
            time_aligned_position_error_m=position_error,
            speed_error_mps=None if speed is None else float(self.state[3, 0])-speed,
            **self.distance_scale.status()), allow_nan=False)))

    def gps_motion_gate_allowed(self, x: float, y: float, stamp: float) -> bool:
        if not self.enable_gps_jump_gate:
            return True
        if (
            self.last_accepted_gps_position is None
            or self.last_accepted_gps_stamp is None
        ):
            return True
        dt = stamp - self.last_accepted_gps_stamp
        distance = math.hypot(
            x - self.last_accepted_gps_position[0],
            y - self.last_accepted_gps_position[1],
        )
        if dt <= 1e-3:
            return distance <= self.gps_jump_gate_m
        return distance <= self.gps_jump_gate_m and distance / dt <= self.gps_speed_gate_mps

    def predict(self, gyro_z: float, accel_x: float, dt: float) -> None:
        x, y, yaw, velocity, gyro_bias, accel_bias = self.state[:, 0]
        gyro_bias = 0.0 if self.trust_imu else gyro_bias
        accel_bias = 0.0 if self.trust_imu else max(-self.accel_bias_limit, min(self.accel_bias_limit, accel_bias))
        yaw_rate = gyro_z-gyro_bias
        acceleration = accel_x-accel_bias
        mid_yaw = yaw+0.5*yaw_rate*dt
        raw_distance = velocity*dt+0.5*acceleration*dt*dt
        c, sn = math.cos(mid_yaw), math.sin(mid_yaw)
        self.distance_scale.add_motion(
            self.last_imu_stamp, raw_distance*c, raw_distance*sn, dt,
            valid=self.use_vehicle_speed_aiding and self.fresh_vehicle_speed() is not None)
        factor = self.distance_scale.factor
        distance = raw_distance*factor
        self.state[0, 0] = x+distance*c
        self.state[1, 0] = y+distance*sn
        self.state[2, 0] = wrap_angle(yaw+yaw_rate*dt)
        self.state[3, 0] = velocity+acceleration*dt
        f = np.eye(6)
        f[0, 2] = -distance*sn; f[0, 3] = factor*dt*c
        f[1, 2] = distance*c; f[1, 3] = factor*dt*sn
        if not self.trust_imu:
            f[0, 4] = 0.5*dt*distance*sn; f[1, 4] = -0.5*dt*distance*c
            f[0, 5] = -0.5*factor*dt*dt*c; f[1, 5] = -0.5*factor*dt*dt*sn
            f[2, 4] = -dt; f[3, 5] = -dt
        q = np.diag([.02*dt, .02*dt, self.gyro_noise_std**2*dt*dt,
                     self.accel_noise_std**2*dt*dt,
                     0.0 if self.trust_imu else self.bias_random_walk*dt,
                     0.0 if self.trust_imu else self.bias_random_walk*dt])
        self.covariance = f @ self.covariance @ f.T+q

    def update_position(self, x: float, y: float, variance_x: float, variance_y: float) -> None:
        h = np.zeros((2, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        measurement = np.array([[x], [y]], dtype=float)
        expected = h @ self.state
        innovation = measurement - expected
        r = np.diag([variance_x, variance_y])
        s = h @ self.covariance @ h.T + r
        gain = self.covariance @ h.T @ np.linalg.inv(s)
        if self.trust_imu:
            gain[2:6, :] = 0.0  # GPS corrects XY, never exact IMU heading/speed/bias
        self.state = self.state + gain @ innovation
        self.state[2, 0] = wrap_angle(self.state[2, 0])
        identity = np.eye(6, dtype=float)
        residual = identity-gain @ h
        self.covariance = residual @ self.covariance @ residual.T+gain @ r @ gain.T

    def update_yaw(self, yaw_measurement: float) -> None:
        if self.trust_imu:
            self.state[2, 0] = wrap_angle(yaw_measurement)
            self.covariance[2, :] = 0.0
            self.covariance[:, 2] = 0.0
            self.covariance[2, 2] = 1e-9
            return
        h = np.zeros((1, 6), dtype=float)
        h[0, 2] = 1.0
        innovation = np.array([[wrap_angle(yaw_measurement - self.state[2, 0])]], dtype=float)
        s = h @ self.covariance @ h.T + self.default_yaw_variance
        gain = self.covariance @ h.T / float(s[0, 0])
        self.state = self.state + gain @ innovation
        self.state[2, 0] = wrap_angle(self.state[2, 0])
        self.covariance = (np.eye(6) - gain @ h) @ self.covariance

    def publish(self, stamp_seconds_value: float) -> None:
        stamp = rospy.Time.from_sec(stamp_seconds_value)
        x, y, yaw, velocity, _, _ = self.state[:, 0]
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = self.latest_z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = velocity * math.cos(yaw)
        odom.twist.twist.linear.y = velocity * math.sin(yaw)
        odom.twist.twist.angular.z = self.latest_gyro_z-(0.0 if self.trust_imu else float(self.state[4, 0]))
        odom.twist.covariance[0] = self.covariance[3, 3]*math.cos(yaw)**2
        odom.twist.covariance[7] = self.covariance[3, 3]*math.sin(yaw)**2
        odom.pose.covariance[0] = self.covariance[0, 0]
        odom.pose.covariance[7] = self.covariance[1, 1]
        odom.pose.covariance[35] = self.covariance[2, 2]
        self.odom_pub.publish(odom)

        pose = PoseWithCovarianceStamped()
        pose.header = odom.header
        pose.pose.pose = odom.pose.pose
        pose.pose.covariance = odom.pose.covariance
        self.pose_pub.publish(pose)

        if self.publish_tf:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = self.latest_z
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(transform)

        rospy.loginfo_throttle(
            2.0,
            "EKF pose map=(%.3f, %.3f) yaw=%.2f deg v=%.3f",
            x,
            y,
            math.degrees(yaw),
            velocity,
        )


if __name__ == "__main__":
    try:
        LocalEnuEkf()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
