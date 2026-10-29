#!/usr/bin/env python3
"""Global MGeo route + dynamic avoidance/lane-change path Pure Pursuit.

The competition route remains read-only.  A path manager can publish a rolling
``nav_msgs/Path`` generated from that route (Frenet bypass or lane change); this
node recomputes curvature and the spatial speed profile for the active path and
uses the result for steering and accel/brake control.
"""

from __future__ import annotations

import math
import json
import os
import threading
import time
from typing import List, Optional, Tuple

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64, String

from purepursuit_mgeo.longitudinal_controller import (
    VehicleSpeedSource,
    LongitudinalCommand,
    MPS_TO_KPH,
    SpeedPIController,
)
from purepursuit_mgeo.plan_transport import read_trajectory, plan_locked
from curvature_speed_purepursuit.planner import (
    PathPoint,
    build_speed_profile,
    adaptive_lookahead_m,
    max_abs_curvature_ahead,
    clean_consecutive_duplicates,
    cumulative_arc_lengths,
    curvature_profile,
    interpolate_by_s,
    load_path_file,
    nearest_projection,
    profile_value_at_s,
)


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AdaptiveCurvaturePurePursuit:
    def __init__(self) -> None:
        rospy.init_node("adaptive_curvature_purepursuit", anonymous=False)

        default_path = os.path.join(
            os.environ.get("HOME", "/home"),
            "morai_ws",
            "data",
            "routes",
            "2026_molit_comp_global_path.txt",
        )
        path_file = rospy.get_param("~path_file", default_path)
        self.base_points = clean_consecutive_duplicates(
            load_path_file(path_file),
            float(rospy.get_param("~duplicate_epsilon_m", 1.0e-6)),
        )
        self.base_s = cumulative_arc_lengths(self.base_points)
        self.base_curvature = curvature_profile(
            self.base_points,
            half_window_points=max(1, int(rospy.get_param("~curvature_half_window_points", 1))),
            smoothing_window=max(1, int(rospy.get_param("~curvature_smoothing_window", 5))),
        )

        self.vehicle_speed_topic = str(rospy.get_param("~vehicle_speed_topic", "/Ego_topic"))
        self.require_vehicle_speed = bool(rospy.get_param("~require_vehicle_speed", True))
        self.vehicle_speed_source = VehicleSpeedSource(
            timeout_sec=float(rospy.get_param("~vehicle_speed_timeout_sec", 0.5)))
        self.speed_source_name = "unavailable"
        if self.require_vehicle_speed and not self.vehicle_speed_topic:
            raise ValueError("Required vehicle speed needs a topic")
        self.max_speed_kph = max(0.0, float(rospy.get_param("~max_speed_kph", 30.0)))
        self.lateral_accel_limit = max(
            1.0e-6, float(rospy.get_param("~lateral_accel_limit_mps2", 1.0))
        )
        self.max_accel_mps2 = max(1.0e-6, float(rospy.get_param("~max_accel_mps2", 1.0)))
        self.max_decel_mps2 = max(1.0e-6, float(rospy.get_param("~max_decel_mps2", 1.5)))
        self.final_speed_kph = max(0.0, float(rospy.get_param("~final_speed_kph", 0.0)))
        self.curvature_half_window = max(1, int(rospy.get_param("~curvature_half_window_points", 1)))
        self.curvature_smoothing_window = max(1, int(rospy.get_param("~curvature_smoothing_window", 5)))

        self.wheelbase_m = max(0.1, float(rospy.get_param("~wheelbase_m", 3.0)))
        self.lookahead_min_m = max(0.1, float(rospy.get_param("~lookahead_min_m", 4.0)))
        self.lookahead_gain = max(0.0, float(rospy.get_param("~lookahead_gain", 0.35)))
        self.max_steering_rad = max(
            0.01, float(rospy.get_param("~max_steering_rad", math.radians(40.0)))
        )
        self.steering_sign = 1.0 if float(rospy.get_param("~steering_sign", 1.0)) >= 0.0 else -1.0
        self.lookahead_curvature_gain = float(rospy.get_param("~lookahead_curvature_gain", 6.0))
        self.lookahead_tight_min_m = float(rospy.get_param("~lookahead_tight_min_m", 2.2))
        self.lookahead_max_m = float(rospy.get_param("~lookahead_max_m", 12.0))
        self.steering_feedforward_weight = _clamp(float(rospy.get_param("~steering_feedforward_weight", 0.35)), 0.0, 1.0)
        self.max_steering_rate_rad_s = max(0.0, float(rospy.get_param("~max_steering_rate_rad_s", 0.5)))
        self.last_steering_rad = 0.0
        self.status_pub = rospy.Publisher("/control/curvature_status", String, queue_size=1)
        self.goal_tolerance_m = max(0.0, float(rospy.get_param("~goal_tolerance_m", 1.5)))

        self.rate_hz = max(1.0, float(rospy.get_param("~control_rate_hz", 20.0)))
        self.pose_topic = rospy.get_param("~pose_topic", "/localization/odometry")
        self.command_topic = rospy.get_param("~command_topic", "/control/ctrl_cmd")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.publish_command = bool(rospy.get_param("~publish_command", True))
        self.longl_cmd_type = int(rospy.get_param("~longl_cmd_type", 1))

        self.use_active_path = bool(rospy.get_param("~use_active_path", True))
        self.active_path_topic = rospy.get_param(
            "~active_path_topic", "/highway_lane_strategy/active_path"
        )
        self.active_path_timeout_sec = max(
            0.1, float(rospy.get_param("~active_path_timeout_sec", 1.0))
        )
        self.require_active_path = bool(rospy.get_param("~require_active_path", True))
        self.stop_required_topic = rospy.get_param(
            "~stop_required_topic", "/highway_lane_strategy/stop_required"
        )
        self.require_fresh_stop_status = bool(
            rospy.get_param("~require_fresh_stop_status", True)
        )
        self.stop_status_timeout_sec = max(
            0.1, float(rospy.get_param("~stop_status_timeout_sec", 1.0))
        )

        self.use_target_speed_override = bool(
            rospy.get_param("~use_target_speed_override", False)
        )
        self.target_speed_override_topic = rospy.get_param(
            "~target_speed_override_topic", "/highway_lane_strategy/target_speed_mps"
        )
        self.target_speed_override_timeout_sec = max(
            0.1, float(rospy.get_param("~target_speed_override_timeout_sec", 1.0))
        )
        self.require_route_mission_ready = bool(rospy.get_param("~require_route_mission_ready", False))
        self.route_mission_ready_topic = rospy.get_param("~route_mission_ready_topic", "/route_mission_gate/ready")
        self.route_mission_timeout_sec = max(0.1, float(rospy.get_param("~route_mission_timeout_sec", 0.8)))
        self.enable_merge_gate = bool(rospy.get_param("~enable_merge_gate", False))
        self.merge_stop_topic = rospy.get_param("~merge_stop_topic", "/roundabout_merge_gate/stop_required")
        self.merge_gate_timeout_sec = max(0.1, float(rospy.get_param("~merge_gate_timeout_sec", 0.8)))

        self.speed_controller = SpeedPIController(
            kp=max(0.0, float(rospy.get_param("~speed_kp", 0.8))),
            ki=max(0.0, float(rospy.get_param("~speed_ki", 0.05))),
            max_accel_mps2=self.max_accel_mps2,
            max_decel_mps2=self.max_decel_mps2,
            integral_limit_kph_s=max(
                0.0, float(rospy.get_param("~speed_integral_limit_kph_s", 10.8))
            ),
            speed_error_deadband_kph=max(
                0.0, float(rospy.get_param("~speed_error_deadband_kph", 0.1))
            ),
        )

        self._lock = threading.RLock()
        self._plan_lock = self._lock
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "")
        self.trajectory_reason = ""
        self.latest_odom: Optional[Odometry] = None
        self.latest_odom_wall_time: Optional[float] = None
        self.active_points: List[PathPoint] = list(self.base_points)
        self.active_s: List[float] = list(self.base_s)
        self.active_curvature: List[float] = list(self.base_curvature)
        self.active_speed_profile = self._make_speed_profile(
            self.active_s,
            self.active_curvature,
            final_speed_mps=self.final_speed_kph / MPS_TO_KPH,
        )
        self.active_path_received = not self.use_active_path
        self.active_path_wall_time: Optional[float] = None
        self.active_path_seq: Optional[int] = None
        self.active_source = "global"
        self.stop_required = False
        self.stop_status_wall_time: Optional[float] = None
        self.target_speed_override_mps = self.max_speed_kph / MPS_TO_KPH
        self.target_speed_override_wall_time: Optional[float] = None
        self.last_control_wall_time: Optional[float] = None
        self.command_speed_mps = 0.0
        self.route_mission_ready = False
        self.route_mission_wall_time: Optional[float] = None
        self.merge_stop_required = True
        self.merge_status_wall_time: Optional[float] = None

        self.command_pub = rospy.Publisher(self.command_topic, CtrlCmd, queue_size=1)
        self.lookahead_pub = rospy.Publisher(
            "/experimental/curvature_lookahead_point", PointStamped, queue_size=1
        )
        self.reference_pub = rospy.Publisher(
            "/experimental/active_reference_path", Path, queue_size=1
        )
        self.global_reference_pub = rospy.Publisher(
            "/experimental/curvature_reference_path", Path, queue_size=1, latch=True
        )
        self.progress_pub = rospy.Publisher("/experimental/curvature_progress", Float64, queue_size=1)
        self.curvature_pub = rospy.Publisher(
            "/experimental/curvature_value", Float64, queue_size=1
        )
        self.speed_limit_pub = rospy.Publisher(
            "/experimental/curvature_speed_limit", Float64, queue_size=1
        )
        self.speed_command_pub = rospy.Publisher(
            "/experimental/curvature_speed_command", Float64, queue_size=1
        )
        self.accel_pub = rospy.Publisher(
            "/experimental/curvature_accel_command", Float64, queue_size=1
        )
        self.brake_pub = rospy.Publisher(
            "/experimental/curvature_brake_command", Float64, queue_size=1
        )
        self.steering_pub = rospy.Publisher(
            "/experimental/curvature_steering", Float64, queue_size=1
        )
        self.source_pub = rospy.Publisher(
            "/experimental/active_path_source", String, queue_size=1
        )
        self.seq_pub = rospy.Publisher(
            "/experimental/active_path_seq", String, queue_size=1
        )

        rospy.Subscriber(self.pose_topic, Odometry, self._odom_cb, queue_size=10)
        if self.require_route_mission_ready:
            rospy.Subscriber(self.route_mission_ready_topic, Bool, self._route_mission_cb, queue_size=1)
        if self.enable_merge_gate:
            rospy.Subscriber(self.merge_stop_topic, Bool, self._merge_stop_cb, queue_size=1)
        if self.trajectory_topic:
            rospy.Subscriber(self.trajectory_topic, String, self._trajectory_cb, queue_size=1)
        elif self.use_active_path:
            rospy.Subscriber(self.active_path_topic, Path, self._active_path_cb, queue_size=1)
        if self.require_fresh_stop_status and not self.trajectory_topic:
            rospy.Subscriber(self.stop_required_topic, Bool, self._stop_cb, queue_size=1)
        if self.use_target_speed_override and not self.trajectory_topic:
            rospy.Subscriber(
                self.target_speed_override_topic,
                Float64,
                self._target_speed_cb,
                queue_size=1,
            )

        self._publish_global_path()
        if self.vehicle_speed_topic:
            rospy.Subscriber(self.vehicle_speed_topic, EgoVehicleStatus, self._vehicle_speed_cb, queue_size=1)
        self._publish_active_path()
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._control_cb
        )
        rospy.logwarn(
            "Adaptive curvature PP path=%s active=%s topic=%s max=%.1fkm/h",
            path_file,
            self.use_active_path,
            self.active_path_topic,
            self.max_speed_kph,
        )

    def _make_speed_profile(
        self,
        s_values: List[float],
        curvatures: List[float],
        final_speed_mps: float,
    ) -> List[float]:
        # A rolling maneuver path ends at its planning horizon, so it must not
        # inherit the global route's terminal zero-speed constraint.
        return build_speed_profile(
            s_values,
            curvatures,
            max_speed_mps=self.max_speed_kph / MPS_TO_KPH,
            lateral_accel_limit_mps2=self.lateral_accel_limit,
            max_accel_mps2=self.max_accel_mps2,
            max_decel_mps2=self.max_decel_mps2,
            initial_speed_mps=None,
            final_speed_mps=max(0.0, float(final_speed_mps)),
        )

    def _odom_cb(self, msg: Odometry) -> None:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        try:
            age = float(rospy.get_time()) - float(stamp.to_sec())
            values = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                      msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
                      msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.latest_odom_wall_time = None
            return
        if (header.frame_id != self.map_frame
                or not all(math.isfinite(float(v)) for v in values)
                or not math.isfinite(age) or not -0.05 <= age <= 0.5):
            self.latest_odom_wall_time = None
            return
        self.latest_odom = msg
        self.latest_odom_wall_time = time.monotonic()

    def _stop_cb(self, msg: Bool) -> None:
        if self.trajectory_topic:
            return
        self.stop_required = bool(msg.data)
        self.stop_status_wall_time = time.monotonic()

    def _route_mission_cb(self, msg: Bool) -> None:
        self.route_mission_ready = bool(msg.data)
        self.route_mission_wall_time = time.monotonic()

    def _merge_stop_cb(self, msg: Bool) -> None:
        self.merge_stop_required = bool(msg.data)
        self.merge_status_wall_time = time.monotonic()

    def _vehicle_speed_cb(self, msg):
        speed = math.hypot(float(msg.velocity.x), float(msg.velocity.y))
        self.vehicle_speed_source.observe(speed, msg.header.stamp.to_sec(),
                                          rospy.get_time(), time.monotonic())

    def _measured_speed(self, ros_now, wall_now):
        speed = self.vehicle_speed_source.sample(ros_now, wall_now)
        if speed is not None:
            self.speed_source_name = "ego_vehicle_status"
            return speed, None
        if self.require_vehicle_speed:
            self.speed_source_name = "unavailable"
            return 0.0, "vehicle_speed_stale_or_unavailable"
        self.speed_source_name = "ekf_imu_fallback"
        v = self.latest_odom.twist.twist.linear
        return math.hypot(v.x, v.y), None

    def _target_speed_cb(self, msg: Float64) -> None:
        if self.trajectory_topic:
            return
        value = float(msg.data)
        if math.isfinite(value):
            self.target_speed_override_mps = max(0.0, value)
            self.target_speed_override_wall_time = time.monotonic()

    def _active_path_cb(self, msg: Path) -> None:
        if self.trajectory_topic:
            return
        self._accept_active_path(msg)

    @plan_locked
    def _trajectory_cb(self, msg: String) -> None:
        try:
            path, stop, speed, reason = read_trajectory(
                msg.data, self.map_frame, rospy.get_time(),
                min(self.active_path_timeout_sec, self.stop_status_timeout_sec,
                    self.target_speed_override_timeout_sec),
                path_type=Path, pose_type=PoseStamped, stamp_type=rospy.Time)
            # A rejected new plan must not inherit the previous plan's permit.
            self.active_path_received = False
            self._accept_active_path(path)
        except (KeyError, TypeError, ValueError) as exc:
            self.active_path_received = False
            self.stop_required = True
            self.trajectory_reason = 'invalid_trajectory:' + str(exc)
            return
        source_age = max(0.0, rospy.get_time() - path.header.stamp.to_sec())
        received_at = time.monotonic() - source_age
        self.active_path_wall_time = received_at
        self.stop_required, self.target_speed_override_mps = stop, speed
        self.stop_status_wall_time = self.target_speed_override_wall_time = received_at
        self.trajectory_reason = reason

    def _accept_active_path(self, msg: Path) -> None:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        try:
            age = float(rospy.get_time()) - float(stamp.to_sec())
        except (AttributeError, TypeError, ValueError, OverflowError):
            rospy.logwarn_throttle(2.0, "active path ignored: missing or invalid timestamp")
            return
        if (header.frame_id != self.map_frame or not math.isfinite(age)
                or not -0.05 <= age <= self.active_path_timeout_sec):
            rospy.logwarn_throttle(
                2.0,
                "active path frame=%s ignored; expected %s",
                msg.header.frame_id,
                self.map_frame,
            )
            return
        if any(not all(math.isfinite(float(v)) for v in (p.pose.position.x,p.pose.position.y,p.pose.position.z)) for p in msg.poses):
            return
        points = [
            PathPoint(
                float(p.pose.position.x),
                float(p.pose.position.y),
                float(p.pose.position.z),
            )
            for p in msg.poses
            if all(
                math.isfinite(float(v))
                for v in (
                    p.pose.position.x,
                    p.pose.position.y,
                    p.pose.position.z,
                )
            )
        ]
        if len(points) < 2:
            # Empty output is a deliberate stop/no-safe-path signal.  Keep the
            # previous geometry for diagnostics, but let freshness expire.
            rospy.logwarn_throttle(1.0, "active path is empty")
            return
        try:
            points = clean_consecutive_duplicates(points)
            s_values = cumulative_arc_lengths(points)
            curvatures = curvature_profile(
                points,
                half_window_points=self.curvature_half_window,
                smoothing_window=self.curvature_smoothing_window,
            )
            speed_profile = self._make_speed_profile(
                s_values, curvatures, final_speed_mps=self.max_speed_kph / MPS_TO_KPH
            )
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(1.0, "active path rejected: %s", exc)
            return
        with self._lock:
            self.active_points = points
            self.active_s = s_values
            self.active_curvature = curvatures
            self.active_speed_profile = speed_profile
            self.active_path_received = True
            self.active_path_wall_time = time.monotonic()
            self.active_path_seq = int(msg.header.seq)
            self.active_source = "dynamic"
        self._publish_active_path()

    def _publish_active_path(self) -> None:
        with self._lock:
            points = list(self.active_points)
            source = self.active_source
            seq = self.active_path_seq
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        if seq is not None:
            msg.header.seq = seq
        for p in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = p.x
            pose.pose.position.y = p.y
            pose.pose.position.z = p.z
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.reference_pub.publish(msg)
        self.source_pub.publish(String(data=source))
        self.seq_pub.publish(String(data="" if seq is None else str(seq)))

    def _publish_global_path(self) -> None:
        """Keep the immutable competition route available for route context and RViz."""
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        for point in self.base_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.global_reference_pub.publish(msg)

    def _managed_fault(self, now_wall: float) -> Optional[str]:
        if self.require_route_mission_ready:
            if self.route_mission_wall_time is None:
                return "route_mission_missing"
            if not 0.0 <= now_wall - self.route_mission_wall_time <= self.route_mission_timeout_sec:
                return "route_mission_stale"
            if not self.route_mission_ready:
                return "route_mission_not_ready"
        if self.enable_merge_gate:
            if self.merge_status_wall_time is None:
                return "merge_gate_missing"
            if not 0.0 <= now_wall - self.merge_status_wall_time <= self.merge_gate_timeout_sec:
                return "merge_gate_stale"
        if self.require_active_path and self.use_active_path:
            if not self.active_path_received or self.active_path_wall_time is None:
                return "active_path_missing"
            if now_wall - self.active_path_wall_time > self.active_path_timeout_sec:
                return "active_path_stale"
        if self.require_fresh_stop_status:
            if self.stop_status_wall_time is None:
                return "stop_status_missing"
            if now_wall - self.stop_status_wall_time > self.stop_status_timeout_sec:
                return "stop_status_stale"
        if self.use_target_speed_override:
            if self.target_speed_override_wall_time is None:
                return "target_speed_override_missing"
            if now_wall - self.target_speed_override_wall_time > self.target_speed_override_timeout_sec:
                return "target_speed_override_stale"
        return None

    def _compute_steering(
        self,
        points: List[PathPoint],
        s_values: List[float],
        x: float,
        y: float,
        yaw: float,
        speed_mps: float,
        progress_s: float,
        curvatures: List[float],
    ) -> Tuple[float, PathPoint, float]:
        preview = max_abs_curvature_ahead(s_values, curvatures, progress_s,
            max(8.0, self.lookahead_min_m+self.lookahead_gain*speed_mps), 0.5)
        lookahead = adaptive_lookahead_m(speed_mps, self.lookahead_min_m, self.lookahead_gain,
            preview, self.lookahead_curvature_gain, self.lookahead_tight_min_m, self.lookahead_max_m)
        target, _ = interpolate_by_s(
            points, s_values, min(s_values[-1], progress_s + lookahead)
        )
        dx, dy = target.x - x, target.y - y
        c, s = math.cos(yaw), math.sin(yaw)
        target_x_body = c * dx + s * dy
        target_y_body = -s * dx + c * dy
        actual_lookahead = max(math.hypot(target_x_body, target_y_body), 1.0e-3)
        alpha = math.atan2(target_y_body, target_x_body)
        curvature = 2.0 * math.sin(alpha) / actual_lookahead
        ff = profile_value_at_s(s_values, curvatures, min(s_values[-1],progress_s+0.75*lookahead))
        w = self.steering_feedforward_weight
        steering = ((1-w)*math.atan(self.wheelbase_m*curvature)+w*math.atan(self.wheelbase_m*ff))*self.steering_sign
        return _clamp(steering, -self.max_steering_rad, self.max_steering_rad), target, actual_lookahead

    def _rate_limited_speed(self, target: float, dt: float) -> float:
        target = max(0.0, float(target))
        delta = target - self.command_speed_mps
        limit = self.max_accel_mps2 if delta >= 0.0 else self.max_decel_mps2
        self.command_speed_mps += _clamp(delta, -limit * max(dt, 1.0e-3), limit * max(dt, 1.0e-3))
        self.command_speed_mps = max(0.0, self.command_speed_mps)
        return self.command_speed_mps

    def _make_command(
        self,
        steering: float,
        pedal: LongitudinalCommand,
        stop: bool,
    ) -> CtrlCmd:
        msg = CtrlCmd()
        if hasattr(msg, "longlCmdType"):
            msg.longlCmdType = self.longl_cmd_type
        if hasattr(msg, "steering"):
            msg.steering = 0.0 if stop else steering
        if hasattr(msg, "accel"):
            msg.accel = 0.0 if stop else pedal.accel
        if hasattr(msg, "brake"):
            msg.brake = 1.0 if stop else pedal.brake
        if hasattr(msg, "acceleration"):
            msg.acceleration = 0.0
        if hasattr(msg, "velocity"):
            msg.velocity = 0.0
        return msg

    @plan_locked
    def _control_cb(self, _event) -> None:
        if self.latest_odom is None or self.latest_odom_wall_time is None:
            self.command_speed_mps = 0.0
            self.speed_controller.reset()
            if self.publish_command: self.command_pub.publish(self._make_command(0.0, None, True))
            self.status_pub.publish(String(data=json.dumps({"stop":True,"reason":"odometry_missing_or_invalid"})))
            return
        now_wall = time.monotonic()
        if now_wall - self.latest_odom_wall_time > 0.5:
            fault = "odometry_stale"
        else:
            fault = self._managed_fault(now_wall)

        now = rospy.Time.now().to_sec()
        dt = 1.0 / self.rate_hz if self.last_control_wall_time is None else now_wall - self.last_control_wall_time
        self.last_control_wall_time = now_wall
        dt = _clamp(dt, 1.0e-3, 0.25)

        pose = self.latest_odom.pose.pose
        x, y = float(pose.position.x), float(pose.position.y)
        yaw = _yaw_from_quaternion(pose.orientation)
        speed_mps, speed_fault = self._measured_speed(now, now_wall)
        if speed_fault:
            fault = fault or speed_fault

        with self._lock:
            points = list(self.active_points)
            s_values = list(self.active_s)
            curvatures = list(self.active_curvature)
            speed_profile = list(self.active_speed_profile)

        stop = fault is not None or self.stop_required or (self.enable_merge_gate and self.merge_stop_required)
        steering = 0.0
        target = points[0]
        lookahead = 0.0
        curvature = 0.0
        target_speed_kph = 0.0
        path_speed_limit_kph = 0.0
        progress_s = 0.0
        global_projection = nearest_projection(self.base_points, self.base_s, x, y)
        global_remaining = max(0.0, self.base_s[-1] - global_projection.progress_s)
        self.progress_pub.publish(Float64(global_projection.progress_s))

        if not stop and len(points) >= 2:
            projection = nearest_projection(points, s_values, x, y)
            progress_s = projection.progress_s
            steering, target, lookahead = self._compute_steering(
                points, s_values, x, y, yaw, speed_mps, progress_s, curvatures
            )
            curvature = profile_value_at_s(s_values, curvatures, progress_s)
            path_limit = profile_value_at_s(s_values, speed_profile, progress_s)
            path_speed_limit_kph = path_limit * MPS_TO_KPH
            target_speed_mps = min(
                path_limit,
                self.target_speed_override_mps
                if self.use_target_speed_override
                else self.max_speed_kph / MPS_TO_KPH,
            )
            target_speed_kph = self._rate_limited_speed(target_speed_mps, dt) * MPS_TO_KPH
            if global_remaining <= self.goal_tolerance_m:
                stop = True
        else:
            self.command_speed_mps = 0.0
            self.speed_controller.reset()

        if stop:
            self.command_speed_mps, target_speed_kph = 0.0, 0.0
            self.speed_controller.reset()
            self.last_steering_rad = 0.0
        else:
            delta = self.max_steering_rate_rad_s * dt
            if delta > 0:
                steering = _clamp(steering,self.last_steering_rad-delta,self.last_steering_rad+delta)
            self.last_steering_rad = steering
        self.status_pub.publish(String(data=json.dumps({"stop":bool(stop),
            "reason":fault or ("merge_gate_stop" if self.enable_merge_gate and self.merge_stop_required else
                               "managed_stop" if self.stop_required else "goal" if stop else "tracking"),
            "trajectory_reason":self.trajectory_reason,"trajectory_seq":self.active_path_seq,
            "target_speed_kph":target_speed_kph,"measured_speed_kph":speed_mps*MPS_TO_KPH,
            "speed_source":self.speed_source_name,
            "path_speed_limit_kph":path_speed_limit_kph,"curvature":curvature})))
        pedal = self.speed_controller.update(
            target_speed_kph,
            speed_mps * MPS_TO_KPH,
            dt,
            stop=stop,
        )
        self.curvature_pub.publish(Float64(curvature))
        self.speed_limit_pub.publish(Float64(path_speed_limit_kph))
        self.speed_command_pub.publish(Float64(target_speed_kph))
        self.accel_pub.publish(Float64(pedal.accel))
        self.brake_pub.publish(Float64(pedal.brake if not stop else 1.0))
        self.steering_pub.publish(Float64(0.0 if stop else steering))
        self.source_pub.publish(String(data=self.active_source))
        self.seq_pub.publish(String(data="" if self.active_path_seq is None else str(self.active_path_seq)))

        lookahead_msg = PointStamped()
        lookahead_msg.header.stamp = rospy.Time.now()
        lookahead_msg.header.frame_id = self.map_frame
        lookahead_msg.point.x = target.x
        lookahead_msg.point.y = target.y
        lookahead_msg.point.z = target.z
        self.lookahead_pub.publish(lookahead_msg)

        if self.publish_command:
            self.command_pub.publish(self._make_command(steering, pedal, stop))
        rospy.loginfo_throttle(
            1.0,
            "adaptive PP source=%s seq=%s s=%.1f global=%.1f/%.1f kappa=%.4f "
            "target=%.1f measured=%.1f accel=%.2f brake=%.2f steer=%.3f stop=%s reason=%s speed_source=%s",
            self.active_source,
            "-" if self.active_path_seq is None else str(self.active_path_seq),
            progress_s,
            global_projection.progress_s,
            self.base_s[-1],
            curvature,
            target_speed_kph,
            speed_mps * MPS_TO_KPH,
            pedal.accel,
            1.0 if stop else pedal.brake,
            0.0 if stop else steering,
            stop,
            fault or ("path_manager_stop" if self.stop_required else ""),
            self.speed_source_name,
        )


if __name__ == "__main__":
    try:
        AdaptiveCurvaturePurePursuit()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

