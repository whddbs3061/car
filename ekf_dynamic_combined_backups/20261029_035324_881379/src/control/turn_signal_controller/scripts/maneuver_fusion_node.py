#!/usr/bin/env python3
"""Final type-1 command overlay: route, directional lights, stop line and safety.

The route remains the sole steering plan. This node owns stop-line braking
when enabled, downstream of lane fallback, and transmits the separate MORAI
lamp protocol. No network transmission occurs when lamp_output_enabled=false.
"""

import copy
import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path

import rospy
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from morai_perception_msgs.msg import TrafficLight, StopLineDetection, SafetyStop, SensorQuality
from nav_msgs.msg import Odometry, Path as RosPath
from std_msgs.msg import String
from common.msg import ObjectInfoArray
from camera_perception.traffic_signal import directional_observation, count_signal_housings
from purepursuit_mgeo.dynamic_zone_guard import DynamicZoneGuard, constrain_pedals
from purepursuit_mgeo.longitudinal_controller import VehicleSpeedSource
from purepursuit_mgeo.plan_transport import read_trajectory

from curvature_speed_purepursuit.planner import (
    load_path_file, clean_consecutive_duplicates, cumulative_arc_lengths, nearest_projection,
    nearest_projection_on_loop,
)
from stopline_control.core import AccelRiseLimiter, StopLineControllerCore, Sample, Decision, finite, clamp
from turn_signal_controller.highway_route import corridor_matches
from turn_signal_controller.fusion import (
    IndicatorLead, build_lamp_packet, route_intent, signal_permits, signal_allowed_directions,
)
from turn_signal_controller.route_contract import reference_path_reason, route_event_key, route_checked_maneuvers
from turn_signal_controller.scheduler import parse_maneuvers
from turn_signal_controller.route_context import load_route_contexts
from turn_signal_controller.turn_motion import TurnMotion, TurnMotionPlanner, quaternion_yaw
from turn_signal_controller.signal_association import (
    Selection, project_signal, associate, SignalConfirmation,
)


def overlay(nominal, decision):
    output = copy.deepcopy(nominal)
    output.longlCmdType = 1
    output.velocity = output.acceleration = 0.0
    output.accel = min(clamp(float(nominal.accel), 0.0, 1.0), decision.accel_limit)
    output.brake = max(clamp(float(nominal.brake), 0.0, 1.0), decision.brake)
    if output.brake > 0:
        output.accel = 0.0
    return output


class ManeuverFusionNode:
    def __init__(self):
        rospy.init_node("maneuver_fusion", anonymous=False)
        self.lock = threading.RLock()
        self.points = clean_consecutive_duplicates(load_path_file(rospy.get_param("~path_file")))
        self.s_values = cumulative_arc_lengths(self.points)
        self.loop_route = bool(rospy.get_param("~loop_route", False))
        if self.loop_route:
            seam_gap = math.hypot(self.points[0].x - self.points[-1].x,
                                  self.points[0].y - self.points[-1].y)
            first, last = self.points[1], self.points[-2]
            first_yaw = math.atan2(first.y - self.points[0].y, first.x - self.points[0].x)
            last_yaw = math.atan2(self.points[-1].y - last.y, self.points[-1].x - last.x)
            seam_yaw = math.atan2(math.sin(first_yaw - last_yaw), math.cos(first_yaw - last_yaw))
            if seam_gap > 1.0 or abs(seam_yaw) > math.radians(20):
                raise ValueError("loop_route requires connected endpoints and matching heading")
        self.timeout = float(rospy.get_param("~input_timeout_sec", 0.5))
        self.signal_timeout = float(rospy.get_param("~signal_timeout_sec", 0.8))
        self.vehicle_speed_topic = str(rospy.get_param("~vehicle_speed_topic", ""))
        self.require_vehicle_speed = bool(rospy.get_param("~require_vehicle_speed", bool(self.vehicle_speed_topic)))
        self.vehicle_speed_timeout = float(rospy.get_param("~vehicle_speed_timeout_sec", 0.5))
        if self.require_vehicle_speed and not self.vehicle_speed_topic:
            raise ValueError("Required vehicle speed needs vehicle_speed_topic")
        self.vehicle_speed_source = VehicleSpeedSource(timeout_sec=self.vehicle_speed_timeout)
        self.vehicle_speed_mps = self.odometry_speed_mps = None
        self.preview_m = float(rospy.get_param("~route_preview_m", 30.0))
        self.turn_threshold = float(rospy.get_param("~turn_threshold_deg", 25.0))
        self.max_route_error = float(rospy.get_param("~max_route_error_m", 3.0))
        self.highway_route_enabled = bool(rospy.get_param("~highway_route_enabled", False))
        self.highway_route_start = float(rospy.get_param("~highway_route_start_s_m", 1118.742))
        self.highway_route_end = float(rospy.get_param("~highway_route_end_s_m", 1741.721))
        self.highway_path_sample = self.highway_state_sample = None
        self.frenet_route_enabled = bool(rospy.get_param("~frenet_route_enabled", True))
        self.frenet_trajectory_sample = self.executed_path_sample = None
        self.frenet_route_accepted = False
        self.tunnel_route_accepted = False
        self.tunnel_plan_samples = []
        self.tunnel_mode_sample = None
        self.route_rejection_reason = "not_initialized"
        self.route_global_error_m = None
        self.highway_route_accepted = False

        self.route_stopline_match_tolerance = float(rospy.get_param(
            "~route_stopline_match_tolerance_m", 4.0))
        self.green_release_max_sec = float(rospy.get_param("~green_release_max_sec", 10.0))
        self.green_release_max_m = float(rospy.get_param("~green_release_max_m", 55.0))
        self.green_release_start_distance_m = float(rospy.get_param(
            "~green_release_start_distance_m", 45.0))
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.timeout, self.signal_timeout, self.preview_m, self.turn_threshold,
                    self.max_route_error, self.route_stopline_match_tolerance,
                    self.green_release_max_sec, self.green_release_max_m,
                    self.green_release_start_distance_m)):
            raise ValueError("Fusion distances, thresholds and timeouts must be positive and finite")
        self.require_quality = bool(rospy.get_param("~require_sensor_quality", True))
        self.require_safety = bool(rospy.get_param("~require_fresh_safety", True))
        self.require_camera_stream = bool(rospy.get_param("~require_fresh_camera_stream", True))
        self.right_on_green = bool(rospy.get_param("~right_on_green", True))
        # Strict mode never uses screen-centre/route-curvature as a signal ID.
        self.require_context = bool(rospy.get_param("~require_route_signal_context", True))
        # Sensor-only mode requires a current, explicit signal observation and
        # a stop-line observation together. A painted line by itself is not a
        # stop command at an unsignalized junction.
        self.stopline_requires_detected_signal = bool(
            rospy.get_param("~stopline_requires_detected_signal", False))
        self.require_reference_path = bool(rospy.get_param("~require_reference_path_match", self.require_context))
        self.reference_path_match = False
        self.reference_path_reason = "reference_path_not_received"
        self.allow_blackout_lane = bool(rospy.get_param("~allow_blackout_lane_corridor", False))
        self.blackout_max_duration = float(rospy.get_param("~blackout_max_duration_sec", 15.0))
        self.blackout_max_distance = float(rospy.get_param("~blackout_max_distance_m", 30.0))
        if not all(finite(v) and v > 0 for v in (self.blackout_max_duration, self.blackout_max_distance)):
            raise ValueError("Blackout corridor budgets must be positive and finite")
        self.corridor_anchor = None
        self.corridor_distance = 0.0
        self.corridor_last = None
        self.corridor_fault = False
        self.corridor_reason = "disabled"
        self.camera = rospy.get_param("~signal_camera", {})
        self.contexts, self.map_heads = [], []
        self.context_load_error = ""
        mgeo_dir = rospy.get_param("~signal_mgeo_path", "")
        if self.require_context and mgeo_dir:
            try:
                self.contexts = load_route_contexts(mgeo_dir, self.points, self.s_values)
                with (Path(mgeo_dir) / "traffic_light_set.json").open(encoding="utf-8-sig") as stream:
                    self.map_heads = [dict(id=str(h["idx"]), x=h["point"][0], y=h["point"][1], z=h["point"][2],
                                             **({"facing_yaw_deg": h["heading"]} if h.get("heading") is not None else {}))
                                      for h in json.load(stream) if h.get("type") == "car"]
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                self.contexts, self.map_heads = [], []
                self.context_load_error = str(exc)
                rospy.logerr("Signal map unavailable; fusion will brake: %s", exc)
        # The curvature profile can use mapped stop-line locations and route
        # directions without claiming an uncalibrated CAM4 pixel projection.
        # Only explicitly selected route contexts are signal-controlled.
        context_ids = rospy.get_param("~signalized_context_ids", [])
        require_signalized = bool(rospy.get_param("~require_signalized_contexts", False))
        if (not isinstance(context_ids, list)
                or any(not isinstance(identifier, str) or not identifier for identifier in context_ids)
                or len(context_ids) != len(set(context_ids))):
            raise ValueError("signalized_context_ids must be a list of unique route context IDs")
        if context_ids and (self.require_context or not mgeo_dir):
            raise ValueError("Route-scoped camera mode requires signal_mgeo_path and no projection mode")
        if require_signalized and not context_ids:
            raise ValueError("Required signalized route context list is empty")
        self.route_signal_contexts = []
        correlated_signal_heads = rospy.get_param("~correlated_signal_heads", {})
        if not isinstance(correlated_signal_heads, dict):
            raise ValueError("correlated_signal_heads must map context IDs to two head IDs")
        if context_ids:
            mapped = {item["id"]: item for item in load_route_contexts(
                mgeo_dir, self.points, self.s_values)}
            missing = set(context_ids) - set(mapped)
            if missing:
                raise ValueError("Unknown signalized route contexts: " + ", ".join(sorted(missing)))
            self.route_signal_contexts = sorted(
                (mapped[identifier] for identifier in context_ids),
                key=lambda item: item["start"],
            )
            for context in self.route_signal_contexts:
                if not context.get("direction_geometry_verified"):
                    raise ValueError("MGeo junction direction disagrees with forward route geometry: "
                                     + context["id"] + " (" + context["direction"] + "/"
                                     + context.get("geometry_direction", "UNKNOWN") + ")")
                context["direction"] = context["geometry_direction"]
                context["direction_source"] = "matched_route_curvature"
            if any(item["stop_s"] is None or item["direction"] not in
                   ("STRAIGHT", "LEFT", "RIGHT") or not item["signal_ids"]
                   for item in self.route_signal_contexts):
                raise ValueError("Signalized contexts need a mapped stop line, direction and signal")
        contexts_by_id = {item["id"]: item for item in self.route_signal_contexts}
        for context_id, head_ids in correlated_signal_heads.items():
            if (not isinstance(context_id, str) or context_id not in contexts_by_id
                    or not isinstance(head_ids, list) or len(head_ids) != 2
                    or any(not isinstance(identifier, str) or not identifier for identifier in head_ids)
                    or len(set(head_ids)) != 2
                    or set(head_ids) != set(contexts_by_id[context_id]["signal_ids"])):
                raise ValueError("Correlated heads must be exactly the two mapped heads of a selected context: "
                                 + str(context_id))
        if correlated_signal_heads:
            with (Path(mgeo_dir) / "intersection_controller_set.json").open(encoding="utf-8-sig") as stream:
                controller_groups = [set(phase) for controller in json.load(stream)
                                     for phase in controller.get("TL", [])]
            for context_id, head_ids in correlated_signal_heads.items():
                if not any(set(head_ids) <= group for group in controller_groups):
                    raise ValueError("Correlated heads do not share an MGeo controller phase: "
                                     + context_id)
        self.correlated_signal_heads = correlated_signal_heads
        self.route_camera_calibrated = bool(
            self.route_signal_contexts and isinstance(self.camera, dict)
            and self.camera.get("calibrated") is True)
        self.require_calibrated_route_camera = bool(rospy.get_param("~require_calibrated_route_camera", False))
        if self.require_calibrated_route_camera and not self.route_camera_calibrated:
            raise ValueError("Required mapped-route CAM4 calibration is missing; install the matching config")
        if self.route_camera_calibrated:
            with (Path(mgeo_dir) / "traffic_light_set.json").open(encoding="utf-8-sig") as stream:
                self.map_heads = [dict(id=str(h["idx"]), x=h["point"][0], y=h["point"][1], z=h["point"][2],
                                             **({"facing_yaw_deg": h["heading"]} if h.get("heading") is not None else {}))
                                  for h in json.load(stream) if h.get("type") == "car"]
        self.pose_history = deque(maxlen=100)
        self.confirmation = SignalConfirmation()
        self.selection = Selection("UNKNOWN", 0.0, False, None, "route_context_unavailable")
        self.selection_stamp = 0.0
        self.selection_context = None
        # Keep callback order until the control tick can associate every image.
        # Latest-only caching can erase a RED between two permissive images.
        self.signal_observations = deque()
        self.route_object_observations = deque(maxlen=32)
        self.signal_queue_limit = 32
        self.signal_queue_fault = None
        self.signal_queue_overflows = 0
        self.next_guard_id = None
        self.next_guard_core = None
        # Explicit simulator-only opt-out when the MORAI build has no lamp API.
        # This must never disable route, signal, stop-line or sensor checks.
        self.test_without_turn_signals = bool(rospy.get_param("~test_without_turn_signals", False))
        self.lamp_enabled = (bool(rospy.get_param("~lamp_output_enabled", False))
                             and not self.test_without_turn_signals)
        if self.test_without_turn_signals:
            rospy.logwarn("Simulator test mode: turn-signal UDP and indicator lead requirement disabled")
        self.remote = (rospy.get_param("~remote_ip", "192.168.0.147"),
                       int(rospy.get_param("~remote_port", 9097)))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.lamp_enabled else None
        self.lead = IndicatorLead(float(rospy.get_param("~lead_time_sec", 5.0)))
        self.core_params = dict(
            front_reference_offset_m=float(rospy.get_param("~front_reference_offset_m", 3.845)),
            hold_distance_m=float(rospy.get_param("~hold_distance_m", 4.0)),
            max_decel_mps2=float(rospy.get_param("~max_decel_mps2", 1.5)),
            planning_decel_mps2=float(rospy.get_param("~planning_decel_mps2", 1.0)),
            brake_ramp_distance_m=float(rospy.get_param("~brake_ramp_distance_m", 0.0)),
            settle_distance_m=float(rospy.get_param("~settle_distance_m", 0.0)),
            stop_tolerance_m=float(rospy.get_param("~stop_tolerance_m", 0.03)),
            signal_timeout_sec=self.signal_timeout,
        )
        self.core = StopLineControllerCore(**self.core_params)
        turn_defaults = dict(max_heading_error_deg=60., max_lateral_error_m=1.5,
                             exit_heading_error_deg=15., exit_lateral_error_m=.75,
                             exit_overrun_m=3.)
        for old_parameter in ("turn_left_speed_kph", "turn_right_speed_kph", "turn_lateral_accel_mps2"):
            if rospy.get_param("~" + old_parameter, None) is not None:
                rospy.logwarn("%s is obsolete and ignored; use shared max_speed_kph and lateral_accel_limit_mps2",
                              old_parameter)
        self.turn_motion = TurnMotionPlanner(self.points, self.s_values,
            **{key: float(rospy.get_param("~turn_" + key, value)) for key, value in turn_defaults.items()},
            max_speed_kph=float(rospy.get_param("~max_speed_kph", 7.2)),
            lateral_accel_limit_mps2=float(rospy.get_param("~lateral_accel_limit_mps2", 1.0)),
            planning_decel_mps2=self.core.planning_decel_mps2,
            max_decel_mps2=self.core.max_decel_mps2,
            reaction_time_sec=self.core.reaction_time_sec)
        self.speed_governor_enabled = bool(rospy.get_param("~speed_governor_enabled", False))
        self.mapped_approach_speed_kph = float(rospy.get_param(
            "~mapped_stopline_approach_speed_kph", 30.0))
        self.mapped_approach_distance_m = float(rospy.get_param(
            "~mapped_stopline_approach_distance_m", 80.0))
        self.mapped_cap_release_after_m = float(rospy.get_param(
            "~mapped_stopline_cap_release_after_m", 5.0))
        if (not all(finite(value) and value > 0 for value in
                    (self.mapped_approach_speed_kph, self.mapped_approach_distance_m))
                or not finite(self.mapped_cap_release_after_m)
                or self.mapped_cap_release_after_m < 0):
            raise ValueError("Invalid mapped stopline speed governor settings")
        self.governor_target_kph = None
        self.governor_stopline_distance_m = None
        # Stopping clearance uses the bumper; signal entry uses the front axle.
        # Both offsets are measured from the same rear-axle base_link origin.
        self.front_axle_offset_m = float(rospy.get_param("~front_axle_offset_m", 3.0))
        if (not finite(self.front_axle_offset_m)
                or not 0 <= self.front_axle_offset_m <= self.core.front_reference_offset_m):
            raise ValueError("front_axle_offset_m must be finite and between zero and the bumper offset")
        self.entry_dropout_grace_sec = float(rospy.get_param("~entry_signal_dropout_grace_sec", 0.6))
        if not finite(self.entry_dropout_grace_sec) or not 0 < self.entry_dropout_grace_sec <= 1.0:
            raise ValueError("entry_signal_dropout_grace_sec must be in (0, 1]")
        self.accel_limiter = AccelRiseLimiter(
            float(rospy.get_param("~accel_rise_rate_per_sec", 0.5)), self.core.max_update_gap_sec)
        self.manual = parse_maneuvers(rospy.get_param("~maneuvers", []))
        if self.route_signal_contexts and self.manual:
            raise ValueError("Route-scoped camera mode cannot mix manual maneuvers")
        if any(m.end_s_m is None or m.end_s_m <= m.start_s_m for m in self.manual):
            raise ValueError("Fusion route events require end_s_m > start_s_m")
        if any(m.kind not in ("turn", "lane_change") or m.end_s_m > self.s_values[-1] for m in self.manual):
            raise ValueError("Maneuver kind must be turn/lane_change and endpoints must be on the route")
        if len({m.identifier for m in self.manual}) != len(self.manual):
            raise ValueError("Duplicate maneuver id")
        if any(a.end_s_m > b.start_s_m for a, b in zip(self.manual, self.manual[1:])):
            raise ValueError("Overlapping route maneuvers")
        if self.require_context:
            self.manual = route_checked_maneuvers(self.manual, self.contexts)
        self.samples, self.last_stamps = {}, {}
        self.nominal = None
        self.nominal_at = None
        self.progress = self.segment = self.previous_ros = None
        self.startup_inside_contexts = set()
        self.last_projection = None
        self.event = None
        self.completed = set()
        self.ignore_line_until_s = -1.0
        self.last_observed_line_stamp = 0.0
        self.enter_permission = False
        self.entry_ticket = None
        self.entry_grant = None
        self.green_release = None
        self.merge_gap = None
        self.merge_gap_at = None
        self.dynamic_zone_guard = None
        self.dynamic_zone_sample = None
        self.dynamic_zone_status = None
        if bool(rospy.get_param("~dynamic_zone_enabled", False)):
            self.dynamic_zone_guard = DynamicZoneGuard(
                load_path_file(rospy.get_param("~path_file")), rospy.get_param("~avoidance_regions"),
                expected_fingerprint=rospy.get_param("~avoidance_route_fingerprint", ""),
                speed_kph=float(rospy.get_param("~dynamic_zone_speed_kph", 20.)),
                approach_m=float(rospy.get_param("~dynamic_zone_approach_m", 40.)),
                lookahead_m=float(rospy.get_param("~dynamic_zone_lookahead_m", 30.)),
                vehicle_width_m=float(rospy.get_param("~dynamic_zone_vehicle_width_m", 1.892)),
                margin_m=float(rospy.get_param("~dynamic_zone_margin_m", .3)),
                clearance_extra_m=float(rospy.get_param("~dynamic_zone_clearance_extra_m", .3)),
                clear_confirm_s=float(rospy.get_param("~dynamic_zone_clear_confirm_s", .6)),
                timeout_s=float(rospy.get_param("~dynamic_zone_input_timeout_s", .6)),
                prediction_s=float(rospy.get_param("~dynamic_zone_prediction_s", 2.)),
                ego_front_m=float(rospy.get_param("~front_reference_offset_m", 3.845)))
            rospy.Subscriber("/integration/avoidance/obstacle_states", String,
                             self.dynamic_obstacles_callback, queue_size=1)
        self.state_pub = rospy.Publisher("/control/maneuver_status", String, queue_size=1)
        self.lamp_pub = rospy.Publisher("/control/turn_signal_state", String, queue_size=1)
        self.output_pub = rospy.Publisher(rospy.get_param("~output_command_topic", "/control/maneuver_cmd"), CtrlCmd, queue_size=1)
        for key, topic, message_type in (
            ("odom", rospy.get_param("~odometry_topic", "/localization/odometry"), Odometry),
            ("signal", rospy.get_param("~signal_topic", "/perception/traffic_light/directional_state"), TrafficLight),
            ("line", rospy.get_param("~stopline_topic", "/perception/camera/stopline"), StopLineDetection),
            ("safety", rospy.get_param("~safety_topic", "/detection/fused_safety_stop"), SafetyStop),
            ("quality", "/localization/sensor_quality", SensorQuality),
            ("objects", rospy.get_param("~signal_objects_topic", "/detection/traffic_light"), ObjectInfoArray),
        ):
            rospy.Subscriber(topic, message_type, self.observe, callback_args=key,
                             queue_size=self.signal_queue_limit if key in ("signal", "objects") else 1)
        rospy.Subscriber("/integration/avoidance/trajectory", String, self.frenet_trajectory_callback, queue_size=1)
        rospy.Subscriber("/integration/avoidance/status", String, self.tunnel_mode_callback, queue_size=1)
        rospy.Subscriber("/experimental/active_reference_path", RosPath, self.executed_path_callback, queue_size=1)
        if self.highway_route_enabled:
            rospy.Subscriber("/highway_lane_strategy/active_path", RosPath, self.highway_path_callback, queue_size=1)
            rospy.Subscriber("/highway_lane_strategy/state", String, self.highway_state_callback, queue_size=1)
        if self.vehicle_speed_topic:
            rospy.Subscriber(self.vehicle_speed_topic, EgoVehicleStatus, self.vehicle_speed_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~nominal_command_topic", "/control/camera_fallback_cmd"),
                         CtrlCmd, self.command_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~reference_path_topic", "/experimental/curvature_reference_path"),
                         RosPath, self.reference_path_callback, queue_size=1)
        rospy.Subscriber("/morai/lidar/merge_gap/results", String, self.merge_gap_callback, queue_size=1)
        if self.allow_blackout_lane:
            rospy.Subscriber("/stability/camera_fallback_status", String, self.fallback_status_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.tick)

    def dynamic_obstacles_callback(self, message):
        try:
            payload=json.loads(message.data)
            if not isinstance(payload,dict):raise ValueError("not an object")
        except (ValueError,TypeError):
            payload={"timestamp":rospy.get_time(),"obstacles":None}
        with self.lock:
            self.dynamic_zone_sample=(payload,time.monotonic())

    def dynamic_constraint(self, odom, now, ros_now):
        guard=getattr(self,"dynamic_zone_guard",None)
        if guard is None:return None
        pose=None
        if odom is not None and odom.value.header.frame_id=="map":
            p=odom.value.pose.pose.position
            pose=(float(p.x),float(p.y))
        measured=(self.vehicle_speed_mps if self.vehicle_speed_mps is not None
                  else self.odometry_speed_mps)
        def current_path():
            sample=self.executed_path_sample
            if sample is None:return None
            path,received=sample
            if (0<=now-received<=self.timeout and path.header.frame_id=="map"
                    and -.05<=ros_now-path.header.stamp.to_sec()<=self.timeout
                    and len(path.poses)<=20000):
                points=[(float(q.pose.position.x),float(q.pose.position.y)) for q in path.poses]
                return points if all(math.isfinite(v) for p in points for v in p) else None
            return None
        return guard.step(now,ros_now,pose,measured,self.dynamic_zone_sample,current_path)

    def highway_path_callback(self, message):
        with self.lock:
            self.highway_path_sample = (message, time.monotonic())

    def highway_state_callback(self, message):
        try:
            state = json.loads(message.data)
            if not isinstance(state, dict):
                return
        except (ValueError, TypeError):
            return
        with self.lock:
            self.highway_state_sample = (state, time.monotonic())

    def highway_route_matches(self, projection, pose, now, ros_now):
        if (not self.highway_route_enabled or self.event is not None
                or not self.highway_route_start <= projection.progress_s <= self.highway_route_end
                or not math.isfinite(projection.distance_m) or projection.distance_m > 11.0
                or self.highway_path_sample is None or self.highway_state_sample is None):
            return False
        path, path_at = self.highway_path_sample
        state, state_at = self.highway_state_sample
        try:
            if (not 0 <= now-path_at <= self.timeout or not 0 <= now-state_at <= self.timeout
                    or not -0.05 <= ros_now-path.header.stamp.to_sec() <= self.timeout
                    or not -0.05 <= ros_now-float(state.get("timestamp", float("nan"))) <= self.timeout
                    or state.get("active") is not True or state.get("stop") is not False
                    or state.get("state") not in ("LANE_CHANGE", "INNER_HOLD", "REJOIN")
                    or path.header.frame_id != "map"):
                return False
            q = pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            return corridor_matches(points, pose.position.x, pose.position.y, yaw)
        except (ValueError, TypeError, AttributeError):
            return False

    def tunnel_mode_callback(self, message):
        # Tunnel mode is independent of a successful avoidance candidate.
        # In particular NORMAL/reference fallback must not revoke the mode.
        now, ros_now = time.monotonic(), rospy.get_time()
        try:
            data = json.loads(message.data)
            stamp = float(data["timestamp"])
            mode = data["tunnel_mode"]
            if (data.get("version") != "TUNNEL_MOTION_V3"
                    or type(mode) is not bool or not finite(stamp)
                    or not -0.05 <= ros_now-stamp <= self.timeout):
                return
        except (ValueError, TypeError, KeyError, OverflowError):
            return
        with self.lock:
            old = getattr(self, "tunnel_mode_sample", None)
            if old is None or stamp > old[0]:
                self.tunnel_mode_sample = (stamp, now, mode)

    def tunnel_mode_active(self, now, ros_now):
        sample = getattr(self, "tunnel_mode_sample", None)
        return bool(sample is not None and sample[2]
                    and 0 <= now-sample[1] <= self.timeout
                    and -0.05 <= ros_now-sample[0] <= self.timeout)

    def frenet_trajectory_callback(self, message):
        tunnel_sample = None
        try:
            data = json.loads(message.data)
            path, stop, speed, reason = read_trajectory(
                message.data, "map", rospy.get_time(), self.timeout)
            if (data.get("tunnel_mode") is True and not stop
                    and data.get("version") == "TUNNEL_MOTION_V3"
                    and data.get("state") in ("NORMAL", "AVOIDANCE", "RETURN")
                    and reason in ("v61_reference", "frenet_avoidance", "frenet_return")):
                tunnel_sample = (path, time.monotonic(), data.get("active") is True)
            if (data.get("active") is not True or stop
                    or data.get("state") not in ("AVOIDANCE", "RETURN")
                    or reason not in ("frenet_avoidance", "frenet_return")):
                raise ValueError("not an active avoidance plan")
            sample = (path, time.monotonic())
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            sample = None
        with self.lock:
            self.frenet_trajectory_sample = sample
            # A plan and the controller's executed path arrive on independent
            # topics. Keep recent candidates to tolerate that one-tick ordering.
            recent = getattr(self, "tunnel_plan_samples", [])
            self.tunnel_plan_samples = ([tunnel_sample]+recent)[:8] if tunnel_sample else []

    def tunnel_route_matches(self, pose, now, ros_now, avoidance_only=False):
        if (self.event is not None or self.executed_path_sample is None
                or self.highway_state_sample is None):
            return False
        executed, executed_at = self.executed_path_sample
        state, state_at = self.highway_state_sample
        if (not 0 <= now-executed_at <= self.timeout
                or not 0 <= now-state_at <= self.timeout
                or not -0.05 <= ros_now-executed.header.stamp.to_sec() <= self.timeout
                or not -0.05 <= ros_now-float(state.get("timestamp", 0)) <= self.timeout
                or state.get("active") is not False or state.get("stop") is not False
                or state.get("state") not in ("OFF", "DONE")
                or executed.header.frame_id != "map"):
            return False
        for path, received, active in getattr(self, "tunnel_plan_samples", []):
            if (avoidance_only and not active) or not 0 <= now-received <= self.timeout:
                continue
            if (not -0.05 <= ros_now-path.header.stamp.to_sec() <= self.timeout
                    or len(path.poses) != len(executed.poses)):
                continue
            if any(abs(getattr(a.pose.position, axis)-getattr(b.pose.position, axis)) > 1e-6
                   or not finite(getattr(b.pose.position, axis))
                   for a, b in zip(path.poses, executed.poses) for axis in ("x", "y", "z")):
                continue
            q = pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            if corridor_matches(points, pose.position.x, pose.position.y, yaw):
                return True
        return False

    def tunnel_obstacle_stop_overridden(self, safety, odom, now, ros_now):
        # Only a simple forward-ROI obstacle stop can yield to a current detour.
        # Pedestrians, signal stops, stale inputs and missing plans still apply.
        return bool(safety is not None and safety.value.stop_required
                    and safety.value.reason == "roi_lidar_forward_obstacle"
                    and odom is not None and self.tunnel_route_accepted
                    and self.tunnel_route_matches(odom.value.pose.pose, now, ros_now, True))

    def executed_path_callback(self, message):
        with self.lock:
            self.executed_path_sample = (message, time.monotonic())

    def frenet_route_matches(self, projection, pose, now, ros_now):
        # A valid detour changes geometry, never traffic-light permission.
        if (not self.frenet_route_enabled or self.event is not None
                or not math.isfinite(projection.distance_m) or projection.distance_m > 8.0
                or self.frenet_trajectory_sample is None or self.executed_path_sample is None
                or self.highway_state_sample is None):
            return False
        path, path_at = self.frenet_trajectory_sample
        executed, executed_at = self.executed_path_sample
        state, state_at = self.highway_state_sample
        try:
            if (not all(0 <= now-t <= self.timeout for t in (path_at, executed_at, state_at))
                    or not all(-0.05 <= ros_now-t <= self.timeout for t in
                        (path.header.stamp.to_sec(), executed.header.stamp.to_sec(), float(state['timestamp'])))
                    or state.get("state") not in ("OFF", "DONE")
                    or state.get("active") is not False or state.get("stop") is not False
                    or path.header.frame_id != "map" or executed.header.frame_id != "map"
                    or len(path.poses) != len(executed.poses)):
                return False
            # Compare actual controller geometry, not separate boolean permits.
            for a, b in zip(path.poses, executed.poses):
                for axis in ("x", "y", "z"):
                    av, bv = float(getattr(a.pose.position, axis)), float(getattr(b.pose.position, axis))
                    if not math.isfinite(av) or not math.isfinite(bv) or abs(av-bv) > 1e-6:
                        return False
            q = pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            return corridor_matches(points, pose.position.x, pose.position.y, yaw)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return False

    def vehicle_speed_callback(self, message):
        # morai_udp_drive_bridge converts the UDP km/h velocity into ROS m/s.
        velocity = message.velocity
        speed = math.hypot(float(velocity.x), float(velocity.y))
        with self.lock:
            self.vehicle_speed_source.observe(speed, message.header.stamp.to_sec(),
                                              rospy.get_time(), time.monotonic())

    def observe(self, message, key):
        stamp, now, ros_now = message.header.stamp.to_sec(), time.monotonic(), rospy.get_time()
        sample = Sample(stamp, now, message)
        with self.lock:
            timeout = self.signal_timeout if key in ("signal", "line", "objects") else self.timeout
            ordered_key = "objects" if self.require_context or self.route_camera_calibrated else "signal"
            previous = self.samples.get(key)
            if (key == ordered_key and previous is not None and stamp == previous.stamp
                    and sample.fresh(ros_now, now, timeout)
                    and self.signal_evidence(message, key) != self.signal_evidence(previous.value, key)):
                # Same source frame cannot simultaneously prove RED and GREEN.
                self.signal_observations.clear()
                self.signal_queue_fault = "signal_duplicate_conflict"
                self.deny_entry_permission()
                return
            if sample.fresh(ros_now, now, timeout) and stamp > self.last_stamps.get(key, 0.0):
                self.samples[key] = sample
                self.last_stamps[key] = stamp
                if key == "odom":
                    self.pose_history.append(sample)
                if key == "objects" and self.route_signal_contexts:
                    self.route_object_observations.append(sample)
                if key == ordered_key:
                    if len(self.signal_observations) >= self.signal_queue_limit:
                        self.signal_observations.clear()
                        self.signal_queue_fault = "signal_queue_overflow"
                        self.signal_queue_overflows += 1
                        self.deny_entry_permission()
                    self.signal_observations.append(sample)

    @staticmethod
    def signal_evidence(message, key):
        """Compare detection evidence, not Python identity or a transport seq."""
        if key == "signal":
            return message.state, message.confidence, message.valid
        fields = ("class_name", "conf", "x_center", "y_center", "width", "height")
        return (message.header.frame_id,
                tuple(tuple(getattr(obj, field, None) for field in fields) for obj in message.objects))

    def reference_path_callback(self, message):
        with self.lock:
            self.reference_path_reason = reference_path_reason(self.points, message)
            self.reference_path_match = self.reference_path_reason == "matched"
            if self.require_reference_path and not self.reference_path_match:
                # Revoke in the callback, even if a matching path returns before
                # the next tick. Old signal evidence must not revive permission.
                self.enter_permission = False
                self.entry_ticket = None
                self.entry_grant = None
                self.core.revoke_permission()
                self.confirmation.reset()
                self.selection = Selection(reason=self.reference_path_reason)
                self.selection_stamp = max(self.selection_stamp, rospy.get_time())
                self.signal_observations.clear()
                self.lead.reset()
                self.corridor_anchor = self.corridor_last = None

    def command_callback(self, message):
        with self.lock:
            self.nominal, self.nominal_at = copy.deepcopy(message), time.monotonic()

    def merge_gap_callback(self, message):
        with self.lock:
            try:
                payload = json.loads(message.data)
                if not isinstance(payload, dict):
                    raise ValueError("gap result must be an object")
                self.merge_gap = payload
                self.merge_gap_at = time.monotonic()
            except (ValueError, TypeError):
                self.merge_gap = self.merge_gap_at = None

    def fallback_status_callback(self, message):
        with self.lock:
            now, ros_now = time.monotonic(), rospy.get_time()
            try:
                data = json.loads(message.data)
                stamp = float(data["stamp"])
                if not isinstance(data, dict) or not isinstance(data.get("mode"), str):
                    raise ValueError("invalid fallback status")
                sample = Sample(stamp, now, data)
                if sample.fresh(ros_now, now, self.timeout) and stamp > self.last_stamps.get("fallback", 0.):
                    self.samples["fallback"] = sample
                    self.last_stamps["fallback"] = stamp
            except (ValueError, KeyError, TypeError):
                self.samples.pop("fallback", None)

    def blackout_corridor(self, quality, fallback, speed, route_valid, line, objects, now, ros_now):
        """Bounded lane keeping only outside any route decision/visible signal.

        This is not permission to cross a junction with bad GPS. It only keeps
        a recently verified, map-aligned approach corridor alive at low speed.
        """
        self.corridor_reason = "disabled"
        if not self.allow_blackout_lane:
            return False
        if self.require_reference_path and not self.reference_path_match:
            self.corridor_reason = self.reference_path_reason
            return False
        if route_valid:
            self.corridor_anchor = (now, ros_now, self.progress)
            self.corridor_last = (now, ros_now)
            self.corridor_distance = 0.
            self.corridor_fault = False
            self.corridor_reason = "normal_route"
            return False
        self.corridor_reason = "no_verified_normal_anchor"
        if self.corridor_anchor is None or quality is None or fallback is None or speed is None:
            return False
        q, f = quality.value, fallback.value
        if self.corridor_last is not None:
            dt, wall_dt = ros_now - self.corridor_last[1], now - self.corridor_last[0]
            self.corridor_fault |= not (0 <= dt <= self.timeout and 0 <= wall_dt <= self.timeout)
            self.corridor_distance += speed * max(0., dt)
        self.corridor_last = (now, ros_now)
        elapsed = max(now - self.corridor_anchor[0], ros_now - self.corridor_anchor[1])
        self.corridor_fault |= (elapsed > self.blackout_max_duration
                                or self.corridor_distance > self.blackout_max_distance
                                or abs(self.progress - self.corridor_anchor[2]) > self.blackout_max_distance)
        self.corridor_reason = "blackout_corridor_budget_or_clock"
        if self.corridor_fault:
            return False
        self.corridor_reason = "fallback_not_verified"
        modes = ("gps_blackout_camera_fallback", "lane_loss_braking", "recovery_camera", "recovery_blend")
        if (q.state not in ("GPS_BLACKOUT", "NORMAL") or getattr(q, "imu_stale", True)
                or (q.state == "GPS_BLACKOUT" and not getattr(q, "gps_blackout", False))
                or (q.state == "NORMAL" and (not getattr(q, "gps_valid", False)
                    or getattr(q, "gps_blackout", True) or getattr(q, "gps_recovering", True)))
                or f.get("mode") not in modes or f.get("nominal_fresh") is not True
                or f.get("blackout_budget_exhausted") is not False
                or not all(finite(f.get(k)) and 0 <= f[k] <= 1. for k in ("accel", "brake"))):
            return False
        if f["mode"] != "recovery_blend" and f.get("camera_used") is not True:
            return False
        if f["mode"] in ("gps_blackout_camera_fallback", "recovery_camera") and f.get("lane_primary_usable") is not True:
            return False
        if f["mode"] == "recovery_blend" and q.state != "NORMAL":
            return False
        if f["mode"] == "lane_loss_braking" and not (f.get("accel") == 0 and finite(f.get("brake"))
                                                      and 0 < f["brake"] <= 1.):
            return False
        self.corridor_reason = "junction_or_observation_guard"
        if (not self.require_context or not self.contexts or self.event is not None
                or line is None or objects is None or line.value.valid or objects.value.objects):
            return False
        horizon = max(40., speed * 6. + speed**2 / (2. * self.core.planning_decel_mps2))
        guards = list(self.contexts) + [dict(id=m.identifier, start=m.start_s_m, end=m.end_s_m) for m in self.manual]
        if any(c["id"] not in self.completed and c["end"] > self.progress
               and c["start"] - self.progress <= horizon for c in guards):
            return False
        self.corridor_reason = "bounded_lane_corridor"
        return True

    def lane_change_clear(self, direction, now):
        # Missing right-side results cannot be interpreted as a free right lane.
        return bool(self.merge_gap is not None and self.merge_gap_at is not None
                    and 0 <= now - self.merge_gap_at <= self.timeout
                    and self.merge_gap.get("valid") is True
                    and isinstance(self.merge_gap.get(direction.lower()), dict)
                    and self.merge_gap[direction.lower()].get("confirmed_available") is True)

    def fresh(self, key, now, ros_now):
        sample = self.samples.get(key)
        timeout = self.signal_timeout if key in ("signal", "line", "objects") else self.timeout
        return sample if sample is not None and sample.fresh(ros_now, now, timeout) else None

    def revoke_entry_permission(self):
        """A later green must not revive an approach-side entry ticket."""
        if self.event is not None and not self.event["committed"]:
            self.enter_permission = False
            self.entry_ticket = None
            self.green_release = None
            self.core.revoke_permission()

    def deny_entry_permission(self):
        """A fresh conflicting indication also cancels dropout tolerance."""
        self.entry_grant = None
        self.revoke_entry_permission()

    def route_release_available(self, event, now, ros_now):
        """Carry a confirmed green only while its mapped lamp leaves CAM4 view."""
        ticket = self.green_release
        return bool(self.route_camera_calibrated and event and ticket
                    and not event.get("committed")
                    and ticket[0] == route_event_key(event)
                    and self.progress is not None
                    and self.progress <= self.crossing_s(event) + 1.0
                    and 0 <= self.progress - ticket[1] <= self.green_release_max_m
                    and 0 <= now - ticket[2] <= self.green_release_max_sec
                    and 0 <= ros_now - ticket[3] <= self.green_release_max_sec)

    def route_green_visibility_loss(self, selection, objects, event, now, ros_now):
        """Keep an earlier mapped GREEN through a bounded detection dropout.

        A fresh unmatched frame is not a new green vote. An observed stop,
        unknown indication, or ambiguous mapped head cancels this carry.
        """
        if (selection.reason not in ("no_match", "no_projected_targets")
                or objects.stamp != self.selection_stamp
                or not self.route_release_available(event, now, ros_now)):
            return False
        if selection.reason == "no_match":
            for obj in objects.value.objects:
                try:
                    confidence = float(obj.conf)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return False
                if not finite(confidence) or confidence > 1.0:
                    return False
                if confidence >= 0.5:
                    state, _ = directional_observation((obj,))
                    if not signal_permits(state, event["direction"], self.right_on_green):
                        return False
        return True

    def suspend_entry_permission(self):
        """Withhold a crossing ticket while a CAM4 frame awaits its object pair.

        The paired previous GREEN remains historical confirmation, but cannot
        authorize movement until the newest source frame has been checked.
        """
        if self.event is not None and not self.event["committed"]:
            self.enter_permission = False
            self.entry_ticket = None

    def consume_legacy_signals(self, direction, ready, route_valid, now, ros_now):
        """Legacy mode also consumes every accepted callback, once and in order."""
        if self.event is not None and self.event["committed"]:
            self.signal_observations.clear()
            self.signal_queue_fault = None
            return  # Existing committed-turn and live-stream guards own this phase.
        if self.signal_queue_fault:
            self.signal_queue_fault = None
            self.signal_observations.clear()
            self.core.revoke_permission()
            self.revoke_entry_permission()
            return
        while self.signal_observations:
            sample = self.signal_observations.popleft()
            if not sample.fresh(ros_now, now, self.signal_timeout):
                self.core.revoke_permission()
                self.revoke_entry_permission()
                continue
            message = sample.value
            valid = (message.valid and finite(message.confidence)
                     and .5 <= message.confidence <= 1.)
            allowed = (valid and ready and route_valid
                       and signal_permits(message.state, direction, self.right_on_green))
            if not allowed:
                self.revoke_entry_permission()
            self.core.observe_signal("GREEN" if allowed else "RED", message.confidence,
                                     valid, sample.stamp, sample.received, ros_now)

    @staticmethod
    def valid_sensor_signal(sample):
        if sample is None:
            return False
        message = sample.value
        state = str(getattr(message, "state", "UNKNOWN")).upper()
        return bool(getattr(message, "valid", False)
                    and finite(getattr(message, "confidence", None))
                    and 0.5 <= message.confidence <= 1.0
                    and state in ("RED", "YELLOW", "GREEN", "LEFT", "RIGHT",
                                  "GREEN_LEFT", "GREEN_RIGHT", "RED_LEFT",
                                  "RED_RIGHT", "YELLOW_LEFT", "YELLOW_RIGHT"))

    @staticmethod
    def valid_sensor_stopline(sample):
        if sample is None:
            return False
        line = sample.value
        return bool(line.valid and line.header.frame_id == "base_link"
                    and finite(line.distance_m) and 0 <= line.distance_m <= 50.0
                    and finite(line.confidence) and 0.5 <= line.confidence <= 1.0)

    def valid_sensor_stop_pair(self, line, signal):
        return bool(self.valid_sensor_stopline(line) and self.valid_sensor_signal(signal)
                    and abs(line.stamp - signal.stamp) <= self.signal_timeout)

    def unique_route_signal(self, signal, objects):
        """Require one fresh CAM4 head in the same frame as the directional state.

        Pixel-to-map head association is unavailable without camera calibration.
        Multiple plausible heads therefore cannot grant route entry.
        """
        if (not self.valid_sensor_signal(signal) or objects is None
                or abs(signal.stamp - objects.stamp) > 0.05):
            return False
        state, _ = directional_observation(objects.value.objects)
        return (self.route_signal_candidate_count(objects) == 1
                and state == str(signal.value.state).upper())

    @staticmethod
    def route_signal_candidate_count(objects):
        return count_signal_housings(objects.value.objects) if objects is not None else 0

    def route_objects_at(self, stamp):
        return next((sample for sample in self.route_object_observations
                     if abs(sample.stamp - stamp) <= 1e-6), None)

    def constrain_route_speed(self, decision, speed, released_context_id=None):
        """Bound measured speed even if the nominal PI pedal or path profile lags."""
        self.governor_target_kph = self.governor_stopline_distance_m = None
        if not self.speed_governor_enabled or speed is None or self.progress is None:
            return decision
        target_kph = self.turn_motion.max_speed_kph
        for context in self.route_signal_contexts:
            distance = context["stop_s"] - self.progress
            if distance < -self.mapped_cap_release_after_m:
                continue
            if context["id"] == released_context_id:
                # A confirmed, direction-compatible green has released this
                # mapped line. Keep the normal curvature speed profile.
                continue
            if distance <= self.mapped_approach_distance_m:
                self.governor_stopline_distance_m = distance
                # Retain the approach cap when the mapped signal is red,
                # unknown, or has not yet passed source-frame confirmation.
                fraction = clamp((distance - 20.0) /
                                 max(self.mapped_approach_distance_m - 20.0, 1.0),
                                 0.0, 1.0)
                target_kph = min(target_kph,
                                 self.mapped_approach_speed_kph +
                                 max(0.0, target_kph - self.mapped_approach_speed_kph) * fraction)
            break
        self.governor_target_kph = target_kph
        target_mps = target_kph / 3.6
        error = target_mps - speed
        accel_limit = min(decision.accel_limit, clamp(error * 0.45, 0.0, 1.0))
        # The nominal controller is a request, not a guarantee. Overspeed
        # therefore adds a measured-speed brake, including outside junctions.
        brake = max(decision.brake, clamp((-error - 0.25) * 0.30, 0.0, 1.0))
        target = target_kph if decision.target_speed_kph is None else min(
            target_kph, decision.target_speed_kph)
        reason = "speed_governor" if decision.mode == "NOMINAL" and (
            accel_limit < decision.accel_limit or brake > decision.brake) else decision.reason
        return Decision(decision.mode, reason, accel_limit, brake,
                        target, decision.distance_m)

    def consume_route_signals(self, event, ready, route_valid, now, ros_now):
        """Consume every CAM4 frame in order; unmatched frames never grant green."""
        if self.route_camera_calibrated:
            self.consume_calibrated_route_signals(event, ready, route_valid, now, ros_now)
            return
        if event is None or event.get("committed"):
            self.signal_observations.clear()
            self.signal_queue_fault = None
            return
        if self.signal_queue_fault:
            self.signal_queue_fault = None
            self.signal_observations.clear()
            self.revoke_entry_permission()
            return
        while self.signal_observations:
            sample = self.signal_observations[0]
            objects = self.route_objects_at(sample.stamp)
            if objects is None and sample.fresh(ros_now, now, self.signal_timeout):
                self.suspend_entry_permission()
                return  # The matching ObjectInfoArray callback may arrive next.
            self.signal_observations.popleft()
            if (not sample.fresh(ros_now, now, self.signal_timeout)
                    or objects is None or not objects.fresh(ros_now, now, self.signal_timeout)
                    or not self.unique_route_signal(sample, objects)):
                self.revoke_entry_permission()
                continue
            allowed = bool(event["camera_line_confirmed"] and ready and route_valid
                           and signal_permits(sample.value.state, event["direction"],
                                              self.right_on_green))
            if not allowed:
                self.revoke_entry_permission()
            self.core.observe_signal("GREEN" if allowed else "RED",
                                     sample.value.confidence, True,
                                     sample.stamp, sample.received, ros_now)

    def consume_calibrated_route_signals(self, event, ready, route_valid, now, ros_now):
        """Associate every source image to this route's MGeo signal group.

        Global CAM4 aggregation can be UNKNOWN when another junction is also
        visible. Only projected heads controlling this matched route link vote;
        the core confirms allowed states over distinct original image stamps.
        """
        self.synchronize_route_intent(event)
        if event is None or event.get("committed"):
            self.signal_observations.clear()
            self.signal_queue_fault = None
            if event is None:
                self.selection = Selection(reason="no_active_route_junction")
            return
        objects = self.fresh("objects", now, ros_now)
        if not route_valid or objects is None:
            self.signal_observations.clear()
            self.selection = Selection(reason="route_context_unavailable" if not route_valid
                                       else "signal_observation_stale")
            self.revoke_entry_permission()
            return
        if self.signal_queue_fault:
            self.selection = Selection(reason=self.signal_queue_fault)
            self.signal_queue_fault = None
            self.signal_observations.clear()
            self.selection_stamp = objects.stamp
            self.deny_entry_permission()
            return
        if not self.signal_observations and objects.stamp > self.selection_stamp:
            self.signal_observations.append(objects)
        while self.signal_observations:
            sample = self.signal_observations[0]
            if sample.stamp <= self.selection_stamp:
                self.signal_observations.popleft()
                continue
            if not sample.fresh(ros_now, now, self.signal_timeout):
                self.signal_observations.popleft()
                self.selection_stamp = sample.stamp
                self.selection = Selection(reason="signal_observation_stale")
                self.revoke_entry_permission()
                continue
            chosen = self.associate_signal_frame(sample, event)
            if chosen is None:
                self.selection = Selection(reason="signal_pose_unsynchronized")
                # A short pose-pair wait is not a negative observation.
                # Preserve confirmation history, but the tick passes
                # permission_pending=True to withhold the braking release.
                self.suspend_entry_permission()
                return
            self.signal_observations.popleft()
            self.selection_stamp = sample.stamp
            self.selection = chosen
            if (ready and route_valid and self.route_green_visibility_loss(
                    chosen, sample, event, now, ros_now)):
                # The confirmed route signal can leave the calibrated gate
                # before the front axle crosses. An explicit stop indication
                # or ambiguous mapped head still revokes permission below.
                self.core.observe_signal("GREEN", 1.0, True,
                                         sample.stamp, sample.received, ros_now)
                continue
            allowed = bool(chosen.valid and ready
                           and route_valid and signal_permits(chosen.state, event["direction"],
                                                              self.right_on_green))
            if not allowed:
                if chosen.valid or chosen.reason not in ("no_match", "no_projected_targets"):
                    self.deny_entry_permission()
                else:
                    self.revoke_entry_permission()
            self.core.observe_signal("GREEN" if allowed else "RED", chosen.confidence,
                                     chosen.valid, sample.stamp, sample.received, ros_now)

    def consume_sensor_stop_pair(self, line, signal, direction, now, ros_now):
        """In map-free mode, use only a temporally paired line + known light.

        No stop-line-only event is passed to the braking core. Signals without
        a nearby measured stop line also cannot create a stop target.
        """
        self.signal_observations.clear()
        self.signal_queue_fault = None
        if not self.valid_sensor_stop_pair(line, signal):
            return
        state = str(signal.value.state).upper()
        permitted = signal_permits(state, direction, self.right_on_green)
        self.core.observe_line(line.value.distance_m, line.value.confidence, True,
                               line.stamp, line.received, ros_now)
        self.core.observe_signal("GREEN" if permitted else "RED", signal.value.confidence,
                                 True, signal.stamp, signal.received, ros_now)

    def synchronize_route_intent(self, event):
        """Reset signal evidence and indicator lead before evaluating new intent."""
        context_id = route_event_key(event)
        if context_id != self.selection_context:
            if self.selection_context is not None:
                self.revoke_entry_permission()
                self.lead.reset()
                self.signal_observations.clear()
                self.selection_stamp = max(self.selection_stamp, rospy.get_time())
            self.green_release = None
            self.entry_grant = None
            self.confirmation.reset()
            self.selection = Selection(reason="route_intent_changed")
            self.selection_context = context_id

    def selected_signal(self, objects, event, route_valid):
        """Consume ordered images at image-time poses; defer, never skip, a hole."""
        self.synchronize_route_intent(event)
        reason = None
        if not route_valid or not event or not event.get("signal_points"):
            reason = "route_context_unavailable"
        elif not isinstance(self.camera, dict) or self.camera.get("calibrated") is not True:
            reason = "signal_camera_uncalibrated"
        elif objects is None:
            reason = "signal_observation_stale"
        if reason:
            self.confirmation.reset()
            self.signal_observations.clear()
            self.revoke_entry_permission()
            self.selection = Selection("UNKNOWN", 0.0, False, None, reason)
            return self.selection
        if self.signal_queue_fault:
            fault, self.signal_queue_fault = self.signal_queue_fault, None
            self.signal_observations.clear()
            self.confirmation.reset()
            self.selection_stamp = objects.stamp
            self.revoke_entry_permission()
            self.selection = Selection(reason=fault)
            return self.selection
        # Reconsider a still-fresh cached frame after context acquisition. The
        # finalized watermark prevents counting an image twice on timer ticks.
        if not self.signal_observations and objects.stamp > self.selection_stamp:
            self.signal_observations.append(objects)
        now, ros_now = time.monotonic(), rospy.get_time()
        while self.signal_observations:
            observation = self.signal_observations[0]
            if observation.stamp <= self.selection_stamp:
                self.signal_observations.popleft()
                continue
            if not observation.fresh(ros_now, now, self.signal_timeout):
                self.signal_observations.popleft()
                self.selection_stamp = observation.stamp
                self.confirmation.reset()
                self.revoke_entry_permission()
                self.selection = Selection(reason="signal_observation_stale")
                continue
            selection = self.associate_signal_frame(observation, event)
            if selection is None:
                # Preserve the original stamp and receipt time. Later odometry
                # may complete this pair, but cannot refresh the image's age.
                self.confirmation.reset()
                self.revoke_entry_permission()
                self.selection = Selection(reason="signal_pose_unsynchronized")
                return self.selection
            self.signal_observations.popleft()
            self.selection_stamp = observation.stamp
            self.selection = self.confirmation.update(selection, observation.stamp)
            if not (self.selection.valid and signal_permits(
                    self.selection.state, event["direction"], self.right_on_green)):
                self.revoke_entry_permission()
        return self.selection

    def associate_signal_frame(self, objects, event):
        """None means await matching pose, not consume an UNKNOWN image."""
        # Closest timestamp must be within 50ms; no extrapolation across turns.
        pose_sample = min(self.pose_history, key=lambda p: abs(p.stamp - objects.stamp), default=None)
        if (pose_sample is None or pose_sample.value.header.frame_id != "map"
                or abs(pose_sample.stamp - objects.stamp) > 0.05):
            return None
        pose = pose_sample.value.pose.pose
        try:
            q = pose.orientation
            norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
                raise ValueError("invalid pose quaternion")
            roll = math.atan2(2*(q.w*q.x + q.y*q.z), 1-2*(q.x*q.x + q.y*q.y))
            pitch = math.asin(clamp(2*(q.w*q.y - q.z*q.x), -1.0, 1.0))
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
            ego = dict(x=pose.position.x, y=pose.position.y, z=pose.position.z,
                       yaw=yaw, roll=roll, pitch=pitch)
            targets = {h["id"]: project_signal(h, ego, self.camera) for h in event["signal_points"]}
            others = {h["id"]: project_signal(h, ego, self.camera) for h in self.map_heads
                      if h["id"] not in targets}
            selection = associate(objects.value.objects, targets, others,
                                  allow_target_group=event.get("source") == "route_camera",
                                  allow_correlated_pair=(event.get("source") == "route_camera"
                                                         and event.get("id") in self.correlated_signal_heads))
        except (ValueError, TypeError, AttributeError, KeyError):
            selection = Selection("UNKNOWN", 0.0, False, None, "signal_projection_invalid")
        return selection

    def projected_route_heads(self, objects, event):
        """Expose map-head pixel projections for visual calibration only."""
        if objects is None or not event or not event.get("signal_points"):
            return {}
        pose_sample = min(self.pose_history, key=lambda p: abs(p.stamp - objects.stamp), default=None)
        if (pose_sample is None or pose_sample.value.header.frame_id != "map"
                or abs(pose_sample.stamp - objects.stamp) > 0.05):
            return {}
        pose = pose_sample.value.pose.pose
        try:
            q = pose.orientation
            norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
                return {}
            ego = dict(
                x=pose.position.x, y=pose.position.y, z=pose.position.z,
                yaw=math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z)),
                roll=math.atan2(2*(q.w*q.x + q.y*q.z), 1-2*(q.x*q.x + q.y*q.y)),
                pitch=math.asin(clamp(2*(q.w*q.y - q.z*q.x), -1.0, 1.0)),
            )
            result = {}
            for head in event["signal_points"]:
                pixel = project_signal(head, ego, self.camera)
                if pixel is not None:
                    result[str(head["id"])] = [round(pixel[0], 1), round(pixel[1], 1)]
            return result
        except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
            return {}

    def send_lamp(self, direction, now, ros_now):
        success = False
        if self.socket is not None:
            try:
                packet = build_lamp_packet(direction)
                success = self.socket.sendto(packet, self.remote) == len(packet)
            except OSError as exc:
                rospy.logerr_throttle(2.0, "LampControl send failed: %s", exc)
        self.lamp_requested, self.lamp_transmit_ok = direction, success
        self.lead.sent(direction, now, ros_now, success)
        # This is transmitted state, never a claim that MORAI acknowledged it.
        self.lamp_pub.publish(String(data=direction if success else "OFF"))

    def update_route(self, odom, now, ros_now):
        self.route_rejection_reason = "odometry_missing_or_wrong_frame"
        self.route_global_error_m = None
        self.highway_route_accepted = False
        self.frenet_route_accepted = False
        self.tunnel_route_accepted = False
        self.odometry_speed_mps = None
        self.vehicle_speed_mps = (self.vehicle_speed_source.sample(ros_now, now)
                                  if self.vehicle_speed_topic else None)
        if odom is None or odom.value.header.frame_id != "map":
            return None
        pose, velocity = odom.value.pose.pose, odom.value.twist.twist.linear
        x, y = float(pose.position.x), float(pose.position.y)
        odometry_speed = math.hypot(velocity.x, velocity.y)
        self.odometry_speed_mps = odometry_speed if finite(odometry_speed) else None
        if self.require_vehicle_speed and self.vehicle_speed_mps is None:
            self.route_rejection_reason = "vehicle_speed_unavailable"
            return None
        speed = (self.vehicle_speed_mps if self.vehicle_speed_mps is not None
                 else self.odometry_speed_mps)
        if not all(finite(v) for v in (x, y, speed)):
            return None
        projection = (nearest_projection_on_loop(self.points, self.s_values, x, y,
                                                self.segment, 250)
                      if self.loop_route else
                      nearest_projection(self.points, self.s_values, x, y,
                                         start_segment=max(0, (self.segment or 0) - 250),
                                         end_segment=None if self.segment is None else self.segment + 250))
        self.route_global_error_m = projection.distance_m
        if self.tunnel_mode_active(now, ros_now):
            # GPS blackout mode alone disables GLOBAL route offset/progress
            # rejection, including NORMAL/reference fallback after plan failure.
            # Do not require proximity to that same global path indirectly.
            # Freeze event progress; no traffic permission is earned by drift.
            self.tunnel_route_accepted = True
            self._tunnel_reanchor_pending = True
            self.last_projection = None
            self.route_rejection_reason = "tunnel_global_route_check_bypassed"
            return speed
        if projection.distance_m > self.max_route_error:
            self.highway_route_accepted = self.highway_route_matches(projection, pose, now, ros_now)
            self.frenet_route_accepted = self.frenet_route_matches(projection, pose, now, ros_now)
            if not (self.highway_route_accepted or self.frenet_route_accepted):
                self.route_rejection_reason = "global_route_offset_without_valid_mission_path"
                return None
        if getattr(self, "_tunnel_reanchor_pending", False) and self.event is None:
            # Re-enter global progress only after the ordinary map-corridor
            # check above passed. Preserve completed traffic events.
            self.progress = None
            self.last_projection = None
            self._tunnel_reanchor_pending = False
        self.route_rejection_reason = "progress_discontinuity"

        length = self.s_values[-1]
        wrapped = (self.loop_route and self.progress is not None
                   and self.progress > length - 10.0 and projection.progress_s < 10.0)
        reverse_seam = (self.loop_route and self.progress is not None
                        and self.progress < 10.0 and projection.progress_s > length - 10.0)
        if self.last_projection is not None:
            previous_s, previous_time, previous_speed = self.last_projection
            dt = ros_now - previous_time
            delta = (projection.progress_s + length - previous_s if wrapped else
                     projection.progress_s - length - previous_s if reverse_seam else
                     projection.progress_s - previous_s)
            if dt < 0 or abs(delta) > max(speed, previous_speed) * dt + 2.0:
                return None
        if self.progress is not None and projection.progress_s < self.progress - 2.0 and not wrapped:
            return None  # Require restart after a teleport/backtrack, not replay old events.
        if reverse_seam:
            return speed  # Position noise cannot replay the previous lap.
        if wrapped:
            # A new lap must earn fresh junction and signal permissions.
            self.event = None
            self.completed.clear()
            self.startup_inside_contexts.clear()
            self.ignore_line_until_s = -1.0
            self.enter_permission = False
            self.entry_ticket = self.entry_grant = self.green_release = None
            self.lead.reset()
            self.core = StopLineControllerCore(**self.core_params)
            self.next_guard_id = self.next_guard_core = None
            self.confirmation.reset()
            self.selection = Selection(reason="new_lap_awaiting_signal")
            self.selection_stamp = 0.0
            self.selection_context = None
            self.signal_observations.clear()
            self.route_object_observations.clear()
            self.signal_queue_fault = None
            self.samples.pop("signal", None)
            self.samples.pop("objects", None)
            self.samples.pop("line", None)
            rospy.loginfo("Maneuver fusion completed a lap; reset junction permissions")
        if self.progress is None:
            # A fresh process may start after the rear axle has already passed
            # a mapped stop line. It must not reinterpret that line as a new
            # approach, while future junctions still need fresh permissions.
            self.startup_inside_contexts = {
                context["id"] for context in self.route_signal_contexts
                if (context["stop_s"] - self.front_axle_offset_m
                    < projection.progress_s < context["end"])
            }
        self.route_rejection_reason = "accepted"
        self.progress = projection.progress_s
        self.segment = projection.segment_index
        self.last_projection = (self.progress, ros_now, speed)
        return speed

    def discover_event(self, line_sample, speed, now, ros_now):
        if self.event is not None or self.progress is None:
            return
        if self.require_context:
            horizon = max(40.0, speed * (self.lead.lead_sec + 1.0) + speed**2 / 2.0)
            candidates = [dict(c, kind="turn", committed=False) for c in self.contexts
                          if c["id"] not in self.completed and c["end"] > self.progress]
            candidates.extend(dict(id=m.identifier, start=m.start_s_m, end=m.end_s_m,
                                   kind=m.kind, direction=m.direction, committed=False)
                              for m in self.manual
                              if m.identifier not in self.completed and m.end_s_m > self.progress)
            if candidates:
                candidate = min(candidates, key=lambda c: (c["start"], c.get("source") != "mgeo"))
                objects = self.fresh("objects", now, ros_now)
                matched_head_visible = False
                if (objects is not None and candidate.get("source") == "mgeo"
                        and candidate.get("signal_points")):
                    selection = self.associate_signal_frame(objects, candidate)
                    route_signal_ids = candidate.get("signal_ids") or [
                        point.get("id") for point in candidate["signal_points"]]
                    matched_head_visible = bool(selection is not None and selection.valid
                                                and selection.selected_id in route_signal_ids)
                line_matches_candidate = False
                if (line_sample is not None and line_sample.value.valid
                        and line_sample.value.header.frame_id == "base_link"
                        and finite(line_sample.value.distance_m)
                        and 0 <= line_sample.value.distance_m <= 50.0
                        and finite(line_sample.value.confidence)
                        and 0.5 <= line_sample.value.confidence <= 1.0):
                    line_s = self.line_route_s(line_sample)
                    target_s = candidate.get("stop_s", candidate["start"])
                    line_matches_candidate = line_s is not None and abs(line_s - target_s) <= 2.0
                if (candidate["start"] - self.progress <= horizon
                        or matched_head_visible or line_matches_candidate):
                    # Pre-arm only a route-matched signal, route-matched line,
                    # or a junction already inside its stopping horizon.
                    self.event = candidate
            return
        if self.route_signal_contexts:
            # Arm the mapped target before its comfortable braking distance,
            # including bumper clearance and the gradual brake ramp.
            core = self.core
            stop_horizon = (speed * core.reaction_time_sec
                            + speed**2 / (2.0 * core.planning_decel_mps2)
                            + core.front_reference_offset_m + core.hold_distance_m
                            + core.trigger_margin_m + core.brake_ramp_distance_m)
            horizon = max(40.0, speed * 6.0 + speed**2 / 2.0, stop_horizon)
            for context in self.route_signal_contexts:
                if context["id"] in self.completed:
                    continue
                if self.progress >= context["end"]:
                    self.completed.add(context["id"])
                    continue
                if context["start"] - self.progress <= horizon:
                    self.event = dict(
                        id=context["id"], source="route_camera", kind="turn",
                        start=context["stop_s"], stop_s=context["stop_s"],
                        end=context["end"], direction=context["direction"],
                        expected_signal_ids=context["signal_ids"],
                        signal_points=context["signal_points"],
                        entry_s=context["entry_s"], exit_s=context["exit_s"],
                        direction_source=context["direction_source"],
                        heading_change_deg=context["heading_change_deg"],
                        geometry_direction=context["geometry_direction"],
                        camera_line_confirmed=False,
                        committed=context["id"] in self.startup_inside_contexts,
                        resumed_inside=context["id"] in self.startup_inside_contexts,
                    )
                break
            return
        for maneuver in self.manual:
            if maneuver.identifier in self.completed:
                continue
            if self.progress >= maneuver.end_s_m:
                self.completed.add(maneuver.identifier)
                continue
            # Enough horizon to signal and brake BEFORE the PP lookahead reaches
            # a lane-change start. Events describe the existing route only.
            horizon = max(25.0, speed * (self.lead.lead_sec + 1.0) + speed**2 / 2.0)
            if maneuver.start_s_m - self.progress <= horizon:
                self.event = dict(id=maneuver.identifier, direction=maneuver.direction,
                                  start=maneuver.start_s_m, end=maneuver.end_s_m,
                                  kind=maneuver.kind, committed=False)
            break
        if self.require_context:
            # No semantic map match: do not turn a missing line into STRAIGHT.
            return
        if self.event is not None or line_sample is None or self.progress < self.ignore_line_until_s:
            return
        line = line_sample.value
        if (not line.valid or line.header.frame_id != "base_link"
                or not finite(line.distance_m) or not 0 <= line.distance_m <= 50.0
                or not finite(line.confidence) or line.confidence < 0.5):
            return
        line_s = self.line_route_s(line_sample)
        if line_s is None:
            return
        if self.stopline_requires_detected_signal:
            signal_sample = self.fresh("signal", now, ros_now)
            if not self.valid_sensor_stop_pair(line_sample, signal_sample):
                return
        intent = route_intent(self.points, self.s_values, line_s, self.preview_m, self.turn_threshold)
        self.event = dict(id="junction_%.1f" % line_s, start=line_s, end=intent.end_s_m,
                          direction=intent.direction, kind="turn", committed=False)

    def line_route_s(self, line_sample):
        """Match a camera distance to the route pose at its source timestamp.

        A current pose minus latest_speed*delay is wrong while braking. Use
        the same bounded 50ms pose matching as signal-head association.
        """
        pose_sample = min(self.pose_history, key=lambda p: abs(p.stamp - line_sample.stamp), default=None)
        if (pose_sample is None or abs(pose_sample.stamp - line_sample.stamp) > 0.05
                or pose_sample.value.header.frame_id != "map"):
            return None
        point = pose_sample.value.pose.pose.position
        if not all(finite(v) for v in (point.x, point.y, line_sample.value.distance_m)):
            return None
        projection = nearest_projection(self.points, self.s_values, point.x, point.y,
                                        start_segment=max(0, (self.segment or 0) - 250),
                                        end_segment=None if self.segment is None else self.segment + 250)
        if projection.distance_m > self.max_route_error:
            return None
        return projection.progress_s + line_sample.value.distance_m

    def crossing_s(self, event):
        if event["kind"] == "lane_change":
            return event["start"]
        # Camera noise may bring the STOP guard upstream, but cannot move a
        # verified map crossing upstream and bypass a still-relevant signal.
        line_s = event.get("stop_s")
        return (event["start"] if line_s is None else line_s) - self.front_axle_offset_m

    def turn_assessment(self, odom, speed):
        if not self.event or self.event["kind"] != "turn" or self.event["direction"] not in ("LEFT", "RIGHT"):
            return TurnMotion()
        pose = odom.value.pose.pose if odom is not None else None
        position = getattr(pose, "position", None)
        return self.turn_motion.evaluate(self.event, self.progress, speed,
            getattr(position, "x", None), getattr(position, "y", None),
            quaternion_yaw(getattr(pose, "orientation", None)), self.crossing_s(self.event))

    def entry_candidate(self, event, permitted, ready, route_valid, now, ros_now):
        """Provisional entry only; the final safety checks must also approve.

        A short-lived ticket proves permission on the approach side. A green
        confirmed near the crossing survives a brief missing camera frame;
        a fresh conflicting indication cancels that grant.
        """
        if event["committed"]:
            return True
        if not route_valid or self.progress <= self.crossing_s(event):
            return False
        ticket = self.entry_ticket
        live = bool(permitted and self.enter_permission and ticket is not None
                    and ticket[0] == route_event_key(event)
                    and ticket[1] <= self.crossing_s(event)
                    and 0 <= now - ticket[2] <= self.timeout
                    and 0 < ros_now - ticket[3] <= self.timeout)
        grant = self.entry_grant
        carried = bool(self.route_camera_calibrated and grant is not None
                       and grant[0] == route_event_key(event)
                       and grant[1] <= self.crossing_s(event)
                       and 0 <= now - grant[2] <= self.entry_dropout_grace_sec
                       and 0 < ros_now - grant[3] <= self.entry_dropout_grace_sec
                       and 0 <= self.progress - grant[1] <= 5.0)
        valid = bool(not event.get("entry_fault") and ready and (live or carried))
        if not valid:
            event["entry_fault"] = True
        return valid

    def tick(self, _event):
        with self.lock:
            now, ros_now = time.monotonic(), rospy.get_time()
            reset = self.previous_ros is not None and ros_now < self.previous_ros
            self.previous_ros = ros_now
            if reset:
                self.samples.clear()
                self.last_stamps.clear()
                self.vehicle_speed_source = VehicleSpeedSource(timeout_sec=self.vehicle_speed_timeout)
                self.vehicle_speed_mps = self.odometry_speed_mps = None
                self.progress = self.segment = self.event = None
                self.startup_inside_contexts.clear()
                self.last_projection = None
                self.merge_gap = self.merge_gap_at = None
                self.completed.clear()
                self.ignore_line_until_s = -1.0
                self.enter_permission = False
                self.entry_ticket = None
                self.entry_grant = None
                self.green_release = None
                self.lead.reset()
                self.core = StopLineControllerCore(**self.core_params)
                self.last_observed_line_stamp = 0.0
                self.pose_history.clear()
                self.confirmation.reset()
                self.selection_context, self.selection_stamp = None, 0.0
                self.signal_observations.clear()
                self.route_object_observations.clear()
                self.signal_queue_fault = None
                self.next_guard_id = self.next_guard_core = None
                self.corridor_anchor = self.corridor_last = None
                self.corridor_distance = 0.
                self.corridor_fault = False

            odom = self.fresh("odom", now, ros_now)
            speed = self.update_route(odom, now, ros_now)
            quality = self.fresh("quality", now, ros_now)
            route_valid = (speed is not None
                           and (not self.require_reference_path or self.reference_path_match))
            if self.require_quality:
                route_valid = route_valid and quality is not None and quality.value.state == "NORMAL"
            fallback = self.fresh("fallback", now, ros_now) if self.allow_blackout_lane else None
            if self.allow_blackout_lane:
                route_valid = bool(route_valid and quality is not None and quality.value.state == "NORMAL"
                                   and not quality.value.imu_stale
                                   and quality.value.gps_valid and not quality.value.gps_blackout
                                   and not quality.value.gps_recovering and fallback is not None
                                   and fallback.value.get("mode") == "normal_nominal"
                                   and fallback.value.get("nominal_fresh") is True)
            line = self.fresh("line", now, ros_now)
            signal = self.fresh("signal", now, ros_now)
            objects = self.fresh("objects", now, ros_now)
            corridor_ok = self.blackout_corridor(quality, fallback, speed, route_valid, line, objects, now, ros_now)
            if route_valid and not self.tunnel_route_accepted:
                self.discover_event(line, speed, now, ros_now)
                if self.route_signal_contexts and self.event is None and self.core.stop_requested:
                    # No selected map junction may retain an old camera-only
                    # stop request after a process restart or route transition.
                    self.core = StopLineControllerCore(**self.core_params)
                    self.green_release = None

            motion = self.turn_assessment(odom, speed)
            completed_this_tick = None
            if (self.event and self.event["committed"] and route_valid
                    and self.progress >= self.event["end"]
                    and (motion.phase == "NONE" or (motion.exit_ready and not motion.fault))):
                completed_this_tick = self.event["id"]
                self.completed.add(self.event["id"])
                self.ignore_line_until_s = self.progress + 3.0
                self.event = None
                self.enter_permission = False
                self.entry_ticket = None
                self.entry_grant = None
                self.green_release = None
                self.lead.reset()
                self.core = StopLineControllerCore(**self.core_params)
                # Only a genuinely newer line may arm the next intersection.
                self.last_observed_line_stamp = line.stamp if line else self.last_observed_line_stamp
                self.confirmation.reset()
                self.selection_context = None
                if self.route_signal_contexts:
                    # The next junction must earn its own source frames.
                    self.signal_observations.clear()
                    self.route_object_observations.clear()
                    self.signal_queue_fault = None
                # Discover a close following junction in this SAME tick.
                if self.require_context or self.route_signal_contexts:
                    self.discover_event(line, speed, now, ros_now)
                motion = self.turn_assessment(odom, speed)

            # A pose fault before entry also cancels previously accumulated
            # signal permission. Recovery must confirm new observations.
            if motion.fault and self.event and not self.event["committed"]:
                self.revoke_entry_permission()
                self.confirmation.reset()

            if self.require_context:
                self.synchronize_route_intent(self.event)
            direction = self.event["direction"] if self.event else (
                "UNKNOWN" if self.require_context or self.route_signal_contexts else "STRAIGHT")
            lamp = direction if direction in ("LEFT", "RIGHT") else "OFF"
            if ((self.require_context or self.route_signal_contexts) and self.event and self.progress is not None
                    and self.event["start"] - self.progress > max(40., (speed or 0.) * 6. + (speed or 0.)**2 / 2.)):
                lamp = "OFF"  # Early map association is not early lamp activation.
            # A localization fault brakes the vehicle, but does not cancel an
            # already entered turn's indication before its exit is verified.
            keep_lamp = bool(self.event and self.event["committed"])
            self.send_lamp(lamp if route_valid or keep_lamp else "OFF", now, ros_now)
            ready = direction == "STRAIGHT" or (direction in ("LEFT", "RIGHT") and
                    (self.test_without_turn_signals or self.lead.ready(lamp, now, ros_now)))
            signal_ok = (signal is not None and signal.value.valid
                         and finite(signal.value.confidence) and 0.5 <= signal.value.confidence <= 1.0)
            permitted = bool(signal_ok and signal_permits(signal.value.state, direction, self.right_on_green))
            event = self.event
            if self.route_signal_contexts:
                if self.route_camera_calibrated:
                    signal_ok = False  # The ordered object frames are consumed after line matching below.
                    permitted = False
                else:
                    signal_ok = self.unique_route_signal(
                        signal, self.route_objects_at(signal.stamp) if signal else None)
                    permitted = bool(event and signal_ok and
                                     signal_permits(signal.value.state, direction, self.right_on_green))
            elif not self.require_context:
                if self.stopline_requires_detected_signal:
                    self.consume_sensor_stop_pair(line, signal, direction, now, ros_now)
                else:
                    self.consume_legacy_signals(direction, ready, route_valid and not motion.fault, now, ros_now)
            if self.require_context:
                chosen = self.selected_signal(objects, event, route_valid and not motion.fault)
                signal_ok = chosen.valid
                permitted = chosen.valid and signal_permits(chosen.state, direction, self.right_on_green)
            committed = bool(event and event["committed"])

            if event and event["kind"] == "lane_change":
                # The PP lookahead is 4 + 0.35*v. Stop the rear axle before it
                # reaches the start, until five continuous seconds have elapsed.
                guard = max(5.0, 4.0 + 0.35 * (speed or 0.0))
                virtual_distance = (event["start"] - guard - self.progress
                                    + self.core.front_reference_offset_m + self.core.hold_distance_m)
                if not committed:
                    self.core.observe_line(max(0.0, virtual_distance), 1.0, True, ros_now, now, ros_now,
                                           target_id=event["id"])
                gap_clear = self.lane_change_clear(direction, now)
                # Do not use a lane-change event to bypass a visible red light.
                permitted = not signal_ok or signal_permits(signal.value.state, "STRAIGHT", self.right_on_green)
                if self.require_context:
                    # Manual lane changes have no assigned traffic-light head.
                    # UNKNOWN association must not mean a visible red is absent.
                    permitted = (objects is not None and not objects.value.objects
                                 and line is not None and not line.value.valid)
                committed = self.entry_candidate(event, permitted and gap_clear, ready, route_valid, now, ros_now)
                effective = "GREEN" if ready and route_valid and gap_clear and permitted else "RED"
                self.core.observe_signal(effective, 1.0, True, ros_now, now, ros_now)
            else:
                if self.route_signal_contexts:
                    if event and route_valid and not committed:
                        # MGeo selects the intersection, stop target and route
                        # direction. CAM1 can corroborate this line, but a
                        # different lane's marking must not veto its green.
                        if self.valid_sensor_stopline(line):
                            measured_s = self.line_route_s(line)
                            if measured_s is not None:
                                event["stopline_match_error_m"] = measured_s - event["stop_s"]
                                if abs(event["stopline_match_error_m"]) <= self.route_stopline_match_tolerance:
                                    event["camera_line_confirmed"] = True
                                    # CAM1 confirms the selected map line; it
                                    # cannot pull the braking target onto a
                                    # nearer marking in an adjacent lane.
                        self.core.observe_line(max(0.0, event["start"] - self.progress),
                                               1.0, True, ros_now, now, ros_now,
                                               target_id=event["id"])
                    self.consume_route_signals(event, ready, route_valid and not motion.fault,
                                               now, ros_now)
                    if self.route_camera_calibrated and not committed:
                        release_latched = bool(objects is not None and
                                               self.route_green_visibility_loss(
                                                   self.selection, objects, event, now, ros_now))
                        signal_ok = bool(objects is not None and (self.selection.valid or release_latched)
                                         and not self.signal_observations
                                         and -0.05 <= ros_now - self.selection_stamp <= self.signal_timeout)
                        permitted = bool(signal_ok and (release_latched or signal_permits(
                            self.selection.state, direction, self.right_on_green)))
                    if not committed:
                        permitted = bool(permitted and event and
                                         (self.route_camera_calibrated or event["camera_line_confirmed"])
                                         and ready and route_valid
                                         and not motion.fault and self.core.signal is not None
                                         and self.core.signal.value == "GREEN"
                                         and self.core.signal.fresh(ros_now, now, self.signal_timeout))
                    if event and not committed and not permitted:
                        if self.signal_observations:
                            self.suspend_entry_permission()
                        else:
                            self.revoke_entry_permission()
                elif self.require_context and event and route_valid and not committed:
                    # Map target survives camera loss. Unverified entry is only
                    # a conservative guard, NOT a measured stop-line location.
                    distance = event["start"] - self.progress
                    if line is not None and line.value.valid and line.value.header.frame_id == "base_link":
                        measured = line.value.distance_m
                        if (finite(measured) and 0 <= measured <= 50
                                and finite(line.value.confidence) and 0.5 <= line.value.confidence <= 1.0):
                            candidate = self.line_route_s(line)
                            if candidate is not None and abs(candidate-event["start"]) <= 2.0:
                                # A noisy observation cannot push the guard ahead.
                                event["start"] = min(event["start"], candidate)
                                event["camera_line_confirmed"] = True
                                distance = event["start"] - self.progress
                    self.core.observe_line(max(0.0, distance), 1.0, True, ros_now, now, ros_now,
                                           target_id=event["id"])
                    permitted = permitted and (event.get("stop_s") is not None
                                               or event.get("camera_line_confirmed", False))
                elif (not self.require_context and not self.route_signal_contexts
                        and (not self.stopline_requires_detected_signal
                             or self.valid_sensor_stop_pair(line, signal))
                        and line is not None and line.stamp > self.last_observed_line_stamp
                        and line.value.header.frame_id == "base_link" and not committed
                        and (self.progress is None or self.progress >= self.ignore_line_until_s)):
                    self.core.observe_line(line.value.distance_m, line.value.confidence, line.value.valid,
                                           line.stamp, line.received, ros_now)
                    self.last_observed_line_stamp = line.stamp
                if event:
                    committed = self.entry_candidate(event, permitted, ready, route_valid, now, ros_now)
                if self.require_context and event and not committed:
                    effective = "GREEN" if permitted and ready and route_valid else "RED"
                    self.core.observe_signal(effective, 1.0, True, ros_now, now, ros_now)
                elif (signal is not None and not self.require_context and not self.route_signal_contexts
                      and (not self.stopline_requires_detected_signal
                           or self.valid_sensor_stop_pair(line, signal))):
                    effective = "GREEN" if permitted and ready and route_valid else "RED"
                    self.core.observe_signal(effective, signal.value.confidence, signal_ok,
                                             signal.stamp, signal.received, ros_now)
                if event and not ready:
                    self.core.stop_requested = True

            signal_frame_pending = bool(self.route_signal_contexts and event and not committed
                                        and self.signal_observations)
            # Once the front axle has entered this mapped intersection,
            # its light and stop line no longer govern this crossing. Route,
            # obstacle and upstream-command safety checks still apply below.
            decision = (Decision("NOMINAL", "junction_entered")
                        if committed and event and event["kind"] == "turn" else
                        self.core.update(now, ros_now, speed,
                                         permission_pending=signal_frame_pending))
            # A later junction may impose an APPROACH speed/brake overlay.
            # It cannot erase permission already confirmed for this junction.
            # Safety faults below still revoke the ticket before publication.
            current_junction_released = decision.mode == "NOMINAL"
            current_green_released = decision.reason == "green"
            # The next junction can be close to this turn. Plan a speed that
            # still permits stopping at its mapped line, but leave the stop
            # decision to that junction's own signal after route handoff.
            future = list(self.contexts if self.require_context else self.route_signal_contexts) + [
                dict(id=m.identifier, start=m.start_s_m, end=m.end_s_m,
                                                kind=m.kind) for m in self.manual]
            future = sorted((c for c in future if event and c["id"] != event["id"]
                             and c["id"] not in self.completed and c["start"] > event["start"]
                             and c["end"] > (self.progress or 0)), key=lambda c: c["start"])
            upcoming = future[0] if future else None
            if ((self.require_context or self.route_signal_contexts) and event
                    and event.get("committed") and route_valid and upcoming
                    and upcoming.get("kind") != "lane_change"
                    and self.progress < event["end"]):
                self.next_guard_id = upcoming["id"]
                self.next_guard_core = None
                a = self.core.planning_decel_mps2
                tau = self.core.reaction_time_sec
                # Bumper clearance and a small model-error reserve are kept
                # even though this overlay never issues the future STOP.
                available = (upcoming["start"] - self.progress
                             - self.core.front_reference_offset_m
                             - self.core.hold_distance_m - 2.0)
                if available <= 0:
                    decision = Decision("SAFE_STOP", "next_junction_spacing_unavailable",
                                        0.0, 1.0, 0.0, available)
                else:
                    cap_mps = max(0.0, math.sqrt((a * tau)**2 + 2.0 * a * available) - a * tau)
                    error = cap_mps - speed
                    accel_limit = min(decision.accel_limit, clamp(error * 0.45, 0.0, 1.0))
                    brake = max(decision.brake, clamp((-error - 0.25) * 0.30, 0.0, 1.0))
                    cap_kph = cap_mps * 3.6
                    target = (cap_kph if decision.target_speed_kph is None else
                              min(decision.target_speed_kph, cap_kph))
                    reason = ("next_junction_speed_envelope"
                              if accel_limit < decision.accel_limit or brake > decision.brake
                              else decision.reason)
                    decision = Decision(decision.mode, reason, accel_limit, brake,
                                        target, available)
            else:
                self.next_guard_id = self.next_guard_core = None
            # An allowed arrow controls entry, not cornering speed. Retain the
            # original core mode so a speed limit cannot erase an entry ticket.
            decision = motion.constrain(decision)
            released_context_id = (event["id"] if self.route_camera_calibrated
                                   and event and permitted and current_green_released
                                   else None)
            decision = self.constrain_route_speed(decision, speed, released_context_id)
            reasons = [motion.fault] if motion.fault else []
            # UNKNOWN/no line is a valid new image observation. No new stamped
            # observations is a dead camera/inference stream, including mid-turn.
            if (self.require_camera_stream and not self.require_context
                    and not self.stopline_requires_detected_signal
                    and (signal is None or line is None)):
                reasons.append("camera_observation_stream_stale")
            if self.require_reference_path and not self.reference_path_match:
                reasons.append(self.reference_path_reason)
            if self.require_context:
                if not route_valid and not corridor_ok:
                    reasons.append("signal_localization_unreliable")
                if (event and route_valid and line is not None and line.value.valid
                        and line.value.header.frame_id == "base_link"
                        and finite(line.value.distance_m) and 0 <= line.value.distance_m <= 50
                        and finite(line.value.confidence) and 0.5 <= line.value.confidence <= 1.):
                    observed_s = self.line_route_s(line)
                    if observed_s is None and event.get("stop_s") is None:
                        reasons.append("stopline_pose_unsynchronized")
            if self.require_vehicle_speed and self.vehicle_speed_source.sample(ros_now, now) is None:
                reasons.append("vehicle_speed_stale_or_unavailable")
            if speed is None:
                reasons.append("odometry_or_route_unavailable")
            if self.allow_blackout_lane and fallback is None:
                reasons.append("camera_fallback_status_stale")
            if self.allow_blackout_lane and not route_valid and not corridor_ok:
                reasons.append("blackout_corridor_unavailable")
            if event and not route_valid:
                reasons.append("maneuver_localization_unreliable")
            if event and direction == "UNKNOWN":
                reasons.append("route_direction_unknown")
            if (self.route_signal_contexts and event and not committed
                    and self.signal_observations and self.core.distance is not None
                    and now - self.signal_observations[0].received >= 0.15
                    and self.core.distance - self.core.hold_distance_m
                    <= max(5.0, (speed or 0.0))):
                reasons.append("signal_pose_pair_pending" if self.route_camera_calibrated
                               else "signal_frame_pair_pending")
            if event and event["kind"] == "lane_change" and not self.lane_change_clear(direction, now):
                reasons.append("adjacent_lane_gap_unavailable_or_stale")
            safety = self.fresh("safety", now, ros_now)
            if self.require_safety and safety is None:
                reasons.append("fused_safety_stale")
            if (safety is not None and safety.value.stop_required
                    and not self.tunnel_obstacle_stop_overridden(safety, odom, now, ros_now)):
                reasons.append(safety.value.reason or "fused_obstacle_stop")
            nominal = self.nominal
            valid_nominal = (nominal is not None and self.nominal_at is not None
                             and 0 <= now - self.nominal_at <= self.timeout
                             and nominal.longlCmdType == 1
                             and all(finite(float(getattr(nominal, k))) for k in ("accel", "brake", "steering")))
            if not valid_nominal:
                reasons.append("nominal_stale_or_not_type1")
                nominal = CtrlCmd()
            if event and committed and not event["committed"]:
                # The tentative crossing must not survive a current obstacle,
                # dead stream, invalid pose or invalid upstream command.
                if reasons:
                    event["entry_fault"] = True
                    committed = False
                else:
                    event["committed"] = True
                    self.entry_grant = None
            if event and event.get("entry_fault"):
                reasons.append("entry_without_current_permission")
            if reasons:
                self.green_release = None
            elif (self.route_camera_calibrated and event and not committed
                    and self.selection.valid and permitted and ready and route_valid
                    and self.core._green_confirmed(now, ros_now)
                    and self.progress is not None
                    and 0 <= event["stop_s"] - self.progress <= self.green_release_start_distance_m
                    and self.progress <= self.crossing_s(event)):
                self.green_release = (route_event_key(event), self.progress, now, ros_now)
            self.enter_permission = bool(event and route_valid and ready and (permitted or committed)
                                         and current_junction_released and not reasons)
            self.entry_ticket = ((route_event_key(event), self.progress, now, ros_now)
                                 if self.enter_permission and not committed
                                 and self.progress <= self.crossing_s(event) else None)
            if (self.route_camera_calibrated and event and self.enter_permission
                    and not committed and self.progress is not None
                    and 0 <= self.crossing_s(event) - self.progress <= 3.0):
                self.entry_grant = (route_event_key(event), self.progress, now, ros_now)
            if reasons:
                decision = Decision("SAFE_STOP", ",".join(reasons), 0.0, 1.0, 0.0, decision.distance_m)
            if corridor_ok:
                # CtrlCmd has no header: status and command can arrive one tick
                # apart. A fresh braking status must constrain an older pedal.
                decision = Decision(decision.mode, decision.reason,
                                    min(decision.accel_limit, fallback.value["accel"]),
                                    max(decision.brake, fallback.value["brake"]),
                                    decision.target_speed_kph, decision.distance_m)
            self.dynamic_zone_status=self.dynamic_constraint(odom,now,ros_now)
            if self.dynamic_zone_status is not None and self.dynamic_zone_status["stop"]:
                decision=Decision("SAFE_STOP",decision.reason+"|"+self.dynamic_zone_status["reason"],
                                  0.,1.,0.,decision.distance_m)
            if (self.dynamic_zone_status is not None and not self.dynamic_zone_status["stop"]
                    and self.dynamic_zone_status["cap_mps"] is not None):
                cap_kph=self.dynamic_zone_status["cap_mps"]*3.6
                target=(cap_kph if decision.target_speed_kph is None
                        else min(decision.target_speed_kph,cap_kph))
                decision=Decision(decision.mode,decision.reason,decision.accel_limit,decision.brake,
                                  target,decision.distance_m)
            output = overlay(nominal, decision)
            if self.dynamic_zone_status is not None:
                measured=(self.vehicle_speed_mps if self.vehicle_speed_mps is not None
                          else self.odometry_speed_mps)
                output=constrain_pedals(output,self.dynamic_zone_status,measured or 0.)
            requested_accel = output.accel
            output.accel = self.accel_limiter.limit(output.accel, output.brake, now, ros_now)
            self.output_pub.publish(output)
            rospy.loginfo_throttle(1.0,
                "FINAL_CTRL mode=%s reason=%s route=%s global_error=%s frenet=%s tunnel=%s accel=%.2f brake=%.2f",
                decision.mode, decision.reason, self.route_rejection_reason,
                self.route_global_error_m, self.frenet_route_accepted, self.tunnel_route_accepted,
                output.accel, output.brake)
            route_signal_pixels = self.projected_route_heads(objects, event)
            permission_state = (self.selection.state if self.require_context or self.route_camera_calibrated
                                else signal.value.state if signal else "UNKNOWN")
            allowed_directions = sorted(signal_allowed_directions(permission_state, self.right_on_green)) if signal_ok else []
            route_objects = (objects if self.route_camera_calibrated else
                             self.route_objects_at(signal.stamp) if signal and self.route_signal_contexts else None)
            if self.require_context or self.route_camera_calibrated:
                signal_selection_reason = self.selection.reason
            elif self.route_signal_contexts:
                if event and not event["camera_line_confirmed"]:
                    signal_selection_reason = "awaiting_cam1_stopline_match"
                elif self.signal_observations:
                    signal_selection_reason = "awaiting_cam4_object_pair"
                elif not signal_ok:
                    signal_selection_reason = "cam4_head_count_%d" % self.route_signal_candidate_count(route_objects)
                elif not permitted:
                    signal_selection_reason = "directional_signal_not_permitted"
                else:
                    signal_selection_reason = "unique_cam4_head_and_mapped_stopline"
            elif self.stopline_requires_detected_signal:
                signal_selection_reason = "awaiting_paired_signal_stopline"
            else:
                signal_selection_reason = "legacy_unassociated"
            self.state_pub.publish(String(data=json.dumps({
                "mode": decision.mode, "reason": decision.reason, "event": event,
                "dynamic_zone_guard": self.dynamic_zone_status,
                "effective_speed_target_kph": decision.target_speed_kph,
                "direction": direction, "signal": permission_state,
                "global_camera_signal": signal.value.state if signal else "STALE",
                "route_direction": direction, "signal_allowed_directions": allowed_directions,
                "route_direction_source": event.get("direction_source") if event else None,
                "route_heading_change_deg": event.get("heading_change_deg") if event else None,
                "route_rejection_reason": self.route_rejection_reason,
                "route_global_error_m": self.route_global_error_m,
                "highway_route_accepted": self.highway_route_accepted,
                "frenet_route_accepted": self.frenet_route_accepted,
                "tunnel_route_accepted": self.tunnel_route_accepted,
                "tunnel_motion_version": "TUNNEL_ROUTE_V4",
                "tunnel_mode_active": self.tunnel_mode_active(now, ros_now),
                "route_link_version": "FRENET_ROUTE_LINK_V1",
                "vehicle_speed_topic": self.vehicle_speed_topic,
                "vehicle_speed_required": self.require_vehicle_speed,
                "vehicle_speed_kph": None if self.vehicle_speed_mps is None else self.vehicle_speed_mps * 3.6,
                "odometry_speed_kph": None if self.odometry_speed_mps is None else self.odometry_speed_mps * 3.6,
                "route_signal_compatible": direction in allowed_directions,
                "reference_path_required": self.require_reference_path,
                "reference_path_match": self.reference_path_match,
                "reference_path_reason": self.reference_path_reason,
                "indicator_ready": ready, "lamp_udp_enabled": self.lamp_enabled,
                "test_without_turn_signals": self.test_without_turn_signals,
                "indicator_lead_required": not self.test_without_turn_signals,
                "lamp_requested": self.lamp_requested, "lamp_transmit_ok": self.lamp_transmit_ok,
                "permission": self.enter_permission, "progress_s_m": self.progress,
                "current_junction_released": current_junction_released,
                "entry_fault": bool(event and event.get("entry_fault")),
                "turn_phase": ("TURNING" if committed and motion.phase == "APPROACH" else motion.phase),
                "turn_speed_limit_kph": motion.speed_limit_kph,
                "turn_curve_speed_kph": motion.curve_speed_kph,
                "turn_curvature_abs_m_inv": motion.curvature_abs_m_inv,
                "max_speed_kph": self.turn_motion.max_speed_kph,
                "speed_governor_target_kph": self.governor_target_kph,
                "mapped_stopline_distance_m": self.governor_stopline_distance_m,
                "lateral_accel_limit_mps2": self.turn_motion.lateral_accel_limit_mps2,
                "turn_heading_error_deg": motion.heading_error_deg,
                "turn_lateral_error_m": motion.lateral_error_m,
                "turn_exit_heading_error_deg": motion.exit_heading_error_deg,
                "completed_event_id": completed_this_tick,
                "blackout_lane_corridor": corridor_ok,
                "blackout_corridor_reason": self.corridor_reason,
                "blackout_corridor_distance_m": self.corridor_distance,
                "entry_reference_offset_m": self.front_axle_offset_m,
                "entry_crossing_s_m": self.crossing_s(event) if event else None,
                "front_bumper_distance_m": decision.distance_m,
                "target_clearance_m": self.core.hold_distance_m,
                "stopline_planning_decel_mps2": self.core.planning_decel_mps2,
                "stopline_brake_ramp_distance_m": self.core.brake_ramp_distance_m,
                "stopline_settle_distance_m": self.core.settle_distance_m,
                "stopline_tracking_fault": self.core.tracking_fault,
                "accel_rise_limited": output.accel < requested_accel,
                "route_context_count": len(self.contexts if self.require_context else self.route_signal_contexts),
                "route_context_error": self.context_load_error,
                # Expose the live gate profile so a stale map-based launch is
                # immediately distinguishable from the camera-only profile.
                "controller_profile": (
                    "mgeo" if self.require_context else
                    "route_camera" if self.route_signal_contexts else
                    "sensor_only" if self.stopline_requires_detected_signal else "legacy_sensor"),
                "require_route_signal_context": self.require_context,
                "stopline_requires_detected_signal": self.stopline_requires_detected_signal,
                "next_junction_guard_id": self.next_guard_id,
                "signal_selection_reason": signal_selection_reason,
                "route_cam4_candidate_count": self.route_signal_candidate_count(route_objects),
                "signal_pending_frames": len(self.signal_observations),
                "signal_release_suspended": signal_frame_pending,
                "signal_release_latched": bool(objects is not None and event and
                                               self.route_green_visibility_loss(
                                                   self.selection, objects, event, now, ros_now)),
                "green_confirmation_samples": self.core.green_samples,
                "signal_queue_overflows": self.signal_queue_overflows,
                "signal_processed_stamp": self.selection_stamp if self.require_context or self.route_camera_calibrated else self.core.last_signal_stamp,
                "selected_signal_id": self.selection.selected_id if self.require_context or self.route_camera_calibrated else None,
                "selected_signal_state": self.selection.state if self.require_context or self.route_camera_calibrated else None,
                "route_camera_calibrated": self.route_camera_calibrated,
                "mapped_stopline_confirmed": bool(event and event.get("camera_line_confirmed")),
                "stopline_match_tolerance_m": self.route_stopline_match_tolerance,
                "mapped_signal_ids": event.get("expected_signal_ids", []) if event else [],
                "route_signal_projection_px": route_signal_pixels,
                "accel": output.accel, "brake": output.brake,
            }, allow_nan=False)))

    def shutdown(self):
        with self.lock:
            if self.socket is not None:
                try:
                    self.socket.sendto(build_lamp_packet("OFF"), self.remote)
                except OSError:
                    pass
                self.socket.close()
                self.socket = None


if __name__ == "__main__":
    ManeuverFusionNode()
    rospy.spin()
