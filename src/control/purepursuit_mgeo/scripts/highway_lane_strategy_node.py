#!/usr/bin/env python3
# V9-priority overlay with v61 ROS transport compatibility.
# Source: morai_highway_team_handoff(1).zip, V9 lane strategy.
# Only the ROS interfaces/gates and non-drive safety publications are adapted.
"""Highway-only proactive lane-change + lane-hold + longitudinal safety manager.

Camera-team files are treated as read-only inputs. This node consumes:
  /perception/camera/highway_environment
  /perception/camera/lane_info
  /perception/merge_gap/available
  /perception/merge_gap/unavailable
and LiDAR tracked obstacles / odometry.

Outside the highway scenario, it simply republishes the existing avoidance
PathManager path/stop and the cruise speed. During the highway scenario it:
  1) waits for a safe LEFT merge gap while staying on the base path;
  2) generates a committed quintic lateral shift into the inner lane;
  3) keeps following the camera-reported current lane centerline after the
     change, so the vehicle does NOT get pulled back to the original outer
     global path;
  4) adapts target speed to the lead vehicle and uses front/rear vehicle speed
     when selecting a lane-change speed;
  5) returns to the base/global path only after the physical lanes converge
     and the ego vehicle is again close to the global path.

It never publishes /ctrl_cmd.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from std_msgs.msg import Bool, Float64, String

from lidar_perception.msg import LidarObstacleArray
from purepursuit_mgeo.path import PathPoint, load_mgeo_path
from purepursuit_mgeo.plan_transport import plan_locked, read_trajectory, trajectory_payload
from purepursuit_mgeo.motion import lead_brake_decision


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def smoothstep5(u: float) -> float:
    u = clamp(float(u), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def polyline_arclength(points: Sequence[Tuple[float, float]]) -> List[float]:
    out = [0.0]
    for i in range(1, len(points)):
        out.append(out[-1] + math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]))
    return out



def resample_polyline_max_gap(points: Sequence[Tuple[float, float]], max_gap: float = 1.0) -> List[Tuple[float, float]]:
    """Add collinear intermediate points; leave the physical path unchanged."""
    if not points:
        return []
    gap = max(0.2, float(max_gap))
    result = [tuple(points[0])]
    for end in points[1:]:
        x0, y0 = result[-1]
        x1, y1 = float(end[0]), float(end[1])
        n = max(1, int(math.ceil(math.hypot(x1-x0, y1-y0) / gap)))
        for k in range(1, n+1):
            t = float(k)/float(n)
            result.append((x0 + (x1-x0)*t, y0 + (y1-y0)*t))
    return result


def required_lane_change_length(speed_mps: float, lane_width_m: float,
                                accel_limit_mps2: float, safety_factor: float = 1.15) -> float:
    """Conservative flat-road quintic length bound; final path still passes exact curvature check.

    The 0->1 quintic smoothstep has max |d²y/du²| ≈ 5.7735. The
    estimate assumes one lane-width lateral translation and is NOT a
    substitute for evaluating actual curvature of the camera-shaped path.
    """
    v = max(0.0, float(speed_mps))
    width = max(0.0, float(lane_width_m))
    a = max(0.15, float(accel_limit_mps2))
    return float(safety_factor) * math.sqrt(5.774 * width * v * v / a)


def completed_change_handoff(settled: bool, camera_centered: bool, endpoint_guard: bool) -> bool:
    """Prefer center-confirmed handoff; never drive to the finite-path PP goal stop."""
    return bool((settled and camera_centered) or endpoint_guard)


def lane_center_speed_limit(lateral_error_m: float, base_mps: float,
                            conservative_mps: float = 1.5) -> float:
    """Continuous, monotone lateral-recovery speed limit (never speeds above base)."""
    e=abs(float(lateral_error_m))
    if not math.isfinite(e):
        return min(base_mps, conservative_mps)
    if e >= 1.30:
        cap=conservative_mps
    elif e >= 0.90:
        # 1.30m -> 1.5m/s, 0.90m -> 2.8m/s
        cap=conservative_mps+(1.30-e)/0.40*(2.8-conservative_mps)
    elif e >= 0.45:
        # 0.90m -> 2.8m/s, 0.45m -> 6.0m/s
        cap=2.8+(0.90-e)/0.45*3.2
    else:
        # Only good centering may release the full requested speed.
        cap=float(base_mps)
    return max(0.0,min(float(base_mps),cap))


def heading_on_polyline(points: Sequence[Tuple[float, float]], x: float, delta: float = 2.0) -> Optional[float]:
    a = interp_y(points, x - delta)
    b = interp_y(points, x + delta)
    if a is None or b is None:
        return None
    return math.atan2(b - a, 2.0 * delta)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def lane_change_observation_gate(info: dict, near_max_m: float = 8.0) -> Tuple[bool, str, dict]:
    """Only authorize *new* lane changes from recently observed near divider geometry.

    HELD/INVALID camera output may serve limited lane-hold recovery, but cannot
    authorize a new lateral manoeuvre. A boundary first seen 11m ahead was
    associated with a sparse >11m initial committed trajectory in the field bag.
    """
    if not isinstance(info, dict) or info.get("output_status") != "FRESH":
        return False, "change_camera_not_fresh", {"output_status": None if not info else info.get("output_status")}
    raw = info.get("left_boundary_points") or []
    xs = []
    for p in raw:
        try:
            x = float(p[0])
            y = float(p[1])
            if math.isfinite(x) and math.isfinite(y) and x > 0.0:
                xs.append(x)
        except (ValueError, TypeError, IndexError):
            continue
    if len(xs) < 3:
        return False, "change_near_divider_missing", {"usable_points": len(xs)}
    nearest = min(xs)
    diag = {"divider_first_x_m": round(nearest, 3), "max_first_x_m": round(float(near_max_m), 3), "usable_points": len(xs)}
    if nearest > near_max_m:
        return False, "change_divider_starts_too_far", diag
    return True, "ok", diag


def tangent_at(points: Sequence[Tuple[float, float]], i: int) -> Tuple[float, float]:
    if len(points) < 2:
        return 1.0, 0.0
    if i <= 0:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
    elif i >= len(points) - 1:
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
    else:
        dx = points[i+1][0] - points[i-1][0]
        dy = points[i+1][1] - points[i-1][1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 1.0, 0.0
    return dx / n, dy / n


def interp_y(points: Sequence[Tuple[float, float]], x: float) -> Optional[float]:
    if not points:
        return None
    pts = sorted(points, key=lambda p: p[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        x0, y0 = pts[i-1]
        x1, y1 = pts[i]
        if x0 <= x <= x1 and x1 > x0 + 1e-6:
            u = (x - x0) / (x1 - x0)
            return y0 + u * (y1 - y0)
    return None


@dataclass
class LocalObstacle:
    oid: int
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    yaw: float


class HighwayLaneStrategyNode:
    OFF = "OFF"
    WAIT_GAP = "WAIT_GAP"
    LANE_CHANGE = "LANE_CHANGE"
    INNER_HOLD = "INNER_HOLD"
    REJOIN = "REJOIN"
    DONE = "DONE"

    def __init__(self) -> None:
        rospy.init_node("highway_lane_strategy", anonymous=False)
        # v61 ROS interface compatibility. V9 lane planning remains the primary algorithm.
        self._plan_lock = threading.RLock()
        self.base_trajectory_topic = rospy.get_param("~base_trajectory_topic", "")
        self.base_reason = ""
        self.trajectory_seq = 0
        self.route_gate_required = bool(rospy.get_param("~route_gate_required", False))
        self.route_gate_topic = rospy.get_param("~route_gate_topic", "/planning/highway_route_active")
        self.route_gate_timeout_s = float(rospy.get_param("~route_gate_timeout_s", 0.6))
        self.route_gate_active = False
        self.route_gate_at = None
        self.handoff_gate_required = bool(rospy.get_param("~handoff_gate_required", False))
        self.handoff_topic = rospy.get_param("~handoff_topic", "/planning/highway_handoff_due")
        self.handoff_timeout_s = float(rospy.get_param("~handoff_timeout_s", 0.6))
        self.handoff_due = False
        self.handoff_at = None
        self.force_highway_active = bool(rospy.get_param("~force_highway_active", False))
        self.bypass_sensor_merge_gate = bool(rospy.get_param("~bypass_sensor_merge_gate", False))
        self.rrt_lidar_only_mode = bool(rospy.get_param("~rrt_lidar_only_mode", False))

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.path_file = rospy.get_param("~path_file")
        self.global_points = load_mgeo_path(self.path_file)

        self.base_path_topic = rospy.get_param("~base_path_topic", "/avoidance_path_manager/active_path")
        self.base_stop_topic = rospy.get_param("~base_stop_topic", "/avoidance_path_manager/stop_required")
        self.odom_topic = rospy.get_param("~odom_topic", "/localization/odometry")
        self.obstacle_topic = rospy.get_param("~obstacle_topic", "/perception/lidar/tracked_obstacles_map")
        self.lane_info_topic = rospy.get_param("~lane_info_topic", "/perception/camera/lane_info")
        self.highway_topic = rospy.get_param("~highway_topic", "/perception/camera/highway_environment")
        self.highway_request_topic = rospy.get_param("~highway_request_topic", "/planning/highway_lane_change_request")
        self.merge_available_topic = rospy.get_param("~merge_available_topic", "/perception/merge_gap/available")
        self.merge_unavailable_topic = rospy.get_param("~merge_unavailable_topic", "/perception/merge_gap/unavailable")

        self.cruise_speed_mps = float(rospy.get_param("~cruise_speed_mps", 6.0))
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.highway_confirm_s = float(rospy.get_param("~highway_confirm_s", 0.5))
        self.ready_confirm_s = float(rospy.get_param("~ready_confirm_s", 0.5))
        self.lane_info_timeout_s = float(rospy.get_param("~lane_info_timeout_s", 0.6))
        self.odom_timeout_s = float(rospy.get_param("~odom_timeout_s", 0.5))
        self.obstacle_timeout_s = float(rospy.get_param("~obstacle_timeout_s", 0.6))
        self.base_timeout_s = float(rospy.get_param("~base_timeout_s", 0.8))
        self.merge_timeout_s = float(rospy.get_param("~merge_timeout_s", 0.8))
        self.mission_request_bypass_sensor_merge_gate = bool(
            rospy.get_param("~mission_request_bypass_sensor_merge_gate", True)
        )

        self.min_lane_confidence = float(rospy.get_param("~min_lane_confidence", 0.45))
        self.lane_width_min_m = float(rospy.get_param("~lane_width_min_m", 2.7))
        self.lane_width_max_m = float(rospy.get_param("~lane_width_max_m", 4.2))
        # Current/target lanes need not have the same physical width.
        # 0 = preserve V9 behaviour; use an independently verified target width
        # when entering a different-width highway lane.
        self.target_lane_width_m = float(rospy.get_param("~target_lane_width_m", 0.0))
        if self.target_lane_width_m != 0.0 and not (
            self.lane_width_min_m <= self.target_lane_width_m <= self.lane_width_max_m
        ):
            raise ValueError("target_lane_width_m must be 0 (legacy) or within validated lane width limits")
        self.max_heading_error_rad = float(rospy.get_param("~max_heading_error_rad", math.radians(22.0)))
        self.require_left_dashed = bool(rospy.get_param("~require_left_dashed", True))

        # Geometry sanity: the detected LEFT dashed divider must really be the
        # divider immediately next to the ego lane. The lane-info publisher can
        # fall back to a one-sided lane estimate; blindly shifting that estimated
        # centerline by one full lane width can command an accidental two-lane
        # jump. Highway lane-change geometry therefore uses the LEFT divider
        # itself and rejects a divider that is not adjacent to the ego lane.
        self.left_divider_expected_tol_m = float(
            rospy.get_param("~left_divider_expected_tol_m", 0.90)
        )
        self.inner_center_max_abs_y_m = float(
            rospy.get_param("~inner_center_max_abs_y_m", 0.90)
        )
        self.inner_lane_invalid_grace_s = float(
            rospy.get_param("~inner_lane_invalid_grace_s", 1.20)
        )
        self.inner_lane_grace_speed_mps = float(
            rospy.get_param("~inner_lane_grace_speed_mps", 1.50)
        )

        # V8 lane-hold stabilization.
        #
        # V7 stacked a temporal median + EMA + another path blend on top of the
        # camera publisher's own EMA.  That suppresses one-frame spikes, but it
        # also adds phase delay to a closed-loop lane-following controller: the
        # ego moves left while the delayed center estimate still asks for left,
        # then the same thing happens to the right.  The result can be a slow
        # left/right weave even though steering no longer spikes.
        #
        # V8 therefore does NOT temporally average ego-frame y samples.  It uses
        # only spatial smoothing inside the current camera frame, then stitches
        # that new geometry into the already-committed MAP-frame path.  The next
        # few metres remain physically fixed in the map; camera corrections are
        # blended in only farther ahead.  This gives Pure Pursuit a stable near
        # horizon without freezing the whole lane model.
        self.inner_stitch_keep_m = float(rospy.get_param("~inner_stitch_keep_m", 7.0))
        self.inner_stitch_blend_m = float(rospy.get_param("~inner_stitch_blend_m", 10.0))
        # V9 rollback: camera-following during INNER_HOLD, with short pose-
        # compensated temporal median instead of a hard map-reference lock.
        # The parameters above remain for the older mode/diagnostics; these
        # shorter look-ahead blending distances make camera correction effective.
        self.inner_camera_keep_m = max(1.0, float(rospy.get_param("~inner_camera_keep_m", 2.5)))
        self.inner_camera_blend_m = max(3.0, float(rospy.get_param("~inner_camera_blend_m", 7.5)))
        self.inner_camera_history_s = max(0.2, float(rospy.get_param("~inner_camera_history_s", 0.42)))
        self.inner_camera_max_jump_m = max(0.6, float(rospy.get_param("~inner_camera_max_jump_m", 1.4)))
        # Continuous map-aligned camera model over the existing robust 5-frame median.
        # Updates once per NEW camera frame, never once per 20Hz timer tick.
        self.inner_camera_smooth_tau_s = max(0.08,float(rospy.get_param("~inner_camera_smooth_tau_s",0.20)))
        self.inner_camera_max_step_m = max(0.10,float(rospy.get_param("~inner_camera_max_step_m",0.30)))
        self.inner_camera_smooth_rate_mps = max(0.7,float(rospy.get_param("~inner_camera_smooth_rate_mps",2.8)))
        self.inner_camera_reacquire_ramp_s = max(0.3,float(rospy.get_param("~inner_camera_reacquire_ramp_s",0.65)))
        # Conditional speed recovery only during stable camera-confirmed lane hold.
        self.inner_stable_speed_confirm_s = max(0.5,float(rospy.get_param("~inner_stable_speed_confirm_s",0.9)))
        self.inner_stable_speed_rise_mps2 = max(self.speed_rise_mps2 if hasattr(self,'speed_rise_mps2') else 0.8,
            float(rospy.get_param("~inner_stable_speed_rise_mps2",1.4)))
        # Only the INITIAL camera observation is compared to the committed lane
        # path. This is a coarse wrong-neighbour-lane reject, not V9's 0.75 m
        # freeze (which prevented correcting normal ~1 m offset).
        self.inner_camera_initial_max_delta_m = max(1.3, float(rospy.get_param(
            "~inner_camera_initial_max_delta_m", 2.1)))
        self.inner_spatial_window = int(rospy.get_param("~inner_spatial_window", 3))
        self.inner_spatial_max_dy_per_m = float(rospy.get_param("~inner_spatial_max_dy_per_m", 0.22))
        self.inner_recovery_timeout_s = float(rospy.get_param("~inner_recovery_timeout_s", 3.0))
        # Camera data are accepted as the SAME physical lane only after their
        # geometry agrees with the MAP-fixed target over consecutive observations.
        self.inner_association_max_y_m = max(0.2, float(rospy.get_param("~inner_association_max_y_m", 0.75)))
        self.inner_association_max_heading_rad = max(0.03, float(rospy.get_param(
            "~inner_association_max_heading_rad", 0.14)))
        self.inner_association_confirm_s = max(0.1, float(rospy.get_param(
            "~inner_association_confirm_s", 0.45)))
        self.inner_unverified_max_travel_m = max(1.0, float(rospy.get_param(
            "~inner_unverified_max_travel_m", 9.0)))
        self.inner_reference_max_cross_track_m = max(0.8, float(rospy.get_param(
            "~inner_reference_max_cross_track_m", 2.0)))
        self.inner_handoff_speed_mps = max(0.5, float(rospy.get_param(
            "~inner_handoff_speed_mps", 1.5)))
        self.center_sample_xs = (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0)

        self.vehicle_length_m = float(rospy.get_param("~vehicle_length_m", 4.635))
        self.vehicle_width_m = float(rospy.get_param("~vehicle_width_m", 1.892))
        self.vehicle_center_from_base_m = float(rospy.get_param("~vehicle_center_from_base_m", 1.50))

        self.change_start_m = float(rospy.get_param("~change_start_m", 3.0))
        self.change_entry_align_m = max(6.0, float(rospy.get_param("~change_entry_align_m", 18.0)))
        self.change_length_safety_factor = max(1.0, float(rospy.get_param(
            "~change_length_safety_factor", 1.15)))
        # Do not extrapolate a 25m camera observation to >100m just because
        # speed calls for a longer lane change; higher-speed use needs verified
        # long-range lane geometry/map alignment.
        self.change_max_unobserved_extension_m = max(0.0, float(rospy.get_param(
            "~change_max_unobserved_extension_m", 45.0)))
        self.change_min_length_m = float(rospy.get_param("~change_min_length_m", 22.0))
        self.change_max_length_m = float(rospy.get_param("~change_max_length_m", 26.0))
        self.change_time_s = float(rospy.get_param("~change_time_s", 5.5))
        self.change_post_hold_m = float(rospy.get_param("~change_post_hold_m", 10.0))
        # New manoeuvres must start from an actual near-field observation.
        self.change_observation_near_max_m = max(3.0, float(rospy.get_param(
            "~change_observation_near_max_m", 8.0)))
        # Avoid the ~11m first-waypoint gap seen in the third lane-change bag.
        self.change_max_waypoint_step_m = max(0.2, float(rospy.get_param(
            "~change_max_waypoint_step_m", 1.0)))
        self.change_complete_min_ratio = float(rospy.get_param("~change_complete_min_ratio", 0.72))
        self.change_center_error_m = float(rospy.get_param("~change_center_error_m", 0.45))
        self.change_heading_error_rad = float(rospy.get_param("~change_heading_error_rad", math.radians(10.0)))
        self.change_complete_confirm_s = float(rospy.get_param("~change_complete_confirm_s", 0.45))
        # Never let the finite committed lane-change path reach Pure Pursuit's
        # ordinary goal-stop condition.  Once the lateral transition is done and
        # only a few metres of post-hold remain, hand over to the receding-horizon
        # INNER_HOLD path.
        self.change_endpoint_guard_m = float(rospy.get_param("~change_endpoint_guard_m", 6.0))

        self.front_min_gap_m = float(rospy.get_param("~front_min_gap_m", 6.0))
        self.rear_min_gap_m = float(rospy.get_param("~rear_min_gap_m", 7.0))
        self.time_headway_s = float(rospy.get_param("~time_headway_s", 1.5))
        self.min_ttc_s = float(rospy.get_param("~min_ttc_s", 3.0))
        self.gap_search_range_m = float(rospy.get_param("~gap_search_range_m", 50.0))
        # Lane-membership gates use obstacle CENTER distance from each lane center.
        # The old footprint-style gate was ~3 m wide and could classify an
        # adjacent-lane car as the current-lane lead, which caused false stops.
        self.current_lane_center_gate_m = float(rospy.get_param("~current_lane_center_gate_m", 1.15))
        self.target_lane_center_gate_m = float(rospy.get_param("~target_lane_center_gate_m", 1.50))
        self.target_lane_lateral_extra_m = float(rospy.get_param("~target_lane_lateral_extra_m", 0.35))

        self.follow_standstill_gap_m = float(rospy.get_param("~follow_standstill_gap_m", 4.0))
        self.follow_time_headway_s = float(rospy.get_param("~follow_time_headway_s", 1.6))
        self.follow_gain = float(rospy.get_param("~follow_gain", 0.35))
        self.follow_search_m = float(rospy.get_param("~follow_search_m", 60.0))
        self.emergency_gap_m = float(rospy.get_param("~emergency_gap_m", 1.5))
        self.emergency_ttc_s = float(rospy.get_param("~emergency_ttc_s", 1.0))
        self.lead_overlap_tolerance_m = float(rospy.get_param("~lead_overlap_tolerance_m", 0.25))
        self.change_settle_m = float(rospy.get_param("~change_settle_m", 6.0))
        self.speed_rise_mps2 = float(rospy.get_param("~speed_rise_mps2", 0.8))
        self.speed_fall_mps2 = float(rospy.get_param("~speed_fall_mps2", 1.8))

        self.collision_long_margin_m = float(rospy.get_param("~collision_long_margin_m", 0.5))
        self.collision_lat_margin_m = float(rospy.get_param("~collision_lat_margin_m", 0.35))
        self.max_lateral_accel_mps2 = float(rospy.get_param("~max_lateral_accel_mps2", 2.5))

        self.release_global_d_m = float(rospy.get_param("~release_global_d_m", 0.55))
        self.release_confirm_s = float(rospy.get_param("~release_confirm_s", 0.6))
        # Event-based re-arm after each completed lane change.  A new left
        # change is considered only after the ego has actually settled in the
        # new lane; there is no fixed target_left_lane_changes counter anymore.
        self.left_change_rearm_min_s = float(rospy.get_param("~left_change_rearm_min_s", 2.5))
        self.left_change_rearm_min_m = float(rospy.get_param("~left_change_rearm_min_m", 8.0))
        self.left_change_stable_lateral_m = float(rospy.get_param("~left_change_stable_lateral_m", 0.35))
        self.left_change_stable_heading_rad = float(rospy.get_param("~left_change_stable_heading_rad", math.radians(7.0)))
        self.left_change_stable_confirm_s = float(rospy.get_param("~left_change_stable_confirm_s", 0.8))
        # A 1-2 frame camera dropout must not restart the whole re-arm timer.
        # Safety/geometry checks are still repeated at the moment of commitment.
        self.inner_rearm_dropout_grace_s = max(0.0, float(rospy.get_param(
            "~inner_rearm_dropout_grace_s", 0.20)))
        # Optional conservative gap-shaping in INNER_HOLD while waiting for a
        # second safe left change. Keep disabled by default on the working launch.
        self.inner_hold_gap_shaping_enabled = bool(rospy.get_param(
            "~inner_hold_gap_shaping_enabled", False))
        self.inner_hold_gap_max_slowdown_mps = max(0.0, float(rospy.get_param(
            "~inner_hold_gap_max_slowdown_mps", 0.75)))
        self.inner_hold_gap_rear_min_ttc_s = max(self.min_ttc_s, float(rospy.get_param(
            "~inner_hold_gap_rear_min_ttc_s", 4.0)))
        self.rejoin_start_global_d_m = float(rospy.get_param("~rejoin_start_global_d_m", 1.8))
        self.rejoin_length_m = float(rospy.get_param("~rejoin_length_m", 18.0))
        self.rejoin_complete_global_d_m = float(rospy.get_param("~rejoin_complete_global_d_m", 0.45))

        self.latest_base_path: Optional[RosPath] = None
        self.base_path_at: Optional[rospy.Time] = None
        self.base_stop = True
        self.base_stop_at: Optional[rospy.Time] = None
        self.latest_odom: Optional[Odometry] = None
        self.odom_at: Optional[rospy.Time] = None
        self.latest_obstacles: Optional[LidarObstacleArray] = None
        self.obstacles_at: Optional[rospy.Time] = None
        self.lane_info = None
        self.lane_info_at: Optional[rospy.Time] = None
        self.highway_environment = False
        self.highway_at: Optional[rospy.Time] = None
        self.highway_request = False
        self.merge_available = False
        self.merge_unavailable = True
        self.merge_at: Optional[rospy.Time] = None

        self.state = self.OFF
        self.highway_true_since: Optional[rospy.Time] = None
        self.ready_since: Optional[rospy.Time] = None
        self.complete_since: Optional[rospy.Time] = None
        self.release_since: Optional[rospy.Time] = None
        self.lane_changes_done = 0
        self.completed_once = False
        self.inner_hold_started_at: Optional[rospy.Time] = None
        self.inner_stable_since: Optional[rospy.Time] = None
        self.inner_unstable_since: Optional[rospy.Time] = None

        self.committed_path: Optional[RosPath] = None
        self.committed_speed_mps = self.cruise_speed_mps
        self.committed_change_length_m = self.change_min_length_m
        self.committed_rejoin_path: Optional[RosPath] = None
        self.rejoin_travel_m = 0.0
        self.last_rejoin_xy: Optional[Tuple[float, float]] = None
        self.change_travel_m = 0.0
        self.last_change_xy: Optional[Tuple[float, float]] = None
        self.inner_hold_travel_m = 0.0
        self.last_hold_xy: Optional[Tuple[float, float]] = None

        self.last_output_speed = self.cruise_speed_mps
        self.last_timer_time: Optional[rospy.Time] = None
        self.last_inner_path: Optional[RosPath] = None
        self.lane_invalid_since: Optional[rospy.Time] = None
        self.inner_handoff_pending = False
        self.inner_camera_match_since: Optional[rospy.Time] = None
        self.inner_camera_match_last_msg_at: Optional[rospy.Time] = None
        self.inner_camera_match_samples = 0
        self.inner_association_diag: dict = {}
        self.inner_unverified_travel_m = 0.0
        self.inner_reference_cross_track_m: Optional[float] = None

        # V9 camera-follow recovery: diagnostic samples and map-aligned median.
        self.center_sample_history: Deque[Dict[float, float]] = deque(maxlen=3)
        self.filtered_center_y: Dict[float, float] = {}
        self.last_good_center_local: Optional[List[Tuple[float, float]]] = None
        # (receipt time, map-frame observed lane centerline). Store one sample
        # only for each new FRESH camera frame, never once per 20Hz timer tick.
        self.inner_camera_history: Deque[Tuple[float,List[Tuple[float,float]]]] = deque(maxlen=5)
        self.inner_camera_filter_reason = "not_started"
        self.inner_camera_rejected_count = 0
        self.inner_camera_samples = 0
        self.inner_smoothed_camera_map: Optional[List[Tuple[float,float]]] = None
        self.inner_camera_last_smoothed_at: Optional[float] = None
        self.inner_camera_last_step_m = 0.0
        self.inner_camera_reacquire_started_at: Optional[float] = None
        self.inner_speed_stable_since: Optional[rospy.Time] = None
        self.inner_filtered_y8_m: Optional[float] = None
        self.inner_speed_lateral_limit_mps: Optional[float] = None
        # EKF pose history for timestamp-aligned camera observations. The v2
        # publisher's measured camera timestamp precedes ROS delivery (~0.1 s).
        self.inner_camera_pose_history: Deque[Tuple[float,float,float,float]] = deque(maxlen=200)

        self.path_pub = rospy.Publisher("~active_path", RosPath, queue_size=1)
        self.trajectory_pub = rospy.Publisher("~trajectory", String, queue_size=1)
        self.fast_change_pub = rospy.Publisher("~fast_change_active", Bool, queue_size=1)
        self.lead_brake_pub = rospy.Publisher("~lead_brake_required", Bool, queue_size=1)
        self.lead_emergency_pub = rospy.Publisher("~lead_emergency_brake", Bool, queue_size=1)
        self.stop_pub = rospy.Publisher("~stop_required", Bool, queue_size=1)
        self.speed_pub = rospy.Publisher("~target_speed_mps", Float64, queue_size=1)
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.state_pub = rospy.Publisher("~state", String, queue_size=1)

        if self.base_trajectory_topic:
            rospy.Subscriber(self.base_trajectory_topic, String, self._base_trajectory_cb, queue_size=1)
        else:
            rospy.Subscriber(self.base_path_topic, RosPath, self._base_path_cb, queue_size=1)
            rospy.Subscriber(self.base_stop_topic, Bool, self._base_stop_cb, queue_size=1)
        if self.route_gate_required:
            rospy.Subscriber(self.route_gate_topic, Bool, self._route_gate_cb, queue_size=1)
        if self.handoff_gate_required:
            rospy.Subscriber(self.handoff_topic, Bool, self._handoff_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber(self.obstacle_topic, LidarObstacleArray, self._obstacles_cb, queue_size=1)
        rospy.Subscriber(self.lane_info_topic, String, self._lane_info_cb, queue_size=1)
        rospy.Subscriber(self.highway_topic, Bool, self._highway_cb, queue_size=1)
        rospy.Subscriber(self.highway_request_topic, Bool, self._highway_request_cb, queue_size=1)
        rospy.Subscriber(self.merge_available_topic, Bool, self._merge_available_cb, queue_size=1)
        rospy.Subscriber(self.merge_unavailable_topic, Bool, self._merge_unavailable_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1.0)), self._tick)
        # Keep safety publishers fresh independently of expensive V9 planning.
        self.lead_safety_timer = rospy.Timer(rospy.Duration(0.1), self._publish_lead_safety)
        rospy.logwarn(
            "Highway lane strategy V8: map-stitched lane hold + auto recovery + event LEFT dashed re-arm, cruise=%.2f m/s; camera files are read-only inputs",
            self.cruise_speed_mps,
        )

    @plan_locked
    def _base_trajectory_cb(self, msg: String) -> None:
        try:
            path, stop, _speed, reason = read_trajectory(
                msg.data, self.map_frame, rospy.get_time(), self.base_timeout_s,
                path_type=RosPath, pose_type=PoseStamped, stamp_type=rospy.Time,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.latest_base_path = None
            self.base_path_at = None
            self.base_stop = True
            self.base_stop_at = None
            self.base_reason = "invalid_base_trajectory:" + str(exc)
            rospy.logwarn_throttle(1.0, "%s", self.base_reason)
            return
        self.latest_base_path = path
        self.base_path_at = path.header.stamp
        self.base_stop = bool(stop)
        self.base_stop_at = path.header.stamp
        self.base_reason = str(reason)

    def _route_gate_cb(self, msg: Bool) -> None:
        self.route_gate_active = bool(msg.data)
        self.route_gate_at = rospy.Time.now()

    def _handoff_cb(self, msg: Bool) -> None:
        self.handoff_due = bool(msg.data)
        self.handoff_at = rospy.Time.now()

    def _route_gate_present(self) -> bool:
        return not self.route_gate_required or (
            self.route_gate_active and self.route_gate_at is not None
            and 0.0 <= (rospy.Time.now() - self.route_gate_at).to_sec() <= self.route_gate_timeout_s
        )

    def _handoff_permitted(self) -> bool:
        return not self.handoff_gate_required or (
            self.handoff_due and self.handoff_at is not None
            and 0.0 <= (rospy.Time.now() - self.handoff_at).to_sec() <= self.handoff_timeout_s
        )

    @plan_locked
    def _publish_lead_safety(self, _event) -> None:
        now = rospy.Time.now()
        active = self.state not in (self.OFF, self.DONE)
        brake = emergency = False
        if active:
            if not self._fresh(self.odom_at, self.odom_timeout_s, now) or not self._fresh(
                self.obstacles_at, self.obstacle_timeout_s, now
            ):
                brake = emergency = True
            else:
                lead, gap, ttc = self._current_lane_lead()
                if lead is not None and gap is not None:
                    _, _, _, ego_speed = self._odom_pose()
                    brake, emergency = lead_brake_decision({
                        "lead": lead.oid, "gap": gap,
                        "desired_gap": self.follow_standstill_gap_m + self.follow_time_headway_s * ego_speed,
                        "ttc": ttc,
                    }, self.emergency_gap_m, self.emergency_ttc_s)
        self.lead_brake_pub.publish(Bool(data=bool(brake)))
        self.lead_emergency_pub.publish(Bool(data=bool(emergency)))

    def _base_path_cb(self, msg: RosPath) -> None:
        if self.base_trajectory_topic:
            return
        self.latest_base_path = msg
        self.base_path_at = rospy.Time.now()

    def _base_stop_cb(self, msg: Bool) -> None:
        if self.base_trajectory_topic:
            return
        self.base_stop = bool(msg.data)
        self.base_stop_at = rospy.Time.now()

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.odom_at = rospy.Time.now()
        pose=msg.pose.pose
        try:
            stamp=float(msg.header.stamp.to_sec())
        except (AttributeError,TypeError,ValueError):
            stamp=self.odom_at.to_sec()
        if abs(stamp-self.odom_at.to_sec())>0.5:
            stamp=self.odom_at.to_sec()
        self.inner_camera_pose_history.append((stamp,
            float(pose.position.x),float(pose.position.y),yaw_from_quaternion(pose.orientation)))

    def _obstacles_cb(self, msg: LidarObstacleArray) -> None:
        self.latest_obstacles = msg
        self.obstacles_at = rospy.Time.now()

    def _lane_info_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.lane_info = data
                self.lane_info_at = rospy.Time.now()
                # The committed lane-change path must not be contaminated by
                # the camera detector switching which lane it calls "ego" while
                # we are physically crossing the divider.  Freeze the temporal
                # lane-center model during LANE_CHANGE and resume sampling after
                # the maneuver.
                if self.state != self.LANE_CHANGE:
                    self._update_center_filter(data)
                if self.state == self.INNER_HOLD:
                    self._accept_temporal_camera_observation(data)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "lane_info JSON parse failed: %s", exc)

    def _highway_cb(self, msg: Bool) -> None:
        self.highway_environment = bool(msg.data)
        self.highway_at = rospy.Time.now()

    def _highway_request_cb(self, msg: Bool) -> None:
        self.highway_request = bool(msg.data)

    def _merge_available_cb(self, msg: Bool) -> None:
        self.merge_available = bool(msg.data)
        self.merge_at = rospy.Time.now()

    def _merge_unavailable_cb(self, msg: Bool) -> None:
        self.merge_unavailable = bool(msg.data)
        self.merge_at = rospy.Time.now()

    def _fresh(self, stamp: Optional[rospy.Time], timeout: float, now: rospy.Time) -> bool:
        return stamp is not None and (now - stamp).to_sec() <= timeout

    def _activation_present(self) -> bool:
        return self._route_gate_present() and bool(
            self.force_highway_active or self.highway_environment or self.highway_request
        )

    def _lane_valid(self, now: rospy.Time) -> Tuple[bool, str]:
        if self.lane_info is None or not self._fresh(self.lane_info_at, self.lane_info_timeout_s, now):
            return False, "lane_info_missing_or_stale"
        d = self.lane_info
        if not bool(d.get("lane_valid", False)):
            return False, "lane_invalid"
        if float(d.get("confidence", 0.0) or 0.0) < self.min_lane_confidence:
            return False, "lane_confidence"
        width = d.get("lane_width_m")
        if width is None or not (self.lane_width_min_m <= float(width) <= self.lane_width_max_m):
            return False, "lane_width"
        heading = d.get("heading_error_rad")
        if heading is None or abs(float(heading)) > self.max_heading_error_rad:
            return False, "lane_heading"
        pts = d.get("centerline_points") or []
        if len(pts) < 3:
            return False, "centerline_short"
        return True, "ok"

    def _left_dashed_ok(self) -> Tuple[bool, str]:
        if not self.require_left_dashed:
            return True, "disabled"
        left = (self.lane_info or {}).get("left_lane") or {}
        if not bool(left.get("detected", False)):
            return False, "left_not_detected"
        if left.get("dashed") is True:
            return True, "ok"
        return False, "left_not_dashed"

    def _extract_centerline_local(self, data: Optional[dict] = None) -> List[Tuple[float, float]]:
        raw = (data if data is not None else (self.lane_info or {})).get("centerline_points") or []
        pts: List[Tuple[float, float]] = [(0.0, 0.0)]
        for p in raw:
            try:
                x, y = float(p[0]), float(p[1])
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y) and x > 0.5:
                pts.append((x, y))
        pts.sort(key=lambda q: q[0])
        out: List[Tuple[float, float]] = []
        for p in pts:
            if not out or math.hypot(p[0]-out[-1][0], p[1]-out[-1][1]) > 0.15:
                out.append(p)
        return out

    def _update_center_filter(self, data: dict) -> None:
        """Keep a tiny diagnostic history only; V8 does no temporal y filtering."""
        raw = self._extract_centerline_local(data)
        if len(raw) < 3:
            return
        sample: Dict[float, float] = {}
        for x in self.center_sample_xs:
            y = interp_y(raw, x)
            if y is not None and math.isfinite(float(y)):
                sample[float(x)] = float(y)
        if sample:
            self.center_sample_history.append(sample)

    def _spatial_centerline_local(self) -> List[Tuple[float, float]]:
        """Smooth only along x inside the newest camera frame.

        This removes local waypoint wiggles without mixing ego-frame samples from
        different times/poses.  The camera publisher already applies temporal EMA.
        """
        raw = self._extract_centerline_local()
        if len(raw) < 3:
            return raw

        xs = []
        x = 1.0
        xmax = min(25.0, max(p[0] for p in raw))
        while x <= xmax + 1e-6:
            y = interp_y(raw, x)
            if y is not None and math.isfinite(float(y)):
                xs.append((x, float(y)))
            x += 1.0
        if len(xs) < 3:
            return raw

        # Odd spatial median window (default 3 samples == 3 m).  This is not a
        # temporal filter, so it adds essentially no steering phase lag.
        win = max(1, int(self.inner_spatial_window))
        if win % 2 == 0:
            win += 1
        half = win // 2
        med = []
        for i, (x, _) in enumerate(xs):
            lo = max(0, i-half)
            hi = min(len(xs), i+half+1)
            med.append((x, float(median([xs[j][1] for j in range(lo, hi)]))))

        # Bound local slope so a bad far waypoint cannot create a sharp kink.
        out: List[Tuple[float, float]] = [(0.0, 0.0)]
        prev_x, prev_y = 0.0, 0.0
        for x, y in med:
            dx = max(1e-3, x-prev_x)
            max_dy = self.inner_spatial_max_dy_per_m * dx
            y = prev_y + clamp(y-prev_y, -max_dy, max_dy)
            out.append((x, y))
            prev_x, prev_y = x, y
        return out

    def _centerline_local(self) -> List[Tuple[float, float]]:
        return self._spatial_centerline_local()

    def _lane_hold_valid(self, now: rospy.Time) -> Tuple[bool, str]:
        """Validity needed only to KEEP the current lane.

        Unlike a new lane-change decision, holding the lane does not require a
        measured lane_width.  This avoids permanent stops when an overtaking car
        briefly occludes one boundary and the publisher switches to a one-sided
        centerline estimate.
        """
        if self.lane_info is None or not self._fresh(self.lane_info_at, self.lane_info_timeout_s, now):
            return False, "lane_info_missing_or_stale"
        d = self.lane_info
        if d.get("output_status") != "FRESH":
            return False, "camera_not_fresh"
        if not bool(d.get("lane_valid", False)):
            return False, "lane_invalid"
        if float(d.get("confidence", 0.0) or 0.0) < self.min_lane_confidence:
            return False, "lane_confidence"
        heading = d.get("heading_error_rad")
        if heading is None or abs(float(heading)) > self.max_heading_error_rad:
            return False, "lane_heading"
        if len(self._centerline_local()) < 3:
            return False, "centerline_short"
        return True, "ok"

    def _camera_raw_center_local(self, data: dict) -> List[Tuple[float,float]]:
        """The measured center, WITHOUT injecting ego (0,0) into the lane."""
        points=[]
        for p in data.get("centerline_points") or []:
            try:
                x,y=float(p[0]),float(p[1])
            except (TypeError,ValueError,IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y) and 0.5 <= x <= 35.0:
                points.append((x,y))
        points.sort()
        return [p for i,p in enumerate(points) if i == 0 or p[0]-points[i-1][0]>0.15]

    def _map_camera_polyline(self, local: Sequence[Tuple[float,float]],
                             measured_pose: Optional[Tuple[float,float,float]] = None) -> List[Tuple[float,float]]:
        ex,ey,yaw=(self._odom_pose()[:3] if measured_pose is None else measured_pose)
        c,si=math.cos(yaw),math.sin(yaw)
        return [(ex+c*x-si*y,ey+si*x+c*y) for x,y in local]

    def _pose_at_camera_timestamp(self, source_time: float) -> Optional[Tuple[float,float,float]]:
        """Interpolate the EKF pose at the camera receive-wall observation time."""
        history=list(self.inner_camera_pose_history)
        if not history:
            return None
        for a,b in zip(history,history[1:]):
            ta,xa,ya,ha=a
            tb,xb,yb,hb=b
            if ta<=source_time<=tb and tb>ta+1e-4:
                if tb-ta>0.16:
                    break
                alpha=(source_time-ta)/(tb-ta)
                return (xa+alpha*(xb-xa),ya+alpha*(yb-ya),
                        ha+alpha*wrap_angle(hb-ha))
        nearest=min(history,key=lambda row:abs(row[0]-source_time))
        if abs(nearest[0]-source_time)<=0.06:
            return nearest[1:]
        return None

    def _local_camera_polyline(self, mapped: Sequence[Tuple[float,float]]) -> List[Tuple[float,float]]:
        ex,ey,yaw,_=self._odom_pose()
        c,si=math.cos(yaw),math.sin(yaw)
        out=[]
        for mx,my in mapped:
            dx,dy=mx-ex,my-ey
            out.append((c*dx+si*dy,-si*dx+c*dy))
        out.sort()
        return out

    @staticmethod
    def _camera_interp_within(points: Sequence[Tuple[float,float]], x: float) -> Optional[float]:
        """Never claim a boundary was observed outside its x-range."""
        if len(points)<2 or x<points[0][0] or x>points[-1][0]:
            return None
        return interp_y(points,x)

    def _accept_temporal_camera_observation(self, info: dict) -> None:
        """Register a *new* FRESH camera center at the ego pose at receipt.

        Storing the observation in map coordinates prevents the simple but
        incorrect operation of averaging ego-relative y across vehicle motion.
        """
        now=rospy.Time.now()
        if (self.latest_odom is None or self.odom_at is None or
                not self._fresh(self.odom_at,self.odom_timeout_s,now)):
            self.inner_camera_filter_reason="odom_not_fresh"
            return
        if info.get("output_status") != "FRESH" or not info.get("lane_valid",False):
            self.inner_camera_filter_reason="camera_not_fresh"
            return
        if float(info.get("confidence",0) or 0)<self.min_lane_confidence:
            self.inner_camera_filter_reason="low_confidence"
            return
        heading=info.get("heading_error_rad")
        if heading is None or abs(float(heading))>self.max_heading_error_rad:
            self.inner_camera_filter_reason="bad_heading"
            return
        raw=self._camera_raw_center_local(info)
        if len(raw)<3:
            self.inner_camera_filter_reason="camera_path_short"
            return
        measured_pose=None
        if info.get("observation_time_source")=="camera_receive_wall":
            try:
                cam_time=float(info["timestamp"])
            except (ValueError,KeyError,TypeError):
                self.inner_camera_filter_reason="camera_timestamp_missing"
                return
            if abs(now.to_sec()-cam_time)>0.55:
                self.inner_camera_filter_reason="camera_timestamp_stale"
                return
            measured_pose=self._pose_at_camera_timestamp(cam_time)
            if measured_pose is None:
                self.inner_camera_filter_reason="camera_pose_sync_missing"
                return
        current=self._map_camera_polyline(raw, measured_pose)
        cutoff=now.to_sec()-self.inner_camera_history_s
        while self.inner_camera_history and self.inner_camera_history[0][0]<cutoff:
            self.inner_camera_history.popleft()
        previous=None
        if self.inner_camera_history:
            previous=self._local_camera_polyline(self.inner_camera_history[-1][1])
        else:
            # On FIRST post-change observation, reject only a drastic ~two-lane
            # disagreement with the committed physical lane. A 1 m lane offset
            # is deliberately allowed so the camera can correct it.
            previous=self._reference_map_to_local(self.last_inner_path)
        if len(previous)>=3:
            differences=[]
            for x in (8.0,14.0):
                a=self._camera_interp_within(raw,x)
                b=self._camera_interp_within(previous,x)
                if a is not None and b is not None:
                    differences.append(abs(a-b))
            limit=(self.inner_camera_max_jump_m if self.inner_camera_history
                   else self.inner_camera_initial_max_delta_m)
            if differences and max(differences)>limit:
                self.inner_camera_rejected_count+=1
                self.inner_camera_filter_reason="camera_lane_jump_rejected"
                return
        self.inner_camera_history.append((now.to_sec(),current))
        self.inner_camera_filter_reason="accepted"

    def _temporal_camera_center_local(self, now: rospy.Time) -> List[Tuple[float,float]]:
        """Map-aligned median + one update per fresh frame of a causal low-lag smoother.

        A median has step discontinuities when its representative sample swaps.
        This smoother imposes BOTH a time constant and a bounded per-observation
        innovation. All histories are transformed using odometry to the *current*
        ego frame before combining; no averaging of old raw ego coordinates.
        """
        cutoff=now.to_sec()-self.inner_camera_history_s
        while self.inner_camera_history and self.inner_camera_history[0][0]<cutoff:
            self.inner_camera_history.popleft()
        self.inner_camera_samples=len(self.inner_camera_history)
        if self.inner_camera_samples<2:
            # Do not claim a camera path when observations expired. Keep normal
            # 3-second lane-loss recovery in the original INNER_HOLD branch.
            if (not self.inner_camera_history or
                    (self.inner_camera_last_smoothed_at is not None and
                     now.to_sec()-self.inner_camera_last_smoothed_at>0.65)):
                self.inner_smoothed_camera_map=None
                self.inner_camera_last_smoothed_at=None
            return []
        aligned=[self._local_camera_polyline(points) for _,points in self.inner_camera_history]
        median_local=[]
        for x in range(3,26):
            ys=[self._camera_interp_within(pts,float(x)) for pts in aligned]
            ys=[z for z in ys if z is not None and math.isfinite(z)]
            if len(ys)>=2:
                median_local.append((float(x),float(median(ys))))
        if len(median_local)<6:
            return []

        newest=float(self.inner_camera_history[-1][0])
        if (self.inner_camera_last_smoothed_at is not None and
                newest-self.inner_camera_last_smoothed_at>0.65):
            self.inner_smoothed_camera_map=None
            self.inner_camera_last_smoothed_at=None
        if (self.inner_camera_last_smoothed_at is not None and
                newest<=self.inner_camera_last_smoothed_at+1e-6):
            return self._local_camera_polyline(self.inner_smoothed_camera_map) if self.inner_smoothed_camera_map else median_local

        previous=[]
        if self.inner_smoothed_camera_map:
            previous=self._local_camera_polyline(self.inner_smoothed_camera_map)
        # After a real camera dropout, never initialize an observation model
        # from a potentially stale/extrapolated committed path. The existing
        # short-path stitch already supplies continuity at handoff.
        if self.inner_camera_last_smoothed_at is None:
            # Short soft handoff after initial detection / a camera dropout.
            self.inner_camera_reacquire_started_at=now.to_sec()
            dt=0.12
        else:
            dt=clamp(newest-self.inner_camera_last_smoothed_at,0.04,0.25)
        alpha=1.0-math.exp(-dt/self.inner_camera_smooth_tau_s)
        step_limit=min(self.inner_camera_max_step_m,self.inner_camera_smooth_rate_mps*dt)
        out=[]; max_step=0.0
        for x,y in median_local:
            old=self._camera_interp_within(previous,x)
            if old is None:
                # Near the very end of previous camera paths, use new geometry.
                yn=y
            else:
                change=clamp(alpha*(y-old),-step_limit,step_limit)
                yn=old+change
                max_step=max(max_step,abs(change))
            out.append((x,yn))
        self.inner_smoothed_camera_map=self._map_camera_polyline(out,measured_pose=None)
        self.inner_camera_last_smoothed_at=newest
        self.inner_camera_last_step_m=max_step
        return out

    def _inner_adaptive_speed_limit(self, raw_center_y: Optional[float],
                                    filtered_local: Sequence[Tuple[float,float]],
                                    adaptive: float, heading: Optional[float],
                                    front_follow: dict, now: rospy.Time) -> float:
        """Reduce prolonged crawl only if alignment has recovered. Never bypass lead speed."""
        y=interp_y(filtered_local,8.0)
        self.inner_filtered_y8_m=y
        if y is None or not math.isfinite(y):
            self.inner_speed_stable_since=None
            self.inner_speed_lateral_limit_mps=self.inner_handoff_speed_mps
            return min(adaptive,self.inner_handoff_speed_mps)
        raw=float(raw_center_y) if raw_center_y is not None and math.isfinite(float(raw_center_y)) else float(y)
        # Retain some caution from the *current* frame, even if median is calm.
        err=max(abs(y),0.65*abs(raw))
        limit=lane_center_speed_limit(err,adaptive,self.inner_handoff_speed_mps)
        if heading is None or not math.isfinite(float(heading)) or abs(float(heading))>0.15:
            limit=min(limit,3.5)
        self.inner_speed_lateral_limit_mps=limit
        stable=(err<=0.38 and heading is not None and
                abs(float(heading))<0.09 and self.inner_camera_samples>=3 and
                self.inner_camera_last_step_m<0.11 and
                not front_follow.get('lead'))
        if stable:
            if self.inner_speed_stable_since is None:
                self.inner_speed_stable_since=now
        else:
            self.inner_speed_stable_since=None
        return min(adaptive,limit)

    def _reference_map_to_local(self, path: Optional[RosPath]) -> List[Tuple[float, float]]:
        """Map-fixed centerline expressed in ego axes WITHOUT an invented (0,0) point."""
        if path is None or self.latest_odom is None:
            return []
        ex, ey, yaw, _ = self._odom_pose()
        c, si = math.cos(yaw), math.sin(yaw)
        points = []
        for ps in path.poses:
            dx = float(ps.pose.position.x) - ex
            dy = float(ps.pose.position.y) - ey
            x, y = c*dx + si*dy, -si*dx + c*dy
            if math.isfinite(x) and math.isfinite(y) and x >= -4.0:
                points.append((x,y))
        points.sort(key=lambda p: p[0])
        # Do not substitute the ego position for the lane center at x=0.
        return [p for i,p in enumerate(points) if i==0 or p[0] - points[i-1][0] > 0.08]

    def _reference_cross_track(self, path: Optional[RosPath]) -> Optional[float]:
        """Distance from ego to nearest physical segment of a map-frame reference."""
        if path is None or self.latest_odom is None or len(path.poses) < 2:
            return None
        ex,ey,_,_ = self._odom_pose()
        pts = [(float(z.pose.position.x), float(z.pose.position.y)) for z in path.poses]
        best = float('inf')
        for (ax,ay),(bx,by) in zip(pts,pts[1:]):
            dx,dy=bx-ax,by-ay
            den=dx*dx+dy*dy
            if den < 1e-9: continue
            t=clamp(((ex-ax)*dx+(ey-ay)*dy)/den,0.0,1.0)
            best=min(best,math.hypot(ex-(ax+t*dx),ey-(ay+t*dy)))
        return best if math.isfinite(best) else None

    def _maintain_reference_map(self, path: Optional[RosPath], now: rospy.Time) -> Optional[RosPath]:
        """Receding map horizon: trim passed points and extend tangent, never move lane onto ego."""
        if path is None or self.latest_odom is None or len(path.poses) < 2:
            return None
        ex,ey,_,_ = self._odom_pose()
        pts=[(float(z.pose.position.x),float(z.pose.position.y)) for z in path.poses]
        best=(float('inf'), 0, 0.0)
        for i in range(len(pts)-1):
            ax,ay=pts[i]; bx,by=pts[i+1]
            dx,dy=bx-ax,by-ay; den=dx*dx+dy*dy
            if den < 1e-8: continue
            t=clamp(((ex-ax)*dx+(ey-ay)*dy)/den,0.0,1.0)
            dist=math.hypot(ex-(ax+t*dx),ey-(ay+t*dy))
            if dist < best[0]: best=(dist,i,t)
        if not math.isfinite(best[0]): return None
        i=best[1]; t=best[2]
        ax,ay=pts[i]; bx,by=pts[i+1]
        foot=(ax+t*(bx-ax),ay+t*(by-ay))
        ahead=[foot]+pts[i+1:]
        # Discard duplicates at the projected foot and keep a fixed physical line.
        ahead=[p for j,p in enumerate(ahead) if j==0 or math.hypot(p[0]-ahead[j-1][0],p[1]-ahead[j-1][1])>0.05]
        if len(ahead)<2: return None
        arc=polyline_arclength(ahead)
        # Model extrapolation is bounded by the preexisting 3-second recovery
        # timeout / unverified travel gate. The curve itself is never translated.
        while arc[-1]<45.0:
            tx,ty=tangent_at(ahead,len(ahead)-1)
            x,y=ahead[-1]
            ahead.append((x+tx,y+ty))
            arc.append(arc[-1]+1.0)
        out=RosPath();out.header.stamp=now;out.header.frame_id=self.map_frame
        for x,y in ahead:
            ps=PoseStamped();ps.header=out.header
            ps.pose.position.x=x;ps.pose.position.y=y;ps.pose.orientation.w=1.0
            out.poses.append(ps)
        return out

    def _camera_reference_agreement(self) -> Tuple[bool,str,dict]:
        """Reject lane-ID flips by comparing camera geometry to the held physical lane."""
        info=self.lane_info or {}
        diag={"camera_status": info.get("output_status")}
        if info.get("output_status") != "FRESH":
            return False,"camera_not_fresh",diag
        # Use observed lane geometry, NOT a target recomputed by lateral offsets.
        camera=self._centerline_local()
        reference=self._reference_map_to_local(self.last_inner_path)
        if len(camera)<3 or len(reference)<3:
            return False,"association_path_short",diag
        errors=[]
        for x in (8.0,15.0):
            yc,yr=interp_y(camera,x),interp_y(reference,x)
            if yc is None or yr is None:
                return False,"association_missing_sample",diag
            errors.append(round(float(yc-yr),3))
        hc=heading_on_polyline(camera,12.0)
        hr=heading_on_polyline(reference,12.0)
        if hc is None or hr is None:
            return False,"association_heading_missing",diag
        dh=abs(wrap_angle(hc-hr))
        diag.update({"delta_y8_m":errors[0],"delta_y15_m":errors[1],
                     "delta_heading_rad":round(dh,3)})
        if max(abs(z) for z in errors)>self.inner_association_max_y_m:
            return False,"different_physical_lane",diag
        if dh>self.inner_association_max_heading_rad:
            return False,"different_lane_heading",diag
        return True,"ok",diag

    def _smooth_inner_path(self, target_local: Sequence[Tuple[float, float]], now: rospy.Time) -> Optional[RosPath]:
        """Blend pose-compensated camera median into a SHORT map-fixed near horizon."""
        target=self._extend_local_polyline(target_local,45.0)
        prev=self._reference_map_to_local(self.last_inner_path)
        if len(target)<3 or len(prev)<3:
            return None
        keep=self.inner_camera_keep_m
        blend_len=self.inner_camera_blend_m
        phase=(1.0 if self.inner_camera_reacquire_started_at is None else
               smoothstep5((now.to_sec()-self.inner_camera_reacquire_started_at)/self.inner_camera_reacquire_ramp_s))
        joined=[]
        x=0.0
        while x<=45.0+1e-6:
            yp=interp_y(prev,x);yt=interp_y(target,x)
            if yp is None: return None
            if yt is None: yt=yp
            if x<=keep: y=yp
            elif x>=keep+blend_len:y=(1.0-phase)*yp+phase*yt
            else:
                w=smoothstep5((x-keep)/blend_len)*phase
                y=(1.0-w)*yp+w*yt
            joined.append((x,float(y)))
            x+=1.0
        # Crucially, do not prepend (0,0): ego can be offset from lane centre.
        return self._local_to_map(joined,now)

    def _recover_inner_path(self, now: rospy.Time) -> Optional[RosPath]:
        """Advance the same map-frame lane during short camera dropouts; no ego recenter."""
        return self._maintain_reference_map(self.last_inner_path,now)

    def _boundary_local(self, key: str) -> List[Tuple[float, float]]:
        """Return a lane boundary in base_link and extrapolate it back to x=0."""
        raw = (self.lane_info or {}).get(key) or []
        pts: List[Tuple[float, float]] = []
        for q in raw:
            try:
                x, y = float(q[0]), float(q[1])
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y) and x > 0.2:
                pts.append((x, y))
        pts.sort(key=lambda q: q[0])
        out: List[Tuple[float, float]] = []
        for q in pts:
            if not out or math.hypot(q[0]-out[-1][0], q[1]-out[-1][1]) > 0.15:
                out.append(q)
        if len(out) >= 2 and out[0][0] > 0.5:
            x0, y0 = out[0]
            x1, y1 = out[1]
            dx = x1 - x0
            slope = (y1-y0)/dx if abs(dx) > 1e-6 else 0.0
            out.insert(0, (0.0, y0 - slope*x0))
        return out

    def _target_lane_width(self, current_lane_width: float) -> float:
        """Width of the LEFT target lane. 0/unset preserves original V9 behaviour.

        Only trust a positive override measured independently from the current
        lane (e.g. simulator map, lane-marking geometry or calibrated survey).
        """
        target = float(getattr(self, "target_lane_width_m", 0.0))
        return target if target > 0.0 else float(current_lane_width)

    def _target_lane_center_shift(self, current_lane_width: float) -> float:
        """Centre-to-centre distance across their SHARED divider."""
        return 0.5 * (float(current_lane_width) + self._target_lane_width(current_lane_width))

    def _left_divider_sanity(self, lane_width: float) -> Tuple[bool, str, dict]:
        divider = self._boundary_local("left_boundary_points")
        if len(divider) < 3:
            return False, "left_divider_short", {"n": len(divider)}
        y5 = interp_y(divider, 5.0)
        if y5 is None:
            y5 = divider[min(1, len(divider)-1)][1]
        expected = 0.5*lane_width
        err = abs(float(y5) - expected)
        diag = {
            "left_divider_y5_m": round(float(y5), 3),
            "expected_half_width_m": round(expected, 3),
            "divider_error_m": round(err, 3),
        }
        if float(y5) <= 0.0:
            return False, "left_divider_wrong_side", diag
        if err > self.left_divider_expected_tol_m:
            return False, "left_divider_not_adjacent", diag
        return True, "ok", diag

    def _inner_center_sanity(self) -> Tuple[bool, str, Optional[float]]:
        center = self._centerline_local()
        if len(center) < 3:
            return False, "inner_center_short", None
        y = interp_y(center, 8.0)
        if y is None:
            y = interp_y(center, 5.0)
        if y is None:
            return False, "inner_center_missing", None
        if abs(float(y)) > self.inner_center_max_abs_y_m:
            return False, "inner_center_not_ego_lane", float(y)
        return True, "ok", float(y)

    def _odom_pose(self) -> Tuple[float, float, float, float]:
        odom = self.latest_odom
        pose = odom.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        speed = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
        return float(pose.position.x), float(pose.position.y), float(yaw), float(speed)

    def _local_to_map(self, pts: Sequence[Tuple[float, float]], stamp: rospy.Time) -> RosPath:
        ex, ey, yaw, _ = self._odom_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        msg = RosPath()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        for x, y in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = ex + c*x - s*y
            ps.pose.position.y = ey + s*x + c*y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _extend_local_polyline(self, points: Sequence[Tuple[float, float]], target_length_m: float) -> List[Tuple[float, float]]:
        pts = list(points)
        if len(pts) < 2:
            return pts
        arc = polyline_arclength(pts)
        if arc[-1] >= target_length_m:
            return pts
        tx, ty = tangent_at(pts, len(pts)-1)
        x, y = pts[-1]
        remain = target_length_m - arc[-1]
        step = 1.0
        while remain > 1e-6:
            ds = min(step, remain)
            x += tx * ds
            y += ty * ds
            pts.append((x, y))
            remain -= ds
        return pts

    def _path_map_to_local(self, path: Optional[RosPath]) -> List[Tuple[float, float]]:
        if path is None or self.latest_odom is None:
            return []
        ex, ey, yaw, _ = self._odom_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        out: List[Tuple[float, float]] = [(0.0, 0.0)]
        for ps in path.poses:
            dx = float(ps.pose.position.x) - ex
            dy = float(ps.pose.position.y) - ey
            x = c*dx + s*dy
            y = -s*dx + c*dy
            if x > 0.5 and math.isfinite(x) and math.isfinite(y):
                out.append((x, y))
        out.sort(key=lambda q: q[0])
        dedup: List[Tuple[float, float]] = []
        for q in out:
            if not dedup or math.hypot(q[0]-dedup[-1][0], q[1]-dedup[-1][1]) > 0.15:
                dedup.append(q)
        return dedup

    def _generate_rejoin_path(self, now: rospy.Time) -> Optional[RosPath]:
        """Blend the camera lane centerline into the base/global path smoothly.

        Rejoin starts only when the two physical lanes are already close, so a
        local-y quintic blend is enough and avoids a hard path-source switch.
        """
        inner = self._centerline_local()
        base = self._path_map_to_local(self.latest_base_path)
        if len(inner) < 3 or len(base) < 3:
            return None
        target_len = max(self.rejoin_length_m + 10.0, 30.0)
        inner = self._extend_local_polyline(inner, target_len)
        base = self._extend_local_polyline(base, target_len)
        arc = polyline_arclength(inner)
        blended: List[Tuple[float, float]] = []
        for i, (x, yi) in enumerate(inner):
            yb = interp_y(base, x)
            if yb is None:
                yb = yi
            u = (arc[i] - 1.5) / max(self.rejoin_length_m, 1e-6)
            w = smoothstep5(u)
            blended.append((x, (1.0-w)*yi + w*yb))
        return self._local_to_map(blended, now)

    def _generate_lane_change_local(self, lane_width: float, speed_mps: float) -> Tuple[List[Tuple[float, float]], float]:
        """Generate exactly ONE left-lane move from the detected left divider."""
        divider = self._boundary_local("left_boundary_points")
        # Curvature/acceleration estimate must use ACTUAL centre shift,
        # not the source lane width alone.
        required=required_lane_change_length(speed_mps,self._target_lane_center_shift(lane_width),
                     self.max_lateral_accel_mps2,self.change_length_safety_factor)
        length=max(self.change_min_length_m,max(speed_mps,1.0)*self.change_time_s,required)
        if length>self.change_max_length_m+1e-6:
            # Higher-speed change needs longer visibility / geometry than this
            # launch currently permits. Reject candidate, never silently clamp.
            return [],length
        target_len = self.change_start_m + length + self.change_post_hold_m
        if not divider or target_len-max(p[0] for p in divider) > self.change_max_unobserved_extension_m:
            return [], length
        divider = self._extend_local_polyline(divider, target_len)
        if len(divider) < 3:
            return [], length

        current: List[Tuple[float, float]] = []
        target: List[Tuple[float, float]] = []
        target_width = self._target_lane_width(lane_width)
        for i, (x, y) in enumerate(divider):
            tx, ty = tangent_at(divider, i)
            nx, ny = -ty, tx
            # Keep current-lane validation tied to its OBSERVED width,
            # but build the desired centre from the independently sized TARGET lane.
            current.append((x - 0.5*lane_width*nx, y - 0.5*lane_width*ny))
            target.append((x + 0.5*target_width*nx, y + 0.5*target_width*ny))

        # The inferred divider may put the present lane centre 0.3-0.6 m
        # away from the ego at x=0.  Merely forcing current[0] to (0,0)
        # leaves an artificial near-field kink in the committed trajectory.
        # Blend that offset out along the entry, before/while moving left;
        # preserve the full measured boundary-based target lane at the end.
        measured_current_y0 = float(current[0][1])
        current[0] = (0.0, 0.0)
        arc = polyline_arclength(current)

        shifted: List[Tuple[float, float]] = []
        for i, (cx, cy) in enumerate(current):
            u = (arc[i] - self.change_start_m) / max(length, 1e-6)
            w = smoothstep5(u)
            if i > 0:
                entry_ratio = arc[i] / max(self.change_entry_align_m, 1e-6)
                cy -= measured_current_y0 * (1.0 - smoothstep5(entry_ratio))
            tx, ty = target[i]
            shifted.append(((1.0-w)*cx + w*tx, (1.0-w)*cy + w*ty))
        return resample_polyline_max_gap(shifted, self.change_max_waypoint_step_m), length

    def _map_obstacles_local(self) -> List[LocalObstacle]:
        if self.latest_obstacles is None or self.latest_odom is None:
            return []
        ex, ey, yaw, _ = self._odom_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for o in self.latest_obstacles.obstacles:
            dx = float(o.center_x_map) - ex
            dy = float(o.center_y_map) - ey
            x = c*dx + s*dy
            y = -s*dx + c*dy
            vx = c*float(o.velocity_x_map) + s*float(o.velocity_y_map)
            vy = -s*float(o.velocity_x_map) + c*float(o.velocity_y_map)
            out.append(LocalObstacle(
                int(o.id), x, y, vx, vy,
                max(0.5, float(o.length)), max(0.4, float(o.width)), float(o.yaw),
            ))
        return out

    def _target_lane_neighbors(self, lane_width: float) -> Tuple[Optional[LocalObstacle], Optional[LocalObstacle], List[LocalObstacle]]:
        center = self._centerline_local()
        obs = self._map_obstacles_local()
        front = None
        rear = None
        considered = []
        for o in obs:
            if abs(o.x) > self.gap_search_range_m:
                continue
            cy = interp_y(center, clamp(o.x, 0.0, 25.0))
            if cy is None:
                cy = 0.0
            # Use exactly the same target centre as the generated path.
            target_y = cy + self._target_lane_center_shift(lane_width)
            # Center-to-center lane membership.  The previous footprint-style
            # allowance (~3 m for a normal lane/car) could mix current- and
            # target-lane vehicles.
            allowance = self.target_lane_center_gate_m
            if abs(o.y - target_y) > allowance:
                continue
            considered.append(o)
            if o.x >= 0.0 and (front is None or o.x < front.x):
                front = o
            if o.x < 0.0 and (rear is None or o.x > rear.x):
                rear = o
        return front, rear, considered

    def _gap_safe_for_speed(self, candidate_speed: float, lane_width: float, change_length: float) -> Tuple[bool, str, dict]:
        front, rear, considered = self._target_lane_neighbors(lane_width)
        t = (self.change_start_m + change_length) / max(candidate_speed, 0.5)
        diag = {"candidate_speed": round(candidate_speed, 2), "t_change": round(t, 2), "objects": [o.oid for o in considered]}

        if front is not None:
            rel_now = front.x - (self.vehicle_center_from_base_m + 0.5*self.vehicle_length_m) - 0.5*front.length
            rel_future = rel_now + (front.vx - candidate_speed) * t
            required = max(self.front_min_gap_m, self.time_headway_s * candidate_speed)
            closing = candidate_speed - front.vx
            ttc = rel_now / closing if closing > 0.05 and rel_now > 0.0 else float("inf")
            diag["front"] = {"id": front.oid, "gap": round(rel_now,2), "future_gap": round(rel_future,2), "v": round(front.vx,2), "ttc": None if not math.isfinite(ttc) else round(ttc,2)}
            if rel_now < required or rel_future < required * 0.75:
                return False, "front_gap", diag
            if ttc < self.min_ttc_s:
                return False, "front_ttc", diag

        if rear is not None:
            ego_rear_from_base = self.vehicle_center_from_base_m - 0.5*self.vehicle_length_m
            rel_now = -rear.x - 0.5*rear.length + ego_rear_from_base
            # Positive means separation behind ego; relative separation evolves by ego - rear speed.
            rel_future = rel_now + (candidate_speed - rear.vx) * t
            required = max(self.rear_min_gap_m, self.time_headway_s * max(rear.vx, 0.0))
            closing = rear.vx - candidate_speed
            ttc = rel_now / closing if closing > 0.05 and rel_now > 0.0 else float("inf")
            diag["rear"] = {"id": rear.oid, "gap": round(rel_now,2), "future_gap": round(rel_future,2), "v": round(rear.vx,2), "ttc": None if not math.isfinite(ttc) else round(ttc,2)}
            if rel_now < required or rel_future < required * 0.75:
                return False, "rear_gap", diag
            if ttc < self.min_ttc_s:
                return False, "rear_ttc", diag

        return True, "ok", diag

    def _path_curvature_ok(self, pts: Sequence[Tuple[float, float]], speed_mps: float) -> Tuple[bool, float]:
        max_k = 0.0
        for i in range(1, len(pts)-1):
            ax, ay = pts[i-1]
            bx, by = pts[i]
            cx, cy = pts[i+1]
            ab = math.hypot(bx-ax, by-ay)
            bc = math.hypot(cx-bx, cy-by)
            ac = math.hypot(cx-ax, cy-ay)
            denom = ab*bc*ac
            if denom < 1e-6:
                continue
            cross = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
            k = abs(2.0*cross/denom)
            max_k = max(max_k, k)
        return speed_mps*speed_mps*max_k <= self.max_lateral_accel_mps2 + 1e-6, max_k

    def _dynamic_path_safe(self, path: RosPath, candidate_speed: float, max_arc_m: Optional[float] = None) -> Tuple[bool, str]:
        if self.latest_obstacles is None or len(path.poses) < 3:
            return False, "no_obstacles_or_short_path"
        # Arc length / time along the ego base_link path.
        arc = [0.0]
        for i in range(1, len(path.poses)):
            a = path.poses[i-1].pose.position
            b = path.poses[i].pose.position
            arc.append(arc[-1] + math.hypot(b.x-a.x, b.y-a.y))

        obs = list(self.latest_obstacles.obstacles)
        for i, ps in enumerate(path.poses):
            if i == 0:
                continue
            if max_arc_m is not None and arc[i] > max_arc_m:
                break
            p = ps.pose.position
            p0 = path.poses[max(0, i-1)].pose.position
            p1 = path.poses[min(len(path.poses)-1, i+1)].pose.position
            yaw = math.atan2(p1.y-p0.y, p1.x-p0.x)
            c, s = math.cos(yaw), math.sin(yaw)
            t = arc[i] / max(candidate_speed, 0.5)
            ego_cx = p.x + self.vehicle_center_from_base_m*c
            ego_cy = p.y + self.vehicle_center_from_base_m*s
            for o in obs:
                ox = float(o.center_x_map) + float(o.velocity_x_map)*t
                oy = float(o.center_y_map) + float(o.velocity_y_map)*t
                dx, dy = ox-ego_cx, oy-ego_cy
                lon = c*dx + s*dy
                lat = -s*dx + c*dy
                lon_lim = 0.5*self.vehicle_length_m + 0.5*max(0.5,float(o.length)) + self.collision_long_margin_m
                lat_lim = 0.5*self.vehicle_width_m + 0.5*max(0.4,float(o.width)) + self.collision_lat_margin_m
                if abs(lon) <= lon_lim and abs(lat) <= lat_lim:
                    # Allow a very short prefix if the obstacle box already overlaps
                    # because LiDAR boxes can transiently touch the ego footprint.
                    if arc[i] > 2.0:
                        return False, "predicted_collision_id_%d" % int(o.id)
        return True, "ok"

    def _choose_lane_change(self, now: rospy.Time) -> Tuple[Optional[RosPath], Optional[float], Optional[float], str, dict]:
        if not self._route_gate_present():
            return None, None, None, "route_gate_inactive", {}
        ok, reason = self._lane_valid(now)
        if not ok:
            return None, None, None, reason, {}
        near_ok, near_reason, near_diag = lane_change_observation_gate(
            self.lane_info, self.change_observation_near_max_m)
        if not near_ok:
            return None, None, None, near_reason, near_diag
        dashed, dreason = self._left_dashed_ok()
        if not dashed:
            return None, None, None, dreason, {}
        # The sensor-team merge-gap node is itself gated by
        # /perception/camera/highway_environment.  Therefore an explicit mission
        # request would otherwise deadlock whenever that upstream highway gate is
        # false.  Under an explicit request we may skip only that upstream veto;
        # the lane-change still must pass this node's own LiDAR front/rear gap,
        # TTC, predicted-collision, curvature and dashed-line checks below.
        use_sensor_merge_gate = not (
            self.bypass_sensor_merge_gate
            or ((self.highway_request or self.force_highway_active)
                and self.mission_request_bypass_sensor_merge_gate)
        )
        if use_sensor_merge_gate:
            if not self._fresh(self.merge_at, self.merge_timeout_s, now):
                return None, None, None, "merge_gap_stale", {}
            if self.merge_unavailable or not self.merge_available:
                return None, None, None, "sensor_merge_gap_unavailable", {}
        if not self._fresh(self.obstacles_at, self.obstacle_timeout_s, now):
            return None, None, None, "obstacles_stale", {}

        width = float(self.lane_info.get("lane_width_m"))
        divider_ok, divider_reason, divider_diag = self._left_divider_sanity(width)
        divider_diag.update({
            "current_lane_width_m": round(width, 3),
            "target_lane_width_m": round(self._target_lane_width(width), 3),
            "target_center_shift_m": round(self._target_lane_center_shift(width), 3),
        })
        if not divider_ok:
            return None, None, None, divider_reason, {"divider": divider_diag}

        _, _, _, ego_speed = self._odom_pose()
        raw_candidates = [
            min(self.cruise_speed_mps, max(ego_speed, 2.0) + 0.5),
            min(self.cruise_speed_mps, max(ego_speed, 2.0)),
            max(1.5, min(self.cruise_speed_mps, ego_speed - 1.0)),
            max(1.5, min(self.cruise_speed_mps, ego_speed - 2.0)),
        ]
        candidates = sorted({round(v, 2) for v in raw_candidates if v > 0.1}, reverse=True)
        diagnostics = {}
        for v in candidates:
            local, length = self._generate_lane_change_local(width, v)
            if len(local) < 3:
                observed_x=max((p[0] for p in self._boundary_local("left_boundary_points")),default=0.0)
                extension=self.change_start_m+length+self.change_post_hold_m-observed_x
                reject=("change_length_exceeds_configured_max" if length>self.change_max_length_m
                        else "change_divider_extrapolation_excess" if extension>self.change_max_unobserved_extension_m
                        else "lane_change_geometry_short")
                diagnostics[str(v)]={"divider":divider_diag,"reason":reject,
                     "required_length_m":round(length,2),"required_extrapolation_m":round(extension,2)}
                continue
            curv_ok, max_k = self._path_curvature_ok(local, v)
            gap_ok, gap_reason, gap_diag = self._gap_safe_for_speed(v, width, length)
            path = self._local_to_map(local, now)
            dyn_ok, dyn_reason = self._dynamic_path_safe(
                path, v, self.change_start_m + length + 4.0
            )
            diagnostics[str(v)] = {
                "curvature": round(max_k,5),
                "change_length_m": round(length,2),
                "lateral_accel_mps2": round(v*v*max_k,2),
                "gap": gap_diag,
                "gap_reason": gap_reason,
                "dyn": dyn_reason,
                "divider": divider_diag,
            }
            if curv_ok and gap_ok and dyn_ok:
                return path, v, length, "ok", diagnostics
        return None, None, None, "no_safe_speed_path_pair", diagnostics

    def _current_lane_lead(self) -> Tuple[Optional[LocalObstacle], Optional[float], Optional[float]]:
        center = self._centerline_local()
        lane_width = float((self.lane_info or {}).get("lane_width_m") or 3.5)
        best = None
        best_gap = None
        best_ttc = None
        _, _, _, ego_speed = self._odom_pose()
        for o in self._map_obstacles_local():
            if o.x <= 0.0 or o.x > self.follow_search_m:
                continue
            cy = interp_y(center, clamp(o.x, 0.0, 25.0))
            if cy is None:
                cy = 0.0
            # Strict current-lane center gate.  Adjacent-lane vehicles must
            # never trigger emergency following/stop while ego is still in the
            # current lane.
            if abs(o.y - cy) > self.current_lane_center_gate_m:
                continue
            # Lead following is bumper-to-bumper, not center-x based.  A vehicle
            # whose rear bumper is still alongside/behind the ego front must not be
            # treated as a front lead merely because its CENTER transformed to x>0.
            ego_front_x = self.vehicle_center_from_base_m + 0.5*self.vehicle_length_m
            obstacle_rear_x = o.x - 0.5*o.length
            gap = obstacle_rear_x - ego_front_x
            if gap < -self.lead_overlap_tolerance_m:
                continue
            if best_gap is None or gap < best_gap:
                closing = ego_speed - o.vx
                ttc = max(gap, 0.0)/closing if closing > 0.05 else float("inf")
                best, best_gap, best_ttc = o, gap, ttc
        return best, best_gap, best_ttc

    def _gap_shaping_speed(self, lane_width: float) -> Tuple[float, dict]:
        """Choose a safe longitudinal speed that tends to create a LEFT-lane slot.

        This never relaxes the merge/TTC rules. It only changes ego speed within
        the normal cruise limit while WAIT_GAP so the vehicle does not passively
        arrive at the physical lane merge with no usable slot.
        """
        _, _, _, ego_speed = self._odom_pose()
        front, rear, _ = self._target_lane_neighbors(lane_width)
        horizon = 3.0
        candidates = []
        v = 1.5
        while v < self.cruise_speed_mps + 1e-6:
            candidates.append(round(v, 2))
            v += 0.5
        candidates.extend([round(clamp(ego_speed, 1.5, self.cruise_speed_mps), 2), round(self.cruise_speed_mps, 2)])
        candidates = sorted(set(candidates), reverse=True)

        best_v = min(self.cruise_speed_mps, max(1.5, ego_speed))
        best_score = -1e9
        best_diag = {}
        for cand in candidates:
            score = 10.0
            diag = {"candidate": cand}
            if front is not None:
                gap0 = front.x - (self.vehicle_center_from_base_m + 0.5*self.vehicle_length_m) - 0.5*front.length
                gapf = gap0 + (front.vx-cand)*horizon
                req = max(self.front_min_gap_m, self.time_headway_s*cand)
                score = min(score, gapf/max(req,0.1))
                diag["front_future_gap"] = round(gapf,2)
            if rear is not None:
                ego_rear_from_base = self.vehicle_center_from_base_m - 0.5*self.vehicle_length_m
                gap0 = -rear.x - 0.5*rear.length + ego_rear_from_base
                gapf = gap0 + (cand-rear.vx)*horizon
                req = max(self.rear_min_gap_m, self.time_headway_s*max(rear.vx,0.0))
                score = min(score, gapf/max(req,0.1))
                diag["rear_future_gap"] = round(gapf,2)
            # Prefer the higher speed when two candidates produce similar gap quality.
            score += 0.02*cand
            if score > best_score:
                best_score = score
                best_v = cand
                best_diag = diag
        best_diag["score"] = round(best_score,3)
        return float(best_v), best_diag

    def _gap_shaping_rear_clear(self, proposed_speed: float, lane_width: float) -> Tuple[bool, str, dict]:
        """Do not deliberately slow for a target-lane gap if rear TTC is short.

        This is a conservative veto on optional speed shaping, *not* permission
        to merge. The real front/rear, merge-sensor and collision gates in
        _choose_lane_change are always unchanged.
        """
        if self.latest_obstacles is None or self.latest_odom is None:
            return False, "rear_obstacles_missing", {}
        center = self._centerline_local()
        ego_rear = self.vehicle_center_from_base_m - 0.5*self.vehicle_length_m
        closest = None
        for o in self._map_obstacles_local():
            if not (-self.gap_search_range_m <= o.x < 0.0):
                continue
            cy = interp_y(center, clamp(o.x, 0.0, 25.0))
            if cy is None:
                cy = 0.0
            current_lane = abs(o.y - cy) <= self.current_lane_center_gate_m
            target_lane = abs(o.y - (cy + self._target_lane_center_shift(lane_width))) <= self.target_lane_center_gate_m
            if not (current_lane or target_lane):
                continue
            gap = ego_rear - (o.x + 0.5*o.length)
            closing = o.vx - proposed_speed
            ttc = max(gap, 0.0)/closing if closing > 0.05 else float("inf")
            required = max(self.rear_min_gap_m, self.time_headway_s*max(0.0,o.vx))
            if gap < required or ttc < self.inner_hold_gap_rear_min_ttc_s:
                return False, "rear_ttc_or_gap_short", {
                    "id": int(o.oid), "lane": "current" if current_lane else "target",
                    "gap_m": round(gap, 2),
                    "ttc_s": None if not math.isfinite(ttc) else round(ttc, 2),
                    "required_gap_m": round(required, 2),
                }
            if closest is None or gap < closest:
                closest = gap
        return True, "ok", {"closest_rear_gap_m": None if closest is None else round(closest, 2)}

    def _candidate_summary(self, candidates: dict) -> dict:
        """Compact JSON diagnostics for every tested speed / safety gate."""
        summary = {}
        for v, d in candidates.items():
            if not isinstance(d, dict):
                continue
            gap = d.get("gap", {})
            summary[str(v)] = {
                "gap_reason": d.get("gap_reason"),
                "dynamic_reason": d.get("dyn"),
                "curvature_ok": (float(v)**2 * float(d.get("curvature", 0.0))
                                 <= self.max_lateral_accel_mps2 + 1e-6),
                "front": gap.get("front"), "rear": gap.get("rear"),
            }
        return summary

    def _adaptive_speed(self, cruise: float) -> Tuple[float, bool, dict]:
        if self.latest_odom is None:
            return 0.0, True, {"reason": "no_odom"}
        _, _, _, ego_speed = self._odom_pose()
        lead, gap, ttc = self._current_lane_lead()
        if lead is None or gap is None:
            return cruise, False, {"lead": None}
        desired = self.follow_standstill_gap_m + self.follow_time_headway_s * ego_speed
        target = min(cruise, max(0.0, lead.vx + self.follow_gain*(gap-desired)))
        emergency = gap < self.emergency_gap_m or (ttc is not None and math.isfinite(ttc) and ttc < self.emergency_ttc_s)
        return target, emergency, {
            "lead": lead.oid,
            "gap": round(gap,2),
            "lead_v": round(lead.vx,2),
            "ttc": None if ttc is None or not math.isfinite(ttc) else round(ttc,2),
            "desired_gap": round(desired,2),
        }

    def _fallback_front_emergency(self) -> Tuple[bool, dict]:
        """Minimal collision guard for temporary lane-camera dropouts.

        When lane geometry is unavailable we do not try to do normal car
        following, but we still must not creep into an object directly in front
        of the ego vehicle.  This guard is intentionally narrow and only latches
        for the current tick; it clears automatically when the object clears.
        """
        if self.latest_odom is None:
            return True, {"reason": "no_odom"}
        _, _, _, ego_speed = self._odom_pose()
        ego_front_x = self.vehicle_center_from_base_m + 0.5*self.vehicle_length_m
        best = None
        for o in self._map_obstacles_local():
            if o.x <= 0.0 or abs(o.y) > 1.8:
                continue
            gap = (o.x - 0.5*o.length) - ego_front_x
            closing = ego_speed - o.vx
            ttc = max(gap, 0.0)/closing if closing > 0.05 else float("inf")
            if best is None or gap < best[1]:
                best = (o, gap, ttc)
        if best is None:
            return False, {"lead": None}
        o, gap, ttc = best
        emergency = gap < self.emergency_gap_m or (math.isfinite(ttc) and ttc < self.emergency_ttc_s)
        return emergency, {
            "lead": o.oid,
            "gap": round(gap, 2),
            "lead_v": round(o.vx, 2),
            "ttc": None if not math.isfinite(ttc) else round(ttc, 2),
            "fallback": True,
        }

    def _limit_speed_rate(self, target: float, dt: float) -> float:
        target = clamp(target, 0.0, self.cruise_speed_mps)
        if target > self.last_output_speed:
            rise=self.speed_rise_mps2
            if (self.state == self.INNER_HOLD and
                    self.inner_speed_stable_since is not None and
                    (rospy.Time.now()-self.inner_speed_stable_since).to_sec() >= self.inner_stable_speed_confirm_s):
                rise=max(rise,self.inner_stable_speed_rise_mps2)
            out = min(target, self.last_output_speed + rise*dt)
        else:
            out = max(target, self.last_output_speed - self.speed_fall_mps2*dt)
        self.last_output_speed = out
        return out

    def _global_signed_d(self) -> Optional[float]:
        if self.latest_odom is None or len(self.global_points) < 2:
            return None
        x, y, _, _ = self._odom_pose()
        best_d2 = float("inf")
        best_signed = None
        n = len(self.global_points)
        for i in range(n-1):
            a = self.global_points[i]
            b = self.global_points[i+1]
            vx, vy = b.x-a.x, b.y-a.y
            l2 = vx*vx + vy*vy
            if l2 < 1e-9:
                continue
            t = clamp(((x-a.x)*vx + (y-a.y)*vy)/l2, 0.0, 1.0)
            px, py = a.x+t*vx, a.y+t*vy
            dx, dy = x-px, y-py
            d2 = dx*dx+dy*dy
            if d2 < best_d2:
                l = math.sqrt(l2)
                tx, ty = vx/l, vy/l
                signed = tx*dy - ty*dx  # +left
                best_d2 = d2
                best_signed = signed
        return best_signed

    def _publish(self, path: Optional[RosPath], stop: bool, speed: float, active: bool, status: dict, now: rospy.Time, dt: float) -> None:
        # V9 decision and speed profile; v61 transport is an extra output only.
        output_path = copy.deepcopy(path) if path is not None else RosPath()
        if path is None or len(output_path.poses) < 2:
            stop = True
            status["reason"] = "trajectory_path_missing_or_short"
        output_path.header.stamp = now
        output_path.header.frame_id = self.map_frame
        self.trajectory_seq = (self.trajectory_seq + 1) & 0xFFFFFFFF
        output_path.header.seq = self.trajectory_seq
        speed_out = self._limit_speed_rate(0.0 if stop else speed, dt)
        self.trajectory_pub.publish(String(data=trajectory_payload(
            output_path, bool(stop), float(speed_out), status.get("reason", self.state)
        )))
        if not stop or path is not None:
            if path is not None and len(output_path.poses) >= 2:
                self.path_pub.publish(output_path)
        self.stop_pub.publish(Bool(data=bool(stop)))
        self.speed_pub.publish(Float64(data=float(speed_out)))
        self.active_pub.publish(Bool(data=bool(active)))
        self.fast_change_pub.publish(Bool(data=bool(active and self.state == self.LANE_CHANGE)))
        status.update({"state": self.state, "active": bool(active), "stop": bool(stop),
                       "target_speed_mps": round(speed_out,2), "lane_changes_done": self.lane_changes_done,
                       "handoff_permitted": self._handoff_permitted(),
                       "overlay": "V9_logic_v61_transport"})
        self.state_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))
        if stop:
            rospy.logwarn_throttle(0.5, "HIGHWAY STOP state=%s reason=%s follow=%s", self.state, str(status.get("reason")), json.dumps(status.get("follow", {}), separators=(",", ":")))

    @plan_locked
    def _tick(self, _event) -> None:
        now = rospy.Time.now()
        if self.last_timer_time is None:
            dt = 1.0/max(self.rate_hz,1.0)
        else:
            dt = clamp((now-self.last_timer_time).to_sec(), 0.001, 0.2)
        self.last_timer_time = now

        base_fresh = self.latest_base_path is not None and self._fresh(self.base_path_at, self.base_timeout_s, now)
        base_stop_fresh = self._fresh(self.base_stop_at, self.base_timeout_s, now)
        odom_fresh = self.latest_odom is not None and self._fresh(self.odom_at, self.odom_timeout_s, now)
        obs_fresh = self.latest_obstacles is not None and self._fresh(self.obstacles_at, self.obstacle_timeout_s, now)
        activation = self._activation_present()

        if not odom_fresh:
            self._publish(self.latest_base_path if base_fresh else None, True, 0.0, self.state != self.OFF, {"reason":"odom_stale"}, now, dt)
            return

        # Distance accumulation for committed states.
        ex, ey, _, ego_speed = self._odom_pose()
        if self.state == self.LANE_CHANGE:
            if self.last_change_xy is not None:
                self.change_travel_m += math.hypot(ex-self.last_change_xy[0], ey-self.last_change_xy[1])
            self.last_change_xy = (ex,ey)
        elif self.state == self.INNER_HOLD:
            if self.last_hold_xy is not None:
                moved=math.hypot(ex-self.last_hold_xy[0],ey-self.last_hold_xy[1])
                self.inner_hold_travel_m+=moved
                self.inner_unverified_travel_m+=moved
            self.last_hold_xy=(ex,ey)
        elif self.state == self.REJOIN:
            if self.last_rejoin_xy is not None:
                self.rejoin_travel_m += math.hypot(ex-self.last_rejoin_xy[0], ey-self.last_rejoin_xy[1])
            self.last_rejoin_xy = (ex,ey)

        if self.completed_once and self.state != self.DONE:
            self.state = self.DONE

        # OFF: transparent pass-through until highway is confidently detected/requested.
        if self.state == self.OFF:
            if activation and not self.completed_once:
                if self.highway_true_since is None:
                    self.highway_true_since = now
                elif (now-self.highway_true_since).to_sec() >= self.highway_confirm_s:
                    self.state = self.WAIT_GAP
                    self.ready_since = None
            else:
                self.highway_true_since = None
            stop = (not base_fresh) or (not base_stop_fresh) or self.base_stop
            self._publish(self.latest_base_path if base_fresh else None, stop, self.cruise_speed_mps, False, {"reason":"base_pass"}, now, dt)
            return

        # DONE: do not re-arm in the same latched-highway scenario.
        if self.state == self.DONE:
            stop = (not base_fresh) or (not base_stop_fresh) or self.base_stop
            self._publish(self.latest_base_path if base_fresh else None, stop, self.cruise_speed_mps, False, {"reason":"highway_strategy_done"}, now, dt)
            return

        # WAIT_GAP: stay on the existing global/avoidance path, but start adapting speed.
        if self.state == self.WAIT_GAP:
            adaptive, emergency, follow = self._adaptive_speed(self.cruise_speed_mps) if obs_fresh and self.lane_info is not None else (self.cruise_speed_mps, False, {})
            shaping = self.cruise_speed_mps
            shaping_diag = {}
            lane_ok_for_shape, _ = self._lane_valid(now)
            if obs_fresh and lane_ok_for_shape:
                shaping, shaping_diag = self._gap_shaping_speed(float(self.lane_info.get("lane_width_m")))
            wait_speed = min(adaptive, shaping)
            path, cand_speed, length, reason, diag = self._choose_lane_change(now)
            if path is not None:
                if self.ready_since is None:
                    self.ready_since = now
                elif (now-self.ready_since).to_sec() >= self.ready_confirm_s:
                    self.committed_path = path
                    self.committed_speed_mps = float(cand_speed)
                    self.committed_change_length_m = float(length)
                    self.change_travel_m = 0.0
                    self.last_change_xy = (ex,ey)
                    self.complete_since = None
                    self.inner_handoff_pending = False
                    self.inner_camera_history.clear()
                    self.inner_smoothed_camera_map=None
                    self.inner_camera_reacquire_started_at=None
                    self.inner_camera_last_smoothed_at=None
                    self.inner_speed_stable_since=None
                    self.inner_camera_filter_reason = "lane_change_committed"
                    self.inner_camera_match_since = None
                    self.state = self.LANE_CHANGE
                    rospy.logwarn("HIGHWAY lane change COMMITTED speed=%.2f length=%.1f", cand_speed, length)
            else:
                self.ready_since = None
            stop = (not base_fresh) or (not base_stop_fresh) or self.base_stop or emergency
            self._publish(self.latest_base_path if base_fresh else None, stop, wait_speed, True, {"reason":reason, "follow":follow, "gap_shaping":shaping_diag, "candidate_diag":diag}, now, dt)
            return

        if self.state == self.LANE_CHANGE:
            # Geometry remains committed. Do not regenerate from camera because the
            # camera's ego-lane identity can switch midway through the maneuver.
            lane_ok, lane_reason = self._lane_valid(now)
            adaptive, emergency, follow = self._adaptive_speed(self.committed_speed_mps) if obs_fresh and lane_ok else (self.committed_speed_mps, False, {})
            speed = min(self.committed_speed_mps, adaptive)

            # Only front/imminent safety may stop a committed change. Rear-gap
            # changes after commitment must not make us brake into the approaching car.
            stop = emergency or not obs_fresh or self.committed_path is None

            lat = None if self.lane_info is None else self.lane_info.get("lateral_error_m")
            head = None if self.lane_info is None else self.lane_info.get("heading_error_rad")
            transition_done = self.change_travel_m >= (self.change_start_m + self.committed_change_length_m)
            settle_needed = self.change_start_m + self.committed_change_length_m + min(self.change_settle_m, self.change_post_hold_m)
            settled = self.change_travel_m >= settle_needed
            progressed = settled
            centered = lane_ok and lat is not None and head is not None and abs(float(lat)) <= self.change_center_error_m and abs(float(head)) <= self.change_heading_error_rad

            # The committed lane-change path is finite, while the sensor-team
            # Pure Pursuit intentionally stops at the end of any finite path.
            # Do not wait for a camera-centering condition all the way to the
            # endpoint: after the planned lateral transition plus the settle
            # distance, the manoeuvre is geometrically complete.  A second
            # endpoint-distance guard guarantees hand-over before PP enters its
            # ~1.5 m goal-stop zone even if odometry distance accumulation is a
            # little noisy.
            remaining_to_end = None
            if self.committed_path is not None and self.committed_path.poses:
                ep = self.committed_path.poses[-1].pose.position
                remaining_to_end = math.hypot(float(ep.x)-ex, float(ep.y)-ey)
            endpoint_guard_complete = (
                transition_done
                and remaining_to_end is not None
                and remaining_to_end <= self.change_endpoint_guard_m
            )
            # Prefer real camera-centred landing while still keeping the
            # endpoint guard: a finite PP path must not reach its goal-stop.
            geometry_complete = completed_change_handoff(settled, centered, endpoint_guard_complete)

            if geometry_complete:
                if self.complete_since is None:
                    self.complete_since = now
                elif (now-self.complete_since).to_sec() >= self.change_complete_confirm_s:
                    self.lane_changes_done += 1
                    self.state = self.INNER_HOLD
                    self.inner_hold_travel_m = 0.0
                    self.last_hold_xy = (ex,ey)
                    self.inner_hold_started_at = now
                    self.inner_stable_since = None
                    self.inner_unstable_since = None
                    self.ready_since = None
                    self.release_since = None
                    self.lane_invalid_since = None
                    self.last_inner_path = self._maintain_reference_map(self.committed_path,now) or self.committed_path
                    self.inner_camera_history.clear()
                    self.inner_smoothed_camera_map=None
                    self.inner_camera_reacquire_started_at=None
                    self.inner_camera_last_smoothed_at=None
                    self.inner_speed_stable_since=None
                    self.inner_camera_last_step_m=0.0
                    self.inner_camera_filter_reason = "awaiting_post_change_camera"
                    self.inner_handoff_pending = True
                    self.inner_camera_match_since = None
                    self.inner_camera_match_last_msg_at = None
                    self.inner_camera_match_samples = 0
                    self.inner_unverified_travel_m = 0.0
                    why = "settled" if settled else "endpoint_guard"
                    rospy.logwarn(
                        "HIGHWAY lane change COMPLETE count=%d reason=%s remaining=%.2fm",
                        self.lane_changes_done, why,
                        -1.0 if remaining_to_end is None else remaining_to_end,
                    )
            else:
                self.complete_since = None

            stop_reason = "lead_emergency" if emergency else ("obstacles_stale" if not obs_fresh else lane_reason)
            self._publish(self.committed_path, stop, speed, True, {
                "reason":stop_reason,
                "travel_m":round(self.change_travel_m,2),
                "transition_done":transition_done,
                "settled":settled,
                "settle_needed_m":round(settle_needed,2),
                "progressed":progressed,
                "centered":centered,
                "remaining_to_end_m":None if remaining_to_end is None else round(remaining_to_end,2),
                "endpoint_guard_complete":endpoint_guard_complete,
                "geometry_complete":geometry_complete,
                "follow":follow,
            }, now, dt)
            return

        if self.state == self.INNER_HOLD:
            # Restore camera-following in the hold state: no V9 reference-
            # agreement latch, no 9 m "unverified travel" hard stop. Camera
            # observations are median-filtered AFTER ego-motion compensation.
            # A lost/invalid camera retains only the original bounded 3 s
            # low-speed recovery and the unchanged LiDAR emergency stop.
            lane_ok,lane_reason=self._lane_hold_valid(now)
            center_ok=False
            center_y=None
            if lane_ok:
                center_ok,_,center_y=self._inner_center_sanity()
            filtered=self._temporal_camera_center_local(now)
            camera_ready=lane_ok and len(filtered)>=6
            if camera_ready:
                smoothed=self._smooth_inner_path(filtered,now)
                if smoothed is not None:
                    self.last_inner_path=smoothed
                    self.last_good_center_local=list(filtered)
                    self.inner_handoff_pending=False
                    self.inner_unverified_travel_m=0.0
                    self.lane_invalid_since=None
                    self.inner_camera_filter_reason="camera_following"
                else:
                    camera_ready=False
                    lane_reason="camera_stitch_failed"
            else:
                if lane_ok:
                    lane_reason="temporal_camera_unconfirmed_"+self.inner_camera_filter_reason
            lane_ok=camera_ready
            if not lane_ok:
                if self.lane_invalid_since is None:
                    self.lane_invalid_since=now
                if self.last_inner_path is not None:
                    recovered=self._recover_inner_path(now)
                    if recovered is not None:
                        self.last_inner_path=recovered
            path=self.last_inner_path
            self.inner_reference_cross_track_m=self._reference_cross_track(path)
            invalid_age=(0.0 if self.lane_invalid_since is None else
                         max(0.0,(now-self.lane_invalid_since).to_sec()))
            lane_grace=(not lane_ok and path is not None and
                        self.lane_invalid_since is not None and
                        invalid_age<=self.inner_recovery_timeout_s)
            # Preserve the existing collision stop and cautious speed while
            # re-acquiring the camera lane (even if center_y8 > 0.9 m).
            if obs_fresh and lane_ok:
                adaptive,emergency,follow=self._adaptive_speed(self.cruise_speed_mps)
                h=(self.lane_info or {}).get("heading_error_rad")
                adaptive=self._inner_adaptive_speed_limit(center_y,filtered,adaptive,h,follow,now)
            elif obs_fresh and lane_grace:
                emergency,follow=self._fallback_front_emergency()
                adaptive=self.inner_lane_grace_speed_mps
            elif obs_fresh:
                adaptive,emergency,follow=0.0,False,{"lead":None}
            else:
                adaptive,emergency,follow=0.0,False,{}
            if not lane_ok:
                self.inner_speed_stable_since=None
                self.inner_filtered_y8_m=None
                self.inner_speed_lateral_limit_mps=None
            if emergency:
                stop=True
                inner_reason="lead_emergency"
            elif not obs_fresh:
                stop=True
                inner_reason="obstacles_stale"
            elif lane_ok:
                stop=path is None
                inner_reason="ok_camera_temporal" if path is not None else "inner_path_missing"
            elif lane_grace:
                stop=False
                adaptive=min(adaptive,self.inner_lane_grace_speed_mps)
                inner_reason="lane_recovery_"+lane_reason
            else:
                stop=True
                inner_reason="lane_recovery_timeout_"+lane_reason
            self.inner_association_diag={"mode":"camera_temporal_follow",
                                         "reason":self.inner_camera_filter_reason,
                                         "sample_count":self.inner_camera_samples,
                                         "rejected_jumps":self.inner_camera_rejected_count}

            # Event-based re-arm: do not count lane changes.  Once the ego has
            # settled in the new lane, the *current ego-left boundary* decides
            # what happens next.  Dashed + safe gap => another left change.
            # Reliably detected non-dashed boundary => stay in this lane and
            # wait for the global route to naturally converge for REJOIN.
            lat = None if self.lane_info is None else self.lane_info.get("lateral_error_m")
            head = None if self.lane_info is None else self.lane_info.get("heading_error_rad")
            hold_elapsed_s = 0.0 if self.inner_hold_started_at is None else max(0.0, (now-self.inner_hold_started_at).to_sec())
            stable_now = (
                lane_ok
                and center_ok
                and lat is not None
                and head is not None
                and abs(float(lat)) <= self.left_change_stable_lateral_m
                and abs(float(head)) <= self.left_change_stable_heading_rad
                and hold_elapsed_s >= self.left_change_rearm_min_s
                and self.inner_hold_travel_m >= self.left_change_rearm_min_m
            )
            if stable_now:
                if self.inner_stable_since is None:
                    self.inner_stable_since = now
                self.inner_unstable_since = None
            else:
                if self.inner_unstable_since is None:
                    self.inner_unstable_since = now
                if (now-self.inner_unstable_since).to_sec() > self.inner_rearm_dropout_grace_s:
                    self.inner_stable_since = None
                # Even for a brief dropout, restart *merge-gap confirmation*.
                # Re-arming memory may persist; unsafe merge commitment cannot.
                self.ready_since = None

            rearmed = (
                stable_now
                and self.inner_stable_since is not None
                and (now-self.inner_stable_since).to_sec() >= self.left_change_stable_confirm_s
            )

            left = (self.lane_info or {}).get("left_lane") or {}
            left_detected = bool(left.get("detected", False))
            left_is_dashed = left.get("dashed") is True
            left_is_nondashed = left_detected and left.get("dashed") is False

            # Report the exact reason whenever a second change was not
            # committed, including every candidate speed's rejected safety gate.
            # Keep this separate from the lane-HOLD / STOP reason.
            next_change_reason = "not_rearmed"
            next_change_diag = {}
            next_change_ready_s = 0.0
            gap_shaping_diag = {"enabled": self.inner_hold_gap_shaping_enabled, "applied": False}
            if rearmed and left_is_dashed:
                p, v, length, reason, diag = self._choose_lane_change(now)
                next_change_reason = reason
                next_change_diag = self._candidate_summary(diag)
                if p is not None:
                    if self.ready_since is None:
                        self.ready_since = now
                    next_change_ready_s = max(0.0, (now-self.ready_since).to_sec())
                    if next_change_ready_s >= self.ready_confirm_s:
                        self.committed_path = p
                        self.committed_speed_mps = float(v)
                        self.committed_change_length_m = float(length)
                        self.change_travel_m = 0.0
                        self.last_change_xy = (ex,ey)
                        self.complete_since = None
                        self.inner_handoff_pending = False
                        self.inner_camera_history.clear()
                        self.inner_smoothed_camera_map=None
                        self.inner_camera_reacquire_started_at=None
                        self.inner_camera_last_smoothed_at=None
                        self.inner_speed_stable_since=None
                        self.inner_camera_filter_reason = "lane_change_committed"
                        self.inner_camera_match_since = None
                        self.inner_stable_since = None
                        self.inner_unstable_since = None
                        self.ready_since = None
                        self.state = self.LANE_CHANGE
                        rospy.logwarn(
                            "HIGHWAY next LEFT lane change COMMITTED by dashed boundary speed=%.2f length=%.1f",
                            v, length,
                        )
                        self._publish(self.committed_path, False, self.committed_speed_mps, True,
                                      {"reason":"left_dashed_rearmed", "candidate_diag":diag}, now, dt)
                        return
                    next_change_reason = "gap_confirmation_wait"
                else:
                    self.ready_since = None
            else:
                self.ready_since = None
                if not lane_ok:
                    next_change_reason = "lane_hold_invalid_" + lane_reason
                elif not stable_now:
                    next_change_reason = "not_stable_in_new_lane"
                elif not rearmed:
                    next_change_reason = "rearm_confirmation_wait"
                elif not left_detected:
                    next_change_reason = "left_boundary_missing"
                else:
                    next_change_reason = "left_not_dashed"

            # Optional traffic gap shaping in INNER_HOLD. It can only reduce
            # speed mildly; it NEVER bypasses lane-change safety checks. Do not
            # shape on camera loss, a STOP, uncertain divider geometry, or when
            # a faster car is close behind in the current/target lane.
            if (self.inner_hold_gap_shaping_enabled and rearmed and left_is_dashed
                    and lane_ok and obs_fresh and not stop
                    and next_change_reason in ("no_safe_speed_path_pair", "sensor_merge_gap_unavailable")):
                strict_ok, _ = self._lane_valid(now)
                width = float(self.lane_info.get("lane_width_m")) if strict_ok else 0.0
                divider_ok, _, _ = self._left_divider_sanity(width) if strict_ok else (False, "no_width", {})
                if divider_ok:
                    wanted, score_diag = self._gap_shaping_speed(width)
                    proposed = max(0.0, adaptive - self.inner_hold_gap_max_slowdown_mps)
                    proposed = min(adaptive, max(proposed, wanted))
                    rear_ok, rear_reason, rear_diag = self._gap_shaping_rear_clear(proposed, width)
                    if rear_ok and proposed < adaptive - 0.05:
                        gap_shaping_diag.update({"applied": True, "original_mps": round(adaptive,2),
                                                 "new_mps": round(proposed,2)})
                        adaptive = proposed
                    gap_shaping_diag.update({"wanted_mps": round(wanted,2), "rear_gate": rear_reason,
                                             "rear": rear_diag, "score": score_diag})
                else:
                    gap_shaping_diag["skip_reason"] = "lane_geometry_unverified"

            global_d = self._global_signed_d()
            can_start_rejoin = (
                self._handoff_permitted()
                and rearmed
                and left_is_nondashed
                and global_d is not None
                and abs(global_d) <= self.rejoin_start_global_d_m
                and lane_ok
                and base_fresh
            )
            if can_start_rejoin:
                rejoin = self._generate_rejoin_path(now)
                if rejoin is not None:
                    self.committed_rejoin_path = rejoin
                    self.rejoin_travel_m = 0.0
                    self.last_rejoin_xy = (ex, ey)
                    self.release_since = None
                    self.state = self.REJOIN
                    rospy.logwarn("HIGHWAY REJOIN COMMITTED global_d=%.2f length=%.1f", global_d, self.rejoin_length_m)
                    self._publish(rejoin, False, adaptive, True, {"reason":"rejoin_committed", "global_d":round(global_d,2), "follow":follow}, now, dt)
                    return

            # Failsafe direct release only when the two paths are already almost
            # coincident. This also prevents a lane-info dropout at the physical
            # merge from stopping the car forever.
            can_direct_release = (
                self._handoff_permitted()
                and rearmed
                and left_is_nondashed
                and global_d is not None
                and abs(global_d) <= self.release_global_d_m
                and base_fresh
            )
            if can_direct_release:
                if self.release_since is None:
                    self.release_since = now
                elif (now-self.release_since).to_sec() >= self.release_confirm_s:
                    self.completed_once = True
                    self.state = self.DONE
                    self._publish(self.latest_base_path, (not base_stop_fresh) or self.base_stop, adaptive, False, {"reason":"direct_release_near_global", "global_d":global_d, "follow":follow}, now, dt)
                    rospy.logwarn("HIGHWAY strategy DONE: direct global release d=%.2f", global_d)
                    return
            else:
                self.release_since = None

            self._publish(path, stop, adaptive, True, {
                "reason": inner_reason,
                "global_d": None if global_d is None else round(global_d,2),
                "inner_hold_travel_m": round(self.inner_hold_travel_m,2),
                "center_y8_m": None if center_y is None else round(center_y,3),
                "lane_grace": lane_grace,
                "lane_invalid_age_s": round(invalid_age,2),
                "center_filter_ready": len(self._centerline_local()) >= 3,
                "center_filter_frames": len(self.center_sample_history),
                "lane_hold_mode": "camera_temporal_follow",
                "handoff_pending": self.inner_handoff_pending,
                "camera_association": self.inner_association_diag,
                "camera_agreement_samples": self.inner_camera_match_samples,
                "reference_cross_track_m": None if self.inner_reference_cross_track_m is None else round(self.inner_reference_cross_track_m,3),
                "unverified_travel_m":round(self.inner_unverified_travel_m,2),
                "stitch_keep_m": round(self.inner_camera_keep_m,2),
                "stitch_blend_m": round(self.inner_camera_blend_m,2),
                "camera_filter_samples": self.inner_camera_samples,
                "camera_filter_reason": self.inner_camera_filter_reason,
                "camera_rejected_jumps": self.inner_camera_rejected_count,
                "camera_smoothed_y8_m": None if self.inner_filtered_y8_m is None else round(self.inner_filtered_y8_m,3),
                "camera_path_update_step_m": round(self.inner_camera_last_step_m,3),
                "lane_speed_limit_mps": None if self.inner_speed_lateral_limit_mps is None else round(self.inner_speed_lateral_limit_mps,2),
                "speed_recovery_confirmed": bool(self.inner_speed_stable_since is not None and
                    (now-self.inner_speed_stable_since).to_sec() >= self.inner_stable_speed_confirm_s),
                "rearmed": rearmed,
                "hold_elapsed_s": round(hold_elapsed_s,2),
                "left_detected": left_detected,
                "left_dashed": left_is_dashed,
                "left_nondashed": left_is_nondashed,
                "stable_now": stable_now,
                "rearm_dropout_age_s": None if self.inner_unstable_since is None
                    else round(max(0.0, (now-self.inner_unstable_since).to_sec()), 2),
                "next_change_reason": next_change_reason,
                "next_change_ready_s": round(next_change_ready_s, 2),
                "next_change_candidate_diag": next_change_diag,
                "gap_shaping": gap_shaping_diag,
                "follow": follow,
            }, now, dt)
            return

        if self.state == self.REJOIN:
            adaptive, emergency, follow = self._adaptive_speed(self.cruise_speed_mps) if obs_fresh and self.lane_info is not None else (self.cruise_speed_mps, False, {})
            global_d = self._global_signed_d()
            stop = emergency or not obs_fresh or self.committed_rejoin_path is None
            close_enough = global_d is not None and abs(global_d) <= self.rejoin_complete_global_d_m
            progressed = self.rejoin_travel_m >= 0.65*self.rejoin_length_m
            if (progressed or close_enough) and base_fresh and self._handoff_permitted():
                if self.release_since is None:
                    self.release_since = now
                elif (now-self.release_since).to_sec() >= self.release_confirm_s:
                    self.completed_once = True
                    self.state = self.DONE
                    self._publish(self.latest_base_path, (not base_stop_fresh) or self.base_stop, adaptive, False, {"reason":"rejoin_complete", "global_d":global_d, "travel_m":round(self.rejoin_travel_m,2), "follow":follow}, now, dt)
                    rospy.logwarn("HIGHWAY REJOIN COMPLETE global_d=%s travel=%.1f", "n/a" if global_d is None else "%.2f" % global_d, self.rejoin_travel_m)
                    return
            else:
                self.release_since = None
            self._publish(self.committed_rejoin_path, stop, adaptive, True, {"reason":"rejoining", "global_d":None if global_d is None else round(global_d,2), "travel_m":round(self.rejoin_travel_m,2), "progressed":progressed, "follow":follow}, now, dt)
            return


if __name__ == "__main__":
    try:
        HighwayLaneStrategyNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
