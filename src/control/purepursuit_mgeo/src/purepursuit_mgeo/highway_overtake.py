"""Camera-lane highway overtaking planner (ROS-free core).

Inside a fixed highway zone of the global route the ego vehicle moves one lane
to the LEFT at a time, passing the left-lane vehicle and merging in front of
it, until it reaches the lane whose left boundary is solid (at most
``max_changes`` changes).  It then only follows the lead vehicle as fast as
allowed, slows down for the zone end and hands control back to the global
path where that path meets the current lane.

Frames
------
map   MGeo local ENU: /localization/odometry, LiDAR tracks, published paths.
ego   base_link (rear-axle centre), x forward, y left: camera lane_info.
lane  Frenet frame (s along the current-lane centre, d to the left) built
      from the camera lane centre.  It is frozen in the map while a lane
      change is in progress, so camera lane-identity switches cannot bend the
      committed manoeuvre.

Nearby vehicles are assumed to keep their lane at constant speed, so each
LiDAR track is reduced to (s, d, v_s) in the lane frame.

Integration gates (``gate_required``)
-------------------------------------
Like the integration's highway_lane_strategy, the planner then activates only
while route_mission_gate reports the highway route (/planning/highway_route_active)
and commits a lane change only while it also grants the lane-change request
(/planning/highway_lane_change_request).  A gate that drops mid-zone blocks
new commits; a committed change is still finished.  The published state uses
the integration names (INTEGRATION_STATE); the planner's own is "phase".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from purepursuit_mgeo.lane_geometry import pose_at
from purepursuit_mgeo.motion import diagonal_progress, lead_brake_decision

Point = Tuple[float, float]
Pose = Tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OvertakeConfig:
    # Highway zone on the global route, as route arc length [m].  The ROS
    # node resolves these from fixed map points (zone_*_xy).
    zone_start_s: float = 0.0
    zone_end_s: float = 0.0
    # Entry lane: its right edge converges onto the divider between these two
    # route positions.  A change out of the entry lane must keep the front
    # corner off that converging solid edge.
    entry_taper_start_s: Optional[float] = None
    entry_taper_end_s: Optional[float] = None
    entry_wheel_margin_m: float = 0.25
    rearm_back_m: float = 100.0
    # route_mission_gate zone/permission flags must be fresh and true.
    gate_required: bool = False
    gate_timeout_s: float = 0.6

    # Speed [m/s], acceleration [m/s^2]
    cruise_speed_mps: float = 15.0
    max_speed_mps: float = 22.0
    end_speed_mps: float = 15.0
    end_decel_mps2: float = 1.5
    end_margin_m: float = 10.0
    target_accel_mps2: float = 1.5
    target_decel_mps2: float = 4.0
    stale_obstacle_speed_mps: float = 8.0

    # Lane change geometry
    max_changes: int = 3
    max_entry_angle_deg: float = 5.0
    max_lateral_accel_mps2: float = 1.5
    change_start_m: float = 2.0
    min_plan_speed_mps: float = 3.0
    capture_tolerance_m: float = 0.35
    room_margin_m: float = 10.0
    commit_confirm_ticks: int = 1
    solid_confirm_frames: int = 3

    # Gap acceptance (aggressive): as soon as the ego is even slightly ahead
    # of the left-lane car and faster than it (the gap is opening), only the
    # minimum bumper gap is required.  A closing gap additionally needs a short
    # time gap plus the braking distance to match speeds.
    gap_front_min_m: float = 3.0
    gap_rear_min_m: float = 1.0
    gap_time_s: float = 0.5
    gap_open_time_s: float = 0.0
    gap_comfort_decel_mps2: float = 3.0
    gap_post_change_s: float = 2.0
    gap_check_accel_mps2: float = 1.0
    gap_check_slow_accel_mps2: float = 0.5
    gap_check_dt_s: float = 0.1
    front_speed_margin_mps: float = 0.5
    rear_speed_margin_mps: float = 0.5
    lateral_margin_m: float = 0.3
    permission_hold_m: float = 40.0

    # Lead following
    follow_standstill_m: float = 4.0
    follow_time_gap_s: float = 1.6
    follow_gain: float = 0.35
    follow_search_m: float = 120.0
    emergency_gap_m: float = 1.5
    emergency_ttc_s: float = 1.0

    # Vehicle
    vehicle_length_m: float = 4.635
    vehicle_width_m: float = 1.892
    vehicle_center_from_base_m: float = 1.5

    # Camera lane reference
    lane_timeout_s: float = 0.6
    min_track_age: int = 2
    lane_width_default_m: float = 3.5
    lane_width_min_m: float = 2.8
    lane_width_max_m: float = 4.2
    eval_x_m: float = 7.0
    ref_filter_tau_s: float = 0.6
    ref_jump_reject_m: float = 0.9
    ref_heading_reject_deg: float = 4.0
    reacquire_tolerance_m: float = 0.8
    max_ref_curvature: float = 5e-4
    ref_far_default_m: float = 40.0
    ref_back_m: float = 30.0

    # Input freshness
    odom_timeout_s: float = 0.5
    obstacle_timeout_s: float = 0.6
    base_timeout_s: float = 1.0
    lane_hysteresis_m: float = 0.4

    # Output path
    path_back_m: float = 3.0
    path_min_ahead_m: float = 80.0
    path_ahead_time_s: float = 5.0
    path_step_m: float = 1.0
    recenter_min_m: float = 5.0

    # Hand-over: at the zone end control returns to the global path.  An
    # offset above this is logged (the ego did not reach the final lane).
    handover_warn_m: float = 1.0

    # The zone end is a lane funnel: lanes short of the final one end there.
    # Within deadline_window_m of the last point where a change still fits,
    # deadline_mode decides how the ego gets out of such a lane:
    #   off    keep chasing a gap ahead; otherwise hand over at the zone end
    #          (the first design)
    #   yield  slow to the end speed, and below a blocking target-lane car to
    #          merge behind it
    #   force  slow to the end speed; at the last point commit unless a
    #          target-lane car is alongside the slot right now
    #   both   yield, and force at the last point
    deadline_mode: str = "yield"
    deadline_window_m: float = 100.0
    yield_speed_drop_mps: float = 3.0
    yield_min_speed_mps: float = 8.0
    force_margin_m: float = 6.0

    @classmethod
    def from_mapping(cls, values: Dict) -> "OvertakeConfig":
        config = cls()
        for item in fields(cls):
            if item.name in values and values[item.name] is not None:
                default = getattr(config, item.name)
                value = values[item.name]
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int) and not isinstance(default, bool):
                    value = int(value)
                elif isinstance(default, float) or default is None:
                    value = float(value)
                setattr(config, item.name, value)
        return config

    def has_taper(self) -> bool:
        return (self.entry_taper_start_s is not None and self.entry_taper_end_s is not None
                and self.entry_taper_end_s > self.entry_taper_start_s)

    def entry_edge_d(self, route_s: float, current_width: float) -> float:
        """Entry-lane right edge, lateral to the current lane centre (+ left).

        The divider lies at +current_width/2; the edge starts one nominal
        lane width to its right and converges onto it over the taper.
        """
        u = clamp((route_s - self.entry_taper_start_s)
                  / (self.entry_taper_end_s - self.entry_taper_start_s), 0.0, 1.0)
        return 0.5*current_width - self.lane_width_default_m*(1.0 - u)

    @property
    def front_extent_m(self) -> float:
        return self.vehicle_center_from_base_m + 0.5*self.vehicle_length_m

    @property
    def rear_extent_m(self) -> float:
        """Signed s of the rear bumper relative to base_link (usually < 0)."""
        return self.vehicle_center_from_base_m - 0.5*self.vehicle_length_m


def align_with_controller(cfg: OvertakeConfig, lateral_mps2: float, accel_mps2: float,
                          decel_mps2: float, lateral_margin: float = 0.9,
                          end_decel_margin: float = 0.8) -> Dict:
    """Keep the planner's ego prediction within the speed controller's limits.

    adaptive_curvature_purepursuit slows a curved path to its lateral
    acceleration limit and slews its speed command by max_accel/max_decel (a
    stop brakes at once).  A change planned or gap-checked beyond these limits
    would be driven slower than predicted, and an end-of-zone braking curve at
    the controller's own limit leaves nothing for the vehicle's response lag.
    Lowers the matching fields of `cfg` in place (limits <= 0 are ignored);
    returns {field: (old, new)}.
    """
    limits = {
        "max_lateral_accel_mps2": lateral_margin*lateral_mps2,
        "target_accel_mps2": accel_mps2,
        "gap_check_accel_mps2": accel_mps2,
        "gap_check_slow_accel_mps2": accel_mps2,
        "target_decel_mps2": decel_mps2,
        "end_decel_mps2": end_decel_margin*decel_mps2,
    }
    changed = {}
    for name, limit in limits.items():
        old = getattr(cfg, name)
        if math.isfinite(limit) and 0.0 < limit < old:
            setattr(cfg, name, limit)
            changed[name] = (old, limit)
    return changed


# ---------------------------------------------------------------------------
# Global route (zone flag, hand-over)
# ---------------------------------------------------------------------------
class Route:
    """Polyline with arc length; windowed projection that follows the ego."""

    def __init__(self, points: Sequence[Point]) -> None:
        self.points = [(float(x), float(y)) for x, y in points]
        if len(self.points) < 2:
            raise ValueError("route needs at least two points")
        self.s = [0.0]
        for (ax, ay), (bx, by) in zip(self.points, self.points[1:]):
            self.s.append(self.s[-1] + math.hypot(bx-ax, by-ay))
        self._hint: Optional[int] = None

    def _project_range(self, x: float, y: float, lo: int, hi: int):
        best = None
        for i in range(max(0, lo), min(len(self.points)-1, hi)):
            ax, ay = self.points[i]
            bx, by = self.points[i+1]
            vx, vy = bx-ax, by-ay
            l2 = vx*vx + vy*vy
            if l2 < 1e-12:
                continue
            t = clamp(((x-ax)*vx + (y-ay)*vy)/l2, 0.0, 1.0)
            px, py = ax+t*vx, ay+t*vy
            d2 = (x-px)**2 + (y-py)**2
            if best is None or d2 < best[0]:
                length = math.sqrt(l2)
                lateral = (vx*(y-ay) - vy*(x-ax))/length
                best = (d2, self.s[i] + t*length, lateral, i)
        return best

    def project(self, x: float, y: float, window: int = 300) -> Tuple[float, float]:
        """Return (arc length s, signed lateral offset, + = left of route)."""
        best = None
        if self._hint is not None:
            best = self._project_range(x, y, self._hint-window, self._hint+window)
        if best is None or best[0] > 20.0**2:
            best = self._project_range(x, y, 0, len(self.points))
        self._hint = best[3]
        return best[1], best[2]

    def s_of_xy(self, x: float, y: float) -> float:
        return self._project_range(x, y, 0, len(self.points))[1]


def polyline_offset(points: Sequence[Point], x: float, y: float) -> Optional[float]:
    """Signed lateral offset (+ left) of (x, y) from the nearest segment."""
    best = None
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        vx, vy = bx-ax, by-ay
        l2 = vx*vx + vy*vy
        if l2 < 1e-12:
            continue
        t = clamp(((x-ax)*vx + (y-ay)*vy)/l2, 0.0, 1.0)
        d2 = (x-ax-t*vx)**2 + (y-ay-t*vy)**2
        if best is None or d2 < best[0]:
            best = (d2, (vx*(y-ay) - vy*(x-ax))/math.sqrt(l2))
    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# Camera lane observation
# ---------------------------------------------------------------------------
@dataclass
class LaneObservation:
    stamp: float                 # wall time the frame was observed / received
    stamp_is_capture: bool       # True: camera receive time; False: publish time
    left_coef: Tuple[float, float, float]
    right_coef: Tuple[float, float, float]
    left_type: str
    left_dashed: bool
    right_type: str
    width: float
    x_far: float

    @property
    def left_solid(self) -> bool:
        return self.left_type in ("white_solid", "yellow")


def _poly3(coef) -> Optional[Tuple[float, float, float]]:
    if not isinstance(coef, (list, tuple)) or not 1 <= len(coef) <= 3:
        return None
    if not _finite(*coef):
        return None
    padded = [0.0]*(3-len(coef)) + [float(v) for v in coef]
    return padded[0], padded[1], padded[2]


def poly_y(coef: Sequence[float], x: float) -> float:
    return (coef[0]*x + coef[1])*x + coef[2]


def parse_lane_info(info, receive_time: float,
                    cfg: OvertakeConfig) -> Tuple[Optional[LaneObservation], str]:
    """Validate one /perception/camera/lane_info JSON payload.

    Accepts both lane publishers in the repository (post_processing
    real_lane_node and lane/live_lane_info_publisher_v2): only the shared
    contract keys are used.  A frame is usable only when both ego boundaries
    are freshly measured, bracket the vehicle and are one lane apart.
    """
    if not isinstance(info, dict):
        return None, "not_a_dict"
    status = info.get("output_status")
    if status is not None and str(status).upper() != "FRESH":
        return None, "status_" + str(status).lower()
    if info.get("lane_valid") is False:
        return None, "lane_invalid"
    straddle = info.get("straddling_lane")
    if isinstance(straddle, dict) and straddle.get("detected"):
        return None, "straddling"
    stamp = info.get("timestamp")
    capture = info.get("observation_time_source") == "camera_receive_wall"
    stamp = float(stamp) if _finite(stamp) else float(receive_time)
    if not capture:
        stamp = min(stamp, float(receive_time))
    if receive_time - stamp > cfg.lane_timeout_s or stamp - receive_time > 0.2:
        return None, "stale"

    lanes = {}
    for side in ("left", "right"):
        lane = info.get(side + "_lane")
        if not isinstance(lane, dict) or not lane.get("detected"):
            return None, "no_" + side
        if lane.get("coasted") or lane.get("from_guide"):
            return None, side + "_predicted"
        age = lane.get("age")
        if age is not None and _finite(age) and int(age) < cfg.min_track_age:
            return None, side + "_young"
        coef = _poly3(lane.get("coef"))
        if coef is None:
            return None, side + "_coef"
        lanes[side] = (lane, coef)

    x = cfg.eval_x_m
    y_left = poly_y(lanes["left"][1], x)
    y_right = poly_y(lanes["right"][1], x)
    if not (y_left > 0.3 and y_right < -0.3):
        return None, "not_bracketing"
    width = y_left - y_right
    if not cfg.lane_width_min_m <= width <= cfg.lane_width_max_m:
        return None, "width_%.2f" % width

    far = []
    for lane, _ in lanes.values():
        x_range = lane.get("x_range_m")
        if isinstance(x_range, (list, tuple)) and len(x_range) >= 2 and _finite(x_range[1]):
            far.append(float(x_range[1]))
    x_far = clamp(min(far) if far else cfg.ref_far_default_m, 10.0, 60.0)
    left, right = lanes["left"][0], lanes["right"][0]
    left_type = str(left.get("type") or "")
    return LaneObservation(
        stamp=stamp,
        stamp_is_capture=capture,
        left_coef=lanes["left"][1],
        right_coef=lanes["right"][1],
        left_type=left_type,
        left_dashed=bool(left.get("dashed")) or left_type == "white_dashed",
        right_type=str(right.get("type") or ""),
        width=width,
        x_far=x_far,
    ), "ok"


# ---------------------------------------------------------------------------
# Lane reference (Frenet frame)
# ---------------------------------------------------------------------------
@dataclass
class LaneReference:
    """Lane centre: origin O, heading psi, curvature k, all in map.

    The centre's lateral offset from the tangent line is c(s) = k s^2 / 2,
    extrapolated linearly outside [-back, far] so a noisy curvature cannot
    bend the far field.
    """
    ox: float
    oy: float
    psi: float
    k: float
    far: float
    back: float

    @property
    def u(self) -> Point:
        return math.cos(self.psi), math.sin(self.psi)

    @property
    def n(self) -> Point:
        return -math.sin(self.psi), math.cos(self.psi)

    def _c(self, s: float) -> float:
        if s > self.far:
            return 0.5*self.k*self.far**2 + self.k*self.far*(s-self.far)
        if s < -self.back:
            return 0.5*self.k*self.back**2 - self.k*self.back*(s+self.back)
        return 0.5*self.k*s*s

    def _dc(self, s: float) -> float:
        return self.k*clamp(s, -self.back, self.far)

    def to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        ux, uy = self.u
        dx, dy = x-self.ox, y-self.oy
        s = dx*ux + dy*uy
        return s, (-dx*uy + dy*ux) - self._c(s)

    def point(self, s: float, d: float = 0.0) -> Point:
        ux, uy = self.u
        lateral = self._c(s) + d
        return self.ox + s*ux - lateral*uy, self.oy + s*uy + lateral*ux

    def heading(self, s: float) -> float:
        return self.psi + math.atan(self._dc(s))

    def anchored(self, s: float) -> "LaneReference":
        """Same lane centre, origin moved to arc length s."""
        x, y = self.point(s)
        return LaneReference(x, y, self.heading(s), self.k,
                             max(0.0, self.far-s), self.back)

    @classmethod
    def from_observation(cls, obs: LaneObservation, pose: Pose,
                         cfg: OvertakeConfig) -> "LaneReference":
        """Origin on the lane centre at the ego; direction of the LEFT line.

        The left line is the divider every manoeuvre crosses and runs parallel
        to the target lanes.  The right line is averaged in only when it is
        parallel too: a converging edge (entry-lane taper, exit) would tilt a
        midpoint line and the tilt grows with distance.
        """
        al, bl, cl = obs.left_coef
        ar, br, cr = obs.right_coef
        c = 0.5*(cl + cr)
        if abs(bl - br) < 0.01 and abs(al - ar) < 2e-4:
            a, b = 0.5*(al + ar), 0.5*(bl + br)
        else:
            a, b = al, bl
        px, py, yaw = pose
        k = clamp(2.0*a, -cfg.max_ref_curvature, cfg.max_ref_curvature)
        return cls(px - math.sin(yaw)*c, py + math.cos(yaw)*c,
                   wrap_angle(yaw + math.atan(b)), k, obs.x_far, cfg.ref_back_m)

    @classmethod
    def from_pose(cls, pose: Pose, cfg: OvertakeConfig) -> "LaneReference":
        return cls(pose[0], pose[1], pose[2], 0.0, cfg.ref_far_default_m, cfg.ref_back_m)


def lane_path(ref: LaneReference, s0: float, s1: float, d: float, step: float) -> List[Point]:
    n = max(2, int(math.ceil((s1-s0)/step))+1)
    return [ref.point(s0 + (s1-s0)*i/(n-1), d) for i in range(n)]


def recenter_length(d0: float, slope0: float, speed: float, cfg: OvertakeConfig) -> float:
    """Shortest cubic join from (d0, slope0) to the lane centre within the lateral
    acceleration limit.  A cubic's d'' is linear, so its extremes are at the
    ends: |6 d0 + 4 L m0| / L^2 and |6 d0 + 2 L m0| / L^2."""
    k_max = cfg.max_lateral_accel_mps2/max(speed, cfg.min_plan_speed_mps)**2
    length = cfg.recenter_min_m
    while length < 300.0 and max(abs(6.0*d0 + 4.0*length*slope0),
                                 abs(6.0*d0 + 2.0*length*slope0)) > k_max*length*length:
        length += 1.0
    return length


def recenter_path(ref: LaneReference, ego_s: float, d0: float, slope0: float,
                  length: float, s1: float, cfg: OvertakeConfig) -> List[Point]:
    """Lane-centre path that starts at the ego's offset and heading.

    A cubic Hermite joins (d0, slope0) to the centre over `length`; behind the
    ego the path continues the ego's heading.  A fresh lane measurement or the
    end of a lane change therefore never steps the path sideways.
    """
    points = []
    s = ego_s - cfg.path_back_m
    while s <= s1:
        u = (s - ego_s)/length
        if u < 0.0:
            d = d0 + slope0*(s - ego_s)
        elif u < 1.0:
            d = d0*(2.0*u**3 - 3.0*u**2 + 1.0) + length*slope0*(u**3 - 2.0*u**2 + u)
        else:
            d = 0.0
        points.append(ref.point(s, d))
        s += cfg.path_step_m
    return points


# ---------------------------------------------------------------------------
# Lane change geometry
# ---------------------------------------------------------------------------
def change_geometry(width: float, speed: float, cfg: OvertakeConfig) -> Tuple[float, float]:
    """Shortest (length, ramp ratio) of a diagonal_progress shift of `width`.

    diagonal_progress has max slope 1/(1-r) and max second derivative
    1.5/(r(1-r)) (normalised).  The slope bounds the entry angle; the second
    derivative bounds lateral acceleration v^2*kappa at `speed`.
    """
    v = max(speed, cfg.min_plan_speed_mps)
    tan_max = math.tan(math.radians(clamp(cfg.max_entry_angle_deg, 0.5, 30.0)))
    a_lat = max(0.1, cfg.max_lateral_accel_mps2)
    best = None
    for i in range(35):
        r = 0.15 + i*(0.49-0.15)/34
        slope_length = width/((1.0-r)*tan_max)
        accel_length = math.sqrt(1.5*width*v*v/(a_lat*r*(1.0-r)))
        length = max(slope_length, accel_length)
        if best is None or length < best[0]:
            best = (length, r)
    return best


@dataclass
class ChangePlan:
    ref: LaneReference          # frozen lane frame of the source lane
    s_commit: float             # ego s at commitment
    s_start: float              # start of the lateral transition
    length: float
    ramp: float
    d0: float
    target_d: float
    v_plan: float
    path: List[Point] = field(default_factory=list)

    @property
    def s_end(self) -> float:
        return self.s_start + self.length

    def offset_at(self, s: float) -> float:
        if s <= self.s_start:
            return self.d0
        u = clamp((s-self.s_start)/self.length, 0.0, 1.0)
        return self.d0 + (self.target_d-self.d0)*diagonal_progress(u, self.ramp)


def make_change_plan(ref: LaneReference, ego_s: float, ego_d: float, target_d: float,
                     v_plan: float, cfg: OvertakeConfig, with_path: bool = True) -> ChangePlan:
    """Shift from the ego's current offset to the target-lane centre."""
    length, ramp = change_geometry(max(1.0, target_d - ego_d), v_plan, cfg)
    plan = ChangePlan(ref, ego_s, ego_s + cfg.change_start_m, length, ramp,
                      ego_d, target_d, v_plan)
    if not with_path:
        return plan
    tail = max(cfg.path_min_ahead_m, cfg.path_ahead_time_s*v_plan)
    s = ego_s - cfg.path_back_m
    while s <= plan.s_end + tail:
        plan.path.append(ref.point(s, plan.offset_at(s)))
        s += cfg.path_step_m
    return plan


# ---------------------------------------------------------------------------
# Tracks in the lane frame and gap acceptance
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float


@dataclass
class LaneTrack:
    id: int
    s: float          # centre
    d: float
    v: float          # along the lane
    length: float
    width: float
    lane: int         # lane index relative to the ego lane (+ = left)
    x: float = 0.0
    y: float = 0.0


@dataclass
class GapDecision:
    ok: bool
    reason: str
    blocker: Optional[int] = None
    time_s: Optional[float] = None
    gap_m: Optional[float] = None
    required_m: Optional[float] = None

    def as_dict(self) -> Dict:
        out = {"ok": self.ok, "reason": self.reason}
        if self.blocker is not None:
            out.update(blocker=self.blocker, t=round(self.time_s, 2),
                       gap=round(self.gap_m, 2), required=round(self.required_m, 2))
        return out


def evaluate_change(plan: ChangePlan, ego_s: float, v0: float,
                    tracks: Iterable[LaneTrack], cfg: OvertakeConfig) -> GapDecision:
    """Is it safe to start `plan` now?

    Tracks keep their lane at constant speed, predicted twice: as measured,
    and pessimistically (slower if now in front, faster if now behind).  The
    ego follows the plan with two acceleration profiles toward the plan speed
    and the same lead-following law the planner applies while changing, over
    the lateral transition plus gap_post_change_s.  Whenever the ego
    footprint overlaps a track laterally, the track is in front of or behind
    the ego at that instant, and the bumper gap must cover a minimum distance,
    a time gap and the braking distance needed to match speeds.  A track can
    therefore be passed (merge ahead of it) or let go by (merge behind it).
    """
    tracks = list(tracks)
    v_plan = max(plan.v_plan, cfg.min_plan_speed_mps)
    horizon = max(0.0, plan.s_end - ego_s)/v_plan + cfg.gap_post_change_s
    ego_center0 = ego_s + cfg.vehicle_center_from_base_m
    half_w = 0.5*cfg.vehicle_width_m
    brake = 2.0*max(0.1, cfg.gap_comfort_decel_mps2)
    dt = cfg.gap_check_dt_s
    relevant = [trk for trk in tracks if trk.lane in (0, 1)
                and not (trk.lane == 0 and trk.s < ego_center0)]   # skip our own follower

    def speed(trk: LaneTrack, pessimistic: bool) -> float:
        if not pessimistic:
            return max(0.0, trk.v)
        if trk.s >= ego_center0:
            return max(0.0, trk.v - cfg.front_speed_margin_mps)
        return max(0.0, trk.v + cfg.rear_speed_margin_mps)

    for pessimistic in (True, False):
        speeds = [(trk, speed(trk, pessimistic)) for trk in relevant]
        for accel in (cfg.gap_check_slow_accel_mps2, cfg.gap_check_accel_mps2):
            s, v, t = ego_s, v0, 0.0
            while t <= horizon + 1e-9:
                d_ego = plan.offset_at(s)
                center = s + cfg.vehicle_center_from_base_m
                lead_gap, lead_v = None, None
                for trk, v_trk in speeds:
                    s_trk = trk.s + v_trk*t
                    reach = half_w + 0.5*trk.width + cfg.lateral_margin_m
                    overlap = abs(trk.d - d_ego) < reach
                    front = s_trk >= center
                    if front:
                        gap = (s_trk - 0.5*trk.length) - (s + cfg.front_extent_m)
                        if (overlap or (abs(trk.d - plan.target_d) < reach and gap > 0.0)) \
                                and (lead_gap is None or gap < lead_gap):
                            lead_gap, lead_v = gap, v_trk
                    if not overlap:
                        continue
                    if front:
                        if v_trk >= v:      # pulling away: the gap is opening
                            need = cfg.gap_front_min_m + cfg.gap_open_time_s*v
                        else:
                            need = (cfg.gap_front_min_m + cfg.gap_time_s*v
                                    + (v-v_trk)**2/brake)
                    else:
                        gap = (s + cfg.rear_extent_m) - (s_trk + 0.5*trk.length)
                        if v_trk > v:
                            need = (cfg.gap_rear_min_m + cfg.gap_time_s*v_trk
                                    + (v_trk-v)**2/brake)
                        else:
                            need = cfg.gap_rear_min_m + cfg.gap_open_time_s*v_trk
                    if gap < need:
                        kind = ("current_front" if trk.lane == 0
                                else "target_front" if front else "target_rear")
                        return GapDecision(False, kind, trk.id, t, gap, need)
                # Same speed law as the planner during the change.
                command = v_plan
                if lead_gap is not None:
                    desired = cfg.follow_standstill_m + cfg.follow_time_gap_s*v
                    command = min(command, max(0.0, lead_v + cfg.follow_gain*(lead_gap-desired)))
                if command > v:
                    v = min(command, v + accel*dt)
                else:
                    v = max(command, v - cfg.target_decel_mps2*dt)
                s += v*dt
                t += dt
    return GapDecision(True, "clear")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
@dataclass
class PlannerOutput:
    state: str
    path: Optional[List[Point]] = None
    target_speed: float = 0.0
    stop: bool = False
    active: bool = False
    zone_active: bool = False
    fast_change: bool = False
    lead_brake: bool = False
    lead_emergency: bool = False
    status: Dict = field(default_factory=dict)
    events: List[str] = field(default_factory=list)


class HighwayOvertakePlanner:
    OFF = "OFF"
    WAIT_LANE = "WAIT_LANE"
    CHASE = "CHASE"
    CHANGE = "CHANGE"
    REACQUIRE = "REACQUIRE"
    FOLLOW = "FOLLOW"
    DONE = "DONE"
    # Names of highway_lane_strategy; route_mission_gate reads "LANE_CHANGE".
    # Like it, WAIT_GAP means "still on the global path's lane": after the
    # first change the wait for the next gap is INNER_HOLD, the state in which
    # the v61 fusion accepts the ego's offset from the global path.
    INTEGRATION_STATE = {
        OFF: "OFF", WAIT_LANE: "WAIT_GAP", CHASE: "WAIT_GAP", CHANGE: "LANE_CHANGE",
        REACQUIRE: "INNER_HOLD", FOLLOW: "INNER_HOLD", DONE: "DONE",
    }

    def integration_state(self, state: str) -> str:
        if state == self.CHASE and self.lane_index > 0:
            return "INNER_HOLD"
        return self.INTEGRATION_STATE[state]

    def __init__(self, cfg: OvertakeConfig, route_points: Sequence[Point]) -> None:
        self.cfg = cfg
        self.route = Route(route_points)
        self.pose_history: List[Tuple[float, Pose]] = []
        self.odom: Optional[Tuple[float, float, float, float, float]] = None
        self.obstacles: Optional[List[Obstacle]] = None
        self.obstacles_at: Optional[float] = None
        self.base_path: Optional[List[Point]] = None
        self.base_stop = False
        self.base_at: Optional[float] = None
        self.pending_lane: Optional[Tuple[float, Dict]] = None
        self.lane_reject = "none"
        self.route_gate: Optional[Tuple[float, bool]] = None
        self.lane_change_request: Optional[Tuple[float, bool]] = None
        self.gates: Dict[str, bool] = {}
        self.reset()

    # ----- state --------------------------------------------------------
    def reset(self) -> None:
        self.state = self.OFF
        self.ref: Optional[LaneReference] = None
        self.width = self.cfg.lane_width_default_m
        self.lane_index = 0                 # 0 = entry lane, +1 per left change
        self.plan: Optional[ChangePlan] = None
        self.last_obs: Optional[LaneObservation] = None
        self.last_obs_at: Optional[float] = None
        self.last_obs_xy: Optional[Point] = None
        self.solid_frames = 0
        self.confirm_ticks = 0
        self.track_lanes: Dict[int, int] = {}
        self.speed_out: Optional[float] = None
        self.last_step: Optional[float] = None
        self.last_gap: Optional[GapDecision] = None
        self.reject_since: Optional[float] = None
        self.deadline = False

    # ----- inputs -------------------------------------------------------
    def on_odom(self, t: float, x: float, y: float, yaw: float, speed: float) -> None:
        self.odom = (t, x, y, yaw, speed)
        self.pose_history.append((t, (x, y, yaw)))
        if len(self.pose_history) > 400:
            del self.pose_history[:100]

    def on_lane_info(self, t: float, info: Dict) -> None:
        self.pending_lane = (t, info)

    def on_obstacles(self, t: float, obstacles: List[Obstacle]) -> None:
        self.obstacles = list(obstacles)
        self.obstacles_at = t

    def on_base_path(self, t: float, points: Optional[List[Point]],
                     stop: Optional[bool] = None) -> None:
        if points is not None:
            self.base_path = list(points)
            self.base_at = t
        if stop is not None:
            self.base_stop = bool(stop)

    def on_route_gate(self, t: float, active: bool) -> None:
        self.route_gate = (t, bool(active))

    def on_lane_change_request(self, t: float, requested: bool) -> None:
        self.lane_change_request = (t, bool(requested))

    # ----- helpers ------------------------------------------------------
    def _fresh(self, stamp: Optional[float], timeout: float, now: float) -> bool:
        return stamp is not None and now - stamp <= timeout

    def _gate(self, sample: Optional[Tuple[float, bool]], now: float) -> bool:
        if not self.cfg.gate_required:
            return True
        return (sample is not None and sample[1]
                and 0.0 <= now - sample[0] <= self.cfg.gate_timeout_s)

    def _pose_at(self, stamp: float) -> Optional[Pose]:
        pose = pose_at(self.pose_history, stamp, tolerance=0.15)
        if pose is None and self.odom is not None:
            pose = self.odom[1:4]
        return pose

    def target_offset(self) -> float:
        """Lateral offset of the left neighbour lane's centre in the lane frame."""
        return 0.5*self.width + 0.5*self.cfg.lane_width_default_m

    def _lane_band(self, lane: int) -> Tuple[float, float]:
        """Lateral band [low, high] of a lane relative to the ego lane frame."""
        half, nominal = 0.5*self.width, self.cfg.lane_width_default_m
        if lane == 0:
            return -half, half
        if lane > 0:
            return half + (lane-1)*nominal, half + lane*nominal
        return -half + lane*nominal, -half + (lane+1)*nominal

    def _lane_of(self, d: float) -> int:
        half, nominal = 0.5*self.width, self.cfg.lane_width_default_m
        if abs(d) <= half:
            return 0
        if d > 0:
            return 1 + int((d - half)//nominal)
        return -1 - int((-d - half)//nominal)

    def _lane_tracks(self, ref: LaneReference, frame_lane: int, now: float,
                     remember: bool = True) -> Optional[List[LaneTrack]]:
        """LiDAR tracks in the frame of `ref`, whose d=0 lane is `frame_lane`.

        Lane membership is kept per track id in absolute lane numbers
        (0 = entry lane) with hysteresis, so a car riding near a lane line does
        not flicker between lanes.  LaneTrack.lane is relative to the ego lane.
        """
        cfg = self.cfg
        if self.obstacles is None or not self._fresh(self.obstacles_at, cfg.obstacle_timeout_s, now):
            return None
        tracks = []
        seen = set()
        for ob in self.obstacles:
            s, d = ref.to_frenet(ob.x, ob.y)
            if not -cfg.follow_search_m <= s <= cfg.follow_search_m + 20.0:
                continue
            heading = ref.heading(s)
            v = ob.vx*math.cos(heading) + ob.vy*math.sin(heading)
            in_frame = self._lane_of(d)
            previous = self.track_lanes.get(ob.id) if remember else None
            if previous is not None:
                low, high = self._lane_band(previous - frame_lane)
                if low - cfg.lane_hysteresis_m <= d <= high + cfg.lane_hysteresis_m:
                    in_frame = previous - frame_lane
            if remember:
                self.track_lanes[ob.id] = frame_lane + in_frame
            seen.add(ob.id)
            tracks.append(LaneTrack(ob.id, s, d, v, max(0.5, ob.length), max(0.4, ob.width),
                                    frame_lane + in_frame - self.lane_index, ob.x, ob.y))
        if remember:
            for oid in list(self.track_lanes):
                if oid not in seen:
                    del self.track_lanes[oid]
        return tracks

    def _follow(self, tracks: Optional[List[LaneTrack]], ego_s: float, ego_d: float,
                v: float, preview_d: Optional[float] = None) -> Tuple[Optional[float], bool, Dict]:
        """Lead-vehicle speed (gap control).

        Leads are tracks overlapping the ego footprint laterally (ego_d) and,
        while changing lanes, tracks entirely ahead in the target lane
        (preview_d).  A target-lane track still alongside is not a lead: the
        committed plan only reaches its lane after it has gone.  Only a lead
        in the ego's path can trigger the emergency stop; a target-lane lead
        the ego does not overlap yet only lowers the speed.
        """
        cfg = self.cfg
        if tracks is None:
            return None, False, {"lead": None, "reason": "obstacles_stale"}
        desired = cfg.follow_standstill_m + cfg.follow_time_gap_s*v
        best = None     # (target speed, emergency, info) of the binding lead
        for trk in tracks:
            reach = 0.5*(cfg.vehicle_width_m+trk.width) + cfg.lateral_margin_m
            g = (trk.s - 0.5*trk.length) - (ego_s + cfg.front_extent_m)
            if trk.s < ego_s + cfg.vehicle_center_from_base_m or g > cfg.follow_search_m:
                continue
            in_path = abs(trk.d - ego_d) < reach
            ahead_in_target = preview_d is not None and abs(trk.d - preview_d) < reach and g > 0.0
            if not (in_path or ahead_in_target):
                continue
            closing = v - trk.v
            ttc = g/closing if closing > 0.1 else float("inf")
            target = clamp(trk.v + cfg.follow_gain*(g-desired), 0.0, cfg.max_speed_mps)
            # A lead pulling away is no emergency, however close it is.
            emergency = in_path and closing > -0.3 and (
                g < cfg.emergency_gap_m or ttc < cfg.emergency_ttc_s)
            if best is None or (emergency, -target) > (best[1], -best[0]):
                best = (target, emergency, {
                    "lead": trk.id, "gap": round(g, 2), "desired_gap": round(desired, 2),
                    "lead_v": round(trk.v, 2), "in_path": in_path,
                    "ttc": None if not math.isfinite(ttc) else round(ttc, 2),
                })
        if best is None:
            return None, False, {"lead": None}
        return best

    def _entry_path_clear(self, plan: ChangePlan, ego_s: float, route_s: float) -> bool:
        """Does the planned path keep the front corner off the converging edge?"""
        cfg = self.cfg
        half = 0.5*cfg.vehicle_width_m + cfg.entry_wheel_margin_m
        s = ego_s
        end = ego_s + (cfg.entry_taper_end_s - route_s) + 1.0
        while s <= end:
            corner_route_s = route_s + (s - ego_s) + cfg.front_extent_m
            if plan.offset_at(s) - half < cfg.entry_edge_d(corner_route_s, self.width):
                return False
            s += 1.0
        return True

    def _entry_design_speed(self, ref: LaneReference, ego_s: float, ego_d: float,
                            route_s: float, low: float, high: float) -> Tuple[float, bool]:
        """Highest design speed in [low, high] whose path clears the entry edge.

        A faster design has longer ramps and leaves the entry lane later, so
        near the end of the entry lane only slower (shorter) changes fit.
        """
        def clear(v: float) -> bool:
            plan = make_change_plan(ref, ego_s, ego_d, self.target_offset(), v, self.cfg,
                                    with_path=False)
            return self._entry_path_clear(plan, ego_s, route_s)

        if clear(high):
            return high, True
        if high <= low or not clear(low):
            return low, False
        for _ in range(12):
            mid = 0.5*(low+high)
            if clear(mid):
                low = mid
            else:
                high = mid
        return low, True

    def _speed_caps(self, route_s: float, tracks_fresh: bool) -> Dict[str, float]:
        cfg = self.cfg
        caps = {"max": cfg.max_speed_mps}
        remaining = max(0.0, cfg.zone_end_s - cfg.end_margin_m - route_s)
        caps["end"] = math.sqrt(cfg.end_speed_mps**2 + 2.0*cfg.end_decel_mps2*remaining)
        if not tracks_fresh:
            caps["obstacles_stale"] = cfg.stale_obstacle_speed_mps
        return caps

    def _accept_reference(self, obs: LaneObservation, now: float, ego: Pose,
                          mode: str) -> Tuple[bool, str]:
        """Fold a camera frame into the lane reference according to `mode`.

        init      adopt the frame directly;
        blend     low-pass filter, reject jumps (camera lane switches);
        reacquire accept only a frame centred on the frozen target lane.
        """
        cfg = self.cfg
        pose = self._pose_at(obs.stamp)
        if pose is None:
            return False, "no_pose"
        new = LaneReference.from_observation(obs, pose, cfg)
        s_new, d_new = new.to_frenet(ego[0], ego[1])
        new = new.anchored(s_new)
        width = clamp(obs.width, cfg.lane_width_min_m, cfg.lane_width_max_m)
        if mode == "init" or self.ref is None:
            self.ref, self.width = new, width
            return True, "init"
        s_old, _ = self.ref.to_frenet(ego[0], ego[1])
        target = self.plan.target_d if (mode == "reacquire" and self.plan) else 0.0
        ox, oy = self.ref.point(s_old, target)
        psi_old = self.ref.heading(s_old)
        nx, ny = -math.sin(psi_old), math.cos(psi_old)
        shift = (new.ox-ox)*nx + (new.oy-oy)*ny
        turn = wrap_angle(new.psi - psi_old)
        if abs(turn) > math.radians(cfg.ref_heading_reject_deg):
            return False, "heading_jump_%.1fdeg" % math.degrees(turn)
        if mode == "reacquire":
            # The frozen frame drifts with its heading error, so past the
            # planned end the lane the ego occupies is the new lane (and
            # nearer the frozen target than the source lane is).
            past_end = self.plan is not None and s_old >= self.plan.s_end
            inside = (abs(d_new) < 0.5*width - cfg.lane_hysteresis_m
                      and abs(shift) < 0.5*cfg.lane_width_default_m)
            if abs(shift) > cfg.reacquire_tolerance_m and not (past_end and inside):
                return False, "not_target_lane_%.2fm" % shift
            self.ref, self.width = new, width
            return True, "reacquired"
        if abs(shift) > cfg.ref_jump_reject_m:
            # A persistent, consistent jump is accepted only when the new
            # centre is the lane the ego physically occupies.
            if self.reject_since is None:
                self.reject_since = now
            if now - self.reject_since > 1.5 and abs(d_new) < 0.5*width:
                self.ref, self.width = new, width
                self.reject_since = None
                return True, "reinit_after_jump"
            return False, "jump_%.2fm" % shift
        self.reject_since = None
        dt = 0.0 if self.last_obs_at is None else clamp(obs.stamp - self.last_obs_at, 0.0, 1.0)
        alpha = 1.0 - math.exp(-dt/max(1e-3, cfg.ref_filter_tau_s)) if dt > 0 else 0.3
        self.ref = LaneReference(
            ox + alpha*shift*nx, oy + alpha*shift*ny,
            wrap_angle(psi_old + alpha*turn),
            self.ref.k + alpha*(new.k-self.ref.k),
            new.far, cfg.ref_back_m,
        )
        self.width += alpha*(width-self.width)
        return True, "blend"

    def _process_lane(self, now: float, ego: Pose, events: List[str]) -> None:
        if self.pending_lane is None:
            return
        received, info = self.pending_lane
        self.pending_lane = None
        obs, reason = parse_lane_info(info, received, self.cfg)
        if obs is None:
            self.lane_reject = reason
            return
        if self.state == self.WAIT_LANE:
            mode = "init"
        elif self.state == self.REACQUIRE:
            mode = "reacquire"
        elif self.state in (self.CHASE, self.FOLLOW):
            mode = "blend"
        else:
            return          # CHANGE keeps the frozen frame; OFF/DONE ignore
        ok, why = self._accept_reference(obs, now, ego, mode)
        self.lane_reject = "none" if ok else why
        if not ok:
            return
        self.last_obs, self.last_obs_at = obs, obs.stamp
        self.last_obs_xy = (ego[0], ego[1])
        self.solid_frames = self.solid_frames + 1 if obs.left_solid else 0
        if mode == "init":
            events.append("lane reference acquired (width %.2fm, left=%s)"
                          % (self.width, obs.left_type))
            self.state = self.CHASE
        elif mode == "reacquire":
            events.append("new lane %d measured (width %.2fm, left=%s)"
                          % (self.lane_index, self.width, obs.left_type))
            self.state = self.CHASE

    # ----- main step ----------------------------------------------------
    def step(self, now: float) -> PlannerOutput:
        cfg = self.cfg
        dt = 0.05 if self.last_step is None else clamp(now - self.last_step, 0.0, 0.2)
        self.last_step = now
        out = PlannerOutput(state=self.state)
        if self.odom is None or now - self.odom[0] > cfg.odom_timeout_s:
            out.stop = True
            out.status = {"state": self.integration_state(self.state), "phase": self.state,
                          "reason": "odom_stale"}
            return out
        _, ex, ey, eyaw, ev = self.odom
        ego = (ex, ey, eyaw)
        route_s, _ = self.route.project(ex, ey)
        in_zone = cfg.zone_start_s <= route_s < cfg.zone_end_s
        route_ok = self._gate(self.route_gate, now)
        permitted = route_ok and self._gate(self.lane_change_request, now)
        self.gates = {"route_gate": route_ok, "lane_change_permitted": permitted}
        out.zone_active = in_zone and route_ok
        events = out.events
        base_fresh = self.base_path is not None and self._fresh(self.base_at, cfg.base_timeout_s, now)

        if self.state == self.DONE and route_s < cfg.zone_start_s - cfg.rearm_back_m:
            self.reset()
            events.append("re-armed for a new lap")
        if self.state == self.OFF and in_zone and route_ok:
            self.state = self.WAIT_LANE
            self.lane_index = 0
            events.append("zone entered at s=%.1f: waiting for camera lane" % route_s)
        if self.state == self.WAIT_LANE and route_s >= cfg.zone_end_s:
            self.state = self.DONE
            events.append("zone ended without a camera lane")

        self._process_lane(now, ego, events)

        if self.state in (self.OFF, self.DONE, self.WAIT_LANE):
            return self._pass_through(out, now, dt, ego, ev, route_s, base_fresh)

        # ----- controlled states --------------------------------------
        if self.state in (self.CHANGE, self.REACQUIRE):
            # Frozen source-lane frame; REACQUIRE has already counted the change.
            ref = self.plan.ref
            frame_lane = self.lane_index - (1 if self.state == self.REACQUIRE else 0)
        else:
            ref, frame_lane = self.ref, self.lane_index
        ego_s, ego_d = ref.to_frenet(ex, ey)
        tracks = self._lane_tracks(ref, frame_lane, now)
        tracks_fresh = tracks is not None
        lane_fresh = self._fresh(self.last_obs_at, cfg.lane_timeout_s, now)
        # The divider type does not change within a few tens of metres, so a
        # recently confirmed dashed line still permits a change while the
        # camera briefly loses the lane (e.g. the narrowing entry lane).
        moved = (math.hypot(ex - self.last_obs_xy[0], ey - self.last_obs_xy[1])
                 if self.last_obs_xy is not None else float("inf"))
        left_dashed = (self.last_obs is not None and self.last_obs.left_dashed
                       and (lane_fresh or moved <= cfg.permission_hold_m))
        status = {
            "route_s": round(route_s, 1), "lane_index": self.lane_index,
            "ego_d": round(ego_d, 2), "lane_width": round(self.width, 2),
            "left_type": self.last_obs.left_type if (self.last_obs and lane_fresh) else None,
            "lane_reject": self.lane_reject,
            "base_stop_ignored": bool(self.base_stop),
        }

        # Zone end: control returns to the global path, which meets the final
        # lane there.  A change still in progress is finished first.
        if route_s >= cfg.zone_end_s and self.state != self.CHANGE:
            offset = polyline_offset(self.base_path, ex, ey) if base_fresh else None
            self.state = self.DONE
            events.append("hand-over to global path at zone end (offset %s)%s" % (
                "n/a" if offset is None else "%.2fm" % offset,
                "" if offset is not None and abs(offset) <= cfg.handover_warn_m
                else "  WARNING: not on the global path's lane"))
            return self._pass_through(out, now, dt, ego, ev, route_s, base_fresh)

        if self.state == self.CHASE and (
            self.lane_index >= cfg.max_changes or self.solid_frames >= cfg.solid_confirm_frames
        ):
            self.state = self.FOLLOW
            events.append("final lane reached (changes=%d, left=%s): follow only"
                          % (self.lane_index, status["left_type"]))

        if self.state == self.CHASE and not self.deadline and self._deadline_near(route_s, ev):
            self.deadline = True
            events.append("lane ends ahead: deadline mode '%s'" % cfg.deadline_mode)
        preview = self.plan.target_d if self.state == self.CHANGE else None
        follow_v, emergency, follow = self._follow(tracks, ego_s, ego_d, ev, preview)
        caps = self._speed_caps(route_s, tracks_fresh)
        if self.deadline and self.state == self.CHASE:
            caps["deadline"] = cfg.end_speed_mps       # a slower change is shorter
        free_speed = min(caps.values())
        if self.state == self.CHANGE:
            caps["plan"] = self.plan.v_plan
        v_cap = min(caps.values())
        target = v_cap if follow_v is None else min(v_cap, follow_v)
        status["caps"] = {k: round(v, 2) for k, v in caps.items()}
        status["follow"] = follow
        status["deadline"] = self.deadline

        path: Optional[List[Point]] = None
        ahead = max(cfg.path_min_ahead_m, cfg.path_ahead_time_s*max(ev, target))
        if self.state == self.CHASE:
            room = cfg.zone_end_s - route_s
            reason = "chasing"
            if not left_dashed:
                reason = "left_not_dashed" if lane_fresh else "lane_not_fresh"
            elif not tracks_fresh:
                reason = "obstacles_stale"
            elif not permitted:
                reason = "lane_change_not_requested" if route_ok else "route_gate_inactive"
            else:
                # Plan for the free-flow speed so the ego can accelerate to
                # the target-lane traffic during the change.  In the entry
                # lane the path must leave the lane before it closes, which
                # bounds the design speed (a faster plan is a longer path).
                v_plan = max(ev, free_speed)
                entry_clear = True
                if self.lane_index == 0 and cfg.has_taper():
                    v_plan, entry_clear = self._entry_design_speed(
                        ref, ego_s, ego_d, route_s, ev, v_plan)
                plan = make_change_plan(ref, ego_s, ego_d, self.target_offset(), v_plan, cfg)
                if not entry_clear:
                    reason = "entry_edge_too_close"
                elif room < (plan.s_end - ego_s) + cfg.room_margin_m:
                    reason = "no_room_before_zone_end"
                else:
                    decision = evaluate_change(plan, ego_s, ev, tracks, cfg)
                    if (not decision.ok and self.deadline and cfg.deadline_mode in ("force", "both")
                            and room < (plan.s_end - ego_s) + cfg.room_margin_m + cfg.force_margin_m
                            and self._slot_clear(ego_s, tracks)):
                        decision = GapDecision(True, "forced_at_lane_end")
                    self.last_gap = decision
                    status["gap"] = decision.as_dict()
                    if decision.ok:
                        self.confirm_ticks += 1
                        reason = "gap_clear_%d" % self.confirm_ticks
                        if self.confirm_ticks >= cfg.commit_confirm_ticks:
                            self.plan = plan
                            self.state = self.CHANGE
                            self.confirm_ticks = 0
                            self.deadline = False
                            events.append(
                                "LANE CHANGE %d->%d committed: v=%.1f plan_v=%.1f length=%.0fm "
                                "ramp=%.2f d0=%+.2f%s" % (
                                    self.lane_index, self.lane_index+1, ev, v_plan, plan.length,
                                    plan.ramp, ego_d, "" if decision.reason == "clear"
                                    else " (" + decision.reason + ")"))
                    else:
                        reason = "gap_" + decision.reason
                        yield_v = (self._yield_speed(decision, tracks, ev)
                                   if self.deadline and cfg.deadline_mode in ("yield", "both")
                                   else None)
                        if yield_v is not None:
                            target = min(target, yield_v)
                            status["caps"]["yield"] = round(yield_v, 2)
            if not reason.startswith("gap_clear"):
                self.confirm_ticks = 0
            status["reason"] = reason
            if self.state == self.CHASE:
                path = self._hold_path(ref, ego_s, ego_d, eyaw, ev, ahead)

        if self.state == self.CHANGE:
            plan = self.plan
            done = ego_s >= plan.s_end or (
                abs(ego_d - plan.target_d) <= cfg.capture_tolerance_m
                and ego_s >= plan.s_start + 0.7*plan.length
            )
            if done:
                self.lane_index += 1
                self.state = self.REACQUIRE
                self.solid_frames = 0
                events.append("lane change complete: now in lane %d (d=%+.2f)"
                              % (self.lane_index, ego_d - plan.target_d))
            else:
                status["reason"] = "changing_%.0f%%" % (
                    100.0*clamp((ego_s - plan.s_start)/plan.length, 0.0, 1.0))
                path = plan.path

        if self.state == self.REACQUIRE:
            # Finish the committed S-curve: the change is counted from 70 %,
            # but cutting its exit ramp short would swing the car sideways.
            status["reason"] = "waiting_new_lane_measurement(%s)" % self.lane_reject
            plan = self.plan
            n = int((ahead + cfg.path_back_m)/cfg.path_step_m) + 1
            path = [plan.ref.point(s, plan.offset_at(s)) for s in
                    (ego_s - cfg.path_back_m + i*cfg.path_step_m for i in range(n))]

        if self.state == self.FOLLOW:
            status.setdefault("reason", "follow")
            path = self._hold_path(ref, ego_s, ego_d, eyaw, ev, ahead)

        out.state = self.state
        out.path = path
        out.active = True
        out.fast_change = self.state == self.CHANGE
        out.stop = bool(emergency)
        lead_brake, lead_emergency = lead_brake_decision(
            follow, cfg.emergency_gap_m, cfg.emergency_ttc_s)
        out.lead_brake, out.lead_emergency = bool(lead_brake), bool(lead_emergency)
        out.target_speed = self._slew(0.0 if emergency else target, ev, dt)
        if emergency:
            events.append("EMERGENCY stop: %s" % follow)
        status.update(self._common_status(out, route_s))
        status["tracks"] = [
            {"id": t.id, "lane": t.lane, "s": round(t.s - ego_s, 1), "d": round(t.d, 2),
             "v": round(t.v, 2), "x": round(t.x, 2), "y": round(t.y, 2)}
            for t in (tracks or []) if t.lane in (-1, 0, 1, 2) and abs(t.s - ego_s) < 80.0
        ]
        out.status = status
        return out

    def _deadline_near(self, route_s: float, ev: float) -> bool:
        """Within deadline_window_m of the last point where a change fits."""
        cfg = self.cfg
        if cfg.deadline_mode not in ("yield", "force", "both"):
            return False
        length, _ = change_geometry(self.target_offset(), max(ev, cfg.end_speed_mps), cfg)
        slack = (cfg.zone_end_s - route_s) - (cfg.change_start_m + length + cfg.room_margin_m)
        return slack < cfg.deadline_window_m

    def _yield_speed(self, decision: GapDecision, tracks: Optional[List[LaneTrack]],
                     ev: float) -> Optional[float]:
        """Speed that lets the blocking target-lane car get ahead of the ego:
        below it, and below the ego's own speed while a faster one closes in."""
        if decision.reason not in ("target_front", "target_rear"):
            return None
        blocker = next((t for t in tracks or [] if t.id == decision.blocker), None)
        if blocker is None:
            return None
        return max(self.cfg.yield_min_speed_mps,
                   min(blocker.v, ev) - self.cfg.yield_speed_drop_mps)

    def _slot_clear(self, ego_s: float, tracks: Optional[List[LaneTrack]]) -> bool:
        """No target-lane car overlaps the ego's slot (bumper minima) right now."""
        cfg = self.cfg
        for trk in tracks or []:
            if trk.lane != 1:
                continue
            front_gap = (trk.s - 0.5*trk.length) - (ego_s + cfg.front_extent_m)
            rear_gap = (ego_s + cfg.rear_extent_m) - (trk.s + 0.5*trk.length)
            if front_gap < cfg.gap_front_min_m and rear_gap < cfg.gap_rear_min_m:
                return False
        return True

    def _hold_path(self, ref: LaneReference, ego_s: float, ego_d: float, ego_yaw: float,
                   speed: float, ahead: float) -> List[Point]:
        """Current-lane path joining the ego's offset and heading to the centre."""
        slope = math.tan(clamp(wrap_angle(ego_yaw - ref.heading(ego_s)), -0.3, 0.3))
        length = recenter_length(ego_d, slope, speed, self.cfg)
        return recenter_path(ref, ego_s, ego_d, slope, length, ego_s + ahead, self.cfg)

    def _pass_through(self, out: PlannerOutput, now: float, dt: float, ego: Pose,
                      ev: float, route_s: float, base_fresh: bool) -> PlannerOutput:
        cfg = self.cfg
        out.state = self.state
        out.path = self.base_path if base_fresh else None
        target = cfg.cruise_speed_mps
        follow = {"lead": None}
        emergency = False
        if self.state == self.WAIT_LANE:
            out.active = True
            tracks = self._lane_tracks(LaneReference.from_pose(ego, cfg), 0, now, remember=False)
            follow_v, emergency, follow = self._follow(tracks, 0.0, 0.0, ev)
            if follow_v is not None:
                target = min(target, follow_v)
        out.stop = (not base_fresh) or self.base_stop or emergency
        lead_brake, lead_emergency = lead_brake_decision(
            follow, cfg.emergency_gap_m, cfg.emergency_ttc_s)
        out.lead_brake = bool(out.active and lead_brake)
        out.lead_emergency = bool(out.active and lead_emergency)
        out.target_speed = self._slew(target, ev, dt)
        reason = "base_pass" if self.state == self.OFF else (
            "highway_strategy_done" if self.state == self.DONE else
            "waiting_camera_lane(%s)" % self.lane_reject)
        out.status = {"reason": reason, "follow": follow, "route_s": round(route_s, 1)}
        out.status.update(self._common_status(out, route_s))
        return out

    def _slew(self, target: float, ego_speed: float, dt: float) -> float:
        cfg = self.cfg
        if self.speed_out is None:
            self.speed_out = ego_speed
        step_up = cfg.target_accel_mps2*dt
        step_down = cfg.target_decel_mps2*dt
        self.speed_out = clamp(target, self.speed_out - step_down, self.speed_out + step_up)
        if target <= 0.0 and ego_speed < 0.5:
            self.speed_out = 0.0
        return max(0.0, self.speed_out)

    def _common_status(self, out: PlannerOutput, route_s: float) -> Dict:
        status = {
            "state": self.integration_state(out.state),
            "phase": out.state,
            "active": out.active,
            "zone_active": out.zone_active,
            "stop": out.stop,
            "target_speed_mps": round(out.target_speed, 2),
            "lane_changes_done": self.lane_index,
            "zone": [round(self.cfg.zone_start_s, 1), round(self.cfg.zone_end_s, 1)],
        }
        status.update(self.gates)
        return status
