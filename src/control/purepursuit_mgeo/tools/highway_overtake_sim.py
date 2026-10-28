#!/usr/bin/env python3
"""Offline closed-loop test of the highway overtaking planner (no ROS).

The real K-city highway section is reproduced as a straight road (road frame:
x along the road, y to the left) and placed in the map frame where the real
section is, so the planner sees map coordinates like in MORAI:

    y = 15.75  ─────────── solid (median side)
    y = 14.0     L4  forbidden lane (traffic allowed, ego must never enter)
    y = 12.25  ─────────── solid
    y = 10.5     L3  destination lane
    y =  8.75  ─ ─ ─ ─ ─ ─ dashed
    y =  7.0     L2
    y =  5.25  ─ ─ ─ ─ ─ ─ dashed
    y =  3.5     L1
    y =  1.75  ─ ─ ─ ─ ─ ─ entry divider (solid for x<-35), L1 edge after x=127
    y =  0.0     entry lane, narrows to zero width between x=55 and x=127
    y = -1.75  ─────────── solid right edge

x = 0 is the zone start (route s~1160), x = 430 the zone end (route s~1590)
where the global path reaches the L3 centre.  The planner code under test is
exactly purepursuit_mgeo.highway_overtake; Pure Pursuit, its steering limits
and the camera/LiDAR outputs are emulated.

Integration emulation: route_mission_gate opens the zone and grants the
lane-change request from route s 1118.7 to the handoff at the zone end, and
--controller emulates adaptive_curvature_purepursuit's speed handling (path
curvature limit, max_accel/max_decel slew) with the highway-test.env values
of a preset; the planner is aligned to those limits like the ROS node.

    python tools/highway_overtake_sim.py                      # morai traffic, 10 seeds
    python tools/highway_overtake_sim.py --scenario all --seeds 5 --out sim_out
    python tools/highway_overtake_sim.py --scenario morai --seed 3 --view
    python tools/highway_overtake_sim.py --controller example # LATERAL 1.0 / ACCEL 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "experimental/curvature_speed_purepursuit/src"))

from curvature_speed_purepursuit import planner as cp  # noqa: E402
from purepursuit_mgeo.highway_overtake import (  # noqa: E402
    HighwayOvertakePlanner, Obstacle, OvertakeConfig, Route, align_with_controller,
)
from purepursuit_mgeo.motion import diagonal_progress  # noqa: E402

# Road frame -> map frame, matching the real section (zone start = entry lane
# centre at (62.4, 215.0), road heading ~south).
MAP_THETA = math.atan2(-474.5, 2.3)
MAP_ORIGIN = (62.4, 215.0)
LANE_W = 3.5
ZONE_START_X, ZONE_END_X = 0.0, 430.0
# Entry lane right edge converges onto the divider between these x (map:
# route s ~1215 -> ~1287).  The divider is a solid+dashed pair, dashed on the
# entry side, from x=-35 (s~1125).
ENTRY_TAPER = (55.0, 127.0)
GLOBAL_MERGE = (75.0, 125.0)
WHEELBASE = 3.0
EGO_L, EGO_W, EGO_CENTER = 4.635, 1.892, 1.5
TRACK_HALF = 0.8
CAMERA_X_RANGE = (4.0, 40.0)
# route_mission_regions_overtake.json: the gate opens 41.5 m before the zone
# start (s 1118.742 vs 1160.239) and hands over at the zone end.
GATE_OPEN_BEFORE_ZONE_M = 1160.239 - 1118.742
MAX_ROUTE_OFFSET_M = 11.0
MAX_STEER = 0.6981317008
# adaptive_curvature_purepursuit settings from highway-test.env: LATERAL_ACCEL_LIMIT_MPS2,
# MAX_ACCEL_MPS2, MAX_DECEL_MPS2, LOOKAHEAD_MIN_M/GAIN/MAX_M/TIGHT_MIN_M/CURVATURE_GAIN,
# STEERING_FEEDFORWARD_WEIGHT, MAX_STEERING_RATE_RAD_S.
_EXAMPLE = dict(lateral=1.0, accel=1.0, decel=1.5, look_min=4.0, look_gain=0.35,
                look_max=12.0, look_tight=2.2, look_curv=6.0, ff=0.35, steer_rate=0.5)
CONTROLLERS = {
    "example": _EXAMPLE,                                    # highway-test.env.example
    "mission": dict(_EXAMPLE, lateral=3.8, accel=1.4, look_gain=0.25,
                    steer_rate=1.2),                        # bash run_test.sh turn mission
}


def road_to_map(x: float, y: float) -> Tuple[float, float]:
    c, s = math.cos(MAP_THETA), math.sin(MAP_THETA)
    return MAP_ORIGIN[0] + c*x - s*y, MAP_ORIGIN[1] + s*x + c*y


def map_to_road(x: float, y: float) -> Tuple[float, float]:
    c, s = math.cos(MAP_THETA), math.sin(MAP_THETA)
    dx, dy = x-MAP_ORIGIN[0], y-MAP_ORIGIN[1]
    return c*dx + s*dy, -s*dx + c*dy


# ---------------------------------------------------------------------------
# Road
# ---------------------------------------------------------------------------
@dataclass
class Line:
    name: str
    xs: List[float]
    ys: List[float]
    kinds: List[Tuple[float, float, str]]

    def y_at(self, x):
        x = np.asarray(x, float)
        y = np.interp(x, self.xs, self.ys)
        return np.where((x >= self.xs[0]) & (x <= self.xs[-1]), y, np.nan)

    def kind_at(self, x: float) -> Optional[str]:
        for x0, x1, kind in self.kinds:
            if x0 <= x <= x1:
                return kind
        return None


# Zone end = lane funnel (MGeo R_KR_PR_K-city_2025, road frame shifted by
# -0.44 m so its highway lanes match the idealised ones above): L1, L2 and L3
# converge on one node at x 432.9 (links A2256W000445/422/423).  The L1/L2
# line ends at x 345, a hatched island whose left edge (B2256W000025/1693)
# sweeps from the right edge across L1 and L2 to the end of the L2/L3 line,
# then recedes (B2256W000021) while the lane splits into two toll lanes
# (B2256W000184 between them, B2256W000159 left edge).  From x 210 an exit
# lane opens right of L1 (right edge B2256W001695).
FUNNEL_X = 432.9
SCORE_END_X = 520.0       # collisions and lines are scored into the toll lanes
ISLAND_EDGE = [(358.0, 1.66), (360.0, 1.76), (370.0, 2.46), (380.0, 3.47), (390.0, 4.47),
               (400.0, 5.50), (410.0, 6.51), (420.0, 7.49), (430.0, 8.52), (432.9, 8.83),
               (440.0, 8.07), (450.0, 7.10), (460.0, 6.09), (470.0, 5.09), (480.0, 4.10),
               (490.0, 3.29), (500.0, 3.25), (520.0, 3.14), (540.0, 3.06), (560.0, 2.96),
               (580.0, 2.89)]
TOLL_SPLIT = [(467.6, 8.53), (470.0, 8.34), (480.0, 7.51), (490.0, 6.85), (500.0, 6.82),
              (547.6, 6.86), (580.0, 6.85)]
TOLL_LEFT_EDGE = [(459.1, 12.32), (470.0, 11.91), (480.0, 11.56), (490.0, 11.36), (500.0, 11.33),
                  (550.0, 11.34), (560.0, 10.91), (570.0, 10.66), (580.0, 10.65)]
EXIT_EDGE = [(210.0, 1.70), (220.0, 1.11), (230.0, 0.49), (240.0, -0.08), (250.0, -0.66),
             (260.0, -1.24), (270.0, -1.82), (280.0, -2.25), (520.0, -2.22), (530.0, -2.27),
             (540.0, -2.62), (550.0, -3.46), (560.0, -4.90), (570.0, -7.39)]


def _line(name, points, kinds):
    return Line(name, [p[0] for p in points], [p[1] for p in points], kinds)


LINES = [
    Line("edge_right", [-400, ENTRY_TAPER[0], ENTRY_TAPER[1], 800], [-1.75, -1.75, 1.75, 1.75],
         [(-400, 210, "solid"), (210, 357, "dashed"), (357, 800, "solid")]),
    Line("entry_divider", [-400, ENTRY_TAPER[1]], [1.75, 1.75],
         [(-400, -35, "solid"), (-35, ENTRY_TAPER[1], "dashed")]),
    Line("l1_l2", [-400, 345], [5.25, 5.25], [(-400, 345, "dashed")]),
    Line("l2_l3", [-400, FUNNEL_X], [8.75, 8.75], [(-400, FUNNEL_X, "dashed")]),
    Line("l3_l4", [-400, 800], [12.25, 12.25], [(-400, 800, "solid")]),
    Line("l4_median", [-400, 800], [15.75, 15.75], [(-400, 800, "solid")]),
    _line("island", ISLAND_EDGE, [(358.0, 580.0, "solid")]),
    _line("toll_split", TOLL_SPLIT, [(467.6, 547.6, "dashed"), (547.6, 580.0, "solid")]),
    _line("toll_left", TOLL_LEFT_EDGE, [(459.1, 580.0, "solid")]),
    _line("exit_edge", EXIT_EDGE, [(210.0, 570.0, "solid")]),
]
LANE_TYPE = {"solid": "white_solid", "dashed": "white_dashed"}


def island_edge_y(x: float) -> float:
    """Left edge of the hatched island at x (nan outside it)."""
    return float(LINES[6].y_at(x))


def global_path_y(x: float) -> float:
    def ease(u):
        return diagonal_progress(min(1.0, max(0.0, u)), 0.3)
    if x < GLOBAL_MERGE[0]:
        return 0.0
    if x < GLOBAL_MERGE[1]:
        return LANE_W*ease((x-GLOBAL_MERGE[0])/(GLOBAL_MERGE[1]-GLOBAL_MERGE[0]))
    if x < 355.0:
        return LANE_W
    if x < ZONE_END_X:
        return LANE_W + 2*LANE_W*ease((x-355.0)/(ZONE_END_X-355.0))
    # Past the funnel the route takes the right toll lane (A2256W000153);
    # the first 20 m blend from the L3 centre onto the recorded path.
    blend = max(0.0, 1.0 - (x - ZONE_END_X)/20.0)
    tail = float(np.interp(x, [p[0] for p in GLOBAL_TAIL], [p[1] for p in GLOBAL_TAIL]))
    return tail + blend*(3*LANE_W - GLOBAL_TAIL[0][1])


# data/routes/2026_molit_comp_global_path.txt from s 1590, sim frame.
GLOBAL_TAIL = [(430.0, 10.11), (439.4, 9.89), (449.3, 8.98), (459.3, 7.94), (469.2, 6.86),
               (479.2, 5.83), (489.2, 5.10), (499.2, 5.04), (509.1, 4.99), (519.1, 4.98),
               (539.1, 4.95), (559.1, 4.75), (569.1, 4.57), (589.3, 4.55), (800.0, 4.55)]

# Where each lane's NPCs drive in the sample scenario (their end links, sim
# frame): L1 cars take the exit (A2256W000179/001105) and end at x 399.4, L2
# cars end on their merge link A2256W000422 at x 387.4 before it bends into
# L3, L3 cars follow A2256W000432 into the left toll lane and end at x 555.9.
NPC_PATHS = {
    1: [(208.4, 3.49), (233.4, 2.08), (258.3, 0.67), (283.3, -0.37), (308.3, -0.39),
        (357.9, -0.42), (399.4, -0.55)],
    2: [(357.9, 7.04), (370.4, 7.06), (382.9, 7.12), (387.4, 7.27)],
    3: [(432.9, 10.48), (457.4, 10.49), (482.4, 9.41), (506.9, 9.07), (531.9, 9.09),
        (555.9, 9.11)],
}
NPC_END_X = {1: 399.4, 2: 387.4, 3: 555.9}


def npc_path_y(lane: int, x: float) -> float:
    path = NPC_PATHS.get(lane)
    if path is None or x <= path[0][0]:
        return lane*LANE_W
    return float(np.interp(x, [p[0] for p in path], [p[1] for p in path]))


def npc_path_yaw(lane: int, x: float) -> float:
    return math.atan2(npc_path_y(lane, x + 0.5) - npc_path_y(lane, x - 0.5), 1.0)


GLOBAL_ROAD = [(x, global_path_y(x)) for x in np.arange(-300.0, 800.0, 0.5)]
GLOBAL_MAP = [road_to_map(x, y) for x, y in GLOBAL_ROAD]


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------
@dataclass
class Npc:
    id: int
    x: float
    y: float
    v: float
    length: float
    width: float
    v_des: float = 0.0
    lane: int = 0

    @property
    def yaw(self) -> float:
        return npc_path_yaw(self.lane, self.x)


@dataclass
class LaneTraffic:
    lane_y: float
    speed_kmh: float
    spacing: Tuple[float, float] = (0.0, 0.0)
    first_x: float = -250.0
    speed_jitter_kmh: float = 0.0
    truck_ratio: float = 0.0
    # Spawn-period traffic: each car's desired speed and the gap to the next
    # one (= speed x spawn period) vary by +-speed_spread / +-period_spread.
    period_s: Optional[float] = None
    speed_spread: float = 0.10
    period_spread: float = 0.15


@dataclass
class Scenario:
    name: str
    description: str
    traffic: List[LaneTraffic] = field(default_factory=list)
    camera_dropout: float = 0.05
    camera_noise: float = 0.03
    camera_format: str = "real_lane"
    ego_start_x: float = -80.0
    ego_start_v: float = 15.0
    extra_npcs: List[Tuple[float, float, float]] = field(default_factory=list)  # x, y, kmh
    # NPCs follow the vehicle ahead (ego included) with the IDM using MORAI's
    # SafetyDist as standstill gap and TTC as time gap; False: never brake.
    reactive: bool = True
    npc_safety_dist_m: float = 5.0
    npc_ttc_s: float = 1.0


# Highway traffic modelled on data/scenarios/2026_molit_comp_sample_scene.json:
# spawn points 6, 8 and 10 emit one car every 10 s into L1, L2 and L3 at 40, 50
# and 60 km/h, without lane changes (gaps ~111, 139 and 167 m); nothing drives
# in the entry lane or in L4; NPCs keep SafetyDist 5 m / TTC 1 s.  Every run
# draws each car's speed (+-10 %), its gap (period +-15 %) and the phase of
# each lane's stream, so no two seeds meet the same traffic.
MORAI_TRAFFIC = [
    LaneTraffic(3.5, 40.0, period_s=10.0),     # spawn point 6
    LaneTraffic(7.0, 50.0, period_s=10.0),     # spawn point 8
    LaneTraffic(10.5, 60.0, period_s=10.0),    # spawn point 10
]

SCENARIOS = {
    "morai": Scenario("morai", "sample-scenario highway traffic, randomised per seed", MORAI_TRAFFIC),
    "morai_camera_noise": Scenario(
        "morai_camera_noise", "same traffic, 30% camera dropouts, noisy fit, v2 publisher",
        MORAI_TRAFFIC, camera_dropout=0.30, camera_noise=0.06, camera_format="v2"),
    "empty": Scenario("empty", "no traffic: three changes as early as possible"),
}

# Invented stress cases, not in the sample scenario; NPCs never brake.
STRESS = {
    "light": Scenario("light", "light traffic, every lane slower than the ego", [
        LaneTraffic(3.5, 60, (60, 90)), LaneTraffic(7.0, 70, (70, 100)),
        LaneTraffic(10.5, 80, (80, 120)), LaneTraffic(14.0, 80, (60, 90)),
    ]),
    "dense_entry": Scenario("dense_entry", "dense L1 next to the short entry lane", [
        LaneTraffic(3.5, 55, (28, 40)), LaneTraffic(7.0, 65, (45, 70)),
        LaneTraffic(10.5, 75, (60, 90)),
    ]),
    "slow_leads": Scenario("slow_leads", "slow vehicles in every lane: overtake, then follow", [
        LaneTraffic(3.5, 40, (50, 70)), LaneTraffic(7.0, 50, (50, 70)),
        LaneTraffic(10.5, 60, (45, 60)),
    ]),
    "fast_rear": Scenario("fast_rear", "target lanes faster than the ego: only rear gaps", [
        LaneTraffic(3.5, 90, (90, 140)), LaneTraffic(7.0, 95, (90, 140)),
        LaneTraffic(10.5, 100, (90, 140)),
    ]),
}
for _stress in STRESS.values():
    _stress.reactive = False
    for _lane in _stress.traffic:
        _lane.speed_jitter_kmh, _lane.truck_ratio = 3.0, 0.15
ALL_SCENARIOS = {**SCENARIOS, **STRESS}


def make_traffic(scenario: Scenario, rng: random.Random) -> List[Npc]:
    npcs, next_id = [], 100
    for lane in scenario.traffic:
        if lane.period_s is not None:
            # Spawn stream: random phase, then one car per (randomised) period.
            base_v = lane.speed_kmh/3.6
            x = lane.first_x + rng.uniform(0.0, base_v*lane.period_s)
            while x < 900.0:
                v = base_v*rng.uniform(1.0 - lane.speed_spread, 1.0 + lane.speed_spread)
                npcs.append(Npc(next_id, x, lane.lane_y, v, 4.6, 1.9, v, lane_number(lane.lane_y)))
                next_id += 1
                x += base_v*lane.period_s*rng.uniform(1.0 - lane.period_spread,
                                                       1.0 + lane.period_spread)
            continue
        x = lane.first_x + rng.uniform(0.0, lane.spacing[1])
        while x < 900.0:
            truck = rng.random() < lane.truck_ratio
            v = (lane.speed_kmh + rng.uniform(-lane.speed_jitter_kmh, lane.speed_jitter_kmh))/3.6
            npcs.append(Npc(next_id, x, lane.lane_y, v, 12.0 if truck else 4.6,
                            2.5 if truck else 1.9, v, lane_number(lane.lane_y)))
            next_id += 1
            x += rng.uniform(*lane.spacing) + (8.0 if truck else 0.0)
    for x, y, kmh in scenario.extra_npcs:
        npcs.append(Npc(next_id, x, y, kmh/3.6, 4.6, 1.9, kmh/3.6, lane_number(y)))
        next_id += 1
    for npc in npcs:
        npc.y = npc_path_y(npc.lane, npc.x)
    npcs[:] = [n for n in npcs if n.x < NPC_END_X.get(n.lane, float("inf"))]
    return npcs


def lane_number(y: float) -> int:
    return int(round(y/LANE_W))


def update_traffic(npcs: List[Npc], scenario: Scenario, ego_box_center: Tuple[float, float],
                   ego_v: float, dt: float) -> None:
    """Advance NPCs; reactive ones follow the vehicle ahead with the IDM.

    The IDM keeps a gap of about SafetyDist + v * TTC (the scenario's NPC
    settings).  The ego counts as the vehicle ahead once its footprint
    overlaps the NPC's lane.
    """
    if not scenario.reactive:
        for npc in npcs:
            npc.x += npc.v*dt
            npc.y = npc_path_y(npc.lane, npc.x)
        npcs[:] = [n for n in npcs if n.x < NPC_END_X.get(n.lane, float("inf"))]
        return
    cx, cy = ego_box_center
    lanes: Dict[int, List[Npc]] = {}
    for npc in npcs:
        lanes.setdefault(npc.lane, []).append(npc)
    a_max, b_comf = 1.5, 3.0
    for lane, members in lanes.items():
        members.sort(key=lambda n: n.x)
        ego_in_lane = abs(cy - npc_path_y(lane, cx)) < 0.5*(EGO_W + 1.9)
        for i, npc in enumerate(members):
            lead_gap, lead_v = None, None
            if i + 1 < len(members):
                ahead = members[i+1]
                lead_gap = (ahead.x - 0.5*ahead.length) - (npc.x + 0.5*npc.length)
                lead_v = ahead.v
            if ego_in_lane and cx > npc.x:
                gap = (cx - 0.5*EGO_L) - (npc.x + 0.5*npc.length)
                if lead_gap is None or gap < lead_gap:
                    lead_gap, lead_v = gap, ego_v
            accel = a_max*(1.0 - (npc.v/max(npc.v_des, 0.1))**4)
            if lead_gap is not None:
                desired = (scenario.npc_safety_dist_m + npc.v*scenario.npc_ttc_s
                           + npc.v*(npc.v - lead_v)/(2.0*math.sqrt(a_max*b_comf)))
                accel -= a_max*(max(desired, 0.0)/max(lead_gap, 0.1))**2
            npc.v = max(0.0, npc.v + max(-8.0, accel)*dt)
            npc.x += npc.v*dt
            npc.y = npc_path_y(npc.lane, npc.x)
    npcs[:] = [n for n in npcs if n.x < NPC_END_X.get(n.lane, float("inf"))]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
class CameraSim:
    """lane_info JSON like real_lane_node / live_lane_info_publisher_v2."""

    def __init__(self, rng: random.Random, scenario: Scenario, rate_hz: float = 12.0,
                 latency_s: float = 0.08) -> None:
        self.rng = rng
        self.scenario = scenario
        self.period = 1.0/rate_hz
        self.latency = latency_s
        self.next_capture = 0.0
        self.ages: Dict[str, int] = {}
        self.queue: List[Tuple[float, Dict]] = []

    def update(self, t: float, ego_road: Tuple[float, float, float]) -> List[Tuple[float, Dict]]:
        if t >= self.next_capture:
            self.next_capture = t + self.period
            self.queue.append((t + self.latency, self._frame(t, ego_road)))
        ready = [item for item in self.queue if item[0] <= t]
        self.queue = [item for item in self.queue if item[0] > t]
        return ready

    def _frame(self, t: float, ego) -> Dict:
        ex, ey, eyaw = ego
        stamp_capture, stamp_publish = t, t + self.latency
        stamp = stamp_capture if self.scenario.camera_format == "v2" else stamp_publish
        base = {"timestamp": stamp, "frame_id": "base_link"}
        if self.scenario.camera_format == "v2":
            base["observation_time_source"] = "camera_receive_wall"
        if self.rng.random() < self.scenario.camera_dropout:
            self.ages.clear()
            base.update(output_status="INVALID", lane_valid=False,
                        left_lane={"detected": False}, right_lane={"detected": False},
                        left_boundary_points=[], right_boundary_points=[])
            return base
        c, s = math.cos(eyaw), math.sin(eyaw)
        fits = []
        for line in LINES:
            xs_road = np.arange(ex-5.0, ex+60.0, 1.0)
            ys_road = line.y_at(xs_road)
            dx, dy = xs_road-ex, ys_road-ey
            xe, ye = c*dx + s*dy, -s*dx + c*dy
            keep = (xe >= CAMERA_X_RANGE[0]) & (xe <= CAMERA_X_RANGE[1]) & np.isfinite(ye)
            if keep.sum() < 8:
                continue
            coef = np.polyfit(xe[keep], ye[keep], 2)
            coef[2] += self.rng.gauss(0.0, self.scenario.camera_noise)
            coef[1] += self.rng.gauss(0.0, self.scenario.camera_noise/15.0)
            y7 = float(np.polyval(coef, 7.0))
            if abs(y7) > 9.0:
                continue
            kind = line.kind_at(ex + 10.0) or "solid"
            fits.append((y7, float(np.polyval(coef, 1.0)), line.name, coef, kind,
                         float(xe[keep].max())))
        seen = {f[2] for f in fits}
        for name in list(self.ages):
            if name not in seen:
                del self.ages[name]
        for f in fits:
            self.ages[f[2]] = self.ages.get(f[2], 0) + 1
        fits.sort(key=lambda f: f[0])

        def meta(f):
            if f is None:
                return {"detected": False, "type": None, "dashed": None, "coef": None, "age": 0}
            return {"detected": True, "type": LANE_TYPE[f[4]], "dashed": f[4] == "dashed",
                    "coef": [round(float(v), 8) for v in f[3]],
                    "x_range_m": [CAMERA_X_RANGE[0], round(f[5], 2)],
                    "age": self.ages[f[2]], "confidence": 0.9,
                    "from_guide": False, "coasted": False}

        straddle = next((f for f in fits if abs(f[1]) < 0.45), None)
        if straddle is not None:
            # real_lane: the straddled line is lane 0, left/right are beyond it.
            left = next((f for f in fits if f[0] > straddle[0]), None)
            right = next((f for f in reversed(fits) if f[0] < straddle[0]), None)
        else:
            left = next((f for f in fits if f[0] > 0.0), None)
            right = next((f for f in reversed(fits) if f[0] < 0.0), None)

        def points(f):
            if f is None:
                return []
            return [[float(x), round(float(np.polyval(f[3], x)), 3)] for x in np.arange(3.0, f[5], 0.5)]

        base.update(
            output_status="FRESH" if (left or right) else "INVALID",
            lane_valid=bool(left or right),
            left_lane=meta(left), right_lane=meta(right),
            straddling_lane=meta(straddle) if straddle is not None else None,
            lane_width_m=(round(left[0]-right[0], 3) if (left and right and straddle is None) else None),
            left_boundary_points=points(left), right_boundary_points=points(right),
        )
        return base


class LidarSim:
    def __init__(self, rng: random.Random, rate_hz: float = 10.0) -> None:
        self.rng = rng
        self.period = 1.0/rate_hz
        self.next_t = 0.0
        self.age: Dict[int, int] = {}

    def update(self, t: float, ego_road, npcs: List[Npc]) -> Optional[List[Obstacle]]:
        if t < self.next_t:
            return None
        self.next_t = t + self.period
        ex, ey, _ = ego_road
        out, seen = [], set()
        for npc in npcs:
            dx = npc.x - ex
            if not -60.0 <= dx <= 80.0 or abs(npc.y - ey) > 18.0:
                continue
            seen.add(npc.id)
            self.age[npc.id] = self.age.get(npc.id, 0) + 1
            # A new Kalman track reports no velocity for its first frames.
            v = 0.0 if self.age[npc.id] <= 2 else npc.v + self.rng.gauss(0.0, 0.2)
            mx, my = road_to_map(npc.x + self.rng.gauss(0, 0.1), npc.y + self.rng.gauss(0, 0.1))
            heading = MAP_THETA + npc.yaw
            out.append(Obstacle(npc.id, mx, my, math.cos(heading)*v, math.sin(heading)*v,
                                npc.length, npc.width))
        for oid in list(self.age):
            if oid not in seen:
                del self.age[oid]
        return out


class ControllerSim:
    """adaptive_curvature_purepursuit on the received path, with its functions.

    Steering: adaptive-lookahead Pure Pursuit blended with the path curvature
    (feed-forward), rate-limited.  Speed: the path's curvature profile capped
    by the controller's max_speed_kph, the minimum with the trajectory speed,
    slewed by max_accel / max_decel; a stop brakes at once.
    """

    def __init__(self, preset: Dict, cap_mps: float) -> None:
        self.p = preset
        self.cap = cap_mps
        self.profile = None
        self.command: Optional[float] = None
        self.steering = 0.0

    def set_path(self, points: List[Tuple[float, float]]) -> None:
        pts = cp.clean_consecutive_duplicates([cp.PathPoint(x, y, 0.0) for x, y in points])
        if len(pts) < 2:
            self.profile = None
            return
        p = self.p
        s = cp.cumulative_arc_lengths(pts)
        k = cp.curvature_profile(pts, half_window_points=1, smoothing_window=5)
        v = cp.build_speed_profile(s, k, self.cap, p["lateral"], p["accel"], p["decel"],
                                   None, self.cap)
        self.profile = (pts, s, k, v)

    def control(self, x: float, y: float, yaw: float, ego_v: float, target: float,
                stop: bool, dt: float) -> Tuple[float, float, float]:
        """(steering, speed command, path speed limit) for one control cycle."""
        if self.command is None:
            self.command = ego_v
        if stop or self.profile is None:
            self.command = self.steering = 0.0
            return 0.0, 0.0, 0.0
        p = self.p
        pts, s, k, v = self.profile
        progress = cp.nearest_projection(pts, s, x, y).progress_s
        preview = cp.max_abs_curvature_ahead(
            s, k, progress, max(8.0, p["look_min"] + p["look_gain"]*ego_v), 0.5)
        look = cp.adaptive_lookahead_m(ego_v, p["look_min"], p["look_gain"], preview,
                                       p["look_curv"], p["look_tight"], p["look_max"])
        target_point, _ = cp.interpolate_by_s(pts, s, min(s[-1], progress + look))
        dx, dy = target_point.x - x, target_point.y - y
        c, sn = math.cos(yaw), math.sin(yaw)
        bx, by = c*dx + sn*dy, -sn*dx + c*dy
        curvature = 2.0*math.sin(math.atan2(by, bx))/max(math.hypot(bx, by), 1e-3)
        ff = cp.profile_value_at_s(s, k, min(s[-1], progress + 0.75*look))
        steer = ((1.0 - p["ff"])*math.atan(WHEELBASE*curvature)
                 + p["ff"]*math.atan(WHEELBASE*ff))
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        rate = p["steer_rate"]*dt
        self.steering = max(self.steering - rate, min(self.steering + rate, steer))
        limit = cp.profile_value_at_s(s, v, progress)
        delta = min(limit, target) - self.command
        step = (p["accel"] if delta >= 0.0 else p["decel"])*dt
        self.command = max(0.0, self.command + max(-step, min(step, delta)))
        return self.steering, self.command, limit


# ---------------------------------------------------------------------------
# Geometry helpers for scoring
# ---------------------------------------------------------------------------
def box_corners(cx, cy, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for lx, ly in ((0.5*length, 0.5*width), (0.5*length, -0.5*width),
                   (-0.5*length, -0.5*width), (-0.5*length, 0.5*width)):
        out.append((cx + c*lx - s*ly, cy + s*lx + c*ly))
    return out


def boxes_overlap(a, b) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 4]
            nx, ny = y1-y2, x2-x1
            pa = [nx*x + ny*y for x, y in a]
            pb = [nx*x + ny*y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
def default_config(route_points) -> OvertakeConfig:
    route = Route(route_points)
    cfg = OvertakeConfig(gate_required=True)
    cfg.zone_start_s = route.s_of_xy(*road_to_map(ZONE_START_X, global_path_y(ZONE_START_X)))
    cfg.zone_end_s = route.s_of_xy(*road_to_map(ZONE_END_X, global_path_y(ZONE_END_X)))
    cfg.entry_taper_start_s = route.s_of_xy(*road_to_map(ENTRY_TAPER[0], 0.0))
    cfg.entry_taper_end_s = route.s_of_xy(*road_to_map(ENTRY_TAPER[1], 0.5*LANE_W))
    return cfg


def run(scenario: Scenario, seed: int, cfg_overrides: Optional[Dict] = None,
        record: bool = True, controller: str = "mission", align: bool = True) -> Dict:
    rng = random.Random(seed)
    cfg = default_config(GLOBAL_MAP)
    for key, value in (cfg_overrides or {}).items():
        setattr(cfg, key, value)
    preset = CONTROLLERS[controller]
    aligned = (align_with_controller(cfg, preset["lateral"], preset["accel"], preset["decel"])
               if align else {})
    planner = HighwayOvertakePlanner(cfg, GLOBAL_MAP)
    # The launch caps the controller at max(max_speed_kph, highway_max_speed_kph).
    ctrl = ControllerSim(preset, max(cfg.max_speed_mps, cfg.cruise_speed_mps))
    npcs = make_traffic(scenario, rng)
    camera, lidar = CameraSim(rng, scenario), LidarSim(rng)
    # route_mission_gate: progress on the global path, zone and request flags.
    gate_route = Route(GLOBAL_MAP)
    gate_window = (cfg.zone_start_s - GATE_OPEN_BEFORE_ZONE_M, cfg.zone_end_s)
    next_gate = 0.0
    max_offset, min_limit = 0.0, None

    # Ego (road frame, rear axle); the controller above drives it.
    ex, ey, eyaw, ev, delta = scenario.ego_start_x, 0.0, 0.0, scenario.ego_start_v, 0.0
    out = None
    steer_cmd, stop_cmd, v_cmd = 0.0, False, ev
    dt, t = 0.02, 0.0
    next_plan = next_pp = next_base = 0.0
    log: List[Dict] = []
    frames: List[Dict] = []
    events: List[Tuple[float, float, str]] = []
    collisions, solid_touch, island_s = [], 0.0, 0.0
    touched: Dict[str, float] = {}
    max_alat, max_head, max_alat_after = 0.0, 0.0, 0.0
    t_zone_start = t_zone_end = None
    v_zone_end = None
    commits = []
    lane_entered_l4 = False

    while t < 90.0 and ex < SCORE_END_X + 40.0:
        # ---- sensors ----------------------------------------------------
        mx, my = road_to_map(ex, ey)
        myaw = eyaw + MAP_THETA
        planner.on_odom(t, mx, my, myaw, ev)
        for stamp, info in camera.update(t, (ex, ey, eyaw)):
            planner.on_lane_info(stamp, json.loads(json.dumps(info)))
        obstacles = lidar.update(t, (ex, ey, eyaw), npcs)
        if obstacles is not None:
            planner.on_obstacles(t, obstacles)
        if t >= next_base:
            next_base = t + 0.1
            i = min(range(len(GLOBAL_ROAD)), key=lambda k: abs(GLOBAL_ROAD[k][0]-ex))
            planner.on_base_path(t, GLOBAL_MAP[max(0, i-6):i+160], False)
        if t >= next_gate:
            next_gate = t + 0.05
            route_s, offset = gate_route.project(mx, my)
            in_gate = gate_window[0] <= route_s < gate_window[1]
            planner.on_route_gate(t, in_gate)
            planner.on_lane_change_request(t, in_gate)
            if in_gate:
                max_offset = max(max_offset, abs(offset))

        # ---- planner (20 Hz) --------------------------------------------
        if t >= next_plan:
            next_plan = t + 0.05
            out = planner.step(t)
            for text in out.events:
                events.append((round(t, 2), round(ex, 1), text))
                if "committed" in text:
                    commits.append({"t": round(t, 2), "x": round(ex, 1), "lane_from": int(round(ey/LANE_W)),
                                    "gap": (out.status.get("gap") or {}), "v": round(ev, 2)})
            if out.path:
                ctrl.set_path(out.path)
            if record and (not frames or t - frames[-1]["t"] >= 0.099):
                follow = out.status.get("follow") or {}
                frames.append({
                    "t": t, "x": ex, "y": ey, "yaw": eyaw, "v": ev, "v_cmd": v_cmd,
                    "state": out.state, "reason": out.status.get("reason"),
                    "gap": out.status.get("gap"), "lead": follow.get("lead"),
                    "changes": out.status.get("lane_changes_done", 0), "stop": out.stop,
                    "deadline": out.status.get("deadline"),
                    "path": [map_to_road(px, py) for px, py in (out.path or [])[::2]],
                    "npcs": [(n.id, round(n.x, 2), round(n.y, 3), round(n.v, 2), n.length, n.width,
                              round(n.yaw, 4)) for n in npcs if abs(n.x - ex) < 170.0],
                })

        # ---- adaptive_curvature_purepursuit (20 Hz) ----------------------
        if t >= next_pp and out is not None:
            next_pp = t + 0.05
            stop_cmd = bool(out.stop or out.path is None)
            steer_cmd, v_cmd, path_limit = ctrl.control(mx, my, myaw, ev, out.target_speed,
                                                        stop_cmd, 0.05)
            if ZONE_START_X <= ex <= ZONE_END_X and not stop_cmd:
                min_limit = path_limit if min_limit is None else min(min_limit, path_limit)

        # ---- vehicle ----------------------------------------------------
        delta += (steer_cmd - delta)*min(1.0, dt/0.12)
        accel = -7.0 if stop_cmd else max(-5.0, min(2.0, 1.5*(v_cmd - ev)))
        ev = max(0.0, ev + accel*dt)
        ex += ev*math.cos(eyaw)*dt
        ey += ev*math.sin(eyaw)*dt
        eyaw += ev*math.tan(delta)/WHEELBASE*dt
        update_traffic(npcs, scenario, (ex + EGO_CENTER*math.cos(eyaw),
                                        ey + EGO_CENTER*math.sin(eyaw)), ev, dt)
        t += dt

        # ---- scoring ----------------------------------------------------
        a_lat = ev*ev*math.tan(delta)/WHEELBASE
        if ZONE_START_X - 20 <= ex <= ZONE_END_X:
            max_alat = max(max_alat, abs(a_lat))
            max_head = max(max_head, abs(math.degrees(eyaw)))
        elif ZONE_END_X < ex <= SCORE_END_X:
            # The route itself bends into the right toll lane here.
            max_alat_after = max(max_alat_after, abs(a_lat))
        if t_zone_start is None and ex >= ZONE_START_X:
            t_zone_start = t
        if t_zone_end is None and ex >= ZONE_END_X:
            t_zone_end, v_zone_end = t, ev
        cx = ex + EGO_CENTER*math.cos(eyaw)
        cy = ey + EGO_CENTER*math.sin(eyaw)
        # Scored through the funnel and into the toll lanes.
        scored = ex <= SCORE_END_X
        ego_box = box_corners(cx, cy, eyaw, EGO_L, EGO_W)
        for npc in npcs:
            if scored and abs(npc.x - cx) < 15 and abs(npc.y - cy) < 4:
                if boxes_overlap(ego_box, box_corners(npc.x, npc.y, npc.yaw, npc.length, npc.width)):
                    if not collisions or collisions[-1]["id"] != npc.id:
                        collisions.append({"t": round(t, 2), "x": round(ex, 1), "id": npc.id})
        if scored and cy > 12.25 - 0.3:
            lane_entered_l4 = True
        edge = island_edge_y(cx)
        if scored and math.isfinite(edge) and LANE_W/2 < cy < edge:
            island_s += dt
        wheels = []
        for ax in (0.0, WHEELBASE):
            for side in (TRACK_HALF, -TRACK_HALF):
                wheels.append((ex + ax*math.cos(eyaw) - side*math.sin(eyaw),
                               ey + ax*math.sin(eyaw) + side*math.cos(eyaw)))
        touching = False
        for line in LINES:
            for wx, wy in wheels:
                if line.kind_at(wx) != "solid" or not ZONE_START_X - 40 <= ex <= SCORE_END_X:
                    continue
                yl = float(line.y_at(wx))
                if math.isfinite(yl) and abs(wy - yl) < 0.075 + 0.12:
                    touching = True
                    touched[line.name] = touched.get(line.name, 0.0) + dt/4
        if touching:
            solid_touch += dt

        if record and len(log) < 100000:
            log.append({"t": t, "x": ex, "y": ey, "yaw": eyaw, "v": ev, "v_cmd": v_cmd,
                        "state": out.state if out else "-", "a_lat": a_lat,
                        "stop": stop_cmd,
                        "reason": out.status.get("reason") if out else None,
                        "gap": out.status.get("gap") if out else None,
                        "follow": out.status.get("follow") if out else None,
                        "caps": out.status.get("caps") if out else None})

    handover = next(((e[0], e[1], e[2]) for e in events if "hand-over" in e[2]), None)
    result = {
        "scenario": scenario.name, "seed": seed,
        "collisions": collisions,
        "solid_touch_s": round(solid_touch, 2),
        "solid_lines_touched": {k: round(v, 2) for k, v in touched.items()},
        "island_s": round(island_s, 2),
        "entered_forbidden_lane": lane_entered_l4,
        "lane_changes": len(commits),
        "commits": commits,
        "zone_time_s": None if (t_zone_start is None or t_zone_end is None)
        else round(t_zone_end - t_zone_start, 2),
        "speed_at_zone_end_kmh": None if v_zone_end is None else round(v_zone_end*3.6, 1),
        "max_lateral_accel_mps2": round(max_alat, 2),
        "max_lateral_accel_after_mps2": round(max_alat_after, 2),
        "max_heading_deg": round(max_head, 2),
        "final_lane_y": round(ey, 2),
        "handover": handover,
        "max_route_offset_m": round(max_offset, 2),
        "controller": controller,
        "aligned": {k: [round(a, 2), round(b, 2)] for k, (a, b) in aligned.items()},
        "min_controller_limit_kmh": None if min_limit is None else round(min_limit*3.6, 1),
        "events": events,
    }
    if record:
        result["_log"] = log
        result["_frames"] = frames
    return result


def verdict(r: Dict) -> List[str]:
    problems = []
    if r["collisions"]:
        problems.append("collision %s" % r["collisions"][0])
    if r["solid_touch_s"] > 0.0:
        problems.append("solid line touched %.2fs %s" % (r["solid_touch_s"], r["solid_lines_touched"]))
    if r.get("island_s", 0.0) > 0.0:
        problems.append("drove onto the hatched island %.2fs (lane ended)" % r["island_s"])
    if r["entered_forbidden_lane"]:
        problems.append("entered forbidden lane")
    if r["handover"] is None:
        problems.append("never reached the zone end")
    elif "WARNING" in r["handover"][2]:
        problems.append("not in the final lane at the zone end (%s)" % r["handover"][2].split("(")[1].split(")")[0])
    if r["speed_at_zone_end_kmh"] is not None and r["speed_at_zone_end_kmh"] > 60.0:
        problems.append("zone end speed %.1f km/h" % r["speed_at_zone_end_kmh"])
    if r["max_lateral_accel_mps2"] > 2.5:
        problems.append("lateral accel %.2f" % r["max_lateral_accel_mps2"])
    # Past the zone end the route's own bend into the toll lane takes up to
    # the controller's limit (3.8 with the mission preset); more is a swerve.
    if r.get("max_lateral_accel_after_mps2", 0.0) > 4.0:
        problems.append("lateral accel %.2f after the zone end" % r["max_lateral_accel_after_mps2"])
    if r["max_route_offset_m"] >= MAX_ROUTE_OFFSET_M:
        problems.append("route offset %.2f m: route_mission_gate stops the car" % r["max_route_offset_m"])
    return problems


def draw_island(ax) -> None:
    """Hatch the island between the right edge and its left edge."""
    xs = [p[0] for p in ISLAND_EDGE]
    ax.fill_between(xs, [LANE_W/2]*len(xs), [p[1] for p in ISLAND_EDGE], facecolor="none",
                    edgecolor="0.6", hatch="///", lw=0.0, zorder=1)


def plot(result: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = result["_log"]
    xs = np.array([p["x"] for p in log])
    ys = np.array([p["y"] for p in log])
    ts = np.array([p["t"] for p in log])
    states = [p["state"] for p in log]
    colors = {"OFF": "0.5", "WAIT_LANE": "purple", "CHASE": "tab:blue", "CHANGE": "tab:orange",
              "REACQUIRE": "tab:pink", "FOLLOW": "tab:green", "DONE": "black", "-": "0.5"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for line in LINES:
        for x0, x1, kind in line.kinds:
            xx = np.linspace(max(x0, -120), min(x1, 560), 300)
            yy = line.y_at(xx)
            ax1.plot(xx, yy, color="black" if kind == "solid" else "0.5",
                     ls="-" if kind == "solid" else (0, (5, 5)), lw=1.4 if kind == "solid" else 1.0)
    draw_island(ax1)
    gx = np.array([p[0] for p in GLOBAL_ROAD])
    gy = np.array([p[1] for p in GLOBAL_ROAD])
    m = (gx > -120) & (gx < 560)
    ax1.plot(gx[m], gy[m], color="red", lw=1.0, alpha=0.6, label="global path")
    # NPC positions near the ego, sampled from the recorded frames.
    for frame in result["_frames"][::5]:
        near = [(n[1], n[2]) for n in frame["npcs"] if abs(n[1] - frame["x"]) < 60]
        if near:
            ax1.plot([n[0] for n in near], [n[1] for n in near], ".", ms=2, color="0.6")
    for state in colors:
        sel = np.array([s == state for s in states])
        if sel.any():
            ax1.scatter(xs[sel][::3], ys[sel][::3], s=4, color=colors[state], label=state)
    for c in result["commits"]:
        ax1.axvline(c["x"], color="tab:orange", lw=0.8, ls=":")
    ax1.axvline(ZONE_START_X, color="blue", lw=1.2, ls="--")
    ax1.axvline(ZONE_END_X, color="blue", lw=1.2, ls="--")
    ax1.set_ylim(-3, 17)
    ax1.set_ylabel("y [m] (left +)")
    ax1.legend(loc="upper left", ncol=9, fontsize=8)
    ax1.set_title("%s seed=%d | changes=%d zone_time=%ss end_speed=%skm/h | %s" % (
        result["scenario"], result["seed"], result["lane_changes"], result["zone_time_s"],
        result["speed_at_zone_end_kmh"], "; ".join(verdict(result)) or "OK"))
    ax2.plot(xs, [p["v"]*3.6 for p in log], label="ego speed")
    ax2.plot(xs, [p["v_cmd"]*3.6 for p in log], label="target speed", alpha=0.7)
    ax2.axhline(60, color="red", lw=0.8, ls=":")
    ax2.set_ylabel("km/h")
    ax2.set_xlabel("x along road [m] (zone %g..%g)" % (ZONE_START_X, ZONE_END_X))
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


STATE_COLORS = {"OFF": "0.55", "WAIT_LANE": "purple", "CHASE": "tab:blue",
                "CHANGE": "tab:orange", "REACQUIRE": "deeppink", "FOLLOW": "tab:green",
                "DONE": "black", "-": "0.55"}


def view(result: Dict, gif_path: str = "", fps: int = 10, window: Tuple[float, float] = (-40.0, 110.0)) -> None:
    """Replay a run: top view around the ego, planner path, state and reasons.

    Interactive keys: space play/pause, left/right -/+1 s, up/down speed x2/x0.5.
    With gif_path the replay is written to an animated GIF instead.
    """
    import matplotlib
    if gif_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    frames = result["_frames"]
    fig = plt.figure(figsize=(15, 7.5) if not gif_path else (12, 6))
    ax = fig.add_axes([0.05, 0.40, 0.93, 0.50])
    axv = fig.add_axes([0.05, 0.08, 0.93, 0.22])

    for line in LINES:
        for xa, xb, kind in line.kinds:
            xx = np.linspace(max(xa, -150), min(xb, 600), 400)
            ax.plot(xx, line.y_at(xx), color="black" if kind == "solid" else "0.55",
                    ls="-" if kind == "solid" else (0, (6, 6)), lw=1.6 if kind == "solid" else 1.2)
    draw_island(ax)
    gx = np.array([p[0] for p in GLOBAL_ROAD])
    ax.plot(gx, [p[1] for p in GLOBAL_ROAD], color="red", lw=1.0, alpha=0.35, label="global path")
    for xz, name in ((ZONE_START_X, "zone start"), (ZONE_END_X, "zone end")):
        ax.axvline(xz, color="blue", ls="--", lw=1.0)
        ax.text(xz + 1, 16.2, name, color="blue", fontsize=8)
    for y, name in ((0.0, "entry"), (3.5, "L1"), (7.0, "L2"), (10.5, "L3"), (14.0, "L4 (forbidden)")):
        ax.text(0.003, (y + 3.0)/20.0, name, transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_ylim(-3.0, 17.0)
    ax.set_ylabel("y [m] (left +) - lateral scale exaggerated")
    path_line, = ax.plot([], [], color="magenta", lw=1.5, marker="o", ms=2.5, label="planner path")
    ego_patch = Polygon(np.zeros((4, 2)), closed=True, color="tab:blue", zorder=5)
    ax.add_patch(ego_patch)
    npc_patches = {}
    for frame in frames:
        for npc in frame["npcs"]:
            if npc[0] not in npc_patches:
                patch = Polygon(np.zeros((4, 2)), closed=True, color="0.7", zorder=4)
                ax.add_patch(patch)
                npc_patches[npc[0]] = patch
    title = ax.set_title("", loc="left", fontsize=10, family="monospace")
    ax.legend(loc="upper right", fontsize=8)

    ts = [f["t"] for f in frames]
    axv.plot(ts, [f["v"]*3.6 for f in frames], label="ego speed")
    axv.plot(ts, [f["v_cmd"]*3.6 for f in frames], label="target speed", alpha=0.7)
    axv.axhline(60, color="red", lw=0.8, ls=":")
    for i, f in enumerate(frames[1:], 1):
        if f["state"] != frames[i-1]["state"]:
            axv.axvline(f["t"], color=STATE_COLORS.get(f["state"], "0.5"), lw=0.8)
            axv.text(f["t"], 5, f["state"], rotation=90, fontsize=7,
                     color=STATE_COLORS.get(f["state"], "0.5"))
    cursor = axv.axvline(ts[0], color="black", lw=1.2)
    axv.set_xlabel("time [s]")
    axv.set_ylabel("km/h")
    axv.legend(loc="upper left", fontsize=8)
    axv.grid(alpha=0.3)

    def draw(i: int):
        f = frames[i]
        ego_patch.set_xy(box_corners(f["x"] + EGO_CENTER*math.cos(f["yaw"]),
                                     f["y"] + EGO_CENTER*math.sin(f["yaw"]),
                                     f["yaw"], EGO_L, EGO_W))
        ego_patch.set_color(STATE_COLORS.get(f["state"], "0.5"))
        gap = f["gap"] or {}
        blocker = None if gap.get("ok", True) else gap.get("blocker")
        present = {npc[0]: npc for npc in f["npcs"]}
        for oid, patch in npc_patches.items():
            npc = present.get(oid)
            if npc is None:
                patch.set_visible(False)
                continue
            _, nx, ny, nv, nl, nw, nyaw = npc
            patch.set_visible(True)
            patch.set_xy(box_corners(nx, ny, nyaw, nl, nw))
            patch.set_color("red" if oid == blocker else "darkorange" if oid == f["lead"]
                            else "0.45" if ny > 12.5 else "0.72")
        if f["path"]:
            path_line.set_data([p[0] for p in f["path"]], [p[1] for p in f["path"]])
        else:
            path_line.set_data([], [])
        ax.set_xlim(f["x"] + window[0], f["x"] + window[1])
        cursor.set_xdata([f["t"], f["t"]])
        gap_txt = "-" if not gap else ("clear" if gap.get("ok") else
                                       "%s id=%s gap %.1f<%.1fm at +%.1fs" % (
                                           gap.get("reason"), gap.get("blocker"), gap.get("gap", 0),
                                           gap.get("required", 0), gap.get("t", 0)))
        title.set_text("%s seed %d | t=%5.1fs x=%6.1fm | %-9s changes=%d | v=%3.0f -> %3.0f km/h%s\n"
                       "reason: %s\ngap   : %s  (red = blocking car, orange = lead)" % (
                           result["scenario"], result["seed"], f["t"], f["x"], f["state"],
                           f["changes"], f["v"]*3.6, f["v_cmd"]*3.6, "  STOP" if f["stop"] else "",
                           f["reason"], gap_txt))
        return [ego_patch, path_line, cursor, title] + list(npc_patches.values())

    if gif_path:
        anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000/fps, blit=False)
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=75)
        plt.close(fig)
        return

    control = {"i": 0, "playing": True, "speed": 1}

    def on_key(event):
        if event.key == " ":
            control["playing"] = not control["playing"]
        elif event.key == "right":
            control["i"] = min(len(frames)-1, control["i"] + 10)
        elif event.key == "left":
            control["i"] = max(0, control["i"] - 10)
        elif event.key == "up":
            control["speed"] = min(8, control["speed"]*2)
        elif event.key == "down":
            control["speed"] = max(1, control["speed"]//2)
        draw(control["i"])
        fig.canvas.draw_idle()

    def tick(_):
        if control["playing"]:
            control["i"] = min(len(frames)-1, control["i"] + control["speed"])
        return draw(control["i"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, tick, interval=100, blit=False, cache_frame_data=False)
    fig._anim = anim  # keep a reference while the window is open
    plt.show()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="morai",
                    help="morai (default), all (= %s), stress (= %s) or one name"
                    % ("/".join(SCENARIOS), "/".join(STRESS)))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="", help="directory for PNG plots and summary.json")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an OvertakeConfig field, e.g. --set max_speed_mps=25")
    ap.add_argument("--seed", type=int, default=None, help="run only this seed")
    ap.add_argument("--view", action="store_true",
                    help="replay the run in a window (space: pause, arrows: seek/speed)")
    ap.add_argument("--gif", default="", help="write the replay of the run to this GIF file")
    ap.add_argument("--controller", default="mission", choices=sorted(CONTROLLERS),
                    help="adaptive_curvature_purepursuit settings: mission (turn mission "
                         "preset, default) or example (highway-test.env.example)")
    ap.add_argument("--no-align", action="store_true",
                    help="do not lower the planner's limits to the controller's")
    args = ap.parse_args(argv)
    options = dict(controller=args.controller, align=not args.no_align)
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            overrides[key] = float(value)
        except ValueError:
            overrides[key] = value          # e.g. --set deadline_mode=yield
    names = (list(SCENARIOS) if args.scenario == "all" else
             list(STRESS) if args.scenario == "stress" else [args.scenario])
    unknown = [n for n in names if n not in ALL_SCENARIOS]
    if unknown:
        ap.error("unknown scenario %s" % unknown)
    if args.view or args.gif:
        if len(names) != 1:
            ap.error("--view/--gif need one --scenario")
        result = run(ALL_SCENARIOS[names[0]], args.seed or 0, overrides, record=True, **options)
        print("%s seed=%d changes=%d -> %s" % (names[0], args.seed or 0, result["lane_changes"],
                                              "; ".join(verdict(result)) or "OK"))
        for event in result["events"]:
            print("  t=%6.2f x=%6.1f  %s" % event)
        view(result, gif_path=args.gif)
        return 0
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    summary, failures = [], 0
    seeds = [args.seed] if args.seed is not None else range(args.seeds)
    for name in names:
        for seed in seeds:
            result = run(ALL_SCENARIOS[name], seed, overrides, record=bool(args.out), **options)
            if not summary:
                print("controller=%s aligned=%s" % (args.controller, result["aligned"] or "-"))
            problems = verdict(result)
            failures += bool(problems)
            print("%-13s seed=%d changes=%d zone_time=%5ss end=%5skm/h a_lat=%.2f head=%.1fdeg "
                  "offset=%.1fm limit=%skm/h -> %s" % (
                      name, seed, result["lane_changes"], result["zone_time_s"],
                      result["speed_at_zone_end_kmh"], result["max_lateral_accel_mps2"],
                      result["max_heading_deg"], result["max_route_offset_m"],
                      result["min_controller_limit_kmh"], "; ".join(problems) or "OK"))
            if args.out:
                plot(result, os.path.join(args.out, "%s_seed%d.png" % (name, seed)))
            summary.append({k: v for k, v in result.items() if not k.startswith("_")})
    if args.out:
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=1)
    print("%d/%d runs with problems" % (failures, len(summary)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
