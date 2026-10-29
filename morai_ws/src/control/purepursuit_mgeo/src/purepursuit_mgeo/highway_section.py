"""The highway planner inside final_ws_curvature_signal.launch (ROS-free core).

final_ws_curvature_signal_highway.launch keeps the team's curvature controller
for the whole route (/control/ctrl_cmd_main) and adds highway_lane_strategy
with a controller of its own (/control/ctrl_cmd_highway).
highway_command_mux_node publishes /control/ctrl_cmd, the fusion's nominal
command, from one of the two (mux_choice).  highway_base_path_node gives
highway_lane_strategy the global route around the ego (route_window): the base
path it passes through outside its zone, follows until it has the camera lane,
and hands over to.
"""
from bisect import bisect_left, bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def mux_choice(state: Optional[Dict], state_age_s: float, highway_cmd_age_s: float,
               highway_end_s: float, timeout_s: float) -> Tuple[bool, str]:
    """(drive with the highway controller, reason) for one command.

    The highway controller drives from the zone entry of highway_lane_strategy
    (any phase but OFF) until the ego has passed highway_end_s after the zone
    (phase DONE).  Otherwise, or while its state or the highway command is
    older than timeout_s, the main controller drives.
    """
    if state is None or not 0.0 <= state_age_s <= timeout_s:
        return False, "highway_state_stale"
    phase = state.get("phase")
    if phase in (None, "OFF"):
        return False, "outside_highway"
    route_s = state.get("route_s")
    if phase == "DONE" and (route_s is None or float(route_s) >= highway_end_s):
        return False, "past_highway_end"
    if not 0.0 <= highway_cmd_age_s <= timeout_s:
        return False, "highway_command_stale"
    return True, str(phase)


def merged_mux_choice(highway_state: Optional[Dict], highway_state_age_s: float,
                      avoidance_state: Optional[Dict], avoidance_state_age_s: float,
                      main_cmd_age_s: float, secondary_cmd_age_s: float,
                      highway_end_s: float, timeout_s: float,
                      highway_enabled: bool = True,
                      avoidance_enabled: bool = False) -> Tuple[str, str]:
    """Choose ``main``, ``secondary`` or fail-safe ``stop``.

    The secondary controller follows the avoidance manager outside the highway
    and the escape4 strategy inside it.  Required state/command streams are
    fail-closed: losing one never silently gives obstacle or highway authority
    back to the nominal controller.
    """
    if highway_enabled:
        if highway_state is None or not 0.0 <= highway_state_age_s <= timeout_s:
            return "stop", "highway_state_stale"
        phase = highway_state.get("phase")
        route_s = highway_state.get("route_s")
        highway_active = phase not in (None, "OFF")
        if phase == "DONE" and (route_s is None or float(route_s) >= highway_end_s):
            highway_active = False
        if highway_active:
            if not 0.0 <= secondary_cmd_age_s <= timeout_s:
                return "stop", "secondary_command_stale_highway"
            return "secondary", "highway_%s" % phase

    if avoidance_enabled:
        if avoidance_state is None or not 0.0 <= avoidance_state_age_s <= timeout_s:
            return "stop", "avoidance_state_stale"
        state = avoidance_state.get("state")
        if not state:
            return "stop", "avoidance_state_invalid"
        if state != "NORMAL":
            if not 0.0 <= secondary_cmd_age_s <= timeout_s:
                return "stop", "secondary_command_stale_avoidance"
            return "secondary", "avoidance_%s" % state

    if not 0.0 <= main_cmd_age_s <= timeout_s:
        return "stop", "main_command_stale"
    return "main", "nominal"


def route_window(points: Sequence[Point], s_values: Sequence[float], s: float,
                 back_m: float, ahead_m: float) -> List[Point]:
    """Route points from back_m behind to ahead_m ahead of arc length s,
    with one point beyond each end so the window covers both limits."""
    lo = max(0, bisect_left(s_values, s - back_m) - 1)
    hi = min(len(points), bisect_right(s_values, s + ahead_m) + 1)
    return list(points[lo:hi])
