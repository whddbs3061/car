#!/usr/bin/env python3
"""escape2: entry double line and the map lane check, next to escape.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake_escape2.py" -v
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from purepursuit_mgeo import highway_overtake_escape as escape  # noqa: E402
from purepursuit_mgeo import highway_overtake_escape2 as escape2  # noqa: E402

LANES_FILE = PACKAGE / "config/highway_lanes.json"
# Road frame of the highway zone (x along the road from the zone start, y left).
THETA = math.atan2(-474.5, 2.3)


def road_to_map(x, y):
    c, s = math.cos(THETA), math.sin(THETA)
    return 62.4 + c*x - s*y, 215.0 + s*x + c*y


class LanesFileTest(unittest.TestCase):
    def test_file_is_current(self):
        import make_highway_lanes
        self.assertEqual(make_highway_lanes.main(["--check"]), 0, "run tools/make_highway_lanes.py")

    def test_map_lane_of_lane_centres(self):
        lanes = escape2.load_map_lanes(str(LANES_FILE))
        # Real lane centres (MGeo) at road x = 200: L1..L4 at y ~3.9, 7.4, 10.9, 14.4.
        for index, y in ((1, 3.9), (2, 7.4), (3, 10.9), (4, 14.4)):
            near = escape2.nearest_lanes(lanes, *road_to_map(200.0, y))
            self.assertEqual(near[0][1], index)
            self.assertLess(near[0][0], 0.6)
            self.assertGreater(near[1][0] - near[0][0], 2.5)
        near = escape2.nearest_lanes(lanes, *road_to_map(20.0, 0.0))     # entry lane
        self.assertEqual(near[0][1], 0)

    def test_funnel_end_is_ambiguous(self):
        lanes = escape2.load_map_lanes(str(LANES_FILE))
        near = escape2.nearest_lanes(lanes, *road_to_map(428.0, 10.6))
        self.assertLess(near[1][0] - near[0][0], 1.5)       # no clear match: no correction


class MapLaneCorrectionTest(unittest.TestCase):
    """The counted lane follows the map once the mismatch has lasted."""

    def planner(self):
        route = [road_to_map(float(x), 3.9) for x in range(-100, 500, 5)]
        cfg = escape2.OvertakeConfig(lanes_file=str(LANES_FILE))
        return escape2.HighwayOvertakePlanner(cfg, route)

    def test_mismatch_is_corrected_after_the_confirm_time(self):
        p = self.planner()
        p.state, p.lane_index = p.CHASE, 0               # counted: entry lane
        x, y = road_to_map(200.0, 3.9)                   # physically: L1
        events = []
        p._check_map_lane(10.0, x, y, events)
        p._check_map_lane(10.5, x, y, events)
        self.assertEqual(p.lane_index, 0)
        p._check_map_lane(11.1, x, y, events)
        self.assertEqual(p.lane_index, 1)
        self.assertIn("lane count corrected by the map: 0 -> 1", events)

    def test_no_correction_while_changing_or_when_unclear(self):
        p = self.planner()
        p.state, p.lane_index = p.CHANGE, 0
        x, y = road_to_map(200.0, 3.9)
        for t in (10.0, 11.0, 12.0):
            p._check_map_lane(t, x, y, [])
        self.assertEqual(p.lane_index, 0)
        p.state = p.CHASE
        x, y = road_to_map(200.0, 5.65)                  # on the L1/L2 line
        for t in (13.0, 14.0, 15.0):
            p._check_map_lane(t, x, y, [])
        self.assertEqual(p.lane_index, 0)

    def test_follow_resumes_changes_when_the_map_says_fewer(self):
        p = self.planner()
        p.state, p.lane_index = p.FOLLOW, 3
        x, y = road_to_map(200.0, 7.4)                   # physically: L2
        for t in (10.0, 11.1):
            p._check_map_lane(t, x, y, [])
        self.assertEqual((p.lane_index, p.state), (2, p.CHASE))


class EscapeTwoSimulationTest(unittest.TestCase):
    def test_entry_double_line_read_as_solid(self):
        import highway_overtake_sim as sim
        for planner, changes in (("escape", 0), ("escape2", 3)):
            r = sim.run(sim.ALL_SCENARIOS["entry_solid"], 0, record=False, controller="mission",
                        planner=planner)
            self.assertEqual(r["lane_changes"], changes, planner)
            if planner == "escape2":
                self.assertEqual(sim.verdict(r), [])

    def test_lost_lane_count_is_restored(self):
        import highway_overtake_sim as sim
        state = {"done": False}

        def lose_count(p, t, ex, ey):
            if not state["done"] and p.lane_index == 1 and p.state == p.REACQUIRE:
                p.lane_index, state["done"] = 0, True

        r = sim.run(sim.ALL_SCENARIOS["morai"], 0, record=False, controller="mission",
                    planner="escape2", on_step=lose_count)
        self.assertTrue(state["done"])
        self.assertTrue(any("corrected by the map: 0 -> 1" in e[2] for e in r["events"]))
        self.assertEqual(sim.verdict(r), [])


class EscapeTwoNodeTest(unittest.TestCase):
    def test_node_runs_escape2_with_the_lanes_file(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        wrapper, _pubs = load_node("highway_overtake_escape2_node",
                                   overtake_params(**{"~lanes_file": str(LANES_FILE)}), clock)
        wrapper.node.time = NS(time=lambda: clock.now)
        node = wrapper.node.HighwayOvertakeNode()
        self.assertIsInstance(node.planner, escape2.HighwayOvertakePlanner)
        self.assertNotIsInstance(node.planner, escape.HighwayOvertakePlanner)
        self.assertEqual([i for i, _ in node.planner.map_lanes], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
