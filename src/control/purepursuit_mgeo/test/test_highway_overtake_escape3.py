#!/usr/bin/env python3
"""escape3: the lane count from the camera's line crossings, next to escape2.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake_escape3.py" -v
"""
import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from purepursuit_mgeo import highway_overtake_escape2 as escape2  # noqa: E402
from purepursuit_mgeo import highway_overtake_escape3 as escape3  # noqa: E402

ESCAPE3_CONFIG = PACKAGE / "config/highway_overtake_escape3.yaml"


def line(track, y):
    """A straight line y (left +) metres beside the ego, like real_lane_node's lane meta."""
    return {"detected": True, "type": "white_dashed", "dashed": True, "track_id": track,
            "age": 10, "coef": [0.0, 0.0, y], "x_range_m": [4.0, 40.0],
            "from_guide": False, "coasted": False}


class Frames:
    """A planner in the zone fed camera frames: (track id, y) per real_lane label."""

    def __init__(self, state="CHASE", crossed=0, lane_index=0):
        route = [(float(x), 0.0) for x in range(-50, 500, 5)]
        self.p = escape3.HighwayOvertakePlanner(escape3.OvertakeConfig(), route)
        self.p.state, self.p.crossed, self.p.lane_index = state, crossed, lane_index
        self.t, self.x, self.events = 100.0, 0.0, []
        self.p.on_odom(self.t, self.x, 0.0, 0.0, 20.0)

    def __call__(self, left=None, right=None, under=None):
        self.t += 0.083
        self.x += 1.5
        self.p.on_odom(self.t, self.x, 0.0, 0.0, 20.0)
        info = {"timestamp": self.t, "output_status": "FRESH", "lane_valid": True,
                "left_lane": line(*left) if left else {"detected": False},
                "right_lane": line(*right) if right else {"detected": False},
                "straddling_lane": line(*under) if under else None}
        lines = escape3.parse_lane_lines(info, self.t, self.p.cfg)
        self.p._track_crossings(lines, self.t, (self.x, 0.0, 0.0), self.events)
        return self.p.crossed


class CrossingCountTest(unittest.TestCase):
    def test_left_line_under_the_ego_is_one_crossing(self):
        f = Frames()
        self.assertEqual(f(left=(7, 0.6), right=(8, -2.9)), 0)
        self.assertEqual(f(left=(9, 3.9), right=(8, -3.0), under=(7, 0.45)), 1)   # +1 -> 0
        self.assertEqual(f(left=(9, 3.6), right=(8, -3.3), under=(7, 0.1)), 1)
        self.assertEqual(f(left=(9, 2.9), right=(7, -0.6)), 1)                     # now -1
        self.assertIn("crossing 1 complete: the line is on the right", f.events)

    def test_right_line_under_the_ego_is_no_crossing(self):
        # The entry lane's right edge under the ego (-1 -> 0 -> -1); the merge
        # divider after it is still the first crossing.
        f = Frames()
        f(left=(1, 2.9), right=(2, -0.6))
        self.assertEqual(f(left=(1, 3.0), under=(2, -0.45)), 0)
        self.assertEqual(f(left=(1, 2.9), right=(2, -0.6)), 0)
        self.assertEqual(f(left=(1, 0.6), right=(2, -2.9)), 0)
        self.assertEqual(f(left=(3, 3.9), right=(2, -3.0), under=(1, 0.45)), 1)

    def test_over_the_right_edge_and_back_is_no_crossing(self):
        f = Frames()
        f(left=(1, 2.9), right=(2, -0.6))
        f(left=(1, 3.0), under=(2, -0.4))
        self.assertEqual(f(left=(2, 0.6), right=(3, -2.9)), -1)                    # right of it
        self.assertEqual(f(left=(1, 3.9), right=(3, -3.0), under=(2, 0.45)), 0)    # back over it
        self.assertEqual(f(left=(1, 2.9), right=(2, -0.6)), 0)
        for _ in range(20):                                                        # 30 m on
            f(left=(1, 1.75), right=(2, -1.75))
        self.assertEqual(f(left=(1, 0.6), right=(2, -2.9)), 0)
        self.assertEqual(f(left=(4, 3.9), right=(2, -3.0), under=(1, 0.45)), 1)

    def test_on_a_line_at_the_zone_start_is_no_crossing(self):
        f = Frames()
        self.assertEqual(f(left=(1, 3.0), under=(2, -0.2)), 0)                     # no frame before
        self.assertEqual(f(left=(1, 2.9), right=(2, -0.6)), 0)

    def test_stepping_back_off_the_left_line_is_uncounted(self):
        f = Frames()
        f(left=(7, 0.6), right=(8, -2.9))
        self.assertEqual(f(left=(9, 3.9), right=(8, -3.0), under=(7, 0.45)), 1)
        self.assertEqual(f(left=(7, 0.55), right=(8, -2.95)), 0)
        self.assertIn("stepped back off the left line: crossing 1 undone", f.events)
        self.assertEqual(f(left=(9, 3.9), right=(8, -3.0), under=(7, 0.45)), 1)

    def test_double_line_counts_once(self):
        # The entry divider seen as its two lines (11 broken, 12 solid) 0.3 m apart.
        f = Frames()
        f(left=(11, 0.6), right=(8, -2.9))
        self.assertEqual(f(left=(12, 0.9), right=(8, -2.9), under=(11, 0.45)), 1)
        self.assertEqual(f(left=(12, 0.6), right=(8, -3.0), under=(11, 0.3)), 1)
        self.assertEqual(f(left=(13, 3.7), right=(11, -0.2), under=(12, 0.1)), 1)
        self.assertEqual(f(left=(13, 3.5), right=(11, -0.6), under=(12, -0.3)), 1)
        self.assertEqual(f(left=(13, 2.9), right=(12, -0.6)), 1)
        self.assertEqual(f.events.count("left line under the ego (camera): crossing 1"), 1)
        self.assertIn("crossing 1 complete: the line is on the right", f.events)

    def test_untracked_lines_by_position(self):
        f = Frames()
        f(left=(None, 0.6), right=(None, -2.9))
        self.assertEqual(f(left=(None, 3.9), right=(None, -3.0), under=(None, 0.45)), 1)
        self.assertEqual(f(left=(None, 3.7), right=(None, -3.2), under=(None, 0.25)), 1)
        self.assertEqual(f(left=(None, 2.9), right=(None, -0.6)), 1)
        self.assertIn("crossing 1 complete: the line is on the right", f.events)


class FinalLaneTest(unittest.TestCase):
    def test_the_final_lanes_left_line_is_not_crossed(self):
        f = Frames(state="FOLLOW", crossed=3, lane_index=3)
        f(left=(5, 0.6), right=(4, -2.9))
        self.assertEqual(f(left=(6, 3.9), right=(4, -3.0), under=(5, 0.45)), 3)
        self.assertTrue(f.p.guard)
        self.assertEqual(f.p.state, f.p.FOLLOW)
        # The lane held is the one right of line 5: the ego is 1.3 m left of its centre.
        self.assertAlmostEqual(f.p.ref.to_frenet(f.x, 0.0)[1], 0.5*f.p.width - 0.45, places=2)
        f(left=(5, 0.6), right=(4, -2.9))
        self.assertFalse(f.p.guard)
        self.assertIn("final lane held", f.events)


class GeometryTest(unittest.TestCase):
    def test_gentler_changes_set_over_the_shared_yaml(self):
        cfg = escape3.OvertakeConfig()
        for v, escape2_m, escape3_m in ((15.0, 60, 85), (20.0, 75, 98), (22.0, 82, 105)):
            self.assertAlmostEqual(escape2.change_geometry(3.5, v, escape2.OvertakeConfig())[0],
                                   escape2_m, delta=1.0)
            self.assertAlmostEqual(escape3.change_geometry(3.5, v, cfg)[0], escape3_m, delta=1.0)
        values = yaml.safe_load(ESCAPE3_CONFIG.read_text(encoding="utf-8"))
        for key, value in values.items():
            self.assertEqual(value, getattr(cfg, key), key)


class EscapeThreeSimulationTest(unittest.TestCase):
    def test_three_changes_counted_by_the_camera(self):
        import highway_overtake_sim as sim
        r = sim.run(sim.ALL_SCENARIOS["real_camera"], 0, record=False, controller="mission",
                    planner="escape3")
        self.assertEqual(sim.verdict(r), [])
        texts = [e[2] for e in r["events"]]
        # The entry double line is seen as two lines and counts once.
        self.assertEqual(sum("left line under the ego" in t for t in texts), 3)
        self.assertEqual(sum("lane change complete" in t and "crossed line on the right" in t
                             for t in texts), 3)
        self.assertFalse(any("undone" in t or "WARNING" in t for t in texts))

    def test_a_frame_heading_error_does_not_carry_the_ego_past_the_lane(self):
        # Every line seen 1 deg to the left: the frozen frame of a change drifts
        # over 4 m in 100 m; the camera lane after the crossed line does not.
        import highway_overtake_sim as sim
        scenario = dataclasses.replace(sim.ALL_SCENARIOS["camera_yaw_bias"], camera_yaw_bias_deg=1.0)
        r = sim.run(scenario, 1, record=False, controller="mission", planner="escape3")
        self.assertEqual(sim.verdict(r), [])
        self.assertLess(r["max_overshoot_m"], 1.0)
        frozen = sim.run(scenario, 1, record=False, controller="mission", planner="escape3",
                         cfg_overrides={"crossing_reacquire": False})
        self.assertGreater(frozen["max_overshoot_m"], 2.0)
        self.assertTrue(frozen["entered_forbidden_lane"])


class EscapeThreeNodeTest(unittest.TestCase):
    def test_node_runs_escape3_with_its_config(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        extra = {"~" + k: v for k, v in
                 yaml.safe_load(ESCAPE3_CONFIG.read_text(encoding="utf-8")).items()}
        wrapper, _pubs = load_node("highway_overtake_escape3_node", overtake_params(**extra), clock)
        wrapper.node.time = NS(time=lambda: clock.now)
        node = wrapper.node.HighwayOvertakeNode()
        self.assertIsInstance(node.planner, escape3.HighwayOvertakePlanner)
        self.assertNotIsInstance(node.planner, escape2.HighwayOvertakePlanner)
        self.assertEqual(node.planner.cfg.max_entry_angle_deg, 3.0)
        self.assertTrue(node.planner.cfg.crossing_reacquire)


if __name__ == "__main__":
    unittest.main()
