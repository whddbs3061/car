#!/usr/bin/env python3
"""escape4: escape3 whose changes are flatter at speed (sideways speed limit).

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake_escape4.py" -v
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from purepursuit_mgeo import highway_overtake_escape3 as escape3  # noqa: E402
from purepursuit_mgeo import highway_overtake_escape4 as escape4  # noqa: E402

ESCAPE4_CONFIG = PACKAGE / "config/highway_overtake_escape4.yaml"


class GeometryTest(unittest.TestCase):
    def test_flatter_at_speed(self):
        cfg = escape4.OvertakeConfig()
        for kmh, length in ((54.0, 85), (63.0, 98), (72.0, 112), (79.2, 123)):
            v = kmh/3.6
            L, r = escape4.change_geometry(3.5, v, cfg)
            self.assertAlmostEqual(L, length, delta=1.0, msg=kmh)
            # diagonal_progress: steepest slope 1/(1-r) of the normalised shift
            self.assertLessEqual(v*3.5/((1.0 - r)*L), cfg.max_lateral_speed_mps + 1e-6, kmh)
        # Up to 54 km/h the 3 deg of escape3 already keeps 0.8 m/s.
        self.assertAlmostEqual(escape4.change_geometry(3.5, 15.0, cfg)[0],
                               escape3.change_geometry(3.5, 15.0, escape3.OvertakeConfig())[0])

    def test_yaml_matches_the_defaults(self):
        cfg = escape4.OvertakeConfig()
        values = yaml.safe_load(ESCAPE4_CONFIG.read_text(encoding="utf-8"))
        for key, value in values.items():
            self.assertEqual(value, getattr(cfg, key), key)


class PassOrMergeBehindTest(unittest.TestCase):
    def setUp(self):
        route = [(float(x), 0.0) for x in range(-50, 500, 5)]
        self.cfg = escape4.OvertakeConfig()
        self.p = escape4.HighwayOvertakePlanner(self.cfg, route)

    def car(self, s, v, d=3.5):
        return escape4.LaneTrack(7, s, d, v, 4.6, 1.9, 1, 0.0, 0.0)

    def test_pass_time_from_the_relative_speed(self):
        blocked = escape4.GapDecision(False, "target_rear", 7)
        # Beside the ego (centre at its rear axle): 3.1 m to make up plus the
        # rear gap 1 m + 0.5 s x v.
        self.assertAlmostEqual(self.p._pass_time(blocked, [self.car(0.0, 11.0)], 0.0, 15.0),
                               (1.0 + 0.5*11.0 + 3.12)/4.0, places=1)       # 2.4 s: pass
        self.assertGreater(self.p._pass_time(blocked, [self.car(0.0, 14.0)], 0.0, 15.0),
                           self.cfg.pass_time_max_s)                         # 11 s: let it past
        self.assertTrue(math.isinf(self.p._pass_time(blocked, [self.car(0.0, 16.0)], 0.0, 15.0)))
        self.assertEqual(self.p._pass_time(escape4.GapDecision(False, "current_front", 7),
                                           [self.car(0.0, 14.0)], 0.0, 15.0), 0.0)

    def test_a_target_lane_car_alongside_is_followed_while_changing(self):
        # Its centre 3 m ahead of the ego's rear axle, bumper gap < 0.
        car = self.car(3.0, 14.7)
        v, _, info = self.p._follow([car], 0.0, 0.3, 11.0, preview_d=3.5)
        self.assertEqual(info["lead"], 7)
        self.assertLess(v, 14.7)
        old = escape3.HighwayOvertakePlanner(escape3.OvertakeConfig(), [(0.0, 0.0), (500.0, 0.0)])
        self.assertIsNone(old._follow([car], 0.0, 0.3, 11.0, preview_d=3.5)[0])   # escape3 kept pace


class EscapeFourSimulationTest(unittest.TestCase):
    def test_three_changes_slower_sideways_than_escape3(self):
        import highway_overtake_sim as sim
        sideways = {}
        for planner in ("escape3", "escape4"):
            r = sim.run(sim.ALL_SCENARIOS["real_camera"], 0, record=True, controller="mission",
                        planner=planner)
            self.assertEqual(sim.verdict(r), [], planner)
            self.assertEqual(r["lane_changes"], 3, planner)
            sideways[planner] = max(p["v"]*math.sin(p["yaw"]) for p in r["_log"]
                                    if 0.0 <= p["x"] <= 360.0)
        self.assertGreater(sideways["escape3"], 1.0)
        self.assertLess(sideways["escape4"], 0.95)       # 0.8 planned, the rest is tracking

    def test_no_emergency_stop_beside_the_cars_it_merges_among(self):
        # Seeds where escape4 designed for 79 km/h ran into a car beside it.
        import highway_overtake_sim as sim
        for seed in (3, 4):
            r = sim.run(sim.ALL_SCENARIOS["real_camera"], seed, record=False, controller="mission",
                        planner="escape4")
            self.assertEqual(sim.verdict(r), [], seed)
            self.assertFalse([e for e in r["events"] if "EMERGENCY" in e[2]], seed)


class EscapeFourNodeTest(unittest.TestCase):
    def test_node_runs_escape4_with_its_config(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        extra = {"~" + k: v for k, v in
                 yaml.safe_load(ESCAPE4_CONFIG.read_text(encoding="utf-8")).items()}
        wrapper, _pubs = load_node("highway_overtake_escape4_node", overtake_params(**extra), clock)
        wrapper.node.time = NS(time=lambda: clock.now)
        node = wrapper.node.HighwayOvertakeNode()
        self.assertIsInstance(node.planner, escape4.HighwayOvertakePlanner)
        self.assertNotIsInstance(node.planner, escape3.HighwayOvertakePlanner)
        self.assertEqual(node.planner.cfg.max_lateral_speed_mps, 0.8)


class HighwayEndTest(unittest.TestCase):
    """28 km/h at the highway end (A2256W000153 end); 30 km/h to the toll gate beyond."""

    def planner(self, state):
        route = [(float(x), 0.0) for x in range(-50, 2000, 5)]
        p = escape4.HighwayOvertakePlanner(escape4.OvertakeConfig(), route)
        p.state = state
        return p

    def test_cap_toward_the_highway_end(self):
        p = self.planner("DONE")
        cfg = p.cfg
        end = cfg.highway_end_s
        self.assertAlmostEqual(p._highway_end_cap(end - 5.0), cfg.highway_end_speed_mps)
        self.assertAlmostEqual(p._highway_end_cap(end - 110.0),
                               math.sqrt(cfg.highway_end_speed_mps**2 + 2.0*1.0*100.0))
        self.assertIsNone(p._highway_end_cap(end + 1.0))
        self.assertIsNone(self.planner("FOLLOW")._highway_end_cap(end - 5.0))

    def test_node_pass_through_speed_is_capped(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        extra = {"~" + k: v for k, v in
                 yaml.safe_load(ESCAPE4_CONFIG.read_text(encoding="utf-8")).items()}
        wrapper, _pubs = load_node("highway_overtake_escape4_node", overtake_params(**extra), clock)
        wrapper.node.time = NS(time=lambda: clock.now)
        node = wrapper.node.HighwayOvertakeNode()
        node.planner.pass_through_cap_mps = 5.0
        self.assertAlmostEqual(node._limit_speed_rate(12.0, 100.0), 5.0)
        node.planner.pass_through_cap_mps = None
        self.assertAlmostEqual(node._limit_speed_rate(30.0, 100.0), node.cruise_speed_mps)

    def test_simulated_speed_at_the_highway_end(self):
        import highway_overtake_sim as sim
        cfg = sim.default_config(sim.GLOBAL_MAP, escape4)
        end_x = sim.ZONE_END_X + 100.0          # on the simulated road
        end = sim.Route(sim.GLOBAL_MAP).s_of_xy(*sim.road_to_map(end_x, sim.global_path_y(end_x)))
        r = sim.run(sim.ALL_SCENARIOS["real_camera"], 0, record=True, controller="mission",
                    planner="escape4", cfg_overrides={"highway_end_s": end})
        self.assertEqual(sim.verdict(r), [])
        at_end = next(p for p in r["_log"] if p["x"] >= end_x)
        self.assertLessEqual(at_end["v"]*3.6, 30.0)
        self.assertGreaterEqual(at_end["v"]*3.6, 26.0)


if __name__ == "__main__":
    unittest.main()
