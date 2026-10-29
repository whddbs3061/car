#!/usr/bin/env python3
"""The lead-escape planner variant and its node, next to the stable planner.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake_escape.py" -v
"""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from purepursuit_mgeo import highway_overtake as stable  # noqa: E402
from purepursuit_mgeo import highway_overtake_escape as escape  # noqa: E402

REF = stable.LaneReference(0.0, 0.0, 0.0, 0.0, 200.0, 200.0)


def decide(module, lead_s, v=22.0):
    cfg = module.OvertakeConfig()
    plan = module.make_change_plan(REF, 0.0, 0.0, 3.5, v, cfg)
    lead = module.LaneTrack(1, lead_s, 0.0, 11.0, 4.6, 1.9, 0)      # 40 km/h, our lane
    return module.evaluate_change(plan, 0.0, v, [lead], cfg)


class LeadEscapeTest(unittest.TestCase):
    """A slower car ahead in the ego's lane is passed sideways, not braked for."""

    def test_escapable_lead_allows_the_change_without_braking(self):
        # Bumper gap ~32 m, closing 11 m/s: beside it before the gap is gone.
        self.assertEqual(decide(stable, 38.0).reason, "current_front")   # match speed first
        self.assertTrue(decide(escape, 38.0).ok)

    def test_too_close_to_escape_still_waits(self):
        self.assertEqual(decide(escape, 28.0).reason, "current_front")

    def test_far_leads_are_unchanged(self):
        for s in (50.0, 70.0):
            self.assertEqual(decide(stable, s).ok, decide(escape, s).ok)

    def test_lane_being_left_is_not_followed_but_still_stops_when_close(self):
        route = [(float(x), 0.0) for x in range(0, 1001, 10)]
        planner = escape.HighwayOvertakePlanner(escape.OvertakeConfig(), route)
        reference = stable.HighwayOvertakePlanner(stable.OvertakeConfig(), route)

        def lead(module, gap):
            cfg = module.OvertakeConfig()
            return module.LaneTrack(1, cfg.front_extent_m + gap + 2.3, 0.0, 11.0, 4.6, 1.9, 0)

        target, emergency, info = planner._follow([lead(escape, 10.0)], 0.0, 1.0, 22.0, 3.5,
                                                  escape_lane=0)
        self.assertEqual((target, emergency, info["lead"]), (None, False, None))
        self.assertTrue(planner._follow([lead(escape, 1.0)], 0.0, 1.0, 22.0, 3.5,
                                        escape_lane=0)[1])
        target, emergency, _ = reference._follow([lead(stable, 10.0)], 0.0, 1.0, 22.0, 3.5)
        self.assertTrue(emergency)                         # stable: TTC < 1 s
        self.assertLess(target, 22.0)


class EscapeSimulationTest(unittest.TestCase):
    def test_sample_traffic_is_faster_and_still_clean(self):
        import highway_overtake_sim as sim
        scenario = sim.ALL_SCENARIOS["morai"]
        slow = sim.run(scenario, 1, record=False, controller="mission", planner="stable")
        fast = sim.run(scenario, 1, record=False, controller="mission", planner="escape")
        self.assertEqual(sim.verdict(fast), [])
        self.assertEqual(fast["lane_changes"], 3)
        self.assertLess(fast["zone_time_s"], slow["zone_time_s"] - 3.0)
        self.assertGreater(fast["min_front_gap_m"], 3.0)

    def test_funnel_case_reaches_the_final_lane(self):
        import highway_overtake_sim as sim
        r = sim.run(sim.ALL_SCENARIOS["morai"], 3, record=False, controller="example",
                    planner="escape")
        self.assertEqual(sim.verdict(r), [])
        self.assertEqual(r["lane_changes"], 3)


class EscapeNodeTest(unittest.TestCase):
    """highway_overtake_escape_node.py is highway_overtake_node.py with the escape planner."""

    def test_node_runs_the_escape_planner(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        wrapper, _pubs = load_node("highway_overtake_escape_node", overtake_params(), clock)
        wrapper.node.time = NS(time=lambda: clock.now)
        node = wrapper.node.HighwayOvertakeNode()
        self.assertIsInstance(node.planner, escape.HighwayOvertakePlanner)
        self.assertNotIsInstance(node.planner, stable.HighwayOvertakePlanner)

    def test_stable_node_keeps_the_stable_planner(self):
        from test_highway_overtake_node import Clock, load_node, overtake_params
        clock = Clock()
        module, _pubs = load_node("highway_overtake_node", overtake_params(), clock)
        node = module.HighwayOvertakeNode()
        self.assertIsInstance(node.planner, stable.HighwayOvertakePlanner)


if __name__ == "__main__":
    unittest.main()
