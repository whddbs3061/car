#!/usr/bin/env python3
"""Checks for purepursuit_mgeo.highway_overtake (no ROS needed).

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake.py" -v
"""
import math
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))

from purepursuit_mgeo.highway_overtake import (  # noqa: E402
    HighwayOvertakePlanner, LaneReference, LaneTrack, OvertakeConfig, align_with_controller,
    change_geometry, evaluate_change, make_change_plan, parse_lane_info,
)


def lane(coef, kind="white_dashed", age=5, **extra):
    meta = {"detected": True, "type": kind, "dashed": kind == "white_dashed",
            "coef": list(coef), "x_range_m": [4.0, 40.0], "age": age,
            "from_guide": False, "coasted": False}
    meta.update(extra)
    return meta


def lane_info(stamp, left=(0.0, 0.0, 1.75), right=(0.0, 0.0, -1.75), **extra):
    info = {"timestamp": stamp, "output_status": "FRESH", "lane_valid": True,
            "left_lane": lane(left), "right_lane": lane(right, "white_solid"),
            "straddling_lane": None}
    info.update(extra)
    return info


class ParseLaneInfoTest(unittest.TestCase):
    cfg = OvertakeConfig()

    def test_real_lane_payload(self):
        obs, reason = parse_lane_info(lane_info(100.0), 100.05, self.cfg)
        self.assertEqual(reason, "ok")
        self.assertTrue(obs.left_dashed)
        self.assertAlmostEqual(obs.width, 3.5)
        self.assertFalse(obs.stamp_is_capture)

    def test_v2_capture_stamp(self):
        info = lane_info(99.9, observation_time_source="camera_receive_wall")
        obs, _ = parse_lane_info(info, 100.0, self.cfg)
        self.assertTrue(obs.stamp_is_capture)
        self.assertAlmostEqual(obs.stamp, 99.9)

    def test_rejections(self):
        cases = {
            "straddling": lane_info(100.0, straddling_lane=lane((0, 0, 0.1))),
            "status_held": lane_info(100.0, output_status="HELD"),
            "left_predicted": lane_info(100.0, left_lane=lane((0, 0, 1.75), coasted=True)),
            "left_young": lane_info(100.0, left_lane=lane((0, 0, 1.75), age=1)),
            "width_2.40": lane_info(100.0, left=(0, 0, 1.2), right=(0, 0, -1.2)),
            "stale": lane_info(99.0),
        }
        for expected, info in cases.items():
            obs, reason = parse_lane_info(info, 100.0, self.cfg)
            self.assertIsNone(obs, expected)
            self.assertEqual(reason, expected)


class LaneReferenceTest(unittest.TestCase):
    cfg = OvertakeConfig()

    def obs(self, right):
        info = lane_info(100.0, left=(0.0, 0.0, 1.75), right=right)
        return parse_lane_info(info, 100.0, self.cfg)[0]

    def test_round_trip_in_rotated_map(self):
        ref = LaneReference.from_observation(self.obs((0, 0, -1.75)), (10.0, 5.0, 1.0), self.cfg)
        x, y = ref.point(25.0, 3.5)
        s, d = ref.to_frenet(x, y)
        self.assertAlmostEqual(s, 25.0, places=6)
        self.assertAlmostEqual(d, 3.5, places=6)

    def test_converging_right_edge_does_not_tilt_the_frame(self):
        # Entry-lane taper: right edge converges at 3 deg, the divider is straight.
        ref = LaneReference.from_observation(
            self.obs((0.0, math.tan(math.radians(3.0)), -1.75)), (0.0, 0.0, 0.0), self.cfg)
        self.assertAlmostEqual(ref.psi, 0.0, places=9)


class ChangeGeometryTest(unittest.TestCase):
    def test_entry_angle_and_lateral_acceleration(self):
        cfg = OvertakeConfig()
        for v in (5.0, 10.0, 15.0, 22.0, 25.0):
            length, ramp = change_geometry(3.5, v, cfg)
            angle = math.degrees(math.atan(3.5/((1.0 - ramp)*length)))
            a_lat = v*v*1.5*3.5/(ramp*(1.0 - ramp)*length*length)
            self.assertLessEqual(angle, cfg.max_entry_angle_deg + 1e-6, v)
            self.assertLessEqual(a_lat, cfg.max_lateral_accel_mps2 + 1e-6, v)


class GapAcceptanceTest(unittest.TestCase):
    """Aggressive acceptance: merge as soon as the left car is passed."""
    cfg = OvertakeConfig()
    ref = LaneReference(0.0, 0.0, 0.0, 0.0, 200.0, 200.0)

    def decide(self, *tracks, v=20.0):
        plan = make_change_plan(self.ref, 0.0, 0.0, 3.5, v, self.cfg)
        return evaluate_change(plan, 0.0, v, list(tracks), self.cfg)

    @staticmethod
    def car(oid, s, lane, v):
        return LaneTrack(oid, s, 3.5*lane, v, 4.6, 1.9, lane)

    def test_empty_target_lane(self):
        self.assertTrue(self.decide().ok)

    def test_car_alongside_blocks(self):
        decision = self.decide(self.car(1, 1.5, 1, 20.0))
        self.assertFalse(decision.ok)
        self.assertEqual(decision.blocker, 1)

    def test_just_passed_slower_car_is_accepted(self):
        # Our rear bumper (s=-0.82) is ~1.5 m ahead of its front bumper.
        self.assertTrue(self.decide(self.car(1, -4.6, 1, 15.0)).ok)

    def test_faster_car_close_behind_blocks(self):
        decision = self.decide(self.car(1, -12.0, 1, 26.0))
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "target_rear")

    def test_own_lane_follower_is_ignored(self):
        self.assertTrue(self.decide(self.car(1, -8.0, 0, 25.0)).ok)


class IntegrationGateTest(unittest.TestCase):
    """route_mission_gate zone/request flags and the published state names."""

    def setUp(self):
        cfg = OvertakeConfig(gate_required=True, zone_start_s=100.0, zone_end_s=900.0)
        self.planner = HighwayOvertakePlanner(cfg, [(float(x), 0.0) for x in range(0, 1001, 10)])

    def step(self, t, x, route_gate=None, request=None):
        p = self.planner
        p.on_odom(t, x, 0.0, 0.0, 15.0)
        p.on_obstacles(t, [])
        p.on_base_path(t, [(x - 5.0 + k, 0.0) for k in range(100)], False)
        p.on_lane_info(t, lane_info(t))
        if route_gate is not None:
            p.on_route_gate(t, route_gate)
        if request is not None:
            p.on_lane_change_request(t, request)
        return p.step(t)

    def test_zone_needs_the_route_gate(self):
        out = self.step(0.0, 200.0)
        self.assertEqual((out.state, out.status["state"]), ("OFF", "OFF"))
        self.assertFalse(out.active)
        out = self.step(0.05, 201.0, route_gate=False)
        self.assertEqual(out.state, "OFF")
        out = self.step(0.10, 202.0, route_gate=True)
        self.assertEqual(out.status["state"], "WAIT_GAP")

    def test_stale_gate_is_closed(self):
        self.step(0.0, 200.0, route_gate=True)
        out = self.step(1.0, 215.0)                  # gate last seen 1.0 s ago
        self.assertFalse(out.status["route_gate"])

    def test_change_needs_the_request(self):
        t = 0.0
        for k in range(4):
            out = self.step(t, 200.0 + k, route_gate=True, request=False)
            t += 0.05
        self.assertEqual((out.state, out.status["reason"]), ("CHASE", "lane_change_not_requested"))
        out = self.step(t, 205.0, route_gate=True, request=True)
        self.assertEqual((out.state, out.status["state"], out.status["phase"]),
                         ("CHANGE", "LANE_CHANGE", "CHANGE"))

    def test_state_names(self):
        self.assertEqual(set(HighwayOvertakePlanner.INTEGRATION_STATE.values()),
                         {"OFF", "WAIT_GAP", "LANE_CHANGE", "INNER_HOLD", "DONE"})


class ControllerAlignmentTest(unittest.TestCase):
    def test_lowers_only_the_limits_the_controller_cannot_follow(self):
        cfg = OvertakeConfig()
        changed = align_with_controller(cfg, 1.0, 1.0, 1.5)
        self.assertAlmostEqual(cfg.max_lateral_accel_mps2, 0.9)
        self.assertEqual((cfg.target_accel_mps2, cfg.target_decel_mps2), (1.0, 1.5))
        self.assertAlmostEqual(cfg.end_decel_mps2, 1.2)       # margin for the response lag
        self.assertNotIn("gap_check_accel_mps2", changed)        # already 1.0
        self.assertEqual(cfg.gap_check_slow_accel_mps2, 0.5)

    def test_generous_or_missing_limits_change_nothing(self):
        cfg = OvertakeConfig()
        self.assertEqual(align_with_controller(cfg, 3.8, 0.0, 9.0), {})
        self.assertEqual(cfg.max_lateral_accel_mps2, 1.5)


class SimulatorSmokeTest(unittest.TestCase):
    """Closed loop on the sample-scenario traffic: no collision, no solid line."""

    def test_scenarios(self):
        import highway_overtake_sim as sim
        for name, seed, controller in (("morai", 0, "mission"), ("morai", 1, "example"),
                                       ("empty", 0, "mission")):
            result = sim.run(sim.ALL_SCENARIOS[name], seed, record=False, controller=controller)
            self.assertEqual(sim.verdict(result), [], (name, seed, controller))
            self.assertEqual(result["lane_changes"], 3, (name, seed, controller))
            self.assertLess(result["max_route_offset_m"], sim.MAX_ROUTE_OFFSET_M - 2.0)


class LaneFunnelTest(unittest.TestCase):
    """The zone end is a lane funnel (MGeo): L1/L2 end on a hatched island."""

    def test_yield_merges_before_the_island(self):
        import highway_overtake_sim as sim
        scenario = sim.ALL_SCENARIOS["morai"]
        held = sim.run(scenario, 3, {"deadline_mode": "off"}, record=False, controller="example")
        self.assertGreater(held["island_s"], 0.0)          # the first design drives onto it
        yielded = sim.run(scenario, 3, record=False, controller="example")
        self.assertEqual(sim.verdict(yielded), [])
        self.assertEqual(yielded["lane_changes"], 3)

    def test_drifted_frame_is_reacquired(self):
        # Frozen frame ~0.5 deg off: re-measurement failed for 130 m before.
        import highway_overtake_sim as sim
        r = sim.run(sim.ALL_SCENARIOS["morai_camera_noise"], 6, record=False, controller="example")
        self.assertEqual(sim.verdict(r), [])
        self.assertTrue(any("new lane 3 measured" in e[2] for e in r["events"]))


if __name__ == "__main__":
    unittest.main()
