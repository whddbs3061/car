import math
from pathlib import Path
import unittest

from purepursuit_mgeo.path import PathPoint, load_mgeo_path
from purepursuit_mgeo.route_avoidance_gate import RouteAvoidanceGate, route_fingerprint


REGIONS = [
    {"name": "static_obstacle", "start_xyz": [-66.77, -198.78, 29.90],
     "end_xyz": [-59.60, -86.96, 29.86]},
    {"name": "dynamic_obstacle", "start_xyz": [-59.36, -63.85, 29.86],
     "end_xyz": [-60.07, 16.88, 29.87]},
]
EXPECTED = "478510b0d72f42d3aad8454da7eb27f9288b6c4d95122aa5eb20e2e6bf1220df"


def actual_route():
    workspace = Path(__file__).resolve().parents[1]
    return load_mgeo_path(str(workspace / "data/routes/2026_molit_comp_global_path.txt"))


class RouteAvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.points = [PathPoint(float(i), 0., 0.) for i in range(201)]
        self.regions = [
            {"name": "static", "start_xyz": [10, 0, 1], "end_xyz": [20, 0, 1]},
            {"name": "dynamic", "start_xyz": [30, 0, 1], "end_xyz": [40, 0, 1]},
        ]
        self.gate = RouteAvoidanceGate(self.points, self.regions)

    def test_boundary_crossing_not_nearest_midpoint_or_lookahead(self):
        self.assertFalse(self.gate.update(9.9, 0)["allowed"])
        self.assertEqual(self.gate.update(10, 0)["regions"], ["static"])
        self.assertTrue(self.gate.update(19.9, 0)["allowed"])
        self.assertFalse(self.gate.update(20, 0)["allowed"])
        self.assertEqual(self.gate.update(30, 0)["regions"], ["dynamic"])
        self.assertFalse(self.gate.update(40, 0)["allowed"])

    def test_skipped_boundary_samples_and_jitter_do_not_reenter(self):
        self.gate.update(9, 0)
        self.assertTrue(self.gate.update(12, 0)["allowed"])
        self.assertFalse(self.gate.update(21, 0)["allowed"])
        self.assertFalse(self.gate.update(19.5, 0)["allowed"])

    def test_blackout_allows_outside_and_route_unmatched_by_default(self):
        self.assertTrue(self.gate.update(5, 50, True)["allowed"])
        self.assertTrue(self.gate.update(5, 0, True)["allowed"])
        self.assertFalse(self.gate.update(5, 0, False)["allowed"])
        gate = RouteAvoidanceGate(self.points, self.regions, gps_blackout_override=False)
        self.assertFalse(gate.update(5, 0, True)["allowed"])
        self.assertTrue(gate.update(15, 0, True)["allowed"])

    def test_initialization_inside_region_and_lateral_detour(self):
        result = self.gate.update(15, 6)
        self.assertTrue(result["allowed"])
        self.assertAlmostEqual(result["progress_s_m"], 15.)

    def test_invalid_pose_does_not_advance_progress(self):
        self.gate.update(5, 0)
        self.assertFalse(self.gate.update(math.nan, 0)["allowed"])
        self.assertEqual(self.gate.progress_s, 5)

    def test_continuity_does_not_jump_to_future_parallel_branch(self):
        points = [PathPoint(float(i), 0, 0) for i in range(101)]
        points += [PathPoint(float(i), 1, 0) for i in reversed(range(101))]
        gate = RouteAvoidanceGate(points, self.regions)
        gate.update(5, 0)
        status = gate.update(15, .9)
        self.assertLess(status["progress_waypoint_index"], 100)
        self.assertTrue(status["allowed"])

    def test_rejects_wrong_route_or_reversed_region(self):
        with self.assertRaises(ValueError):
            RouteAvoidanceGate(self.points, self.regions, expected_fingerprint="wrong")
        with self.assertRaises(ValueError):
            RouteAvoidanceGate(self.points, [dict(name="bad", start_xyz=[20,0,0],
                                                end_xyz=[10,0,0])])

    def test_actual_pdf_landmarks_and_complete_forward_lap(self):
        points = actual_route()
        self.assertEqual(len(points), 4430)
        self.assertEqual(route_fingerprint(points), EXPECTED)
        gate = RouteAvoidanceGate(points, REGIONS, EXPECTED)
        self.assertEqual([(r["start_index"], r["end_index"]) for r in gate.regions],
                         [(553, 782), (828, 998)])
        for i, point in enumerate(points):
            status = gate.update(point.x, point.y)
            self.assertEqual(status["allowed"], 553 <= i < 782 or 828 <= i < 998,
                             "unexpected region at waypoint %d" % i)


if __name__ == "__main__":
    unittest.main()
