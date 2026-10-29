#!/usr/bin/env python3
"""final_ws_highway_overtake_bringup.launch differs from the integration launch
only where intended, and the controller speed ceiling is neutral outside the zone.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_overtake_launch.py" -v
"""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(REPO / "src/experimental/curvature_speed_purepursuit/src"))

from curvature_speed_purepursuit import planner as cp  # noqa: E402
from purepursuit_mgeo.mission_regions import load_mission_regions  # noqa: E402
from purepursuit_mgeo.route_geometry import RoutePolyline  # noqa: E402

LAUNCH_DIR = REPO / "src/bringup/morai_bringup/launch"
SOURCE = LAUNCH_DIR / "final_ws_highway_bringup.launch"
OUTPUT = LAUNCH_DIR / "final_ws_highway_overtake_bringup.launch"
ROUTE_FILE = REPO / "data/routes/2026_molit_comp_global_path.txt"


def generator():
    spec = importlib.util.spec_from_file_location(
        "make_overtake_launch", REPO / "docker/final_ws/make_overtake_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(element):
    """Comparable form: tag, attributes and children, whitespace ignored."""
    return (element.tag, tuple(sorted(element.attrib.items())),
            tuple(canonical(child) for child in element))


def parse(path):
    return ET.parse(str(path)).getroot()


class GeneratedLaunchTest(unittest.TestCase):
    def setUp(self):
        self.source, self.output = parse(SOURCE), parse(OUTPUT)

    def test_written_files_are_current(self):
        self.assertEqual(generator().main(["--check"]), 0, "run docker/final_ws/make_overtake_launch.py")

    def test_only_the_intended_elements_differ(self):
        new_args = {"overtake_config_file", "highway_cruise_kph", "highway_max_speed_kph",
                    "controller_max_speed_kph"}
        source = [e for e in self.source]
        output = [e for e in self.output if not (e.tag == "arg" and e.get("name") in new_args)]
        self.assertEqual(len(source), len(output))
        for a, b in zip(source, output):
            name = a.get("name")
            if a.tag == "node" and name == "highway_lane_strategy":
                self.assertEqual(b.get("type"), "highway_overtake_node.py")
                self.assertEqual((b.get("pkg"), b.get("name"), b.get("output")),
                                 (a.get("pkg"), a.get("name"), a.get("output")))
                continue
            if a.tag == "arg" and name == "mission_regions_file":
                self.assertEqual(b.get("default"),
                                 "$(find purepursuit_mgeo)/config/route_mission_regions_overtake.json")
                continue
            if a.tag == "node" and name == "adaptive_curvature_purepursuit":
                for pa, pb in zip(a, b):
                    if pa.get("name") == "max_speed_kph":
                        self.assertEqual(pb.get("value"), "$(arg controller_max_speed_kph)")
                    else:
                        self.assertEqual(canonical(pa), canonical(pb))
                continue
            self.assertEqual(canonical(a), canonical(b), (a.tag, name))

    def test_planner_keeps_the_integration_wiring(self):
        old = next(n for n in self.source.iter("node") if n.get("name") == "highway_lane_strategy")
        new = next(n for n in self.output.iter("node") if n.get("name") == "highway_lane_strategy")
        old_params = {p.get("name"): p.get("value") for p in old.findall("param")}
        new_params = {p.get("name"): p.get("value") for p in new.findall("param")}
        for name in ("path_file", "base_trajectory_topic", "odom_topic", "obstacle_topic",
                     "lane_info_topic", "highway_request_topic", "route_gate_required",
                     "route_gate_topic", "handoff_topic", "cruise_speed_mps"):
            self.assertEqual(new_params[name], old_params[name], name)
        controller = next(n for n in self.output.iter("node")
                          if n.get("name") == "adaptive_curvature_purepursuit")
        trajectory = next(p.get("value") for p in controller.findall("param")
                          if p.get("name") == "trajectory_topic")
        self.assertEqual(trajectory, "/highway_lane_strategy/trajectory")


class RegionsTest(unittest.TestCase):
    def test_overtake_regions_only_add_the_handoff(self):
        original = json.loads((PACKAGE / "config/route_mission_regions.json").read_text(encoding="utf-8"))
        overtake = json.loads((PACKAGE / "config/route_mission_regions_overtake.json").read_text(encoding="utf-8"))
        self.assertTrue(overtake["highway"].pop("handoff_enabled"))
        handoff = overtake["highway"].pop("handoff_s_m")
        original["highway"].pop("handoff_enabled")
        original["highway"].pop("handoff_s_m")
        self.assertEqual(original, overtake)
        length = RoutePolyline.from_path_file(str(ROUTE_FILE)).length   # as route_mission_gate
        try:
            load_mission_regions(PACKAGE / "config/route_mission_regions.json", ROUTE_FILE, length)
        except ValueError as exc:
            self.skipTest("the team's own regions file does not load here: %s" % exc)
        config = load_mission_regions(PACKAGE / "config/route_mission_regions_overtake.json",
                                      ROUTE_FILE, length)
        self.assertEqual(config["highway"]["handoff_s_m"], handoff)


class ControllerCeilingTest(unittest.TestCase):
    """Raising the controller's max_speed_kph cannot change a capped command.

    The controller drives min(curvature profile capped by max_speed_kph,
    trajectory speed).  Outside the zone the trajectory speed is the
    integration cruise = max_speed_kph, so a higher ceiling changes nothing.
    """

    def test_min_with_cruise_is_unchanged(self):
        rng = random.Random(3)
        for _ in range(40):
            x = y = heading = 0.0
            points = []
            for _ in range(rng.randint(20, 200)):
                heading += rng.uniform(-0.15, 0.15)
                x += math.cos(heading)*rng.uniform(0.5, 2.0)
                y += math.sin(heading)*rng.uniform(0.5, 2.0)
                points.append(cp.PathPoint(x, y, 0.0))
            s = cp.cumulative_arc_lengths(points)
            k = cp.curvature_profile(points, 1, 5)
            cruise = rng.uniform(2.0, 16.0)
            lateral, accel, decel = rng.uniform(0.5, 4.0), rng.uniform(0.5, 1.5), rng.uniform(1.0, 2.0)
            low = cp.build_speed_profile(s, k, cruise, lateral, accel, decel, None, cruise)
            high = cp.build_speed_profile(s, k, 22.0, lateral, accel, decel, None, 22.0)
            for a, b in zip(low, high):
                self.assertAlmostEqual(min(a, cruise), min(b, cruise), places=9)


if __name__ == "__main__":
    unittest.main()
