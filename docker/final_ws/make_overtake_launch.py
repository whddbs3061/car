#!/usr/bin/env python3
"""Derive the overtaking launch and mission regions from the integration ones.

final_ws_highway_overtake_bringup.launch is final_ws_highway_bringup.launch
with every element copied unchanged except:
  1. highway_lane_strategy runs highway_overtake_node.py (same name, topics,
     gates and pass-through cruise speed);
  2. adaptive_curvature_purepursuit max_speed_kph becomes
     max(max_speed_kph, highway_max_speed_kph); outside the zone the node still
     publishes cruise = max_speed_kph, and the controller takes the minimum;
  3. mission_regions_file defaults to route_mission_regions_overtake.json;
  4. new args highway_cruise_kph, highway_max_speed_kph, overtake_config_file.
route_mission_regions_overtake.json is route_mission_regions.json with the
highway handoff enabled at the planner's zone end (zone_end_xy), so
route_mission_gate keeps the vehicle driving after the highway.

Paths follow this file's workspace, so inside the container the outputs are
derived from the container's own launch and regions.

    python make_overtake_launch.py            # write both files
    python make_overtake_launch.py --check    # exit 1 if a written file is stale
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
LAUNCH_DIR = WS / "src/bringup/morai_bringup/launch"
SOURCE = LAUNCH_DIR / "final_ws_highway_bringup.launch"
OUTPUT = LAUNCH_DIR / "final_ws_highway_overtake_bringup.launch"
PACKAGE = WS / "src/control/purepursuit_mgeo"
REGIONS_SOURCE = PACKAGE / "config/route_mission_regions.json"
REGIONS_OUTPUT = PACKAGE / "config/route_mission_regions_overtake.json"
PLANNER_CONFIG = PACKAGE / "config/highway_overtake.yaml"
ROUTE_FILE = WS / "data/routes/2026_molit_comp_global_path.txt"
HIGHWAY_NODE = "highway_lane_strategy"
CONTROLLER_NODE = "adaptive_curvature_purepursuit"
REGIONS = "$(find purepursuit_mgeo)/config/route_mission_regions_overtake.json"
NEW_ARGS = (
    ("overtake_config_file", "$(find purepursuit_mgeo)/config/highway_overtake.yaml"),
    ("highway_cruise_kph", "54.0"),
    ("highway_max_speed_kph", "79.2"),
    ("controller_max_speed_kph",
     "$(eval max(float(arg('max_speed_kph')), float(arg('highway_max_speed_kph'))))"),
)
# Wiring copied from the replaced node: topics, gates, pass-through speed.
COPIED_PARAMS = (
    "path_file", "map_frame", "base_path_topic", "base_stop_topic", "base_trajectory_topic",
    "odom_topic", "obstacle_topic", "lane_info_topic", "highway_request_topic",
    "route_gate_required", "route_gate_topic", "route_gate_timeout_s",
    "handoff_topic", "handoff_timeout_s", "cruise_speed_mps",
    "vehicle_length_m", "vehicle_width_m", "vehicle_center_from_base_m",
)
# The planner keeps its predictions within these controller limits.
CONTROLLER_LIMITS = (
    ("lateral_accel_limit_mps2", "controller_lateral_accel_limit_mps2"),
    ("max_accel_mps2", "controller_max_accel_mps2"),
    ("max_decel_mps2", "controller_max_decel_mps2"),
)


def _only(items, what):
    if len(items) != 1:
        raise ValueError("expected exactly one %s, found %d" % (what, len(items)))
    return items[0]


def _param(node, name):
    return _only([p for p in node.findall("param") if p.get("name") == name],
                 "param %s in node %s" % (name, node.get("name")))


def _element(tag, tail, **attrib):
    element = ET.Element(tag, attrib)
    element.tail = tail
    return element


def transform(text: str) -> str:
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if root.tag != "launch":
        raise ValueError("not a launch file")
    parents = {child: parent for parent in root.iter() for child in parent}
    args = [a for a in root if a.tag == "arg"]
    names = {a.get("name") for a in args}
    for required in ("max_speed_kph", "mission_regions_file", "path_file"):
        if required not in names:
            raise ValueError("source launch has no arg %s" % required)
    for name, _ in NEW_ARGS:
        if name in names:
            raise ValueError("source launch already declares %s" % name)
    old = _only([n for n in root.iter("node") if n.get("name") == HIGHWAY_NODE],
                "node %s" % HIGHWAY_NODE)
    controller = _only([n for n in root.iter("node") if n.get("name") == CONTROLLER_NODE],
                       "node %s" % CONTROLLER_NODE)

    # 3. regions with the handoff at the zone end
    _only([a for a in args if a.get("name") == "mission_regions_file"],
          "arg mission_regions_file").set("default", REGIONS)
    # 4. new args after the last declared one (defaults may use any earlier arg)
    position = list(root).index(args[-1]) + 1
    for offset, (name, default) in enumerate(NEW_ARGS):
        root.insert(position + offset, _element("arg", args[-1].tail, name=name, default=default))
    # 2. controller speed ceiling
    _param(controller, "max_speed_kph").set("value", "$(arg controller_max_speed_kph)")
    # 1. planner node in place of the old one
    node = ET.Element("node", dict(old.attrib))
    node.set("type", "highway_overtake_node.py")
    indent = (old.text or "\n    ")
    node.text = indent
    node.tail = old.tail
    children = [_element("rosparam", indent, command="load", file="$(arg overtake_config_file)")]
    for name in COPIED_PARAMS:
        found = [p for p in old.findall("param") if p.get("name") == name]
        if found:
            children.append(_element("param", indent, name=name, value=found[0].get("value")))
    for source_name, name in CONTROLLER_LIMITS:
        children.append(_element("param", indent, name=name,
                                 value=_param(controller, source_name).get("value")))
    children += [
        _element("param", indent, name="mission_regions_file", value="$(arg mission_regions_file)"),
        _element("param", indent, name="highway_cruise_speed_mps",
                 value="$(eval float(arg('highway_cruise_kph')) / 3.6)"),
        _element("param", indent, name="highway_max_speed_mps",
                 value="$(eval float(arg('highway_max_speed_kph')) / 3.6)"),
    ]
    children[-1].tail = (old[-1].tail if len(old) else "\n  ")
    node.extend(children)
    parent = parents[old]
    parent.insert(list(parent).index(old), node)
    parent.remove(old)

    header = ET.Comment(
        " GENERATED by docker/final_ws/make_overtake_launch.py from final_ws_highway_bringup.launch;"
        " do not edit, regenerate.  Only highway_lane_strategy (highway_overtake_node.py),"
        " the controller max_speed_kph and the mission_regions_file default differ. ")
    header.tail = root.text
    root.insert(0, header)
    return ET.tostring(root, encoding="unicode") + "\n"


def zone_end_s(route_file: Path = ROUTE_FILE, config_file: Path = PLANNER_CONFIG) -> float:
    """Route arc length of the planner's zone_end_xy, as the ROS node resolves it."""
    import yaml
    sys.path.insert(0, str(PACKAGE / "src"))
    from purepursuit_mgeo.highway_overtake import Route
    from purepursuit_mgeo.path import load_mgeo_path
    x, y = yaml.safe_load(config_file.read_text(encoding="utf-8"))["zone_end_xy"]
    route = Route([(p.x, p.y) for p in load_mgeo_path(str(route_file))])
    return route.s_of_xy(float(x), float(y))


def derive_regions(text: str, handoff_s: float) -> str:
    config = json.loads(text)
    config["highway"]["handoff_enabled"] = True
    config["highway"]["handoff_s_m"] = round(handoff_s, 3)
    return json.dumps(config, indent=2) + "\n"


def main(argv) -> int:
    if [a for a in argv if a != "--check"]:
        print(__doc__, file=sys.stderr)
        return 2
    outputs = (
        (OUTPUT, transform(SOURCE.read_text(encoding="utf-8")), SOURCE),
        (REGIONS_OUTPUT, derive_regions(REGIONS_SOURCE.read_text(encoding="utf-8"), zone_end_s()),
         REGIONS_SOURCE),
    )
    stale = 0
    for output, text, source in outputs:
        if "--check" in argv:
            current = output.is_file() and output.read_text(encoding="utf-8") == text
            stale += not current
            print("%s: %s" % ("CURRENT" if current else "STALE (run make_overtake_launch.py)", output))
        else:
            with open(output, "w", encoding="utf-8", newline="\n") as stream:   # LF on every OS
                stream.write(text)
            print("wrote %s from %s" % (output, source))
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
