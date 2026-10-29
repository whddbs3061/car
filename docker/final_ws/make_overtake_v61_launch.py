#!/usr/bin/env python3
"""Derive the overtaking launch from the team's v61 obstacle integration launch.

integrated_obstacle_v61_overtake.launch is integrated_obstacle_v61.launch with
every element copied unchanged except three nodes, each kept under its name:
  1. highway_lane_strategy runs highway_overtake_node.py instead of the team's
     highway strategy (v61_highway_with_avoidance_node.py): same input and
     output topics, pass-through of the avoidance path outside its zone;
  2. curvature_speed_purepursuit runs curvature_pp_overtake_node.py, the team's
     controller loaded unchanged that uses highway_max_speed_kph and
     highway_lateral_accel_mps2 only for paths received inside the zone;
  3. curvature_signal_controller runs maneuver_fusion_overtake_node.py, the
     team's fusion loaded unchanged whose speed governor caps at
     highway_max_speed_kph only inside the zone;
and, per the competition network rules, the vehicle status UDP port: the
drive bridge listens on 9099 (Competition Vehicle Status, sent from 9088;
its 181-byte layout is the one the bridge parses) instead of 1911 (Ego
Vehicle Status, not allowed); control stays 9094 -> 9093, sensors unchanged.
New args highway_cruise_kph, highway_max_speed_kph,
highway_lateral_accel_mps2, overtake_config_file.  The zone flag is
/highway_lane_strategy/zone_active; without it the team's behaviour applies.

Paths follow this file's workspace, so inside the container the output is
derived from the container's own launch.

    python make_overtake_v61_launch.py            # write the launch
    python make_overtake_v61_launch.py --check    # exit 1 if it is stale
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_overtake_launch import _element, _only, _param  # noqa: E402

WS = Path(__file__).resolve().parents[2]
LAUNCH_DIR = WS / "src/bringup/morai_bringup/launch"
SOURCE = LAUNCH_DIR / "integrated_obstacle_v61.launch"
OUTPUT = LAUNCH_DIR / "integrated_obstacle_v61_overtake.launch"
HIGHWAY_NODE = "highway_lane_strategy"
CONTROLLER_NODE = "curvature_speed_purepursuit"
FUSION_NODE = "curvature_signal_controller"
# Competition network: status Sim 9088 -> User 9099, Ctrl Cmd User 9094 -> Sim 9093.
STATUS_PORT = "9099"
CONTROL_PORTS = {"control_remote_port": "9093", "control_source_port": "9094"}
NEW_ARGS = (
    ("overtake_config_file", "$(find purepursuit_mgeo)/config/highway_overtake.yaml"),
    ("highway_cruise_kph", "54.0"),
    ("highway_max_speed_kph", "79.2"),
    ("highway_lateral_accel_mps2", "2.0"),
)
# Wiring copied from the replaced node.  Its planner tuning is not copied:
# several names coincide with the overtaking planner's own fields.
COPIED_PARAMS = (
    "path_file", "map_frame", "odom_topic", "obstacle_topic", "lane_info_topic",
    "highway_request_topic", "cruise_speed_mps", "base_path_topic", "base_stop_topic",
)
WRAPPERS = {
    CONTROLLER_NODE: ("curvature_pp_overtake_node.py", (
        ("highway_max_speed_kph", "$(arg highway_max_speed_kph)"),
        ("highway_lateral_accel_mps2", "$(arg highway_lateral_accel_mps2)"),
    )),
    FUSION_NODE: ("maneuver_fusion_overtake_node.py", (
        ("highway_max_speed_kph", "$(arg highway_max_speed_kph)"),
    )),
}


def _node(root, name):
    return _only([n for n in root.iter("node") if n.get("name") == name], "node %s" % name)


def _append(node, params):
    indent = node.text or "\n    "
    last_tail = node[-1].tail if len(node) else "\n  "
    if len(node):
        node[-1].tail = indent
    for k, (name, value) in enumerate(params):
        node.append(_element("param", indent if k < len(params) - 1 else last_tail,
                             name=name, value=value))


def transform(text: str) -> str:
    root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if root.tag != "launch":
        raise ValueError("not a launch file")
    parents = {child: parent for parent in root.iter() for child in parent}
    args = [a for a in root if a.tag == "arg"]
    names = {a.get("name") for a in args}
    for required in ("max_speed_kph", "path_file"):
        if required not in names:
            raise ValueError("source launch has no arg %s" % required)
    for name, _ in NEW_ARGS:
        if name in names:
            raise ValueError("source launch already declares %s" % name)
    status = _only([a for a in args if a.get("name") == "ego_status_port"], "arg ego_status_port")
    status.set("default", STATUS_PORT)
    for name, port in CONTROL_PORTS.items():
        arg = _only([a for a in args if a.get("name") == name], "arg " + name)
        if arg.get("default") != port:
            raise ValueError("%s is %s, the competition uses %s" % (name, arg.get("default"), port))
    old = _node(root, HIGHWAY_NODE)
    if old.get("type") != "v61_highway_with_avoidance_node.py":
        raise ValueError("unexpected highway node type %s" % old.get("type"))
    controller = _node(root, CONTROLLER_NODE)
    if controller.get("type") != "adaptive_curvature_purepursuit_node.py":
        raise ValueError("unexpected controller type %s" % controller.get("type"))
    fusion = _node(root, FUSION_NODE)
    if fusion.get("type") != "maneuver_fusion_node.py":
        raise ValueError("unexpected fusion type %s" % fusion.get("type"))

    # New args after the leading arg block: roslaunch resolves $(arg) in order.
    children = list(root)
    first_user = next(i for i, e in enumerate(children) if e.tag not in ("arg",) and
                      not callable(e.tag))
    leading = [a for a in args if children.index(a) < first_user]
    position = children.index(leading[-1]) + 1
    for offset, (name, default) in enumerate(NEW_ARGS):
        root.insert(position + offset, _element("arg", leading[-1].tail, name=name, default=default))

    # 2, 3: the team's controller and fusion through their zone-only wrappers
    for name, (script, params) in WRAPPERS.items():
        node = _node(root, name)
        node.set("pkg", "purepursuit_mgeo")
        node.set("type", script)
        _append(node, params)

    # 1: the overtaking planner in place of the team's highway strategy
    node = ET.Element("node", dict(old.attrib))
    node.set("type", "highway_overtake_node.py")
    indent = old.text or "\n    "
    node.text = indent
    node.tail = old.tail
    children = [_element("rosparam", indent, command="load", file="$(arg overtake_config_file)")]
    for name in COPIED_PARAMS:
        found = [p for p in old.findall("param") if p.get("name") == name]
        if found:
            children.append(_element("param", indent, name=name, value=found[0].get("value")))
    children += [
        _element("param", indent, name="base_trajectory_topic", value=""),
        _element("param", indent, name="route_gate_required", value="false"),
        _element("param", indent, name="controller_lateral_accel_limit_mps2",
                 value="$(arg highway_lateral_accel_mps2)"),
        _element("param", indent, name="controller_max_accel_mps2",
                 value=_param(controller, "max_accel_mps2").get("value")),
        _element("param", indent, name="controller_max_decel_mps2",
                 value=_param(controller, "max_decel_mps2").get("value")),
        _element("param", indent, name="highway_cruise_speed_mps",
                 value="$(eval float(arg('highway_cruise_kph')) / 3.6)"),
        _element("param", indent, name="highway_max_speed_mps",
                 value="$(eval float(arg('highway_max_speed_kph')) / 3.6)"),
    ]
    children[-1].tail = old[-1].tail if len(old) else "\n  "
    node.extend(children)
    parent = parents[old]
    parent.insert(list(parent).index(old), node)
    parent.remove(old)

    header = ET.Comment(
        " GENERATED by docker/final_ws/make_overtake_v61_launch.py from integrated_obstacle_v61.launch;"
        " do not edit, regenerate.  Only highway_lane_strategy (highway_overtake_node.py) and the"
        " zone-only wrappers of curvature_speed_purepursuit and curvature_signal_controller differ,"
        " and ego_status_port is 9099 (Competition Vehicle Status). ")
    header.tail = root.text
    root.insert(0, header)
    return ET.tostring(root, encoding="unicode") + "\n"


def main(argv) -> int:
    if [a for a in argv if a != "--check"]:
        print(__doc__, file=sys.stderr)
        return 2
    text = transform(SOURCE.read_text(encoding="utf-8"))
    if "--check" in argv:
        current = OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == text
        print("%s: %s" % ("CURRENT" if current else "STALE (run make_overtake_v61_launch.py)", OUTPUT))
        return 0 if current else 1
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as stream:   # LF on every OS
        stream.write(text)
    print("wrote %s from %s" % (OUTPUT, SOURCE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
