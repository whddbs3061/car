"""Offline arbitration and launch-contract regression checks (no ROS runtime)."""
import ast
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'src/bringup/morai_bringup/launch'
source = ROOT / 'src/control/purepursuit_mgeo/scripts/no_obstacle_integration_node.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'choose')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
choose = ns['choose']

class MissionsTests(unittest.TestCase):
    def test_highway_selects_distinct_controller(self):
        self.assertEqual(choose(10, ('normal', 10), ({'active': False}, 10), None,
                                highway=('highway', 10), highway_required=True),
                         ('highway', 'highway'))

    def test_stale_highway_never_falls_back_to_wrong_lane(self):
        for sample in (None, ('highway', 9), ('highway', 11)):
            self.assertEqual(choose(10, ('normal', 10), ({'active': False}, 10), None,
                                    highway=sample, highway_required=True),
                             (None, 'highway_command_stale'))

    def test_roundabout_priority(self):
        driver = {'active': True, 'driver_input_fresh': True}
        self.assertEqual(choose(10, ('normal', 10), ({'active': True}, 10), (driver, 10),
                                highway=('highway', 10), highway_required=True),
                         (driver, 'roundabout_v61'))

    def test_highway_exit_returns_to_normal(self):
        self.assertEqual(choose(10, ('normal', 10), ({'active': False}, 10), None,
                                highway=('highway', 10), highway_required=False)[0], 'normal')

    def test_launch_include_arguments(self):
        packages = {}
        for p in (ROOT / 'src').rglob('package.xml'):
            packages[ET.parse(p).getroot().findtext('name')] = p.parent
        seen = set()
        def visit(p):
            if p in seen: return
            seen.add(p)
            root = ET.parse(p).getroot()
            declared = {a.get('name') for a in root.findall('arg')}
            for node in root.iter():
                for value in node.attrib.values():
                    refs = re.findall(r'\$\(arg ([^)]+)\)', value)
                    refs += re.findall(r"arg\('([^']+)'\)", value)
                    self.assertFalse(set(refs)-declared, (p, refs, declared))
            for inc in root.iter('include'):
                m = re.fullmatch(r'\$\(find ([^)]+)\)/(.*)', inc.get('file'))
                if not m: continue
                target = packages[m[1]] / m[2]
                accepted = {a.get('name') for a in ET.parse(target).getroot().findall('arg')}
                supplied = {a.get('name') for a in inc.findall('arg')}
                self.assertFalse(supplied-accepted, (target, supplied-accepted))
                visit(target)
        visit(LAUNCH / 'final_ws_missions_v61.launch')

    def test_dry_run_and_no_duplicate_sensor_receivers(self):
        root = ET.parse(LAUNCH / 'final_ws_missions_v61.launch').getroot()
        self.assertEqual(root.find("arg[@name='enable_control']").get('default'), 'false')
        self.assertEqual(root.find("include/arg[@name='publish_nominal_command']").get('value'), 'true')
        self.assertFalse([n for n in root.findall('node') if n.get('pkg') in ('camera_perception', 'lidar_perception')])
        node = root.find("node[@name='highway_curvature_purepursuit']")
        self.assertEqual(node.find("param[@name='command_topic']").get('value'), '/integration/highway_cmd')
        self.assertIsNotNone(node.find("remap[@from='/experimental/curvature_reference_path']"))
        bridge = ET.parse(LAUNCH / 'morai_udp_ekf_purepursuit.launch').getroot()
        # Publish previews while retaining the independent UDP enable_control gate.
        self.assertEqual(bridge.find("include/arg[@name='control_output_enabled']").get('value'), '$(arg enable_control)')

if __name__ == '__main__': unittest.main()
