#!/usr/bin/env python3
"""Resolve complete launch includes with real ROS; no sensors/control started."""
import os
from pathlib import Path
import subprocess


CURVATURE_REQUIRED = {
    '/curvature_speed_purepursuit',
    '/curvature_signal_controller',
    '/highway_lane_camera',
    '/lane_info_contract',
    '/yolo_camera',
    '/traffic_light_stop',
    '/camera_debug_dashboard',
}
CURVATURE_EXCLUDED = {
    '/control_mux',
    '/final_ws_stopline_controller',
    '/avoidance_path_manager',
    '/bypass_lane_guard',
    '/avoidance_frenet_debug',
    '/highway_lane_strategy',
    '/roi_sensor_safety_adapter',
    '/intersection_environment',
    '/pedestrian_crossing_fusion',
    '/highway_environment_gate',
}
ROUNDABOUT_REQUIRED = {
    '/morai_lidar_udp_source',
    '/roundabout_lidar_tracking',
    '/roundabout_merge_gate',
}
ROUNDABOUT_LIDAR_NODES = {
    '/morai_lidar_udp_source',
    '/roundabout_lidar_tracking',
}
# Escape4 and ZIP avoidance remain separate situations/controllers.
HIGHWAY_NODES = {
    '/highway_base_path',
    '/highway_lane_strategy',
    '/highway_curvature_purepursuit',
}
AVOIDANCE_NODES = {
    '/avoidance_frenet_debug',
    '/bypass_lane_guard',
    '/avoidance_path_manager',
    '/avoidance_curvature_purepursuit',
}
MUX_NODES = {'/highway_command_mux'}


def validate_curvature_names(names, output, roundabout_enabled, highway_enabled=False,
                             avoidance_enabled=False):
    assert len(names) == len(set(names)), output
    for required in CURVATURE_REQUIRED:
        assert required in names, (required, output)
    optional = set()
    if highway_enabled:
        optional |= HIGHWAY_NODES
    if avoidance_enabled:
        optional |= AVOIDANCE_NODES
    if highway_enabled or avoidance_enabled:
        optional |= MUX_NODES
    for excluded in CURVATURE_EXCLUDED - optional:
        assert excluded not in names, (excluded, output)
    expected_optional = optional
    known_optional = HIGHWAY_NODES | AVOIDANCE_NODES | MUX_NODES
    present_optional = known_optional.intersection(names)
    assert present_optional == expected_optional, (
        'optional nodes mismatch', 'missing', expected_optional - present_optional,
        'unexpected', present_optional - expected_optional, output)

    present_roundabout = ROUNDABOUT_REQUIRED.intersection(names)
    if roundabout_enabled:
        assert present_roundabout == ROUNDABOUT_REQUIRED, (
            'roundabout nodes missing', ROUNDABOUT_REQUIRED - present_roundabout, output)
        expected_lidar = ROUNDABOUT_LIDAR_NODES
    else:
        # The highway keeps the roundabout's LiDAR nodes, never its gate.
        expected_lidar = (ROUNDABOUT_LIDAR_NODES
                          if highway_enabled or avoidance_enabled else set())
        left = present_roundabout - expected_lidar
        assert not left, ('roundabout nodes remained disabled', left, output)

    present_lidar = {name for name in names if 'lidar' in name.lower()}
    assert present_lidar == expected_lidar, (
        'unexpected curvature-signal LiDAR nodes', present_lidar - expected_lidar,
        'missing required LiDAR nodes', expected_lidar - present_lidar, output)


def main():
    ws=Path(__file__).resolve().parents[2]
    env=dict(os.environ, MORAI_WS=str(ws))
    for control,yolo in (('false','true'),('true','true'),('false','false')):
        output=subprocess.check_output(['roslaunch','--nodes','morai_bringup',
            'final_ws_highway_bringup.launch','workspace_path:='+str(ws),
            'enable_control:='+control,'enable_yolo:='+yolo],env=env,text=True)
        names=output.splitlines()
        assert len(names)==len(set(names)), output
        for required in ('/adaptive_curvature_purepursuit','/control_mux',
                         '/avoidance_path_manager','/highway_lane_strategy'):
            assert required in names, (required,output)
        # The highway launch uses the ROI camera stack, whose lane runner is
        # named lane_camera (sometimes under an inherited namespace). The
        # dedicated highway_lane_camera node belongs to curvature_signal.
        assert any(name.endswith('/lane_camera') for name in names), ('ROI lane camera missing',output)
        assert ('/yolo_camera' in names) == (yolo == 'true'), (
            'ROI YOLO camera configuration mismatch', yolo, output)
        assert '/purepursuit_mgeo' not in names and '/curvature_speed_purepursuit' not in names
    print('HIGHWAY_LAUNCH_OK: ROS resolved all includes; control was not started')
    for control in ('false', 'true'):
        output=subprocess.check_output(['roslaunch','--nodes','morai_bringup',
            'final_ws_curvature_signal.launch','workspace_path:='+str(ws),
            'enable_control:='+control],env=env,text=True)
        names=output.splitlines()
        validate_curvature_names(names, output, roundabout_enabled=True, highway_enabled=True,
                                 avoidance_enabled=True)

    for roundabout, highway, avoidance in (
            ('false', 'true', 'true'), ('true', 'false', 'true'),
            ('true', 'true', 'false'), ('false', 'false', 'false')):
        output=subprocess.check_output(['roslaunch','--nodes','morai_bringup',
            'final_ws_curvature_signal.launch','workspace_path:='+str(ws),
            'enable_control:=false','enable_roundabout_mission:='+roundabout,
            'enable_highway:='+highway,
            'enable_obstacle_avoidance:='+avoidance],env=env,text=True)
        validate_curvature_names(output.splitlines(), output, roundabout_enabled=roundabout == 'true',
                                 highway_enabled=highway == 'true',
                                 avoidance_enabled=avoidance == 'true')
    print('CURVATURE_SIGNAL_LAUNCH_OK: ROS resolved monitor/drive with and without the highway; '
          'control was not started')


if __name__=='__main__': main()

