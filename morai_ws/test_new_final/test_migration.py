"""Packet-to-controller and real GPS/IMU EKF tests with ROS I/O replaced."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
import numpy as np

WS=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WS/'src/localization/morai_udp_drive_bridge/test'),
              str(WS/'src/experimental/curvature_speed_purepursuit/test')]
import test_transport_safety as bridge_fixture
import test_competition_status as wire_fixture
import test_node_startup as control_fixture


class CompetitionInputTests(unittest.TestCase):
    def test_general_status_mode_cannot_be_enabled(self):
        for name in ('ego_legacy','ego','auto'):
            fixture=bridge_fixture.TransportSafetyTest();fixture.setUp()
            fixture.parameters['~status_protocol']=name
            with self.assertRaises(ValueError):fixture.bridge(False)
            fixture.assert_no_sends()

    def test_short_status_port_is_rejected(self):
        fixture=bridge_fixture.TransportSafetyTest();fixture.setUp()
        fixture.parameters['~status_port']=909
        with self.assertRaises(ValueError):fixture.bridge(False)
        fixture.assert_no_sends()

    def test_real_packet_projection_feeds_speed_controller_with_nan_reserved_fields(self):
        fixture=bridge_fixture.TransportSafetyTest();fixture.setUp();fixture.now=100.
        bridge=fixture.bridge(False)
        from morai_udp_drive_bridge.protocol import parse_competition_vehicle_status
        bridge.publish_status(parse_competition_vehicle_status(wire_fixture.packet(229)))
        msg=bridge.status_pub.publish.call_args.args[0]
        case=control_fixture.StartupTest();case.setUp()
        case.module.rospy.get_time.side_effect=lambda:case.now
        case.params['~require_vehicle_speed']=True
        case.params['~vehicle_speed_topic']='/Ego_topic'
        case.node=case.module.CurvatureSpeedPurePursuitNode()
        case.node.vehicle_speed_callback(msg)
        self.assertEqual(case.node.vehicle_speed_source.sample(case.now,case.now),10.)
        self.assertTrue(math.isnan(msg.position.x))
        self.assertTrue(math.isnan(msg.velocity.y))
        # Stale Competition speed is not reinterpreted as measured zero.
        case.now+=.6
        case.tick()
        command=case.node.command_pub.publish.call_args.args[0]
        self.assertEqual((command.accel,command.brake),(0.,1.))

    def test_every_status_launch_default_uses_competition_port(self):
        found=0
        for path in (WS/'src').rglob('*.launch'):
            root=ET.parse(path).getroot()
            for arg in root.findall('arg'):
                if arg.get('name') in ('ego_status_port','status_port') and arg.get('default'):
                    self.assertEqual(arg.get('default'),'9099',str(path))
                    found+=1
        self.assertGreaterEqual(found,10)

    def test_no_general_status_parser_remains_in_the_bridge(self):
        from morai_udp_drive_bridge import protocol
        self.assertFalse(hasattr(protocol,'parse_ego_vehicle_status'))
        fields=protocol.CompetitionVehicleStatusMeasurement.__dataclass_fields__
        for forbidden in ('pos_x_m','pos_y_m','pos_z_m','velocity_y_kmh','velocity_z_kmh',
                          'acceleration_x_mps2','acceleration_y_mps2','acceleration_z_mps2',
                          'tire_lateral_force','side_slip_angle','tire_cornering_stiffness'):
            self.assertNotIn(forbidden,fields)

    def test_legacy_practice_evaluator_ignores_unavailable_lateral_speed(self):
        deps={'rospy':Mock(), 'morai_msgs.msg':NS(EgoVehicleStatus=object),
              'morai_perception_msgs.msg':NS(SensorQuality=object),
              'std_msgs.msg':NS(Bool=object,Float64=object,String=object)}
        spec=importlib.util.spec_from_file_location('_new_final_legacy_judge',
            WS/'src/mission/mission_evaluator/scripts/mission_evaluator_node.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,deps):spec.loader.exec_module(module)
        judge=module.MissionEvaluator.__new__(module.MissionEvaluator)
        judge.ego_callback(NS(velocity=NS(x=-10.,y=float('nan'),z=float('nan'))))
        self.assertEqual(judge.last_speed_kph,36.)
        judge.ego_callback(NS(velocity=NS(x=float('nan'))))
        self.assertIsNone(judge.last_speed_kph)


class LocalizationSourceTests(unittest.TestCase):
    def setUp(self):
        self.ros=Mock()
        self.ros.get_param.side_effect=lambda key,default=None:default
        self.ros.get_time.return_value=100.
        self.ros.Time.now.return_value=NS(to_sec=lambda:100.)
        deps={'rospy':self.ros,'tf2_ros':Mock(),
              'geometry_msgs.msg':NS(PoseWithCovarianceStamped=object,TransformStamped=object),
              'nav_msgs.msg':NS(Odometry=object),'sensor_msgs.msg':NS(Imu=object),
              'morai_perception_msgs.msg':NS(SensorQuality=object)}
        spec=importlib.util.spec_from_file_location('_new_final_ekf',
            WS/'src/localization/ekf_local_enu/scripts/ekf_local_enu_node.py')
        self.module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,deps):spec.loader.exec_module(self.module)
        self.node=self.module.LocalEnuEkf()
        self.node.publish=Mock()

    @staticmethod
    def imu(t):
        return NS(header=NS(stamp=NS(to_sec=lambda:t)),
                  orientation=NS(x=0.,y=0.,z=0.,w=1.),
                  angular_velocity=NS(z=0.),linear_acceleration=NS(x=0.))

    def test_estimator_subscribes_gps_and_imu_without_ego_status(self):
        topics=[c.args[0] for c in self.ros.Subscriber.call_args_list]
        self.assertEqual(topics,['/localization/gps','/Imu'])

    def test_position_is_initialized_from_gps_and_propagated_with_imu(self):
        self.node.imu_callback(self.imu(100.))
        gps=NS(header=NS(stamp=NS(to_sec=lambda:100.)),
               pose=NS(pose=NS(position=NS(x=123.,y=456.,z=3.)),covariance=[1.]*36),
               twist=NS(twist=NS(linear=NS(x=10.,y=0.))))
        self.node.gps_callback(gps)
        self.assertAlmostEqual(self.node.state[0,0],123.)
        self.assertAlmostEqual(self.node.state[1,0],456.)
        self.node.imu_callback(self.imu(100.1))
        self.assertAlmostEqual(self.node.state[0,0],124.)
        self.assertAlmostEqual(self.node.state[1,0],456.)

    def test_missing_gps_does_not_invent_an_initial_position(self):
        self.node.imu_callback(self.imu(100.))
        self.assertFalse(self.node.initialized)
        self.node.publish.assert_not_called()


class SavedSettingsTests(unittest.TestCase):
    def test_old_saved_1911_cannot_override_the_new_competition_port(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe')
        executable=str(bash) if bash.exists() else shutil.which('bash')
        if not executable:self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scripts=root/'docker/final_ws';scripts.mkdir(parents=True)
            source=WS/'docker/final_ws'
            for name in ('run_test.sh','highway-test.env.example','curvature_runtime_files.sh'):
                shutil.copyfile(source/name,scripts/name)
            (scripts/'highway.env').write_text('UBUNTU_IP=192.168.10.21\nMORAI_IP=192.168.10.14\nCONTAINER_NAME=test_only\n',encoding='utf8')
            saved=scripts/'highway-test.env'
            saved.write_text('EGO_STATUS_PORT=1911\n',encoding='utf8')
            command=[executable,str(scripts/'run_test.sh'),'show','2']
            result=subprocess.run(command,cwd=root,capture_output=True)
            output=(result.stdout+result.stderr).decode('utf8',errors='replace')
            self.assertEqual(result.returncode,0,output)
            self.assertIn('ego_status_port:=9099',output)
            self.assertNotIn('ego_status_port:=1911',output)
            saved.write_text('EGO_STATUS_PORT=1911\nCOMPETITION_STATUS_PORT=9082\n',encoding='utf8')
            result=subprocess.run(command,cwd=root,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(b'ego_status_port:=9082',result.stdout)

    def test_deployment_manifest_covers_new_bridge_and_protocol(self):
        manifest=(WS/'docker/final_ws/curvature_runtime_files.sh').read_text(encoding='utf8')
        self.assertIn('src/localization/morai_udp_drive_bridge',manifest)
        self.assertIn('find "$package/src" "$package/scripts"',manifest)
