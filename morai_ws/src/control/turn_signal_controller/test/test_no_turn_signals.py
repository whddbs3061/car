"""Simulator opt-out removes only indicator requirements, using real callbacks."""
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import test_maneuver_fusion as fixtures
from test_route_contract import reference_message


class NoTurnSignalsTest(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.StrictFusionTest()
        self.case.setUp()
        self.c = self.case.fixture
        previous = self.case.node
        self.c.params["~test_without_turn_signals"] = True
        # Even a conflicting lamp_output_enabled=true must not create a socket.
        self.factory = Mock(return_value=self.c.socket)
        self.c.module.socket.socket = self.factory
        self.node = self.c.module.ManeuverFusionNode()
        self.c.node = self.case.node = self.node
        self.node.require_context = self.node.require_camera_stream = True
        self.node.require_reference_path = True
        self.node.camera = previous.camera
        self.node.contexts, self.node.map_heads = previous.contexts, previous.map_heads
        self.node.reference_path_callback(reference_message(self.node.points))

    def frames(self, count=20, **kwargs):
        for _ in range(count):
            result = self.case.frame(**kwargs)
        return result

    def test_no_socket_or_packet_including_shutdown_and_no_fake_lamp_success(self):
        _, status = self.frames()
        self.node.shutdown()
        self.factory.assert_not_called()
        self.c.socket.sendto.assert_not_called()
        self.assertIsNone(self.node.socket)
        self.assertFalse(status["lamp_udp_enabled"])
        self.assertFalse(status["lamp_transmit_ok"])
        self.assertTrue(status["test_without_turn_signals"])
        self.assertFalse(status["indicator_lead_required"])
        self.assertEqual(self.node.lamp_pub.publish.call_args.args[0].data, "OFF")

    def test_route_signal_matrix_without_five_second_lamp_delay(self):
        allowed = {"GREEN": {"STRAIGHT", "RIGHT"},
                   "GREEN_LEFT": {"STRAIGHT", "LEFT", "RIGHT"},
                   "LEFT": {"LEFT"}, "RIGHT": {"RIGHT"},
                   "RED": set(), "YELLOW": set(), "UNKNOWN": set()}
        for direction in ("STRAIGHT", "LEFT", "RIGHT"):
            for signal, directions in allowed.items():
                with self.subTest(route=direction, signal=signal):
                    self.setUp()
                    self.case.context["direction"] = direction
                    output, status = self.frames(state=signal)
                    self.assertLess(self.c.now - 100., 5.)
                    self.assertEqual(status["route_direction"], direction)
                    self.assertEqual(status["permission"], direction in directions)
                    self.assertEqual(output.brake, 0. if direction in directions else 1.)
                    self.assertEqual(output.steering, fixtures.Command().steering)

    def test_green_still_needs_temporal_confirmation_and_red_revokes_it(self):
        output, status = self.case.frame()
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)
        _, status = self.frames()
        self.assertTrue(status["permission"])
        output, status = self.case.frame("RED")
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)

    def test_duplicate_image_cannot_authorize_entry(self):
        output, status = self.frames(objects_stamp=100.)
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)

    def test_missing_map_event_keeps_curvature_route_active(self):
        self.node.contexts = []
        output, status = self.frames(state="GREEN")
        self.assertEqual(status["route_direction"], "UNKNOWN")
        self.assertFalse(status["indicator_ready"])
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 0.)

    def test_reference_mismatch_still_stops(self):
        self.frames()
        wrong = reference_message(self.node.points)
        wrong.poses[-1].pose.position.y += 10.
        self.node.reference_path_callback(wrong)
        output, status = self.case.frame()
        self.assertFalse(status["reference_path_match"])
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)

    def test_camera_calibration_and_localization_and_safety_still_gate_motion(self):
        for fault in ("uncalibrated", "blackout", "obstacle", "stale"):
            with self.subTest(fault=fault):
                self.setUp()
                self.frames()
                if fault == "uncalibrated":
                    self.node.camera["calibrated"] = False
                    output, _ = self.case.frame()
                elif fault == "blackout":
                    output, _ = self.case.frame(quality="GPS_BLACKOUT")
                elif fault == "obstacle":
                    output, _ = self.case.frame(obstacle=True)
                else:
                    self.c.now += 1.
                    output, _ = self.c.tick(refresh=False)
                self.assertEqual((output.accel, output.brake), (0., 1.))

    def test_unverified_map_stopline_still_requires_camera_stopline(self):
        self.case.context["stop_s"] = None
        output, status = self.frames()
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)
        _, status = self.frames(line=True)
        self.assertTrue(status["permission"])

    def test_default_remains_indicator_gated_and_lamp_false_alone_is_not_opt_out(self):
        self.c.params.pop("~test_without_turn_signals")
        self.c.params["~lamp_output_enabled"] = False
        self.node = self.c.module.ManeuverFusionNode()
        self.c.node = self.case.node = self.node
        output, status = self.c.run_ticks(120)
        self.assertFalse(status["test_without_turn_signals"])
        self.assertTrue(status["indicator_lead_required"])
        self.assertFalse(status["indicator_ready"])
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)


class NoTurnSignalsLaunchTest(unittest.TestCase):
    def test_opt_out_reaches_fusion_and_both_lamp_writers_are_disabled(self):
        directory = fixtures.SOURCE / "bringup/morai_bringup/launch"
        native = ET.parse(directory / "final_ws_native_no_lamps.launch").getroot()
        final = ET.parse(directory / "final_ws_bringup.launch").getroot()
        perception = ET.parse(directory / "perception_control_bringup.launch").getroot()
        self.assertEqual(native.find("arg[@name='ego_status_port']").get("default"), "9099")
        self.assertEqual(native.find("arg[@name='enable_control']").get("default"), "false")
        self.assertEqual(native.find("include/arg[@name='test_without_turn_signals']").get("value"), "true")
        self.assertEqual(native.find("include/arg[@name='enable_turn_signal']").get("value"), "false")
        for root in (final, perception):
            self.assertEqual(root.find("arg[@name='test_without_turn_signals']").get("default"), "false")
            self.assertEqual(root.find("include/arg[@name='ego_status_port']").get("value"), "$(arg ego_status_port)")
        self.assertEqual(final.find("include/arg[@name='test_without_turn_signals']").get("value"), "$(arg test_without_turn_signals)")
        fusion = perception.find("node[@type='maneuver_fusion_node.py']")
        self.assertEqual(fusion.find("param[@name='test_without_turn_signals']").get("value"), "$(arg test_without_turn_signals)")
        self.assertIn("not arg('test_without_turn_signals')", fusion.find("param[@name='lamp_output_enabled']").get("value"))
        self.assertIn("not arg('test_without_turn_signals')", perception.find("node[@type='turn_signal_node.py']").get("if"))
