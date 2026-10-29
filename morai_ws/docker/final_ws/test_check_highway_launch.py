#!/usr/bin/env python3
import unittest

import check_highway_launch as check


class CurvatureLaunchValidationTest(unittest.TestCase):
    def base_names(self):
        return sorted(check.CURVATURE_REQUIRED)

    def test_roundabout_nodes_are_required_and_are_the_only_lidar_nodes(self):
        names = self.base_names() + sorted(check.ROUNDABOUT_REQUIRED)
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=True)

    def test_unexpected_lidar_node_is_rejected(self):
        names = self.base_names() + sorted(check.ROUNDABOUT_REQUIRED) + ['/legacy_lidar_avoidance']
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=True)

    def test_missing_roundabout_node_is_rejected(self):
        names = self.base_names() + sorted(check.ROUNDABOUT_REQUIRED - {'/roundabout_merge_gate'})
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=True)

    def test_roundabout_nodes_are_absent_when_disabled(self):
        names = self.base_names()
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=False)

    def test_roundabout_node_left_enabled_is_rejected(self):
        names = self.base_names() + ['/roundabout_merge_gate']
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=False)

    def test_highway_nodes_are_required_when_enabled(self):
        names = (self.base_names() + sorted(check.ROUNDABOUT_REQUIRED) +
                 sorted(check.HIGHWAY_NODES | check.HIGHWAY_BASE_NODES |
                        check.SECONDARY_CONTROL_NODES))
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=True,
                                       highway_enabled=True)
        partial = [n for n in names if n != '/highway_command_mux']
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(partial, '\n'.join(partial), roundabout_enabled=True,
                                           highway_enabled=True)

    def test_highway_keeps_the_lidar_without_the_roundabout(self):
        names = (self.base_names() + sorted(check.ROUNDABOUT_LIDAR_NODES) +
                 sorted(check.HIGHWAY_NODES | check.HIGHWAY_BASE_NODES |
                        check.SECONDARY_CONTROL_NODES))
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=False,
                                       highway_enabled=True)

    def test_highway_node_left_enabled_is_rejected(self):
        names = self.base_names() + ['/highway_lane_strategy']
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=False)

    def test_avoidance_reuses_secondary_controller_and_lidar(self):
        names = (self.base_names() + sorted(check.ROUNDABOUT_LIDAR_NODES) +
                 sorted(check.AVOIDANCE_NODES | check.SECONDARY_CONTROL_NODES))
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=False,
                                       avoidance_enabled=True)

    def test_highway_with_avoidance_does_not_start_parallel_base_node(self):
        names = (self.base_names() + sorted(check.ROUNDABOUT_REQUIRED) +
                 sorted(check.AVOIDANCE_NODES | check.HIGHWAY_NODES |
                        check.SECONDARY_CONTROL_NODES))
        check.validate_curvature_names(names, '\n'.join(names), roundabout_enabled=True,
                                       highway_enabled=True, avoidance_enabled=True)
        with self.assertRaises(AssertionError):
            check.validate_curvature_names(names + ['/highway_base_path'], '\n'.join(names),
                                           roundabout_enabled=True, highway_enabled=True,
                                           avoidance_enabled=True)


if __name__ == '__main__':
    unittest.main()
