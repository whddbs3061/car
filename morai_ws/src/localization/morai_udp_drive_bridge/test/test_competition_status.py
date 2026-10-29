"""Wire fixtures are written by offsets, independently of the bridge parser.

These synthetic packets validate the adopted team schema, not a live MORAI
or organiser-forwarder capture. Forbidden slots deliberately contain NaNs.
"""
import json
import math
import struct
import unittest

from morai_udp_drive_bridge.protocol import ProtocolError, parse_competition_vehicle_status
import test_transport_safety as transport


def packet(size=181):
    raw = bytearray(size)
    raw[:11] = b'#MoraiInfo$'
    struct.pack_into('<I', raw, 11, size-29)
    struct.pack_into('<II', raw, 27, 123, 250_000_000)
    struct.pack_into('<bb', raw, 35, 2, 4)
    values = {37:36.,45:.2,49:0.,53:4.635,57:1.892,61:2.434,
              65:.9,69:3.,73:.735,89:1.,93:2.,97:90.,101:36.,
              113:.1,117:.2,121:.3,137:-5.}
    for offset, value in values.items():
        struct.pack_into('<f', raw, offset, value)
    for offset in (77,81,85,105,109,125,129,133):
        struct.pack_into('<f', raw, offset, float('nan'))
    if size == 229:
        struct.pack_into('<12f', raw, 179, *([float('nan')]*12))
    raw[-2:] = b'\r\n'
    return bytes(raw)


class CompetitionParserTests(unittest.TestCase):
    def test_both_layouts_extract_only_available_fields(self):
        for size in (181,229):
            value = parse_competition_vehicle_status(packet(size))
            self.assertEqual((value.sec,value.nsec,value.ctrl_mode,value.gear),
                             (123,250_000_000,2,4))
            self.assertEqual(value.velocity_x_kmh,36.)
            self.assertEqual(value.heading_deg,90.)
            self.assertEqual(value.steer_deg,-5.)
            self.assertEqual(value.packet_length,size)
            for forbidden in ('pos_x_m','velocity_y_kmh','acceleration_x_mps2','link_id'):
                self.assertFalse(hasattr(value,forbidden))

    def test_envelope_validation(self):
        for size in (181,229):
            base=packet(size)
            bad_header=bytearray(base);bad_header[0]=0
            bad_length=bytearray(base);struct.pack_into('<I',bad_length,11,102)
            bad_nsec=bytearray(base);struct.pack_into('<I',bad_nsec,31,1_000_000_000)
            bad_speed=bytearray(base);struct.pack_into('<f',bad_speed,101,float('nan'))
            for bad in (bad_header,bad_length,bad_nsec,bad_speed,base[:-1],
                        base+b'padding',base[:-2]+b'xx'):
                with self.assertRaises(ProtocolError):
                    parse_competition_vehicle_status(bytes(bad))

    def test_extended_packet_cannot_be_truncated_into_a_base_packet(self):
        with self.assertRaises(ProtocolError):
            parse_competition_vehicle_status(packet(229)[:181])


class CompetitionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.harness=transport.TransportSafetyTest()
        self.harness.setUp()
        self.node=self.harness.bridge(False)

    def test_default_uses_competition_9088_to_9099(self):
        self.assertEqual((self.node.status_protocol,self.node.status_source_port,
                          self.node.status_port),('competition',9088,9099))

    def test_ros_projection_marks_unavailable_fields_and_converts_once(self):
        for size in (181,229):
            self.node.publish_status(parse_competition_vehicle_status(packet(size)))
            msg=self.node.status_pub.publish.call_args.args[0]
            self.assertEqual(msg.velocity.x,10.)
            self.assertEqual(msg.heading,90.)
            self.assertEqual(msg.wheel_angle,-5.)
            self.assertTrue(all(math.isnan(v) for v in (
                msg.velocity.y,msg.velocity.z,msg.position.x,msg.position.y,msg.position.z,
                msg.acceleration.x,msg.acceleration.y,msg.acceleration.z)))
            metadata=json.loads(self.node.metadata_pub.publish.call_args.args[0].data)
            self.assertEqual(metadata['packet_length'],size)
            self.assertEqual(metadata['velocity_x_mps'],10.)
            self.assertEqual(metadata['expected_source_port'],9088)
            self.assertFalse(metadata['position_available'])
            self.assertFalse(metadata['velocity_yz_available'])
            self.assertEqual((metadata['ctrl_mode'],metadata['gear']),(2,4))

    def test_receive_reads_complete_datagram_and_rejects_oversized_input(self):
        receive=self.harness.make_socket()
        self.harness.socket_factory.side_effect=None
        self.harness.socket_factory.return_value=receive
        second=bytearray(packet(229));struct.pack_into("<I",second,31,250_000_001)
        receive.recvfrom.side_effect=[
            (packet(181),('192.0.2.1',9087)),
            (packet()+b'garbage',('192.0.2.1',9088)),
            (packet(181),('192.0.2.1',9088)),(packet(181),('192.0.2.1',9088)),
            (bytes(second),('192.0.2.1',9088))]
        values=[]
        def published(msg):
            values.append(msg)
            if len(values)==2:self.node.stop_event.set()
        self.node.status_pub.publish.side_effect=published
        self.node.status_receive_loop()
        self.assertEqual(len(values),2)
        self.assertEqual(receive.recvfrom.call_count,5)
        for call in receive.recvfrom.call_args_list:self.assertEqual(call.args,(65535,))
        receive.bind.assert_called_once_with(('0.0.0.0',9099))
        self.harness.assert_no_sends()
