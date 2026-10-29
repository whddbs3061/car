#!/usr/bin/env python3
"""Receive-only CollisionData UDP -> timestamped internal JSON snapshots."""

import json
import socket
import threading

import rospy
from std_msgs.msg import String
from mission_evaluator.collision_udp import parse_collision_packet


class CollisionBridge:
    def __init__(self):
        self.publisher = rospy.Publisher(rospy.get_param("~output_topic","/evaluation/collision_snapshot"),String,queue_size=100)
        self.header = rospy.get_param("~packet_header","#CollisionData$").encode("ascii")
        if len(self.header) != 15:
            raise ValueError("Collision header must contain exactly 15 bytes")
        self.verified = bool(rospy.get_param("~protocol_verified",False))
        self.remote = rospy.get_param("~allowed_sender_ip","")
        self.source_port = int(rospy.get_param("~allowed_sender_port",9091))
        if not 0 <= self.source_port <= 65535:
            raise ValueError("CollisionData source port must be 0..65535")
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        # Intentionally no SO_REUSEADDR: never steal/share another receiver's port.
        self.socket.bind((rospy.get_param("~bind_ip","0.0.0.0"),int(rospy.get_param("~port",9092))))
        self.socket.settimeout(0.2)
        self.stop = threading.Event()
        self.last_packet_time = None
        self.sequence = 0
        rospy.on_shutdown(self.close)
        self.thread = threading.Thread(target=self.receive,daemon=True)
        self.thread.start()
        rospy.logwarn("Practice collision receiver only. Verify protocol/empty snapshots in MORAI; verified=%s",self.verified)

    def receive(self):
        while not self.stop.is_set() and not rospy.is_shutdown():
            try:
                packet,sender = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if self.remote and sender[0] != self.remote:
                continue
            if self.source_port and sender[1] != self.source_port:
                rospy.logwarn_throttle(2,"CollisionData source port mismatch: %d != %d",
                                       sender[1],self.source_port)
                continue
            try:
                data = parse_collision_packet(packet,self.header)
                stamp = data["packet_timestamp_sec"]
                if self.last_packet_time is not None and stamp <= self.last_packet_time:
                    rospy.logwarn_throttle(2,"Duplicate/backward CollisionData timestamp rejected; reset receiver after simulator restart")
                    continue
                self.last_packet_time = stamp
                self.sequence += 1
                data.update(timestamp_sec=rospy.get_time(),sequence=self.sequence,valid=True,
                            protocol_verified=self.verified,source="collision_udp")
                self.publisher.publish(String(data=json.dumps(data,allow_nan=False)))
            except (ValueError,TypeError) as exc:
                rospy.logwarn_throttle(2,"CollisionData not scored: %s",exc)

    def close(self):
        self.stop.set()
        self.socket.close()


if __name__ == "__main__":
    rospy.init_node("collision_evaluation_bridge")
    CollisionBridge()
    rospy.spin()
