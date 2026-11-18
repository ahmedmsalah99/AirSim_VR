"""Publish confirmed victims over ZeroMQ.

Each message is two frames: the topic (e.g. b"/rescue/victims") and a JSON
body shaped like a ROS geometry_msgs/PoseWithCovarianceStamped, so a ROS 2
bridge can republish it without reshaping:

    {"header": {"stamp": 1700000000.123, "frame_id": "airsim_world_ned"},
     "id": 1, "position": {"x": 12.3, "y": -4.5, "z": 0.2},
     "covariance": [9 values, row-major 3x3, m^2],
     "confidence": 0.71, "hits": 6, "frames": 8}
"""
import json
import time


class VictimPublisher:
    def __init__(self, endpoint, topic):
        import zmq

        self.topic = topic.encode()
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.bind(endpoint)

    def publish(self, victim_id, position, std, confidence, hits, frames):
        msg = {
            "header": {"stamp": time.time(), "frame_id": "airsim_world_ned"},
            "id": victim_id,
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "covariance": [float(std[0]) ** 2, 0, 0, 0, float(std[1]) ** 2, 0, 0, 0, float(std[2]) ** 2],
            "confidence": float(confidence),
            "hits": hits,
            "frames": frames,
        }
        self.socket.send_multipart([self.topic, json.dumps(msg).encode()])
        return msg

    def close(self):
        self.socket.close(linger=500)
