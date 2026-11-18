"""Print the victims the mission publishes. Run it in a second terminal before or during mission.py:

    python listen_victims.py
"""
import json

import zmq

from survey_plan import DEFAULT_CONFIG, load_config


def main():
    pub = load_config(DEFAULT_CONFIG)["publish"]
    endpoint = pub["endpoint"].replace("*", "localhost")
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.connect(endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, pub["topic"].encode())
    print(f"listening on {endpoint} {pub['topic']} (Ctrl+C to stop)")
    while True:
        topic, body = socket.recv_multipart()
        msg = json.loads(body)
        p = msg["position"]
        print(f"{topic.decode()}  victim {msg['id']}  at x={p['x']:.1f} y={p['y']:.1f} z={p['z']:.1f}  "
              f"({msg['hits']}/{msg['frames']} frames, confidence {msg['confidence']:.2f})")


if __name__ == "__main__":
    main()
