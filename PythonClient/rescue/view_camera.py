"""Show a live view of the drone's downward camera in its own window.

AirSim's built-in camera subwindow froze Unreal in Both mode, so watch the
camera from Python instead. Run it next to fly_survey.py while Play is on:

    python view_camera.py

Press q or Esc in the window to close it.
"""
import argparse
import time

import cv2
import numpy as np

import setup_path  # noqa: F401  (puts this repo's airsim package on the path)
import airsim

from survey_plan import DEFAULT_CONFIG, load_config


def main():
    cfg = load_config(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=cfg["camera"]["name"])
    parser.add_argument("--width", type=int, default=640, help="display width in pixels")
    args = parser.parse_args()

    drone = cfg["drone"]
    client = airsim.MultirotorClient(port=drone["port"])
    client.confirmConnection()
    request = [airsim.ImageRequest(args.camera, airsim.ImageType.Scene)]  # PNG, decoded below
    window = f"{drone['name']} {args.camera}"
    last = time.time()

    while True:
        response = client.simGetImages(request, vehicle_name=drone["name"])[0]
        image = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("no image from AirSim; is Play running?")
            time.sleep(1)
            continue

        image = cv2.resize(image, (args.width, args.width * image.shape[0] // image.shape[1]))
        now = time.time()
        fps, last = 1 / max(now - last, 1e-6), now
        z = client.getMultirotorState(vehicle_name=drone["name"]).kinematics_estimated.position.z_val
        cv2.putText(image, f"{fps:4.1f} fps   {-z:5.1f} m above spawn point", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, image)
        if cv2.waitKey(1) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
