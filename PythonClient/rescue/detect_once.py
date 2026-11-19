"""Test the person detector on one drone camera frame (or on an image file).

With Play running in Unreal:

    python detect_once.py                      # frame from where the drone is now
    python detect_once.py --target Victim_01   # fly above that actor first, then detect
    python detect_once.py --target Victim_01 --tilt 45   # look at it at 45 deg instead of straight down

Without AirSim, on a saved image:

    python detect_once.py --image bottom.png

The image with boxes drawn is saved in output/. --target takes the actor's
name or tag as AirSim sees it (give the actor a tag in Details > Actor > Tags).
With --tilt the camera looks diagonally forward; the drone then hovers back
from the target, facing it, so the target is in the middle of the frame. The
camera is turned straight down again afterwards. The drone is left hovering;
run mission.py or land it from Python afterwards.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2

from detector import PersonDetector, draw
from survey_plan import DEFAULT_CONFIG, load_config, plan_survey

OUTPUT_DIR = Path(__file__).with_name("output")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="run on this image file instead of the drone camera")
    parser.add_argument("--target", help="actor name or tag to hover above before capturing")
    parser.add_argument("--altitude", type=float, help="height above ground to capture from (default: survey altitude)")
    parser.add_argument("--tilt", type=float, default=90,
                        help="camera angle below the horizon in degrees: 90 = straight down (default), 45 = diagonal")
    parser.add_argument("--all-classes", action="store_true", help="also show vehicles")
    args = parser.parse_args()

    cfg = load_config(DEFAULT_CONFIG)
    detector = PersonDetector(cfg["detector"])
    print(f"detector on {detector.device}: {detector.seconds_per_tile:.2f} s per tile")
    classes = None if not args.all_classes else [0, 1]

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            sys.exit(f"can't read {args.image}")
        name = Path(args.image).stem
    else:
        altitude = args.altitude or plan_survey(cfg).altitude
        image = TestFlight(cfg).capture(args.target, altitude, args.tilt)
        name = f"{args.target or 'here'}_tilt{args.tilt:g}"

    detector.detect(image, classes)  # warm-up: the first run includes ONNX Runtime setup
    start = time.time()
    detections = detector.detect(image, classes)
    elapsed = time.time() - start

    print(f"{len(detections)} detection(s) in {elapsed * 1000:.0f} ms "
          f"({cfg['detector']['tiles_x']}x{cfg['detector']['tiles_y']} tiles on {image.shape[1]}x{image.shape[0]})")
    for d in detections:
        cx, cy = d.center
        print(f"  class {d.class_id}  conf {d.confidence:.2f}  {d.width:.0f}x{d.height:.0f} px  at ({cx:.0f}, {cy:.0f})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_{stamp}_raw.png"), image)
    out = OUTPUT_DIR / f"{name}_{stamp}_detections.png"
    cv2.imwrite(str(out), draw(image, detections))
    print(f"saved {out}")


class TestFlight:
    def __init__(self, cfg):
        import setup_path  # noqa: F401  (puts this repo's airsim package on the path)
        import airsim
        from drone_camera import DroneCamera

        self.airsim = airsim
        self.cfg = cfg
        self.name = cfg["drone"]["name"]
        self.spawn = cfg["drone"]["spawn_ned"]
        self.client = airsim.MultirotorClient(port=cfg["drone"]["port"])
        self.client.confirmConnection()
        self.camera = DroneCamera(self.client, airsim, cfg)

    def capture(self, target, altitude, tilt):
        from drone_camera import STRAIGHT_DOWN

        if target:
            self.hover_near(target, altitude, tilt)
        self.camera.set_tilt(STRAIGHT_DOWN)
        height = self.camera.center_depth()
        try:
            if tilt < STRAIGHT_DOWN:
                if not self.camera.set_tilt(tilt):
                    sys.exit(f"couldn't point the camera {tilt:g} deg down")
                time.sleep(0.5)
            image = self.camera.grab().image
        finally:
            if tilt < STRAIGHT_DOWN and not self.camera.set_tilt(STRAIGHT_DOWN):
                print("WARNING: couldn't turn the camera straight down again; restart Play before flying")
        print(f"captured {image.shape[1]}x{image.shape[0]} at {tilt:g} deg below the horizon, "
              f"{height:.1f} m above the ground")
        return image

    def hover_near(self, target, altitude, tilt):
        """Hover south of the target, facing north, so the tilted camera's centre is on it."""
        from drone_camera import STRAIGHT_DOWN

        c, airsim = self.client, self.airsim
        pose = c.simGetObjectPose(target)
        if any(math.isnan(v) for v in (pose.position.x_val, pose.position.y_val)):
            sys.exit(f"AirSim can't find an actor named or tagged '{target}'. "
                     f"Add the tag in the editor (Details > Actor > Tags), Save All, and restart Play.")
        back = altitude / math.tan(math.radians(tilt)) if tilt < 90 else 0.0
        x = pose.position.x_val - back - self.spawn[0]
        y = pose.position.y_val - self.spawn[1]

        c.enableApiControl(True, self.name)
        c.armDisarm(True, self.name)
        state = c.getMultirotorState(vehicle_name=self.name)
        ground_z = state.kinematics_estimated.position.z_val
        if state.landed_state == airsim.LandedState.Landed:
            print("taking off")
            c.takeoffAsync(vehicle_name=self.name).join()
        else:
            self.camera.set_tilt(STRAIGHT_DOWN)
            ground_z += self.camera.center_depth()
        z = ground_z - altitude

        print(f"flying to {altitude:.1f} m up, {back:.1f} m south of {target}")
        north = airsim.YawMode(False, 0)
        c.moveToZAsync(z, self.cfg["drone"]["climb_speed"], yaw_mode=north, vehicle_name=self.name).join()
        c.moveToPositionAsync(x, y, z, self.cfg["drone"]["transit_speed"], yaw_mode=north,
                              vehicle_name=self.name).join()
        c.hoverAsync(vehicle_name=self.name).join()
        time.sleep(2)  # let it settle so the frame isn't tilted


if __name__ == "__main__":
    main()
