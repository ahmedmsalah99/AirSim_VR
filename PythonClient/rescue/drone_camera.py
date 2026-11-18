"""The drone's search camera in AirSim: tilt it, grab frames, locate pixels on the ground."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from camera_geometry import Intrinsics, depth_point, ground_point, rotate

# Exactly 90 deg is gimbal lock for Unreal's rotators: heading and roll become ambiguous, so the
# image comes back rotated about the view axis. Half a degree off avoids it.
STRAIGHT_DOWN = 89.5
MAX_DEPTH_M = 500  # farther than this is sky
# simSetCameraOrientation's pitch sign and the camera's mounting aren't documented for this
# AirSim version; one of these (pitch sign, roll, yaw) combinations points it the way we want.
CONVENTIONS = [(sign, roll, yaw) for sign in (-1, 1)
               for roll, yaw in ((0, 0), (math.pi, math.pi), (0, math.pi), (math.pi, 0))]
FIND_TOLERANCE = 0.98   # cos of the allowed error while finding the convention (~11 deg)
REUSE_TOLERANCE = 0.95  # later read-backs further off than this (~18 deg) are reported


@dataclass
class Frame:
    image: np.ndarray       # BGR
    pos: np.ndarray         # camera position, drone frame (NED, relative to its spawn point)
    q: object               # camera orientation, airsim Quaternionr
    depth: np.ndarray = None  # planar depth in metres, same size as image


class DroneCamera:
    def __init__(self, client, airsim, cfg):
        self.client, self.airsim = client, airsim
        cam = cfg["camera"]
        self.camera = cam["name"]
        self.vehicle = cfg["drone"]["name"]
        self.intr = Intrinsics(cam["width_px"], cam["height_px"], cam["hfov_deg"])
        self.convention = None  # the CONVENTIONS entry that worked; reused for every later tilt

    def set_tilt(self, tilt_deg):
        """Point the camera tilt_deg below the horizon, straight ahead of the drone, image upright.

        simSetCameraOrientation's pitch sign and the camera's mounting aren't documented for
        this AirSim version, and a pitch angle can't tell "45 deg down, forward" from "135 deg
        down, backwards, upside down". So the first call tries each convention and keeps the
        one whose actual forward and up directions, read back from AirSim, are right (best
        done with the drone still on the ground). Later calls apply that convention directly:
        it's a rotation relative to the drone, so it holds for any tilt, and the read-back
        check proved unreliable in flight. A disagreeing read-back is only printed.
        Returns whether the tilt was set.
        """
        t = math.radians(min(tilt_deg, STRAIGHT_DOWN))
        if self.convention is not None:
            q = self._quat(t, self.convention)
            self.client.simSetCameraOrientation(self.camera, q, vehicle_name=self.vehicle)
            fwd, up = self._agreement(t)
            if min(fwd, up) < REUSE_TOLERANCE:
                print(f"note: camera read-back after tilting to {tilt_deg:.0f} deg disagrees "
                      f"(forward {math.degrees(math.acos(min(fwd, 1))):.0f} deg off, "
                      f"up {math.degrees(math.acos(min(up, 1))):.0f} deg off); using it anyway")
            return True
        for convention in CONVENTIONS:
            q = self._quat(t, convention)
            self.client.simSetCameraOrientation(self.camera, q, vehicle_name=self.vehicle)
            if min(self._agreement(t)) > FIND_TOLERANCE:
                self.convention = convention
                return True
        return False

    def _quat(self, t, convention):
        sign, roll, yaw = convention
        return self.airsim.to_quaternion(sign * t, roll, yaw)

    def _agreement(self, t):
        """cos of the angle between the camera's actual forward/up and the wanted ones."""
        cam = self.client.simGetCameraInfo(self.camera, vehicle_name=self.vehicle).pose.orientation
        body = self.client.getMultirotorState(vehicle_name=self.vehicle).kinematics_estimated.orientation
        # In the drone's frame (x forward, z down): forward tilted down by t, image top towards the sky.
        fwd = np.dot(rotate(cam, (1, 0, 0)), rotate(body, (math.cos(t), 0, math.sin(t))))
        up = np.dot(rotate(cam, (0, 0, -1)), rotate(body, (math.sin(t), 0, -math.cos(t))))
        return float(fwd), float(up)

    def grab(self, with_depth=False):
        """One frame. Scene and depth come from the same request, so they share one camera pose."""
        airsim = self.airsim
        requests = [airsim.ImageRequest(self.camera, airsim.ImageType.Scene)]
        if with_depth:
            requests.append(airsim.ImageRequest(self.camera, airsim.ImageType.DepthPlanner, True, False))
        responses = self.client.simGetImages(requests, vehicle_name=self.vehicle)
        scene = responses[0]
        image = cv2.imdecode(np.frombuffer(scene.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("AirSim returned no image; is Play running?")
        p, q = scene.camera_position, scene.camera_orientation
        frame = Frame(image, np.array([p.x_val, p.y_val, p.z_val]), q)
        if with_depth:
            d = responses[1]
            frame.depth = np.array(d.image_data_float, np.float32).reshape(d.height, d.width)
        return frame

    def ground_point(self, frame, u, v, ground_z):
        return ground_point(self.intr, frame.pos, frame.q, u, v, ground_z)

    def depth_point(self, frame, u, v):
        """3D point at image pixel (u, v) from the frame's depth image, or None where there's no surface.

        The depth image may be smaller than the scene image (settings.json keeps it at 320x180:
        AirSim's Python client decodes float images very slowly), with the same field of view,
        so (u, v) is scaled to it.
        """
        h, w = frame.depth.shape
        x = int(np.clip(u * w / self.intr.width, 0, w - 1))
        y = int(np.clip(v * h / self.intr.height, 0, h - 1))
        window = frame.depth[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        d = float(np.median(window))
        if not 0 < d < MAX_DEPTH_M:
            return None
        return depth_point(self.intr, frame.pos, frame.q, u, v, d)

    def center_depth(self):
        """Median planar depth in the middle of the view (height above ground when looking down)."""
        depth = self.grab(with_depth=True).depth
        h, w = depth.shape
        return float(np.median(depth[2 * h // 5:3 * h // 5, 2 * w // 5:3 * w // 5]))
