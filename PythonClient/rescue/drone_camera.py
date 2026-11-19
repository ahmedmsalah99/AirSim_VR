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
# How a wanted tilt t maps to simSetCameraOrientation(to_quaternion(pitch, roll, yaw)) isn't
# documented for this AirSim version: the pitch sign, and the camera's own mounting (the
# downward camera is already turned -90 deg, so pitch p gives a tilt of 90 - p). One of these
# (pitch sign, pitch offset, roll, yaw) combinations, pitch = sign * t + offset, is right.
CONVENTIONS = [(sign, offset, roll, yaw) for sign in (-1, 1) for offset in (0, math.pi / 2, -math.pi / 2)
               for roll, yaw in ((0, 0), (math.pi, math.pi), (0, math.pi), (math.pi, 0))]
# Two tilts are needed to tell the conventions apart: at 45 deg, "t" and "90 - t" coincide.
CALIBRATION_TILTS_DEG = (60, 30)
FIND_TOLERANCE = 0.98   # cos of the allowed error while calibrating (~11 deg)
REUSE_TOLERANCE = 0.95  # in flight, allow for the drone swaying while it's read back (~18 deg)


@dataclass
class Frame:
    image: np.ndarray       # BGR
    pos: np.ndarray         # camera position, drone frame (NED, relative to its spawn point)
    q: object               # camera orientation, airsim Quaternionr
    intr: Intrinsics        # the camera's intrinsics (zoom) when the frame was taken
    depth: np.ndarray = None  # planar depth in metres, may be smaller than image (same FOV)


class DroneCamera:
    def __init__(self, client, airsim, cfg):
        self.client, self.airsim = client, airsim
        cam = cfg["camera"]
        self.camera = cam["name"]
        self.vehicle = cfg["drone"]["name"]
        self.cam = cam
        self.intr = Intrinsics(cam["width_px"], cam["height_px"], cam["hfov_deg"])
        self.convention = None  # the CONVENTIONS entry that worked; reused for every later tilt

    def set_tilt(self, tilt_deg):
        """Point the camera tilt_deg below the horizon, straight ahead of the drone, image upright.

        The first call calibrates: it finds the convention (see CONVENTIONS) that gives the
        right forward and up directions, read back from AirSim, at two different tilts (best
        with the drone still on the ground). Later calls apply that convention and check the
        read-back again with a looser tolerance; if it disagrees, every convention is tried
        for this tilt. Returns whether the camera ended up pointing the right way.
        """
        t = math.radians(min(tilt_deg, STRAIGHT_DOWN))
        if self.convention is None and not self.calibrate():
            return False
        if self._apply(t, self.convention, REUSE_TOLERANCE):
            return True
        for convention in CONVENTIONS:
            if self._apply(t, convention, REUSE_TOLERANCE):
                print(f"note: the camera tilt convention changed at {tilt_deg:.0f} deg; using the new one")
                self.convention = convention
                return True
        fwd, up = self._agreement(t)
        print(f"WARNING: couldn't tilt the camera to {tilt_deg:.0f} deg (read-back: forward "
              f"{math.degrees(math.acos(min(fwd, 1))):.0f} deg off, up {math.degrees(math.acos(min(up, 1))):.0f} deg off)")
        return False

    def calibrate(self):
        """Find the convention that's right at every calibration tilt."""
        for convention in CONVENTIONS:
            if all(self._apply(math.radians(t), convention, FIND_TOLERANCE) for t in CALIBRATION_TILTS_DEG):
                self.convention = convention
                return True
        return False

    def set_fov(self, fov_deg):
        """Zoom: set the horizontal field of view (all image types), and the intrinsics with it."""
        self.client.simSetCameraFov(self.camera, fov_deg, vehicle_name=self.vehicle)
        self.intr = Intrinsics(self.cam["width_px"], self.cam["height_px"], fov_deg)

    def _apply(self, t, convention, tolerance):
        sign, offset, roll, yaw = convention
        q = self.airsim.to_quaternion(sign * t + offset, roll, yaw)
        self.client.simSetCameraOrientation(self.camera, q, vehicle_name=self.vehicle)
        return min(self._agreement(t)) > tolerance

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
        frame = Frame(image, np.array([p.x_val, p.y_val, p.z_val]), q, self.intr)
        if with_depth:
            d = responses[1]
            frame.depth = np.array(d.image_data_float, np.float32).reshape(d.height, d.width)
        return frame

    def ground_point(self, frame, u, v, ground_z):
        return ground_point(frame.intr, frame.pos, frame.q, u, v, ground_z)

    def depth_point(self, frame, u, v):
        """3D point at image pixel (u, v) from the frame's depth image, or None where there's no surface.

        The depth image may be smaller than the scene image (settings.json keeps it at 320x180:
        AirSim's Python client decodes float images very slowly), with the same field of view,
        so (u, v) is scaled to it.
        """
        h, w = frame.depth.shape
        x = int(np.clip(u * w / frame.intr.width, 0, w - 1))
        y = int(np.clip(v * h / frame.intr.height, 0, h - 1))
        window = frame.depth[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        d = float(np.median(window))
        if not 0 < d < MAX_DEPTH_M:
            return None
        return depth_point(frame.intr, frame.pos, frame.q, u, v, d)

    def center_depth(self):
        """Median planar depth in the middle of the view (height above ground when looking down).

        Fetches only the depth image: float images are slow through AirSim's Python client,
        so keep them small (320x180 in settings.json).
        """
        request = self.airsim.ImageRequest(self.camera, self.airsim.ImageType.DepthPlanner, True, False)
        d = self.client.simGetImages([request], vehicle_name=self.vehicle)[0]
        depth = np.array(d.image_data_float, np.float32).reshape(d.height, d.width)
        h, w = depth.shape
        return float(np.median(depth[2 * h // 5:3 * h // 5, 2 * w // 5:3 * w // 5]))
