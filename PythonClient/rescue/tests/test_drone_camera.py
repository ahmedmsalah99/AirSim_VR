"""set_tilt against a fake AirSim whose downward camera is mounted at -90 deg, like AirSim's.

With that mounting, pitch p gives a tilt of 90 - p: at 45 deg the wrong mapping "pitch = t"
looks right, which is what fooled a calibration done at one tilt only.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_geometry import rotate  # noqa: E402
from drone_camera import DroneCamera  # noqa: E402


def quat(pitch, roll, yaw):
    """Same formula as airsim.to_quaternion."""
    t0, t1 = math.cos(yaw / 2), math.sin(yaw / 2)
    t2, t3 = math.cos(roll / 2), math.sin(roll / 2)
    t4, t5 = math.cos(pitch / 2), math.sin(pitch / 2)
    return types.SimpleNamespace(w_val=t0 * t2 * t4 + t1 * t3 * t5, x_val=t0 * t3 * t4 - t1 * t2 * t5,
                                 y_val=t0 * t2 * t5 + t1 * t3 * t4, z_val=t1 * t2 * t4 - t0 * t3 * t5)


def mul(a, b):
    w1, x1, y1, z1 = a.w_val, a.x_val, a.y_val, a.z_val
    w2, x2, y2, z2 = b.w_val, b.x_val, b.y_val, b.z_val
    return types.SimpleNamespace(w_val=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, x_val=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                 y_val=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, z_val=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


MOUNT = quat(-math.pi / 2, 0, 0)  # the downward camera's own -90 deg pitch


class FakeClient:
    def __init__(self):
        self.body = quat(0, 0, 0)
        self.applied = quat(0, 0, 0)
        self.broken = False

    def simSetCameraOrientation(self, camera, q, vehicle_name):
        self.applied = q

    def simGetCameraInfo(self, camera, vehicle_name):
        rel = quat(0, 0, 0) if self.broken else self.applied  # broken: the camera ignores requests
        return types.SimpleNamespace(pose=types.SimpleNamespace(orientation=mul(mul(self.body, MOUNT), rel)))

    def getMultirotorState(self, vehicle_name):
        return types.SimpleNamespace(kinematics_estimated=types.SimpleNamespace(orientation=self.body))


CFG = {"camera": {"name": "bottom_center", "width_px": 1280, "height_px": 720, "hfov_deg": 60},
       "drone": {"name": "Drone1"}}


def looks(client, tilt_deg):
    """cos of the angle between where the camera looks and tilt_deg below the horizon, ahead."""
    cam = client.simGetCameraInfo("", "").pose.orientation
    t = math.radians(tilt_deg)
    return np.dot(rotate(cam, (1, 0, 0)), rotate(client.body, (math.cos(t), 0, math.sin(t))))


def camera(client):
    return DroneCamera(client, types.SimpleNamespace(to_quaternion=quat), CFG)


def test_calibration_finds_the_mapping_that_works_at_every_tilt():
    client = FakeClient()
    cam = camera(client)
    assert cam.set_tilt(45)
    sign, offset, _, _ = cam.convention
    assert (sign, offset) == (-1, pytest.approx(math.pi / 2))  # pitch = 90 - t
    for tilt in (45, 63, 70, 30, 89.5):
        assert cam.set_tilt(tilt)
        assert looks(client, tilt) > 0.99, tilt


def test_tilts_hold_while_the_drone_sways_and_turns():
    client = FakeClient()
    cam = camera(client)
    assert cam.set_tilt(45)
    client.body = quat(math.radians(6), math.radians(-4), math.radians(90))
    assert cam.set_tilt(64)
    assert looks(client, 64) > 0.99


def test_a_camera_that_ignores_requests_is_reported(capsys):
    client = FakeClient()
    cam = camera(client)
    assert cam.set_tilt(45)
    client.broken = True
    assert not cam.set_tilt(30)
    assert "couldn't tilt" in capsys.readouterr().out


def test_no_convention_found_returns_false():
    client = FakeClient()
    client.broken = True
    cam = camera(client)
    assert not cam.set_tilt(45)
    assert cam.convention is None


def test_depth_point_scales_pixels_to_a_smaller_depth_image():
    cam = camera(FakeClient())
    depth = np.full((180, 320), 50.0, np.float32)
    depth[134:137, 239:242] = 10.0             # around image pixel (960, 540) of a 1280x720 frame
    frame = types.SimpleNamespace(depth=depth, pos=np.zeros(3), q=quat(0, 0, 0), intr=cam.intr)
    assert cam.depth_point(frame, 960, 540)[0] == pytest.approx(10.0)  # planar depth along the forward axis
    assert cam.depth_point(frame, 100, 100)[0] == pytest.approx(50.0)
