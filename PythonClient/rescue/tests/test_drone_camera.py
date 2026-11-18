"""set_tilt against a fake AirSim whose camera pitch sign is flipped and whose drone sways."""
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
    """airsim.to_quaternion, keeping its arguments so the fake can reinterpret them."""
    t0, t1 = math.cos(yaw / 2), math.sin(yaw / 2)
    t2, t3 = math.cos(roll / 2), math.sin(roll / 2)
    t4, t5 = math.cos(pitch / 2), math.sin(pitch / 2)
    return types.SimpleNamespace(w_val=t0 * t2 * t4 + t1 * t3 * t5, x_val=t0 * t3 * t4 - t1 * t2 * t5,
                                 y_val=t0 * t2 * t5 + t1 * t3 * t4, z_val=t1 * t2 * t4 - t0 * t3 * t5,
                                 args=(pitch, roll, yaw))


def mul(a, b):
    w1, x1, y1, z1 = a.w_val, a.x_val, a.y_val, a.z_val
    w2, x2, y2, z2 = b.w_val, b.x_val, b.y_val, b.z_val
    return types.SimpleNamespace(w_val=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, x_val=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                 y_val=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, z_val=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


class FakeClient:
    """The camera's relative rotation is the requested one with the pitch sign flipped."""

    def __init__(self):
        self.body = quat(0, 0, 0)
        self.applied = quat(0, 0, 0)
        self.broken = False

    def simSetCameraOrientation(self, camera, q, vehicle_name):
        self.applied = q

    def simGetCameraInfo(self, camera, vehicle_name):
        if self.broken:
            rel = quat(0, 0, 0)  # the camera ignores every request
        else:
            p, r, y = self.applied.args
            rel = quat(-p, r, y)
        return types.SimpleNamespace(pose=types.SimpleNamespace(orientation=mul(self.body, rel)))

    def getMultirotorState(self, vehicle_name):
        return types.SimpleNamespace(kinematics_estimated=types.SimpleNamespace(orientation=self.body))


CFG = {"camera": {"name": "bottom_center", "width_px": 1280, "height_px": 720, "hfov_deg": 60},
       "drone": {"name": "Drone1"}}


def looks(client, tilt_deg):
    cam = client.simGetCameraInfo("", "").pose.orientation
    t = math.radians(tilt_deg)
    want = rotate(client.body, (math.cos(t), 0, math.sin(t)))
    return np.dot(rotate(cam, (1, 0, 0)), want)


def test_convention_is_found_once_then_applied_even_if_the_read_back_disagrees(capsys):
    client = FakeClient()
    cam = DroneCamera(client, types.SimpleNamespace(to_quaternion=quat), CFG)

    assert cam.set_tilt(45)
    assert cam.convention[0] == 1          # the fake flips the pitch sign, so +pitch looks down
    assert looks(client, 45) > 0.99

    client.body = quat(math.radians(10), math.radians(-5), math.radians(90))  # braking and turned east
    assert cam.set_tilt(64)
    assert looks(client, 64) > 0.95

    client.broken = True                   # the read-back no longer matches what was applied
    assert cam.set_tilt(30)
    assert client.applied.args == pytest.approx((math.radians(30), 0, 0))
    assert "disagrees" in capsys.readouterr().out


def test_no_convention_found_returns_false():
    client = FakeClient()
    client.broken = True
    cam = DroneCamera(client, types.SimpleNamespace(to_quaternion=quat), CFG)
    assert not cam.set_tilt(45)
    assert cam.convention is None
