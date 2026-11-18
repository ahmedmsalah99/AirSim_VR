import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_geometry import Intrinsics, bearing_deg, depth_point, ground_point, rotate  # noqa: E402


def quat(pitch, roll, yaw):
    """Same formula as airsim.to_quaternion (radians)."""
    t0, t1 = math.cos(yaw / 2), math.sin(yaw / 2)
    t2, t3 = math.cos(roll / 2), math.sin(roll / 2)
    t4, t5 = math.cos(pitch / 2), math.sin(pitch / 2)
    return types.SimpleNamespace(w_val=t0 * t2 * t4 + t1 * t3 * t5, x_val=t0 * t3 * t4 - t1 * t2 * t5,
                                 y_val=t0 * t2 * t5 + t1 * t3 * t4, z_val=t1 * t2 * t4 - t0 * t3 * t5)


INTR = Intrinsics(1280, 720, 60)


def test_rotate_yaw_and_pitch():
    assert rotate(quat(0, 0, math.pi / 2), (1, 0, 0)) == pytest.approx([0, 1, 0], abs=1e-9)  # yaw 90: east
    down45 = rotate(quat(-math.pi / 4, 0, 0), (1, 0, 0))
    assert down45 == pytest.approx([math.sqrt(0.5), 0, math.sqrt(0.5)])  # 45 deg below the horizon (z down)


def test_image_centre_at_45_deg_hits_ground_one_altitude_ahead():
    cam_pos = np.array([0.0, 0.0, -20.0])  # 20 m above ground z = 0
    p = ground_point(INTR, cam_pos, quat(-math.pi / 4, 0, 0), INTR.cx, INTR.cy, 0.0)
    assert p == pytest.approx([20, 0, 0], abs=1e-9)


def test_pixel_right_of_centre_lands_east_when_facing_north():
    cam_pos = np.array([0.0, 0.0, -20.0])
    p = ground_point(INTR, cam_pos, quat(-math.pi / 4, 0, 0), INTR.cx + 200, INTR.cy, 0.0)
    assert p[1] > 0


def test_ray_above_horizon_misses_ground():
    cam_pos = np.array([0.0, 0.0, -20.0])
    assert ground_point(INTR, cam_pos, quat(-math.radians(5), 0, 0), INTR.cx, 0, 0.0) is None


def test_depth_point_matches_ground_point_on_flat_ground():
    cam_pos = np.array([3.0, -2.0, -15.0])
    q = quat(-math.radians(40), 0, math.radians(30))
    u, v = 900, 500
    g = ground_point(INTR, cam_pos, q, u, v, 0.0)
    planar = rotate_inverse_forward(q, g - cam_pos)
    assert depth_point(INTR, cam_pos, q, u, v, planar) == pytest.approx(g, abs=1e-9)


def rotate_inverse_forward(q, v):
    """Distance along the camera's optical axis to point offset v (planar depth)."""
    return float(np.dot(rotate(q, (1, 0, 0)), v))


def test_bearing():
    assert bearing_deg((0, 0), (10, 0)) == pytest.approx(0)
    assert bearing_deg((0, 0), (0, 10)) == pytest.approx(90)
    assert bearing_deg((0, 0), (-10, 0)) == pytest.approx(180)
