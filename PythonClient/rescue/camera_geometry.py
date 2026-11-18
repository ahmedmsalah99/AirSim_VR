"""Pixel <-> world geometry for an AirSim camera.

Frames are NED (x north, y east, z down). AirSim cameras use x forward,
y right, z down; image u grows to the right and v downwards. Camera poses
from AirSim image responses are in the drone's own frame (relative to its
spawn point), the same frame as its kinematics.
"""
import math

import numpy as np


def quat_wxyz(q):
    """(w, x, y, z) from an airsim Quaternionr or anything with w_val..z_val."""
    return q.w_val, q.x_val, q.y_val, q.z_val


def rotate(q, v):
    """Rotate vector v by quaternion q (airsim Quaternionr)."""
    w, x, y, z = quat_wxyz(q)
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class Intrinsics:
    def __init__(self, width_px, height_px, hfov_deg):
        self.width, self.height = width_px, height_px
        self.f = width_px / (2 * math.tan(math.radians(hfov_deg) / 2))
        self.cx, self.cy = width_px / 2, height_px / 2

    def ray(self, u, v):
        """Direction of pixel (u, v) in the camera frame (not normalised; forward component 1)."""
        return np.array([1.0, (u - self.cx) / self.f, (v - self.cy) / self.f])


def ground_point(intr, cam_pos, cam_q, u, v, ground_z):
    """Where the ray through pixel (u, v) hits the flat ground z = ground_z, or None if it doesn't."""
    d = rotate(cam_q, intr.ray(u, v))
    if d[2] <= 1e-6:
        return None  # at or above the horizon
    t = (ground_z - cam_pos[2]) / d[2]
    return np.asarray(cam_pos, float) + t * d if t > 0 else None


def depth_point(intr, cam_pos, cam_q, u, v, depth_planar):
    """3D point for pixel (u, v) given its planar depth (distance along the camera's optical axis)."""
    return np.asarray(cam_pos, float) + rotate(cam_q, intr.ray(u, v) * depth_planar)


def bearing_deg(frm, to):
    """Heading in degrees (0 = north, 90 = east) from point frm to point to."""
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


def confirm_viewpoint(drone, target, distance, lowest_z):
    """Where to confirm a person from, approaching along the line of sight.

    drone, target: positions (NED, z down). Returns the point `distance` from the target
    on the straight line to the drone (never farther than the drone already is, and
    never below lowest_z), the heading that faces the target, and the line-of-sight
    angle below the horizon in degrees.
    """
    d = np.asarray(drone, float) - np.asarray(target, float)
    r = float(np.linalg.norm(d))
    unit = d / r
    tilt = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
    s = min(distance, r)
    if target[2] + unit[2] * s > lowest_z:
        # Too low: stop on the line where it reaches the lowest allowed height.
        s = min(r, (lowest_z - target[2]) / unit[2]) if unit[2] < 0 else r
    return np.asarray(target, float) + unit * s, bearing_deg(drone, target), tilt
