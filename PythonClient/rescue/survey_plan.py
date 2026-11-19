"""Plan the drone's lawnmower search over a rectangle.

The camera looks forward along the lane, tilted camera.tilt_deg below the
horizon (90 = straight down). On flat ground it sees a trapezoid; a person
is only big enough for the detector out to the slant range where they
still cover min_person_px pixels. The part of the view inside that range
(the detection band, near_m to far_m ahead of the drone) sets how wide a
strip one lane covers and how long each ground point stays detectable.

From that the plan picks the lane spacing and the speed, tries every
altitude from the lowest safe one up to max_altitude, keeps the fastest
mission, and records which constraint decided each value.

All positions are in AirSim's world NED frame: metres from the PlayerStart,
x north (Unreal +X), y east (Unreal +Y), z down. Altitudes are metres above
the ground at the takeoff point, which is assumed flat across the area.

    python survey_plan.py [config.yaml]    # print the plan without flying
"""
from dataclasses import dataclass, field
import math
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
ALTITUDE_STEP = 0.5


class InfeasiblePlan(Exception):
    pass


@dataclass
class Lane:
    start: tuple        # (x, y) world NED
    end: tuple
    heading_deg: float  # direction of travel; the camera looks this way


@dataclass
class ViewBand:
    """Where on the ground a person is detectable, for one altitude and tilt."""
    near_m: float       # ahead of the drone (negative: behind it)
    far_m: float
    swath_m: float      # width of ground covered at the far edge
    person_px_near: float
    person_px_far: float
    far_limited_by: str  # "detector pixels" or "image edge"


@dataclass
class SurveyPlan:
    altitude: float                 # m above ground
    min_safe_altitude: float
    altitude_reason: str
    tilt_deg: float
    band: ViewBand
    lane_spacing: float
    lane_axis: str                  # "x" or "y": the direction lanes run in
    speed: float
    speed_limits: dict              # name -> m/s
    lanes: list = field(default_factory=list)
    path_length: float = 0.0
    est_time_s: float = 0.0

    @property
    def speed_reason(self):
        return min(self.speed_limits, key=self.speed_limits.get)

    @property
    def lane_count(self):
        return len(self.lanes)

    @property
    def waypoints(self):
        return [p for lane in self.lanes for p in (lane.start, lane.end)]


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def tile_size(length, tiles, overlap):
    """Size of each of `tiles` equal tiles that cover `length` with neighbours overlapping by `overlap`."""
    return length / (1 + (tiles - 1) * (1 - overlap))


def detector_scale(cam, det):
    """Detector-input pixels per image pixel: each tile is resized to fit det['input_size']."""
    tile_w = tile_size(cam["width_px"], det["tiles_x"], det["tile_overlap"])
    tile_h = tile_size(cam["height_px"], det["tiles_y"], det["tile_overlap"])
    return min(1.0, det["input_size"] / max(tile_w, tile_h))


def focal_px(cam):
    """Focal length in image pixels (square pixels, hfov across the width)."""
    return cam["width_px"] / (2 * math.tan(math.radians(cam["hfov_deg"]) / 2))


def fov_for_pixels(cam, det, distance, person_px):
    """Horizontal FOV (deg) at which a person `distance` away covers person_px detector pixels."""
    f = person_px * distance / (det["person_size_m"] * detector_scale(cam, det))
    return math.degrees(2 * math.atan(cam["width_px"] / (2 * f)))


def vertical_fov(cam):
    return 2 * math.atan(math.tan(math.radians(cam["hfov_deg"]) / 2) * cam["height_px"] / cam["width_px"])


def detection_range(cam, det):
    """Slant range at which a person shrinks to min_person_px detector pixels."""
    return det["person_size_m"] * focal_px(cam) * detector_scale(cam, det) / det["min_person_px"]


def view_band(altitude, tilt_deg, cam, det):
    """The detection band at this altitude and tilt, or None if no person would be detectable."""
    tilt = math.radians(tilt_deg)
    half_v = vertical_fov(cam) / 2
    r_max = detection_range(cam, det)
    if altitude >= r_max:
        return None  # even straight below, people are too small

    a_near = tilt + half_v                          # steepest ray: bottom edge of the image
    a_far, limited_by = tilt - half_v, "image edge"  # shallowest ray: top edge
    a_pixels = math.asin(altitude / r_max)          # shallower than this, people are too small
    if a_pixels > a_far:
        a_far, limited_by = a_pixels, "detector pixels"
    if a_far >= min(a_near, math.pi / 2 + half_v):
        return None

    def ahead(a):
        return altitude / math.tan(a) if abs(a - math.pi / 2) > 1e-9 else 0.0

    def person_px(a):
        return det["person_size_m"] * focal_px(cam) * detector_scale(cam, det) * math.sin(a) / altitude

    # Ground width covered at the far edge: depth along the optical axis times the horizontal FOV.
    depth = altitude / math.sin(a_far) * math.cos(a_far - tilt)
    swath = 2 * depth * math.tan(math.radians(cam["hfov_deg"]) / 2)
    return ViewBand(ahead(a_near), ahead(a_far), swath, person_px(a_near), person_px(a_far), limited_by)


def plan_survey(cfg):
    area, search = cfg["area"], cfg["search"]
    min_safe = max(area["max_obstacle_height"] + search["safety_margin"], search["min_altitude"])
    if min_safe > search["max_altitude"]:
        raise InfeasiblePlan(f"the lowest safe altitude ({min_safe:.1f} m) is above max_altitude "
                             f"({search['max_altitude']} m)")

    plans = []
    h = min_safe
    while h <= search["max_altitude"] + 1e-9:
        plan = plan_at_altitude(cfg, h)
        if plan:
            plans.append(plan)
        h += ALTITUDE_STEP
    if not plans:
        r_max = detection_range(cfg["camera"], cfg["detector"])
        raise InfeasiblePlan(
            f"at {cfg['camera']['tilt_deg']} deg tilt no altitude between {min_safe:.1f} and "
            f"{search['max_altitude']} m keeps people detectable (detection range {r_max:.1f} m). Use more "
            f"detector tiles, a higher camera resolution, a narrower field of view or a steeper tilt.")

    # Fastest mission; among near-ties (within 2%) the lowest altitude, which sees people biggest.
    fastest = min(p.est_time_s for p in plans)
    best = next(p for p in plans if p.est_time_s <= fastest * 1.02)
    best.min_safe_altitude = min_safe
    best.altitude_reason = ("lowest safe altitude" if best.altitude == min_safe
                            else "fastest mission between the lowest safe altitude and max_altitude")
    return best


def plan_at_altitude(cfg, altitude):
    area, cam, det, drone, search = cfg["area"], cfg["camera"], cfg["detector"], cfg["drone"], cfg["search"]
    band = view_band(altitude, cam["tilt_deg"], cam, det)
    if band is None:
        return None

    frame_rate = det["fps_per_tile"] / (det["tiles_x"] * det["tiles_y"])
    a_near = math.radians(cam["tilt_deg"]) + vertical_fov(cam) / 2
    speed_limits = {
        "drone max speed": drone["max_speed"],
        # Image motion of a ground point is fastest at the near edge: v sin^2(a) / h radians per second.
        "motion blur": search["max_blur_px"] * altitude / (focal_px(cam) * math.sin(a_near) ** 2 * cam["exposure_s"]),
        # A ground point crosses the detection band in (far - near) / v seconds and must be seen in min_frames.
        "frames per person": (band.far_m - band.near_m) * frame_rate / search["min_frames_per_point"],
    }
    speed = min(speed_limits.values())
    max_spacing = band.swath_m * (1 - search["side_overlap"])

    # Lanes along the axis set in the config, or ("auto") whichever gives the shorter mission.
    fixed = search.get("lane_axis", "auto")
    if fixed not in ("x", "y", "auto"):
        raise InfeasiblePlan(f"search.lane_axis must be x, y or auto, not {fixed!r}")
    best = None
    for axis in (("x", "y") if fixed == "auto" else (fixed,)):
        lanes, spacing = _lanes(area, axis, band, max_spacing)
        length = _path_length(lanes)
        time_s = length / speed + (len(lanes) - 1) * drone["turn_time_s"]
        if best is None or time_s < best[3]:
            best = (axis, lanes, spacing, time_s, length)
    axis, lanes, spacing, sweep_time, length = best

    home = tuple(drone["spawn_ned"][:2])
    transit = math.dist(home, lanes[0].start) + math.dist(lanes[-1].end, home)
    est_time = sweep_time + 2 * altitude / drone["climb_speed"] + transit / drone["transit_speed"]
    return SurveyPlan(altitude=altitude, min_safe_altitude=altitude, altitude_reason="", tilt_deg=cam["tilt_deg"],
                      band=band, lane_spacing=spacing, lane_axis=axis, speed=speed, speed_limits=speed_limits,
                      lanes=lanes, path_length=length, est_time_s=est_time)


def _lanes(area, axis, band, max_spacing):
    """Lanes parallel to `axis`, spread evenly across the area, alternating direction.

    The camera sees band.near_m..band.far_m ahead, so each lane is shifted back by
    near_m: at its start the near edge of the band is on the area's edge, and at its
    end the band has just passed the far edge.
    """
    (x0, y0), (x1, y1) = area["corner_a"], area["corner_b"]
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    lo, hi, start, end = (y0, y1, x0, x1) if axis == "x" else (x0, x1, y0, y1)

    width, swath = hi - lo, band.swath_m
    if width <= swath:
        offsets, spacing = [(lo + hi) / 2], 0.0
    else:
        # The outer lanes cover their edge with half a swath; the rest are evenly spaced.
        n = math.ceil((width - swath) / max_spacing) + 1
        spacing = (width - swath) / (n - 1)
        offsets = [lo + swath / 2 + i * spacing for i in range(n)]

    lanes = []
    for i, c in enumerate(offsets):
        forward = i % 2 == 0
        a, b = (start - band.near_m, end - band.near_m) if forward else (end + band.near_m, start + band.near_m)
        heading = (0 if forward else 180) if axis == "x" else (90 if forward else -90)
        p, q = ((a, c), (b, c)) if axis == "x" else ((c, a), (c, b))
        lanes.append(Lane(p, q, heading))
    return lanes, spacing


def _path_length(lanes):
    points = [p for lane in lanes for p in (lane.start, lane.end)]
    return sum(math.dist(p, q) for p, q in zip(points, points[1:]))


def format_plan(plan):
    b = plan.band
    limits = "\n".join(f"    {name:<18} {v:6.1f} m/s" for name, v in plan.speed_limits.items())
    return f"""Survey plan
  altitude          {plan.altitude:6.1f} m   ({plan.altitude_reason}; lowest safe {plan.min_safe_altitude:.1f} m)
  camera tilt       {plan.tilt_deg:6.1f} deg below the horizon, looking along the lane
  detection band    {b.near_m:6.1f} to {b.far_m:.1f} m ahead (far edge set by {b.far_limited_by})
  person size       {b.person_px_near:6.1f} px at the near edge, {b.person_px_far:.1f} px at the far edge
  swath             {b.swath_m:6.1f} m wide at the far edge
  lanes             {plan.lane_count} lanes along {plan.lane_axis}, {plan.lane_spacing:.1f} m apart
  speed             {plan.speed:6.1f} m/s  (limited by {plan.speed_reason})
{limits}
  path length       {plan.path_length:6.0f} m
  estimated time    {plan.est_time_s / 60:6.1f} min (climb, transit, sweep, turns, return; without confirmations)"""


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    try:
        print(format_plan(plan_survey(load_config(config_path))))
    except InfeasiblePlan as e:
        sys.exit(f"Plan is infeasible: {e}")
