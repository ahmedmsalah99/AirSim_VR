"""Plan the drone's lawnmower search over a rectangle.

The plan picks the survey altitude, the spacing between lanes and the
ground speed from the camera, the detector and the drone's limits, and
records which constraint decided each value.

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


class InfeasiblePlan(Exception):
    pass


@dataclass
class SurveyPlan:
    altitude: float                 # m above ground
    min_safe_altitude: float
    max_detect_altitude: float
    altitude_reason: str
    footprint_across: float         # ground width seen by the camera, across the lane
    footprint_along: float          # ground length seen by the camera, along the lane
    gsd: float                      # metres per image pixel
    person_px: float                # person size in detector-input pixels at this altitude
    lane_spacing: float
    lane_axis: str                  # "x" or "y": the direction lanes run in
    speed: float
    speed_limits: dict              # name -> m/s
    waypoints: list = field(default_factory=list)  # [(x, y)] world NED, lane ends in order
    path_length: float = 0.0
    est_time_s: float = 0.0

    @property
    def speed_reason(self):
        return min(self.speed_limits, key=self.speed_limits.get)

    @property
    def lane_count(self):
        return len(self.waypoints) // 2


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def camera_footprint(altitude, hfov_deg, width_px, height_px):
    """Ground (across, along) size seen by a nadir camera; image width is across the lane."""
    half_h = math.tan(math.radians(hfov_deg) / 2)
    half_v = half_h * height_px / width_px
    return 2 * altitude * half_h, 2 * altitude * half_v


def detector_width_px(cam, det):
    """Image width in pixels as the detector sees it.

    Each tile is resized to fit det['input_size'], so a 1280 px image read as
    one tile reaches a 640 px detector at half resolution.
    """
    tile_w = cam["width_px"] / det["tiles_x"]
    tile_h = cam["height_px"] / det["tiles_y"]
    scale = min(1.0, det["input_size"] / max(tile_w, tile_h))
    return cam["width_px"] * scale


def plan_survey(cfg):
    area, cam, det, drone, search = cfg["area"], cfg["camera"], cfg["detector"], cfg["drone"], cfg["search"]

    # 1. Lowest altitude that clears every obstacle in the area.
    min_safe = max(area["max_obstacle_height"] + search["safety_margin"], search["min_altitude"])

    # 2. Highest altitude where a person still covers min_person_px pixels at the detector input:
    #    person_px = person_size / gsd, gsd = footprint_across / width_px, footprint grows linearly with h.
    det_width = detector_width_px(cam, det)
    tan_half = math.tan(math.radians(cam["hfov_deg"]) / 2)
    max_detect = det["person_size_m"] * det_width / (det["min_person_px"] * 2 * tan_half)

    if min_safe > max_detect:
        raise InfeasiblePlan(
            f"the lowest safe altitude ({min_safe:.1f} m) is above the highest altitude where people are "
            f"still detectable ({max_detect:.1f} m). Use more detector tiles, a higher camera resolution "
            f"or a narrower field of view, or lower max_obstacle_height if it's overestimated.")

    # Fly as high as detection allows: wider lanes, fewer turns, shorter mission.
    altitude, altitude_reason = max_detect, "highest altitude where people stay detectable"
    if altitude > search["max_altitude"]:
        altitude, altitude_reason = search["max_altitude"], "max_altitude in config"

    across, along = camera_footprint(altitude, cam["hfov_deg"], cam["width_px"], cam["height_px"])
    gsd = across / cam["width_px"]
    person_px = det["person_size_m"] / (across / det_width)

    # 3. Lanes overlap by side_overlap so a person on a lane edge is fully in at least one image.
    max_spacing = across * (1 - search["side_overlap"])

    # 4. Speed: the slowest of the drone's limit, motion blur and frames per ground point.
    frame_rate = det["fps_per_tile"] / (det["tiles_x"] * det["tiles_y"])
    speed_limits = {
        "drone max speed": drone["max_speed"],
        # Blur: distance moved during one exposure must stay under max_blur_px pixels.
        "motion blur": search["max_blur_px"] * gsd / cam["exposure_s"],
        # Every ground point stays in view for footprint_along / v seconds and must be seen in min_frames frames.
        "frames per person": along * frame_rate / search["min_frames_per_point"],
    }
    speed = min(speed_limits.values())

    # Run lanes along whichever axis gives the shorter mission (usually the long side: fewer turns).
    candidates = [_lanes(area, axis, across, max_spacing) for axis in ("x", "y")]
    best = None
    for axis, waypoints, spacing in candidates:
        length = _path_length(waypoints)
        time_s = length / speed + (len(waypoints) // 2 - 1) * drone["turn_time_s"]
        if best is None or time_s < best[3]:
            best = (axis, waypoints, spacing, time_s, length)
    axis, waypoints, spacing, sweep_time, length = best

    climb_time = altitude / drone["climb_speed"]
    transit = math.dist(area_home(cfg), waypoints[0]) + math.dist(waypoints[-1], area_home(cfg))
    est_time = sweep_time + 2 * climb_time + transit / drone["transit_speed"]

    return SurveyPlan(altitude=altitude, min_safe_altitude=min_safe, max_detect_altitude=max_detect,
                      altitude_reason=altitude_reason, footprint_across=across, footprint_along=along,
                      gsd=gsd, person_px=person_px, lane_spacing=spacing, lane_axis=axis, speed=speed,
                      speed_limits=speed_limits, waypoints=waypoints, path_length=length,
                      est_time_s=est_time)


def area_home(cfg):
    return tuple(cfg["drone"]["spawn_ned"][:2])


def _lanes(area, axis, footprint_across, max_spacing):
    """Lane end points for lanes parallel to `axis`, spread evenly over the area."""
    (x0, y0), (x1, y1) = sorted([tuple(area["corner_a"]), tuple(area["corner_b"])])
    y0, y1 = min(y0, y1), max(y0, y1)
    lo, hi, start, end = (y0, y1, x0, x1) if axis == "x" else (x0, x1, y0, y1)

    width = hi - lo
    if width <= footprint_across:
        offsets, spacing = [(lo + hi) / 2], 0.0
    else:
        # First and last lanes sit half a footprint inside the edges; the rest are evenly spaced.
        n = math.ceil((width - footprint_across) / max_spacing) + 1
        spacing = (width - footprint_across) / (n - 1)
        offsets = [lo + footprint_across / 2 + i * spacing for i in range(n)]

    waypoints = []
    for i, c in enumerate(offsets):
        a, b = (start, end) if i % 2 == 0 else (end, start)
        waypoints += [(a, c), (b, c)] if axis == "x" else [(c, a), (c, b)]
    return axis, waypoints, spacing


def _path_length(waypoints):
    return sum(math.dist(p, q) for p, q in zip(waypoints, waypoints[1:]))


def format_plan(plan):
    limits = "\n".join(f"    {name:<18} {v:6.1f} m/s" for name, v in plan.speed_limits.items())
    return f"""Survey plan
  altitude          {plan.altitude:6.1f} m   ({plan.altitude_reason})
    lowest safe     {plan.min_safe_altitude:6.1f} m   (tallest obstacle + safety margin)
    highest detect  {plan.max_detect_altitude:6.1f} m   (person still >= min_person_px at the detector)
  camera footprint  {plan.footprint_across:6.1f} m across x {plan.footprint_along:.1f} m along
  ground sample     {plan.gsd * 100:6.2f} cm/px, a person is {plan.person_px:.1f} px at the detector
  lanes             {plan.lane_count} lanes along {plan.lane_axis}, {plan.lane_spacing:.1f} m apart
  speed             {plan.speed:6.1f} m/s  (limited by {plan.speed_reason})
{limits}
  path length       {plan.path_length:6.0f} m
  estimated time    {plan.est_time_s / 60:6.1f} min (climb, transit, sweep, turns, return)"""


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    try:
        print(format_plan(plan_survey(load_config(config_path))))
    except InfeasiblePlan as e:
        sys.exit(f"Plan is infeasible: {e}")
