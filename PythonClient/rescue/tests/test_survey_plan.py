import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey_plan import (InfeasiblePlan, detection_range, fov_for_pixels, load_config, plan_at_altitude,  # noqa: E402
                         plan_survey, vertical_fov, view_band)

BASE = load_config()


def cfg(**overrides):
    c = copy.deepcopy(BASE)
    for key, value in overrides.items():
        section, name = key.split("__")
        c[section][name] = value
    return c


def test_nadir_swath_is_the_camera_footprint_when_not_pixel_limited():
    c = cfg(camera__tilt_deg=90)
    band = view_band(10, 90, c["camera"], c["detector"])
    assert band.far_limited_by == "image edge"
    assert band.swath_m == pytest.approx(2 * 10 * math.tan(math.radians(30)))  # 2 h tan(hfov/2)
    assert band.near_m == pytest.approx(-band.far_m)


def test_far_edge_is_where_a_person_shrinks_to_min_px():
    c = cfg(camera__tilt_deg=45)
    band = view_band(20, 45, c["camera"], c["detector"])
    assert band.far_limited_by == "detector pixels"
    assert band.person_px_far == pytest.approx(c["detector"]["min_person_px"])
    slant = math.hypot(20, band.far_m)
    assert slant == pytest.approx(detection_range(c["camera"], c["detector"]))


def test_near_edge_is_the_bottom_of_the_image():
    c = cfg(camera__tilt_deg=45)
    band = view_band(20, 45, c["camera"], c["detector"])
    a_near = math.radians(45) + vertical_fov(c["camera"]) / 2
    assert band.near_m == pytest.approx(20 / math.tan(a_near))


def test_too_high_to_detect_anyone_has_no_band():
    c = cfg()
    r = detection_range(c["camera"], c["detector"])
    assert view_band(r + 1, 45, c["camera"], c["detector"]) is None


def test_obstacles_too_tall_is_infeasible():
    with pytest.raises(InfeasiblePlan):
        plan_survey(cfg(area__max_obstacle_height=100, search__max_altitude=110))


def test_altitude_within_bounds_and_fastest():
    c = cfg(search__max_altitude=30)
    plan = plan_survey(c)
    assert plan.min_safe_altitude <= plan.altitude <= 30
    others = [plan_at_altitude(c, h) for h in (plan.min_safe_altitude, 25, 30)]
    assert all(plan.est_time_s <= o.est_time_s * 1.02 for o in others if o)


def test_speed_is_slowest_limit():
    plan = plan_survey(cfg(drone__max_speed=100, camera__exposure_s=0.01))
    assert plan.speed == min(plan.speed_limits.values())
    assert plan.speed_reason == "motion blur"


def test_lanes_cover_area_across_with_overlap():
    c = cfg(area__corner_a=[-37, 10], area__corner_b=[81, 95])
    plan = plan_survey(c)
    across = 1 if plan.lane_axis == "x" else 0
    offsets = sorted(lane.start[across] for lane in plan.lanes)
    lo, hi = (10, 95) if plan.lane_axis == "x" else (-37, 81)
    half = plan.band.swath_m / 2
    assert offsets[0] - half == pytest.approx(lo)
    assert offsets[-1] + half == pytest.approx(hi)
    for a, b in zip(offsets, offsets[1:]):
        assert b - a <= plan.band.swath_m * (1 - c["search"]["side_overlap"]) + 1e-9


def test_lanes_are_shifted_back_so_the_band_covers_the_area_edges():
    c = cfg(area__corner_a=[0, 0], area__corner_b=[300, 60])
    plan = plan_survey(c)
    assert plan.lane_axis == "x"
    near = plan.band.near_m
    first, second = plan.lanes[0], plan.lanes[1]
    assert (first.start[0], first.end[0], first.heading_deg) == pytest.approx((0 - near, 300 - near, 0))
    assert (second.start[0], second.end[0], second.heading_deg) == pytest.approx((300 + near, 0 + near, 180))


def test_lanes_run_along_long_side_on_auto():
    assert plan_survey(cfg(area__corner_a=[0, 0], area__corner_b=[300, 60], search__lane_axis="auto")).lane_axis == "x"
    plan = plan_survey(cfg(area__corner_a=[0, 0], area__corner_b=[60, 300], search__lane_axis="auto"))
    assert plan.lane_axis == "y"
    assert {lane.heading_deg for lane in plan.lanes} == {90, -90}


def test_zoom_fov_gives_the_wanted_person_size():
    c = cfg()
    fov = fov_for_pixels(c["camera"], c["detector"], distance=25, person_px=40)
    zoomed = dict(c["camera"], hfov_deg=fov)
    assert detection_range(zoomed, dict(c["detector"], min_person_px=40)) == pytest.approx(25)
    assert fov_for_pixels(c["camera"], c["detector"], detection_range(c["camera"], c["detector"]),
                          c["detector"]["min_person_px"]) == pytest.approx(c["camera"]["hfov_deg"])


def test_fixed_lane_axis_gives_more_shorter_lanes_across_a_wide_area():
    wide = dict(area__corner_a=[-25, -60], area__corner_b=[25, 60])
    auto = plan_survey(cfg(search__lane_axis="auto", **wide))
    along_x = plan_survey(cfg(search__lane_axis="x", **wide))
    assert auto.lane_axis == "y" and along_x.lane_axis == "x"
    assert along_x.lane_count > auto.lane_count
    assert {abs(lane.end[0] - lane.start[0]) for lane in along_x.lanes} == {50}


def test_bad_lane_axis_is_rejected():
    with pytest.raises(InfeasiblePlan):
        plan_survey(cfg(search__lane_axis="z"))
