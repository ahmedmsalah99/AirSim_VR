import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survey_plan import InfeasiblePlan, camera_footprint, load_config, plan_survey  # noqa: E402

BASE = load_config()


def cfg(**overrides):
    c = copy.deepcopy(BASE)
    for key, value in overrides.items():
        section, name = key.split("__")
        c[section][name] = value
    return c


def test_footprint_matches_hand_calculation():
    across, along = camera_footprint(25, 60, 1280, 720)
    assert across == pytest.approx(28.87, abs=0.01)
    assert along == pytest.approx(across * 720 / 1280)


def test_person_is_exactly_min_px_at_max_detect_altitude():
    plan = plan_survey(cfg(search__max_altitude=1000))
    assert plan.altitude == pytest.approx(plan.max_detect_altitude)
    assert plan.person_px == pytest.approx(BASE["detector"]["min_person_px"])


def test_altitude_capped_by_config():
    plan = plan_survey(cfg(search__max_altitude=30))
    assert plan.altitude == 30
    assert plan.person_px > BASE["detector"]["min_person_px"]


def test_more_tiles_allow_higher_altitude():
    one = plan_survey(cfg(detector__tiles_x=1, area__max_obstacle_height=5))
    two = plan_survey(cfg(detector__tiles_x=2, area__max_obstacle_height=5))
    assert two.max_detect_altitude > one.max_detect_altitude


def test_obstacles_above_detect_altitude_is_infeasible():
    with pytest.raises(InfeasiblePlan):
        plan_survey(cfg(area__max_obstacle_height=100))


def test_speed_is_slowest_limit():
    plan = plan_survey(cfg(drone__max_speed=100, camera__exposure_s=0.01))
    assert plan.speed == min(plan.speed_limits.values())
    assert plan.speed_reason == "motion blur"


def test_lanes_cover_whole_area_with_overlap():
    c = cfg(area__corner_a=[-37, 10], area__corner_b=[81, 95])
    plan = plan_survey(c)
    half = plan.footprint_across / 2
    offsets = sorted({p[1] if plan.lane_axis == "x" else p[0] for p in plan.waypoints})
    lo, hi = (10, 95) if plan.lane_axis == "x" else (-37, 81)
    assert offsets[0] - half == pytest.approx(lo)
    assert offsets[-1] + half == pytest.approx(hi)
    for a, b in zip(offsets, offsets[1:]):
        assert b - a <= plan.footprint_across * (1 - c["search"]["side_overlap"]) + 1e-9


def test_lanes_run_along_long_side():
    plan = plan_survey(cfg(area__corner_a=[0, 0], area__corner_b=[300, 60]))
    assert plan.lane_axis == "x"
    plan = plan_survey(cfg(area__corner_a=[0, 0], area__corner_b=[60, 300]))
    assert plan.lane_axis == "y"


def test_waypoints_alternate_direction():
    plan = plan_survey(BASE)
    for i in range(0, len(plan.waypoints) - 2, 2):
        start, end, next_start = plan.waypoints[i], plan.waypoints[i + 1], plan.waypoints[i + 2]
        axis = 0 if plan.lane_axis == "x" else 1
        assert next_start[axis] == end[axis]
        assert math.copysign(1, end[axis] - start[axis]) != math.copysign(1, plan.waypoints[i + 3][axis] - next_start[axis])
