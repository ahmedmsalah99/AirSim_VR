import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from set_area import long_side, update_config_text  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_corners_and_lane_axis_are_replaced_and_comments_kept():
    text = CONFIG.read_text()
    new = update_config_text(text, (-31.5, 12), (44, -150.5), "y")
    cfg = yaml.safe_load(new)
    assert cfg["area"]["corner_a"] == [-31.5, 12] and cfg["area"]["corner_b"] == [44, -150.5]
    assert cfg["search"]["lane_axis"] == "y"
    assert "# any two opposite corners" in new and "# lanes run along x or y" in new
    assert len(new.splitlines()) == len(text.splitlines())


def test_lanes_follow_the_long_side():
    assert long_side((0, 0), (100, 40)) == "x"
    assert long_side((0, 0), (-40, -100)) == "y"
