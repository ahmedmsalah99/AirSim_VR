"""Set the search area in config.yaml from two markers placed in the level.

In the Unreal editor, with Play stopped:
  1. Place Actors > search "Target Point", drag one onto each of two opposite
     corners of the area to search.
  2. Select each, Details > Actor > Tags > +, and tag them Area_A and Area_B.
Then press Play and run

    python set_area.py

It reads both markers' positions from AirSim (metres from the PlayerStart, the
config's frame), writes them as area.corner_a / corner_b, points the lanes
along the area's long side (search.lane_axis), and prints the new plan.
"""
import argparse
import math
import re
import sys

from survey_plan import DEFAULT_CONFIG, InfeasiblePlan, format_plan, load_config, plan_survey


def update_config_text(text, corner_a, corner_b, lane_axis):
    """config.yaml's text with the corners and lane axis replaced, comments kept."""
    def put(key, value, text):
        new, n = re.subn(rf"^(  {key}: )[^#\n]*?(\s*(#.*)?)$", lambda m: m.group(1) + value + m.group(2),
                         text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"couldn't find {key} in the config")
        return new

    fmt = lambda p: f"[{p[0]:g}, {p[1]:g}]"  # noqa: E731
    text = put("corner_a", fmt(corner_a), text)
    text = put("corner_b", fmt(corner_b), text)
    return put("lane_axis", lane_axis, text)


def long_side(corner_a, corner_b):
    """The axis ("x" or "y") of the rectangle's longer side."""
    return "x" if abs(corner_b[0] - corner_a[0]) >= abs(corner_b[1] - corner_a[1]) else "y"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="*", default=["Area_A", "Area_B"], help="the two corner markers' tags")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if len(args.tags) != 2:
        sys.exit("give two tags, one per opposite corner")

    import setup_path  # noqa: F401
    import airsim

    cfg = load_config(args.config)
    client = airsim.MultirotorClient(port=cfg["drone"]["port"])
    client.confirmConnection()
    corners = []
    for tag in args.tags:
        p = client.simGetObjectPose(tag).position
        if math.isnan(p.x_val):
            sys.exit(f"no actor tagged {tag!r} found: tag it in Details > Actor > Tags, then press Play again")
        corners.append((round(p.x_val * 2) / 2, round(p.y_val * 2) / 2))  # to 0.5 m
        print(f"{tag}: x={corners[-1][0]:g} y={corners[-1][1]:g}")
    (ax, ay), (bx, by) = corners
    print(f"area {abs(bx - ax):g} m along x by {abs(by - ay):g} m along y")
    axis = long_side(*corners)

    with open(args.config) as f:
        text = f.read()
    with open(args.config, "w") as f:
        f.write(update_config_text(text, corners[0], corners[1], axis))
    print(f"wrote corner_a, corner_b and lane_axis: {axis} (the long side) to {args.config}\n")
    try:
        print(format_plan(plan_survey(load_config(args.config))))
    except InfeasiblePlan as e:
        print(f"Plan is infeasible: {e}")


if __name__ == "__main__":
    main()
