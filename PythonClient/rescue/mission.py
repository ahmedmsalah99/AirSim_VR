"""Search-and-rescue drone mission: search, confirm, publish, resume.

Start Play in Unreal (SimMode "Both"), optionally `python listen_victims.py`
in another terminal, then:

    python mission.py               # full mission
    python mission.py --no-detect   # just fly the search pattern

1. Take off, climb to the planned altitude, tilt the camera (survey_plan.py).
2. Fly the lanes facing forward. Every frame goes through the detector; a
   person's feet pixel is projected onto the (flat) ground to a position.
3. A person not seen before interrupts the lane. The drone turns to face them,
   tilts the camera to the line-of-sight angle, and descends along the line of
   sight (the path it just saw them through) until they are confirm.min_person_px
   tall, but no lower than confirm.min_altitude.
4. It hovers and checks N frames. If the person is found near the same spot in
   k of them, the position measured with the depth camera is published over
   ZeroMQ and marked green in the scene; otherwise the candidate is rejected (red).
5. It climbs back along the same line to where it stopped and carries on.
6. After the last lane it flies home and lands, and prints the position error
   against the tagged people in ground_truth (config.yaml).

Ctrl+C stops the drone in a hover. A summary is saved in output/.
"""
import argparse
from dataclasses import dataclass, field
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import setup_path  # noqa: F401  (puts this repo's airsim package on the path)
import airsim

from camera_geometry import confirm_viewpoint
from drone_camera import STRAIGHT_DOWN, DroneCamera
from survey_plan import (DEFAULT_CONFIG, InfeasiblePlan, detector_scale, focal_px, format_plan, load_config,
                         plan_survey)

OUTPUT_DIR = Path(__file__).with_name("output")
ARRIVE_RADIUS = 1.5
MOVE_MARGIN_S = 15  # extra time a move gets beyond distance / speed before it's given up
LANE_COLOR = [0.0, 1.0, 0.2, 1.0]
AREA_COLOR = [1.0, 0.6, 0.0, 1.0]
CANDIDATE_COLOR = [1.0, 1.0, 0.0, 1.0]
CONFIRMED_COLOR = [0.0, 1.0, 0.0, 1.0]
REJECTED_COLOR = [1.0, 0.0, 0.0, 1.0]


@dataclass
class Candidate:
    id: int
    seen_at: np.ndarray             # first estimate, drone frame, from the flat-ground projection
    confidence: float
    status: str = "pending"         # -> "confirmed" or "rejected"
    position: np.ndarray = None     # confirmed position, drone frame, from the depth camera
    std: np.ndarray = None
    hits: int = 0
    frames: int = 0
    events: list = field(default_factory=list)


class Mission:
    def __init__(self, cfg, plan, detect=True):
        self.cfg, self.plan = cfg, plan
        drone = cfg["drone"]
        self.name = drone["name"]
        self.spawn = np.array(drone["spawn_ned"], float)
        self.client = airsim.MultirotorClient(port=drone["port"])
        self.camera = DroneCamera(self.client, airsim, cfg)
        self.detector = self.publisher = None
        if detect:
            from detector import PersonDetector
            from publisher import VictimPublisher
            self.detector = PersonDetector(cfg["detector"])
            self.publisher = VictimPublisher(cfg["publish"]["endpoint"], cfg["publish"]["topic"])
        self.candidates = []
        self.events = []
        self.t0 = time.time()

    # ---- bookkeeping -------------------------------------------------------------------

    def log(self, msg):
        t = time.time() - self.t0
        self.events.append((round(t, 1), msg))
        print(f"[{t:7.1f}s] {msg}", flush=True)

    def to_local(self, xy):
        return np.asarray(xy, float) - self.spawn[:2]

    def to_world(self, p):
        """Drone frame -> world NED (relative to the PlayerStart)."""
        return np.asarray(p, float) + self.spawn

    def position(self):
        p = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    def in_area(self, p):
        (ax, ay), (bx, by) = self.cfg["area"]["corner_a"], self.cfg["area"]["corner_b"]
        x, y = self.to_world(p)[:2]
        margin = self.cfg["confirm"]["merge_radius"]
        return min(ax, bx) - margin <= x <= max(ax, bx) + margin and min(ay, by) - margin <= y <= max(ay, by) + margin

    # ---- mission -----------------------------------------------------------------------

    def run(self):
        c, plan = self.client, self.plan
        c.confirmConnection()
        c.enableApiControl(True, self.name)
        c.armDisarm(True, self.name)

        # Altitudes are measured from the ground under the takeoff point.
        state = c.getMultirotorState(vehicle_name=self.name)
        self.ground_z = self.position()[2]
        if state.landed_state != airsim.LandedState.Landed:
            self.camera.set_tilt(STRAIGHT_DOWN)
            self.ground_z += self.camera.center_depth()
            self.log(f"already flying, {self.ground_z - self.position()[2]:.1f} m above the ground")
        self.z = self.ground_z - plan.altitude
        self.draw_plan()
        if not self.camera.set_tilt(plan.tilt_deg):
            raise RuntimeError(f"couldn't tilt the camera {plan.tilt_deg} deg down")

        try:
            self.log("taking off")
            c.takeoffAsync(vehicle_name=self.name).join()
            heading = plan.lanes[0].heading_deg
            self.log(f"climbing to {plan.altitude:.1f} m")
            climb_s = plan.altitude / self.cfg["drone"]["climb_speed"] * 2 + MOVE_MARGIN_S
            c.moveToZAsync(self.z, self.cfg["drone"]["climb_speed"], timeout_sec=climb_s,
                           yaw_mode=airsim.YawMode(False, heading), vehicle_name=self.name).join()

            for i, lane in enumerate(plan.lanes):
                self.log(f"to the start of lane {i + 1}/{plan.lane_count}")
                self.goto(self.at_altitude(lane.start), self.cfg["drone"]["transit_speed"], lane.heading_deg,
                          what=f"transit to lane {i + 1}")
                self.log(f"lane {i + 1}/{plan.lane_count} at {plan.speed:.1f} m/s, heading {lane.heading_deg:.0f} deg")
                self.sweep(self.at_altitude(lane.end), plan.speed, lane.heading_deg)

            self.log("search complete, returning home")
            self.goto(self.at_altitude(self.spawn[:2]), self.cfg["drone"]["transit_speed"], heading=None,
                      what="return home")
            c.landAsync(vehicle_name=self.name).join()
            c.armDisarm(False, self.name)
            self.log("landed")
        except KeyboardInterrupt:
            self.log("stopped by user: hovering")
            c.cancelLastTask(self.name)
            c.hoverAsync(vehicle_name=self.name).join()
        finally:
            self.camera.set_tilt(STRAIGHT_DOWN)
            self.report()

    def at_altitude(self, xy_world):
        x, y = self.to_local(xy_world)
        return np.array([x, y, self.z])

    def goto(self, target, speed, heading, what="move"):
        """Fly to target and wait, but no longer than twice the expected time plus a margin.

        AirSim's moves never finish if the drone can't reach the point (blocked by a tree or a
        roof, or after bumping into something), which would hang the mission. Returns whether
        the drone got there.
        """
        yaw = airsim.YawMode(False, heading) if heading is not None else airsim.YawMode()
        timeout = 2 * np.linalg.norm(np.asarray(target) - self.position()) / speed + MOVE_MARGIN_S
        self.client.moveToPositionAsync(*target, speed, timeout_sec=timeout, yaw_mode=yaw,
                                        vehicle_name=self.name).join()
        missed = np.linalg.norm(np.asarray(target) - self.position())
        if missed > ARRIVE_RADIUS:
            self.log(f"WARNING: {what} stopped {missed:.1f} m short (blocked?); carrying on from here")
            return False
        return True

    def face(self, heading):
        self.client.rotateToYawAsync(heading, timeout_sec=10, vehicle_name=self.name).join()

    def sweep(self, target, speed, heading):
        """Fly to target facing heading, checking every frame for people; confirm each new one."""
        if self.detector is None:
            self.goto(target, speed, heading)
            return
        c = self.client
        self.face(heading)
        c.moveToPositionAsync(*target, speed, yaw_mode=airsim.YawMode(False, heading), vehicle_name=self.name)
        deadline = time.time() + 2 * np.linalg.norm(target - self.position()) / speed + 30
        while np.linalg.norm(self.position()[:2] - target[:2]) > ARRIVE_RADIUS:
            if time.time() > deadline:
                self.log("WARNING: lane took too long; moving on")
                break
            candidate = self.scan()
            if candidate is None:
                continue
            self.confirm(candidate)
            self.face(heading)
            c.moveToPositionAsync(*target, speed, yaw_mode=airsim.YawMode(False, heading), vehicle_name=self.name)
            deadline = time.time() + 2 * np.linalg.norm(target - self.position()) / speed + 30

    def scan(self):
        """One frame: the first person not seen before, as a new Candidate, or None."""
        frame = self.camera.grab()
        for det in self.detector.detect(frame.image):
            feet = self.camera.ground_point(frame, det.center[0], det.y2, self.ground_z)
            if feet is None or not self.in_area(feet) or self.known(feet):
                continue
            cand = Candidate(len(self.candidates) + 1, feet, det.confidence)
            self.candidates.append(cand)
            w = self.to_world(feet)
            self.log(f"candidate {cand.id} at x={w[0]:.1f} y={w[1]:.1f} (confidence {det.confidence:.2f})")
            self.mark(feet, CANDIDATE_COLOR, f"?{cand.id}")
            return cand
        return None

    def known(self, p):
        r = self.cfg["confirm"]["merge_radius"]
        for cand in self.candidates:
            ref = cand.position if cand.position is not None else cand.seen_at
            if np.linalg.norm(ref[:2] - p[:2]) < r:
                return True
        return False

    def confirm(self, cand):
        """Approach along the line of sight, check k of N frames, publish, and climb back."""
        c, conf = self.client, self.cfg["confirm"]
        stop = self.position()
        c.hoverAsync(vehicle_name=self.name).join()

        cam, det = self.cfg["camera"], self.cfg["detector"]
        distance = det["person_size_m"] * focal_px(cam) * detector_scale(cam, det) / conf["min_person_px"]
        point, heading, tilt = confirm_viewpoint(stop, cand.seen_at, distance, self.ground_z - conf["min_altitude"])
        self.log(f"confirming {cand.id}: line of sight {tilt:.0f} deg down, approaching to "
                 f"{np.linalg.norm(point - cand.seen_at):.1f} m from it, {self.ground_z - point[2]:.1f} m up")

        self.face(heading)
        if not self.camera.set_tilt(tilt):
            self.log("WARNING: couldn't tilt the camera to the line of sight; keeping the search tilt")
        self.goto(point, conf["speed"], heading, what="approach")
        c.hoverAsync(vehicle_name=self.name).join()
        time.sleep(1)  # settle so frames aren't tilted

        positions, confidences = [], []
        for i in range(conf["frames"]):
            frame = self.camera.grab(with_depth=True)
            dets = self.detector.detect(frame.image)
            hit = None
            for d in dets:
                u, v = d.center[0], d.y2
                p = self.camera.depth_point(frame, u, v)
                if p is None:
                    p = self.camera.ground_point(frame, u, v, self.ground_z)
                if p is not None and np.linalg.norm(p[:2] - cand.seen_at[:2]) < conf["match_radius"]:
                    positions.append(p)
                    confidences.append(d.confidence)
                    hit = np.linalg.norm(p[:2] - cand.seen_at[:2])
                    break
            self.log(f"  frame {i + 1}/{conf['frames']}: " + (f"hit, {hit:.1f} m from the candidate" if hit is not None
                                                             else f"miss ({len(dets)} person(s) elsewhere)"))
            time.sleep(conf["frame_interval_s"])

        cand.hits, cand.frames = len(positions), conf["frames"]
        if cand.hits >= conf["required"]:
            pts = np.array(positions)
            cand.status, cand.position = "confirmed", np.median(pts, axis=0)
            cand.std = np.maximum(pts.std(axis=0), 0.05)
            w = self.to_world(cand.position)
            self.log(f"CONFIRMED victim {cand.id} at x={w[0]:.1f} y={w[1]:.1f} ({cand.hits}/{cand.frames} frames)")
            self.publisher.publish(cand.id, w, cand.std, float(np.mean(confidences)), cand.hits, cand.frames)
            self.mark(cand.position, CONFIRMED_COLOR, f"victim {cand.id}")
        else:
            cand.status = "rejected"
            self.log(f"rejected candidate {cand.id}: found in {cand.hits}/{cand.frames} frames, "
                     f"{conf['required']} needed")
            self.mark(cand.seen_at, REJECTED_COLOR, f"x{cand.id}")

        self.log("climbing back to the lane")
        self.goto(stop, conf["speed"], heading, what="climb back")
        if not self.camera.set_tilt(self.plan.tilt_deg):
            self.log("WARNING: couldn't restore the camera tilt")

    # ---- drawing and reporting ------------------------------------------------------------

    def draw_plan(self):
        ground = self.ground_z + self.spawn[2]
        z = ground - self.plan.altitude
        lanes = [airsim.Vector3r(x, y, z) for x, y in self.plan.waypoints]
        (ax, ay), (bx, by) = self.cfg["area"]["corner_a"], self.cfg["area"]["corner_b"]
        area = [airsim.Vector3r(x, y, ground - 0.5) for x, y in [(ax, ay), (ax, by), (bx, by), (bx, ay), (ax, ay)]]
        self.client.simFlushPersistentMarkers()
        self.client.simPlotLineStrip(area, color_rgba=AREA_COLOR, thickness=15, is_persistent=True)
        self.client.simPlotLineStrip(lanes, color_rgba=LANE_COLOR, thickness=8, is_persistent=True)

    def mark(self, p, color, label):
        w = self.to_world(p)
        z = self.ground_z + self.spawn[2] - 1
        self.client.simPlotPoints([airsim.Vector3r(w[0], w[1], z)], color_rgba=color, size=25, is_persistent=True)
        self.client.simPlotStrings([label], [airsim.Vector3r(w[0], w[1], z - 2)], scale=3,
                                   color_rgba=color, duration=3600)

    def report(self):
        confirmed = [c for c in self.candidates if c.status == "confirmed"]
        truth = self.ground_truth()
        print(f"\n{len(confirmed)} victim(s) confirmed, "
              f"{sum(c.status == 'rejected' for c in self.candidates)} candidate(s) rejected")
        rows = []
        for c in confirmed:
            w = self.to_world(c.position)
            nearest = min(truth.items(), key=lambda kv: math.dist(kv[1], w[:2]), default=None)
            err = math.dist(nearest[1], w[:2]) if nearest else None
            rows.append({"id": c.id, "x": round(w[0], 2), "y": round(w[1], 2), "hits": c.hits,
                         "nearest_truth": nearest[0] if nearest else None,
                         "error_m": round(err, 2) if err is not None else None})
            print(f"  victim {c.id}: x={w[0]:.1f} y={w[1]:.1f}"
                  + (f"  -> {nearest[0]}, error {err:.2f} m" if nearest else ""))
        found = {r["nearest_truth"] for r in rows if r["error_m"] is not None and r["error_m"] < 3}
        for tag in truth:
            if tag not in found:
                print(f"  {tag}: not found")

        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / f"mission_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps({
            "plan": {"altitude": self.plan.altitude, "tilt_deg": self.plan.tilt_deg,
                     "lanes": self.plan.lane_count, "speed": self.plan.speed},
            "victims": rows,
            "candidates": [{"id": c.id, "status": c.status, "hits": c.hits, "frames": c.frames,
                            "seen_at": [round(v, 2) for v in self.to_world(c.seen_at)]} for c in self.candidates],
            "ground_truth": {k: list(v) for k, v in truth.items()},
            "events": self.events,
        }, indent=2))
        print(f"saved {out}")

    def ground_truth(self):
        truth = {}
        for tag in self.cfg.get("ground_truth") or []:
            p = self.client.simGetObjectPose(tag).position
            if not math.isnan(p.x_val):
                truth[tag] = (p.x_val, p.y_val)
        return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--no-detect", action="store_true", help="fly the search pattern without detecting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        plan = plan_survey(cfg)
    except InfeasiblePlan as e:
        sys.exit(f"Plan is infeasible: {e}")
    print(format_plan(plan))
    Mission(cfg, plan, detect=not args.no_detect).run()


if __name__ == "__main__":
    main()
