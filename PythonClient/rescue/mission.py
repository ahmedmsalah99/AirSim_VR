"""Search-and-rescue drone mission: search, confirm, publish, resume.

Start Play in Unreal (SimMode "Both"), optionally `python listen_victims.py`
in another terminal, then:

    python mission.py               # full mission
    python mission.py --no-detect   # just fly the search pattern
    python mission.py --victim Victim_01   # test one person: view, detect, confirm, land

1. Take off, climb to the planned altitude, tilt the camera (survey_plan.py).
2. Fly the lanes facing forward. Every frame goes through the detector; a
   person's feet pixel is projected onto the (flat) ground to a position.
3. A person not seen before interrupts the lane. The drone turns to face them,
   tilts the camera to the line-of-sight angle, and descends along the line of
   sight (the path it just saw them through) towards where they'd be
   confirm.min_person_px tall, but never below confirm.min_altitude. The camera
   zooms in for the rest.
4. It hovers and checks N frames. If the person is found near the same spot in
   k of them, their position is published over ZeroMQ and marked green in the
   scene; otherwise the candidate is rejected (red).
5. It zooms back out, climbs back along the same line to where it stopped and
   carries on.
6. After the last lane it flies home and lands, and prints the position error
   against the tagged people in ground_truth (config.yaml).

Ctrl+C (or an error) ends the mission early: the drone flies home and lands; a
second Ctrl+C lands it where it is. A summary is saved in output/.
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

from camera_geometry import arc_point, bearing_deg, confirm_viewpoint
from drone_camera import STRAIGHT_DOWN, DroneCamera
from survey_plan import (DEFAULT_CONFIG, InfeasiblePlan, detector_scale, focal_px, format_plan, fov_for_pixels,
                         load_config, plan_survey)

OUTPUT_DIR = Path(__file__).with_name("output")
ARRIVE_RADIUS = 1.5
LANDED_HEIGHT_M = 0.5
SLOW_DESCENT_M = 1.5      # the last bit of a landing is flown slowly
FAST_DESCENT_SPEED = 3    # m/s above that
MIN_SEARCH_SPEED = 0.3  # m/s; slower than this the search would never finish
REPORT_EVERY_S = 5  # how often the search logs what it has been detecting
STILL_SPEED = 0.3   # m/s
MOVE_MARGIN_S = 15  # extra time a move gets beyond distance / speed before it's given up
LANE_COLOR = [0.0, 1.0, 0.2, 1.0]
AREA_COLOR = [1.0, 0.6, 0.0, 1.0]
CONFIRMED_COLOR = [0.0, 1.0, 0.0, 1.0]
REJECTED_COLOR = [1.0, 0.0, 0.0, 1.0]
MARKER_HEIGHT_M = 6


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
    attempts: int = 0
    last_stop: np.ndarray = None    # where the drone left its lane for the last attempt
    events: list = field(default_factory=list)


class Mission:
    def __init__(self, cfg, plan, detect=True, victim=None):
        self.cfg, self.plan, self.victim = cfg, plan, victim
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
        self.speed = plan.speed
        self.scans = {"frames": 0, "people": 0}
        self.image_dir = OUTPUT_DIR / f"mission_{time.strftime('%Y%m%d_%H%M%S')}"
        if getattr(self.detector, "device", None):
            self.log(f"detector on {self.detector.device}: {self.detector.seconds_per_tile:.2f} s per tile")

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
        if self.victim:
            return True  # testing one person, who needn't be inside the search area
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

        # Altitudes, ground positions and the landing all use the ground under the take-off
        # point. Measure it with the depth camera instead of assuming the drone starts on it
        # (it can be held at its spawn height): a wrong ground puts every detected person's
        # feet in the wrong place and lands the drone in mid-air.
        self.log("measuring the ground below with the depth camera")
        start = time.time()
        below = self.height_above_ground()
        if time.time() - start > 5:
            self.log(f"WARNING: that took {time.time() - start:.0f} s; depth images are probably still "
                     f"1280x720. Copy the repo's settings.json to Documents\\AirSim (depth is 320x180 there)")
        if below is None:
            below = 0.0
            self.log("WARNING: couldn't measure the ground below; assuming the drone is on it")
        self.ground_z = self.position()[2] + below
        self.log(f"ground is {below:.1f} m below the drone")
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

            for i, lane in (() if self.victim else enumerate(plan.lanes)):
                self.log(f"to the start of lane {i + 1}/{plan.lane_count}")
                if i == 0:
                    self.goto(self.at_altitude(lane.start), self.cfg["drone"]["transit_speed"], lane.heading_deg,
                              what=f"transit to lane {i + 1}")
                else:
                    # A short hop to the next lane: stop at the end of the last one first (a new
                    # move given at search speed carried the drone on past the corner), then hop at
                    # turn_speed holding the heading; sweep() turns to the new lane in place.
                    self.wait_until_still()
                    self.goto(self.at_altitude(lane.start), min(self.speed, self.cfg["drone"]["turn_speed"]),
                              heading=None, what=f"transit to lane {i + 1}")
                if i == 0:
                    self.speed = self.speed_for_frame_rate()
                self.log(f"lane {i + 1}/{plan.lane_count} at {self.speed:.1f} m/s, heading {lane.heading_deg:.0f} deg")
                self.sweep(self.at_altitude(lane.end), self.speed, lane.heading_deg)

            if self.victim:
                self.check_victim(self.victim)
            else:
                self.log("search complete")
            self.return_and_land()
        except KeyboardInterrupt:
            self.log("stopped by user")
            self.reconnect()
            self.abort()
        except Exception as e:
            self.log(f"ERROR: {e!r}")
            try:
                self.abort()
            except Exception as again:  # e.g. the connection to AirSim is gone
                self.log(f"couldn't return home either: {again!r}")
            raise
        finally:
            try:
                self.camera.set_fov(self.cfg["camera"]["hfov_deg"])
                self.camera.set_tilt(STRAIGHT_DOWN)
            except Exception as e:
                self.log(f"couldn't reset the camera: {e!r}")
            self.report()

    def check_victim(self, tag, max_frames=10):
        """Test the detection and confirmation on one tagged person: fly to where a lane would
        see them (mid detection band, at the search altitude and tilt, facing them, coming from
        the drone's side), look until they're detected, then confirm them as the search would."""
        p = self.client.simGetObjectPose(tag).position
        if math.isnan(p.x_val):
            raise RuntimeError(f"no actor tagged {tag!r} in the level (Details > Actor > Tags)")
        target = np.append(self.to_local((p.x_val, p.y_val)), self.ground_z)
        here = self.position()
        away = here[:2] - target[:2]
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.array([-1.0, 0.0])
        ahead = (self.plan.band.near_m + self.plan.band.far_m) / 2
        view = np.append(target[:2] + away * ahead, self.z)
        heading = bearing_deg(view, target)
        self.log(f"to the search view of {tag}: {ahead:.0f} m back from it, {self.plan.altitude:.0f} m up, "
                 f"facing it (heading {heading:.0f} deg)")
        self.goto(view, self.cfg["drone"]["transit_speed"], heading, what=f"move to view {tag}")
        self.wait_until_still()
        for i in range(max_frames):
            candidate = self.scan()
            if candidate is not None:
                self.confirm(candidate)
                return
        self.report_scans()
        self.log(f"{tag} wasn't detected in {max_frames} frames from the search view")

    def return_and_land(self):
        self.log("returning home")
        self.goto(self.at_altitude(self.spawn[:2]), self.cfg["drone"]["transit_speed"], heading=None,
                  what="return home")
        self.land()

    def land(self):
        """Land, and check it: landAsync can return with the drone still in the air."""
        c = self.client
        c.landAsync(timeout_sec=60, vehicle_name=self.name).join()
        height = self.height_above_ground()
        if height is not None and height > LANDED_HEIGHT_M:
            self.log(f"land returned {height:.1f} m above the ground; descending the rest")
            if height > SLOW_DESCENT_M:  # fast down to just above the ground, then gently
                c.moveToZAsync(self.position()[2] + height - SLOW_DESCENT_M, FAST_DESCENT_SPEED,
                               timeout_sec=height / FAST_DESCENT_SPEED * 2 + MOVE_MARGIN_S, vehicle_name=self.name).join()
                height = self.height_above_ground() or SLOW_DESCENT_M
            c.moveToZAsync(self.position()[2] + height, 1, timeout_sec=height * 2 + MOVE_MARGIN_S,
                           vehicle_name=self.name).join()
            c.landAsync(timeout_sec=30, vehicle_name=self.name).join()
            height = self.height_above_ground()
        if height is not None and height > LANDED_HEIGHT_M:
            self.log(f"WARNING: still {height:.1f} m above the ground")
        c.armDisarm(False, self.name)
        self.log("landed" + (f" ({max(height, 0):.1f} m above the ground)" if height is not None else ""))

    def height_above_ground(self):
        """Height above whatever is straight below, from the depth camera, or None."""
        if not self.camera.set_tilt(STRAIGHT_DOWN):
            return None
        d = self.camera.center_depth()
        return d if 0 <= d < 200 else None

    def wait_until_still(self, timeout_s=6):
        """Hover until the drone has stopped moving, so frames aren't taken mid-sway."""
        self.client.hoverAsync(vehicle_name=self.name).join()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            v = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.linear_velocity
            if math.hypot(v.x_val, v.y_val, v.z_val) < STILL_SPEED:
                return True
            time.sleep(0.2)
        self.log("WARNING: still moving after hovering; taking the frames anyway")
        return False

    def save(self, name, frame, detections):
        """Save a frame with its boxes for inspection; never let that stop the flight."""
        if not self.cfg["confirm"].get("save_images", True):
            return
        try:
            import cv2
            from detector import draw
            self.image_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.image_dir / f"{name}.jpg"), draw(frame.image, detections))
        except Exception as e:
            self.log(f"WARNING: couldn't save {name}: {e!r}")

    def abort(self):
        """Stopped early: fly home at the current altitude and land there (not into whatever is
        below). A second Ctrl+C while it's on the way lands it where it is."""
        self.client.cancelLastTask(self.name)
        try:
            self.return_and_land()
        except KeyboardInterrupt:
            self.log("stopped again: landing here")
            self.reconnect()
            self.client.cancelLastTask(self.name)
            self.land()

    def reconnect(self):
        """A Ctrl+C in the middle of an AirSim call leaves the client unusable ("IOLoop is
        already running"), so open a new connection before flying home."""
        self.client = airsim.MultirotorClient(port=self.cfg["drone"]["port"])
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.camera.client = self.client

    def frame_rate(self, frames=3):
        """Search frames per second actually achieved: capture plus detection, timed."""
        self.camera.grab()  # the first frame includes one-off setup
        capture = detection = 0.0
        for _ in range(frames):
            start = time.time()
            image = self.camera.grab().image
            capture += time.time() - start
            start = time.time()
            self.detector.detect(image)
            detection += time.time() - start
        self.log(f"per frame: capture {capture / frames:.2f} s (AirSim), detection {detection / frames:.2f} s (YOLO)")
        return frames / max(capture + detection, 1e-6)

    def speed_for_frame_rate(self):
        """The planned speed, or slower if frames come too slowly for it.

        A ground point stays in the detection band for (far - near) / v seconds and has to be
        in min_frames_per_point frames. The plan assumes detector.fps_per_tile; if capture and
        detection are slower in practice (e.g. Unreal throttled in the background), people
        would pass between frames unseen.
        """
        if self.detector is None:
            return self.plan.speed
        fps = self.frame_rate()
        band = self.plan.band.far_m - self.plan.band.near_m
        limit = band * fps / self.cfg["search"]["min_frames_per_point"]
        self.log(f"search frames: {fps:.1f} per second")
        if limit >= self.plan.speed:
            return self.plan.speed
        self.log(f"WARNING: too slow for {self.plan.speed:.1f} m/s; searching at {limit:.1f} m/s so every point "
                 f"is in {self.cfg['search']['min_frames_per_point']} frames. If Unreal isn't the active window, "
                 f"turn off Edit > Editor Preferences > Performance > 'Use Less CPU when in Background'")
        return max(limit, MIN_SEARCH_SPEED)

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
        last_report = time.time()
        while np.linalg.norm(self.position()[:2] - target[:2]) > ARRIVE_RADIUS:
            if time.time() > deadline:
                self.log("WARNING: lane took too long; moving on")
                break
            candidate = self.scan()
            if time.time() - last_report > REPORT_EVERY_S:
                self.report_scans()
                last_report = time.time()
            if candidate is None:
                continue
            self.confirm(candidate)
            self.face(heading)
            c.moveToPositionAsync(*target, speed, yaw_mode=airsim.YawMode(False, heading), vehicle_name=self.name)
            deadline = time.time() + 2 * np.linalg.norm(target - self.position()) / speed + 30
        if self.scans["frames"]:
            self.report_scans()

    def scan(self):
        """One frame: the first person not seen before, as a new Candidate, or None."""
        frame = self.camera.grab()
        detections = self.detector.detect(frame.image)
        self.scans["frames"] += 1
        self.scans["people"] += len(detections)
        min_conf = self.cfg["search"]["min_candidate_confidence"]
        for det in detections:
            if det.confidence < min_conf:
                reason = f"below {min_conf} confidence"
                self.scans[reason] = self.scans.get(reason, 0) + 1
                continue
            feet = self.camera.ground_point(frame, det.center[0], det.y2, self.ground_z)
            if feet is None or not self.in_area(feet):
                reason = "not on the ground" if feet is None else "outside the area"
                self.scans[reason] = self.scans.get(reason, 0) + 1
                continue
            near = self.nearby(feet)
            if near is not None:
                if self.can_retry(near):
                    near.seen_at = feet
                    w = self.to_world(feet)
                    self.log(f"candidate {near.id} (rejected before) seen again at x={w[0]:.1f} y={w[1]:.1f} "
                             f"from a new viewpoint; trying again")
                    self.save(f"candidate{near.id}_search{near.attempts + 1}", frame, [det])
                    return near
                self.scans["already known"] = self.scans.get("already known", 0) + 1
                continue
            cand = Candidate(len(self.candidates) + 1, feet, det.confidence)
            self.candidates.append(cand)
            w = self.to_world(feet)
            self.log(f"candidate {cand.id} at x={w[0]:.1f} y={w[1]:.1f} (confidence {det.confidence:.2f})")
            self.save(f"candidate{cand.id}_search", frame, [det])
            return cand
        return None

    def report_scans(self):
        """Log what the search frames since the last report saw, including what was dropped and why."""
        s = self.scans
        dropped = ", ".join(f"{n} {k}" for k, n in s.items() if k not in ("frames", "people"))
        self.log(f"  {s['frames']} frames, {s['people']} person detection(s)" + (f" (dropped: {dropped})" if dropped else ""))
        self.scans = {"frames": 0, "people": 0}

    def nearby(self, p):
        """The known candidate within merge_radius of p, or None."""
        r = self.cfg["confirm"]["merge_radius"]
        for cand in self.candidates:
            ref = cand.position if cand.position is not None else cand.seen_at
            if np.linalg.norm(ref[:2] - p[:2]) < r:
                return cand
        return None

    def can_retry(self, cand):
        """A rejected candidate gets another attempt (up to max_attempts) once it's seen again from
        retry_distance or more from where it was last tried: from there the view may be clear."""
        conf = self.cfg["confirm"]
        return (cand.status == "rejected" and cand.attempts < conf["max_attempts"]
                and np.linalg.norm(self.position()[:2] - cand.last_stop[:2]) >= conf["retry_distance"])

    def confirm(self, cand):
        """Approach along the line of sight (never below min_altitude) and zoom, then look from
        the approach point and, if the view is blocked, from points along an arc around the person
        (same distance and height, still facing them) until one viewpoint sees them in `required`
        of `frames` frames. Publish or reject, then go back along the arc and the line of sight
        to where the lane was left."""
        c, conf = self.client, self.cfg["confirm"]
        cand.attempts += 1
        cand.status = "pending"
        stop = cand.last_stop = self.position()
        c.hoverAsync(vehicle_name=self.name).join()

        # Descend along the line of sight towards the distance where the person is min_person_px,
        # but never below min_altitude: there's no obstacle map, and trees and roofs stand around
        # people. Zooming the camera makes up the rest.
        cam, det = self.cfg["camera"], self.cfg["detector"]
        distance = det["person_size_m"] * focal_px(cam) * detector_scale(cam, det) / conf["min_person_px"]
        lowest_z = self.ground_z - conf["min_altitude"]
        point, heading, tilt = confirm_viewpoint(stop, cand.seen_at, distance, lowest_z)
        range_m = float(np.linalg.norm(point - cand.seen_at))
        fov = min(cam["hfov_deg"], max(conf["min_fov_deg"], fov_for_pixels(cam, det, range_m, conf["min_person_px"])))
        moves = np.linalg.norm(point - stop) > ARRIVE_RADIUS
        if not moves:
            point = stop
        self.log(f"confirming {cand.id}" + (f" (attempt {cand.attempts})" if cand.attempts > 1 else "")
                 + f": line of sight {tilt:.0f} deg down, "
                 + (f"approaching to {range_m:.1f} m from it, {self.ground_z - point[2]:.1f} m up"
                    if moves else f"staying at {range_m:.1f} m from it")
                 + (f", zooming to {fov:.0f} deg" if fov < cam["hfov_deg"] else ""))

        self.face(heading)
        if not self.camera.set_tilt(tilt):
            self.log("WARNING: couldn't tilt the camera to the line of sight; keeping the search tilt")
        if moves:
            self.goto(point, conf["speed"], heading, what="approach")
        if fov < cam["hfov_deg"]:
            self.camera.set_fov(fov)

        positions, confidences, frames, angle, confirmed = [], [], 0, 0.0, False
        for offset in conf["arc_offsets_deg"]:
            if offset != angle:
                self.log(f"  view blocked? moving {offset:+.0f} deg around it")
                self.arc(cand.seen_at, point, angle, offset)
                angle = offset
            self.wait_until_still()
            hits, confs = self.look(cand, f"{offset:+.0f}")
            positions += hits
            confidences += confs
            frames += conf["frames"]
            if len(hits) >= conf["required"]:
                confirmed = True
                break

        cand.hits, cand.frames = len(positions), frames
        if confirmed:
            pts = np.array(positions)
            cand.status, cand.position = "confirmed", np.median(pts, axis=0)
            cand.std = np.maximum(pts.std(axis=0), 0.05)
            measured = self.depth_position(cand.position, conf["match_radius"]) if conf["use_depth"] else None
            if measured is not None:
                cand.position = measured
            source = "depth camera" if measured is not None else f"ground projection, median of {cand.hits}"
            w = self.to_world(cand.position)
            self.log(f"CONFIRMED victim {cand.id} at x={w[0]:.1f} y={w[1]:.1f} ({len(hits)}/{conf['frames']} frames "
                     f"from the {angle:+.0f} deg view, position from the {source})")
            self.publisher.publish(cand.id, w, cand.std, float(np.mean(confidences)), cand.hits, cand.frames)
            self.mark(cand.position, CONFIRMED_COLOR, f"victim {cand.id}")
        else:
            cand.status = "rejected"
            self.log(f"rejected candidate {cand.id}: no viewpoint saw it in {conf['required']} of "
                     f"{conf['frames']} frames ({cand.hits} hits in {cand.frames} frames)")
            self.mark(cand.seen_at, REJECTED_COLOR, f"x{cand.id}")

        if angle != 0:
            self.arc(cand.seen_at, point, angle, 0)
        if fov < cam["hfov_deg"]:
            self.camera.set_fov(cam["hfov_deg"])
        if moves:
            self.log("climbing back to the lane")
            self.goto(stop, conf["speed"], heading, what="climb back")
        if not self.camera.set_tilt(self.plan.tilt_deg):
            self.log("WARNING: couldn't restore the camera tilt")

    def arc(self, target, start, from_deg, to_deg):
        """Fly around target from angle from_deg to to_deg (relative to start), in arc_step_deg
        steps at start's distance and height, facing the target. The camera tilt stays valid:
        it's relative to the drone, and the line-of-sight angle doesn't change on the arc."""
        conf = self.cfg["confirm"]
        steps = max(1, math.ceil(abs(to_deg - from_deg) / conf["arc_step_deg"]))
        for i in range(1, steps + 1):
            p = arc_point(target, start, from_deg + (to_deg - from_deg) * i / steps)
            self.goto(p, conf["speed"], bearing_deg(p, target), what="arc")

    def look(self, cand, view):
        """frames frames from here: the positions and confidences of those that see the candidate.
        Hit or miss per frame from the scene image and the flat-ground projection, which is fast;
        float depth images are very slow through AirSim's Python client."""
        conf = self.cfg["confirm"]
        positions, confidences = [], []
        for i in range(conf["frames"]):
            frame = self.camera.grab()
            dets = self.detector.detect(frame.image)
            self.save(f"candidate{cand.id}_try{cand.attempts}_view{view}_{i + 1}", frame, dets)
            hit = None
            for d in dets:
                p = self.camera.ground_point(frame, d.center[0], d.y2, self.ground_z)
                if p is not None and np.linalg.norm(p[:2] - cand.seen_at[:2]) < conf["match_radius"]:
                    positions.append(p)
                    confidences.append(d.confidence)
                    hit = np.linalg.norm(p[:2] - cand.seen_at[:2])
                    break
            self.log(f"  view {view}: frame {i + 1}/{conf['frames']}: "
                     + (f"hit, {hit:.1f} m from the candidate" if hit is not None
                        else f"miss ({len(dets)} person(s) elsewhere)"))
            time.sleep(conf["frame_interval_s"])
        return positions, confidences

    def depth_position(self, near, radius):
        """The person's feet measured with the depth camera (one frame), or None if not found."""
        frame = self.camera.grab(with_depth=True)
        for d in self.detector.detect(frame.image):
            p = self.camera.depth_point(frame, d.center[0], d.y2)
            if p is not None and np.linalg.norm(p[:2] - near[:2]) < radius:
                return p
        return None

    # ---- drawing and reporting ------------------------------------------------------------

    def draw_plan(self):
        ground = self.ground_z + self.spawn[2]
        z = ground - self.plan.altitude
        lanes = [airsim.Vector3r(x, y, z) for x, y in self.plan.waypoints]
        (ax, ay), (bx, by) = self.cfg["area"]["corner_a"], self.cfg["area"]["corner_b"]
        area = [airsim.Vector3r(x, y, ground - 0.5) for x, y in [(ax, ay), (ax, by), (bx, by), (bx, ay), (ax, ay)]]
        self.client.simFlushPersistentMarkers()
        self.client.simPlotLineStrip(area, color_rgba=AREA_COLOR, thickness=5, is_persistent=True)
        self.client.simPlotLineStrip(lanes, color_rgba=LANE_COLOR, thickness=8, is_persistent=True)

    def mark(self, p, color, label):
        """Mark a decided candidate, MARKER_HEIGHT_M above the ground.

        The camera sees debug drawings too: a marker on a person hides them, so candidates
        aren't marked until they're decided, and markers float above the spot.
        """
        w = self.to_world(p)
        z = self.ground_z + self.spawn[2] - MARKER_HEIGHT_M
        self.client.simPlotPoints([airsim.Vector3r(w[0], w[1], z)], color_rgba=color, size=15, is_persistent=True)
        self.client.simPlotStrings([label], [airsim.Vector3r(w[0], w[1], z - 1)], scale=3,
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
                     "lanes": self.plan.lane_count, "speed": self.speed},
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
    parser.add_argument("--victim", metavar="TAG",
                        help="instead of the search, test one tagged person: view them as a lane would, "
                             "detect, confirm, land")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        plan = plan_survey(cfg)
    except InfeasiblePlan as e:
        sys.exit(f"Plan is infeasible: {e}")
    print(format_plan(plan))
    if args.victim and args.no_detect:
        sys.exit("--victim needs the detector; drop --no-detect")
    Mission(cfg, plan, detect=not args.no_detect, victim=args.victim).run()


if __name__ == "__main__":
    main()
