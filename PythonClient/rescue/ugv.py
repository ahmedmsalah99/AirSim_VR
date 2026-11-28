"""The Husky UGV: drive a medical kit to every victim the drone confirms.

Start Play in Unreal, then (before or alongside mission.py):

    python ugv.py
    python ugv.py --victim Victim_01   # test: drive to a tagged person without the drone

It listens for confirmed victims on the drone's ZeroMQ topic (publish in
config.yaml) and keeps a queue:

1. At home it waits. When a victim arrives it drives to them, with no prior
   map: a 2D occupancy grid is built live from the lidar (obstacles between
   ugv.obstacle_min/max_height_m above the ground), and an A* path is planned
   on it and replanned as obstacles appear. Unknown ground counts as free.
   The Husky follows the path with pure-pursuit steering.
2. It stops ugv.stop_distance_m from the victim and moves one of the kits
   placed in the level (tags in ugv.medkits) to the ground beside them.
3. If more victims are queued it drives to the nearest; otherwise it heads
   home, and a victim published on the way (or while it waits at home) sends
   it there instead.

The UGV's position comes from AirSim (its pose is the SLAM pose); the map is
built from the lidar alone. Ctrl+C stops it where it is.
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from occupancy_grid import OccupancyGrid
from survey_plan import DEFAULT_CONFIG, load_config

OUTPUT_DIR = Path(__file__).with_name("output")
STUCK_MOVE_M = 0.3      # moving less than this while driving ...
STUCK_TIME_S = 4.0      # ... for this long is stuck
REVERSE_TIME_S = 2.0
TIMING_EVERY_S = 10     # log how long a control step takes, this often
STEADY_DEG = 10      # leaning less than this, it stands steady: a place to right it at ...
RIGHT_BACK_S = 2.0   # ... from at least this long before it tipped
TIPPED_DEG = 60       # leaning more than this: it has tipped over ...
MAX_RIGHTINGS = 3     # ... and is set back on its wheels up to this many times per trip
LEAN_DEG = 25         # leaning more than this: ease off and straighten up
NEAR_ARRIVE_M = 1.0             # within stop distance + clearance + this, a long way round is arrival
BUMP_ARRIVE_M = 1.5             # stuck within stop_distance + this of the goal counts as arrived
REVERSE_ON = math.radians(110)   # a path target this far off the nose starts a multi-point turn ...
REVERSE_OFF = math.radians(30)   # ... until it's within this of the nose
PATH_COLOR = [0.1, 0.4, 1.0, 1.0]
KIT_HEIGHT_M = 0.15     # the kit's origin above the ground


class TippedOver(Exception):
    pass


def yaw_of(q):
    """Heading (rad, NED: clockwise from north) of an AirSim quaternion."""
    return math.atan2(2 * (q.w_val * q.z_val + q.x_val * q.y_val), 1 - 2 * (q.y_val ** 2 + q.z_val ** 2))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lookahead_point(path, pos, distance):
    """The point distance metres along the path past the point on it nearest to pos: what pure
    pursuit steers at. (Aiming at the next waypoint instead cuts the corners, into the obstacles
    the corner goes around.)"""
    pts = np.asarray(path, float)
    if len(pts) == 1:
        return pts[0]
    best, seg, frac = math.inf, 0, 0.0
    for k, (a, b) in enumerate(zip(pts, pts[1:])):
        ab = b - a
        t = float(np.clip((pos - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
        d = np.linalg.norm(a + t * ab - pos)
        if d < best:
            best, seg, frac = d, k, t
    left = distance
    point = pts[seg] + frac * (pts[seg + 1] - pts[seg])
    for k in range(seg, len(pts) - 1):
        end = pts[k + 1]
        step = np.linalg.norm(end - point)
        if step >= left:
            return point + (end - point) * (left / step)
        left -= step
        point = end
    return pts[-1]


class VictimSubscriber:
    """Confirmed victims from the drone, as dicts (see publisher.py)."""

    def __init__(self, endpoint, topic):
        import zmq

        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.connect(endpoint)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.zmq = zmq

    def poll(self):
        out = []
        while self.socket.poll(0):
            _, body = self.socket.recv_multipart()
            out.append(json.loads(body))
        return out


class Ugv:
    def __init__(self, cfg, client=None, subscriber=None):
        self.cfg = cfg
        u = self.u = cfg["ugv"]
        self.name = u["name"]
        self.spawn = np.array(u["spawn_ned"], float)
        import setup_path  # noqa: F401  (puts this repo's airsim package on the path)
        import airsim
        self.airsim = airsim
        self.client = client or airsim.CarClient(port=u["port"])
        self.subscriber = subscriber or VictimSubscriber(u["subscribe"], cfg["publish"]["topic"])
        self.grid = OccupancyGrid((0.0, 0.0), u["map_size_m"], u["map_resolution_m"])
        # Stay within the search area (plus a margin): beyond it the map may end or drop away.
        (ax, ay), (bx, by) = cfg["area"]["corner_a"], cfg["area"]["corner_b"]
        m = u["area_margin_m"]
        self.grid.set_bounds(self.to_local((min(ax, bx) - m, min(ay, by) - m)),
                             self.to_local((max(ax, bx) + m, max(ay, by) + m)))
        self.radius = u["robot_radius_m"] + u["safety_margin_m"]
        self.queue, self.served, self.failed, self.seen_ids = [], [], [], set()
        self.kits = list(u["medkits"])
        self.path, self.last_plan, self.squeezing, self.backing = None, -math.inf, False, False
        self._orientation = None
        self.last_points = 0
        self.turn_leg = "back"
        self.t0 = self.now()

    # ---- bookkeeping -------------------------------------------------------------------

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)

    def log(self, msg):
        print(f"[{self.now() - self.t0:7.1f}s] UGV: {msg}", flush=True)

    def to_local(self, xy):
        return np.asarray(xy, float)[:2] - self.spawn[:2]

    def to_world(self, xy):
        return np.asarray(xy, float)[:2] + self.spawn[:2]

    # ---- sensing -----------------------------------------------------------------------

    def state(self):
        """(position xy, heading rad, speed m/s, z) in the UGV's local NED frame."""
        s = self.client.getCarState(vehicle_name=self.name)
        p = s.kinematics_estimated.position
        self._orientation = s.kinematics_estimated.orientation  # for tilt_deg(), without another call
        return np.array([p.x_val, p.y_val]), yaw_of(s.kinematics_estimated.orientation), s.speed, p.z_val

    def right(self, pos, heading, z):
        """Put the UGV back on its wheels at pos (local xy) and height z (local NED), facing
        heading, a little above the ground."""
        a = self.airsim
        h = heading / 2
        upright = a.Pose(a.Vector3r(float(pos[0]), float(pos[1]), float(z) - 0.3),
                         a.Quaternionr(0.0, 0.0, math.sin(h), math.cos(h)))
        self.client.simSetVehiclePose(upright, True, vehicle_name=self.name)
        self.sleep(1.0)

    def tilt_deg(self):
        """How far the UGV leans from upright (deg) at the last state(): over ~60 it has tipped over."""
        q = self._orientation
        # The body's down axis in the world: its z component is cos(tilt).
        down_z = 1 - 2 * (q.x_val ** 2 + q.y_val ** 2)
        return math.degrees(math.acos(max(-1.0, min(1.0, down_z))))

    def sense(self, pos=None, z=None):
        """Add the latest lidar sweep to the map (pos and z: the UGV's, if just read). Returns the
        time the fetch from AirSim took."""
        u = self.u
        start = self.now()
        data = self.client.getLidarData(lidar_name=u["lidar"], vehicle_name=self.name)
        pts = np.asarray(data.point_cloud, float)
        fetch = self.now() - start
        self.last_points = len(pts) // 3
        if len(pts) < 3:
            return fetch
        pts = pts[: len(pts) // 3 * 3].reshape(-1, 3)
        if pos is None:
            pos, _, _, z = self.state()
        ground_z = z + u["base_height_m"]   # NED: down is +z
        height = ground_z - pts[:, 2]
        # Obstacles stick up from the ground; a drop (an edge, a ditch) shows as hits well below it.
        obstacles = pts[((height >= u["obstacle_min_height_m"]) & (height <= u["obstacle_max_height_m"]))
                        | (height < -u["drop_m"]), :2]
        self.grid.update(pos, obstacles)
        # Terrain: every surface point up to obstacle height goes into the height map, which marks
        # steps and slopes too steep to drive (a ramp, a bank, a kerb) as obstacles.
        surface = height <= u["obstacle_max_height_m"]
        self.grid.update_ground(pts[surface, :2], -pts[surface, 2], u["max_step_m"], u["max_slope_deg"])
        return fetch

    def check_victims(self):
        """Queue newly published victims (by id; one within 3 m of a known one is the same)."""
        for msg in self.subscriber.poll():
            if msg.get("id") in self.seen_ids:
                continue
            self.seen_ids.add(msg.get("id"))
            p = msg["position"]
            world = np.array([p["x"], p["y"], p["z"]])
            local = self.to_local(world)
            if any(np.linalg.norm(v["local"] - local) < 3 for v in self.queue + self.served):
                continue
            self.queue.append({"id": msg.get("id"), "world": world, "local": local})
            self.log(f"victim {msg.get('id')} published at x={world[0]:.1f} y={world[1]:.1f}; "
                     f"{len(self.queue)} in the queue")

    # ---- driving -----------------------------------------------------------------------

    def controls(self, throttle=0.0, steering=0.0, brake=0.0):
        """throttle < 0 reverses. AirSim needs the reverse gear for that, as its hello_car.py
        does: in automatic gear a negative throttle only brakes (the Husky stayed put)."""
        c = self.airsim.CarControls()
        c.throttle, c.steering, c.brake = float(throttle), float(steering), float(brake)
        c.is_manual_gear = throttle < 0
        c.manual_gear = -1 if throttle < 0 else 0
        c.gear_immediate = True
        self.client.setCarControls(c, vehicle_name=self.name)

    def stop(self):
        self.controls(brake=1.0)

    def replan(self, pos, goal, tolerance):
        self.path = self.grid.plan(pos, goal, self.radius, tolerance)
        if self.path is None:
            # Squeeze through: only the robot's radius, without the safety margin.
            self.path = self.grid.plan(pos, goal, self.u["robot_radius_m"], tolerance)
            if self.path is not None and not self.squeezing:
                self.log(f"no way with the {self.u['safety_margin_m']} m safety margin; going through a "
                         f"narrow gap without it")
            self.squeezing = self.path is not None
        else:
            self.squeezing = False
        self.last_plan = self.now()
        if self.path is not None:
            self.draw(self.path)

    def explain_no_path(self, pos, goal, tolerance, what):
        """Why there's no path (logged), and a picture of the map saved to output/."""
        r = self.u["robot_radius_m"]
        reasons = []
        if self.grid.enclosed(pos, r):
            reasons.append("the UGV is boxed in where it is")
        if self.grid.goal_blocked(goal, r, tolerance):
            reasons.append(f"everywhere within {tolerance:.1f} m of {what} is blocked")
        reason = " and ".join(reasons) or "every way there is blocked"
        name = OUTPUT_DIR / f"ugv_noway_{time.strftime('%Y%m%d_%H%M%S')}.png"
        try:
            import cv2
            img = self.grid.picture(r, [(pos, (255, 0, 0)), (goal, (0, 0, 255))])
            OUTPUT_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(name), img)
            saved = f"; map saved to {name} (UGV blue, goal red, obstacles black, too close grey, north up)"
        except Exception as e:  # noqa: BLE001 - the picture is only a help
            saved = f" (couldn't save the map: {e!r})"
        return reason + saved

    def draw(self, path):
        try:
            z = self.spawn[2] - 0.5
            pts = [self.airsim.Vector3r(*self.to_world(p), z) for p in path]
            self.client.simPlotLineStrip(pts, color_rgba=PATH_COLOR, thickness=8, duration=self.u["replan_s"] * 2)
        except Exception:  # noqa: BLE001 - drawing is only a help
            pass

    def steer(self, pos, heading, speed):
        """Pure pursuit along self.path: throttle (< 0 in reverse), steering and brake for this step.

        The Husky steers like a car and can't turn on the spot, so when the path starts behind it
        it makes a 3-point turn: it reverses on full lock until the path is ahead, then goes on
        forwards. A forward U-turn in a tight spot ran it into the fence and the house beside its
        start."""
        u = self.u
        path = self.path
        target = lookahead_point(path, pos, u["lookahead_m"])
        d = np.asarray(target) - pos
        bearing = math.atan2(d[1], d[0])
        alpha = wrap(bearing - heading)  # > 0: the target is to the right (NED)
        if self.backing and abs(alpha) < REVERSE_OFF:
            self.backing = False
        elif not self.backing and (abs(alpha) > REVERSE_ON
                                   or (abs(alpha) > REVERSE_OFF and self.close(pos, heading))):
            # Far off the nose, or turning with something close ahead: turn on the spot instead.
            self.backing, self.turn_leg = True, "back"
        forward = True
        if self.backing:
            # A multi-point turn, until the target is within REVERSE_OFF of the nose: back up on
            # full lock so the nose swings round towards it (reversing, a car turns the other way
            # for the same steering) until something's close behind, then forward on the opposite
            # lock until something's close in front, and so on. (Reversing the whole way, rear
            # first, took it 13 m backwards into a dip it couldn't get out of.)
            side = float(np.sign(alpha)) or 1.0
            if self.turn_leg == "back" and self.close(pos, heading + math.pi):
                self.turn_leg = "forward"
            elif self.turn_leg == "forward" and self.close(pos, heading):
                self.turn_leg = "back"
            forward = self.turn_leg == "forward"
            steering = side * u["steering_sign"] * (1.0 if forward else -1.0)
        else:
            steering = float(np.clip(u["steering_gain"] * alpha * u["steering_sign"], -1, 1))
        want = u["max_speed"] if abs(alpha) < 0.5 and not self.backing else u["slow_speed"]
        want = min(want, max(u["slow_speed"], np.linalg.norm(np.asarray(path[-1]) - pos) / 2))
        mag = abs(speed)
        # Proportional throttle, and a brake only when well over: on/off throttle and a brake at the
        # first sign of overspeed made it surge and stop.
        throttle = float(np.clip(0.25 + 0.5 * (want - mag), 0, u["max_throttle"]))
        brake = 0.2 if mag > want + 0.8 else 0.0
        # Less steering the faster it goes: full lock only at walking pace (full lock at speed
        # is what rolls a vehicle).
        limit = 1.0 / (1.0 + max(mag - 0.5, 0.0))
        steering = float(np.clip(steering, -limit, limit))
        return (throttle if forward else -throttle), steering, brake

    def close(self, pos, direction, clearance=0.6):
        """Whether the map has an obstacle within clearance of the UGV's body (robot_radius_m)
        in direction (rad, NED)."""
        d = np.array([math.cos(direction), math.sin(direction)])
        side = np.array([-d[1], d[0]])
        reach = self.u["robot_radius_m"] + clearance
        pts = [pos + d * a + side * w for a in np.arange(0.3, reach + 0.01, 0.25) for w in (-0.4, 0, 0.4)]
        cells = self.grid.to_cell(pts)
        cells = cells[self.grid.inside(cells)]
        return bool(self.grid.occupied()[cells[:, 0], cells[:, 1]].any()) if len(cells) else False

    def drive_to(self, goal, tolerance, what, interrupt=None):
        """Drive to within tolerance of goal (local xy). Returns "arrived", "interrupted" (when
        interrupt() turns true) or "failed" (no path, or too long)."""
        u = self.u
        goal = np.asarray(goal, float)
        start = self.now()
        pos = self.state()[0]
        deadline = start + np.linalg.norm(goal - pos) / u["slow_speed"] * 3 + 60
        self.path, self.last_plan, slow_since, no_path_since = None, -math.inf, None, None
        self.backing = False
        rightings = 0
        stable = collections.deque()   # (time, pos, heading, z) while it stood steady, recent first last
        last_safe = None               # where it was last righted
        timing = {"steps": 0, "sense": 0.0, "fetch": 0.0, "plan": 0.0, "since": self.now()}
        self.log(f"driving to {what} ({np.linalg.norm(goal - pos):.0f} m away)")
        while True:
            timing["steps"] += 1
            if self.now() - timing["since"] >= TIMING_EVERY_S:
                n, span = timing["steps"], self.now() - timing["since"]
                self.log(f"  control loop: {span / n:.2f} s per step (lidar fetch {timing['fetch'] / n:.2f} s, "
                         f"map {(timing['sense'] - timing['fetch']) / n:.2f} s, planning {timing['plan'] / n:.2f} s, "
                         f"{self.last_points} points); want {u['control_period_s']:.2f} s")
                timing.update(steps=0, sense=0.0, fetch=0.0, plan=0.0, since=self.now())
            step_start = self.now()
            pos, heading, speed, z = self.state()
            tilt = self.tilt_deg()
            if tilt > TIPPED_DEG:
                self.stop()
                if rightings >= MAX_RIGHTINGS:
                    self.log(f"WARNING: tipped over ({tilt:.0f} deg from upright) on the way to {what} "
                             f"{rightings + 1} times; stopping (restart Play to put it back)")
                    raise TippedOver(what)
                rightings += 1
                # Back on its wheels where it last stood steady (on the spot where it fell, it
                # rolled straight off again), and that spot marked so the next plan avoids it.
                # (Tipping again before it stood steady anywhere: the place it was last righted at.)
                back = [s for s in stable if self.now() - s[0] >= RIGHT_BACK_S] or list(stable) or \
                       ([last_safe] if last_safe else [(self.now(), pos, heading, z)])
                last_safe = back[-1]
                _, safe_pos, safe_heading, safe_z = last_safe
                self.log(f"tipped over ({tilt:.0f} deg from upright, {self.touching()}, at "
                         f"x={self.to_world(pos)[0]:.1f} y={self.to_world(pos)[1]:.1f}); setting it back on its wheels "
                         f"{np.linalg.norm(pos - safe_pos):.1f} m back, where it was steady, and marking "
                         f"the spot ({rightings}/{MAX_RIGHTINGS})")
                self.grid.mark([pos + np.array([dx, dy]) for dx in np.arange(-0.75, 0.76, 0.25)
                                for dy in np.arange(-0.75, 0.76, 0.25) if dx * dx + dy * dy <= 0.6])
                self.right(safe_pos, safe_heading, safe_z)
                stable.clear()
                self.path, self.last_plan = None, -math.inf
                continue
            if tilt < STEADY_DEG:
                stable.append((self.now(), pos.copy(), heading, z))
                while stable and self.now() - stable[0][0] > 2 * RIGHT_BACK_S:
                    stable.popleft()
            if tilt > LEAN_DEG:
                # Leaning hard: ease off and straighten up before it rolls.
                self.controls(throttle=0.0, steering=0.0, brake=0.5)
                self.sleep(max(0.0, u["control_period_s"] - (self.now() - step_start)))
                continue
            if np.linalg.norm(goal - pos) <= tolerance + 0.3:
                self.stop()
                return "arrived"
            if self.now() > deadline:
                self.stop()
                self.log(f"WARNING: gave up on {what}: took too long")
                return "failed"
            t_sense = self.now()
            timing["fetch"] += self.sense(pos, z)
            timing["sense"] += self.now() - t_sense
            self.check_victims()
            if interrupt is not None and interrupt():
                return "interrupted"
            # Replan every replan_s (also while there's no path: a failed search can take a
            # second), and at once when the lidar puts something across the current path.
            if (self.now() - self.last_plan >= u["replan_s"] or (
                    self.path is not None
                    and self.grid.path_blocked([tuple(pos)] + list(self.path[1:]), self.radius))):
                t_plan = self.now()
                self.replan(pos, goal, tolerance)
                timing["plan"] += self.now() - t_plan
                # Next to the goal but the plan goes a long way round (the near side of the ring
                # around a victim is blocked, by the person and what stands around them): it's
                # there. It used to drive off round the victim to the far side instead.
                straight = float(np.linalg.norm(goal - pos))
                near = tolerance + self.radius + NEAR_ARRIVE_M   # as close as clearance may allow
                if self.path is None and straight <= near:
                    self.stop()
                    self.log(f"{straight:.1f} m from {what}, and no way closer: close enough")
                    return "arrived"
                if self.path is not None and straight <= near:
                    length = sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(self.path, self.path[1:]))
                    if length > 2 * max(straight - tolerance, 0.5) + 1.0:
                        self.stop()
                        self.log(f"{straight:.1f} m from {what}, and the way to {tolerance:.1f} m from it is "
                                 f"{length:.0f} m round: close enough")
                        return "arrived"
            if self.path is None:
                self.stop()
                no_path_since = no_path_since or self.now()
                if self.now() - no_path_since > u["no_path_timeout_s"]:
                    self.log(f"WARNING: no way to {what} in the map: "
                             f"{self.explain_no_path(pos, goal, tolerance, what)}; giving up")
                    return "failed"
                self.sleep(max(0.0, u["control_period_s"] - (self.now() - step_start)))
                continue
            no_path_since = None
            throttle, steering, brake = self.steer(pos, heading, speed)
            # Stuck: driving but not getting anywhere (against something the lidar misses). Judged
            # by distance moved, not speed: pushing against a pole it bounced, and its speed
            # flickered above any threshold. Back off the other way and replan.
            # (A change of direction restarts it: a multi-point turn goes back and forth in place.)
            if (throttle == 0 or slow_since is None or np.sign(throttle) != slow_since[2]
                    or np.linalg.norm(pos - slow_since[1]) > STUCK_MOVE_M):
                slow_since = (self.now(), pos.copy(), np.sign(throttle))
            elif self.now() - slow_since[0] > STUCK_TIME_S:
                way = "reversing" if throttle < 0 else "going forward"
                left = float(np.linalg.norm(goal - pos))
                if left <= tolerance + BUMP_ARRIVE_M:
                    # Against something right by the goal: most likely the victim themself,
                    # a little off where the drone placed them.
                    self.stop()
                    self.log(f"stuck {way} {left:.1f} m from {what} ({self.touching()}): close enough")
                    return "arrived"
                self.log(f"stuck {way} ({self.touching()}); marking it in the map and backing off")
                self.mark_bump(pos, heading + (math.pi if throttle < 0 else 0.0))
                self.escape(-1 if throttle > 0 else 1, -steering)
                slow_since, self.path, self.last_plan = None, None, -math.inf
                continue
            self.controls(throttle, steering, brake)
            self.sleep(max(0.0, u["control_period_s"] - (self.now() - step_start)))

    def mark_bump(self, pos, direction):
        """Put what the UGV is pushing against into the map, so the next plan goes around it
        instead of into it again: where AirSim says it hit (a corner or a side touches a pole as
        often as the bumper), else a short line across its path just beyond its bumper."""
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=self.name)
            p = info.impact_point
            hit = np.array([p.x_val, p.y_val])
            if (info.has_collided and "ground" not in info.object_name.lower()
                    and np.linalg.norm(hit - pos) < self.u["robot_radius_m"] + 1.0):
                self.grid.mark([hit + np.array([dx, dy]) for dx in (-0.25, 0, 0.25) for dy in (-0.25, 0, 0.25)])
                return
        except Exception:  # noqa: BLE001 - fall back to the bumper
            pass
        ahead = np.array([math.cos(direction), math.sin(direction)])
        side = np.array([-ahead[1], ahead[0]])
        front = pos + ahead * (self.u["robot_radius_m"] + 0.3)
        self.grid.mark([front + side * w for w in np.arange(-0.5, 0.51, 0.25)])

    def escape(self, direction, steering):
        """Drive REVERSE_TIME_S the given way (1 forward, -1 back), then stop; not if the map has
        something close that way."""
        pos, heading, _, _ = self.state()
        if self.close(pos, heading + (0.0 if direction > 0 else math.pi)):
            self.log("  something close that way too; not moving, replanning")
            self.stop()
            return
        end = self.now() + REVERSE_TIME_S
        while self.now() < end:
            self.controls(throttle=0.5 * direction, steering=steering)
            self.sleep(self.u["control_period_s"])
        self.stop()

    def touching(self):
        """What AirSim last saw the UGV collide with, for the log."""
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=self.name)
            return f"last collision: {info.object_name}" if info.has_collided else "no collision reported"
        except Exception:  # noqa: BLE001
            return "collision info unavailable"

    # ---- the job -----------------------------------------------------------------------

    def drop_kit(self, victim):
        """Move a pre-placed kit next to the victim, on the UGV's side."""
        pos = self.state()[0]
        if not self.kits:
            self.log(f"WARNING: no medical kits left for victim {victim['id']} (add more to ugv.medkits)")
            return False
        tag = self.kits.pop(0)
        away = pos - victim["local"]
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0.0])
        xy = self.to_world(victim["local"] + away * self.u["kit_offset_m"])
        z = victim["world"][2] - KIT_HEIGHT_M
        pose = self.airsim.Pose(self.airsim.Vector3r(xy[0], xy[1], z), self.airsim.Quaternionr())
        if not self.client.simSetObjectPose(tag, pose, True):
            self.log(f"WARNING: couldn't move {tag}: is an actor tagged {tag} in the level, with "
                     f"Mobility set to Movable?")
            return False
        self.log(f"medical kit {tag} placed beside victim {victim['id']} at x={xy[0]:.1f} y={xy[1]:.1f}")
        return True

    def check(self):
        """Check the lidar and the driving before a real run: where one sweep's points are, then a
        short drive with steering +0.5 to see whether and which way the Husky moves."""
        c, u = self.client, self.u
        c.confirmConnection()
        c.enableApiControl(True, self.name)
        pos, heading, _, z = self.state()
        self.log(f"pose: x={pos[0]:.2f} y={pos[1]:.2f} z={z:.2f} (local NED), heading {math.degrees(heading):.0f} deg")
        data = c.getLidarData(lidar_name=u["lidar"], vehicle_name=self.name)
        pts = np.asarray(data.point_cloud, float)
        if len(pts) < 3:
            self.log(f"lidar {u['lidar']}: NO POINTS. Is Vehicles.Car1.Sensors.{u['lidar']} in the "
                     f"settings.json AirSim reads (Documents\\AirSim), and Play restarted since?")
        else:
            pts = pts[: len(pts) // 3 * 3].reshape(-1, 3)
            height = z + u["base_height_m"] - pts[:, 2]
            rng = np.linalg.norm(pts[:, :2] - pos, axis=1)
            band = (height >= u["obstacle_min_height_m"]) & (height <= u["obstacle_max_height_m"])
            self.log(f"lidar: {len(pts)} points in one sweep, {band.sum()} in the obstacle band "
                     f"({u['obstacle_min_height_m']}..{u['obstacle_max_height_m']} m above the assumed ground)")
            self.log("  range       points  in band  height above the assumed ground (10% / median / 90%)")
            for lo, hi in ((0, 1.5), (1.5, 5), (5, 15), (15, 1e9)):
                m = (rng >= lo) & (rng < hi)
                if m.any():
                    q = np.percentile(height[m], [10, 50, 90])
                    label = f"{lo:g}-{hi:g} m" if hi < 1e9 else f"{lo:g}+ m"
                    self.log(f"  {label:<10} {m.sum():7d}  {band[m].sum():7d}  "
                             f"{q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f} m")
            ground_band = (rng >= 3) & (rng < 15)
            if ground_band.sum() > 20:
                low = float(np.percentile(height[ground_band], 5))   # the lowest hits: the ground
                self.log(f"  the ground (lowest 5% of the hits 3-15 m away) comes out {low:+.2f} m above the "
                         f"assumed ground: set ugv.base_height_m to {u['base_height_m'] - low:.2f}"
                         + ("" if abs(low) > 0.15 else " (or leave it: that's close enough)"))
            if (rng < 1.5).sum() > 0.05 * len(pts):
                self.log("  WARNING: many points within 1.5 m: the lidar may be hitting the Husky itself "
                         "(e.g. a mesh added as a child actor). Raise the lidar (Z in settings.json) or "
                         "turn off collision on that mesh")
        self.log("driving 3 s: throttle 0.4, steering +0.5 ...")
        top = 0.0
        for _ in range(30):
            self.controls(throttle=0.4, steering=0.5)
            self.sleep(0.1)
            top = max(top, abs(self.state()[2]))
        self.stop()
        new_pos, new_heading, _, _ = self.state()
        moved = float(np.linalg.norm(new_pos - pos))
        turned = math.degrees(wrap(new_heading - heading))
        self.log(f"moved {moved:.2f} m, top speed {top:.2f} m/s, turned {turned:+.0f} deg")
        if moved < 0.3:
            self.log("  the Husky DIDN'T MOVE, so the steering direction is unknown. Check that it drives "
                     "at all, e.g. with PythonClient/both/car_go_back.py, and that nothing is in front of it")
        elif abs(turned) < 3:
            self.log("  it moved but didn't turn: steering may not reach the wheels")
        else:
            side = "right" if turned > 0 else "left"
            self.log(f"  it turned {side}. Pure pursuit expects steering > 0 to turn right: "
                     f"ugv.steering_sign should be {1 if turned > 0 else -1}")
        c.enableApiControl(False, self.name)

    def nearest(self):
        pos = self.state()[0]
        return min(self.queue, key=lambda v: np.linalg.norm(v["local"] - pos))

    def queue_actor(self, tag):
        """Queue the actor tagged tag (a person placed in the level) as a victim, at its true
        position, without waiting for the drone."""
        p = self.client.simGetObjectPose(tag).position
        if math.isnan(p.x_val):
            raise SystemExit(f"no actor tagged {tag!r} in the level (Details > Actor > Tags)")
        # The actor's origin may be at its waist; the kit goes on the ground the UGV stands on.
        world = np.array([p.x_val, p.y_val, self.spawn[2] + self.state()[3] + self.u["base_height_m"]])
        self.queue.append({"id": tag, "world": world, "local": self.to_local(world)})
        self.log(f"{tag} queued directly at x={world[0]:.1f} y={world[1]:.1f} (not waiting for the drone)")

    def run(self, once=False):
        """Serve victims until Ctrl+C (or, with once, until home with an empty queue)."""
        c, u = self.client, self.u
        c.confirmConnection()
        c.enableApiControl(True, self.name)
        self.log(f"ready at home; listening for victims on {u['subscribe']}")
        home = np.zeros(2)
        at_home = True
        try:
            while True:
                self.check_victims()
                if self.queue:
                    victim = self.nearest()
                    result = self.drive_to(victim["local"], u["stop_distance_m"], f"victim {victim['id']}")
                    self.queue.remove(victim)
                    if result == "arrived":
                        self.drop_kit(victim)
                        self.served.append(victim)
                    else:
                        self.failed.append(victim)
                    at_home = False
                    continue
                if not at_home:
                    result = self.drive_to(home, u["home_radius_m"], "home", interrupt=lambda: bool(self.queue))
                    if result == "interrupted":
                        self.log("new victim on the way home")
                        continue
                    at_home = True
                    self.log(f"home; {len(self.served)} victim(s) served, {len(self.failed)} unreachable")
                if once:
                    return
                self.stop()
                self.sense()
                self.sleep(u["control_period_s"])
        except KeyboardInterrupt:
            self.log("stopped by user")
        except TippedOver:
            self.log(f"{len(self.served)} victim(s) served before it tipped over; "
                     f"{len(self.queue)} still waiting")
        finally:
            try:
                self.stop()
            except Exception:  # noqa: BLE001
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true",
                        help="check the lidar heights and the steering direction (drives 3 s), then exit")
    parser.add_argument("--victim", metavar="TAG", action="append",
                        help="test without the drone: drive to this tagged person (repeat for several), "
                             "drop the kits, come home and exit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if "ugv" not in cfg:
        sys.exit("config.yaml has no ugv section")
    ugv = Ugv(cfg)
    if args.check:
        ugv.check()
    elif args.victim:
        for tag in args.victim:
            ugv.queue_actor(tag)
        ugv.run(once=True)
    else:
        ugv.run()


if __name__ == "__main__":
    main()
