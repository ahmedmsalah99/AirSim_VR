"""The Husky UGV: drive a medical kit to every victim the drone confirms.

Start Play in Unreal, then (before or alongside mission.py):

    python ugv.py

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
import json
import math
import sys
import time

import numpy as np

from occupancy_grid import OccupancyGrid
from survey_plan import DEFAULT_CONFIG, load_config

STUCK_SPEED = 0.15      # m/s: below this while driving counts as stuck ...
STUCK_TIME_S = 3.0      # ... for this long
REVERSE_TIME_S = 2.0
PATH_COLOR = [0.1, 0.4, 1.0, 1.0]
KIT_HEIGHT_M = 0.15     # the kit's origin above the ground


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
        self.radius = u["robot_radius_m"] + u["safety_margin_m"]
        self.queue, self.served, self.failed, self.seen_ids = [], [], [], set()
        self.kits = list(u["medkits"])
        self.path, self.last_plan = None, -math.inf
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
        return np.array([p.x_val, p.y_val]), yaw_of(s.kinematics_estimated.orientation), s.speed, p.z_val

    def sense(self):
        """Add the latest lidar sweep to the map."""
        u = self.u
        data = self.client.getLidarData(lidar_name=u["lidar"], vehicle_name=self.name)
        pts = np.asarray(data.point_cloud, float)
        if len(pts) < 3:
            return
        pts = pts[: len(pts) // 3 * 3].reshape(-1, 3)
        pos, _, _, z = self.state()
        ground_z = z + u["base_height_m"]   # NED: down is +z
        height = ground_z - pts[:, 2]
        obstacles = pts[(height >= u["obstacle_min_height_m"]) & (height <= u["obstacle_max_height_m"]), :2]
        self.grid.update(pos, obstacles)

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
        c = self.airsim.CarControls()
        c.throttle, c.steering, c.brake = float(throttle), float(steering), float(brake)
        self.client.setCarControls(c, vehicle_name=self.name)

    def stop(self):
        self.controls(brake=1.0)

    def replan(self, pos, goal, tolerance):
        self.path = self.grid.plan(pos, goal, self.radius, tolerance)
        self.last_plan = self.now()
        if self.path is not None:
            self.draw(self.path)

    def draw(self, path):
        try:
            z = self.spawn[2] - 0.5
            pts = [self.airsim.Vector3r(*self.to_world(p), z) for p in path]
            self.client.simPlotLineStrip(pts, color_rgba=PATH_COLOR, thickness=8, duration=self.u["replan_s"] * 2)
        except Exception:  # noqa: BLE001 - drawing is only a help
            pass

    def steer(self, pos, heading, speed):
        """Pure pursuit along self.path: throttle and steering for this step."""
        u = self.u
        path = self.path
        target = lookahead_point(path, pos, u["lookahead_m"])
        d = np.asarray(target) - pos
        alpha = wrap(math.atan2(d[1], d[0]) - heading)  # > 0: the target is to the right (NED)
        steering = float(np.clip(u["steering_gain"] * alpha * u["steering_sign"], -1, 1))
        want = u["max_speed"] if abs(alpha) < 0.5 else u["slow_speed"]
        want = min(want, max(u["slow_speed"], np.linalg.norm(np.asarray(path[-1]) - pos) / 2))
        throttle = float(np.clip(0.3 + 0.4 * (want - speed), 0, 1)) if speed < want + 0.3 else 0.0
        brake = 0.0 if speed < want + 0.3 else 0.3
        return throttle, steering, brake

    def drive_to(self, goal, tolerance, what, interrupt=None):
        """Drive to within tolerance of goal (local xy). Returns "arrived", "interrupted" (when
        interrupt() turns true) or "failed" (no path, or too long)."""
        u = self.u
        goal = np.asarray(goal, float)
        start = self.now()
        pos = self.state()[0]
        deadline = start + np.linalg.norm(goal - pos) / u["slow_speed"] * 3 + 60
        self.path, self.last_plan, slow_since, no_path_since = None, -math.inf, None, None
        self.log(f"driving to {what} ({np.linalg.norm(goal - pos):.0f} m away)")
        while True:
            pos, heading, speed, _ = self.state()
            if np.linalg.norm(goal - pos) <= tolerance + 0.3:
                self.stop()
                return "arrived"
            if self.now() > deadline:
                self.stop()
                self.log(f"WARNING: gave up on {what}: took too long")
                return "failed"
            self.sense()
            self.check_victims()
            if interrupt is not None and interrupt():
                return "interrupted"
            # Replan every replan_s (also while there's no path: a failed search can take a
            # second), and at once when the lidar puts something across the current path.
            if (self.now() - self.last_plan >= u["replan_s"] or (
                    self.path is not None
                    and self.grid.path_blocked([tuple(pos)] + list(self.path[1:]), self.radius))):
                self.replan(pos, goal, tolerance)
            if self.path is None:
                self.stop()
                no_path_since = no_path_since or self.now()
                if self.now() - no_path_since > u["no_path_timeout_s"]:
                    self.log(f"WARNING: no way to {what} in the map; giving up")
                    return "failed"
                self.sleep(u["control_period_s"])
                continue
            no_path_since = None
            throttle, steering, brake = self.steer(pos, heading, speed)
            # Stuck: pushing but not moving (against something the lidar misses). Back up.
            if throttle > 0 and speed < STUCK_SPEED:
                slow_since = slow_since or self.now()
                if self.now() - slow_since > STUCK_TIME_S:
                    self.log("stuck; backing up")
                    self.reverse(-steering)
                    slow_since, self.path = None, None
                    continue
            else:
                slow_since = None
            self.controls(throttle, steering, brake)
            self.sleep(u["control_period_s"])

    def reverse(self, steering):
        end = self.now() + REVERSE_TIME_S
        while self.now() < end:
            self.controls(throttle=-0.5, steering=steering)  # auto gear: negative throttle reverses
            self.sleep(self.u["control_period_s"])
        self.stop()

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
        """Check the lidar and the steering before a real run: one sweep's heights, then a short
        drive with steering +0.5 to see which way the Husky turns."""
        c, u = self.client, self.u
        c.confirmConnection()
        c.enableApiControl(True, self.name)
        pos, heading, _, z = self.state()
        data = c.getLidarData(lidar_name=u["lidar"], vehicle_name=self.name)
        pts = np.asarray(data.point_cloud, float)
        if len(pts) < 3:
            self.log(f"lidar {u['lidar']}: NO POINTS. Is Vehicles.Car1.Sensors.{u['lidar']} in the "
                     f"settings.json AirSim reads (Documents\\AirSim), and Play restarted since?")
        else:
            pts = pts[: len(pts) // 3 * 3].reshape(-1, 3)
            height = z + u["base_height_m"] - pts[:, 2]
            rng = np.linalg.norm(pts[:, :2] - pos, axis=1)
            near = height[rng < 15]
            obstacle = (height >= u["obstacle_min_height_m"]) & (height <= u["obstacle_max_height_m"])
            self.log(f"lidar: {len(pts)} points in one sweep, {obstacle.sum()} in the obstacle band")
            if len(near):
                self.log(f"  heights above the assumed ground (within 15 m): lowest 10% "
                         f"{np.percentile(near, 10):+.2f} m, median {np.median(near):+.2f} m. The ground "
                         f"should come out near 0; if it's off, change ugv.base_height_m by that much")
        self.log("driving 3 s with steering +0.5 ...")
        for _ in range(30):
            self.controls(throttle=0.4, steering=0.5)
            self.sleep(0.1)
        self.stop()
        turned = math.degrees(wrap(self.state()[1] - heading))
        side = "right" if turned > 0 else "left"
        self.log(f"turned {abs(turned):.0f} deg to the {side} (NED heading). Pure pursuit expects steering > 0 to turn "
                 f"right: ugv.steering_sign should be {1 if turned > 0 else -1}")
        c.enableApiControl(False, self.name)

    def nearest(self):
        pos = self.state()[0]
        return min(self.queue, key=lambda v: np.linalg.norm(v["local"] - pos))

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
    args = parser.parse_args()
    cfg = load_config(args.config)
    if "ugv" not in cfg:
        sys.exit("config.yaml has no ugv section")
    ugv = Ugv(cfg)
    ugv.check() if args.check else ugv.run()


if __name__ == "__main__":
    main()
