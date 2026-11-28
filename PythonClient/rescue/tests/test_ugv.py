"""ugv.py and occupancy_grid.py against a simulated Husky: a bicycle-model car with a 2D lidar
that sees round obstacles, in simulated time."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy_grid import OccupancyGrid  # noqa: E402

DT = 0.1          # simulated seconds per control step
OUT = None        # set per test: where ugv.py saves its map pictures
WHEELBASE = 0.8
MAX_STEER = math.radians(35)


# ---- occupancy grid ---------------------------------------------------------------------

def test_a_hit_marks_its_cell_and_the_ray_clears_the_way():
    g = OccupancyGrid((0, 0), 40, 0.25)
    g.update((0, 0), [(5, 0)])
    occ = g.occupied()
    assert occ[tuple(g.to_cell((5, 0)))]
    assert g.logodds[tuple(g.to_cell((2.5, 0)))] < 0      # seen free
    assert g.logodds[tuple(g.to_cell((8, 0)))] == 0       # behind the hit: unknown


def test_plan_goes_around_a_wall_and_keeps_its_distance():
    g = OccupancyGrid((0, 0), 40, 0.25)
    wall = [(5, y) for y in np.arange(-4, 4.01, 0.1)]
    for _ in range(2):
        g.update((0, 0), wall)
    path = g.plan((0, 0), (10, 0), radius_m=1.0)
    assert path is not None
    assert np.linalg.norm(np.asarray(path[-1]) - (10, 0)) < 0.3
    dense = np.concatenate([np.linspace(a, b, 50) for a, b in zip(path, path[1:])])
    closest = min(np.min(np.linalg.norm(dense - w, axis=1)) for w in wall)
    assert closest >= 1.0 - 0.25                        # grown by the radius, give or take a cell


def test_goal_tolerance_stops_short_of_an_occupied_goal():
    g = OccupancyGrid((0, 0), 40, 0.25)
    g.update((0, 0), [(8, 0)])                           # the victim themself shows up in the lidar
    path = g.plan((0, 0), (8, 0), radius_m=1.0, goal_tolerance_m=2.0)
    assert path is not None and 1.7 <= np.linalg.norm(np.asarray(path[-1]) - (8, 0)) <= 2.3


def test_pure_pursuit_aims_along_the_path_not_at_the_next_corner():
    from ugv import lookahead_point
    path = [(0, 0), (10, 0), (10, 10)]
    assert lookahead_point(path, np.array([2.0, 0.5]), 2.5) == pytest.approx([4.5, 0])
    assert lookahead_point(path, np.array([9.0, 0.0]), 2.5) == pytest.approx([10, 1.5])
    assert lookahead_point(path, np.array([10.0, 9.5]), 2.5) == pytest.approx([10, 10])


def test_no_path_when_walled_in():
    g = OccupancyGrid((0, 0), 20, 0.25)
    ring = [(5 * math.cos(a), 5 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 400)]
    g.update((0, 0), ring)
    assert g.plan((0, 0), (8, 0), radius_m=0.5) is None


# ---- simulated Husky --------------------------------------------------------------------

class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val, self.y_val, self.z_val = x, y, z


class Quat:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w_val, self.x_val, self.y_val, self.z_val = w, x, y, z


class SimCar:
    """AirSim CarClient stand-in. Every setCarControls() runs the car for DT."""

    def __init__(self, obstacles, spawn_z=-2.0):
        self.obstacles = obstacles  # [(x, y, radius)] in the car's local frame
        self.x = self.y = self.yaw = self.v = 0.0
        self.z = 0.0
        self.t = 0.0
        self.cmd = None
        self.kits = {}
        self.collisions = 0

    # time
    def step(self):
        c = self.cmd
        if c is not None:
            accel = 3.0 * c.throttle - 4.0 * c.brake * np.sign(self.v) - 0.5 * self.v
            self.v = float(np.clip(self.v + accel * DT, -1.5, 4.0))
            steer = float(np.clip(c.steering, -1, 1)) * MAX_STEER
            nx = self.x + self.v * math.cos(self.yaw) * DT
            ny = self.y + self.v * math.sin(self.yaw) * DT
            if any(math.hypot(nx - ox, ny - oy) < r + 0.5 for ox, oy, r in self.obstacles):
                self.v, self.collisions = 0.0, self.collisions + 1
            else:
                self.x, self.y = nx, ny
            self.yaw += self.v / WHEELBASE * math.tan(steer) * DT
        self.t += DT

    # API
    def confirmConnection(self):
        pass

    def enableApiControl(self, on, vehicle_name=""):
        pass

    def setCarControls(self, controls, vehicle_name=""):
        self.cmd = controls
        self.step()

    def getCarState(self, vehicle_name=""):
        h = self.yaw / 2
        k = types.SimpleNamespace(position=Vec(self.x, self.y, self.z), orientation=Quat(math.cos(h), 0, 0, math.sin(h)))
        return types.SimpleNamespace(kinematics_estimated=k, speed=abs(self.v))

    def getLidarData(self, lidar_name="", vehicle_name=""):
        """A 360 deg sweep at 1 m above the ground: hits on the obstacles within 30 m, plus a
        ground hit ring (below the obstacle height band, so not an obstacle)."""
        a = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        d = np.stack([np.cos(a), np.sin(a)], axis=1)                      # (rays, 2)
        best = np.full(len(a), 30.0)
        if self.obstacles:
            obs = np.asarray(self.obstacles, float)                        # (n, 3)
            rel = obs[:, :2] - [self.x, self.y]                            # (n, 2)
            along = d @ rel.T                                              # (rays, n)
            perp2 = (rel ** 2).sum(1)[None, :] - along ** 2
            r2 = obs[:, 2][None, :] ** 2
            hit = (along > 0) & (perp2 <= r2)
            dist = np.where(hit, along - np.sqrt(np.maximum(r2 - perp2, 0)), np.inf)
            best = np.minimum(best, dist.min(axis=1))
        pts = []
        for k in range(len(a)):
            if best[k] < 30:
                pts += [self.x + d[k, 0] * best[k], self.y + d[k, 1] * best[k], self.z - 0.2]
            pts += [self.x + d[k, 0] * 6, self.y + d[k, 1] * 6, self.z + 0.3]   # the ground
        return types.SimpleNamespace(point_cloud=pts)

    def simSetObjectPose(self, tag, pose, teleport=True):
        self.kits[tag] = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
        return True

    def simPlotLineStrip(self, *a, **k):
        pass


class Inbox:
    """VictimSubscriber stand-in: messages appear at given simulated times."""

    def __init__(self, car, schedule):
        self.car, self.schedule = car, sorted(schedule, key=lambda m: m[0])

    def poll(self):
        due = [m for t, m in self.schedule if t <= self.car.t]
        self.schedule = [(t, m) for t, m in self.schedule if t > self.car.t]
        return due


def victim(vid, x, y, z=0.0):
    return {"id": vid, "position": {"x": x, "y": y, "z": z}}


@pytest.fixture
def ugv_mod(monkeypatch, tmp_path):
    global OUT
    OUT = tmp_path
    fake = types.ModuleType("airsim")

    class CarControls:
        throttle = steering = brake = 0.0

    fake.CarControls = CarControls
    fake.Vector3r = Vec
    fake.Quaternionr = Quat
    fake.Pose = lambda position, orientation: types.SimpleNamespace(position=position, orientation=orientation)
    monkeypatch.setitem(sys.modules, "airsim", fake)
    monkeypatch.setitem(sys.modules, "setup_path", types.ModuleType("setup_path"))
    sys.modules.pop("ugv", None)
    import ugv
    from survey_plan import load_config
    return ugv, load_config()


def make(ugv_mod, obstacles, schedule):
    mod, cfg = ugv_mod
    mod.OUTPUT_DIR = OUT
    cfg["ugv"]["spawn_ned"] = [0, 0, -2]
    car = SimCar(obstacles)
    u = mod.Ugv(cfg, client=car, subscriber=Inbox(car, schedule))
    u.now = lambda: car.t
    u.t0 = 0.0
    u.sleep = lambda s: None
    return u, car


def test_drives_around_trees_to_the_victim_drops_a_kit_and_returns_home(ugv_mod, capsys):
    trees = [(12, y, 0.6) for y in (-3, -1.5, 0, 1.5, 3)] + [(20, 6, 1.0)]   # a row of trees across the way
    u, car = make(ugv_mod, trees, [(0.0, victim(1, 25, 0))])
    u.run(once=True)
    out = capsys.readouterr().out
    assert car.collisions == 0
    assert "MedKit_1 placed beside victim 1" in out
    kx, ky, kz = car.kits["MedKit_1"]
    assert math.hypot(kx - 25, ky) == pytest.approx(u.u["kit_offset_m"], abs=0.05)
    assert kz == pytest.approx(-0.15)
    assert math.hypot(car.x, car.y) <= u.u["home_radius_m"] + 0.3   # back home
    assert len(u.served) == 1


def test_the_next_victim_is_served_before_going_home(ugv_mod, capsys):
    u, car = make(ugv_mod, [], [(0.0, victim(1, 15, 0)), (0.0, victim(2, 15, 10))])
    u.run(once=True)
    out = capsys.readouterr().out
    assert [v["id"] for v in u.served] == [1, 2]
    assert out.index("victim 2") < out.index("driving to home")


def test_a_victim_published_on_the_way_home_turns_the_ugv_around(ugv_mod, capsys):
    u, car = make(ugv_mod, [], [(0.0, victim(1, 20, 0))])
    # Publish victim 2 once the UGV is on its way home (after its first kit).
    original = u.drop_kit

    def drop_and_schedule(v):
        ok = original(v)
        if v["id"] == 1:
            u.subscriber.schedule.append((car.t + 3.0, victim(2, 18, -12)))
        return ok
    u.drop_kit = drop_and_schedule
    u.run(once=True)
    out = capsys.readouterr().out
    assert "new victim on the way home" in out
    assert [v["id"] for v in u.served] == [1, 2]
    assert "MedKit_2 placed beside victim 2" in out


def test_duplicates_are_ignored_and_running_out_of_kits_is_reported(ugv_mod, capsys):
    mod, cfg = ugv_mod
    u, car = make(ugv_mod, [], [(0.0, victim(1, 10, 0)), (0.0, victim(1, 10, 0)), (0.0, victim(7, 11, 1))])
    u.kits = []
    u.run(once=True)
    out = capsys.readouterr().out
    assert len(u.served) == 1             # id 1 twice, and 7 within 3 m of it: one victim
    assert "no medical kits left" in out


def test_unreachable_victim_is_given_up_and_the_ugv_goes_home(ugv_mod, capsys):
    ring = [(30 + 4 * math.cos(a), 4 * math.sin(a), 0.5) for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    u, car = make(ugv_mod, ring, [(0.0, victim(1, 30, 0))])
    u.grid = OccupancyGrid((0, 0), 90, 0.25)  # a smaller map: a failed search covers all of it
    u.u["no_path_timeout_s"] = 3
    u.run(once=True)
    out = capsys.readouterr().out
    assert "giving up" in out or "took too long" in out
    if "giving up" in out:
        assert "every way there is blocked" in out    # the victim's surroundings are free; the ring isn't
        assert list(OUT.glob("ugv_noway_*.png"))
    assert len(u.failed) == 1 and not u.served
    assert car.collisions == 0
    assert math.hypot(car.x, car.y) <= u.u["home_radius_m"] + 0.3


def test_check_reports_points_and_the_steering_sign(ugv_mod, capsys):
    u, car = make(ugv_mod, [(8, 0, 0.5)], [])
    car.enableApiControl = lambda *a, **k: None
    u.check()
    out = capsys.readouterr().out
    assert "points in one sweep" in out
    assert "steering_sign should be 1" in out       # the simulated car turns right for steering > 0


def test_check_says_when_the_husky_doesnt_move(ugv_mod, capsys):
    u, car = make(ugv_mod, [], [])
    car.enableApiControl = lambda *a, **k: None
    car.setCarControls = lambda controls, vehicle_name="": car.step()   # controls never reach the car
    u.check()
    out = capsys.readouterr().out
    assert "DIDN'T MOVE" in out and "steering_sign should be" not in out


def test_a_gap_too_narrow_for_the_safety_margin_is_squeezed_through(ugv_mod, capsys):
    # A wall of posts across the whole map with a 2.4 m gap: wide enough for the robot (radius
    # 0.6 m) but not with the 0.4 m margin on top. (With a way around the wall's end, it would
    # rather take that.)
    posts = [(10, y, 0.25) for y in np.arange(-31, -1.2 + 0.01, 0.4)] + \
            [(10, y, 0.25) for y in np.arange(1.2, 31.01, 0.4)]
    u, car = make(ugv_mod, posts, [(0.0, victim(1, 20, 0))])
    u.grid = OccupancyGrid((0, 0), 60, 0.25)
    u.run(once=True)
    out = capsys.readouterr().out
    assert "narrow gap" in out
    assert len(u.served) == 1 and car.collisions == 0


def test_boxed_in_is_explained(ugv_mod):
    u, car = make(ugv_mod, [], [])
    ring = [(3 * math.cos(a), 3 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 300)]
    u.grid.update((0, 0), ring)
    why = u.explain_no_path(np.zeros(2), np.array([20.0, 0]), 2.0, "victim 1")
    assert why.startswith("the UGV is boxed in where it is")


def test_branches_above_the_husky_are_not_obstacles(ugv_mod):
    u, car = make(ugv_mod, [], [])
    ground = car.z + u.u["base_height_m"]          # NED: up is -z
    pts = [5, 0, ground - 2.0,                     # a branch 2 m up: the Husky passes under
           6, 0, ground - 0.8]                     # a trunk or wall at 0.8 m: an obstacle
    car.getLidarData = lambda **k: types.SimpleNamespace(point_cloud=pts)
    u.sense()
    occ = u.grid.occupied()
    assert not occ[tuple(u.grid.to_cell((5, 0)))]
    assert occ[tuple(u.grid.to_cell((6, 0)))]
