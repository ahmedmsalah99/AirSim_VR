"""Run mission.py against a fake AirSim, camera, detector and publisher."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUND_Z = 4.0                       # drone frame: the drone settled 4 m below its spawn point
PERSON = np.array([10.0, 3.0, GROUND_Z])  # drone frame


class Future:
    def __init__(self, client, target=None):
        self.client, self.target = client, target

    def join(self):
        if self.target is not None:
            self.client.pos = np.array(self.target, float)
            self.client.moving = None


class FakeClient:
    def __init__(self, port):
        self.pos = np.array([0.0, 0.0, GROUND_Z])
        self.moving = None
        self.moves, self.markers = [], []

    def __getattr__(self, name):
        return lambda *a, **k: Future(self)

    def getMultirotorState(self, vehicle_name):
        if self.moving is not None:  # an unjoined move advances 2 m per poll
            step = self.moving - self.pos
            n = np.linalg.norm(step)
            self.pos = self.moving.copy() if n < 2 else self.pos + step / n * 2
        p = types.SimpleNamespace(x_val=self.pos[0], y_val=self.pos[1], z_val=self.pos[2])
        landed = 0 if self.pos[2] >= GROUND_Z - 0.1 else 1
        return types.SimpleNamespace(landed_state=landed, kinematics_estimated=types.SimpleNamespace(position=p))

    def moveToPositionAsync(self, x, y, z, v, **kw):
        self.moves.append((np.array([x, y, z]), v))
        self.moving = np.array([x, y, z], float)
        return Future(self, (x, y, z))

    def moveToZAsync(self, z, v, **kw):
        return Future(self, (self.pos[0], self.pos[1], z))

    def landAsync(self, **kw):
        return Future(self, (self.pos[0], self.pos[1], GROUND_Z))

    def hoverAsync(self, **kw):
        self.moving = None
        return Future(self)

    def simGetObjectPose(self, tag):
        nan = float("nan")
        world = PERSON[:2] + SPAWN[:2] if tag == "Victim_01" else (nan, nan)
        return types.SimpleNamespace(position=types.SimpleNamespace(x_val=world[0], y_val=world[1]))

    def simPlotPoints(self, points, **kw):
        self.markers.append(kw.get("color_rgba"))


class FakeCamera:
    def __init__(self, client, airsim, cfg):
        self.tilts = []
        self.depth_grabs = 0

    def set_tilt(self, tilt):
        self.tilts.append(tilt)
        return True

    def grab(self, with_depth=False):
        self.depth_grabs += with_depth
        return types.SimpleNamespace(image=None)

    def ground_point(self, frame, u, v, ground_z):
        return PERSON.copy()

    def depth_point(self, frame, u, v):
        return PERSON + np.array([0.2, -0.1, -0.05])


class FakeDetector:
    def __init__(self, cfg):
        pass

    def detect(self, image):
        return [types.SimpleNamespace(center=(640, 400), y2=420, confidence=0.8)]


PUBLISHED = []


class FakePublisher:
    def __init__(self, endpoint, topic):
        pass

    def publish(self, victim_id, position, std, confidence, hits, frames):
        PUBLISHED.append((victim_id, np.array(position), hits, frames))


SPAWN = None


@pytest.fixture
def mission(monkeypatch, tmp_path):
    global SPAWN
    fake = types.ModuleType("airsim")
    fake.MultirotorClient = FakeClient
    fake.YawMode = lambda *a: a
    fake.Vector3r = lambda *a: a
    fake.LandedState = types.SimpleNamespace(Landed=0, Flying=1)
    monkeypatch.setitem(sys.modules, "airsim", fake)
    monkeypatch.setitem(sys.modules, "setup_path", types.ModuleType("setup_path"))
    cam_mod = types.ModuleType("drone_camera")
    cam_mod.DroneCamera, cam_mod.STRAIGHT_DOWN = FakeCamera, 89.5
    monkeypatch.setitem(sys.modules, "drone_camera", cam_mod)
    det_mod = types.ModuleType("detector")
    det_mod.PersonDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "detector", det_mod)
    pub_mod = types.ModuleType("publisher")
    pub_mod.VictimPublisher = FakePublisher
    monkeypatch.setitem(sys.modules, "publisher", pub_mod)
    monkeypatch.setattr("time.sleep", lambda s: None)
    sys.modules.pop("mission", None)
    import mission
    monkeypatch.setattr(mission, "OUTPUT_DIR", tmp_path)
    PUBLISHED.clear()
    from survey_plan import load_config
    cfg = load_config()
    SPAWN = np.array(cfg["drone"]["spawn_ned"], float)
    return mission, cfg


def test_confirm_viewpoint_on_line_of_sight():
    from camera_geometry import confirm_viewpoint
    drone, target = np.array([0.0, 0.0, -20.0]), np.array([20.0, 0.0, 0.0])
    point, heading, tilt = confirm_viewpoint(drone, target, 10, lowest_z=-2)
    assert tilt == pytest.approx(45)
    assert heading == pytest.approx(0)
    assert np.linalg.norm(point - target) == pytest.approx(10)
    assert np.cross(point - target, drone - target) == pytest.approx([0, 0, 0], abs=1e-9)  # on the line

    point, _, _ = confirm_viewpoint(drone, target, 1, lowest_z=-6)   # would go below 6 m
    assert point[2] == pytest.approx(-6)
    point, _, _ = confirm_viewpoint(drone, target, 100, lowest_z=-2)  # never farther than the drone
    assert point == pytest.approx(drone)


def test_mission_without_detection_flies_every_lane_and_lands(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    m = mod.Mission(cfg, plan, detect=False)
    m.run()
    visited = [p[:2] + SPAWN[:2] for p, _ in m.client.moves]
    assert visited[:-1] == [pytest.approx(w) for w in plan.waypoints]
    assert visited[-1] == pytest.approx(SPAWN[:2])
    assert m.client.pos[2] == pytest.approx(GROUND_Z)


def test_person_is_confirmed_published_once_and_the_lane_resumes(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    m = mod.Mission(cfg, plan, detect=True)
    m.run()

    assert len(PUBLISHED) == 1, "seen in every frame, but published only once"
    vid, world, hits, frames = PUBLISHED[0]
    assert hits == frames == cfg["confirm"]["frames"]
    assert world[:2] == pytest.approx(PERSON[:2] + SPAWN[:2] + [0.2, -0.1])  # from the depth camera
    assert m.camera.depth_grabs == 1  # float depth is slow: once, for the final position

    confirm_moves = [p for p, v in m.client.moves if v == cfg["confirm"]["speed"]]
    approach, back = confirm_moves
    assert np.linalg.norm(approach - PERSON) < np.linalg.norm(back - PERSON)  # went closer, then returned
    assert back[2] == pytest.approx(GROUND_Z - plan.altitude)                # back at survey altitude

    tilts = m.camera.tilts
    assert tilts[0] == plan.tilt_deg and tilts[-1] == 89.5
    los = tilts[1]
    assert los != plan.tilt_deg and tilts[2] == plan.tilt_deg               # line of sight, then restored
    assert m.client.pos == pytest.approx([SPAWN[0] * 0, 0, GROUND_Z])       # landed back at home
    assert any(math.isclose(c[1], 1.0) and c[0] == 0.0 for c in m.client.markers)  # green marker


def test_unreachable_confirm_point_does_not_hang_the_mission(mission, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey

    class Blocked(FakeClient):
        def moveToPositionAsync(self, x, y, z, v, **kw):
            if v == cfg["confirm"]["speed"] and len([m for m in self.moves if m[1] == v]) == 0:
                assert "timeout_sec" in kw  # a waited-on move must be able to give up
                self.moves.append((np.array([x, y, z]), v))
                return Future(self)  # the approach never gets anywhere
            return super().moveToPositionAsync(x, y, z, v, **kw)

    monkeypatch_client = mod.airsim.MultirotorClient
    mod.airsim.MultirotorClient = Blocked
    try:
        m = mod.Mission(cfg, plan_survey(cfg), detect=True)
        m.run()
    finally:
        mod.airsim.MultirotorClient = monkeypatch_client
    out = capsys.readouterr().out
    assert "approach stopped" in out and "landed" in out
