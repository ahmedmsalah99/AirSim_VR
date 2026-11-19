"""Run mission.py against a fake AirSim, camera, detector and publisher."""
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
        v = types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0)
        landed = 0 if self.pos[2] >= GROUND_Z - 0.1 else 1
        return types.SimpleNamespace(landed_state=landed,
                                     kinematics_estimated=types.SimpleNamespace(position=p, linear_velocity=v))

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
        self.client = client
        self.tilts = []
        self.fovs = []
        self.depth_grabs = 0

    def set_tilt(self, tilt):
        self.tilts.append(tilt)
        return True

    def center_depth(self):
        return GROUND_Z - self.client.pos[2]  # looking straight down from the fake drone

    def set_fov(self, fov):
        self.fovs.append(fov)

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


def one_drone(cls):
    """A MultirotorClient stand-in: every connection talks to the same fake drone, as
    reconnecting to AirSim does."""
    drone = {}

    def connect(port):
        if "it" not in drone:
            drone["it"] = cls(port)
        return drone["it"]
    return connect


@pytest.fixture
def mission(monkeypatch, tmp_path):
    global SPAWN
    fake = types.ModuleType("airsim")
    fake.MultirotorClient = one_drone(FakeClient)
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
    cfg["confirm"]["save_images"] = False
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
    assert world[:2] == pytest.approx(PERSON[:2] + SPAWN[:2])  # median of the ground projections
    assert m.camera.depth_grabs == 0  # depth is off by default (slow in AirSim)
    zoom, unzoom = m.camera.fovs[:2]  # (the mission resets the zoom once more at the end)
    assert zoom < cfg["camera"]["hfov_deg"] and unzoom == cfg["camera"]["hfov_deg"]

    confirm_moves = [p for p, v in m.client.moves if v == cfg["confirm"]["speed"]]
    approach, back = confirm_moves
    assert np.linalg.norm(approach - PERSON) < np.linalg.norm(back - PERSON)  # went closer, then returned
    assert back[2] == pytest.approx(GROUND_Z - plan.altitude)                # back at survey altitude
    assert GROUND_Z - approach[2] >= cfg["confirm"]["min_altitude"] - 1e-9   # never below min_altitude

    tilts = m.camera.tilts
    assert tilts[:2] == [89.5, plan.tilt_deg]          # straight down to measure the ground, then search
    los = tilts[2]
    assert los != plan.tilt_deg and tilts[3] == plan.tilt_deg  # line of sight, then restored
    assert tilts[-1] == 89.5
    assert m.client.pos == pytest.approx([SPAWN[0] * 0, 0, GROUND_Z])       # landed back at home
    assert m.client.markers == [mod.CONFIRMED_COLOR]  # nothing drawn on the person before it's decided


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

    mod.airsim.MultirotorClient = one_drone(Blocked)
    try:
        m = mod.Mission(cfg, plan_survey(cfg), detect=True)
        m.run()
    finally:
        mod.airsim.MultirotorClient = one_drone(FakeClient)
    out = capsys.readouterr().out
    assert "approach stopped" in out and "landed" in out


def test_depth_is_used_once_when_enabled(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    cfg["confirm"]["use_depth"] = True
    m = mod.Mission(cfg, plan_survey(cfg), detect=True)
    m.run()
    assert m.camera.depth_grabs == 1
    assert PUBLISHED[0][1][:2] == pytest.approx(PERSON[:2] + SPAWN[:2] + [0.2, -0.1])


class Interrupt(FakeCamera):
    """Ctrl+C on the 3rd frame."""
    fired = False

    def __init__(self, client, airsim, cfg):
        super().__init__(client, airsim, cfg)
        self.grabs = 0

    def grab(self, with_depth=False):
        self.grabs += 1
        if self.grabs == 3:
            Interrupt.fired = True
            raise KeyboardInterrupt
        return super().grab(with_depth)


class SecondInterrupt(FakeClient):
    """Ctrl+C again on the first move after the first one (the flight home)."""

    def __init__(self, port):
        super().__init__(port)
        self.stopped_at = None

    def moveToPositionAsync(self, x, y, z, v, **kw):
        if Interrupt.fired and self.stopped_at is None:
            self.stopped_at = self.pos.copy()
            raise KeyboardInterrupt
        return super().moveToPositionAsync(x, y, z, v, **kw)


def run_interrupted(mod, cfg, client=FakeClient):
    from survey_plan import plan_survey
    Interrupt.fired = False
    mod.DroneCamera = Interrupt
    mod.airsim.MultirotorClient = one_drone(client)
    try:
        m = mod.Mission(cfg, plan_survey(cfg), detect=True)
        m.run()
    finally:
        mod.airsim.MultirotorClient = one_drone(FakeClient)
    return m


def test_ctrl_c_flies_home_and_lands(mission, capsys):
    mod, cfg = mission
    m = run_interrupted(mod, cfg)
    out = capsys.readouterr().out
    assert "stopped by user" in out and "returning home" in out and "landed" in out
    assert m.client.pos == pytest.approx([0, 0, GROUND_Z])        # home, on the ground
    assert m.camera.fovs[-1] == cfg["camera"]["hfov_deg"]         # zoom reset


def test_second_ctrl_c_lands_where_it_is(mission, capsys):
    mod, cfg = mission
    m = run_interrupted(mod, cfg, SecondInterrupt)
    out = capsys.readouterr().out
    assert "stopped again: landing here" in out and "landed" in out
    assert m.client.pos[:2] == pytest.approx(m.client.stopped_at[:2])  # landed where it was stopped
    assert m.client.pos[2] == pytest.approx(GROUND_Z)


class StuckLanding(FakeClient):
    """landAsync returns at once the first time, leaving the drone in the air."""

    def __init__(self, port):
        super().__init__(port)
        self.lands = 0

    def landAsync(self, **kw):
        self.lands += 1
        return Future(self) if self.lands == 1 else super().landAsync(**kw)


def test_landing_is_checked_and_finished(mission, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    mod.airsim.MultirotorClient = one_drone(StuckLanding)
    try:
        m = mod.Mission(cfg, plan_survey(cfg), detect=False)
        m.run()
    finally:
        mod.airsim.MultirotorClient = one_drone(FakeClient)
    out = capsys.readouterr().out
    assert "descending the rest" in out
    assert m.client.pos[2] == pytest.approx(GROUND_Z)
    assert "landed (0.0 m above the ground)" in out


class HeldInTheAir(FakeClient):
    """The drone starts 5 m above the ground, held at its spawn height."""

    def __init__(self, port):
        super().__init__(port)
        self.pos = np.array([0.0, 0.0, GROUND_Z - 5])

    def landAsync(self, **kw):
        if self.pos[2] < GROUND_Z - 5 - 0.1:     # AirSim's land stops at the height it started from
            return Future(self, (self.pos[0], self.pos[1], GROUND_Z - 5))
        return Future(self)


def test_ground_is_measured_not_assumed(mission, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    mod.airsim.MultirotorClient = one_drone(HeldInTheAir)
    try:
        plan = plan_survey(cfg)
        m = mod.Mission(cfg, plan, detect=False)
        m.run()
    finally:
        mod.airsim.MultirotorClient = one_drone(FakeClient)
    out = capsys.readouterr().out
    assert "ground is 5.0 m below the drone" in out
    assert m.ground_z == pytest.approx(GROUND_Z)
    lane_z = {round(p[2], 6) for p, v in m.client.moves if v == plan.speed}
    assert lane_z == {round(GROUND_Z - plan.altitude, 6)}       # searched at the planned height above the real ground
    assert "descending the rest" in out
    assert m.client.pos[2] == pytest.approx(GROUND_Z)          # landed on the real ground


def test_search_logs_detections_and_why_they_were_dropped(mission, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    cfg["area"]["corner_a"], cfg["area"]["corner_b"] = [100, 100], [150, 150]  # the person is outside it
    mod.REPORT_EVERY_S = 0
    m = mod.Mission(cfg, plan_survey(cfg), detect=True)
    m.run()
    out = capsys.readouterr().out
    assert "person detection(s) (dropped:" in out and "outside the area" in out
    assert not m.candidates


def test_search_slows_down_when_frames_are_slow(mission, capsys, monkeypatch):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    monkeypatch.setattr(mod.Mission, "frame_rate", lambda self, frames=3: 0.2)  # one frame every 5 s
    m = mod.Mission(cfg, plan, detect=True)
    m.run()
    band = plan.band.far_m - plan.band.near_m
    expected = band * 0.2 / cfg["search"]["min_frames_per_point"]
    assert expected < plan.speed
    assert m.speed == pytest.approx(expected)
    assert {v for _, v in m.client.moves if abs(v - expected) < 1e-9}       # the lanes were flown at it
    assert "Use Less CPU when in Background" in capsys.readouterr().out


def test_fast_frames_keep_the_planned_speed(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    m = mod.Mission(cfg, plan, detect=True)
    m.run()
    assert m.speed == plan.speed


def test_blocked_first_view_is_confirmed_from_the_arc(mission, monkeypatch, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    real_look = mod.Mission.look
    monkeypatch.setattr(mod.Mission, "look",
                        lambda self, cand, view: ([], []) if view == "+0" else real_look(self, cand, view))
    m = mod.Mission(cfg, plan, detect=True)
    m.run()

    assert len(PUBLISHED) == 1
    out = capsys.readouterr().out
    assert "moving +30 deg around it" in out and "from the +30 deg view" in out
    slow = [p for p, v in m.client.moves if v == cfg["confirm"]["speed"]]
    approach, arc, back = slow[0], slow[1:-1], slow[-1]
    radius = np.hypot(*(approach - PERSON)[:2])
    assert len(arc) >= 6                                       # out to +30 and back, in 10 deg steps
    for p in arc:
        assert np.hypot(*(p - PERSON)[:2]) == pytest.approx(radius)  # same distance from the person
        assert p[2] == pytest.approx(approach[2])                    # same height
    assert arc[-1] == pytest.approx(approach)                  # back to the approach point ...
    assert back[2] == pytest.approx(GROUND_Z - plan.altitude)  # ... then up to the lane


def test_rejected_candidate_is_retried_from_a_new_viewpoint(mission, monkeypatch, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    real_look = mod.Mission.look
    monkeypatch.setattr(mod.Mission, "look",
                        lambda self, cand, view: ([], []) if cand.attempts == 1 else real_look(self, cand, view))
    m = mod.Mission(cfg, plan_survey(cfg), detect=True)
    m.run()

    out = capsys.readouterr().out
    assert "rejected candidate 1" in out and "trying again" in out
    (cand,) = m.candidates
    assert cand.attempts == 2 and cand.status == "confirmed"
    assert len(PUBLISHED) == 1


def test_rejected_candidate_is_not_retried_from_the_same_place_or_too_often(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    m = mod.Mission(cfg, plan_survey(cfg), detect=True)
    cand = mod.Candidate(1, PERSON.copy(), 0.5, status="rejected", attempts=1, last_stop=m.position())
    assert not m.can_retry(cand)                       # still where it was tried
    m.client.pos = m.client.pos + [cfg["confirm"]["retry_distance"] + 1, 0, 0]
    assert m.can_retry(cand)
    cand.attempts = cfg["confirm"]["max_attempts"]
    assert not m.can_retry(cand)


def test_weak_detections_dont_start_a_confirmation(mission, monkeypatch, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    monkeypatch.setattr(FakeDetector, "detect",
                        lambda self, image: [types.SimpleNamespace(center=(640, 400), y2=420, confidence=0.3)])
    m = mod.Mission(cfg, plan_survey(cfg), detect=True)
    m.run()
    assert not m.candidates and not PUBLISHED
    assert "below 0.4 confidence" in capsys.readouterr().out


def test_one_victim_is_viewed_from_the_search_band_confirmed_and_landed(mission, capsys):
    mod, cfg = mission
    from survey_plan import plan_survey
    plan = plan_survey(cfg)
    m = mod.Mission(cfg, plan, detect=True, victim="Victim_01")
    m.run()

    assert len(PUBLISHED) == 1
    view = m.client.moves[0][0]  # first move: to the search view, no lanes flown
    ahead = (plan.band.near_m + plan.band.far_m) / 2
    assert np.linalg.norm(view[:2] - PERSON[:2]) == pytest.approx(ahead)
    assert view[2] == pytest.approx(GROUND_Z - plan.altitude)
    assert "lane 1/" not in capsys.readouterr().out
    assert m.client.pos == pytest.approx([0, 0, GROUND_Z])  # landed back at home


def test_unknown_victim_tag_aborts_and_lands(mission):
    mod, cfg = mission
    from survey_plan import plan_survey
    m = mod.Mission(cfg, plan_survey(cfg), detect=True, victim="Nobody")
    with pytest.raises(RuntimeError, match="Nobody"):
        m.run()
    assert m.client.pos[2] == pytest.approx(GROUND_Z)
