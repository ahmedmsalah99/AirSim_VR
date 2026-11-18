"""Run fly_survey against a fake AirSim client and check the commands it sends."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Done:
    def join(self):
        pass


class FakeClient:
    GROUND_Z = 4.0  # drone settled 4 m below its spawn point (spawned 5 m up)

    def __init__(self, port):
        self.moves = []
        self.z_moves = []

    def __getattr__(self, name):
        return lambda *a, **k: Done()

    def getMultirotorState(self, vehicle_name):
        pos = types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=self.GROUND_Z)
        return types.SimpleNamespace(kinematics_estimated=types.SimpleNamespace(position=pos))

    def simGetCameraInfo(self, name, vehicle_name):
        return types.SimpleNamespace(fov=60.0)

    def moveToPositionAsync(self, x, y, z, v, **kw):
        self.moves.append((x, y, z, v))
        return Done()

    def moveToZAsync(self, z, v, **kw):
        self.z_moves.append(z)
        return Done()


@pytest.fixture
def fly_survey(monkeypatch):
    fake = types.ModuleType("airsim")
    fake.MultirotorClient = FakeClient
    fake.YawMode = lambda *a: a
    fake.Vector3r = lambda *a: a
    monkeypatch.setitem(sys.modules, "airsim", fake)
    monkeypatch.setitem(sys.modules, "setup_path", types.ModuleType("setup_path"))
    sys.modules.pop("fly_survey", None)
    import fly_survey
    return fly_survey


def test_flies_every_lane_in_drone_frame(fly_survey):
    from survey_plan import load_config, plan_survey
    cfg = load_config()
    cfg["drone"]["spawn_ned"] = [10, -20, -5]
    plan = plan_survey(cfg)

    survey = fly_survey.Survey(cfg, plan)
    survey.log = lambda msg: None
    survey.run()
    client = survey.client

    expected_z = FakeClient.GROUND_Z - plan.altitude
    assert client.z_moves == [pytest.approx(expected_z)]
    assert all(z == pytest.approx(expected_z) for _, _, z, _ in client.moves)

    visited = [(x + 10, y - 20) for x, y, _, _ in client.moves]
    assert visited[:-1] == [pytest.approx(p) for p in plan.waypoints]
    assert visited[-1] == pytest.approx((10, -20))       # back over the spawn point
    lane_speeds = {v for _, _, _, v in client.moves[1:-1]}
    assert lane_speeds == {plan.speed}
