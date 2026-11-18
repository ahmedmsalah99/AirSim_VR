"""Fly the lawnmower search planned by survey_plan.py and draw it in the scene.

Start Play in Unreal first (SimMode "Both"), then:

    python fly_survey.py [config.yaml]

The drone takes off where it spawned, climbs straight up to the survey
altitude, flies the lanes, comes back and lands. Ctrl+C stops it in a hover.
"""
import sys
import time

import setup_path  # noqa: F401  (puts this repo's airsim package on the path)
import airsim

from survey_plan import DEFAULT_CONFIG, InfeasiblePlan, format_plan, load_config, plan_survey

LANE_COLOR = [0.0, 1.0, 0.2, 1.0]
AREA_COLOR = [1.0, 0.6, 0.0, 1.0]


class Survey:
    def __init__(self, cfg, plan):
        self.cfg, self.plan = cfg, plan
        drone = cfg["drone"]
        self.name = drone["name"]
        # Commands and state are relative to the drone's spawn point; the plan and the
        # debug drawing use world NED (relative to PlayerStart).
        self.spawn = drone["spawn_ned"]
        self.client = airsim.MultirotorClient(port=drone["port"])
        self.t0 = time.time()

    def log(self, msg):
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def to_local(self, x, y):
        return x - self.spawn[0], y - self.spawn[1]

    def run(self):
        c = self.client
        c.confirmConnection()
        self.check_camera()
        self.start_camera_window()
        c.enableApiControl(True, self.name)
        c.armDisarm(True, self.name)

        # Ground under the takeoff point, in the drone's frame; altitudes are measured from it.
        ground_z = c.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position.z_val
        self.z = ground_z - self.plan.altitude
        self.draw(ground_z + self.spawn[2])

        yaw = airsim.YawMode(False, 0 if self.plan.lane_axis == "x" else 90)
        drone = self.cfg["drone"]

        self.log("taking off")
        c.takeoffAsync(vehicle_name=self.name).join()
        self.log(f"climbing to {self.plan.altitude:.1f} m")
        c.moveToZAsync(self.z, drone["climb_speed"], yaw_mode=yaw, vehicle_name=self.name).join()

        first = self.plan.waypoints[0]
        self.log(f"transit to the first lane at {drone['transit_speed']} m/s")
        self.move_to(first, drone["transit_speed"], yaw)

        for lane in range(self.plan.lane_count):
            end = self.plan.waypoints[2 * lane + 1]
            if lane > 0:
                self.move_to(self.plan.waypoints[2 * lane], self.plan.speed, yaw)
            self.log(f"lane {lane + 1}/{self.plan.lane_count} at {self.plan.speed:.1f} m/s")
            self.move_to(end, self.plan.speed, yaw)

        self.log("search done, returning home")
        self.move_to(tuple(self.spawn[:2]), drone["transit_speed"], yaw)
        c.landAsync(vehicle_name=self.name).join()
        c.armDisarm(False, self.name)
        self.log("landed")

    def move_to(self, xy, speed, yaw):
        x, y = self.to_local(*xy)
        self.client.moveToPositionAsync(x, y, self.z, speed, yaw_mode=yaw, vehicle_name=self.name).join()

    def check_camera(self):
        cam = self.cfg["camera"]
        info = self.client.simGetCameraInfo(cam["name"], vehicle_name=self.name)
        if abs(info.fov - cam["hfov_deg"]) > 0.5:
            self.log(f"WARNING: camera {cam['name']} has FOV {info.fov:.1f} deg in AirSim but "
                     f"{cam['hfov_deg']} in config.yaml; the plan's footprint is wrong. "
                     f"Copy the repo's settings.json to Documents\\AirSim again.")

    def start_camera_window(self):
        # AirSim's camera subwindow stays black until its camera has served one API image request.
        request = airsim.ImageRequest(self.cfg["camera"]["name"], airsim.ImageType.Scene)
        self.client.simGetImages([request], vehicle_name=self.name)

    def draw(self, ground_world_z):
        z = ground_world_z - self.plan.altitude
        lanes = [airsim.Vector3r(x, y, z) for x, y in self.plan.waypoints]
        (ax, ay), (bx, by) = self.cfg["area"]["corner_a"], self.cfg["area"]["corner_b"]
        gz = ground_world_z - 0.5
        area = [airsim.Vector3r(x, y, gz) for x, y in [(ax, ay), (ax, by), (bx, by), (bx, ay), (ax, ay)]]
        self.client.simFlushPersistentMarkers()
        self.client.simPlotLineStrip(area, color_rgba=AREA_COLOR, thickness=15, is_persistent=True)
        self.client.simPlotLineStrip(lanes, color_rgba=LANE_COLOR, thickness=8, is_persistent=True)
        self.client.simPlotStrings(["start"], [lanes[0]], scale=3, color_rgba=LANE_COLOR, duration=600)

    def stop(self):
        self.log("stopping: hovering in place (API control kept)")
        self.client.cancelLastTask(self.name)
        self.client.hoverAsync(vehicle_name=self.name).join()


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
    try:
        plan = plan_survey(cfg)
    except InfeasiblePlan as e:
        sys.exit(f"Plan is infeasible: {e}")
    print(format_plan(plan))

    survey = Survey(cfg, plan)
    try:
        survey.run()
    except KeyboardInterrupt:
        survey.stop()


if __name__ == "__main__":
    main()
