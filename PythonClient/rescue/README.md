# Rescue mission

The drone searches a rectangle in a lawnmower pattern with its camera tilted forward. When it sees a person, it approaches along its line of sight to confirm them, publishes their position over ZeroMQ, and carries on with the search. The UGV comes next.

| File | What it does |
|---|---|
| `config.yaml` | Search area, drone limits, camera, detector, confirmation and publishing settings |
| `mission.py` | The mission: search, confirm, publish, resume, return home, report |
| `set_area.py` | Sets the search area from two corner markers placed in the level |
| `ugv.py` | The Husky UGV: drives a medical kit to each confirmed victim, mapping with its lidar as it goes |
| `occupancy_grid.py` | The UGV's lidar occupancy grid and A* path planner |
| `survey_plan.py` | Computes altitude, detection band, lane spacing, speed and lanes, and explains each choice |
| `camera_geometry.py` | Pixel ↔ ground geometry, and the line-of-sight confirm point |
| `drone_camera.py` | The drone camera in AirSim: tilt it, grab frames with depth, locate pixels |
| `detector.py` | Runs the YOLO11 ONNX person detector over image tiles |
| `publisher.py` | Publishes confirmed victims over ZeroMQ |
| `listen_victims.py` | Prints what the mission publishes |
| `detect_once.py` | Tests the detector on one drone camera frame or an image file |
| `compare_detectors.py` | Runs the `.pt` with Ultralytics and the ONNX pipeline on the same image, to check the pipeline |
| `export_onnx.py` | Converts the YOLO11 `.pt` detector to ONNX |

## Setup

```
pip install msgpack-rpc-python numpy pyyaml onnxruntime-directml opencv-python pyzmq
```

`onnxruntime-directml` runs the detector on the GPU (any DirectX 12 GPU on Windows, no CUDA needed). On the CPU, next to Unreal, YOLO took 3.6 s per frame. It replaces plain `onnxruntime`; if that's installed, `pip uninstall onnxruntime` first, since the two clash. The detector uses CUDA if available, else DirectML GPU 1, else GPU 0, else the CPU, and logs which (`detector on DirectML GPU 1: 0.01 s per tile`). On a laptop GPU 0 is the integrated GPU that draws the screen; running the detector there froze the machine, so it's only used when it's the only one. `detector.device` in `config.yaml` can pin one (`dml:0`, `dml:1`, `cuda`, `cpu`).

Copy the repo's `settings.json` to `<Documents>\AirSim` (see the top-level README). Its `CameraDefaults` give every camera, including the drone's `bottom_center`, 1280×720 images with a 60° field of view, which the plan assumes. Don't configure cameras under `Vehicles` → `Cameras`: in this AirSim version that spawns a new camera actor and crashes on Play. The settings also spawn the drone 4 m east of the Husky so it doesn't land on it.

**Turn off Unreal's background throttling:** Edit → Editor Preferences → General → Performance → untick **Use Less CPU when in Background**. While you're in the terminal, the editor otherwise drops to a few frames per second, every camera image waits for a frame, and the search gets a frame every several seconds, so people pass between frames unseen. `mission.py` measures the frame rate before the first lane and slows down if it's too low for the planned speed (and says so). On a slow machine, Settings → Engine Scalability Settings → Medium helps too.

Put the converted model at `models/person_yolo11s.onnx` (see `export_onnx.py`). Tag the people placed in the level `Victim_01`, `Victim_02`, … (Details → Actor → Tags) and list the tags under `ground_truth` in `config.yaml`, so the mission can report its position error.

## Use

1. Set the search rectangle. Easiest: in the editor drag two **Target Points** (Place Actors) onto opposite corners of the area, tag them `Area_A` and `Area_B` (Details → Actor → Tags), press Play and run `python set_area.py`. It writes the corners into `config.yaml`, runs the lanes along the long side, and prints the plan. Or set it in `config.yaml` by hand. Positions are metres from the PlayerStart: `x = (Unreal X - PlayerStart X) / 100`, and the same for `y`. Read the Unreal X/Y off actors in the editor. Also set `max_obstacle_height` to the tallest building or tree inside the rectangle.
2. Check the plan without flying:
   ```
   python survey_plan.py
   ```
3. Press Play in Unreal. In a second terminal, start listening for victims:
   ```
   python listen_victims.py
   ```
4. Run the mission:
   ```
   python mission.py               # search, confirm, publish
   python mission.py --no-detect   # only fly the search pattern
   python mission.py --victim Victim_01   # test one person, no lanes
   ```
   `--victim` flies straight to where a lane would see that tagged person (mid detection band, at the search altitude and tilt, facing them), looks until they're detected (up to 10 frames), confirms them exactly as the search does, then flies home and lands.
   The area (orange) and the lanes (green) are drawn in the scene; confirmed victims are marked green and rejected candidates red, a few metres above the spot. Candidates aren't marked before they're decided: the camera sees debug drawings, and a marker on a person would hide them. Ctrl+C (or an error) ends the mission early: the drone reconnects (a Ctrl+C mid-call breaks AirSim's client), flies home at its current altitude and lands; a second Ctrl+C lands it where it is. Before taking off it measures the ground below with the depth camera (`ground is … m below the drone`); every height, every person's ground position and the landing are relative to it. During the lanes it logs every 5 s how many people it detected and how many it dropped and why. At the end it lands (checking that it's really down) and prints each victim's position and its error against the tagged people. It saves a summary (`output/mission_<time>.json`) and, for every candidate, the search frame that triggered it and each confirm frame with the detections drawn (`output/mission_<time>/candidate<N>_*.jpg`; `confirm.save_images`).
   The window at the bottom of the viewport shows the drone's camera; press `1` in the viewport to hide or show it.

## How the plan is computed

The camera looks forward along the lane, `camera.tilt_deg` below the horizon (90 = straight down). On flat ground it sees a trapezoid. A person covers `person_size × f × sin(a) / h` detector pixels, where `a` is the ray's angle below the horizon, `h` the altitude and `f` the focal length in detector pixels (each image tile is resized to the 640 px input, so `f` shrinks with it). So people are only detectable out to the slant range where they shrink to `min_person_px`.

1. **Detection band:** from the bottom edge of the image (`near`) to wherever comes first, the top edge of the image or that range (`far`). `near` and `far` are distances ahead of the drone.
2. **Swath:** the ground width covered at the **near** edge of the band, `2 × depth × tan(hfov / 2)`, its narrowest. The band is wider at its far edge, but a point out at the far edge's sides is only seen at the far end, where people are smallest and in the fewest frames; spaced by the near-edge width, every point between the lanes crosses the whole band.
3. **Lane spacing** = swath × (1 − `side_overlap`), spread evenly so the outer lanes just reach the edges. Lanes run along `search.lane_axis` (`x` or `y`; `auto` picks whichever gives the shorter mission, usually the long side) and alternate direction; the drone faces its direction of travel. Each lane is shifted back by `near`, so the band starts on the area's edge and ends just past the far edge.
4. **Speed** is the slowest of:
   - `max_speed` of the drone,
   - motion blur: image motion is fastest at the near edge, `v·sin²(a)/h` rad/s, so `v ≤ max_blur_px × h / (f × sin²(a_near) × exposure_s)`,
   - frames per person: a ground point crosses the band in `(far − near) / v` seconds and must be seen in `min_frames_per_point` frames at the detector's frame rate (`fps_per_tile / tiles`).
5. **Altitude:** every altitude from the lowest safe one (`max_obstacle_height + safety_margin`, at least `min_altitude`) up to `max_altitude` is tried in 0.5 m steps, and the fastest mission wins (the lowest altitude among near-ties). If no altitude keeps people detectable, the plan is rejected with suggestions.

## How a person is confirmed

1. **Candidate:** during a lane, each detection's feet pixel (bottom centre of the box) is projected onto the flat ground through the camera pose, which gives a position. A detection with at least `search.min_candidate_confidence` (0.4), more than `merge_radius` from every known one and inside the area, counts towards a new person; once that person has been detected in `search.frames_to_confirm` (3) search frames, within `merge_radius` of each other and none more than 5 frames apart, they become a candidate at the median of those positions (so one stray detection doesn't stop the drone); weaker ones are counted in the search log but don't start a confirmation (confirm frames still count hits down to the detector's `conf_threshold`, 0.25, since a zoomed, partly hidden person can score lower). A **rejected** candidate seen again is tried again (up to `max_attempts`) if the drone is now at least `retry_distance` from where it last tried: the new view may be clear.
2. **Approach:** the drone stops, turns to face the candidate, tilts the camera to the line-of-sight angle, and flies down **along the line of sight** towards the distance where a person is `confirm.min_person_px` tall (`distance = person_size × f / min_person_px`). It never goes below `confirm.min_altitude` (17 m): there's no obstacle map, and trees and roofs stand around people, so going lower than the tallest of them risks collisions.
3. **Zoom:** whatever the approach couldn't do, the camera's zoom does: the field of view narrows until a person at that range is `min_person_px` tall, down to `confirm.min_fov_deg`. The intrinsics used to locate detections follow the zoom.
4. **Check, then arc:** it hovers until it has stopped moving and runs the detector on `frames` frames, `frame_interval_s` apart. A frame is a hit if a person is detected within `match_radius` of the candidate (feet projected onto the flat ground). If fewer than `required` frames hit, something may be in the way (a pole, a branch), so it flies **along an arc around the person**, at the same distance and height and still facing them, to the next angle in `arc_offsets_deg` (30° to one side, then 30° to the other, in `arc_step_deg` steps) and looks again. The camera tilt stays right on the arc, since the line-of-sight angle doesn't change.
5. **Decision:** as soon as one viewpoint has at least `required` hits, the person is confirmed: the median of the hits' positions is published (with their spread as the covariance) and marked green. If no viewpoint does, the candidate is rejected and marked red. With `use_depth: true`, one more frame with depth measures the feet directly, with no flat-ground assumption; it's off by default because AirSim's Python client decodes float images very slowly (depth images are 320×180 in `settings.json` to limit that).
6. **Resume:** it flies back along the arc to the approach point, zooms back out, flies back up the line of sight to where it stopped, restores the camera tilt and heading, and continues the lane.

## The UGV

`ugv.py` drives the Husky (`Car1`) to every victim the drone confirms and leaves a medical kit beside them.

**Setup, once:**
1. Copy the repo's `settings.json` to `<Documents>\AirSim` again: it gives `Car1` a 16-channel lidar (`Lidar1`, 30 m range, 1 m above the car), and restart Play.
2. AirSim 1.x can't spawn objects, so the kits are placed in the level in advance and moved when needed. Put a few kit actors (any mesh) somewhere out of sight, e.g. under the map. Set each one's **Mobility** to **Movable**, and tag them `MedKit_1`, `MedKit_2`, ... (Details → Actor → Tags), as listed in `ugv.medkits` in `config.yaml`.
3. With Play running, and some open ground in front of the Husky:
   ```
   python ugv.py --check
   ```
   It reads one lidar sweep, finds the ground in it, and prints the `ugv.base_height_m` to set. If that's wrong, the ground is mapped as obstacles, in rings around the UGV, and it finds itself boxed in. Then it drives for 3 s with the steering at +0.5 and says what `ugv.steering_sign` should be.

**Run:** start the UGV first (ZeroMQ doesn't keep messages for late subscribers), then the drone:
```
python ugv.py          # terminal 1: waits at home
python mission.py      # terminal 2
```
To test the UGV alone, without the drone: `python ugv.py --victim Victim_01` drives to that tagged person (at their true position), drops a kit, comes home and exits. Repeat `--victim` for several.
What it does:
1. **Queue:** each confirmed victim is queued. Ids already seen, and anyone within 3 m of a known victim, are ignored. At home it waits; when a victim arrives it drives to the nearest queued one.
2. **Mapping, with no prior map:** every lidar sweep updates a 2D occupancy grid (`map_resolution_m` cells, log-odds). A point between `obstacle_min_height_m` and `obstacle_max_height_m` above the ground **around it** (the lowest ground seen within 0.5 m, from the height map) marks its cell occupied, and the cells its ray passed through get freer. Measuring from the local ground, not from under the UGV, keeps sloping ground ground on a hilly map. Lower points are ground; higher ones (branches, eaves) pass over the Husky. Edges and ditches show up as steps in the height map (below). The UGV also stays within the search area plus `area_margin_m`. **Terrain:** every surface point also goes into a height map (the lowest height seen in each cell). Where the ground rises more than `max_step_m` within 0.5 m (a kerb, a ramp's edge) or more than `max_slope_deg` over 1 m (a ramp, a bank), the higher cell is too steep and counts as an obstacle; gentle slopes stay drivable. The UGV's position comes from AirSim, so this is mapping with a known pose: the map is built only from what the lidar has seen.
3. **Planning:** A* on the grid, with obstacles grown by `robot_radius_m + safety_margin_m`. Unknown cells count as free, so it heads straight for the goal and goes around what the lidar finds. It replans every `replan_s`, and immediately if something appears across its path. The path is drawn blue in the scene.
4. **Driving:** pure pursuit. It steers at the point `lookahead_m` along the path, at `max_speed`, and slows to `slow_speed` in sharp turns and near the goal. If it pushes but doesn't move for 3 s, it backs up and replans.
5. **Kit:** it stops `stop_distance_m` from the victim and moves the next kit to the ground `kit_offset_m` from them, on its side.
6. **Next:** it drives to the nearest queued victim. With none left it heads home, and a victim published on the way, or while it waits at home, sends it there. A victim it can't reach (no path for `no_path_timeout_s`, or far too long) is skipped and reported.

## Messages

`mission.py` publishes on `publish.endpoint` (default `tcp://*:5556`), topic `/rescue/victims`. Each message is JSON shaped like a ROS `PoseWithCovarianceStamped`:

```json
{"header": {"stamp": 1700000000.12, "frame_id": "airsim_world_ned"},
 "id": 1, "position": {"x": 12.3, "y": -4.5, "z": 0.2},
 "covariance": [0.09, 0, 0, 0, 0.04, 0, 0, 0, 0.01],
 "confidence": 0.71, "hits": 6, "frames": 8}
```

## Testing the detector

With Play running:

```
python detect_once.py                      # one frame from where the drone is now
python detect_once.py --target Victim_01   # fly above that actor first
python detect_once.py --target Victim_01 --tilt 45   # look at it at 45 deg, from back and facing it
python detect_once.py --image some.png     # an image file, no AirSim needed
```

It prints each detection's confidence and size in pixels and saves the image with boxes in `output/`. The image is split into `tiles_x` × `tiles_y` overlapping tiles (`detector` in `config.yaml`), each resized to the model's 640×640 input, so small people keep their pixels. The overlap must be bigger than the largest person in pixels; partial boxes from a neighbouring tile are merged into the full one.

To check the ONNX pipeline against Ultralytics itself on the same image (run it in the environment with `ultralytics` installed, e.g. the one used for `export_onnx.py`):

```
python compare_detectors.py --image output/<name>_raw.png --pt models/person_yolo11s.pt
```

## Tests

```
python -m pytest tests
```
