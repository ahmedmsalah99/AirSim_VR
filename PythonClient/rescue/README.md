# Rescue mission

The drone searches a rectangle in a lawnmower pattern with its camera tilted forward. When it sees a person, it approaches along its line of sight to confirm them, publishes their position over ZeroMQ, and carries on with the search. The UGV comes next.

| File | What it does |
|---|---|
| `config.yaml` | Search area, drone limits, camera, detector, confirmation and publishing settings |
| `mission.py` | The mission: search, confirm, publish, resume, return home, report |
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
pip install msgpack-rpc-python numpy pyyaml onnxruntime opencv-python pyzmq
```

Copy the repo's `settings.json` to `<Documents>\AirSim` (see the top-level README). Its `CameraDefaults` give every camera, including the drone's `bottom_center`, 1280×720 images with a 60° field of view, which the plan assumes. Don't configure cameras under `Vehicles` → `Cameras`: in this AirSim version that spawns a new camera actor and crashes on Play. The settings also spawn the drone 4 m east of the Husky so it doesn't land on it.

Put the converted model at `models/person_yolo11s.onnx` (see `export_onnx.py`). Tag the people placed in the level `Victim_01`, `Victim_02`, … (Details → Actor → Tags) and list the tags under `ground_truth` in `config.yaml`, so the mission can report its position error.

## Use

1. Set the search rectangle in `config.yaml`. Positions are metres from the PlayerStart: `x = (Unreal X - PlayerStart X) / 100`, and the same for `y`. Read the Unreal X/Y off actors in the editor. Also set `max_obstacle_height` to the tallest building or tree inside the rectangle.
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
   ```
   The area (orange) and the lanes (green) are drawn in the scene; confirmed victims are marked green and rejected candidates red, a few metres above the spot. Candidates aren't marked before they're decided: the camera sees debug drawings, and a marker on a person would hide them. Ctrl+C (or an error) ends the mission early: the drone flies home at its current altitude and lands; a second Ctrl+C lands it where it is. Before taking off it measures the ground below with the depth camera (`ground is … m below the drone`); every height, every person's ground position and the landing are relative to it. During the lanes it logs every 5 s how many people it detected and how many it dropped and why. At the end it lands (checking that it's really down) and prints each victim's position and its error against the tagged people. It saves a summary (`output/mission_<time>.json`) and, for every candidate, the search frame that triggered it and each confirm frame with the detections drawn (`output/mission_<time>/candidate<N>_*.jpg`; `confirm.save_images`).
   The window at the bottom of the viewport shows the drone's camera; press `1` in the viewport to hide or show it.

## How the plan is computed

The camera looks forward along the lane, `camera.tilt_deg` below the horizon (90 = straight down). On flat ground it sees a trapezoid. A person covers `person_size × f × sin(a) / h` detector pixels, where `a` is the ray's angle below the horizon, `h` the altitude and `f` the focal length in detector pixels (each image tile is resized to the 640 px input, so `f` shrinks with it). So people are only detectable out to the slant range where they shrink to `min_person_px`.

1. **Detection band:** from the bottom edge of the image (`near`) to wherever comes first, the top edge of the image or that range (`far`). `near` and `far` are distances ahead of the drone.
2. **Swath:** the ground width covered at the far edge, `2 × depth × tan(hfov / 2)`. Every point within half a swath of the lane crosses the band.
3. **Lane spacing** = swath × (1 − `side_overlap`), spread evenly so the outer lanes just reach the edges. Lanes run along whichever axis gives the shorter mission and alternate direction; the drone faces its direction of travel. Each lane is shifted back by `near`, so the band starts on the area's edge and ends just past the far edge.
4. **Speed** is the slowest of:
   - `max_speed` of the drone,
   - motion blur: image motion is fastest at the near edge, `v·sin²(a)/h` rad/s, so `v ≤ max_blur_px × h / (f × sin²(a_near) × exposure_s)`,
   - frames per person: a ground point crosses the band in `(far − near) / v` seconds and must be seen in `min_frames_per_point` frames at the detector's frame rate (`fps_per_tile / tiles`).
5. **Altitude:** every altitude from the lowest safe one (`max_obstacle_height + safety_margin`, at least `min_altitude`) up to `max_altitude` is tried in 0.5 m steps, and the fastest mission wins (the lowest altitude among near-ties). If no altitude keeps people detectable, the plan is rejected with suggestions.

## How a person is confirmed

1. **Candidate:** during a lane, each detection's feet pixel (bottom centre of the box) is projected onto the flat ground through the camera pose, which gives a position. A detection more than `merge_radius` from every known one, inside the area, is a new candidate.
2. **Approach:** the drone stops, turns to face the candidate, tilts the camera to the line-of-sight angle, and flies down **along the line of sight** towards the distance where a person is `confirm.min_person_px` tall (`distance = person_size × f / min_person_px`). It never goes below `confirm.min_altitude` (12 m): there's no obstacle map, and trees and roofs stand around people, so going lower than the tallest of them risks collisions.
3. **Zoom:** whatever the approach couldn't do, the camera's zoom does: the field of view narrows until a person at that range is `min_person_px` tall, down to `confirm.min_fov_deg`. The intrinsics used to locate detections follow the zoom.
4. **Check:** it hovers until it has stopped moving, then runs the detector on `frames` frames, `frame_interval_s` apart. A frame is a hit if a person is detected within `match_radius` of the candidate (feet projected onto the flat ground).
5. **Decision:** with at least `required` hits, the person is confirmed: the median of the hits' positions is published (with their spread as the covariance) and marked green. Otherwise the candidate is rejected and marked red. With `use_depth: true`, one more frame with depth measures the feet directly, with no flat-ground assumption; it's off by default because AirSim's Python client decodes float images very slowly (depth images are 320×180 in `settings.json` to limit that).
6. **Resume:** it zooms back out, flies back up the same line to where it stopped, restores the camera tilt and heading, and continues the lane.

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
