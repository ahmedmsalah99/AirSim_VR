# Rescue mission

The drone searches a rectangle in a lawnmower pattern. The UGV and the confirm/publish steps come next.

| File | What it does |
|---|---|
| `config.yaml` | Search area, drone limits, camera and detector parameters |
| `survey_plan.py` | Computes altitude, lane spacing, speed and waypoints, and explains each choice |
| `fly_survey.py` | Flies the plan in AirSim and draws it in the scene |
| `view_camera.py` | Live view of the drone's downward camera in its own window |
| `export_onnx.py` | Converts the YOLO11 `.pt` detector to ONNX |

## Setup

```
pip install msgpack-rpc-python numpy pyyaml opencv-python
```

Copy the repo's `settings.json` to `<Documents>\AirSim` (see the top-level README). Its `CameraDefaults` give every camera, including the drone's `bottom_center`, 1280×720 images with a 60° field of view, which the plan assumes. Don't configure cameras under `Vehicles` → `Cameras`: in this AirSim version that spawns a new camera actor and crashes on Play. The settings also spawn the drone 4 m east of the Husky so it doesn't land on it.

## Use

1. Set the search rectangle in `config.yaml`. Positions are metres from the PlayerStart: `x = (Unreal X - PlayerStart X) / 100`, and the same for `y`. Read the Unreal X/Y off actors in the editor. Also set `max_obstacle_height` to the tallest building or tree inside the rectangle.
2. Check the plan without flying:
   ```
   python survey_plan.py
   ```
3. Press Play in Unreal, then fly it:
   ```
   python fly_survey.py
   ```
   The area (orange) and the lanes (green) are drawn in the scene. Ctrl+C stops the drone in a hover.
4. To watch the downward camera, run this in a second terminal (q or Esc closes it):
   ```
   python view_camera.py
   ```
   Don't use AirSim's built-in camera subwindows (`SubWindows` in `settings.json`): in Both mode they stayed black and froze Unreal.

## How the plan is computed

1. **Lowest safe altitude** = `max_obstacle_height + safety_margin` (at least `min_altitude`).
2. **Highest altitude where people stay detectable.** The detector needs a person to cover `min_person_px` pixels in its 640 px input. Each image tile is resized to 640, so the effective image width is `width_px × min(1, 640 / tile size)`. The camera sees `2·h·tan(FOV/2)` of ground across, which gives `h_max = person_size × effective_width / (min_person_px × 2·tan(FOV/2))`. If `h_max` is below the safe altitude, the plan is rejected with suggestions. Splitting the image into more tiles raises `h_max`, at the cost of detector time per frame.
3. **Altitude**: the drone flies at `h_max` (capped at `max_altitude`). Higher means wider lanes and fewer of them.
4. **Lane spacing** = ground width × (1 − `side_overlap`), spread evenly so the outer lanes' footprints just reach the edges. Lanes run along whichever axis gives the shorter mission, counting `turn_time_s` per turn.
5. **Speed** is the slowest of:
   - `max_speed` of the drone,
   - motion blur: `max_blur_px × ground per pixel / exposure_s`,
   - frames per person: every ground point stays in view for `footprint along / v` seconds and must be seen in `min_frames_per_point` frames at the detector's frame rate (`fps_per_tile / tiles`).

## Tests

```
python -m pytest tests
```
