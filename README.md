# AirSim_VR: drone + UGV rescue sim

This fork of AirSim adds a `Both` sim mode that runs a drone and a ground vehicle (the Husky UGV) in the same Unreal level, each with its own API port:

| Vehicle | Name in `settings.json` | Python client | Port |
|---|---|---|---|
| Drone | `Drone1` | `airsim.MultirotorClient(port=41451)` | 41451 |
| UGV (Husky body on the AirSim car) | `Car1` | `airsim.CarClient(port=41452)` | 41452 |

The Unreal project is `Unreal/Environments/Blocks` (UE 4.24). It opens on the `Rural_Cabins` map, and the Husky assets are in `Content/Husky`.

The sections below are for Windows. The original AirSim README follows them.

## Requirements

- Unreal Engine **4.24**, installed from the Epic launcher to `C:\Program Files\Epic Games\UE_4.24`.
- Visual Studio **2019** with the "Desktop development with C++" workload. The default VS 2019 toolset is enough, VS 2017 (v141) is not needed.
- Python 3 with `pip install msgpack-rpc-python numpy`. The scripts in `PythonClient/both` use the `airsim` package from this repo, so there's no need to `pip install airsim`.

## First-time setup

1. Copy `settings.json` from the repo root to `<Documents>\AirSim\settings.json`. AirSim never reads the copy in the repo. `<Documents>` is your Windows Documents folder, wherever Windows keeps it. With OneDrive that's `%USERPROFILE%\OneDrive\Documents`, not `%USERPROFILE%\Documents`. From the repo root, in `cmd`:
   ```
   for /f "delims=" %d in ('powershell -c "[Environment]::GetFolderPath('MyDocuments')"') do (mkdir "%d\AirSim" 2>nul & copy /Y settings.json "%d\AirSim\settings.json")
   ```
   To see which folder that is, run `powershell -c "[Environment]::GetFolderPath('MyDocuments')"`. A `settings.json` next to `UE4Editor.exe` (`UE_4.24\Engine\Binaries\Win64`) takes priority over this one, so make sure there isn't one there.
2. Do the full build below.

## What to rebuild after a change

After a `git pull`, look at which files changed (`git diff --stat HEAD@{1}`) and do the matching row. When several rows apply, do the one highest in the table.

| What changed | What to do |
|---|---|
| `AirLib/`, `Unreal/Plugins/AirSim/`, `MavLinkCom/` or `build.cmd` (C++) | **Full build** (all 3 steps below) |
| `Unreal/Environments/Blocks/Source/` (C++ of the project) | Steps 2 and 3 |
| Maps, Blueprints, meshes (`.umap`, `.uasset` in `Content/`) | Nothing to build. Open the editor (step 3). Close it first if it's open |
| `settings.json` | Copy it to `<Documents>\AirSim` again (first-time setup, step 1), then restart Play |
| Python scripts | Nothing. Run them |

### Full build

Close the Unreal editor first.

1. **Build AirLib and the plugin.** Use the **"x64 Native Tools Command Prompt for VS 2019"** from the Start menu; `build.cmd` refuses to run in a normal terminal.
   ```
   cd /d %USERPROFILE%\Projects\AirSim_VR
   build.cmd
   ```
2. **Copy the plugin into Blocks and regenerate the project files.**
   ```
   cd Unreal\Environments\Blocks
   update_from_git.bat
   ```
   This mirrors `Unreal/Plugins/AirSim` and `AirLib` into `Blocks/Plugins/AirSim`. It then deletes `Binaries`, `Intermediate`, `Saved` and `Blocks.sln` and regenerates the project files. Assets in `Blocks/Content` are not touched. It should print `Using ...\UE_4.24\Engine\Binaries\DotNET\UnrealBuildTool.exe`.
3. **Compile and open the editor.** Double-click `Blocks.uproject` and answer **Yes** when it asks to rebuild the missing modules. If that rebuild fails, open `Blocks.sln` in Visual Studio and build **Development Editor / Win64** to see the actual compile error.

## How to run

1. Make sure `<Documents>\AirSim\settings.json` matches the repo's `settings.json`.
2. Open `Blocks.uproject` and press **Play**. The API ports only open while Play is running.
3. In another terminal, with your Python environment active:
   ```
   python %USERPROFILE%\Projects\AirSim_VR\PythonClient\both\multi_agent_drone_car.py
   ```

## Drone rescue mission

`PythonClient/rescue` is the drone's search-and-rescue mission, in Python only (no PX4 or ROS):

1. The drone flies a lawnmower search over a rectangle with its camera tilted 45° forward.
2. A YOLO11 person detector runs on the GPU.
3. When someone is seen in 3 frames, the drone flies down the line of sight, zooms in and checks them from up to 3 viewpoints.
4. Confirmed victims are published over ZeroMQ.
5. The drone goes back to where it stopped and carries on, then flies home and lands.

Quick start, with Play running:
```
cd %USERPROFILE%\Projects\AirSim_VR\PythonClient\rescue
pip install msgpack-rpc-python numpy pyyaml onnxruntime-directml opencv-python pyzmq
python survey_plan.py                  # print the search plan without flying
python mission.py                      # the full mission
python mission.py --victim Victim_01   # test one tagged person, no lanes
python listen_victims.py               # (another terminal) print the published victims
```
The detector model goes in `PythonClient/rescue/models/person_yolo11s.onnx`. The search area, speeds and thresholds are in `PythonClient/rescue/config.yaml`. To fit the area to a new map, use `set_area.py`. **[`PythonClient/rescue/README.md`](PythonClient/rescue/README.md)** has the full setup and explains how the plan, detection and confirmation work.

## Troubleshooting

- **`WSAECONNREFUSED` from Python:** nothing is listening on the ports. Check that Play is running, then run `netstat -ano | findstr 4145`; you should see both 41451 and 41452. If they're missing, AirSim failed to start. Look for a popup in Unreal or red lines in **Window → Developer Tools → Output Log**. Common causes:
  - `settings.json` is missing from `<Documents>\AirSim`, or doesn't have `"SimMode": "Both"`.
  - The plugin wasn't rebuilt after a C++ change.
- **The car is the SUV instead of the Husky, or settings changes have no effect:** AirSim is reading a different `settings.json`. Usually it's in the OneDrive Documents folder while you edited `%USERPROFILE%\Documents`. Copy it again with the command from first-time setup, step 1. AirSim's own messages (such as "Loaded settings from …") only show briefly on screen after Play; they're not in the Output Log.
- **The mission flies very slowly (`search frames: 0.2 per second`):** the line above it says which step is slow. If **capture** is slow (several seconds), the plugin is older than the image-capture fix on `rescue-dev`, so do a full build. If **detection** is slow, YOLO is on the CPU. Install `onnxruntime-directml` instead of `onnxruntime`. On a laptop, check that the log says `detector on DirectML GPU 1`. GPU 0 is the integrated GPU that drives the screen, and using it froze the machine.
- **The Unreal editor crashes or freezes with no error during a mission:** older builds of the `Both` mode updated the car from two threads at once. That's fixed on `rescue-dev`, so do a full build. To catch the next crash, run these once in an **administrator** cmd. Windows then saves a dump to `C:\CrashDumps`, which you can open in Visual Studio with **Debug with Native Only** to see the call stack:
  ```
  reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\UE4Editor.exe" /v DumpFolder /t REG_EXPAND_SZ /d "C:\CrashDumps" /f
  reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\UE4Editor.exe" /v DumpType /t REG_DWORD /d 1 /f
  ```
- **`MSB8020 ... v141 cannot be found`:** `build.cmd` was changed to force the VS 2017 toolset. Keep it on the default VS 2019 toolset.
- **"Failed to generate project files ... UnrealBuildTool.exe":** `Unreal/Environments/Blocks/GenerateProjectFiles.bat` must be the version that calls the engine's UnrealBuildTool directly, as on `rescue-dev`, not the one that goes through `UnrealVersionSelector`.

---

# Welcome to AirSim

AirSim is a simulator for drones, cars and more, built on [Unreal Engine](https://www.unrealengine.com/) (we now also have an experimental [Unity](https://unity3d.com/) release). It is open-source, cross platform, and supports hardware-in-loop with popular flight controllers such as PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped into any Unreal environment. Similarly, we have an experimental release for a Unity plugin.

Our goal is to develop AirSim as a platform for AI research to experiment with deep learning, computer vision and reinforcement learning algorithms for autonomous vehicles. For this purpose, AirSim also exposes APIs to retrieve data and control vehicles in a platform independent way.

**Check out the quick 1.5 minute demo**

Drones in AirSim

[![AirSim Drone Demo Video](docs/images/demo_video.png)](https://youtu.be/-WfTr1-OBGQ)

Cars in AirSim

[![AirSim Car Demo Video](docs/images/car_demo_video.png)](https://youtu.be/gnz1X3UNM5Y)

## What's New
* A ROS wrapper for multirotors is available. See [airsim_ros_pkgs](https://github.com/microsoft/AirSim/blob/master/ros/src/airsim_ros_pkgs) for the ROS API, and [airsim_tutorial_pkgs](https://github.com/microsoft/AirSim/blob/master/ros/src/airsim_tutorial_pkgs) for tutorials.
* [Added sensor APIs for Barometer, IMU, GPS, Magnetometer, Distance Sensor](https://microsoft.github.io/AirSim/sensors)
* Added support for [docker in ubuntu](https://github.com/microsoft/AirSim/blob/master/docs/docker_ubuntu.md)
* Added Weather Effects and [APIs](https://microsoft.github.io/AirSim/apis#weather-apis)
* Added [Time of Day API](https://microsoft.github.io/AirSim/apis#time-of-day-api)
* An experimental integration of [AirSim on Unity](https://github.com/Microsoft/AirSim/tree/master/Unity) is now available. Learn more in [Unity blog post](https://blogs.unity3d.com/2018/11/14/airsim-on-unity-experiment-with-autonomous-vehicle-simulation).
* [New environments](https://github.com/Microsoft/AirSim/releases/tag/v1.2.1): Forest, Plains (windmill farm), TalkingHeads (human head simulation), TrapCam (animal detection via camera)
* Highly efficient [NoDisplay view mode](https://microsoft.github.io/AirSim/settings#viewmode) to turn off main screen rendering so you can capture images at high rate
* [Lidar Sensor](https://microsoft.github.io/AirSim/lidar)
* Case Study: [Formula Student Technion Driverless](https://github.com/Microsoft/AirSim/wiki/technion)
* [Multi-Vehicle Capability](https://microsoft.github.io/AirSim/multi_vehicle)
* [ROS publisher](https://github.com/Microsoft/AirSim/pull/1135)

For complete list of changes, view our [Changelog](CHANGELOG.md)

## How to Get It

### Windows
* [Download binaries](https://microsoft.github.io/AirSim/use_precompiled)
* [Build it](https://microsoft.github.io/AirSim/build_windows)

### Linux
* [Download binaries](https://microsoft.github.io/AirSim/use_precompiled)
* [Build it](https://microsoft.github.io/AirSim/build_linux)

### macOS
* [Build it](https://microsoft.github.io/AirSim/build_linux)

[![Build Status](https://travis-ci.org/Microsoft/AirSim.svg?branch=master)](https://travis-ci.org/Microsoft/AirSim)

## How to Use It

### Documentation

View our [detailed documentation](https://microsoft.github.io/AirSim/) on all aspects of AirSim.

### Manual drive

If you have remote control (RC) as shown below, you can manually control the drone in the simulator. For cars, you can use arrow keys to drive manually.

[More details](https://microsoft.github.io/AirSim/remote_control/)

![record screenshot](docs/images/AirSimDroneManual.gif)

![record screenshot](docs/images/AirSimCarManual.gif)


### Programmatic control

AirSim exposes APIs so you can interact with the vehicle in the simulation programmatically. You can use these APIs to retrieve images, get state, control the vehicle and so on. The APIs are exposed through the RPC, and are accessible via a variety of languages, including C++, Python, C# and Java.

These APIs are also available as part of a separate, independent cross-platform library, so you can deploy them on a companion computer on your vehicle. This way you can write and test your code in the simulator, and later execute it on the real vehicles. Transfer learning and related research is one of our focus areas.

Note that you can use [SimMode setting](https://microsoft.github.io/AirSim/settings#simmode) to specify the default vehicle or the new [ComputerVision mode](https://microsoft.github.io/AirSim/image_apis#computer-vision-mode-1) so you don't get prompted each time you start AirSim.

[More details](https://microsoft.github.io/AirSim/apis/)

### Gathering training data

There are two ways you can generate training data from AirSim for deep learning. The easiest way is to simply press the record button in the lower right corner. This will start writing pose and images for each frame. The data logging code is pretty simple and you can modify it to your heart's content.

![record screenshot](docs/images/record_data.png)

A better way to generate training data exactly the way you want is by accessing the APIs. This allows you to be in full control of how, what, where and when you want to log data.

### Computer Vision mode

Yet another way to use AirSim is the so-called "Computer Vision" mode. In this mode, you don't have vehicles or physics. You can use the keyboard to move around the scene, or use APIs to position available cameras in any arbitrary pose, and collect images such as depth, disparity, surface normals or object segmentation.

[More details](https://microsoft.github.io/AirSim/image_apis/)

### Weather Effects

Press F10 to see various options available for weather effects. You can also control the weather using [APIs](https://microsoft.github.io/AirSim/apis#weather-apis). Press F1 to see other options available.

![record screenshot](docs/images/weather_menu.png)

## Tutorials

- [Video - Setting up AirSim with Pixhawk Tutorial](https://youtu.be/1oY8Qu5maQQ) by Chris Lovett
- [Video - Using AirSim with Pixhawk Tutorial](https://youtu.be/HNWdYrtw3f0) by Chris Lovett
- [Video - Using off-the-self environments with AirSim](https://www.youtube.com/watch?v=y09VbdQWvQY) by Jim Piavis
- [Reinforcement Learning with AirSim](https://microsoft.github.io/AirSim/docs/reinforcement_learning) by Ashish Kapoor
- [The Autonomous Driving Cookbook](https://aka.ms/AutonomousDrivingCookbook) by Microsoft Deep Learning and Robotics Garage Chapter
- [Using TensorFlow for simple collision avoidance](https://github.com/simondlevy/AirSimTensorFlow) by Simon Levy and WLU team

## Participate

### Paper

More technical details are available in [AirSim paper (FSR 2017 Conference)](https://arxiv.org/abs/1705.05065). Please cite this as:
```
@inproceedings{airsim2017fsr,
  author = {Shital Shah and Debadeepta Dey and Chris Lovett and Ashish Kapoor},
  title = {AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles},
  year = {2017},
  booktitle = {Field and Service Robotics},
  eprint = {arXiv:1705.05065},
  url = {https://arxiv.org/abs/1705.05065}
}
```

### Contribute

Please take a look at [open issues](https://github.com/microsoft/airsim/issues) if you are looking for areas to contribute to.

* [More on AirSim design](https://microsoft.github.io/AirSim/design)
* [More on code structure](https://microsoft.github.io/AirSim/code_structure)
* [Contribution Guidelines](CONTRIBUTING.md)
* [Trello Board](https://trello.com/b/1t2qCeaA/wishlist-by-community-for-community)

### Who is Using AirSim?

We are maintaining a [list](https://microsoft.github.io/AirSim/who_is_using) of a few projects, people and groups that we are aware of. If you would like to be featured in this list please [make a request here](https://github.com/microsoft/airsim/issues).

## Contact

Join the AirSim group on [Facebook](https://www.facebook.com/groups/1225832467530667/) to stay up to date or ask any questions.

## FAQ

If you run into problems, check the [FAQ](https://microsoft.github.io/AirSim/faq) and feel free to post issues in the  [AirSim](https://github.com/Microsoft/AirSim/issues) repository.

## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
