# WashyBoy

A petri-dish robot that grabs and cleans dishes, built on an SO-101 arm with a camera. This README covers the SO-101 interface. For a full record of how everything was set up, see [docs/SETUP_LOG.md](docs/SETUP_LOG.md).

A thin Python layer on top of **Hugging Face LeRobot**, the official SO-101 software. LeRobot does the motor driving, calibration and camera capture. This layer adds easy commands for picking up dishes.

| Piece | Where it comes from |
|---|---|
| Motor control, calibration, cameras | [LeRobot](https://huggingface.co/docs/lerobot/so101) |
| x/y/z motion (inverse kinematics) | LeRobot `RobotKinematics` (placo) + [official SO-101 URDF](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) |
| Dish detection | OpenCV `HoughCircles` |
| Plate dimensions | your `Stunning Amur.stl`: 70 × 70 mm, 12.4 mm tall |

## 1. Install (once)

LeRobot recommends conda and Python 3.12 ([install guide](https://huggingface.co/docs/lerobot/installation)):

```bash
conda create -y -n lerobot python=3.12 ffmpeg -c conda-forge
conda activate lerobot
pip install -r requirements.txt
conda install -y -c conda-forge placo   # pip can't build placo on Windows
python download_assets.py               # SO-101 URDF + example training data
```

The arm's calibration is in `calibration/petri_follower.json`. To use this arm on another PC without recalibrating, copy that file to `%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\` (Windows) or `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` (Mac/Linux).

The plate model is in `hardware/petri_plate_70x70x12.4mm.stl`.

## Training data

`download_assets.py` fetches [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace), Hugging Face's official SO-101 dataset: 50 pick-and-place episodes with 2 cameras at 30 fps. Its joint names match `petri_arm.py` exactly. Use it to test the training pipeline. A policy that actually handles petri dishes needs your own recordings (see "Next steps").

## Camera-driven mode (recommended)

The robot finds plates and stations with the camera; you teach only one pose. See **[docs/TRAINING.md](docs/TRAINING.md)**:
1. `python make_markers.py`, then print `markers_A4.pdf` at 100%.
2. `setup_wizard.bat`
3. `autonomous.bat --dry-run`, then `autonomous.bat`
4. `autonomous.bat --record` to collect training data, then `lerobot-train`.

**Windows NumPy note:** if NumPy crashes with `0xc06d007f` after installing placo, run `conda install -n lerobot -c conda-forge "libopenblas=*=pthreads*" "libblas=*=*openblas"`.

## Quick launchers (Windows, already set up on this PC)

- `check_arm.bat` runs a read-only test of all 6 motors and tells you which cable to check.
- `recalibrate.bat` runs a full recalibration. Add `--setup-motors` to also re-assign motor IDs.

## 2. Set up the arm (once, using LeRobot's own tools)

```bash
lerobot-find-port
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=petri_follower
lerobot-find-cameras opencv
```

Replace `COM5` with the port that `lerobot-find-port` printed. Only run `setup-motors` on new motors. `--robot.id` must match `arm_id` in `PetriArm` (default `petri_follower`).

## 3. Use it

```python
from petri_arm import PetriArm

with PetriArm(port="COM5", camera_index=0) as arm:
    arm.home()
    print(arm.get_joints())          # degrees; gripper 0 = closed, 100 = open
    print(arm.get_xyz())             # metres, arm base frame

    arm.move_joints(shoulder_pan=30, duration=1.5)
    arm.open_gripper()
    arm.move_to_xyz(0.20, 0.00, 0.06)
    arm.move_relative(dz=-0.03)
    arm.close_gripper(20)

    arm.go_to("above_stack")         # a pose you taught
    arm.save_frame("look.png")
    print(arm.find_dishes(show=True))  # [(x_px, y_px, radius_px), ...]
```

### Commands

| Command | What it does |
|---|---|
| `get_joints()` / `move_joints(**j, duration=)` | Read or smoothly move joints (degrees) |
| `open_gripper()` / `close_gripper(pct)` / `set_gripper(pct)` | Open, close or set the gripper |
| `teach()` | Turns torque off so you can move the arm by hand, then saves named poses to `poses.json` |
| `go_to(name)` / `home()` | Replay a taught pose |
| `get_xyz()` / `move_to_xyz(x,y,z)` / `move_relative(dx,dy,dz)` | Move in metres |
| `stack_height(level, table_z)` | Z of the top of plate number `level` (12.4 mm per plate) |
| `get_frame()` / `save_frame()` / `find_dishes()` | Camera image and circle detection |
| `pixel_to_table(u, v, calib)` | Convert a camera pixel to table x/y (after `calibrate_camera.py`) |

### Suggested workflow

1. **Teach poses** such as `home`, `above_stack` and `clean_station`. This is the most reliable method and needs no maths:
   ```bash
   python -c "from petri_arm import PetriArm; PetriArm('COM5').connect().teach()"
   ```
2. **Calibrate the camera** so a dish seen in the image becomes an arm x/y: `python calibrate_camera.py COM5 0`
3. **Run the example routine**: `python example_pick_and_clean.py`

## Safety notes

- `max_step_deg=10` caps each command, using LeRobot's `max_relative_target`. Raise it only once your moves are tested.
- `disconnect()` turns torque off and the arm drops. Hold it, or end in a resting pose.
- The SO-101 has only 5 joints plus the gripper, so it cannot reach every orientation. `move_to_xyz` keeps the current orientation as closely as it can. Start from a taught "gripper pointing down" pose.
- Z = 0 is the arm's base frame, not the table. To measure the table height, touch the table with the tip and call `get_xyz()`. Pass that value as `table_z`.
- If `move_to_xyz` lands off-target, the URDF frame may not match your gripper. The SO-ARM100 repo has [an open issue about this](https://github.com/TheRobotStudio/SO-ARM100/issues/146). Teach poses instead.
