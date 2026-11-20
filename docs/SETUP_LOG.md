# WashyBoy setup log

A record of every step taken to set up the SO-101 petri-dish robot: what was done, why, and what came out of it. Newest entries go at the bottom.

**Date:** 2026-11-20 · **PC:** Windows 11

---

## 1. Plate model inspected
- **File:** `Stunning Amur.stl`, saved in this repo as `hardware/petri_plate_70x70x12.4mm.stl`.
- **Method:** read the binary STL header and computed the bounding box of all triangles.
- **Result:** 1,442 triangles, **70 × 70 mm footprint, 12.4 mm tall**.
- **Used for:** `PLATE_SIZE_M` and `PLATE_HEIGHT_M` in `petri_arm.py`, and stack heights (12.4 mm per plate).

## 2. Research: existing software for the SO-101
The rule was to use existing online resources instead of writing drivers from scratch.

| Need | Resource used |
|---|---|
| Motor control, calibration, cameras | Hugging Face **LeRobot**: https://huggingface.co/docs/lerobot/so101 |
| Camera API | https://huggingface.co/docs/lerobot/cameras |
| x/y/z motion (inverse kinematics) | LeRobot `RobotKinematics` (uses the **placo** library) |
| Arm geometry (URDF + meshes) | https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101 |
| Dish detection | OpenCV `HoughCircles` |

**Key API facts confirmed from the LeRobot source:**
- Import path: `from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig`.
- Joint keys: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`.
- Units: joints in degrees (`use_degrees=True`); gripper from 0 (closed) to 100 (open).
- Safety: `max_relative_target` caps how far each command can move a joint.

## 3. Code written
| File | Purpose |
|---|---|
| `petri_arm.py` | `PetriArm` class: joints, gripper, teach and replay, x/y/z moves, stack heights, camera, dish finder |
| `calibrate_camera.py` | Maps camera pixels to table x/y using 4 markers and a homography |
| `example_pick_and_clean.py` | Example routine: find dish → pick → scrub → stack |
| `recalibrate.py` | Runs LeRobot's calibration; auto-detects the USB port; optional `--setup-motors` |
| `check_arm.py` | Read-only health check of all 6 motors; says which cable is at fault |
| `download_assets.py` | Re-downloads the URDF and the example dataset, which are not stored in the repo |
| `check_arm.bat`, `recalibrate.bat` | One-click launchers that need no conda activation |
| `requirements.txt` | pip dependencies |

## 4. Software installed on this PC
At the start, Python was not installed; only the Microsoft Store shortcut existed.

1. **Miniforge3** (conda), installed for the current user only via `winget install CondaForge.Miniforge3 --scope user`.
2. **Conda environment `lerobot`** created with `conda create -n lerobot python=3.12 ffmpeg -c conda-forge`.
3. **LeRobot and OpenCV:** `pip install "lerobot[feetech,dataset]" opencv-python numpy`. This installed LeRobot 0.7.0, OpenCV 5.1.0 and PyTorch 2.15 (CPU only).
4. **placo:** `pip install placo` **failed**, because it needs a C++ compiler on Windows. **Fix:** `conda install -c conda-forge placo` (version 0.10.0, prebuilt).
5. **git 2.56.1 and the GitHub CLI 2.105.0**, installed with `conda install -c conda-forge git gh`. git was not on the PC.

**Python path:** `%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe`

## 5. SO-101 URDF downloaded
- git wasn't available at this point, so only `Simulation/SO101` (41 files) was fetched through the GitHub API into `SO-ARM100/Simulation/SO101/`.
- **Kinematics test:** forward kinematics with all joints at 0° puts the gripper at x=0.391 m, y=0, z=0.227 m.
- **Problem:** LeRobot's IK with default settings ended **12 mm off** target, and up to 162 mm off on harder targets.
- **Cause:** the SO-101 has 5 joints, so it cannot always hold the gripper orientation and the position at the same time.
- **Fix:** added `PetriArm.solve_ik()`, which iterates in two passes. Pass 1 tries to keep the orientation (weight 0.001). If that leaves more than 2 mm of error, pass 2 solves for position only.
- **Result (30 random targets, simulation):** median error **0.4 mm**, worst 1.4 mm; gripper tilt changed by 7° on median.

## 6. First arm: hardware fault
- **Port:** COM4 (USB adapter chip: CH343, `1a86:55d3`).
- **Read-only motor scan** (no motion):
  - Motors 1 and 2 answered every time at 11.9 V.
  - **Motors 3 and 4 never answered.**
  - Motors 5 and 6 answered only some of the time.
- **Checked every ID (0–30) at every Feetech baud rate:** no motors with wrong IDs or wrong speeds were found.
- **Diagnosis:** a bad cable or connector between motor 2 (shoulder_lift) and motor 3 (elbow_flex). This cannot be fixed in software.
- **Outcome:** the team replaced the arm.

## 7. New arm: calibration
- **Port:** COM5. All 6 motors scored 5/5 on `check_arm.py`: 11.9–12.0 V, 24–26 °C.
- **First calibration attempt:** run in the Claude app's built-in terminal. **It got stuck** at the "move all joints" step.
  - **Cause:** LeRobot detects Enter with a Windows console key check (`enter_pressed()`), which cannot see key presses in that embedded terminal.
  - **Fix:** ran `recalibrate.bat --port COM5` in a normal Windows console window instead.
- **Result:** the calibration was saved, and a copy is stored in this repo as `calibration/petri_follower.json`.

**Recorded ranges (raw encoder values):**

| Joint | Min | Max | Homing offset |
|---|---|---|---|
| shoulder_pan | 763 | 3491 | -1924 |
| shoulder_lift | 892 | 3287 | 1945 |
| elbow_flex | 747 | 2958 | 1976 |
| wrist_flex | 958 | 3302 | 1893 |
| wrist_roll | 0 | 4095 (full turn) | 1383 |
| gripper | 2029 (closed stop) | 3513 | 801 |

**Check after calibration** (read-only, torque off), with the arm in the middle position:
- Joints read pan −3.4°, lift −2.4°, elbow 11.2°, wrist flex 0.5°, roll 0.0°; gripper 0.1% open.
- Forward kinematics put the gripper tip at (0.382, 0.021, 0.181) m.

## 8. Training data
- **Searched Hugging Face** for SO-101 and petri-dish datasets. **No petri-dish robot dataset exists.**
- **Datasets considered:**

| Dataset | Size | Notes |
|---|---|---|
| `lerobot/svla_so101_pickplace` | 82 MB | 50 episodes, 2 cameras, official, **chosen** |
| `sreetz-nv/so101_teleop_vials_rack_left` | 392 MB | 75 episodes, lab vials |
| `youliangtan/so101-table-cleanup` | 500 MB | 80 episodes |
| `DecisionFacts/Physical_AI_SO101_Test_Tube_Nesting_Task` | 4.1 GB | 160 episodes |

- **Downloaded** `lerobot/svla_so101_pickplace` into `data/` (not committed; use `download_assets.py`).
- **Verified with `LeRobotDataset`:** 11,939 frames at 30 fps; cameras `up` and `side` at 640×480. The task is "pink lego brick into the transparent box". **Its joint names match `petri_arm.py` exactly.**

## 9. GitHub
- Logged in with `gh auth login` (browser device code), then ran `gh auth setup-git`.
- **Target repository:** `MakOltz/WashyBoy` (public). It contained only `README.md` (`# WashyBoy`).
- **Status:** the logged-in account has **read-only** access to this repo, so pushing needs the owner to grant write access (Settings → Collaborators).
- **Not committed:** `data/` and `SO-ARM100/` (large, downloadable). No passwords, tokens or email addresses are in the repo.
- **Write access to `MakOltz/WashyBoy` never came through:** the account stayed read-only, with no pending invite.
- **Created our own repository instead:** **`Mako7667/WashyBoy`** (private), with `gh repo create Mako7667/WashyBoy --private`.
- **Remotes:** `origin` is now `Mako7667/WashyBoy`, where we push. `upstream` is `MakOltz/WashyBoy`, which we can still pull from.
- **Pushed `main`:** the original "Initial commit" plus the project commit. Commits use GitHub's no-reply email address.
- **To give teammates access:** on GitHub, open the repo's **Settings → Collaborators → Add people**.

---

## Open items / next steps
1. **Teach poses** with `PetriArm.teach()`: `home`, `above_stack`, `above_clean_station`, `clean_station`.
2. **Mount the camera** and run `calibrate_camera.py`.
3. **Test** `example_pick_and_clean.py` with one plate.
4. **Optional, for an AI policy:** get an SO-101 leader arm, record about 50 petri-dish runs with `lerobot-record`, and fine-tune SmolVLA. This needs an NVIDIA GPU.
