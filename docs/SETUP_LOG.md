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
- **Made the repository public** at the user's request, with `gh repo edit --visibility public`. Before that, the repo was scanned for tokens, passwords and email addresses; none were found. It is now at https://github.com/Mako7667/WashyBoy. Anyone can read and clone it; only collaborators can push.

## 10. Camera-driven control and training pipeline
**Goal:** the robot decides where to go from the camera, not from hand-taught positions.

- **Cameras found:** only the laptop's built-in webcam (index 0). **A USB robot camera is still needed.**

**`vision.py` (OpenCV ArUco + contours):**
- Table markers 0–3 give a fresh pixel→table mapping every frame, so the camera re-aligns itself if bumped. With only 3 markers visible, it falls back to an affine mapping.
- Station markers: 10 is the cleaning station, 11 is the clean stack.
- Plates are found as ~70 mm squares, with position and angle. Dishes are found as circles.
- **Bug fixed during testing:** the table's own outline hid every plate, because only the outermost shapes were being searched. Fixed by searching all shapes and removing duplicates.
- **Synthetic test** (camera at an angle): plate positions within 1 mm, angles within 2°, stations exact.

**Top-down picking (`petri_arm.py`):**
- `solve_down`, `move_down_to`, `pick_plate`, `place_plate`, `look`, `in_pick_zone`.
- **Bug fixed:** the first "pointing down" reference pose was wrong (the gripper's sideways axis pointed down, not its approach axis). The URDF search found the right one: wrist_flex about +80°.
- **Bug fixed:** the jaw-angle convention was off by 180°, and the wrist-roll direction was guessed. Now it's measured from the URDF.
- **Result (40 random targets):** position median 0.16 mm, jaw angle exact, tilt median 4.8°, wrist roll ≤ 48°.
- **Reach map:** top-down works 14–30 cm from the base, up to about 8.5 cm high. Hover height is capped to stay reachable, so a stack is limited to about 5 plates.

**Environment fix: NumPy matrix maths crashed** (`0xc06d007f`, a missing function in a library file). Installing placo had switched NumPy to an Intel MKL version that didn't match. **Fix:** switched to OpenBLAS (pthreads build); the OpenMP build clashed with PyTorch's own OpenMP.

**Other new pieces:**
- **Autonomous loop** (`autonomous.py`): park → look → pick the nearest dirty plate → scrub at marker 10 → stack at marker 11 → check with the camera that the plate actually moved → repeat. Options: `--dry-run`, `--max`, `--record`.
- **Setup wizard** (`setup_wizard.py`): pick the camera, teach only `home`, touch the 4 table markers (which also measures table height), live check.
- **Marker sheet** (`make_markers.py`): `markers_A4.pdf` with six 40 mm markers.

**Training data recorder** (`recorder.py`):
- Saves each successful autonomous cycle as a LeRobot episode (camera video, joint state, action, 30 fps); failed cycles are discarded; later sessions add to the same dataset.
- **Tested** with fake frames: 3 episodes and 50 frames, as expected; it reloads correctly.

**Training pipeline:**
- Installed `lerobot[training]` (needed for `accelerate`).
- `--wandb.enable` no longer exists in LeRobot 0.7; the replacement is `--trackio.enable`.
- **Smoke test:** a 2-step ACT run on the CPU with `lerobot/svla_so101_pickplace` trained and saved the model (loss 77.9 → 70.5). The `checkpoints/last` symlink then failed with WinError 1314, because Windows needs Developer Mode for symlinks. The workaround (use the step-numbered folder) is documented.
- **Guide:** `docs/TRAINING.md` (ACT and SmolVLA commands, and `lerobot-rollout` to run a trained policy).
- **Planner end-to-end test** (synthetic image → vision → target choice → IK for each move): it chose the right plate and skipped the out-of-reach one. Every move was reachable after the hover cap.

## 11. Team script: `Petri Dish Cleaner.py` → `petri_dish_cleaner.py`
**What it is:** a wrist-camera routine (teach with a leader arm → visual-servo pick → wipe on a microfiber cloth → rack slots), with a grip check that detects a missed or dropped dish.

**Read before running.** It could not move the arm as written:
- **Import paths:** `lerobot.robots.so101_follower` / `lerobot.teleoperators.so101_leader` don't exist in LeRobot 0.7 (now `so_follower` / `so_leader`), and the fallback path doesn't exist either. **Fixed** in the repo copy.
- **Calibration ID:** the default `--follower-id my_follower` would have started a new calibration. **Changed** to `petri_follower`.
- **Poses file:** renamed to `cleaner_poses.json`, so it doesn't overwrite `petri_arm.py`'s `poses.json`.
- **Not run on the arm:** `teach` needs a leader arm, and only one arm (COM5) is connected. `run` needs the poses that `teach` creates. Running `run` now would only connect, fail, and disconnect, which switches the torque off and drops the arm.

**Verified offline:**
- The script loads (`--help` works).
- `detect_dish` found a synthetic dish at (353, 301) with radius 110 px; expected about (350, 300) and 100–110 px.

**Design difference from `autonomous.py`:**

| | This script | `autonomous.py` |
|---|---|---|
| Camera | Wrist-mounted | Overhead |
| Teaching | 11 poses with a leader arm | 1 pose plus markers |
| Gripping | Holds dishes by the rim | Holds the 70 mm plate |

The two can coexist; which to use depends on where the camera is mounted.

## 12. Side-mounted camera
The user confirmed the camera will be fixed on the **left side** of the arm, not overhead or on the wrist.

**Problem: parallax.** The camera sees plate tops 12.4 mm above the table, so an angled view shifts them.

**Simulation** (pinhole camera, 30 random plates):

| Camera position | View angle | Markers on table | Markers raised 12.4 mm |
|---|---|---|---|
| 45 cm out, 35 cm up | 52° | 17.3 mm median error | 0.0 mm |
| 40 cm out, 55 cm up | 36° | 9.7 mm median error | 0.0 mm |

**Fix:**
- The setup wizard now asks whether markers 0–3 are raised to plate height, and subtracts 12.4 mm when working out the table height.
- `make_markers.py --size` allows 30–60 mm markers; the sheet was regenerated at 55 mm.

**Already built on existing projects:** LeRobot (motors, cameras, datasets, training), OpenCV (ArUco markers and contours), placo (IK) and the SO-ARM100 URDF. No further libraries were added before the 3:30 deadline, to avoid new risk.

## 13. Camera plugged in, first powered arm motion
**Cameras:**
- Camera 0 is the laptop webcam; **camera 1 is the robot camera**.
- Its frame is **rotated 90°** (mounted sideways), sits **low**, mostly sees the room, window and arm, and has strong window backlight. The work area is only a thin strip.
- **Added `camera_rotation`** to `PetriArm` and saved `robot_setup.json` with port COM5, camera 1, rotation 90.
- LeRobot wants the frame size *after* rotation (480×640). The first attempt failed on this and was fixed.

**`test_move.py`:** small relative moves (base ±10°, gripper open/close, return) with a 5° safety clamp. Motors stay on afterwards (`disable_torque_on_disconnect=False`), so the arm doesn't drop.
- **Run 1:** base 11.3 → 21.1 → 1.8 → 11.0°. The gripper reached 40.6% because the clamp limited the speed. Returned within 0.4°.
- **Run 2:** same result, within 0.4°.
- **Loop mode** (`--loops 0`, stop with Ctrl+C): gripper move slowed to 2.5 s, now reaches 59.6%. Loop 1 returned within 0.5°.

## 14. Wizard fix and dish-only mode
- **Wizard crash:** `JSONDecodeError` when reading `robot_setup.json`. **Cause:** I wrote the file from PowerShell, which added an invisible UTF-8 byte-order mark. **Fix:** the file was rewritten without it, and the wizard now reads with `utf-8-sig`, which accepts either form.
- **The plates are still printing,** so the user wants to move and stack the petri dishes directly. **New `stack_dishes.py` / `.bat`:** the camera finds round dishes (40–110 mm) → the arm picks the nearest reachable one with a whole-dish grip from above → places it on stack marker 11 at level × dish height → the camera checks that the dish moved → repeat. Options: `--dry-run`, `--max`, `--dish-height`, `--grip-open/closed/depth`, `--record`.
- **Offline test** (two 90 mm clear dishes on a dark mat, angled camera): both detected, the nearest chosen, centre within about 7 mm.
- **Setup advice for dishes:** raise table markers 0–3 by taping them onto upside-down spare dishes (removes parallax; answer "y" in the wizard). Put a dark mat under the work area.

---

## Open items / next steps
1. **Get a USB camera** and mount it above the work area, looking down.
2. **Print `markers_A4.pdf`** at 100% and check that a black square measures 40 mm. Tape the markers down.
3. **Run `setup_wizard.bat`.**
4. **Check the gripper:** can it open to grip the 70 mm plate?
5. **Run `autonomous.bat --dry-run`,** then `--max 1`.
6. **Run `autonomous.bat --record`** to collect 50–100 episodes, then train on a GPU (`docs/TRAINING.md`).
