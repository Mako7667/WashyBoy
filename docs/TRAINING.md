# Camera-driven WashyBoy: setup and training

WashyBoy decides what to do from the **camera**. You teach it one pose; everything else comes from what it sees. There are two layers:

| Layer | What decides the motion | Status |
|---|---|---|
| **1. Vision + planner** (`autonomous.py`) | The camera finds plates and stations; the code computes the arm motion | Works without training |
| **2. Learned policy** (ACT / SmolVLA) | A neural network maps camera images to arm motion | Trained on recordings from layer 1, or from teleoperation |

Layer 1 also produces the training data for layer 2, so no leader arm is required.

---

## Part 1: Camera setup (one time)

### What you need
- **A USB webcam** for the robot. The PC currently has only the laptop's built-in camera. Any 640×480 or better webcam works. Mount it **above the work area looking down**, rigidly: tape or clamp it, and don't move it afterwards.
- **The printed markers:** run `python make_markers.py`, print `markers_A4.pdf` at **100% / actual size**, and check that a black square measures 40 mm.
- Even, steady lighting. Avoid direct sunlight and moving shadows.

### Camera on the side of the arm (your setup)
A camera fixed to the **left side** of the arm works. It doesn't have to be overhead.
- **Mount it high,** looking down at the work area: the higher, the better. Aim for at least 40 cm up.
- **Raise markers 0–3 to plate height:** tape each onto a spare plate or a 12.4 mm block, then answer **y** in the wizard. Without this, a side view misplaces plate centres by about 10–20 mm because of parallax. Raised markers remove that error completely in simulation.
- **Print bigger markers:** `python make_markers.py --size 55`. They look smaller and squashed from the side.
- **Make sure the `home` pose clears the camera's view,** so the arm doesn't hide the plates.

### Lay out the table
```
            camera (above, looking down)

   [0]                              [1]
          [10 CLEAN]      plates
                          to clean
     [11 STACK]
   [3]                              [2]
                 ===ARM BASE===
```
- **Markers 0–3** (table): near the corners of the work area, all visible to the camera, all **14–30 cm from the arm base** (the zone the arm can reach pointing straight down). Tape them flat.
- **Marker 10:** the cleaning station (put your brush or sponge here).
- **Marker 11:** where clean plates are stacked.
- Station markers can be moved at any time; the robot finds them every cycle.

### Run the wizard
Double-click **`setup_wizard.bat`**. It runs in a normal console window, which is needed for keyboard input.
1. **Pick the camera:** each camera is previewed; type the robot camera's number.
2. **Teach `home`:** the motors go limp; fold the arm out of the camera's view; press Enter. This is the only pose you teach.
3. **Touch the markers:** for each table marker 0–3, put the gripper tip on its centre and press SPACE in the camera window. This tells the camera where the arm is and measures the table height.
4. **Live check:** green boxes show plates with their arm coordinates. Plates marked "out of reach" are outside the 14–30 cm zone.

If the camera or table markers move, run `setup_wizard.bat --markers-only`.

### Run it
```bash
autonomous.bat --dry-run
```
```bash
autonomous.bat --max 1
```
```bash
autonomous.bat
```
- `--dry-run`: shows what the camera sees and what it would do; the arm doesn't move.
- `--max 1`: one plate, with a hand on the power switch.
- No options: runs until no dirty plates are left.

**Things to tune in the code:**

| Setting | Where | What it does |
|---|---|---|
| `grip_open`, `grip_closed`, `grip_depth` | `pick_plate()` | How wide the gripper opens, how hard it grips, how deep it reaches |
| `SCRUB_*` | `autonomous.py` | The cleaning motion |
| `STATION_CLEARANCE_M` | `autonomous.py` | How close to a station marker a plate counts as "at" it |

**Known limits (measured in simulation):**
- **Reach:** top-down picks work 14–30 cm from the base, up to about 8.5 cm high. That limits a stack to **about 5 plates**; start a new stack marker beyond that.
- **Spacing:** keep plates **at least 2 cm apart**. Touching plates merge into one shape and are skipped.
- **Covered markers:** don't let plates cover a table marker. The code falls back to 3 markers, but plate detection near a covered marker was unreliable in testing.

> ⚠️ **Check the gripper width first.** The plate is 70 mm square. The stock SO-101 gripper may not open that wide. If it can't, grip the plate by a raised edge or handle, or redesign the plate with a grip tab.

---

## Part 2: Collecting training data

```bash
autonomous.bat --record
```
- Each **successful** cycle (camera confirms the plate moved) is saved as one episode in `data/washyboy_pick_clean/`. Failed cycles are discarded.
- **Format:** LeRobot, the same as `lerobot-record`: `observation.images.cam` (the camera), `observation.state` (6 joints), `action` (6 joint commands), at 30 fps.
- **Target:** 50–100 episodes. Vary the plate positions and angles, the lighting, and the number of plates.

**Alternative: teleoperation.** If you get an SO-101 leader arm, `lerobot-record` records human demonstrations. Human demos teach recovery behaviours that the script never shows. Mixing both kinds is good.

---

## Part 3: Training

Training needs an **NVIDIA GPU**. This PC has a CPU-only PyTorch: fine for testing the pipeline, far too slow for real training. Options: a PC with an RTX card, Google Colab, or a cloud GPU.

Copy `data/washyboy_pick_clean/` to the GPU machine, or push it to the Hugging Face Hub.

**ACT** (small, fast, good first choice; about 1–2 hours on an RTX 3060 for 50k steps):
```bash
lerobot-train --dataset.repo_id=local/washyboy_pick_clean --dataset.root=data/washyboy_pick_clean --policy.type=act --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --steps=50000 --batch_size=8 --output_dir=outputs/train/act_washyboy --job_name=act_washyboy
```

**SmolVLA** (fine-tunes Hugging Face's pretrained vision-language-action model; uses the task text):
```bash
lerobot-train --dataset.repo_id=local/washyboy_pick_clean --dataset.root=data/washyboy_pick_clean --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --steps=20000 --batch_size=32 --output_dir=outputs/train/smolvla_washyboy --job_name=smolvla_washyboy
```

**Windows note:** at the end of training, LeRobot creates a `checkpoints/last` shortcut (a symlink). Windows refuses unless **Developer Mode** is on (Settings → System → For developers), and shows `WinError 1314`. The model is still saved: use `checkpoints/<step number>/pretrained_model` instead of `checkpoints/last/...`.

**Tested on this PC:** a 2-step ACT run on the CPU with the downloaded dataset. It took about 3 s per step, the loss fell from 77.9 to 70.5, and a 207 MB model was saved. The pipeline works; real training needs a GPU.

**Practise first:** run the same commands with `--dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=data/svla_so101_pickplace` (downloaded by `download_assets.py`).

## Part 4: Running a trained policy

The camera name must match the one used in recording (`cam`):
```bash
lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=petri_follower --robot.cameras="{cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --policy.path=outputs/train/act_washyboy/checkpoints/last/pretrained_model --task="Pick up the petri dish plate, clean it, and place it on the clean stack." --duration=60
```
Replace `index_or_path: 1` with the camera number chosen in the wizard (saved in `robot_setup.json`).

**Recommended for the competition:** keep the vision + planner as the main controller, because it is predictable, and use the learned policy where the planner struggles, for example the scrubbing motion. Compare their success rates over 20 runs each before deciding.
