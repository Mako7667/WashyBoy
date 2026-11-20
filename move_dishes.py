"""
move_dishes.py - move the dish on spot 1 to A and the dish on spot 2 to B.

  python move_dishes.py teach   motors go loose; put the gripper (jaws OPEN) where it should
                                grip/release at 1, 2, A, B and press Enter at each.
  python move_dishes.py run     does 1 -> A, then 2 -> B (lifts LIFT_M between spots).

Options for run: --open 60 --closed 20 (gripper %), --lift 0.06 (m), --speed 1.0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from petri_arm import MOTORS, PetriArm

SPOTS_FILE = Path(__file__).with_name("dish_spots.json")
LEARNED_FILE = Path(__file__).with_name("grip_learned.json")  # squeeze that worked, per spot
SPOTS = ["1", "2", "A", "B"]


def teach(arm, leader_port=None, only=None):
    spots = json.loads(SPOTS_FILE.read_text()) if SPOTS_FILE.exists() else {}
    stop, thread, leader = None, None, None
    if leader_port:
        # Black LEADER arm drives the white FOLLOWER (LeRobot teleoperation).
        import threading
        import time
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        leader = SO101Leader(SO101LeaderConfig(port=leader_port, id="petri_leader"))
        leader.connect()  # first time: runs LeRobot's leader calibration (needs a real console window)
        stop = threading.Event()

        import logging
        logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)  # hide "clamped" spam
        arm.robot.config.max_relative_target = 30.0  # less lag than 10, still safety-checked
        arm._bus_lock = threading.Lock()  # saving a spot and following never use the bus at once

        def mirror():  # follow the leader (no left/right flip)
            while not stop.is_set():
                try:
                    act = leader.get_action()
                    with arm._bus_lock:
                        arm.robot.send_action(act)
                except Exception as e:  # a dropped USB packet must not stop following
                    print(f"\n  (bus hiccup, retrying: {str(e)[:60]})")
                    time.sleep(0.1)
                time.sleep(1 / 30)

        thread = threading.Thread(target=mirror, daemon=True)
        thread.start()
        print("White arm now follows the BLACK leader arm. Move the black arm to each spot.")
    else:
        arm.robot.bus.disable_torque()
        print("Motors loose. For each spot: put the gripper around/at the dish rim height, jaws OPEN.")
    try:
        for s in (only or SPOTS):
            input(f"  Gripper at spot {s}? Press Enter to save...")
            with getattr(arm, '_bus_lock', None) or __import__('contextlib').nullcontext():
                spots[s] = arm.get_joints()
            SPOTS_FILE.write_text(json.dumps(spots, indent=2))
            print(f"    saved {s}")
        input("  Now put the arm in a safe resting pose clear of the dishes, press Enter (saved as 'rest')...")
        with getattr(arm, '_bus_lock', None) or __import__('contextlib').nullcontext():
            spots["rest"] = arm.get_joints()
        SPOTS_FILE.write_text(json.dumps(spots, indent=2))
    finally:
        if stop is not None:
            stop.set()
            thread.join()
            leader.disconnect()
        print("Done teaching.")


def above(arm, joints, lift):
    """Same pose lifted `lift` metres straight up (IK keeps the taught gripper angle)."""
    q = np.array([joints[m] for m in MOTORS])
    T = arm.kinematics.forward_kinematics(q).copy()
    T[2, 3] += lift
    q2 = arm.solve_ik(q, T)
    return {m: float(v) for m, v in zip(MOTORS, q2) if m != "gripper"}


def go(arm, joints, t):
    arm.move_joints(duration=t, **{m: v for m, v in joints.items() if m != "gripper"})


def fast_camera(idx):
    """Open a camera asking for 30 fps with MJPG (USB webcams often drop to ~5-15 fps in raw YUY2)."""
    import cv2
    c = cv2.VideoCapture(idx)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(cv2.CAP_PROP_FPS, 30)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, less lag
    return c


def check(arm, spot):
    """Go to a taught grip pose and HOLD there, so you can see if the jaws are around the dish."""
    spots = json.loads(SPOTS_FILE.read_text())
    S = spots[spot]
    arm.set_gripper(min(100.0, S["gripper"] + 15))
    go(arm, above(arm, S, 0.06), 2.0)
    go(arm, S, 1.5)
    print(f"Holding at spot {spot}. Are the open jaws around the dish rim? (motors stay on)")


def see_dish(cap, label):
    """Wrist camera check: find round dish rims in view. Saves seen_<label>.png. Returns count."""
    import cv2
    if cap is None:
        return -1
    for _ in range(4):  # flush stale frames
        cap.read()
    ok, f = cap.read()
    if not ok:
        print("  (camera read failed)")
        return -1
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    # Clear plastic on white paper is low-contrast: boost local contrast first (CLAHE).
    g = cv2.GaussianBlur(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray), (7, 7), 1.5)
    h, w = g.shape
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.3, minDist=int(w * 0.15), param1=90, param2=45,
                         minRadius=int(w * 0.08), maxRadius=int(w * 0.30))
    circles = [] if c is None else [tuple(int(v) for v in x) for x in c[0]]
    # Ignore the top third (background, cables, the arm) and circles mostly off-frame.
    circles = [(x, y, r) for x, y, r in circles if y > h / 3 and r < x < w - r * 0.5 + w * 0.1]
    for x, y, r in circles:
        cv2.circle(f, (x, y), r, (0, 255, 0), 3)
    cv2.putText(f, f"{label}: {len(circles)} dish(es) seen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"seen_{label}.png", f)
    cv2.imshow("wrist camera", f)
    cv2.waitKey(300)
    print(f"  camera at {label}: {len(circles)} dish(es) seen -> seen_{label}.png")
    return len(circles)


def run(arm, a):
    spots = json.loads(SPOTS_FILE.read_text())
    t = 2.0 / a.speed

    # Gentle gripper: cap the gripper motor's force (LeRobot's default is 50% = 500).
    with arm.robot.bus.torque_disabled("gripper"):  # only the gripper relaxes for a moment
        arm.robot.bus.write("Max_Torque_Limit", "gripper", int(a.grip_force * 10))
    print(f"Gripper force limited to {a.grip_force:.0f}%")

    cap, wrist_idx = None, None
    if a.camera >= 0 or a.camera == -2:
        # auto: the wrist camera currently enumerates as 0; a newly plugged judge camera comes after
        for idx in ([a.camera] if a.camera >= 0 else [0, 1, 2, 3]):
            c = fast_camera(idx)
            ok, fr = c.read() if c.isOpened() else (False, None)
            if ok and fr.std() > 12:  # skip 'camera disabled' placeholders
                cap, wrist_idx = c, idx
                print(f"Wrist camera: index {idx}")
                break
            c.release()
        if cap is None:
            print("No working camera - running without camera checks")

    def grip_values(spot):
        """Open/close based on the jaw opening TAUGHT at this spot (dish width)."""
        g = spot["gripper"]
        opening = a.open if a.open is not None else min(100.0, g + a.open_margin)
        closed = a.closed if a.closed is not None else max(0.0, g - a.squeeze)
        return opening, closed

    # Judge camera: a second, fixed camera that watches the table (any working camera that is
    # not the wrist camera). Optional - the gripper's own position is the main judge.
    judge = None
    if a.judge_camera != -1:
        for idx in ([a.judge_camera] if a.judge_camera >= 0 else [1, 2, 3, 0]):
            if idx == wrist_idx:
                continue
            c = fast_camera(idx)
            ok, fr = c.read() if c.isOpened() else (False, None)
            if ok and fr.std() > 12:
                judge = c
                print(f"Judge camera: index {idx}")
                break
            c.release()
    learned = json.loads(LEARNED_FILE.read_text()) if LEARNED_FILE.exists() else {}

    for src, dst in [(p[0], p[1]) for p in a.pairs.split(",")]:
        S, D = spots[src], spots[dst]
        g_open, _ = grip_values(S)
        squeeze = learned.get(src, a.squeeze)
        print(f"Moving dish {src} -> {dst}  (open {g_open:.0f}%, squeeze {squeeze:.0f}% past taught)")
        arm.set_gripper(g_open)
        go(arm, above(arm, S, a.lift), t)
        # Taught positions come first: the camera only reports, it never skips a dish
        # (unless --trust-camera is given).
        if see_dish(cap, f"spot{src}") == 0 and a.trust_camera:
            print(f"  No dish seen at spot {src} - skipping it")
            continue
        holding = False
        before = see_dish(judge, f"judge_{src}_before")
        for attempt in range(1, a.attempts + 1):
            g_closed = a.closed if a.closed is not None else max(0.0, S["gripper"] - squeeze)
            go(arm, S, t * 0.75)
            arm.move_joints(duration=1.2, gripper=g_closed)  # close slowly so the dish centres itself
            time.sleep(0.3)
            g_now = arm.get_joints()["gripper"]
            # Gripper judge: stalled above its target => something (the dish) is between the jaws.
            grip_ok = g_now > g_closed + a.stall_margin
            holding = grip_ok
            last = attempt == a.attempts
            print(f"  attempt {attempt}: gripper stopped at {g_now:.1f}% (target {g_closed:.1f}%) -> "
                  f"{'HOLDING' if grip_ok else 'no contact felt'}"
                  f"{'' if holding or last else ', squeezing harder'}")
            if holding or last:  # last try: keep this (tightest) grip and carry on to the taught drop spot
                if holding:
                    learned[src] = squeeze
                    LEARNED_FILE.write_text(json.dumps(learned, indent=2))
                go(arm, above(arm, S, a.lift), t * 0.75)
                break
            # Improve: open, squeeze a bit more next time (gripper already at the taught grip pose).
            arm.set_gripper(g_open)
            squeeze = min(squeeze + a.squeeze_step, S["gripper"])
        after = see_dish(judge, f"judge_{src}_after")
        if before >= 0 and after >= 0:
            print(f"  judge camera: {before} -> {after} round objects ({'dish lifted' if after < before else 'unclear'})")
        go(arm, above(arm, D, a.lift), t)
        go(arm, D, t * 0.75)
        arm.set_gripper(g_open)
        go(arm, above(arm, D, a.lift), t * 0.75)
        n = see_dish(cap, f"spot{dst}")
        if n == 0:
            print(f"  WARNING: camera does not see a dish at {dst} after placing")
    go(arm, spots.get("rest", spots["A"]), t)
    for c in (cap, judge):
        if c is not None:
            c.release()
    print(f"Done: {a.pairs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teach", "run", "check"])
    ap.add_argument("spot", nargs="?", default="1", help="for check: which spot (1, 2, A, B)")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--open", type=float, default=None, help="fixed open %% (default: taught opening + margin)")
    ap.add_argument("--closed", type=float, default=None, help="fixed closed %% (default: taught opening - squeeze)")
    ap.add_argument("--open-margin", type=float, default=15, help="open this much wider than taught before descending")
    ap.add_argument("--squeeze", type=float, default=16, help="close this much past the taught opening")
    ap.add_argument("--grip-force", type=float, default=35, help="gripper motor force limit %% (LeRobot default 50)")
    ap.add_argument("--judge-camera", type=int, default=-2, help="fixed judge camera index; -2 auto, -1 none")
    ap.add_argument("--attempts", type=int, default=2, help="grip tries per dish before skipping")
    ap.add_argument("--only", nargs="+", help="teach only these spots, e.g. --only 1 A")
    ap.add_argument("--pairs", default="1A,2B", help="moves to run, e.g. 1A or 1A,2B")
    ap.add_argument("--squeeze-step", type=float, default=4, help="extra squeeze %% after a failed grip")
    ap.add_argument("--stall-margin", type=float, default=1.0, help="gripper stops this far above target = holding")
    ap.add_argument("--lift", type=float, default=0.06)
    ap.add_argument("--speed", type=float, default=1.6)
    ap.add_argument("--camera", type=int, default=-2, help="wrist camera index; -2 = auto, -1 = no camera")
    ap.add_argument("--ignore-camera", action="store_true", help="(default now) camera never skips a dish")
    ap.add_argument("--trust-camera", action="store_true", help="skip a spot if the camera sees no dish")
    ap.add_argument("--leader", nargs="?", const="auto", help="teach with the black leader arm (port or auto)")
    a = ap.parse_args()
    leader_port = a.leader
    if leader_port == "auto":
        from serial.tools import list_ports
        cands = [p.device for p in list_ports.comports() if p.vid == 0x1A86 and p.device != a.port]
        if not cands:
            raise SystemExit("Leader (black) arm not found. Plug in its USB and power.")
        leader_port = cands[0]
        print(f"Leader arm on {leader_port}")
    arm = PetriArm(port=a.port)
    arm.robot.config.disable_torque_on_disconnect = (a.mode == "teach")  # after run: keep holding
    arm.connect()
    try:
        if a.mode == "teach":
            teach(arm, leader_port, a.only)
        elif a.mode == "check":
            check(arm, a.spot)
        else:
            run(arm, a)
    finally:
        arm.disconnect()
