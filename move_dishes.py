"""
move_dishes.py - move the dish on spot 1 to A and the dish on spot 2 to B.

  python move_dishes.py teach   motors go loose; put the gripper (jaws OPEN) where it should
                                grip/release at 1, 2, A, B and press Enter at each.
  python move_dishes.py run     does 1 -> A, then 2 -> B (lifts LIFT_M between spots).

Options for run: --open 60 --closed 20 (gripper %), --lift 0.06 (m), --speed 1.0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from petri_arm import MOTORS, PetriArm

SPOTS_FILE = Path(__file__).with_name("dish_spots.json")
SPOTS = ["1", "2", "A", "B"]


def teach(arm, leader_port=None, mirror=False, hz=60):
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

        # Responsive following: relax the per-command safety clamp while a human drives it.
        arm.robot.config.max_relative_target = 25.0
        flip = ("shoulder_pan.pos", "wrist_roll.pos") if mirror else ()

        def follow():
            while not stop.is_set():
                t0 = time.perf_counter()
                act = leader.get_action()
                for k in flip:  # mirror image: arms facing each other
                    act[k] = -act[k]
                arm.robot.send_action(act)
                time.sleep(max(0.0, 1 / hz - (time.perf_counter() - t0)))

        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        print(f"White arm now follows the BLACK leader arm at {hz} Hz"
              f"{' (MIRRORED left/right)' if mirror else ''}. Move the black arm to each spot.")
    else:
        arm.robot.bus.disable_torque()
        print("Motors loose. For each spot: put the gripper around/at the dish rim height, jaws OPEN.")
    try:
        for s in SPOTS:
            input(f"  Gripper at spot {s}? Press Enter to save...")
            spots[s] = arm.get_joints()
            SPOTS_FILE.write_text(json.dumps(spots, indent=2))
            print(f"    saved {s}")
        input("  Now put the arm in a safe resting pose clear of the dishes, press Enter (saved as 'rest')...")
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
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=110, param2=38,
                         minRadius=40, maxRadius=230)
    circles = [] if c is None else [tuple(int(v) for v in x) for x in c[0]]
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
    cap = None
    if a.camera >= 0:
        import cv2
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            print("Camera not available - running without camera checks")
            cap = None
    for src, dst in [("1", "A"), ("2", "B")]:
        print(f"Moving dish {src} -> {dst}")
        S, D = spots[src], spots[dst]
        arm.set_gripper(a.open)
        go(arm, above(arm, S, a.lift), t)
        if see_dish(cap, f"spot{src}") == 0 and not a.ignore_camera:
            print(f"  No dish seen at spot {src} - skipping it (use --ignore-camera to grab anyway)")
            continue
        go(arm, S, t * 0.75)
        arm.close_gripper(a.closed)
        go(arm, above(arm, S, a.lift), t * 0.75)
        go(arm, above(arm, D, a.lift), t)
        go(arm, D, t * 0.75)
        arm.set_gripper(a.open)
        go(arm, above(arm, D, a.lift), t * 0.75)
        n = see_dish(cap, f"spot{dst}")
        if n == 0:
            print(f"  WARNING: camera does not see a dish at {dst} after placing")
    go(arm, spots.get("rest", spots["A"]), t)
    if cap is not None:
        cap.release()
    print("Done: 1 -> A, 2 -> B")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teach", "run"])
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--open", type=float, default=60)
    ap.add_argument("--closed", type=float, default=20)
    ap.add_argument("--lift", type=float, default=0.06)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--camera", type=int, default=1, help="wrist camera index; -1 = no camera")
    ap.add_argument("--ignore-camera", action="store_true", help="grab even if no dish is seen")
    ap.add_argument("--leader", nargs="?", const="auto", help="teach with the black leader arm (port or auto)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="turn off left/right mirroring (on by default)")
    ap.add_argument("--hz", type=int, default=60, help="leader-follow update rate")
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
        teach(arm, leader_port, a.mirror, a.hz) if a.mode == "teach" else run(arm, a)
    finally:
        arm.disconnect()
