#!/usr/bin/env python3
"""
SO-ARM101 petri-dish dry / rack routine (LeRobot, leader + follower,
wrist camera, microfiber cloth).

  teach   Drive the FOLLOWER with the LEADER arm, press Enter to save each pose.
  vision  Live camera view with the dish detector overlaid (arm does not move).
  run     Pick (vision-aligned), wipe dry on the cloth, rack - for each dish.

  python petri_dish_cleaner.py teach  --follower-port P1 --leader-port P2 --slots 0 1 2
  python petri_dish_cleaner.py vision --camera 0
  python petri_dish_cleaner.py run    --follower-port P1 --camera 0 --slots 0 1 2 --speed 0.5

Physical setup: dish stack at a known spot on a contrasting mat; a small
microfiber cloth fixed flat (taped/clipped onto a block); a slotted rack.
The dish is held by its rim and wiped against the fixed cloth.

Verify LeRobot class/import names against your installed version.
Keep electronics away from water; run at low --speed first.
"""
import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:  # LeRobot 0.7+ layout (SO-100/101 share one module)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
except ImportError:  # older layout
    from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]
POSES_FILE = Path("cleaner_poses.json")  # separate from petri_arm.py poses.json

# ---- motion tunables --------------------------------------------------------
GRIPPER_OPEN = 60.0
LOOP_HZ = 30
MOVE_S = 2.0
CONTACT_S = 1.2
GRIP_CLOSE_S = 0.8
MAX_RELATIVE_TARGET = 10.0
DRY_CYCLES, DRY_ROLL_AMP, DRY_PAN_AMP, DRY_PERIOD = 6, 30.0, 4.0, 1.0

# ---- grasp verification -----------------------------------------------------
# If the gripper closes MORE than the taught "on the dish" value by this margin,
# the jaws closed on nothing (missed) or the dish was dropped.
GRIP_MISS_MARGIN = 6.0

# ---- vision tunables (wrist camera looking down at the dish) ----------------
CAM_W, CAM_H = 640, 480
DISH_MIN_R, DISH_MAX_R = 60, 200      # dish radius in pixels at the "dish_above" height
TARGET_PX = (CAM_W // 2, int(CAM_H * 0.6))  # where the dish should sit in frame before descent
PIX_TOL = 8
SERVO_ITERS = 20
PAN_GAIN = 0.04          # joint units per pixel of horizontal error (sign: tune!)
REACH_GAIN = 0.04        # joint units per pixel of vertical error (sign: tune!)
REACH_COUPLING = 0.8     # shoulder_lift vs elbow_flex ratio for in/out reach (tune)


# =============================================================================
# robot helpers
# =============================================================================
def make_follower(port, rid):
    r = SO101Follower(SO101FollowerConfig(
        port=port, id=rid, max_relative_target=MAX_RELATIVE_TARGET))
    r.connect()
    return r


def read_pose(robot):
    obs = robot.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in JOINTS}


def send(robot, pose):
    robot.send_action({f"{j}.pos": float(pose[j]) for j in JOINTS})


def smoothstep(t):
    return t * t * (3 - 2 * t)


def move_to(robot, target, duration=MOVE_S, gripper=None):
    start = read_pose(robot)
    goal = {j: target[j] for j in ARM_JOINTS}
    goal["gripper"] = start["gripper"] if gripper is None else gripper
    steps = max(1, int(duration * LOOP_HZ))
    for i in range(1, steps + 1):
        s = smoothstep(i / steps)
        send(robot, {j: start[j] + (goal[j] - start[j]) * s for j in JOINTS})
        time.sleep(1 / LOOP_HZ)


def set_gripper(robot, value, duration=GRIP_CLOSE_S):
    cur = read_pose(robot)
    steps = max(1, int(duration * LOOP_HZ))
    for i in range(1, steps + 1):
        p = dict(cur)
        p["gripper"] = cur["gripper"] + (value - cur["gripper"]) * i / steps
        send(robot, p)
        time.sleep(1 / LOOP_HZ)


def oscillate(robot, base, cycles, period, roll_amp, pan_amp=0.0):
    n = int(period * LOOP_HZ)
    grip = read_pose(robot)["gripper"]
    for k in range(cycles * n):
        ph = 2 * math.pi * k / n
        p = dict(base)
        p["wrist_roll"] = base["wrist_roll"] + roll_amp * math.sin(ph)
        p["shoulder_pan"] = base["shoulder_pan"] + pan_amp * math.sin(ph / 2)
        p["gripper"] = grip
        send(robot, p)
        time.sleep(1 / LOOP_HZ)


class DishLost(RuntimeError):
    pass


def check_holding(robot, grip_closed, where):
    g = read_pose(robot)["gripper"]
    if g < grip_closed - GRIP_MISS_MARGIN:
        raise DishLost(f"Gripper closed to {g:.1f} (< {grip_closed:.1f}) at '{where}': "
                       "no dish in the jaws.")


# =============================================================================
# vision
# =============================================================================
def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}")
    return cap


def grab(cap, flush=3):
    for _ in range(flush):  # drop stale buffered frames
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return frame


def detect_dish(frame):
    """Return (cx, cy, r) of the most central circle, or None.
    Works best with a clear/translucent plastic dish on a dark, matte mat."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=120, param2=40,
                               minRadius=DISH_MIN_R, maxRadius=DISH_MAX_R)
    if circles is None:
        return None
    c = min(circles[0], key=lambda c: (c[0] - CAM_W / 2) ** 2 + (c[1] - CAM_H / 2) ** 2)
    return float(c[0]), float(c[1]), float(c[2])


def servo_to_dish(robot, cap, base_pose):
    """Nudge the arm until the dish is centred at TARGET_PX. Returns the corrected
    pose. Raises if the dish can't be found/centred (never grab blindly)."""
    pose = {j: base_pose[j] for j in JOINTS}
    for _ in range(SERVO_ITERS):
        det = detect_dish(grab(cap))
        if det is None:
            raise RuntimeError("Dish not detected - check the stack/lighting.")
        ex, ey = det[0] - TARGET_PX[0], det[1] - TARGET_PX[1]
        if abs(ex) < PIX_TOL and abs(ey) < PIX_TOL:
            return pose
        pose["shoulder_pan"] -= PAN_GAIN * ex
        d = REACH_GAIN * ey
        pose["shoulder_lift"] += d
        pose["elbow_flex"] -= d * REACH_COUPLING
        move_to(robot, pose, duration=0.4)
        time.sleep(0.15)
    raise RuntimeError("Vision alignment did not converge.")


def vision_test(index):
    cap = open_camera(index)
    print("Live detection. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detect_dish(frame)
        cv2.drawMarker(frame, TARGET_PX, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if det:
            cv2.circle(frame, (int(det[0]), int(det[1])), int(det[2]), (0, 255, 0), 2)
        cv2.imshow("dish detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# teaching with the leader arm
# =============================================================================
def required_poses(slots):
    names = ["home", "dish_above", "dish_grasp", "dry_above", "dry_contact"]
    for s in slots:
        names += [f"rack_above_{s}", f"rack_slot_{s}"]
    return names


def teach(follower, leader, slots):
    poses = json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else {}
    stop = threading.Event()

    def mirror():  # follower tracks the leader
        while not stop.is_set():
            act = leader.get_action()
            follower.send_action(act)
            time.sleep(1 / LOOP_HZ)

    t = threading.Thread(target=mirror, daemon=True)
    t.start()
    print("Follower now mirrors the leader.")
    print("dish_above: hold the camera over the dish stack so the dish is in view.")
    print("dish_grasp: lower the follower so the jaws close on the dish rim; that")
    print("            gripper value is used for carrying. Other poses: gripper ignored.\n")
    try:
        for name in required_poses(slots):
            tag = " (saved)" if name in poses else ""
            a = input(f"'{name}'{tag} - Enter to record, s to skip: ").strip().lower()
            if a == "s":
                continue
            poses[name] = read_pose(follower)
            POSES_FILE.write_text(json.dumps(poses, indent=2))
            print(f"  saved {name}")
    finally:
        stop.set()
        t.join()


# =============================================================================
# the routine
# =============================================================================
def process_dish(robot, cap, P, slot):
    grip_closed = P["dish_grasp"]["gripper"]

    # 1. Vision-aligned pick
    move_to(robot, P["dish_above"], gripper=GRIPPER_OPEN)
    above = servo_to_dish(robot, cap, P["dish_above"])
    # Re-apply the taught descent (grasp - above) on top of the corrected pose
    grasp = {j: above[j] + (P["dish_grasp"][j] - P["dish_above"][j]) for j in ARM_JOINTS}
    move_to(robot, grasp, CONTACT_S, gripper=GRIPPER_OPEN)
    set_gripper(robot, grip_closed)
    check_holding(robot, grip_closed, "pickup")
    move_to(robot, above, CONTACT_S)
    check_holding(robot, grip_closed, "lift")

    # 2. Dry: press the dish onto the fixed microfiber cloth and wipe by
    #    rolling the wrist with a slight sideways sweep
    move_to(robot, P["dry_above"])
    move_to(robot, P["dry_contact"], CONTACT_S)
    oscillate(robot, P["dry_contact"], DRY_CYCLES, DRY_PERIOD,
              DRY_ROLL_AMP, DRY_PAN_AMP)
    check_holding(robot, grip_closed, "dry")
    move_to(robot, P["dry_above"], CONTACT_S)

    # 3. Rack
    move_to(robot, P[f"rack_above_{slot}"])
    move_to(robot, P[f"rack_slot_{slot}"], CONTACT_S)
    set_gripper(robot, GRIPPER_OPEN)
    move_to(robot, P[f"rack_above_{slot}"], CONTACT_S)
    move_to(robot, P["home"])


def run(robot, cap, slots, speed):
    global MOVE_S, CONTACT_S
    MOVE_S /= speed
    CONTACT_S /= speed
    if not POSES_FILE.exists():
        raise SystemExit("poses.json missing - run 'teach' first.")
    P = json.loads(POSES_FILE.read_text())
    missing = [n for n in required_poses(slots) if n not in P]
    if missing:
        raise SystemExit(f"Missing poses: {missing}")

    move_to(robot, P["home"], gripper=GRIPPER_OPEN)
    for n, slot in enumerate(slots, 1):
        print(f"Dish {n}/{len(slots)} -> slot {slot}")
        try:
            process_dish(robot, cap, P, slot)
        except (DishLost, RuntimeError) as e:
            print(f"STOPPED: {e}")
            set_gripper(robot, GRIPPER_OPEN)
            move_to(robot, P["home"])
            return
    print("Done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["teach", "vision", "run"])
    ap.add_argument("--follower-port")
    ap.add_argument("--leader-port")
    ap.add_argument("--follower-id", default="petri_follower")  # matches calibration/petri_follower.json
    ap.add_argument("--leader-id", default="my_leader")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--slots", type=int, nargs="+", default=[0])
    ap.add_argument("--speed", type=float, default=0.5)
    a = ap.parse_args()

    if a.mode == "vision":
        return vision_test(a.camera)

    if not a.follower_port:
        raise SystemExit("--follower-port required")
    follower = make_follower(a.follower_port, a.follower_id)
    try:
        if a.mode == "teach":
            if not a.leader_port:
                raise SystemExit("--leader-port required for teach")
            leader = SO101Leader(SO101LeaderConfig(port=a.leader_port, id=a.leader_id))
            leader.connect()
            try:
                teach(follower, leader, a.slots)
            finally:
                leader.disconnect()
        else:
            cap = open_camera(a.camera)
            try:
                run(follower, cap, a.slots, a.speed)
            finally:
                cap.release()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
