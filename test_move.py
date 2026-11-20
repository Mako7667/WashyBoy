"""
test_move.py - small, slow motion test. Moves relative to wherever the arm is now:
base +-10 deg, gripper open/close, then back to the start pose.
Motors are LEFT ON afterwards (arm holds its pose, does not drop). To release:
power-cycle the arm or run any script that disconnects normally while holding the arm.
Usage:  python test_move.py [--port COM5]
"""
import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM5")
ap.add_argument("--loops", type=int, default=1, help="repeat count; 0 = forever (stop with Ctrl+C)")
args = ap.parse_args()
port = args.port

robot = SO101Follower(SO101FollowerConfig(port=port, id="petri_follower", use_degrees=True,
                                          max_relative_target=5.0, disable_torque_on_disconnect=False))
robot.connect(calibrate=False)


def joints():
    o = robot.get_observation()
    return {m: float(o[f"{m}.pos"]) for m in MOTORS}


def move(goal, duration=1.5, hz=30):
    start = joints()
    steps = int(duration * hz)
    for i in range(1, steps + 1):
        s = i / steps
        s = s * s * (3 - 2 * s)
        robot.send_action({f"{m}.pos": start[m] + (goal[m] - start[m]) * s for m in MOTORS})
        time.sleep(1 / hz)


import logging
logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)  # hide clamp chatter

try:
    home = joints()
    print("start:", {k: round(v, 1) for k, v in home.items()})
    n = 0
    while args.loops == 0 or n < args.loops:
        n += 1
        for label, delta in [("pan +10", {"shoulder_pan": 10}), ("pan -10", {"shoulder_pan": -10}), ("pan back", {})]:
            move({**home, **{k: home[k] + v for k, v in delta.items()}})
        g_open = (move({**home, "gripper": 60.0}, duration=2.5), joints()["gripper"])[1]
        move({**home, "gripper": home["gripper"]}, duration=2.5)
        end = joints()
        drift = max(abs(end[m] - home[m]) for m in MOTORS[:-1])
        print(f"loop {n}: gripper opened to {g_open:5.1f}, back to start within {drift:.1f} deg", flush=True)
except KeyboardInterrupt:
    print("\nStopped by Ctrl+C - returning to start pose")
    move(home, duration=1.5)
finally:
    robot.disconnect()  # torque stays ON (disable_torque_on_disconnect=False)
    print("Done. Motors still holding.")
