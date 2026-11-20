"""save_spot.py - save the white arm's CURRENT pose as a dish spot (no motion, torque unchanged).
Usage: python save_spot.py 1        (or A, 2, B, rest)"""
import json
import sys
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

name = sys.argv[1]
f = Path(__file__).with_name("dish_spots.json")
robot = SO101Follower(SO101FollowerConfig(port="COM5", id="petri_follower", use_degrees=True))
bus = robot.bus
bus.connect(handshake=False)
try:
    pos = bus.sync_read("Present_Position", num_retry=5)  # calibrated degrees / gripper %
finally:
    bus.disconnect(disable_torque=False)
spots = json.loads(f.read_text()) if f.exists() else {}
spots[name] = {k: float(v) for k, v in pos.items()}
f.write_text(json.dumps(spots, indent=2))
print(f"saved spot {name}:", {k: round(v, 1) for k, v in spots[name].items()})
