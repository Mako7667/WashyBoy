"""lock_motors.py - hold the SO-101 exactly where it is now (torque ON, no jump).
Usage: python lock_motors.py [--port COM5].  Undo with release_motors.py."""
import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM5")
bus = SO101Follower(SO101FollowerConfig(port=ap.parse_args().port, id="petri_follower")).bus
bus.connect(handshake=False)
pos = bus.sync_read("Present_Position", normalize=False, num_retry=5)
bus.sync_write("Goal_Position", pos, normalize=False)  # target = where it is -> no jump
bus.enable_torque()
print("Locked at raw positions:", pos)
print("Torque:", bus.sync_read("Torque_Enable", normalize=False))
bus.disconnect(disable_torque=False)
