"""release_motors.py - switch all SO-101 motor torque OFF so the arm can be moved by hand.
HOLD THE ARM FIRST: it goes limp immediately. Usage: python release_motors.py [--port COM5]"""
import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM5")
bus = SO101Follower(SO101FollowerConfig(port=ap.parse_args().port, id="petri_follower")).bus
bus.connect(handshake=False)
bus.disable_torque()
print("Torque:", bus.sync_read("Torque_Enable", normalize=False))
bus.disconnect(disable_torque=True)
print("Motors loose.")
