"""
One-time camera -> table calibration.

1. Put 4 small markers on the table where the arm can reach.
2. For each marker: click it in the camera window, then jog the gripper tip onto it by hand
   (torque is off) and press Enter. The pixel and the arm's xyz are paired up.
3. Saves camera_calib.json, used by petri_arm.pixel_to_table().

Usage:  python calibrate_camera.py COM5 0
"""

import json
import sys

import cv2

from petri_arm import PetriArm

port, cam = sys.argv[1], int(sys.argv[2])
clicked = []

with PetriArm(port=port, camera_index=cam) as arm:
    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", lambda e, x, y, *_: e == cv2.EVENT_LBUTTONDOWN and clicked.append([x, y]))
    pixels, table = [], []
    while len(pixels) < 4:
        print(f"Marker {len(pixels) + 1}/4: click it in the window...")
        clicked.clear()
        while not clicked:
            cv2.imshow("calib", cv2.cvtColor(arm.get_frame(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(30)
        arm.robot.bus.disable_torque()
        input("Now move the gripper tip onto that marker by hand and press Enter")
        x, y, _ = arm.get_xyz()
        arm.robot.bus.enable_torque()
        pixels.append(clicked[0])
        table.append([x, y])
        print(f"  pixel {clicked[0]} -> table ({x:.3f}, {y:.3f}) m")

json.dump({"pixels": pixels, "table_m": table}, open("camera_calib.json", "w"), indent=2)
print("Saved camera_calib.json")
