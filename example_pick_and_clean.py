"""
Example routine: find a dish with the camera, pick its plate, take it to the cleaning
station, and put it back on a stack. Edit PORT / CAMERA and the taught pose names.

Before running, teach these poses once with:  python -c "from petri_arm import PetriArm; PetriArm('COM5').connect().teach()"
  home, above_clean_station, clean_station, above_stack
"""

import json

from petri_arm import PetriArm, pixel_to_table

PORT = "COM5"      # the arm's port on this PC (check with check_arm.bat)
CAMERA = 0         # from `lerobot-find-cameras opencv`
HOVER = 0.08       # metres above the plate before descending
GRIP_OPEN = 90     # % open, wide enough for the 70 mm plate
GRIP_CLOSED = 20   # % open when holding the plate (tune so the motor doesn't stall)

calib = json.load(open("camera_calib.json"))

with PetriArm(port=PORT, camera_index=CAMERA) as arm:
    arm.home()

    dishes = arm.find_dishes(show=True)
    if not dishes:
        raise SystemExit("No dish seen")
    u, v, r = dishes[0]
    x, y = pixel_to_table(u, v, calib)
    z_top = arm.stack_height(0)
    print(f"Dish at pixel ({u},{v}) -> table ({x:.3f}, {y:.3f}) m")

    # pick
    arm.set_gripper(GRIP_OPEN)
    arm.move_to_xyz(x, y, z_top + HOVER)
    arm.move_to_xyz(x, y, z_top - 0.005, duration=1.5)
    arm.close_gripper(GRIP_CLOSED)
    arm.move_relative(dz=HOVER)

    # clean
    arm.go_to("above_clean_station")
    arm.go_to("clean_station")
    for _ in range(3):                       # simple scrub motion
        arm.move_joints(duration=0.4, wrist_roll=arm.get_joints()["wrist_roll"] + 30)
        arm.move_joints(duration=0.4, wrist_roll=arm.get_joints()["wrist_roll"] - 30)
    arm.go_to("above_clean_station")

    # place on top of a stack (here: on top of 2 plates already there)
    arm.go_to("above_stack")
    sx, sy, _ = arm.get_xyz()
    arm.move_to_xyz(sx, sy, arm.stack_height(2) + 0.003, duration=1.5)
    arm.set_gripper(GRIP_OPEN)
    arm.move_relative(dz=HOVER)
    arm.home()
