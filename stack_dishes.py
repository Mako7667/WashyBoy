"""
stack_dishes.py - camera-driven: find round petri dishes, pick each one, stack it on marker 11.
No printed plates needed. Run setup_wizard.bat first (only the stack marker, ID 11, is required
besides the 4 table markers).

  python stack_dishes.py --dry-run          camera only: shows the dish it would pick (arm still)
  python stack_dishes.py --max 1            one dish
  python stack_dishes.py                    until no dishes are left in reach

Tune for your dishes:
  --dish-height 15     mm, height of one dish (base + lid) = stacking step
  --grip-open 100      % gripper opening before descending (must clear the dish diameter)
  --grip-closed 30     % when holding (higher = gentler; raise it if the motor strains)
  --grip-depth 7       mm below the dish top where the jaws close
Tips: dark mat under the dishes (clear plastic shows up best on dark), even lighting.
"""

import argparse
import time

import cv2
import numpy as np

from petri_arm import PetriArm
from setup_wizard import load_setup

DISH_R_MIN, DISH_R_MAX = 0.020, 0.055  # metres: accept 40-110 mm diameter dishes
STACK_CLEARANCE_M = 0.06  # dishes within this of the stack marker are already stacked


def choose_dish(arm, scene):
    stack = scene.stations.get("stack")
    best = None
    for x, y, r in scene.dishes:
        if not (DISH_R_MIN <= r <= DISH_R_MAX) or not arm.in_pick_zone(x, y):
            continue
        if stack and np.hypot(x - stack[0], y - stack[1]) < STACK_CLEARANCE_M:
            continue
        d = np.hypot(x, y)
        if best is None or d < best[0]:
            best = (d, x, y, r)
    return best[1:] if best else None


def look(arm, tries=20):
    for _ in range(tries):
        scene = arm.look()
        cv2.imshow("WashyBoy camera", scene.image)
        cv2.waitKey(1)
        if scene.homography is not None and "stack" in scene.stations:
            return scene
        time.sleep(0.2)
    raise RuntimeError("Camera can't see " + ("the table markers" if scene.homography is None else "the stack marker (11)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--stack-start", type=int, default=0, help="dishes already on the stack")
    ap.add_argument("--dish-height", type=float, default=15.0)
    ap.add_argument("--grip-open", type=float, default=100.0)
    ap.add_argument("--grip-closed", type=float, default=30.0)
    ap.add_argument("--grip-depth", type=float, default=7.0)
    ap.add_argument("--record", action="store_true", help="save successful runs as LeRobot training data")
    a = ap.parse_args()

    setup = load_setup()
    if not {"port", "camera_index", "table_z"} <= setup.keys():
        raise SystemExit("Run setup_wizard.bat first.")
    table_z, h, depth = setup["table_z"], a.dish_height / 1000, a.grip_depth / 1000

    with PetriArm(port=setup["port"], camera_index=setup["camera_index"],
                  camera_rotation=setup.get("camera_rotation", 0)) as arm:
        rec = None
        if a.record:
            from recorder import EpisodeRecorder
            rec = EpisodeRecorder(arm.robot.observation_features, arm.robot.action_features,
                                  root="data/washyboy_stack_dishes",
                                  task="Pick up the petri dish and place it on the stack.")
        level, done = a.stack_start, 0
        try:
            while done < a.max:
                arm.home()
                scene = look(arm)
                target = choose_dish(arm, scene)
                if target is None:
                    print("No dishes in reach (14-30 cm from the base). Finished.")
                    break
                x, y, r = target
                sx, sy = scene.stations["stack"]
                print(f"Dish at ({x:.3f}, {y:.3f}) diameter {2 * r * 1000:.0f} mm -> stack level {level} at ({sx:.3f}, {sy:.3f})")
                if a.dry_run:
                    print("Dry run: press any key in the camera window to exit.")
                    cv2.waitKey(0)
                    break
                arm.recorder = rec
                try:
                    yaw = arm.pick_plate(x, y, 0.0, table_z + h, grip_open=a.grip_open,
                                         grip_closed=a.grip_closed, grip_depth=depth)
                    arm.place_plate(sx, sy, table_z + (level + 1) * h, yaw,
                                    grip_open=a.grip_open, grip_depth=depth)
                finally:
                    arm.recorder = None
                arm.home()
                after = arm.look()
                moved = not any(np.hypot(dx - x, dy - y) < 0.02 for dx, dy, _ in after.dishes)
                if rec is not None:
                    rec.save_episode() if moved else rec.discard_episode()
                if not moved:
                    print("  Dish still there - pick missed. Check --grip-open / --grip-depth. Retrying.")
                    continue
                level += 1
                done += 1
                print(f"  Stacked. {done} done.")
        finally:
            if rec is not None:
                rec.close()
            cv2.destroyAllWindows()
            input("Hold the arm, then press Enter to switch the motors off...")


if __name__ == "__main__":
    main()
