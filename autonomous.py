"""
autonomous.py - camera-driven WashyBoy loop. Needs setup_wizard.py to have been run once.

  Loop:  park at home (out of view) -> camera looks at the table -> pick the nearest dirty
         plate (any plate not on the stack / cleaning station) -> scrub it at the cleaning
         station (marker 10) -> stack it on the clean stack (marker 11) -> repeat.

  The robot decides WHERE to go from the camera every cycle: move a plate, the stack marker
  or the cleaning marker and it adapts. Nothing is hard-coded except the home pose.

Options
  --record          save every successful cycle as a LeRobot training episode (data/washyboy_pick_clean)
  --stack-start N   plates already on the clean stack (default 0)
  --max N           stop after N plates
  --dry-run         only show what the camera sees and what it would do; the arm does not move
"""

import argparse
import time

import cv2
import numpy as np

from petri_arm import PLATE_HEIGHT_M, PetriArm
from setup_wizard import load_setup

STATION_CLEARANCE_M = 0.08  # plates closer than this to a station marker are treated as "at" it
SCRUB_HEIGHT_M = 0.04  # held plate height above the table while scrubbing
SCRUB_STROKES = 4


def choose_target(arm: PetriArm, scene):
    """Nearest reachable plate that is not sitting on the stack or the cleaning station."""
    candidates = []
    for p in scene.plates:
        near_station = any(np.hypot(p.x - sx, p.y - sy) < STATION_CLEARANCE_M for sx, sy in scene.stations.values())
        if near_station or not arm.in_pick_zone(p.x, p.y):
            continue
        candidates.append((np.hypot(p.x, p.y), p))
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def scrub(arm: PetriArm, x: float, y: float, table_z: float, jaw_yaw: float):
    """Simple scrub: hold the plate on the cleaning station and wipe it side to side."""
    z = table_z + SCRUB_HEIGHT_M
    arm.move_down_to(x, y, z, jaw_yaw, duration=1.5)
    for _ in range(SCRUB_STROKES):
        arm.move_down_to(x + 0.02, y, z - 0.02, jaw_yaw, duration=0.5)
        arm.move_down_to(x - 0.02, y, z - 0.02, jaw_yaw, duration=0.5)
    arm.move_down_to(x, y, z, jaw_yaw, duration=0.5)


def look_until_ready(arm: PetriArm, need_stations=("clean_station", "stack"), tries=20):
    for _ in range(tries):
        scene = arm.look()
        cv2.imshow("WashyBoy camera", scene.image)
        cv2.waitKey(1)
        if scene.homography is not None and all(s in scene.stations for s in need_stations):
            return scene
        time.sleep(0.2)
    missing = [s for s in need_stations if s not in scene.stations]
    raise RuntimeError(f"Camera can't see: {'table markers' if scene.homography is None else missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--stack-start", type=int, default=0)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    setup = load_setup()
    if not {"port", "camera_index", "table_z"} <= setup.keys():
        raise SystemExit("Run setup_wizard.bat first.")
    table_z = setup["table_z"]
    plate_top = table_z + PLATE_HEIGHT_M

    with PetriArm(port=setup["port"], camera_index=setup["camera_index"], camera_rotation=setup.get("camera_rotation", 0)) as arm:
        rec = None
        if args.record:
            from recorder import EpisodeRecorder

            rec = EpisodeRecorder(arm.robot.observation_features, arm.robot.action_features,
                                  root="data/washyboy_pick_clean")
            print(f"Recording training data: {rec.num_episodes} episodes so far")

        level, done = args.stack_start, 0
        try:
            while done < args.max:
                arm.home()
                scene = look_until_ready(arm)
                target = choose_target(arm, scene)
                if target is None:
                    print("No dirty plates in reach. Finished.")
                    break
                sx, sy = scene.stations["stack"]
                cx, cy = scene.stations["clean_station"]
                for name, (px, py) in scene.stations.items():
                    if not arm.in_pick_zone(px, py):
                        raise RuntimeError(f"Station '{name}' is {np.hypot(px, py) * 100:.0f} cm from the base; "
                                           f"move its marker to {arm.PICK_R_MIN * 100:.0f}-{arm.PICK_R_MAX * 100:.0f} cm.")
                print(f"Plate at ({target.x:.3f}, {target.y:.3f}) angle {target.yaw_deg:+.0f} deg "
                      f"-> clean at ({cx:.3f}, {cy:.3f}) -> stack level {level} at ({sx:.3f}, {sy:.3f})")
                if args.dry_run:
                    cv2.waitKey(0)
                    break

                arm.recorder = rec
                try:
                    yaw = arm.pick_plate(target.x, target.y, target.yaw_deg, plate_top)
                    scrub(arm, cx, cy, table_z, yaw)
                    arm.place_plate(sx, sy, arm.stack_height(level, table_z), yaw)
                    arm.recorder = None
                    arm.home()
                finally:
                    arm.recorder = None

                # verify with the camera: the plate should be gone from where it was
                after = arm.look()
                still_there = any(np.hypot(p.x - target.x, p.y - target.y) < 0.02 for p in after.plates)
                if rec is not None:
                    if still_there:
                        rec.discard_episode()
                        print("  Pick failed (plate still there) - episode discarded")
                    else:
                        rec.save_episode()
                        print(f"  Saved episode {rec.num_episodes}")
                if still_there:
                    print("  Plate still at its old spot - retrying.")
                    continue
                level += 1
                done += 1
        finally:
            if rec is not None:
                rec.close()
            cv2.destroyAllWindows()
            arm.home()
            input("Hold the arm, then press Enter to switch the motors off...")


if __name__ == "__main__":
    main()
