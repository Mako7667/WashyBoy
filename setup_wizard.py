"""
setup_wizard.py - one-time guided setup for camera-driven WashyBoy. Run in a normal console
window (setup_wizard.bat), not inside an IDE terminal.

Steps
  1. Pick the robot camera (a preview of each camera is shown).
  2. Teach ONE pose, "home": arm parked where it does NOT block the camera's view.
  3. Touch the 4 table markers (IDs 0-3) with the gripper tip so the camera knows where the
     arm is. The table height is measured at the same time.
  4. Live check: the camera view shows detected plates and stations with arm coordinates.

Everything is saved to robot_setup.json, table_markers.json and poses.json.
Re-run any time the camera or table markers are moved (step 3 only: --markers-only).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import vision
from petri_arm import PLATE_HEIGHT_M, POSES_FILE, PetriArm
from recalibrate import find_arm_port

SETUP_FILE = Path(__file__).with_name("robot_setup.json")


def load_setup() -> dict:
    # utf-8-sig: tolerate a byte-order mark (files saved by Windows PowerShell/Notepad)
    return json.loads(SETUP_FILE.read_text(encoding="utf-8-sig")) if SETUP_FILE.exists() else {}


def save_setup(d: dict):
    SETUP_FILE.write_text(json.dumps(d, indent=2))


def choose_camera() -> int:
    print("\n[1/4] Choose the robot camera. Each camera is shown for a few seconds.")
    found = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            found.append(idx)
            cv2.putText(frame, f"camera {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("camera preview", frame)
            cv2.waitKey(2500)
        cap.release()
    cv2.destroyAllWindows()
    if not found:
        raise SystemExit("No cameras found. Plug in the robot camera and try again.")
    print(f"Cameras found: {found}. (On a laptop, 0 is usually the built-in webcam.)")
    while True:
        s = input(f"Which camera is the ROBOT camera? {found}: ").strip()
        if s.isdigit() and int(s) in found:
            return int(s)


def teach_home(arm: PetriArm):
    print("\n[2/4] Teach the HOME pose.")
    print("  Motors will go limp. Fold the arm so it does not block the camera's view of the")
    print("  work area (e.g. pulled back and up), then press Enter. Hold the arm!")
    input("  Press Enter to turn torque OFF...")
    arm.robot.bus.disable_torque()
    input("  Arm in HOME position? Press Enter to save...")
    arm.poses["home"] = arm.get_joints()
    POSES_FILE.write_text(json.dumps(arm.poses, indent=2))
    arm.robot.bus.enable_torque()
    print(f"  Saved home: { {k: round(v, 1) for k, v in arm.poses['home'].items()} }")


def touch_markers(arm: PetriArm, raised: bool) -> float:
    print("\n[3/4] Table markers. Tape markers 0-3 near the corners of the work area, all inside")
    print("  the camera view and within ~30 cm of the robot base. Tape them down flat.")
    if raised:
        print("  Markers are RAISED to plate height (on spare plates / 12.4 mm blocks): touch the")
        print("  marker paper itself, on top of the block.")
    input("  Press Enter to turn torque OFF (hold the arm)...")
    arm.robot.bus.disable_torque()
    markers, zs = {}, []
    try:
        for mid in vision.TABLE_IDS:
            print(f"  Move the gripper TIP onto the CENTRE of marker {mid}, touching the paper.")
            while True:
                frame = arm.get_frame()
                px, corners, ids = vision.detect_markers(frame)
                vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.putText(vis, f"touch marker {mid}, then press SPACE in this window", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("setup", vis)
                if cv2.waitKey(30) & 0xFF == ord(" "):
                    break
            x, y, z = arm.get_xyz()
            markers[mid] = (x, y)
            zs.append(z)
            print(f"    marker {mid}: x={x:.3f} y={y:.3f} z={z:.3f} m")
    finally:
        arm.robot.bus.enable_torque()
        cv2.destroyAllWindows()
    vision.save_table_markers(markers)
    table_z = float(np.median(zs)) - (PLATE_HEIGHT_M if raised else 0.0)
    if np.ptp(zs) > 0.015:
        print(f"  WARNING: marker heights differ by {np.ptp(zs) * 1000:.0f} mm - is the table level / tip touching?")
    print(f"  Table height in arm frame: {table_z:.3f} m. Saved table_markers.json")
    return table_z


def live_check(arm: PetriArm):
    print("\n[4/4] Live check. Place a plate and the station markers (10, 11) in view.")
    print("  Green boxes = plates seen (with arm x,y). Press ESC in the window to finish.")
    arm.home()
    while True:
        scene = arm.look()
        for p in scene.plates:
            if not arm.in_pick_zone(p.x, p.y):
                cv2.putText(scene.image, "out of reach", (p.px[0] - 40, p.px[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("live check", scene.image)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--markers-only", action="store_true", help="only redo step 3 (camera/markers moved)")
    args = ap.parse_args()
    setup = load_setup()
    setup["port"] = args.port or setup.get("port") or find_arm_port()
    if not args.markers_only or "camera_index" not in setup:
        setup["camera_index"] = choose_camera()
    save_setup(setup)

    with PetriArm(port=setup["port"], camera_index=setup["camera_index"], camera_rotation=setup.get("camera_rotation", 0)) as arm:
        if not args.markers_only:
            teach_home(arm)
        print("\nIs the camera at an angle (e.g. on the side of the arm) rather than straight overhead?")
        print("If so, raise markers 0-3 to plate height (tape each onto a spare plate or a 12.4 mm")
        print("block). This removes parallax error on the plate tops (~12 mm at 45 degrees).")
        raised = input("Are markers 0-3 raised to plate height? [y/N]: ").strip().lower() == "y"
        setup["markers_raised"] = raised
        setup["table_z"] = touch_markers(arm, raised)
        save_setup(setup)
        live_check(arm)
        input("\nSetup done. Hold the arm, then press Enter to switch the motors off...")
    print(f"Saved {SETUP_FILE.name}. Next: autonomous.bat (or autonomous.bat --record to collect training data)")


if __name__ == "__main__":
    main()
