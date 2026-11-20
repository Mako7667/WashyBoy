"""
recalibrate.py - fully recalibrate the SO-101 follower arm using LeRobot's own calibration routine.

Usage:
    python recalibrate.py                    # auto-detects the arm's USB port (it was COM4 on this PC)
    python recalibrate.py --port COM4
    python recalibrate.py --setup-motors     # ALSO re-assign motor IDs (only for new or swapped motors)

What happens (LeRobot's standard procedure, https://huggingface.co/docs/lerobot/so101#calibrate):
  1. Torque turns off. Move the arm so EVERY joint is in the middle of its range, then press Enter.
  2. Move EACH joint slowly through its full range of motion (end stop to end stop),
     including opening/closing the gripper. Press Enter when done.
  3. The new calibration is saved and overwrites the old file for this --id.
"""

import argparse
import sys

from serial.tools import list_ports  # pyserial, installed with lerobot

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# USB-serial chips used by SO-101 controller boards (Waveshare / Feetech): WCH CH340/CH343, FTDI
KNOWN_ADAPTERS = {(0x1A86, 0x55D3), (0x1A86, 0x7523), (0x1A86, 0x55D4), (0x0403, 0x6001)}


def find_arm_port() -> str:
    ports = [p for p in list_ports.comports() if (p.vid, p.pid) in KNOWN_ADAPTERS]
    if len(ports) == 1:
        return ports[0].device
    all_ports = [f"{p.device} ({p.description})" for p in list_ports.comports()]
    if not ports:
        sys.exit(f"No SO-101 controller board found. Is USB + power plugged in? Ports seen: {all_ports}")
    sys.exit(f"Several possible arms found: {[p.device for p in ports]}. Pick one with --port.")


def main():
    ap = argparse.ArgumentParser(description="Recalibrate the SO-101 follower arm")
    ap.add_argument("--port", help="serial port, e.g. COM4 (auto-detected if omitted)")
    ap.add_argument("--id", default="petri_follower", help="calibration name; must match PetriArm(arm_id=...)")
    ap.add_argument("--setup-motors", action="store_true", help="re-assign motor IDs/baudrate first")
    args = ap.parse_args()

    port = args.port or find_arm_port()
    print(f"Using arm on {port}, calibration id '{args.id}'")

    robot = SO101Follower(SO101FollowerConfig(port=port, id=args.id, use_degrees=True))

    if args.setup_motors:
        print("\n=== Motor ID setup: you'll be asked to connect ONE motor at a time ===")
        robot.setup_motors()
        input("\nNow daisy-chain all motors again (shoulder_pan -> controller), then press Enter...")

    robot.connect(calibrate=False)
    try:
        print("\n=== Calibration ===")
        robot.calibrate()  # always runs a fresh calibration and saves it
        print(f"\nDone. Calibration saved to: {robot.calibration_fpath}")
        print("Joint readings now (degrees; gripper 0-100):")
        obs = robot.get_observation()
        for k, v in obs.items():
            print(f"  {k:20s} {v:8.1f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
