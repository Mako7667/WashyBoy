"""
check_arm.py - read-only health check of the SO-101 servo chain. Nothing moves.

Pings all 6 motors several times and reports which answer, plus voltage and temperature.
Usage:  python check_arm.py [--port COM4]
"""

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from recalibrate import find_arm_port

ROUNDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    args = ap.parse_args()
    port = args.port or find_arm_port()

    robot = SO101Follower(SO101FollowerConfig(port=port, id="petri_follower"))
    bus = robot.bus
    bus.connect(handshake=False)
    results = {}
    try:
        for name in bus.motors:
            ok, info = 0, ""
            for _ in range(ROUNDS):
                try:
                    if bus.ping(name, num_retry=1) is not None:
                        ok += 1
                        v = bus.read("Present_Voltage", name, normalize=False)
                        t = bus.read("Present_Temperature", name, normalize=False)
                        info = f"{v / 10:.1f} V, {t} C"
                except Exception:
                    pass
            results[name] = ok
            status = "OK" if ok == ROUNDS else ("FLAKY" if ok else "NO ANSWER")
            print(f"  id {bus.motors[name].id}  {name:14s} {ok}/{ROUNDS}  {status:10s} {info}")
    finally:
        bus.disconnect(disable_torque=False)

    names = list(bus.motors)
    bad = [n for n in names if results[n] < ROUNDS]
    print()
    if not bad:
        print(f"All 6 motors healthy on {port}. Ready to calibrate:  python recalibrate.py")
        return
    first_bad = names.index(bad[0])
    if first_bad > 0 and all(results[n] == ROUNDS for n in names[:first_bad]):
        print(f"Problem starts at '{bad[0]}'. Check the 3-pin cable between '{names[first_bad - 1]}'"
              f" and '{bad[0]}' (reseat both ends), and that motor's two connectors.")
    elif first_bad == 0:
        print("Even the first motor fails: check the controller-board power (12 V for SO-101)"
              " and the cable from the board to 'shoulder_pan'.")
    print("If one motor never answers even with good cables, it may have the wrong ID:"
          " run  python recalibrate.py --setup-motors")


if __name__ == "__main__":
    main()
