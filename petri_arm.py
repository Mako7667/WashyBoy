"""
petri_arm.py - a simple interface to the SO-101 arm for the petri-dish robot.

Uses existing libraries instead of custom drivers:
  * Hugging Face LeRobot  -> motor control, calibration, cameras
      https://huggingface.co/docs/lerobot/so101
  * LeRobot RobotKinematics (placo backend) + the official SO-101 URDF
      https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
  * OpenCV HoughCircles  -> find the round petri dish in the camera image

Units used everywhere in this file:
  joints  = degrees (gripper = 0 closed ... 100 open)
  xyz     = metres, in the arm's base frame (from the URDF)

Quick start (see README.md for setup):

    from petri_arm import PetriArm
    with PetriArm(port="COM5", camera_index=0) as arm:
        arm.home()
        arm.open_gripper()
        arm.move_to_xyz(0.20, 0.0, 0.05)
        arm.close_gripper()
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Your 3D-printed plate ("Stunning Amur.stl"): 70 x 70 mm footprint, 12.4 mm tall.
PLATE_SIZE_M = 0.070
PLATE_HEIGHT_M = 0.0124

POSES_FILE = Path(__file__).with_name("poses.json")


class PetriArm:
    def __init__(
        self,
        port: str,
        arm_id: str = "petri_follower",
        camera_index: int | str | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        urdf_path: str | None = str(Path(__file__).with_name("SO-ARM100") / "Simulation/SO101/so101_new_calib.urdf"),
        max_step_deg: float = 10.0,
    ):
        """
        port          serial port of the follower arm (find it with `lerobot-find-port`), e.g. "COM5"
        arm_id        must match the --robot.id you used for `lerobot-calibrate`
        camera_index  OpenCV camera index (find it with `lerobot-find-cameras opencv`), or None
        urdf_path     SO-101 URDF, needed only for move_to_xyz()/get_xyz()
        max_step_deg  safety limit: max joint change per command (LeRobot's max_relative_target)
        """
        cameras = {}
        if camera_index is not None:
            from lerobot.cameras.opencv import OpenCVCameraConfig

            cameras["cam"] = OpenCVCameraConfig(
                index_or_path=camera_index, fps=camera_fps, width=camera_width, height=camera_height
            )

        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=arm_id,
                cameras=cameras,
                use_degrees=True,
                max_relative_target=max_step_deg,
            )
        )
        self.urdf_path = urdf_path
        self._kin = None
        self.poses: dict[str, dict[str, float]] = (
            json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else {}
        )

    # ------------------------------------------------------------------ connection
    def connect(self):
        self.robot.connect()  # runs calibration automatically the first time
        return self

    def disconnect(self):
        if self.robot.is_connected:
            self.robot.disconnect()  # turns torque off (arm goes limp!)

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    # ------------------------------------------------------------------ joints
    def get_joints(self) -> dict[str, float]:
        """Current joint positions, e.g. {'shoulder_pan': 3.2, ..., 'gripper': 40.0}."""
        obs = self.robot.get_observation()
        return {m: float(obs[f"{m}.pos"]) for m in MOTORS}

    def move_joints(self, duration: float = 1.5, hz: int = 50, **targets: float):
        """
        Smoothly move to joint targets over `duration` seconds.
        Only the joints you pass are moved, e.g. arm.move_joints(shoulder_pan=30, gripper=80)
        """
        unknown = set(targets) - set(MOTORS)
        if unknown:
            raise ValueError(f"Unknown joints {unknown}; valid: {MOTORS}")
        start = self.get_joints()
        goal = {**start, **targets}
        steps = max(1, int(duration * hz))
        for i in range(1, steps + 1):
            t = i / steps
            s = t * t * (3 - 2 * t)  # smoothstep easing
            self.robot.send_action({f"{m}.pos": start[m] + (goal[m] - start[m]) * s for m in MOTORS})
            time.sleep(1 / hz)

    # ------------------------------------------------------------------ gripper
    def set_gripper(self, percent_open: float, duration: float = 0.6):
        self.move_joints(duration=duration, gripper=float(np.clip(percent_open, 0, 100)))

    def open_gripper(self):
        self.set_gripper(100)

    def close_gripper(self, percent_open: float = 0):
        """Close on the dish. Use a value >0 if the plate is wide and you don't want to stall the motor."""
        self.set_gripper(percent_open)

    # ------------------------------------------------------------------ named poses (teach & repeat)
    def teach(self):
        """
        Teach mode: torque turns off so you can move the arm by hand.
        Type a name + Enter to save the current pose, or just Enter to finish.
        Poses are saved to poses.json and can be replayed with go_to("name").
        """
        self.robot.bus.disable_torque()
        print("Torque OFF - move the arm by hand. Type a pose name to save, empty line to quit.")
        try:
            while True:
                name = input("pose name> ").strip()
                if not name:
                    break
                self.poses[name] = self.get_joints()
                POSES_FILE.write_text(json.dumps(self.poses, indent=2))
                print(f"saved '{name}': {self.poses[name]}")
        finally:
            self.robot.bus.enable_torque()
            print("Torque ON")

    def go_to(self, pose_name: str, duration: float = 2.0, include_gripper: bool = False):
        pose = dict(self.poses[pose_name])
        if not include_gripper:
            pose.pop("gripper", None)
        self.move_joints(duration=duration, **pose)

    def home(self, duration: float = 2.0):
        """Go to the pose saved as 'home', or to all-zero (the calibration middle) if none saved."""
        if "home" in self.poses:
            self.go_to("home", duration)
        else:
            self.move_joints(duration=duration, **{m: 0.0 for m in MOTORS[:-1]})

    # ------------------------------------------------------------------ Cartesian (x, y, z)
    @property
    def kinematics(self):
        if self._kin is None:
            from lerobot.model.kinematics import RobotKinematics

            if not self.urdf_path or not Path(self.urdf_path).exists():
                raise FileNotFoundError(
                    f"URDF not found at {self.urdf_path!r}. Run: git clone https://github.com/TheRobotStudio/SO-ARM100"
                )
            self._kin = RobotKinematics(
                urdf_path=str(self.urdf_path), target_frame_name="gripper_frame_link", joint_names=MOTORS
            )
        return self._kin

    def get_pose(self) -> np.ndarray:
        """4x4 transform of the gripper in the base frame."""
        q = np.array(list(self.get_joints().values()))
        return self.kinematics.forward_kinematics(q)

    def get_xyz(self) -> tuple[float, float, float]:
        x, y, z = self.get_pose()[:3, 3]
        return float(x), float(y), float(z)

    def solve_ik(self, q: np.ndarray, target: np.ndarray, tol_m: float = 0.001) -> np.ndarray:
        """
        Iterate LeRobot's IK until the tip is within tol_m of the target.
        Pass 1 also tries to keep the gripper orientation; if that leaves >2 mm of
        position error (5-DOF arm can't do both), pass 2 solves for position only.
        """
        kin = self.kinematics
        err = lambda qq: np.linalg.norm(kin.forward_kinematics(qq)[:3, 3] - target[:3, 3])
        for ow in (0.001, 0.0):
            for _ in range(30):
                q = kin.inverse_kinematics(q, target, orientation_weight=ow)
                if err(q) < tol_m:
                    return q
            if err(q) < 0.002:
                return q
        return q

    def move_to_xyz(self, x: float, y: float, z: float, duration: float = 2.0):
        """
        Move the gripper tip to (x, y, z) metres, keeping its current orientation as closely
        as the 5-DOF arm allows. Tip: first go_to() a taught pose with the gripper pointing
        down, then use small xyz moves from there.
        """
        joints = self.get_joints()
        q = np.array(list(joints.values()))
        target = self.kinematics.forward_kinematics(q).copy()
        target[:3, 3] = [x, y, z]
        q_new = self.solve_ik(q, target)
        sol = {m: float(v) for m, v in zip(MOTORS, q_new) if m != "gripper"}
        self.move_joints(duration=duration, **sol)
        err = np.linalg.norm(np.array(self.get_xyz()) - np.array([x, y, z]))
        if err > 0.01:
            print(f"warning: ended {err * 1000:.0f} mm from target (out of reach?)")

    def move_relative(self, dx: float = 0, dy: float = 0, dz: float = 0, duration: float = 1.0):
        x, y, z = self.get_xyz()
        self.move_to_xyz(x + dx, y + dy, z + dz, duration)

    # ------------------------------------------------------------------ stacking helpers
    @staticmethod
    def stack_height(level: int, table_z: float = 0.0) -> float:
        """Z (metres) of the top of plate number `level` in a stack (level 0 = bottom plate)."""
        return table_z + (level + 1) * PLATE_HEIGHT_M

    # ------------------------------------------------------------------ camera
    def get_frame(self) -> np.ndarray:
        """Latest RGB camera frame as an (H, W, 3) numpy array."""
        obs = self.robot.get_observation()
        if "cam" not in obs:
            raise RuntimeError("No camera configured - pass camera_index=... to PetriArm()")
        return obs["cam"]

    def save_frame(self, path: str = "frame.png"):
        import cv2

        cv2.imwrite(path, cv2.cvtColor(self.get_frame(), cv2.COLOR_RGB2BGR))
        return path

    def find_dishes(self, min_radius_px: int = 40, max_radius_px: int = 200, show: bool = False):
        """
        Detect round petri dishes in the camera image with OpenCV's Hough circle transform.
        Returns a list of (x_px, y_px, radius_px), largest first. Tune the radii for your
        camera height. Converting pixels -> metres needs a one-time calibration (see README).
        """
        import cv2

        frame = self.get_frame()
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius_px * 2,
            param1=100, param2=40, minRadius=min_radius_px, maxRadius=max_radius_px,
        )
        found = [] if circles is None else sorted(
            [tuple(int(v) for v in c) for c in circles[0]], key=lambda c: -c[2]
        )
        if show:
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for x, y, r in found:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("dishes", vis)
            cv2.waitKey(1)
        return found


def pixel_to_table(px: float, py: float, calib: dict) -> tuple[float, float]:
    """
    Map a camera pixel to table (x, y) metres using a homography from 4+ known points.
    calib = {"pixels": [[u,v],...], "table_m": [[x,y],...]}  (make it with calibrate_camera.py)
    """
    import cv2

    H, _ = cv2.findHomography(np.array(calib["pixels"], float), np.array(calib["table_m"], float))
    p = H @ np.array([px, py, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])
