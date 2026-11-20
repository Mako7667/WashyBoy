"""
recorder.py - save robot runs as a LeRobot dataset (same format as `lerobot-record`).

Used by autonomous.py --record: every time the camera-driven script successfully picks,
cleans and stacks a plate, that run is saved as one training episode (camera video +
joint states + joint commands). Failed runs are thrown away. The result can be used to
train a camera-based policy with `lerobot-train` (see docs/TRAINING.md).
"""

from __future__ import annotations

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import hw_to_dataset_features
from lerobot.utils.feature_utils import build_dataset_frame

DEFAULT_TASK = "Pick up the petri dish plate, clean it, and place it on the clean stack."


class EpisodeRecorder:
    def __init__(self, observation_features: dict, action_features: dict, root: str | Path,
                 repo_id: str = "local/washyboy_pick_clean", fps: int = 30, task: str = DEFAULT_TASK):
        self.root, self.task, self.fps = Path(root), task, fps
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root, image_writer_threads=4)
        else:
            features = {
                **hw_to_dataset_features(observation_features, "observation", use_video=True),
                **hw_to_dataset_features(action_features, "action", use_video=True),
            }
            self.ds = LeRobotDataset.create(repo_id, fps=fps, features=features, root=self.root,
                                            robot_type="so101_follower", use_videos=True, image_writer_threads=4)
        self.frames = 0

    def add(self, observation: dict, action: dict):
        frame = {
            **build_dataset_frame(self.ds.features, observation, "observation"),
            **build_dataset_frame(self.ds.features, action, "action"),
            "task": self.task,
        }
        self.ds.add_frame(frame)
        self.frames += 1

    def save_episode(self):
        if self.frames:
            self.ds.save_episode()
        self.frames = 0

    def discard_episode(self):
        self.ds.clear_episode_buffer()
        self.frames = 0

    def close(self):
        self.ds.finalize()

    @property
    def num_episodes(self) -> int:
        return self.ds.meta.total_episodes
