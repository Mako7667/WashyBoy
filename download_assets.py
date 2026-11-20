"""
download_assets.py - fetch the files that are not stored in this repo.

  * SO-101 URDF + meshes (for move_to_xyz)  from github.com/TheRobotStudio/SO-ARM100
  * Example SO-101 training data            from huggingface.co/datasets/lerobot/svla_so101_pickplace

Usage:  python download_assets.py
"""

import json
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent


def fetch_urdf():
    api = "https://api.github.com/repos/TheRobotStudio/SO-ARM100/contents/Simulation/SO101"
    out = ROOT / "SO-ARM100" / "Simulation" / "SO101"
    for sub in ("", "/assets"):
        req = urllib.request.Request(api + sub, headers={"User-Agent": "petri-arm-setup"})
        for item in json.load(urllib.request.urlopen(req)):
            if item["type"] != "file":
                continue
            dest = out / item["path"].removeprefix("Simulation/SO101/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                urllib.request.urlretrieve(item["download_url"], dest)
    print("URDF ready:", out / "so101_new_calib.urdf")


def fetch_dataset():
    path = snapshot_download(
        "lerobot/svla_so101_pickplace", repo_type="dataset", local_dir=ROOT / "data" / "svla_so101_pickplace"
    )
    print("Dataset ready:", path)


if __name__ == "__main__":
    fetch_urdf()
    fetch_dataset()
