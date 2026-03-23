"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_left_only \
    --robot_type Unitree_G1_Dex3 \
    --side left
"""

import os
import cv2
import tqdm
import tyro
import json
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS, RobotConfig


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_config: RobotConfig) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_config is not None, "Robot config cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"
        self.json_state_data_name = robot_config.json_state_data_name
        self.json_action_data_name = robot_config.json_action_data_name
        self.camera_to_image_key = robot_config.camera_to_image_key

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.available_cameras = self._infer_available_cameras()

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []
        root_path = Path(self.data_dirs)
        if not root_path.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {root_path}")

        # Support both layouts:
        # 1) raw_dir/episode_xxxx/data.json
        # 2) raw_dir/task_name/episode_xxxx/data.json
        for candidate in sorted(root_path.iterdir()):
            if not candidate.is_dir():
                continue

            if (candidate / self.json_file).is_file():
                self.episode_paths.append(str(candidate))
                continue

            nested_episode_paths = [
                str(subdir)
                for subdir in sorted(candidate.iterdir())
                if subdir.is_dir() and (subdir / self.json_file).is_file()
            ]
            if nested_episode_paths:
                self.task_paths.append(str(candidate))
                self.episode_paths.extend(nested_episode_paths)

        if not self.episode_paths:
            raise FileNotFoundError(
                f"No episode directories containing '{self.json_file}' were found under: {root_path}"
            )

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> list:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _infer_available_cameras(self) -> list[str]:
        """Infer available mapped camera names from cached JSON episodes."""
        available = set()
        for episode_data in self.episodes_data_cached:
            for sample_data in episode_data.get("data", []):
                for camera_key in sample_data.get("colors", {}).keys():
                    mapped_key = self.camera_to_image_key.get(camera_key)
                    if mapped_key:
                        available.add(mapped_key)

        available_list = sorted(available)
        if available_list:
            print(f"==> Detected cameras in JSON: {available_list}")
        else:
            print("==> Warning: No mapped color cameras detected in JSON.")
        return available_list

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height = cam_width = None
        if cameras:
            cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    robot_config: RobotConfig | None = None,
    cameras: list[str] | None = None,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    resolved_robot_config = robot_config if robot_config is not None else ROBOT_CONFIGS[robot_type]
    motors = resolved_robot_config.motors
    cameras = cameras if cameras is not None else resolved_robot_config.cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    json_dataset: JsonDataset,
) -> LeRobotDataset:
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def select_robot_config(robot_type: str, side: Literal["both", "left", "right"]) -> RobotConfig:
    base_config = ROBOT_CONFIGS[robot_type]
    if side == "both":
        return base_config

    side_title = side.title()
    side_prefix = f"{side}_"
    side_camera_prefix = f"cam_{side}_"

    motors = [motor for motor in base_config.motors if motor.startswith(f"k{side_title}")]
    cameras = [camera for camera in base_config.cameras if camera.startswith(side_camera_prefix)]
    camera_to_image_key = {key: val for key, val in base_config.camera_to_image_key.items() if val in cameras}
    json_state_data_name = [part for part in base_config.json_state_data_name if part.startswith(side_prefix)]
    json_action_data_name = [part for part in base_config.json_action_data_name if part.startswith(side_prefix)]

    if not motors:
        raise ValueError(
            f"Robot type '{robot_type}' does not expose any motors for side='{side}'. "
            "Use side='both' or update the robot config."
        )
    if not json_state_data_name or not json_action_data_name:
        raise ValueError(
            f"Robot type '{robot_type}' does not expose any state/action parts for side='{side}'. "
            "Use side='both' or update the robot config."
        )

    if not cameras:
        print(f"==> Warning: No cameras matched side='{side}'. Continuing with state/action only.")

    print(
        "==> Using side filter:",
        {
            "side": side,
            "motors": motors,
            "state_parts": json_state_data_name,
            "action_parts": json_action_data_name,
            "cameras": cameras,
        },
    )

    return RobotConfig(
        motors=motors,
        cameras=cameras,
        camera_to_image_key=camera_to_image_key,
        json_state_data_name=json_state_data_name,
        json_action_data_name=json_action_data_name,
    )


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    side: Literal["both", "left", "right"] = "both",
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    selected_robot_config = select_robot_config(robot_type, side)
    json_dataset = JsonDataset(raw_dir, selected_robot_config)
    target_cameras = json_dataset.available_cameras or selected_robot_config.cameras

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        robot_config=selected_robot_config,
        cameras=target_cameras,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        json_dataset,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
