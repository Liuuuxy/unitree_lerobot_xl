"""Safe real-robot evaluation for G1 policies.

This keeps the same high-level flow as ``eval_g1.py`` while adding:
- dataset-aware action bounds
- per-step delta limits relative to current measured state
- gradual initialization to the reference episode start pose
- left/right-aware action routing based on dataset feature names
- saved safety logs and metrics
"""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from multiprocessing.sharedctypes import SynchronizedArray
from torch import nn

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device, init_logging

from unitree_lerobot.eval_robot.make_robot import setup_image_client, setup_robot_interface
from unitree_lerobot.eval_robot.robot_control.robot_arm import (
    G1_23_JointArmIndex,
    G1_29_JointArmIndex,
)
from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import (
    Brainco_Left_Hand_JointIndex,
    Brainco_Right_Hand_JointIndex,
)
from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import (
    Inspire_Left_Hand_JointIndex,
    Inspire_Right_Hand_JointIndex,
)
from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
    Dex3_1_Left_JointIndex,
    Dex3_1_Right_JointIndex,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import (
    EvalRealConfig,
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
)

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def canonicalize_joint_name(name: str) -> str:
    return name.replace("Wristyaw", "WristYaw")


def build_enum_index_map(enum_cls) -> dict[str, int]:
    return {canonicalize_joint_name(member.name): idx for idx, member in enumerate(enum_cls)}


ARM_NAME_TO_INDEX = {
    "G1_29": build_enum_index_map(G1_29_JointArmIndex),
    "G1_23": build_enum_index_map(G1_23_JointArmIndex),
}

EE_NAME_TO_INDEX = {
    "dex1": (
        {"kLeftGripper": 0},
        {"kRightGripper": 0},
    ),
    "dex3": (
        build_enum_index_map(Dex3_1_Left_JointIndex),
        build_enum_index_map(Dex3_1_Right_JointIndex),
    ),
    "inspire1": (
        build_enum_index_map(Inspire_Left_Hand_JointIndex),
        build_enum_index_map(Inspire_Right_Hand_JointIndex),
    ),
    "brainco": (
        build_enum_index_map(Brainco_Left_Hand_JointIndex),
        build_enum_index_map(Brainco_Right_Hand_JointIndex),
    ),
}


@dataclass
class EvalRealSafeConfig(EvalRealConfig):
    max_steps: int = 0
    init_from_dataset: bool = True

    safety_lower_stat: str = "q01"
    safety_upper_stat: str = "q99"
    safety_margin: float = 0.02

    safety_arm_delta_limit: float = 0.03
    safety_ee_delta_limit: float = 0.05

    safety_init_arm_delta_limit: float = 0.02
    safety_init_ee_delta_limit: float = 0.03
    safety_init_tolerance: float = 0.05
    safety_max_init_seconds: float = 12.0

    safety_log_every: int = 30


def flatten_names(raw_names: Any) -> list[str]:
    if raw_names is None:
        return []
    if isinstance(raw_names, str):
        return [raw_names]

    flattened: list[str] = []
    for item in raw_names:
        if isinstance(item, str):
            flattened.append(item)
        else:
            flattened.extend(flatten_names(item))
    return flattened


def get_meta_info(dataset: LeRobotDataset) -> dict[str, Any]:
    info = getattr(dataset.meta, "info", {})
    return info if isinstance(info, dict) else {}


def get_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    info = get_meta_info(dataset)
    feature_info = info.get("features", {}).get(key, {})
    names = None
    if isinstance(feature_info, dict):
        names = feature_info.get("names")

    if not names and hasattr(dataset, "features") and key in dataset.features:
        names = getattr(dataset.features[key], "names", None)

    return [canonicalize_joint_name(name) for name in flatten_names(names)]


def get_dataset_image_keys(dataset: LeRobotDataset) -> list[str]:
    info = get_meta_info(dataset)
    features = info.get("features", {})
    return sorted(key for key in features if key.startswith("observation.images."))


def get_policy_name(cfg: EvalRealSafeConfig) -> str:
    if cfg.policy is None or not cfg.policy.pretrained_path:
        return "random_policy"

    policy_path = Path(str(cfg.policy.pretrained_path))
    if policy_path.name == "pretrained_model" and len(policy_path.parents) >= 3:
        return policy_path.parents[2].name

    return policy_path.parent.name


def make_output_dir(cfg: EvalRealSafeConfig) -> Path:
    repo_name = cfg.repo_id.replace("/", "__")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.save_dir) / repo_name / get_policy_name(cfg) / f"real_robot_ep_{cfg.episodes:04d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_feature_names(
    cfg: EvalRealSafeConfig,
    state_names: list[str],
    action_names: list[str],
):
    arm_map = ARM_NAME_TO_INDEX[cfg.arm]
    left_ee_map, right_ee_map = EE_NAME_TO_INDEX.get(cfg.ee.lower(), ({}, {})) if cfg.ee else ({}, {})
    available_names = set(arm_map) | set(left_ee_map) | set(right_ee_map)

    unknown_state = sorted(set(state_names) - available_names)
    unknown_action = sorted(set(action_names) - available_names)
    if unknown_state:
        raise ValueError(f"Unsupported observation.state features for {cfg.arm}/{cfg.ee}: {unknown_state}")
    if unknown_action:
        raise ValueError(f"Unsupported action features for {cfg.arm}/{cfg.ee}: {unknown_action}")


def get_action_bounds(
    dataset: LeRobotDataset,
    action_names: list[str],
    lower_stat: str,
    upper_stat: str,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    stats = getattr(dataset.meta, "stats", {})
    action_stats = stats.get("action", {}) if isinstance(stats, dict) else {}
    dataset_action_names = get_feature_names(dataset, "action")

    if lower_stat not in action_stats or upper_stat not in action_stats:
        missing = [name for name in (lower_stat, upper_stat) if name not in action_stats]
        raise ValueError(f"Dataset action stats missing requested keys: {missing}")

    lower_values = np.asarray(action_stats[lower_stat], dtype=np.float64)
    upper_values = np.asarray(action_stats[upper_stat], dtype=np.float64)
    if lower_values.shape[0] != len(dataset_action_names) or upper_values.shape[0] != len(dataset_action_names):
        raise ValueError("Dataset action stats shape does not match action feature names.")

    lower_by_name = {name: float(value) for name, value in zip(dataset_action_names, lower_values, strict=True)}
    upper_by_name = {name: float(value) for name, value in zip(dataset_action_names, upper_values, strict=True)}

    lower = np.array([lower_by_name[name] - margin for name in action_names], dtype=np.float64)
    upper = np.array([upper_by_name[name] + margin for name in action_names], dtype=np.float64)
    return lower, upper


def read_ee_state(ee_shared_mem: dict[str, Any], ee_dof: int) -> tuple[np.ndarray, np.ndarray]:
    if not ee_shared_mem or ee_dof == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    with ee_shared_mem["lock"]:
        full_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)

    if full_state.size < 2 * ee_dof:
        raise ValueError(f"Expected {2 * ee_dof} end-effector state values, got {full_state.size}.")

    return full_state[:ee_dof].copy(), full_state[ee_dof : 2 * ee_dof].copy()


def build_current_state_map(
    cfg: EvalRealSafeConfig,
    current_arm_q: np.ndarray,
    left_ee_state: np.ndarray,
    right_ee_state: np.ndarray,
) -> dict[str, float]:
    state_map: dict[str, float] = {}

    arm_map = ARM_NAME_TO_INDEX[cfg.arm]
    for name, index in arm_map.items():
        if index < len(current_arm_q):
            state_map[name] = float(current_arm_q[index])

    if cfg.ee:
        # left_ee_map, right_ee_map = EE_NAME_TO_INDEX[cfg.ee.lower()]
        left_ee_map, right_ee_map = EE_NAME_TO_INDEX.get(cfg.ee.lower(), ({}, {}))
        for name, index in left_ee_map.items():
            if index < len(left_ee_state):
                state_map[name] = float(left_ee_state[index])
        for name, index in right_ee_map.items():
            if index < len(right_ee_state):
                state_map[name] = float(right_ee_state[index])

    return state_map


def read_live_observation(
    tv_img_array,
    wrist_img_array,
    tv_img_shape,
    wrist_img_shape,
    is_binocular: bool,
    has_wrist_cam: bool,
    arm_ctrl,
) -> tuple[dict[str, Any], np.ndarray]:
    current_tv_image = tv_img_array.copy()
    current_wrist_image = wrist_img_array.copy() if has_wrist_cam and wrist_img_array is not None else None

    left_top_cam = current_tv_image[:, : tv_img_shape[1] // 2] if is_binocular else current_tv_image
    right_top_cam = current_tv_image[:, tv_img_shape[1] // 2 :] if is_binocular else None
    left_top_cam = np.ascontiguousarray(left_top_cam)
    if right_top_cam is not None:
        right_top_cam = np.ascontiguousarray(right_top_cam)

    left_wrist_cam = right_wrist_cam = None
    if has_wrist_cam and current_wrist_image is not None and wrist_img_shape is not None:
        left_wrist_cam = np.ascontiguousarray(current_wrist_image[:, : wrist_img_shape[1] // 2])
        right_wrist_cam = np.ascontiguousarray(current_wrist_image[:, wrist_img_shape[1] // 2 :])

    observation = {
        "observation.images.cam_left_high": torch.from_numpy(left_top_cam),
        "observation.images.cam_right_high": torch.from_numpy(right_top_cam) if right_top_cam is not None else None,
        "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_cam) if left_wrist_cam is not None else None,
        "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_cam) if right_wrist_cam is not None else None,
    }
    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    return observation, current_arm_q


def build_state_tensor(
    state_names: list[str],
    current_state_map: dict[str, float],
) -> torch.Tensor:
    state_values = np.array([current_state_map[name] for name in state_names], dtype=np.float32)
    return torch.from_numpy(state_values)


def build_live_observation(
    dataset_image_keys: list[str],
    state_names: list[str],
    raw_observation: dict[str, Any],
    current_state_map: dict[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    observation = {}
    missing_image_keys = [key for key in dataset_image_keys if raw_observation.get(key) is None]
    if missing_image_keys:
        raise ValueError(
            "Live robot observation is missing image keys required by the dataset: "
            f"{missing_image_keys}. Check camera setup or use a dataset with matching cameras."
        )

    for key in dataset_image_keys:
        observation[key] = raw_observation[key]

    state_tensor = build_state_tensor(state_names, current_state_map)
    observation["observation.state"] = state_tensor
    return observation, state_tensor.numpy()


def set_ee_targets(ee_shared_mem: dict[str, Any], left_target: np.ndarray, right_target: np.ndarray):
    if not ee_shared_mem:
        return

    if isinstance(ee_shared_mem["left"], SynchronizedArray):
        ee_shared_mem["left"][:] = to_list(left_target)
        ee_shared_mem["right"][:] = to_list(right_target)
    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
        ee_shared_mem["left"].value = to_scalar(left_target)
        ee_shared_mem["right"].value = to_scalar(right_target)


class SafetyLimiter:
    def __init__(
        self,
        cfg: EvalRealSafeConfig,
        action_names: list[str],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        self.cfg = cfg
        self.action_names = action_names
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.arm_map = ARM_NAME_TO_INDEX[cfg.arm]
        self.left_ee_map, self.right_ee_map = EE_NAME_TO_INDEX.get(cfg.ee.lower(), ({}, {})) if cfg.ee else ({}, {})
        self.arm_names = set(self.arm_map)
        self.left_ee_names = set(self.left_ee_map)
        self.right_ee_names = set(self.right_ee_map)

        self.abs_clip_counts = np.zeros(len(action_names), dtype=np.int64)
        self.delta_clip_counts = np.zeros(len(action_names), dtype=np.int64)

    def _apply_named_targets(
        self,
        named_targets: dict[str, float],
        current_state_map: dict[str, float],
        current_arm_q: np.ndarray,
        current_left_ee: np.ndarray,
        current_right_ee: np.ndarray,
        arm_delta_limit: float,
        ee_delta_limit: float,
        update_counters: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        arm_target = current_arm_q.astype(np.float64, copy=True)
        left_target = current_left_ee.astype(np.float64, copy=True)
        right_target = current_right_ee.astype(np.float64, copy=True)

        safe_action = np.zeros(len(self.action_names), dtype=np.float64)
        abs_hits = 0
        delta_hits = 0

        for idx, name in enumerate(self.action_names):
            if name not in named_targets:
                safe_action[idx] = current_state_map.get(name, 0.0)
                continue

            raw_value = float(named_targets[name])

            bounded_value = float(np.clip(raw_value, self.lower_bounds[idx], self.upper_bounds[idx]))
            if not np.isclose(bounded_value, raw_value):
                abs_hits += 1
                if update_counters:
                    self.abs_clip_counts[idx] += 1

            current_value = current_state_map.get(name, bounded_value)
            delta_limit = arm_delta_limit if name in self.arm_names else ee_delta_limit
            if delta_limit > 0:
                delta_low = current_value - delta_limit
                delta_high = current_value + delta_limit
                safe_value = float(np.clip(bounded_value, delta_low, delta_high))
                if not np.isclose(safe_value, bounded_value):
                    delta_hits += 1
                    if update_counters:
                        self.delta_clip_counts[idx] += 1
            else:
                safe_value = bounded_value

            safe_action[idx] = safe_value

            if name in self.arm_map:
                arm_target[self.arm_map[name]] = safe_value
            elif name in self.left_ee_map:
                left_target[self.left_ee_map[name]] = safe_value
            elif name in self.right_ee_map:
                right_target[self.right_ee_map[name]] = safe_value

        return arm_target, left_target, right_target, {
            "safe_action": safe_action.astype(np.float32),
            "num_abs_clipped": abs_hits,
            "num_delta_clipped": delta_hits,
        }

    def apply_policy_action(
        self,
        raw_action: np.ndarray,
        current_state_map: dict[str, float],
        current_arm_q: np.ndarray,
        current_left_ee: np.ndarray,
        current_right_ee: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        named_targets = {name: float(value) for name, value in zip(self.action_names, raw_action, strict=True)}
        arm_target, left_target, right_target, info = self._apply_named_targets(
            named_targets=named_targets,
            current_state_map=current_state_map,
            current_arm_q=current_arm_q,
            current_left_ee=current_left_ee,
            current_right_ee=current_right_ee,
            arm_delta_limit=self.cfg.safety_arm_delta_limit,
            ee_delta_limit=self.cfg.safety_ee_delta_limit,
            update_counters=True,
        )
        return info["safe_action"], arm_target, left_target, right_target, info

    def apply_reference_state(
        self,
        target_state: dict[str, float],
        current_state_map: dict[str, float],
        current_arm_q: np.ndarray,
        current_left_ee: np.ndarray,
        current_right_ee: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arm_target, left_target, right_target, _ = self._apply_named_targets(
            named_targets=target_state,
            current_state_map=current_state_map,
            current_arm_q=current_arm_q,
            current_left_ee=current_left_ee,
            current_right_ee=current_right_ee,
            arm_delta_limit=self.cfg.safety_init_arm_delta_limit,
            ee_delta_limit=self.cfg.safety_init_ee_delta_limit,
            update_counters=False,
        )
        return arm_target, left_target, right_target

    def make_summary(self, raw_actions: np.ndarray, safe_actions: np.ndarray) -> dict[str, Any]:
        if raw_actions.size == 0 or safe_actions.size == 0:
            return {
                "num_steps": 0,
                "action_dim": len(self.action_names),
                "fraction_values_clipped": 0.0,
                "fraction_steps_with_clipping": 0.0,
                "mae_raw_to_safe": 0.0,
                "mse_raw_to_safe": 0.0,
                "rmse_raw_to_safe": 0.0,
                "max_abs_correction": 0.0,
                "abs_clip_counts_per_dim": self.abs_clip_counts.tolist(),
                "delta_clip_counts_per_dim": self.delta_clip_counts.tolist(),
            }

        diff = safe_actions - raw_actions
        abs_diff = np.abs(diff)
        sq_diff = diff**2
        clipped_mask = abs_diff > 1e-8

        return {
            "num_steps": int(raw_actions.shape[0]),
            "action_dim": int(raw_actions.shape[1]),
            "fraction_values_clipped": float(clipped_mask.mean()),
            "fraction_steps_with_clipping": float(clipped_mask.any(axis=1).mean()),
            "mae_raw_to_safe": float(abs_diff.mean()),
            "mse_raw_to_safe": float(sq_diff.mean()),
            "rmse_raw_to_safe": float(np.sqrt(sq_diff.mean())),
            "max_abs_correction": float(abs_diff.max()),
            "abs_clip_counts_per_dim": self.abs_clip_counts.tolist(),
            "delta_clip_counts_per_dim": self.delta_clip_counts.tolist(),
        }


def move_to_reference_pose(
    cfg: EvalRealSafeConfig,
    limiter: SafetyLimiter,
    robot_interface: dict[str, Any],
    reference_state: dict[str, float],
):
    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    ee_shared_mem = robot_interface["ee_shared_mem"]
    ee_dof = robot_interface["ee_dof"]

    logger_mp.info("Gradually moving robot toward the reference episode start pose.")
    start_time = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        current_arm_q = arm_ctrl.get_current_dual_arm_q()
        left_ee_state, right_ee_state = read_ee_state(ee_shared_mem, ee_dof)
        current_state_map = build_current_state_map(cfg, current_arm_q, left_ee_state, right_ee_state)

        arm_target, left_target, right_target = limiter.apply_reference_state(
            target_state=reference_state,
            current_state_map=current_state_map,
            current_arm_q=current_arm_q,
            current_left_ee=left_ee_state,
            current_right_ee=right_ee_state,
        )

        tau = arm_ik.solve_tau(arm_target)
        arm_ctrl.ctrl_dual_arm(arm_target, tau)
        set_ee_targets(ee_shared_mem, left_target, right_target)

        max_error = max(
            (abs(reference_state[name] - current_state_map.get(name, reference_state[name])) for name in reference_state),
            default=0.0,
        )
        if max_error <= cfg.safety_init_tolerance:
            logger_mp.info("Reference pose reached within tolerance %.4f.", cfg.safety_init_tolerance)
            return

        if (time.perf_counter() - start_time) >= cfg.safety_max_init_seconds:
            logger_mp.warning(
                "Stopped initialization after %.2fs with max remaining error %.4f.",
                cfg.safety_max_init_seconds,
                max_error,
            )
            return

        time.sleep(max(0.0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start)))


def build_reference_state(step: dict[str, Any], state_names: list[str]) -> dict[str, float]:
    state_values = step["observation.state"].detach().cpu().numpy().astype(np.float64)
    return {name: float(value) for name, value in zip(state_names, state_values, strict=True)}


def save_artifacts(
    output_dir: Path,
    cfg: EvalRealSafeConfig,
    action_names: list[str],
    state_names: list[str],
    image_keys: list[str],
    raw_actions: list[np.ndarray],
    safe_actions: list[np.ndarray],
    measured_states: list[np.ndarray],
    commanded_arm_targets: list[np.ndarray],
    commanded_left_ee: list[np.ndarray],
    commanded_right_ee: list[np.ndarray],
    limiter: SafetyLimiter,
):
    raw_action_arr = np.asarray(raw_actions, dtype=np.float32)
    safe_action_arr = np.asarray(safe_actions, dtype=np.float32)
    measured_state_arr = np.asarray(measured_states, dtype=np.float32)
    commanded_arm_arr = np.asarray(commanded_arm_targets, dtype=np.float32)
    commanded_left_arr = np.asarray(commanded_left_ee, dtype=np.float32)
    commanded_right_arr = np.asarray(commanded_right_ee, dtype=np.float32)

    summary = limiter.make_summary(raw_action_arr, safe_action_arr)
    summary.update(
        {
            "repo_id": cfg.repo_id,
            "episode_index": cfg.episodes,
            "policy_name": get_policy_name(cfg),
            "send_real_robot": cfg.send_real_robot,
            "safety_lower_stat": cfg.safety_lower_stat,
            "safety_upper_stat": cfg.safety_upper_stat,
            "safety_margin": cfg.safety_margin,
            "safety_arm_delta_limit": cfg.safety_arm_delta_limit,
            "safety_ee_delta_limit": cfg.safety_ee_delta_limit,
            "init_from_dataset": cfg.init_from_dataset,
        }
    )

    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "action_names": action_names,
                "state_names": state_names,
                "image_keys": image_keys,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.save(output_dir / "raw_policy_actions.npy", raw_action_arr)
    np.save(output_dir / "safe_policy_actions.npy", safe_action_arr)
    np.save(output_dir / "measured_states.npy", measured_state_arr)
    np.save(output_dir / "commanded_arm_targets.npy", commanded_arm_arr)
    np.save(output_dir / "commanded_left_ee.npy", commanded_left_arr)
    np.save(output_dir / "commanded_right_ee.npy", commanded_right_arr)

    logger_mp.info("Saved evaluation artifacts to %s", output_dir)
    logger_mp.info(
        "Safety summary: clipped %.2f%% of action values, step clip rate %.2f%%, raw->safe MAE %.6f",
        100.0 * summary["fraction_values_clipped"],
        100.0 * summary["fraction_steps_with_clipping"],
        summary["mae_raw_to_safe"],
    )


def eval_policy(
    cfg: EvalRealSafeConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn.Module."

    logger_mp.info("Arguments: %s", cfg)

    output_dir = make_output_dir(cfg)
    state_names = get_feature_names(dataset, "observation.state")
    action_names = get_feature_names(dataset, "action")
    image_keys = get_dataset_image_keys(dataset)
    validate_feature_names(cfg, state_names, action_names)
    lower_bounds, upper_bounds = get_action_bounds(
        dataset=dataset,
        action_names=action_names,
        lower_stat=cfg.safety_lower_stat,
        upper_stat=cfg.safety_upper_stat,
        margin=cfg.safety_margin,
    )
    limiter = SafetyLimiter(cfg, action_names, lower_bounds, upper_bounds)

    if cfg.visualization:
        rerun_logger = RerunLogger()

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_info = None
    robot_interface = None
    rerun_logger = None
    raw_actions: list[np.ndarray] = []
    safe_actions: list[np.ndarray] = []
    measured_states: list[np.ndarray] = []
    commanded_arm_targets: list[np.ndarray] = []
    commanded_left_ee: list[np.ndarray] = []
    commanded_right_ee: list[np.ndarray] = []

    try:
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]
        ee_dof = robot_interface["ee_dof"]

        reference_step = dataset[0]
        reference_state = build_reference_state(reference_step, state_names)
        task = reference_step["task"]

        if cfg.init_from_dataset and cfg.send_real_robot:
            move_to_reference_pose(cfg, limiter, robot_interface, reference_state)

        user_input = input(
            "Enter 's' to start safe real-robot evaluation "
            f"({'commanding' if cfg.send_real_robot else 'dry-run only'} mode): "
        )
        if user_input.lower() != "s":
            logger_mp.info("Evaluation cancelled by user before start.")
            return

        tv_img_array = image_info["tv_img_array"]
        wrist_img_array = image_info["wrist_img_array"]
        tv_img_shape = image_info["tv_img_shape"]
        wrist_img_shape = image_info["wrist_img_shape"]
        is_binocular = image_info["is_binocular"]
        has_wrist_cam = image_info["has_wrist_cam"]

        logger_mp.info(
            "Starting safe evaluation loop at %.2f Hz using bounds [%s, %s] + margin %.4f.",
            cfg.frequency,
            cfg.safety_lower_stat,
            cfg.safety_upper_stat,
            cfg.safety_margin,
        )
        if cfg.max_steps > 0:
            logger_mp.info("Loop will stop after %d steps.", cfg.max_steps)
        else:
            logger_mp.info("Loop will run until interrupted.")

        step_idx = 0
        while True:
            if cfg.max_steps > 0 and step_idx >= cfg.max_steps:
                break

            loop_start_time = time.perf_counter()
            raw_observation, current_arm_q = read_live_observation(
                tv_img_array,
                wrist_img_array,
                tv_img_shape,
                wrist_img_shape,
                is_binocular,
                has_wrist_cam,
                arm_ctrl,
            )

            left_ee_state, right_ee_state = read_ee_state(ee_shared_mem, ee_dof)
            current_state_map = build_current_state_map(cfg, current_arm_q, left_ee_state, right_ee_state)
            observation, state_vector = build_live_observation(
                dataset_image_keys=image_keys,
                state_names=state_names,
                raw_observation=raw_observation,
                current_state_map=current_state_map,
            )

            action = predict_action(
                observation=observation,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                use_dataset=False,
                robot_type=None,
            )
            raw_action = action.detach().cpu().numpy().astype(np.float32).flatten()
            safe_action, arm_target, left_target, right_target, clip_info = limiter.apply_policy_action(
                raw_action=raw_action,
                current_state_map=current_state_map,
                current_arm_q=current_arm_q,
                current_left_ee=left_ee_state,
                current_right_ee=right_ee_state,
            )

            if cfg.send_real_robot:
                tau = arm_ik.solve_tau(arm_target)
                arm_ctrl.ctrl_dual_arm(arm_target, tau)
                set_ee_targets(ee_shared_mem, left_target, right_target)

            raw_actions.append(raw_action.copy())
            safe_actions.append(safe_action.copy())
            measured_states.append(state_vector.copy())
            commanded_arm_targets.append(arm_target.astype(np.float32))
            commanded_left_ee.append(left_target.astype(np.float32))
            commanded_right_ee.append(right_target.astype(np.float32))

            if cfg.visualization:
                visualization_data(step_idx, observation, state_vector, safe_action, rerun_logger)

            if cfg.safety_log_every > 0 and (step_idx + 1) % cfg.safety_log_every == 0:
                logger_mp.info(
                    "Step %d: abs clipped=%d, delta clipped=%d, max raw->safe correction=%.5f",
                    step_idx + 1,
                    clip_info["num_abs_clipped"],
                    clip_info["num_delta_clipped"],
                    float(np.max(np.abs(safe_action - raw_action))) if raw_action.size else 0.0,
                )

            step_idx += 1
            time.sleep(max(0.0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except KeyboardInterrupt:
        logger_mp.info("Safe evaluation interrupted by user.")
    finally:
        save_artifacts(
            output_dir=output_dir,
            cfg=cfg,
            action_names=action_names,
            state_names=state_names,
            image_keys=image_keys,
            raw_actions=raw_actions,
            safe_actions=safe_actions,
            measured_states=measured_states,
            commanded_arm_targets=commanded_arm_targets,
            commanded_left_ee=commanded_left_ee,
            commanded_right_ee=commanded_right_ee,
            limiter=limiter,
        )
        if image_info:
            cleanup_resources(image_info)
        if robot_interface:
            cleanup_resources(robot_interface)


@parser.wrap()
def eval_main(cfg: EvalRealSafeConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset_root = cfg.root if cfg.root else None
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=dataset_root, episodes=[cfg.episodes])

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of safe eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
