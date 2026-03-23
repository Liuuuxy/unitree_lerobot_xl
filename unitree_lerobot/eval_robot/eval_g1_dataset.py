"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import torch
import tqdm
import logging
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from typing import Any
from dataclasses import asdict
from pathlib import Path
from torch import nn
from contextlib import nullcontext
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def make_eval_output_dir(cfg: EvalRealConfig) -> Path:
    policy_name = "random_policy"
    if cfg.policy is not None and cfg.policy.pretrained_path:
        policy_path = Path(str(cfg.policy.pretrained_path))
        if policy_path.name == "pretrained_model" and len(policy_path.parents) >= 3:
            policy_name = policy_path.parents[2].name
        else:
            policy_name = policy_path.parent.name

    repo_name = cfg.repo_id.replace("/", "__")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.save_dir) / repo_name / policy_name / f"episode_{cfg.episodes:04d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compute_action_metrics(
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
) -> dict[str, float | list[float] | int]:
    diff = predicted_actions - ground_truth_actions
    abs_diff = np.abs(diff)
    sq_diff = diff**2

    return {
        "num_steps": int(ground_truth_actions.shape[0]),
        "action_dim": int(ground_truth_actions.shape[1]),
        "mae": float(abs_diff.mean()),
        "mse": float(sq_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "max_abs_error": float(abs_diff.max()),
        "mae_per_dim": abs_diff.mean(axis=0).tolist(),
        "mse_per_dim": sq_diff.mean(axis=0).tolist(),
        "rmse_per_dim": np.sqrt(sq_diff.mean(axis=0)).tolist(),
    }


def save_eval_artifacts(
    output_dir: Path,
    episode_index: int,
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
    metrics: dict[str, float | list[float] | int],
) -> dict[str, str]:
    gt_path = output_dir / f"episode_{episode_index:04d}_ground_truth_actions.npy"
    pred_path = output_dir / f"episode_{episode_index:04d}_predicted_actions.npy"
    metrics_path = output_dir / f"episode_{episode_index:04d}_metrics.json"
    figure_path = output_dir / f"episode_{episode_index:04d}_action_comparison.png"

    np.save(gt_path, ground_truth_actions)
    np.save(pred_path, predicted_actions)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "ground_truth_actions": str(gt_path),
        "predicted_actions": str(pred_path),
        "metrics": str(metrics_path),
        "figure": str(figure_path),
    }


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    output_dir = make_eval_output_dir(cfg)
    logger_mp.info(f"Saving eval artifacts to: {output_dir}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    if len(dataset) == 0:
        raise ValueError(f"Episode {cfg.episodes} produced an empty dataset selection.")

    step = dataset[0]

    ground_truth_actions = []
    predicted_actions = []

    if cfg.send_real_robot:
        from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

        robot_interface = setup_robot_interface(cfg)
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

    # ===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        if cfg.send_real_robot:
            # Initialize robot to starting pose
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)

            time.sleep(1)

        for step_idx in tqdm.tqdm(range(len(dataset))):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if cfg.send_real_robot:
                # Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                # logger_mp.info(f"Arm Action: {arm_action}")

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)
        metrics = compute_action_metrics(ground_truth_actions, predicted_actions)
        saved_paths = save_eval_artifacts(
            output_dir=output_dir,
            episode_index=cfg.episodes,
            ground_truth_actions=ground_truth_actions,
            predicted_actions=predicted_actions,
            metrics=metrics,
        )

        # Get the number of timesteps and action dimensions
        n_timesteps, n_dims = ground_truth_actions.shape

        # Create a figure with subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        fig.suptitle("Ground Truth vs Predicted Actions")

        # Plot each dimension
        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
            ax.plot(predicted_actions[:, i], label="Predicted", color="red", linestyle="--")
            ax.set_ylabel(f"Dim {i + 1}")
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        # plt.show()

        time.sleep(1)
        plt.savefig(saved_paths["figure"])
        plt.close(fig)

        logger_mp.info("Evaluation metrics:")
        logger_mp.info(
            "  MAE=%.6f MSE=%.6f RMSE=%.6f MaxAbsErr=%.6f",
            metrics["mae"],
            metrics["mse"],
            metrics["rmse"],
            metrics["max_abs_error"],
        )
        logger_mp.info("Saved files:")
        logger_mp.info("  ground truth actions: %s", saved_paths["ground_truth_actions"])
        logger_mp.info("  predicted actions: %s", saved_paths["predicted_actions"])
        logger_mp.info("  metrics json: %s", saved_paths["metrics"])
        logger_mp.info("  comparison figure: %s", saved_paths["figure"])


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
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

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
