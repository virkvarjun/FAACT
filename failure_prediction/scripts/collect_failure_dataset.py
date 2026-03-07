#!/usr/bin/env python
"""Collect failure prediction training data by running ACT rollouts.

Loads a trained ACT checkpoint, runs evaluation rollouts, and logs per-step
data including observations, action chunks, ACT embeddings, and outcomes.

Example:
    python -m failure_prediction.scripts.collect_failure_dataset \
        --checkpoint /path/to/checkpoints/100000/pretrained_model \
        --task AlohaTransferCube-v0 \
        --num_episodes 200 \
        --output_dir failure_dataset/transfer_cube \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add the Research directory to path so we can import failure_prediction
SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.utils.failure_dataset_logger import FailureDatasetLogger
from failure_prediction.utils.success_inference import infer_episode_outcome


def parse_args():
    p = argparse.ArgumentParser(description="Collect failure prediction dataset from ACT rollouts")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to pretrained_model directory")
    p.add_argument("--task", type=str, default="AlohaTransferCube-v0",
                    help="Gymnasium task ID")
    p.add_argument("--env_type", type=str, default="aloha",
                    help="Environment type (aloha, pusht, etc.)")
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--output_dir", type=str, default="failure_dataset")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=None,
                    help="Override max episode steps (default: env default)")
    p.add_argument("--failure_horizon", type=int, default=10,
                    help="K for failure_within_k labels")
    p.add_argument("--save_images", action="store_true", default=False)
    p.add_argument("--save_embeddings", action="store_true", default=True)
    p.add_argument("--no_save_embeddings", dest="save_embeddings", action="store_false")
    p.add_argument("--save_action_chunks", action="store_true", default=True)
    p.add_argument("--no_save_action_chunks", dest="save_action_chunks", action="store_false")
    p.add_argument("--dataset_name", type=str, default=None,
                    help="Optional dataset name for metadata")
    p.add_argument("--perturbation_mode", type=str, default="none",
                    choices=["none", "obs_noise", "action_noise"],
                    help="Perturbation mode (placeholder for future experiments)")
    return p.parse_args()


def make_single_env(task: str, env_type: str, max_steps: int | None = None):
    """Create a single (non-vectorized) gym environment."""
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
    }
    if max_steps is not None:
        gym_kwargs["max_episode_steps"] = max_steps

    gym_id = f"gym_{env_type}/{task}"
    env = gym.make(gym_id, **gym_kwargs)
    return env


def load_policy_and_processors(checkpoint_path: str, device: str):
    """Load ACT policy and its pre/post processors from a checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=checkpoint_path)
    policy.to(device)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": device},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    return policy, preprocessor, postprocessor


def preprocess_obs(obs: dict) -> dict[str, torch.Tensor]:
    """Convert raw env observation to policy-compatible tensor dict."""
    result = {}

    if "pixels" in obs:
        if isinstance(obs["pixels"], dict):
            for key, img in obs["pixels"].items():
                img_t = torch.from_numpy(img)
                if img_t.ndim == 3:
                    img_t = img_t.unsqueeze(0)
                img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
                result[f"observation.images.{key}"] = img_t
        else:
            img_t = torch.from_numpy(obs["pixels"])
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
            result["observation.image"] = img_t

    if "agent_pos" in obs:
        state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)
        result["observation.state"] = state

    return result


def features_to_numpy(features: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert model features dict from GPU tensors to CPU numpy arrays.

    Reduces high-dimensional features to manageable sizes:
    - latent_sample: kept as-is (B, latent_dim)
    - encoder_out: first token only (B, dim_model) - the latent encoding
    - decoder_out: mean-pooled over chunk dim (B, dim_model)
    """
    result = {}
    for key, val in features.items():
        v = val.detach().cpu()
        if key == "encoder_out":
            result["encoder_latent_token"] = v[:, 0, :].squeeze(0).numpy()
        elif key == "decoder_out":
            result["decoder_mean"] = v.mean(dim=1).squeeze(0).numpy()
        elif key == "latent_sample":
            result["latent_sample"] = v.squeeze(0).numpy()
        else:
            result[key] = v.squeeze(0).numpy()
    return result


def run_collection(args):
    """Main collection loop."""
    logger.info(f"Loading policy from {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.checkpoint, args.device
    )

    chunk_size = policy.config.chunk_size
    n_action_steps = policy.config.n_action_steps
    logger.info(f"Policy loaded: chunk_size={chunk_size}, n_action_steps={n_action_steps}")

    logger.info(f"Creating environment: {args.env_type}/{args.task}")
    env = make_single_env(args.task, args.env_type, args.max_steps)

    dataset_logger = FailureDatasetLogger(
        output_dir=args.output_dir,
        save_embeddings=args.save_embeddings,
        save_action_chunks=args.save_action_chunks,
        save_images=args.save_images,
    )

    collection_meta = {
        "checkpoint_path": args.checkpoint,
        "task": args.task,
        "env_type": args.env_type,
        "num_episodes": args.num_episodes,
        "device": args.device,
        "seed": args.seed,
        "failure_horizon": args.failure_horizon,
        "save_embeddings": args.save_embeddings,
        "save_action_chunks": args.save_action_chunks,
        "save_images": args.save_images,
        "perturbation_mode": args.perturbation_mode,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "dataset_name": args.dataset_name,
        "collection_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    all_successes = []
    all_episode_lengths = []

    for ep_idx in trange(args.num_episodes, desc="Collecting episodes"):
        ep_seed = args.seed + ep_idx

        dataset_logger.start_episode(
            episode_id=ep_idx,
            checkpoint_path=args.checkpoint,
            task_name=args.task,
            seed=ep_seed,
        )

        policy.reset()
        raw_obs, info = env.reset(seed=ep_seed)

        episode_rewards = []
        episode_successes = []
        episode_dones = []
        episode_terminated = []
        episode_truncated = []

        current_chunk = None
        current_features = None
        chunk_step_idx = 0

        max_ep_steps = env.spec.max_episode_steps or 400
        if args.max_steps:
            max_ep_steps = args.max_steps

        done = False
        step = 0

        while not done and step < max_ep_steps:
            obs_dict = preprocess_obs(raw_obs)
            obs_processed = preprocessor(obs_dict)

            need_new_chunk = (current_chunk is None) or (chunk_step_idx >= n_action_steps)

            if need_new_chunk:
                with torch.inference_mode():
                    action_chunk, features = policy.predict_action_chunk_with_features(obs_processed)
                current_chunk = action_chunk
                current_features = features_to_numpy(features) if args.save_embeddings else None
                chunk_step_idx = 0
                new_chunk = True
            else:
                new_chunk = False

            action = current_chunk[:, chunk_step_idx]
            action = postprocessor(action)

            action_np = action.detach().cpu().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]

            raw_obs, reward, terminated, truncated, info = env.step(action_np)

            success_this_step = False
            if "is_success" in info:
                success_this_step = bool(info["is_success"])

            done = terminated or truncated

            obs_state = None
            if "agent_pos" in (raw_obs if isinstance(raw_obs, dict) else {}):
                obs_state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)

            chunk_np = None
            if args.save_action_chunks and new_chunk:
                chunk_np = current_chunk.detach().cpu().numpy()
                if chunk_np.ndim == 3:
                    chunk_np = chunk_np[0]

            dataset_logger.log_step(
                timestep=step,
                executed_action=action_np,
                reward=float(reward),
                done=done,
                success=success_this_step,
                obs_state=obs_state,
                predicted_action_chunk=chunk_np,
                chunk_step_idx=chunk_step_idx,
                new_chunk_generated=new_chunk,
                features=current_features if new_chunk else None,
            )

            episode_rewards.append(float(reward))
            episode_successes.append(success_this_step)
            episode_dones.append(done)
            episode_terminated.append(terminated)
            episode_truncated.append(truncated)

            chunk_step_idx += 1
            step += 1

        outcome = infer_episode_outcome(
            rewards=np.array(episode_rewards),
            successes=np.array(episode_successes),
            dones=np.array(episode_dones),
            terminated=np.array(episode_terminated),
            truncated=np.array(episode_truncated),
            env_name=args.task,
        )

        dataset_logger.end_episode(
            success=outcome["success"],
            termination_reason=outcome["termination_reason"],
        )
        dataset_logger.save_episode()

        all_successes.append(outcome["success"])
        all_episode_lengths.append(step)

    env.close()

    collection_meta["collection_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    collection_meta["total_episodes"] = len(all_successes)
    collection_meta["total_successes"] = sum(all_successes)
    collection_meta["total_failures"] = len(all_successes) - sum(all_successes)
    collection_meta["success_rate"] = float(np.mean(all_successes))
    collection_meta["avg_episode_length"] = float(np.mean(all_episode_lengths))

    meta_path = Path(args.output_dir) / "collection_meta.json"
    with open(meta_path, "w") as f:
        json.dump(collection_meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    logger.info(f"  Episodes:       {len(all_successes)}")
    logger.info(f"  Successes:      {sum(all_successes)}")
    logger.info(f"  Failures:       {len(all_successes) - sum(all_successes)}")
    logger.info(f"  Success rate:   {np.mean(all_successes) * 100:.1f}%")
    logger.info(f"  Avg ep length:  {np.mean(all_episode_lengths):.1f}")
    logger.info(f"  Output dir:     {args.output_dir}")
    logger.info(f"  Metadata:       {meta_path}")

    if sum(all_successes) == len(all_successes):
        logger.warning("No failures collected! Consider increasing perturbation or using harder tasks.")
    if sum(all_successes) == 0:
        logger.warning("No successes collected! Check if the policy checkpoint is valid.")

    return collection_meta


if __name__ == "__main__":
    args = parse_args()
    run_collection(args)
