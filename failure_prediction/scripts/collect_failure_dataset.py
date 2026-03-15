#!/usr/bin/env python
"""Collect failure prediction training data by running policy rollouts (ACT or π₀).

Loads a trained checkpoint, runs evaluation rollouts, and logs per-step
data including observations, action chunks, embeddings, and outcomes.

Supports --policy_type act|pi0 for failure-aware wrapper around ACT or π₀ (PI0).

Example (ACT):
    python -m failure_prediction.scripts.collect_failure_dataset \
        --checkpoint /path/to/act_checkpoint/pretrained_model \
        --task AlohaTransferCube-v0 --policy_type act \
        --num_episodes 200 --output_dir failure_dataset/transfer_cube

Example (π₀):
    python -m failure_prediction.scripts.collect_failure_dataset \
        --checkpoint lerobot/pi0_base --policy_type pi0 \
        --task AlohaTransferCube-v0 --task_desc "transfer the cube to the target" \
        --num_episodes 200 --output_dir failure_dataset/pi0_transfer_cube
"""

from __future__ import annotations

import argparse
import importlib
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


def extract_obs_images(obs: dict) -> dict[str, np.ndarray]:
    """Extract pixels from env obs. Handles dict (multi-cam) or single array."""
    images = {}
    raw_pixels = obs.get("pixels")
    if isinstance(raw_pixels, dict):
        for key, value in raw_pixels.items():
            images[str(key)] = np.asarray(value)
    elif raw_pixels is not None:
        images["main"] = np.asarray(raw_pixels)
    return images


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
    p.add_argument("--policy_type", type=str, default="act", choices=["act", "pi0"],
                    help="Base policy: act (LeRobot ACT) or pi0 (π₀)")
    p.add_argument("--task_desc", type=str, default="",
                    help="Language task for π₀ (e.g. 'transfer the cube to the target'). Required for pi0.")
    return p.parse_args()


# Env: gym_aloha/AlohaTransferCube-v0 etc.
def make_single_env(task: str, env_type: str, max_steps: int | None = None):
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
    }
    if max_steps is not None:
        gym_kwargs["max_episode_steps"] = max_steps

    gym_id = f"gym_{env_type}/{task}"
    env = gym.make(gym_id, **gym_kwargs)
    return env


def load_act_policy_and_processors(checkpoint_path: str, device: str):
    """Load ACTPolicy from checkpoint."""
    from pathlib import Path

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. "
            "If trained on RunPod, copy the checkpoint locally or run this script on RunPod."
        )
    ckpt_str = str(path)
    policy = ACTPolicy.from_pretrained(
        pretrained_name_or_path=ckpt_str,
        local_files_only=True,
    )
    policy.to(device)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_str,
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy, preprocessor, postprocessor


def load_pi0_policy_and_processors(checkpoint_path: str, device: str):
    """Load PI0Policy from checkpoint. π₀ needs task string for language conditioning."""
    from pathlib import Path

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    path = Path(checkpoint_path).resolve()
    ckpt_str = str(path) if path.exists() else checkpoint_path
    policy = PI0Policy.from_pretrained(
        pretrained_name_or_path=ckpt_str,
        local_files_only=path.exists(),
    )
    policy.to(device)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_str,
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy, preprocessor, postprocessor


def load_policy_and_processors(checkpoint_path: str, device: str, policy_type: str = "act"):
    """Load policy and processors by type (act or pi0)."""
    if policy_type == "act":
        return load_act_policy_and_processors(checkpoint_path, device)
    if policy_type == "pi0":
        return load_pi0_policy_and_processors(checkpoint_path, device)
    raise ValueError(f"Unknown policy_type: {policy_type}")


# Env obs -> policy format: pixels -> (1,C,H,W) float [0,1], agent_pos -> state
def preprocess_obs(obs: dict, task_desc: str | None = None, policy_type: str = "act"):
    """Build obs dict for ACT, or transition for π₀ (with task in complementary_data)."""
    result = {}

    if "pixels" in obs:
        if isinstance(obs["pixels"], dict):
            for key, img in obs["pixels"].items():
                img_t = torch.from_numpy(np.asarray(img))
                if img_t.ndim == 3:
                    img_t = img_t.unsqueeze(0)
                img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
                result[f"observation.images.{key}"] = img_t
        else:
            img_t = torch.from_numpy(np.asarray(obs["pixels"]))
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
            result["observation.images.main"] = img_t

    if "agent_pos" in obs:
        state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)
        result["observation.state"] = state

    if policy_type == "pi0" and task_desc:
        from lerobot.processor.converters import create_transition
        task = task_desc if task_desc.endswith("\n") else f"{task_desc}\n"
        return create_transition(observation=result, complementary_data={"task": task})
    return result


def _predict_act_with_features(policy, obs_processed):
    if hasattr(policy, "predict_action_chunk_with_features"):
        return policy.predict_action_chunk_with_features(obs_processed)
    # Fallback 1: model supports return_features=True (custom lerobot fork)
    from lerobot.utils.constants import OBS_IMAGES

    # ACT expects OBS_IMAGES as list of tensors; config lists which keys to use
    batch = dict(obs_processed)
    if getattr(policy, "config", None) and getattr(policy.config, "image_features", None):
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    with torch.inference_mode():
        try:
            out = policy.model(batch, return_features=True)
        except TypeError:
            pass  # Fall through to fallback 2
        else:
            actions = out[0]
            features = out[2] if len(out) >= 3 else {}
            return actions, features
        # Fallback 2: hooks on encoder/decoder output; works with vanilla lerobot
        captured = {}

        def make_hook(name):
            def hook(_m, _in, out):
                captured[name] = out.detach()

            return hook

        handles = []
        if hasattr(policy.model, "encoder"):
            handles.append(policy.model.encoder.register_forward_hook(make_hook("encoder_out")))
        if hasattr(policy.model, "decoder"):
            handles.append(policy.model.decoder.register_forward_hook(make_hook("decoder_out")))

        actions = policy.model(batch)[0]
        for h in handles:
            h.remove()
        batch_size = actions.shape[0]
        cfg = getattr(policy.model, "config", policy.config)
        latent_dim = getattr(cfg, "latent_dim", 32)
        dim_model = getattr(cfg, "dim_model", 512)

        def _t(x):
            if x.dim() == 3:
                return x.transpose(0, 1)  # (S,B,C) -> (B,S,C) to match expected layout
            return x

        enc = captured.get("encoder_out")
        dec = captured.get("decoder_out")
        # latent_sample zeros when no VAE; encoder/decoder from hooks
        features = {
            "latent_sample": torch.zeros(batch_size, latent_dim, device=actions.device, dtype=actions.dtype),
            "encoder_out": _t(enc) if enc is not None else torch.zeros(batch_size, 1, dim_model, device=actions.device, dtype=actions.dtype),
            "decoder_out": _t(dec) if dec is not None else torch.zeros(batch_size, actions.shape[1], dim_model, device=actions.device, dtype=actions.dtype),
        }
        return actions, features


def _predict_pi0_with_features(policy, obs_processed):
    """π₀: predict chunk and use action_chunk_mean as feature (no internal embeddings exposed)."""
    with torch.inference_mode():
        actions = policy.predict_action_chunk(obs_processed)
    n_steps = getattr(policy.config, "n_action_steps", actions.shape[1])
    actions = actions[:, :n_steps]
    # Mean over chunk dim -> fixed-size vector for failure predictor
    mean_vec = actions.mean(dim=1)
    features = {"action_chunk_mean": mean_vec}
    return actions, features


def predict_action_chunk_with_features(policy, obs_processed, policy_type: str = "act"):
    """Get actions and features for failure prediction. Dispatches by policy type."""
    if policy_type == "pi0":
        return _predict_pi0_with_features(policy, obs_processed)
    return _predict_act_with_features(policy, obs_processed)


# encoder_out -> first token only; decoder_out -> mean over chunk. action_chunk_mean for π₀.
def features_to_numpy(features: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    result = {}
    for key, val in features.items():
        v = val.detach().cpu()
        if key == "encoder_out":
            result["encoder_latent_token"] = v[:, 0, :].squeeze(0).numpy()
        elif key == "decoder_out":
            result["decoder_mean"] = v.mean(dim=1).squeeze(0).numpy()
        elif key == "latent_sample":
            result["latent_sample"] = v.squeeze(0).numpy()
        elif key == "action_chunk_mean":
            result["action_chunk_mean"] = v.squeeze(0).numpy()  # π₀ fallback
        else:
            result[key] = v.squeeze(0).numpy()
    return result


def _default_task_desc(task_id: str) -> str:
    """Default language task for π₀ from gym task ID."""
    lower = task_id.lower()
    if "transfer" in lower and "cube" in lower:
        return "transfer the cube to the target"
    if "aloha" in lower:
        return "pick up the object and complete the task"
    return "complete the task"


# Main loop: load policy, run episodes, log features + outcomes per step to raw/episode_*.npz
def run_collection(args):
    if args.policy_type == "pi0" and not args.task_desc:
        args.task_desc = _default_task_desc(args.task)
        logger.info(f"Using default task_desc for π₀: {args.task_desc!r}")

    pkg = f"gym_{args.env_type}"
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Environment package '{pkg}' not found. Install with: pip install gym-{args.env_type}"
        ) from e

    logger.info(f"Loading {args.policy_type} policy from {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.checkpoint, args.device, policy_type=args.policy_type
    )

    chunk_size = getattr(policy.config, "chunk_size", getattr(policy.config, "n_action_steps", 100))
    n_action_steps = getattr(policy.config, "n_action_steps", chunk_size)
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
        "policy_type": args.policy_type,
        "task": args.task,
        "task_desc": getattr(args, "task_desc", ""),
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
            obs_dict = preprocess_obs(raw_obs, task_desc=args.task_desc, policy_type=args.policy_type)
            obs_processed = preprocessor(obs_dict)

            # Request new chunk when we've exhausted the current one (chunk_size actions, n_action_steps per rollout)
            need_new_chunk = (current_chunk is None) or (chunk_step_idx >= n_action_steps)

            if need_new_chunk:
                action_chunk, features = predict_action_chunk_with_features(
                    policy, obs_processed, policy_type=args.policy_type
                )
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
            if args.save_action_chunks and current_chunk is not None:
                chunk_np = current_chunk.detach().cpu().numpy()
                if chunk_np.ndim == 3:
                    chunk_np = chunk_np[0]

            dataset_logger.log_step(
                timestep=step,
                executed_action=action_np,
                reward=float(reward),
                done=done,
                success=success_this_step,
                terminated=bool(terminated),
                truncated=bool(truncated),
                obs_state=obs_state,
                obs_images=extract_obs_images(raw_obs) if args.save_images else None,
                env_info=deepcopy(info),
                predicted_action_chunk=chunk_np,
                chunk_length=current_chunk.shape[1] if current_chunk is not None else None,
                chunk_step_idx=chunk_step_idx,
                new_chunk_generated=new_chunk,
                features=current_features,  # same features for all steps in chunk (no new pred)
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
            terminal_step=outcome["terminal_step"],
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
