#!/usr/bin/env python
"""Run failure-aware ACT evaluation: baseline, monitor-only, or intervention.

Compares three modes:
- baseline: standard ACT rollout
- monitor_only: risk model runs, alarms logged, no behavior change
- intervention: when risk > threshold, re-sample N candidate chunks and pick lowest-risk

Example:
    PYTHONPATH=Research MUJOCO_GL=egl python -m failure_prediction.scripts.run_failure_aware_eval \\
        --checkpoint outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \\
        --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \\
        --mode intervention --risk_model_ckpt failure_prediction_runs/transfer_cube_supervised \\
        --risk_threshold 0.5 --num_candidate_chunks 5 --output_dir failure_prediction_runs/online_eval_intervention
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.utils.success_inference import infer_episode_outcome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import collect helpers (same package)
from failure_prediction.scripts.collect_failure_dataset import (  # noqa: E402
    load_policy_and_processors,
    make_single_env,
    preprocess_obs,
    predict_action_chunk_with_features,
    features_to_numpy,
    _default_task_desc,
)


# For intervention: add noise to obs so we get N different chunks from ACT.
def add_obs_noise(obs_dict: dict, noise_std: float = 0.03, rng: np.random.Generator | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    out = dict(obs_dict)
    if "pixels" in out:
        scale = 255.0 * noise_std  # pixel range 0-255
        if isinstance(out["pixels"], dict):
            out["pixels"] = {
                k: np.clip(
                    np.asarray(v, dtype=np.float32) + rng.normal(0, scale, np.asarray(v).shape).astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                for k, v in out["pixels"].items()
            }
        else:
            arr = np.asarray(out["pixels"], dtype=np.float32)
            out["pixels"] = np.clip(arr + rng.normal(0, scale, arr.shape), 0, 255).astype(np.uint8)
    return out


# Load best_model.pt + config.json. Maps feat_* -> key in features dict.
def load_risk_model(ckpt_dir: Path, device: str, feature_field: str = "feat_decoder_mean"):
    import json
    from failure_prediction.models.failure_predictor import FailurePredictorMLP

    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Risk model config not found: {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    input_dim = config.get("input_dim", 512)
    hidden_dims = config.get("hidden_dims", [256, 128])
    dropout = config.get("dropout", 0.1)

    model = FailurePredictorMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    weight_path = ckpt_dir / "best_model.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"Risk model weights not found: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device)
    model.eval()

    field_map = {
        "feat_decoder_mean": "decoder_mean",
        "feat_encoder_latent_token": "encoder_latent_token",
        "feat_latent_sample": "latent_sample",
        "feat_action_chunk_mean": "action_chunk_mean",  # π₀
    }
    score_key = field_map.get(feature_field, "decoder_mean")
    return model, score_key


# baseline / monitor_only (log alarms) / intervention (resample + pick lowest risk)
def run_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    n_action_steps: int,
    device: str,
    risk_model=None,
    risk_key: str = "decoder_mean",
    risk_threshold: float | None = None,
    mode: str = "baseline",
    num_candidate_chunks: int = 5,
    obs_noise_std: float = 0.03,
    rng: np.random.Generator | None = None,
    policy_type: str = "act",
    task_desc: str | None = None,
) -> dict:
    """Run one episode and return metrics."""
    if rng is None:
        rng = np.random.default_rng()

    policy.reset()
    raw_obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
    current_chunk = None
    chunk_step_idx = 0
    max_ep_steps = env.spec.max_episode_steps or 400

    episode_rewards = []
    episode_successes = []
    episode_dones = []
    episode_terminated = []
    episode_truncated = []
    interventions = []
    alarms = []
    step_scores = [] if risk_model else None

    done = False
    step = 0

    while not done and step < max_ep_steps:
        obs_dict = preprocess_obs(raw_obs, task_desc=task_desc, policy_type=policy_type)
        obs_processed = preprocessor(obs_dict)
        need_new_chunk = (current_chunk is None) or (chunk_step_idx >= n_action_steps)

        if need_new_chunk:
            action_chunk, features = predict_action_chunk_with_features(
                policy, obs_processed, policy_type=policy_type
            )
            feat_np = features_to_numpy(features)
            feat_vec = feat_np.get(risk_key)
            risk_prob = None
            if risk_model is not None and feat_vec is not None:
                with torch.no_grad():
                    x = torch.from_numpy(feat_vec).float().unsqueeze(0).to(device)
                    logit = risk_model(x).cpu().item()
                    risk_prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
                if step_scores is not None:
                    step_scores.append(risk_prob)
                alarmed = risk_prob >= risk_threshold

                if mode == "intervention" and alarmed:
                    candidates = []
                    for _ in range(num_candidate_chunks):
                        noisy_obs = add_obs_noise(raw_obs, noise_std=obs_noise_std, rng=rng)
                        noisy_dict = preprocess_obs(noisy_obs, task_desc=task_desc, policy_type=policy_type)
                        noisy_processed = preprocessor(noisy_dict)
                        chunk_cand, feat_cand = predict_action_chunk_with_features(
                            policy, noisy_processed, policy_type=policy_type
                        )
                        feat_cand_np = features_to_numpy(feat_cand)
                        fv = feat_cand_np.get(risk_key)
                        if fv is not None:
                            with torch.no_grad():
                                xc = torch.from_numpy(fv).float().unsqueeze(0).to(device)
                                lc = risk_model(xc).cpu().item()
                                pc = 1.0 / (1.0 + np.exp(-np.clip(lc, -500, 500)))
                            candidates.append((chunk_cand, pc))
                    if candidates:
                        best_idx = np.argmin([c[1] for c in candidates])  # lowest risk prob
                        action_chunk = candidates[best_idx][0]
                        interventions.append({
                            "step": step,
                            "baseline_risk": risk_prob,
                            "n_candidates": num_candidate_chunks,
                            "candidate_risks": [c[1] for c in candidates],
                            "chosen_idx": int(best_idx),
                        })
                alarms.append(alarmed)
            else:
                alarms.append(False)

            current_chunk = action_chunk
            chunk_step_idx = 0
        else:
            if risk_model and step_scores is not None and len(alarms) > 0:
                alarms.append(alarms[-1])

        action = current_chunk[:, chunk_step_idx]
        action = postprocessor(action)
        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]

        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        success_this_step = bool(info.get("is_success", False))
        done = terminated or truncated

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
        env_name=env.unwrapped.spec.id if hasattr(env, "unwrapped") else "",
    )

    return {
        "success": outcome["success"],
        "terminal_step": outcome["terminal_step"],
        "episode_length": step,
        "interventions": interventions,
        "n_interventions": len(interventions),
        "alarms": alarms,
        "step_scores": step_scores,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Failure-aware policy evaluation (ACT or π₀)")
    p.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    p.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    p.add_argument("--env_type", type=str, default="aloha")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--mode", type=str, default="baseline", choices=["baseline", "monitor_only", "intervention"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=None)
    # Risk model (required for monitor_only and intervention)
    p.add_argument("--risk_model_ckpt", type=str, default=None)
    p.add_argument("--risk_feature_field", type=str, default="feat_decoder_mean",
                    help="feat_decoder_mean for ACT, feat_action_chunk_mean for π₀")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--policy_type", type=str, default="act", choices=["act", "pi0"])
    p.add_argument("--task_desc", type=str, default="", help="Language task for π₀")
    p.add_argument("--num_candidate_chunks", type=int, default=5)
    p.add_argument("--obs_noise_std", type=float, default=0.03)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("monitor_only", "intervention") and not args.risk_model_ckpt:
        raise ValueError("--risk_model_ckpt required for monitor_only and intervention modes")

    if args.mode in ("monitor_only", "intervention"):
        ckpt = Path(args.risk_model_ckpt)
        if not (ckpt / "config.json").exists():
            raise FileNotFoundError(f"Risk model config not found: {ckpt / 'config.json'}")
        if not (ckpt / "best_model.pt").exists():
            raise FileNotFoundError(f"Risk model weights not found: {ckpt / 'best_model.pt'}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists() and "lerobot/" not in args.checkpoint:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (use HF repo ID like lerobot/pi0_base for π₀)")

    pkg = f"gym_{args.env_type}"
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Install gym-{args.env_type}") from e

    if args.policy_type == "pi0" and not args.task_desc:
        args.task_desc = _default_task_desc(args.task)
        logger.info(f"Using default task_desc for π₀: {args.task_desc!r}")

    logger.info(f"Loading {args.policy_type} from {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.checkpoint, args.device, policy_type=args.policy_type
    )
    n_action_steps = getattr(policy.config, "n_action_steps", 100)

    risk_model = None
    risk_key = "decoder_mean" if args.policy_type == "act" else "action_chunk_mean"
    if args.risk_model_ckpt:
        ckpt_dir = Path(args.risk_model_ckpt)
        risk_model, risk_key = load_risk_model(ckpt_dir, args.device, args.risk_feature_field)
        logger.info(f"Loaded risk model from {ckpt_dir}")

    env = make_single_env(args.task, args.env_type, args.max_steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_results = []
    n_ep = args.num_episodes
    if n_ep <= 0:
        raise ValueError("num_episodes must be positive")
    for ep_idx in trange(n_ep, desc=f"Eval {args.mode}"):
        ep_seed = args.seed + ep_idx
        rng_ep = np.random.default_rng(ep_seed)
        result = run_episode(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_action_steps=n_action_steps,
            device=args.device,
            risk_model=risk_model if args.mode != "baseline" else None,
            risk_key=risk_key,
            risk_threshold=args.risk_threshold if args.mode != "baseline" else None,
            mode=args.mode,
            num_candidate_chunks=args.num_candidate_chunks,
            obs_noise_std=args.obs_noise_std,
            rng=rng_ep,
            policy_type=args.policy_type,
            task_desc=args.task_desc or None,
        )
        result["episode_id"] = ep_idx
        all_results.append(result)

    env.close()

    successes = [r["success"] for r in all_results]
    n_success = sum(successes)
    n_fail = len(successes) - n_success
    success_rate = n_success / len(successes) if successes else 0

    n_interventions = sum(r["n_interventions"] for r in all_results)
    failed_ep_ids = [r["episode_id"] for r in all_results if not r["success"]]
    # How often did we alarm on failures (catch) vs successes (false alarm)
    failed_with_alarm = sum(1 for r in all_results if not r["success"] and any(r.get("alarms", [])))
    success_ep_ids = [r["episode_id"] for r in all_results if r["success"]]
    success_with_false_alarm = sum(1 for r in all_results if r["success"] and any(r.get("alarms", [])))

    metrics = {
        "mode": args.mode,
        "num_episodes": len(all_results),
        "success_rate": success_rate,
        "n_success": n_success,
        "n_fail": n_fail,
        "total_interventions": n_interventions,
        "avg_interventions_per_episode": n_interventions / len(all_results) if all_results else 0,
        "pct_failed_with_alarm": (failed_with_alarm / n_fail * 100) if n_fail > 0 else 0,
        "pct_success_false_alarm": (success_with_false_alarm / n_success * 100) if n_success > 0 else 0,
        "checkpoint": args.checkpoint,
        "risk_model_ckpt": args.risk_model_ckpt,
        "risk_threshold": args.risk_threshold,
        "num_candidate_chunks": args.num_candidate_chunks,
    }

    # Lead time: steps from first intervention to actual failure (for failed episodes)
    if all_results and any(r.get("interventions") for r in all_results):
        all_lead = []
        for r in all_results:
            if not r["success"] and r.get("interventions"):
                first_int = r["interventions"][0]
                lead = r["terminal_step"] - first_int["step"]
                all_lead.append(max(0, lead))
        metrics["lead_time_mean"] = float(np.mean(all_lead)) if all_lead else 0
        metrics["lead_time_median"] = float(np.median(all_lead)) if all_lead else 0
        metrics["recovery_after_intervention"] = sum(
            1 for r in all_results
            if r["n_interventions"] > 0 and r["success"]
        )

    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "episode_results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "alarms" and k != "step_scores"} for r in all_results], f, indent=2)

    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Success rate: {n_success}/{len(all_results)} = {success_rate*100:.1f}%")
    logger.info(f"Total interventions: {n_interventions}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
