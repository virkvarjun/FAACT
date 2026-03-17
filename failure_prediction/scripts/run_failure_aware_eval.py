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
sys.path.insert(0, str(RESEARCH_DIR / "faact"))

from failure_prediction.utils.success_inference import infer_episode_outcome
from failure_prediction.runtime_components import (
    ThresholdInterventionPolicy,
    load_supervised_risk_runtime,
)
from faact.backbone.factory import make_backbone_wrapper
from faact.evaluation.online_runner import EpisodeRunnerConfig, run_episode as run_wrapper_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import collect helpers (same package)
from failure_prediction.scripts.collect_failure_dataset import (  # noqa: E402
    make_single_env,
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


def _logit_to_prob(logit: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))


def score_feature_vector(risk_model, feat_vec: np.ndarray | None, device: str) -> float | None:
    if risk_model is None or feat_vec is None:
        return None
    with torch.no_grad():
        x = torch.from_numpy(feat_vec).float().unsqueeze(0).to(device)
        logit = risk_model(x).cpu().item()
    return _logit_to_prob(logit)


def chunk_to_numpy(action_chunk) -> np.ndarray:
    chunk_np = action_chunk.detach().cpu().numpy()
    if chunk_np.ndim == 3:
        chunk_np = chunk_np[0]
    return chunk_np.astype(np.float32, copy=False)


def compute_candidate_diversity(
    baseline_chunk: np.ndarray, candidate_chunks: list[np.ndarray]
) -> dict[str, float | list[float]]:
    if not candidate_chunks:
        return {
            "candidate_l2_to_baseline": [],
            "candidate_l2_to_baseline_mean": 0.0,
            "candidate_l2_to_baseline_min": 0.0,
            "candidate_l2_to_baseline_max": 0.0,
            "candidate_pairwise_l2_mean": 0.0,
            "candidate_pairwise_l2_max": 0.0,
        }

    base_flat = baseline_chunk.reshape(-1)
    dists_to_baseline = [
        float(np.linalg.norm(candidate.reshape(-1) - base_flat)) for candidate in candidate_chunks
    ]

    pairwise_dists = []
    for idx, first in enumerate(candidate_chunks):
        first_flat = first.reshape(-1)
        for second in candidate_chunks[idx + 1 :]:
            pairwise_dists.append(float(np.linalg.norm(first_flat - second.reshape(-1))))

    return {
        "candidate_l2_to_baseline": dists_to_baseline,
        "candidate_l2_to_baseline_mean": float(np.mean(dists_to_baseline)),
        "candidate_l2_to_baseline_min": float(np.min(dists_to_baseline)),
        "candidate_l2_to_baseline_max": float(np.max(dists_to_baseline)),
        "candidate_pairwise_l2_mean": float(np.mean(pairwise_dists)) if pairwise_dists else 0.0,
        "candidate_pairwise_l2_max": float(np.max(pairwise_dists)) if pairwise_dists else 0.0,
    }


def predict_action_chunk_with_sampling(
    policy,
    obs_processed,
    policy_type: str,
    use_dropout: bool = False,
):
    """Optionally enable MC dropout to sample diverse ACT chunks at inference time."""
    if not use_dropout:
        return predict_action_chunk_with_features(policy, obs_processed, policy_type=policy_type)

    was_training = policy.training
    policy.train()
    try:
        with torch.no_grad():
            return predict_action_chunk_with_features(policy, obs_processed, policy_type=policy_type)
    finally:
        if not was_training:
            policy.eval()


def add_action_noise(
    action_chunk: torch.Tensor,
    noise_std: float = 0.05,
    prefix_steps: int = 10,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Perturb only the early action prefix to search for local self-corrections."""
    if rng is None:
        rng = np.random.default_rng()
    candidate = action_chunk.clone()
    prefix = min(prefix_steps, candidate.shape[1])
    noise = torch.from_numpy(
        rng.normal(0.0, noise_std, size=tuple(candidate[:, :prefix].shape)).astype(np.float32)
    ).to(device=candidate.device, dtype=candidate.dtype)
    candidate[:, :prefix] = candidate[:, :prefix] + noise
    return candidate


# Load best_model.pt + config.json. Maps feat_* -> key in features dict.
def load_risk_model(ckpt_dir: Path, device: str, feature_field: str = "feat_decoder_mean"):
    scorer, meta = load_supervised_risk_runtime(ckpt_dir, device, feature_field=feature_field)
    return scorer, meta["feature_keys"]


# baseline / monitor_only (log alarms) / intervention (resample + pick lowest risk)
def run_episode(
    env,
    backbone,
    risk_scorer=None,
    intervention_policy=None,
    risk_threshold: float | None = None,
    mode: str = "baseline",
    num_candidate_chunks: int = 5,
    obs_noise_std: float = 0.03,
    switch_margin: float = 0.0,
    replan_interval: int | None = None,
    candidate_source: str = "obs_noise",
    temporal_ensemble_coeff: float | None = None,
    action_noise_std: float = 0.05,
    action_noise_prefix_steps: int = 10,
    min_candidate_l2_to_baseline: float = 0.0,
    rng: np.random.Generator | None = None,
    policy_type: str = "act",
    task_desc: str | None = None,
) -> dict:
    """Run one episode through the shared FAACT wrapper runtime."""
    config = EpisodeRunnerConfig(
        mode=mode,
        num_candidate_chunks=num_candidate_chunks,
        obs_noise_std=obs_noise_std,
        switch_margin=switch_margin,
        replan_interval=replan_interval,
        candidate_source=candidate_source,
        action_noise_std=action_noise_std,
        action_noise_prefix_steps=action_noise_prefix_steps,
        task_desc=task_desc,
        score_every_step=("remaining" in (getattr(risk_scorer, "feature_key", "") or "")),
        temporal_ensemble_coeff=temporal_ensemble_coeff if policy_type == "act" else None,
        cooldown_steps=getattr(intervention_policy, "cooldown_steps", 0),
        max_interventions_per_episode=getattr(intervention_policy, "max_interventions_per_episode", None),
        boundary_only_intervention=getattr(intervention_policy, "boundary_only", False),
        min_candidate_l2_to_baseline=min_candidate_l2_to_baseline,
    )
    result, _frames = run_wrapper_episode(
        env=env,
        backbone=backbone,
        rng=rng,
        risk_scorer=risk_scorer if mode != "baseline" else None,
        intervention_policy=intervention_policy if mode == "intervention" else None,
        config=config,
    )
    return result


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
    p.add_argument(
        "--risk_feature_field",
        type=str,
        default="",
        help="Single feat_* field or comma-separated list. Defaults to the feature field(s) saved in the risk model config.",
    )
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--policy_type", type=str, default="act", choices=["act", "pi0"])
    p.add_argument("--task_desc", type=str, default="", help="Language task for π₀")
    p.add_argument("--num_candidate_chunks", type=int, default=5)
    p.add_argument("--obs_noise_std", type=float, default=0.03)
    p.add_argument("--switch_margin", type=float, default=0.0,
                    help="Minimum risk improvement required to replace the baseline chunk")
    p.add_argument("--replan_interval", type=int, default=None,
                    help="How many executed actions before replanning. Defaults to policy n_action_steps.")
    p.add_argument("--candidate_source", type=str, default="obs_noise",
                    choices=["obs_noise", "dropout", "obs_noise_dropout", "action_noise", "hybrid"],
                    help="How intervention candidates are diversified")
    p.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                    help="If set for ACT, replan every step and ensemble overlapping chunks online")
    p.add_argument("--action_noise_std", type=float, default=0.05,
                    help="Std of Gaussian noise for action-space candidate search")
    p.add_argument("--action_noise_prefix_steps", type=int, default=10,
                    help="Only perturb this many early steps during action-space candidate search")
    p.add_argument("--cooldown_steps", type=int, default=0,
                    help="Minimum executed steps between accepted interventions")
    p.add_argument("--max_interventions_per_episode", type=int, default=None,
                    help="Optional cap on accepted interventions per episode")
    p.add_argument("--boundary_only_intervention", action="store_true",
                    help="Only allow alarms/interventions when proposing a fresh chunk")
    p.add_argument("--min_candidate_l2_to_baseline", type=float, default=0.0,
                    help="Reject candidate swaps that are too similar to baseline")
    return p.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _mean(values):
    return float(np.mean(values)) if values else 0.0


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

    logger.info(f"Loading {args.policy_type} wrapper from {args.checkpoint}")
    backbone = make_backbone_wrapper(
        args.policy_type,
        args.checkpoint,
        device=args.device,
        task_desc=args.task_desc or None,
    )
    n_action_steps = backbone.chunk_size

    risk_scorer = None
    risk_keys = []
    if args.risk_model_ckpt:
        ckpt_dir = Path(args.risk_model_ckpt)
        risk_scorer, risk_keys = load_risk_model(ckpt_dir, args.device, args.risk_feature_field or None)
        logger.info(f"Loaded risk model from {ckpt_dir}")
    intervention_policy = (
        ThresholdInterventionPolicy(
            args.risk_threshold,
            cooldown_steps=args.cooldown_steps,
            max_interventions_per_episode=args.max_interventions_per_episode,
            boundary_only=args.boundary_only_intervention,
        )
        if args.mode == "intervention" and risk_scorer is not None
        else None
    )

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
            backbone=backbone,
            risk_scorer=risk_scorer,
            intervention_policy=intervention_policy,
            risk_threshold=args.risk_threshold if args.mode != "baseline" else None,
            mode=args.mode,
            num_candidate_chunks=args.num_candidate_chunks,
            obs_noise_std=args.obs_noise_std,
            switch_margin=args.switch_margin,
            replan_interval=args.replan_interval,
            candidate_source=args.candidate_source,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            action_noise_std=args.action_noise_std,
            action_noise_prefix_steps=args.action_noise_prefix_steps,
            min_candidate_l2_to_baseline=args.min_candidate_l2_to_baseline,
            rng=rng_ep,
            policy_type=args.policy_type,
            task_desc=args.task_desc or None,
        )
        result["config_overrides"] = {
            "cooldown_steps": args.cooldown_steps,
            "max_interventions_per_episode": args.max_interventions_per_episode,
            "boundary_only_intervention": args.boundary_only_intervention,
            "min_candidate_l2_to_baseline": args.min_candidate_l2_to_baseline,
        }
        result["episode_id"] = ep_idx
        all_results.append(result)

    env.close()

    successes = [r["success"] for r in all_results]
    n_success = sum(successes)
    n_fail = len(successes) - n_success
    success_rate = n_success / len(successes) if successes else 0

    n_interventions = sum(r["n_interventions"] for r in all_results)
    n_intervention_attempts = sum(r.get("n_intervention_attempts", 0) for r in all_results)
    n_rejected_interventions = sum(
        r.get("episode_summary", {}).get("n_rejected_interventions", 0) for r in all_results
    )
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
        "total_intervention_attempts": n_intervention_attempts,
        "total_rejected_interventions": n_rejected_interventions,
        "avg_interventions_per_episode": n_interventions / len(all_results) if all_results else 0,
        "avg_intervention_attempts_per_episode": n_intervention_attempts / len(all_results) if all_results else 0,
        "accepted_intervention_rate": (
            n_interventions / n_intervention_attempts if n_intervention_attempts > 0 else 0
        ),
        "pct_failed_with_alarm": (failed_with_alarm / n_fail * 100) if n_fail > 0 else 0,
        "pct_success_false_alarm": (success_with_false_alarm / n_success * 100) if n_success > 0 else 0,
        "checkpoint": args.checkpoint,
        "risk_model_ckpt": args.risk_model_ckpt,
        "risk_threshold": args.risk_threshold,
        "risk_feature_field": args.risk_feature_field or None,
        "risk_feature_keys": risk_keys,
        "num_candidate_chunks": args.num_candidate_chunks,
        "switch_margin": args.switch_margin,
        "replan_interval": args.replan_interval if args.replan_interval is not None else n_action_steps,
        "candidate_source": args.candidate_source,
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "action_noise_std": args.action_noise_std,
        "action_noise_prefix_steps": args.action_noise_prefix_steps,
        "cooldown_steps": args.cooldown_steps,
        "max_interventions_per_episode": args.max_interventions_per_episode,
        "boundary_only_intervention": args.boundary_only_intervention,
        "min_candidate_l2_to_baseline": args.min_candidate_l2_to_baseline,
    }

    alarm_risks = []
    accepted_deltas = []
    best_candidate_deltas = []
    better_candidate_attempts = 0
    total_alarm_steps = 0
    total_boundary_alarms = 0
    rejection_reason_counts = {}
    for result in all_results:
        episode_summary = result.get("episode_summary", {})
        total_alarm_steps += int(episode_summary.get("n_alarm_steps", 0))
        total_boundary_alarms += int(episode_summary.get("n_boundary_alarms", 0))
        better_candidate_attempts += int(
            round(
                episode_summary.get("better_candidate_available_rate", 0.0)
                * result.get("n_intervention_attempts", 0)
            )
        )
        for event in result.get("alarm_events", []):
            if event.get("risk_prob") is not None:
                alarm_risks.append(float(event["risk_prob"]))
        for item in result.get("interventions", []):
            if item.get("best_candidate_risk_delta") is not None:
                best_candidate_deltas.append(float(item["best_candidate_risk_delta"]))
            if item.get("accepted", False) and item.get("risk_delta") is not None:
                accepted_deltas.append(float(item["risk_delta"]))
            reason = item.get("rejection_reason", "")
            if reason:
                rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

    metrics["alarm_steps_total"] = total_alarm_steps
    metrics["boundary_alarm_rate"] = (
        total_boundary_alarms / total_alarm_steps if total_alarm_steps > 0 else 0.0
    )
    metrics["better_candidate_available_rate"] = (
        better_candidate_attempts / n_intervention_attempts if n_intervention_attempts > 0 else 0.0
    )
    metrics["mean_alarm_risk"] = _mean(alarm_risks)
    metrics["mean_accepted_risk_delta"] = _mean(accepted_deltas)
    metrics["mean_best_candidate_risk_delta"] = _mean(best_candidate_deltas)
    metrics["rejection_reason_counts"] = rejection_reason_counts

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
        json.dump(
            [
                _json_ready({k: v for k, v in r.items() if k != "alarms" and k != "step_scores"})
                for r in all_results
            ],
            f,
            indent=2,
        )
    with open(output_dir / "diagnostics_summary.json", "w") as f:
        json.dump(
            _json_ready(
                {
                    "aggregate_metrics": metrics,
                    "episode_summaries": [
                        {
                            "episode_id": r["episode_id"],
                            "success": r["success"],
                            "terminal_step": r["terminal_step"],
                            "episode_length": r["episode_length"],
                            "episode_summary": r.get("episode_summary", {}),
                        }
                        for r in all_results
                    ],
                }
            ),
            f,
            indent=2,
        )

    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Success rate: {n_success}/{len(all_results)} = {success_rate*100:.1f}%")
    logger.info(f"Total interventions: {n_interventions}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
