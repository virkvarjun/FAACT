#!/usr/bin/env python
"""Threshold sweep and lead-time analysis for the supervised failure predictor.

Loads validation/test predictions and processed dataset to compute:
- Threshold sweep: precision, recall, F1, FPR, alarms per episode at each threshold
- Lead-time analysis: mean/median/min/max lead time, detection fractions
- Recommended deployment operating point

Example:
    python -m failure_prediction.scripts.analyze_failure_predictor \
        --predictions_dir failure_prediction_runs/transfer_cube_supervised \
        --processed_dir failure_dataset/transfer_cube/processed \
        --output_dir failure_prediction_runs/transfer_cube_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import load_processed_dataset
from failure_prediction.utils.eval_metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return float(a / b)


def load_predictions_and_metadata(predictions_dir: Path, processed_dir: Path):
    """Load test predictions and join with processed dataset for episode-level info."""
    pred_path = predictions_dir / "test_predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    pred = dict(np.load(pred_path, allow_pickle=True))
    logits = pred.get("logits", None)
    probs = pred.get("probs", None)
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64).ravel()
    if probs is None and logits is not None:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    elif probs is None:
        raise ValueError("No probs or logits in predictions")
    probs = np.asarray(probs, dtype=np.float64).ravel()

    episode_ids = np.asarray(pred["episode_ids"]).ravel()
    timesteps = np.asarray(pred["timesteps"]).ravel()
    labels = np.asarray(pred.get("labels", np.zeros(len(probs)))).ravel()

    data, _ = load_processed_dataset(processed_dir)
    proc_ep = np.asarray(data["episode_id"]).ravel()
    proc_ts = np.asarray(data["timestep"]).ravel()
    proc_steps_to_failure = np.asarray(data["steps_to_failure"]).ravel()
    proc_episode_failed = np.asarray(data["episode_failed"]).ravel()

    # Build lookup: (episode_id, timestep) -> (steps_to_failure, episode_failed)
    lookup = {}
    for i in range(len(proc_ep)):
        k = (int(proc_ep[i]), int(proc_ts[i]))
        lookup[k] = (int(proc_steps_to_failure[i]), int(proc_episode_failed[i]))

    steps_to_failure = np.array([lookup.get((int(e), int(t)), (-1, 0))[0] for e, t in zip(episode_ids, timesteps)])
    episode_failed = np.array([lookup.get((int(e), int(t)), (-1, 0))[1] for e, t in zip(episode_ids, timesteps)])

    unique_eps = np.unique(episode_ids)
    ep_outcome = {}
    for e in unique_eps:
        mask = episode_ids == e
        ep_outcome[int(e)] = bool(episode_failed[mask][0]) if mask.any() else False

    logits_arr = pred.get("logits")
    if logits_arr is not None:
        logits_arr = np.asarray(logits_arr, dtype=np.float64).ravel()
    else:
        logits_arr = np.clip(np.log(probs / (1.0 - probs + 1e-12)), -500, 500)

    return {
        "probs": probs,
        "logits": logits_arr,
        "labels": labels,
        "episode_ids": episode_ids,
        "timesteps": timesteps,
        "steps_to_failure": steps_to_failure,
        "episode_failed": episode_failed,
        "ep_outcome": ep_outcome,
    }


def threshold_sweep(probs: np.ndarray, labels: np.ndarray, episode_ids: np.ndarray, ep_failed: dict, thresholds: np.ndarray, logits: np.ndarray | None = None):
    """Sweep thresholds and compute metrics per threshold."""
    if logits is None:
        logits = np.clip(np.log(probs / (1.0 - probs + 1e-12)), -500, 500)
    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(np.int64)
        m = compute_binary_metrics(logits, labels, threshold=thresh)
        n_pos = int((labels > 0.5).sum())
        n_neg = len(labels) - n_pos
        fp_rate = _safe_divide(m["fp"], n_neg, 0.0)

        unique_eps = np.unique(episode_ids)
        alarms_per_ep = {}
        for ep in unique_eps:
            mask = episode_ids == ep
            alarms_per_ep[int(ep)] = int(preds[mask].sum())

        success_eps = [e for e in unique_eps if not ep_failed.get(int(e), True)]
        fail_eps = [e for e in unique_eps if ep_failed.get(int(e), False)]
        alarms_per_success = [alarms_per_ep[int(e)] for e in success_eps] if success_eps else [0]
        alarms_per_fail = [alarms_per_ep[int(e)] for e in fail_eps] if fail_eps else [0]

        results.append({
            "threshold": float(thresh),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fp_rate": fp_rate,
            "alarms_per_success_ep_mean": float(np.mean(alarms_per_success)),
            "alarms_per_success_ep_median": float(np.median(alarms_per_success)),
            "alarms_per_fail_ep_mean": float(np.mean(alarms_per_fail)),
            "alarms_per_fail_ep_median": float(np.median(alarms_per_fail)),
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
        })
    return results


def lead_time_analysis(probs: np.ndarray, episode_ids: np.ndarray, timesteps: np.ndarray,
                       steps_to_failure: np.ndarray, ep_failed: dict, threshold: float):
    """Compute lead-time statistics for failed episodes."""
    failed_eps = [e for e in np.unique(episode_ids) if ep_failed.get(int(e), False)]
    success_eps = [e for e in np.unique(episode_ids) if not ep_failed.get(int(e), True)]

    lead_times = []
    never_alarmed_fail = 0
    alarmed_success = 0

    for ep in failed_eps:
        mask = episode_ids == ep
        ep_probs = probs[mask]
        ep_ts = timesteps[mask]
        ep_st2f = steps_to_failure[mask]
        alarms = ep_probs >= threshold
        if not alarms.any():
            never_alarmed_fail += 1
            continue
        first_alarm_idx = np.argmax(alarms)
        first_alarm_step = int(ep_ts[first_alarm_idx])
        st2f = int(ep_st2f[first_alarm_idx])
        failure_step = first_alarm_step + st2f if st2f >= 0 else int(ep_ts[-1]) + 1
        lead = failure_step - first_alarm_step
        lead_times.append(max(0, lead))

    for ep in success_eps:
        mask = episode_ids == ep
        ep_probs = probs[mask]
        if (ep_probs >= threshold).any():
            alarmed_success += 1

    lead_times = np.array(lead_times) if lead_times else np.array([0])

    det_1 = len(lead_times[lead_times >= 1]) if len(lead_times) > 0 else 0
    det_3 = len(lead_times[lead_times >= 3]) if len(lead_times) > 0 else 0
    det_5 = len(lead_times[lead_times >= 5]) if len(lead_times) > 0 else 0
    det_10 = len(lead_times[lead_times >= 10]) if len(lead_times) > 0 else 0
    n_fail = len(failed_eps)
    n_success = len(success_eps)

    return {
        "threshold": threshold,
        "n_failed_episodes": n_fail,
        "n_success_episodes": n_success,
        "lead_time_mean": float(np.mean(lead_times)) if len(lead_times) > 0 else 0.0,
        "lead_time_median": float(np.median(lead_times)) if len(lead_times) > 0 else 0.0,
        "lead_time_min": int(np.min(lead_times)) if len(lead_times) > 0 else 0,
        "lead_time_max": int(np.max(lead_times)) if len(lead_times) > 0 else 0,
        "pct_failed_detected_1_step": _safe_divide(det_1, n_fail, 0.0) * 100,
        "pct_failed_detected_3_step": _safe_divide(det_3, n_fail, 0.0) * 100,
        "pct_failed_detected_5_step": _safe_divide(det_5, n_fail, 0.0) * 100,
        "pct_failed_detected_10_step": _safe_divide(det_10, n_fail, 0.0) * 100,
        "pct_failed_never_alarmed": _safe_divide(never_alarmed_fail, n_fail, 0.0) * 100,
        "pct_success_false_alarm": _safe_divide(alarmed_success, n_success, 0.0) * 100 if n_success > 0 else 0.0,
        "lead_times": lead_times.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Threshold sweep and lead-time analysis for failure predictor")
    p.add_argument("--predictions_dir", type=str, required=True, help="Dir with test_predictions.npz")
    p.add_argument("--processed_dir", type=str, required=True, help="Processed dataset dir (for steps_to_failure)")
    p.add_argument("--output_dir", type=str, default=None, help="Output dir (default: predictions_dir/analysis)")
    p.add_argument("--n_thresholds", type=int, default=101, help="Number of thresholds to sweep")
    return p.parse_args()


def main():
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading predictions from {predictions_dir}, processed from {processed_dir}")
    data = load_predictions_and_metadata(predictions_dir, processed_dir)

    probs = data["probs"]
    logits = data.get("logits")
    labels = data["labels"]
    episode_ids = data["episode_ids"]
    timesteps = data["timesteps"]
    steps_to_failure = data["steps_to_failure"]
    ep_outcome = data["ep_outcome"]

    thresholds = np.linspace(0.0, 1.0, args.n_thresholds)
    sweep_results = threshold_sweep(probs, labels, episode_ids, ep_outcome, thresholds, logits=logits)
    np.savez(output_dir / "threshold_sweep.npz", thresholds=thresholds)

    best_f1_idx = np.argmax([r["f1"] for r in sweep_results])
    best_f1_threshold = sweep_results[best_f1_idx]["threshold"]

    high_recall_threshold = best_f1_threshold
    for r in reversed(sweep_results):
        if r["recall"] >= 0.7:
            high_recall_threshold = r["threshold"]
            break

    low_fp_threshold = best_f1_threshold
    for r in sweep_results:
        if r["fp_rate"] <= 0.05:
            low_fp_threshold = r["threshold"]
            break

    recommended = best_f1_threshold
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump({
            "sweep": sweep_results,
            "best_f1_threshold": best_f1_threshold,
            "best_f1_metrics": sweep_results[best_f1_idx],
            "high_recall_threshold": high_recall_threshold,
            "low_fp_threshold": low_fp_threshold,
            "recommended_threshold": recommended,
            "recommended_reason": "Best F1 on test set",
        }, f, indent=2)

    lead = lead_time_analysis(probs, episode_ids, timesteps, steps_to_failure, ep_outcome, recommended)
    with open(output_dir / "lead_time.json", "w") as f:
        json.dump(lead, f, indent=2)

    summary = f"""# Failure Predictor Analysis

## Threshold Sweep Summary
- **Best F1 threshold:** {best_f1_threshold:.3f} (F1={sweep_results[best_f1_idx]['f1']:.3f})
- **High-recall threshold:** {high_recall_threshold:.3f}
- **Low false-alarm threshold:** {low_fp_threshold:.3f}
- **Recommended deployment threshold:** {recommended:.3f}

## Lead-Time Analysis (threshold={recommended:.3f})
- Mean lead time: {lead['lead_time_mean']:.1f} steps
- Median lead time: {lead['lead_time_median']:.1f} steps
- Min/Max: {lead['lead_time_min']} / {lead['lead_time_max']} steps
- Failed episodes detected ≥1 step early: {lead['pct_failed_detected_1_step']:.1f}%
- Failed episodes detected ≥10 step early: {lead['pct_failed_detected_10_step']:.1f}%
- Failed episodes never alarmed: {lead['pct_failed_never_alarmed']:.1f}%
- Success episodes with false alarm: {lead['pct_success_false_alarm']:.1f}%
"""
    with open(output_dir / "analysis_summary.md", "w") as f:
        f.write(summary)

    logger.info(f"Analysis saved to {output_dir}")
    logger.info(summary)


if __name__ == "__main__":
    main()
