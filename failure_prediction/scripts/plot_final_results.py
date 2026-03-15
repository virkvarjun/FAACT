#!/usr/bin/env python
"""Generate final plots for failure-aware ACT: offline predictor + online evaluation.

Example:
    python -m failure_prediction.scripts.plot_final_results \\
        --run_dirs failure_prediction_runs/transfer_cube_supervised \\
                   failure_prediction_runs/transfer_cube_analysis \\
                   failure_prediction_runs/online_eval_baseline \\
                   failure_prediction_runs/online_eval_monitor \\
                   failure_prediction_runs/online_eval_intervention \\
        --output_dir failure_prediction_runs/final_plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", type=str, nargs="+", help="Run directories")
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/final_plots")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        sys.exit(0)

    run_dirs = [Path(d) for d in args.run_dirs]

    # --- Offline: supervised predictor (ROC, PR, threshold sweep, histogram) ---
    for rd in run_dirs:
        pred_npz = rd / "test_predictions.npz"
        if pred_npz.exists():
            data = dict(np.load(pred_npz, allow_pickle=True))
            probs = data.get("probs")
            if probs is None and "logits" in data:
                probs = 1.0 / (1.0 + np.exp(-np.clip(data["logits"], -500, 500)))
            labels = data.get("labels", np.zeros(len(probs)))
            ep_ids = np.asarray(data.get("episode_ids", np.arange(len(probs)))).ravel()

            # ROC
            try:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(labels, probs)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(5, 5))
                plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "roc_curve.png", dpi=100)
                plt.close()
            except Exception:
                pass

            # PR curve
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                prec, rec, _ = precision_recall_curve(labels, probs)
                ap = average_precision_score(labels, probs)
                plt.figure(figsize=(5, 5))
                plt.plot(rec, prec, label=f"AUPRC={ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "pr_curve.png", dpi=100)
                plt.close()
            except Exception:
                pass

            # Risk histogram
            plt.figure(figsize=(6, 4))
            plt.hist(probs[labels <= 0.5], bins=30, alpha=0.6, label="Negative", density=True)
            plt.hist(probs[labels > 0.5], bins=30, alpha=0.6, label="Positive", density=True)
            plt.xlabel("Risk score")
            plt.ylabel("Density")
            plt.title("Risk score distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "risk_histogram.png", dpi=100)
            plt.close()

            # Score-over-time for first failed episode
            fail_eps = [e for e in np.unique(ep_ids) if (labels[ep_ids == e] > 0.5).any()]
            if fail_eps:
                ep0 = fail_eps[0]
                mask = ep_ids == ep0
                t = np.arange(mask.sum())
                plt.figure(figsize=(8, 4))
                plt.plot(t, probs[mask], label="Risk")
                plt.axhline(0.5, color="gray", linestyle="--", label="Threshold")
                plt.xlabel("Timestep")
                plt.ylabel("Risk score")
                plt.title(f"Failed episode {ep0} - risk over time")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "risk_failed_episode.png", dpi=100)
                plt.close()
            break

    # --- Analysis: threshold sweep, lead-time ---
    for rd in run_dirs:
        thresh_json = rd / "threshold_sweep.json"
        lead_json = rd / "lead_time.json"
        if thresh_json.exists():
            sweep = load_json(thresh_json)
            if sweep and "sweep" in sweep:
                rows = sweep["sweep"]
                thresh = [r["threshold"] for r in rows]
                f1 = [r["f1"] for r in rows]
                prec = [r["precision"] for r in rows]
                rec = [r["recall"] for r in rows]
                plt.figure(figsize=(7, 4))
                plt.plot(thresh, f1, label="F1")
                plt.plot(thresh, prec, label="Precision")
                plt.plot(thresh, rec, label="Recall")
                plt.xlabel("Threshold")
                plt.ylabel("Metric")
                plt.title("Threshold sweep")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "threshold_sweep.png", dpi=100)
                plt.close()
        if lead_json.exists():
            lead = load_json(lead_json)
            if lead and "lead_times" in lead and lead["lead_times"]:
                lt = np.array(lead["lead_times"])
                plt.figure(figsize=(6, 4))
                plt.hist(lt, bins=min(20, max(5, len(np.unique(lt)))), edgecolor="black", alpha=0.7)
                plt.xlabel("Lead time (steps)")
                plt.ylabel("Count")
                plt.title("Lead time distribution")
                plt.tight_layout()
                plt.savefig(out_dir / "lead_time_histogram.png", dpi=100)
                plt.close()
            break

    # --- Online: success rate comparison ---
    baseline_metrics = None
    monitor_metrics = None
    intervention_metrics = None
    for rd in run_dirs:
        mpath = rd / "eval_metrics.json"
        if mpath.exists():
            m = load_json(mpath)
            if m:
                mode = m.get("mode", "")
                if mode == "baseline":
                    baseline_metrics = m
                elif mode == "monitor_only":
                    monitor_metrics = m
                elif mode == "intervention":
                    intervention_metrics = m

    if baseline_metrics or monitor_metrics or intervention_metrics:
        modes = []
        rates = []
        for name, m in [("ACT baseline", baseline_metrics), ("ACT+monitor", monitor_metrics), ("ACT+intervention", intervention_metrics)]:
            if m:
                modes.append(name)
                rates.append(m.get("success_rate", 0) * 100)
        if modes:
            plt.figure(figsize=(6, 4))
            plt.bar(modes, rates, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(modes)])
            plt.ylabel("Success rate (%)")
            plt.title("Online evaluation comparison")
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(out_dir / "online_success_rates.png", dpi=100)
            plt.close()

        # Interventions per episode (intervention mode)
        if intervention_metrics:
            ep_path = None
            for rd in run_dirs:
                p = rd / "episode_results.json"
                if p.exists():
                    mpath = rd / "eval_metrics.json"
                    if mpath.exists():
                        m = load_json(mpath)
                        if m and m.get("mode") == "intervention":
                            ep_path = p
                            break
            if ep_path:
                eps = load_json(ep_path)
                if eps:
                    n_int = [r.get("n_interventions", 0) for r in eps]
                    plt.figure(figsize=(6, 4))
                    plt.hist(n_int, bins=range(0, max(n_int) + 2 if n_int else 2), edgecolor="black", alpha=0.7)
                    plt.xlabel("Interventions per episode")
                    plt.ylabel("Count")
                    plt.title("Intervention distribution")
                    plt.tight_layout()
                    plt.savefig(out_dir / "interventions_per_episode.png", dpi=100)
                    plt.close()

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
