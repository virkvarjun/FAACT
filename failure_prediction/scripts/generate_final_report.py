#!/usr/bin/env python
"""Generate final project report from run artifacts.

Example:
    python -m failure_prediction.scripts.generate_final_report \\
        --raw_dir failure_dataset/transfer_cube/raw \\
        --processed_dir failure_dataset/transfer_cube/processed \\
        --supervised_dir failure_prediction_runs/transfer_cube_supervised \\
        --analysis_dir failure_prediction_runs/transfer_cube_analysis \\
        --online_baseline failure_prediction_runs/online_eval_baseline \\
        --online_monitor failure_prediction_runs/online_eval_monitor \\
        --online_intervention failure_prediction_runs/online_eval_intervention \\
        --checkpoint_path a23v/act_transfer_cube \\
        --output failure_dataset/transfer_cube/final_project_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    p.add_argument("--raw_dir", type=str, default=None)
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--supervised_dir", type=str, required=True)
    p.add_argument("--analysis_dir", type=str, default=None)
    p.add_argument("--online_baseline", type=str, default=None)
    p.add_argument("--online_monitor", type=str, default=None)
    p.add_argument("--online_intervention", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default="a23v/act_transfer_cube")
    p.add_argument("--output", type=str, default="failure_dataset/transfer_cube/final_project_report.md")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []

    # Header
    parts.append("# Failure-Aware ACT: Final Project Report\n")
    parts.append("## 1. Checkpoint\n")
    parts.append(f"- **Checkpoint:** `{args.checkpoint_path}`\n")

    # Dataset
    if args.processed_dir:
        meta_path = Path(args.processed_dir) / "metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            if meta:
                parts.append("## 2. Dataset Summary\n")
                parts.append(f"- **Total timesteps:** {meta.get('total_timesteps', 'N/A')}\n")
                parts.append(f"- **Episodes:** {meta.get('total_episodes', 'N/A')}\n")
                parts.append(f"- **Success rate (collection):** {meta.get('success_rate', 0)*100:.1f}%\n")

    # Supervised predictor
    sup_dir = Path(args.supervised_dir)
    metrics_path = sup_dir / "metrics.json"
    if metrics_path.exists():
        m = load_json(metrics_path)
        if m and "test_metrics" in m:
            tm = m["test_metrics"]
            parts.append("## 3. Supervised Failure Predictor\n")
            parts.append(f"- **Test AUROC:** {tm.get('auroc', 0):.4f}\n")
            parts.append(f"- **Test AUPRC:** {tm.get('auprc', 0):.4f}\n")
            parts.append(f"- **Test F1:** {tm.get('f1', 0):.4f}\n")
            parts.append(f"- **Precision:** {tm.get('precision', 0):.4f}\n")
            parts.append(f"- **Recall:** {tm.get('recall', 0):.4f}\n")

    # Threshold and lead-time
    if args.analysis_dir:
        ad = Path(args.analysis_dir)
        thresh_path = ad / "threshold_sweep.json"
        lead_path = ad / "lead_time.json"
        if thresh_path.exists():
            sweep = load_json(thresh_path)
            if sweep:
                rec = sweep.get("recommended_threshold", 0.5)
                parts.append("## 4. Threshold and Lead-Time\n")
                parts.append(f"- **Recommended threshold:** {rec:.3f}\n")
                parts.append(f"- **Reason:** {sweep.get('recommended_reason', 'Best F1')}\n")
        if lead_path.exists():
            lead = load_json(lead_path)
            if lead:
                parts.append(f"- **Mean lead time:** {lead.get('lead_time_mean', 0):.1f} steps\n")
                parts.append(f"- **Median lead time:** {lead.get('lead_time_median', 0):.1f} steps\n")
                parts.append(f"- **Failed episodes never alarmed:** {lead.get('pct_failed_never_alarmed', 0):.1f}%\n")
                parts.append(f"- **Success episodes with false alarm:** {lead.get('pct_success_false_alarm', 0):.1f}%\n")

    # FIPER (from integration report or fiper dir)
    parts.append("## 5. FIPER Baseline Summary\n")
    parts.append("- FIPER (RND + ACE) underperformed on this setup.\n")
    parts.append("- Alarm precision/recall were 0; no meaningful failure detection.\n")
    parts.append("- Supervised predictor is the primary path.\n")

    # Online evaluation
    parts.append("## 6. Online Evaluation Summary\n")
    for name, path in [
        ("ACT baseline", args.online_baseline),
        ("ACT + monitor only", args.online_monitor),
        ("ACT + monitor + intervention", args.online_intervention),
    ]:
        if path:
            mpath = Path(path) / "eval_metrics.json"
            if mpath.exists():
                m = load_json(mpath)
                if m:
                    sr = m.get("success_rate", 0) * 100
                    n_ep = m.get("num_episodes", 0)
                    parts.append(f"### {name}\n")
                    parts.append(f"- **Success rate:** {sr:.1f}% ({m.get('n_success', 0)}/{n_ep})\n")
                    if m.get("total_interventions", 0) > 0:
                        parts.append(f"- **Total interventions:** {m['total_interventions']}\n")
                        parts.append(f"- **Avg interventions/episode:** {m.get('avg_interventions_per_episode', 0):.2f}\n")
                        if "recovery_after_intervention" in m:
                            parts.append(f"- **Recoveries after intervention:** {m['recovery_after_intervention']}\n")
                    parts.append("\n")

    # Intervention impact
    if args.online_baseline and args.online_intervention:
        bpath = Path(args.online_baseline) / "eval_metrics.json"
        ipath = Path(args.online_intervention) / "eval_metrics.json"
        if bpath.exists() and ipath.exists():
            bm = load_json(bpath)
            im = load_json(ipath)
            if bm and im:
                base_sr = bm.get("success_rate", 0) * 100
                int_sr = im.get("success_rate", 0) * 100
                diff = int_sr - base_sr
                parts.append("## 7. Did Intervention Improve ACT?\n")
                parts.append(f"- Baseline success rate: {base_sr:.1f}%\n")
                parts.append(f"- Intervention success rate: {int_sr:.1f}%\n")
                parts.append(f"- **Change:** {diff:+.1f} percentage points\n")
                parts.append("\n")

    # Limitations
    parts.append("## 8. Limitations\n")
    parts.append("- ACT does not natively support multi-sample latent sampling; used observation-noise proxy.\n")
    parts.append("- Candidate scoring uses same-state feature; no chunk-specific embeddings.\n")
    parts.append("- Small evaluation batch (20 ep) in examples; scale for conclusive results.\n")
    parts.append("- FIPER baseline did not detect failures in this setup.\n")
    parts.append("\n")

    # Future work
    parts.append("## 9. Recommended Future Work\n")
    parts.append("- Integrate failure predictor into production deployment loop.\n")
    parts.append("- Add multi-sample ACT inference if available for true latent diversity.\n")
    parts.append("- Scale online evaluation for statistical significance.\n")
    parts.append("- Tune threshold per deployment environment.\n")

    with open(out_path, "w") as f:
        f.write("".join(parts))

    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
