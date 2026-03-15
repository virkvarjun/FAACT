#!/usr/bin/env python
"""Generate a grounded integration report from real failure prediction outputs.

Reads artifacts from collection, postprocessing, supervised training, and FIPER,
and produces a concise technical summary for the integration report.

Example:
    python -m failure_prediction.scripts.generate_integration_report \
        --raw_dir failure_dataset/transfer_cube/raw \
        --processed_dir failure_dataset/transfer_cube/processed \
        --supervised_dir failure_prediction_runs/transfer_cube_supervised \
        --fiper_dir failure_prediction_runs/transfer_cube_fiper \
        --output failure_dataset/transfer_cube/integration_report.md
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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Generate integration report from real outputs")
    p.add_argument("--raw_dir", type=str, default=None)
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--supervised_dir", type=str, default=None)
    p.add_argument("--fiper_dir", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="Output markdown file")
    args = p.parse_args()

    report_lines = [
        "# Failure Prediction Integration Report",
        "",
        "## 1. Checkpoint and Commands",
        "",
    ]

    if args.checkpoint_path:
        report_lines.append(f"- **Checkpoint used:** `{args.checkpoint_path}`")
    else:
        report_lines.append("- **Checkpoint used:** (set via --checkpoint_path)")
    report_lines.append("")

    # Raw / processed stats
    if args.raw_dir:
        raw_path = Path(args.raw_dir)
        episodes = sorted(raw_path.glob("episode_*.npz")) if raw_path.exists() else []
        report_lines.append(f"- **Rollout episodes collected:** {len(episodes)}")
    report_lines.append("")

    if args.processed_dir:
        proc_path = Path(args.processed_dir)
        meta_path = proc_path / "metadata.json"
        meta = load_json(meta_path)
        stats = meta.get("stats", {})
        report_lines.append("## 2. Processed Dataset")
        report_lines.append("")
        report_lines.append(f"- **Total episodes:** {stats.get('total_episodes', 'N/A')}")
        report_lines.append(f"- **Successful:** {stats.get('successful_episodes', 'N/A')}")
        report_lines.append(f"- **Failed:** {stats.get('failed_episodes', 'N/A')}")
        report_lines.append(f"- **Total timesteps:** {stats.get('total_timesteps', 'N/A')}")
        report_lines.append(f"- **failure_within_k positives:** {stats.get('failure_within_k_positive', 'N/A')}")
        report_lines.append("")
        fields = meta.get("fields", {})
        feat_fields = sorted(k for k in fields if k.startswith("feat_"))
        report_lines.append("## 3. Available Feature Fields")
        report_lines.append("")
        for k in feat_fields:
            report_lines.append(f"- `{k}`: {fields[k]}")
        if feat_fields:
            report_lines.append(f"- **Chosen default:** `{feat_fields[0]}` (or set explicitly)")
        report_lines.append("")

        # Action chunks
        if "predicted_action_chunk" in fields:
            report_lines.append("## 4. Action Chunks")
            report_lines.append("")
            report_lines.append("- **Present:** yes")
            report_lines.append(f"- **Shape:** {fields['predicted_action_chunk']}")
        else:
            report_lines.append("## 4. Action Chunks")
            report_lines.append("")
            report_lines.append("- **Present:** no (ACE used chunk_change proxy on available data or zeros)")
        report_lines.append("")

    report_lines.append("## 5. ACE Mode Used")
    report_lines.append("")
    report_lines.append(
        "- **Mode:** `chunk_change` (proxy: magnitude of chunk-to-chunk change over rolling window)"
    )
    report_lines.append(
        "- True runtime multi-sample ACE requires ACT to support multiple chunk samples per input."
    )
    report_lines.append("")

    # Supervised metrics
    if args.supervised_dir:
        sup_path = Path(args.supervised_dir)
        metrics_path = sup_path / "metrics.json"
        metrics = load_json(metrics_path)
        report_lines.append("## 6. Supervised Model Metrics")
        report_lines.append("")
        bva = metrics.get("best_val_auroc")
        report_lines.append(f"- **best_val_auroc:** {bva:.4f}" if isinstance(bva, (int, float)) else f"- **best_val_auroc:** N/A")
        test_m = metrics.get("test_metrics", {})
        for k, v in test_m.items():
            if isinstance(v, float):
                report_lines.append(f"- **test_{k}:** {v:.4f}")
            else:
                report_lines.append(f"- **test_{k}:** {v}")
        report_lines.append("")

    # FIPER metrics
    if args.fiper_dir:
        fiper_path = Path(args.fiper_dir)
        fiper_json = fiper_path / "fiper_results.json"
        fiper_results = load_json(fiper_json)
        report_lines.append("## 7. FIPER Baseline Metrics")
        report_lines.append("")
        for k in [
            "alarm_precision",
            "alarm_recall",
            "false_alarm_rate",
            "pct_failed_eps_with_alarm",
            "pct_success_eps_false_alarm",
            "lead_time_mean",
            "lead_time_median",
        ]:
            if k in fiper_results:
                v = fiper_results[k]
                if isinstance(v, float):
                    report_lines.append(f"- **{k}:** {v:.4f}")
                else:
                    report_lines.append(f"- **{k}:** {v}")
        report_lines.append("")

    report_lines.append("## 8. Caveats and Limitations")
    report_lines.append("")
    report_lines.append("- ACE uses chunk_change proxy; true sample-dispersion ACE needs multi-sample inference.")
    report_lines.append("- Calibration assumes successful episodes are nominal; distribution shift may affect thresholds.")
    report_lines.append("- Episode-boundary effects in ACE when concatenating multiple episodes.")
    report_lines.append("")

    report_lines.append("## 9. Recommended Next Steps")
    report_lines.append("")
    report_lines.append("- Integrate failure predictor into online rollout loop for self-correction.")
    report_lines.append("- Add multi-sample ACT inference if available for true ACE.")
    report_lines.append("- Scale up rollout collection for more robust calibration.")
    report_lines.append("")

    report = "\n".join(report_lines)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
