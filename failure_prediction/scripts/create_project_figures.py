#!/usr/bin/env python
"""Create project figures for failure prediction / failure_within_k explanation.

Generates:
1. Rollout timeline strip - success vs failed episode with failure_within_k shading
2. ACT pipeline diagram - observation → ACT → action chunk + embeddings
3. Feature table - embedding dims, action chunks, labels
4. Episode distribution - bar/donut chart (18 success, 12 failure, 30 total)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))


def _setup_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        sys.exit(1)


def fig1_rollout_timeline(
    output_dir: Path,
    num_steps_success: int = 50,
    num_steps_failed: int = 55,
    failure_horizon_k: int = 10,
    frame_height: float = 1.2,
) -> None:
    """Rollout timeline strip: success + failed episode, last K steps shaded red."""
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), height_ratios=[1, 0.4])
    ax_top, ax_label = axes

    # --- Top: two rows of "frames" ---
    n_succ, n_fail = num_steps_success, num_steps_failed
    step_w = 0.9

    # Success episode row
    y_succ = 1.0
    for t in range(n_succ):
        rect = plt.Rectangle((t * step_w, y_succ - frame_height), step_w * 0.85, frame_height * 0.9,
                             facecolor="steelblue", edgecolor="white", linewidth=0.5)
        ax_top.add_patch(rect)
    ax_top.text(-1.5, y_succ - frame_height / 2, "Success", va="center", fontsize=11, fontweight="bold")
    ax_top.set_xlim(-3, max(n_succ, n_fail) * step_w + 2)
    ax_top.set_ylim(-2, 2.5)
    ax_top.axis("off")

    # Failed episode row: last K steps before failure in red
    y_fail = -0.5
    failure_at = n_fail
    last_k_start = failure_at - failure_horizon_k
    for t in range(n_fail):
        x = t * step_w
        rect = plt.Rectangle((x, y_fail - frame_height), step_w * 0.85, frame_height * 0.9,
                             facecolor="crimson" if last_k_start <= t < failure_at else "steelblue",
                             edgecolor="white", linewidth=0.5)
        ax_top.add_patch(rect)
    ax_top.text(-1.5, y_fail - frame_height / 2, "Failure", va="center", fontsize=11, fontweight="bold",
                color="crimson")
    # Red shade label
    mid_k = (last_k_start + failure_at) / 2
    ax_top.annotate(f"last {failure_horizon_k} steps\nbefore failure", xy=(mid_k * step_w, y_fail),
                    fontsize=9, ha="center", color="darkred", fontweight="bold")
    ax_top.annotate("", xy=((last_k_start + 0.5) * step_w, y_fail - 0.1),
                    xytext=((failure_at - 0.5) * step_w, y_fail - 0.1),
                    arrowprops=dict(arrowstyle="<->", color="darkred", lw=2))

    # --- Bottom: timestep labels ---
    ax_label.set_xlim(ax_top.get_xlim())
    ax_label.set_ylim(0, 1)
    ax_label.axis("off")
    step_interval = max(1, (max(n_succ, n_fail) // 12))
    for t in range(0, max(n_succ, n_fail) + 1, step_interval):
        x = t * step_w
        ax_label.text(x, 0.7, str(t), ha="center", fontsize=8)
    ax_label.text(-1.5, 0.5, "timestep", fontsize=10, style="italic")

    plt.suptitle("Rollout Timeline: failure_within_k Labeling", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "rollout_timeline_strip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'rollout_timeline_strip.png'}")


def fig2_act_pipeline(output_dir: Path) -> None:
    """ACT pipeline: observation → ACT → action chunk + embeddings."""
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Boxes
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def box(x, y, w, h, label, color="lightblue"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", lw=1.5)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, wrap=True)

    box(0.5, 2, 1.5, 1.2, "observation\n(image + state)")
    box(3.5, 1.8, 2.2, 1.6, "ACT\n(action chunk transformer)")
    box(6.8, 2.2, 1.8, 1.2, "action chunk\n(100×14)")
    box(6.8, 0.6, 1.8, 1.2, "internal\nembeddings")

    # Arrows
    ax.annotate("", xy=(3.3, 2.4), xytext=(2.0, 2.6), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(6.7, 2.8), xytext=(5.7, 2.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(6.7, 1.2), xytext=(5.7, 1.8), arrowprops=dict(arrowstyle="->", lw=2))

    # Feature labels
    ax.text(8.2, 2.8, "feat_decoder_mean\n(512-d)", fontsize=9, va="center")
    ax.text(8.2, 1.8, "feat_encoder_latent_token\n(512-d)", fontsize=9, va="center")
    ax.text(8.2, 0.9, "feat_latent_sample\n(32-d)", fontsize=9, va="center")
    ax.plot([7.6, 7.9], [2.2, 2.2], "k-", lw=1)
    ax.plot([7.6, 7.9], [1.5, 1.5], "k-", lw=1)
    ax.plot([7.6, 7.9], [0.7, 0.7], "k-", lw=1)

    ax.set_title("ACT Pipeline: Mining Policy Internal State for Failure Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "act_pipeline_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'act_pipeline_diagram.png'}")


def fig3_feature_table(output_dir: Path) -> None:
    """Feature table / block diagram: embeddings, action chunks, labels."""
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    def block(x, y, w, h, title, items, color="lavender"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="gray", lw=1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", fontsize=11, fontweight="bold")
        for i, line in enumerate(items):
            ax.text(x + 0.2, y + h - 0.7 - i * 0.5, line, fontsize=10)

    block(0.3, 3.5, 2.2, 2.2, "Embeddings", ["feat_decoder_mean: 512-d", "feat_encoder_latent: 512-d", "feat_latent_sample: 32-d"])
    block(2.8, 3.5, 2.2, 2.2, "Action Chunks", ["100 × 14", "(chunk_len × action_dim)"])
    block(5.3, 3.5, 1.5, 2.2, "Labels", ["failure within", "10 steps"])

    ax.text(3.5, 2.0, "→ MLP predictor → risk score", ha="center", fontsize=11, style="italic")
    ax.set_title("Technical Summary: Features & Labels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'feature_table.png'}")


def fig4_episode_distribution(output_dir: Path, n_success: int = 18, n_failure: int = 12) -> None:
    """Episode distribution: bar and donut chart (18 success, 12 failure, 30 total)."""
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    total = n_success + n_failure

    # Bar chart
    ax_bar = axes[0]
    ax_bar.bar(["Success", "Failure"], [n_success, n_failure], color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax_bar.set_ylabel("Episodes")
    ax_bar.set_title("Episode Outcomes")
    for i, v in enumerate([n_success, n_failure]):
        ax_bar.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Donut chart
    ax_donut = axes[1]
    sizes = [n_success, n_failure]
    colors = ["#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax_donut.pie(
        sizes, labels=["Success", "Failure"], autopct="%1.0f%%",
        colors=colors, startangle=90, wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax_donut.set_title(f"Dataset: {total} Episodes")

    plt.suptitle("Transfer Cube Dataset Distribution", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "episode_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'episode_distribution.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/project_figures")
    p.add_argument("--failure_horizon", type=int, default=10)
    p.add_argument("--n_success", type=int, default=18)
    p.add_argument("--n_failure", type=int, default=12)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_rollout_timeline(out_dir, failure_horizon_k=args.failure_horizon)
    fig2_act_pipeline(out_dir)
    fig3_feature_table(out_dir)
    fig4_episode_distribution(out_dir, n_success=args.n_success, n_failure=args.n_failure)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
