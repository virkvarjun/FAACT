#!/usr/bin/env python
"""Visualize the ACT embedding space used for failure prediction.

Reduces high-dimensional embeddings (feat_decoder_mean, etc.) to 2D via t-SNE or PCA,
then scatter-plots colored by failure labels and episode outcome.

Example:
    python -m failure_prediction.scripts.visualize_embedding_space \\
        --processed_dir failure_dataset/transfer_cube/processed \\
        --feature_field feat_decoder_mean \\
        --output_dir failure_prediction_runs/embedding_plots \\
        --method tsne
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import (
    load_processed_dataset,
    load_failure_dataset,
    get_available_feature_fields,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--mock", action="store_true", help="Use synthetic data when processed_dir has no data")
    p.add_argument("--feature_field", type=str, default="feat_decoder_mean")
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/embedding_plots")
    p.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca", "umap"])
    p.add_argument("--max_samples", type=int, default=3000, help="Subsample for speed (t-SNE is slow)")
    p.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (5-50 typical)")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        sys.exit(1)

    # Load real or mock data
    if args.mock or not args.processed_dir:
        features, labels, episode_ids, timesteps, _, _ = load_failure_dataset(
            mock=True,
            mock_num_episodes=30,
            mock_timesteps_per_episode=50,
            mock_feature_dim=512,
            mock_positive_ratio=0.3,
        )
        ep_failed_per_ep = np.array(
            [1.0 if np.any(labels[episode_ids == e] > 0.5) else 0.0 for e in np.unique(episode_ids)]
        )
        ep_failed_per_step = ep_failed_per_ep[episode_ids]
        data = {
            args.feature_field: features,
            "failure_within_k": labels,
            "episode_failed": ep_failed_per_step,
        }
        print("Using mock synthetic data for visualization")
    else:
        data, _ = load_processed_dataset(Path(args.processed_dir))

    if args.feature_field not in data:
        available = get_available_feature_fields(data)
        print(f"Feature '{args.feature_field}' not found. Available: {available}")
        sys.exit(1)

    X = np.asarray(data[args.feature_field], dtype=np.float32)
    labels = np.asarray(data["failure_within_k"]).ravel()
    ep_failed = np.asarray(data["episode_failed"]).ravel()

    n = len(X)
    if n > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, args.max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
        ep_failed = ep_failed[idx]
        n = args.max_samples
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Dimensionality reduction
    if args.method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        print(f"PCA: explained variance ratio = {reducer.explained_variance_ratio_.sum():.3f}")
    elif args.method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(args.perplexity, max(5, n // 4))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_2d = reducer.fit_transform(X)
        print("t-SNE done")
    elif args.method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            print("UMAP done")
        except ImportError:
            print("UMAP not installed. Use: pip install umap-learn")
            sys.exit(1)
    else:
        raise ValueError(args.method)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: colored by failure_within_k
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = labels > 0.5
    neg = ~pos
    if neg.any():
        ax.scatter(X_2d[neg, 0], X_2d[neg, 1], c="C0", alpha=0.4, s=10, label="No failure soon")
    if pos.any():
        ax.scatter(X_2d[pos, 0], X_2d[pos, 1], c="C3", alpha=0.6, s=15, label="Failure within k")
    ax.set_xlabel(f"{args.method.upper()} 1")
    ax.set_ylabel(f"{args.method.upper()} 2")
    ax.set_title(f"Embedding space ({args.feature_field}) - colored by failure_within_k")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_by_failure_within_k.png", dpi=120)
    plt.close()

    # Plot 2: colored by episode outcome
    fig, ax = plt.subplots(figsize=(8, 6))
    fail_eps = ep_failed > 0.5
    succ_eps = ~fail_eps
    if succ_eps.any():
        ax.scatter(X_2d[succ_eps, 0], X_2d[succ_eps, 1], c="C2", alpha=0.4, s=10, label="Success episode")
    if fail_eps.any():
        ax.scatter(X_2d[fail_eps, 0], X_2d[fail_eps, 1], c="C1", alpha=0.6, s=15, label="Failed episode")
    ax.set_xlabel(f"{args.method.upper()} 1")
    ax.set_ylabel(f"{args.method.upper()} 2")
    ax.set_title(f"Embedding space ({args.feature_field}) - colored by episode outcome")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_by_episode_outcome.png", dpi=120)
    plt.close()

    # Plot 3: combined view (episode outcome + failure_within_k overlay)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[succ_eps & neg, 0], X_2d[succ_eps & neg, 1], c="C2", alpha=0.3, s=8, label="Success, no failure soon")
    ax.scatter(X_2d[succ_eps & pos, 0], X_2d[succ_eps & pos, 1], c="C0", alpha=0.5, s=12, label="Success, failure soon (near end)")
    ax.scatter(X_2d[fail_eps & neg, 0], X_2d[fail_eps & neg, 1], c="gray", alpha=0.4, s=8, label="Failed, no alarm yet")
    ax.scatter(X_2d[fail_eps & pos, 0], X_2d[fail_eps & pos, 1], c="C3", alpha=0.7, s=15, label="Failed, failure soon")
    ax.set_xlabel(f"{args.method.upper()} 1")
    ax.set_ylabel(f"{args.method.upper()} 2")
    ax.set_title(f"Embedding space ({args.feature_field}) - outcome × failure_within_k")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_combined.png", dpi=120)
    plt.close()

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
