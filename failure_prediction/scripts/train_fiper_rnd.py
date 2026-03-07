#!/usr/bin/env python
"""Train RND (Random Network Distillation) on policy embeddings.

Trains only on successful episodes for FIPER-style OOD detection.
Example mock:
    python -m failure_prediction.scripts.train_fiper_rnd \
        --mock_data --feature_dim 256 --output_dir failure_prediction_runs/mock_fiper_rnd --epochs 5

Example real:
    python -m failure_prediction.scripts.train_fiper_rnd \
        --processed_dir failure_dataset/transfer_cube/processed \
        --feature_field feat_decoder_mean \
        --output_dir failure_prediction_runs/transfer_cube_fiper_rnd \
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import load_processed_dataset, load_failure_dataset, get_available_feature_fields
from failure_prediction.data.splits import create_episode_splits
from failure_prediction.models.rnd import RNDModule, compute_rnd_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train RND for FIPER")
    p.add_argument("--mock_data", action="store_true")
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--feature_field", type=str, default="feat_decoder_mean")
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/fiper_rnd")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dims", type=str, default="256,256")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--num_mock_episodes", type=int, default=50)
    p.add_argument("--timesteps_per_episode", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mock_data and not args.processed_dir:
        logger.error("Either --mock_data or --processed_dir is required")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mock_data:
        features, _, episode_ids, _, input_dim, meta = load_failure_dataset(
            mock=True,
            mock_num_episodes=args.num_mock_episodes,
            mock_timesteps_per_episode=args.timesteps_per_episode,
            mock_feature_dim=args.feature_dim,
            mock_seed=args.seed,
        )
        success_mask = np.ones(len(features), dtype=bool)
        success_mask[: len(features) // 3] = False
        np.random.default_rng(args.seed).shuffle(success_mask)
    else:
        data, meta = load_processed_dataset(args.processed_dir)
        if args.feature_field not in data:
            raise ValueError(
                f"Feature field '{args.feature_field}' not found. "
                f"Available: {get_available_feature_fields(data)}"
            )
        features = np.asarray(data[args.feature_field], dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
        if "success" in data:
            success_mask = data["success"].astype(bool)
        else:
            success_mask = ~(data["episode_failed"].astype(bool))
        input_dim = features.shape[1]

    success_only = features[success_mask]
    if len(success_only) == 0:
        raise ValueError(
            "No successful episode timesteps. RND requires successful rollouts for training."
        )

    logger.info(f"RND training on {len(success_only)} successful timesteps, input_dim={input_dim}")

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    model = RNDModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        seed=args.seed,
    ).to(args.device)

    X = torch.from_numpy(success_only).float()
    train_ds = TensorDataset(X)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(args.device)
            optimizer.zero_grad()
            loss = model.loss(xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X)
        logger.info(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}")

    train_scores = compute_rnd_scores(model, success_only, args.device)

    config = {
        "mock_data": args.mock_data,
        "processed_dir": str(args.processed_dir) if args.processed_dir else None,
        "feature_field": args.feature_field,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    torch.save(model.state_dict(), out_dir / "rnd_model.pt")
    np.savez(out_dir / "train_rnd_scores.npz", scores=train_scores)
    logger.info(f"Saved RND model and scores to {out_dir}")


if __name__ == "__main__":
    main()
