#!/usr/bin/env python
"""Train failure predictor MLP on processed failure dataset.

Supports mock mode (synthetic data) and real processed dataset mode.
Example mock:
    python -m failure_prediction.scripts.train_failure_predictor \
        --mock_data --output_dir failure_prediction_runs/mock_run \
        --feature_dim 256 --num_mock_episodes 50 --timesteps_per_episode 40 --epochs 5

Example real:
    python -m failure_prediction.scripts.train_failure_predictor \
        --processed_dir failure_dataset/transfer_cube/processed \
        --feature_field feat_policy_embedding --label_field failure_within_k \
        --output_dir failure_prediction_runs/transfer_cube_risk_mlp \
        --epochs 20 --batch_size 256
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.data.failure_dataset import load_failure_dataset
from failure_prediction.data.splits import create_episode_splits, split_summary
from failure_prediction.models.failure_predictor import FailurePredictorMLP
from failure_prediction.utils.eval_metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train failure predictor MLP")
    p.add_argument("--mock_data", action="store_true", help="Use synthetic mock dataset")
    p.add_argument("--processed_dir", type=str, default=None, help="Processed dataset directory")
    p.add_argument("--feature_field", type=str, default="feat_decoder_mean")
    p.add_argument("--label_field", type=str, default="failure_within_k")
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/default")
    p.add_argument("--run_name", type=str, default=None, help="Subdir under output_dir")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dims", type=str, default="256,128", help="Comma-separated")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=None, help="BCE pos_weight for imbalance")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Mock-only
    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--num_mock_episodes", type=int, default=50)
    p.add_argument("--timesteps_per_episode", type=int, default=40)
    p.add_argument("--mock_positive_ratio", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mock_data and not args.processed_dir:
        logger.error("Either --mock_data or --processed_dir is required")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load_failure_dataset: processed dir or mock. Episode-level splits to avoid leakage.
    features, labels, episode_ids, timesteps, input_dim, metadata = load_failure_dataset(
        processed_dir=args.processed_dir,
        feature_field=args.feature_field,
        label_field=args.label_field,
        mock=args.mock_data,
        mock_num_episodes=args.num_mock_episodes,
        mock_timesteps_per_episode=args.timesteps_per_episode,
        mock_feature_dim=args.feature_dim,
        mock_positive_ratio=args.mock_positive_ratio,
        mock_seed=args.seed,
    )

    # Episode-level splits (not timestep) to avoid train/val leakage
    train_mask, val_mask, test_mask = create_episode_splits(
        episode_ids,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    split_stats = split_summary(labels, train_mask, val_mask, test_mask, episode_ids)

    # Convert to tensors; train only on train split
    X_train = torch.from_numpy(features[train_mask]).float()
    y_train = torch.from_numpy(labels[train_mask]).float().unsqueeze(1)
    X_val = torch.from_numpy(features[val_mask]).float()
    y_val = torch.from_numpy(labels[val_mask]).float()
    X_test = torch.from_numpy(features[test_mask]).float()
    y_test = torch.from_numpy(labels[test_mask]).float()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    model = FailurePredictorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = args.pos_weight
    if pos_weight is None and not args.mock_data:  # auto pos_weight for class imbalance
        n_pos = int((labels[train_mask] > 0.5).sum())
        n_neg = int(train_mask.sum()) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = float(n_neg) / n_pos
    if pos_weight is not None:
        pos_weight_t = torch.tensor([pos_weight], device=args.device)
    else:
        pos_weight_t = None

    out_dir = Path(args.output_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "mock_data": args.mock_data,
        "processed_dir": str(args.processed_dir) if args.processed_dir else None,
        "feature_field": args.feature_field,
        "label_field": args.label_field,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "pos_weight": pos_weight,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    best_val_auroc = -1.0
    metrics_history = []  # AUROC/AUPRC per epoch

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(args.device), yb.to(args.device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = F.binary_cross_entropy_with_logits(  # BCE, no sigmoid (logits in)
                logits, yb.squeeze(-1), pos_weight=pos_weight_t
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val.to(args.device)).cpu().numpy()
        val_metrics = compute_binary_metrics(logits_val, y_val.numpy())
        val_loss = float(
            F.binary_cross_entropy_with_logits(
                torch.from_numpy(logits_val).float(),
                y_val,
            ).item()
        )

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "last_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val.to(args.device)).cpu().numpy()
        logits_test = model(X_test.to(args.device)).cpu().numpy()

    probs_val = 1.0 / (1.0 + np.exp(-np.clip(logits_val, -500, 500)))
    probs_test = 1.0 / (1.0 + np.exp(-np.clip(logits_test, -500, 500)))

    # Save predictions for analysis (episode_ids, timesteps allow per-episode breakdown)

    val_ep = episode_ids[val_mask]
    val_ts = timesteps[val_mask]
    test_ep = episode_ids[test_mask]
    test_ts = timesteps[test_mask]

    np.savez(
        out_dir / "val_predictions.npz",
        logits=logits_val,
        probs=probs_val,
        labels=y_val.numpy(),
        episode_ids=val_ep,
        timesteps=val_ts,
    )
    np.savez(
        out_dir / "test_predictions.npz",
        logits=logits_test,
        probs=probs_test,
        labels=y_test.numpy(),
        episode_ids=test_ep,
        timesteps=test_ts,
    )

    test_metrics = compute_binary_metrics(logits_test, y_test.numpy())
    final_metrics = {
        "metrics_history": metrics_history,
        "best_val_auroc": best_val_auroc,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(
        f"Done. Best val AUROC={best_val_auroc:.4f}. "
        f"Test AUROC={test_metrics['auroc']:.4f} F1={test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
