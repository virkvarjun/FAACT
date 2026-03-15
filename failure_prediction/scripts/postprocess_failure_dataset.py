#!/usr/bin/env python
"""Postprocess raw episode files into a dense timestep dataset with failure labels.

Loads raw .npz episode files from the collection script, computes per-timestep
failure-within-horizon labels, and saves a processed dataset suitable for
training a failure prediction model.

Example:
    python -m failure_prediction.scripts.postprocess_failure_dataset \
        --input_dir failure_dataset/transfer_cube/raw \
        --output_dir failure_dataset/transfer_cube/processed \
        --failure_horizon 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.utils.failure_dataset_logger import FailureDatasetLogger
from failure_prediction.utils.failure_labeling import label_failure_windows


def parse_args():
    p = argparse.ArgumentParser(description="Postprocess raw episodes into labeled timestep dataset")
    p.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing raw episode .npz files")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for processed dataset")
    p.add_argument("--failure_horizon", type=int, default=10,
                    help="K for failure_within_k labels")
    p.add_argument("--near_failure_horizon", type=int, default=None,
                    help="Horizon for softer near_failure label (default: 2*K)")
    return p.parse_args()


def load_all_episodes(input_dir: Path) -> list[dict]:
    """Load all raw episode .npz files. Each returns {meta, arrays}."""
    files = sorted(input_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode files found in {input_dir}")

    episodes = []
    for f in files:
        ep = FailureDatasetLogger.load_episode(f)
        ep["source_file"] = str(f)
        episodes.append(ep)

    logger.info(f"Loaded {len(episodes)} episodes from {input_dir}")
    return episodes


# Call label_failure_windows per episode; copy feat_* from raw. Output: timestep_dataset.npz
def process_episodes(
    episodes: list[dict],
    failure_horizon: int,
    near_failure_horizon: int | None,
) -> dict:
    all_rows = {
        "episode_id": [],
        "timestep": [],
        "success": [],
        "episode_failed": [],
        "failure_within_k": [],
        "steps_to_failure": [],
        "near_failure": [],
        "reward": [],
        "done": [],
        "terminated": [],
        "truncated": [],
        "chunk_length": [],
        "chunk_step_idx": [],
        "new_chunk_generated": [],
    }

    array_fields = {}

    for ep in episodes:
        meta = ep["meta"]
        arrays = ep["arrays"]

        ep_id = meta["episode_id"]
        num_steps = meta["num_steps"]
        ep_failed = meta.get("episode_failed", not meta.get("success", False))
        terminal_step = meta.get("terminal_step", meta.get("num_steps", num_steps) - 1)

        # Per-timestep labels: failure_within_k, steps_to_failure, near_failure
        labels = label_failure_windows(
            num_steps=num_steps,
            episode_failed=ep_failed,
            terminal_step=terminal_step,
            failure_horizon=failure_horizon,
            near_failure_horizon=near_failure_horizon,
        )

        all_rows["episode_id"].extend([ep_id] * num_steps)
        all_rows["timestep"].extend(range(num_steps))
        all_rows["success"].extend([int(meta.get("success", False))] * num_steps)
        all_rows["episode_failed"].extend(labels["episode_failed"].tolist())
        all_rows["failure_within_k"].extend(labels["failure_within_k"].tolist())
        all_rows["steps_to_failure"].extend(labels["steps_to_failure"].tolist())
        all_rows["near_failure"].extend(labels["near_failure"].tolist())

        if "reward" in arrays:
            all_rows["reward"].extend(arrays["reward"].tolist())
        else:
            all_rows["reward"].extend([0.0] * num_steps)

        if "done" in arrays:
            all_rows["done"].extend(arrays["done"].astype(int).tolist())
        else:
            dones = [0] * num_steps
            dones[-1] = 1
            all_rows["done"].extend(dones)

        if "terminated" in arrays:
            all_rows["terminated"].extend(arrays["terminated"].astype(int).tolist())
        else:
            all_rows["terminated"].extend([0] * num_steps)

        if "truncated" in arrays:
            all_rows["truncated"].extend(arrays["truncated"].astype(int).tolist())
        else:
            all_rows["truncated"].extend([0] * num_steps)

        if "chunk_length" in arrays:
            all_rows["chunk_length"].extend(arrays["chunk_length"].tolist())
        else:
            all_rows["chunk_length"].extend([-1] * num_steps)

        if "chunk_step_idx" in arrays:
            all_rows["chunk_step_idx"].extend(arrays["chunk_step_idx"].tolist())
        else:
            all_rows["chunk_step_idx"].extend([0] * num_steps)

        if "new_chunk_generated" in arrays:
            all_rows["new_chunk_generated"].extend(arrays["new_chunk_generated"].astype(int).tolist())
        else:
            all_rows["new_chunk_generated"].extend([0] * num_steps)

        # Copy embedding arrays (feat_decoder_mean, etc.) to output
        for key in arrays:
            if key in ("reward", "done", "terminated", "truncated", "timestep", "success", "chunk_length", "chunk_step_idx",
                       "new_chunk_generated", "_meta_json"):
                continue
            if arrays[key].shape[0] != num_steps:
                continue
            if arrays[key].ndim == 1:
                # 1D scalars per step -> extend to all_rows
                if arrays[key].dtype.kind in {"U", "S", "O"}:
                    result_key = key
                    if result_key not in all_rows:
                        all_rows[result_key] = []
                    all_rows[result_key].extend(arrays[key].tolist())
                else:
                    result_key = key
                    if result_key not in all_rows:
                        all_rows[result_key] = []
                    all_rows[result_key].extend(arrays[key].tolist())
                continue
            # 2D+ arrays (feat_decoder_mean etc.) -> concatenate across episodes later
            if key not in array_fields:
                array_fields[key] = []
            array_fields[key].append(arrays[key])

    result = {}
    for key, vals in all_rows.items():
        result[key] = np.array(vals)

    for key, val_list in array_fields.items():
        try:
            result[key] = np.concatenate(val_list, axis=0)  # stack episodes vertically
        except ValueError:
            logger.warning(f"Skipping array field '{key}' due to inconsistent shapes")

    return result


def compute_stats(dataset: dict, episodes: list[dict]) -> dict:
    """Compute summary stats: episode counts, class balance, avg lengths."""
    n_episodes = len(episodes)
    n_success = sum(1 for ep in episodes if ep["meta"].get("success", False))
    n_failed = n_episodes - n_success
    total_timesteps = len(dataset["episode_id"])

    n_positive = int(dataset["failure_within_k"].sum())
    n_negative = total_timesteps - n_positive

    success_lengths = [ep["meta"]["num_steps"] for ep in episodes if ep["meta"].get("success", False)]
    failure_lengths = [ep["meta"]["num_steps"] for ep in episodes if not ep["meta"].get("success", False)]

    stats = {
        "total_episodes": n_episodes,
        "successful_episodes": n_success,
        "failed_episodes": n_failed,
        "success_rate": n_success / max(n_episodes, 1),
        "total_timesteps": total_timesteps,
        "failure_within_k_positive": n_positive,
        "failure_within_k_negative": n_negative,
        "class_balance": n_positive / max(total_timesteps, 1),
        "avg_episode_length": total_timesteps / max(n_episodes, 1),
        "avg_success_length": float(np.mean(success_lengths)) if success_lengths else 0,
        "avg_failure_length": float(np.mean(failure_lengths)) if failure_lengths else 0,
    }

    for key, val in dataset.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            stats[f"shape_{key}"] = list(val.shape)

    return stats


def run_postprocessing(args):
    """Main postprocessing pipeline."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_all_episodes(input_dir)

    logger.info(f"Processing with failure_horizon={args.failure_horizon}")
    dataset = process_episodes(
        episodes,
        failure_horizon=args.failure_horizon,
        near_failure_horizon=args.near_failure_horizon,
    )

    stats = compute_stats(dataset, episodes)

    dataset_path = output_dir / "timestep_dataset.npz"
    np.savez_compressed(dataset_path, **dataset)
    logger.info(f"Saved processed dataset to {dataset_path}")

    metadata = {
        "failure_horizon": args.failure_horizon,
        "near_failure_horizon": args.near_failure_horizon or 2 * args.failure_horizon,
        "input_dir": str(input_dir),
        "stats": stats,
        "fields": {key: {"shape": list(val.shape), "dtype": str(val.dtype)}
                   for key, val in dataset.items()},
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info("=" * 60)
    logger.info("Postprocessing Summary")
    logger.info("=" * 60)
    logger.info(f"  Total episodes:          {stats['total_episodes']}")
    logger.info(f"  Successful:              {stats['successful_episodes']}")
    logger.info(f"  Failed:                  {stats['failed_episodes']}")
    logger.info(f"  Success rate:            {stats['success_rate'] * 100:.1f}%")
    logger.info(f"  Total timesteps:         {stats['total_timesteps']}")
    logger.info(f"  failure_within_k (+):    {stats['failure_within_k_positive']}")
    logger.info(f"  failure_within_k (-):    {stats['failure_within_k_negative']}")
    logger.info(f"  Class balance:           {stats['class_balance']:.4f}")
    logger.info(f"  Avg episode length:      {stats['avg_episode_length']:.1f}")
    logger.info(f"  Avg success length:      {stats['avg_success_length']:.1f}")
    logger.info(f"  Avg failure length:      {stats['avg_failure_length']:.1f}")

    for key in dataset:
        if isinstance(dataset[key], np.ndarray) and dataset[key].ndim >= 2:
            logger.info(f"  {key}: shape={dataset[key].shape}")

    if stats["failed_episodes"] == 0:
        logger.warning("No failed episodes! Cannot generate positive failure_within_k labels.")
    if stats["successful_episodes"] == 0:
        logger.warning("No successful episodes! Dataset will have no negative examples.")

    return stats


if __name__ == "__main__":
    args = parse_args()
    run_postprocessing(args)
