"""Dataset loader for processed failure prediction data.

Supports real processed .npz datasets and mock synthetic mode for testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Labels/metadata; excluded when picking feature fields
LABEL_AND_META_KEYS = {
    "episode_id",
    "timestep",
    "success",
    "episode_failed",
    "failure_within_k",
    "steps_to_failure",
    "near_failure",
    "reward",
    "done",
    "terminated",
    "truncated",
    "chunk_length",
    "chunk_step_idx",
    "new_chunk_generated",
}


def get_available_feature_fields(dataset: dict) -> list[str]:
    """Return feature fields (feat_*, etc.) excluding labels and metadata. Must be numeric, 2D or 1D-numeric."""
    candidates = []
    for key in dataset:
        if key in LABEL_AND_META_KEYS:
            continue
        arr = dataset[key]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim < 1 or arr.dtype.kind not in "bifu":
            continue
        if arr.ndim == 1 and arr.dtype.kind in "b":
            continue
        candidates.append(key)
    return sorted(candidates)


def load_processed_dataset(processed_dir: str | Path) -> tuple[dict, dict]:
    """Load processed timestep dataset and metadata from disk.

    Args:
        processed_dir: Directory containing timestep_dataset.npz and metadata.json.

    Returns:
        (dataset_dict, metadata_dict)
    """
    processed_dir = Path(processed_dir)
    dataset_path = processed_dir / "timestep_dataset.npz"
    meta_path = processed_dir / "metadata.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    data = dict(np.load(dataset_path, allow_pickle=True))

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return data, metadata


def load_failure_dataset(
    processed_dir: str | Path | None = None,
    feature_field: str = "feat_decoder_mean",
    feature_fields: list[str] | None = None,
    label_field: str = "failure_within_k",
    mock: bool = False,
    mock_num_episodes: int = 50,
    mock_timesteps_per_episode: int = 40,
    mock_feature_dim: int = 256,
    mock_positive_ratio: float = 0.3,
    mock_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Load failure prediction dataset for training.

    Args:
        processed_dir: Path to processed dataset directory (required if mock=False).
        feature_field: Single feature field (used if feature_fields is None).
        feature_fields: Multiple feature fields to concatenate (overrides feature_field).
        label_field: Name of label array (default: failure_within_k).
        mock: If True, generate synthetic data instead of loading from disk.
        mock_*: Parameters for synthetic data generation.

    Returns:
        (features, labels, episode_ids, timesteps, input_dim, metadata)
    """
    if mock:
        return _create_mock_dataset(
            num_episodes=mock_num_episodes,
            timesteps_per_episode=mock_timesteps_per_episode,
            feature_dim=mock_feature_dim,
            positive_ratio=mock_positive_ratio,
            seed=mock_seed,
        )

    if processed_dir is None:
        raise ValueError("processed_dir is required when mock=False")

    data, metadata = load_processed_dataset(processed_dir)
    fields_to_use = feature_fields if feature_fields else [feature_field]
    available = get_available_feature_fields(data)

    for f in fields_to_use:
        if f not in data:
            raise ValueError(
                f"Feature field '{f}' not found. Available: {available}"
            )

    if label_field not in data:
        raise ValueError(
            f"Label field '{label_field}' not found. "
            f"Available: {list(k for k in data if k not in LABEL_AND_META_KEYS or k == 'failure_within_k')}"
        )

    parts = []
    for f in fields_to_use:
        arr = np.asarray(data[f], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        parts.append(arr)
    features = np.concatenate(parts, axis=1)  # single or multi-field -> (N, input_dim)
    labels = np.asarray(data[label_field], dtype=np.float32)
    episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
    timesteps = np.asarray(data["timestep"], dtype=np.int64) if "timestep" in data else np.arange(len(labels))

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if labels.ndim > 1:
        labels = labels.ravel()

    n = len(labels)
    if len(features) != n or len(episode_ids) != n:
        raise ValueError(
            f"Mismatched lengths: features={len(features)}, labels={n}, episode_ids={len(episode_ids)}"
        )

    if np.any(~np.isfinite(features)):
        raise ValueError("Features contain NaN or Inf")
    if np.any(~np.isfinite(labels)):
        raise ValueError("Labels contain NaN or Inf")

    input_dim = features.shape[1]
    metadata["feature_field"] = fields_to_use[0] if len(fields_to_use) == 1 else None
    metadata["feature_fields"] = fields_to_use
    metadata["label_field"] = label_field
    metadata["input_dim"] = input_dim
    metadata["n_samples"] = n

    logger.info(
        f"Loaded dataset: {n} samples, input_dim={input_dim}, "
        f"positives={int(labels.sum())}, negatives={n - int(labels.sum())}"
    )

    return features, labels, episode_ids, timesteps, input_dim, metadata


def _create_mock_dataset(
    num_episodes: int = 50,
    timesteps_per_episode: int = 40,
    feature_dim: int = 256,
    positive_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Create synthetic dataset for testing training pipeline."""
    rng = np.random.default_rng(seed)

    total_steps = num_episodes * timesteps_per_episode
    n_positive = int(total_steps * positive_ratio)
    n_negative = total_steps - n_positive

    episode_ids = np.repeat(np.arange(num_episodes), timesteps_per_episode)
    timesteps = np.tile(np.arange(timesteps_per_episode), num_episodes)

    labels = np.zeros(total_steps, dtype=np.float32)
    labels[:n_positive] = 1.0
    rng.shuffle(labels)

    # Positive samples shifted +0.5 so model can learn separable distributions
    features = np.zeros((total_steps, feature_dim), dtype=np.float32)
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    features[neg_mask] = rng.standard_normal((n_negative, feature_dim)).astype(np.float32)
    features[pos_mask] = rng.standard_normal((n_positive, feature_dim)).astype(np.float32) + 0.5

    metadata = {
        "mock": True,
        "num_episodes": num_episodes,
        "timesteps_per_episode": timesteps_per_episode,
        "feature_dim": feature_dim,
        "input_dim": feature_dim,
        "n_samples": total_steps,
        "n_positive": int(pos_mask.sum()),
        "n_negative": int(neg_mask.sum()),
    }

    logger.info(
        f"Mock dataset: {total_steps} samples, dim={feature_dim}, "
        f"pos={int(pos_mask.sum())}, neg={int(neg_mask.sum())}"
    )

    return features, labels, episode_ids, timesteps, feature_dim, metadata
