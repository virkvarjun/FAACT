"""Feature field statistics for processed failure datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from failure_prediction.data.failure_dataset import LABEL_AND_META_KEYS, get_available_feature_fields

logger = logging.getLogger(__name__)


def compute_feature_stats(arr: np.ndarray) -> dict[str, Any]:
    """Compute per-field statistics for a numeric array (shape, dtype, min/max, mean, std)."""
    arr = np.asarray(arr)
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_samples": arr.shape[0] if arr.ndim >= 1 else 0,
    }
    if arr.size == 0:
        stats["has_nan"] = False
        stats["has_inf"] = False
        stats["min"] = np.nan
        stats["max"] = np.nan
        stats["mean"] = np.nan
        stats["std"] = np.nan
        return stats

    # Flatten to (n_samples, n_features) for per-sample stats
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    stats["has_nan"] = bool(np.isnan(flat).any())
    stats["has_inf"] = bool(np.isinf(flat).any())
    finite = np.isfinite(flat)
    if finite.any():
        stats["min"] = float(np.min(flat[finite]))
        stats["max"] = float(np.max(flat[finite]))
        stats["mean"] = float(np.mean(flat[finite]))
        stats["std"] = float(np.std(flat[finite]))
    else:
        stats["min"] = stats["max"] = stats["mean"] = stats["std"] = np.nan
    if flat.shape[1] > 1:
        stats["dim"] = flat.shape[1]
    return stats


def inspect_dataset_features(dataset: dict) -> dict[str, Any]:
    """Inspect all feature fields (feat_*, etc.) and compute stats. Used for debugging/visualization."""
    feature_fields = get_available_feature_fields(dataset)
    result: dict[str, Any] = {
        "feature_fields": feature_fields,
        "field_stats": {},
        "embedding_candidates": [f for f in feature_fields if f.startswith("feat_")],
    }
    for key in feature_fields:
        arr = dataset[key]
        result["field_stats"][key] = compute_feature_stats(arr)
    return result
