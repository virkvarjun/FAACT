"""Episode-level train/val/test splits for failure prediction.

Splits by unique episode_id to avoid timestep leakage across splits.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def create_episode_splits(
    episode_ids: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create episode-level train/val/test masks.

    Args:
        episode_ids: (N,) array of episode IDs per timestep.
        train_frac: Fraction of episodes for training.
        val_frac: Fraction of episodes for validation.
        test_frac: Fraction of episodes for test.
        seed: Random seed for reproducibility.

    Returns:
        (train_mask, val_mask, test_mask) each of shape (N,) bool
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got train={train_frac} val={val_frac} test={test_frac}"
        )

    # Split by episode ID so all timesteps of one episode stay in same split (no leakage)
    unique_episodes = np.unique(episode_ids)
    n_episodes = len(unique_episodes)

    if n_episodes == 0:
        raise ValueError("No unique episodes found")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_episodes)

    n_train = max(1, int(n_episodes * train_frac))
    n_val = max(0, int(n_episodes * val_frac))
    n_test = n_episodes - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n_episodes - n_train

    train_eps = set(perm[:n_train])
    val_eps = set(perm[n_train : n_train + n_val])
    test_eps = set(perm[n_train + n_val :])
    # Masks: bool per timestep, True if that timestep's episode is in split
    train_mask = np.array([e in train_eps for e in episode_ids])
    val_mask = np.array([e in val_eps for e in episode_ids])
    test_mask = np.array([e in test_eps for e in episode_ids])

    return train_mask, val_mask, test_mask


def split_summary(
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute and log split statistics."""
    def _stats(mask: np.ndarray, name: str) -> dict[str, Any]:
        n = int(mask.sum())
        lab = labels[mask]
        n_pos = int((lab > 0.5).sum())
        n_neg = n - n_pos
        n_eps = len(np.unique(episode_ids[mask]))
        return {
            f"{name}_n_samples": n,
            f"{name}_n_episodes": n_eps,
            f"{name}_n_positive": n_pos,
            f"{name}_n_negative": n_neg,
        }

    stats = {}
    stats.update(_stats(train_mask, "train"))
    stats.update(_stats(val_mask, "val"))
    stats.update(_stats(test_mask, "test"))

    for split in ["train", "val", "test"]:
        n_pos = stats[f"{split}_n_positive"]
        n_neg = stats[f"{split}_n_negative"]
        n = stats[f"{split}_n_samples"]
        logger.info(
            f"  {split}: {n} samples, {stats[f'{split}_n_episodes']} episodes, "
            f"+={n_pos} -={n_neg}"
        )
        if n > 0 and (n_pos == 0 or n_neg == 0):
            logger.warning(
                f"  {split} has no {'positives' if n_pos == 0 else 'negatives'}!"
            )

    return stats
