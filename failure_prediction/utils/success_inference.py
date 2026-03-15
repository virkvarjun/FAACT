"""Utilities for inferring episode success/failure from environment signals."""

from __future__ import annotations

import numpy as np


# Infer success/failure from env signals (rewards, is_success, terminated, truncated).
# Used when labeling episodes after collection.
def infer_episode_outcome(
    rewards: np.ndarray,
    successes: np.ndarray,
    dones: np.ndarray,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    env_name: str | None = None,
) -> dict:
    """Determine episode outcome from environment signals.

    Uses a priority-based approach:
    1. Explicit success flags from the environment
    2. Reward threshold (task-specific)
    3. Termination reason if available

    Args:
        rewards: (T,) per-step rewards.
        successes: (T,) per-step success flags (True when env reports is_success).
        dones: (T,) cumulative done flags.
        terminated: (T,) per-step termination flags (optional).
        truncated: (T,) per-step truncation flags (optional).
        env_name: Environment name for task-specific logic.

    Returns:
        Dict with:
            success: bool
            episode_failed: bool
            termination_reason: str
            terminal_step: int
    """
    num_steps = len(rewards)

    # Find first done step (episode end)
    done_indices = np.where(dones)[0]
    terminal_step = int(done_indices[0]) if len(done_indices) > 0 else num_steps - 1

    # Primary signal: env sets is_success when task completes
    success = bool(np.any(successes))

    if success:
        termination_reason = "success"
    # truncated = hit max steps (timeout) or env-specific truncation
    elif truncated is not None and len(truncated) > 0 and truncated[terminal_step]:
        termination_reason = "timeout_or_failure"
    # terminated = hard failure (e.g. drop, collision)
    elif terminated is not None and len(terminated) > 0 and terminated[terminal_step]:
        termination_reason = "terminated_failure"
    else:
        termination_reason = "unknown"

    episode_failed = not success

    return {
        "success": success,
        "episode_failed": episode_failed,
        "termination_reason": termination_reason,
        "terminal_step": terminal_step,
        "total_reward": float(np.sum(rewards[: terminal_step + 1])),
        "max_reward": float(np.max(rewards[: terminal_step + 1])),
    }
