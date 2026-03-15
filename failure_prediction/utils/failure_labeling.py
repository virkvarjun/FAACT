"""Dense per-timestep failure labeling for downstream risk model training.

The core training target for the future failure predictor is:
    r_t = P(failure within next K steps | x_t)

This module generates the ground-truth labels from completed episodes.
"""

from __future__ import annotations

import numpy as np


# Label each timestep: 1 if failure happens within K steps from here.
# steps_to_failure = distance to failure; -1 for success episodes.
def label_failure_windows(
    num_steps: int,
    episode_failed: bool,
    terminal_step: int,
    failure_horizon: int = 10,
    near_failure_horizon: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute dense per-timestep failure labels from a completed episode.

    Args:
        num_steps: Total timesteps in the episode.
        episode_failed: Whether the episode ultimately failed.
        terminal_step: The step at which the episode ended.
        failure_horizon: K in "failure within next K steps".
        near_failure_horizon: Optional separate horizon for softer near_failure label.
            Defaults to 2 * failure_horizon.

    Returns:
        Dict with:
            failure_within_k: (num_steps,) binary array, 1 if failure within K steps.
            steps_to_failure: (num_steps,) int array, distance to failure. -1 for success episodes.
            near_failure: (num_steps,) binary array, softer warning window.
            episode_failed: (num_steps,) repeated episode-level label for convenience.
    """
    if near_failure_horizon is None:
        near_failure_horizon = 2 * failure_horizon

    # Initialize: success episodes get -1 for steps_to_failure; failed episodes get distance
    failure_within_k = np.zeros(num_steps, dtype=np.int32)
    steps_to_failure = np.full(num_steps, -1, dtype=np.int32)
    near_failure = np.zeros(num_steps, dtype=np.int32)
    episode_failed_arr = np.full(num_steps, int(episode_failed), dtype=np.int32)

    # Only failed episodes have positive labels; success episodes stay all zeros
    if episode_failed:
        for t in range(num_steps):
            dist = terminal_step - t  # steps until failure at terminal_step
            steps_to_failure[t] = max(dist, 0)

            if dist <= failure_horizon and dist >= 0:
                failure_within_k[t] = 1  # within K steps of failure

            if dist <= near_failure_horizon and dist >= 0:
                near_failure[t] = 1  # softer warning window (2K by default)

    return {
        "failure_within_k": failure_within_k,
        "steps_to_failure": steps_to_failure,
        "near_failure": near_failure,
        "episode_failed": episode_failed_arr,
    }
