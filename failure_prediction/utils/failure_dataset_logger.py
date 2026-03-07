"""Logger for collecting failure prediction training data during ACT rollouts.

Handles per-episode buffering and serialization to .npz files.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FailureDatasetLogger:
    """Collects and saves per-episode rollout data for failure prediction.

    Usage:
        logger = FailureDatasetLogger(output_dir, save_embeddings=True)
        logger.start_episode(episode_id=0, checkpoint_path="...", task_name="...", seed=42)
        for step in range(max_steps):
            logger.log_step(timestep=step, ...)
        logger.end_episode(success=False, termination_reason="timeout_or_failure")
        logger.save_episode()
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_embeddings: bool = True,
        save_action_chunks: bool = True,
        save_images: bool = False,
        save_obs_state: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.save_embeddings = save_embeddings
        self.save_action_chunks = save_action_chunks
        self.save_images = save_images
        self.save_obs_state = save_obs_state

        self._episode_meta = {}
        self._step_data = []
        self._episode_active = False

    def start_episode(
        self,
        episode_id: int,
        checkpoint_path: str,
        task_name: str,
        seed: int | None = None,
    ):
        """Begin recording a new episode."""
        if self._episode_active:
            logger.warning("Starting new episode without ending previous one. Discarding.")

        self._episode_meta = {
            "episode_id": episode_id,
            "checkpoint_path": checkpoint_path,
            "task_name": task_name,
            "seed": seed,
            "start_time": time.time(),
        }
        self._step_data = []
        self._episode_active = True

    def log_step(
        self,
        timestep: int,
        executed_action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        obs_state: np.ndarray | None = None,
        predicted_action_chunk: np.ndarray | None = None,
        chunk_step_idx: int = 0,
        new_chunk_generated: bool = False,
        features: dict[str, np.ndarray] | None = None,
    ):
        """Record a single timestep of data.

        Args:
            timestep: Current step index.
            executed_action: (action_dim,) action sent to env.
            reward: Scalar reward from env.
            done: Whether episode is done after this step.
            success: Whether success was flagged at this step.
            obs_state: (state_dim,) robot proprioceptive state.
            predicted_action_chunk: (chunk_size, action_dim) full action chunk.
            chunk_step_idx: Position within the current chunk being executed.
            new_chunk_generated: Whether a new chunk was predicted this step.
            features: Dict of ACT internal features (numpy arrays, already on CPU).
        """
        step = {
            "timestep": timestep,
            "executed_action": np.asarray(executed_action, dtype=np.float32),
            "reward": float(reward),
            "done": bool(done),
            "success": bool(success),
            "chunk_step_idx": int(chunk_step_idx),
            "new_chunk_generated": bool(new_chunk_generated),
        }

        if self.save_obs_state and obs_state is not None:
            step["obs_state"] = np.asarray(obs_state, dtype=np.float32)

        if self.save_action_chunks and predicted_action_chunk is not None:
            step["predicted_action_chunk"] = np.asarray(predicted_action_chunk, dtype=np.float32)

        if self.save_embeddings and features is not None:
            for key, val in features.items():
                if val is not None:
                    step[f"feat_{key}"] = np.asarray(val, dtype=np.float32)

        self._step_data.append(step)

    def end_episode(self, success: bool, termination_reason: str):
        """Finalize episode metadata."""
        self._episode_meta["success"] = success
        self._episode_meta["episode_failed"] = not success
        self._episode_meta["termination_reason"] = termination_reason
        self._episode_meta["num_steps"] = len(self._step_data)
        self._episode_meta["end_time"] = time.time()
        self._episode_meta["total_reward"] = sum(s["reward"] for s in self._step_data)

    def save_episode(self) -> Path:
        """Serialize the current episode to disk as .npz and return the file path."""
        ep_id = self._episode_meta["episode_id"]
        filename = self.raw_dir / f"episode_{ep_id:06d}.npz"

        arrays = {}
        scalars_per_step = {}

        all_keys = set()
        for s in self._step_data:
            all_keys.update(s.keys())

        array_keys = set()
        scalar_keys = set()
        for key in all_keys:
            sample_val = next((s[key] for s in self._step_data if key in s), None)
            if sample_val is None:
                continue
            if isinstance(sample_val, np.ndarray):
                array_keys.add(key)
            else:
                scalar_keys.add(key)

        for key in array_keys:
            vals = []
            for s in self._step_data:
                if key in s:
                    vals.append(s[key])
                else:
                    vals.append(np.zeros_like(vals[-1]) if vals else np.array([]))
            if vals and all(v.shape == vals[0].shape for v in vals):
                arrays[key] = np.stack(vals)

        for key in scalar_keys:
            vals = [s.get(key, None) for s in self._step_data]
            try:
                arrays[key] = np.array(vals)
            except (ValueError, TypeError):
                pass

        meta_json = json.dumps(self._episode_meta, default=str)
        arrays["_meta_json"] = np.array([meta_json])

        np.savez_compressed(filename, **arrays)

        self._episode_active = False
        self._step_data = []

        logger.info(
            f"Saved episode {ep_id} ({self._episode_meta['num_steps']} steps, "
            f"success={self._episode_meta['success']}) -> {filename}"
        )
        return filename

    @staticmethod
    def load_episode(path: str | Path) -> dict:
        """Load a saved episode .npz file and return structured data."""
        data = dict(np.load(path, allow_pickle=True))
        meta = json.loads(str(data.pop("_meta_json")[0]))
        return {"meta": meta, "arrays": data}
