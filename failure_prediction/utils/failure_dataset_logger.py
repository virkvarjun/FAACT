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


# Buffers steps per episode, writes raw/episode_XXXXXX.npz. Keys: feat_decoder_mean, feat_encoder_latent_token, reward, done, etc.
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

        self._episode_meta = {}  # checkpoint_path, task_name, success, etc.
        self._step_data = []    # list of step dicts; flushed on save
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
        terminated: bool | None = None,
        truncated: bool | None = None,
        obs_state: np.ndarray | None = None,
        obs_images: dict[str, np.ndarray] | None = None,
        env_info: dict | None = None,
        predicted_action_chunk: np.ndarray | None = None,
        chunk_length: int | None = None,
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
            "terminated": bool(terminated) if terminated is not None else False,
            "truncated": bool(truncated) if truncated is not None else False,
            "env_info_json": json.dumps(env_info or {}, default=str),
            "chunk_length": int(chunk_length) if chunk_length is not None else -1,
            "chunk_step_idx": int(chunk_step_idx),
            "new_chunk_generated": bool(new_chunk_generated),
        }

        if self.save_obs_state and obs_state is not None:
            step["obs_state"] = np.asarray(obs_state, dtype=np.float32)

        if self.save_images and obs_images:
            for key, image in obs_images.items():
                if image is None:
                    continue
                step[f"image_{key}"] = np.asarray(image)

        if self.save_action_chunks and predicted_action_chunk is not None:
            step["predicted_action_chunk"] = np.asarray(predicted_action_chunk, dtype=np.float32)

        # features come from features_to_numpy: decoder_mean, encoder_latent_token, latent_sample
        if self.save_embeddings and features is not None:
            for key, val in features.items():
                if val is not None:
                    step[f"feat_{key}"] = np.asarray(val, dtype=np.float32)

        self._step_data.append(step)

    def end_episode(self, success: bool, termination_reason: str, terminal_step: int | None = None):
        """Finalize episode metadata."""
        self._episode_meta["success"] = success
        self._episode_meta["episode_failed"] = not success
        self._episode_meta["termination_reason"] = termination_reason
        self._episode_meta["num_steps"] = len(self._step_data)
        self._episode_meta["terminal_step"] = (
            int(terminal_step) if terminal_step is not None else max(len(self._step_data) - 1, 0)
        )
        self._episode_meta["end_time"] = time.time()
        episode_return = sum(s["reward"] for s in self._step_data)
        self._episode_meta["return"] = episode_return
        self._episode_meta["total_reward"] = episode_return

    def save_episode(self) -> Path:
        """Serialize the current episode to disk as .npz and return the file path."""
        ep_id = self._episode_meta["episode_id"]
        filename = self.raw_dir / f"episode_{ep_id:06d}.npz"

        arrays = {}
        # Collect all keys across steps; separate array vs scalar per step
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

        # Stack per-step arrays; fill missing with zeros to keep consistent shape
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

        # Store meta as single-element array so npz can hold it
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
        """Load episode from .npz. Returns {meta: {...}, arrays: {feat_*, reward, done, ...}}."""
        data = dict(np.load(path, allow_pickle=True))
        meta = json.loads(str(data.pop("_meta_json")[0]))
        return {"meta": meta, "arrays": data}
