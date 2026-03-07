# Failure Dataset Pipeline

Data collection and labeling pipeline for training a failure prediction model on top of ACT policy rollouts.

## Overview

This pipeline:
1. Runs a trained ACT policy in simulation for many episodes
2. Logs per-step observations, action chunks, ACT embeddings, and outcomes
3. Labels each timestep with dense failure-within-horizon targets
4. Produces a clean dataset for training a risk prediction model

The future risk model target is:
```
r_t = P(failure within next K steps | x_t)
```

## How to Run

### Step 1: Collect Raw Rollouts

```bash
python -m failure_prediction.scripts.collect_failure_dataset \
    --checkpoint /path/to/checkpoints/100000/pretrained_model \
    --task AlohaTransferCube-v0 \
    --num_episodes 200 \
    --output_dir failure_dataset/transfer_cube \
    --device cuda \
    --save_embeddings \
    --save_action_chunks \
    --failure_horizon 10
```

Run from the `Research/` directory.

### Step 2: Postprocess into Labeled Dataset

```bash
python -m failure_prediction.scripts.postprocess_failure_dataset \
    --input_dir failure_dataset/transfer_cube/raw \
    --output_dir failure_dataset/transfer_cube/processed \
    --failure_horizon 10
```

## Output Structure

```
failure_dataset/transfer_cube/
├── collection_meta.json          # Collection parameters and summary stats
├── raw/
│   ├── episode_000000.npz        # Per-episode raw data
│   ├── episode_000001.npz
│   └── ...
└── processed/
    ├── timestep_dataset.npz      # Flattened timestep dataset
    └── metadata.json             # Field schemas and summary stats
```

## Data Fields

### Per-Episode Raw File (episode_XXXXXX.npz)

| Field | Shape | Description |
|-------|-------|-------------|
| `executed_action` | (T, action_dim) | Actions sent to the environment |
| `reward` | (T,) | Per-step reward |
| `done` | (T,) | Done flag |
| `success` | (T,) | Per-step success flag from env |
| `terminated` | (T,) | Per-step terminated flag |
| `truncated` | (T,) | Per-step truncated flag |
| `env_info_json` | (T,) | JSON-serialized environment info dict |
| `obs_state` | (T, state_dim) | Robot proprioceptive state (joint positions) |
| `image_<camera>` | (T, H, W, 3) | Optional raw images when `--save_images` is enabled |
| `predicted_action_chunk` | (T, chunk_size, action_dim) | Full predicted action chunks (at chunk boundaries) |
| `chunk_length` | (T,) | Length of the currently active action chunk |
| `chunk_step_idx` | (T,) | Position within current chunk |
| `new_chunk_generated` | (T,) | Whether a new chunk was predicted this step |
| `feat_latent_sample` | (T, latent_dim) | VAE latent representation (at chunk boundaries) |
| `feat_encoder_latent_token` | (T, dim_model) | Encoder's latent token output |
| `feat_decoder_mean` | (T, dim_model) | Mean-pooled decoder output |
| `_meta_json` | (1,) | JSON string with episode metadata |

### Processed Timestep Dataset (timestep_dataset.npz)

| Field | Shape | Description |
|-------|-------|-------------|
| `episode_id` | (N,) | Episode index |
| `timestep` | (N,) | Step within episode |
| `success` | (N,) | Episode-level success (repeated) |
| `episode_failed` | (N,) | Episode-level failure (repeated) |
| `failure_within_k` | (N,) | **Main label**: 1 if failure within K steps |
| `steps_to_failure` | (N,) | Distance to failure (-1 for successful episodes) |
| `near_failure` | (N,) | Softer warning: 1 if failure within 2K steps |
| `reward` | (N,) | Per-step reward |
| `done` | (N,) | Done flag |
| `executed_action` | (N, action_dim) | Actions sent to env |
| `obs_state` | (N, state_dim) | Robot state |
| `feat_latent_sample` | (N, latent_dim) | ACT latent (at chunk boundaries) |
| `feat_encoder_latent_token` | (N, dim_model) | Encoder latent token |
| `feat_decoder_mean` | (N, dim_model) | Mean-pooled decoder |

## ACT Embeddings

Three intermediate representations are extracted from the ACT model:

1. **`latent_sample`** (32-dim): The VAE latent vector. During inference with `use_vae=True`, this is zeros (no action conditioning). Captures the "mode" of the policy.

2. **`encoder_latent_token`** (512-dim): First token of the transformer encoder output. This is the projected latent after attending to all observation tokens. Encodes the full scene understanding.

3. **`decoder_mean`** (512-dim): Mean-pooled transformer decoder output (before the action head). Summarizes the planned trajectory across all chunk positions.

These are extracted via `ACTPolicy.predict_action_chunk_with_features()`, which calls `ACT.forward(batch, return_features=True)`. The default forward API is unchanged.

## Label Definitions

### `failure_within_k`
For failed episodes: 1 for timesteps where `terminal_step - t <= K`.
For successful episodes: always 0.

### `steps_to_failure`
For failed episodes: `terminal_step - t` (distance to end).
For successful episodes: -1.

### `near_failure`
Same as `failure_within_k` but with 2K horizon. Provides a softer warning signal.

## Success/Failure Inference

Episode outcomes are determined by priority:
1. Explicit `is_success` flag from the environment info
2. Termination type (terminated vs truncated)
3. Fallback: `unknown`

Termination reasons: `success`, `timeout_or_failure`, `terminated_failure`, `unknown`.

Episode metadata includes `episode_id`, `checkpoint_path`, `task_name`, `seed`, `success`,
`episode_failed`, `num_steps`, `terminal_step`, `termination_reason`, and `return`.

## Known Limitations

- Embeddings are only logged at chunk boundaries (when `new_chunk_generated=True`). Between chunk boundaries, the embedding fields contain zeros. Downstream models should account for this.
- Image observations are not saved by default (`--save_images=False`) to keep file sizes manageable.
- The `latent_sample` during inference is always zeros when `use_vae=True` (no action conditioning at test time). It may still be useful as a baseline feature.
- Perturbation modes (`obs_noise`, `action_noise`) are placeholders and not yet implemented.

## Typical Shapes (ALOHA)

| Feature | Shape |
|---------|-------|
| `obs_state` | (14,) — 7 joints per arm × 2 arms |
| `executed_action` | (14,) |
| `predicted_action_chunk` | (100, 14) — chunk_size=100 |
| `feat_latent_sample` | (32,) — latent_dim=32 |
| `feat_encoder_latent_token` | (512,) — dim_model=512 |
| `feat_decoder_mean` | (512,) — dim_model=512 |
