# Failure Predictor Training

This document describes how to train the baseline failure predictor MLP on the processed failure dataset.

## Overview

The failure predictor estimates:

\[
r_t = P(\text{failure within next } K \text{ steps} \mid x_t)
\]

where \(x_t\) is a selected ACT embedding (e.g. `feat_decoder_mean`). The model is a small MLP trained with BCE loss on binary `failure_within_k` labels.

## Expected Processed Dataset Format

The loader expects a processed directory containing:

- **`timestep_dataset.npz`**: NumPy archive with:
  - `episode_id` (N,) — unique episode identifier per timestep
  - `timestep` (N,) — step index within episode
  - `failure_within_k` (N,) — binary label (1 = failure within K steps)
  - `steps_to_failure`, `episode_failed`, etc.
  - Feature fields: `feat_decoder_mean`, `feat_latent_sample`, `feat_encoder_latent_token`, etc.

- **`metadata.json`** (optional): Stats and field metadata.

Feature fields are inferred automatically. Use `--feature_field feat_decoder_mean` (or another `feat_*` key) to select the embedding.

## Mock Mode (No Real Data)

To validate the training pipeline without ACT or collected rollouts:

```bash
python -m failure_prediction.scripts.train_failure_predictor \
  --mock_data \
  --output_dir failure_prediction_runs/mock_run \
  --feature_dim 256 \
  --num_mock_episodes 50 \
  --timesteps_per_episode 40 \
  --epochs 5
```

This generates synthetic embeddings and labels, runs the full training loop, and saves outputs.

## Real Data Mode

Once ACT is trained and rollouts are collected and postprocessed:

```bash
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean \
  --label_field failure_within_k \
  --output_dir failure_prediction_runs/transfer_cube_risk_mlp \
  --epochs 20 \
  --batch_size 256
```

## Outputs

Each run writes to `output_dir/` (or `output_dir/run_name/`):

| File | Description |
|------|-------------|
| `config.json` | Run configuration |
| `split_summary.json` | Episode/timestep/positive/negative counts per split |
| `metrics.json` | Train/val metrics per epoch, best val AUROC, test metrics |
| `best_model.pt` | Best checkpoint (by val AUROC) |
| `last_model.pt` | Final epoch checkpoint |
| `val_predictions.npz` | Val logits, probs, labels, episode_ids, timesteps |
| `test_predictions.npz` | Test logits, probs, labels, episode_ids, timesteps |

## Assumptions

- Splits are **episode-level** only (no timestep leakage).
- Single feature field for MVP; multiple fields can be added later.
- Labels are binary `failure_within_k`; other labels (e.g. `steps_to_failure`) not yet supported.
- Input dimension is inferred from the chosen feature field.

## Next Stage

After this scaffold:

1. Train ACT and collect rollouts.
2. Postprocess to produce `timestep_dataset.npz`.
3. Run training with real data.
4. (Later) Integrate predictor into online self-correction loop.
