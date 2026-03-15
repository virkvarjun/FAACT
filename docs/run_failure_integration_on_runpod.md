# Running the Failure Prediction Integration on RunPod

This guide runs the full failure prediction pipeline wired to the trained ACT checkpoint.

## Prerequisites

- RunPod instance with GPU (used for ACT training)
- ACT checkpoint at: `/root/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model`
- Environment: `gym_aloha`, `lerobot` installed, MuJoCo with EGL

## One-Shot Pipeline

From the Research directory:

```bash
cd /root/Research

# Set PYTHONPATH so lerobot and failure_prediction can be imported
export PYTHONPATH=lerobot/src:$PYTHONPATH

# Optional: reduce episodes for faster first pass (default 30)
export NUM_EPISODES=20

bash scripts/run_failure_prediction_integration.sh
```

## Per-Step Commands (if running manually)

### 1. Collect raw rollout data

```bash
cd /root/Research
export PYTHONPATH=lerobot/src:$PYTHONPATH
export MUJOCO_GL=egl

python -m failure_prediction.scripts.collect_failure_dataset \
  --checkpoint /root/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \
  --task AlohaTransferCube-v0 \
  --env_type aloha \
  --num_episodes 30 \
  --output_dir failure_dataset/transfer_cube \
  --device cuda \
  --save_embeddings \
  --save_action_chunks
```

### 2. Postprocess

```bash
python -m failure_prediction.scripts.postprocess_failure_dataset \
  --input_dir failure_dataset/transfer_cube/raw \
  --output_dir failure_dataset/transfer_cube/processed \
  --failure_horizon 10
```

### 3. Inspect

```bash
python -m failure_prediction.scripts.inspect_failure_dataset \
  --raw_dir failure_dataset/transfer_cube/raw \
  --processed_dir failure_dataset/transfer_cube/processed \
  --failure_horizon 10 \
  --sample_episode random \
  --json_report failure_dataset/transfer_cube/inspection_report.json
```

### 4. Inspect feature fields

```bash
python -m failure_prediction.scripts.inspect_feature_fields \
  --processed_dir failure_dataset/transfer_cube/processed \
  --json_out failure_dataset/transfer_cube/feature_report.json
```

### 5. Train supervised predictor

```bash
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean \
  --label_field failure_within_k \
  --output_dir failure_prediction_runs/transfer_cube_supervised \
  --epochs 20 \
  --batch_size 256
```

### 6. Run FIPER baseline

```bash
python -m failure_prediction.scripts.run_fiper_offline_eval \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean \
  --action_chunk_field predicted_action_chunk \
  --output_dir failure_prediction_runs/transfer_cube_fiper \
  --window_size 3 \
  --alpha 0.1
```

### 7. Generate plots and report

```bash
python -m failure_prediction.scripts.plot_failure_results \
  --run_dir failure_prediction_runs/transfer_cube_supervised \
  --output_dir failure_prediction_runs/plots/supervised

python -m failure_prediction.scripts.plot_failure_results \
  --run_dir failure_prediction_runs/transfer_cube_fiper \
  --output_dir failure_prediction_runs/plots/fiper

python -m failure_prediction.scripts.generate_integration_report \
  --raw_dir failure_dataset/transfer_cube/raw \
  --processed_dir failure_dataset/transfer_cube/processed \
  --supervised_dir failure_prediction_runs/transfer_cube_supervised \
  --fiper_dir failure_prediction_runs/transfer_cube_fiper \
  --checkpoint_path /root/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \
  --output failure_dataset/transfer_cube/integration_report.md
```

## Output Locations

| Artifact | Path |
|---------|------|
| Raw episodes | `failure_dataset/transfer_cube/raw/` |
| Processed dataset | `failure_dataset/transfer_cube/processed/` |
| Supervised run | `failure_prediction_runs/transfer_cube_supervised/` |
| FIPER run | `failure_prediction_runs/transfer_cube_fiper/` |
| Plots | `failure_prediction_runs/plots/` |
| Integration report | `failure_dataset/transfer_cube/integration_report.md` |

## Feature Fields (from real ACT)

The collect script logs:

- `feat_decoder_mean` — mean-pooled decoder output (512-d)
- `feat_encoder_latent_token` — first encoder token (512-d)
- `feat_latent_sample` — VAE latent (32-d, zeros at inference)

Default: `feat_decoder_mean` (closest to action head, stable).

## ACE Mode

- **Used:** `chunk_change` — magnitude of chunk-to-chunk change over a rolling window
- **Limitation:** True sample-dispersion ACE needs multi-sample ACT inference (not yet available)
