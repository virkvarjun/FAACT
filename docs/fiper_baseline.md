# FIPER Baseline

FIPER-style failure prediction using **RND** (Random Network Distillation) for OOD detection in the policy embedding space, **ACE** (Action-Chunk Entropy) proxies for action uncertainty, **conformal calibration** on successful rollouts only, and **windowed alarm** aggregation.

## Ingredients

1. **RND in policy embedding space**: Fixed random target network + trainable predictor. Score = L2 prediction error. Trained only on successful episodes.
2. **ACE (Action-Chunk Entropy)**: Mode B proxy when only one logged chunk per timestep:
   - `chunk_change`: magnitude of chunk-to-chunk change over rolling window
   - `chunk_norm_var`: variance of chunk L2 norm over window
3. **Conformal calibration**: Thresholds = empirical quantile(1-α) of scores on successful episodes only.
4. **Windowed alarm**: Rolling mean over `window_size`; alarm when both RND and ACE exceed thresholds (AND rule).

## What is exact vs approximate

- **Exact**: RND score, conformal thresholding, alarm aggregation logic
- **Approximate**: ACE uses logged-chunk proxies (chunk_change, chunk_norm_var) because we have only one predicted chunk per timestep. True multi-sample ACE requires runtime policy sampling.
- **Placeholder**: Online `CandidateChunkScorer.score_candidates(chunk_samples)` for when ACT supports multi-sample chunk generation at runtime.

## Mock mode

```bash
# Train RND
python -m failure_prediction.scripts.train_fiper_rnd \
  --mock_data --feature_dim 256 \
  --output_dir failure_prediction_runs/mock_fiper_rnd --epochs 5

# Full FIPER eval (trains RND if no checkpoint)
python -m failure_prediction.scripts.run_fiper_offline_eval \
  --mock_data --output_dir failure_prediction_runs/mock_fiper \
  --window_size 3 --alpha 0.1
```

## Real data mode

```bash
# 1. Train RND on processed data
python -m failure_prediction.scripts.train_fiper_rnd \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean \
  --output_dir failure_prediction_runs/transfer_cube_fiper_rnd --epochs 20

# 2. Run FIPER offline eval
python -m failure_prediction.scripts.run_fiper_offline_eval \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean \
  --rnd_checkpoint failure_prediction_runs/transfer_cube_fiper_rnd/rnd_model.pt \
  --output_dir failure_prediction_runs/transfer_cube_fiper \
  --window_size 3 --alpha 0.1
```

## Outputs

- `fiper_results.json`: alarm precision, recall, false alarm rate, lead-time stats
- `fiper_artifacts.npz`: test RND scores, ACE scores, alarms, episode_ids, timesteps, labels

## Metrics

- **Alarm precision**: TP / (TP + FP) for alarms vs failure_within_k
- **Alarm recall**: TP / (TP + FN)
- **False alarm rate**: FP / negatives
- **% failed episodes with at least one alarm**
- **% successful episodes with false alarm**
- **Lead time**: T_failure - T_first_alarm for failed episodes with early alarm
