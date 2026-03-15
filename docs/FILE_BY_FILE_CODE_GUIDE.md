# File-by-File Code Guide

A walkthrough of every important file: where to go, what each does, and how the code works.

---

## Reading Order (Suggested Path)

1. **Pipeline overview** — README.md, failure_dataset_pipeline.md  
2. **Core utilities** — failure_labeling.py, success_inference.py  
3. **Data flow** — collect → FailureDatasetLogger → postprocess → load_failure_dataset  
4. **Models** — FailurePredictorMLP, ACT feature extraction  
5. **Evaluation** — run_failure_aware_eval  
6. **Supporting** — configs, scripts, experiments

---

# 1. PIPELINE SCRIPTS

## `failure_prediction/scripts/collect_failure_dataset.py`

**What it does:** Runs a trained ACT policy in simulation and logs per-step data for failure prediction.

**Entry point:** `python -m failure_prediction.scripts.collect_failure_dataset --checkpoint ... --task AlohaTransferCube-v0 --num_episodes 200 --output_dir failure_dataset/transfer_cube`

**Key flow:**

1. **`load_policy_and_processors(checkpoint_path, device)`** (lines 98–127)
   - Loads `ACTPolicy.from_pretrained()`, pre/post processors
   - Returns `(policy, preprocessor, postprocessor)`

2. **`make_single_env(task, env_type)`** (lines 84–95)
   - Builds `gym_id = f"gym_{env_type}/{task}"` (e.g. `gym_aloha/AlohaTransferCube-v0`)
   - Creates env with `obs_type="pixels_agent_pos"`, `render_mode="rgb_array"`

3. **`preprocess_obs(obs)`** (lines 132–157)
   - Converts raw env obs to policy format
   - `pixels` → `observation.images.<camera>` as (1, C, H, W) tensor, float [0,1]
   - `agent_pos` → `observation.state` as (1, state_dim)

4. **`predict_action_chunk_with_features(policy, obs_processed)`** (lines 160–214)
   - Tries: `policy.predict_action_chunk_with_features()` (our fork)
   - Fallback 1: `policy.model(batch, return_features=True)` (custom lerobot)
   - Fallback 2: Forward hooks on encoder/decoder to capture `encoder_out`, `decoder_out`
   - Returns `(actions, features)` where features has `latent_sample`, `encoder_out`, `decoder_out`

5. **`features_to_numpy(features)`** (lines 216–236)
   - `encoder_out` → first token only → `encoder_latent_token` (B, dim_model)
   - `decoder_out` → mean over chunk dim → `decoder_mean` (B, dim_model)
   - These names match what postprocess expects (`feat_decoder_mean`, etc.)

6. **`run_collection(args)`** (lines 238+)
   - For each episode: `env.reset()` → step loop:
     - When `need_new_chunk`: call `predict_action_chunk_with_features`, log features
     - Execute `action = current_chunk[:, chunk_step_idx]`, postprocess, `env.step(action_np)`
   - Uses `FailureDatasetLogger.log_step(...)` each step
   - At episode end: `infer_episode_outcome()` → `logger.end_episode()`, `logger.save_episode()`

**Output:** `failure_dataset/<task>/raw/episode_000000.npz`, etc.

---

## `failure_prediction/scripts/postprocess_failure_dataset.py`

**What it does:** Converts raw episode `.npz` files into a single labeled timestep dataset.

**Entry point:** `python -m failure_prediction.scripts.postprocess_failure_dataset --input_dir .../raw --output_dir .../processed --failure_horizon 10`

**Key flow:**

1. **`load_all_episodes(input_dir)`** (lines 48–61)
   - `FailureDatasetLogger.load_episode(f)` for each `episode_*.npz`

2. **`process_episodes(episodes, failure_horizon, near_failure_horizon)`** (lines 64–145)
   - For each episode: `label_failure_windows(num_steps, episode_failed, terminal_step, failure_horizon)`
   - Collects: `episode_id`, `timestep`, `success`, `episode_failed`, `failure_within_k`, `steps_to_failure`, `near_failure`, reward/done/terminated/truncated, chunk info
   - Copies embedding arrays (`feat_*`) from raw into `array_fields` with same names

3. **`label_failure_windows()`** (from `failure_labeling.py`)
   - See failure_labeling section below

4. **Save:** `timestep_dataset.npz` (all arrays), `metadata.json`

**Output:** `failure_dataset/<task>/processed/timestep_dataset.npz`

---

## `failure_prediction/scripts/train_failure_predictor.py`

**What it does:** Trains the MLP failure predictor on processed data (or mock data).

**Entry point:** `python -m failure_prediction.scripts.train_failure_predictor --processed_dir .../processed --output_dir failure_prediction_runs/run --epochs 20`

**Key flow:**

1. **`load_failure_dataset(...)`** (lines 80–90)
   - Loads from `processed_dir` or generates mock data
   - Returns `(features, labels, episode_ids, timesteps, input_dim, metadata)`

2. **`create_episode_splits(episode_ids, train_frac, val_frac, test_frac)`** (from `data/splits.py`)
   - Splits by episode (not timestep) to avoid leakage

3. **Model:** `FailurePredictorMLP(input_dim, hidden_dims, dropout)` — see models section

4. **Training loop:** BCEWithLogitsLoss, Adam, standard epoch loop

5. **Save:** `best_model.pt`, `config.json` (input_dim, hidden_dims, dropout, feature_field, label_field)

**Output:** `failure_prediction_runs/<run>/best_model.pt`, `config.json`

---

## `failure_prediction/scripts/run_failure_aware_eval.py`

**What it does:** Online evaluation: baseline vs monitor_only vs intervention.

**Entry point:** `python -m failure_prediction.scripts.run_failure_aware_eval --checkpoint ... --mode intervention --risk_model_ckpt ... --risk_threshold 0.5 --output_dir ...`

**Key flow:**

1. **`load_risk_model(ckpt_dir, device)`** (lines 72–96)
   - Loads `config.json`, builds `FailurePredictorMLP`, loads `best_model.pt`
   - Maps `feat_decoder_mean` → `decoder_mean` (key used in features dict)

2. **`run_episode(..., risk_model, risk_threshold, mode)`** (lines 99–227)
   - Step loop similar to collect, but:
     - When `need_new_chunk`: get `action_chunk, features` from policy
     - Score: `risk_prob = sigmoid(risk_model(feat_vec))`
     - If `mode == "intervention"` and `risk_prob >= risk_threshold`:
       - **Intervention logic:** generate `num_candidate_chunks` chunks by adding `obs_noise_std` to images
       - For each: `predict_action_chunk_with_features(policy, noisy_processed)` → score
       - Pick chunk with **lowest** risk: `best_idx = argmin(candidate_risks)`
     - Otherwise use original chunk
   - Returns dict with `success`, `terminal_step`, `interventions`, `alarms`, `step_scores`

3. **`add_obs_noise(obs_dict, noise_std)`** (lines 51–70)
   - Adds Gaussian noise to `pixels` (scaled by 255 * noise_std) for diversity

**Output:** JSON in `output_dir` with per-episode metrics

---

# 2. CORE UTILITIES

## `failure_prediction/utils/failure_labeling.py`

**What it does:** Produces dense per-timestep labels for failure prediction.

**Function:** `label_failure_windows(num_steps, episode_failed, terminal_step, failure_horizon, near_failure_horizon)`

**Logic:**

- `failure_within_k[t] = 1` iff episode failed AND `terminal_step - t <= failure_horizon` and `>= 0`
- `steps_to_failure[t] = terminal_step - t` if failed, else `-1`
- `near_failure[t] = 1` with `near_failure_horizon` (default 2*K)

**Return:** Dict of numpy arrays.

---

## `failure_prediction/utils/success_inference.py`

**What it does:** Infers episode success/failure from env signals.

**Function:** `infer_episode_outcome(rewards, successes, dones, terminated, truncated, env_name)`

**Logic:**

- `terminal_step` = first index where `dones` is True
- `success = any(successes)` (env’s `is_success`)
- `episode_failed = not success`
- `termination_reason`: "success" | "timeout_or_failure" | "terminated_failure" | "unknown"

**Return:** `{success, episode_failed, termination_reason, terminal_step, total_reward, max_reward}`

---

## `failure_prediction/utils/failure_dataset_logger.py`

**What it does:** Logs and saves per-episode rollout data during collection.

**Key methods:**

- `start_episode(episode_id, checkpoint_path, task_name, seed)` — reset buffers
- `log_step(timestep, executed_action, reward, done, success, ..., features)` — append step dict
- `end_episode(success, ...)` — set meta
- `save_episode()` — write `raw/episode_XXXXXX.npz`
- `load_episode(path)` — static, load .npz back

**Step dict keys:** timestep, executed_action, reward, done, success, terminated, truncated, env_info_json, chunk_*, and (if `save_embeddings`) feat_* arrays.

---

# 3. DATA

## `failure_prediction/data/failure_dataset.py`

**What it does:** Load processed dataset or create mock data.

**Key functions:**

- `load_processed_dataset(processed_dir)` → `(dict of arrays, metadata)`
- `load_failure_dataset(processed_dir, feature_field, label_field, mock=True|False, mock_*...)`  
  → `(features, labels, episode_ids, timesteps, input_dim, metadata)`
- `get_available_feature_fields(dataset)` — lists candidate feature columns (excludes LABEL_AND_META_KEYS)

**Mock mode:** Random features + synthetic `failure_within_k` with `mock_positive_ratio`.

---

## `failure_prediction/data/splits.py`

**What it does:** Episode-level train/val/test splits.

- `create_episode_splits(episode_ids, train_frac, val_frac, test_frac, seed)` → `(train_mask, val_mask, test_mask)` — boolean masks over timesteps, grouped by episode
- `split_summary(...)` — basic stats per split

---

# 4. MODELS

## `failure_prediction/models/failure_predictor.py`

**What it does:** MLP that predicts P(failure within K | embedding).

**Class:** `FailurePredictorMLP(input_dim, hidden_dims, dropout)`

- `dims = [input_dim] + hidden_dims + [1]`
- Sequential: Linear → ReLU → Dropout (repeat) → Linear
- `forward(x)` → logits (squeezed to 1D)

**Input:** Single embedding vector (e.g. `feat_decoder_mean`, 512-dim).  
**Output:** Scalar logit.

---

## `Research/lerobot/src/lerobot/policies/act/modeling_act.py`

**What it does:** ACT policy with optional feature extraction for failure prediction.

**ACTPolicy:**

- `predict_action_chunk(batch)` — standard action prediction (unchanged)
- `predict_action_chunk_with_features(batch)` (lines 137–156):
  - Builds batch, calls `self.model(batch, return_features=True)`
  - Returns `(actions, features)` where features = `{latent_sample, encoder_out, decoder_out}`

**ACT (model):**

- `forward(batch, return_features=False)` (lines 401–547)
- Standard flow: latent → encoder → decoder → action_head
- If `return_features=True` (lines 539–545):
  ```python
  features = {
      "latent_sample": latent_sample,      # (B, 32)
      "encoder_out": encoder_out.T,        # (B, seq, 512)
      "decoder_out": decoder_out,         # (B, chunk_size, 512)
  }
  return actions, (mu, log_sigma_x2), features
  ```

**Feature semantics:**

- `latent_sample`: VAE latent; often zeros at inference
- `encoder_out`: Transformer encoder output
- `decoder_out`: Decoder output before action head; mean-pooled → `decoder_mean` (primary risk input)

---

# 5. INTERFACES & FIPER

## `failure_prediction/interfaces.py`

**What it does:** Abstract interfaces for pluggable runtime components.

- `RiskScorer.predict_step(features) -> RiskScore` — supervised risk
- `FiperScorer.compute_scores(embedding, action_chunk) -> FiperScores` — RND + ACE
- `InterventionPolicy.should_interrupt(...) -> InterventionDecision` — placeholder

---

## `failure_prediction/fiper/`

**What it does:** FIPER-style baseline (RND OOD + ACE uncertainty + conformal calibration).

- `ace.py` — `compute_ace_scores` (uncertainty)
- `alarm.py` — `WindowedAlarmAggregator`
- `baseline.py` — `FIPERBaseline`
- `conformal.py` — `calibrate_thresholds`

Used for ablation vs supervised MLP.

---

# 6. CONFIGS & SCRIPTS

## `failure_prediction/configs/failure_predictor_mlp.yaml`

- `processed_dir`, `feature_field`, `label_field`
- `hidden_dims`, `epochs`, `batch_size`, splits

## `failure_prediction/configs/deployment.yaml`

- `risk_threshold`, `risk_feature_field`
- `num_candidate_chunks`, `obs_noise_std`

## `Research/scripts/` (top-level)

- `train_sim.sh`, `train_act_persistent.sh` — ACT training
- `eval_sim.sh` — ACT evaluation
- `run_failure_aware_full_pipeline.sh` — full pipeline
- `setup_runpod_gpu.sh` — RunPod setup

## `Research/experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh`

- Karpathy-style training with optional vanilla/cpu modes
- Uses `lerobot/aloha_sim_transfer_cube_human`, `--dataset.video_backend=pyav`, `--policy.push_to_hub=False`

---

# 7. QUICK REFERENCE: DATA FLOW

```
Raw obs (env)
    → preprocess_obs() → observation.images.*, observation.state
    → preprocessor()   → policy-ready batch
    → predict_action_chunk_with_features()
        → ACT.forward(return_features=True)
        → actions, features = {latent_sample, encoder_out, decoder_out}
    → features_to_numpy() → {decoder_mean, encoder_latent_token, latent_sample}
    → FailureDatasetLogger.log_step() → raw/episode_*.npz

postprocess_failure_dataset
    → label_failure_windows() → failure_within_k, steps_to_failure
    → timestep_dataset.npz (feat_* + labels)

train_failure_predictor
    → load_failure_dataset() → features, labels
    → FailurePredictorMLP(feat_decoder_mean) → risk logit
    → best_model.pt

run_failure_aware_eval (intervention)
    → risk_model(feat) >= threshold?
    → Yes: add_obs_noise() × N → N chunks → pick lowest risk
    → No: use original chunk
```

---

*Use this as a roadmap: open each file and follow the line numbers and function names above.*
