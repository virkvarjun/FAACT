# Failure Prediction Pipeline — Presentation Walkthrough

Use this order when presenting. Each file has comments; reference the line ranges below.

---

## 1. Pipeline Overview (30 sec)

**Flow:** Collect → Postprocess → Train → Eval

- **Collect:** ACT runs in sim, logs embeddings per step → `raw/episode_*.npz`
- **Postprocess:** Labels timesteps with `failure_within_k` → `processed/timestep_dataset.npz`
- **Train:** MLP on embeddings → `best_model.pt`
- **Eval:** Baseline / monitor / intervention (resample chunks when risk high)

---

## 2. Utilities (2 min)

### `failure_prediction/utils/failure_labeling.py`
- **What:** Per-timestep labels for "failure within K steps"
- **Key:** `label_failure_windows` — produces `failure_within_k`, `steps_to_failure`, `near_failure`
- **Comment to show:** L14–16 (label each timestep; -1 for success episodes)

### `failure_prediction/utils/success_inference.py`
- **What:** Infers success/failure from env signals (rewards, is_success, terminated)
- **Key:** `infer_episode_outcome` — used after each episode in collection
- **Comment to show:** L8–9 (priority: success flags → reward → termination)

### `failure_prediction/utils/failure_dataset_logger.py`
- **What:** Buffers steps, writes `raw/episode_*.npz`
- **Key:** `log_step`, `save_episode`, `load_episode`
- **Comment to show:** L18 (feat_decoder_mean, etc. stored as feat_*)

---

## 3. Collection (3 min)

### `failure_prediction/scripts/collect_failure_dataset.py`

| Section | Lines | What to say |
|---------|-------|-------------|
| Entry | 1–14 | Docstring: runs ACT, logs per-step data |
| make_single_env | 85–96 | Gym env: gym_aloha/AlohaTransferCube-v0 |
| load_policy_and_processors | 99–131 | ACT + preprocessor + postprocessor |
| preprocess_obs | 133–158 | Pixels → (1,C,H,W) [0,1], agent_pos → state |
| predict_action_chunk_with_features | 161–215 | Actions + features. Policy method, else return_features, else hooks |
| features_to_numpy | 217–230 | decoder_out → mean → decoder_mean (primary risk input) |
| run_collection | 233–284 | Main loop: need_new_chunk → predict → log step → env.step |

---

## 4. Postprocess (2 min)

### `failure_prediction/scripts/postprocess_failure_dataset.py`

| Section | Lines | What to say |
|---------|-------|-------------|
| load_all_episodes | 48–61 | Load raw episode .npz files |
| process_episodes | 64–182 | For each episode: `label_failure_windows`, copy feat_* |
| L98 | 98–99 | Copy embeddings (feat_decoder_mean, etc.) to output |

**Output:** `timestep_dataset.npz` with features + labels

---

## 5. Model & Training (2 min)

### `failure_prediction/models/failure_predictor.py`
- **Comment:** L12 — 512→256→128→1, BCEWithLogitsLoss
- **Input:** feat_decoder_mean. **Output:** logit

### `failure_prediction/data/failure_dataset.py`
- **load_failure_dataset:** Loads processed dir or mock. Returns features, labels, episode_ids
- **LABEL_AND_META_KEYS:** Excluded from feature selection

### `failure_prediction/data/splits.py`
- **create_episode_splits:** Episode-level splits (not timestep) to avoid leakage

### `failure_prediction/scripts/train_failure_predictor.py`
- L80–81: load_failure_dataset, episode-level splits
- L123: auto pos_weight for class imbalance
- L167: BCE, no sigmoid (logits in)

---

## 6. Evaluation (3 min)

### `failure_prediction/scripts/run_failure_aware_eval.py`

| Section | Lines | What to say |
|---------|-------|-------------|
| Docstring | 1–15 | Three modes: baseline, monitor_only, intervention |
| add_obs_noise | 49–50 | Noise on obs → N different chunks (for intervention) |
| load_risk_model | 72–96 | best_model.pt + config; feat_decoder_mean → decoder_mean |
| run_episode | 99–100 | baseline / monitor / intervention |
| need_new_chunk | ~139 | Chunk-based execution |
| Intervention block | ~155–181 | N noisy obs → N chunks → score → pick lowest risk |

---

## 7. Interfaces (1 min)

### `failure_prediction/interfaces.py`
- **Docstring:** RiskScorer (MLP), FiperScorer (RND+ACE), InterventionPolicy (placeholder)
- Pluggable so you can swap in other risk models later

---

## Quick Reference: Comments by File

| File | Key comments |
|------|--------------|
| failure_labeling | L14–16: per-timestep labels |
| success_inference | L8–9: infer from env signals |
| failure_dataset_logger | L18: feat_* keys |
| collect_failure_dataset | L133, L161, L217, L233 |
| postprocess_failure_dataset | L48, L65, L98 |
| failure_predictor | L12: architecture |
| failure_dataset | L17: LABEL_AND_META_KEYS |
| splits | L1–4: episode-level |
| train_failure_predictor | L80, L93, L123, L167 |
| run_failure_aware_eval | L49, L72, L99, ~139, ~156 |
| interfaces | L1–6: three interfaces |
