# Codebase Walkthrough — Interview Prep

A guided tour of the entire repo for interview preparation. Use this to navigate, explain components, and answer "how does X work?" questions.

---

## 1. High-Level: What Is This Project?

**Failure-Aware ACT** — A runtime monitoring and intervention layer for Action Chunking Transformers (ACT). Instead of only asking "did the robot fail?", we ask:

- Is the **current predicted chunk** still valid?
- If we interrupt now, can the task still be **recovered**?

The system:
1. Runs ACT to predict action chunks
2. Monitors internal ACT embeddings for failure risk
3. Interrupts and replans when risk exceeds a threshold

**Paper inspiration:** "Failure Prediction at Runtime for Generative Robot Policies" (conceptual alignment).

---

## 2. Repo Navigation

### 2.1 Where Is Everything?

```
lerobot/                    # Git root (pushed to virkvarjun/lerobot-test)
├── Research/               # Main research code (YOU SPEND MOST TIME HERE)
│   ├── failure_prediction/ # Failure prediction pipeline
│   ├── experiments/       # Experimental work (Karpathy, etc.)
│   ├── scripts/           # Top-level scripts
│   ├── docs/              # Documentation
│   ├── configs/            # Config files
│   ├── lerobot/           # LeRobot fork (ACT extensions, submodule)
│   └── failure_prediction_runs/  # Outputs (checkpoints, plots, videos)
├── lerobot/               # Upstream LeRobot (at project root)
├── lerobot-sim/           # Simulation stack
└── failure_policies_research/  # Older failure-related work
```

**Start here for interviews:** `Research/` — that's where all the novel work lives.

### 2.2 Quick Paths by Topic

| Topic | Paths to Know |
|-------|---------------|
| Failure prediction pipeline | `Research/failure_prediction/` |
| ACT embeddings / feature extraction | `Research/lerobot/src/lerobot/policies/act/modeling_act.py` |
| Data collection | `failure_prediction/scripts/collect_failure_dataset.py` |
| Postprocessing / labeling | `failure_prediction/scripts/postprocess_failure_dataset.py` |
| Risk model | `failure_prediction/models/failure_predictor.py` |
| Online eval + intervention | `failure_prediction/scripts/run_failure_aware_eval.py` |
| Karpathy experiments | `Research/experiments/karpathy_autoresearch/` |
| Training scripts | `Research/scripts/`, `experiments/karpathy_autoresearch/scripts/` |

---

## 3. Failure Prediction Pipeline (End-to-End)

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. COLLECT (raw rollouts)                                                    │
│    collect_failure_dataset.py                                                │
│    → ACT in AlohaTransferCube-v0 → log obs, actions, embeddings, outcomes    │
│    → Saves: failure_dataset/<task>/raw/episode_*.npz                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. POSTPROCESS (labeling)                                                    │
│    postprocess_failure_dataset.py                                            │
│    → failure_labeling.label_failure_windows()                                 │
│    → Labels: failure_within_k, steps_to_failure                                │
│    → Saves: failure_dataset/<task>/processed/timestep_dataset.npz             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRAIN (risk model)                                                       │
│    train_failure_predictor.py                                                │
│    → FailurePredictorMLP on feat_decoder_mean                                │
│    → Label: failure_within_k                                                  │
│    → Saves: failure_prediction_runs/<run>/best_model.pt                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. EVAL (online)                                                             │
│    run_failure_aware_eval.py                                                 │
│    → baseline | monitor_only | intervention                                  │
│    → intervention: risk > threshold → re-sample chunks, pick lowest risk    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Files by Stage

| Stage | Script | What It Does |
|-------|--------|--------------|
| Collect | `collect_failure_dataset.py` | Runs ACT in sim, logs `feat_decoder_mean`, `feat_encoder_latent_token`, `feat_latent_sample`, rewards, done, success |
| Postprocess | `postprocess_failure_dataset.py` | Loads raw episodes, labels with `failure_within_k`, `steps_to_failure` |
| Train | `train_failure_predictor.py` | Trains MLP on embeddings → risk; supports `--mock` for synthetic data |
| Eval | `run_failure_aware_eval.py` | baseline / monitor_only / intervention modes |

### 3.3 ACT Embeddings (Why They Matter)

ACT is extended to **return features** during forward pass:

| Feature | Dim | Use |
|---------|-----|-----|
| `latent_sample` | 32 | VAE latent (often zeros at inference) |
| `encoder_latent_token` | 512 | First encoder token |
| `decoder_mean` | 512 | Mean-pooled decoder output — **primary feature for risk model** |

**Where extracted:** `Research/lerobot/src/lerobot/policies/act/modeling_act.py`  
- `predict_action_chunk_with_features()` — returns `(action_chunk, features)`  
- `ACT.forward(batch, return_features=True)` — returns dict with these features

### 3.4 Labeling Logic (`failure_prediction/utils/failure_labeling.py`)

- **`failure_within_k`**: 1 if failure occurs within next K steps from this timestep
- **`steps_to_failure`**: Distance to failure (−1 if success)
- Success/failure from env info, termination type, or fallback `unknown`

---

## 4. failure_prediction/ — Structure

```
failure_prediction/
├── __init__.py
├── interfaces.py          # RiskScorer, FiperScorer, InterventionPolicy ABCs
├── scripts/               # All pipeline scripts (see table below)
├── models/                # FailurePredictorMLP, RNDModule
├── fiper/                 # FIPER baseline: RND, ACE, conformal, WindowedAlarmAggregator
├── data/                  # FailureDataset, load_failure_dataset, episode splits
├── utils/                 # failure_labeling, success_inference, eval_metrics, etc.
├── configs/               # failure_predictor_mlp.yaml, fiper_baseline.yaml, deployment.yaml
└── docs/                  # failure_dataset_pipeline.md
```

### Scripts Quick Reference

| Script | Purpose |
|--------|---------|
| `collect_failure_dataset.py` | Collect raw rollouts with embeddings |
| `postprocess_failure_dataset.py` | Label episodes with failure_within_k |
| `train_failure_predictor.py` | Train MLP risk model |
| `run_failure_aware_eval.py` | Online eval: baseline / monitor / intervention |
| `analyze_failure_predictor.py` | Threshold sweep, lead-time analysis |
| `plot_final_results.py` | Plot results |
| `generate_final_report.py` | Generate project report markdown |
| `record_eval_videos.py` | Record intervention videos with labels |
| `train_fiper_rnd.py` | Train RND OOD model |
| `run_fiper_offline_eval.py` | FIPER baseline offline eval |
| `visualize_embedding_space.py` | t-SNE/UMAP of embeddings |
| `create_project_figures.py` | Rollout timeline, ACT pipeline diagram, etc. |

---

## 5. experiments/ — Karpathy Autoresearch

**Location:** `Research/experiments/karpathy_autoresearch/`

**Goal:** Adapt Karpathy’s nanochat/autoresearch ideas to FAACT (optimizer groups, warmdown scheduler).

**What was planned/added (may vary by branch):**

| Component | Change |
|-----------|--------|
| Optimizer | 4 param groups: backbone, action_head, embeddings, rest |
| Scheduler | `KarpathyWarmdownSchedulerConfig` — absolute warmup, warmdown, nonzero final LR |
| ACT config | `use_karpathy_scheduler`, `scheduler_warmup_steps`, etc. |

**Train script:** `scripts/train_faact_karpathy.sh`  
- Modes: `smoke`, `runpod`, `runpod vanilla`, `runpod vanilla cpu`  
- Uses `lerobot/aloha_sim_transfer_cube_human`, `--dataset.video_backend=pyav`, `--policy.push_to_hub=False`

**Note:** On RunPod, vanilla lerobot is used (no Karpathy args) unless you install a modified fork.

---

## 6. Top-Level Scripts (`Research/scripts/`)

| Script | Purpose |
|--------|---------|
| `train_sim.sh` | Simulation training (transfer_cube, insertion) |
| `train_act_persistent.sh` | ACT on RunPod with persistent storage |
| `eval_sim.sh` | Evaluate ACT in sim |
| `run_failure_aware_full_pipeline.sh` | Full pipeline: analyze → eval → plots → report |
| `run_failure_prediction_integration.sh` | Failure prediction integration |
| `setup_runpod_gpu.sh` | RunPod GPU setup (clone, deps, lerobot install) |
| `setup_runpod_persistent.sh` | RunPod persistent volume setup |
| `record_5_interventions.sh` | Record 5 intervention videos |

---

## 7. LeRobot Package (`Research/lerobot/`)

**Submodule:** HuggingFace lerobot with research-specific changes.

**Key extensions:**

1. **ACT forward with features** (`policies/act/modeling_act.py`)
   - `return_features=True` → dict with `latent_sample`, `encoder_out`, `decoder_out`
   - `predict_action_chunk_with_features()` for inference

2. **Optimizer groups** (if Karpathy branch)
   - 4 groups: backbone, action_head, embeddings, rest

3. **Karpathy scheduler** (if present)
   - Absolute warmup, warmdown, nonzero final LR

**Core modules you might discuss:**

- `policies/act/` — ACT implementation
- `datasets/` — LeRobotDataset, video backends (torchcodec, pyav)
- `processor/` — normalization, image transforms, device placement
- `envs/` — make_env, gym_aloha, gym_pusht
- `optim/` — optimizers, schedulers

---

## 8. Configs

| Config | Path | Purpose |
|--------|------|---------|
| `failure_predictor_mlp.yaml` | `failure_prediction/configs/` | MLP: feature_field, label_field, hidden dims, epochs |
| `deployment.yaml` | `failure_prediction/configs/` | risk_threshold, num_candidate_chunks, obs_noise_std |
| `fiper_baseline.yaml` | `failure_prediction/configs/` | FIPER baseline settings |

---

## 9. Interfaces (`failure_prediction/interfaces.py`)

Abstract interfaces for pluggable runtime intervention:

- **`RiskScorer`** — `predict_step(features) -> RiskScore`
- **`FiperScorer`** — `compute_scores(embedding, action_chunk) -> FiperScores` (RND + ACE)
- **`InterventionPolicy`** — `should_interrupt(...) -> InterventionDecision`

Designed so runtime intervention can plug in without rewriting core logic.

---

## 10. Common Interview Questions — Cheat Sheet

**Q: Walk me through the failure prediction pipeline.**  
A: Collect raw rollouts with ACT embeddings → postprocess with failure_within_k labels → train MLP on feat_decoder_mean → online eval with baseline / monitor / intervention. Intervention: if risk > threshold, re-sample N chunks with observation noise and pick lowest risk.

**Q: Where are ACT embeddings extracted?**  
A: `Research/lerobot/src/lerobot/policies/act/modeling_act.py` — `predict_action_chunk_with_features()` and `forward(..., return_features=True)`.

**Q: What is failure_within_k?**  
A: Per-timestep binary label: 1 if failure occurs within the next K steps, 0 otherwise. Used as the supervised target for the risk model.

**Q: How does intervention work at runtime?**  
A: When risk > threshold, we re-sample N candidate action chunks (add obs noise, query ACT), score each with the risk model, and execute the chunk with lowest risk.

**Q: What’s in experiments/karpathy_autoresearch/?**  
A: Experimental optimizer/scheduler changes inspired by Karpathy’s nanochat: per-group LRs, warmdown scheduler. Separate from main project; for ablation studies.

**Q: How do you run training on RunPod?**  
A: SSH in, run `setup_runpod_gpu.sh`, then `train_faact_karpathy.sh runpod vanilla`. Use tmux for persistence.

---

## 11. Running Things (Commands to Know)

```bash
# From Research/
cd Research

# Collect failure data
python -m failure_prediction.scripts.collect_failure_dataset \
  --checkpoint /path/to/pretrained_model --task AlohaTransferCube-v0 \
  --num_episodes 200 --output_dir failure_dataset/transfer_cube

# Postprocess
python -m failure_prediction.scripts.postprocess_failure_dataset \
  --input_dir failure_dataset/transfer_cube/raw \
  --output_dir failure_dataset/transfer_cube/processed

# Train risk model
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir failure_dataset/transfer_cube/processed \
  --output_dir failure_prediction_runs/my_run

# Run evaluation
python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint /path/to/act --risk_model_ckpt failure_prediction_runs/my_run \
  --mode intervention

# Karpathy training (RunPod)
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod vanilla
```

---

*Generated for interview prep. Update paths if repo structure changes.*
