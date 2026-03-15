# Karpathy nanochat/autoresearch → FAACT Integration Plan

**Goal:** Integrate the most promising ideas from Karpathy's nanochat/autoresearch tuning into FAACT (Failure-Aware ACT) for improved training stability, sample efficiency, and downstream failure prediction accuracy.

**Philosophy:** Do not blindly port LLM-specific changes. Only keep changes that make sense for ACT's architecture and imitation learning dynamics.

---

## 1. Current FAACT Stack Summary

### 1.1 Model Architecture (lerobot ACT)

| Component | Location | Notes |
|-----------|----------|-------|
| ACT policy | `lerobot/src/lerobot/policies/act/modeling_act.py` | `ACTPolicy`, `ACT` |
| Encoder | `ACTEncoder` + `ACTEncoderLayer` | Self-attention, pre/post-norm |
| Decoder | `ACTDecoder` + `ACTDecoderLayer` | Self + cross-attention |
| Action head | `self.action_head = nn.Linear(dim_model, action_dim)` | Single linear layer |
| VAE encoder | Optional, `ACTEncoder(is_vae_encoder=True)` | For latent sampling |
| Positional embeddings | Sinusoidal (1D + 2D), **no RoPE** | `create_sinusoidal_pos_embedding`, `ACTSinusoidalPositionEmbedding2d` |
| Attention | `nn.MultiheadAttention` | **Full attention**, no sliding window |
| Init | Xavier uniform for encoder/decoder | `_reset_parameters()` |
| Backbone | ResNet via torchvision | Layer4 features |

**Key findings:**
- No RoPE (uses sinusoidal)
- No local/sliding attention (full context)
- No QK normalization
- No value embedding gates or logit softcap

### 1.2 Optimizer & Scheduler

| Component | Location | Current |
|-----------|----------|---------|
| Optimizer creation | `lerobot/optim/factory.py` | `make_optimizer_and_scheduler()` |
| Policy param groups | `modeling_act.py::get_optim_params()` | 2 groups: backbone vs rest |
| ACT optimizer preset | `configuration_act.py::get_optimizer_preset()` | AdamW, lr=1e-5, wd=1e-4 |
| ACT scheduler preset | `configuration_act.py::get_scheduler_preset()` | **Returns None** (no scheduler) |
| AdamW config | `lerobot/optim/optimizers.py` | Single lr, wd, betas for all groups |
| Cosine + warmup | `lerobot/optim/schedulers.py` | `CosineDecayWithWarmupSchedulerConfig` exists but ACT doesn't use it |

### 1.3 Failure Predictor

| Component | Location | Notes |
|-----------|----------|-------|
| MLP | `failure_prediction/models/failure_predictor.py` | input_dim → [256,128] → 1 |
| Training | `failure_prediction/scripts/train_failure_predictor.py` | Adam, single lr=1e-3, no scheduler |

### 1.4 Training Loop

- **Entry:** `lerobot-train` → `lerobot/src/lerobot/scripts/lerobot_train.py`
- **Flow:** `update_policy()` → forward, backward, gradient clip, optimizer.step(), scheduler.step()
- **Config:** draccus + `TrainPipelineConfig`, `use_policy_training_preset=True` loads optimizer from policy

---

## 2. Karpathy Ideas: Transferability Assessment

### 2.1 Directly Transferable (high confidence)

| Idea | Nanochat | FAACT adaptation | ROI |
|------|----------|------------------|-----|
| **Per-parameter-group LR** | embedding, unembedding, matrix, scalars | backbone, encoder/decoder, action_head, embeddings, norm/bias | High |
| **Stronger output-head LR** | lm_head at lower LR (scaled by √(dim/768)) | Action head: try slightly lower or higher LR; commonly output heads benefit from different scaling | High |
| **Absolute warmup steps** | `--warmup-steps 40` | Add `warmup_steps` (int) instead of ratio-only; ACT often trains 100k steps | Medium |
| **Warmdown with nonzero floor** | `--warmdown-ratio 0.65`, `--final-lr-frac 0.05` | Cosine/warmdown decay with final_lr_frac so LR doesn't go to zero | High |
| **Cosine weight decay schedule** | `0.5 * (1 + cos(π * step / steps))` | Add optional cosine WD schedule (currently static) | Medium |

### 2.2 Maybe Transferable (needs ablation)

| Idea | Nanochat | FAACT adaptation | Ablation priority |
|------|----------|------------------|-------------------|
| **QK post-norm scaling** | Q,K normalized, optional scaler | nn.MultiheadAttention doesn't expose Q/K; would need custom attention or post-scale of attn output | 2nd (after opt/scheduler) |
| **Smaller init scales** | embeddings, MLP, output head | Xavier → smaller gain; try 0.9x or init action head smaller | 3rd |
| **Betas tuning** | beta1=0.96 for some groups | ACT uses default (0.9, 0.999); try (0.9, 0.99) or group-specific | 4th |

### 2.3 Not Applicable to FAACT

| Idea | Reason |
|------|--------|
| RoPE theta increase | ACT uses sinusoidal embeddings, not RoPE |
| Local/sliding attention | ACT uses full attention; window pattern is LLM-specific |
| Logit softcap | ACT outputs actions (regression), not logits |
| Value embedding gates | No analogous structure in ACT |
| Muon optimizer | Keep AdamW; Muon is for LLM scaling |

---

## 3. Implementation Plan

### Phase 1: Repo Understanding ✅

- [x] Trace model, optimizer, scheduler, training loop
- [x] Identify param groups, attention, init
- [x] Document current state

### Phase 2: Optimizer / Scheduler Adaptations ✅

**Files edited:**
1. `lerobot/src/lerobot/policies/act/modeling_act.py` — extend `get_optim_params()` for more granular groups
2. `lerobot/src/lerobot/policies/act/configuration_act.py` — add optimizer/scheduler options
3. `lerobot/src/lerobot/optim/schedulers.py` — add `KarpathyWarmdownSchedulerConfig`
4. ACT config wired to use new scheduler when enabled

**Commit strategy (done):**
- [x] Refactor `get_optim_params` into explicit groups (backbone, transformer, action_head, embeddings)
- [x] Add config fields for per-group LR
- [x] Add `KarpathyWarmdownSchedulerConfig`
- [x] Wire ACT config to use new scheduler when enabled

### Phase 3: Attention / Architecture Adaptations

**Deferred:** QK post-norm scaling, init scale reduction.

### Phase 4: Configs and Ablations

**Experiment scripts:** `experiments/karpathy_autoresearch/scripts/`

### Phase 5: Evaluation

**Metrics:** Training/val loss, rollout success rate, failure predictor AUROC, gradient norms, NaNs.

---

## 4. Ablation Order (Safest → Riskiest)

1. **Scheduler only** — Enable `use_karpathy_scheduler=True`
2. **Param groups** — Add action_head, embeddings groups with configurable LR
3. **Cosine weight decay** — Optional cosine WD schedule
4. **Init scale** — Smaller init for action head
5. **QK post-norm** — Custom attention path (highest risk)

---

## 5. Files Reference

| Purpose | Path |
|---------|------|
| ACT model | `lerobot/src/lerobot/policies/act/modeling_act.py` |
| ACT config | `lerobot/src/lerobot/policies/act/configuration_act.py` |
| Schedulers | `lerobot/src/lerobot/optim/schedulers.py` |
| Experiment scripts | `experiments/karpathy_autoresearch/scripts/` |

---

## 6. Run Commands

**From `Research/` (project root):**
```bash
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
```

**RunPod:**
```bash
cd /workspace/Research
export MUJOCO_GL=egl
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
```
