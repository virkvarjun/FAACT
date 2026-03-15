# FAACT: Reuse RunPod ACT + Wrapper Around π₀ (PI0)

**Goal:** Use your existing RunPod-trained ACT checkpoint from the start (no retraining), and architect FAACT so it can wrap π₀ (PI0) as the base policy instead of (or in addition to) ACT.

---

## π₀ (PI0) — OpenPI Repository

The π₀ model comes from **[Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)**. Clone it when you need to:

- Use openpi's native checkpoints (base models, DROID/ALOHA fine-tuned)
- Inspect the original model architecture for embedding extraction
- Run openpi's policy server (`serve_policy.py`)

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

**LeRobot integration:** LeRobot already has a port of π₀ in `lerobot/src/lerobot/policies/pi0/` (and `pi0_fast`, `pi05`). It loads HF checkpoints like `lerobot/pi0_base`. For FAACT wrapping π₀, we work with the **LeRobot PI0 policy**—no need to clone openpi unless you're debugging against the upstream implementation or using openpi-specific checkpoints/tools.

---

## Part 1: Reusing RunPod ACT — No Retraining

**Yes, you can use the same ACT checkpoint from RunPod without retraining.**

The pipeline is already designed for this. Every script takes `--checkpoint` and loads the policy from that path.

### Checkpoint Location (RunPod Persistent Volume)

After training with `scripts/train_act_persistent.sh`:

```
/workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model
```

### How to Use It

| Scenario | What to do |
|----------|------------|
| **Run on RunPod** | Use the path above directly; `/workspace` is persistent. |
| **Run locally** | Copy checkpoint from RunPod: `scp -r <pod>@ssh.runpod.io:/workspace/Research/outputs/train/act_transfer_cube ./` |
| **Upload to HuggingFace** | `huggingface-cli upload <user>/act_transfer_cube /workspace/.../pretrained_model .` then use `--checkpoint lerobot/<user>/act_transfer_cube` |

### Full FAACT Pipeline Using Existing Checkpoint

```bash
# Set checkpoint once (local path or HF ID)
export ACT_CHECKPOINT="/path/to/pretrained_model"   # or your HF repo

# 1. Collect (no training — uses checkpoint as-is)
python -m failure_prediction.scripts.collect_failure_dataset \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha \
  --num_episodes 200 --output_dir failure_dataset/transfer_cube

# 2. Postprocess
python -m failure_prediction.scripts.postprocess_failure_dataset \
  --input_dir failure_dataset/transfer_cube/raw \
  --output_dir failure_dataset/transfer_cube/processed

# 3. Train only the failure predictor (small MLP, fast)
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir failure_dataset/transfer_cube/processed \
  --feature_field feat_decoder_mean --output_dir failure_prediction_runs/transfer_cube_supervised

# 4. Online eval (intervention mode)
python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha \
  --mode intervention --risk_model_ckpt failure_prediction_runs/transfer_cube_supervised \
  --risk_threshold 0.5 --num_candidate_chunks 5
```

**What you do NOT need:** ACT training. The base policy is read-only; only the small failure-predictor MLP is trained.

---

## Part 2: FAACT as Wrapper Around π₀ (PI0)

Today FAACT is **ACT-specific** because it relies on:

1. `predict_action_chunk_with_features()` → returns `(action_chunk, features)` with `decoder_mean` (512-d)
2. Chunk-based execution: `n_action_steps` per chunk
3. Failure predictor MLP trained on `feat_decoder_mean`

π₀ (PI0) is a different policy (flow-matching, PaliGemma-based). To make FAACT a **policy-agnostic wrapper**, you need:

### Architecture Change: Policy-Agnostic FAACT

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FAACT Wrapper (policy-agnostic)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Base Policy Interface:                                             │
│    - predict_action_chunk(obs) → actions                             │
│    - predict_action_chunk_with_features(obs) → (actions, features) │
├─────────────────────────────────────────────────────────────────────┤
│  Supported base policies:                                            │
│    - ACT  (existing: decoder_mean, encoder_latent_token)             │
│    - PI0  (new: need analogous embedding)                            │
│    - PI0Fast, SmolVLA (future)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Add `predict_action_chunk_with_features` for PI0

PI0 / PI0Fast use flow-matching and output action chunks via `predict_action_chunk`. To get embeddings:

- **Option A (preferred):** Add `return_features=True` (or similar) to PI0’s forward path and expose an embedding (e.g. mean of action-conditioning hidden states before the action head).
- **Option B:** Use forward hooks on a chosen PI0 layer and pool (e.g. mean) to produce a fixed-size vector.

The embedding does not need to match ACT’s `decoder_mean` semantics; it must be a **stable, fixed-dim representation** of the current observation/state that correlates with failure. You then train a **new failure predictor** on PI0 embeddings (separate from the ACT one).

**Reference:** When implementing, inspect [openpi](https://github.com/Physical-Intelligence/openpi) (`src/openpi/models_pytorch/pi0_pytorch.py`, `sample_actions`) to identify which hidden states to expose for the failure predictor.

### Step 2: Policy-Agnostic Collect / Eval Scripts

Refactor `collect_failure_dataset.py` and `run_failure_aware_eval.py` to:

- Accept `--policy_type act|pi0|pi0_fast`
- Dispatch to the right loader: `load_policy_and_processors(checkpoint, device, policy_type=...)`
- Call `predict_action_chunk_with_features` via the common interface (implemented for each policy)

### Step 3: Per-Policy Failure Predictor

| Base policy | Feature field | Failure predictor |
|-------------|----------------|-------------------|
| ACT | `feat_decoder_mean` (512-d) | `FailurePredictorMLP` (existing) |
| PI0 | e.g. `feat_pi0_action_conditioning` (TBD) | New MLP trained on PI0 rollouts |

Same training pipeline: collect rollouts → postprocess → train MLP. Only the feature field and possibly `input_dim` change.

### Step 4: Intervention Logic (Unchanged)

Intervention logic (risk threshold, resampling with obs noise, pick lowest-risk chunk) stays the same. It operates on `(action_chunk, risk_score)`, independent of the base policy.

---

## Implementation Plan

### Phase 1: Reuse ACT (Immediate — No Code Changes)

1. Ensure RunPod volume has the checkpoint at  
   `/workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model`
2. Run `scripts/run_failure_prediction_integration.sh` on RunPod (or locally with a copied checkpoint).
3. Alternatively, upload checkpoint to HuggingFace and use the HF path as `--checkpoint`.

**Effort:** 0 code changes. Config/path only.

---

### Phase 2: FAACT Wrapper Around PI0

| Task | File(s) | Description |
|------|---------|-------------|
| 1 | `lerobot/.../policies/pi0/modeling_pi0.py` (or `pi0_fast`) | Add `predict_action_chunk_with_features()` returning `(actions, features)` with a suitable embedding key |
| 2 | `failure_prediction/scripts/collect_failure_dataset.py` | Add `--policy_type`; `load_policy_and_processors(policy_type=...)` |
| 3 | `failure_prediction/scripts/run_failure_aware_eval.py` | Same `--policy_type` and loader |
| 4 | `failure_prediction/utils/feature_stats.py` or similar | Ensure feature dim is detected for PI0 (or pass `--feature_dim`) |
| 5 | Docs / config | Document `feat_pi0_*` and recommended `input_dim` for the failure predictor |

**Estimated effort:** 2–4 days, depending on how easily PI0’s internals expose an embedding.

---

## Deployment Notes

### Raspberry Pi Zero

If “Pi Zero” meant **Raspberry Pi Zero** (512 MB RAM):

- ACT / PI0 cannot run on-device; they need GBs of RAM and a GPU.
- Use **async inference**: run the policy (ACT or PI0) on a GPU server (e.g. RunPod, your laptop), and the Pi as `RobotClient` that streams observations and receives action chunks.
- FAACT wrapper runs on the server: policy + failure predictor + intervention logic. The Pi only executes actions.

### RunPod as Policy Server

You can run the policy server on RunPod and connect a robot client from anywhere:

```bash
# On RunPod (GPU)
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# Robot client connects with --policy_type=act --pretrained_name_or_path=/workspace/.../pretrained_model
```

Today the async server does not include FAACT (risk model + intervention). To support it, the server would need to:
- Load the risk model
- Run `predict_action_chunk_with_features`
- Apply intervention logic before returning chunks to the client

---

## Summary

| Question | Answer |
|----------|--------|
| Can I reuse RunPod ACT without retraining? | **Yes.** Point `--checkpoint` at the pretrained model path. |
| Can FAACT wrap π₀ (PI0)? | **Yes**, with a policy-agnostic wrapper: add `predict_action_chunk_with_features` for PI0 and train a PI0-specific failure predictor. |
| Can FAACT run on Raspberry Pi Zero? | Base policy no; use async inference (policy on GPU, Pi as client). FAACT logic would live on the server. |

---

## Quick Start: Reuse ACT Today

```bash
cd /workspace/Research  # or your local clone

export ACT_CHECKPOINT="/workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model"
export RISK_MODEL="failure_prediction_runs/transfer_cube_supervised"

# If you haven’t done collect + train yet:
bash scripts/run_failure_prediction_integration.sh

# Or just run intervention eval with existing risk model:
python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha \
  --mode intervention --risk_model_ckpt "$RISK_MODEL" \
  --risk_threshold 0.5 --num_candidate_chunks 5 \
  --output_dir failure_prediction_runs/online_eval_intervention
```

## Quick Start: Failure-Aware Wrapper Around π₀ (PI0)

```bash
cd /workspace/Research

export PI0_CHECKPOINT="lerobot/pi0_base"
export RISK_MODEL="failure_prediction_runs/pi0_transfer_cube_supervised"

# 1. Collect rollouts from π₀
python -m failure_prediction.scripts.collect_failure_dataset \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy_type pi0 \
  --task AlohaTransferCube-v0 --env_type aloha \
  --task_desc "transfer the cube to the target" \
  --num_episodes 200 --output_dir failure_dataset/pi0_transfer_cube \
  --device cuda

# 2. Postprocess
python -m failure_prediction.scripts.postprocess_failure_dataset \
  --input_dir failure_dataset/pi0_transfer_cube/raw \
  --output_dir failure_dataset/pi0_transfer_cube/processed \
  --failure_horizon 10

# 3. Train failure predictor on π₀ features (action_chunk_mean)
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir failure_dataset/pi0_transfer_cube/processed \
  --feature_field feat_action_chunk_mean \
  --output_dir "$RISK_MODEL"

# 4. Online eval with intervention
python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$PI0_CHECKPOINT" \
  --policy_type pi0 \
  --task AlohaTransferCube-v0 --env_type aloha \
  --task_desc "transfer the cube to the target" \
  --mode intervention --risk_model_ckpt "$RISK_MODEL" \
  --risk_feature_field feat_action_chunk_mean \
  --risk_threshold 0.5 --num_candidate_chunks 5 \
  --output_dir failure_prediction_runs/pi0_online_eval_intervention
```
