# Experiment: faact_karpathy_opt_v1

**Location:** `experiments/karpathy_autoresearch/experiments/`

## Hypothesis

Karpathy-style optimizer/scheduler changes improve ACT training stability and sample efficiency:
- Per-parameter-group LR (backbone, action head, embeddings)
- Absolute warmup steps instead of ratio-only
- Warmdown phase with nonzero final LR floor
- Cosine-like LR decay avoids abrupt end-of-training collapse

## Config diff from baseline

| Parameter | Baseline | Karpathy v1 |
|-----------|----------|-------------|
| `use_karpathy_scheduler` | (N/A) | True |
| `scheduler_warmup_steps` | - | 500 |
| `scheduler_warmdown_ratio` | - | 0.5 |
| `scheduler_final_lr_frac` | - | 0.05 |
| Param groups | 2 (backbone, rest) | 4 (backbone, action_head, embeddings, rest) |
| `optimizer_lr_action_head` | - | None (uses base) |
| `optimizer_lr_embeddings` | - | None (uses base) |

## Run command

**Smoke (local, 500 steps):**
```bash
cd Research
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
```

**Full (RunPod):**
```bash
cd /workspace/Research
export MUJOCO_GL=egl
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
```

**Manual (if lerobot-train is in PATH):**
```bash
lerobot-train --policy.type=act --policy.use_karpathy_scheduler=True \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_warmdown_ratio=0.5 \
  --policy.scheduler_final_lr_frac=0.05 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube \
  --output_dir=outputs/train/faact_karpathy_opt --steps=100000
```

## Metrics to monitor

- Training L1 loss (action prediction)
- KL loss (if VAE)
- Validation loss
- Gradient norm / NaN checks
- Rollout success rate (post-training eval)
- Failure predictor AUROC (on extracted embeddings)

## Result summary

(To be filled after run)

- [ ] Keep
- [ ] Revert
- [ ] Iterate
