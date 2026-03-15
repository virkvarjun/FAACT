# Karpathy FAACT Integration: Final Summary

**Location:** `experiments/karpathy_autoresearch/` — experimental, isolated from main project.

## What was implemented

### 1. Optimizer param groups (lerobot)

- **Before:** 2 groups (backbone, rest)
- **After:** 4 groups — rest (encoder/decoder), backbone, action_head, embeddings
- Config: `optimizer_lr_action_head`, `optimizer_lr_embeddings` (optional overrides)

### 2. Karpathy-style scheduler (lerobot)

- `KarpathyWarmdownSchedulerConfig`: absolute warmup, warmdown phase, nonzero final LR floor
- ACT config: `use_karpathy_scheduler`, `scheduler_warmup_steps`, `scheduler_warmdown_ratio`, `scheduler_final_lr_frac`
- Default: off for backward compatibility

### 3. Experiment infrastructure

- `experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh`
- `experiments/karpathy_autoresearch/experiments/faact_karpathy_opt_v1.md`

## What was NOT implemented (deferred)

- QK post-norm scaling
- Cosine weight decay schedule
- Init scale reduction
- RoPE / local attention (not applicable)

## Run commands

**Smoke (500 steps):**
```bash
cd Research
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
```

**Full RunPod:**
```bash
cd /workspace/Research
export MUJOCO_GL=egl
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
```

## Next ablations

1. Scheduler only
2. Action head LR: `--policy.optimizer_lr_action_head=2e-5`
3. Embeddings LR: `--policy.optimizer_lr_embeddings`
4. Warmdown ratio: 0.4, 0.5, 0.65
