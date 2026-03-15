# Karpathy Autoresearch → FAACT (Experimental)

**Status:** Experimental — isolated from main project. Results may be reverted.

This folder contains experiments adapting ideas from Andrej Karpathy's nanochat/autoresearch tuning to the FAACT (Failure-Aware ACT) policy. The main project codebase is unchanged for production use; all experimental scripts, configs, and docs live here.

## What this is

- **Optimizer/scheduler adaptations** inspired by nanochat: per-param-group LR, absolute warmup, warmdown with nonzero LR floor
- **Optional ACT training flags** in lerobot (gated, backward compatible)
- **Ablation scripts** for smoke tests and RunPod runs

## Structure

```
experiments/karpathy_autoresearch/
├── README.md           # This file
├── integration_plan.md # Full technical plan
├── final_summary.md    # What was implemented, next steps
├── scripts/            # Run helpers
│   └── train_faact_karpathy.sh
└── experiments/        # Per-experiment writeups
    └── faact_karpathy_opt_v1.md
```

## Dependencies on main project

The lerobot package (`Research/lerobot`) contains optional Karpathy support, enabled via config flags:

- `--policy.use_karpathy_scheduler=True`
- `--policy.scheduler_warmup_steps=500`
- `--policy.scheduler_warmdown_ratio=0.5`
- `--policy.scheduler_final_lr_frac=0.05`
- `--policy.optimizer_lr_action_head` (optional)
- `--policy.optimizer_lr_embeddings` (optional)

These are additive; default behavior is unchanged.

## Quick start

From `Research/` (project root):

```bash
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
```

From RunPod (`/workspace/Research` or repo root):

```bash
export MUJOCO_GL=egl
bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
```

## Disclaimers

- This is **not** the main failure prediction / ACT pipeline
- Ablations are for research; production training uses baseline config
- See `integration_plan.md` for full technical details
