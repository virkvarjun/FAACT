#!/bin/bash
# Train ACT with Karpathy-inspired optimizer/scheduler adaptations.
# EXPERIMENTAL — lives in experiments/karpathy_autoresearch/
#
# Usage (local smoke, 500 steps):
#   cd Research && bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh smoke
#
# Usage (RunPod, full ablation):
#   cd /workspace/lerobot-test/Research && bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod
#
# Usage (vanilla ACT — when Karpathy args not in installed lerobot):
#   bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod vanilla
#
# Usage (RunPod with CPU — if "no kernel image" CUDA error, use to verify pipeline):
#   bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod vanilla cpu
#
# Prerequisites: lerobot installed, dataset and policy configured.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root (Research/) — 3 levels up from experiments/.../scripts/
RESEARCH_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(dirname "$RESEARCH_DIR")"
# lerobot submodule: at repo root (lerobot-test) or inside Research (local dev)
LEROBOT_DIR="${REPO_ROOT}/lerobot"
[ -d "${RESEARCH_DIR}/lerobot" ] && LEROBOT_DIR="${RESEARCH_DIR}/lerobot"
cd "$RESEARCH_DIR"

# Default: transfer cube (human demos)
DATASET_REPO="${DATASET_REPO:-lerobot/aloha_sim_transfer_cube_human}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/train}"
STEPS="${STEPS:-100000}"

if [ "$1" = "smoke" ]; then
  STEPS=500
  EVAL_FREQ=200
  SAVE_FREQ=500
  echo "Smoke test: $STEPS steps"
elif [ "$1" = "runpod" ]; then
  export MUJOCO_GL=egl
  STEPS=100000
  EVAL_FREQ=10000
  SAVE_FREQ=20000
  [ "$2" = "vanilla" ] && USE_VANILLA=1 || USE_VANILLA=0
  [ "$3" = "cpu" ] && USE_CPU=1 || USE_CPU=0
  [ "$USE_CPU" = 1 ] && DEVICE_ARG="--policy.device=cpu" || DEVICE_ARG=""
  [ "$USE_VANILLA" = 1 ] && echo "RunPod run (vanilla ACT): $STEPS steps" || echo "RunPod run: $STEPS steps"
  [ "$USE_CPU" = 1 ] && echo "Using CPU (slow — fix PyTorch/GPU for real training)"
  shift  # consume runpod
  [ "$USE_VANILLA" = 1 ] && shift  # consume vanilla
  [ "$USE_CPU" = 1 ] && shift  # consume cpu
else
  EVAL_FREQ=10000
  SAVE_FREQ=20000
  USE_VANILLA=0
  USE_CPU=0
  DEVICE_ARG=""
fi

# Build output dir and base args
OUTPUT_DIR="${OUTPUT_BASE}/faact_$([ "$USE_VANILLA" = 1 ] && echo 'vanilla' || echo 'karpathy_opt')_$(date +%Y%m%d_%H%M)"

KARPATHY_ARGS=()
[ "$USE_VANILLA" = 0 ] && KARPATHY_ARGS=(
  --policy.use_karpathy_scheduler=True
  --policy.scheduler_warmup_steps=500
  --policy.scheduler_warmdown_ratio=0.5
  --policy.scheduler_final_lr_frac=0.05
)

PYTHONPATH="${LEROBOT_DIR}:${RESEARCH_DIR}" lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.push_to_hub=False \
  $DEVICE_ARG \
  "${KARPATHY_ARGS[@]}" \
  --output_dir="$OUTPUT_DIR" \
  --steps=$STEPS \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --batch_size=8 \
  --num_workers=4 \
  "$@"
