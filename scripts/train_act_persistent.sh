#!/bin/bash
# ACT training with PERSISTENT storage on /workspace.
# Survives Pod restarts. Safe to close laptop (run inside tmux).
set -e

# Use /workspace (persistent volume) for all outputs
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
OUTPUT_BASE="${WORKSPACE_ROOT}/Research/outputs"

# Conda (adjust if you use a different path)
if [ -f "/root/miniforge3/bin/conda" ]; then
    eval "$(/root/miniforge3/bin/conda shell.bash hook)"
    conda activate lerobot
fi

export MUJOCO_GL=egl

TASK=${1:-"transfer_cube"}
STEPS=${2:-100000}
BATCH_SIZE=${3:-8}

case $TASK in
  transfer_cube)
    DATASET="lerobot/aloha_sim_transfer_cube_human"
    ENV_TYPE="aloha"
    ENV_TASK="AlohaTransferCube-v0"
    ;;
  insertion)
    DATASET="lerobot/aloha_sim_insertion_human"
    ENV_TYPE="aloha"
    ENV_TASK="AlohaInsertion-v0"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Available tasks: transfer_cube, insertion"
    exit 1
    ;;
esac

OUTPUT_DIR="${OUTPUT_BASE}/train/act_${TASK}"
JOB_NAME="act_${TASK}"
# Clear previous run (lerobot-train refuses to overwrite when resume=False)
rm -rf "${OUTPUT_DIR}"

echo "================================================"
echo "  LeRobot ACT Training (PERSISTENT)"
echo "================================================"
echo "  Task:       $ENV_TASK"
echo "  Dataset:    $DATASET"
echo "  Output:     $OUTPUT_DIR  [on /workspace - PERSISTENT]"
echo "  Steps:      $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "================================================"

lerobot-train \
  --dataset.repo_id=${DATASET} \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --env.type=${ENV_TYPE} \
  --env.task=${ENV_TASK} \
  --steps=${STEPS} \
  --batch_size=${BATCH_SIZE} \
  --wandb.enable=false

echo ""
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "Checkpoint path: ${OUTPUT_DIR}/checkpoints/100000/pretrained_model"
