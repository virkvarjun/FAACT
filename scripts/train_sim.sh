#!/bin/bash
set -e

eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda activate lerobot
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

OUTPUT_DIR="/root/Research/outputs/train/act_${TASK}"
JOB_NAME="act_${TASK}"

echo "================================================"
echo "  LeRobot ACT Training"
echo "================================================"
echo "  Task:       $ENV_TASK"
echo "  Dataset:    $DATASET"
echo "  Env type:   $ENV_TYPE"
echo "  Steps:      $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output:     $OUTPUT_DIR"
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
