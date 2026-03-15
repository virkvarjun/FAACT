#!/bin/bash
# Record 5 intervention videos. Run from Research/ on RunPod or similar.
#
# Usage:
#   cd /workspace/Research && bash scripts/record_5_interventions.sh

set -e
PYTHONPATH="$(pwd)" MUJOCO_GL=egl python -m failure_prediction.scripts.record_eval_videos \
  --checkpoint a23v/act_transfer_cube \
  --risk_model_ckpt failure_prediction_runs/transfer_cube_supervised \
  --output_dir failure_prediction_runs/videos \
  --n_intervention 5 \
  --n_failure 0 \
  --n_success 0 \
  --device cuda

echo "Done. Videos saved to failure_prediction_runs/videos/"
echo "Commit and push:"
echo "  git add Research/failure_prediction_runs/videos/intervention_*.mp4"
echo "  git commit -m 'Add 5 intervention eval videos'"
echo "  git push origin main"
