#!/bin/bash
# Full failure-aware ACT pipeline: analysis -> online eval -> plots -> report
# Run from repo root with PYTHONPATH including Research (or Research/Research for failure_prediction)
#
# Prerequisites: ACT checkpoint, processed dataset, trained supervised predictor
# Set these or pass as env vars:
ACT_CHECKPOINT="${ACT_CHECKPOINT:-outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model}"
RISK_MODEL="${RISK_MODEL:-failure_prediction_runs/transfer_cube_supervised}"
PROCESSED_DIR="${PROCESSED_DIR:-failure_dataset/transfer_cube/processed}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$RISK_MODEL}"
BASE="failure_prediction_runs"

set -e
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-Research:Research/Research}"

echo "=== 1. Offline analysis (threshold sweep + lead-time) ==="
python -m failure_prediction.scripts.analyze_failure_predictor \
  --predictions_dir "$PREDICTIONS_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --output_dir "${BASE}/transfer_cube_analysis"

THRESHOLD=$(python -c "
import json
p = '${BASE}/transfer_cube_analysis/threshold_sweep.json'
try:
    print(json.load(open(p))['recommended_threshold'])
except Exception:
    print('0.5')
")
echo "Recommended threshold: $THRESHOLD"

echo "=== 2. Online evaluation ==="
python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \
  --mode baseline --output_dir "${BASE}/online_eval_baseline" --device cuda

python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \
  --mode monitor_only --risk_model_ckpt "$RISK_MODEL" \
  --risk_threshold "$THRESHOLD" --output_dir "${BASE}/online_eval_monitor" --device cuda

python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint "$ACT_CHECKPOINT" \
  --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \
  --mode intervention --risk_model_ckpt "$RISK_MODEL" \
  --risk_threshold "$THRESHOLD" --num_candidate_chunks 5 \
  --output_dir "${BASE}/online_eval_intervention" --device cuda

echo "=== 3. Plots ==="
python -m failure_prediction.scripts.plot_final_results \
  --run_dirs \
    "$RISK_MODEL" \
    "${BASE}/transfer_cube_analysis" \
    "${BASE}/online_eval_baseline" \
    "${BASE}/online_eval_monitor" \
    "${BASE}/online_eval_intervention" \
  --output_dir "${BASE}/final_plots"

echo "=== 4. Final report ==="
python -m failure_prediction.scripts.generate_final_report \
  --processed_dir "${PROCESSED_DIR}" \
  --supervised_dir "$RISK_MODEL" \
  --analysis_dir "${BASE}/transfer_cube_analysis" \
  --online_baseline "${BASE}/online_eval_baseline" \
  --online_monitor "${BASE}/online_eval_monitor" \
  --online_intervention "${BASE}/online_eval_intervention" \
  --checkpoint_path a23v/act_transfer_cube \
  --output failure_dataset/transfer_cube/final_project_report.md

echo "Done. Report: failure_dataset/transfer_cube/final_project_report.md"
