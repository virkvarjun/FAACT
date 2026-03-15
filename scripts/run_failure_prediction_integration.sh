#!/usr/bin/env bash
# Run the full failure prediction integration pipeline on real ACT rollout data.
#
# Prerequisites:
#   - Trained ACT checkpoint (e.g. outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model)
#   - Run from Research/ directory
#   - PYTHONPATH must include lerobot/src for ACT imports
#
# Usage (RunPod, where checkpoint lives):
#   cd /root/Research
#   PYTHONPATH=lerobot/src:$PYTHONPATH bash scripts/run_failure_prediction_integration.sh
#
# Usage (local with copied checkpoint):
#   CHECKPOINT=/path/to/pretrained_model OUTPUT_BASE=./failure_dataset bash scripts/run_failure_prediction_integration.sh

set -e

# Paths (override via env)
WORKSPACE="${WORKSPACE:-$(pwd)}"
CHECKPOINT="${CHECKPOINT:-${WORKSPACE}/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model}"
OUTPUT_BASE="${OUTPUT_BASE:-${WORKSPACE}/failure_dataset}"
RUNS_BASE="${RUNS_BASE:-${WORKSPACE}/failure_prediction_runs}"
PYTHONPATH="${PYTHONPATH:-}"
if [[ -z "$PYTHONPATH" ]] && [[ -d "${WORKSPACE}/lerobot" ]]; then
  export PYTHONPATH="${WORKSPACE}/lerobot/src:${PYTHONPATH}"
fi

# Pipeline config
TASK="${TASK:-AlohaTransferCube-v0}"
ENV_TYPE="${ENV_TYPE:-aloha}"
NUM_EPISODES="${NUM_EPISODES:-30}"
POLICY_TYPE="${POLICY_TYPE:-act}"  # act or pi0
TASK_DESC="${TASK_DESC:-}"  # Language task for π₀ (optional; default from task ID)
FAILURE_HORIZON="${FAILURE_HORIZON:-10}"
DEVICE="${DEVICE:-cuda}"
MUJOCO_GL="${MUJOCO_GL:-egl}"  # Use EGL for headless (e.g. RunPod)

RAW_DIR="${OUTPUT_BASE}/transfer_cube/raw"
PROCESSED_DIR="${OUTPUT_BASE}/transfer_cube/processed"
INSPECTION_JSON="${OUTPUT_BASE}/transfer_cube/inspection_report.json"
FEATURE_REPORT="${OUTPUT_BASE}/transfer_cube/feature_report.json"

# Default feature field (inspect_feature_fields will recommend; we use decoder_mean as default)
FEATURE_FIELD="${FEATURE_FIELD:-feat_decoder_mean}"

echo "=== Failure Prediction Integration Pipeline ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Episodes: ${NUM_EPISODES}"
echo ""

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
  echo "Set CHECKPOINT env var or ensure the path exists."
  exit 1
fi

# --- Phase 2: Collect raw rollout data ---
echo ">>> Phase 2: Collecting ${NUM_EPISODES} rollout episodes (policy=${POLICY_TYPE})..."
mkdir -p "${OUTPUT_BASE}/transfer_cube"
COLLECT_ARGS=(
  --checkpoint "$CHECKPOINT"
  --task "$TASK"
  --env_type "$ENV_TYPE"
  --num_episodes "$NUM_EPISODES"
  --output_dir "${OUTPUT_BASE}/transfer_cube"
  --device "$DEVICE"
  --failure_horizon "$FAILURE_HORIZON"
  --policy_type "$POLICY_TYPE"
  --save_embeddings
  --save_action_chunks
)
[[ -n "$TASK_DESC" ]] && COLLECT_ARGS+=(--task_desc "$TASK_DESC")
MUJOCO_GL="$MUJOCO_GL" python -m failure_prediction.scripts.collect_failure_dataset "${COLLECT_ARGS[@]}"

# --- Phase 2: Postprocess ---
echo ""
echo ">>> Postprocessing raw episodes..."
python -m failure_prediction.scripts.postprocess_failure_dataset \
  --input_dir "$RAW_DIR" \
  --output_dir "$PROCESSED_DIR" \
  --failure_horizon "$FAILURE_HORIZON"

# --- Phase 2: Inspect and validate ---
echo ""
echo ">>> Inspecting and validating dataset..."
python -m failure_prediction.scripts.inspect_failure_dataset \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --failure_horizon "$FAILURE_HORIZON" \
  --sample_episode random \
  --json_report "$INSPECTION_JSON"

# --- Phase 3: Inspect feature fields and choose default ---
echo ""
echo ">>> Inspecting feature fields..."
python -m failure_prediction.scripts.inspect_feature_fields \
  --processed_dir "$PROCESSED_DIR" \
  --json_out "$FEATURE_REPORT"

# Use recommended field if available
if [[ -f "$FEATURE_REPORT" ]]; then
  REC=$(python -c "
import json
with open('$FEATURE_REPORT') as f:
  r = json.load(f)
cands = r.get('embedding_candidates', [])
print(cands[0] if cands else 'feat_decoder_mean')
" 2>/dev/null || echo "feat_decoder_mean")
  FEATURE_FIELD="$REC"
  echo "Using recommended feature field: $FEATURE_FIELD"
fi

# --- Phase 4: Train supervised predictor ---
echo ""
echo ">>> Phase 4: Training supervised failure predictor..."
SUPERVISED_DIR="${RUNS_BASE}/transfer_cube_supervised"
mkdir -p "$SUPERVISED_DIR"
python -m failure_prediction.scripts.train_failure_predictor \
  --processed_dir "$PROCESSED_DIR" \
  --feature_field "$FEATURE_FIELD" \
  --label_field failure_within_k \
  --output_dir "$SUPERVISED_DIR" \
  --epochs 20 \
  --batch_size 256 \
  --device "$DEVICE"

# --- Phase 5: Run FIPER baseline (trains RND if no checkpoint, calibrates, evaluates) ---
echo ""
echo ">>> Phase 5: Running FIPER baseline..."
FIPER_DIR="${RUNS_BASE}/transfer_cube_fiper"
mkdir -p "$FIPER_DIR"
python -m failure_prediction.scripts.run_fiper_offline_eval \
  --processed_dir "$PROCESSED_DIR" \
  --feature_field "$FEATURE_FIELD" \
  --action_chunk_field predicted_action_chunk \
  --output_dir "$FIPER_DIR" \
  --window_size 3 \
  --alpha 0.1 \
  --device "$DEVICE"

# --- Phase 6: Generate plots ---
echo ""
echo ">>> Phase 6: Generating plots..."
PLOTS_DIR="${RUNS_BASE}/plots"
mkdir -p "$PLOTS_DIR"
python -m failure_prediction.scripts.plot_failure_results \
  --run_dir "$SUPERVISED_DIR" \
  --output_dir "${PLOTS_DIR}/supervised"
python -m failure_prediction.scripts.plot_failure_results \
  --run_dir "$FIPER_DIR" \
  --output_dir "${PLOTS_DIR}/fiper"

# --- Phase 7: Generate integration report ---
echo ""
echo ">>> Phase 7: Generating integration report..."
REPORT_PATH="${OUTPUT_BASE}/transfer_cube/integration_report.md"
python -m failure_prediction.scripts.generate_integration_report \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --supervised_dir "$SUPERVISED_DIR" \
  --fiper_dir "$FIPER_DIR" \
  --checkpoint_path "$CHECKPOINT" \
  --output "$REPORT_PATH"

echo ""
echo "=== Integration pipeline complete ==="
echo "  Raw data:        $RAW_DIR"
echo "  Processed data:  $PROCESSED_DIR"
echo "  Supervised run:  $SUPERVISED_DIR"
echo "  FIPER run:      $FIPER_DIR"
echo "  Plots:          ${PLOTS_DIR}"
echo "  Report:         ${REPORT_PATH}"
