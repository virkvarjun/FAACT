#!/usr/bin/env bash
# Step 1: smoke test (follower + camera, PI0, dry_run, 30 steps).
# Usage: set PY to your conda Python, export HF_TOKEN if needed, then:
#   bash faact_hardware/scripts/run_step1_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/faact_hardware:${ROOT}/faact:${ROOT}:${PYTHONPATH:-}"

: "${PY:?Set PY to conda env python, e.g. export PY=\$CONDA_PREFIX/bin/python}"

cd "$ROOT"
"$PY" faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_smoke_test.yaml \
  --use-real-robot \
  --seed 0
