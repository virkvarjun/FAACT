#!/bin/bash
# RunPod: install deps + run failure-aware eval in tmux.
# Use when Research is on network volume at /workspace/Research.
# Usage: bash scripts/setup_and_run_eval.sh
#
set -e

WORKSPACE="${WORKSPACE:-/workspace}"
RESEARCH_DIR="${WORKSPACE}/Research"
CHECKPOINT="${RESEARCH_DIR}/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model"
RISK_CKPT="${RESEARCH_DIR}/failure_prediction_runs/transfer_cube_100ep_supervised"
OUTPUT_DIR="${RESEARCH_DIR}/failure_prediction_runs/online_eval_intervention_th03"

echo "=== [1/5] System deps (Mesa EGL, tmux) ==="
apt-get update -qq && apt-get install -y -qq \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    ffmpeg tmux \
    > /dev/null 2>&1

export MUJOCO_GL=egl
grep -q "MUJOCO_GL" ~/.bashrc 2>/dev/null || echo 'export MUJOCO_GL=egl' >> ~/.bashrc

echo "=== [2/5] Conda + lerobot env ==="
if [ ! -f "/root/miniforge3/bin/conda" ]; then
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /root/miniforge3
    rm /tmp/miniforge.sh
fi
eval "$(/root/miniforge3/bin/conda shell.bash hook)"

if ! conda env list | grep -q "lerobot"; then
    conda create -y -n lerobot python=3.12
fi
conda activate lerobot

echo "=== [3/5] Install lerobot + aloha + gymnasium ==="
cd "${RESEARCH_DIR}"
git submodule update --init 2>/dev/null || true
pip install -e ./lerobot 2>/dev/null || pip install -e "${RESEARCH_DIR}/lerobot"
pip install gymnasium gym-aloha "shimmy[gym-v26]" 2>/dev/null || true

echo "=== [4/5] Verify ==="
python -c "
import torch, gymnasium
from lerobot.policies.act.modeling_act import ACTPolicy
print('OK: CUDA', torch.cuda.is_available())
"

echo "=== [5/5] Launch eval in tmux ==="
WRAPPER="${RESEARCH_DIR}/.run_eval_tmp.sh"
cat > "${WRAPPER}" << EOF
eval "\$(/root/miniforge3/bin/conda shell.bash hook)"
conda activate lerobot
cd ${RESEARCH_DIR} && MUJOCO_GL=egl PYTHONPATH=${RESEARCH_DIR}:\$PYTHONPATH python -m failure_prediction.scripts.run_failure_aware_eval \\
  --checkpoint ${CHECKPOINT} \\
  --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \\
  --mode intervention \\
  --risk_model_ckpt failure_prediction_runs/transfer_cube_100ep_supervised \\
  --risk_threshold 0.3 --num_candidate_chunks 5 \\
  --output_dir failure_prediction_runs/online_eval_intervention_th03 \\
  --device cuda
echo ""
echo "=== Done ==="
exec bash
EOF
chmod +x "${WRAPPER}"

if [ -n "${TMUX:-}" ]; then
    echo "Already in tmux, running directly..."
    source "${WRAPPER}"
else
    echo "Starting tmux session 'eval'..."
    tmux new-session -d -s eval "bash -l ${WRAPPER}"
    echo ""
    echo "Eval running in tmux. Attach with:  tmux attach -t eval"
    echo "Detach (leave running):  Ctrl+B, then D"
fi
