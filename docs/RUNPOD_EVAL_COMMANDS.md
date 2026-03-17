# RunPod: Full Setup + Run Failure-Aware Eval

When you have a **new GPU pod** with **Research on the network volume** at `/workspace/Research`, use these commands.

---

## Option 1: One-command script (recommended)

```bash
# 1. SSH in
ssh 2wwjqxy535aad2-64411d3e@ssh.runpod.io -i ~/.ssh/id_ed25519

# 2. Start tmux first (so setup survives disconnects)
tmux new -s setup

# 3. Run the setup + eval script
cd /workspace/Research && bash scripts/setup_and_run_eval.sh
```

The script installs deps, then launches the eval in a **second tmux session** named `eval`. You can detach (`Ctrl+B`, `D`) and reconnect later with `tmux attach -t eval`.

---

## Option 2: Manual commands

If the script fails or you prefer step-by-step:

```bash
# 1. SSH + tmux
ssh 2wwjqxy535aad2-64411d3e@ssh.runpod.io -i ~/.ssh/id_ed25519
tmux new -s work

# 2. System deps
apt-get update && apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev ffmpeg tmux
export MUJOCO_GL=egl

# 3. Conda + env
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /root/miniforge3 && rm /tmp/miniforge.sh
eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda create -y -n lerobot python=3.12
conda activate lerobot

# 4. Install lerobot + deps
cd /workspace/Research
git submodule update --init
pip install -e ./lerobot
pip install gymnasium gym-aloha "shimmy[gym-v26]"

# 5. Run eval (stays in tmux)
cd /workspace/Research && MUJOCO_GL=egl PYTHONPATH=/workspace/Research:$PYTHONPATH python -m failure_prediction.scripts.run_failure_aware_eval \
  --checkpoint /workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \
  --task AlohaTransferCube-v0 --env_type aloha --num_episodes 20 \
  --mode intervention \
  --risk_model_ckpt failure_prediction_runs/transfer_cube_100ep_supervised \
  --risk_threshold 0.3 --num_candidate_chunks 5 \
  --output_dir failure_prediction_runs/online_eval_intervention_th03 \
  --device cuda
```

---

## Tmux cheat sheet

| Action | Keys |
|--------|------|
| Detach (leave running) | `Ctrl+B`, then `D` |
| Attach to session | `tmux attach -t eval` (or `setup`, `work`) |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t eval` |

---

## If using direct SSH (not RunPod proxy)

```bash
ssh root@<PUBLIC_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

Get `<PUBLIC_IP>` and `<PORT>` from RunPod dashboard → Connect → SSH over exposed TCP.
