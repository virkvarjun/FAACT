# RunPod ACT Training — Persistent & Laptop-Safe

This guide ensures checkpoints are saved to `/workspace` (persistent) and training runs in the background so you can close your laptop.

---

## Step 1: Start a New GPU Pod

1. Go to https://www.runpod.io
2. **Terminate** your current CPU Pod (if running)
3. **Deploy** a new Pod:
   - **GPU**: RTX 4090, A40, or A100 (24GB+ VRAM recommended)
   - **Container**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` or similar
   - **Volume**: 50–100 GB (persistent storage — **required**)
   - **Mount path**: `/workspace`
4. Wait for the Pod to start
5. Copy your SSH command from the RunPod Connect section

---

## Step 2: SSH In and Run Setup

```bash
# From your Mac terminal
ssh -i ~/.ssh/id_ed25519 <YOUR_POD_ID>@ssh.runpod.io
```

Once connected, run the setup (clones to `/workspace` for persistence):

```bash
# Install git if needed
apt-get update -qq && apt-get install -y -qq git

# Clone to /workspace (persistent)
cd /workspace
if [ -d "Research" ]; then rm -rf Research; fi
git clone --recurse-submodules https://github.com/virkvarjun/Research.git

# Run setup script (installs Miniforge, conda env, lerobot)
cd /workspace/Research
bash setup_runpod.sh
```

**Note:** If `setup_runpod.sh` clones to `/root/Research`, edit it or run:

```bash
cd /workspace
rm -rf Research 2>/dev/null
git clone --recurse-submodules https://github.com/virkvarjun/Research.git
cd Research
# Then run the rest of setup (miniforge, conda, pip install) - 
# you may need to adjust paths in setup_runpod.sh to use /workspace
```

---

## Step 3: Start a tmux Session (So You Can Close Laptop)

```bash
# Start tmux
tmux new -s train

# You're now in a persistent session. Closing SSH won't kill it.
```

---

## Step 4: Run Training (Persistent Output)

```bash
cd /workspace/Research
bash scripts/train_act_persistent.sh transfer_cube 100000 8
```

- Checkpoints save to: `/workspace/Research/outputs/train/act_transfer_cube/`
- This path is on the volume — it survives Pod restarts
- Training runs in tmux — safe to close laptop

---

## Step 5: Detach from tmux (Close Laptop)

**To leave training running and disconnect:**

```
Press: Ctrl+B, then D
```

You can close your laptop. Training continues on the server.

---

## Step 6: Reconnect Later

```bash
# SSH back in
ssh -i ~/.ssh/id_ed25519 <YOUR_POD_ID>@ssh.runpod.io

# Reattach to tmux
tmux attach -t train
```

---

## Step 7: After Training — Checkpoint Location

Checkpoint path:

```
/workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model
```

Use this for the failure prediction collect script:

```bash
--checkpoint /workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model
```

---

## Quick Reference

| What | Command |
|------|---------|
| Start tmux | `tmux new -s train` |
| Detach (leave running) | `Ctrl+B` then `D` |
| Reattach | `tmux attach -t train` |
| List sessions | `tmux ls` |

---

## Optional: Extra Backup to Hugging Face

After training finishes:

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli upload <YOUR_USERNAME>/act_transfer_cube \
  /workspace/Research/outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \
  .
```
