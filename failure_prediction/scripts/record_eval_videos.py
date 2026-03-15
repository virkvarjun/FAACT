#!/usr/bin/env python
"""Record video samples of failure, intervention, and success episodes.

Runs episodes until it collects: 2 failure, 2 intervention, 1 success.
Saves 5 mp4 videos to the output directory.

Example:
    PYTHONPATH=Research MUJOCO_GL=egl python -m failure_prediction.scripts.record_eval_videos \\
        --checkpoint outputs/train/act_transfer_cube/checkpoints/100000/pretrained_model \\
        --risk_model_ckpt failure_prediction_runs/transfer_cube_supervised \\
        --risk_threshold 0.5 \\
        --output_dir failure_prediction_runs/videos \\
        --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.scripts.run_failure_aware_eval import (
    load_risk_model,
    add_obs_noise,
)
from failure_prediction.scripts.collect_failure_dataset import (
    load_policy_and_processors,
    make_single_env,
    preprocess_obs,
    predict_action_chunk_with_features,
    features_to_numpy,
)
from failure_prediction.utils.success_inference import infer_episode_outcome


def get_frame(env, obs=None):
    """Get current frame from env (rgb array, HWC uint8)."""
    try:
        frame = env.render()
        if frame is not None and frame.size > 0:
            arr = np.asarray(frame)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 4:
                return arr[0]
    except Exception:
        pass
    if obs is not None and "pixels" in obs:
        pix = obs["pixels"]
        if isinstance(pix, dict):
            v = next(iter(pix.values()))
        else:
            v = pix
        arr = np.asarray(v).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return arr[..., :3]
        if arr.ndim == 4:
            return arr[0, ..., :3]
    return np.zeros((480, 640, 3), dtype=np.uint8)


def run_episode_with_frames(
    env,
    policy,
    preprocessor,
    postprocessor,
    n_action_steps: int,
    device: str,
    risk_model,
    risk_key: str,
    risk_threshold: float,
    num_candidate_chunks: int,
    obs_noise_std: float,
    seed: int,
) -> tuple[dict, list[np.ndarray]]:
    """Run one episode in intervention mode, return result and frames."""
    rng = np.random.default_rng(seed)
    policy.reset()
    raw_obs, info = env.reset(seed=int(seed))
    frames = [get_frame(env, raw_obs)]
    current_chunk = None
    chunk_step_idx = 0
    max_ep_steps = env.spec.max_episode_steps or 400

    episode_rewards = []
    episode_successes = []
    episode_dones = []
    episode_terminated = []
    episode_truncated = []
    interventions = []

    done = False
    step = 0

    while not done and step < max_ep_steps:
        obs_dict = preprocess_obs(raw_obs)
        obs_processed = preprocessor(obs_dict)
        need_new_chunk = (current_chunk is None) or (chunk_step_idx >= n_action_steps)

        if need_new_chunk:
            action_chunk, features = predict_action_chunk_with_features(policy, obs_processed)
            feat_np = features_to_numpy(features)
            feat_vec = feat_np.get(risk_key)
            risk_prob = 0.0
            if risk_model is not None and feat_vec is not None:
                with torch.no_grad():
                    x = torch.from_numpy(feat_vec).float().unsqueeze(0).to(device)
                    logit = risk_model(x).cpu().item()
                    risk_prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
                if risk_prob >= risk_threshold:
                    candidates = []
                    for _ in range(num_candidate_chunks):
                        noisy_obs = add_obs_noise(raw_obs, noise_std=obs_noise_std, rng=rng)
                        noisy_dict = preprocess_obs(noisy_obs)
                        noisy_processed = preprocessor(noisy_dict)
                        chunk_cand, feat_cand = predict_action_chunk_with_features(policy, noisy_processed)
                        feat_cand_np = features_to_numpy(feat_cand)
                        fv = feat_cand_np.get(risk_key)
                        if fv is not None:
                            with torch.no_grad():
                                xc = torch.from_numpy(fv).float().unsqueeze(0).to(device)
                                lc = risk_model(xc).cpu().item()
                                pc = 1.0 / (1.0 + np.exp(-np.clip(lc, -500, 500)))
                            candidates.append((chunk_cand, pc))
                    if candidates:
                        best_idx = np.argmin([c[1] for c in candidates])
                        action_chunk = candidates[best_idx][0]
                        interventions.append({"step": step})
            current_chunk = action_chunk
            chunk_step_idx = 0

        action = current_chunk[:, chunk_step_idx]
        action = postprocessor(action)
        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]

        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        frames.append(get_frame(env, raw_obs))
        success_this_step = bool(info.get("is_success", False))
        done = terminated or truncated

        episode_rewards.append(float(reward))
        episode_successes.append(success_this_step)
        episode_dones.append(done)
        episode_terminated.append(terminated)
        episode_truncated.append(truncated)
        chunk_step_idx += 1
        step += 1

    outcome = infer_episode_outcome(
        rewards=np.array(episode_rewards),
        successes=np.array(episode_successes),
        dones=np.array(episode_dones),
        terminated=np.array(episode_terminated),
        truncated=np.array(episode_truncated),
        env_name=env.unwrapped.spec.id if hasattr(env, "unwrapped") else "",
    )

    return {
        "success": outcome["success"],
        "n_interventions": len(interventions),
    }, frames


def _draw_label_on_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw label text (SUCCESS / INTERVENTION / FAILURE) on frame. Returns RGB copy."""
    frame = np.asarray(frame).copy()
    try:
        import cv2
        # cv2 uses BGR; frame is RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(frame.shape[0], frame.shape[1]) / 400.0
        thick = max(1, int(scale * 2))
        color_bgr = (0, 200, 0) if label == "success" else (0, 140, 255) if label == "intervention" else (0, 0, 255)
        cv2.putText(bgr, label.upper(), (20, 40), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(bgr, label.upper(), (20, 40), font, scale, color_bgr, thick, cv2.LINE_AA)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return frame


def _write_videos_readme(out_dir: Path, collected: dict[str, int]) -> None:
    """Write README listing videos with their labels."""
    readme = out_dir / "README.md"
    videos = sorted(out_dir.glob("*.mp4"))
    lines = [
        "# Eval Videos",
        "",
        "| Label | Count | Description |",
        "|-------|-------|-------------|",
        "| **Success** | {} | Episode completed successfully |".format(collected.get("success", 0)),
        "| **Intervention** | {} | Risk-triggered chunk re-selection prevented failure |".format(collected.get("intervention", 0)),
        "| **Failure** | {} | Episode failed (no recovery) |".format(collected.get("failure", 0)),
        "",
        "## Files",
        "",
    ]
    for v in videos:
        # Parse label from filename: success_0_ep12.mp4 -> success
        parts = v.stem.split("_")
        lb = parts[0] if parts else "?"
        lines.append(f"- `{v.name}` — **{lb.upper()}**")
    lines.extend([
        "",
        "Videos are named `{label}_{idx}_ep{ep}.mp4`. The label is also overlaid on each frame.",
    ])
    with open(readme, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {readme}")


def save_video(frames: list[np.ndarray], path: Path, fps: int = 30, label: str | None = None):
    """Save frames to mp4. If label is given, overlay it on every frame."""
    if label:
        frames = [_draw_label_on_frame(f, label) for f in frames]
    try:
        import imageio
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(np.asarray(f))
        writer.close()
        return True
    except ImportError:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            return True
        except Exception:
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--risk_model_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="failure_prediction_runs/videos")
    p.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    p.add_argument("--env_type", type=str, default="aloha")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--num_candidate_chunks", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_episodes", type=int, default=100, help="Max episodes to run before giving up")
    p.add_argument("--n_failure", type=int, default=2)
    p.add_argument("--n_intervention", type=int, default=2)
    p.add_argument("--n_success", type=int, default=1)
    args = p.parse_args()

    importlib.import_module(f"gym_{args.env_type}")
    policy, preprocessor, postprocessor = load_policy_and_processors(args.checkpoint, args.device)
    risk_model, risk_key = load_risk_model(Path(args.risk_model_ckpt), args.device)
    env = make_single_env(args.task, args.env_type)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = {"failure": 0, "intervention": 0, "success": 0}
    rng = np.random.default_rng(args.seed)

    for ep in range(args.max_episodes):
        seed = int(rng.integers(0, 2**31))
        result, frames = run_episode_with_frames(
            env, policy, preprocessor, postprocessor,
            policy.config.n_action_steps, args.device,
            risk_model, risk_key, args.risk_threshold,
            args.num_candidate_chunks, 0.03, seed,
        )
        success = result["success"]
        n_int = result["n_interventions"]

        label = None
        if not success and collected["failure"] < args.n_failure:
            label = "failure"
        elif n_int > 0 and collected["intervention"] < args.n_intervention:
            label = "intervention"
        elif success and collected["success"] < args.n_success:
            label = "success"

        if label:
            idx = collected[label]
            collected[label] += 1
            out_path = out_dir / f"{label}_{idx}_ep{ep}.mp4"
            if save_video(frames, out_path, label=label):
                print(f"Saved {out_path} ({len(frames)} frames)")

        if all(
            collected[k] >= v
            for k, v in [("failure", args.n_failure), ("intervention", args.n_intervention), ("success", args.n_success)]
        ):
            break

    env.close()
    _write_videos_readme(out_dir, collected)
    print(f"Done. Collected: {collected}")


if __name__ == "__main__":
    main()
