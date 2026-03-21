# FAACT on SO101: rollout checklist

Use [`run_hardware_faact.py`](../scripts/run_hardware_faact.py) as the main entry point. It mirrors the simulation stack: `BackbonePolicyWrapper`, `merge_feature_dicts`, `TorchMLPRiskScorer`, `ThresholdInterventionPolicy`, and (when `runtime.mode: intervene`) shared [`run_intervention_search`](../../faact/faact/evaluation/online_runner.py) from `faact.evaluation.online_runner`.

## Phase 0 — Preconditions

- **LeRobot install**: `pip install -e /path/to/lerobot` with PI0/ACT dependencies as needed.
- **HF token** (if using gated checkpoints): `export HF_TOKEN=...`
- **GPU**: set `backbone.device: cuda` or `cpu` in YAML.
- **Calibration**: use `robot.connect_calibrate: false` and matching `robot_id` so existing calibration is applied without prompts (see [HARDWARE_SAFETY.md](HARDWARE_SAFETY.md)).
- **Observation contract**: real runs use `agent_pos` + `pixels` via [`so101_bridge`](../faact_hardware/so101_bridge.py). PI0/ACT wrappers accept `observation.state` or `agent_pos`.

## Step 1 — Physical smoke test (quick)

Use [`so101_smoke_test.yaml`](../configs/so101_smoke_test.yaml): **`risk.enabled: false`**, **`dry_run: true`**, **`max_steps: 30`**. Edit **`robot.cameras.top.index_or_path`** if the iPhone / Continuity camera is not index `0`.

```bash
cd /path/to/Research
export PY="$CONDA_PREFIX/bin/python"   # after: conda activate your-env
export HF_TOKEN="..."                  # if hub download needs it
export PYTHONPATH="faact_hardware:faact:."

$PY faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_smoke_test.yaml \
  --use-real-robot \
  --seed 0
```

Optional CLI overrides: `--max-steps 20`, `--dry-run`, `--no-dry-run`.

Or: `bash faact_hardware/scripts/run_step1_smoke.sh` (requires `export PY=...` first).

**Exit criteria**: process exits 0, logs under `faact_hardware/runs/so101_smoke_test/`.

## Phase 1 — Shadow (policy + risk, no physical motion)

Use **dummy observations** and `DryRunRobot` (default without `--use-real-robot`):

```bash
cd /path/to/Research
pip install -e faact_hardware
PYTHONPATH="faact_hardware:faact:." python faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_transfer_cube.yaml
```

**Exit criteria**: episode completes, `hardware_episode.jsonl` written, no uncaught exceptions.

## Phase 2 — Alarm-only on real hardware (logging only)

1. Edit YAML: `robot.dry_run: true`, `runtime.mode: shadow` (or `alarm_only` for labeling), `risk.enabled: true`, valid `risk.risk_model_ckpt`.
2. Configure `robot.cameras` (see example in `configs/so101_transfer_cube.yaml`).
3. Run:

```bash
PYTHONPATH="faact_hardware:faact:." python faact_hardware/scripts/run_hardware_faact.py \
  --config faact_hardware/configs/so101_transfer_cube.yaml \
  --use-real-robot
```

**Exit criteria**: alarms in JSONL roughly align with visible instability; tune `risk.risk_threshold` if needed (domain shift from simulation).

## Phase 3 — Closed-loop (follower executes policy actions)

1. Set `robot.dry_run: false` in YAML.
2. Keep `runtime.mode: shadow` first to verify actions in logs, or switch to short `runtime.max_steps`.
3. Supervise closely; use `robot.max_relative_target` (e.g. `0.1`–`0.3`) for safer goal jumps.

**Exit criteria**: smooth motion, no comms faults, `max_abs_action_value` appropriate for your policy’s output scale.

## Phase 4 — Intervention mode

1. Set `runtime.mode: intervene` in YAML.
2. Ensure `risk.enabled: true` and intervention hyperparameters match your sim sweep (`num_candidate_chunks`, `switch_margin`, locality knobs).
3. Run with `--use-real-robot` only after Phases 1–3 pass.

**Exit criteria**: `hardware_interventions.jsonl` records attempts; `accepted_interventions` reflects successful chunk replacements.

## Camera: Mac built-in, iPhone (Continuity), third-party apps

LeRobot uses **OpenCV**-style camera indices (`robot.cameras.*.index_or_path` in YAML).

- **Built-in FaceTime camera** on Mac is often index **`0`**.
- **iPhone as Mac webcam** (Continuity Camera) or **third-party apps** (Camo, EpocCam, etc. that install a **virtual webcam**) usually appear as **another index** — try **`0`**, **`1`**, **`2`** until the image in logs matches what you expect.
- Third-party apps do **not** need custom FAACT code: they expose a normal macOS camera device; only the **index** in YAML changes.
- Grant **Camera** permission for Terminal / your IDE (macOS **Privacy & Security**).

See also [SO101_USB_PORTS.md](SO101_USB_PORTS.md) (camera subsection).

## Simulation vs your physical bench

If FAACT’s **risk model** was trained in **simulation** and/or the **policy** is a **base** checkpoint (`lerobot/pi0_base`) while your **task, objects, lighting, and camera view** (e.g. iPhone on a tripod) differ from sim:

- **Risk scores** may be poorly calibrated or noisy; treat **`risk.enabled: true`** as experimental until you validate or retrain on **real** trajectories.
- **Policy behavior** may be weak or irrelevant until you **fine-tune** (or fully train) on **your** SO101 dataset for **your** task.
- **One leader + one follower** is normal: teleop uses both; **autonomous FAACT** uses the **follower** + **camera** only (leader can stay disconnected when debugging USB if needed).

This is **expected domain shift**, not a sign the wiring is wrong.

## Tuning notes (domain shift)

- Simulation-trained risk models are often miscalibrated on real images; lower or raise `risk_threshold`, or recalibrate / fine-tune on a small real dataset.
- Point `backbone.checkpoint` at a policy trained on **your** SO101 task for meaningful behavior.

## Next steps (recommended order for your situation)

1. **Stabilize hardware**: follower USB + camera index (iPhone/virtual cam = try indices until correct). Keep **`robot.dry_run: true`** until the run completes without comms errors.
2. **Run policy-only**: leave **`risk.enabled: false`**, **`runtime.mode: shadow`**, short **`max_steps`**. Confirm logs and that observations look sane.
3. **Closed-loop (supervised)**: **`robot.dry_run: false`**, conservative **`max_relative_target`**, still short episodes.
4. **Data**: record a **LeRobot dataset** on **this** task with **this** camera + follower (`lerobot-record` or your usual pipeline).
5. **Policy**: fine-tune (or train) a checkpoint on that dataset; point **`backbone.checkpoint`** at it.
6. **Risk**: collect failure labels or proxy outcomes on **real** features; retrain the MLP (or recalibrate threshold); set **`risk.enabled: true`** and validate alarms.
7. **FAACT intervention**: set **`runtime.mode: intervene`** only after alarms look trustworthy; tune **`switch_margin`**, candidate counts, locality knobs as in sim.
