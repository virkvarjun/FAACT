# FAACT on SO101: rollout checklist

Use [`run_hardware_faact.py`](../scripts/run_hardware_faact.py) as the main entry point. It mirrors the simulation stack: `BackbonePolicyWrapper`, `merge_feature_dicts`, `TorchMLPRiskScorer`, `ThresholdInterventionPolicy`, and (when `runtime.mode: intervene`) shared [`run_intervention_search`](../../faact/faact/evaluation/online_runner.py) from `faact.evaluation.online_runner`.

## Phase 0 — Preconditions

- **LeRobot install**: `pip install -e /path/to/lerobot` with PI0/ACT dependencies as needed.
- **HF token** (if using gated checkpoints): `export HF_TOKEN=...`
- **GPU**: set `backbone.device: cuda` or `cpu` in YAML.
- **Calibration**: use `robot.connect_calibrate: false` and matching `robot_id` so existing calibration is applied without prompts (see [HARDWARE_SAFETY.md](HARDWARE_SAFETY.md)).
- **Observation contract**: real runs use `agent_pos` + `pixels` via [`so101_bridge`](../faact_hardware/so101_bridge.py). PI0/ACT wrappers accept `observation.state` or `agent_pos`.

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

## Tuning notes (domain shift)

- Simulation-trained risk models are often miscalibrated on real images; lower or raise `risk_threshold`, or recalibrate / fine-tune on a small real dataset.
- Point `backbone.checkpoint` at a policy trained on **your** SO101 task for meaningful behavior.
