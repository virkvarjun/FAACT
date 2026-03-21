#!/usr/bin/env python3
"""Run FAACT on physical SO101 (or dummy obs) with shadow / alarm / intervene modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))
sys.path.insert(0, str(RESEARCH_DIR / "faact"))
sys.path.insert(0, str(RESEARCH_DIR / "faact_hardware"))

from faact.backbone.factory import make_backbone_wrapper
from faact_hardware.config import load_config
from faact_hardware.runtime import DryRunRobot, HardwareRuntime
from faact_hardware.so101_bridge import (
    LeRobotSO101FollowerRobot,
    LeRobotSO101ObservationAdapter,
    lerobot_available,
    make_so101_follower_robot,
)
from failure_prediction.runtime_components import (
    ThresholdInterventionPolicy,
    load_supervised_risk_runtime,
)


class DummyObservationAdapter:
    """Synthetic obs for CI / tests without hardware."""

    def reset(self) -> None:
        return None

    def get_observation(self) -> dict[str, np.ndarray] | None:
        return {
            "agent_pos": np.zeros(6, dtype=np.float32),
            "pixels": {
                "top": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAACT hardware runner (SO101)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--use-real-robot",
        action="store_true",
        help="Connect to SO101 follower via LeRobot (requires cameras + USB)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override config runtime.max_steps (e.g. 30 for smoke tests)",
    )
    dry = p.add_mutually_exclusive_group()
    dry.add_argument(
        "--dry-run",
        dest="dry_run_override",
        action="store_true",
        help="Force robot.dry_run=true (no follower motion)",
    )
    dry.add_argument(
        "--no-dry-run",
        dest="no_dry_run",
        action="store_true",
        help="Force robot.dry_run=false (follower executes commands)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.runtime.max_steps = int(args.max_steps)
    if args.dry_run_override:
        cfg.robot.dry_run = True
    elif args.no_dry_run:
        cfg.robot.dry_run = False
    rng = np.random.default_rng(args.seed)

    backbone = make_backbone_wrapper(
        cfg.backbone.policy_type,
        cfg.backbone.checkpoint,
        device=cfg.backbone.device,
        task_desc=cfg.backbone.task_desc,
    )

    risk_scorer = None
    intervention_policy = None
    if cfg.risk.enabled and cfg.risk.risk_model_ckpt:
        risk_scorer, _meta = load_supervised_risk_runtime(
            cfg.risk.risk_model_ckpt,
            cfg.backbone.device,
            feature_field=cfg.risk.risk_feature_field or None,
        )
        intervention_policy = ThresholdInterventionPolicy(
            cfg.risk.risk_threshold,
            cooldown_steps=cfg.risk.cooldown_steps,
            max_interventions_per_episode=cfg.risk.max_interventions_per_episode,
            boundary_only=cfg.risk.boundary_only_intervention,
        )

    if args.use_real_robot:
        if not lerobot_available():
            raise RuntimeError("LeRobot is not installed in this environment.")
        robot_handle = make_so101_follower_robot(cfg.robot)
        obs_adapter = LeRobotSO101ObservationAdapter(robot_handle)
        motor_names = list(robot_handle.bus.motors.keys())
        robot = LeRobotSO101FollowerRobot(
            robot_handle,
            motor_names,
            dry_run=cfg.robot.dry_run,
            control_hz=cfg.robot.control_hz,
            max_abs_action_value=cfg.safety.max_abs_action_value,
        )
    else:
        obs_adapter = DummyObservationAdapter()
        robot = DryRunRobot()

    runtime = HardwareRuntime(
        config=cfg,
        backbone=backbone,
        observation_adapter=obs_adapter,
        robot=robot,
        risk_scorer=risk_scorer,
        intervention_policy=intervention_policy,
        rng=rng,
    )
    print(json.dumps(runtime.run_episode(), indent=2))


if __name__ == "__main__":
    main()
