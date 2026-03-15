#!/usr/bin/env python
"""Inspect and validate raw/processed failure-dataset outputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_DIR))

from failure_prediction.utils.failure_dataset_checks import (
    combined_status,
    inspect_processed_dataset,
    inspect_raw_episode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect failure-dataset raw and processed outputs")
    parser.add_argument("--raw_episode", type=str, default=None, help="Path to a raw episode_XXXXXX.npz file")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory containing raw episode_*.npz files")
    parser.add_argument(
        "--sample_episode",
        type=str,
        default="first",
        choices=["first", "last", "random"],
        help="How to choose an episode when --raw_dir is used",
    )
    parser.add_argument("--processed_dir", type=str, default=None, help="Processed dataset directory")
    parser.add_argument("--processed_file", type=str, default=None, help="Path to timestep_dataset.npz")
    parser.add_argument("--failure_horizon", type=int, default=10, help="K for failure_within_k validation")
    parser.add_argument("--max_failed_examples_to_check", type=int, default=2)
    parser.add_argument("--max_success_examples_to_check", type=int, default=1)
    parser.add_argument("--json_report", type=str, default=None, help="Optional path to save a JSON report")
    return parser.parse_args()


def choose_raw_episode(raw_dir: Path, mode: str) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw directory does not exist: {raw_dir}. "
            "Run collection first or point --raw_dir at the dataset's actual raw/ directory."
        )
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"--raw_dir is not a directory: {raw_dir}")
    episodes = sorted(raw_dir.glob("episode_*.npz"))
    if not episodes:
        raise FileNotFoundError(
            f"No episode_*.npz files found in {raw_dir}. "
            "Check that collection completed and that you are pointing at the raw/ subdirectory."
        )
    if mode == "first":
        return episodes[0]
    if mode == "last":
        return episodes[-1]
    return random.choice(episodes)


def print_section_header(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def print_findings(findings: list[dict[str, str]]):
    for finding in findings:
        print(f"[{finding['severity']}] {finding['message']}")


def print_raw_report(report: dict):
    print_section_header("Raw Episode Summary")
    meta = report["meta"]
    print(f"path: {report['path']}")
    print(f"num_timesteps: {report['num_timesteps']}")
    print(f"episode_id: {meta.get('episode_id')}")
    print(f"success: {meta.get('success')}")
    print(f"episode_failed: {meta.get('episode_failed')}")
    print(f"terminal_step: {meta.get('terminal_step')}")
    print(f"termination_reason: {meta.get('termination_reason')}")
    print(f"return: {meta.get('return')}")
    print(f"embedding_fields: {report['embedding_fields']}")
    print(f"action_fields: {report['action_fields']}")
    print("array_shapes:")
    for key in sorted(report["array_shapes"]):
        print(f"  {key}: {report['array_shapes'][key]}")
    if report["embedding_summaries"]:
        print("embedding_stats:")
        for key in report["embedding_fields"]:
            stats = report["embedding_summaries"][key]
            print(
                f"  {key}: shape={stats['shape']} dtype={stats['dtype']} "
                f"min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')} std={stats.get('std')} "
                f"avg_l2_norm={stats.get('avg_l2_norm')} avg_step_diff_l2={stats.get('avg_step_diff_l2')} "
                f"all_rows_identical={stats.get('all_rows_identical')}"
            )
    print(f"status: {report['section']['status']}")
    print_findings(report["section"]["findings"])


def print_processed_report(report: dict):
    print_section_header("Processed Dataset Summary")
    print(f"path: {report['path']}")
    print(f"dataset_path: {report['dataset_path']}")
    print(f"available_files: {report['available_files']}")
    print(f"num_rows: {report['num_rows']}")
    print(f"unique_episode_count: {report['unique_episode_count']}")
    print(f"successful_episode_count: {report['successful_episode_count']}")
    print(f"failed_episode_count: {report['failed_episode_count']}")
    print(f"failure_within_k_positive: {report['failure_within_k_positive']}")
    print(f"failure_within_k_negative: {report['failure_within_k_negative']}")
    print(f"class_balance: {report['class_balance']:.6f}")
    print(f"embedding_fields: {report['embedding_fields']}")
    print(f"action_fields: {report['action_fields']}")
    print("array_shapes:")
    for key in sorted(report["array_shapes"]):
        print(f"  {key}: {report['array_shapes'][key]}")
    if report["embedding_summaries"]:
        print("embedding_stats:")
        for key in report["embedding_fields"]:
            stats = report["embedding_summaries"][key]
            print(
                f"  {key}: shape={stats['shape']} dtype={stats['dtype']} "
                f"min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')} std={stats.get('std')} "
                f"avg_l2_norm={stats.get('avg_l2_norm')} avg_step_diff_l2={stats.get('avg_step_diff_l2')} "
                f"all_rows_identical={stats.get('all_rows_identical')}"
            )
    if report["checked_episode_examples"]:
        print("checked_episode_examples:")
        for example in report["checked_episode_examples"]:
            print(f"  {example}")
    print(f"schema_status: {report['schema_section']['status']}")
    print_findings(report["schema_section"]["findings"])
    print(f"label_status: {report['label_section']['status']}")
    print_findings(report["label_section"]["findings"])


def main():
    args = parse_args()
    raw_report = None
    processed_report = None

    if args.raw_episode:
        raw_report = inspect_raw_episode(args.raw_episode)
    elif args.raw_dir:
        raw_report = inspect_raw_episode(choose_raw_episode(Path(args.raw_dir), args.sample_episode))

    if args.processed_dir:
        if not Path(args.processed_dir).exists():
            raise FileNotFoundError(
                f"Processed directory does not exist: {args.processed_dir}. "
                "Run postprocessing first or pass the correct processed dataset path."
            )
        processed_report = inspect_processed_dataset(
            args.processed_dir,
            failure_horizon=args.failure_horizon,
            max_failed_examples_to_check=args.max_failed_examples_to_check,
            max_success_examples_to_check=args.max_success_examples_to_check,
        )
    elif args.processed_file:
        if not Path(args.processed_file).exists():
            raise FileNotFoundError(
                f"Processed dataset file does not exist: {args.processed_file}. "
                "Run postprocessing first or pass the correct timestep_dataset.npz path."
            )
        processed_report = inspect_processed_dataset(
            args.processed_file,
            failure_horizon=args.failure_horizon,
            max_failed_examples_to_check=args.max_failed_examples_to_check,
            max_success_examples_to_check=args.max_success_examples_to_check,
        )

    if raw_report is None and processed_report is None:
        raise SystemExit("Provide at least one of --raw_episode/--raw_dir or --processed_dir/--processed_file")

    if raw_report is not None:
        print_raw_report(raw_report)
    if processed_report is not None:
        print_processed_report(processed_report)

    sections = []
    if raw_report is not None:
        sections.append(raw_report["section"])
    if processed_report is not None:
        sections.extend([processed_report["schema_section"], processed_report["label_section"]])

    overall = combined_status(*sections)
    print_section_header("Validation Result")
    print(f"overall_status: {overall}")
    for section in sections:
        print(f"{section['name']}: {section['status']}")

    if args.json_report:
        report = {
            "raw_report": raw_report,
            "processed_report": processed_report,
            "overall_status": overall,
        }
        Path(args.json_report).write_text(json.dumps(report, indent=2))
        print(f"json_report: {args.json_report}")


if __name__ == "__main__":
    main()
