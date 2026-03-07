"""Inspection helpers for failure-dataset raw rollouts and processed datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from failure_prediction.utils.failure_dataset_logger import FailureDatasetLogger

NUMERIC_KINDS = {"b", "i", "u", "f", "c"}
RAW_REQUIRED_META_KEYS = {
    "episode_id",
    "checkpoint_path",
    "task_name",
    "seed",
    "success",
    "episode_failed",
    "num_steps",
    "termination_reason",
    "return",
}
RAW_REQUIRED_ARRAY_KEYS = {"timestep", "executed_action", "reward", "done"}
PROCESSED_REQUIRED_KEYS = {"episode_id", "timestep", "episode_failed", "failure_within_k", "steps_to_failure"}


def _status(severity: str, message: str) -> dict[str, str]:
    return {"severity": severity, "message": message}


def _shape_list(arr: np.ndarray) -> list[int]:
    return list(arr.shape)


def _is_numeric(arr: np.ndarray) -> bool:
    return arr.dtype.kind in NUMERIC_KINDS


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _array_summary(arr: np.ndarray) -> dict[str, Any]:
    summary = {
        "shape": _shape_list(arr),
        "dtype": str(arr.dtype),
        "contains_nan": False,
        "contains_inf": False,
    }
    if _is_numeric(arr):
        finite_mask = np.isfinite(arr)
        summary["contains_nan"] = bool(np.isnan(arr).any())
        summary["contains_inf"] = bool(np.isinf(arr).any())
        if finite_mask.any():
            finite_vals = arr[finite_mask]
            summary["min"] = float(np.min(finite_vals))
            summary["max"] = float(np.max(finite_vals))
            summary["mean"] = float(np.mean(finite_vals))
            summary["std"] = float(np.std(finite_vals))
    return summary


def _embedding_stats(arr: np.ndarray) -> dict[str, Any]:
    stats = _array_summary(arr)
    if arr.ndim == 0:
        stats["all_rows_identical"] = True
        stats["avg_l2_norm"] = _safe_float(np.linalg.norm(arr))
        stats["avg_step_diff_l2"] = None
        return stats

    rows = arr.reshape(arr.shape[0], -1)
    if rows.shape[0] == 0:
        stats["all_rows_identical"] = True
        stats["avg_l2_norm"] = 0.0
        stats["avg_step_diff_l2"] = None
        return stats

    ref = rows[0]
    stats["all_rows_identical"] = bool(np.allclose(rows, ref[None, :], atol=1e-8, rtol=1e-6))
    stats["avg_l2_norm"] = float(np.mean(np.linalg.norm(rows, axis=1)))
    if rows.shape[0] > 1:
        diffs = rows[1:] - rows[:-1]
        stats["avg_step_diff_l2"] = float(np.mean(np.linalg.norm(diffs, axis=1)))
    else:
        stats["avg_step_diff_l2"] = None
    return stats


def _finalize_section(name: str, findings: list[dict[str, str]]) -> dict[str, Any]:
    severity_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    overall = "PASS"
    for finding in findings:
        if severity_order[finding["severity"]] > severity_order[overall]:
            overall = finding["severity"]
    return {"name": name, "status": overall, "findings": findings}


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def inspect_raw_episode(raw_episode_path: str | Path) -> dict[str, Any]:
    path = Path(raw_episode_path)
    loaded = FailureDatasetLogger.load_episode(path)
    meta = loaded["meta"]
    arrays = loaded["arrays"]

    findings: list[dict[str, str]] = []
    missing_meta = sorted(RAW_REQUIRED_META_KEYS - set(meta))
    if missing_meta:
        findings.append(_status("WARN", f"Missing episode metadata keys: {missing_meta}"))

    num_steps = int(meta.get("num_steps", -1))
    first_dims: dict[str, int] = {}
    array_summaries = {}
    invalid_numeric_keys = []
    image_warnings = []
    for key, arr in arrays.items():
        array_summaries[key] = _array_summary(arr)
        if arr.ndim > 0:
            first_dims[key] = int(arr.shape[0])
        if _is_numeric(arr) and (np.isnan(arr).any() or np.isinf(arr).any()):
            invalid_numeric_keys.append(key)
        if key.startswith("image_") and arr.ndim not in (4, 5):
            image_warnings.append(f"{key} has unexpected ndim={arr.ndim}, expected 4 or 5")

    if invalid_numeric_keys:
        findings.append(_status("WARN", f"Numeric arrays contain NaN/Inf: {sorted(invalid_numeric_keys)}"))
    if image_warnings:
        findings.extend(_status("WARN", msg) for msg in image_warnings)

    per_step_lengths = {key: dim for key, dim in first_dims.items() if key != "_meta_json"}
    unique_lengths = sorted(set(per_step_lengths.values()))
    if len(unique_lengths) > 1:
        findings.append(_status("FAIL", f"Inconsistent first-dimension lengths across arrays: {per_step_lengths}"))
    inferred_steps = unique_lengths[0] if unique_lengths else 0
    if num_steps >= 0 and inferred_steps and num_steps != inferred_steps:
        findings.append(
            _status("WARN", f"Metadata num_steps={num_steps} does not match inferred timestep count={inferred_steps}")
        )
    if num_steps < 0:
        num_steps = inferred_steps

    missing_arrays = sorted(key for key in RAW_REQUIRED_ARRAY_KEYS if key not in arrays)
    if missing_arrays:
        findings.append(_status("WARN", f"Missing common per-step arrays: {missing_arrays}"))

    if "success" not in meta and "episode_failed" not in meta:
        findings.append(_status("FAIL", "Episode metadata is missing both success and episode_failed"))

    embedding_fields = sorted(key for key in arrays if key.startswith("feat_"))
    action_fields = [key for key in ("executed_action", "predicted_action_chunk") if key in arrays]
    if not embedding_fields:
        findings.append(_status("WARN", "No embedding fields found in raw episode"))
    if "predicted_action_chunk" not in arrays:
        findings.append(_status("WARN", "No predicted_action_chunk field found in raw episode"))

    if "chunk_length" in arrays and "predicted_action_chunk" in arrays:
        chunk_length = arrays["chunk_length"]
        chunk_shape = arrays["predicted_action_chunk"].shape
        if len(chunk_shape) >= 2 and chunk_length.shape[0] == chunk_shape[0]:
            valid = chunk_length >= 0
            if np.any(valid):
                expected = chunk_shape[1]
                mismatched = np.where(valid & (chunk_length != expected))[0]
                if mismatched.size > 0:
                    findings.append(
                        _status(
                            "WARN",
                            f"chunk_length disagrees with predicted_action_chunk.shape[1]={expected} at "
                            f"{mismatched.size} timesteps",
                        )
                    )

    embedding_summaries = {key: _embedding_stats(arrays[key]) for key in embedding_fields}
    for key, stats in embedding_summaries.items():
        if stats["contains_nan"] or stats["contains_inf"]:
            findings.append(_status("WARN", f"Embedding field {key} contains NaN/Inf"))
        if stats.get("all_rows_identical"):
            findings.append(_status("WARN", f"Embedding field {key} is constant across timesteps"))
        avg_diff = stats.get("avg_step_diff_l2")
        if avg_diff is not None and math.isclose(avg_diff, 0.0, abs_tol=1e-12):
            findings.append(_status("WARN", f"Embedding field {key} has zero step-to-step difference"))

    if not findings:
        findings.append(_status("PASS", "Raw episode schema and values look consistent"))

    return {
        "path": str(path),
        "meta": meta,
        "num_timesteps": num_steps,
        "array_shapes": {key: _shape_list(arr) for key, arr in arrays.items()},
        "array_summaries": array_summaries,
        "embedding_fields": embedding_fields,
        "embedding_summaries": embedding_summaries,
        "action_fields": action_fields,
        "section": _finalize_section("raw_schema", findings),
    }


def _group_episode_indices(episode_ids: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for idx, ep_id in enumerate(episode_ids.tolist()):
        groups.setdefault(int(ep_id), []).append(idx)
    return {ep_id: np.array(indices, dtype=np.int64) for ep_id, indices in groups.items()}


def _episode_label_check(
    dataset: dict[str, np.ndarray],
    ep_id: int,
    indices: np.ndarray,
    failure_horizon: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    findings: list[dict[str, str]] = []
    timesteps = dataset["timestep"][indices]
    failure_labels = dataset["failure_within_k"][indices]
    steps_to_failure = dataset["steps_to_failure"][indices]
    episode_failed = bool(dataset["episode_failed"][indices][0])
    terminal_timestep = int(np.max(timesteps))
    expected_steps = terminal_timestep - timesteps

    if episode_failed:
        expected_failure = (expected_steps <= failure_horizon).astype(np.int32)
        if not np.array_equal(failure_labels.astype(np.int32), expected_failure):
            findings.append(
                _status(
                    "FAIL",
                    f"Episode {ep_id} failure_within_k does not match expected failure window at horizon {failure_horizon}",
                )
            )
        if not np.array_equal(steps_to_failure.astype(np.int32), expected_steps.astype(np.int32)):
            findings.append(_status("FAIL", f"Episode {ep_id} steps_to_failure is inconsistent with terminal timestep"))
    else:
        if np.any(failure_labels):
            findings.append(_status("FAIL", f"Successful episode {ep_id} has positive failure_within_k labels"))
        if not np.all(steps_to_failure == -1):
            findings.append(_status("FAIL", f"Successful episode {ep_id} has non-sentinel steps_to_failure values"))

    if not findings:
        findings.append(_status("PASS", f"Episode {ep_id} label consistency check passed"))

    return findings, {
        "episode_id": ep_id,
        "episode_failed": episode_failed,
        "terminal_timestep": terminal_timestep,
        "num_steps": int(indices.shape[0]),
        "num_positive_failure_within_k": int(np.sum(failure_labels)),
    }


def inspect_processed_dataset(
    processed_path: str | Path,
    failure_horizon: int,
    max_failed_examples_to_check: int = 2,
    max_success_examples_to_check: int = 1,
) -> dict[str, Any]:
    path = Path(processed_path)
    metadata = None
    files = []
    if path.is_dir():
        files = sorted(p.name for p in path.iterdir())
        dataset_path = path / "timestep_dataset.npz"
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
    else:
        dataset_path = path
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        files = [path.name]

    dataset = _load_npz_dict(dataset_path)
    findings: list[dict[str, str]] = []

    missing_keys = sorted(PROCESSED_REQUIRED_KEYS - set(dataset))
    if missing_keys:
        findings.append(_status("FAIL", f"Missing required processed dataset keys: {missing_keys}"))

    num_rows = int(dataset["episode_id"].shape[0]) if "episode_id" in dataset and dataset["episode_id"].ndim > 0 else 0
    array_shapes = {key: _shape_list(arr) for key, arr in dataset.items()}

    inconsistent_keys = []
    invalid_numeric_keys = []
    for key, arr in dataset.items():
        if arr.ndim > 0 and arr.shape[0] != num_rows:
            inconsistent_keys.append(key)
        if _is_numeric(arr) and (np.isnan(arr).any() or np.isinf(arr).any()):
            invalid_numeric_keys.append(key)
    if inconsistent_keys:
        findings.append(_status("FAIL", f"Processed arrays with inconsistent row count: {sorted(inconsistent_keys)}"))
    if invalid_numeric_keys:
        findings.append(_status("WARN", f"Processed numeric arrays contain NaN/Inf: {sorted(invalid_numeric_keys)}"))

    if missing_keys:
        if not findings:
            findings.append(_status("FAIL", "Processed dataset is missing required keys"))
        return {
            "path": str(path),
            "dataset_path": str(dataset_path),
            "available_files": files,
            "metadata": metadata,
            "num_rows": num_rows,
            "unique_episode_count": 0,
            "successful_episode_count": 0,
            "failed_episode_count": 0,
            "failure_within_k_positive": 0,
            "failure_within_k_negative": 0,
            "class_balance": 0.0,
            "array_shapes": array_shapes,
            "embedding_fields": sorted(key for key in dataset if key.startswith("feat_")),
            "embedding_summaries": {},
            "action_fields": [key for key in ("predicted_action_chunk", "executed_action") if key in dataset],
            "checked_episode_examples": [],
            "schema_section": _finalize_section("processed_schema", findings),
            "label_section": _finalize_section(
                "label_consistency",
                [_status("WARN", "Label consistency checks skipped because required fields are missing")],
            ),
        }

    episode_groups = _group_episode_indices(dataset["episode_id"])
    unique_episodes = sorted(episode_groups)

    failed_episode_ids = []
    successful_episode_ids = []
    for ep_id in unique_episodes:
        indices = episode_groups[ep_id]
        failed = bool(dataset["episode_failed"][indices][0])
        if failed:
            failed_episode_ids.append(ep_id)
        else:
            successful_episode_ids.append(ep_id)

    n_positive = int(np.sum(dataset["failure_within_k"])) if "failure_within_k" in dataset else 0
    n_negative = num_rows - n_positive
    class_balance = n_positive / num_rows if num_rows else 0.0

    embedding_fields = sorted(key for key in dataset if key.startswith("feat_"))
    action_chunk_fields = [key for key in ("predicted_action_chunk", "executed_action") if key in dataset]
    embedding_summaries = {key: _embedding_stats(dataset[key]) for key in embedding_fields}
    for key, stats in embedding_summaries.items():
        if stats["contains_nan"] or stats["contains_inf"]:
            findings.append(_status("WARN", f"Processed embedding field {key} contains NaN/Inf"))
        if stats.get("all_rows_identical"):
            findings.append(_status("WARN", f"Processed embedding field {key} is constant across rows"))

    label_check_findings: list[dict[str, str]] = []
    label_check_examples = []
    for ep_id in failed_episode_ids[:max_failed_examples_to_check]:
        ep_findings, ep_summary = _episode_label_check(dataset, ep_id, episode_groups[ep_id], failure_horizon)
        label_check_findings.extend(ep_findings)
        label_check_examples.append(ep_summary)
    for ep_id in successful_episode_ids[:max_success_examples_to_check]:
        ep_findings, ep_summary = _episode_label_check(dataset, ep_id, episode_groups[ep_id], failure_horizon)
        label_check_findings.extend(ep_findings)
        label_check_examples.append(ep_summary)

    if not failed_episode_ids:
        label_check_findings.append(_status("WARN", "No failed episodes available for positive-window label checks"))
    if not successful_episode_ids:
        label_check_findings.append(_status("WARN", "No successful episodes available for negative label checks"))

    if not findings:
        findings.append(_status("PASS", "Processed dataset schema and row counts look consistent"))
    if not label_check_findings:
        label_check_findings.append(_status("PASS", "Label consistency checks passed"))

    return {
        "path": str(path),
        "dataset_path": str(dataset_path),
        "available_files": files,
        "metadata": metadata,
        "num_rows": num_rows,
        "unique_episode_count": len(unique_episodes),
        "successful_episode_count": len(successful_episode_ids),
        "failed_episode_count": len(failed_episode_ids),
        "failure_within_k_positive": n_positive,
        "failure_within_k_negative": n_negative,
        "class_balance": class_balance,
        "array_shapes": array_shapes,
        "embedding_fields": embedding_fields,
        "embedding_summaries": embedding_summaries,
        "action_fields": action_chunk_fields,
        "checked_episode_examples": label_check_examples,
        "schema_section": _finalize_section("processed_schema", findings),
        "label_section": _finalize_section("label_consistency", label_check_findings),
    }


def combined_status(*sections: dict[str, Any]) -> str:
    severity_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    result = "PASS"
    for section in sections:
        status = section["status"]
        if severity_order[status] > severity_order[result]:
            result = status
    return result
