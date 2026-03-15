"""Evaluation metrics for failure prediction.

AUROC, AUPRC, accuracy, precision, recall, F1, confusion matrix.
Handles class imbalance and edge cases.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return float(a / b)


def _auroc_numpy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Pure numpy AUROC (area under ROC curve). Fallback when sklearn not available."""
    order = np.argsort(-probs)  # descending by predicted prob
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc)


def _auprc_numpy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Pure numpy AUPRC (area under precision-recall curve)."""
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(labels_sorted)
    prec = tp / np.arange(1, len(labels) + 1, dtype=np.float64)
    rec = tp / n_pos
    # trapezoidal rule (integrate prec over rec)
    auc = np.trapz(prec, rec)
    return float(auc)


def compute_binary_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute binary classification metrics.

    Args:
        logits: Raw model outputs (N,).
        labels: Ground truth binary labels (N,) in {0, 1}.
        threshold: Classification threshold for accuracy/precision/recall/F1.

    Returns:
        Dict with auroc, auprc, accuracy, precision, recall, f1,
        tp, tn, fp, fn, n_positive, n_negative.
    """
    labels = np.asarray(labels, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()

    n = len(labels)
    n_pos = int((labels > 0.5).sum())
    n_neg = n - n_pos

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # sigmoid; clip avoids overflow

    result: dict[str, float | int] = {
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    if n == 0:
        result["auroc"] = 0.0
        result["auprc"] = 0.0
        result["accuracy"] = 0.0
        result["precision"] = 0.0
        result["recall"] = 0.0
        result["f1"] = 0.0
        result["tp"] = result["tn"] = result["fp"] = result["fn"] = 0
        return result

    if n_pos == 0 or n_neg == 0:
        result["auroc"] = 0.0
        result["auprc"] = 0.0
    else:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            result["auroc"] = float(roc_auc_score(labels, probs))
            result["auprc"] = float(average_precision_score(labels, probs))
        except ImportError:
            result["auroc"] = _auroc_numpy(probs, labels)
            result["auprc"] = _auprc_numpy(probs, labels)
        except Exception as e:
            logger.warning(f"AUROC/AUPRC failed: {e}")
            result["auroc"] = _auroc_numpy(probs, labels)
            result["auprc"] = _auprc_numpy(probs, labels)

    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels > 0.5)).sum())
    tn = int(((preds == 0) & (labels <= 0.5)).sum())
    fp = int(((preds == 1) & (labels <= 0.5)).sum())
    fn = int(((preds == 0) & (labels > 0.5)).sum())

    result["tp"] = tp
    result["tn"] = tn
    result["fp"] = fp
    result["fn"] = fn

    result["accuracy"] = _safe_divide(tp + tn, n, 0.0)
    result["precision"] = _safe_divide(tp, tp + fp, 0.0)
    result["recall"] = _safe_divide(tp, tp + fn, 0.0)
    p, r = result["precision"], result["recall"]
    result["f1"] = _safe_divide(2 * p * r, p + r, 0.0) if (p + r) > 0 else 0.0

    return result


def confusion_matrix_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, int]:
    """Return confusion matrix as dict."""
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
