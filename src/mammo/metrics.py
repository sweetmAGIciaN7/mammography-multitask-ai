"""Evaluation metrics (pure numpy / scikit-learn, no torch)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE with equal-width bins on P(y=1) (reliability of the positive-class probability)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "accuracy": float((pred == y).mean()),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
    }
    out["balanced_accuracy"] = float(np.nanmean([out["sensitivity"], out["specificity"]]))
    return out


def multiclass_metrics(y: np.ndarray, prob: np.ndarray, ignore: int = -1) -> dict:
    y, prob = np.asarray(y), np.asarray(prob, float)
    keep = y != ignore
    y, prob = y[keep], prob[keep]
    if len(y) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}
    pred = prob.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(prob.shape[1])), zero_division=0)),
        "n": int(len(y)),
    }


def per_original(group: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average predictions over all augmented copies of the same original image."""
    group = np.asarray(group)
    uniq, inv = np.unique(group, return_inverse=True)
    p_mean = np.bincount(inv, weights=np.asarray(p, float)) / np.bincount(inv)
    y_first = np.zeros(len(uniq), dtype=int)
    y_first[inv] = np.asarray(y).astype(int)
    return y_first, p_mean


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn=roc_auc_score, n: int = 1000, seed: int = 0,
                 groups: np.ndarray | None = None) -> tuple[float, float]:
    """95% percentile CI. With ``groups``, resample whole groups (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    if groups is None:
        units = [np.array([i]) for i in range(len(y))]
    else:
        g = np.asarray(groups)
        units = [np.flatnonzero(g == u) for u in np.unique(g)]
    stats = []
    for _ in range(n):
        pick = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        if len(np.unique(y[pick])) < 2:
            continue
        stats.append(fn(y[pick], p[pick]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
