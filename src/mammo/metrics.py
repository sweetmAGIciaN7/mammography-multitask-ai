"""Evaluation metrics (pure numpy / scikit-learn, no torch)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


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
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "qwk": float("nan"), "n": 0}
    pred = prob.argmax(1)
    labels = list(range(prob.shape[1]))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "qwk": quadratic_kappa(y, pred, labels),
        "n": int(len(y)),
    }


def quadratic_kappa(y: np.ndarray, pred: np.ndarray, labels=None) -> float:
    """Quadratic-weighted Cohen's kappa: agreement for ordered classes (density A<B<C<D).
    0 = chance level, 1 = perfect; being off by two grades costs 4x as much as being off by one."""
    y, pred = np.asarray(y), np.asarray(pred)
    if len(np.unique(np.concatenate([y, pred]))) < 2:
        return float("nan")
    return float(cohen_kappa_score(y, pred, labels=labels, weights="quadratic"))


def auc_fn(y, p) -> float:
    return float(roc_auc_score(y, p))


def density_acc_fn(y, prob) -> float:
    return float((np.asarray(prob).argmax(1) == np.asarray(y)).mean())


def density_qwk_fn(y, prob) -> float:
    return quadratic_kappa(y, np.asarray(prob).argmax(1), list(range(np.asarray(prob).shape[1])))


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, fn=auc_fn,
                     n: int = 2000, seed: int = 0) -> dict:
    """Is model A better than model B on the *same* test images?

    Resamples whole patients (both models see the same resample each time) and reports
    diff = fn(A) - fn(B), its 95% CI, and the share of resamples where A is not better
    (a one-sided bootstrap p-value)."""
    rng = np.random.default_rng(seed)
    y, a, b, g = np.asarray(y), np.asarray(a), np.asarray(b), np.asarray(groups)
    _, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    diffs = []
    for _ in range(n):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if y.ndim == 1 and len(np.unique(y[pick])) < 2:
            continue
        diffs.append(fn(y[pick], a[pick]) - fn(y[pick], b[pick]))
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": float(fn(y, a) - fn(y, b)), "ci95": [float(lo), float(hi)],
            "p_not_better": float(np.mean(diffs <= 0)), "n_boot": int(len(diffs))}


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
