"""Probability calibration (pure numpy / scipy, no torch).

A model is *calibrated* when its probabilities can be taken literally: of all
mammograms it calls "70% malignant", about 70% should really be malignant.
Discrimination (AUC) and calibration are different things. A model can rank
cases well (high AUC) and still be over- or under-confident.

Temperature scaling (Guo et al., ICML 2017) is the simplest fix: divide every
logit by one number ``T`` fitted on held-out predictions by minimising the
negative log-likelihood. ``T > 1`` softens over-confident predictions, ``T < 1``
sharpens under-confident ones. Because it is a single monotone rescaling it
**never changes the ranking**, so AUC and argmax accuracy stay exactly the same;
only the probabilities move.

To keep this honest, ``T`` is never fitted on the images it is evaluated on:
``crossfit_temperature`` fits it on the other cross-validation folds.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_softmax, softmax

from .metrics import expected_calibration_error

_EPS = 1e-12


# ----------------------------------------------------------------------------------------------- scores
def binary_nll(y, logit) -> float:
    """Mean negative log-likelihood (log loss) of binary labels given logits."""
    y, z = np.asarray(y, float), np.asarray(logit, float)
    # log(1 + exp(-z)) for y=1, log(1 + exp(z)) for y=0, computed stably
    return float(np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))


def multiclass_nll(y, logits) -> float:
    y, z = np.asarray(y, int), np.asarray(logits, float)
    return float(-np.mean(log_softmax(z, axis=1)[np.arange(len(y)), y]))


def multiclass_brier(y, prob) -> float:
    """Multi-class Brier score: mean over images of sum_k (p_k - 1[y=k])^2. Ranges 0 (perfect) to 2."""
    y, p = np.asarray(y, int), np.asarray(prob, float)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(((p - onehot) ** 2).sum(1)))


def top_label_ece(y, prob, n_bins: int = 15) -> float:
    """ECE of the *predicted* class: is the model right 80% of the time when it is 80% confident?"""
    y, p = np.asarray(y, int), np.asarray(prob, float)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    return expected_calibration_error(correct, conf, n_bins)


def calibration_summary_binary(y, logit, T: float = 1.0, n_bins: int = 15) -> dict:
    z = np.asarray(logit, float) / T
    p = expit(z)
    y = np.asarray(y, float)
    return {"nll": binary_nll(y, z), "brier": float(np.mean((p - y) ** 2)), "ece": expected_calibration_error(y, p, n_bins),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def calibration_summary_multiclass(y, logits, T: float = 1.0, n_bins: int = 15) -> dict:
    z = np.asarray(logits, float) / T
    p = softmax(z, axis=1)
    return {"nll": multiclass_nll(y, z), "brier": multiclass_brier(y, p), "ece": top_label_ece(y, p, n_bins),
            "mean_confidence": float(p.max(1).mean()), "accuracy": float((p.argmax(1) == np.asarray(y)).mean())}


# ----------------------------------------------------------------------------------------------- fitting
def fit_temperature(y, logits, binary: bool | None = None, bounds=(0.05, 20.0)) -> float:
    """The T that minimises NLL of ``logits / T``. 1-D logits = binary, 2-D = multi-class."""
    z = np.asarray(logits, float)
    if binary is None:
        binary = z.ndim == 1
    nll = (lambda t: binary_nll(y, z / t)) if binary else (lambda t: multiclass_nll(y, z / t))
    # optimise log T so that "twice as hot" and "twice as cold" are symmetric for the optimiser
    res = minimize_scalar(lambda lt: nll(np.exp(lt)), bounds=np.log(bounds), method="bounded",
                          options={"xatol": 1e-5})
    return float(np.exp(res.x))


def fit_platt(y, logit) -> tuple[float, float]:
    """Platt scaling for a binary model: calibrated logit = a * logit + b, fitted by minimising NLL.

    Temperature scaling is the special case b = 0, a = 1/T. The extra intercept ``b`` can also fix a model
    that is *biased* (predicts "malignant" too often on average), which a temperature cannot."""
    from scipy.optimize import minimize
    y, z = np.asarray(y, float), np.asarray(logit, float)
    res = minimize(lambda ab: binary_nll(y, ab[0] * z + ab[1]), x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def crossfit_platt(y, logit, folds) -> tuple[np.ndarray, dict]:
    """Like ``crossfit_temperature`` but with Platt scaling (binary only)."""
    y, z, folds = np.asarray(y), np.asarray(logit, float), np.asarray(folds)
    out = np.empty_like(z)
    params = {}
    for k in np.unique(folds):
        te = folds == k
        a, b = fit_platt(y[~te], z[~te])
        params[int(k)] = (a, b)
        out[te] = a * z[te] + b
    return out, params


def crossfit_temperature(y, logits, folds) -> tuple[np.ndarray, dict]:
    """Calibrate out-of-fold predictions without ever using an image's own label.

    For every fold k, ``T_k`` is fitted on the predictions of all *other* folds and applied to fold k.
    Returns (calibrated logits, {fold: T_k})."""
    y, z, folds = np.asarray(y), np.asarray(logits, float), np.asarray(folds)
    out = np.empty_like(z)
    temps = {}
    for k in np.unique(folds):
        te = folds == k
        t = fit_temperature(y[~te], z[~te])
        temps[int(k)] = t
        out[te] = z[te] / t
    return out, temps


# ----------------------------------------------------------------------------------------------- curves
def reliability_curve(y, p, n_bins: int = 10) -> dict:
    """Per equal-width bin: mean predicted probability, observed frequency, count (empty bins dropped)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_p, freq, count = [], [], []
    for b in range(n_bins):
        m = which == b
        if m.any():
            mean_p.append(float(p[m].mean())); freq.append(float(y[m].mean())); count.append(int(m.sum()))
    return {"mean_predicted": mean_p, "observed": freq, "count": count, "edges": edges.tolist()}


# ----------------------------------------------------------------------------------------------- thresholds
def threshold_for_sensitivity(y, p, target: float) -> float:
    """Largest threshold whose sensitivity on (y, p) is still >= target."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1          # need at least k+1 positives at or above the threshold
    return float(pos[min(max(k, 0), len(pos) - 1)])


def crossfit_operating_point(y, p, folds, target_sensitivity: float = 0.90) -> dict:
    """Pick the threshold on the other folds (to reach the target sensitivity there), apply it to fold k.

    This is how an operating point would be chosen before deployment, and it shows how well the
    promised sensitivity holds on unseen patients."""
    y, p, folds = np.asarray(y, int), np.asarray(p, float), np.asarray(folds)
    pred = np.zeros(len(y), int)
    thresholds = {}
    for k in np.unique(folds):
        te = folds == k
        t = threshold_for_sensitivity(y[~te], p[~te], target_sensitivity)
        thresholds[int(k)] = t
        pred[te] = (p[te] >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"target_sensitivity": target_sensitivity,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "thresholds": thresholds, "pred": pred}
