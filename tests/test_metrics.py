import numpy as np
import pandas as pd

from mammo.metrics import binary_metrics, bootstrap_ci, expected_calibration_error, multiclass_metrics, per_original


def test_binary_metrics_perfect_and_counts():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_metrics(y, p)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_brier_is_at_least_ece_squared():
    """The inequality the paper's Table 3 violates (Brier 0.010 with ECE 0.372)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.random(300); y = (rng.random(300) < rng.random()).astype(int)
        # with a single bin ECE is exactly |mean(p)-mean(y)| and Brier >= ECE^2 must hold
        ece1 = expected_calibration_error(y, p, n_bins=1)
        assert np.mean((p - y) ** 2) >= ece1 ** 2 - 1e-12
        ece = expected_calibration_error(y, p)
        assert np.mean((p - y) ** 2) >= ece ** 2 - 1e-12


def test_multiclass_ignores_missing():
    y = np.array([0, 1, -1, 2]); prob = np.eye(4)[[0, 1, 3, 2]]
    m = multiclass_metrics(y, prob)
    assert m["n"] == 3 and m["accuracy"] == 1.0


def test_per_original_averages_copies():
    g = np.array(["a", "a", "b", "b"]); y = np.array([1, 1, 0, 0]); p = np.array([0.8, 0.6, 0.2, 0.4])
    yo, po = per_original(g, y, p)
    assert list(yo) == [1, 0] and np.allclose(po, [0.7, 0.3])


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo, hi = bootstrap_ci(y, p, n=200)
    auc = binary_metrics(y, p)["auc"]
    assert lo < auc < hi


def test_leakage_plot_renders(tmp_path):
    from mammo.plots import plot_leakage
    rows = [{"protocol": pr, "fold": k, "file_auc": a + 0.01 * k, "file_accuracy": a - 0.05}
            for pr, a in [("random", 0.95), ("image", 0.7), ("patient", 0.65)] for k in range(5)]
    plot_leakage(pd.DataFrame(rows), tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 10_000
