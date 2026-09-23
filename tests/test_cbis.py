import json

import numpy as np
import pandas as pd
import pytest

from mammo.cbis import index_cbis, locate_cbis
from mammo.metrics import multiclass_metrics, paired_bootstrap, quadratic_kappa
from mammo.preprocess import breast_bbox, cached_images, load_mammogram
from mammo.splits import make_folds


def _index(fake_cbis):
    loc = locate_cbis(fake_cbis)
    return index_cbis(loc["csv"], loc["jpeg"])


def test_index_merges_duplicates_and_labels(fake_cbis):
    df, stats = _index(fake_cbis)
    assert stats["abnormalities_without_jpeg"] == 0
    assert len(df) == 26 and df["image_id"].is_unique            # 12 patients x 2 images + 2 extra views
    assert df["patient"].nunique() == 12
    row = df.set_index("image_id").loc["P_00002_LEFT_CC"]
    assert row["pathology"] == 1                                  # benign mass + malignant calc -> malignant
    assert row["n_mass"] == 1 and row["n_calc"] == 1              # the two stored copies became one image
    assert row["density"] == 2                                    # BI-RADS density 3 -> class index 2
    assert stats["images_stored_twice_mass_and_calc"] == 1
    assert df.set_index("image_id").loc["P_00003_RIGHT_CC", "density"] == -1   # density 0 is invalid
    assert set(df["density"]) <= {-1, 0, 1, 2, 3}
    # BENIGN_WITHOUT_CALLBACK counts as benign
    assert df.set_index("image_id").loc["P_00004_LEFT_CC", "pathology"] == 0
    # found through the DICOM PatientID fallback
    assert df.set_index("image_id").loc["P_00004_RIGHT_CC", "path"].endswith("777777/1-1.jpg")
    # the lesion crops are never used as whole images
    assert not df["path"].str.contains("cropped").any()
    for p in df["path"]:
        assert pd.notna(p)


def test_official_split_is_patient_level(fake_cbis):
    df, stats = _index(fake_cbis)
    assert stats["patients_in_official_train_and_test"] == 1      # P_00003
    assert df.loc[df["patient"] == "P_00003", "official_split"].eq("train").all()
    test_pat = set(df.loc[df["official_split"] == "test", "patient"])
    train_pat = set(df.loc[df["official_split"] == "train", "patient"])
    assert test_pat == {"P_00011", "P_00012"} and not test_pat & train_pat


def test_patient_folds_never_share_a_patient(fake_cbis):
    df, _ = _index(fake_cbis)
    folds = make_folds(df, "patient", n_splits=3, seed=0)
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(df)))
    for tr, te in folds:
        assert not set(df.iloc[tr]["patient"]) & set(df.iloc[te]["patient"])


def test_preprocessing_crops_and_orients(fake_cbis):
    df, _ = _index(fake_cbis)
    right = df[df["side"] == "RIGHT"]["path"].iloc[0]
    a = load_mammogram(right, 160, 96)
    assert a.shape == (160, 96) and a.dtype == np.uint8
    assert a[:, :48].mean() > a[:, 48:].mean()                    # breast now on the left
    assert a[:3].mean() < 200                                     # scanner strip cropped away
    left = load_mammogram(df[df["side"] == "LEFT"]["path"].iloc[0], 160, 96)
    assert left[:, :48].mean() > left[:, 48:].mean()


def test_breast_bbox_ignores_border_strip():
    from conftest import _fake_mammogram
    a = _fake_mammogram(False)
    t, b, l, r = breast_bbox(a)
    assert t > 6 and l == 0 and b > 400


def test_image_cache_is_reused(fake_cbis, tmp_path):
    df, _ = _index(fake_cbis)
    msgs = []
    x1 = cached_images(df["path"][:5], 64, 40, tmp_path / "cache", workers=2, log=msgs.append)
    x2 = cached_images(df["path"][:5], 64, 40, tmp_path / "cache", workers=2, log=msgs.append)
    assert x1.shape == (5, 64, 40) and np.array_equal(x1, x2)
    assert any("using cached" in m for m in msgs)


def test_quadratic_kappa_and_density_metrics():
    y = np.array([0, 1, 2, 3, 1, 2])
    assert quadratic_kappa(y, y) == pytest.approx(1.0)
    off_by_one = quadratic_kappa(y, np.clip(y + 1, 0, 3))
    off_by_two = quadratic_kappa(y, np.clip(y + 2, 0, 3))
    assert 1 > off_by_one > off_by_two
    m = multiclass_metrics(np.array([0, 1, -1]), np.eye(4)[[0, 1, 2]])
    assert m["n"] == 2 and m["qwk"] == pytest.approx(1.0)


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(n // 2), 2)
    good = np.clip(y * 0.5 + rng.random(n) * 0.6, 0, 1)
    noise = rng.random(n)
    res = paired_bootstrap(y, good, noise, groups, n=300)
    assert res["diff"] > 0.2 and res["ci95"][0] > 0 and res["p_not_better"] < 0.01
    same = paired_bootstrap(y, good, good, groups, n=100)
    assert same["diff"] == 0


def test_ablation_plot_and_table(tmp_path):
    from mammo.experiments.cbis import render_markdown
    from mammo.plots import plot_ablation
    mal = {"auc_pooled": 0.7, "auc_ci95": [0.65, 0.75], "accuracy": 0.66, "majority_baseline_accuracy": 0.55}
    den = {"accuracy": 0.6, "accuracy_ci95": [0.55, 0.65], "qwk": 0.5, "qwk_ci95": [0.4, 0.6],
           "majority_baseline_accuracy": 0.42}
    table = {"mt_cbam": {"label": "A", "malignancy": mal, "density": den},
             "mt_plain": {"label": "B", "malignancy": mal, "density": den},
             "st_path": {"label": "C", "malignancy": mal}, "st_dens": {"label": "D", "density": den}}
    comps = [{"question": "q?", "a": "mt_cbam", "b": "st_path", "metric": "AUC", "diff": 0.01,
              "ci95": [-0.02, 0.04], "p_not_better": 0.3, "n_boot": 100}]
    plot_ablation(table, comps, tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 10_000
    md = render_markdown({"results": table, "comparisons": comps,
                          "official_split": {"mt_cbam": {"n_test": 10, "test_patients": 5, "path_auc": 0.7}}})
    assert "0.700 (0.650–0.750)" in md and "q? (AUC)" in md and "Official" in md


def test_end_to_end_smoke(fake_cbis, tmp_path):
    """prepare -> run all four variants (+ official split) -> summarise, on CPU with an untrained backbone."""
    pytest.importorskip("torch")
    from mammo.experiments.cbis import main
    common = ["--data-root", str(fake_cbis), "--out", str(tmp_path / "res"), "--cache", str(tmp_path / "cache"),
              "--checkpoints", str(tmp_path / "ckpt"), "--smoke", "--no-pretrained", "--workers", "2"]
    assert main(["prepare", *common]) == 0
    assert main(["run", "mt_cbam", "mt_plain", "st_path", "st_dens", "--official", *common]) == 0
    assert main(["summarise", *common]) == 0
    out = tmp_path / "res_smoke"
    s = json.load(open(out / "summary.json"))
    assert set(s["results"]) == {"mt_cbam", "mt_plain", "st_path", "st_dens"}
    assert "malignancy" not in s["results"]["st_dens"] and "density" not in s["results"]["st_path"]
    assert len(s["comparisons"]) == 4
    oof = pd.read_csv(out / "mt_cbam" / "oof_predictions.csv")
    assert len(oof) == 26 and oof["image_id"].is_unique           # every image tested exactly once
    assert (out / "ablation_chart.png").exists() and (out / "preprocessing_preview.png").exists()
    assert (tmp_path / "ckpt" / "cbis_mt_cbam_official.pt").exists()
