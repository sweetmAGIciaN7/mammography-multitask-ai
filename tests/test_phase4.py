"""Phase 4: calibration maths, INbreast indexing / DICOM windowing, and both experiment summaries (no torch)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mammo.calibration import (binary_nll, calibration_summary_binary, crossfit_operating_point, crossfit_platt,
                               crossfit_temperature, fit_platt, fit_temperature, multiclass_brier, reliability_curve,
                               threshold_for_sensitivity, top_label_ece)


def _overconfident(n=4000, scale=2.5, bias=0.0, seed=0):
    """True logit ~ N(0, 1.5); labels drawn from it; the 'model' reports scale * true logit + bias."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, 1.5, n)
    y = (rng.random(n) < expit(true)).astype(int)
    return y, scale * true + bias


def test_temperature_recovers_overconfidence():
    y, z = _overconfident(scale=2.5)
    T = fit_temperature(y, z)
    assert 2.2 < T < 2.8
    before, after = calibration_summary_binary(y, z), calibration_summary_binary(y, z, T)
    assert after["nll"] < before["nll"] and after["ece"] < before["ece"]


def test_temperature_never_changes_ranking():
    y, z = _overconfident()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, z) == pytest.approx(roc_auc_score(y, z / fit_temperature(y, z)))


def test_platt_fixes_bias_that_temperature_cannot():
    y, z = _overconfident(scale=1.0, bias=1.0)
    t = calibration_summary_binary(y, z, fit_temperature(y, z))
    a, b = fit_platt(y, z)
    p = calibration_summary_binary(y, a * z + b)
    assert b == pytest.approx(-1.0, abs=0.2)
    assert p["ece"] < t["ece"] and abs(p["mean_predicted"] - p["observed_rate"]) < 0.01


def test_multiclass_temperature():
    rng = np.random.default_rng(1)
    true = rng.normal(0, 1.2, (3000, 4))
    y = np.array([rng.choice(4, p=np.exp(r) / np.exp(r).sum()) for r in true])
    assert 0.4 < fit_temperature(y, 0.5 * true) < 0.6                  # under-confident -> T < 1
    assert multiclass_brier(y, np.eye(4)[y]) == 0
    assert 0 <= top_label_ece(y, np.full((len(y), 4), 0.25)) <= 1


def test_crossfit_never_uses_the_folds_own_labels():
    y, z = _overconfident(n=1000)
    folds = np.repeat(np.arange(1, 6), 200)
    _, t1 = crossfit_temperature(y, z, folds)
    y2 = y.copy(); y2[folds == 3] = 1 - y2[folds == 3]                  # scramble fold 3's labels
    _, t2 = crossfit_temperature(y2, z, folds)
    assert t1[3] == t2[3] and t1[1] != t2[1]
    _, p1 = crossfit_platt(y, z, folds)
    _, p2 = crossfit_platt(y2, z, folds)
    assert p1[3] == p2[3]


def test_operating_point_threshold():
    y, z = _overconfident(n=2000)
    p = expit(z)
    t = threshold_for_sensitivity(y, p, 0.9)
    assert ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= 0.9
    op = crossfit_operating_point(y, p, np.arange(len(y)) % 5, 0.9)
    assert 0.85 < op["sensitivity"] < 0.95 and op["tp"] + op["fn"] == y.sum()


def test_reliability_curve_and_nll():
    curve = reliability_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.15], n_bins=10)
    assert sum(curve["count"]) == 4 and curve["observed"][0] == 0
    assert binary_nll([1], [0.0]) == pytest.approx(np.log(2))


# ----------------------------------------------------------------------------------------------- CBIS calibration
def _fake_phase3(root, seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    y = rng.integers(0, 2, n)
    dens = rng.integers(0, 4, n)
    dens[:10] = -1
    base = pd.DataFrame({"fold": np.arange(n) % 5 + 1, "row": np.arange(n), "image_id": [f"I{i}" for i in range(n)],
                         "patient": [f"P{i // 2}" for i in range(n)], "y": y, "density": dens})
    for name, tasks in {"mt_cbam": "pd", "st_path": "p", "st_dens": "d"}.items():
        d = base.copy(); d.insert(0, "config", name)
        if "p" in tasks:
            d["logit_malignant"] = 3 * (y - 0.5) + rng.normal(0, 2, n) + 0.5
            d["p_malignant"] = expit(d["logit_malignant"])
        if "d" in tasks:
            lg = rng.normal(0, 1, (n, 4)); lg[np.arange(n), np.clip(dens, 0, 3)] += 1.5
            for k in range(4):
                d[f"logit_density_{k}"] = lg[:, k]
                d[f"p_density_{k}"] = np.exp(lg[:, k]) / np.exp(lg).sum(1)
        (root / name).mkdir(parents=True)
        d.to_csv(root / name / "oof_predictions.csv", index=False)
        if name == "mt_cbam":
            d.assign(fold=0).head(200).to_csv(root / name / "official_predictions.csv", index=False)
    return root


def test_calibration_experiment_end_to_end(tmp_path):
    from mammo.experiments import calibration as cal
    cbis = _fake_phase3(tmp_path / "cbis")
    assert cal.main(["--cbis", str(cbis), "--out", str(tmp_path / "out"), "--nboot", "50"]) == 0
    s = json.load(open(tmp_path / "out" / "summary.json"))
    m = s["variants"]["mt_cbam"]["malignancy"]
    assert m["platt"]["nll"] <= m["uncalibrated"]["nll"] + 1e-9
    assert set(s["variants"]) == {"mt_cbam", "st_path", "st_dens"}
    assert s["official_transfer"]["n"] == 200
    assert (tmp_path / "out" / "reliability_diagram.png").stat().st_size > 10_000


# ----------------------------------------------------------------------------------------------- INbreast
def test_birads_parsing_and_index(fake_kaggle):
    from mammo.inbreast import birads_number, index_inbreast_full, locate_inbreast
    assert birads_number("4a") == 4 and birads_number(" 6 ") == 6 and np.isnan(birads_number(""))
    df = index_inbreast_full(locate_inbreast(fake_kaggle))
    assert len(df) == 12 and df["patient"].nunique() == 6
    assert df["path"].str.endswith("_ANON.dcm").all()
    assert set(df["suspicious"]) <= {0, 1} and (df["suspicious"] == (df["birads_num"] >= 4)).all()
    assert (df["density"] == -1).sum() == 1


def test_to_uint8_window_and_inversion():
    from mammo.inbreast import to_uint8
    a = np.zeros((200, 100), np.uint16)
    a[20:180, :60] = np.linspace(2000, 12000, 60, dtype=np.uint16)     # 'breast' on a zero background
    out = to_uint8(a)
    assert out.dtype == np.uint8 and out[:, 80:].max() == 0 and out[100, 59] == 255
    assert 0 < out[100, 0] < out[100, 30] < out[100, 59]                # tissue contrast kept, skin not black
    inv = to_uint8(a.max() - a, invert=True)
    assert inv[100, 59] == 255 and inv[:, 80:].max() == 0


def test_dicom_roundtrip(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from mammo.inbreast import load_inbreast_image
    h, w = 400, 260
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.where(((yy - h / 2) / (0.45 * h)) ** 2 + ((xx - w + 1) / (0.8 * w)) ** 2 < 1, 9000, 0).astype(np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns, ds.SamplesPerPixel = h, w, 1
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 14, 13, 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = img.tobytes()
    f = tmp_path / "x.dcm"
    try:
        ds.save_as(f, enforce_file_format=True)
    except TypeError:  # pydicom < 3
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(f, write_like_original=False)
    out = load_inbreast_image(f, 160, 96)
    assert out.shape == (160, 96) and out.dtype == np.uint8
    assert out[:, :30].mean() > out[:, -30:].mean()                    # breast was on the right -> now on the left


def _fake_external(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    base = pd.DataFrame({"image_id": [str(20000000 + i) for i in range(n)], "patient": [f"{i // 4:016x}" for i in range(n)],
                         "side": "L", "view": "CC", "density": rng.integers(0, 4, n)})
    base.loc[:3, "density"] = -1
    bn = rng.choice([1, 2, 3, 4, 5, 6], n)
    base["birads"], base["birads_num"], base["suspicious"] = bn.astype(str), bn.astype(float), (bn >= 4).astype(int)
    out = tmp_path / "external"
    for name, tasks in {"mt_cbam": "pd", "st_dens": "d", "st_path": "p"}.items():
        d = base.copy(); d.insert(0, "config", name)
        if "p" in tasks:
            d["logit_malignant"] = 2 * (d["suspicious"] - 0.5) + rng.normal(0, 1.5, n)
            d["p_malignant"] = expit(d["logit_malignant"])
        if "d" in tasks:
            lg = rng.normal(0, 1, (n, 4)); lg[np.arange(n), np.clip(base["density"], 0, 3)] += 1.0
            for k in range(4):
                d[f"logit_density_{k}"] = lg[:, k]
                d[f"p_density_{k}"] = np.exp(lg[:, k]) / np.exp(lg).sum(1)
        (out / name).mkdir(parents=True)
        d.to_csv(out / name / "inbreast_predictions.csv", index=False)
        off = d.rename(columns={"suspicious": "y"}).drop(columns=["birads", "birads_num"])
        off.to_csv(out / name / "cbis_official_predictions.csv", index=False)
    json.dump({"path_auc": 0.7, "dens_qwk": 0.7}, open(out / "mt_cbam" / "cbis_official_metrics.json", "w"))
    base.to_csv(out / "inbreast_index.csv", index=False)
    return out


def test_external_summary(tmp_path):
    from mammo.experiments import external as E
    out = _fake_external(tmp_path)
    cbis = _fake_phase3(tmp_path / "cbis")
    json.dump({"path_auc": 0.746, "dens_qwk": 0.711}, open(cbis / "mt_cbam" / "official_metrics.json", "w"))
    assert E.main(["summarise", "--out", str(out), "--cbis-results", str(cbis), "--nboot", "50"]) == 0
    s = json.load(open(out / "summary.json"))
    r = s["results"]["mt_cbam"]
    assert r["inbreast_density"]["n_images"] == 116
    assert sum(map(sum, r["inbreast_density"]["confusion"])) == 116
    assert "cbis_temperature" in r["inbreast_density"]["calibration"]
    assert 0.5 < r["inbreast_malignancy_proxy"]["auc"] <= 1
    assert {c["question"] for c in s["comparisons"]} >= {"Does multi-task training help density transfer?"}
    assert s["official_reproduced"]["path_auc"] == {"phase3": 0.746, "phase4": 0.7}
    assert (out / "external_chart.png").stat().st_size > 10_000
    assert "INbreast" in (out / "summary.md").read_text()
