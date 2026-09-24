"""Phase 6: single-image inference and rendering for the public demo (Hugging Face Space, ``demo/app.py``).

The demo runs the **paper-design model trained on the official CBIS-DDSM training split** (``mt_cbam``, Phase 4)
on one uploaded mammogram and shows:

* the malignancy score, raw and after Platt scaling. The Platt parameters were fitted on the Phase 3
  cross-validation predictions and applied unchanged to this model (Phase 4a "official transfer": ECE
  0.206 -> 0.090 on the official test split). They are an honest best effort, not a guarantee;
* the four density-grade probabilities;
* Grad-CAM for the malignancy output (the map that localised lesions in Phase 5);
* the CBAM spatial gate on a **fixed 0-1 scale**, so a viewer can see for themselves that it is open almost
  everywhere (Phase 5), instead of a min-max-stretched heatmap that always looks "focused".

Everything except :func:`load_demo_model` / :func:`predict_file` / :func:`export_demo_checkpoint` is numpy-only,
so the rendering is unit-tested without torch.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np

DENSITY_NAMES = ("A: almost entirely fatty", "B: scattered fibroglandular", "C: heterogeneously dense",
                 "D: extremely dense")
DISCLAIMER = ("Research demo only. Not a medical device; must not be used for diagnosis, screening or any "
              "clinical decision.")


# ------------------------------------------------------------------------------------------------ torch parts
def export_demo_checkpoint(ckpt_path: str | Path, out_path: str | Path, calibration: dict | None = None,
                           meta: dict | None = None) -> dict:
    """Slim, self-describing copy of a Phase 4 checkpoint: float32 weights + training config + the calibration
    the demo applies + provenance. Returns the saved dict without the weights."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {"state_dict": {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()},
           "config": ck["config"], "height": int(ck.get("height", 640)), "width": int(ck.get("width", 384)),
           "calibration": calibration or {}, "meta": meta or {}}
    torch.save(out, out_path)
    return {k: v for k, v in out.items() if k != "state_dict"}


def load_demo_model(path: str | Path, device: str = "cpu"):
    """(model in eval mode, checkpoint dict). Never downloads ImageNet weights: they are in the state dict."""
    import torch
    from .train import TrainConfig, build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: (tuple(v) if k == "tasks" else v) for k, v in ck["config"].items()})
    cfg.pretrained = False
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def predict_file(model, ck: dict, path: str | Path, device: str = "cpu") -> dict:
    """Preprocess exactly as in training (``load_mammogram``), run the model once, return scores and maps."""
    import torch
    from .explain import explain_batch
    from .preprocess import load_mammogram_with_transform
    from .train import _Prep
    H, W = int(ck.get("height", 640)), int(ck.get("width", 384))
    img, tf = load_mammogram_with_transform(path, H, W)
    x = _Prep(device)(torch.from_numpy(img[None]), train=False)
    out = explain_batch(model, x)
    p = float(out["p_malignant"][0])
    res = {"image": img, "transform": tf.to_dict(), "p_malignant_raw": p,
           "p_malignant": calibrated_probability(p, ck.get("calibration", {}).get("platt")),
           "p_density": [float(v) for v in out["p_density"][0]] if "p_density" in out else None,
           "gradcam": out["gradcam_mal"][0], "cbam": out["cbam"][0] if "cbam" in out else None}
    return res


# ------------------------------------------------------------------------------------------------ numpy parts
def logit(p: float, eps: float = 1e-7) -> float:
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p) - math.log1p(-p)


def calibrated_probability(p_raw: float, platt: dict | None) -> float:
    """Platt scaling on the logit: sigmoid(a * logit + b). ``None`` -> the raw probability."""
    if not platt:
        return float(p_raw)
    z = platt["a"] * logit(p_raw) + platt["b"]
    return 1.0 / (1.0 + math.exp(-z))


def density_labels(p_density) -> dict:
    return {name: float(v) for name, v in zip(DENSITY_NAMES, p_density)}


def render_panels(res: dict, dpi: int = 110) -> np.ndarray:
    """Three panels as one RGB uint8 image: model input | Grad-CAM (malignancy) | CBAM gate on a fixed 0-1 scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .localization import breast_region, upsample
    from .plots import MAP_CMAP, SURFACE, TEXT, TEXT2

    img = res["image"]
    H, W = img.shape
    breast = breast_region(img)
    panels = [("Model input\n(cropped to the breast, chest wall left)", None)]
    cam = upsample(np.asarray(res["gradcam"], float), H, W)
    inside = cam[breast] if breast.any() else cam.ravel()
    hi = float(inside.max()) if inside.size else 0.0
    panels.append(("Grad-CAM, malignancy output\n(scaled to its own maximum)",
                   np.where(breast, cam / hi if hi > 0 else 0.0, np.nan)))
    if res.get("cbam") is not None:
        g = upsample(np.asarray(res["cbam"], float), H, W)
        raw = np.asarray(res["cbam"], float)
        panels.append((f"CBAM attention gate, fixed 0–1 scale\n(actual range {raw.min():.2f}–{raw.max():.2f})",
                       np.where(breast, g, np.nan)))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 5.2), facecolor=SURFACE)
    for ax, (title, m) in zip(np.atleast_1d(axes), panels):
        ax.axis("off")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        if m is not None:
            ax.imshow(m, cmap=MAP_CMAP, vmin=0, vmax=1, alpha=0.55)
            if np.isfinite(m).any():
                yx = np.unravel_index(np.nanargmax(m), m.shape)
                ax.scatter([yx[1]], [yx[0]], marker="x", s=46, c="white", linewidths=2.2, zorder=5)
                ax.scatter([yx[1]], [yx[0]], marker="x", s=46, c=TEXT, linewidths=0.8, zorder=6)
        ax.set_title(title, fontsize=8.5, color=TEXT, loc="left")
    fig.text(0.01, 0.01, "x = maximum of the map inside the breast.  " + DISCLAIMER, fontsize=6.5, color=TEXT2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.asarray(Image.open(buf).convert("RGB"))


def summary_markdown(res: dict) -> str:
    """Plain-language read-out of one prediction, with the caveats that matter."""
    p, raw = res["p_malignant"], res["p_malignant_raw"]
    lines = [f"**Malignancy score: {p:.0%}** (calibrated; raw model output {raw:.0%})."]
    if res.get("p_density") is not None:
        k = int(np.argmax(res["p_density"]))
        lines.append(f"**Most likely density grade: {DENSITY_NAMES[k]}** ({res['p_density'][k]:.0%}).")
    lines += [
        "",
        "How to read this:",
        "- The model was trained on CBIS-DDSM, **scanned film** mammograms that all contain a finding (a mass or "
        "calcifications). On that task it separates biopsy-confirmed malignant from benign findings with AUC "
        "0.75–0.78. That is useful for research and far from a radiologist.",
        "- The score is calibrated on CBIS-DDSM only. Phase 4 showed probabilities do **not** transfer to other "
        "scanners or populations, even when the ranking does.",
        "- Grad-CAM points within one cell of the lesion in about 44% of test images, so it can be wrong. The CBAM gate "
        "is shown on its true 0–1 scale: it is open almost everywhere, which is why it doesn't localise lesions.",
        "",
        f"_{DISCLAIMER}_",
    ]
    return "\n".join(lines)
