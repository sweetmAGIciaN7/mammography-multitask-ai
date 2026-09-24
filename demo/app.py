"""Gradio demo: multi-task mammography model (paper design, trained on CBIS-DDSM, Phase 4).

Space layout (built by notebooks/06_demo_export.ipynb):
    app.py  requirements.txt  README.md  model.pt  mammo/  examples/*.jpg  examples/examples.csv
Run locally from the repo:  python demo/app.py  (needs demo/model.pt and gradio).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]  # Space: ./mammo ; repo: ../src/mammo

import gradio as gr  # noqa: E402

from mammo.demo import (DISCLAIMER, density_labels, load_demo_model, predict_file,  # noqa: E402
                        render_panels, summary_markdown)

REPO = "https://github.com/sweetmAGIciaN7/mammography-multitask-ai"
MODEL, CK = load_demo_model(HERE / "model.pt")


def analyse(path):
    if not path:
        raise gr.Error("Upload a mammogram (PNG or JPEG) or pick one of the examples.")
    try:
        res = predict_file(MODEL, CK, path)
    except Exception as e:  # unreadable file etc.
        raise gr.Error(f"Could not read this image: {e}")
    labels = {"malignant": res["p_malignant"], "benign": 1 - res["p_malignant"]}
    dens = density_labels(res["p_density"]) if res["p_density"] is not None else {}
    return render_panels(res), labels, dens, summary_markdown(res)


def _examples():
    f = HERE / "examples" / "examples.csv"
    if not f.exists():
        return [], ""
    rows = list(csv.DictReader(open(f)))
    table = ["| Example | Biopsy result | Lesion | Density (radiologist) |", "|---|---|---|---|"]
    for r in rows:
        table.append(f"| `{r['file']}` | {r['pathology']} | {r['lesion_type']} | {r['density']} |")
    return [[str(HERE / "examples" / r["file"])] for r in rows], "\n".join(table)


EXAMPLES, EXAMPLE_TABLE = _examples()

INTRO = f"""
# Multi-task mammography model: malignancy + breast density

A replication of the multi-task, attention-guided CNN of Esen et al. (IEEE Access 2025), rebuilt with a
**leakage-free, patient-level evaluation**. Upload a full mammogram (one view) to get a malignancy score, a
breast-density grade and two heatmaps. Code, experiments and the full story are on
[GitHub]({REPO}): the paper reported 93.6% accuracy; with an honest split the same design reaches AUC 0.78.

**{DISCLAIMER}**
"""

EXAMPLES_NOTE = """
**Examples:** 6 images drawn at random (fixed seed, not selected) from the official CBIS-DDSM **test** split,
which the model never saw: 3 biopsy-confirmed malignant, 3 benign. Some are misclassified; that is expected.
CBIS-DDSM: Lee et al., *Scientific Data* 2017, CC BY 3.0.
"""

with gr.Blocks(title="Multi-task mammography model") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="filepath", label="Mammogram (PNG / JPEG)", height=420)
            btn = gr.Button("Analyse", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp)
                gr.Markdown(EXAMPLES_NOTE + "\n" + EXAMPLE_TABLE)
        with gr.Column(scale=2):
            out_img = gr.Image(label="What the model sees", type="numpy")
            with gr.Row():
                out_mal = gr.Label(label="Malignancy (calibrated on CBIS-DDSM)")
                out_dens = gr.Label(label="Breast density (BI-RADS)")
            out_txt = gr.Markdown()
    btn.click(analyse, inp, [out_img, out_mal, out_dens, out_txt])

if __name__ == "__main__":
    demo.launch()
