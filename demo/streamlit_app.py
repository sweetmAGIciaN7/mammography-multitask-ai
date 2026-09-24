"""Streamlit demo: the paper-design model (EfficientNet-B0 + CBAM, two heads) trained on CBIS-DDSM (Phase 4).

Deployed on Streamlit Community Cloud from this repository (entrypoint: demo/streamlit_app.py).
Run locally:  pip install -r demo/requirements.txt && streamlit run demo/streamlit_app.py
Needs demo/model.pt and demo/examples/ (built by notebooks/06_demo_export.ipynb).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from mammo.demo import (DENSITY_NAMES, DISCLAIMER, load_demo_model, predict_file,  # noqa: E402
                        render_panels, summary_markdown)

REPO = "https://github.com/sweetmAGIciaN7/mammography-multitask-ai"
st.set_page_config(page_title="Multi-task mammography model", page_icon="🩻", layout="wide")


@st.cache_resource
def model():
    return load_demo_model(HERE / "model.pt")


@st.cache_data
def examples() -> pd.DataFrame:
    f = HERE / "examples" / "examples.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(max_entries=20)
def analyse(path: str) -> dict:
    m, ck = model()
    res = predict_file(m, ck, path)
    res["panels"] = render_panels(res)
    return res


st.title("Multi-task mammography model")
st.markdown(
    f"Malignancy and breast density from one mammogram. The model replicates the multi-task, attention-guided CNN "
    f"of Esen et al. (IEEE Access 2025), rebuilt with a **leakage-free, patient-level evaluation**. The paper reported "
    f"93.6% accuracy; with an honest split the same design reaches AUC 0.75–0.78. Full study: [GitHub]({REPO}).")
st.warning(DISCLAIMER, icon="⚠️")

ex = examples()
left, right = st.columns([1, 2.2], gap="large")
with left:
    st.subheader("1. Choose an image")
    source = st.radio("Source", ["Example from the test set", "Upload your own"], label_visibility="collapsed")
    path, truth = None, None
    if source.startswith("Example") and len(ex):
        labels = [f"Example {i + 1}" for i in range(len(ex))]
        k = st.selectbox("Example", range(len(ex)), format_func=lambda i: labels[i])
        row = ex.iloc[k]
        path = str(HERE / "examples" / row["file"])
        truth = row
        st.image(path)
        st.caption("Six images drawn **at random** (fixed seed, not selected) from the official CBIS-DDSM test split, "
                   "which the model never saw: 3 biopsy-proven malignant, 3 benign. Some are misclassified. "
                   "CBIS-DDSM: Lee et al., *Scientific Data* 2017, CC BY 3.0.")
    else:
        up = st.file_uploader("Mammogram (PNG or JPEG, one view)", type=["png", "jpg", "jpeg"])
        if up is not None:
            tmp = Path(st.session_state.get("_tmpdir", "/tmp")) / f"upload_{up.file_id}{Path(up.name).suffix}"
            tmp.write_bytes(up.getvalue())
            path = str(tmp)
        st.caption("Trained on scanned-film mammograms that contain a finding. Other images (digital systems, "
                   "screening exams, non-mammograms) give unreliable or meaningless output.")

with right:
    st.subheader("2. What the model says")
    if path is None:
        st.info("Pick an example or upload an image.")
    else:
        try:
            with st.spinner("Running the model (a few seconds on CPU)..."):
                res = analyse(path)
        except Exception as e:  # unreadable file etc.
            st.error(f"Could not analyse this image: {e}")
            st.stop()
        c1, c2, c3 = st.columns(3)
        c1.metric("Malignancy score (calibrated)", f"{res['p_malignant']:.0%}",
                  help="Platt scaling fitted on CBIS-DDSM cross-validation predictions. Raw model output: "
                       f"{res['p_malignant_raw']:.0%}.")
        kd = int(max(range(4), key=lambda i: res["p_density"][i]))
        c2.metric("Density grade", DENSITY_NAMES[kd].split(":")[0])
        c2.caption(f"{DENSITY_NAMES[kd].split(': ')[1]} ({res['p_density'][kd]:.0%})")
        if truth is not None:
            c3.metric("Ground truth (biopsy)", str(truth["pathology"]))
            c3.caption(f"{truth['lesion_type']}; radiologist's density grade {truth['density']}")
        st.image(res["panels"])
        st.bar_chart(pd.DataFrame({"probability": res["p_density"]}, index=[n.replace(": ", " · ") for n in DENSITY_NAMES]),
                     horizontal=True, height=180)
        st.markdown(summary_markdown(res))
