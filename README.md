<div align="center">

# Multi-Task Mammography Classification with Attention and Ordinal Assessment

**Interpretable multi-task deep learning for joint BI-RADS assessment and breast-density classification on INbreast**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--educational-yellow)]()

</div>

> ⚠️ **Scope disclaimer.** This is an educational research / engineering project, **not** a clinical diagnostic system. The main target is derived from BI-RADS *assessment categories*, which are **not biopsy-confirmed pathology ground truth**. Nothing in this repository has undergone clinical validation and it must not be used for diagnosis, screening, or treatment decisions. See [§ Limitations](#12-limitations) and [§ Disclaimer](#disclaimer).

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Dataset](#2-dataset)
3. [Problem Reformulation](#3-problem-reformulation)
4. [Data Split](#4-data-split)
5. [Preprocessing](#5-preprocessing)
6. [Models](#6-models)
7. [Training](#7-training)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Results](#9-results)
10. [Cumulative-Threshold Consistency Audit](#10-cumulative-threshold-consistency-audit)
11. [Grad-CAM Analysis](#11-grad-cam-analysis)
12. [Limitations](#12-limitations)
13. [Reproducibility / Quickstart](#13-reproducibility--quickstart)
14. [Repository Structure](#14-repository-structure)
15. [Relation to Reference Work](#15-relation-to-reference-work)
16. [Future Work](#16-future-work)
17. [Citation](#17-citation)
18. [License](#18-license)
19. [Disclaimer](#disclaimer)

---

## 1. Motivation

Mammograms carry information relevant to several related radiological tasks. Rather than training one network per label, multi-task learning tests whether a **shared visual representation** helps a model learn complementary structure across tasks.

This project studies:

- Can a single EfficientNet-B0 backbone jointly predict **BI-RADS assessment** and **breast density**?
- Does class weighting reduce collapse toward the majority assessment class?
- Does a **cumulative-threshold (ordinal-inspired)** formulation help an inherently ordered target?
- Does shared channel-spatial attention improve on a plain shared backbone?
- Does a more complex **task-specific dual-attention** design outperform shared attention?
- Which regions drive the model's predictions, per Grad-CAM?

The project was inspired by recent attention-guided multi-task work on mammography (see [§ 15](#15-relation-to-reference-work)), but it is **not a reproduction** of that work — the accessible dataset and metadata required a different, more conservative problem formulation.

## 2. Dataset

[**INbreast**](https://doi.org/10.1016/j.acra.2011.09.014) — a full-field digital mammography dataset: **410 mammograms from 115 cases**.

> Moreira, I. C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M. J., & Cardoso, J. S. *INbreast: Toward a Full-field Digital Mammographic Database.* Academic Radiology, 19(2), 236–248, 2012.

A reproducibility audit (`scripts/audit_data.py`) verifies, on the local PNG extract:

- 410 samples, 410 unique image IDs, 410 unique image paths
- one-to-one correspondence between PNG IDs and accessible metadata
- no duplicate IDs / paths
- no overlap between train, validation, and test indices

```bash
python -m scripts.audit_data
```

## 3. Problem Reformulation

The paper that motivated this project uses binary pathology classification alongside 4-class density estimation. In the **accessible** INbreast metadata used here, the only reliably present labels are **BI-RADS assessment** and **ACR density** — a binary target derived from BI-RADS is *not* equivalent to biopsy-confirmed malignancy. The task was therefore reformulated rather than mislabeled as "cancer detection."

### Task A — 5-class BI-RADS assessment

| Model class | Source BI-RADS |
|---|---|
| 0 | BI-RADS 1 |
| 1 | BI-RADS 2 |
| 2 | BI-RADS 3 |
| 3 | BI-RADS 4a / 4b / 4c |
| 4 | BI-RADS 5 / 6 |

### Task B — 4-class breast density

| Model class | ACR density |
|---|---|
| 0 | ACR 1 |
| 1 | ACR 2 |
| 2 | ACR 3 |
| 3 | ACR 4 |

One image has a missing density annotation; its target is encoded as `-1` and masked out of both the density loss and density metrics.

## 4. Data Split

Fixed image-level split, seed `42`, stratified by the 5-class assessment target:

| Split | Images | Fraction |
|---|---|---|
| Train | 262 | 64% |
| Validation | 65 | 16% |
| Test | 83 | 20% |
| **Total** | **410** | **100%** |

**Known limitation — image-level, not patient-level split.** The accessible metadata does not expose a usable patient identifier, so patient-level independence between subsets **cannot be guaranteed**. Treat all reported metrics with this caveat in mind (see [§ 12](#12-limitations)).

## 5. Preprocessing

Mammograms are loaded grayscale and converted to 3 channels for ImageNet-pretrained backbones.

| Stage | Transform |
|---|---|
| Train | resize `224×224`, grayscale→RGB, random horizontal flip (p=0.5), brightness jitter (±10%), contrast jitter (±10%), ImageNet normalization |
| Val / Test | resize `224×224`, grayscale→RGB, ImageNet normalization (**no augmentation**) |

## 6. Models

All variants share an **EfficientNet-B0** backbone (ImageNet-initialized) so that architectural comparisons are controlled.

<details>
<summary><b>6.1 Weighted categorical baseline</b></summary>

```
Mammogram → EfficientNet-B0 → shared 256-d FC representation
                                   ├── 5-class BI-RADS assessment (class-weighted)
                                   └── 4-class density
```
</details>

<details>
<summary><b>6.2 Cumulative-threshold assessment variant</b></summary>

The 5-class assessment head is replaced by 4 cumulative binary thresholds:

```
class 0 → [0,0,0,0]      class 3 → [1,1,1,0]
class 1 → [1,0,0,0]      class 4 → [1,1,1,1]
class 2 → [1,1,0,0]
```

Thresholds are optimized independently, so **monotonicity is not structurally enforced** — see the audit in [§ 10](#10-cumulative-threshold-consistency-audit). Density remains a standard 4-class head.
</details>

<details>
<summary><b>6.3 Shared channel-spatial attention</b></summary>

```
EfficientNet features → Channel attention → Spatial attention → GAP → shared representation
                                                                          ├── Assessment
                                                                          └── Density
```

A CBAM-style module inserted before global average pooling; both tasks read the same attended representation.
</details>

<details>
<summary><b>6.4 Task-specific dual-attention model</b></summary>

```
                     ┌─ Assessment attention ─ FC ─ Assessment head
EfficientNet features
                     └─ Density attention ───── FC ─ Density head
```

Branches after the shared backbone with **separate** attention pathways and post-pooling representations per task — a more complex design, not merely "two attention blocks."
</details>

## 7. Training

**Stage 1 — frozen backbone**: train attention / shared layers / heads · AdamW · `lr=1e-4` · `wd=1e-2`
**Stage 2 — full fine-tuning**: unfreeze backbone · AdamW · `lr=5e-5` · `wd=5e-3`

| Setting | Value |
|---|---|
| Batch size | 16 |
| Image size | 224 |
| Gradient clipping | norm 1.0 |
| Task weights (assessment / density) | 1.0 / 1.0 |
| Label smoothing | 0.05 |
| Checkpoint selection | mean of validation assessment macro-F1 and density macro-F1 |

The best validation checkpoint is restored before test evaluation.

## 8. Evaluation Protocol

| Task | Metrics |
|---|---|
| **Assessment** | Accuracy, Balanced Accuracy, Macro-F1, MAE (classes are ordered, so MAE penalizes distant confusions more than adjacent ones) |
| **Density** | Accuracy, Balanced Accuracy, Macro-F1 |

Class-wise F1 and confusion matrices are available in `src/evaluation/metrics.py`.

## 9. Results

All models evaluated on the same fixed 83-image test split (`results/tables/model_comparison.csv`):

| Model | Assessment Acc. | Assessment Bal. Acc. | Assessment Macro-F1 | Assessment MAE ↓ | Density Acc. | Density Bal. Acc. | Density Macro-F1 |
|---|---|---|---|---|---|---|---|
| Weighted categorical baseline | 0.265 | 0.237 | 0.209 | 1.506 | 0.518 | 0.401 | 0.373 |
| Cumulative-threshold assessment | 0.253 | 0.274 | 0.193 | 1.229 | **0.639** | **0.513** | **0.488** |
| **Shared attention** | **0.470** | **0.279** | **0.256** | **1.133** | 0.494 | 0.384 | 0.347 |
| Dual attention | 0.386 | 0.249 | 0.237 | 1.253 | 0.482 | 0.367 | 0.329 |

**Interpretation**

- **Shared attention** gives the strongest assessment results across every assessment metric.
- **Cumulative-threshold** gives the strongest *density* results — but density is a plain categorical head there, so this is **not** evidence that ordinal density modeling helps; treat it as an incidental effect of that run.
- **Dual attention** adds complexity without beating shared attention on this small dataset.
- **Class weighting** meaningfully reduces collapse toward the dominant assessment class relative to the unweighted baseline.

## 10. Cumulative-Threshold Consistency Audit

The cumulative-threshold head does not enforce monotonic outputs by construction. A post-hoc audit found non-monotonic patterns (e.g. `[1,0,1,0]`) in:

- **15 / 65** validation images (23.1%)
- **22 / 83** test images (26.5%)

Predictions are decoded by counting thresholds with sigmoid probability > 0.5. Because this was discovered *after* the main experiment, **the decoder was not changed post-hoc using test-set behavior** — the finding is reported as a methodological limitation, not silently patched. A strictly monotonic formulation (e.g. CORAL/CORN-style shared weights) is listed under [§ 16](#16-future-work).

## 11. Grad-CAM Analysis

Grad-CAM was computed for the shared-attention model on both task heads. These maps are:

- **not** the model's learned attention weights,
- **not** lesion segmentations,
- **not** evidence of clinical validity —

only qualitative saliency showing which backbone regions influenced a given prediction.

| Multi-example gallery | Assessment: mid vs. late layer | Density: mid vs. late layer |
|---|---|---|
| ![gallery](results/figures/gradcam_gallery_shared_attention.png) | ![assessment](results/figures/22580548_assessment_gradcam_comparison.png) | ![density](results/figures/22580548_density_gradcam_comparison.png) |

In the inspected examples, intermediate backbone layers produce more spatially localized maps than the final representation, with saliency often concentrated within breast tissue rather than background — a qualitative observation from selected examples, not a validated localization metric.

## 12. Limitations

| Limitation | Detail |
|---|---|
| **Dataset size** | INbreast has only 410 images; rare assessment/density classes are underrepresented. |
| **Image-level split** | No usable patient identifier in accessible metadata ⇒ same-patient leakage across subsets cannot be excluded. |
| **Single split / seed** | All comparisons use one stratified split (seed 42); no multi-seed or patient-grouped CV yet. |
| **Repeated hold-out inspection** | The same test split was consulted while comparing architectures during development. It was never used for gradient updates or checkpoint selection, but should not be treated as a fully untouched final benchmark. |
| **Modest assessment performance** | Macro-F1 stays low across all variants — useful for *relative* comparison of design choices, not for deployment-level claims. |
| **Ordinal monotonicity** | ~1/4 of cumulative-threshold predictions are non-monotonic (§ 10). |
| **Grad-CAM is qualitative** | No lesion-localization accuracy or radiologist-agreement metric is computed. |

## 13. Reproducibility / Quickstart

```bash
git clone https://github.com/sweetmAGIciaN7/mammography-multitask-ai
cd mammography-multitask-ai
pip install -r requirements.txt

# 1. Verify data integrity
python -m scripts.audit_data

# 2. Train (pick one or more)
python -m src.training.train                  # weighted categorical baseline
python -m src.training.train_ordinal           # cumulative-threshold variant
python -m src.training.train_attention         # shared channel-spatial attention
python -m src.training.train_dual_attention    # task-specific dual attention

# 3. Evaluate all saved experiments
python -m src.evaluation.compare_models

# 4. Grad-CAM
python -m src.evaluation.gradcam
python -m src.evaluation.gradcam_gallery
```

Model checkpoints and the INbreast dataset itself are **intentionally excluded** from version control (obtain INbreast directly from its [official source](http://medicalresearch.inescporto.pt/breastresearch/index.php); usage is subject to its own terms).

## 14. Repository Structure

```
mammography-multitask-ai/
├── configs/                 baseline.yaml
├── data/                    README.md (data access notes; no raw data committed)
├── experiments/             README.md (experiment tracking notes)
├── results/
│   ├── figures/              Grad-CAM galleries and layer comparisons
│   └── tables/                model_comparison.csv
├── scripts/                 audit_data.py
├── src/
│   ├── data/                 dataset.py, loaders.py, preprocessing.py, splits.py
│   ├── evaluation/           compare_models.py, gradcam.py, gradcam_gallery.py,
│   │                         metrics.py, ordinal_metrics.py
│   ├── models/                attention.py, multitask_baseline.py,
│   │                         multitask_attention.py, multitask_dual_attention.py,
│   │                         multitask_ordinal.py
│   └── training/             losses.py, ordinal_losses.py, train.py,
│                             train_attention.py, train_dual_attention.py,
│                             train_ordinal.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 15. Relation to Reference Work

This project was motivated by:

> Esen, G., Nurtas, M., La Paglia, L., Amankulov, J., Matkerim, B., & Altaibek, A. *Multi-Task Attention-Guided Deep Learning for Simultaneous Breast Cancer Detection and Density Estimation in Mammography.* IEEE Access, 13, 2025. DOI: [10.1109/ACCESS.2025.3634473](https://doi.org/10.1109/ACCESS.2025.3634473)

That work combines binary malignancy classification with 4-class BI-RADS density (A–D) using six backbones (ResNet-50, DenseNet-121, EfficientNet-B0/B3/B7, MobileNetV3-Large), CBAM-style channel+spatial attention, stratified hold-out (64/16/20) plus 5-fold CV, and reports AUC up to 0.962 with extensive calibration analysis.

**This repository deliberately differs:**

| Aspect | Reference paper | This repository |
|---|---|---|
| Primary target | Binary malignancy (benign/malignant) | 5-class BI-RADS **assessment** (ordinal-inspired) |
| Ground truth | Framed as pathology | Explicitly **not** biopsy-confirmed; BI-RADS assessment only |
| Backbones compared | 6 (ResNet-50 → EfficientNet-B7) | 1 fixed backbone (EfficientNet-B0) across 4 task/attention variants, for a controlled comparison |
| Data augmentation | Extensive (multi-angle rotation + flips + CLAHE) | Light (flip + brightness/contrast jitter) to limit synthetic-artifact risk on n=410 |
| Ordinality | Not modeled | Explicit cumulative-threshold variant, with a **documented non-monotonicity audit** |
| Reported scope | Clinical framing, deployment recommendations | Educational framing; claims limited to what image-level, n=410 evidence supports |

The goal here is not to reproduce the paper's reported metrics, but to use its multi-task/attention ideas as a starting point for a **technically auditable** student research exercise — including reporting where results are weaker or more ambiguous than a clinical framing would suggest.

## 16. Future Work

- Obtain reliable patient identifiers and repeat evaluation with a patient-grouped split.
- Multi-seed or patient-grouped cross-validation with confidence intervals.
- Evaluate on a second, larger public dataset (e.g. CBIS-DDSM) to stress-test conclusions.
- Implement a strictly monotonic ordinal head (CORAL/CORN-style shared thresholds).
- Quantify predictive uncertainty and calibration (Brier score, ECE, temperature scaling).
- Validate Grad-CAM against expert lesion annotations rather than visual inspection alone.
- Separate architecture selection from a genuinely untouched final test set.

## 17. Citation

If you use this code or refer to this analysis, please cite:

```bibtex
@misc{mammography_multitask_ai,
  author       = {sweetmAGIciaN7},
  title        = {Multi-Task Mammography Classification with Attention and Ordinal Assessment},
  year         = {2026},
  howpublished = {\url{https://github.com/sweetmAGIciaN7/mammography-multitask-ai}},
  note         = {Educational research project; not for clinical use}
}
```

The dataset and the architectural ideas that motivated this project should also be cited:

```bibtex
@article{moreira2012inbreast,
  title   = {INbreast: Toward a Full-field Digital Mammographic Database},
  author  = {Moreira, Inês C. and Amaral, Igor and Domingues, Inês and Cardoso, Ana and Cardoso, Maria J. and Cardoso, Jaime S.},
  journal = {Academic Radiology},
  volume  = {19},
  number  = {2},
  pages   = {236--248},
  year    = {2012},
  doi     = {10.1016/j.acra.2011.09.014}
}

@article{esen2025multitask,
  title   = {Multi-Task Attention-Guided Deep Learning for Simultaneous Breast Cancer Detection and Density Estimation in Mammography},
  author  = {Esen, Gani and Nurtas, Marat and La Paglia, Laura and Amankulov, Jandos and Matkerim, Bazargul and Altaibek, Aizhan},
  journal = {IEEE Access},
  volume  = {13},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3634473}
}
```

## 18. License

Released under the [MIT License](LICENSE). The INbreast dataset is **not** redistributed with this repository and is governed by its own terms — request access from the [official source](http://medicalresearch.inescporto.pt/breastresearch/index.php).

## Disclaimer

This repository is for **educational and research purposes only**. It is not a medical device, has not undergone clinical validation, and must not be used for diagnosis, screening, treatment decisions, or patient care.
