<div align="center">

# Multi-Task Mammography Classification with External Validation

**Auditable multi-task deep learning for joint BI-RADS assessment and breast-density classification**

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--educational-yellow)]()

</div>

> ⚠️ **Scope disclaimer.** This is an educational research and engineering project, **not a clinical diagnostic system**.  
> The primary target is derived from BI-RADS assessment categories, not biopsy-confirmed pathology ground truth.  
> Nothing in this repository has undergone clinical validation and it must not be used for diagnosis, screening, treatment decisions, or patient care.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Primary Dataset: INbreast](#2-primary-dataset-inbreast)
3. [Metadata and Label Audit](#3-metadata-and-label-audit)
4. [Problem Formulation](#4-problem-formulation)
5. [Data Split](#5-data-split)
6. [Models](#6-models)
7. [Training Strategy](#7-training-strategy)
8. [Internal Results](#8-internal-results)
9. [Why Accuracy Was Not Enough](#9-why-accuracy-was-not-enough)
10. [External Validation on CBIS-DDSM](#10-external-validation-on-cbis-ddsm)
11. [External Results](#11-external-results)
12. [Cross-Dataset Generalization](#12-cross-dataset-generalization)
13. [Key Findings](#13-key-findings)
14. [Limitations](#14-limitations)
15. [Repository Structure](#15-repository-structure)
16. [Reproducibility](#16-reproducibility)
17. [Relation to Reference Work](#17-relation-to-reference-work)
18. [Future Work](#18-future-work)
19. [Citation](#19-citation)
20. [License](#20-license)

---

# 1. Motivation

Mammograms contain information relevant to several related radiological tasks. Instead of building a completely separate network for every target, this project investigates whether a **shared visual representation** can support two tasks simultaneously:

1. **BI-RADS assessment classification**
2. **Breast-density classification**

The project began as a multi-task computer-vision experiment, but the main research question gradually became broader:

> **Does reasonable performance on an internal test split transfer to mammograms from an independent dataset?**

This matters because aggregate test accuracy from a single dataset can hide two important problems:

- severe failures on minority classes;
- poor generalization when image acquisition, processing, cohort composition, or annotation conventions change.

The project therefore focuses not only on model architecture, but also on:

- dataset provenance;
- metadata verification;
- class imbalance;
- reproducible splitting;
- class-wise evaluation;
- external validation;
- methodological limitations.

---

# 2. Primary Dataset: INbreast

The primary development dataset is **INbreast**, a full-field digital mammography dataset.

The processed local dataset contains **410 grayscale mammograms** organized by BI-RADS assessment:

| Training class | Images |
|---|---:|
| BI-RADS 1 | 67 |
| BI-RADS 2 | 220 |
| BI-RADS 3 | 23 |
| BI-RADS 4 | 43 |
| BI-RADS 5 | 57 |
| **Total** | **410** |

Images are stored as PNG files with unique 8-digit identifiers.

The class distribution is strongly imbalanced, especially for BI-RADS 3 and BI-RADS 4.

---

# 3. Metadata and Label Audit

A major part of the project was verifying that the processed dataset could actually be traced back to the original INbreast metadata.

The original `INbreast.xls` file contains 412 rows, of which 410 have usable image identifiers.

The audit found:

```text
Local PNG images:             410
Unique local image IDs:       410
Valid metadata image IDs:     410
Exact ID matches:             410 / 410
Missing local IDs:            0
Missing metadata IDs:         0
Duplicate valid image IDs:    0
```

This confirmed a one-to-one correspondence between the processed PNG extract and the usable original metadata.

## Original BI-RADS distribution

| Original BI-RADS | Images |
|---|---:|
| 1 | 67 |
| 2 | 220 |
| 3 | 23 |
| 4a | 13 |
| 4b | 8 |
| 4c | 22 |
| 5 | 49 |
| 6 | 8 |

For the five-class task, labels are normalized as:

| Model target | Original metadata |
|---|---|
| BI-RADS 1 | 1 |
| BI-RADS 2 | 2 |
| BI-RADS 3 | 3 |
| BI-RADS 4 | 4a / 4b / 4c |
| BI-RADS 5 | 5 / 6 |

After normalization:

```text
Folder-derived labels vs metadata labels:
410 / 410 agreement
```

The processed labels were therefore independently verified rather than silently trusted.

---

## Breast-density metadata

ACR density distribution:

| ACR density | Images |
|---|---:|
| 1 | 136 |
| 2 | 146 |
| 3 | 99 |
| 4 | 28 |
| Missing | 1 |

One image has missing density metadata.

The dataset loader uses a **density mask**, allowing the image to remain usable for BI-RADS training without contributing to density loss or density metrics.

---

# 4. Problem Formulation

The project performs **multi-task classification**.

## Task A — BI-RADS assessment

Five model classes:

```text
0 → BI-RADS 1
1 → BI-RADS 2
2 → BI-RADS 3
3 → BI-RADS 4
4 → BI-RADS 5
```

## Task B — Breast density

Four ACR density classes:

```text
0 → ACR 1
1 → ACR 2
2 → ACR 3
3 → ACR 4
```

The project deliberately does **not** call the BI-RADS target "cancer detection."

BI-RADS assessment is not equivalent to biopsy-confirmed malignancy ground truth.

---

# 5. Data Split

A fixed stratified image-level split is used.

Random seed:

```text
42
```

| Split | Images |
|---|---:|
| Train | 287 |
| Validation | 61 |
| Test | 62 |
| **Total** | **410** |

The split is stratified by BI-RADS class.

## Important limitation

The available INbreast metadata does not contain usable patient identifiers.

The `Patient ID` field is recorded as:

```text
removed
```

for the usable image rows.

Because the image filenames do not reliably encode patient identity, a verified patient-level split cannot be reconstructed.

Therefore, this project uses an **image-level split**, and patient-level independence between train, validation, and test subsets cannot be guaranteed.

This limitation is reported explicitly rather than claiming a leak-safe patient-level split.

---

# 6. Models

All primary experiments use a **ResNet18 backbone** with ImageNet initialization.

The main architecture shares a visual representation between the two tasks.

```text
Mammogram
    ↓
ResNet18 backbone
    ↓
Shared representation
   ↙                 ↘
BI-RADS head       Density head
5 classes          4 classes
```

Three model formulations and two baseline training protocols were evaluated.

---

## 6.1 Original multi-task baseline

A pretrained ResNet18 backbone with two linear task heads:

- 5-class BI-RADS classification;
- 4-class density classification.

File:

```text
src/models/multitask_resnet.py
```

---

## 6.2 Improved baseline

The architecture remains the same, but the training protocol introduces:

- `384 × 384` inputs;
- data augmentation;
- weighted sampling;
- class-weighted loss;
- masked density loss;
- learning-rate scheduling;
- early stopping;
- validation selection based on multi-task macro-F1.

This experiment was designed after the original baseline showed that overall accuracy could conceal complete failure on minority classes.

---

## 6.3 Ordinal BI-RADS model

BI-RADS classes are ordered, so a second model treats assessment as a cumulative-threshold problem.

Targets are encoded as:

```text
Class 0 → [0, 0, 0, 0]
Class 1 → [1, 0, 0, 0]
Class 2 → [1, 1, 0, 0]
Class 3 → [1, 1, 1, 0]
Class 4 → [1, 1, 1, 1]
```

The BI-RADS head uses binary cross-entropy across the four thresholds.

Density remains a standard four-class categorical task.

Files:

```text
src/models/ordinal_multitask_resnet.py
scripts/train_ordinal.py
scripts/evaluate_ordinal.py
```

---

## 6.4 Channel-spatial attention model

The attention experiment applies a CBAM-style channel-spatial attention module to the final ResNet18 convolutional features.

```text
ResNet18 features
       ↓
Channel attention
       ↓
Spatial attention
       ↓
Global average pooling
      ↙      ↘
BI-RADS    Density
```

Files:

```text
src/models/attention_multitask_resnet.py
scripts/train_attention.py
scripts/evaluate_attention.py
```

---

# 7. Training Strategy

The improved training protocol uses:

| Setting | Value |
|---|---|
| Input size | 384 × 384 |
| Batch size | 8 |
| Maximum epochs | 30 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Early stopping | Yes |
| BI-RADS sampling | WeightedRandomSampler |
| BI-RADS loss | Class-weighted cross-entropy |
| Density loss | Class-weighted cross-entropy |
| Missing density | Masked |

Training augmentation includes:

- random horizontal flip;
- small rotation;
- small translation;
- small scale perturbation.

Validation and test images receive no augmentation.

Model selection uses the average of:

```text
validation BI-RADS macro-F1
validation density macro-F1
```

---

# 8. Internal Results

All experiments below use the same fixed INbreast test split.

| Model | BI-RADS Accuracy | BI-RADS Macro-F1 | Density Accuracy | Density Macro-F1 |
|---|---:|---:|---:|---:|
| Original baseline | **0.5323** | 0.3385 | 0.6290 | 0.5930 |
| Improved baseline | 0.4032 | **0.3690** | **0.7097** | **0.7335** |
| Ordinal model | 0.4677 | 0.3424 | 0.6613 | 0.6143 |
| Attention model | 0.3387 | 0.3211 | 0.6613 | 0.6663 |

The **improved baseline** was selected for external evaluation.

Its strongest result was breast-density classification:

```text
Density accuracy:   70.97%
Density macro-F1:   73.35%
```

---

# 9. Why Accuracy Was Not Enough

The original baseline achieved the highest overall BI-RADS accuracy:

```text
53.23%
```

However, class-wise evaluation showed:

```text
BI-RADS 3 recall = 0
BI-RADS 4 recall = 0
```

The model was benefiting from the highly imbalanced class distribution.

After stronger class balancing, the improved baseline achieved lower overall BI-RADS accuracy but better macro-F1:

```text
Original BI-RADS macro-F1:   0.3385
Improved BI-RADS macro-F1:   0.3690
```

Minority classes also received non-zero recall.

This demonstrates why **accuracy alone is insufficient** for evaluating a strongly imbalanced medical-image classifier.

---

## Ordinal experiment

Test results:

```text
BI-RADS accuracy:    0.4677
BI-RADS macro-F1:    0.3424
BI-RADS MAE:         0.8548

Density accuracy:    0.6613
Density macro-F1:    0.6143
```

The ordinal formulation did not outperform the improved categorical baseline.

---

## Attention experiment

Test results:

```text
BI-RADS accuracy:    0.3387
BI-RADS macro-F1:    0.3211

Density accuracy:    0.6613
Density macro-F1:    0.6663
```

The attention model also did not outperform the improved baseline.

These negative results are intentionally retained.

Greater model complexity did not automatically improve generalization on a dataset of only 410 mammograms.

---

# 10. External Validation on CBIS-DDSM

To test cross-dataset generalization, the improved INbreast model was evaluated on **CBIS-DDSM**.

No CBIS-DDSM images were used during training.

No fine-tuning was performed.

Only the original INbreast-trained checkpoint was evaluated.

## Downloaded subsets

| CBIS-DDSM subset | Full mammogram series |
|---|---:|
| Mass-Test | 361 |
| Calc-Test | 284 |
| **Total** | **645** |

---

## Manifest verification

The TCIA `.tcia` manifests were parsed to obtain the expected `SeriesInstanceUID` values.

Every downloaded DICOM file was independently read using `pydicom` and matched against the manifests.

Audit result:

```text
Mass manifest series:          361
Calc manifest series:          284
Manifest overlap:              0

Downloaded Mass series:        361 / 361
Downloaded Calc series:        284 / 284

Unknown DICOM files:           0
Unreadable DICOM files:        0
```

This avoids relying on directory names to identify dataset membership.

---

## CBIS-DDSM metadata audit

The official case-description CSV files are lesion-level rather than mammogram-level.

Raw metadata:

```text
Mass lesion rows:    378
Calc lesion rows:    326
```

After aggregation:

```text
Mass mammograms:     361
Calc mammograms:     284
Total mammograms:    645
```

Fifty mammograms contain more than one lesion metadata row.

Four mammograms contain conflicting BI-RADS assessment values across lesion records.

Rather than assigning an arbitrary target, those four mammograms are excluded from BI-RADS evaluation.

---

## Compatible external labels

CBIS-DDSM includes values outside the INbreast task space:

```text
assessment = 0
density = 0
```

These values are **not silently remapped**.

Final compatible evaluation sets:

```text
BI-RADS usable mammograms:   595
Density usable mammograms:   643
```

---

## DICOM audit

All 645 external images share the following observed characteristics:

```text
PhotometricInterpretation: MONOCHROME2
TransferSyntaxUID:         1.2.840.10008.1.2
BitsStored:                16
PixelRepresentation:       unsigned
```

Header errors:

```text
0 / 645
```

Pixel decoding was verified for examples from both the Mass and Calc subsets.

---

# 11. External Results

## Basic DICOM preprocessing

The DICOM pipeline performs:

1. pixel decoding;
2. photometric handling;
3. percentile intensity normalization;
4. conversion to grayscale;
5. resize to `384 × 384`;
6. grayscale-to-RGB conversion;
7. ImageNet normalization.

Results:

| Metric | CBIS-DDSM |
|---|---:|
| BI-RADS Accuracy | 0.1832 |
| BI-RADS Macro-F1 | 0.0993 |
| Density Accuracy | 0.4012 |
| Density Macro-F1 | 0.3380 |

---

## Label-independent harmonization

Visual comparison showed substantial differences between the processed INbreast PNG images and CBIS-DDSM DICOM images.

A second preprocessing pipeline therefore introduced:

- cropping of large background regions;
- preservation of the main breast region;
- wider percentile-based intensity normalization.

Importantly:

- no CBIS labels were used to choose image-specific transformations;
- no model weights were updated;
- the same checkpoint was used.

Results:

| Evaluation | BI-RADS Accuracy | BI-RADS Macro-F1 | Density Accuracy | Density Macro-F1 |
|---|---:|---:|---:|---:|
| INbreast internal | 0.4032 | 0.3690 | 0.7097 | 0.7335 |
| CBIS-DDSM basic | 0.1832 | 0.0993 | 0.4012 | 0.3380 |
| CBIS-DDSM harmonized | 0.1866 | 0.0992 | 0.4028 | 0.3459 |

The harmonization produced only a minor change.

This indicates that simple cropping and intensity scaling alone do not explain the external performance gap.

---

# 12. Cross-Dataset Generalization

The largest failure occurred in BI-RADS transfer.

On harmonized CBIS-DDSM:

```text
BI-RADS 5 recall: 0.9346
BI-RADS 4 recall: 0.0030
```

The model shifted heavily toward predicting BI-RADS 5 even though BI-RADS 4 was the largest compatible external class.

Density transfer was less extreme, but still substantially weaker:

```text
INbreast density macro-F1:    0.7335
CBIS density macro-F1:        0.3459
```

The result is consistent with a substantial **cross-dataset domain shift**.

Potential contributors include:

- acquisition differences;
- scanner differences;
- image-processing differences;
- cohort composition;
- abnormality prevalence;
- annotation conventions;
- task-distribution differences;
- the small size of the original training dataset.

For this reason, the CBIS-DDSM experiment is interpreted as a **cross-dataset transfer test**, not as a direct estimate of clinical performance.

---

# 13. Key Findings

### 1. Dataset provenance matters

The processed INbreast dataset was traced back to the original metadata before training.

All 410 usable image identifiers and normalized BI-RADS labels were verified.

### 2. Accuracy can hide severe failures

The original baseline achieved higher BI-RADS accuracy while completely missing some minority classes.

Macro-F1 and per-class recall revealed this behavior.

### 3. More complex models are not automatically better

Neither the ordinal formulation nor channel-spatial attention outperformed the improved baseline.

### 4. Internal evaluation is not enough

The best internal model performed substantially worse on an independent dataset.

### 5. Simple preprocessing harmonization was not enough

The external gap remained almost unchanged after label-independent background cropping and intensity harmonization.

### 6. Negative results are useful

The project does not hide unsuccessful architectural experiments or poor external performance.

Those results reveal limitations that would be invisible if only the highest internal accuracy were reported.

---

# 14. Limitations

| Limitation | Detail |
|---|---|
| **Dataset size** | INbreast contains only 410 usable mammograms. |
| **Class imbalance** | BI-RADS 2 strongly dominates the training distribution. |
| **Image-level split** | Patient identifiers are unavailable, so patient-level independence cannot be guaranteed. |
| **Single fixed split** | Results come from one seed-42 split rather than repeated grouped cross-validation. |
| **Small minority classes** | BI-RADS 3 and BI-RADS 4 contain relatively few examples. |
| **External cohort differences** | CBIS-DDSM is a curated abnormal-mammography cohort rather than a representative screening population. |
| **Task alignment** | BI-RADS/assessment annotations across datasets should not be assumed to have perfectly identical distributions or annotation processes. |
| **External preprocessing** | DICOM acquisition and image-processing characteristics differ from the processed INbreast PNG images. |
| **No clinical validation** | Neither internal nor external results establish clinical usefulness. |

The external performance should therefore be interpreted cautiously.

It demonstrates poor transfer under the current task alignment, not that a clinically validated mammography system has been evaluated.

---

# 15. Repository Structure

```text
mammography-multitask-ai/
│
├── configs/
│
├── data/
│   ├── Inbreast/
│   │   ├── birads1/
│   │   ├── birads2/
│   │   ├── birads3/
│   │   ├── birads4/
│   │   └── birads5/
│   │
│   ├── metadata.csv
│   ├── metadata_split.csv
│   │
│   └── external/
│       └── cbis_ddsm/
│           └── local metadata / external files
│
├── results/
│   ├── baseline/
│   ├── baseline_improved/
│   ├── ordinal/
│   ├── attention/
│   ├── external_cbis/
│   └── external_cbis_harmonized/
│
├── scripts/
│   ├── check_inbreast_metadata.py
│   ├── build_metadata.py
│   ├── create_split.py
│   ├── test_dataset.py
│   │
│   ├── train_baseline.py
│   ├── train_ordinal.py
│   ├── train_attention.py
│   │
│   ├── evaluate_baseline.py
│   ├── evaluate_ordinal.py
│   ├── evaluate_attention.py
│   │
│   ├── audit_cbis_download.py
│   ├── audit_cbis_metadata.py
│   ├── build_cbis_metadata.py
│   ├── test_cbis_pixels.py
│   ├── audit_cbis_pixels_all.py
│   ├── test_cbis_dataset.py
│   ├── compare_preprocessing.py
│   ├── compare_preprocessing_harmonized.py
│   ├── evaluate_external_cbis.py
│   └── evaluate_external_cbis_harmonized.py
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── cbis_dataset.py
│   │   └── cbis_dataset_harmonized.py
│   │
│   └── models/
│       ├── multitask_resnet.py
│       ├── ordinal_multitask_resnet.py
│       └── attention_multitask_resnet.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Large datasets and trained checkpoints are intentionally excluded from version control.

---

# 16. Reproducibility

## Installation

Python 3.11 was used during development.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install requirements:

```bash
pip install -r requirements.txt
```

For external DICOM evaluation:

```bash
pip install pydicom
```

---

## Build INbreast metadata

```bash
python scripts/check_inbreast_metadata.py
python scripts/build_metadata.py
python scripts/create_split.py
```

---

## Train models

Improved categorical baseline:

```bash
python scripts/train_baseline.py
```

Ordinal variant:

```bash
python scripts/train_ordinal.py
```

Attention variant:

```bash
python scripts/train_attention.py
```

---

## Internal evaluation

```bash
python scripts/evaluate_baseline.py
python scripts/evaluate_ordinal.py
python scripts/evaluate_attention.py
```

Evaluation outputs include:

- JSON metrics;
- classification reports;
- confusion matrices.

---

## External evaluation

CBIS-DDSM is **not redistributed** with this repository.

After downloading the official Mass-Test and Calc-Test full-mammogram subsets and their case-description CSV files:

1. verify the manifests;
2. audit the metadata;
3. build image-level metadata;
4. verify DICOM decoding;
5. run external evaluation.

Example:

```bash
python scripts/evaluate_external_cbis.py
```

Harmonized evaluation:

```bash
python scripts/evaluate_external_cbis_harmonized.py
```

Both use the original INbreast-trained checkpoint without external fine-tuning.

---

# 17. Relation to Reference Work

This project was motivated by:

> Esen, G., Nurtas, M., La Paglia, L., Amankulov, J., Matkerim, B., & Altaibek, A.  
> **Multi-Task Attention-Guided Deep Learning for Simultaneous Breast Cancer Detection and Density Estimation in Mammography.**  
> *IEEE Access*, 2025.

The reference work inspired the use of:

- multi-task learning;
- mammographic image classification;
- attention mechanisms;
- shared feature extraction.

This repository is **not a reproduction** of that paper.

The accessible dataset and metadata required a more conservative task definition.

In particular, this project predicts:

```text
5-class BI-RADS assessment
+
4-class breast density
```

rather than presenting BI-RADS-derived labels as biopsy-confirmed cancer ground truth.

The goal is therefore not to reproduce the paper's reported metrics, but to use its ideas as a starting point for a technically auditable student research project.

---

# 18. Future Work

Possible extensions include:

- obtain reliable patient identifiers and repeat evaluation with patient-grouped splitting;
- multi-seed grouped cross-validation;
- confidence intervals for reported metrics;
- calibration analysis;
- strictly monotonic ordinal methods such as CORAL/CORN;
- external training or domain adaptation as a separate experiment;
- larger mammography datasets;
- investigation of acquisition-specific preprocessing;
- model uncertainty estimation;
- lesion-level localization;
- comparison against additional backbones;
- prospective evaluation on a genuinely independent cohort.

---

# 19. Citation

If you refer to this repository:

```bibtex
@misc{mammography_multitask_ai,
  author       = {sweetmAGIciaN7},
  title        = {Multi-Task Mammography Classification with External Validation},
  year         = {2026},
  howpublished = {\url{https://github.com/sweetmAGIciaN7/mammography-multitask-ai}},
  note         = {Educational research project; not for clinical use}
}
```

INbreast:

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
```

Reference work:

```bibtex
@article{esen2025multitask,
  title   = {Multi-Task Attention-Guided Deep Learning for Simultaneous Breast Cancer Detection and Density Estimation in Mammography},
  author  = {Esen, Gani and Nurtas, Marat and La Paglia, Laura and Amankulov, Jandos and Matkerim, Bazargul and Altaibek, Aizhan},
  journal = {IEEE Access},
  volume  = {13},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3634473}
}
```

---

# 20. License

Released under the [MIT License](LICENSE).

The INbreast and CBIS-DDSM datasets are **not redistributed** with this repository and remain subject to their respective access and usage terms.

---

## Disclaimer

This repository is for **educational and research purposes only**.

It is not a medical device, has not undergone clinical validation, and must not be used for diagnosis, screening, treatment decisions, or patient care.