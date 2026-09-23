# Multi-Task Mammography AI: a replication study

**Joint malignancy and breast-density classification, and why a published 93.6% turned out to be data leakage.**

![python](https://img.shields.io/badge/python-3.11-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.x-red) ![status](https://img.shields.io/badge/status-v2%20in%20progress-orange) ![license](https://img.shields.io/badge/license-MIT-green)

> ⚠️ Research and education only. Not a medical device. Must not be used for diagnosis or screening.

---

## TL;DR

I set out to reproduce [Esen et al., IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3634473), a multi-task
CNN with channel–spatial attention that reports **93.6% accuracy / 0.962 AUC** for benign-vs-malignant
classification on INbreast. My first attempt ([`archive/v1`](archive/v1)) reached only 34–53% accuracy. Instead of
tuning until the numbers looked good, I audited the paper:

1. **The data was augmented before it was split.** Rotated copies of the same mammogram ended up in both the
   training and test sets ([details](PAPER_AUDIT.md#1-main-issue-the-test-set-contains-copies-of-training-images)).
2. **The reported accuracies are arithmetically impossible** on a 20% split of 410 images, and only possible on
   the augmented copies ([§2](PAPER_AUDIT.md#2-the-reported-accuracies-imply-an-augmented-test-set)).
3. **The reported Brier score and ECE contradict each other.** Brier ≥ ECE² always holds; the paper's values
   violate it ([§4.2](PAPER_AUDIT.md#4-internal-inconsistencies)).

Then I **tested it**: the same model, trained on the same images, with only the split changed.

<!-- RESULTS:LEAKAGE -->
![Leakage experiment](results/leakage/leakage_chart.png)

EfficientNet-B0 with CBAM attention and two task heads, following the paper's architecture, loss weights and label
smoothing. The data is the pre-augmented set the paper appears to use: 106 INbreast mass images × 72 copies =
7,632 files from 50 patients. The model and training are identical in all three runs; 5-fold CV; 8 epochs; one
free Kaggle T4.

| Split | Test files whose original is in train | Malignancy AUC | Malignancy acc. | Density acc. |
|---|---:|---:|---:|---:|
| Paper protocol (augment, then split) | **100%** | **1.000** ± 0.000 | **1.000** | **1.000** |
| Grouped by original image | 0% | 0.867 ± 0.066 | 0.828 | 0.689 |
| Grouped by patient | 0% | **0.724** ± 0.093 | **0.697** | **0.403** |
| *Trivial baseline (always predict the most common class)* | | *0.500* | *0.66* | *0.40* |

Pooled out-of-fold AUC for the patient-grouped split: **0.755** (95% CI 0.667–0.835, cluster bootstrap over
original images).

**What this shows**

1. **The paper's protocol produces near-perfect scores regardless of what the model learns.** On average ~57 of each test image's 71 rotated or flipped siblings are in the training set.
   Our replication scores *higher* than the paper (1.000 vs 0.962): the protocol measures memorisation.
2. **Grouping by image is not enough.** With image-level grouping, 97% of test images still had a
   *different mammogram of the same patient* in training. Separating patients drops malignancy AUC from 0.87 to 0.72.
3. **The density "skill" disappears completely.** With a patient-level split, density accuracy (0.403) equals always
   guessing the most common category (0.401). The model had learned to recognise *patients*, not breast density.
   That is unsurprising for 50 patients, and it matters for any multi-task claim made on this data.
4. **Honest numbers on 106 images are uncertain.** Fold-to-fold AUC ranges from 0.62 to 0.82. Phase 3 therefore moves
   to CBIS-DDSM (~1,500 patients, biopsy-confirmed labels) for the model-building work.

## Phase 3: a leakage-free multi-task model (CBIS-DDSM)

Phase 2 showed that on ~106 images honest numbers are too uncertain to compare designs. Phase 3 moves to
**CBIS-DDSM**: about 3,100 mammograms from about 1,500 patients, with **biopsy-confirmed** pathology, a BI-RADS density
grade and a patient ID for every image. The question is no longer "can we hit 93%?" but **"once the split is
honest, does the paper's design actually help?"**

| Variant | Malignancy head | Density head | CBAM attention | Answers |
|---|:-:|:-:|:-:|---|
| `mt_cbam` | ✓ | ✓ | ✓ | the paper's design |
| `mt_plain` | ✓ | ✓ | – | does attention help? |
| `st_path` | ✓ | – | ✓ | does learning density help malignancy? |
| `st_dens` | – | ✓ | ✓ | does learning malignancy help density? |

**Design choices that keep the evaluation honest**

- **Patient-level 5-fold CV.** No patient ever appears in both train and test (checked by an assertion in
  every fold). All four variants share the same folds, so they can be compared image by image.
- **One image = one example.** CBIS-DDSM stores a mammogram twice when it has both a mass and a calcification;
  these copies are merged so the same film can't end up on both sides of a split.
- **No tuning on the test fold.** A fixed schedule of 15 epochs, with no early stopping and no threshold picked on test data.
- **Uncertainty reported.** 95% CIs come from a bootstrap over *patients*. Variant-vs-variant differences use a
  *paired* bootstrap on the same test images.
- **Standard benchmark too.** The main model is also trained on the official CBIS-DDSM training split and tested
  on the official test split. Patients who appear in both are counted as training patients.
- **Whole mammograms, not lesion crops.** Each image is cropped to the breast, oriented chest-wall-left and
  resized to 640×384. Malignancy is therefore predicted from the whole image, which is harder (and more
  realistic) than classifying a radiologist-drawn crop.

<!-- RESULTS:CBIS -->
*Results pending: the full run is in progress on Kaggle.*

## Roadmap

| Phase | Status | What |
|---|---|---|
| 1 | ✅ | Clean, tested codebase (`src/mammo`), v1 archived |
| 2 | ✅ | **Leakage experiment**: paper protocol vs. image-grouped vs. patient-grouped CV |
| 3 | 🏃 | Leakage-free multi-task model on CBIS-DDSM (biopsy-confirmed labels, patient-level split); single- vs. multi-task and attention ablations |
| 4 | ⏳ | Calibration (temperature scaling) and external validation on INbreast |
| 5 | ⏳ | Do attention maps point at lesions? Scored against radiologist ROI outlines |
| 6 | ⏳ | Final report |

## Repository layout

```
src/mammo/            data indexing, leakage-safe splits, model, training, metrics, plots
  experiments/        one script per experiment (python -m mammo.experiments.<name>)
notebooks/            thin Kaggle notebooks that call the scripts
tests/                unit tests (splits never leak, CBIS-DDSM label merging, preprocessing, metrics)
PAPER_AUDIT.md        section-by-section replication notes on the reference paper
archive/v1/           first attempt (kept for transparency)
```

## Reproduce

All experiments run on a free Kaggle GPU. Open the notebook on Kaggle, attach the datasets listed in its first
cell, turn on GPU and Internet, then **Run All**:

- [`notebooks/02_leakage_experiment.ipynb`](notebooks/02_leakage_experiment.ipynb): Phase 2, ~40 min
- [`notebooks/03_cbis_multitask.ipynb`](notebooks/03_cbis_multitask.ipynb): Phase 3, ~2 h on T4 x2

Locally: `pip install -r requirements.txt && PYTHONPATH=src pytest -q tests`

## Data

- **Huang & Lin (2020)**, *Dataset of breast mammography images with masses*, Data in Brief 31:105928 (CC BY-NC-SA 4.0). This is the pre-augmented INbreast mass images.
- **INbreast**: Moreira et al., *Academic Radiology* 19(2):236–248, 2012.
- **CBIS-DDSM**: Lee et al., *Scientific Data* 4:170177, 2017. We use the JPEG version on Kaggle
  ([awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)).

No images are redistributed in this repository.
