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
*Phase 2 results are being computed. The chart will appear here.*

## Roadmap

| Phase | Status | What |
|---|---|---|
| 1 | ✅ | Clean, tested codebase (`src/mammo`), v1 archived |
| 2 | 🔄 | **Leakage experiment**: paper protocol vs. image-grouped vs. patient-grouped CV |
| 3 | ⏳ | Leakage-free multi-task model on CBIS-DDSM (biopsy-confirmed labels, patient-level split); single- vs. multi-task and attention ablations |
| 4 | ⏳ | Calibration (temperature scaling) and external validation on INbreast |
| 5 | ⏳ | Do attention maps point at lesions? Scored against radiologist ROI outlines |
| 6 | ⏳ | Final report |

## Repository layout

```
src/mammo/            data indexing, leakage-safe splits, model, training, metrics, plots
  experiments/        one script per experiment (python -m mammo.experiments.<name>)
notebooks/            thin Kaggle notebooks that call the scripts
tests/                unit tests (splits never leak, metric identities, filename parsing)
PAPER_AUDIT.md        section-by-section replication notes on the reference paper
archive/v1/           first attempt (kept for transparency)
```

## Reproduce

All experiments run on a free Kaggle GPU. Open
[`notebooks/02_leakage_experiment.ipynb`](notebooks/02_leakage_experiment.ipynb) on Kaggle, attach the two datasets
listed at the top, turn on GPU and Internet, then **Run All**.

Locally: `pip install -r requirements.txt && PYTHONPATH=src pytest -q tests`

## Data

- **Huang & Lin (2020)**, *Dataset of breast mammography images with masses*, Data in Brief 31:105928 (CC BY-NC-SA 4.0). This is the pre-augmented INbreast mass images.
- **INbreast**: Moreira et al., *Academic Radiology* 19(2):236–248, 2012.
- **CBIS-DDSM**: Lee et al., *Scientific Data* 4:170177, 2017.

No images are redistributed in this repository.
