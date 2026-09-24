# Replication notes on Esen et al. (2025)

> G. Esen, M. Nurtas, L. La Paglia, J. Amankulov, B. Matkerim, A. Altaibek,
> *"Multi-Task Attention-Guided Deep Learning for Simultaneous Breast Cancer Detection and Density Estimation in Mammography,"*
> IEEE Access, vol. 13, pp. 198938–198951, 2025. DOI: [10.1109/ACCESS.2025.3634473](https://doi.org/10.1109/ACCESS.2025.3634473)

This project started as an attempt to reproduce that paper. While doing so, I found that the reported numbers
cannot be reached with a leakage-free evaluation, and I traced why. These notes are written in the spirit of
a replication study. The architecture proposed in the paper is reasonable and is re-implemented here
([`src/mammo/model.py`](src/mammo/model.py)). The concerns below are about the **evaluation**, which determines
what the reported numbers mean.

Every claim cites a section, table or figure of the paper. Claims marked 🧪 are tested by code in this repository.

---

## 1. Main issue: the test set contains copies of training images

**What the paper says.** Section III-B: each image was rotated to 11 angles (30°–330°), and the original and
rotated images were flipped horizontally and vertically. *"Subsequent to the augmentation process, the dataset
was systematically partitioned into training and testing subsets, as delineated in Table 1."* Table 1 lists
7,208 augmented images split 4,614 / 1,154 / 1,440.

**Why it matters.** If augmentation happens *before* the split, a rotated copy of a test mammogram is almost
always in the training set. The model is then tested on images it has effectively seen already, so the score
measures memorisation, not generalisation to new patients.

**The data source.** Figure 1 cites reference [12], Huang & Lin (2020), *"Dataset of breast mammography images
with masses"* (Data in Brief 31:105928). That public dataset contains **106 INbreast mass images augmented to
7,632 files**, organised into exactly the eight *Density × Benign/Malignant* categories of Table 1 and
preprocessed with CLAHE (as mentioned in Sec. III-B). The paper's dataset is therefore very likely ~106 original
mammograms at ~70 copies each, not 410 independent images as the abstract states.

🧪 **Tested in Phase 2** ([`notebooks/02_leakage_experiment.ipynb`](notebooks/02_leakage_experiment.ipynb),
[results](results/leakage/)). The paper's architecture was trained on that dataset three times, changing only the split.
The paper's split gave malignancy AUC **1.000**. Grouping by patient gave **0.724 ± 0.093**. Density accuracy fell from
1.000 to **0.403**, the same as always predicting the most common class (0.401).

## 2. The reported accuracies imply an augmented test set

A 20% hold-out of 410 images has ~82 images. On 81–83 test images, the achievable accuracies between 0.92 and
0.945 are 0.926–0.928 and 0.938–0.940; nothing in between is possible. The reported **0.936** (Table 2) and the fold accuracies **0.933,
0.931, 0.934** (Table 4) cannot be produced on 81–83 original images. They are only possible on the much larger
augmented test sets. This supports §1 for both the hold-out and the 5-fold cross-validation.

## 3. Image-level rather than patient-level split

INbreast has 115 patients and 410 images: typically CC and MLO views of both breasts. Stratifying by image
(Sec. V-A) places different views of the same breast in train and test, even without augmentation.

## 4. Internal inconsistencies

| # | Where | Observation |
|---|---|---|
| 4.1 | Table 1 | The rows sum to **7,328**, not the stated 7,208. Training sums to 4,321 (stated 4,614), validation to 1,438 (stated 1,154). Each row is split 60/20/20 with validation = test, not the 64/16/20 stated in the text. |
| 4.2 | Table 3 | 🧪 Brier = 0.010 together with ECE = 0.372 is impossible. By Jensen's inequality, **Brier ≥ ECE²** ≈ 0.138 (proved and property-tested in [`tests/test_metrics.py`](tests/test_metrics.py)). |
| 4.3 | Fig. 3 vs Table 2 | ROC legend AUCs differ from the table: ResNet50 0.916 vs 0.947, DenseNet121 0.925 vs 0.944, EfficientNet-B0 0.918 vs 0.937. |
| 4.4 | Sec. IV-B vs Fig. 2 / Sec. VI-A | Two different architectures are described. The first is CBAM (channel + 7×7 spatial attention) with Dense 512→256→128 heads. The second is spatial-only attention (Conv 64→32→1) with Dense 256→128→64 heads. Fig. 2 shows EfficientNet-B3 at 224×224 giving 7×7×**960** features, but B3 outputs 1,536 channels (960 is MobileNetV3-Large), and Sec. V-A states B3 was run at 300×300. |
| 4.5 | Secs. III-A, III-B, V-A | Three different augmentation recipes are given (±15° rotation/±10% scale/±15% brightness; CLAHE + 11 rotations + H/V flips; H-flip + ±10% brightness + 0.9–1.1 contrast). |
| 4.6 | Fig. 4 | Two of the four "correct attention" examples are misclassified: Case B is benign with P(malignant)=0.82, Case C is malignant with P=0.35. |
| 4.7 | Sec. VI-C | Attention statistics (Gini, entropy, p-values) are reported without a method, a sample size, or radiologist ground truth. 🧪 Tested in Phase 5 against CBIS-DDSM ROI masks: the CBAM attention localises lesions barely better than a brightest-tissue baseline, and none of the reported statistics is reproduced ([Phase 5](docs/RESULTS.md#phase-5-do-the-attention-maps-point-at-the-lesion)). |
| 4.8 | Sec. V-D | CUDA 12.5 is stated for an RTX 5090. Blackwell GPUs require CUDA 12.8 or newer. |
| 4.9 | Sec. IV-A | *"λ₁/λ₂ ≈ 2.5 ensures the model prioritizes minimizing false negatives"*: a loss weight between two tasks does not set the sensitivity/specificity trade-off. That is the decision threshold's job. |

## 5. What this does *not* claim

- It does not claim the attention architecture is ineffective. That needs a leakage-free comparison, which is
  Phase 3 of this project: with a patient-level split the design reaches AUC 0.78 on CBIS-DDSM, and CBAM adds a
  small +0.013 ([Phase 3](docs/RESULTS.md#phase-3-a-leakage-free-multi-task-model-cbis-ddsm)).
- It does not claim any intent. Augment-then-split is a common mistake in medical imaging; see e.g.
  Roberts et al., *Nature Machine Intelligence* 3, 199–217 (2021), and Kapoor & Narayanan, *Patterns* 4(9), 2023.
- No code or split files were released with the paper, so exact reproduction is impossible. §1–2 are the most
  likely explanation consistent with the text, the tables, and the cited dataset.
