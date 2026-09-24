# Model card: multi-task mammography model (`mt_cbam`, official-split checkpoint)

Format follows Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Every number links to a committed
result file. Headline numbers are collected in [`results/overview/summary.json`](results/overview/summary.json).

> ⚠️ **Not a medical device.** For research and education only. It must not be used for diagnosis, screening,
> triage or any decision about a patient.

## Model details

| | |
|---|---|
| Architecture | EfficientNet-B0 (ImageNet-pretrained) → CBAM (channel gate, then 7×7 spatial gate) on the last feature map → global average pool → shared MLP (512 → 256) → two heads: malignancy (1 logit) and BI-RADS density (4 logits, A–D). [`src/mammo/model.py`](src/mammo/model.py) |
| Design source | Esen et al., IEEE Access 13 (2025), Sec. IV-B, re-implemented here. Differences: B0 instead of B3, 640×384 input instead of 224×224 / 300×300 |
| Input | One full-field mammogram view (CC or MLO), grayscale. Preprocessing: Otsu breast crop → flip so the chest wall is on the left → aspect-preserving resize to 640×384 with zero padding ([`preprocess.py`](src/mammo/preprocess.py)) |
| Output | P(malignant) for the image's finding(s); probabilities of density A/B/C/D; optional Grad-CAM and CBAM maps |
| Training | Official CBIS-DDSM training split (2,434 images); 22 patients listed in both official splits kept in training only. 15 epochs, AdamW (lr 2e-4, weight decay 0.01, one-cycle schedule), batch 16, mixed precision, loss = 2.0·BCE(malignancy) + 0.8·CE(density, label smoothing 0.05), mammography-safe augmentation (vertical flip, ±10° rotation, zoom, shift, brightness/contrast). Seed 42, one run |
| Calibration | Optional Platt scaling of the malignancy logit, `sigmoid(0.625·z − 0.484)`, fitted on Phase 3 cross-validation predictions (not on the test split) |
| Weights | Produced by [`notebooks/04_external_inbreast.ipynb`](notebooks/04_external_inbreast.ipynb). A slim copy with calibration and provenance is exported by [`notebooks/06_demo_export.ipynb`](notebooks/06_demo_export.ipynb) and served by the demo |
| License | Code MIT. The weights were trained on CBIS-DDSM (CC BY 3.0); cite it when using them |

## Intended use

- **Intended:** studying multi-task learning, calibration, domain shift and explanation methods in mammography;
  teaching; reproducing this replication study.
- **Out of scope:** any clinical use. Screening populations (the training data contain *only* mammograms with a
  finding, so the model has never seen the typical normal screening case as a separate class). Digital breast
  tomosynthesis, ultrasound or MRI. Detecting or localising lesions (it classifies whole images; its maps are
  approximate explanations, not detections).

## Training and evaluation data

| Dataset | Role | Size | Labels |
|---|---|---|---|
| CBIS-DDSM (scanned film, USA, 1990s) | training + internal test | 2,802 images / 1,460 patients after merging duplicates; official test 368 / 212 | biopsy-confirmed pathology per finding; BI-RADS density |
| INbreast (full-field digital, Portugal) | external test only | 410 images / 108 patients | ACR density; BI-RADS assessment (no biopsy for most images) |

Class balance, CBIS-DDSM: 44.8% malignant; density A/B/C/D = 401/1,107/848/446 images.

## Performance

95% CIs come from a bootstrap over patients. The "majority" column always predicts the most common class.

| Evaluation | Metric | Value (95% CI) | Majority |
|---|---|---:|---:|
| CBIS-DDSM official test (this checkpoint) | malignancy AUC | **0.746** (0.681–0.805) | 0.500 |
| | accuracy / sensitivity / specificity at 0.5 (uncalibrated) | 0.622 / 0.761 / 0.521 | 0.579 |
| | density QWK | **0.711** (0.631–0.773) | 0.000 |
| | density accuracy | 0.606 | 0.454 |
| CBIS-DDSM 5-fold patient-level CV (same recipe, 5 models) | malignancy AUC | 0.782 (0.762–0.802) | 0.500 |
| | density QWK | 0.769 (0.746–0.791) | 0.000 |
| INbreast (external, never used for training or tuning) | density QWK | **0.660** (0.550–0.743); 93.4% within ±1 grade | 0.000 |
| | proxy AUC, BI-RADS 4–6 vs 1–3 | 0.822 (0.761–0.877) | 0.500 |

Sources: [`results/external/`](results/external/summary.md), [`results/cbis/`](results/cbis/summary.md),
[`results/overview/summary.json`](results/overview/summary.json).

**Subgroups (5-fold CV).** Images with masses: AUC 0.83. Images with only calcifications: AUC 0.72 (calcifications
are a few pixels across at 640×384). CC and MLO views: both 0.78.

**Calibration.**
- Uncalibrated, the malignancy output is over-confident and biased towards "malignant": on the official test split
  its mean prediction is 0.60 against a true rate of 0.42, and ECE is 0.206.
- The shipped Platt parameters reduce ECE to 0.090.
- On INbreast, probabilities shift again: normal mammograms get a mean score of 0.25.
- Always recalibrate on data from the target setting.

**Operating point.** On cross-validation, a threshold chosen to reach 90% sensitivity gave 89.8% sensitivity and
38.6% specificity on unseen patients ([`results/calibration/`](results/calibration/summary.md)).

**Explanations.** Grad-CAM of the malignancy output points within one 32-pixel cell of the radiologist's outline in
44% (38–50%) of official-test images. The CBAM spatial gate does so in 24% (19–29%), against 18% for a lesion-blind
"brightest tissue" map. Don't read the CBAM map as a lesion localiser
([`results/attention/`](results/attention/summary.md)).

## Limitations and risks

- **One training seed.** Differences smaller than about 0.02 AUC between this and related variants are not reliable.
- **Domain.** Trained on digitised film from one US collection. Performance on modern digital systems, other vendors,
  other populations and other breast densities is only partly known (INbreast: one site, 108 patients).
- **Task definition.** "Malignant" means the image's CBIS-DDSM finding was biopsy-proven malignant. The model has
  not learned to find cancer in an unselected screening exam.
- **Label noise.** CBIS-DDSM density and outlines come from one reading. Calcification ROIs are loose.
- **Preprocessing assumptions.** Some film-edge strips and scanner labels survive the crop. Images that are not
  single-view mammograms produce meaningless output, and the demo does not detect this.
- **Fairness.** CBIS-DDSM has no age, ethnicity or scanner metadata, so subgroup performance on these axes could not
  be measured.

## Ethical considerations

The reference paper's headline results came from a leaky evaluation, and this project exists to make that visible.
The same risk applies to any number in this card if the model is evaluated carelessly. Patient-level separation,
external data and uncertainty estimates are the minimum for trusting a medical-imaging result.

## Citation

If you use this code or model, please cite the datasets (CBIS-DDSM: Lee et al., *Sci. Data* 2017; INbreast: Moreira
et al., *Acad. Radiol.* 2012) and the reference paper (Esen et al., *IEEE Access* 2025), and link this repository.
