### Malignancy (patient-level 5-fold CV, pooled out-of-fold)

| Model | Calibration | ECE (95% CI) | Brier | NLL | Mean predicted P(malignant) |
|---|---|---:|---:|---:|---:|
| Multi-task + CBAM (paper design) | none | 0.109 (0.095–0.131) | 0.205 | 0.610 | 0.542 |
|  | temperature (T≈1.74) | 0.091 (0.072–0.112) | 0.197 | 0.576 | 0.539 |
|  | Platt | 0.028 (0.024–0.049) | 0.188 | 0.555 | 0.448 |
| Multi-task, no attention | none | 0.118 (0.100–0.138) | 0.210 | 0.629 | 0.533 |
|  | temperature (T≈1.86) | 0.085 (0.068–0.108) | 0.201 | 0.586 | 0.532 |
|  | Platt | 0.032 (0.027–0.054) | 0.194 | 0.569 | 0.448 |
| Malignancy only + CBAM | none | 0.108 (0.092–0.127) | 0.206 | 0.611 | 0.520 |
|  | temperature (T≈1.74) | 0.077 (0.059–0.099) | 0.197 | 0.575 | 0.520 |
|  | Platt | 0.031 (0.025–0.054) | 0.191 | 0.564 | 0.447 |

Observed share of malignant images: 0.448.

### Density (top-label calibration: when the model is X% sure of a grade, is it right X% of the time?)

| Model | Calibration | ECE (95% CI) | Brier (4-class) | NLL | Mean confidence | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Multi-task + CBAM (paper design) | none | 0.034 (0.024–0.055) | 0.449 | 0.761 | 0.652 | 0.668 |
|  | temperature (T≈0.88) | 0.030 (0.026–0.054) | 0.450 | 0.759 | 0.681 | 0.668 |
| Multi-task, no attention | none | 0.036 (0.027–0.057) | 0.467 | 0.790 | 0.656 | 0.653 |
|  | temperature (T≈0.94) | 0.036 (0.031–0.059) | 0.468 | 0.789 | 0.672 | 0.653 |
| Density only + CBAM | none | 0.056 (0.042–0.078) | 0.436 | 0.748 | 0.735 | 0.681 |
|  | temperature (T≈1.10) | 0.034 (0.027–0.057) | 0.433 | 0.746 | 0.709 | 0.681 |

### Operating points (paper design, Platt-calibrated, threshold chosen on the other folds)

| Threshold rule | Sensitivity | Specificity | PPV | NPV | Missed cancers | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 on raw probabilities | 0.752 | 0.657 | 0.640 | 0.766 | 311 | 530 |
| 0.5 on calibrated probabilities | 0.615 | 0.799 | 0.713 | 0.719 | 483 | 311 |
| aim for 90% sensitivity | 0.898 | 0.386 | 0.543 | 0.823 | 128 | 950 |
| aim for 95% sensitivity | 0.945 | 0.274 | 0.514 | 0.860 | 69 | 1123 |

### Does a calibration transfer to a newly trained model? (official test split, 368 images)

| Calibration (fitted on the CV predictions) | ECE | Brier | Mean predicted P(malignant) | Observed |
|---|---:|---:|---:|---:|
| uncalibrated | 0.206 | 0.246 | 0.601 | 0.421 |
| temperature | 0.164 | 0.226 | 0.580 | 0.421 |
| platt | 0.090 | 0.203 | 0.489 | 0.421 |

Sanity check against the paper: Brier ≥ ECE² holds for every model here (yes). The paper reports Brier 0.01 with ECE 0.365–0.419, which would need Brier ≥ 0.133.
