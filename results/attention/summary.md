Official CBIS-DDSM test split: 354 of 368 images have a usable ROI mask (206 patients). Subsets: all 354, mass only 316, calcification only 34, mass + calcification 4, benign 200, malignant 154, model correct 225, model wrong 129, lesion < 1 cell 144, lesion 1-4 cells 163, lesion > 4 cells 47.

### Localisation (inside the breast; 95% CI, patient bootstrap)

| Method | Pointing game (strict) | Pointing game (±1 cell) | Energy in lesion | Pixel AUC | IoU, top 10% |
|---|---:|---:|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.065 (0.04–0.09) | 0.237 (0.19–0.29) | 0.019 (0.02–0.02) | 0.694 (0.66–0.72) | 0.043 (0.03–0.05) |
| CBAM attention (st_path) | 0.085 (0.06–0.11) | 0.364 (0.31–0.42) | 0.020 (0.02–0.02) | 0.709 (0.68–0.74) | 0.058 (0.05–0.07) |
| CBAM attention (st_dens) | 0.048 (0.02–0.08) | 0.193 (0.15–0.24) | 0.019 (0.02–0.02) | 0.670 (0.64–0.70) | 0.040 (0.03–0.05) |
| Grad-CAM malignancy (mt_cbam) | 0.292 (0.23–0.35) | 0.441 (0.38–0.50) | 0.103 (0.09–0.12) | 0.722 (0.68–0.76) | 0.078 (0.06–0.09) |
| Grad-CAM malignancy (mt_plain) | 0.295 (0.24–0.35) | 0.418 (0.36–0.48) | 0.106 (0.09–0.12) | 0.707 (0.67–0.75) | 0.082 (0.07–0.10) |
| Grad-CAM malignancy (st_path) | 0.320 (0.27–0.37) | 0.475 (0.42–0.53) | 0.127 (0.11–0.15) | 0.736 (0.70–0.77) | 0.082 (0.07–0.10) |
| Grad-CAM density (mt_cbam) | 0.101 (0.07–0.13) | 0.186 (0.14–0.23) | 0.031 (0.02–0.04) | 0.534 (0.50–0.57) | 0.035 (0.03–0.05) |
| Grad-CAM density (mt_plain) | 0.096 (0.07–0.13) | 0.189 (0.15–0.23) | 0.038 (0.03–0.05) | 0.608 (0.57–0.64) | 0.043 (0.03–0.05) |
| Grad-CAM density (st_dens) | 0.042 (0.02–0.07) | 0.136 (0.10–0.18) | 0.026 (0.02–0.03) | 0.575 (0.54–0.61) | 0.028 (0.02–0.04) |
| Control: CBAM, ImageNet-only net | 0.003 (0.00–0.01) | 0.006 (0.00–0.02) | 0.016 (0.01–0.02) | 0.425 (0.40–0.45) | 0.006 (0.00–0.01) |
| Control: Grad-CAM, ImageNet-only net | 0.063 (0.04–0.09) | 0.150 (0.11–0.19) | 0.028 (0.02–0.03) | 0.612 (0.58–0.64) | 0.034 (0.02–0.04) |
| Ceiling: ROI mask at map resolution | 0.828 (0.78–0.87) | 1.000 (1.00–1.00) | 0.431 (0.41–0.46) | 0.998 (1.00–1.00) | 0.107 (0.08–0.14) |
| Baseline: brightest tissue | 0.139 (0.10–0.18) | 0.182 (0.14–0.22) | 0.022 (0.02–0.03) | 0.666 (0.64–0.70) | 0.051 (0.04–0.06) |
| Baseline: breast centre | 0.034 (0.02–0.05) | 0.141 (0.10–0.18) | 0.022 (0.02–0.03) | 0.622 (0.59–0.66) | 0.028 (0.02–0.04) |
| Baseline: random | 0.021 (0.01–0.03) | 0.075 (0.06–0.09) | 0.018 (0.01–0.02) | 0.505 (0.49–0.52) | 0.013 (0.01–0.02) |
| Baseline: uniform | 0.018 (0.01–0.02) | 0.076 (0.07–0.08) | 0.018 (0.01–0.02) | 0.500 (0.50–0.50) | 0.018 (0.01–0.02) |

The uniform baseline is chance level: a map with no information scores the lesion's share of the breast on strict pointing and energy (here 0.018, 0.014–0.022), the share of the breast within one cell of the lesion on ±1-cell pointing, and 0.5 AUC. The ceiling row is the ROI mask itself pooled to the map grid: the best any map of this resolution could score.

#### mass only

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.247 (0.19–0.30) | 0.016 (0.01–0.02) | 0.707 (0.68–0.74) |
| CBAM attention (st_path) | 0.372 (0.31–0.43) | 0.017 (0.01–0.02) | 0.727 (0.69–0.76) |
| CBAM attention (st_dens) | 0.200 (0.16–0.25) | 0.015 (0.01–0.02) | 0.677 (0.64–0.71) |
| Grad-CAM malignancy (mt_cbam) | 0.468 (0.40–0.54) | 0.099 (0.08–0.12) | 0.727 (0.69–0.77) |
| Grad-CAM malignancy (mt_plain) | 0.424 (0.36–0.49) | 0.101 (0.08–0.12) | 0.712 (0.67–0.75) |
| Grad-CAM malignancy (st_path) | 0.484 (0.43–0.54) | 0.122 (0.10–0.14) | 0.746 (0.71–0.78) |
| Grad-CAM density (mt_cbam) | 0.184 (0.14–0.23) | 0.026 (0.02–0.03) | 0.522 (0.48–0.56) |
| Grad-CAM density (mt_plain) | 0.190 (0.15–0.24) | 0.033 (0.03–0.04) | 0.607 (0.57–0.64) |
| Grad-CAM density (st_dens) | 0.130 (0.09–0.17) | 0.024 (0.02–0.03) | 0.580 (0.55–0.61) |
| Control: CBAM, ImageNet-only net | 0.003 (0.00–0.01) | 0.012 (0.01–0.01) | 0.423 (0.39–0.45) |
| Control: Grad-CAM, ImageNet-only net | 0.146 (0.10–0.19) | 0.025 (0.02–0.03) | 0.615 (0.58–0.65) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.422 (0.40–0.45) | 0.999 (1.00–1.00) |
| Baseline: brightest tissue | 0.179 (0.14–0.22) | 0.018 (0.01–0.02) | 0.668 (0.63–0.70) |
| Baseline: breast centre | 0.120 (0.09–0.16) | 0.018 (0.01–0.02) | 0.620 (0.58–0.66) |
| Baseline: random | 0.062 (0.05–0.08) | 0.014 (0.01–0.02) | 0.507 (0.49–0.52) |
| Baseline: uniform | 0.070 (0.07–0.08) | 0.014 (0.01–0.02) | 0.500 (0.50–0.50) |

#### calcification only

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.147 (0.03–0.29) | 0.046 (0.02–0.08) | 0.566 (0.48–0.65) |
| CBAM attention (st_path) | 0.265 (0.11–0.43) | 0.046 (0.02–0.08) | 0.540 (0.45–0.63) |
| CBAM attention (st_dens) | 0.088 (0.00–0.18) | 0.045 (0.02–0.08) | 0.581 (0.49–0.67) |
| Grad-CAM malignancy (mt_cbam) | 0.176 (0.06–0.30) | 0.129 (0.06–0.21) | 0.689 (0.59–0.78) |
| Grad-CAM malignancy (mt_plain) | 0.324 (0.15–0.50) | 0.138 (0.06–0.22) | 0.660 (0.54–0.78) |
| Grad-CAM malignancy (st_path) | 0.353 (0.19–0.52) | 0.161 (0.07–0.26) | 0.648 (0.56–0.73) |
| Grad-CAM density (mt_cbam) | 0.176 (0.06–0.31) | 0.071 (0.03–0.13) | 0.618 (0.52–0.72) |
| Grad-CAM density (mt_plain) | 0.176 (0.06–0.29) | 0.068 (0.03–0.11) | 0.608 (0.51–0.70) |
| Grad-CAM density (st_dens) | 0.118 (0.03–0.23) | 0.032 (0.01–0.05) | 0.489 (0.40–0.59) |
| Control: CBAM, ImageNet-only net | 0.037 (0.00–0.11) | 0.039 (0.02–0.06) | 0.444 (0.37–0.52) |
| Control: Grad-CAM, ImageNet-only net | 0.176 (0.06–0.32) | 0.041 (0.02–0.07) | 0.584 (0.49–0.68) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.486 (0.38–0.59) | 0.998 (1.00–1.00) |
| Baseline: brightest tissue | 0.176 (0.04–0.31) | 0.052 (0.02–0.08) | 0.638 (0.56–0.70) |
| Baseline: breast centre | 0.265 (0.12–0.43) | 0.054 (0.02–0.09) | 0.627 (0.52–0.73) |
| Baseline: random | 0.174 (0.11–0.23) | 0.045 (0.02–0.07) | 0.487 (0.45–0.52) |
| Baseline: uniform | 0.115 (0.08–0.16) | 0.044 (0.02–0.07) | 0.500 (0.50–0.50) |

#### benign

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.160 (0.10–0.22) | 0.015 (0.01–0.02) | 0.673 (0.63–0.71) |
| CBAM attention (st_path) | 0.229 (0.17–0.29) | 0.016 (0.01–0.02) | 0.653 (0.61–0.70) |
| CBAM attention (st_dens) | 0.155 (0.10–0.21) | 0.015 (0.01–0.02) | 0.662 (0.62–0.70) |
| Grad-CAM malignancy (mt_cbam) | 0.275 (0.21–0.35) | 0.054 (0.04–0.07) | 0.609 (0.56–0.66) |
| Grad-CAM malignancy (mt_plain) | 0.250 (0.19–0.32) | 0.052 (0.04–0.07) | 0.595 (0.55–0.64) |
| Grad-CAM malignancy (st_path) | 0.300 (0.24–0.37) | 0.071 (0.05–0.09) | 0.635 (0.59–0.68) |
| Grad-CAM density (mt_cbam) | 0.140 (0.09–0.19) | 0.023 (0.01–0.03) | 0.527 (0.47–0.58) |
| Grad-CAM density (mt_plain) | 0.155 (0.10–0.21) | 0.031 (0.02–0.04) | 0.616 (0.57–0.66) |
| Grad-CAM density (st_dens) | 0.110 (0.07–0.15) | 0.020 (0.01–0.03) | 0.561 (0.52–0.60) |
| Control: CBAM, ImageNet-only net | 0.001 (0.00–0.00) | 0.012 (0.01–0.02) | 0.420 (0.38–0.46) |
| Control: Grad-CAM, ImageNet-only net | 0.120 (0.07–0.17) | 0.019 (0.01–0.02) | 0.584 (0.54–0.62) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.408 (0.38–0.44) | 0.998 (1.00–1.00) |
| Baseline: brightest tissue | 0.105 (0.06–0.15) | 0.017 (0.01–0.02) | 0.634 (0.59–0.68) |
| Baseline: breast centre | 0.125 (0.08–0.18) | 0.018 (0.01–0.02) | 0.615 (0.57–0.67) |
| Baseline: random | 0.078 (0.06–0.09) | 0.014 (0.01–0.02) | 0.511 (0.50–0.53) |
| Baseline: uniform | 0.069 (0.06–0.08) | 0.014 (0.01–0.02) | 0.500 (0.50–0.50) |

#### malignant

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.338 (0.25–0.42) | 0.025 (0.02–0.03) | 0.722 (0.68–0.76) |
| CBAM attention (st_path) | 0.538 (0.45–0.62) | 0.027 (0.02–0.03) | 0.783 (0.74–0.82) |
| CBAM attention (st_dens) | 0.242 (0.17–0.31) | 0.024 (0.02–0.03) | 0.680 (0.63–0.73) |
| Grad-CAM malignancy (mt_cbam) | 0.656 (0.57–0.74) | 0.166 (0.13–0.20) | 0.868 (0.82–0.91) |
| Grad-CAM malignancy (mt_plain) | 0.636 (0.55–0.72) | 0.176 (0.14–0.21) | 0.852 (0.80–0.89) |
| Grad-CAM malignancy (st_path) | 0.701 (0.62–0.78) | 0.200 (0.16–0.24) | 0.867 (0.83–0.90) |
| Grad-CAM density (mt_cbam) | 0.247 (0.18–0.32) | 0.041 (0.03–0.06) | 0.542 (0.49–0.60) |
| Grad-CAM density (mt_plain) | 0.234 (0.16–0.31) | 0.048 (0.03–0.06) | 0.598 (0.55–0.65) |
| Grad-CAM density (st_dens) | 0.169 (0.10–0.24) | 0.034 (0.02–0.05) | 0.592 (0.54–0.64) |
| Control: CBAM, ImageNet-only net | 0.013 (0.00–0.03) | 0.020 (0.01–0.03) | 0.431 (0.39–0.47) |
| Control: Grad-CAM, ImageNet-only net | 0.188 (0.12–0.26) | 0.039 (0.03–0.05) | 0.648 (0.60–0.69) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.461 (0.42–0.50) | 0.998 (1.00–1.00) |
| Baseline: brightest tissue | 0.282 (0.21–0.35) | 0.029 (0.02–0.04) | 0.707 (0.67–0.75) |
| Baseline: breast centre | 0.162 (0.10–0.23) | 0.027 (0.02–0.04) | 0.632 (0.59–0.68) |
| Baseline: random | 0.072 (0.05–0.10) | 0.023 (0.02–0.03) | 0.497 (0.48–0.51) |
| Baseline: uniform | 0.085 (0.07–0.10) | 0.023 (0.02–0.03) | 0.500 (0.50–0.50) |

#### model correct

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.280 (0.21–0.34) | 0.022 (0.02–0.03) | 0.695 (0.66–0.73) |
| CBAM attention (st_path) | 0.399 (0.33–0.47) | 0.020 (0.02–0.02) | 0.726 (0.69–0.76) |
| CBAM attention (st_dens) | 0.188 (0.14–0.24) | 0.018 (0.01–0.02) | 0.677 (0.64–0.71) |
| Grad-CAM malignancy (mt_cbam) | 0.480 (0.40–0.56) | 0.118 (0.09–0.14) | 0.739 (0.69–0.78) |
| Grad-CAM malignancy (mt_plain) | 0.436 (0.36–0.51) | 0.120 (0.09–0.14) | 0.714 (0.67–0.76) |
| Grad-CAM malignancy (st_path) | 0.515 (0.44–0.58) | 0.135 (0.11–0.16) | 0.753 (0.71–0.80) |
| Grad-CAM density (mt_cbam) | 0.167 (0.11–0.22) | 0.029 (0.02–0.04) | 0.515 (0.47–0.56) |
| Grad-CAM density (mt_plain) | 0.169 (0.12–0.22) | 0.032 (0.02–0.04) | 0.588 (0.54–0.63) |
| Grad-CAM density (st_dens) | 0.142 (0.09–0.19) | 0.024 (0.02–0.03) | 0.587 (0.55–0.62) |
| Control: CBAM, ImageNet-only net | 0.010 (0.00–0.02) | 0.017 (0.01–0.02) | 0.414 (0.38–0.45) |
| Control: Grad-CAM, ImageNet-only net | 0.173 (0.12–0.23) | 0.032 (0.02–0.04) | 0.621 (0.58–0.66) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.448 (0.42–0.48) | 0.999 (1.00–1.00) |
| Baseline: brightest tissue | 0.220 (0.17–0.27) | 0.024 (0.02–0.03) | 0.685 (0.65–0.72) |
| Baseline: breast centre | 0.138 (0.09–0.18) | 0.023 (0.02–0.03) | 0.618 (0.58–0.65) |
| Baseline: random | 0.077 (0.06–0.10) | 0.020 (0.02–0.02) | 0.502 (0.49–0.52) |
| Baseline: uniform | 0.080 (0.07–0.09) | 0.019 (0.02–0.02) | 0.500 (0.50–0.50) |

#### model wrong

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.163 (0.10–0.23) | 0.016 (0.01–0.02) | 0.692 (0.64–0.74) |
| CBAM attention (st_path) | 0.293 (0.21–0.39) | 0.021 (0.01–0.03) | 0.676 (0.62–0.73) |
| CBAM attention (st_dens) | 0.202 (0.14–0.28) | 0.021 (0.01–0.03) | 0.654 (0.60–0.71) |
| Grad-CAM malignancy (mt_cbam) | 0.372 (0.29–0.46) | 0.076 (0.06–0.10) | 0.692 (0.63–0.75) |
| Grad-CAM malignancy (mt_plain) | 0.383 (0.30–0.47) | 0.080 (0.06–0.10) | 0.693 (0.63–0.75) |
| Grad-CAM malignancy (st_path) | 0.395 (0.30–0.49) | 0.111 (0.08–0.14) | 0.703 (0.64–0.76) |
| Grad-CAM density (mt_cbam) | 0.217 (0.15–0.29) | 0.034 (0.02–0.05) | 0.563 (0.51–0.62) |
| Grad-CAM density (mt_plain) | 0.225 (0.15–0.30) | 0.050 (0.03–0.07) | 0.643 (0.58–0.70) |
| Grad-CAM density (st_dens) | 0.123 (0.07–0.19) | 0.031 (0.02–0.05) | 0.547 (0.49–0.60) |
| Control: CBAM, ImageNet-only net | 0.000 (0.00–0.00) | 0.013 (0.01–0.02) | 0.445 (0.40–0.49) |
| Control: Grad-CAM, ImageNet-only net | 0.109 (0.06–0.16) | 0.019 (0.01–0.02) | 0.596 (0.54–0.65) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.403 (0.37–0.44) | 0.998 (1.00–1.00) |
| Baseline: brightest tissue | 0.116 (0.06–0.18) | 0.017 (0.01–0.02) | 0.633 (0.58–0.69) |
| Baseline: breast centre | 0.147 (0.09–0.21) | 0.020 (0.01–0.03) | 0.631 (0.58–0.68) |
| Baseline: random | 0.072 (0.05–0.09) | 0.015 (0.01–0.02) | 0.510 (0.49–0.53) |
| Baseline: uniform | 0.070 (0.06–0.08) | 0.015 (0.01–0.02) | 0.500 (0.50–0.50) |

#### lesion < 1 cell

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.090 (0.04–0.15) | 0.005 (0.00–0.01) | 0.674 (0.62–0.72) |
| CBAM attention (st_path) | 0.188 (0.12–0.25) | 0.005 (0.00–0.01) | 0.667 (0.61–0.71) |
| CBAM attention (st_dens) | 0.111 (0.06–0.17) | 0.005 (0.00–0.00) | 0.649 (0.60–0.70) |
| Grad-CAM malignancy (mt_cbam) | 0.285 (0.21–0.37) | 0.025 (0.02–0.03) | 0.632 (0.57–0.69) |
| Grad-CAM malignancy (mt_plain) | 0.257 (0.18–0.34) | 0.026 (0.02–0.03) | 0.624 (0.56–0.69) |
| Grad-CAM malignancy (st_path) | 0.306 (0.23–0.38) | 0.032 (0.02–0.04) | 0.646 (0.59–0.70) |
| Grad-CAM density (mt_cbam) | 0.139 (0.08–0.21) | 0.007 (0.00–0.01) | 0.520 (0.46–0.58) |
| Grad-CAM density (mt_plain) | 0.097 (0.05–0.15) | 0.009 (0.01–0.01) | 0.569 (0.52–0.62) |
| Grad-CAM density (st_dens) | 0.111 (0.06–0.17) | 0.006 (0.00–0.01) | 0.545 (0.50–0.59) |
| Control: CBAM, ImageNet-only net | 0.002 (0.00–0.01) | 0.004 (0.00–0.00) | 0.423 (0.38–0.47) |
| Control: Grad-CAM, ImageNet-only net | 0.104 (0.06–0.16) | 0.006 (0.00–0.01) | 0.578 (0.53–0.63) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.257 (0.24–0.28) | 0.998 (1.00–1.00) |
| Baseline: brightest tissue | 0.035 (0.01–0.07) | 0.005 (0.00–0.01) | 0.606 (0.56–0.65) |
| Baseline: breast centre | 0.076 (0.04–0.12) | 0.005 (0.00–0.01) | 0.623 (0.58–0.67) |
| Baseline: random | 0.041 (0.02–0.06) | 0.004 (0.00–0.00) | 0.496 (0.48–0.51) |
| Baseline: uniform | 0.046 (0.04–0.05) | 0.004 (0.00–0.00) | 0.500 (0.50–0.50) |

#### lesion 1-4 cells

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.294 (0.22–0.37) | 0.016 (0.02–0.02) | 0.699 (0.66–0.74) |
| CBAM attention (st_path) | 0.451 (0.37–0.54) | 0.017 (0.02–0.02) | 0.733 (0.69–0.78) |
| CBAM attention (st_dens) | 0.197 (0.14–0.27) | 0.015 (0.01–0.02) | 0.666 (0.62–0.71) |
| Grad-CAM malignancy (mt_cbam) | 0.564 (0.47–0.66) | 0.123 (0.10–0.14) | 0.791 (0.74–0.84) |
| Grad-CAM malignancy (mt_plain) | 0.509 (0.42–0.59) | 0.121 (0.10–0.14) | 0.763 (0.71–0.81) |
| Grad-CAM malignancy (st_path) | 0.552 (0.46–0.64) | 0.142 (0.12–0.17) | 0.799 (0.75–0.85) |
| Grad-CAM density (mt_cbam) | 0.166 (0.10–0.23) | 0.024 (0.02–0.03) | 0.534 (0.48–0.59) |
| Grad-CAM density (mt_plain) | 0.233 (0.16–0.31) | 0.037 (0.03–0.04) | 0.641 (0.59–0.69) |
| Grad-CAM density (st_dens) | 0.135 (0.08–0.20) | 0.023 (0.02–0.03) | 0.595 (0.55–0.64) |
| Control: CBAM, ImageNet-only net | 0.006 (0.00–0.02) | 0.013 (0.01–0.01) | 0.418 (0.38–0.46) |
| Control: Grad-CAM, ImageNet-only net | 0.129 (0.07–0.19) | 0.025 (0.02–0.03) | 0.636 (0.59–0.68) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.497 (0.48–0.51) | 0.999 (1.00–1.00) |
| Baseline: brightest tissue | 0.187 (0.13–0.25) | 0.018 (0.02–0.02) | 0.689 (0.64–0.73) |
| Baseline: breast centre | 0.141 (0.09–0.19) | 0.020 (0.02–0.02) | 0.626 (0.57–0.68) |
| Baseline: random | 0.076 (0.06–0.09) | 0.015 (0.01–0.02) | 0.509 (0.49–0.53) |
| Baseline: uniform | 0.077 (0.07–0.08) | 0.015 (0.01–0.02) | 0.500 (0.50–0.50) |

#### lesion > 4 cells

| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |
|---|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.489 (0.32–0.63) | 0.075 (0.06–0.09) | 0.736 (0.68–0.79) |
| CBAM attention (st_path) | 0.596 (0.43–0.76) | 0.078 (0.06–0.10) | 0.756 (0.69–0.81) |
| CBAM attention (st_dens) | 0.426 (0.27–0.57) | 0.072 (0.06–0.09) | 0.747 (0.68–0.81) |
| Grad-CAM malignancy (mt_cbam) | 0.489 (0.31–0.66) | 0.270 (0.20–0.34) | 0.756 (0.65–0.85) |
| Grad-CAM malignancy (mt_plain) | 0.596 (0.43–0.75) | 0.300 (0.23–0.37) | 0.765 (0.67–0.85) |
| Grad-CAM malignancy (st_path) | 0.723 (0.59–0.85) | 0.367 (0.30–0.44) | 0.794 (0.73–0.86) |
| Grad-CAM density (mt_cbam) | 0.404 (0.29–0.52) | 0.129 (0.09–0.17) | 0.572 (0.49–0.64) |
| Grad-CAM density (mt_plain) | 0.319 (0.19–0.44) | 0.135 (0.10–0.17) | 0.613 (0.52–0.69) |
| Grad-CAM density (st_dens) | 0.213 (0.09–0.35) | 0.099 (0.07–0.13) | 0.595 (0.51–0.68) |
| Control: CBAM, ImageNet-only net | 0.021 (0.00–0.07) | 0.062 (0.05–0.08) | 0.455 (0.40–0.52) |
| Control: Grad-CAM, ImageNet-only net | 0.362 (0.20–0.53) | 0.102 (0.08–0.13) | 0.632 (0.55–0.71) |
| Ceiling: ROI mask at map resolution | 1.000 (1.00–1.00) | 0.738 (0.71–0.76) | 0.999 (1.00–1.00) |
| Baseline: brightest tissue | 0.617 (0.48–0.75) | 0.088 (0.07–0.11) | 0.771 (0.71–0.83) |
| Baseline: breast centre | 0.340 (0.20–0.51) | 0.083 (0.06–0.11) | 0.608 (0.53–0.69) |
| Baseline: random | 0.181 (0.13–0.24) | 0.070 (0.06–0.09) | 0.518 (0.50–0.54) |
| Baseline: uniform | 0.167 (0.14–0.19) | 0.069 (0.05–0.09) | 0.500 (0.50–0.50) |

### Paired comparisons (A − B on the same images)

| A | B | Metric | Subset | A − B | 95% CI | P(A not better) |
|---|---|---|---|---:|---:|---:|
| CBAM attention (mt_cbam) | Baseline: uniform | Pointing game (±1 cell) | all | +0.161 | +0.111 to +0.207 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: uniform | Energy in lesion | all | +0.002 | +0.001 to +0.002 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: uniform | Pixel AUC | all | +0.194 | +0.163 to +0.223 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.055 | +0.000 to +0.108 | 0.027 |
| CBAM attention (mt_cbam) | Baseline: brightest tissue | Energy in lesion | all | -0.002 | -0.003 to -0.001 | 1.000 |
| CBAM attention (mt_cbam) | Baseline: brightest tissue | Pixel AUC | all | +0.028 | -0.016 to +0.067 | 0.107 |
| CBAM attention (mt_cbam) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.096 | +0.032 to +0.155 | 0.002 |
| CBAM attention (mt_cbam) | Baseline: breast centre | Energy in lesion | all | -0.003 | -0.005 to -0.001 | 0.999 |
| CBAM attention (mt_cbam) | Baseline: breast centre | Pixel AUC | all | +0.071 | +0.027 to +0.113 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: random | Pointing game (±1 cell) | all | +0.162 | +0.111 to +0.209 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: random | Energy in lesion | all | +0.002 | +0.001 to +0.002 | 0.000 |
| CBAM attention (mt_cbam) | Baseline: random | Pixel AUC | all | +0.189 | +0.158 to +0.219 | 0.000 |
| CBAM attention (st_path) | Baseline: uniform | Pointing game (±1 cell) | all | +0.287 | +0.231 to +0.345 | 0.000 |
| CBAM attention (st_path) | Baseline: uniform | Energy in lesion | all | +0.003 | +0.002 to +0.003 | 0.000 |
| CBAM attention (st_path) | Baseline: uniform | Pixel AUC | all | +0.209 | +0.177 to +0.240 | 0.000 |
| CBAM attention (st_path) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.181 | +0.129 to +0.233 | 0.000 |
| CBAM attention (st_path) | Baseline: brightest tissue | Energy in lesion | all | -0.001 | -0.002 to -0.001 | 1.000 |
| CBAM attention (st_path) | Baseline: brightest tissue | Pixel AUC | all | +0.043 | +0.008 to +0.078 | 0.009 |
| CBAM attention (st_path) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.222 | +0.158 to +0.286 | 0.000 |
| CBAM attention (st_path) | Baseline: breast centre | Energy in lesion | all | -0.002 | -0.004 to -0.000 | 0.976 |
| CBAM attention (st_path) | Baseline: breast centre | Pixel AUC | all | +0.087 | +0.048 to +0.126 | 0.000 |
| CBAM attention (st_path) | Baseline: random | Pointing game (±1 cell) | all | +0.288 | +0.229 to +0.346 | 0.000 |
| CBAM attention (st_path) | Baseline: random | Energy in lesion | all | +0.002 | +0.002 to +0.003 | 0.000 |
| CBAM attention (st_path) | Baseline: random | Pixel AUC | all | +0.205 | +0.172 to +0.236 | 0.000 |
| CBAM attention (st_dens) | Baseline: uniform | Pointing game (±1 cell) | all | +0.117 | +0.076 to +0.160 | 0.000 |
| CBAM attention (st_dens) | Baseline: uniform | Energy in lesion | all | +0.001 | +0.001 to +0.001 | 0.000 |
| CBAM attention (st_dens) | Baseline: uniform | Pixel AUC | all | +0.170 | +0.136 to +0.201 | 0.000 |
| CBAM attention (st_dens) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.010 | -0.038 to +0.057 | 0.372 |
| CBAM attention (st_dens) | Baseline: brightest tissue | Energy in lesion | all | -0.003 | -0.004 to -0.002 | 1.000 |
| CBAM attention (st_dens) | Baseline: brightest tissue | Pixel AUC | all | +0.004 | -0.038 to +0.041 | 0.456 |
| CBAM attention (st_dens) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.051 | -0.000 to +0.106 | 0.028 |
| CBAM attention (st_dens) | Baseline: breast centre | Energy in lesion | all | -0.004 | -0.006 to -0.002 | 1.000 |
| CBAM attention (st_dens) | Baseline: breast centre | Pixel AUC | all | +0.047 | +0.004 to +0.087 | 0.017 |
| CBAM attention (st_dens) | Baseline: random | Pointing game (±1 cell) | all | +0.117 | +0.074 to +0.162 | 0.000 |
| CBAM attention (st_dens) | Baseline: random | Energy in lesion | all | +0.001 | +0.000 to +0.001 | 0.000 |
| CBAM attention (st_dens) | Baseline: random | Pixel AUC | all | +0.165 | +0.134 to +0.196 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: uniform | Pointing game (±1 cell) | all | +0.365 | +0.304 to +0.427 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: uniform | Energy in lesion | all | +0.085 | +0.068 to +0.102 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: uniform | Pixel AUC | all | +0.222 | +0.183 to +0.259 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.258 | +0.195 to +0.320 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: brightest tissue | Energy in lesion | all | +0.081 | +0.065 to +0.097 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: brightest tissue | Pixel AUC | all | +0.056 | +0.011 to +0.095 | 0.010 |
| Grad-CAM malignancy (mt_cbam) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.299 | +0.231 to +0.369 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: breast centre | Energy in lesion | all | +0.081 | +0.064 to +0.097 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: breast centre | Pixel AUC | all | +0.099 | +0.049 to +0.147 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: random | Pointing game (±1 cell) | all | +0.365 | +0.302 to +0.431 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: random | Energy in lesion | all | +0.085 | +0.068 to +0.102 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Baseline: random | Pixel AUC | all | +0.217 | +0.175 to +0.259 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: uniform | Pointing game (±1 cell) | all | +0.342 | +0.283 to +0.400 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: uniform | Energy in lesion | all | +0.088 | +0.071 to +0.105 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: uniform | Pixel AUC | all | +0.207 | +0.168 to +0.245 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.236 | +0.177 to +0.294 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: brightest tissue | Energy in lesion | all | +0.084 | +0.068 to +0.101 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: brightest tissue | Pixel AUC | all | +0.041 | -0.004 to +0.081 | 0.039 |
| Grad-CAM malignancy (mt_plain) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.277 | +0.209 to +0.342 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: breast centre | Energy in lesion | all | +0.084 | +0.067 to +0.101 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: breast centre | Pixel AUC | all | +0.084 | +0.033 to +0.131 | 0.001 |
| Grad-CAM malignancy (mt_plain) | Baseline: random | Pointing game (±1 cell) | all | +0.343 | +0.283 to +0.402 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: random | Energy in lesion | all | +0.088 | +0.071 to +0.105 | 0.000 |
| Grad-CAM malignancy (mt_plain) | Baseline: random | Pixel AUC | all | +0.202 | +0.159 to +0.243 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: uniform | Pointing game (±1 cell) | all | +0.398 | +0.344 to +0.451 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: uniform | Energy in lesion | all | +0.109 | +0.090 to +0.129 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: uniform | Pixel AUC | all | +0.236 | +0.198 to +0.271 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.292 | +0.235 to +0.348 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: brightest tissue | Energy in lesion | all | +0.105 | +0.087 to +0.125 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: brightest tissue | Pixel AUC | all | +0.070 | +0.029 to +0.109 | 0.001 |
| Grad-CAM malignancy (st_path) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.333 | +0.270 to +0.394 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: breast centre | Energy in lesion | all | +0.105 | +0.086 to +0.124 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: breast centre | Pixel AUC | all | +0.114 | +0.064 to +0.159 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: random | Pointing game (±1 cell) | all | +0.399 | +0.342 to +0.456 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: random | Energy in lesion | all | +0.109 | +0.090 to +0.129 | 0.000 |
| Grad-CAM malignancy (st_path) | Baseline: random | Pixel AUC | all | +0.231 | +0.192 to +0.269 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: uniform | Pointing game (±1 cell) | all | +0.110 | +0.071 to +0.152 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: uniform | Energy in lesion | all | +0.013 | +0.008 to +0.019 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: uniform | Pixel AUC | all | +0.034 | -0.004 to +0.071 | 0.043 |
| Grad-CAM density (mt_cbam) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.004 | -0.044 to +0.054 | 0.461 |
| Grad-CAM density (mt_cbam) | Baseline: brightest tissue | Energy in lesion | all | +0.009 | +0.004 to +0.015 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: brightest tissue | Pixel AUC | all | -0.133 | -0.178 to -0.091 | 1.000 |
| Grad-CAM density (mt_cbam) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.045 | -0.006 to +0.097 | 0.048 |
| Grad-CAM density (mt_cbam) | Baseline: breast centre | Energy in lesion | all | +0.009 | +0.004 to +0.015 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: breast centre | Pixel AUC | all | -0.089 | -0.133 to -0.046 | 1.000 |
| Grad-CAM density (mt_cbam) | Baseline: random | Pointing game (±1 cell) | all | +0.111 | +0.068 to +0.156 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: random | Energy in lesion | all | +0.013 | +0.008 to +0.019 | 0.000 |
| Grad-CAM density (mt_cbam) | Baseline: random | Pixel AUC | all | +0.029 | -0.011 to +0.067 | 0.084 |
| Grad-CAM density (mt_plain) | Baseline: uniform | Pointing game (±1 cell) | all | +0.113 | +0.073 to +0.155 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: uniform | Energy in lesion | all | +0.021 | +0.015 to +0.027 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: uniform | Pixel AUC | all | +0.108 | +0.072 to +0.143 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: brightest tissue | Pointing game (±1 cell) | all | +0.007 | -0.042 to +0.053 | 0.430 |
| Grad-CAM density (mt_plain) | Baseline: brightest tissue | Energy in lesion | all | +0.017 | +0.011 to +0.022 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: brightest tissue | Pixel AUC | all | -0.058 | -0.103 to -0.017 | 0.995 |
| Grad-CAM density (mt_plain) | Baseline: breast centre | Pointing game (±1 cell) | all | +0.048 | -0.008 to +0.102 | 0.048 |
| Grad-CAM density (mt_plain) | Baseline: breast centre | Energy in lesion | all | +0.016 | +0.010 to +0.022 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: breast centre | Pixel AUC | all | -0.014 | -0.060 to +0.028 | 0.761 |
| Grad-CAM density (mt_plain) | Baseline: random | Pointing game (±1 cell) | all | +0.114 | +0.070 to +0.158 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: random | Energy in lesion | all | +0.020 | +0.014 to +0.027 | 0.000 |
| Grad-CAM density (mt_plain) | Baseline: random | Pixel AUC | all | +0.103 | +0.066 to +0.139 | 0.000 |
| Grad-CAM density (st_dens) | Baseline: uniform | Pointing game (±1 cell) | all | +0.059 | +0.021 to +0.099 | 0.001 |
| Grad-CAM density (st_dens) | Baseline: uniform | Energy in lesion | all | +0.008 | +0.004 to +0.013 | 0.000 |
| Grad-CAM density (st_dens) | Baseline: uniform | Pixel AUC | all | +0.075 | +0.044 to +0.105 | 0.000 |
| Grad-CAM density (st_dens) | Baseline: brightest tissue | Pointing game (±1 cell) | all | -0.047 | -0.097 to +0.001 | 0.970 |
| Grad-CAM density (st_dens) | Baseline: brightest tissue | Energy in lesion | all | +0.004 | +0.000 to +0.009 | 0.018 |
| Grad-CAM density (st_dens) | Baseline: brightest tissue | Pixel AUC | all | -0.092 | -0.132 to -0.054 | 1.000 |
| Grad-CAM density (st_dens) | Baseline: breast centre | Pointing game (±1 cell) | all | -0.006 | -0.058 to +0.044 | 0.614 |
| Grad-CAM density (st_dens) | Baseline: breast centre | Energy in lesion | all | +0.004 | -0.001 to +0.009 | 0.050 |
| Grad-CAM density (st_dens) | Baseline: breast centre | Pixel AUC | all | -0.048 | -0.090 to -0.009 | 0.989 |
| Grad-CAM density (st_dens) | Baseline: random | Pointing game (±1 cell) | all | +0.060 | +0.021 to +0.099 | 0.001 |
| Grad-CAM density (st_dens) | Baseline: random | Energy in lesion | all | +0.008 | +0.004 to +0.013 | 0.000 |
| Grad-CAM density (st_dens) | Baseline: random | Pixel AUC | all | +0.070 | +0.037 to +0.102 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: uniform | Pointing game (±1 cell) | all | +0.074 | +0.033 to +0.117 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: uniform | Energy in lesion | all | +0.010 | +0.005 to +0.015 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: uniform | Pixel AUC | all | +0.112 | +0.081 to +0.143 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: brightest tissue | Pointing game (±1 cell) | all | -0.032 | -0.088 to +0.022 | 0.892 |
| Control: Grad-CAM, ImageNet-only net | Baseline: brightest tissue | Energy in lesion | all | +0.006 | +0.001 to +0.011 | 0.013 |
| Control: Grad-CAM, ImageNet-only net | Baseline: brightest tissue | Pixel AUC | all | -0.054 | -0.095 to -0.014 | 0.996 |
| Control: Grad-CAM, ImageNet-only net | Baseline: breast centre | Pointing game (±1 cell) | all | +0.008 | -0.048 to +0.065 | 0.414 |
| Control: Grad-CAM, ImageNet-only net | Baseline: breast centre | Energy in lesion | all | +0.005 | -0.000 to +0.011 | 0.030 |
| Control: Grad-CAM, ImageNet-only net | Baseline: breast centre | Pixel AUC | all | -0.011 | -0.056 to +0.034 | 0.665 |
| Control: Grad-CAM, ImageNet-only net | Baseline: random | Pointing game (±1 cell) | all | +0.074 | +0.032 to +0.117 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: random | Energy in lesion | all | +0.010 | +0.005 to +0.014 | 0.000 |
| Control: Grad-CAM, ImageNet-only net | Baseline: random | Pixel AUC | all | +0.107 | +0.075 to +0.142 | 0.000 |
| Control: CBAM, ImageNet-only net | Baseline: uniform | Pointing game (±1 cell) | all | -0.070 | -0.080 to -0.059 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: uniform | Energy in lesion | all | -0.002 | -0.003 to -0.001 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: uniform | Pixel AUC | all | -0.075 | -0.102 to -0.048 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: brightest tissue | Pointing game (±1 cell) | all | -0.176 | -0.218 to -0.138 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: brightest tissue | Energy in lesion | all | -0.006 | -0.008 to -0.005 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: brightest tissue | Pixel AUC | all | -0.241 | -0.286 to -0.195 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: breast centre | Pointing game (±1 cell) | all | -0.135 | -0.173 to -0.098 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: breast centre | Energy in lesion | all | -0.007 | -0.009 to -0.004 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: breast centre | Pixel AUC | all | -0.197 | -0.246 to -0.149 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: random | Pointing game (±1 cell) | all | -0.069 | -0.085 to -0.053 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: random | Energy in lesion | all | -0.002 | -0.003 to -0.002 | 1.000 |
| Control: CBAM, ImageNet-only net | Baseline: random | Pixel AUC | all | -0.080 | -0.109 to -0.049 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pointing game (±1 cell) | all | -0.203 | -0.268 to -0.141 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pointing game (±1 cell) | mass only | -0.222 | -0.293 to -0.152 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pointing game (±1 cell) | calcification only | -0.029 | -0.154 to +0.103 | 0.766 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Energy in lesion | all | -0.083 | -0.100 to -0.067 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Energy in lesion | mass only | -0.083 | -0.100 to -0.067 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Energy in lesion | calcification only | -0.083 | -0.140 to -0.029 | 1.000 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pixel AUC | all | -0.028 | -0.073 to +0.016 | 0.898 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pixel AUC | mass only | -0.020 | -0.070 to +0.028 | 0.806 |
| CBAM attention (mt_cbam) | Grad-CAM malignancy (mt_cbam) | Pixel AUC | calcification only | -0.124 | -0.248 to +0.011 | 0.963 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pointing game (±1 cell) | all | +0.045 | -0.006 to +0.097 | 0.053 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pointing game (±1 cell) | mass only | +0.047 | -0.007 to +0.099 | 0.051 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pointing game (±1 cell) | calcification only | +0.059 | -0.105 to +0.238 | 0.312 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Energy in lesion | all | +0.001 | +0.001 to +0.001 | 0.000 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Energy in lesion | mass only | +0.001 | +0.001 to +0.001 | 0.000 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Energy in lesion | calcification only | +0.001 | +0.000 to +0.002 | 0.009 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pixel AUC | all | +0.024 | +0.004 to +0.045 | 0.008 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pixel AUC | mass only | +0.030 | +0.010 to +0.053 | 0.003 |
| CBAM attention (mt_cbam) | CBAM attention (st_dens) | Pixel AUC | calcification only | -0.016 | -0.073 to +0.052 | 0.691 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pointing game (±1 cell) | all | +0.171 | +0.104 to +0.236 | 0.000 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pointing game (±1 cell) | mass only | +0.172 | +0.104 to +0.242 | 0.000 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pointing game (±1 cell) | calcification only | +0.176 | -0.026 to +0.382 | 0.061 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Energy in lesion | all | +0.002 | +0.001 to +0.002 | 0.000 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Energy in lesion | mass only | +0.002 | +0.001 to +0.002 | 0.000 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Energy in lesion | calcification only | +0.001 | -0.001 to +0.003 | 0.155 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pixel AUC | all | +0.040 | +0.008 to +0.072 | 0.006 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pixel AUC | mass only | +0.050 | +0.018 to +0.084 | 0.003 |
| CBAM attention (st_path) | CBAM attention (st_dens) | Pixel AUC | calcification only | -0.041 | -0.146 to +0.068 | 0.779 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pointing game (±1 cell) | all | +0.023 | -0.017 to +0.061 | 0.158 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pointing game (±1 cell) | mass only | +0.044 | +0.003 to +0.086 | 0.018 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pointing game (±1 cell) | calcification only | -0.147 | -0.300 to -0.028 | 1.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Energy in lesion | all | -0.003 | -0.011 to +0.003 | 0.827 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Energy in lesion | mass only | -0.002 | -0.010 to +0.005 | 0.711 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Energy in lesion | calcification only | -0.009 | -0.026 to +0.009 | 0.836 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pixel AUC | all | +0.015 | -0.008 to +0.040 | 0.099 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pixel AUC | mass only | +0.015 | -0.008 to +0.039 | 0.094 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM malignancy (mt_plain) | Pixel AUC | calcification only | +0.029 | -0.079 to +0.146 | 0.304 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pointing game (±1 cell) | all | +0.254 | +0.184 to +0.323 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pointing game (±1 cell) | mass only | +0.285 | +0.209 to +0.361 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pointing game (±1 cell) | calcification only | +0.000 | -0.146 to +0.147 | 0.590 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Energy in lesion | all | +0.072 | +0.055 to +0.088 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Energy in lesion | mass only | +0.073 | +0.056 to +0.091 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Energy in lesion | calcification only | +0.058 | +0.002 to +0.117 | 0.022 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pixel AUC | all | +0.188 | +0.134 to +0.241 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pixel AUC | mass only | +0.205 | +0.150 to +0.262 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Grad-CAM density (mt_cbam) | Pixel AUC | calcification only | +0.071 | -0.108 to +0.237 | 0.186 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pointing game (±1 cell) | all | +0.291 | +0.221 to +0.362 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pointing game (±1 cell) | mass only | +0.323 | +0.247 to +0.398 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pointing game (±1 cell) | calcification only | +0.000 | -0.167 to +0.156 | 0.588 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Energy in lesion | all | +0.075 | +0.060 to +0.090 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Energy in lesion | mass only | +0.074 | +0.058 to +0.089 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Energy in lesion | calcification only | +0.089 | +0.032 to +0.149 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pixel AUC | all | +0.110 | +0.068 to +0.152 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pixel AUC | mass only | +0.112 | +0.068 to +0.158 | 0.000 |
| Grad-CAM malignancy (mt_cbam) | Control: Grad-CAM, ImageNet-only net | Pixel AUC | calcification only | +0.106 | -0.017 to +0.216 | 0.047 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pointing game (±1 cell) | all | +0.231 | +0.181 to +0.278 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pointing game (±1 cell) | mass only | +0.244 | +0.190 to +0.294 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pointing game (±1 cell) | calcification only | +0.110 | -0.006 to +0.250 | 0.045 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Energy in lesion | all | +0.004 | +0.003 to +0.005 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Energy in lesion | mass only | +0.003 | +0.002 to +0.004 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Energy in lesion | calcification only | +0.008 | +0.002 to +0.015 | 0.001 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pixel AUC | all | +0.269 | +0.222 to +0.313 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pixel AUC | mass only | +0.284 | +0.234 to +0.332 | 0.000 |
| CBAM attention (mt_cbam) | Control: CBAM, ImageNet-only net | Pixel AUC | calcification only | +0.121 | -0.025 to +0.266 | 0.047 |

### The paper's attention statistics, recomputed

Definitions: Gini of the raw map values; Shannon entropy of the map normalised to sum 1, in nats (max = ln n); normalised entropy = entropy / ln n; area80 = smallest share of cells holding 80% of the map's mass; peak = max value. `full` = all 20×12 cells, `breast` = breast cells only, `p7` = map average-pooled to 7×7 (the paper's size).

| Map | Statistic | Paper (malignant vs benign) | Ours malignant | Ours benign | Difference (95% CI) | Cohen's d | Welch p |
|---|---|---|---:|---:|---:|---:|---:|
| CBAM attention (mt_cbam) | full gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.105 ± 0.020 | 0.097 ± 0.015 | +0.007 (+0.004 to +0.012) | +0.44 | 0.000118 |
| CBAM attention (mt_cbam) | full entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 5.453 ± 0.012 | 5.459 ± 0.007 | -0.006 (-0.009 to -0.004) | -0.68 | 9.62e-09 |
| CBAM attention (mt_cbam) | full entropy_norm | – | 0.995 ± 0.002 | 0.996 ± 0.001 | -0.001 (-0.002 to -0.001) | -0.68 | 9.62e-09 |
| CBAM attention (mt_cbam) | full area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.724 ± 0.016 | 0.730 ± 0.012 | -0.007 (-0.010 to -0.004) | -0.49 | 1.52e-05 |
| CBAM attention (mt_cbam) | full peak | – | 0.999 ± 0.002 | 0.999 ± 0.002 | +0.000 (+0.000 to +0.001) | +0.27 | 0.0103 |
| CBAM attention (mt_cbam) | p7 gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.088 ± 0.019 | 0.081 ± 0.014 | +0.007 (+0.003 to +0.010) | +0.40 | 0.000469 |
| CBAM attention (mt_cbam) | p7 entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.876 ± 0.008 | 3.879 ± 0.005 | -0.004 (-0.005 to -0.002) | -0.53 | 5.54e-06 |
| CBAM attention (mt_cbam) | p7 entropy_norm | – | 0.996 ± 0.002 | 0.997 ± 0.001 | -0.001 (-0.001 to -0.001) | -0.53 | 5.54e-06 |
| CBAM attention (mt_cbam) | p7 area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.748 ± 0.017 | 0.754 ± 0.011 | -0.006 (-0.009 to -0.003) | -0.43 | 0.000192 |
| CBAM attention (mt_cbam) | p7 peak | – | 0.994 ± 0.005 | 0.993 ± 0.005 | +0.001 (+0.000 to +0.002) | +0.25 | 0.0199 |
| CBAM attention (mt_cbam) | breast gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.109 ± 0.027 | 0.104 ± 0.018 | +0.005 (-0.001 to +0.011) | +0.21 | 0.0667 |
| CBAM attention (mt_cbam) | breast entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 4.914 ± 0.224 | 4.903 ± 0.208 | +0.011 (-0.047 to +0.063) | +0.05 | 0.644 |
| CBAM attention (mt_cbam) | breast entropy_norm | – | 0.993 ± 0.003 | 0.995 ± 0.002 | -0.001 (-0.002 to -0.001) | -0.43 | 0.000159 |
| CBAM attention (mt_cbam) | breast area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.720 ± 0.022 | 0.725 ± 0.015 | -0.004 (-0.009 to +0.000) | -0.24 | 0.0347 |
| CBAM attention (mt_cbam) | breast peak | – | 0.999 ± 0.002 | 0.998 ± 0.002 | +0.000 (-0.000 to +0.001) | +0.20 | 0.0723 |
| CBAM attention (st_path) | full gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.141 ± 0.026 | 0.126 ± 0.020 | +0.015 (+0.010 to +0.021) | +0.66 | 8.93e-09 |
| CBAM attention (st_path) | full entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 5.438 ± 0.017 | 5.448 ± 0.012 | -0.010 (-0.014 to -0.007) | -0.70 | 2.19e-09 |
| CBAM attention (st_path) | full entropy_norm | – | 0.992 ± 0.003 | 0.994 ± 0.002 | -0.002 (-0.002 to -0.001) | -0.70 | 2.19e-09 |
| CBAM attention (st_path) | full area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.708 ± 0.018 | 0.717 ± 0.015 | -0.008 (-0.012 to -0.004) | -0.50 | 7.9e-06 |
| CBAM attention (st_path) | full peak | – | 0.997 ± 0.005 | 0.995 ± 0.007 | +0.002 (+0.001 to +0.003) | +0.34 | 0.00123 |
| CBAM attention (st_path) | p7 gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.105 ± 0.020 | 0.098 ± 0.019 | +0.007 (+0.003 to +0.012) | +0.38 | 0.000478 |
| CBAM attention (st_path) | p7 entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.871 ± 0.008 | 3.875 ± 0.007 | -0.003 (-0.005 to -0.002) | -0.43 | 0.000117 |
| CBAM attention (st_path) | p7 entropy_norm | – | 0.995 ± 0.002 | 0.996 ± 0.002 | -0.001 (-0.001 to -0.000) | -0.43 | 0.000117 |
| CBAM attention (st_path) | p7 area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.744 ± 0.016 | 0.748 ± 0.016 | -0.004 (-0.007 to -0.001) | -0.25 | 0.0201 |
| CBAM attention (st_path) | p7 peak | – | 0.970 ± 0.023 | 0.971 ± 0.021 | -0.001 (-0.005 to +0.004) | -0.05 | 0.668 |
| CBAM attention (st_path) | breast gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.143 ± 0.039 | 0.123 ± 0.029 | +0.020 (+0.012 to +0.029) | +0.58 | 3.14e-07 |
| CBAM attention (st_path) | breast entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 4.895 ± 0.226 | 4.892 ± 0.210 | +0.003 (-0.055 to +0.056) | +0.01 | 0.9 |
| CBAM attention (st_path) | breast entropy_norm | – | 0.990 ± 0.006 | 0.993 ± 0.004 | -0.003 (-0.004 to -0.002) | -0.56 | 9.9e-07 |
| CBAM attention (st_path) | breast area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.703 ± 0.029 | 0.716 ± 0.024 | -0.013 (-0.019 to -0.006) | -0.48 | 1.71e-05 |
| CBAM attention (st_path) | breast peak | – | 0.997 ± 0.006 | 0.995 ± 0.007 | +0.002 (+0.001 to +0.004) | +0.35 | 0.000942 |
| CBAM attention (st_dens) | full gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.081 ± 0.023 | 0.079 ± 0.020 | +0.002 (-0.003 to +0.007) | +0.09 | 0.425 |
| CBAM attention (st_dens) | full entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 5.465 ± 0.008 | 5.467 ± 0.007 | -0.001 (-0.003 to +0.000) | -0.18 | 0.099 |
| CBAM attention (st_dens) | full entropy_norm | – | 0.997 ± 0.001 | 0.997 ± 0.001 | -0.000 (-0.001 to +0.000) | -0.18 | 0.099 |
| CBAM attention (st_dens) | full area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.742 ± 0.017 | 0.743 ± 0.015 | -0.001 (-0.005 to +0.002) | -0.09 | 0.411 |
| CBAM attention (st_dens) | full peak | – | 1.000 ± 0.001 | 1.000 ± 0.001 | +0.000 (-0.000 to +0.000) | +0.14 | 0.185 |
| CBAM attention (st_dens) | p7 gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.071 ± 0.023 | 0.069 ± 0.019 | +0.002 (-0.003 to +0.007) | +0.09 | 0.393 |
| CBAM attention (st_dens) | p7 entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.881 ± 0.007 | 3.882 ± 0.006 | -0.001 (-0.003 to +0.000) | -0.17 | 0.116 |
| CBAM attention (st_dens) | p7 entropy_norm | – | 0.997 ± 0.002 | 0.998 ± 0.001 | -0.000 (-0.001 to +0.000) | -0.17 | 0.116 |
| CBAM attention (st_dens) | p7 area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.760 ± 0.018 | 0.761 ± 0.016 | -0.002 (-0.006 to +0.002) | -0.10 | 0.361 |
| CBAM attention (st_dens) | p7 peak | – | 0.998 ± 0.003 | 0.998 ± 0.002 | +0.000 (-0.001 to +0.001) | +0.02 | 0.838 |
| CBAM attention (st_dens) | breast gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.052 ± 0.017 | 0.052 ± 0.014 | -0.000 (-0.004 to +0.003) | -0.00 | 0.981 |
| CBAM attention (st_dens) | breast entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 4.938 ± 0.220 | 4.921 ± 0.204 | +0.016 (-0.039 to +0.068) | +0.08 | 0.471 |
| CBAM attention (st_dens) | breast entropy_norm | – | 0.998 ± 0.001 | 0.998 ± 0.001 | -0.000 (-0.000 to +0.000) | -0.11 | 0.322 |
| CBAM attention (st_dens) | breast area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.764 ± 0.013 | 0.765 ± 0.011 | -0.000 (-0.003 to +0.003) | -0.03 | 0.796 |
| CBAM attention (st_dens) | breast peak | – | 1.000 ± 0.001 | 1.000 ± 0.001 | +0.000 (-0.000 to +0.000) | +0.14 | 0.185 |
| Grad-CAM malignancy (mt_cbam) | full gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.853 ± 0.062 | 0.865 ± 0.056 | -0.012 (-0.027 to +0.002) | -0.21 | 0.0531 |
| Grad-CAM malignancy (mt_cbam) | full entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.694 ± 0.467 | 3.642 ± 0.429 | +0.052 (-0.059 to +0.163) | +0.12 | 0.286 |
| Grad-CAM malignancy (mt_cbam) | full entropy_norm | – | 0.674 ± 0.085 | 0.665 ± 0.078 | +0.009 (-0.011 to +0.030) | +0.12 | 0.286 |
| Grad-CAM malignancy (mt_cbam) | full area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.129 ± 0.058 | 0.120 ± 0.051 | +0.009 (-0.004 to +0.023) | +0.17 | 0.113 |
| Grad-CAM malignancy (mt_cbam) | full peak | – | 0.253 ± 0.162 | 0.149 ± 0.112 | +0.104 (+0.071 to +0.137) | +0.76 | 7.16e-11 |
| Grad-CAM malignancy (mt_cbam) | p7 gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.733 ± 0.087 | 0.738 ± 0.081 | -0.005 (-0.024 to +0.015) | -0.06 | 0.607 |
| Grad-CAM malignancy (mt_cbam) | p7 entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 2.766 ± 0.355 | 2.768 ± 0.335 | -0.001 (-0.084 to +0.080) | -0.00 | 0.97 |
| Grad-CAM malignancy (mt_cbam) | p7 entropy_norm | – | 0.711 ± 0.091 | 0.711 ± 0.086 | -0.000 (-0.022 to +0.021) | -0.00 | 0.97 |
| Grad-CAM malignancy (mt_cbam) | p7 area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.238 ± 0.082 | 0.234 ± 0.074 | +0.004 (-0.015 to +0.022) | +0.05 | 0.639 |
| Grad-CAM malignancy (mt_cbam) | p7 peak | – | 0.094 ± 0.059 | 0.057 ± 0.042 | +0.037 (+0.025 to +0.049) | +0.73 | 3.22e-10 |
| Grad-CAM malignancy (mt_cbam) | breast gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.847 ± 0.077 | 0.864 ± 0.069 | -0.017 (-0.035 to +0.000) | -0.23 | 0.0341 |
| Grad-CAM malignancy (mt_cbam) | breast entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.161 ± 0.570 | 3.049 ± 0.620 | +0.113 (-0.032 to +0.257) | +0.19 | 0.0767 |
| Grad-CAM malignancy (mt_cbam) | breast entropy_norm | – | 0.639 ± 0.109 | 0.617 ± 0.116 | +0.022 (-0.005 to +0.049) | +0.19 | 0.0722 |
| Grad-CAM malignancy (mt_cbam) | breast area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.133 ± 0.073 | 0.121 ± 0.063 | +0.012 (-0.004 to +0.029) | +0.18 | 0.102 |
| Grad-CAM malignancy (mt_cbam) | breast peak | – | 0.245 ± 0.170 | 0.127 ± 0.123 | +0.118 (+0.083 to +0.153) | +0.81 | 4.41e-12 |
| Control: CBAM, ImageNet-only net | full gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.310 ± 0.058 | 0.297 ± 0.056 | +0.012 (-0.001 to +0.026) | +0.22 | 0.0444 |
| Control: CBAM, ImageNet-only net | full entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 5.320 ± 0.061 | 5.333 ± 0.056 | -0.013 (-0.027 to +0.001) | -0.23 | 0.0372 |
| Control: CBAM, ImageNet-only net | full entropy_norm | – | 0.971 ± 0.011 | 0.973 ± 0.010 | -0.002 (-0.005 to +0.000) | -0.23 | 0.0372 |
| Control: CBAM, ImageNet-only net | full area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.593 ± 0.050 | 0.603 ± 0.046 | -0.010 (-0.021 to +0.001) | -0.20 | 0.0598 |
| Control: CBAM, ImageNet-only net | full peak | – | 0.643 ± 0.097 | 0.630 ± 0.083 | +0.012 (-0.007 to +0.031) | +0.14 | 0.208 |
| Control: CBAM, ImageNet-only net | p7 gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.241 ± 0.049 | 0.234 ± 0.044 | +0.007 (-0.004 to +0.018) | +0.15 | 0.163 |
| Control: CBAM, ImageNet-only net | p7 entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 3.796 ± 0.040 | 3.802 ± 0.034 | -0.006 (-0.015 to +0.002) | -0.17 | 0.123 |
| Control: CBAM, ImageNet-only net | p7 entropy_norm | – | 0.975 ± 0.010 | 0.977 ± 0.009 | -0.002 (-0.004 to +0.001) | -0.17 | 0.123 |
| Control: CBAM, ImageNet-only net | p7 area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.655 ± 0.043 | 0.661 ± 0.037 | -0.006 (-0.015 to +0.003) | -0.14 | 0.199 |
| Control: CBAM, ImageNet-only net | p7 peak | – | 0.466 ± 0.033 | 0.469 ± 0.033 | -0.003 (-0.009 to +0.004) | -0.08 | 0.458 |
| Control: CBAM, ImageNet-only net | breast gini | 0.68 ± 0.12 vs 0.43 ± 0.15 | 0.312 ± 0.068 | 0.299 ± 0.063 | +0.013 (-0.002 to +0.028) | +0.20 | 0.0724 |
| Control: CBAM, ImageNet-only net | breast entropy | 2.31 ± 0.38 vs 3.74 ± 0.51 | 4.776 ± 0.253 | 4.775 ± 0.221 | +0.002 (-0.060 to +0.058) | +0.01 | 0.952 |
| Control: CBAM, ImageNet-only net | breast entropy_norm | – | 0.965 ± 0.016 | 0.969 ± 0.013 | -0.003 (-0.007 to +0.000) | -0.22 | 0.0469 |
| Control: CBAM, ImageNet-only net | breast area80 | 0.124 ± 0.032 vs 0.287 ± 0.058 | 0.602 ± 0.055 | 0.610 ± 0.052 | -0.008 (-0.020 to +0.004) | -0.15 | 0.167 |
| Control: CBAM, ImageNet-only net | breast peak | – | 0.603 ± 0.103 | 0.596 ± 0.093 | +0.007 (-0.013 to +0.027) | +0.07 | 0.502 |

| Map | Peak (grid) | Paper dense vs non-dense | Ours dense (C/D) | Ours non-dense (A/B) | Difference (95% CI) | Cohen's d | Pearson r with grade (95% CI) | Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CBAM attention (mt_cbam) | full | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.999 ± 0.001 | 0.999 ± 0.002 | +0.000 (-0.000 to +0.000) | +0.06 | +0.021 (-0.09–+0.12) | -0.028 |
| CBAM attention (mt_cbam) | p7 | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.994 ± 0.005 | 0.993 ± 0.005 | +0.001 (-0.000 to +0.002) | +0.15 | +0.089 (-0.02–+0.19) | +0.121 |
| CBAM attention (mt_cbam) | breast | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.999 ± 0.002 | 0.998 ± 0.002 | +0.000 (-0.000 to +0.001) | +0.13 | +0.070 (-0.04–+0.17) | -0.005 |
| CBAM attention (st_path) | full | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.996 ± 0.007 | 0.997 ± 0.006 | -0.001 (-0.002 to +0.001) | -0.13 | -0.052 (-0.14–+0.03) | -0.120 |
| CBAM attention (st_path) | p7 | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.972 ± 0.021 | 0.969 ± 0.023 | +0.002 (-0.002 to +0.007) | +0.10 | +0.080 (-0.01–+0.17) | +0.063 |
| CBAM attention (st_path) | breast | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.995 ± 0.007 | 0.996 ± 0.007 | -0.001 (-0.002 to +0.001) | -0.08 | -0.011 (-0.10–+0.08) | -0.084 |
| CBAM attention (st_dens) | full | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 1.000 ± 0.001 | 1.000 ± 0.001 | +0.000 (+0.000 to +0.000) | +0.25 | +0.215 (+0.09–+0.33) | +0.122 |
| CBAM attention (st_dens) | p7 | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.999 ± 0.003 | 0.998 ± 0.002 | +0.000 (-0.000 to +0.001) | +0.15 | +0.145 (+0.01–+0.31) | +0.141 |
| CBAM attention (st_dens) | breast | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 1.000 ± 0.001 | 1.000 ± 0.001 | +0.000 (+0.000 to +0.000) | +0.23 | +0.210 (+0.08–+0.33) | +0.121 |
| Grad-CAM malignancy (mt_cbam) | full | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.141 ± 0.112 | 0.231 ± 0.155 | -0.090 (-0.121 to -0.057) | -0.65 | -0.282 (-0.37–-0.19) | -0.312 |
| Grad-CAM malignancy (mt_cbam) | p7 | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.055 ± 0.042 | 0.086 ± 0.057 | -0.031 (-0.042 to -0.019) | -0.60 | -0.264 (-0.36–-0.16) | -0.290 |
| Grad-CAM malignancy (mt_cbam) | breast | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.128 ± 0.119 | 0.213 ± 0.170 | -0.085 (-0.118 to -0.049) | -0.56 | -0.241 (-0.33–-0.14) | -0.213 |
| Control: CBAM, ImageNet-only net | full | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.623 ± 0.088 | 0.644 ± 0.090 | -0.021 (-0.040 to -0.003) | -0.24 | -0.084 (-0.19–+0.02) | -0.109 |
| Control: CBAM, ImageNet-only net | p7 | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.466 ± 0.030 | 0.468 ± 0.035 | -0.003 (-0.010 to +0.004) | -0.08 | -0.016 (-0.14–+0.10) | -0.021 |
| Control: CBAM, ImageNet-only net | breast | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | 0.596 ± 0.100 | 0.601 ± 0.095 | -0.005 (-0.026 to +0.015) | -0.05 | -0.013 (-0.13–+0.09) | -0.041 |

### CBAM dynamic range (sigmoid gate values)

| Map | min | max | mean | median range within an image | median SD within an image |
|---|---:|---:|---:|---:|---:|
| CBAM attention (mt_cbam) | 0.000 | 1.000 | 0.851 | 0.809 | 0.164 |
| CBAM attention (st_path) | 0.000 | 1.000 | 0.760 | 0.901 | 0.184 |
| CBAM attention (st_dens) | 0.004 | 1.000 | 0.885 | 0.610 | 0.135 |
| Control: CBAM, ImageNet-only net | 0.000 | 0.956 | 0.233 | 0.595 | 0.125 |

Checkpoint check (same test images and predictions as Phase 4): mt_cbam: AUC 0.746 → 0.746, max |ΔP| 0.0341; mt_plain: AUC 0.772 → 0.774, max |ΔP| 0.0403; st_path: AUC 0.746 → 0.745, max |ΔP| 0.0383
