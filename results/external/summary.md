### Density: CBIS-DDSM official test (internal) vs INbreast (external)

| Model | CBIS-DDSM acc. | CBIS-DDSM QWK | INbreast acc. (95% CI) | INbreast QWK (95% CI) | within ±1 grade |
|---|---:|---:|---:|---:|---:|
| Multi-task + CBAM (paper design) | 0.606 | 0.711 | 0.538 (0.464–0.616) | 0.660 (0.550–0.743) | 0.934 |
| Multi-task, no attention | 0.639 | 0.736 | 0.560 (0.480–0.646) | 0.655 (0.545–0.744) | 0.941 |
| Density only + CBAM | 0.674 | 0.721 | 0.545 (0.472–0.617) | 0.601 (0.489–0.694) | 0.963 |
| *Always predict INbreast's most common grade* | | | *0.357* | *0.000* | |
| *Always predict CBIS-DDSM's most common grade* | | | *0.357* | *0.000* | |

INbreast density: true grade counts A/B/C/D = [136, 146, 99, 28]; the paper-design model predicted [138, 141, 69, 61].

Density calibration on INbreast (paper design, top-label ECE): uncalibrated 0.107, cbis temperature 0.129 (T=0.88), oracle temperature fitted on inbreast 0.083 (T=1.36)

Confusion matrix (paper design, rows = true ACR grade, columns = predicted):

| true \ pred | A | B | C | D |
|---|---:|---:|---:|---:|
| A | 100 | 35 | 1 | 0 |
| B | 29 | 73 | 28 | 16 |
| C | 9 | 32 | 30 | 28 |
| D | 0 | 1 | 10 | 17 |

### Malignancy proxy on INbreast (exploratory: BI-RADS 4–6 vs 1–3, not pathology)

| Model | CBIS-DDSM official AUC | INbreast proxy AUC (95% CI) | BI-RADS 5–6 vs 1–2 AUC |
|---|---:|---:|---:|
| Multi-task + CBAM (paper design) | 0.746 | 0.822 (0.761–0.877) | 0.907 (0.838–0.962) |
| Multi-task, no attention | 0.772 | 0.796 (0.733–0.850) | 0.854 (0.783–0.915) |
| Malignancy only + CBAM | 0.746 | 0.788 (0.722–0.847) | 0.880 (0.809–0.938) |

| Question (INbreast) | Difference (A − B) | 95% CI | P(A not better) |
|---|---:|---:|---:|
| Does multi-task training help density transfer? (QWK) | +0.059 | -0.019 to +0.146 | 0.08 |
| Does CBAM help density transfer? (QWK) | +0.004 | -0.038 to +0.048 | 0.43 |
| Does multi-task training help the malignancy proxy? (proxy AUC) | +0.034 | -0.012 to +0.086 | 0.08 |
| Does CBAM help the malignancy proxy? (proxy AUC) | +0.026 | -0.015 to +0.069 | 0.11 |

Reproducibility of the Phase 3 official-split result (paper design, retrained with the same seed): path_auc 0.746 → 0.746, path_accuracy 0.622 → 0.622, dens_accuracy 0.606 → 0.606, dens_qwk 0.711 → 0.711
