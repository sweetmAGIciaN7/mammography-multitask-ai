---
title: Multi-task Mammography Model
emoji: 🩻
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: Malignancy + breast density, leakage-free replication
---

# Multi-task mammography model (research demo)

EfficientNet-B0 + CBAM attention with two heads (malignancy, BI-RADS density), the design of Esen et al.
(IEEE Access 2025), trained on the official CBIS-DDSM training split with a patient-level, leakage-free protocol.

- Code, experiments, model card and technical report:
  https://github.com/sweetmAGIciaN7/mammography-multitask-ai
- Test performance (official CBIS-DDSM test split, 212 unseen patients): malignancy AUC 0.746,
  density QWK 0.711. External (INbreast): density QWK 0.660.

**Research demo only. Not a medical device; must not be used for diagnosis, screening or any clinical decision.**

Example images come from CBIS-DDSM (Lee et al., *Scientific Data* 4:170177, 2017; The Cancer Imaging Archive,
CC BY 3.0).
