from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import InbreastMultiTaskDataset
from src.models.multitask_resnet import MultiTaskResNet18


# ============================================================
# PATHS
# ============================================================

METADATA_PATH = ROOT / "data" / "metadata_split.csv"

RESULTS_DIR = ROOT / "results" / "baseline"

CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"

BIRADS_CM_PATH = RESULTS_DIR / "birads_confusion_matrix.png"
DENSITY_CM_PATH = RESULTS_DIR / "density_confusion_matrix.png"

BIRADS_REPORT_PATH = RESULTS_DIR / "birads_classification_report.txt"
DENSITY_REPORT_PATH = RESULTS_DIR / "density_classification_report.txt"


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n=== DEVICE ===")
print(device)


# ============================================================
# TEST METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

print("\n=== TEST SET ===")
print("Images:", len(test_df))


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# DATASET / LOADER
# ============================================================

test_dataset = InbreastMultiTaskDataset(
    metadata_csv=METADATA_PATH,
    dataframe=test_df,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# MODEL
# ============================================================

model = MultiTaskResNet18(
    pretrained=False,
)

model = model.to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded checkpoint.")
print("Best epoch:", checkpoint["epoch"])


# ============================================================
# INFERENCE
# ============================================================

birads_true = []
birads_pred = []

density_true = []
density_pred = []


with torch.no_grad():

    for batch in test_loader:

        images = (
            batch["image"]
            .to(device)
        )

        outputs = model(images)


        # ------------------------
        # BI-RADS
        # ------------------------

        birads_predictions = (
            outputs["birads"]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        birads_targets = (
            batch["birads"]
            .numpy()
        )

        birads_true.extend(
            birads_targets.tolist()
        )

        birads_pred.extend(
            birads_predictions.tolist()
        )


        # ------------------------
        # DENSITY
        # ------------------------

        density_predictions = (
            outputs["density"]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        density_targets = (
            batch["density"]
            .numpy()
        )

        density_masks = (
            batch["density_mask"]
            .numpy()
        )

        for (
            target,
            prediction,
            mask,
        ) in zip(
            density_targets,
            density_predictions,
            density_masks,
        ):

            if mask == 1:

                density_true.append(
                    int(target)
                )

                density_pred.append(
                    int(prediction)
                )


# ============================================================
# BI-RADS REPORT
# ============================================================

birads_target_names = [
    "BI-RADS 1",
    "BI-RADS 2",
    "BI-RADS 3",
    "BI-RADS 4",
    "BI-RADS 5",
]

birads_report = classification_report(
    birads_true,
    birads_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=birads_target_names,
    digits=4,
    zero_division=0,
)

print("\n=== BI-RADS CLASSIFICATION REPORT ===")
print(birads_report)

with open(
    BIRADS_REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        birads_report
    )


# ============================================================
# BI-RADS CONFUSION MATRIX
# ============================================================

birads_cm = confusion_matrix(
    birads_true,
    birads_pred,
    labels=[0, 1, 2, 3, 4],
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=birads_cm,
    display_labels=[
        "1",
        "2",
        "3",
        "4",
        "5",
    ],
)

display.plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "BI-RADS Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    BIRADS_CM_PATH,
    dpi=200,
)

plt.close(fig)


# ============================================================
# DENSITY REPORT
# ============================================================

density_target_names = [
    "ACR 1",
    "ACR 2",
    "ACR 3",
    "ACR 4",
]

density_report = classification_report(
    density_true,
    density_pred,
    labels=[0, 1, 2, 3],
    target_names=density_target_names,
    digits=4,
    zero_division=0,
)

print("\n=== DENSITY CLASSIFICATION REPORT ===")
print(density_report)

with open(
    DENSITY_REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        density_report
    )


# ============================================================
# DENSITY CONFUSION MATRIX
# ============================================================

density_cm = confusion_matrix(
    density_true,
    density_pred,
    labels=[0, 1, 2, 3],
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=density_cm,
    display_labels=[
        "1",
        "2",
        "3",
        "4",
    ],
)

display.plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "ACR Density Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    DENSITY_CM_PATH,
    dpi=200,
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("\n=== SAVED ===")

print(BIRADS_CM_PATH)
print(DENSITY_CM_PATH)
print(BIRADS_REPORT_PATH)
print(DENSITY_REPORT_PATH)

print("\nSUCCESS")