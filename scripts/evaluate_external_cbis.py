from pathlib import Path
import sys
import json

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from src.data.cbis_dataset import (
    CbisExternalDataset,
)

from src.models.multitask_resnet import (
    MultiTaskResNet18,
)


METADATA = (
    ROOT
    / "data"
    / "external"
    / "cbis_ddsm"
    / "cbis_metadata.csv"
)

CHECKPOINT = (
    ROOT
    / "results"
    / "baseline_improved"
    / "best_model.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "external_cbis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BATCH_SIZE = 4


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (384, 384)
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


def load_model(device):
    model = MultiTaskResNet18(
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    print(
        "Checkpoint type:",
        type(checkpoint),
    )

    # Support either:
    # 1. raw state_dict
    # 2. training checkpoint dict
    if isinstance(
        checkpoint,
        dict,
    ):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            # Could already be a raw state_dict.
            state_dict = checkpoint
    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return model


def plot_confusion_matrix(
    cm,
    labels,
    title,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    for i in range(
        cm.shape[0]
    ):
        for j in range(
            cm.shape[1]
        ):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Metadata:",
        METADATA,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = CbisExternalDataset(
        METADATA,
        transform=make_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    model = load_model(
        device
    )

    birads_true = []
    birads_pred = []

    density_true = []
    density_pred = []

    processed = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "image"
            ].to(device)

            birads_labels = batch[
                "birads"
            ]

            density_labels = batch[
                "density"
            ]

            birads_mask = batch[
                "birads_mask"
            ].bool()

            density_mask = batch[
                "density_mask"
            ].bool()

            outputs = model(
                images
            )

            # Support common model return formats.
            if isinstance(
                outputs,
                dict,
            ):
                if "birads" in outputs:
                    birads_logits = outputs[
                        "birads"
                    ]
                elif "birads_logits" in outputs:
                    birads_logits = outputs[
                        "birads_logits"
                    ]
                else:
                    raise KeyError(
                        "Could not find BI-RADS "
                        "logits in model output."
                    )

                if "density" in outputs:
                    density_logits = outputs[
                        "density"
                    ]
                elif "density_logits" in outputs:
                    density_logits = outputs[
                        "density_logits"
                    ]
                else:
                    raise KeyError(
                        "Could not find density "
                        "logits in model output."
                    )

            elif isinstance(
                outputs,
                (tuple, list),
            ):
                if len(outputs) != 2:
                    raise RuntimeError(
                        "Expected 2 model outputs."
                    )

                birads_logits = outputs[0]
                density_logits = outputs[1]

            else:
                raise RuntimeError(
                    "Unsupported model output format."
                )

            birads_predictions = (
                birads_logits
                .argmax(dim=1)
                .cpu()
            )

            density_predictions = (
                density_logits
                .argmax(dim=1)
                .cpu()
            )

            if birads_mask.any():
                birads_true.extend(
                    birads_labels[
                        birads_mask
                    ]
                    .numpy()
                    .tolist()
                )

                birads_pred.extend(
                    birads_predictions[
                        birads_mask
                    ]
                    .numpy()
                    .tolist()
                )

            if density_mask.any():
                density_true.extend(
                    density_labels[
                        density_mask
                    ]
                    .numpy()
                    .tolist()
                )

                density_pred.extend(
                    density_predictions[
                        density_mask
                    ]
                    .numpy()
                    .tolist()
                )

            processed += (
                images.size(0)
            )

            if (
                processed % 40 == 0
                or processed
                == len(dataset)
            ):
                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                )

    print(
        "\n=== SAMPLE COUNTS ==="
    )

    print(
        "BI-RADS evaluated:",
        len(birads_true),
    )

    print(
        "Density evaluated:",
        len(density_true),
    )

    birads_accuracy = accuracy_score(
        birads_true,
        birads_pred,
    )

    birads_macro_f1 = f1_score(
        birads_true,
        birads_pred,
        average="macro",
        labels=[
            0,
            1,
            2,
            3,
            4,
        ],
        zero_division=0,
    )

    density_accuracy = accuracy_score(
        density_true,
        density_pred,
    )

    density_macro_f1 = f1_score(
        density_true,
        density_pred,
        average="macro",
        labels=[
            0,
            1,
            2,
            3,
        ],
        zero_division=0,
    )

    print(
        "\n=== EXTERNAL CBIS-DDSM RESULTS ==="
    )

    print(
        f"BI-RADS accuracy: "
        f"{birads_accuracy:.4f}"
    )

    print(
        f"BI-RADS macro-F1: "
        f"{birads_macro_f1:.4f}"
    )

    print(
        f"Density accuracy: "
        f"{density_accuracy:.4f}"
    )

    print(
        f"Density macro-F1: "
        f"{density_macro_f1:.4f}"
    )

    birads_report = (
        classification_report(
            birads_true,
            birads_pred,
            labels=[
                0,
                1,
                2,
                3,
                4,
            ],
            target_names=[
                "BI-RADS 1",
                "BI-RADS 2",
                "BI-RADS 3",
                "BI-RADS 4",
                "BI-RADS 5",
            ],
            digits=4,
            zero_division=0,
        )
    )

    density_report = (
        classification_report(
            density_true,
            density_pred,
            labels=[
                0,
                1,
                2,
                3,
            ],
            target_names=[
                "ACR 1",
                "ACR 2",
                "ACR 3",
                "ACR 4",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print(
        "\n=== BI-RADS REPORT ==="
    )

    print(
        birads_report
    )

    print(
        "\n=== DENSITY REPORT ==="
    )

    print(
        density_report
    )

    birads_cm = confusion_matrix(
        birads_true,
        birads_pred,
        labels=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    density_cm = confusion_matrix(
        density_true,
        density_pred,
        labels=[
            0,
            1,
            2,
            3,
        ],
    )

    plot_confusion_matrix(
        birads_cm,
        [
            "1",
            "2",
            "3",
            "4",
            "5",
        ],
        "External CBIS-DDSM: BI-RADS",
        OUTPUT_DIR
        / "birads_confusion_matrix.png",
    )

    plot_confusion_matrix(
        density_cm,
        [
            "1",
            "2",
            "3",
            "4",
        ],
        "External CBIS-DDSM: Density",
        OUTPUT_DIR
        / "density_confusion_matrix.png",
    )

    metrics = {
        "dataset": "CBIS-DDSM",
        "evaluation_type":
            "external_no_finetuning",

        "total_images": len(dataset),

        "birads_evaluated":
            len(birads_true),

        "density_evaluated":
            len(density_true),

        "birads_accuracy":
            float(
                birads_accuracy
            ),

        "birads_macro_f1":
            float(
                birads_macro_f1
            ),

        "density_accuracy":
            float(
                density_accuracy
            ),

        "density_macro_f1":
            float(
                density_macro_f1
            ),
    }

    with open(
        OUTPUT_DIR
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        OUTPUT_DIR
        / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "EXTERNAL CBIS-DDSM EVALUATION\n"
        )

        f.write(
            "No retraining or fine-tuning "
            "on CBIS-DDSM.\n\n"
        )

        f.write(
            "BI-RADS\n"
        )

        f.write(
            birads_report
        )

        f.write(
            "\n\nDENSITY\n"
        )

        f.write(
            density_report
        )

    print(
        "\nSaved results to:",
        OUTPUT_DIR,
    )

    print(
        "\nSUCCESS"
    )


if __name__ == "__main__":
    main()