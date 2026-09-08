"""Create a Grad-CAM gallery for multiple shared-attention test examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.loaders import make_dataloaders
from src.evaluation.gradcam import GradCAM
from src.models.multitask_attention import (
    MultiTaskAttentionEfficientNetB0,
)


ASSESSMENT_NAMES = [
    "BI-RADS 1",
    "BI-RADS 2",
    "BI-RADS 3",
    "BI-RADS 4",
    "BI-RADS 5/6",
]

DENSITY_NAMES = [
    "ACR 1",
    "ACR 2",
    "ACR 3",
    "ACR 4",
]


def tensor_to_grayscale_image(image_tensor):
    """
    Convert normalized 3xHxW tensor to a displayable grayscale image.
    Since the input started as grayscale repeated into 3 channels,
    we can use the first channel after inverse normalization.
    """
    mean = 0.485
    std = 0.229

    image = image_tensor[0].detach().cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)

    return image


def load_original_png(image_id: str):
    path = next(Path("data/Inbreast").rglob(f"{image_id}.png"))
    image = Image.open(path).convert("L").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr, path


def collect_test_predictions(model, test_loader, device):
    """
    Run the shared-attention model on the whole test set and
    store predictions + tensors for later gallery selection.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            assessment_pred = outputs["assessment_logits"].argmax(dim=1).cpu()
            density_pred = outputs["density_logits"].argmax(dim=1).cpu()

            batch_size = images.shape[0]

            for i in range(batch_size):
                rows.append(
                    {
                        "image_tensor": batch["image"][i : i + 1],
                        "image_id": batch["image_id"][i],
                        "assessment_true": int(batch["assessment_target"][i]),
                        "assessment_pred": int(assessment_pred[i]),
                        "density_true": int(batch["density_target"][i]),
                        "density_pred": int(density_pred[i]),
                    }
                )

    return rows


def choose_examples(rows):
    """
    Pick a small, meaningful gallery:
    1) correct assessment
    2) wrong assessment
    3) correct density
    4) wrong density
    """
    selected = []

    correct_assessment = next(
        (r for r in rows if r["assessment_true"] == r["assessment_pred"]),
        None,
    )

    wrong_assessment = next(
        (r for r in rows if r["assessment_true"] != r["assessment_pred"]),
        None,
    )

    correct_density = next(
        (r for r in rows if r["density_true"] == r["density_pred"]),
        None,
    )

    wrong_density = next(
        (r for r in rows if r["density_true"] != r["density_pred"]),
        None,
    )

    for item, tag in [
        (correct_assessment, "correct_assessment"),
        (wrong_assessment, "wrong_assessment"),
        (correct_density, "correct_density"),
        (wrong_density, "wrong_density"),
    ]:
        if item is None:
            continue

        candidate = dict(item)
        candidate["tag"] = tag

        if candidate["image_id"] not in {x["image_id"] for x in selected}:
            selected.append(candidate)

    return selected


def make_gallery_figure(model, examples, output_path, device):
    """
    Save a 4-column gallery:
    Original | Assessment CAM overlay | Density CAM overlay | Text info
    """
    target_layer = model.features[6]
    gradcam = GradCAM(model, target_layer)

    n_rows = len(examples)
    fig = plt.figure(figsize=(16, 4 * n_rows))

    for row_idx, example in enumerate(examples):
        image = example["image_tensor"].to(device)
        image_id = example["image_id"]

        original_arr, _ = load_original_png(image_id)

        assessment_cam, _, _ = gradcam.generate(
            image,
            task="assessment",
            target_class=example["assessment_pred"],
        )

        density_cam, _, _ = gradcam.generate(
            image,
            task="density",
            target_class=example["density_pred"],
        )

        # Column 1: original
        ax1 = fig.add_subplot(n_rows, 4, row_idx * 4 + 1)
        ax1.imshow(original_arr, cmap="gray")
        ax1.set_title(f"{example['tag']} | Original")
        ax1.axis("off")

        # Column 2: assessment overlay
        ax2 = fig.add_subplot(n_rows, 4, row_idx * 4 + 2)
        ax2.imshow(original_arr, cmap="gray")
        ax2.imshow(assessment_cam, cmap="inferno", alpha=0.45)
        ax2.set_title("Assessment Grad-CAM")
        ax2.axis("off")

        # Column 3: density overlay
        ax3 = fig.add_subplot(n_rows, 4, row_idx * 4 + 3)
        ax3.imshow(original_arr, cmap="gray")
        ax3.imshow(density_cam, cmap="inferno", alpha=0.45)
        ax3.set_title("Density Grad-CAM")
        ax3.axis("off")

        # Column 4: text panel
        ax4 = fig.add_subplot(n_rows, 4, row_idx * 4 + 4)
        ax4.axis("off")

        if example["density_true"] == -1:
            density_true_name = "Missing"
        else:
            density_true_name = DENSITY_NAMES[example["density_true"]]
        
        text = (
            f"Image ID: {image_id}\n\n"
            f"Assessment:\n"
            f"  True: {ASSESSMENT_NAMES[example['assessment_true']]}\n"
            f"  Pred: {ASSESSMENT_NAMES[example['assessment_pred']]}\n\n"
            f"Density:\n"
            f"  True: {density_true_name}\n"
            f"  Pred: {DENSITY_NAMES[example['density_pred']]}"
        )

        ax4.text(
            0.0,
            0.95,
            text,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    fig.suptitle(
        "Shared-Attention Model: Multi-Example Grad-CAM Gallery",
        fontsize=16,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    gradcam.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, _, test_loader = make_dataloaders(
        batch_size=16,
        image_size=224,
        seed=42,
    )

    model = MultiTaskAttentionEfficientNetB0(pretrained=False).to(device)

    checkpoint = torch.load(
        "results/checkpoints/attention_shared_weighted.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = collect_test_predictions(model, test_loader, device)
    examples = choose_examples(rows)

    print("Selected examples:")
    for example in examples:
        print(
            f"  {example['tag']}: "
            f"{example['image_id']} | "
            f"assessment {example['assessment_true']}->{example['assessment_pred']} | "
            f"density {example['density_true']}->{example['density_pred']}"
        )

    output_path = "results/figures/gradcam_gallery_shared_attention.png"
    make_gallery_figure(
        model=model,
        examples=examples,
        output_path=output_path,
        device=device,
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()