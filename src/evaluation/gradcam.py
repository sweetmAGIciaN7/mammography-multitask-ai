"""Grad-CAM visualizations for the shared-attention multi-task model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.loaders import make_dataloaders
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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

        output.register_hook(
            self._save_gradient
        )

    def _save_gradient(
        self,
        gradient,
    ):
        self.gradients = gradient

    def generate(
        self,
        image,
        task="assessment",
        target_class=None,
    ):
        self.model.zero_grad()

        outputs = self.model(image)

        if task == "assessment":
            logits = outputs["assessment_logits"]

        elif task == "density":
            logits = outputs["density_logits"]

        else:
            raise ValueError(
                "task must be 'assessment' or 'density'"
            )

        if target_class is None:
            target_class = logits.argmax(
                dim=1
            ).item()

        score = logits[
            0,
            target_class,
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return (
            cam.detach().cpu().numpy(),
            target_class,
            outputs,
        )

    def close(self):
        self.handle.remove()


def save_comparison_figure(
    original_image_path,
    late_heatmap,
    mid_heatmap,
    output_path,
    title,
):
    original = Image.open(
        original_image_path
    ).convert("L")

    original = original.resize(
        (
            late_heatmap.shape[1],
            late_heatmap.shape[0],
        )
    )

    original_array = np.asarray(
        original,
        dtype=np.float32,
    )

    original_array /= 255.0

    fig = plt.figure(
        figsize=(16, 4)
    )

    ax1 = fig.add_subplot(
        1,
        4,
        1,
    )

    ax1.imshow(
        original_array,
        cmap="gray",
    )

    ax1.set_title(
        "Original"
    )

    ax1.axis("off")

    ax2 = fig.add_subplot(
        1,
        4,
        2,
    )

    ax2.imshow(
        late_heatmap,
        cmap="inferno",
    )

    ax2.set_title(
        "Late-layer Grad-CAM"
    )

    ax2.axis("off")

    ax3 = fig.add_subplot(
        1,
        4,
        3,
    )

    ax3.imshow(
        mid_heatmap,
        cmap="inferno",
    )

    ax3.set_title(
        "Mid-layer Grad-CAM"
    )

    ax3.axis("off")

    ax4 = fig.add_subplot(
        1,
        4,
        4,
    )

    ax4.imshow(
        original_array,
        cmap="gray",
    )

    ax4.imshow(
        mid_heatmap,
        cmap="inferno",
        alpha=0.45,
    )

    ax4.set_title(
        "Mid-layer overlay"
    )

    ax4.axis("off")

    fig.suptitle(
        title
    )

    fig.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
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

    _, _, test_loader = make_dataloaders(
        batch_size=1,
        image_size=224,
        seed=42,
    )

    model = MultiTaskAttentionEfficientNetB0(
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        "results/checkpoints/attention_shared_weighted.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    batch = next(
        iter(test_loader)
    )

    image = batch["image"].to(
        device
    )

    image_id = batch[
        "image_id"
    ][0]

    true_assessment = batch[
        "assessment_target"
    ].item()

    true_density = batch[
        "density_target"
    ].item()

    original_image_path = next(
        Path("data/Inbreast").rglob(
            f"{image_id}.png"
        )
    )

    # Final EfficientNet block.
    late_layer = model.features

    # Earlier block with higher spatial resolution.
    mid_layer = model.features[6]

    late_cam = GradCAM(
        model,
        late_layer,
    )

    mid_cam = GradCAM(
        model,
        mid_layer,
    )

    # ----------------------------------------------------------
    # Assessment
    # ----------------------------------------------------------

    assessment_late, predicted_assessment, _ = (
        late_cam.generate(
            image,
            task="assessment",
        )
    )

    assessment_mid, _, _ = (
        mid_cam.generate(
            image,
            task="assessment",
            target_class=predicted_assessment,
        )
    )

    assessment_title = (
        f"Assessment | "
        f"True: {ASSESSMENT_NAMES[true_assessment]} | "
        f"Predicted: "
        f"{ASSESSMENT_NAMES[predicted_assessment]}"
    )

    assessment_output = (
        Path("results/figures")
        / f"{image_id}_assessment_gradcam_comparison.png"
    )

    save_comparison_figure(
        original_image_path,
        assessment_late,
        assessment_mid,
        assessment_output,
        assessment_title,
    )

    # ----------------------------------------------------------
    # Density
    # ----------------------------------------------------------

    density_late, predicted_density, _ = (
        late_cam.generate(
            image,
            task="density",
        )
    )

    density_mid, _, _ = (
        mid_cam.generate(
            image,
            task="density",
            target_class=predicted_density,
        )
    )

    density_title = (
        f"Density | "
        f"True: {DENSITY_NAMES[true_density]} | "
        f"Predicted: "
        f"{DENSITY_NAMES[predicted_density]}"
    )

    density_output = (
        Path("results/figures")
        / f"{image_id}_density_gradcam_comparison.png"
    )

    save_comparison_figure(
        original_image_path,
        density_late,
        density_mid,
        density_output,
        density_title,
    )

    late_cam.close()
    mid_cam.close()

    print()
    print(
        "Image ID:",
        image_id,
    )

    print(
        "True assessment:",
        ASSESSMENT_NAMES[
            true_assessment
        ],
    )

    print(
        "Predicted assessment:",
        ASSESSMENT_NAMES[
            predicted_assessment
        ],
    )

    print(
        "True density:",
        DENSITY_NAMES[
            true_density
        ],
    )

    print(
        "Predicted density:",
        DENSITY_NAMES[
            predicted_density
        ],
    )

    print()
    print(
        "Saved:",
        assessment_output,
    )

    print(
        "Saved:",
        density_output,
    )


if __name__ == "__main__":
    main()