import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class OrdinalMultiTaskResNet18(nn.Module):
    """
    ResNet18 multi-task model.

    Task 1:
        Ordinal BI-RADS prediction using four cumulative thresholds.

        Original classes:
            1, 2, 3, 4, 5

        Zero-based dataset targets:
            0, 1, 2, 3, 4

        Ordinal questions:
            class > 0?
            class > 1?
            class > 2?
            class > 3?

    Task 2:
        ACR breast density classification:
            4 classes
    """

    def __init__(
        self,
        num_birads_thresholds=4,
        num_density_classes=4,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(
            weights=weights
        )

        feature_dim = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.birads_ordinal_head = nn.Linear(
            feature_dim,
            num_birads_thresholds,
        )

        self.density_head = nn.Linear(
            feature_dim,
            num_density_classes,
        )

    def forward(self, x):
        features = self.backbone(x)

        birads_logits = (
            self.birads_ordinal_head(
                features
            )
        )

        density_logits = (
            self.density_head(
                features
            )
        )

        return {
            "birads_ordinal": birads_logits,
            "density": density_logits,
        }


def ordinal_logits_to_class(
    logits,
    threshold=0.5,
):
    """
    Convert cumulative ordinal logits into
    zero-based BI-RADS class predictions.

    Example:

        sigmoid outputs:
            [0.9, 0.8, 0.7, 0.2]

        thresholded:
            [1, 1, 1, 0]

        sum:
            3

        zero-based class 3 = BI-RADS 4.
    """

    probabilities = torch.sigmoid(
        logits
    )

    passed_thresholds = (
        probabilities >= threshold
    ).long()

    predictions = (
        passed_thresholds.sum(dim=1)
    )

    return predictions