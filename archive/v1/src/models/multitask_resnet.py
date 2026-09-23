import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MultiTaskResNet18(nn.Module):
    """
    ResNet18 backbone with two classification heads:

    1. BI-RADS assessment:
       5 classes

    2. ACR breast density:
       4 classes
    """

    def __init__(
        self,
        num_birads_classes=5,
        num_density_classes=4,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(weights=weights)

        feature_dim = backbone.fc.in_features

        # Remove original ImageNet classifier
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.birads_head = nn.Linear(
            feature_dim,
            num_birads_classes,
        )

        self.density_head = nn.Linear(
            feature_dim,
            num_density_classes,
        )

    def forward(self, x):
        features = self.backbone(x)

        birads_logits = self.birads_head(features)
        density_logits = self.density_head(features)

        return {
            "birads": birads_logits,
            "density": density_logits,
        }