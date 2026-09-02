"""Image preprocessing and augmentation for INbreast mammograms."""

from torchvision import transforms


# ImageNet normalization values used with pretrained CNN backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224):
    """
    Transform used during model training.

    Mammograms are stored as grayscale images, but pretrained ImageNet
    backbones expect 3-channel input, so the grayscale channel is
    replicated to RGB-like 3-channel format.
    """

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transform(image_size: int = 224):
    """
    Deterministic transform used during validation and testing.
    """

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )