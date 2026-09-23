from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from torchvision import transforms

from src.data.cbis_dataset import (
    CbisExternalDataset,
)


ROOT = Path(__file__).resolve().parents[1]

METADATA = (
    ROOT
    / "data"
    / "external"
    / "cbis_ddsm"
    / "cbis_metadata.csv"
)


transform = transforms.Compose(
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


dataset = CbisExternalDataset(
    METADATA,
    transform=transform,
)


print(
    "Dataset length:",
    len(dataset),
)


birads_usable = 0
density_usable = 0


for i in range(
    len(dataset)
):
    sample = dataset[i]

    birads_usable += int(
        sample[
            "birads_mask"
        ].item()
    )

    density_usable += int(
        sample[
            "density_mask"
        ].item()
    )

    if i < 3:
        print(
            "\nSample",
            i,
        )

        print(
            "Image shape:",
            sample["image"].shape,
        )

        print(
            "BI-RADS:",
            sample["birads"],
        )

        print(
            "BI-RADS mask:",
            sample["birads_mask"],
        )

        print(
            "Density:",
            sample["density"],
        )

        print(
            "Density mask:",
            sample["density_mask"],
        )

        print(
            "ID:",
            sample["match_id"],
        )

        print(
            "Dataset:",
            sample["dataset"],
        )


print(
    "\nBI-RADS usable:",
    birads_usable,
)

print(
    "Density usable:",
    density_usable,
)

print(
    "\nSUCCESS"
)