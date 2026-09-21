from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from src.data.cbis_dataset_harmonized import (
    CbisHarmonizedDataset,
)


CBIS_METADATA = (
    ROOT
    / "data"
    / "external"
    / "cbis_ddsm"
    / "cbis_metadata.csv"
)

INBREAST_ROOT = (
    ROOT
    / "data"
    / "Inbreast"
)

OUTPUT = (
    ROOT
    / "results"
    / "external_cbis_harmonized"
    / "preprocessing_comparison.png"
)


def find_inbreast_images():
    images = []

    for folder in sorted(
        INBREAST_ROOT.glob("birads*")
    ):
        images.extend(
            sorted(
                folder.glob("*.png")
            )
        )

    return images


def main():
    cbis_df = pd.read_csv(
        CBIS_METADATA
    )

    inbreast_images = (
        find_inbreast_images()
    )

    inbreast_indices = [
        0,
        len(inbreast_images) // 4,
        len(inbreast_images) // 2,
        3 * len(inbreast_images) // 4,
        len(inbreast_images) - 1,
    ]

    cbis_indices = [
        0,
        len(cbis_df) // 4,
        len(cbis_df) // 2,
        3 * len(cbis_df) // 4,
        len(cbis_df) - 1,
    ]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(16, 8),
    )

    for column, index in enumerate(
        inbreast_indices
    ):
        path = inbreast_images[
            index
        ]

        image = Image.open(
            path
        ).convert("L")

        axes[
            0,
            column
        ].imshow(
            image,
            cmap="gray",
        )

        axes[
            0,
            column
        ].set_title(
            path.parent.name
            + "\n"
            + path.name
        )

        axes[
            0,
            column
        ].axis("off")

    for column, index in enumerate(
        cbis_indices
    ):
        row = cbis_df.iloc[
            index
        ]

        image = (
            CbisHarmonizedDataset
            ._dicom_to_pil(
                row["dicom_path"]
            )
        )

        axes[
            1,
            column
        ].imshow(
            image,
            cmap="gray",
        )

        axes[
            1,
            column
        ].set_title(
            row["dataset"]
            + "\n"
            + row["match_id"]
        )

        axes[
            1,
            column
        ].axis("off")

    axes[
        0,
        0
    ].set_ylabel(
        "INbreast",
        fontsize=14,
    )

    axes[
        1,
        0
    ].set_ylabel(
        "CBIS harmonized",
        fontsize=14,
    )

    fig.suptitle(
        "Label-independent input harmonization",
        fontsize=16,
    )

    fig.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT,
        dpi=200,
    )

    print(
        "Saved:",
        OUTPUT,
    )

    print("SUCCESS")


if __name__ == "__main__":
    main()