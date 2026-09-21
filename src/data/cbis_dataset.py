from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

from PIL import Image
from torch.utils.data import Dataset


class CbisExternalDataset(Dataset):
    """
    External-validation dataset for CBIS-DDSM.

    Important:
    - Uses image-level cbis_metadata.csv
    - Reads original DICOM files
    - Does not perform any training-time augmentation
    - Supports separate masks for:
        * BI-RADS/assessment
        * breast density
    """

    def __init__(
        self,
        metadata_csv,
        transform=None,
    ):
        self.metadata_csv = Path(
            metadata_csv
        )

        self.df = pd.read_csv(
            self.metadata_csv
        )

        self.transform = transform

        required = [
            "dicom_path",
            "assessment",
            "breast_density",
            "birads_usable",
            "density_usable",
            "match_id",
            "dataset",
        ]

        missing = [
            col
            for col in required
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _dicom_to_pil(path):
        ds = pydicom.dcmread(
            path
        )

        image = ds.pixel_array.astype(
            np.float32
        )

        photometric = str(
            getattr(
                ds,
                "PhotometricInterpretation",
                "",
            )
        ).upper()

        # Current audit showed MONOCHROME2 for all files,
        # but keep this correct for MONOCHROME1 too.
        if photometric == "MONOCHROME1":
            image = image.max() - image

        # Robust normalization.
        #
        # Mammograms can contain extreme background /
        # scanner values. Percentile clipping avoids
        # letting a tiny number of extreme pixels define
        # the whole display range.
        nonzero = image[
            image > 0
        ]

        if nonzero.size > 0:
            low = np.percentile(
                nonzero,
                0.5,
            )

            high = np.percentile(
                nonzero,
                99.5,
            )
        else:
            low = float(
                image.min()
            )

            high = float(
                image.max()
            )

        if high <= low:
            low = float(
                image.min()
            )

            high = float(
                image.max()
            )

        if high > low:
            image = np.clip(
                image,
                low,
                high,
            )

            image = (
                image - low
            ) / (
                high - low
            )
        else:
            image = np.zeros_like(
                image,
                dtype=np.float32,
            )

        image = (
            image * 255.0
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            image,
            mode="L",
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.df.iloc[
            index
        ]

        image = self._dicom_to_pil(
            row["dicom_path"]
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        birads_usable = bool(
            row["birads_usable"]
        )

        density_usable = bool(
            row["density_usable"]
        )

        # Placeholder label when unusable.
        # The corresponding mask is 0, so it must
        # not contribute to metrics.
        if birads_usable:
            birads = int(
                row["assessment"]
            ) - 1
        else:
            birads = 0

        if density_usable:
            density = int(
                row["breast_density"]
            ) - 1
        else:
            density = 0

        return {
            "image": image,

            "birads": torch.tensor(
                birads,
                dtype=torch.long,
            ),

            "density": torch.tensor(
                density,
                dtype=torch.long,
            ),

            "birads_mask": torch.tensor(
                float(
                    birads_usable
                ),
                dtype=torch.float32,
            ),

            "density_mask": torch.tensor(
                float(
                    density_usable
                ),
                dtype=torch.float32,
            ),

            "match_id": str(
                row["match_id"]
            ),

            "dataset": str(
                row["dataset"]
            ),
        }