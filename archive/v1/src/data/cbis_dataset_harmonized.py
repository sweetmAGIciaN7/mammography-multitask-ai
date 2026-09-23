from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

from PIL import Image
from torch.utils.data import Dataset


class CbisHarmonizedDataset(Dataset):
    """
    CBIS-DDSM external-validation dataset
    with label-independent input harmonization.

    No labels are used during preprocessing.
    No training or fine-tuning is performed.
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
    def _crop_background(image):
        """
        Crop large black borders while trying to ignore
        small text/markers around image edges.
        """

        nonzero = image[
            image > 0
        ]

        if nonzero.size == 0:
            return image

        threshold = np.percentile(
            nonzero,
            2.0,
        )

        mask = image > threshold

        # A real breast occupies a substantial fraction
        # of a row/column; isolated letters usually do not.
        row_fraction = mask.mean(
            axis=1
        )

        col_fraction = mask.mean(
            axis=0
        )

        valid_rows = np.where(
            row_fraction > 0.02
        )[0]

        valid_cols = np.where(
            col_fraction > 0.02
        )[0]

        if (
            valid_rows.size == 0
            or valid_cols.size == 0
        ):
            return image

        y1 = int(valid_rows[0])
        y2 = int(valid_rows[-1]) + 1

        x1 = int(valid_cols[0])
        x2 = int(valid_cols[-1]) + 1

        height = y2 - y1
        width = x2 - x1

        # Small padding so breast boundary is not cut.
        pad_y = max(
            int(height * 0.02),
            1,
        )

        pad_x = max(
            int(width * 0.02),
            1,
        )

        y1 = max(
            0,
            y1 - pad_y,
        )

        y2 = min(
            image.shape[0],
            y2 + pad_y,
        )

        x1 = max(
            0,
            x1 - pad_x,
        )

        x2 = min(
            image.shape[1],
            x2 + pad_x,
        )

        return image[
            y1:y2,
            x1:x2
        ]

    @staticmethod
    def _normalize_intensity(image):
        """
        Wider percentile window than the original
        CBIS preprocessing to preserve more of the
        mammogram's native gray-scale range.
        """

        nonzero = image[
            image > 0
        ]

        if nonzero.size == 0:
            return np.zeros_like(
                image,
                dtype=np.uint8,
            )

        low = np.percentile(
            nonzero,
            0.1,
        )

        high = np.percentile(
            nonzero,
            99.9,
        )

        if high <= low:
            low = float(
                image.min()
            )

            high = float(
                image.max()
            )

        if high <= low:
            return np.zeros_like(
                image,
                dtype=np.uint8,
            )

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

        image = (
            image * 255.0
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        return image

    @classmethod
    def _dicom_to_pil(
        cls,
        path,
    ):
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

        if photometric == "MONOCHROME1":
            image = (
                image.max()
                - image
            )

        image = cls._crop_background(
            image
        )

        image = cls._normalize_intensity(
            image
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

        if birads_usable:
            birads = (
                int(
                    row["assessment"]
                )
                - 1
            )
        else:
            birads = 0

        if density_usable:
            density = (
                int(
                    row["breast_density"]
                )
                - 1
            )
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