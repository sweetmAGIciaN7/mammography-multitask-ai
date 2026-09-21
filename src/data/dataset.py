from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class InbreastMultiTaskDataset(Dataset):
    """
    INbreast multi-task dataset.

    Returns:
        {
            "image": image,
            "birads": birads_target,
            "density": density_target,
            "density_mask": density_mask,
            "image_id": image_id,
        }

    BI-RADS classes in metadata.csv:
        1, 2, 3, 4, 5

    Converted to zero-based class indices:
        0, 1, 2, 3, 4

    Density ACR classes:
        1, 2, 3, 4

    Converted to zero-based class indices:
        0, 1, 2, 3

    One image has missing density.
    For that sample:
        density = 0   # placeholder only
        density_mask = 0.0

    For valid density samples:
        density_mask = 1.0
    """

    def __init__(
        self,
        metadata_csv,
        root_dir=None,
        transform=None,
        dataframe=None,
    ):
        self.metadata_csv = Path(metadata_csv)

        if root_dir is None:
            # metadata.csv is expected inside data/
            # project root is therefore one level above data/
            self.root_dir = self.metadata_csv.parent.parent
        else:
            self.root_dir = Path(root_dir)

        self.transform = transform

        if dataframe is None:
            self.df = pd.read_csv(self.metadata_csv)
        else:
            self.df = dataframe.reset_index(drop=True).copy()

        self._validate_metadata()

    def _validate_metadata(self):
        required_columns = {
            "image_id",
            "image_path",
            "birads_class",
            "density_acr",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required metadata columns: "
                f"{sorted(missing_columns)}"
            )

        if self.df["image_id"].duplicated().any():
            raise ValueError(
                "Duplicate image IDs found in dataset metadata."
            )

        invalid_birads = self.df[
            ~self.df["birads_class"].isin([1, 2, 3, 4, 5])
        ]

        if len(invalid_birads):
            raise ValueError(
                "Invalid BI-RADS classes found:\n"
                + invalid_birads[
                    ["image_id", "birads_class"]
                ].to_string(index=False)
            )

        valid_density = self.df["density_acr"].dropna()

        invalid_density = valid_density[
            ~valid_density.isin([1, 2, 3, 4])
        ]

        if len(invalid_density):
            raise ValueError(
                "Invalid density ACR classes found."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self.root_dir / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        # Zero-based class index:
        # BI-RADS 1..5 -> 0..4
        birads = int(row["birads_class"]) - 1

        if pd.isna(row["density_acr"]):
            # Placeholder target.
            # Must be ignored in the density loss using density_mask.
            density = 0
            density_mask = 0.0
        else:
            # ACR 1..4 -> 0..3
            density = int(row["density_acr"]) - 1
            density_mask = 1.0

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
            "density_mask": torch.tensor(
                density_mask,
                dtype=torch.float32,
            ),
            "image_id": str(row["image_id"]),
        }