from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class InbreastDataset(Dataset):
    """
    INbreast dataset loader for multi-task learning.

    Targets:
    - diagnostic_target:
        0 = BI-RADS-derived negative
        1 = BI-RADS-derived positive
       -1 = excluded from binary task (BI-RADS 3)

    - density_target:
        0 = ACR 1
        1 = ACR 2
        2 = ACR 3
        3 = ACR 4
       -1 = missing density

    Important:
    diagnostic_target is derived from BI-RADS assessment and is not
    biopsy-confirmed pathology ground truth.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        metadata_csv: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata_csv = Path(metadata_csv)
        self.transform = transform

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset root directory does not exist: {self.root_dir}"
            )

        if not self.metadata_csv.is_file():
            raise FileNotFoundError(
                f"Metadata CSV does not exist: {self.metadata_csv}"
            )

        self.metadata = pd.read_csv(
            self.metadata_csv,
            sep=";",
            dtype=str,
        )

        required_columns = {"File Name", "ACR", "Bi-Rads"}
        missing_columns = required_columns - set(self.metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.metadata["File Name"] = (
            self.metadata["File Name"]
            .fillna("")
            .str.strip()
        )

        self.metadata["ACR"] = (
            self.metadata["ACR"]
            .fillna("")
            .str.strip()
        )

        self.metadata["Bi-Rads"] = (
            self.metadata["Bi-Rads"]
            .fillna("")
            .str.strip()
            .str.lower()
        )

        if self.metadata["File Name"].duplicated().any():
            duplicates = self.metadata.loc[
                self.metadata["File Name"].duplicated(),
                "File Name",
            ].tolist()

            raise ValueError(
                f"Duplicate File Name values found in metadata: {duplicates}"
            )

        self.metadata_by_id = self.metadata.set_index("File Name")

        png_paths = sorted(
            p
            for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".png"
        )

        if not png_paths:
            raise ValueError(
                f"No PNG images found under {self.root_dir}"
            )

        self.samples = []

        for image_path in png_paths:
            image_id = image_path.stem

            if image_id not in self.metadata_by_id.index:
                raise ValueError(
                    f"No metadata row found for image ID {image_id}"
                )

            row = self.metadata_by_id.loc[image_id]

            diagnostic_target = self._encode_diagnostic_target(
                row["Bi-Rads"]
            )
             
            assessment_target = self._encode_assessment_target(
                row["Bi-Rads"]
            )

            density_target = self._encode_density_target(
                row["ACR"]
            )

            self.samples.append(
                {
                    "image_path": image_path,
                    "image_id": image_id,
                    "diagnostic_target": diagnostic_target,
                    "assessment_target": assessment_target,
                    "density_target": density_target,
                }
            )

    @staticmethod
    def _encode_diagnostic_target(birads: str) -> int:
        """
        BI-RADS-derived diagnostic target.

        1, 2 -> negative (0)
        3    -> excluded from binary task (-1)
        4a, 4b, 4c, 5, 6 -> positive (1)
        """

        if birads in {"1", "2"}:
            return 0

        if birads == "3":
            return -1

        if birads in {"4a", "4b", "4c", "5", "6"}:
            return 1

        raise ValueError(
            f"Unexpected Bi-Rads value: {birads!r}"
        )

    @staticmethod
    def _encode_assessment_target(birads: str) -> int:
        """
        BI-RADS assessment target grouped into five classes.

        1          -> 0
        2          -> 1
        3          -> 2
        4a/4b/4c   -> 3
        5/6        -> 4
        """

        if birads == "1":
            return 0

        if birads == "2":
            return 1

        if birads == "3":
            return 2

        if birads in {"4a", "4b", "4c"}:
            return 3

        if birads in {"5", "6"}:
            return 4

        raise ValueError(
            f"Unexpected Bi-Rads value: {birads!r}"
        )

    @staticmethod
    def _encode_density_target(acr: str) -> int:
        """
        ACR density target.

        ACR 1 -> 0
        ACR 2 -> 1
        ACR 3 -> 2
        ACR 4 -> 3
        missing -> -1
        """

        if acr == "":
            return -1

        mapping = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
        }

        if acr not in mapping:
            raise ValueError(
                f"Unexpected ACR value: {acr!r}"
            )

        return mapping[acr]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": sample["image_id"],
            "diagnostic_target": sample["diagnostic_target"],
            "assessment_target": sample["assessment_target"],
            "density_target": sample["density_target"],
        }