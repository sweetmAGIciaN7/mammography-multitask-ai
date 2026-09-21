from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torchvision import transforms
from src.data.dataset import InbreastMultiTaskDataset


metadata_csv = ROOT / "data" / "metadata.csv"

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

dataset = InbreastMultiTaskDataset(
    metadata_csv=metadata_csv,
    transform=transform,
)

print("\n=== DATASET TEST ===")
print("Length:", len(dataset))

sample = dataset[0]

print("\nFirst sample:")
print("Image shape:", sample["image"].shape)
print("BI-RADS target:", sample["birads"])
print("Density target:", sample["density"])
print("Density mask:", sample["density_mask"])
print("Image ID:", sample["image_id"])

missing_density_count = 0

for i in range(len(dataset)):
    if dataset[i]["density_mask"].item() == 0:
        missing_density_count += 1

print("\nMissing density samples:", missing_density_count)

print("\nSUCCESS")