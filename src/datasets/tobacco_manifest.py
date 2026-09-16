import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


SOURCE_LABEL_COLUMNS = (
    "y00_c02_weather_fleck",
    "y01_c03_potato_virus_y",
    "y02_c04_tobacco_mosaic_virus",
    "y03_c05_cucumber_mosaic_virus",
    "y04_c06_alternaria_leaf_spot",
    "y05_c07_black_shank",
    "y06_c08_bacterial_wilt",
    "y07_c09_hollow_stalk",
    "y08_c10_black_root_rot",
    "y09_c11_wildfire",
    "y10_c12_potassium_deficiency",
    "y11_c13_magnesium_deficiency",
    "y12_c14_powdery_mildew",
    "y13_c15_root_knot_nematodes",
    "y14_c16_anthracnose",
    "y15_c17_frogeye_leaf_spot",
    "y16_c18_angular_leaf_spot",
    "y17_c19_sunburn",
    "y18_c20_herbicide_phytotoxicity",
)

EXCLUDED_LABEL_COLUMN = "y13_c15_root_knot_nematodes"
TASK_LABEL_COLUMNS = SOURCE_LABEL_COLUMNS[:13] + SOURCE_LABEL_COLUMNS[14:]
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TobaccoRecord:
    file_name: str
    target: torch.Tensor


class TobaccoManifestDataset(Dataset):
    def __init__(self, manifest_path, data_root, split, transform=None):
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.records = self._read_records()

    def _read_records(self):
        required_columns = {
            "file_name",
            "split",
            "is_healthy",
            *SOURCE_LABEL_COLUMNS,
        }
        records = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"manifest is missing required columns: {sorted(missing)}")

            for csv_line, row in enumerate(reader, start=2):
                if row["split"] != self.split:
                    continue

                source_target = torch.tensor(
                    [self._binary_value(row[column], column, csv_line) for column in SOURCE_LABEL_COLUMNS],
                    dtype=torch.float32,
                )
                is_healthy = self._binary_value(row["is_healthy"], "is_healthy", csv_line)
                if is_healthy == 1 and source_target.sum().item() != 0:
                    raise ValueError(
                        f"manifest line {csv_line} has is_healthy=1 and a positive disease label"
                    )

                target = torch.cat((source_target[:13], source_target[14:]))
                records.append(TobaccoRecord(file_name=row["file_name"], target=target))
        return records

    @staticmethod
    def _binary_value(value, column, csv_line):
        if value not in ("0", "1"):
            raise ValueError(
                f"manifest line {csv_line} column {column!r} must be 0 or 1, got {value!r}"
            )
        return int(value)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.data_root / record.file_name
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.target.clone()
