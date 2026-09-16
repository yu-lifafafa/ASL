import csv

import pytest
import torch
from torchvision import transforms

from src.datasets import SOURCE_LABEL_COLUMNS, TobaccoManifestDataset
from train import (
    build_datasets,
    build_transforms,
    calculate_mAP_from_logits,
    load_pretrained_backbone,
    parse_args,
    prepare_target,
    select_validation_model,
)


def _write_manifest(path):
    fieldnames = ["file_name", "split", "is_healthy", *SOURCE_LABEL_COLUMNS]
    rows = []
    for split, source_index in (("train", 0), ("val", 14), ("test", 18)):
        row = {
            "file_name": f"images/{split}.png",
            "split": split,
            "is_healthy": "0",
            **{column: "0" for column in SOURCE_LABEL_COLUMNS},
        }
        row[SOURCE_LABEL_COLUMNS[source_index]] = "1"
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_tobacco_cli_uses_18_classes_and_384_without_coco_data_argument(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    args = parse_args(
        [
            "--dataset",
            "tobacco",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(tmp_path),
        ]
    )

    assert args.dataset == "tobacco"
    assert args.num_classes == 18
    assert args.image_size == 384
    assert args.manifest == str(manifest_path)
    assert args.data_root == str(tmp_path)


def test_tobacco_datasets_use_manifest_train_and_val_splits(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)
    args = parse_args(
        [
            "--dataset",
            "tobacco",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(tmp_path),
        ]
    )
    train_transform, val_transform = build_transforms(args)

    train_dataset, val_dataset = build_datasets(args, train_transform, val_transform)

    assert isinstance(train_dataset, TobaccoManifestDataset)
    assert isinstance(val_dataset, TobaccoManifestDataset)
    assert train_dataset.split == "train"
    assert val_dataset.split == "val"
    assert [record.file_name for record in train_dataset.records] == ["images/train.png"]
    assert [record.file_name for record in val_dataset.records] == ["images/val.png"]


def test_tobacco_transforms_match_baseline_protocol(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)
    args = parse_args(
        [
            "--dataset",
            "tobacco",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(tmp_path),
        ]
    )

    train_transform, val_transform = build_transforms(args)

    assert [type(step) for step in train_transform.transforms] == [
        transforms.Resize,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
    ]
    assert [type(step) for step in val_transform.transforms] == [
        transforms.Resize,
        transforms.ToTensor,
    ]
    assert train_transform.transforms[0].size == (384, 384)
    assert train_transform.transforms[1].p == 0.5
    assert val_transform.transforms[0].size == (384, 384)


def test_tobacco_target_is_not_reduced_like_coco_target():
    tobacco_target = torch.tensor(
        [[1.0, 0.0] + [0.0] * 16, [0.0, 1.0] + [0.0] * 16],
        dtype=torch.float32,
    )
    coco_target = torch.tensor(
        [[[1, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [0, 1]]],
        dtype=torch.long,
    )

    prepared_tobacco = prepare_target(tobacco_target, dataset="tobacco")
    prepared_coco = prepare_target(coco_target, dataset="coco")

    assert torch.equal(prepared_tobacco, tobacco_target)
    assert prepared_tobacco.shape == (2, 18)
    assert torch.equal(prepared_coco, torch.tensor([[1, 1], [1, 1]]))


def test_validation_map_accepts_raw_logits_and_targets_with_shape_n_by_18():
    targets = torch.eye(18, dtype=torch.float32)
    logits = torch.full((18, 18), -5.0)
    logits[torch.arange(18), torch.arange(18)] = 5.0

    score = calculate_mAP_from_logits(targets, logits)

    assert score == pytest.approx(100.0)


def test_best_validation_model_selects_ema_when_ema_map_is_higher():
    regular = torch.nn.Linear(2, 2)
    ema = type("Ema", (), {"module": torch.nn.Linear(2, 2)})()

    selected_model, selected_name, selected_score = select_validation_model(
        regular,
        ema,
        {"regular": 71.0, "ema": 72.0},
    )

    assert selected_model is ema.module
    assert selected_name == "ema"
    assert selected_score == 72.0


class _TinyClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.body = torch.nn.Linear(3, 4)
        self.head = torch.nn.Module()
        self.head.fc = torch.nn.Linear(4, num_classes)


def test_pretrained_loader_loads_backbone_and_reinitializes_old_head(tmp_path):
    model = _TinyClassifier(num_classes=18)
    original_head = model.head.fc.weight.detach().clone()
    old_model = _TinyClassifier(num_classes=80)
    with torch.no_grad():
        old_model.body.weight.fill_(3.0)
        old_model.body.bias.fill_(2.0)
    checkpoint_path = tmp_path / "pretrained.pth"
    torch.save({"model": old_model.state_dict()}, checkpoint_path)

    load_pretrained_backbone(model, checkpoint_path)

    assert torch.equal(model.body.weight, torch.full_like(model.body.weight, 3.0))
    assert torch.equal(model.body.bias, torch.full_like(model.body.bias, 2.0))
    assert torch.equal(model.head.fc.weight, original_head)


def test_pretrained_loader_rejects_missing_backbone_tensor(tmp_path):
    model = _TinyClassifier(num_classes=18)
    checkpoint = {"model": {"body.weight": torch.ones_like(model.body.weight)}}
    checkpoint_path = tmp_path / "incomplete.pth"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing_backbone"):
        load_pretrained_backbone(model, checkpoint_path)
