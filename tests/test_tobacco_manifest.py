import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.tobacco_manifest import (
    SOURCE_LABEL_COLUMNS,
    TobaccoManifestDataset,
)


def _row(file_name, split, positives=(), is_healthy=0):
    row = {
        "file_name": file_name,
        "split": split,
        "is_healthy": str(is_healthy),
    }
    row.update({column: "0" for column in SOURCE_LABEL_COLUMNS})
    for source_index in positives:
        row[SOURCE_LABEL_COLUMNS[source_index]] = "1"
    return row


def _write_manifest(tmp_path, rows):
    manifest_path = tmp_path / "manifest.csv"
    fieldnames = ["file_name", "split", "is_healthy", *SOURCE_LABEL_COLUMNS]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_image(data_root, relative_path, color=(10, 20, 30)):
    image_path = data_root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(image_path)


@pytest.mark.parametrize(
    ("source_index", "model_index"),
    [(0, 0), (12, 12), (14, 13), (18, 17)],
)
def test_source_boundaries_map_to_expected_model_indices(tmp_path, source_index, model_index):
    relative_path = "images/sample.png"
    _write_image(tmp_path, relative_path)
    manifest_path = _write_manifest(
        tmp_path, [_row(relative_path, "train", positives=(source_index,))]
    )

    dataset = TobaccoManifestDataset(manifest_path, tmp_path, "train", transform=None)
    _, target = dataset[0]

    expected = torch.zeros(18, dtype=torch.float32)
    expected[model_index] = 1.0
    assert torch.equal(target, expected)


def test_source_13_is_excluded(tmp_path):
    relative_path = "images/excluded.png"
    _write_image(tmp_path, relative_path)
    manifest_path = _write_manifest(
        tmp_path, [_row(relative_path, "train", positives=(13,))]
    )

    dataset = TobaccoManifestDataset(manifest_path, tmp_path, "train", transform=None)
    _, target = dataset[0]

    assert torch.equal(target, torch.zeros(18, dtype=torch.float32))


def test_multi_label_mapping_preserves_all_included_positives(tmp_path):
    relative_path = "images/multi.png"
    _write_image(tmp_path, relative_path)
    manifest_path = _write_manifest(
        tmp_path, [_row(relative_path, "train", positives=(0, 12, 13, 14, 18))]
    )

    dataset = TobaccoManifestDataset(manifest_path, tmp_path, "train", transform=None)
    _, target = dataset[0]

    expected = torch.zeros(18, dtype=torch.float32)
    expected[[0, 12, 13, 17]] = 1.0
    assert torch.equal(target, expected)


def test_healthy_sample_has_zero_target_with_float32_shape_18(tmp_path):
    relative_path = "images/healthy.png"
    _write_image(tmp_path, relative_path)
    manifest_path = _write_manifest(
        tmp_path, [_row(relative_path, "train", is_healthy=1)]
    )

    dataset = TobaccoManifestDataset(manifest_path, tmp_path, "train", transform=None)
    _, target = dataset[0]

    assert target.dtype == torch.float32
    assert target.shape == (18,)
    assert target.sum().item() == 0.0


def test_healthy_sample_with_disease_positive_raises_clear_error(tmp_path):
    manifest_path = _write_manifest(
        tmp_path, [_row("images/conflict.png", "train", positives=(0,), is_healthy=1)]
    )

    with pytest.raises(ValueError, match="is_healthy=1.*disease label"):
        TobaccoManifestDataset(manifest_path, tmp_path, "train", transform=None)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_split_filtering_is_exact(tmp_path, split):
    rows = []
    for current_split in ("train", "val", "test"):
        relative_path = f"images/{current_split}.png"
        _write_image(tmp_path, relative_path)
        rows.append(_row(relative_path, current_split, positives=(0,)))
    manifest_path = _write_manifest(tmp_path, rows)

    dataset = TobaccoManifestDataset(manifest_path, tmp_path, split, transform=None)

    assert len(dataset) == 1
    assert dataset.records[0].file_name == f"images/{split}.png"


def test_dataloader_batches_targets_as_b_by_18_float32(tmp_path):
    rows = []
    for index in range(2):
        relative_path = f"images/sample_{index}.png"
        _write_image(tmp_path, relative_path)
        rows.append(_row(relative_path, "train", positives=(index,)))
    manifest_path = _write_manifest(tmp_path, rows)
    dataset = TobaccoManifestDataset(
        manifest_path,
        tmp_path,
        "train",
        transform=transforms.ToTensor(),
    )

    images, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert images.shape == (2, 3, 8, 8)
    assert targets.shape == (2, 18)
    assert targets.dtype == torch.float32


def test_official_tresnet_asl_forward_backward_has_finite_nonzero_gradients(tmp_path):
    from src.loss_functions.losses import AsymmetricLoss
    from src.models import create_model

    rows = []
    for index, positives in enumerate(((0, 14), (12, 18))):
        relative_path = f"images/integration_{index}.png"
        _write_image(tmp_path, relative_path, color=(40 + index, 80, 120))
        rows.append(_row(relative_path, "train", positives=positives))
    manifest_path = _write_manifest(tmp_path, rows)
    dataset = TobaccoManifestDataset(
        manifest_path,
        tmp_path,
        "train",
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        ),
    )
    images, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    args = SimpleNamespace(
        model_name="tresnet_m",
        num_classes=18,
        do_bottleneck_head=False,
    )
    model = create_model(args).cuda().train()
    logits = model(images.cuda())
    criterion = AsymmetricLoss(
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    )
    loss = criterion(logits, targets.cuda())
    loss.backward()

    backbone_grad = model.body.conv1[0].weight.grad
    classifier_grad = model.head.fc.weight.grad
    assert logits.shape == (2, 18)
    assert torch.isfinite(loss)
    assert backbone_grad is not None
    assert torch.isfinite(backbone_grad).all()
    assert torch.count_nonzero(backbone_grad) > 0
    assert classifier_grad is not None
    assert torch.isfinite(classifier_grad).all()
    assert torch.count_nonzero(classifier_grad) > 0
