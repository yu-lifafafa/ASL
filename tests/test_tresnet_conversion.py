import pytest
import torch

from convert_timm_tresnet_to_asl import build_converted_backbone_state


def test_build_converted_backbone_state_copies_every_mapped_tensor_exactly():
    timm_state = {
        "body.block.conv.weight": torch.arange(6).reshape(2, 3),
        "body.block.bn.running_mean": torch.tensor([1.0, 2.0]),
        "head.fc.weight": torch.zeros(11221, 2),
        "head.fc.bias": torch.zeros(11221),
    }
    asl_state = {
        "body.block.0.weight": torch.empty(2, 3),
        "body.block.1.running_mean": torch.empty(2),
        "head.fc.weight": torch.empty(18, 2),
        "head.fc.bias": torch.empty(18),
    }
    mapping = {
        "body.block.conv.weight": "body.block.0.weight",
        "body.block.bn.running_mean": "body.block.1.running_mean",
    }

    converted = build_converted_backbone_state(
        timm_state,
        asl_state,
        mapping,
        expected_count=2,
    )

    assert set(converted) == {
        "body.block.0.weight",
        "body.block.1.running_mean",
    }
    assert torch.equal(
        converted["body.block.0.weight"],
        timm_state["body.block.conv.weight"],
    )
    assert torch.equal(
        converted["body.block.1.running_mean"],
        timm_state["body.block.bn.running_mean"],
    )


def test_build_converted_backbone_state_rejects_incomplete_mapping():
    timm_state = {
        "body.a.weight": torch.ones(1),
        "body.b.weight": torch.ones(1),
        "head.fc.weight": torch.zeros(2, 1),
        "head.fc.bias": torch.zeros(2),
    }
    asl_state = {
        "body.a.0.weight": torch.empty(1),
        "body.b.0.weight": torch.empty(1),
        "head.fc.weight": torch.empty(18, 1),
        "head.fc.bias": torch.empty(18),
    }

    with pytest.raises(RuntimeError, match="mapping does not cover every backbone tensor"):
        build_converted_backbone_state(
            timm_state,
            asl_state,
            {"body.a.weight": "body.a.0.weight"},
            expected_count=2,
        )
