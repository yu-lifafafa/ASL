import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import timm
import torch
from safetensors.torch import load_file

from audit_timm_tresnet_compat import (
    DEFAULT_TIMM_WEIGHT,
    EXPECTED_BACKBONE_TENSORS,
    HEAD_KEYS,
    backbone_state,
    semantic_backbone_audit,
)


DEFAULT_OUTPUT = Path("weights") / "tresnet_m.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert timm TResNet-M ImageNet21K weights for ASL",
    )
    parser.add_argument(
        "--timm-weight",
        type=Path,
        default=DEFAULT_TIMM_WEIGHT,
        help=f"input timm safetensors path (default: {DEFAULT_TIMM_WEIGHT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output ASL checkpoint path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def resolve_from_repo_root(path):
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def build_asl_tresnet_m():
    from src.models import create_model

    model_args = SimpleNamespace(
        model_name="tresnet_m",
        num_classes=18,
        do_bottleneck_head=False,
    )
    return create_model(model_args)


def build_converted_backbone_state(
    timm_state,
    asl_state,
    mapping,
    expected_count=EXPECTED_BACKBONE_TENSORS,
):
    timm_backbone = backbone_state(timm_state)
    asl_backbone = backbone_state(asl_state)

    if (
        len(timm_backbone) != expected_count
        or len(asl_backbone) != expected_count
        or len(mapping) != expected_count
        or set(mapping) != set(timm_backbone)
        or set(mapping.values()) != set(asl_backbone)
        or len(set(mapping.values())) != len(mapping)
    ):
        raise RuntimeError(
            "mapping does not cover every backbone tensor exactly once: "
            f"timm={len(timm_backbone)}, ASL={len(asl_backbone)}, "
            f"mapped={len(mapping)}, unique_targets={len(set(mapping.values()))}, "
            f"expected={expected_count}"
        )

    converted = {}
    for timm_key, asl_key in mapping.items():
        source = timm_backbone[timm_key]
        destination = asl_backbone[asl_key]
        if tuple(source.shape) != tuple(destination.shape):
            raise RuntimeError(
                "mapped tensor shape mismatch: "
                f"{timm_key} {tuple(source.shape)} -> "
                f"{asl_key} {tuple(destination.shape)}"
            )
        converted[asl_key] = source.detach().cpu().clone()

    if any(key in converted for key in HEAD_KEYS):
        raise RuntimeError("converted backbone unexpectedly contains classifier tensors")
    return converted


def exact_backbone_mismatches(converted_state, timm_state, mapping):
    return [
        (timm_key, asl_key)
        for timm_key, asl_key in mapping.items()
        if not torch.equal(
            converted_state[asl_key].detach().cpu(),
            timm_state[timm_key].detach().cpu(),
        )
    ]


def load_and_validate_payload(checkpoint_path, expected_state):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model"}:
        raise RuntimeError("converted checkpoint must contain only the 'model' field")
    checkpoint_state = payload["model"]
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError("converted checkpoint 'model' field must be a state dict")
    if set(checkpoint_state) != set(expected_state):
        raise RuntimeError("serialized checkpoint backbone keys changed during round trip")
    mismatches = [
        key
        for key in expected_state
        if not torch.equal(checkpoint_state[key], expected_state[key])
    ]
    if mismatches:
        raise RuntimeError(
            f"serialized checkpoint tensors changed during round trip: {mismatches}"
        )
    return checkpoint_state


def validate_with_repository_loader(checkpoint_path, timm_state, mapping):
    from train import load_pretrained_backbone

    model = build_asl_tresnet_m()
    before = {
        key: model.state_dict()[key].detach().cpu().clone()
        for key in HEAD_KEYS
    }
    load_pretrained_backbone(model, checkpoint_path)
    loaded_state = model.state_dict()

    backbone_mismatches = [
        (timm_key, asl_key)
        for timm_key, asl_key in mapping.items()
        if not torch.equal(
            loaded_state[asl_key].detach().cpu(),
            timm_state[timm_key].detach().cpu(),
        )
    ]
    head_unchanged = all(
        torch.equal(loaded_state[key].detach().cpu(), before[key])
        for key in HEAD_KEYS
    )
    if backbone_mismatches or not head_unchanged:
        raise RuntimeError(
            "repository loader verification failed: "
            f"backbone_mismatches={backbone_mismatches}, "
            f"head_unchanged={head_unchanged}"
        )
    return backbone_mismatches, head_unchanged


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    cli_args = parse_args()
    timm_weight = resolve_from_repo_root(cli_args.timm_weight)
    output_path = resolve_from_repo_root(cli_args.output)
    if not timm_weight.is_file():
        raise FileNotFoundError(f"timm weights not found: {timm_weight}")

    print("building timm tresnet_m.miil_in21k (pretrained=False)")
    timm_model = timm.create_model(
        "tresnet_m.miil_in21k",
        pretrained=False,
    )
    pretrained_state = load_file(str(timm_weight), device="cpu")
    timm_model.load_state_dict(pretrained_state, strict=True)
    print(f"strict timm checkpoint load: passed ({timm_weight})")

    asl_model = build_asl_tresnet_m()
    mapping_established, mapping = semantic_backbone_audit(
        timm_model,
        asl_model,
    )
    if not mapping_established:
        raise RuntimeError("semantic backbone mapping was not established; no file written")

    timm_state = timm_model.state_dict()
    converted_state = build_converted_backbone_state(
        timm_state,
        asl_model.state_dict(),
        mapping,
    )
    prewrite_mismatches = exact_backbone_mismatches(
        converted_state,
        timm_state,
        mapping,
    )
    if prewrite_mismatches:
        raise RuntimeError(
            f"pre-write exact tensor verification failed: {prewrite_mismatches}"
        )
    print(
        "pre-write exact tensor verification: "
        f"{len(converted_state)}/{EXPECTED_BACKBONE_TENSORS} passed"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        torch.save({"model": converted_state}, temporary_path)

        round_trip_state = load_and_validate_payload(
            temporary_path,
            converted_state,
        )
        serialized_mismatches = exact_backbone_mismatches(
            round_trip_state,
            timm_state,
            mapping,
        )
        if serialized_mismatches:
            raise RuntimeError(
                "serialized tensor verification failed: "
                f"{serialized_mismatches}"
            )
        validate_with_repository_loader(
            temporary_path,
            timm_state,
            mapping,
        )
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if not output_path.is_file():
        raise RuntimeError(f"converted checkpoint was not created: {output_path}")
    load_and_validate_payload(output_path, converted_state)
    backbone_mismatches, head_unchanged = validate_with_repository_loader(
        output_path,
        timm_state,
        mapping,
    )

    exact_count = len(mapping) - len(backbone_mismatches)
    print(f"converted backbone tensors: {len(converted_state)}")
    print(f"exact tensor equality: {exact_count}")
    print(f"backbone mismatches: {backbone_mismatches}")
    print(f"head unchanged: {head_unchanged}")
    print("loader validation passed: True")
    print(f"output path: {output_path}")
    print(f"output size bytes: {output_path.stat().st_size}")
    print(f"SHA256: {sha256_file(output_path)}")


if __name__ == "__main__":
    main()
