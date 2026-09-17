import argparse
from pathlib import Path
from types import SimpleNamespace

import timm
from safetensors.torch import load_file


DEFAULT_TIMM_WEIGHT = Path("weights") / "tresnet_m.miil_in21k.safetensors"
HEAD_KEYS = (
    "head.fc.weight",
    "head.fc.bias",
)
EXPECTED_BACKBONE_TENSORS = 432


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit timm and ASL TResNet-M state_dict compatibility",
    )
    parser.add_argument(
        "--timm-weight",
        type=Path,
        default=DEFAULT_TIMM_WEIGHT,
        help=f"timm safetensors path (default: {DEFAULT_TIMM_WEIGHT})",
    )
    return parser.parse_args()


def compare_state_dicts(source, destination):
    source_keys = set(source)
    destination_keys = set(destination)
    matching = sorted(source_keys & destination_keys)
    missing = sorted(destination_keys - source_keys)
    unexpected = sorted(source_keys - destination_keys)
    shape_mismatches = sorted(
        key
        for key in matching
        if tuple(source[key].shape) != tuple(destination[key].shape)
    )
    exact_matches = [key for key in matching if key not in shape_mismatches]
    return exact_matches, missing, unexpected, shape_mismatches


def ordered_backbone_audit(timm_state, asl_state):
    timm_backbone = [
        (key, tensor) for key, tensor in timm_state.items() if key not in HEAD_KEYS
    ]
    asl_backbone = [
        (key, tensor) for key, tensor in asl_state.items() if key not in HEAD_KEYS
    ]
    ordered_pairs = list(zip(timm_backbone, asl_backbone))
    shape_mismatches = [
        (
            index,
            timm_key,
            tuple(timm_tensor.shape),
            asl_key,
            tuple(asl_tensor.shape),
        )
        for index, ((timm_key, timm_tensor), (asl_key, asl_tensor)) in enumerate(
            ordered_pairs,
            start=1,
        )
        if tuple(timm_tensor.shape) != tuple(asl_tensor.shape)
    ]
    count_matches = (
        len(timm_backbone)
        == len(asl_backbone)
        == EXPECTED_BACKBONE_TENSORS
    )
    mapping_established = count_matches and not shape_mismatches

    print(f"timm ordered backbone tensor count: {len(timm_backbone)}")
    print(f"ASL ordered backbone tensor count: {len(asl_backbone)}")
    print("first 30 ordered backbone pairs:")
    for index, ((timm_key, timm_tensor), (asl_key, asl_tensor)) in enumerate(
        ordered_pairs[:30],
        start=1,
    ):
        print(
            f"  {index:03d}: {timm_key} {tuple(timm_tensor.shape)}"
            f" -> {asl_key} {tuple(asl_tensor.shape)}"
        )
    print(f"ordered backbone shape mismatches: {shape_mismatches}")
    print(f"ordered backbone mapping established: {mapping_established}")
    return mapping_established


def main():
    args = parse_args()
    weights_path = args.timm_weight
    if not weights_path.is_absolute():
        weights_path = Path(__file__).resolve().parent / weights_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"timm weights not found: {weights_path}")

    print("building timm tresnet_m.miil_in21k (pretrained=False)")
    timm_model = timm.create_model(
        "tresnet_m.miil_in21k",
        pretrained=False,
    )
    checkpoint_state = load_file(str(weights_path), device="cpu")
    timm_model.load_state_dict(checkpoint_state, strict=True)
    print(f"strict timm checkpoint load: passed ({weights_path})")

    timm_state = timm_model.state_dict()
    print(f"timm key count: {len(timm_state)}")

    from src.models import create_model

    args = SimpleNamespace(
        model_name="tresnet_m",
        num_classes=18,
        do_bottleneck_head=False,
    )
    asl_model = create_model(args)
    asl_state = asl_model.state_dict()
    exact, missing, unexpected, shape_mismatches = compare_state_dicts(
        timm_state,
        asl_state,
    )

    print(f"ASL key count: {len(asl_state)}")
    print(f"exact same-name+shape matches: {len(exact)}")
    print(f"ASL missing keys: {missing}")
    print(f"timm unexpected keys: {unexpected}")
    print(f"same-name shape mismatch keys: {shape_mismatches}")
    print("classifier shapes:")
    print(f"  timm head.fc.weight: {tuple(timm_state['head.fc.weight'].shape)}")
    print(f"  timm head.fc.bias: {tuple(timm_state['head.fc.bias'].shape)}")
    print(f"  ASL head.fc.weight: {tuple(asl_state['head.fc.weight'].shape)}")
    print(f"  ASL head.fc.bias: {tuple(asl_state['head.fc.bias'].shape)}")

    if not ordered_backbone_audit(timm_state, asl_state):
        raise SystemExit("ordered backbone mapping was not established")


if __name__ == "__main__":
    main()
