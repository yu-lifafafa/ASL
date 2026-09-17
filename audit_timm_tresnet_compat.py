from types import SimpleNamespace
from pathlib import Path

import timm
from safetensors.torch import load_file


DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent
    / "weights"
    / "tresnet_m.miil_in21k.safetensors"
)


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


def main():
    weights_path = DEFAULT_WEIGHTS_PATH
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


if __name__ == "__main__":
    main()
