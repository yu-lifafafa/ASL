from types import SimpleNamespace

import timm


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
    print("building timm tresnet_m.miil_in21k (pretrained=True)")
    timm_model = timm.create_model("tresnet_m.miil_in21k", pretrained=True)
    timm_state = timm_model.state_dict()
    print(f"timm key count: {len(timm_state)}")
    print(f"timm head keys: {[key for key in timm_state if key.startswith('head.')]}")

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
    print(f"exact key+shape matches: {len(exact)}")
    print(f"missing keys: {missing}")
    print(f"unexpected keys: {unexpected}")
    print(f"shape mismatch keys: {shape_mismatches}")


if __name__ == "__main__":
    main()
