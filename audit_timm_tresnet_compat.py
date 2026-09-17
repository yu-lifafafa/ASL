import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import timm
import torch
from torch import nn
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


def backbone_state(state):
    return {key: tensor for key, tensor in state.items() if key not in HEAD_KEYS}


def module_kind(module):
    if isinstance(module, nn.Conv2d):
        return "conv"
    if isinstance(module, nn.Linear):
        return "linear"

    parameter_roles = {name for name, _ in module.named_parameters(recurse=False)}
    buffer_roles = {name for name, _ in module.named_buffers(recurse=False)}
    if (
        {"weight", "bias"}.issubset(parameter_roles)
        and {"running_mean", "running_var"}.issubset(buffer_roles)
    ):
        return "norm"
    return type(module).__name__


def state_owner_kinds(model, state):
    modules = dict(model.named_modules())
    owner_kinds = {}
    for key in state:
        owner_path, _ = key.rsplit(".", 1)
        if owner_path not in modules:
            raise RuntimeError(f"state key has no owning module: {key}")
        owner_kinds[owner_path] = module_kind(modules[owner_path])
    return modules, owner_kinds


def logical_module_path(owner_path, owner_kinds):
    parts = owner_path.split(".")
    logical_path = owner_path
    for length in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:length])
        descendant_kinds = {
            path: kind
            for path, kind in owner_kinds.items()
            if path == candidate or path.startswith(candidate + ".")
        }
        kind_counts = defaultdict(int)
        for kind in descendant_kinds.values():
            kind_counts[kind] += 1
        if kind_counts == {"conv": 1, "norm": 1}:
            logical_path = candidate
    return logical_path


def semantic_descriptors(model, state):
    modules, owner_kinds = state_owner_kinds(model, state)
    descriptors = {}
    for key, tensor in state.items():
        owner_path, role = key.rsplit(".", 1)
        descriptors[key] = {
            "logical_path": logical_module_path(owner_path, owner_kinds),
            "module_type": module_kind(modules[owner_path]),
            "role": role,
            "shape": tuple(tensor.shape),
        }
    return descriptors


def semantic_backbone_audit(timm_model, asl_model):
    timm_state = backbone_state(timm_model.state_dict())
    asl_state = backbone_state(asl_model.state_dict())
    timm_descriptors = semantic_descriptors(timm_model, timm_state)
    asl_descriptors = semantic_descriptors(asl_model, asl_state)

    mapping = {}
    ambiguous = []
    shape_mismatches = set()

    for timm_key, timm_tensor in timm_state.items():
        if timm_key in asl_state:
            asl_shape = tuple(asl_state[timm_key].shape)
            timm_shape = tuple(timm_tensor.shape)
            if timm_shape == asl_shape:
                mapping[timm_key] = timm_key
                continue
            shape_mismatches.add((timm_key, timm_shape, timm_key, asl_shape))

        timm_descriptor = timm_descriptors[timm_key]
        semantic_candidates = [
            asl_key
            for asl_key, asl_descriptor in asl_descriptors.items()
            if (
                asl_descriptor["logical_path"]
                == timm_descriptor["logical_path"]
                and asl_descriptor["module_type"]
                == timm_descriptor["module_type"]
                and asl_descriptor["role"] == timm_descriptor["role"]
            )
        ]
        shape_candidates = [
            asl_key
            for asl_key in semantic_candidates
            if asl_descriptors[asl_key]["shape"] == timm_descriptor["shape"]
        ]
        if len(shape_candidates) == 1:
            mapping[timm_key] = shape_candidates[0]
        elif len(shape_candidates) > 1:
            ambiguous.append((timm_key, shape_candidates))
        else:
            for asl_key in semantic_candidates:
                shape_mismatches.add(
                    (
                        timm_key,
                        timm_descriptor["shape"],
                        asl_key,
                        asl_descriptors[asl_key]["shape"],
                    )
                )

    target_sources = defaultdict(list)
    for timm_key, asl_key in mapping.items():
        target_sources[asl_key].append(timm_key)
    duplicate_targets = {
        asl_key: timm_keys
        for asl_key, timm_keys in target_sources.items()
        if len(timm_keys) > 1
    }
    mapped_targets = set(mapping.values())
    unmapped_timm = [key for key in timm_state if key not in mapping]
    unmapped_asl = [key for key in asl_state if key not in mapped_targets]
    sorted_shape_mismatches = sorted(shape_mismatches)
    established = (
        len(timm_state)
        == len(asl_state)
        == EXPECTED_BACKBONE_TENSORS
        and len(mapping) == EXPECTED_BACKBONE_TENSORS
        and not unmapped_timm
        and not unmapped_asl
        and not ambiguous
        and not sorted_shape_mismatches
        and not duplicate_targets
    )

    print(f"timm backbone tensors: {len(timm_state)}")
    print(f"ASL backbone tensors: {len(asl_state)}")
    print(f"mapped count: {len(mapping)}")
    print(f"unmapped timm keys: {unmapped_timm}")
    print(f"unmapped ASL keys: {unmapped_asl}")
    print(f"ambiguous mappings: {ambiguous}")
    print(f"shape mismatches: {sorted_shape_mismatches}")
    print(f"duplicate target mappings: {duplicate_targets}")
    print("first 50 final semantic mappings:")
    mapped_in_timm_order = [
        (timm_key, mapping[timm_key])
        for timm_key in timm_state
        if timm_key in mapping
    ]
    for index, (timm_key, asl_key) in enumerate(mapped_in_timm_order[:50], start=1):
        print(
            f"  {index:03d}: {timm_key} {tuple(timm_state[timm_key].shape)}"
            f" -> {asl_key} {tuple(asl_state[asl_key].shape)}"
        )
    print(f"semantic backbone mapping established: {established}")
    return established, mapping


def numerical_feature_audit(timm_model, asl_model, mapping):
    timm_state = timm_model.state_dict()
    asl_state = asl_model.state_dict()
    original_head = {key: asl_state[key].clone() for key in HEAD_KEYS}
    for timm_key, asl_key in mapping.items():
        asl_state[asl_key] = timm_state[timm_key].detach().clone()
    asl_model.load_state_dict(asl_state, strict=True)
    for key in HEAD_KEYS:
        if not torch.equal(asl_model.state_dict()[key], original_head[key]):
            raise RuntimeError(f"ASL classifier changed during temporary load: {key}")

    if not hasattr(timm_model, "forward_features") or not hasattr(
        timm_model,
        "forward_head",
    ):
        print("pre-classifier feature audit unavailable: timm feature API missing")
        return
    if not hasattr(asl_model, "body") or not hasattr(asl_model, "global_pool"):
        print("pre-classifier feature audit unavailable: ASL feature modules missing")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_model = timm_model.to(device).eval()
    asl_model = asl_model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(20260917)
    input_tensor = torch.randn(
        1,
        3,
        224,
        224,
        generator=generator,
        device=device,
    )
    with torch.inference_mode():
        timm_features = timm_model.forward_features(input_tensor)
        timm_pooled = timm_model.forward_head(timm_features, pre_logits=True)
        asl_features = asl_model.body(input_tensor)
        asl_pooled = asl_model.global_pool(asl_features)

    if timm_pooled.shape != asl_pooled.shape:
        print(
            "pre-classifier feature audit unavailable: "
            f"shape mismatch timm={tuple(timm_pooled.shape)} "
            f"ASL={tuple(asl_pooled.shape)}"
        )
        return
    absolute_diff = (timm_pooled.float() - asl_pooled.float()).abs()
    print(f"pre-classifier feature shape: {tuple(timm_pooled.shape)}")
    print(f"pre-classifier max_abs_diff: {absolute_diff.max().item():.12g}")
    print(f"pre-classifier mean_abs_diff: {absolute_diff.mean().item():.12g}")


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

    mapping_established, mapping = semantic_backbone_audit(
        timm_model,
        asl_model,
    )
    if not mapping_established:
        raise SystemExit("semantic backbone mapping was not established")
    numerical_feature_audit(timm_model, asl_model, mapping)


if __name__ == "__main__":
    main()
