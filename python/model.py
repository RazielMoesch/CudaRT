"""Exact architecture and preprocessing contract of examples/mnist.cpp."""
from collections.abc import Mapping

import torch
from torch import nn

ARCHITECTURE = "Flatten -> Linear(784,128) -> ReLU -> Linear(128,10)"
PREPROCESSING = "uint8 / 255.0"


def make_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def load_model(path, prefixes=None):
    # State dictionaries only; never unpickle an arbitrary saved Module.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a dictionary")
    if metadata.get("preprocessing", PREPROCESSING) != PREPROCESSING:
        raise ValueError("Checkpoint preprocessing does not match pixel / 255.0")
    if metadata.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("Checkpoint architecture metadata does not match the C++ model")
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError("Invalid state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    choices = [prefixes] if prefixes else [("1", "3"), ("fc1", "fc2"), ("linear0", "linear1")]
    matched = [pair for pair in choices if all(f"{p}.{kind}" in state for p in pair for kind in ("weight", "bias"))]
    if len(matched) != 1:
        raise ValueError("Cannot identify exactly two layers; use --prefixes FIRST SECOND")
    first, second = matched[0]
    mapping = {f"{dst}.{kind}": state[f"{src}.{kind}"]
               for dst, src in (("1", first), ("3", second)) for kind in ("weight", "bias")}
    used = {f"{prefix}.{kind}" for prefix in (first, second) for kind in ("weight", "bias")}
    if set(state) != used:
        raise ValueError("Checkpoint has extra parameters; use a checkpoint of the matching MLP")
    model = make_model()
    for name, expected in model.state_dict().items():
        value = mapping[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
            raise ValueError(f"Wrong tensor shape for {name}; expected {tuple(expected.shape)}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite FP32 weights")
    model.load_state_dict(mapping, strict=True)
    return model.eval(), metadata


def export_model(model, path):
    from crt import write_crt
    parameters = {}
    for i, index in enumerate((1, 3)):
        layer = model[index]
        for kind in ("weight", "bias"):
            tensor = getattr(layer, kind).detach().cpu().contiguous()
            parameters[f"linear{i}.{kind}"] = (tuple(tensor.shape), tensor.flatten().tolist())
    write_crt(path, parameters)
