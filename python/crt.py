"""The little-endian CRT1 format consumed by src/loader.cpp (stdlib only)."""
import math
import struct
from pathlib import Path

PARAMETER_SHAPES = {
    "linear0.weight": (128, 784),
    "linear0.bias": (128,),
    "linear1.weight": (10, 128),
    "linear1.bias": (10,),
}


def write_crt(path, parameters):
    """parameters maps names to (shape, flat row-major float iterable)."""
    if set(parameters) != set(PARAMETER_SHAPES):
        raise ValueError("Expected exactly the four parameters of the 784→128→10 MLP")
    records = []
    for name, expected in PARAMETER_SHAPES.items():
        shape, values = parameters[name]
        values = list(values)
        if tuple(shape) != expected or len(values) != math.prod(expected):
            raise ValueError(f"Invalid shape or element count for {name}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite parameter: {name}")
        encoded = name.encode("utf-8")
        records.append(
            struct.pack("<I", len(encoded)) + encoded
            + struct.pack("<I", len(shape))
            + struct.pack(f"<{len(shape)}Q", *shape)
            + struct.pack(f"<{len(values)}f", *values)
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"CRT1" + struct.pack("<I", len(records)) + b"".join(records))
