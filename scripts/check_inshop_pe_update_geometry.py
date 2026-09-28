#!/usr/bin/env python3
"""Physical relative-update arithmetic check, no model or CUDA."""

import torch
from inspect_inshop_pe_update_geometry import update_geometry

x = {"x": torch.tensor([3.0, 4.0])}
assert update_geometry(x, x) == {
    "initial_norm": 5.0,
    "update_norm": 0.0,
    "relative_update": 0.0,
}
assert update_geometry(x, {"x": 2 * x["x"]}) == {
    "initial_norm": 5.0,
    "update_norm": 5.0,
    "relative_update": 1.0,
}
try:
    update_geometry(x, {"x": torch.tensor([float("nan"), 4.0])})
except AssertionError:
    pass
else:
    raise AssertionError("nonfinite update accepted")
print("PASS zero/scaled update and nonfinite rejection")
