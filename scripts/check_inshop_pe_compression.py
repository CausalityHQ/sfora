#!/usr/bin/env python3
"""Minimal scalar variance fixture for the cached source/PCA diagnostic."""

import torch
from diagnose_inshop_pe_compression import retained_variance

source = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
assert abs(retained_variance(source, torch.tensor([[1.0, 0.0]])) - 0.5) < 1e-12
assert abs(retained_variance(source, torch.eye(2)) - 1) < 1e-12
try:
    retained_variance(source, torch.tensor([[2.0, 0.0]]))
except AssertionError:
    pass
else:
    raise AssertionError("non-orthogonal projection accepted")
print("PASS retained variance .5/1 and invalid basis rejection")
