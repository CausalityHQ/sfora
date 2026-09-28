#!/usr/bin/env python3
"""Small arithmetic and golden-parity falsifiers for saved-state attribution."""

import numpy as np
from diagnose_inshop_pe_learning_attribution import decomposition, golden_parity

states = {
    "00": np.array([1.0, 2.0]),
    "01": np.array([3.0, 3.0]),
    "10": np.array([2.0, 5.0]),
    "11": np.array([6.0, 8.0]),
}
parts = decomposition(states)
assert np.array_equal(parts["interaction"], [2.0, 2.0])
assert np.array_equal(
    parts["total"], sum(parts[k] for k in ("encoder", "head", "interaction"))
)
vectors = np.eye(3, dtype=np.float32)
quality = {"per_query_r1": [1.0, 0.0], "per_query_ap": [0.5, 0.25]}
golden_parity(vectors, vectors, quality, quality)
for changed, scores in (
    (vectors[::-1], quality),
    (vectors, {**quality, "per_query_ap": [0.5, 0.3]}),
):
    try:
        golden_parity(vectors, changed, quality, scores)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad golden fixture accepted")
print("PASS interaction algebra and golden vector/score mismatch rejection")
