import importlib.util
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "measure_nuisance_transfer",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_nuisance_transfer.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_transferable_within_subspace_has_large_ratio() -> None:
    rng = np.random.default_rng(7)
    rows, labels = [], []
    dimension = 16
    for label in range(20):
        centre = np.zeros(dimension)
        centre[4:] = rng.normal(size=dimension - 4)
        for _ in range(8):
            row = centre.copy()
            row[:4] = 2.5 * rng.normal(size=4)
            row[4:] += 0.05 * rng.normal(size=dimension - 4)
            rows.append(row)
            labels.append(label)
    result = _MODULE.measure({"embeddings": np.asarray(rows), "labels": np.asarray(labels)}, 0)
    assert result["a_to_b"]["observed"]["4"]["rho"] > 2.0
    assert result["b_to_a"]["observed"]["4"]["rho"] > 2.0
