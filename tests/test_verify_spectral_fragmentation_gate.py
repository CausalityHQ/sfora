from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_spectral_fragmentation_gate.py"
SPEC = importlib.util.spec_from_file_location("verify_spectral_fragmentation_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture() -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    # Label 0 has two disconnected mutual-nearest-neighbour pairs. Label 1 is a
    # connected 1-NN chain; both provide an exact-size stratum of size four.
    fragmented = np.asarray([[1, 0], [0.99, 0.01], [-1, 0], [-0.99, -0.01]], dtype=float)
    connected = np.asarray([[1, 0], [0.8, 0.6], [0, 1], [-0.8, 0.6]], dtype=float)
    embeddings = np.vstack([fragmented, connected])
    labels = np.asarray([0] * 4 + [1] * 4)
    result: dict[str, float | int] = {
        "eligible_classes": 2,
        "one_nn_fragmented_count": 1,
        "one_nn_fragmented_fraction": 0.5,
        "size_matched_fragmented_minus_connected_top1_points": 2.0,
    }
    return embeddings, labels, result


def test_verify_reports_both_arms_and_common_strata() -> None:
    embeddings, labels, result = _fixture()
    payload = MODULE.verify(
        embeddings,
        labels,
        result,
        minimum_per_arm=1,
        minimum_effect_points=1.0,
    )
    assert payload["fragmented_classes"] == 1
    assert payload["connected_classes"] == 1
    assert payload["common_exact_size_strata"] == [4]
    assert payload["registered_gate_pass"] is True


def test_verify_fails_closed_on_diagnostic_disagreement() -> None:
    embeddings, labels, result = _fixture()
    result["one_nn_fragmented_count"] = 0
    with pytest.raises(ValueError, match="fragmented count differs"):
        MODULE.verify(
            embeddings,
            labels,
            result,
            minimum_per_arm=1,
            minimum_effect_points=1.0,
        )
