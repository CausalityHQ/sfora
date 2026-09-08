from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_sop_relational_linear.py"
_SPEC = importlib.util.spec_from_file_location("probe_sop_relational_linear", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_score_symmetric_excludes_self_and_matches_hand_derived_metrics() -> None:
    values = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.99, 0.10],
                [0.10, 0.99],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )

    result = _MODULE.score_symmetric(
        values,
        (1, 1, 2, 2),
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert result["map_at_r"] == 0.0
    assert result["r1"] == 0.0
    assert result["per_query_ap"] == (0.0, 0.0, 0.0, 0.0)
    assert result["per_query_r1"] == (0.0, 0.0, 0.0, 0.0)


def test_score_symmetric_packed_and_float_agree_on_exact_geometry() -> None:
    values = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = (1, 1, 2, 2)

    floating = _MODULE.score_symmetric(
        values, labels, candidate_width=2, device=torch.device("cpu")
    )
    packed = _MODULE.score_symmetric(
        pack_int8_unit_embeddings(values),
        labels,
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert floating == packed
    assert floating["map_at_r"] == 1.0
    assert floating["r1"] == 1.0


@pytest.mark.parametrize(
    ("labels", "candidate_width", "match"),
    (
        ((1, 2, 2), 2, "singleton"),
        ((1, 1, 1, 2, 2, 2), 1, "candidate"),
        ((1, 1, 2, 2), 4, "candidate"),
    ),
)
def test_score_symmetric_rejects_invalid_positive_or_candidate_authority(
    labels: tuple[int, ...], candidate_width: int, match: str
) -> None:
    values = F.normalize(
        torch.arange(1, len(labels) * 3 + 1).reshape(len(labels), 3).float(), dim=1
    )
    with pytest.raises(ValueError, match=match):
        _MODULE.score_symmetric(
            values,
            labels,
            candidate_width=candidate_width,
            device=torch.device("cpu"),
        )
