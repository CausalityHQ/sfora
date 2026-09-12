from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from sfora.residual_quantization import ResidualQuantizationSpec, ResidualQuantizer

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_additive_codec_preflight.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_additive_codec_preflight", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity_quantizer() -> ResidualQuantizer:
    return ResidualQuantizer.from_codebooks(
        ResidualQuantizationSpec(dimensions=2, stages=1, codebook_size=4),
        torch.tensor(
            [[[-1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]]],
            dtype=torch.float32,
        ),
    )


def test_asymmetric_sop_score_uses_float_queries_and_additive_coded_gallery() -> None:
    subject = _subject()
    quantizer = _identity_quantizer()
    values = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.9, 0.1],
            [0.0, -1.0],
            [0.1, -0.9],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    values = torch.nn.functional.normalize(values, dim=1)

    score = subject.score_asymmetric_additive(
        values,
        quantizer.hard_encode(values),
        quantizer,
        (1, 1, 2, 2, 3, 3, 4, 4),
        device=torch.device("cpu"),
    )

    assert score["map_at_r"] == pytest.approx(1.0)
    assert score["r1"] == pytest.approx(1.0)
    assert score["per_query_ap"] == [1.0] * 8


@pytest.mark.parametrize(
    ("map_at_r", "classification"),
    (
        (0.5709, "greedy-residual-failed"),
        (0.5830, "greedy-residual-viable"),
        (0.5860, "greedy-residual-target-passed"),
    ),
)
def test_additive_preflight_decision_uses_frozen_rate_efficiency_gates(
    map_at_r: float, classification: str
) -> None:
    subject = _subject()
    decision = subject.additive_preflight_decision(
        map_at_r=map_at_r,
        r1=0.83,
        pq32_map_at_r=0.58194932,
        pq32_r1=0.82763569,
    )
    assert decision["classification"] == classification
    assert decision["gates"] == {
        "match_pq32_map_at_r": 0.58194932,
        "recover_0_015_map_at_r": 0.58563,
        "match_pq32_r1": 0.82763569,
    }


def test_additive_preflight_rejects_output_aliases_before_fitting(tmp_path: Path) -> None:
    subject = _subject()
    output = (tmp_path / "result.json").absolute()
    with pytest.raises(ValueError, match="output authority"):
        subject.validate_output_paths(output, output)


def test_additive_preflight_requires_r1_to_pass_the_target() -> None:
    subject = _subject()
    decision = subject.additive_preflight_decision(
        map_at_r=0.59,
        r1=0.82,
        pq32_map_at_r=0.581,
        pq32_r1=0.827,
    )
    assert decision["classification"] == "greedy-residual-viable"
    assert decision["passed"] is False


def test_additive_preflight_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--execute-additive-codec-preflight" in result.stderr
