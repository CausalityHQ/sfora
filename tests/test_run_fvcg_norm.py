from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sfora.asgcv_verdict_marginal import (
    collapsed_verdict_coefficient,
    collapsed_verdict_probability,
    torch_collapsed_grpo_verdict_loss,
)
from sfora.fvcg_direct import FvcgStepAuthority
from sfora.fvcg_norm import FvcgNormAuthority

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fvcg_norm.py"
_SPEC = importlib.util.spec_from_file_location("run_fvcg_norm_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _Adapter:
    def __init__(self, semantic_scale: float = 1.0) -> None:
        torch.manual_seed(7)
        self.visual = torch.nn.Linear(4, 4, bias=False)
        self.pooler = torch.nn.Linear(4, 3, bias=False)
        self.language = torch.nn.Linear(4, 2, bias=False)
        self.semantic_scale = semantic_scale
        for parameter in self.language.parameters():
            parameter.requires_grad_(False)

    def vision_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.visual.parameters())

    def language_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.language.parameters())

    def vision_pool(self, value: object) -> torch.Tensor:
        assert isinstance(value, torch.Tensor)
        return torch.nn.functional.normalize(self.pooler(self.visual(value)), dim=-1)

    def direct_collapsed_verdict_backward(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object:
        assert isinstance(pair, torch.Tensor)
        feature = self.visual(pair).mean(dim=0)
        correct = feature[0].float()
        incorrect = feature[1].float()
        loss = torch_collapsed_grpo_verdict_loss(correct, incorrect)
        (loss * self.semantic_scale).backward()
        probability = collapsed_verdict_probability(
            float(correct.detach()), float(incorrect.detach())
        )
        return SimpleNamespace(
            branch_scores=(float(correct.detach()), float(incorrect.detach())),
            correct_probability=probability,
            coefficient=collapsed_verdict_coefficient(probability),
            loss=float(loss.detach()),
            generated_tokens=0,
        )


def _authority() -> FvcgNormAuthority:
    return FvcgNormAuthority(
        base=FvcgStepAuthority(
            source_commit="1" * 40,
            launch_authority_sha256="2" * 64,
            model_revision="3" * 40,
            fixture_sha256="4" * 64,
            selection_seed_sha256="5" * 64,
            semantic_weight=0.25,
            gradient_clip_norm=10_000.0,
            direct_vjp_atol=0.05,
            direct_vjp_rtol=0.01,
        ),
        rho=0.25,
    ).validated()


def _run(scale: float) -> object:
    adapter = _Adapter(scale)
    microbatch = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pair = torch.tensor([[0.2, 0.5, 0.1, 0.7], [0.8, 0.3, 0.4, 0.2]])
    proxies = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = torch.optim.SGD(
        [*adapter.vision_parameters(), *adapter.pooler.parameters(), proxies], lr=1.0e-3
    )
    return _MODULE.run_norm_combined_step(
        adapter,
        authority=_authority(),
        optimizer=optimizer,
        proxies=proxies,
        proxy_labels=torch.arange(4),
        dml_microbatch=microbatch,
        dml_labels=labels,
        semantic_pair=pair,
        correct_completion_ids=(11,),
        incorrect_completion_ids=(22,),
        ordinal=0,
        direct_vjp_errors=(0.0, 0.0),
        memory_psi_full_avg10_ppm=0,
    )


def _context() -> object:
    adapter = _Adapter()
    microbatch = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pair = torch.tensor([[0.2, 0.5, 0.1, 0.7], [0.8, 0.3, 0.4, 0.2]])
    proxies = torch.nn.Parameter(torch.randn(4, 3))
    return _MODULE.direct.PhaseAContext(
        adapter=adapter,
        optimizer=torch.optim.SGD(
            [*adapter.vision_parameters(), *adapter.pooler.parameters(), proxies],
            lr=1.0e-3,
        ),
        proxies=proxies,
        proxy_labels=torch.arange(4),
        dml_microbatch=microbatch,
        dml_labels=labels,
        semantic_pair=pair,
        correct_completion_ids=(11,),
        incorrect_completion_ids=(22,),
        direct_vjp_errors=(0.0, 0.0),
        memory_psi_full_avg10_ppm=0,
    )


def test_norm_combined_step_applies_registered_ratio_and_updates_all_roles() -> None:
    evidence = _run(1.0)
    assert evidence.applied_to_dml_ratio_ppm == 250_000
    assert evidence.base.vision_nonzero_gradient_parameters > 0
    assert evidence.base.pooler_nonzero_gradient_parameters > 0
    assert evidence.base.proxy_nonzero_gradient_parameters > 0
    assert evidence.base.language_gradient_parameters == 0
    assert evidence.base.vision_state_changed is True
    assert evidence.base.pooler_state_changed is True
    assert evidence.base.proxy_state_changed is True


def test_norm_combined_step_is_scale_stable() -> None:
    ordinary = _run(1.0)
    huge = _run(1_000_000.0)
    assert ordinary.applied_to_dml_ratio_ppm == huge.applied_to_dml_ratio_ppm
    assert abs(
        ordinary.base.combined_gradient_cosine_distance_ppm
        - huge.base.combined_gradient_cosine_distance_ppm
    ) <= 2


def test_norm_cli_rejects_network_and_duplicate_options(tmp_path: Path) -> None:
    for option in ("--model-uri", "--aws-profile", "--s3-prefix"):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([option, "x"])
    with pytest.raises(SystemExit, match="duplicate"):
        _MODULE.parse_args(["--source-commit", "1" * 40, "--source-commit", "1" * 40])


def test_norm_phase_a_replays_step_zero_and_reopens_result(tmp_path: Path) -> None:
    raw = _MODULE.run_phase_a(
        authority=_authority(),
        context_factory=lambda _ordinal: _context(),
        output_directory=tmp_path,
    )
    result = _MODULE.validate_fvcg_norm_phase_a_result_bytes(raw)
    assert result.passed is True
    for name in (
        "safe_semantic_norm",
        "raw_dot",
        "projected_dot",
        "applied_semantic_norm",
        "applied_to_dml_ratio_ppm",
    ):
        assert getattr(result.steps[0], name) == getattr(result.repeated_step_zero, name)
    for name in (
        "correct_score",
        "incorrect_score",
        "loss",
        "gradient_sha256",
        "updated_state_sha256",
        "optimizer_state_sha256",
    ):
        assert getattr(result.steps[0].base, name) == getattr(
            result.repeated_step_zero.base, name
        )
    assert (tmp_path / "result.json").read_bytes() == raw
    assert (
        _MODULE.run_phase_a(
            authority=_authority(),
            context_factory=lambda _ordinal: pytest.fail("resume reran science"),
            output_directory=tmp_path,
        )
        == raw
    )
