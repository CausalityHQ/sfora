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
from sfora.fvcg_direct import FvcgStepAuthority, select_stratum_pair

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fvcg_direct.py"
_SPEC = importlib.util.spec_from_file_location("run_fvcg_direct_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _authority() -> FvcgStepAuthority:
    return FvcgStepAuthority(
        source_commit="1" * 40,
        launch_authority_sha256="2" * 64,
        model_revision="3" * 40,
        fixture_sha256="4" * 64,
        selection_seed_sha256="5" * 64,
        semantic_weight=0.25,
        gradient_clip_norm=10_000.0,
        direct_vjp_atol=1.0e-5,
        direct_vjp_rtol=1.0e-4,
    ).validated()


class _FakeAdapter:
    def __init__(self) -> None:
        torch.manual_seed(7)
        self.visual = torch.nn.Linear(4, 4, bias=False)
        self.pooler = torch.nn.Linear(4, 3, bias=False)
        self.language = torch.nn.Linear(4, 2, bias=False)
        for parameter in self.language.parameters():
            parameter.requires_grad_(False)
        self.calls: list[str] = []

    def vision_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.visual.parameters())

    def language_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.language.parameters())

    def vision_pool(self, microbatch: object) -> torch.Tensor:
        assert isinstance(microbatch, torch.Tensor)
        self.calls.append("dml-forward")
        return torch.nn.functional.normalize(self.pooler(self.visual(microbatch)), dim=-1)

    def direct_collapsed_verdict_backward(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object:
        assert isinstance(pair, torch.Tensor)
        assert correct_completion_ids == (11,)
        assert incorrect_completion_ids == (22,)
        self.calls.append("semantic-backward")
        feature = self.visual(pair).mean(dim=0)
        correct = feature[0].float()
        incorrect = feature[1].float()
        loss = torch_collapsed_grpo_verdict_loss(correct, incorrect)
        loss.backward()
        probability = collapsed_verdict_probability(
            float(correct.detach()), float(incorrect.detach())
        )
        coefficient = collapsed_verdict_coefficient(probability)
        return SimpleNamespace(
            correct_probability=probability,
            coefficient=coefficient,
            branch_scores=(float(correct.detach()), float(incorrect.detach())),
            loss=float(loss.detach()),
            branch_count=2,
            generated_tokens=0,
            vision_nonzero_gradient_parameters=sum(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in self.visual.parameters()
            ),
            language_gradient_parameters=0,
            finite=True,
            gradient_sha256="6" * 64,
            boundary_gradient_sha256="7" * 64,
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    microbatch = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pair = torch.tensor([[0.2, 0.5, 0.1, 0.7], [0.8, 0.3, 0.4, 0.2]])
    proxies = torch.nn.Parameter(torch.randn(4, 3))
    proxy_labels = torch.arange(4)
    return microbatch, labels, pair, proxies, proxy_labels


def test_combined_step_accumulates_losses_clips_once_and_updates_trainable_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    microbatch, labels, pair, proxies, proxy_labels = _inputs()
    optimizer = torch.optim.AdamW(
        [*adapter.vision_parameters(), *adapter.pooler.parameters(), proxies], lr=1.0e-3
    )
    clip_calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters: object, maximum: float) -> torch.Tensor:
        clip_calls.append(maximum)
        return real_clip(parameters, maximum)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    before_visual = _MODULE.parameter_state_sha256(adapter.vision_parameters())
    before_pooler = _MODULE.parameter_state_sha256(tuple(adapter.pooler.parameters()))
    before_proxy = _MODULE.parameter_state_sha256((proxies,))

    evidence = _MODULE.run_combined_step(
        adapter,
        authority=_authority(),
        optimizer=optimizer,
        proxies=proxies,
        proxy_labels=proxy_labels,
        dml_microbatch=microbatch,
        dml_labels=labels,
        semantic_pair=pair,
        correct_completion_ids=(11,),
        incorrect_completion_ids=(22,),
        ordinal=0,
        direct_vjp_errors=(0.0, 0.0),
        memory_psi_full_avg10_ppm=0,
    )

    assert adapter.calls == ["dml-forward", "semantic-backward"]
    assert clip_calls == [_authority().gradient_clip_norm]
    assert evidence.selected_pair == select_stratum_pair(
        tuple(range(8)), seed_sha256=_authority().selection_seed_sha256, step=0
    )
    assert evidence.vision_nonzero_gradient_parameters > 0
    assert evidence.pooler_nonzero_gradient_parameters > 0
    assert evidence.proxy_nonzero_gradient_parameters > 0
    assert evidence.language_gradient_parameters == 0
    assert evidence.generated_tokens == 0
    assert evidence.gradients_finite is True
    assert evidence.semantic_gradient_norm > 0.0
    assert _MODULE.parameter_state_sha256(adapter.vision_parameters()) != before_visual
    assert _MODULE.parameter_state_sha256(tuple(adapter.pooler.parameters())) != before_pooler
    assert _MODULE.parameter_state_sha256((proxies,)) != before_proxy
    assert all(parameter.grad is None for parameter in adapter.language_parameters())


def test_combined_step_rejects_language_gradient_and_nonfinite_field() -> None:
    adapter = _FakeAdapter()
    microbatch, labels, pair, proxies, proxy_labels = _inputs()
    optimizer = torch.optim.SGD(
        [*adapter.vision_parameters(), *adapter.pooler.parameters(), proxies], lr=1.0e-3
    )
    adapter.language.weight.requires_grad_(True)
    adapter.language.weight.grad = torch.ones_like(adapter.language.weight)
    with pytest.raises(ValueError, match="language gradient"):
        _MODULE.run_combined_step(
            adapter,
            authority=_authority(),
            optimizer=optimizer,
            proxies=proxies,
            proxy_labels=proxy_labels,
            dml_microbatch=microbatch,
            dml_labels=labels,
            semantic_pair=pair,
            correct_completion_ids=(11,),
            incorrect_completion_ids=(22,),
            ordinal=0,
            direct_vjp_errors=(0.0, 0.0),
            memory_psi_full_avg10_ppm=0,
        )
