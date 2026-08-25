from __future__ import annotations

import copy

import pytest
import torch
from torch.optim import _muon

from sfora import unicom_proxy_muon as module


def _phase1_rows(
    adamw_means: tuple[float, ...],
    proxy_muon_means: tuple[float, ...],
) -> list[dict[str, object]]:
    learning_rates = (0.000025, 0.00005, 0.0001, 0.0002, 0.0004)
    rows: list[dict[str, object]] = []
    for optimizer, means in (
        ("adamw", adamw_means),
        ("proxy_muon", proxy_muon_means),
    ):
        for learning_rate, mean in zip(learning_rates, means, strict=True):
            for fit_seed in (0, 1, 2):
                rows.append(
                    {
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "fit_seed": fit_seed,
                        "step_64_diagnostic_mean": mean,
                    }
                )
    return rows


def test_frozen_protocol_constants() -> None:
    assert module.LR_GRID == (0.000025, 0.00005, 0.0001, 0.0002, 0.0004)
    assert module.PHASE1_SEEDS == (0, 1, 2)
    assert module.PHASE2_SEEDS == (3, 4, 5)
    assert module.RETAINED_STEPS == (0, 64, 128, 192, 256, 307, 384, 435, 512)
    assert module.VALIDATION_STEPS == (307, 435, 512)


def test_select_learning_rate_uses_three_seed_mean_and_smaller_lr_tie() -> None:
    rows = _phase1_rows(
        adamw_means=(3.0, 2.0, 1.0, 1.0, 4.0),
        proxy_muon_means=(4.0, 3.0, 2.0, 1.0, 5.0),
    )

    adamw = module.select_learning_rate(rows, optimizer="adamw")
    proxy_muon = module.select_learning_rate(rows, optimizer="proxy_muon")

    assert adamw.learning_rate == 0.0001
    assert adamw.mean_step_64_loss == 1.0
    assert adamw.interior is True
    assert proxy_muon.learning_rate == 0.0002
    assert proxy_muon.mean_step_64_loss == 1.0
    assert proxy_muon.interior is True


def test_select_learning_rate_marks_registered_boundaries() -> None:
    rows = _phase1_rows(
        adamw_means=(0.5, 1.0, 2.0, 3.0, 4.0),
        proxy_muon_means=(4.0, 3.0, 2.0, 1.0, 0.5),
    )

    assert module.select_learning_rate(rows, optimizer="adamw").interior is False
    assert module.select_learning_rate(rows, optimizer="proxy_muon").interior is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[-1])),
        lambda rows: rows.__setitem__(0, {**rows[0], "fit_seed": True}),
        lambda rows: rows.__setitem__(0, {**rows[0], "learning_rate": 1}),
        lambda rows: rows.__setitem__(0, {**rows[0], "step_64_diagnostic_mean": float("nan")}),
        lambda rows: rows.__setitem__(slice(0, 2), reversed(rows[:2])),
    ),
)
def test_select_learning_rate_rejects_incomplete_or_noncanonical_rows(mutation) -> None:
    rows = _phase1_rows(
        adamw_means=(3.0, 2.0, 1.0, 1.0, 4.0),
        proxy_muon_means=(4.0, 3.0, 2.0, 1.0, 5.0),
    )
    mutation(rows)

    with pytest.raises(ValueError):
        module.select_learning_rate(rows, optimizer="adamw")


def _adamw_reference_rows(
    selected_loss: float,
    anchor_loss: float,
) -> list[dict[str, object]]:
    return [
        {
            "variant": "adamw_selected",
            "learning_rate": 0.0002,
            "fit_seed": 3,
            "step_512_diagnostic_mean": selected_loss,
            "step_512_accuracy": 0.9,
        },
        {
            "variant": "adamw_anchor",
            "learning_rate": 0.0001,
            "fit_seed": 3,
            "step_512_diagnostic_mean": anchor_loss,
            "step_512_accuracy": 0.8,
        },
    ]


def test_adamw_reference_uses_loss_and_smaller_lr_tie_not_accuracy() -> None:
    reference = module.select_adamw_reference(
        _adamw_reference_rows(selected_loss=1.0, anchor_loss=1.0),
        selected_learning_rate=0.0002,
        fit_seed=3,
    )

    assert reference.variant == "adamw_anchor"
    assert reference.learning_rate == 0.0001
    assert reference.step_512_diagnostic_mean == 1.0
    assert reference.step_512_accuracy == 0.8


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    (
        (0.898, 0.9, True),
        (0.8979999999999999, 0.9, False),
    ),
)
def test_accuracy_noninferiority_uses_registered_negative_point002_boundary(
    candidate: float, reference: float, expected: bool
) -> None:
    assert module.accuracy_noninferior(candidate, reference) is expected


def _decision_evidence() -> dict[str, object]:
    return {
        "structural_valid": True,
        "adamw_selected_lr_interior": True,
        "proxy_muon_selected_lr_interior": True,
        "proxy_muon_reach_steps": {3: 307, 4: 307, 5: 307},
        "proxy_muon_noninferior_at_reach": {3: True, 4: True, 5: True},
        "proxy_muon_step512_noninferior": {3: True, 4: True, 5: True},
        "fp32_reach_steps": {3: 307, 4: 307, 5: 307},
        "fp32_noninferior_at_reach": {3: True, 4: True, 5: True},
        "fp32_step512_noninferior": {3: True, 4: True, 5: True},
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda evidence: evidence.__setitem__("structural_valid", False), "STRUCTURAL_FAILURE"),
        (
            lambda evidence: evidence.__setitem__("adamw_selected_lr_interior", False),
            "UNRESOLVED_LR_BOUNDARY",
        ),
        (lambda _evidence: None, "PROCEED_TRAINING"),
        (
            lambda evidence: evidence["proxy_muon_reach_steps"].__setitem__(3, 435),
            "ROUTE_FP32_ORTHOGONALIZATION",
        ),
        (
            lambda evidence: (
                evidence["proxy_muon_reach_steps"].__setitem__(3, 435),
                evidence["fp32_reach_steps"].__setitem__(3, 435),
            ),
            "ROUTE_MATCHED_LR",
        ),
        (
            lambda evidence: (
                evidence["proxy_muon_reach_steps"].__setitem__(3, ">512"),
                evidence["fp32_reach_steps"].__setitem__(3, 435),
            ),
            "CLOSE_PROXY_MUON",
        ),
    ),
)
def test_decision_uses_exact_registered_cascade(mutation, expected: str) -> None:
    evidence = _decision_evidence()
    mutation(evidence)

    assert module.decide_proxy_muon_f0(evidence) == expected


def test_structural_failure_cannot_bypass_malformed_decision_evidence() -> None:
    evidence = _decision_evidence()
    evidence["structural_valid"] = False
    evidence["fp32_step512_noninferior"] = {3: True, 4: True, 5: 1}

    with pytest.raises(ValueError):
        module.decide_proxy_muon_f0(evidence)


@pytest.mark.parametrize("optimizer_name", ("adamw", "proxy_muon"))
def test_build_head_optimizer_uses_exact_registered_constructor_and_fresh_state(
    optimizer_name: str,
) -> None:
    first_head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))
    second_head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))

    first = module.build_head_optimizer(first_head, optimizer_name, 0.0002)
    second = module.build_head_optimizer(second_head, optimizer_name, 0.0002)

    assert first is not second
    assert first.param_groups[0]["params"] == [first_head]
    assert second.param_groups[0]["params"] == [second_head]
    assert first.state == {}
    assert second.state == {}
    if optimizer_name == "adamw":
        assert type(first) is torch.optim.AdamW
        assert first.param_groups[0]["lr"] == 0.0002
        assert first.param_groups[0]["betas"] == (0.9, 0.999)
        assert first.param_groups[0]["eps"] == 1e-8
        assert first.param_groups[0]["weight_decay"] == 0.0
    else:
        assert type(first) is torch.optim.Muon
        group = first.param_groups[0]
        assert group["lr"] == 0.0002
        assert group["momentum"] == 0.95
        assert group["nesterov"] is True
        assert group["ns_coefficients"] == (3.4445, -4.775, 2.0315)
        assert group["eps"] == 1e-7
        assert group["ns_steps"] == 5
        assert group["adjust_lr_fn"] == "match_rms_adamw"
        assert group["weight_decay"] == 0.0


@pytest.mark.parametrize(
    ("optimizer_name", "expected_state_keys"),
    (
        ("adamw", ("step", "exp_avg", "exp_avg_sq")),
        ("proxy_muon", ("momentum_buffer",)),
    ),
)
def test_build_head_optimizer_creates_exact_state_on_first_step(
    optimizer_name: str, expected_state_keys: tuple[str, ...]
) -> None:
    head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))
    optimizer = module.build_head_optimizer(head, optimizer_name, 0.0002)
    head.grad = torch.full_like(head, 0.25)

    optimizer.step()

    assert tuple(optimizer.state) == (head,)
    assert tuple(optimizer.state[head]) == expected_state_keys


@pytest.mark.parametrize(
    ("head", "optimizer_name", "learning_rate"),
    (
        (torch.ones(4, 3), "adamw", 0.0001),
        (torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float64)), "adamw", 0.0001),
        (torch.nn.Parameter(torch.ones(4)), "proxy_muon", 0.0001),
        (torch.nn.Parameter(torch.ones(4, 3)), "unknown", 0.0001),
        (torch.nn.Parameter(torch.ones(4, 3)), "adamw", True),
    ),
)
def test_build_head_optimizer_rejects_noncanonical_inputs(
    head: torch.Tensor, optimizer_name: str, learning_rate: object
) -> None:
    with pytest.raises(ValueError):
        module.build_head_optimizer(head, optimizer_name, learning_rate)


@pytest.mark.parametrize("shape", ((32, 16), (16, 32)))
def test_precision_muon_bfloat16_is_byte_identical_to_builtin_for_eight_steps(
    shape: tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(20260825)
    initial = torch.randn(shape, generator=generator, dtype=torch.float32)
    gradients = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for _ in range(8)
    )
    builtin_head = torch.nn.Parameter(initial.clone())
    adapter_head = torch.nn.Parameter(initial.clone())
    builtin = module.build_head_optimizer(builtin_head, "proxy_muon", 0.0002)
    adapter = module.PrecisionMuon(
        adapter_head,
        lr=0.0002,
        ns_dtype=torch.bfloat16,
    )

    for gradient in gradients:
        builtin_head.grad = gradient.clone()
        adapter_head.grad = gradient.clone()
        builtin.step()
        adapter.step()

        assert torch.equal(adapter_head, builtin_head)
        assert torch.equal(
            adapter.state[adapter_head]["momentum_buffer"],
            builtin.state[builtin_head]["momentum_buffer"],
        )


def test_builtin_muon_trace_matches_private_helper_and_is_noninterfering() -> None:
    generator = torch.Generator().manual_seed(20260826)
    initial = torch.randn((32, 16), generator=generator, dtype=torch.float32)
    first_gradient = torch.randn((32, 16), generator=generator, dtype=torch.float32)
    traced_gradient = torch.randn((32, 16), generator=generator, dtype=torch.float32)
    traced_head = torch.nn.Parameter(initial.clone())
    control_head = torch.nn.Parameter(initial.clone())
    traced_optimizer = module.build_head_optimizer(traced_head, "proxy_muon", 0.0002)
    control_optimizer = module.build_head_optimizer(control_head, "proxy_muon", 0.0002)
    for head, optimizer in (
        (traced_head, traced_optimizer),
        (control_head, control_optimizer),
    ):
        head.grad = first_gradient.clone()
        optimizer.step()
        head.grad = traced_gradient.clone()

    head_before = traced_head.detach().clone()
    gradient_before = traced_head.grad.clone()
    momentum_before = traced_optimizer.state[traced_head]["momentum_buffer"].clone()
    next_momentum = momentum_before.clone().lerp_(traced_gradient, 0.05)
    effective_update = traced_gradient.lerp(next_momentum, 0.95)
    expected = _muon._zeropower_via_newtonschulz(
        effective_update,
        (3.4445, -4.775, 2.0315),
        5,
        1e-7,
    )

    trace = module.trace_builtin_muon_step(traced_head, traced_optimizer)

    assert torch.equal(trace.orthogonal_update, expected)
    assert trace.update_dtype == "torch.bfloat16"
    assert type(trace.polar_factor_residual) is float
    assert torch.isfinite(torch.tensor(trace.polar_factor_residual))
    assert torch.equal(traced_head, head_before)
    assert torch.equal(traced_head.grad, gradient_before)
    assert torch.equal(
        traced_optimizer.state[traced_head]["momentum_buffer"], momentum_before
    )

    traced_optimizer.step()
    control_optimizer.step()

    assert torch.equal(traced_head, control_head)
    assert torch.equal(
        traced_optimizer.state[traced_head]["momentum_buffer"],
        control_optimizer.state[control_head]["momentum_buffer"],
    )


def test_float32_newton_schulz_clones_instead_of_mutating_its_input() -> None:
    generator = torch.Generator().manual_seed(20260827)
    update = torch.randn((32, 16), generator=generator, dtype=torch.float32)
    before = update.clone()

    orthogonal = module._newton_schulz_zeropower(update, ns_dtype=torch.float32)

    assert torch.equal(update, before)
    assert orthogonal.dtype == torch.float32
    assert orthogonal.data_ptr() != update.data_ptr()


@pytest.mark.parametrize(
    "head",
    (
        torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32)),
        torch.nn.Parameter(
            torch.tensor(
                [[float("nan"), 1.0, 1.0]] * 4,
                dtype=torch.float32,
            )
        ),
    ),
)
def test_precision_muon_rejects_zero_row_or_nonfinite_initial_head(
    head: torch.nn.Parameter,
) -> None:
    with pytest.raises(ValueError):
        module.PrecisionMuon(head, lr=0.0002, ns_dtype=torch.bfloat16)


@pytest.mark.parametrize("state_mutation", ("extra", "nan_momentum", "inf_gradient"))
def test_precision_muon_rejects_nonfinite_or_mutated_live_state(
    state_mutation: str,
) -> None:
    head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))
    optimizer = module.PrecisionMuon(
        head,
        lr=0.0002,
        ns_dtype=torch.bfloat16,
    )
    head.grad = torch.ones_like(head)
    optimizer.step()
    if state_mutation == "extra":
        optimizer.state[head]["unexpected"] = torch.ones(())
    elif state_mutation == "nan_momentum":
        optimizer.state[head]["momentum_buffer"][0, 0] = float("nan")
    else:
        head.grad[0, 0] = float("inf")

    with pytest.raises(ValueError):
        optimizer.step()


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("lr", 0.0003),
        ("weight_decay", 0.1),
        ("momentum", 0.9),
        ("nesterov", False),
        ("ns_coefficients", (3.0, -4.775, 2.0315)),
        ("eps", 1e-6),
        ("ns_steps", 4),
        ("adjust_lr_fn", None),
    ),
)
def test_builtin_muon_trace_rejects_constructor_drift(key: str, value: object) -> None:
    head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))
    optimizer = module.build_head_optimizer(head, "proxy_muon", 0.0002)
    head.grad = torch.ones_like(head)
    optimizer.param_groups[0][key] = value

    with pytest.raises(ValueError):
        module.trace_builtin_muon_step(head, optimizer)


def test_precision_muon_rejects_nonfinite_update_before_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))
    optimizer = module.PrecisionMuon(
        head,
        lr=0.0002,
        ns_dtype=torch.float32,
    )
    head.grad = torch.ones_like(head)
    before = head.detach().clone()
    monkeypatch.setattr(
        module,
        "_newton_schulz_zeropower",
        lambda _update, *, ns_dtype: torch.full_like(head, float("inf")),
    )

    with pytest.raises(ValueError):
        optimizer.step()

    assert torch.equal(head, before)


def test_precision_muon_rejects_unregistered_newton_schulz_dtype() -> None:
    head = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.float32))

    with pytest.raises(ValueError):
        module.PrecisionMuon(head, lr=0.0002, ns_dtype=torch.float16)


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ({307: 1.0, 435: 0.9, 512: 0.8}, 307),
        ({307: 1.1, 435: 1.0, 512: 0.8}, 435),
        ({307: 1.1, 435: 1.01, 512: 1.0}, 512),
        ({307: 1.1, 435: 1.01, 512: 1.001}, ">512"),
    ],
)
def test_compute_reach_step_uses_first_registered_loss_not_above_reference(
    losses: dict[int, float], expected: int | str
) -> None:
    assert module.compute_reach_step(losses, reference_loss=1.0) == expected
