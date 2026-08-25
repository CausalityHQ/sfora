from __future__ import annotations

import copy

import pytest

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
