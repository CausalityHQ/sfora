from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from sfora.unicom_cap import (
    CAP_STEP_GRID,
    CapConstruction,
    CapCosineSummary,
    CapDecision,
    build_cap_heads,
    cap_decision,
    cap_step_equivalence,
    covariance_mask_mismatch,
)
from sfora.unicom_probe import ProbeMetrics
from sfora.unicom_training import sample_shard_masks


def _analytic_features() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[3.0, 4.0], [0.0, 2.0], [4.0, 3.0]], dtype=torch.float32),
        torch.tensor([0, 0, 1], dtype=torch.int64),
    )


def test_build_cap_heads_uses_registered_ledoit_wolf_solve() -> None:
    features, labels = _analytic_features()
    calls: list[tuple[np.ndarray, bool, int]] = []
    covariance = np.diag([2.0, 4.0]).astype(np.float64)

    def fake(
        values: np.ndarray, *, assume_centered: bool, block_size: int
    ) -> tuple[np.ndarray, float]:
        calls.append((values.copy(), assume_centered, block_size))
        return covariance.copy(), 0.25

    result = build_cap_heads(
        features,
        labels,
        row_norm=0.5,
        ledoit_wolf_fn=fake,
    )

    normalized = torch.nn.functional.normalize(features, dim=1).double().numpy()
    class_means = np.stack((normalized[:2].mean(axis=0), normalized[2:].mean(axis=0)))
    global_mean = normalized.mean(axis=0)
    residuals = normalized - class_means[labels.numpy()]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], residuals)
    assert calls[0][1:] == (True, 1000)
    assert result.sample_count == 3
    assert result.feature_count == 2
    assert result.shrinkage == 0.25
    assert result.covariance_trace == 6.0
    assert result.cholesky_diagonal_min == 2.0**0.5
    assert result.cholesky_diagonal_max == 2.0
    assert result.covariance_sha256 == hashlib.sha256(
        covariance.tobytes(order="C")
    ).hexdigest()
    np.testing.assert_array_equal(result.covariance, covariance)
    np.testing.assert_array_equal(result.class_means, class_means)
    np.testing.assert_array_equal(result.global_mean, global_mean)
    assert tuple(result.heads) == ("cap_centered", "cap_uncentered")

    for name, rhs in {
        "cap_centered": class_means - global_mean,
        "cap_uncentered": class_means,
    }.items():
        cholesky = np.linalg.cholesky(covariance)
        first_solution = np.linalg.solve(cholesky, rhs.T)
        expected = np.linalg.solve(cholesky.T, first_solution).T
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        expected = torch.from_numpy(expected * 0.5).float()
        torch.testing.assert_close(result.heads[name], expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(result.heads[name], dim=1),
            torch.full((2,), 0.5, dtype=torch.float32),
            rtol=1e-7,
            atol=1e-8,
        )


def test_build_cap_heads_uses_row_weighted_global_mean_for_unbalanced_classes() -> None:
    features, labels = _analytic_features()

    result = build_cap_heads(
        features,
        labels,
        row_norm=0.5,
        ledoit_wolf_fn=lambda *_args, **_kwargs: (np.eye(2), 0.0),
    )

    normalized = torch.nn.functional.normalize(features, dim=1).double().numpy()
    np.testing.assert_array_equal(result.global_mean, normalized.mean(axis=0))
    assert not np.array_equal(result.global_mean, result.class_means.mean(axis=0))


@pytest.mark.parametrize("mutation", ("nonfinite", "asymmetric", "not_positive"))
def test_build_cap_heads_rejects_invalid_covariance(mutation: str) -> None:
    features, labels = _analytic_features()
    covariance = np.eye(2, dtype=np.float64)
    if mutation == "nonfinite":
        covariance[0, 0] = np.nan
    elif mutation == "asymmetric":
        covariance[0, 1] = 0.1
    else:
        covariance[1, 1] = 0.0

    with pytest.raises(ValueError):
        build_cap_heads(
            features,
            labels,
            row_norm=0.5,
            ledoit_wolf_fn=lambda *_args, **_kwargs: (covariance, 0.0),
        )


def test_cap_step_equivalence_uses_first_registered_loss_at_or_below_cap() -> None:
    losses = {
        step: 1.0 if step < 1 else (0.8 if step < 2 else 0.6)
        for step in CAP_STEP_GRID
    }

    assert cap_step_equivalence(0.6, losses) == 2
    assert cap_step_equivalence(0.1, losses) == ">512"


def _metrics(
    *,
    mean_loss: float,
    accuracy: float,
    non_worse_masks: int = 64,
    unrepresented_mean_loss: float | None = None,
    paired_delta: float = 0.1,
) -> ProbeMetrics:
    mask_losses = tuple(
        1.0 - paired_delta if index < non_worse_masks else 1.1
        for index in range(64)
    )
    image_losses = (1.0 - paired_delta,) * 3_188
    return ProbeMetrics(
        mean_loss=mean_loss,
        accuracy=accuracy,
        correct_count=int(accuracy * 64),
        observation_count=64,
        per_mask_mean_losses=mask_losses,
        per_mask_represented_mean_losses=mask_losses,
        per_mask_unrepresented_mean_losses=mask_losses,
        per_image_mean_losses=image_losses,
        represented_mean_loss=mean_loss,
        unrepresented_mean_loss=(
            mean_loss
            if unrepresented_mean_loss is None
            else unrepresented_mean_loss
        ),
    )


def _decision_inputs(
    *,
    cosine: float = 0.95,
    first_step: int | None = 64,
    non_worse_masks: int = 64,
    paired_delta: float = 0.1,
    loss_delta: float = 0.1,
    accuracy_delta: float = 0.01,
) -> tuple[
    ProbeMetrics,
    dict[str, ProbeMetrics],
    dict[int, dict[str, CapCosineSummary]],
    dict[int, dict[int, float]],
]:
    class_mean = _metrics(mean_loss=1.0, accuracy=0.5, paired_delta=0.0)
    cap = _metrics(
        mean_loss=1.0 - loss_delta,
        accuracy=0.5 + accuracy_delta,
        non_worse_masks=non_worse_masks,
        paired_delta=paired_delta,
        unrepresented_mean_loss=1.0,
    )
    cap_metrics = {"cap_centered": cap, "cap_uncentered": cap}
    summary = CapCosineSummary(
        minimum=cosine,
        p05=cosine,
        median=cosine,
        mean=cosine,
    )
    target_heads = {
        seed: {"cap_centered": summary, "cap_uncentered": summary}
        for seed in (0, 1, 2)
    }
    trajectories = {
        seed: {
            step: cap.mean_loss
            + (0.1 if first_step is None or step < first_step else 0.0)
            for step in CAP_STEP_GRID
        }
        for seed in (0, 1, 2)
    }
    return class_mean, cap_metrics, target_heads, trajectories


def _decide(
    inputs: tuple[
        ProbeMetrics,
        dict[str, ProbeMetrics],
        dict[int, dict[str, CapCosineSummary]],
        dict[int, dict[int, float]],
    ],
) -> CapDecision:
    class_mean, cap_metrics, target_heads, trajectories = inputs
    return cap_decision(
        class_mean=class_mean,
        cap_metrics=cap_metrics,
        target_heads=target_heads,
        trajectories=trajectories,
    )


def test_cap_decision_accepts_exact_registered_boundaries_and_centered_tie() -> None:
    inputs = _decision_inputs(
        cosine=0.95,
        first_step=64,
        non_worse_masks=60,
        loss_delta=0.0501203852609845,
        accuracy_delta=0.006380126646800488,
    )

    actual = _decide(inputs)

    assert actual.status == "PROCEED_STAGE_A"
    assert actual.selected_variant == "cap_centered"
    assert tuple(actual.per_variant) == ("cap_centered", "cap_uncentered")
    for variant in actual.per_variant.values():
        assert tuple(variant.seed_invariant_predicates) == (
            "loss_delta_at_least_0_0501203852609845",
            "accuracy_delta_at_least_0_006380126646800488",
            "mask_and_stratum_consistent",
            "paired_95_lower_bound_positive",
            "identity_95_lower_bound_positive",
        )
        assert all(variant.seed_invariant_predicates.values())
        assert variant.per_seed_step_equivalence == {0: 64, 1: 64, 2: 64}
        assert variant.passes_static is True
        assert variant.passes_all is True
        assert variant.decision_level == 2


def test_cap_decision_routes_static_candidate_when_step_equivalence_is_32() -> None:
    actual = _decide(_decision_inputs(first_step=32))

    assert actual.status == "ROUTE_STAGE_B"
    assert actual.selected_variant == "cap_centered"
    assert all(variant.passes_static for variant in actual.per_variant.values())
    assert all(not variant.passes_all for variant in actual.per_variant.values())


def test_cap_decision_closes_and_selects_null_below_head_cosine_boundary() -> None:
    actual = _decide(_decision_inputs(cosine=float(np.nextafter(0.95, 0.0))))

    assert actual.status == "CLOSE_CAP"
    assert actual.selected_variant is None
    assert all(variant.decision_level == 0 for variant in actual.per_variant.values())


@pytest.mark.parametrize(
    ("mutation", "predicate"),
    (
        ("loss", "loss_delta_at_least_0_0501203852609845"),
        ("accuracy", "accuracy_delta_at_least_0_006380126646800488"),
        ("mask_count", "mask_and_stratum_consistent"),
        ("paired_zero", "paired_95_lower_bound_positive"),
        ("identity_zero", "identity_95_lower_bound_positive"),
    ),
)
def test_cap_decision_rejects_each_static_boundary_below_or_at_zero(
    mutation: str, predicate: str
) -> None:
    class_mean, cap_metrics, target_heads, trajectories = _decision_inputs()
    cap = cap_metrics["cap_centered"]
    if mutation == "loss":
        cap = _metrics(
            mean_loss=1.0 - (0.0501203852609845 - 1e-12),
            accuracy=cap.accuracy,
        )
    elif mutation == "accuracy":
        cap = _metrics(
            mean_loss=cap.mean_loss,
            accuracy=0.5 + (0.006380126646800488 - 1e-12),
        )
    elif mutation == "mask_count":
        cap = _metrics(mean_loss=cap.mean_loss, accuracy=cap.accuracy, non_worse_masks=59)
    elif mutation == "paired_zero":
        cap = _metrics(mean_loss=cap.mean_loss, accuracy=cap.accuracy, paired_delta=0.0)
    else:
        cap = ProbeMetrics(
            **{
                **cap.__dict__,
                "per_image_mean_losses": class_mean.per_image_mean_losses,
            }
        )
    cap_metrics = {**cap_metrics, "cap_centered": cap}

    actual = cap_decision(
        class_mean=class_mean,
        cap_metrics=cap_metrics,
        target_heads=target_heads,
        trajectories=trajectories,
    )

    assert actual.per_variant["cap_centered"].seed_invariant_predicates[predicate] is False


@pytest.mark.parametrize("mutation", ("missing_seed", "reordered_seed", "numpy_scalar"))
def test_cap_decision_rejects_noncanonical_seed_or_scalar_evidence(mutation: str) -> None:
    class_mean, cap_metrics, target_heads, trajectories = _decision_inputs()
    if mutation == "missing_seed":
        target_heads.pop(2)
    elif mutation == "reordered_seed":
        target_heads = {2: target_heads[2], 0: target_heads[0], 1: target_heads[1]}
    else:
        target_heads[0]["cap_centered"] = CapCosineSummary(
            minimum=0.95,
            p05=0.95,
            median=0.95,
            mean=np.float64(0.95),
        )

    with pytest.raises((TypeError, ValueError)):
        cap_decision(
            class_mean=class_mean,
            cap_metrics=cap_metrics,
            target_heads=target_heads,
            trajectories=trajectories,
        )


def test_cap_decision_validates_authenticated_metric_dimensions() -> None:
    class_mean, cap_metrics, target_heads, trajectories = _decision_inputs()

    def resize(metrics: ProbeMetrics) -> ProbeMetrics:
        return ProbeMetrics(
            **{
                **metrics.__dict__,
                "per_mask_mean_losses": metrics.per_mask_mean_losses[:8],
                "per_mask_represented_mean_losses": (
                    metrics.per_mask_represented_mean_losses[:8]
                ),
                "per_mask_unrepresented_mean_losses": (
                    metrics.per_mask_unrepresented_mean_losses[:8]
                ),
                "per_image_mean_losses": metrics.per_image_mean_losses[:8],
            }
        )

    actual = cap_decision(
        class_mean=resize(class_mean),
        cap_metrics={name: resize(metrics) for name, metrics in cap_metrics.items()},
        target_heads=target_heads,
        trajectories=trajectories,
        expected_mask_count=8,
        expected_image_count=8,
    )

    assert actual.status == "CLOSE_CAP"


def test_cap_decision_treats_both_over_512_values_as_centered_tie() -> None:
    actual = _decide(_decision_inputs(first_step=None))

    assert actual.status == "PROCEED_STAGE_A"
    assert actual.selected_variant == "cap_centered"
    assert all(
        variant.min_step_equivalence == ">512"
        for variant in actual.per_variant.values()
    )


def _diagnostic_construction(covariance: np.ndarray) -> CapConstruction:
    dimension = covariance.shape[0]
    first = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
    second = np.linspace(2.0, 0.5, dimension, dtype=np.float64)
    class_means = np.stack((first, second))
    global_mean = class_means.mean(axis=0)
    centered = np.linalg.solve(covariance, (class_means - global_mean).T).T
    uncentered = np.linalg.solve(covariance, class_means.T).T
    heads = {}
    for name, values in (
        ("cap_centered", centered),
        ("cap_uncentered", uncentered),
    ):
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        heads[name] = torch.from_numpy(values).float().contiguous()
    diagonal = np.diag(np.linalg.cholesky(covariance))
    eigenvalues = np.linalg.eigvalsh(covariance)
    probabilities = eigenvalues / eigenvalues.sum()
    return CapConstruction(
        sample_count=4,
        feature_count=dimension,
        shrinkage=0.25,
        covariance_trace=float(np.trace(covariance)),
        cholesky_diagonal_min=float(diagonal.min()),
        cholesky_diagonal_max=float(diagonal.max()),
        covariance_sha256=hashlib.sha256(covariance.tobytes(order="C")).hexdigest(),
        condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        effective_rank=float(np.exp(-(probabilities * np.log(probabilities)).sum())),
        covariance=covariance,
        class_means=class_means,
        global_mean=global_mean,
        heads=heads,
    )


def test_covariance_mask_mismatch_is_one_for_diagonal_covariance() -> None:
    construction = _diagnostic_construction(np.eye(513, dtype=np.float64))

    actual = covariance_mask_mismatch(construction, seed=23_006, mask_sets=1)

    assert actual["seed"] == 23_006
    assert actual["mask_sets"] == 1
    assert len(actual["mask_sha256"]) == 1
    assert len(actual["mask_sha256"][0]) == 8
    generator = torch.Generator(device="cpu").manual_seed(23_006)
    masks = sample_shard_masks(
        dimension=513,
        selected=512,
        shards=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    assert actual["mask_sha256"] == (
        tuple(
            hashlib.sha256(
                np.asarray(mask.numpy(), dtype="<i8").tobytes(order="C")
            ).hexdigest()
            for mask in masks
        ),
    )
    assert actual["condition_number"] == 1.0
    assert actual["effective_rank"] == pytest.approx(513.0)
    for name in ("cap_centered", "cap_uncentered"):
        assert len(actual["cosines"][name]["row_cosines"]) == 2
        assert all(
            value == pytest.approx(1.0, abs=2e-15)
            for value in actual["cosines"][name]["row_cosines"]
        )
        assert actual["cosines"][name]["minimum"] == pytest.approx(1.0, abs=2e-15)
        assert actual["cosines"][name]["p05"] == pytest.approx(1.0, abs=2e-15)
        assert actual["cosines"][name]["median"] == pytest.approx(1.0, abs=2e-15)
        assert actual["cosines"][name]["mean"] == pytest.approx(1.0, abs=2e-15)


def test_covariance_mask_mismatch_detects_off_diagonal_coupling() -> None:
    covariance = np.eye(513, dtype=np.float64) + 0.01 * np.ones(
        (513, 513), dtype=np.float64
    )
    construction = _diagnostic_construction(covariance)

    actual = covariance_mask_mismatch(construction, seed=23_006, mask_sets=1)

    assert actual["cosines"]["cap_uncentered"]["minimum"] < 1.0
    assert actual["cosines"]["cap_uncentered"]["p05"] <= actual["cosines"][
        "cap_uncentered"
    ]["median"]


def test_covariance_mask_mismatch_uses_contiguous_objective_class_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_cap as cap_module

    dimension = 513
    class_count = 16
    covariance = np.eye(dimension, dtype=np.float64)
    class_means = np.repeat(
        np.arange(1, class_count + 1, dtype=np.float64)[:, None],
        dimension,
        axis=1,
    )
    row_norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    heads = {
        name: torch.from_numpy(class_means / row_norms).float().contiguous()
        for name in ("cap_centered", "cap_uncentered")
    }
    construction = CapConstruction(
        sample_count=class_count,
        feature_count=dimension,
        shrinkage=0.0,
        covariance_trace=float(dimension),
        cholesky_diagonal_min=1.0,
        cholesky_diagonal_max=1.0,
        covariance_sha256=hashlib.sha256(covariance.tobytes(order="C")).hexdigest(),
        condition_number=1.0,
        effective_rank=float(dimension),
        covariance=covariance,
        class_means=class_means,
        global_mean=np.zeros(dimension, dtype=np.float64),
        heads=heads,
    )
    observed_rows: list[tuple[int, ...]] = []

    def capture(left: np.ndarray, _right: np.ndarray) -> np.ndarray:
        observed_rows.append(tuple(int(value) for value in left[:, 0]))
        return np.ones(left.shape[0], dtype=np.float64)

    monkeypatch.setattr(cap_module, "_cosine_rows", capture)

    covariance_mask_mismatch(construction, seed=23_006, mask_sets=1)

    assert observed_rows[1::2] == [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
    ]


def test_covariance_mask_mismatch_reuses_two_stage_cholesky_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_cap as cap_module

    covariance = np.eye(513, dtype=np.float64) + 0.001 * np.ones(
        (513, 513), dtype=np.float64
    )
    construction = _diagnostic_construction(covariance)
    calls: list[np.ndarray] = []
    original = np.linalg.solve

    def capture(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        calls.append(left.copy())
        return original(left, right)

    monkeypatch.setattr(cap_module.np.linalg, "solve", capture)

    covariance_mask_mismatch(construction, seed=23_006, mask_sets=1)

    np.testing.assert_array_equal(calls[0], np.linalg.cholesky(covariance))
    np.testing.assert_array_equal(calls[1], np.linalg.cholesky(covariance).T)
