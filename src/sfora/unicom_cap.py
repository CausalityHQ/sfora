"""Pure covariance-adjusted prototype construction and decision logic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf
from torch.nn import functional as F

from sfora.unicom_probe import PROBE_SELECTED_FEATURES, PROBE_SHARDS, ProbeMetrics
from sfora.unicom_training import sample_shard_masks

CAP_VARIANTS = ("cap_centered", "cap_uncentered")
CAP_FIT_SEEDS = (0, 1, 2)
CAP_STEP_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CAP_MASK_SEED = 23_006
CAP_MASK_SETS = 8
CAP_LOSS_DELTA = 0.0501203852609845
CAP_ACCURACY_DELTA = 0.006380126646800488
CAP_HEAD_COSINE_MINIMUM = 0.95
CAP_NON_WORSE_MASK_MINIMUM = 60
CAP_MASK_T_CRITICAL_DF63 = 1.998340542520741
CAP_IDENTITY_T_CRITICAL_DF3187 = 1.9607086212236648


@dataclass(frozen=True)
class CapConstruction:
    sample_count: int
    feature_count: int
    shrinkage: float
    covariance_trace: float
    cholesky_diagonal_min: float
    cholesky_diagonal_max: float
    covariance_sha256: str
    condition_number: float
    effective_rank: float
    covariance: np.ndarray
    class_means: np.ndarray
    global_mean: np.ndarray
    heads: dict[str, torch.Tensor]


@dataclass(frozen=True)
class CapCosineSummary:
    minimum: float
    p05: float
    median: float
    mean: float


@dataclass(frozen=True)
class CapVariantStatistics:
    loss_delta: float
    accuracy_delta: float
    non_worse_mask_count: int
    unrepresented_loss_delta: float
    mask_paired_mean_delta: float
    mask_paired_95_lower_bound: float
    identity_paired_mean_delta: float
    identity_paired_95_lower_bound: float


@dataclass(frozen=True)
class CapVariantDecision:
    statistics: CapVariantStatistics
    seed_invariant_predicates: dict[str, bool]
    per_seed_head_cosine_mean: dict[int, float]
    per_seed_step_equivalence: dict[int, int | str]
    per_seed_predicates: dict[int, dict[str, bool]]
    passes_static: bool
    passes_all: bool
    decision_level: int
    min_step_equivalence: int | str


@dataclass(frozen=True)
class CapDecision:
    status: str
    selected_variant: str | None
    per_variant: dict[str, CapVariantDecision]


LedoitWolf = Callable[..., tuple[np.ndarray, float]]


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} differs")
    return value


def _normalize_solution(values: np.ndarray, row_norm: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise ValueError("CAP solution row norm differs")
    return np.ascontiguousarray(values / norms[:, None] * row_norm)


def build_cap_heads(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    row_norm: float,
    ledoit_wolf_fn: LedoitWolf = ledoit_wolf,
) -> CapConstruction:
    """Build the two registered covariance-adjusted prototype heads."""

    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.dtype != torch.float32
        or features.shape[0] == 0
        or features.shape[1] == 0
        or not torch.isfinite(features).all()
        or torch.any(torch.linalg.vector_norm(features, dim=1) == 0.0)
        or type(labels) is not torch.Tensor
        or labels.ndim != 1
        or labels.dtype != torch.int64
        or labels.shape[0] != features.shape[0]
        or labels.device != features.device
        or type(row_norm) is not float
        or not math.isfinite(row_norm)
        or row_norm <= 0.0
        or not callable(ledoit_wolf_fn)
    ):
        raise ValueError("CAP construction input differs")
    class_count = int(labels.max()) + 1
    if int(labels.min()) != 0:
        raise ValueError("CAP labels differ")
    counts = torch.bincount(labels, minlength=class_count)
    if counts.shape[0] != class_count or torch.any(counts == 0):
        raise ValueError("CAP labels differ")

    normalized = F.normalize(features, dim=1).double()
    class_means_tensor = torch.zeros(
        class_count,
        features.shape[1],
        dtype=torch.float64,
        device=features.device,
    )
    class_means_tensor.index_add_(0, labels, normalized)
    class_means_tensor /= counts.double()[:, None]
    global_mean_tensor = normalized.mean(dim=0)
    normalized_array = np.ascontiguousarray(normalized.cpu().numpy())
    class_means = np.ascontiguousarray(class_means_tensor.cpu().numpy())
    global_mean = np.ascontiguousarray(global_mean_tensor.cpu().numpy())
    label_array = np.ascontiguousarray(labels.cpu().numpy())
    residuals = np.ascontiguousarray(normalized_array - class_means[label_array])

    covariance_value, shrinkage_value = ledoit_wolf_fn(
        residuals,
        assume_centered=True,
        block_size=1000,
    )
    covariance = np.asarray(covariance_value)
    if (
        covariance.dtype != np.float64
        or covariance.shape != (features.shape[1], features.shape[1])
        or not covariance.flags.c_contiguous
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("CAP covariance differs")
    shrinkage = float(shrinkage_value)
    if not math.isfinite(shrinkage):
        raise ValueError("CAP shrinkage differs")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("CAP covariance is not positive definite") from error
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        raise ValueError("CAP covariance eigenvalues differ")
    probabilities = eigenvalues / eigenvalues.sum()

    heads: dict[str, torch.Tensor] = {}
    for name, rhs in (
        ("cap_centered", class_means - global_mean),
        ("cap_uncentered", class_means),
    ):
        first_solution = np.linalg.solve(cholesky, rhs.T)
        solution = np.linalg.solve(cholesky.T, first_solution).T
        normalized_solution = _normalize_solution(solution, row_norm)
        heads[name] = (
            torch.from_numpy(normalized_solution.astype(np.float32, copy=False))
            .to(features.device)
            .contiguous()
        )

    diagonal = np.diag(cholesky)
    return CapConstruction(
        sample_count=int(features.shape[0]),
        feature_count=int(features.shape[1]),
        shrinkage=shrinkage,
        covariance_trace=float(np.trace(covariance)),
        cholesky_diagonal_min=float(diagonal.min()),
        cholesky_diagonal_max=float(diagonal.max()),
        covariance_sha256=hashlib.sha256(covariance.tobytes(order="C")).hexdigest(),
        condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        effective_rank=float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        covariance=covariance.copy(),
        class_means=class_means,
        global_mean=global_mean,
        heads=heads,
    )


def cap_step_equivalence(
    cap_loss: float, trajectory_losses: Mapping[int, float]
) -> int | str:
    """Return the first registered snapshot whose loss reaches the CAP loss."""

    _finite_float(cap_loss, "CAP loss")
    if type(trajectory_losses) is not dict or tuple(trajectory_losses) != CAP_STEP_GRID:
        raise ValueError("CAP trajectory step order differs")
    for step, loss in trajectory_losses.items():
        _finite_float(loss, "CAP trajectory loss")
        if loss <= cap_loss:
            return step
    return ">512"


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(denominator == 0.0) or not np.isfinite(denominator).all():
        raise ValueError("CAP covariance diagnostic row differs")
    return np.clip(numerator / denominator, -1.0, 1.0)


def covariance_mask_mismatch(
    construction: CapConstruction,
    *,
    seed: int = CAP_MASK_SEED,
    mask_sets: int = CAP_MASK_SETS,
) -> dict[str, object]:
    """Compare restricted full solutions with mask-local covariance solves."""

    if (
        type(construction) is not CapConstruction
        or type(seed) is not int
        or seed < 0
        or type(mask_sets) is not int
        or mask_sets <= 0
        or construction.feature_count < PROBE_SELECTED_FEATURES
        or tuple(construction.heads) != CAP_VARIANTS
    ):
        raise ValueError("CAP covariance diagnostic input differs")
    covariance = construction.covariance
    eigenvalues = np.linalg.eigvalsh(covariance)
    probabilities = eigenvalues / eigenvalues.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask_hashes: list[tuple[str, ...]] = []
    cosines = {name: [] for name in CAP_VARIANTS}
    right_hand_sides = {
        "cap_centered": construction.class_means - construction.global_mean,
        "cap_uncentered": construction.class_means,
    }
    full_cholesky = np.linalg.cholesky(covariance)
    full_solutions = {}
    for name, rhs in right_hand_sides.items():
        first_solution = np.linalg.solve(full_cholesky, rhs.T)
        full_solutions[name] = np.linalg.solve(
            full_cholesky.T, first_solution
        ).T
    class_count = construction.class_means.shape[0]
    for _mask_set in range(mask_sets):
        masks = sample_shard_masks(
            dimension=construction.feature_count,
            selected=PROBE_SELECTED_FEATURES,
            shards=PROBE_SHARDS,
            generator=generator,
            device=torch.device("cpu"),
        )
        current_hashes: list[str] = []
        for shard, mask_tensor in enumerate(masks):
            mask = np.asarray(mask_tensor.numpy(), dtype="<i8")
            current_hashes.append(
                hashlib.sha256(np.ascontiguousarray(mask).tobytes(order="C")).hexdigest()
            )
            rows = np.arange(shard, class_count, PROBE_SHARDS)
            if rows.size == 0:
                continue
            principal = covariance[np.ix_(mask, mask)]
            cholesky = np.linalg.cholesky(principal)
            for name in CAP_VARIANTS:
                rhs = right_hand_sides[name][np.ix_(rows, mask)]
                first_solution = np.linalg.solve(cholesky, rhs.T)
                local = np.linalg.solve(cholesky.T, first_solution).T
                restricted = full_solutions[name][np.ix_(rows, mask)]
                cosines[name].extend(_cosine_rows(restricted, local).tolist())
        mask_hashes.append(tuple(current_hashes))

    summaries: dict[str, dict[str, object]] = {}
    for name in CAP_VARIANTS:
        values = np.asarray(cosines[name], dtype=np.float64)
        if values.size != mask_sets * class_count or not np.isfinite(values).all():
            raise ValueError("CAP covariance diagnostic count differs")
        summaries[name] = {
            "row_cosines": values.tolist(),
            "minimum": float(values.min()),
            "p05": float(np.quantile(values, 0.05, method="linear")),
            "median": float(np.quantile(values, 0.5, method="linear")),
            "mean": float(values.mean()),
        }
    return {
        "seed": seed,
        "mask_sets": mask_sets,
        "mask_sha256": tuple(mask_hashes),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "cosines": summaries,
    }


def _validate_probe_metrics(
    metrics: object,
    name: str,
    *,
    expected_mask_count: int,
    expected_image_count: int,
) -> ProbeMetrics:
    if type(metrics) is not ProbeMetrics:
        raise ValueError(f"{name} metrics differ")
    float_values = (
        metrics.mean_loss,
        metrics.accuracy,
        metrics.represented_mean_loss,
        metrics.unrepresented_mean_loss,
        *metrics.per_mask_mean_losses,
        *metrics.per_mask_represented_mean_losses,
        *metrics.per_mask_unrepresented_mean_losses,
        *metrics.per_image_mean_losses,
    )
    if (
        any(type(value) is not float or not math.isfinite(value) for value in float_values)
        or len(metrics.per_mask_mean_losses) != expected_mask_count
        or len(metrics.per_mask_represented_mean_losses) != expected_mask_count
        or len(metrics.per_mask_unrepresented_mean_losses) != expected_mask_count
        or len(metrics.per_image_mean_losses) != expected_image_count
        or type(metrics.correct_count) is not int
        or type(metrics.observation_count) is not int
    ):
        raise ValueError(f"{name} metrics differ")
    return metrics


def _paired_statistics(
    baseline: tuple[float, ...], candidate: tuple[float, ...], critical: float
) -> tuple[float, float]:
    deltas = tuple(
        left - right for left, right in zip(baseline, candidate, strict=True)
    )
    mean = math.fsum(deltas) / len(deltas)
    variance = math.fsum((value - mean) ** 2 for value in deltas) / (
        len(deltas) - 1
    )
    lower = mean - critical * math.sqrt(variance) / math.sqrt(len(deltas))
    return float(mean), float(lower)


def _minimum_step(values: Mapping[int, int | str]) -> int | str:
    numeric = tuple(
        math.inf if value == ">512" else value for value in values.values()
    )
    minimum = min(numeric)
    return ">512" if math.isinf(minimum) else int(minimum)


def cap_decision(
    *,
    class_mean: ProbeMetrics,
    cap_metrics: Mapping[str, ProbeMetrics],
    target_heads: Mapping[int, Mapping[str, CapCosineSummary]],
    trajectories: Mapping[int, Mapping[int, float]],
    expected_mask_count: int = 64,
    expected_image_count: int = 3_188,
) -> CapDecision:
    """Apply the registered CAP F0 predicates and variant selection rule."""

    class_mean = _validate_probe_metrics(
        class_mean,
        "class_mean",
        expected_mask_count=expected_mask_count,
        expected_image_count=expected_image_count,
    )
    if type(cap_metrics) is not dict or tuple(cap_metrics) != CAP_VARIANTS:
        raise ValueError("CAP metric order differs")
    if any(
        type(values) is not dict or tuple(values) != CAP_FIT_SEEDS
        for values in (target_heads, trajectories)
    ):
        raise ValueError("CAP seed order differs")
    for seed in CAP_FIT_SEEDS:
        if (
            type(target_heads[seed]) is not dict
            or tuple(target_heads[seed]) != CAP_VARIANTS
            or type(trajectories[seed]) is not dict
            or tuple(trajectories[seed]) != CAP_STEP_GRID
        ):
            raise ValueError("CAP seed evidence differs")
        for summary in target_heads[seed].values():
            if type(summary) is not CapCosineSummary or any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    summary.minimum,
                    summary.p05,
                    summary.median,
                    summary.mean,
                )
            ):
                raise ValueError("CAP cosine summary differs")
        for loss in trajectories[seed].values():
            _finite_float(loss, "CAP trajectory loss")

    per_variant: dict[str, CapVariantDecision] = {}
    for name in CAP_VARIANTS:
        metrics = _validate_probe_metrics(
            cap_metrics[name],
            name,
            expected_mask_count=expected_mask_count,
            expected_image_count=expected_image_count,
        )
        mask_mean, mask_lower = _paired_statistics(
            class_mean.per_mask_mean_losses,
            metrics.per_mask_mean_losses,
            CAP_MASK_T_CRITICAL_DF63,
        )
        identity_mean, identity_lower = _paired_statistics(
            class_mean.per_image_mean_losses,
            metrics.per_image_mean_losses,
            CAP_IDENTITY_T_CRITICAL_DF3187,
        )
        non_worse = sum(
            baseline >= candidate
            for baseline, candidate in zip(
                class_mean.per_mask_mean_losses,
                metrics.per_mask_mean_losses,
                strict=True,
            )
        )
        statistics = CapVariantStatistics(
            loss_delta=float(class_mean.mean_loss - metrics.mean_loss),
            accuracy_delta=float(metrics.accuracy - class_mean.accuracy),
            non_worse_mask_count=non_worse,
            unrepresented_loss_delta=float(
                class_mean.unrepresented_mean_loss
                - metrics.unrepresented_mean_loss
            ),
            mask_paired_mean_delta=mask_mean,
            mask_paired_95_lower_bound=mask_lower,
            identity_paired_mean_delta=identity_mean,
            identity_paired_95_lower_bound=identity_lower,
        )
        seed_invariant = {
            "loss_delta_at_least_0_0501203852609845": metrics.mean_loss
            <= class_mean.mean_loss - CAP_LOSS_DELTA,
            "accuracy_delta_at_least_0_006380126646800488": metrics.accuracy
            >= class_mean.accuracy + CAP_ACCURACY_DELTA,
            "mask_and_stratum_consistent": non_worse
            >= CAP_NON_WORSE_MASK_MINIMUM
            and metrics.unrepresented_mean_loss
            <= class_mean.unrepresented_mean_loss,
            "paired_95_lower_bound_positive": mask_lower > 0.0,
            "identity_95_lower_bound_positive": identity_lower > 0.0,
        }
        cosine_means = {
            seed: target_heads[seed][name].mean for seed in CAP_FIT_SEEDS
        }
        step_equivalence = {
            seed: cap_step_equivalence(metrics.mean_loss, trajectories[seed])
            for seed in CAP_FIT_SEEDS
        }
        per_seed_predicates = {
            seed: {
                "head_cosine_at_least_0_95": cosine_means[seed]
                >= CAP_HEAD_COSINE_MINIMUM,
                "step_equivalence_at_least_64": step_equivalence[seed] == ">512"
                or step_equivalence[seed] >= 64,
            }
            for seed in CAP_FIT_SEEDS
        }
        passes_static = all(seed_invariant.values()) and all(
            values["head_cosine_at_least_0_95"]
            for values in per_seed_predicates.values()
        )
        passes_all = passes_static and all(
            values["step_equivalence_at_least_64"]
            for values in per_seed_predicates.values()
        )
        level = 2 if passes_all else (1 if passes_static else 0)
        per_variant[name] = CapVariantDecision(
            statistics=statistics,
            seed_invariant_predicates=seed_invariant,
            per_seed_head_cosine_mean=cosine_means,
            per_seed_step_equivalence=step_equivalence,
            per_seed_predicates=per_seed_predicates,
            passes_static=passes_static,
            passes_all=passes_all,
            decision_level=level,
            min_step_equivalence=_minimum_step(step_equivalence),
        )

    maximum_level = max(value.decision_level for value in per_variant.values())
    if maximum_level == 0:
        return CapDecision(
            status="CLOSE_CAP", selected_variant=None, per_variant=per_variant
        )
    candidates = [
        name
        for name, value in per_variant.items()
        if value.decision_level == maximum_level
    ]
    selected = candidates[0]
    for name in candidates[1:]:
        incumbent = per_variant[selected].min_step_equivalence
        challenger = per_variant[name].min_step_equivalence
        incumbent_value = math.inf if incumbent == ">512" else incumbent
        challenger_value = math.inf if challenger == ">512" else challenger
        if challenger_value > incumbent_value:
            selected = name
    return CapDecision(
        status="PROCEED_STAGE_A" if maximum_level == 2 else "ROUTE_STAGE_B",
        selected_variant=selected,
        per_variant=per_variant,
    )
