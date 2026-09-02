"""Deterministic local artifacts for cross-seed denoising experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import torch

from sfora.weight_space_transfer import AlphaEvaluation, SeedInterpolationCurve

_SCHEMA = "sfora-cross-seed-tensor-artifact-v1"
_MANIFEST_KEYS = {
    "bindings",
    "claim_eligible",
    "role",
    "schema",
    "state_sha256",
    "tensors",
}
_TENSOR_KEYS = {"bytes", "dtype", "file", "name", "sha256", "shape"}
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}
_SEEDS = (17, 29, 43)
_CANDIDATE_ROLES = ("tower-soup", "wiener-denoise", "spectral-denoise")
_SQRT2 = 2.0**0.5
_SQRT3 = 3.0**0.5


class SpectralEdgeAmbiguity(ValueError):
    """A singular value is within the preregistered spectral edge tolerance."""


@dataclass(frozen=True)
class GroupEvidence:
    """Deterministic evidence for one named tensorwise Wiener group."""

    name: str
    cosines: tuple[float, float, float]
    rho: float
    beta: float
    g_js: float


@dataclass(frozen=True)
class SpectralEvidence:
    """Deterministic evidence for one matrix-shaped spectral estimate."""

    name: str
    edge: float
    tolerance: float
    kept_rank: int
    total_rank: int
    retained_energy: float
    total_energy: float
    singular_values: tuple[float, ...]
    retained: tuple[bool, ...]


@dataclass(frozen=True)
class CandidateStates:
    """The three preregistered candidate towers and their construction evidence."""

    tower_soup: OrderedDict[str, torch.Tensor]
    wiener_denoise: OrderedDict[str, torch.Tensor]
    spectral_denoise: OrderedDict[str, torch.Tensor]
    groups: tuple[GroupEvidence, ...]
    spectral: tuple[SpectralEvidence, ...]


@dataclass(frozen=True)
class ProjectedEvaluation:
    """One candidate evaluated with one seed's registered retrieval head."""

    seed: int
    correctness: tuple[bool, ...]
    mean_nearest_positive_cosine: float
    mean_nearest_negative_cosine: float
    mean_margin: float
    folded_state_sha256: str
    wall_time_ns: int
    peak_cuda_bytes: int
    peak_rss_bytes: int
    determinism_replay: bool

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in _SEEDS:
            raise ValueError("projected seed differs")
        if (
            type(self.correctness) is not tuple
            or len(self.correctness) != 1345
            or any(type(value) is not bool for value in self.correctness)
        ):
            raise ValueError("projected correctness evidence differs")
        values = (
            self.mean_nearest_positive_cosine,
            self.mean_nearest_negative_cosine,
            self.mean_margin,
        )
        if any(
            type(value) is not float or not torch.isfinite(torch.tensor(value))
            for value in values
        ):
            raise ValueError("projected means must be concrete finite floats")
        if not _is_hex(self.folded_state_sha256, 64):
            raise ValueError("projected state digest differs")
        if (
            type(self.wall_time_ns) is not int
            or self.wall_time_ns <= 0
            or type(self.peak_cuda_bytes) is not int
            or self.peak_cuda_bytes < 0
            or type(self.peak_rss_bytes) is not int
            or self.peak_rss_bytes <= 0
            or type(self.determinism_replay) is not bool
            or not self.determinism_replay
        ):
            raise ValueError("projected resource or determinism evidence differs")


@dataclass(frozen=True)
class CandidateEvaluation:
    """Raw and three-head evaluation evidence for one fixed candidate tower."""

    role: str
    raw_correctness: tuple[bool, ...]
    raw_mean_nearest_positive_cosine: float
    raw_mean_nearest_negative_cosine: float
    raw_mean_margin: float
    raw_wall_time_ns: int
    raw_peak_cuda_bytes: int
    raw_peak_rss_bytes: int
    raw_determinism_replay: bool
    projected: tuple[ProjectedEvaluation, ...]
    tower_state_sha256: str
    construction_evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in (
            "tower-soup",
            "wiener-denoise",
            "spectral-denoise",
        ):
            raise ValueError("candidate role differs")
        if (
            type(self.raw_correctness) is not tuple
            or len(self.raw_correctness) != 1345
            or any(type(value) is not bool for value in self.raw_correctness)
        ):
            raise ValueError("raw correctness evidence differs")
        raw_values = (
            self.raw_mean_nearest_positive_cosine,
            self.raw_mean_nearest_negative_cosine,
            self.raw_mean_margin,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in raw_values):
            raise ValueError("raw means must be concrete finite floats")
        if (
            type(self.raw_wall_time_ns) is not int
            or self.raw_wall_time_ns <= 0
            or type(self.raw_peak_cuda_bytes) is not int
            or self.raw_peak_cuda_bytes < 0
            or type(self.raw_peak_rss_bytes) is not int
            or self.raw_peak_rss_bytes <= 0
            or type(self.raw_determinism_replay) is not bool
            or not self.raw_determinism_replay
        ):
            raise ValueError("raw resource or determinism evidence differs")
        if (
            type(self.projected) is not tuple
            or tuple(row.seed for row in self.projected) != _SEEDS
            or any(type(row) is not ProjectedEvaluation for row in self.projected)
        ):
            raise ValueError("projected seed order differs")
        if not _is_hex(self.tower_state_sha256, 64) or not _is_hex(
            self.construction_evidence_sha256, 64
        ):
            raise ValueError("candidate digest differs")


@dataclass(frozen=True)
class HeadSwapEvaluation:
    """One ordered source-tower/target-head coadaptation control."""

    source_seed: int
    target_seed: int
    own_correctness: tuple[bool, ...]
    swapped_correctness: tuple[bool, ...]
    own_mean_margin: float
    swapped_mean_margin: float

    def __post_init__(self) -> None:
        if (
            type(self.source_seed) is not int
            or type(self.target_seed) is not int
            or self.source_seed not in _SEEDS
            or self.target_seed not in _SEEDS
            or self.source_seed == self.target_seed
        ):
            raise ValueError("swap seed pair differs")
        for correctness in (self.own_correctness, self.swapped_correctness):
            if (
                type(correctness) is not tuple
                or len(correctness) != 1345
                or any(type(value) is not bool for value in correctness)
            ):
                raise ValueError("swap correctness evidence differs")
        if any(
            type(value) is not float or not torch.isfinite(torch.tensor(value))
            for value in (self.own_mean_margin, self.swapped_mean_margin)
        ):
            raise ValueError("swap margins must be concrete finite floats")


@dataclass(frozen=True)
class PairedDenoisingEvidence:
    """Per-seed exact paired evidence against the selected scalar comparator."""

    seed: int
    candidate_only: int
    scalar_only: int
    mcnemar_p_value: float


@dataclass(frozen=True)
class DenoisingDecision:
    """Recomputed terminal classification for the fixed three-candidate experiment."""

    terminal_class: str
    selected_candidate: str | None
    best_scalar_alpha: float
    candidate_passes: tuple[bool, bool, bool]
    reaches_95_percent: tuple[bool, bool, bool]
    head_coadaptation_observed: bool
    paired_evidence: tuple[tuple[PairedDenoisingEvidence, ...], ...]

    def __post_init__(self) -> None:
        terminals = {
            "authority-failure",
            "numerical-failure",
            "resource-failure",
            "spectral-denoise-benefit",
            "wiener-denoise-benefit",
            "tower-soup-only-benefit",
            "no-cross-seed-benefit-with-head-coadaptation",
            "no-cross-seed-benefit",
        }
        if self.terminal_class not in terminals:
            raise ValueError("denoising terminal class differs")
        if self.selected_candidate not in (None, *_CANDIDATE_ROLES):
            raise ValueError("selected candidate differs")
        if type(self.best_scalar_alpha) is not float:
            raise ValueError("best scalar alpha differs")
        if (
            type(self.candidate_passes) is not tuple
            or len(self.candidate_passes) != 3
            or any(type(value) is not bool for value in self.candidate_passes)
            or type(self.reaches_95_percent) is not tuple
            or len(self.reaches_95_percent) != 3
            or any(type(value) is not bool for value in self.reaches_95_percent)
            or type(self.head_coadaptation_observed) is not bool
        ):
            raise ValueError("denoising decision evidence differs")
        if (
            type(self.paired_evidence) is not tuple
            or len(self.paired_evidence) != 3
            or any(tuple(row.seed for row in rows) != _SEEDS for rows in self.paired_evidence)
        ):
            raise ValueError("paired evidence seed order differs")


def wiener_gain(rho: float) -> float:
    """Return the fixed three-replicate Wiener gain for a bounded correlation."""

    if type(rho) is not float or not torch.isfinite(torch.tensor(rho)) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be a finite float in [0, 1]")
    return 3.0 * rho / (1.0 + 2.0 * rho)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = float(torch.dot(left.reshape(-1), right.reshape(-1)) / (left_norm * right_norm))
    endpoint_tolerance = 8.0 * torch.finfo(torch.float64).eps
    if abs(value - 1.0) <= endpoint_tolerance:
        return 1.0
    if abs(value + 1.0) <= endpoint_tolerance:
        return -1.0
    return min(max(value, -1.0), 1.0)


def _wiener_evidence(
    name: str,
    updates: tuple[torch.Tensor, ...],
) -> tuple[GroupEvidence, torch.Tensor]:
    cosines = (
        _cosine(updates[0], updates[1]),
        _cosine(updates[0], updates[2]),
        _cosine(updates[1], updates[2]),
    )
    if any(float(torch.linalg.vector_norm(update)) == 0.0 for update in updates):
        rho = 0.0
    else:
        rho = min(max(sum(cosines) / 3.0, 0.0), 1.0)
    beta = wiener_gain(float(rho))
    mean = sum(updates[1:], updates[0].clone()) / 3.0
    residual_energy = sum(
        (float(torch.dot((update - mean).reshape(-1), (update - mean).reshape(-1))))
        for update in updates
    )
    noise = residual_energy / 6.0
    mean_energy = float(torch.dot(mean.reshape(-1), mean.reshape(-1)))
    positive = max(mean_energy - noise, 0.0)
    denominator = positive + noise
    g_js = positive / denominator if denominator > 0.0 else 0.0
    evidence = GroupEvidence(
        name=name,
        cosines=cosines,
        rho=float(rho),
        beta=beta,
        g_js=g_js,
    )
    return evidence, mean * beta


def _spectral_estimate(
    name: str,
    updates: tuple[torch.Tensor, ...],
) -> tuple[SpectralEvidence, torch.Tensor]:
    original_shape = updates[0].shape
    rows = original_shape[0]
    columns = updates[0].numel() // rows
    matrices = tuple(update.reshape(rows, columns) for update in updates)
    mean = sum(matrices[1:], matrices[0].clone()) / 3.0
    contrasts = (
        (matrices[0] - matrices[1]) / _SQRT2,
        (matrices[0] - matrices[2]) / _SQRT2,
        (matrices[1] - matrices[2]) / _SQRT2,
    )
    edge = max(float(torch.linalg.matrix_norm(contrast, ord=2)) for contrast in contrasts) / _SQRT3
    left, singular_values, right = torch.linalg.svd(mean, full_matrices=False)
    sigma1 = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(rows, columns)
        * max(sigma1, edge, 1.0)
    )
    if sigma1 == 0.0 and edge == 0.0:
        retained = torch.zeros_like(singular_values, dtype=torch.bool)
    else:
        if bool(torch.any(torch.abs(singular_values - edge) <= tolerance)):
            raise SpectralEdgeAmbiguity(
                f"tensor {name!r} has a singular value at the spectral edge"
            )
        retained = singular_values > edge
        # Adjacent values inside the tolerance are one decision cluster. A cluster
        # cannot straddle the edge because that would have triggered the guard above.
        start = 0
        while start < singular_values.numel():
            end = start + 1
            while end < singular_values.numel() and (
                abs(float(singular_values[end - 1] - singular_values[end])) <= tolerance
            ):
                end += 1
            decision = bool(retained[start])
            retained[start:end] = decision
            start = end
    filtered = singular_values * retained.to(singular_values.dtype)
    estimate = ((left * filtered.unsqueeze(0)) @ right).reshape(original_shape)
    energies = singular_values.square()
    total_energy = float(energies.sum())
    retained_energy = float((energies * retained.to(energies.dtype)).sum())
    evidence = SpectralEvidence(
        name=name,
        edge=edge,
        tolerance=tolerance,
        kept_rank=int(retained.sum()),
        total_rank=int(singular_values.numel()),
        retained_energy=retained_energy,
        total_energy=total_energy,
        singular_values=tuple(float(value) for value in singular_values),
        retained=tuple(bool(value) for value in retained),
    )
    return evidence, estimate


def build_cross_seed_candidates(
    initial: object,
    endpoints: object,
) -> CandidateStates:
    """Build the fixed soup, Wiener, and symmetric spectral candidate towers."""

    if not isinstance(initial, OrderedDict) or not initial:
        raise ValueError("initial state must be a non-empty OrderedDict")
    if type(endpoints) is not dict or set(endpoints) != set(_SEEDS):
        raise ValueError("endpoints must contain exactly seeds 17, 29, and 43")
    endpoint_map = cast(dict[int, object], endpoints)
    names = tuple(sorted(initial))
    for seed in _SEEDS:
        state = endpoint_map[seed]
        if not isinstance(state, OrderedDict) or set(state) != set(names):
            raise ValueError(f"seed {seed} state names differ")

    soup: OrderedDict[str, torch.Tensor] = OrderedDict()
    wiener: OrderedDict[str, torch.Tensor] = OrderedDict()
    spectral: OrderedDict[str, torch.Tensor] = OrderedDict()
    group_rows: list[GroupEvidence] = []
    spectral_rows: list[SpectralEvidence] = []
    for name in names:
        initial_value = initial[name]
        if not isinstance(initial_value, torch.Tensor):
            raise TypeError("state values must be tensors")
        values: list[torch.Tensor] = []
        for seed in _SEEDS:
            endpoint_value = cast(OrderedDict[str, object], endpoint_map[seed])[name]
            if not isinstance(endpoint_value, torch.Tensor):
                raise TypeError("state values must be tensors")
            if (
                endpoint_value.shape != initial_value.shape
                or endpoint_value.dtype != initial_value.dtype
            ):
                raise ValueError(f"tensor {name!r} shape or dtype differs")
            values.append(endpoint_value)
        if not initial_value.is_floating_point():
            if any(not torch.equal(initial_value, value) for value in values):
                raise ValueError(f"non-floating tensor {name!r} differs")
            soup[name] = initial_value.detach().cpu().contiguous().clone()
            wiener[name] = initial_value.detach().cpu().contiguous().clone()
            spectral[name] = initial_value.detach().cpu().contiguous().clone()
            continue
        all_values = (initial_value, *values)
        if any(not bool(torch.isfinite(value).all()) for value in all_values):
            raise ValueError(f"floating tensor {name!r} must be finite")
        base = initial_value.detach().cpu().to(torch.float64)
        updates = tuple(value.detach().cpu().to(torch.float64) - base for value in values)
        group, wiener_update = _wiener_evidence(name, updates)
        mean = sum(updates[1:], updates[0].clone()) / 3.0
        if initial_value.ndim >= 2:
            spectral_row, spectral_update = _spectral_estimate(name, updates)
            spectral_rows.append(spectral_row)
        else:
            spectral_update = wiener_update
        soup[name] = (base + mean).to(torch.float32).contiguous()
        wiener[name] = (base + wiener_update).to(torch.float32).contiguous()
        spectral[name] = (base + spectral_update).to(torch.float32).contiguous()
        group_rows.append(group)
    return CandidateStates(
        tower_soup=soup,
        wiener_denoise=wiener,
        spectral_denoise=spectral,
        groups=tuple(group_rows),
        spectral=tuple(spectral_rows),
    )


def _mcnemar(
    candidate: tuple[bool, ...],
    scalar: tuple[bool, ...],
    seed: int,
) -> PairedDenoisingEvidence:
    candidate_only = sum(
        left and not right for left, right in zip(candidate, scalar, strict=True)
    )
    scalar_only = sum(
        right and not left for left, right in zip(candidate, scalar, strict=True)
    )
    disagreements = candidate_only + scalar_only
    if disagreements == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(disagreements, value)
            for value in range(min(candidate_only, scalar_only) + 1)
        ) / (2**disagreements)
        p_value = min(1.0, 2.0 * tail)
    return PairedDenoisingEvidence(
        seed=seed,
        candidate_only=candidate_only,
        scalar_only=scalar_only,
        mcnemar_p_value=p_value,
    )


def _best_scalar_rows(
    curves: object,
) -> tuple[float, tuple[AlphaEvaluation, AlphaEvaluation, AlphaEvaluation]]:
    if (
        type(curves) is not tuple
        or len(curves) != 3
        or any(type(curve) is not SeedInterpolationCurve for curve in curves)
        or tuple(curve.seed for curve in curves) != _SEEDS
    ):
        raise ValueError("scalar curves differ from the registered seed order")
    typed = cast(tuple[SeedInterpolationCurve, ...], curves)
    if any(any(row.queries != 1345 for row in curve.rows) for curve in typed):
        raise ValueError("scalar query cardinality differs")
    choices: list[tuple[int, float, float, tuple[AlphaEvaluation, ...]]] = []
    for index, alpha in enumerate(typed[0].rows):
        rows = tuple(curve.rows[index] for curve in typed)
        if any(row.alpha != alpha.alpha for row in rows):
            raise ValueError("scalar alpha grid differs")
        choices.append(
            (
                sum(row.correct for row in rows),
                sum(row.mean_margin for row in rows) / 3.0,
                alpha.alpha,
                rows,
            )
        )
    _, _, selected_alpha, selected_rows = max(
        choices, key=lambda item: (item[0], item[1], item[2])
    )
    return selected_alpha, cast(
        tuple[AlphaEvaluation, AlphaEvaluation, AlphaEvaluation], selected_rows
    )


def classify_denoising_result(
    scalar_curves: object,
    candidates: object,
    swaps: object,
    *,
    failure: str | None = None,
) -> DenoisingDecision:
    """Recompute all fixed cross-seed quality gates and terminal precedence."""

    best_alpha, scalar_rows = _best_scalar_rows(scalar_curves)
    if (
        type(candidates) is not tuple
        or len(candidates) != 3
        or any(type(candidate) is not CandidateEvaluation for candidate in candidates)
        or tuple(candidate.role for candidate in candidates) != _CANDIDATE_ROLES
    ):
        raise ValueError("candidate order differs")
    typed_candidates = cast(tuple[CandidateEvaluation, ...], candidates)
    expected_swaps = tuple(
        (source, target)
        for source in _SEEDS
        for target in _SEEDS
        if source != target
    )
    if (
        type(swaps) is not tuple
        or len(swaps) != 6
        or any(type(row) is not HeadSwapEvaluation for row in swaps)
        or tuple((row.source_seed, row.target_seed) for row in swaps) != expected_swaps
    ):
        raise ValueError("head swap order differs")
    typed_swaps = cast(tuple[HeadSwapEvaluation, ...], swaps)
    if failure not in (None, "authority-failure", "numerical-failure", "resource-failure"):
        raise ValueError("failure class differs")

    scalar_correct = sum(row.correct for row in scalar_rows)
    scalar_margin = sum(row.mean_margin for row in scalar_rows) / 3.0
    candidate_correct = tuple(
        sum(sum(row.correctness) for row in candidate.projected)
        for candidate in typed_candidates
    )
    candidate_margins = tuple(
        sum(row.mean_margin for row in candidate.projected) / 3.0
        for candidate in typed_candidates
    )
    seed_floor = tuple(
        all(
            sum(row.correctness) >= scalar.correct - 1
            for row, scalar in zip(candidate.projected, scalar_rows, strict=True)
        )
        for candidate in typed_candidates
    )
    soup_pass = (
        candidate_correct[0] - scalar_correct >= 9
        and seed_floor[0]
        and candidate_margins[0] > scalar_margin
    )
    wiener_pass = (
        candidate_correct[1] - scalar_correct >= 9
        and candidate_correct[1] - candidate_correct[0] >= 5
        and seed_floor[1]
        and candidate_margins[1] > scalar_margin
        and candidate_margins[1] > candidate_margins[0]
    )
    spectral_pass = (
        candidate_correct[2] - scalar_correct >= 9
        and candidate_correct[2] - candidate_correct[0] >= 5
        and candidate_correct[2] - candidate_correct[1] >= 5
        and seed_floor[2]
        and candidate_margins[2] > scalar_margin
        and candidate_margins[2] > candidate_margins[0]
        and candidate_margins[2] > candidate_margins[1]
    )
    candidate_passes = (soup_pass, wiener_pass, spectral_pass)
    reaches_95 = tuple(
        all(sum(row.correctness) >= 1278 for row in candidate.projected)
        for candidate in typed_candidates
    )
    coadaptation = sum(sum(row.swapped_correctness) for row in typed_swaps) < sum(
        sum(row.own_correctness) for row in typed_swaps
    )
    paired = tuple(
        tuple(
            _mcnemar(row.correctness, scalar.correctness, row.seed)
            for row, scalar in zip(candidate.projected, scalar_rows, strict=True)
        )
        for candidate in typed_candidates
    )

    if failure is not None:
        terminal = failure
        selected = None
        candidate_passes = (False, False, False)
    elif spectral_pass:
        terminal = "spectral-denoise-benefit"
        selected = "spectral-denoise"
    elif wiener_pass:
        terminal = "wiener-denoise-benefit"
        selected = "wiener-denoise"
    elif soup_pass:
        terminal = "tower-soup-only-benefit"
        selected = "tower-soup"
    elif coadaptation:
        terminal = "no-cross-seed-benefit-with-head-coadaptation"
        selected = None
    else:
        terminal = "no-cross-seed-benefit"
        selected = None
    return DenoisingDecision(
        terminal_class=terminal,
        selected_candidate=selected,
        best_scalar_alpha=best_alpha,
        candidate_passes=candidate_passes,
        reaches_95_percent=cast(tuple[bool, bool, bool], reaches_95),
        head_coadaptation_observed=coadaptation,
        paired_evidence=paired,
    )


def _correctness_payload(correctness: tuple[bool, ...]) -> dict[str, object]:
    raw = bytes(correctness)
    return {
        "bits": raw.hex(),
        "correct": sum(correctness),
        "queries": len(correctness),
        "recall_ppm": sum(correctness) * 1_000_000 // len(correctness),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _projected_payload(row: ProjectedEvaluation) -> dict[str, object]:
    return {
        "correctness": _correctness_payload(row.correctness),
        "determinism_replay": row.determinism_replay,
        "folded_state_sha256": row.folded_state_sha256,
        "mean_margin": row.mean_margin,
        "mean_nearest_negative_cosine": row.mean_nearest_negative_cosine,
        "mean_nearest_positive_cosine": row.mean_nearest_positive_cosine,
        "peak_cuda_bytes": row.peak_cuda_bytes,
        "peak_rss_bytes": row.peak_rss_bytes,
        "seed": row.seed,
        "wall_time_ns": row.wall_time_ns,
    }


def _candidate_payload(candidate: CandidateEvaluation) -> dict[str, object]:
    aggregate_correct = sum(sum(row.correctness) for row in candidate.projected)
    return {
        "aggregate_correct": aggregate_correct,
        "aggregate_queries": 4035,
        "aggregate_recall_ppm": aggregate_correct * 1_000_000 // 4035,
        "construction_evidence_sha256": candidate.construction_evidence_sha256,
        "mean_projected_margin": sum(row.mean_margin for row in candidate.projected) / 3.0,
        "projected": [_projected_payload(row) for row in candidate.projected],
        "raw_correctness": _correctness_payload(candidate.raw_correctness),
        "raw_mean_margin": candidate.raw_mean_margin,
        "raw_mean_nearest_negative_cosine": candidate.raw_mean_nearest_negative_cosine,
        "raw_mean_nearest_positive_cosine": candidate.raw_mean_nearest_positive_cosine,
        "raw_peak_cuda_bytes": candidate.raw_peak_cuda_bytes,
        "raw_peak_rss_bytes": candidate.raw_peak_rss_bytes,
        "raw_determinism_replay": candidate.raw_determinism_replay,
        "raw_wall_time_ns": candidate.raw_wall_time_ns,
        "role": candidate.role,
        "tower_state_sha256": candidate.tower_state_sha256,
    }


def _alpha_payload(row: AlphaEvaluation) -> dict[str, object]:
    return {
        "alpha": row.alpha,
        "correctness": _correctness_payload(row.correctness),
        "folded_state_sha256": row.folded_state_sha256,
        "mean_margin": row.mean_margin,
        "mean_nearest_negative_cosine": row.mean_nearest_negative_cosine,
        "mean_nearest_positive_cosine": row.mean_nearest_positive_cosine,
        "peak_cuda_bytes": row.peak_cuda_bytes,
        "peak_rss_bytes": row.peak_rss_bytes,
        "seed": row.seed,
        "tower_squared_displacement": row.tower_squared_displacement,
        "wall_time_ns": row.wall_time_ns,
    }


def _swap_payload(row: HeadSwapEvaluation) -> dict[str, object]:
    return {
        "own_correctness": _correctness_payload(row.own_correctness),
        "own_mean_margin": row.own_mean_margin,
        "source_seed": row.source_seed,
        "swapped_correctness": _correctness_payload(row.swapped_correctness),
        "swapped_mean_margin": row.swapped_mean_margin,
        "target_seed": row.target_seed,
    }


def _decision_payload(decision: DenoisingDecision) -> dict[str, object]:
    return {
        "best_scalar_alpha": decision.best_scalar_alpha,
        "candidate_passes": list(decision.candidate_passes),
        "head_coadaptation_observed": decision.head_coadaptation_observed,
        "paired_evidence": [
            [
                {
                    "candidate_only": row.candidate_only,
                    "mcnemar_p_value": row.mcnemar_p_value,
                    "scalar_only": row.scalar_only,
                    "seed": row.seed,
                }
                for row in rows
            ]
            for rows in decision.paired_evidence
        ],
        "reaches_95_percent": list(decision.reaches_95_percent),
        "selected_candidate": decision.selected_candidate,
        "terminal_class": decision.terminal_class,
    }


def canonical_denoising_result_bytes(
    scalar_curves: object,
    candidates: object,
    swaps: object,
    decision: object,
    *,
    failure: str | None = None,
) -> bytes:
    """Serialize a claim-ineligible result after complete recomputation."""

    recomputed = classify_denoising_result(
        scalar_curves, candidates, swaps, failure=failure
    )
    if type(decision) is not DenoisingDecision or decision != recomputed:
        raise ValueError("stored denoising decision differs from recomputation")
    typed_curves = cast(tuple[SeedInterpolationCurve, ...], scalar_curves)
    typed_candidates = cast(tuple[CandidateEvaluation, ...], candidates)
    typed_swaps = cast(tuple[HeadSwapEvaluation, ...], swaps)
    payload: dict[str, object] = {
        "candidates": [_candidate_payload(candidate) for candidate in typed_candidates],
        "claim_eligible": False,
        "decision": _decision_payload(recomputed),
        "failure": failure,
        "scalar_curves": [
            {"rows": [_alpha_payload(row) for row in curve.rows], "seed": curve.seed}
            for curve in typed_curves
        ],
        "schema": "sfora-cross-seed-denoising-result-v1",
        "swaps": [_swap_payload(row) for row in typed_swaps],
    }
    raw = _canonical_json_bytes(payload)
    if read_denoising_result(raw) != recomputed:
        raise ValueError("serialized denoising result failed validation")
    return raw


def _parse_correctness(value: object, *, role: str) -> tuple[bool, ...]:
    keys = {"bits", "correct", "queries", "recall_ppm", "sha256"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{role} correctness schema differs")
    typed = cast(dict[str, object], value)
    bits = typed["bits"]
    if type(bits) is not str:
        raise ValueError(f"{role} correctness bits differ")
    try:
        raw = bytes.fromhex(bits)
    except ValueError as exc:
        raise ValueError(f"{role} correctness bits differ") from exc
    if any(item not in (0, 1) for item in raw):
        raise ValueError(f"{role} correctness bits differ")
    correctness = tuple(bool(item) for item in raw)
    correct = sum(correctness)
    if (
        type(typed["correct"]) is not int
        or typed["correct"] != correct
        or type(typed["queries"]) is not int
        or typed["queries"] != len(correctness)
        or type(typed["recall_ppm"]) is not int
        or typed["recall_ppm"] != correct * 1_000_000 // len(correctness)
        or typed["sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError(f"{role} correctness aggregate differs")
    return correctness


def _parse_scalar_curves(value: object) -> tuple[SeedInterpolationCurve, ...]:
    if type(value) is not list or len(value) != 3:
        raise ValueError("scalar curve schema differs")
    curves: list[SeedInterpolationCurve] = []
    row_keys = {
        "alpha",
        "correctness",
        "folded_state_sha256",
        "mean_margin",
        "mean_nearest_negative_cosine",
        "mean_nearest_positive_cosine",
        "peak_cuda_bytes",
        "peak_rss_bytes",
        "seed",
        "tower_squared_displacement",
        "wall_time_ns",
    }
    for raw_curve in cast(list[object], value):
        if type(raw_curve) is not dict or set(raw_curve) != {"rows", "seed"}:
            raise ValueError("scalar curve schema differs")
        curve = cast(dict[str, object], raw_curve)
        if type(curve["rows"]) is not list:
            raise ValueError("scalar curve rows differ")
        rows: list[AlphaEvaluation] = []
        for raw_row in cast(list[object], curve["rows"]):
            if type(raw_row) is not dict or set(raw_row) != row_keys:
                raise ValueError("scalar row schema differs")
            row = cast(dict[str, object], raw_row)
            correctness = _parse_correctness(row["correctness"], role="scalar")
            rows.append(
                AlphaEvaluation(
                    seed=cast(int, row["seed"]),
                    alpha=cast(float, row["alpha"]),
                    correct=sum(correctness),
                    queries=len(correctness),
                    recall_ppm=sum(correctness) * 1_000_000 // len(correctness),
                    mean_nearest_positive_cosine=cast(
                        float, row["mean_nearest_positive_cosine"]
                    ),
                    mean_nearest_negative_cosine=cast(
                        float, row["mean_nearest_negative_cosine"]
                    ),
                    mean_margin=cast(float, row["mean_margin"]),
                    correctness=correctness,
                    folded_state_sha256=cast(str, row["folded_state_sha256"]),
                    tower_squared_displacement=cast(
                        float, row["tower_squared_displacement"]
                    ),
                    wall_time_ns=cast(int, row["wall_time_ns"]),
                    peak_cuda_bytes=cast(int, row["peak_cuda_bytes"]),
                    peak_rss_bytes=cast(int, row["peak_rss_bytes"]),
                )
            )
        curves.append(
            SeedInterpolationCurve(seed=cast(int, curve["seed"]), rows=tuple(rows))
        )
    return tuple(curves)


def _parse_candidates(value: object) -> tuple[CandidateEvaluation, ...]:
    if type(value) is not list or len(value) != 3:
        raise ValueError("candidate order differs")
    candidate_keys = {
        "aggregate_correct",
        "aggregate_queries",
        "aggregate_recall_ppm",
        "construction_evidence_sha256",
        "mean_projected_margin",
        "projected",
        "raw_correctness",
        "raw_mean_margin",
        "raw_mean_nearest_negative_cosine",
        "raw_mean_nearest_positive_cosine",
        "raw_peak_cuda_bytes",
        "raw_peak_rss_bytes",
        "raw_determinism_replay",
        "raw_wall_time_ns",
        "role",
        "tower_state_sha256",
    }
    projected_keys = {
        "correctness",
        "determinism_replay",
        "folded_state_sha256",
        "mean_margin",
        "mean_nearest_negative_cosine",
        "mean_nearest_positive_cosine",
        "peak_cuda_bytes",
        "peak_rss_bytes",
        "seed",
        "wall_time_ns",
    }
    candidates: list[CandidateEvaluation] = []
    for raw_candidate in cast(list[object], value):
        if type(raw_candidate) is not dict or set(raw_candidate) != candidate_keys:
            raise ValueError("candidate schema differs")
        candidate = cast(dict[str, object], raw_candidate)
        if type(candidate["projected"]) is not list:
            raise ValueError("candidate projected schema differs")
        projected: list[ProjectedEvaluation] = []
        for raw_row in cast(list[object], candidate["projected"]):
            if type(raw_row) is not dict or set(raw_row) != projected_keys:
                raise ValueError("candidate projected schema differs")
            row = cast(dict[str, object], raw_row)
            projected.append(
                ProjectedEvaluation(
                    seed=cast(int, row["seed"]),
                    correctness=_parse_correctness(row["correctness"], role="projected"),
                    mean_nearest_positive_cosine=cast(
                        float, row["mean_nearest_positive_cosine"]
                    ),
                    mean_nearest_negative_cosine=cast(
                        float, row["mean_nearest_negative_cosine"]
                    ),
                    mean_margin=cast(float, row["mean_margin"]),
                    folded_state_sha256=cast(str, row["folded_state_sha256"]),
                    wall_time_ns=cast(int, row["wall_time_ns"]),
                    peak_cuda_bytes=cast(int, row["peak_cuda_bytes"]),
                    peak_rss_bytes=cast(int, row["peak_rss_bytes"]),
                    determinism_replay=cast(bool, row["determinism_replay"]),
                )
            )
        built = CandidateEvaluation(
            role=cast(str, candidate["role"]),
            raw_correctness=_parse_correctness(candidate["raw_correctness"], role="raw"),
            raw_mean_nearest_positive_cosine=cast(
                float, candidate["raw_mean_nearest_positive_cosine"]
            ),
            raw_mean_nearest_negative_cosine=cast(
                float, candidate["raw_mean_nearest_negative_cosine"]
            ),
            raw_mean_margin=cast(float, candidate["raw_mean_margin"]),
            raw_wall_time_ns=cast(int, candidate["raw_wall_time_ns"]),
            raw_peak_cuda_bytes=cast(int, candidate["raw_peak_cuda_bytes"]),
            raw_peak_rss_bytes=cast(int, candidate["raw_peak_rss_bytes"]),
            raw_determinism_replay=cast(bool, candidate["raw_determinism_replay"]),
            projected=tuple(projected),
            tower_state_sha256=cast(str, candidate["tower_state_sha256"]),
            construction_evidence_sha256=cast(
                str, candidate["construction_evidence_sha256"]
            ),
        )
        expected = _candidate_payload(built)
        for key in ("aggregate_correct", "aggregate_queries", "aggregate_recall_ppm"):
            if candidate[key] != expected[key]:
                raise ValueError("candidate aggregate differs")
        if candidate["mean_projected_margin"] != expected["mean_projected_margin"]:
            raise ValueError("candidate aggregate margin differs")
        candidates.append(built)
    if tuple(candidate.role for candidate in candidates) != _CANDIDATE_ROLES:
        raise ValueError("candidate order differs")
    return tuple(candidates)


def _parse_swaps(value: object) -> tuple[HeadSwapEvaluation, ...]:
    keys = {
        "own_correctness",
        "own_mean_margin",
        "source_seed",
        "swapped_correctness",
        "swapped_mean_margin",
        "target_seed",
    }
    if type(value) is not list or len(value) != 6:
        raise ValueError("head swap order differs")
    rows: list[HeadSwapEvaluation] = []
    for raw_row in cast(list[object], value):
        if type(raw_row) is not dict or set(raw_row) != keys:
            raise ValueError("head swap schema differs")
        row = cast(dict[str, object], raw_row)
        rows.append(
            HeadSwapEvaluation(
                source_seed=cast(int, row["source_seed"]),
                target_seed=cast(int, row["target_seed"]),
                own_correctness=_parse_correctness(row["own_correctness"], role="swap own"),
                swapped_correctness=_parse_correctness(
                    row["swapped_correctness"], role="swap target"
                ),
                own_mean_margin=cast(float, row["own_mean_margin"]),
                swapped_mean_margin=cast(float, row["swapped_mean_margin"]),
            )
        )
    return tuple(rows)


def read_denoising_result(raw: bytes) -> DenoisingDecision:
    """Authenticate, recompute, and return one canonical denoising decision."""

    if type(raw) is not bytes:
        raise TypeError("denoising result must be concrete bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("denoising result is not valid JSON") from exc
    keys = {
        "candidates",
        "claim_eligible",
        "decision",
        "failure",
        "scalar_curves",
        "schema",
        "swaps",
    }
    if type(value) is not dict or _canonical_json_bytes(value) != raw:
        raise ValueError("denoising result is not canonical")
    if set(value) != keys or value["schema"] != "sfora-cross-seed-denoising-result-v1":
        raise ValueError("denoising result schema differs")
    if type(value["claim_eligible"]) is not bool or value["claim_eligible"] is not False:
        raise ValueError("denoising result claim eligibility differs")
    failure = value["failure"]
    if failure is not None and type(failure) is not str:
        raise ValueError("denoising failure class differs")
    curves = _parse_scalar_curves(value["scalar_curves"])
    candidates = _parse_candidates(value["candidates"])
    swaps = _parse_swaps(value["swaps"])
    decision = classify_denoising_result(curves, candidates, swaps, failure=failure)
    if value["decision"] != _decision_payload(decision):
        raise ValueError("stored denoising decision differs")
    return decision


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_bindings(value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ValueError("bindings must be a non-empty concrete string mapping")
    bindings = cast(dict[object, object], value)
    if any(type(key) is not str or type(item) is not str for key, item in bindings.items()):
        raise ValueError("bindings must contain concrete strings")
    return {cast(str, key): cast(str, item) for key, item in sorted(bindings.items())}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _state_digest(
    *,
    role: str,
    bindings: Mapping[str, str],
    records: list[dict[str, object]],
    payloads: list[bytes],
) -> str:
    digest = hashlib.sha256()
    header = {
        "bindings": dict(bindings),
        "role": role,
        "tensors": [
            {
                "bytes": record["bytes"],
                "dtype": record["dtype"],
                "name": record["name"],
                "shape": record["shape"],
            }
            for record in records
        ],
    }
    header_bytes = _canonical_json_bytes(header)
    digest.update(len(header_bytes).to_bytes(8, "little"))
    digest.update(header_bytes)
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def write_tensor_artifact(
    root: Path,
    state: object,
    *,
    role: str,
    bindings: object,
) -> bytes:
    """Write a deterministic, content-addressed tensor artifact."""

    if type(role) is not str or not role:
        raise ValueError("role must be a non-empty concrete string")
    normalized_bindings = _validate_bindings(bindings)
    if not isinstance(state, OrderedDict) or not state:
        raise ValueError("state must be a non-empty OrderedDict")
    if root.exists():
        raise ValueError("artifact root already exists")

    records: list[dict[str, object]] = []
    payloads: list[bytes] = []
    for ordinal, (name, tensor_value) in enumerate(sorted(state.items())):
        if type(name) is not str or not name:
            raise ValueError("tensor name must be a non-empty concrete string")
        if not isinstance(tensor_value, torch.Tensor):
            raise TypeError("state values must be tensors")
        tensor = tensor_value.detach().cpu().contiguous()
        dtype_name = _DTYPE_NAMES.get(tensor.dtype)
        if dtype_name is None:
            raise ValueError(f"unsupported tensor dtype: {tensor.dtype}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name!r} must be finite")
        payload = _tensor_bytes(tensor)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        records.append(
            {
                "bytes": len(payload),
                "dtype": dtype_name,
                "file": f"tensors/{ordinal:05d}-{payload_sha256}.bin",
                "name": name,
                "sha256": payload_sha256,
                "shape": list(tensor.shape),
            }
        )
        payloads.append(payload)

    state_sha256 = _state_digest(
        role=role,
        bindings=normalized_bindings,
        records=records,
        payloads=payloads,
    )
    manifest = {
        "bindings": normalized_bindings,
        "claim_eligible": False,
        "role": role,
        "schema": _SCHEMA,
        "state_sha256": state_sha256,
        "tensors": records,
    }
    manifest_bytes = _canonical_json_bytes(manifest)

    tensors_root = root / "tensors"
    tensors_root.mkdir(parents=True)
    try:
        for record, payload in zip(records, payloads, strict=True):
            path = root / cast(str, record["file"])
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        with (root / "manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        for record in records:
            path = root / cast(str, record["file"])
            if path.is_file() and not path.is_symlink():
                path.unlink()
        manifest_path = root / "manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            manifest_path.unlink()
        if tensors_root.is_dir():
            tensors_root.rmdir()
        if root.is_dir():
            root.rmdir()
        raise
    return manifest_bytes


def read_tensor_artifact(
    root: Path,
    manifest_bytes: bytes,
    *,
    role: str,
) -> OrderedDict[str, torch.Tensor]:
    """Authenticate and load a deterministic tensor artifact."""

    if type(manifest_bytes) is not bytes:
        raise TypeError("manifest bytes must be concrete bytes")
    try:
        value = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if type(value) is not dict or _canonical_json_bytes(value) != manifest_bytes:
        raise ValueError("manifest is not canonical JSON")
    if set(value) != _MANIFEST_KEYS:
        raise ValueError("manifest schema differs")
    if value["schema"] != _SCHEMA or type(value["claim_eligible"]) is not bool:
        raise ValueError("manifest schema differs")
    if value["claim_eligible"] is not False:
        raise ValueError("manifest claim eligibility differs")
    if type(role) is not str or value["role"] != role:
        raise ValueError("manifest role differs")
    bindings = _validate_bindings(value["bindings"])
    if not _is_hex(value["state_sha256"], 64):
        raise ValueError("state digest differs")
    if type(value["tensors"]) is not list or not value["tensors"]:
        raise ValueError("tensor schema differs")

    tensors = cast(list[object], value["tensors"])
    records: list[dict[str, object]] = []
    payloads: list[bytes] = []
    expected_names: list[str] = []
    for ordinal, raw_record in enumerate(tensors):
        if type(raw_record) is not dict or set(raw_record) != _TENSOR_KEYS:
            raise ValueError("tensor schema differs")
        record = cast(dict[str, object], raw_record)
        name = record["name"]
        if type(name) is not str or not name:
            raise ValueError("tensor name differs")
        expected_names.append(name)
        if expected_names != sorted(expected_names) or len(set(expected_names)) != len(
            expected_names
        ):
            raise ValueError("tensor order differs")
        dtype_name = record["dtype"]
        if type(dtype_name) is not str or dtype_name not in _DTYPES:
            raise ValueError("tensor dtype differs")
        shape = record["shape"]
        if type(shape) is not list or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError("tensor shape differs")
        byte_count = record["bytes"]
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("tensor length differs")
        payload_sha256 = record["sha256"]
        if not _is_hex(payload_sha256, 64):
            raise ValueError("tensor digest differs")
        expected_file = f"tensors/{ordinal:05d}-{payload_sha256}.bin"
        file_value = record["file"]
        if type(file_value) is not str or file_value != expected_file:
            raise ValueError("tensor path differs")
        pure_path = PurePosixPath(file_value)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("tensor path differs")
        path = root / file_value
        if path.is_symlink():
            raise ValueError("tensor payload must not be a symlink")
        if not path.is_file():
            raise ValueError("tensor payload is missing")
        payload = path.read_bytes()
        if len(payload) != byte_count:
            raise ValueError("tensor payload length differs")
        if hashlib.sha256(payload).hexdigest() != payload_sha256:
            raise ValueError("tensor payload digest differs")
        dtype = _DTYPES[dtype_name]
        expected_elements = 1
        for dimension in shape:
            expected_elements *= dimension
        expected_bytes = expected_elements * torch.empty((), dtype=dtype).element_size()
        if expected_bytes != byte_count:
            raise ValueError("tensor shape and length differ")
        records.append(record)
        payloads.append(payload)

    computed_state_sha256 = _state_digest(
        role=role,
        bindings=bindings,
        records=records,
        payloads=payloads,
    )
    if computed_state_sha256 != value["state_sha256"]:
        raise ValueError("bindings or state digest differs")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for record, payload in zip(records, payloads, strict=True):
        dtype = _DTYPES[cast(str, record["dtype"])]
        owned = bytearray(payload)
        tensor = torch.frombuffer(owned, dtype=dtype).clone()
        result[cast(str, record["name"])] = tensor.reshape(cast(list[int], record["shape"]))
    return result
