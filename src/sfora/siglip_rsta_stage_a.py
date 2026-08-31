"""Prospective, claim-ineligible SigLIP RSTA Stage-A authority."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from sfora.pass209_m4 import canonical_json_bytes
from sfora.siglip_proxy_control import (
    PooledProxyAnchorModel,
    recomputed_proxy_anchor_backward,
)

_LABELS = tuple(range(49))
_PRIMARY_CLASS_DOMAIN = "rsta-siglip-a-v1|class|"
_ALTERNATE_CLASS_DOMAIN = "rsta-siglip-a-v1|alternate-class|"
_ROLE_DOMAIN = "rsta-siglip-a-v1|role|"
_BATCH_ORDER_DOMAIN = "rsta-siglip-a-v1|batch-order|"
_ALTERNATE_DISTRACTOR_DOMAIN = "rsta-siglip-a-v1|alternate-distractor|"


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class RstaCheckpointBinding:
    """Outcome-blind identity for one sealed final checkpoint."""

    seed: int
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or self.seed not in (17, 29, 43)
            or not _is_lower_hex(self.sha256, 64)
            or type(self.byte_length) is not int
            or self.byte_length <= 0
        ):
            raise ValueError("RSTA control binding checkpoint differs from authority")


@dataclass(frozen=True)
class RstaControlBinding:
    """Complete outcome-blind authority projected from sealed control receipts."""

    schema: str
    claim_eligible: bool
    control_complete: bool
    source_commit: str
    config_sha256: str
    run_authority_sha256: str
    dataset_id: str
    dataset_revision: str
    environment_sha256: str
    optimization_manifest_sha256: str
    checkpoints: tuple[RstaCheckpointBinding, ...]

    def __post_init__(self) -> None:
        valid = (
            type(self.schema) is str
            and self.schema == "rsta-control-binding-v1"
            and type(self.claim_eligible) is bool
            and self.claim_eligible is False
            and type(self.control_complete) is bool
            and self.control_complete is True
            and _is_lower_hex(self.source_commit, 40)
            and _is_lower_hex(self.config_sha256, 64)
            and _is_lower_hex(self.run_authority_sha256, 64)
            and type(self.dataset_id) is str
            and self.dataset_id == "tanganke/stanford_cars"
            and _is_lower_hex(self.dataset_revision, 40)
            and _is_lower_hex(self.environment_sha256, 64)
            and _is_lower_hex(self.optimization_manifest_sha256, 64)
            and type(self.checkpoints) is tuple
            and len(self.checkpoints) == 3
            and all(type(item) is RstaCheckpointBinding for item in self.checkpoints)
            and tuple(item.seed for item in self.checkpoints) == (17, 29, 43)
            and len({item.sha256 for item in self.checkpoints}) == 3
        )
        if not valid:
            raise ValueError("RSTA control binding differs from frozen authority")


def rsta_control_binding_bytes(binding: RstaControlBinding) -> bytes:
    """Serialize a validated outcome-blind binding as canonical JSON."""

    if type(binding) is not RstaControlBinding:
        raise ValueError("RSTA control binding has the wrong concrete type")
    return canonical_json_bytes(
        {
            "schema": binding.schema,
            "claim_eligible": binding.claim_eligible,
            "control_complete": binding.control_complete,
            "source_commit": binding.source_commit,
            "config_sha256": binding.config_sha256,
            "run_authority_sha256": binding.run_authority_sha256,
            "dataset_id": binding.dataset_id,
            "dataset_revision": binding.dataset_revision,
            "environment_sha256": binding.environment_sha256,
            "optimization_manifest_sha256": binding.optimization_manifest_sha256,
            "checkpoints": [
                {
                    "seed": item.seed,
                    "sha256": item.sha256,
                    "byte_length": item.byte_length,
                }
                for item in binding.checkpoints
            ],
        }
    )


@dataclass(frozen=True)
class RstaStageAConfig:
    """Frozen prospective gates for the claim-ineligible Stage-A falsifier."""

    seeds: tuple[int, int, int] = (17, 29, 43)
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 200
    pass_delta: float = 0.03
    pass_seed_delta: float = 0.02
    pass_rho: float = 0.20
    fail_rho: float = 0.10
    pass_abs_log_ratio: float = math.log(1.10)
    max_abs_deranged_delta: float = 0.01

    def __post_init__(self) -> None:
        expected: tuple[tuple[str, object], ...] = (
            ("seeds", (17, 29, 43)),
            ("bootstrap_replicates", 10_000),
            ("bootstrap_seed", 200),
            ("pass_delta", 0.03),
            ("pass_seed_delta", 0.02),
            ("pass_rho", 0.20),
            ("fail_rho", 0.10),
            ("pass_abs_log_ratio", math.log(1.10)),
            ("max_abs_deranged_delta", 0.01),
        )
        if any(
            type(getattr(self, name)) is not type(value)
            or getattr(self, name) != value
            for name, value in expected
        ):
            raise ValueError("RSTA Stage-A config differs from the frozen contract")


@dataclass(frozen=True)
class RstaReceiverEvidence:
    """One receiver's registered causal measurements."""

    seed: int
    label: int
    receiver_id: str
    delta: float
    self_minus_desc: float
    rho: float
    abs_log_ratio: float
    deranged_delta: float
    alternate_delta: float


@dataclass(frozen=True)
class RstaSeedEvidence:
    """Class-balanced evidence derived for one control seed."""

    seed: int
    delta: float
    self_minus_desc: float
    alternate_delta: float


@dataclass(frozen=True)
class RstaAggregate:
    """Fully recomputed Stage-A aggregate and decision."""

    verdict: str
    first_decisive_clause: str
    pooled_delta: float
    bootstrap_delta_95_lower: float
    pooled_self_minus_desc: float
    bootstrap_self_minus_desc_95_lower: float
    seed_deltas: tuple[float, float, float]
    alternate_pooled_delta: float
    alternate_seed_deltas: tuple[float, float, float]
    median_rho: float
    median_abs_log_ratio: float
    pooled_deranged_delta: float
    bootstrap_delta_sha256: str
    bootstrap_self_minus_desc_sha256: str
    numpy_version: str
    receiver_count: int
    config: RstaStageAConfig
    receiver_evidence: tuple[RstaReceiverEvidence, ...]


@dataclass(frozen=True)
class ContextualDirectionEvidence:
    """Exact contextual cotangents and selected parameter-space descent direction."""

    dbar: torch.Tensor
    parameter_names: tuple[str, ...]
    parameter_direction: tuple[torch.Tensor, ...]
    maximum_score_disagreement: float


@dataclass(frozen=True)
class RstaReceiverFields:
    """Matrix-free contextual and self motion in one receiver output space."""

    descriptor: torch.Tensor
    batch_motion: torch.Tensor
    self_motion: torch.Tensor
    batch_radial_fraction: float
    self_radial_fraction: float
    backend: str


@dataclass(frozen=True)
class RstaJvpBackendEvidence:
    """Outcome-blind backend preflight decision."""

    backend: str
    comparison_available: bool
    maximum_relative_disagreement: float
    forward_error: str | None


@dataclass(frozen=True)
class RstaOutcomeDirection:
    """Proxy-free tangent outcome direction for one receiver."""

    margin: float
    direction: torch.Tensor
    selected_foreign_indices: tuple[int, ...]
    selected_foreign_ids: tuple[str, ...]


@dataclass(frozen=True)
class RstaReceiverScore:
    """All registered causal statistics and controls for one receiver."""

    a_self: float
    a_batch: float
    delta: float
    a_desc: float
    self_minus_desc: float
    cos_batch_self: float
    rho: float
    log_ratio: float
    cross_contribution: float
    random_delta: float
    deranged_delta: float
    batch_radial_fraction: float
    self_radial_fraction: float
    dbar_radial_fraction: float


@dataclass(frozen=True)
class RstaControlDirections:
    """Deterministic tangent controls for one registered receiver batch."""

    random_targets: torch.Tensor
    deranged_directions: torch.Tensor


@dataclass(frozen=True)
class RstaBatchRow:
    """One immutable row in a registered logical batch."""

    example_id: str
    label: int
    role: str


@dataclass(frozen=True)
class RstaBatch:
    """One 120-row logical batch."""

    index: int
    rows: tuple[RstaBatchRow, ...]


@dataclass(frozen=True)
class RstaReceiverRow:
    """Registered context and outcome roles for one receiver."""

    example_id: str
    label: int
    support_ids: tuple[str, str]
    primary_batch: int
    primary_row: int
    alternate_batch: int
    alternate_row: int
    primary_peer_id: str
    alternate_peer_id: str
    primary_foreign_labels: tuple[int, ...]
    alternate_foreign_labels: tuple[int, ...]


@dataclass(frozen=True)
class RstaRolePanel:
    """Complete deterministic role authority for both Stage-A panels."""

    primary_class_order: tuple[int, ...]
    alternate_class_order: tuple[int, ...]
    ranked_ids_by_label: tuple[tuple[str, ...], ...]
    support_ids_by_label: tuple[tuple[str, str], ...]
    primary_batches: tuple[RstaBatch, RstaBatch]
    alternate_batches: tuple[RstaBatch, RstaBatch]
    receivers: tuple[RstaReceiverRow, ...]


def _hash(domain: str, text: str | int) -> bytes:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + str(text).encode("utf-8")
    ).digest()


def _ordered_classes(domain: str) -> tuple[int, ...]:
    return tuple(sorted(_LABELS, key=lambda label: (_hash(domain, label), label)))


def _batch_class_sets(order: tuple[int, ...]) -> tuple[frozenset[int], frozenset[int]]:
    return (frozenset(order[:30]), frozenset((*order[30:], *order[:11])))


def _ordered_batch(index: int, rows: list[RstaBatchRow]) -> RstaBatch:
    domain = f"{_BATCH_ORDER_DOMAIN}{index}|"
    return RstaBatch(
        index=index,
        rows=tuple(sorted(rows, key=lambda row: (_hash(domain, row.example_id), row.example_id))),
    )


def _validate_examples(
    examples: Sequence[tuple[str, int]],
) -> tuple[tuple[str, ...], ...]:
    by_label: list[list[str]] = [[] for _ in _LABELS]
    seen: set[str] = set()
    observed_labels: set[int] = set()
    for item in examples:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("examples must be concrete (example_id, label) tuples")
        example_id, label = item
        if type(example_id) is not str or not example_id:
            raise ValueError("example_id must be a nonempty string")
        if type(label) is not int:
            raise ValueError("label must be a concrete integer")
        observed_labels.add(label)
        if example_id in seen:
            raise ValueError("duplicate example_id")
        seen.add(example_id)
        if label in _LABELS:
            by_label[label].append(example_id)
    if observed_labels != set(_LABELS):
        raise ValueError("exact optimization labels 0..48 are required")
    if any(len(ids) < 15 for ids in by_label):
        raise ValueError("every optimization label requires at least 15 examples")
    return tuple(
        tuple(sorted(ids, key=lambda example_id: (_hash(_ROLE_DOMAIN, example_id), example_id)))
        for ids in by_label
    )


def _panel_batches(
    *,
    class_order: tuple[int, ...],
    ranked: tuple[tuple[str, ...], ...],
    alternate_roles: tuple[tuple[str, tuple[str, ...]], ...] | None,
) -> tuple[RstaBatch, RstaBatch]:
    batches: list[RstaBatch] = []
    for batch_index in (0, 1):
        main_labels = class_order[:30] if batch_index == 0 else class_order[30:]
        rows: list[RstaBatchRow] = []
        for label in main_labels:
            rows.extend(
                RstaBatchRow(example_id, label, "receiver")
                for example_id in ranked[label][2:5]
            )
            peer = ranked[label][5] if alternate_roles is None else alternate_roles[label][0]
            rows.append(RstaBatchRow(peer, label, "peer"))
        if batch_index == 1:
            for label in class_order[:11]:
                refill = (
                    ranked[label][6:10]
                    if alternate_roles is None
                    else alternate_roles[label][1]
                )
                rows.extend(
                    RstaBatchRow(example_id, label, "refill")
                    for example_id in refill
                )
        if len(rows) != 120:
            raise ValueError("RSTA logical batch does not contain 120 rows")
        batches.append(_ordered_batch(batch_index, rows))
    return (batches[0], batches[1])


def select_rsta_roles(examples: Sequence[tuple[str, int]]) -> RstaRolePanel:
    """Select the frozen Stage-A supports, receivers, peers, and refills."""

    ranked = _validate_examples(examples)
    primary_order = _ordered_classes(_PRIMARY_CLASS_DOMAIN)
    alternate_order = _ordered_classes(_ALTERNATE_CLASS_DOMAIN)
    primary_refill_labels = frozenset(primary_order[:11])

    alternate_roles: list[tuple[str, tuple[str, ...]]] = []
    for label in _LABELS:
        excluded = set(ranked[label][:6])
        if label in primary_refill_labels:
            excluded.update(ranked[label][6:10])
        candidates = sorted(
            (example_id for example_id in ranked[label] if example_id not in excluded),
            key=lambda example_id: (
                _hash(_ALTERNATE_DISTRACTOR_DOMAIN, example_id),
                example_id,
            ),
        )
        if len(candidates) < 5:
            raise ValueError("insufficient disjoint alternate roles")
        alternate_roles.append((candidates[0], tuple(candidates[1:5])))
    alternate_roles_tuple = tuple(alternate_roles)

    primary_batches = _panel_batches(
        class_order=primary_order,
        ranked=ranked,
        alternate_roles=None,
    )
    alternate_batches = _panel_batches(
        class_order=alternate_order,
        ranked=ranked,
        alternate_roles=alternate_roles_tuple,
    )
    primary_sets = _batch_class_sets(primary_order)
    alternate_sets = _batch_class_sets(alternate_order)
    primary_locations = {
        row.example_id: (batch.index, row_index)
        for batch in primary_batches
        for row_index, row in enumerate(batch.rows)
        if row.role == "receiver"
    }
    alternate_locations = {
        row.example_id: (batch.index, row_index)
        for batch in alternate_batches
        for row_index, row in enumerate(batch.rows)
        if row.role == "receiver"
    }

    receivers: list[RstaReceiverRow] = []
    for label in _LABELS:
        primary_batch = 0 if label in primary_order[:30] else 1
        alternate_batch = 0 if label in alternate_order[:30] else 1
        primary_foreign = tuple(sorted(primary_sets[primary_batch] - {label}))
        alternate_foreign = tuple(sorted(alternate_sets[alternate_batch] - {label}))
        if primary_foreign == alternate_foreign:
            raise ValueError("alternate foreign-class context is unchanged")
        if len(set(primary_foreign) & set(alternate_foreign)) > 22:
            raise ValueError("alternate foreign-class context is a near duplicate")
        for example_id in ranked[label][2:5]:
            primary_location = primary_locations[example_id]
            alternate_location = alternate_locations[example_id]
            receivers.append(
                RstaReceiverRow(
                    example_id=example_id,
                    label=label,
                    support_ids=(ranked[label][0], ranked[label][1]),
                    primary_batch=primary_location[0],
                    primary_row=primary_location[1],
                    alternate_batch=alternate_location[0],
                    alternate_row=alternate_location[1],
                    primary_peer_id=ranked[label][5],
                    alternate_peer_id=alternate_roles_tuple[label][0],
                    primary_foreign_labels=primary_foreign,
                    alternate_foreign_labels=alternate_foreign,
                )
            )

    return RstaRolePanel(
        primary_class_order=primary_order,
        alternate_class_order=alternate_order,
        ranked_ids_by_label=ranked,
        support_ids_by_label=tuple((ids[0], ids[1]) for ids in ranked),
        primary_batches=primary_batches,
        alternate_batches=alternate_batches,
        receivers=tuple(receivers),
    )


def _class_balanced_matrices(
    rows: Sequence[RstaReceiverEvidence],
    config: RstaStageAConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(rows) is not tuple:
        raise ValueError("receiver evidence must be an immutable tuple")
    if len(rows) != len(config.seeds) * 49 * 3:
        raise ValueError("RSTA requires exactly 147 receiver rows per seed")
    by_cell: dict[tuple[int, int], list[RstaReceiverEvidence]] = {}
    receiver_keys: set[tuple[int, str]] = set()
    for row in rows:
        if type(row) is not RstaReceiverEvidence:
            raise ValueError("receiver evidence has the wrong concrete type")
        if row.seed not in config.seeds or row.label not in _LABELS:
            raise ValueError("receiver seed or label differs from authority")
        if type(row.receiver_id) is not str or not row.receiver_id:
            raise ValueError("receiver_id must be a nonempty string")
        key = (row.seed, row.receiver_id)
        if key in receiver_keys:
            raise ValueError("duplicate receiver evidence")
        receiver_keys.add(key)
        values = (
            row.delta,
            row.self_minus_desc,
            row.rho,
            row.abs_log_ratio,
            row.deranged_delta,
            row.alternate_delta,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError("receiver metrics must be concrete finite floats")
        if not 0.0 <= row.rho <= 1.0 or row.abs_log_ratio < 0.0:
            raise ValueError("receiver rho or absolute log ratio is outside authority")
        by_cell.setdefault((row.seed, row.label), []).append(row)
    expected_cells = {
        (seed, label) for seed in config.seeds for label in _LABELS
    }
    if set(by_cell) != expected_cells or any(
        len(cell_rows) != 3 for cell_rows in by_cell.values()
    ):
        raise ValueError("RSTA requires exactly three receiver rows per seed/class")

    matrices: list[np.ndarray] = []
    for field in ("delta", "self_minus_desc", "alternate_delta"):
        matrices.append(
            np.asarray(
                [
                    [
                        math.fsum(getattr(row, field) for row in by_cell[(seed, label)])
                        / 3.0
                        for label in _LABELS
                    ]
                    for seed in config.seeds
                ],
                dtype=np.float64,
            )
        )
    return (matrices[0], matrices[1], matrices[2])


def _bootstrap_distribution(
    matrix: np.ndarray,
    config: RstaStageAConfig,
) -> np.ndarray:
    indices = np.random.Generator(np.random.PCG64(config.bootstrap_seed)).integers(
        0,
        49,
        size=(config.bootstrap_replicates, 49),
        dtype=np.int64,
    )
    distribution = matrix[:, indices].mean(axis=(0, 2), dtype=np.float64)
    return np.ascontiguousarray(distribution, dtype=np.float64)


def summarize_rsta_stage_a(
    rows: tuple[RstaReceiverEvidence, ...],
    config: RstaStageAConfig,
) -> RstaAggregate:
    """Recompute class-balanced evidence, bootstrap bounds, and verdict."""

    if type(config) is not RstaStageAConfig:
        raise ValueError("RSTA config has the wrong concrete type")
    delta_matrix, self_desc_matrix, alternate_matrix = _class_balanced_matrices(
        rows, config
    )
    seed_deltas = tuple(float(value) for value in delta_matrix.mean(axis=1))
    self_desc_seed = tuple(float(value) for value in self_desc_matrix.mean(axis=1))
    alternate_seed = tuple(float(value) for value in alternate_matrix.mean(axis=1))
    pooled_delta = math.fsum(seed_deltas) / 3.0
    pooled_self_desc = math.fsum(self_desc_seed) / 3.0
    alternate_pooled = math.fsum(alternate_seed) / 3.0
    delta_bootstrap = _bootstrap_distribution(delta_matrix, config)
    self_desc_bootstrap = _bootstrap_distribution(self_desc_matrix, config)
    delta_lower = float(np.quantile(delta_bootstrap, 0.025))
    self_desc_lower = float(np.quantile(self_desc_bootstrap, 0.025))
    median_rho = float(
        np.median(np.asarray([row.rho for row in rows], dtype=np.float64))
    )
    median_log_ratio = float(
        np.median(np.asarray([row.abs_log_ratio for row in rows], dtype=np.float64))
    )
    pooled_deranged = math.fsum(row.deranged_delta for row in rows) / len(rows)

    if pooled_delta <= 0.0:
        verdict, clause = "FAIL", "pooled-delta-nonpositive"
    elif all(value <= 0.0 for value in seed_deltas):
        verdict, clause = "FAIL", "all-seed-deltas-nonpositive"
    elif median_rho < config.fail_rho:
        verdict, clause = "FAIL", "median-rho-below-fail-floor"
    elif alternate_pooled <= 0.0:
        verdict, clause = "FAIL", "alternate-pooled-delta-nonpositive"
    elif all(value <= 0.0 for value in alternate_seed):
        verdict, clause = "FAIL", "all-alternate-seed-deltas-nonpositive"
    elif (
        pooled_delta >= config.pass_delta
        and delta_lower > 0.0
        and all(value >= config.pass_seed_delta for value in seed_deltas)
        and pooled_self_desc > 0.0
        and self_desc_lower > 0.0
        and median_rho >= config.pass_rho
        and median_log_ratio >= config.pass_abs_log_ratio
        and abs(pooled_deranged) <= config.max_abs_deranged_delta
        and alternate_pooled > 0.0
        and all(value > 0.0 for value in alternate_seed)
    ):
        verdict, clause = "PASS_ONWARD", "all-pass-gates"
    else:
        verdict, clause = "UNRESOLVED", "no-decisive-clause"

    return RstaAggregate(
        verdict=verdict,
        first_decisive_clause=clause,
        pooled_delta=pooled_delta,
        bootstrap_delta_95_lower=delta_lower,
        pooled_self_minus_desc=pooled_self_desc,
        bootstrap_self_minus_desc_95_lower=self_desc_lower,
        seed_deltas=seed_deltas,  # type: ignore[arg-type]
        alternate_pooled_delta=alternate_pooled,
        alternate_seed_deltas=alternate_seed,  # type: ignore[arg-type]
        median_rho=median_rho,
        median_abs_log_ratio=median_log_ratio,
        pooled_deranged_delta=pooled_deranged,
        bootstrap_delta_sha256=hashlib.sha256(
            delta_bootstrap.tobytes(order="C")
        ).hexdigest(),
        bootstrap_self_minus_desc_sha256=hashlib.sha256(
            self_desc_bootstrap.tobytes(order="C")
        ).hexdigest(),
        numpy_version=np.__version__,
        receiver_count=len(rows),
        config=config,
        receiver_evidence=rows,
    )


def rsta_stage_a_result_bytes(result: RstaAggregate) -> bytes:
    """Validate and serialize a canonical claim-ineligible Stage-A result."""

    if type(result) is not RstaAggregate:
        raise ValueError("RSTA result has the wrong concrete type")
    if summarize_rsta_stage_a(result.receiver_evidence, result.config) != result:
        raise ValueError("RSTA result differs from recomputed evidence")
    payload: dict[str, object] = {
        "schema": "siglip-rsta-stage-a-result-v1",
        "claim_eligible": False,
        "verdict": result.verdict,
        "first_decisive_clause": result.first_decisive_clause,
        "receiver_count": result.receiver_count,
        "config": {
            "seeds": result.config.seeds,
            "bootstrap_replicates": result.config.bootstrap_replicates,
            "bootstrap_seed": result.config.bootstrap_seed,
            "pass_delta": result.config.pass_delta,
            "pass_seed_delta": result.config.pass_seed_delta,
            "pass_rho": result.config.pass_rho,
            "fail_rho": result.config.fail_rho,
            "pass_abs_log_ratio": result.config.pass_abs_log_ratio,
            "max_abs_deranged_delta": result.config.max_abs_deranged_delta,
        },
        "metrics": {
            "pooled_delta": result.pooled_delta,
            "bootstrap_delta_95_lower": result.bootstrap_delta_95_lower,
            "pooled_self_minus_desc": result.pooled_self_minus_desc,
            "bootstrap_self_minus_desc_95_lower": (
                result.bootstrap_self_minus_desc_95_lower
            ),
            "seed_deltas": result.seed_deltas,
            "alternate_pooled_delta": result.alternate_pooled_delta,
            "alternate_seed_deltas": result.alternate_seed_deltas,
            "median_rho": result.median_rho,
            "median_abs_log_ratio": result.median_abs_log_ratio,
            "pooled_deranged_delta": result.pooled_deranged_delta,
        },
        "bootstrap": {
            "delta_distribution_sha256": result.bootstrap_delta_sha256,
            "self_minus_desc_distribution_sha256": (
                result.bootstrap_self_minus_desc_sha256
            ),
            "numpy_version": result.numpy_version,
        },
        "receiver_evidence": [
            {
                "seed": row.seed,
                "label": row.label,
                "receiver_id": row.receiver_id,
                "delta": row.delta,
                "self_minus_desc": row.self_minus_desc,
                "rho": row.rho,
                "abs_log_ratio": row.abs_log_ratio,
                "deranged_delta": row.deranged_delta,
                "alternate_delta": row.alternate_delta,
            }
            for row in result.receiver_evidence
        ],
    }
    return canonical_json_bytes(payload)


def contextual_rsta_direction(
    model: PooledProxyAnchorModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    microbatch_size: int,
    alpha: float,
    delta: float,
    score_tolerance: float,
) -> ContextualDirectionEvidence:
    """Replay Proxy Anchor and return exact output and parameter descent directions."""

    if type(alpha) is not float or alpha != 32.0 or type(delta) is not float or delta != 0.1:
        raise ValueError("RSTA requires the frozen Proxy Anchor operator")
    if type(model) is not PooledProxyAnchorModel:
        raise ValueError("RSTA direction requires the registered pooled model")
    selected = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith("tower.") or name.startswith("projection.")
            ),
            key=lambda item: item[0],
        )
    )
    if not selected or any(not parameter.requires_grad for _, parameter in selected):
        raise ValueError("RSTA selected parameter authority differs")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("RSTA direction requires cleared gradients")

    try:
        replay = recomputed_proxy_anchor_backward(
            model,
            inputs,
            labels,
            microbatch_size=microbatch_size,
            alpha=alpha,
            delta=delta,
            score_tolerance=score_tolerance,
        )
        normalized_proxies = F.normalize(model.proxies.detach().float(), dim=1)
        dbar = -(replay.score_gradients @ normalized_proxies)
        if not bool(torch.isfinite(dbar).all()):
            raise ValueError("RSTA contextual cotangent must be finite")
        direction: list[torch.Tensor] = []
        for _, parameter in selected:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("RSTA selected parameter gradient is missing or nonfinite")
            direction.append(-parameter.grad.detach().clone())
        return ContextualDirectionEvidence(
            dbar=dbar.detach().clone(),
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=tuple(direction),
            maximum_score_disagreement=replay.maximum_score_disagreement,
        )
    finally:
        for parameter in model.parameters():
            parameter.grad = None


def receiver_rsta_fields(
    model: PooledProxyAnchorModel,
    receiver_input: torch.Tensor,
    dbar: torch.Tensor,
    *,
    parameter_names: tuple[str, ...],
    parameter_direction: tuple[torch.Tensor, ...],
    backend: str,
) -> RstaReceiverFields:
    """Compute ``Jg`` and ``JJ^T dbar`` without materializing a Jacobian."""

    if type(model) is not PooledProxyAnchorModel:
        raise ValueError("RSTA fields require the registered pooled model")
    if backend not in ("forward-mode", "double-backward"):
        raise ValueError("RSTA fields require the preflight-selected backend")
    if (
        not isinstance(receiver_input, torch.Tensor)
        or not receiver_input.is_floating_point()
        or receiver_input.ndim < 2
        or receiver_input.shape[0] != 1
        or not bool(torch.isfinite(receiver_input).all())
    ):
        raise ValueError("RSTA receiver input must be one finite floating row")
    if (
        not isinstance(dbar, torch.Tensor)
        or not dbar.is_floating_point()
        or dbar.ndim != 1
        or not bool(torch.isfinite(dbar).all())
    ):
        raise ValueError("RSTA receiver cotangent must be one finite vector")

    selected = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith("tower.") or name.startswith("projection.")
            ),
            key=lambda item: item[0],
        )
    )
    expected_names = tuple(name for name, _ in selected)
    if not any(name.startswith("tower.") for name in expected_names) or not any(
        name.startswith("projection.") for name in expected_names
    ):
        raise ValueError("RSTA requires the complete trainable tower and projection")
    if (
        type(parameter_names) is not tuple
        or parameter_names != expected_names
        or type(parameter_direction) is not tuple
        or len(parameter_direction) != len(selected)
    ):
        raise ValueError("RSTA selected parameter tuple differs from authority")
    primals = tuple(
        parameter.detach().requires_grad_(backend == "double-backward")
        for _, parameter in selected
    )
    for tangent, primal in zip(parameter_direction, primals, strict=True):
        if (
            type(tangent) is not torch.Tensor
            or tangent.shape != primal.shape
            or tangent.dtype != primal.dtype
            or tangent.device != primal.device
            or not bool(torch.isfinite(tangent).all())
        ):
            raise ValueError("RSTA parameter direction differs from authority")

    tower_buffers = dict(model.tower.named_buffers())
    projection_buffers = dict(model.projection.named_buffers())

    def encode(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        value_by_name = dict(zip(expected_names, values, strict=True))
        tower_state = {
            name: value_by_name[f"tower.{name}"]
            for name, _ in model.tower.named_parameters()
        }
        tower_state.update(tower_buffers)
        projection_state = {
            name: value_by_name[f"projection.{name}"]
            for name, _ in model.projection.named_parameters()
        }
        projection_state.update(projection_buffers)
        pooled = torch.func.functional_call(model.tower, tower_state, (receiver_input,))
        projected = torch.func.functional_call(
            model.projection, projection_state, (pooled,)
        ).float()
        return F.normalize(projected, dim=1).squeeze(0)

    if backend == "forward-mode":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script` is deprecated\..*",
                category=DeprecationWarning,
                module=r"torch\.jit\._script",
            )
            descriptor, batch_motion = torch.func.jvp(
                encode,
                (primals,),
                (parameter_direction,),
            )
        vjp_descriptor, pullback = torch.func.vjp(encode, primals)
        if not torch.equal(descriptor, vjp_descriptor):
            raise ValueError("RSTA receiver descriptor replay differs")
        (self_parameter_direction,) = pullback(dbar.to(descriptor))
        _, self_motion = torch.func.jvp(
            encode,
            (primals,),
            (self_parameter_direction,),
        )
    else:
        descriptor = encode(primals)

        def double_backward_jvp(
            tangent_values: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            output = encode(primals)
            output_cotangent = torch.zeros_like(output, requires_grad=True)
            transposed = torch.autograd.grad(
                output,
                primals,
                grad_outputs=output_cotangent,
                create_graph=True,
            )
            pairing = sum(
                (left * right).sum()
                for left, right in zip(transposed, tangent_values, strict=True)
            )
            return torch.autograd.grad(pairing, output_cotangent)[0]

        batch_motion = double_backward_jvp(parameter_direction)
        self_parameter_direction = torch.autograd.grad(
            descriptor,
            primals,
            grad_outputs=dbar.to(descriptor),
        )
        self_motion = double_backward_jvp(self_parameter_direction)

    def tangent(field: torch.Tensor) -> tuple[torch.Tensor, float]:
        field_norm = torch.linalg.vector_norm(field)
        if not bool(torch.isfinite(field_norm)) or float(field_norm) <= 1.0e-12:
            raise ValueError("RSTA receiver field has zero norm")
        radial = descriptor * torch.dot(descriptor, field)
        radial_fraction = float(
            (torch.linalg.vector_norm(radial) / field_norm).detach()
        )
        if radial_fraction > 1.0e-3:
            raise ValueError("RSTA receiver field radial fraction exceeds authority")
        return field - radial, radial_fraction

    batch_motion, batch_radial_fraction = tangent(batch_motion)
    self_motion, self_radial_fraction = tangent(self_motion)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (descriptor, batch_motion, self_motion)
    ):
        raise ValueError("RSTA receiver fields must be finite")
    return RstaReceiverFields(
        descriptor=descriptor.detach(),
        batch_motion=batch_motion.detach(),
        self_motion=self_motion.detach(),
        batch_radial_fraction=batch_radial_fraction,
        self_radial_fraction=self_radial_fraction,
        backend=backend,
    )


def preflight_rsta_jvp_backend(
    model: PooledProxyAnchorModel,
    receiver_input: torch.Tensor,
    dbar: torch.Tensor,
    *,
    parameter_names: tuple[str, ...],
    parameter_direction: tuple[torch.Tensor, ...],
) -> RstaJvpBackendEvidence:
    """Select the registered JVP backend before scientific rows are opened."""

    try:
        forward = receiver_rsta_fields(
            model,
            receiver_input,
            dbar,
            parameter_names=parameter_names,
            parameter_direction=parameter_direction,
            backend="forward-mode",
        )
    except NotImplementedError as error:
        receiver_rsta_fields(
            model,
            receiver_input,
            dbar,
            parameter_names=parameter_names,
            parameter_direction=parameter_direction,
            backend="double-backward",
        )
        return RstaJvpBackendEvidence(
            backend="double-backward",
            comparison_available=False,
            maximum_relative_disagreement=0.0,
            forward_error=type(error).__name__,
        )

    fallback = receiver_rsta_fields(
        model,
        receiver_input,
        dbar,
        parameter_names=parameter_names,
        parameter_direction=parameter_direction,
        backend="double-backward",
    )

    def relative(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = max(
            float(torch.linalg.vector_norm(left)),
            float(torch.linalg.vector_norm(right)),
            1.0e-12,
        )
        return float(torch.linalg.vector_norm(left - right)) / denominator

    maximum = max(
        relative(forward.descriptor, fallback.descriptor),
        relative(forward.batch_motion, fallback.batch_motion),
        relative(forward.self_motion, fallback.self_motion),
    )
    if not math.isfinite(maximum) or maximum > 1.0e-5:
        raise ValueError("RSTA JVP backends disagree above the registered tolerance")
    return RstaJvpBackendEvidence(
        backend="forward-mode",
        comparison_available=True,
        maximum_relative_disagreement=maximum,
        forward_error=None,
    )


def proxy_free_margin_direction(
    descriptor: torch.Tensor,
    positive_descriptors: torch.Tensor,
    foreign_descriptors: torch.Tensor,
    *,
    foreign_example_ids: tuple[str, ...],
    foreign_role_digests: tuple[str, ...],
) -> RstaOutcomeDirection:
    """Construct the frozen proxy-free margin-ascent tangent direction."""

    if (
        not isinstance(descriptor, torch.Tensor)
        or not descriptor.is_floating_point()
        or descriptor.ndim != 1
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("RSTA outcome descriptor must be one finite vector")
    dimensions = int(descriptor.shape[0])
    if (
        not isinstance(positive_descriptors, torch.Tensor)
        or positive_descriptors.shape != (2, dimensions)
        or positive_descriptors.dtype != descriptor.dtype
        or positive_descriptors.device != descriptor.device
        or not bool(torch.isfinite(positive_descriptors).all())
    ):
        raise ValueError("RSTA outcome requires exactly two finite positive descriptors")
    if (
        not isinstance(foreign_descriptors, torch.Tensor)
        or foreign_descriptors.ndim != 2
        or foreign_descriptors.shape[0] < 32
        or foreign_descriptors.shape[1] != dimensions
        or foreign_descriptors.dtype != descriptor.dtype
        or foreign_descriptors.device != descriptor.device
        or not bool(torch.isfinite(foreign_descriptors).all())
    ):
        raise ValueError("RSTA outcome foreign descriptor authority differs")
    foreign_count = int(foreign_descriptors.shape[0])
    if (
        type(foreign_example_ids) is not tuple
        or len(foreign_example_ids) != foreign_count
        or len(set(foreign_example_ids)) != foreign_count
        or any(type(value) is not str or not value for value in foreign_example_ids)
        or type(foreign_role_digests) is not tuple
        or len(foreign_role_digests) != foreign_count
        or any(not _is_lower_hex(value, 64) for value in foreign_role_digests)
    ):
        raise ValueError("RSTA outcome foreign identity authority differs")
    all_rows = torch.cat(
        (descriptor.unsqueeze(0), positive_descriptors, foreign_descriptors), dim=0
    )
    norm_errors = (torch.linalg.vector_norm(all_rows, dim=1) - 1.0).abs()
    if bool((norm_errors > 2.0e-5).any()):
        raise ValueError("RSTA outcome descriptors must be unit rows")

    foreign_scores = foreign_descriptors @ descriptor
    selected = tuple(
        sorted(
            range(foreign_count),
            key=lambda index: (
                -float(foreign_scores[index]),
                foreign_role_digests[index],
                foreign_example_ids[index],
            ),
        )[:32]
    )
    selected_foreign = foreign_descriptors[list(selected)]
    positive_scores = positive_descriptors @ descriptor
    selected_scores = selected_foreign @ descriptor
    tau = 0.05
    positive_logmeanexp = torch.logsumexp(positive_scores / tau, dim=0) - math.log(2.0)
    negative_logmeanexp = torch.logsumexp(selected_scores / tau, dim=0) - math.log(32.0)
    margin = tau * (positive_logmeanexp - negative_logmeanexp)
    positive_weights = torch.softmax(positive_scores / tau, dim=0)
    negative_weights = torch.softmax(selected_scores / tau, dim=0)
    gradient = positive_weights @ positive_descriptors - negative_weights @ selected_foreign
    tangent = gradient - descriptor * torch.dot(descriptor, gradient)
    tangent_norm = torch.linalg.vector_norm(tangent)
    if not bool(torch.isfinite(tangent_norm)) or float(tangent_norm) <= 1.0e-12:
        raise ValueError("RSTA outcome direction has zero projected norm")
    direction = tangent / tangent_norm
    return RstaOutcomeDirection(
        margin=float(margin),
        direction=direction.detach(),
        selected_foreign_indices=selected,
        selected_foreign_ids=tuple(foreign_example_ids[index] for index in selected),
    )


def score_rsta_receiver(
    *,
    fields: RstaReceiverFields,
    dbar: torch.Tensor,
    outcome_direction: torch.Tensor,
    random_target: torch.Tensor,
    deranged_direction: torch.Tensor,
) -> RstaReceiverScore:
    """Project once and compute the complete frozen receiver statistic family."""

    if not isinstance(fields, RstaReceiverFields):
        raise ValueError("RSTA receiver fields differ from authority")
    descriptor = fields.descriptor
    batch_motion = fields.batch_motion
    self_motion = fields.self_motion
    values = (
        descriptor,
        batch_motion,
        self_motion,
        dbar,
        outcome_direction,
        random_target,
        deranged_direction,
    )
    if (
        any(not isinstance(value, torch.Tensor) for value in values)
        or descriptor.ndim != 1
        or any(value.shape != descriptor.shape for value in values)
        or any(value.dtype != descriptor.dtype for value in values)
        or any(value.device != descriptor.device for value in values)
        or any(not bool(torch.isfinite(value).all()) for value in values)
    ):
        raise ValueError("RSTA receiver score tensors differ from authority")
    descriptor_norm = torch.linalg.vector_norm(descriptor)
    if abs(float(descriptor_norm) - 1.0) > 2.0e-5:
        raise ValueError("RSTA receiver descriptor must be unit normalized")
    if (
        fields.backend not in ("forward-mode", "double-backward")
        or type(fields.batch_radial_fraction) is not float
        or not math.isfinite(fields.batch_radial_fraction)
        or not 0.0 <= fields.batch_radial_fraction <= 1.0e-3
        or type(fields.self_radial_fraction) is not float
        or not math.isfinite(fields.self_radial_fraction)
        or not 0.0 <= fields.self_radial_fraction <= 1.0e-3
    ):
        raise ValueError("RSTA receiver field evidence differs from authority")

    def project_cotangent(value: torch.Tensor) -> tuple[torch.Tensor, float]:
        original_norm = torch.linalg.vector_norm(value)
        if float(original_norm) <= 1.0e-12:
            raise ValueError("RSTA receiver score has zero input norm")
        radial = descriptor * torch.dot(descriptor, value)
        projected = value - radial
        projected_norm = torch.linalg.vector_norm(projected)
        if float(projected_norm) <= 1.0e-12:
            raise ValueError("RSTA receiver score has zero projected norm")
        radial_fraction = float(torch.linalg.vector_norm(radial) / original_norm)
        return projected, radial_fraction

    def validate_tangent(value: torch.Tensor, *, unit: bool) -> torch.Tensor:
        value_norm = torch.linalg.vector_norm(value)
        if float(value_norm) <= 1.0e-12:
            raise ValueError("RSTA receiver score has zero norm")
        radial_fraction = float(torch.abs(torch.dot(descriptor, value)) / value_norm)
        if radial_fraction > 1.0e-3:
            raise ValueError("RSTA receiver score tangent residual exceeds authority")
        if unit and abs(float(value_norm) - 1.0) > 2.0e-5:
            raise ValueError("RSTA receiver score control must be unit normalized")
        return value

    batch = validate_tangent(batch_motion, unit=False)
    self_value = validate_tangent(self_motion, unit=False)
    descriptor_cotangent, dbar_radial = project_cotangent(dbar)
    outcome = validate_tangent(outcome_direction, unit=True)
    random_value = validate_tangent(random_target, unit=True)
    deranged_value = validate_tangent(deranged_direction, unit=True)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 1.0e-12:
            raise ValueError("RSTA receiver score has zero norm")
        return float(torch.dot(left, right) / denominator)

    a_self = cosine(self_value, outcome)
    a_batch = cosine(batch, outcome)
    a_desc = cosine(descriptor_cotangent, outcome)
    cos_batch_self = max(-1.0, min(1.0, cosine(batch, self_value)))
    batch_norm = float(torch.linalg.vector_norm(batch))
    self_norm = float(torch.linalg.vector_norm(self_value))
    return RstaReceiverScore(
        a_self=a_self,
        a_batch=a_batch,
        delta=a_self - a_batch,
        a_desc=a_desc,
        self_minus_desc=a_self - a_desc,
        cos_batch_self=cos_batch_self,
        rho=math.sqrt(max(0.0, 1.0 - cos_batch_self * cos_batch_self)),
        log_ratio=math.log((batch_norm + 1.0e-12) / (self_norm + 1.0e-12)),
        cross_contribution=cosine(batch - self_value, outcome),
        random_delta=cosine(self_value, random_value) - cosine(batch, random_value),
        deranged_delta=cosine(self_value, deranged_value)
        - cosine(batch, deranged_value),
        batch_radial_fraction=fields.batch_radial_fraction,
        self_radial_fraction=fields.self_radial_fraction,
        dbar_radial_fraction=dbar_radial,
    )


def rsta_control_directions(
    descriptors: torch.Tensor,
    outcome_directions: torch.Tensor,
    *,
    receiver_ids: tuple[str, ...],
) -> RstaControlDirections:
    """Construct ID-seeded random and cyclic-derangement tangent controls."""

    if (
        not isinstance(descriptors, torch.Tensor)
        or not descriptors.is_floating_point()
        or descriptors.ndim != 2
        or descriptors.shape[0] < 2
        or not isinstance(outcome_directions, torch.Tensor)
        or outcome_directions.shape != descriptors.shape
        or outcome_directions.dtype != descriptors.dtype
        or outcome_directions.device != descriptors.device
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(outcome_directions).all())
    ):
        raise ValueError("RSTA control direction tensors differ from authority")
    receiver_count, dimensions = descriptors.shape
    if (
        type(receiver_ids) is not tuple
        or len(receiver_ids) != receiver_count
        or len(set(receiver_ids)) != receiver_count
        or any(type(value) is not str or not value for value in receiver_ids)
    ):
        raise ValueError("RSTA control receiver identity authority differs")
    if bool(
        ((torch.linalg.vector_norm(descriptors, dim=1) - 1.0).abs() > 2.0e-5).any()
    ):
        raise ValueError("RSTA control descriptors must be unit rows")
    outcome_norms = torch.linalg.vector_norm(outcome_directions, dim=1)
    outcome_radial = (descriptors * outcome_directions).sum(dim=1).abs()
    if bool((outcome_norms <= 1.0e-12).any()) or bool(
        ((outcome_norms - 1.0).abs() > 2.0e-5).any()
    ) or bool((outcome_radial / outcome_norms > 1.0e-3).any()):
        raise ValueError("RSTA control outcome directions differ from authority")

    def project_and_normalize(
        vectors: torch.Tensor, *, error: str
    ) -> torch.Tensor:
        projected = vectors - descriptors * (descriptors * vectors).sum(
            dim=1, keepdim=True
        )
        norms = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 1.0e-12).any()):
            raise ValueError(error)
        return projected / norms

    random_rows = []
    for receiver_id in receiver_ids:
        seed = int.from_bytes(
            _hash("rsta-siglip-a-v1|random-target|", receiver_id)[:8], "big"
        )
        values = np.random.Generator(np.random.PCG64(seed)).standard_normal(
            dimensions
        )
        random_rows.append(
            torch.as_tensor(values, dtype=descriptors.dtype, device=descriptors.device)
        )
    random_targets = project_and_normalize(
        torch.stack(random_rows), error="RSTA random control has zero projected norm"
    )
    deranged = torch.roll(outcome_directions, shifts=-1, dims=0)
    deranged_directions = project_and_normalize(
        deranged, error="RSTA deranged control has zero projected norm"
    )
    return RstaControlDirections(
        random_targets=random_targets,
        deranged_directions=deranged_directions,
    )
