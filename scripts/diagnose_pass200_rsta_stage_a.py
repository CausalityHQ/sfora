#!/usr/bin/env python3
"""Pure deterministic core for the preregistered Pass200 RSTA diagnostic.

Artifact binding and model execution are deliberately added in later tasks.  The
helpers in this first layer depend only on deterministic NumPy geometry and
statistics, so importing this module cannot touch datasets, checkpoints, or CUDA.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

_ROLE_DOMAIN = "rsta-stage-a-v1|role|"
_IDENTITY_DOMAIN = "rsta-stage-a-v1|identity|"
_DISTRACTOR_DOMAIN = "rsta-stage-a-v1|distractor|"
_ALTERNATE_DISTRACTOR_DOMAIN = "rsta-stage-a-v1|alternate-distractor|"
_BATCH_ORDER_PREFIX = "rsta-stage-a-v1|batch-order|"
_PRIMARY_IDENTITIES = 64
_RECEIVERS_PER_BATCH = 8
_DISTRACTORS_PER_BATCH = 172
_VECTOR_EPS = 1.0e-12
_ROTATION_VECTOR_NAMES = frozenset(("z", "dbar", "b", "s", "q"))
_ROTATION_STATISTIC_NAMES = frozenset(
    ("A_self", "A_batch", "Delta", "A_desc", "rho", "log_ratio", "cos_b_s")
)


def domain_hash(domain: str, text: str) -> bytes:
    """Return ``SHA256(domain.encode('ascii') + NUL + text.encode('utf-8'))``."""
    if not isinstance(domain, str) or not isinstance(text, str):
        raise TypeError("domain and text must be strings")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + text.encode("utf-8")).digest()


def domain_seed(domain: str, text: str) -> int:
    """Extract the registered unsigned big-endian seed from a domain hash."""
    return int.from_bytes(domain_hash(domain, text)[:8], byteorder="big", signed=False)


def _canonical_inputs(
    example_ids: Sequence[str], labels: Sequence[int]
) -> tuple[list[str], list[int]]:
    ids = list(example_ids)
    raw_labels = list(labels)
    if not ids or len(ids) != len(raw_labels):
        raise ValueError("example IDs and labels must be nonempty and aligned")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every example ID must be a nonempty string")
    if len(set(ids)) != len(ids):
        raise ValueError("example IDs must be unique")
    canonical_labels: list[int] = []
    for value in raw_labels:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("identity labels must be unsigned integers")
        label = int(value)
        if label < 0:
            raise ValueError("identity labels must be unsigned integers")
        canonical_labels.append(label)
    return ids, canonical_labels


def _identity_roles(
    example_ids: list[str], labels: list[int]
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    rows_by_label: dict[int, list[str]] = {}
    for example_id, label in zip(example_ids, labels, strict=True):
        rows_by_label.setdefault(label, []).append(example_id)
    eligible_roles: dict[int, list[str]] = {}
    for label, row_ids in rows_by_label.items():
        if len(row_ids) >= 3:
            eligible_roles[label] = sorted(
                row_ids, key=lambda value: (domain_hash(_ROLE_DOMAIN, value), value)
            )
    return rows_by_label, eligible_roles


def _ordered_batch(receiver_ids: list[str], distractor_ids: list[str], batch: int) -> list[str]:
    if len(receiver_ids) != _RECEIVERS_PER_BATCH:
        raise ValueError("each diagnostic batch requires exactly eight receivers")
    if len(distractor_ids) != _DISTRACTORS_PER_BATCH:
        raise ValueError("each diagnostic batch requires exactly 172 distractors")
    domain = f"{_BATCH_ORDER_PREFIX}{batch}|"
    combined = receiver_ids + distractor_ids
    if len(set(combined)) != _RECEIVERS_PER_BATCH + _DISTRACTORS_PER_BATCH:
        raise ValueError("diagnostic batch rows must be unique")
    return sorted(combined, key=lambda value: (domain_hash(domain, value), value))


def select_primary_panel(example_ids: Sequence[str], labels: Sequence[int]) -> dict[str, Any]:
    """Select frozen roles, 64 receivers, and eight official-size batches."""
    ids, canonical_labels = _canonical_inputs(example_ids, labels)
    rows_by_label, roles = _identity_roles(ids, canonical_labels)
    ordered_eligible = sorted(
        roles,
        key=lambda label: (domain_hash(_IDENTITY_DOMAIN, str(label)), label),
    )
    if len(ordered_eligible) < _PRIMARY_IDENTITIES:
        raise ValueError("at least 64 identities with three rows are required")
    selected_labels = ordered_eligible[:_PRIMARY_IDENTITIES]
    selected_set = set(selected_labels)
    receiver_ids = [roles[label][2] for label in selected_labels]
    groups = [
        selected_labels[start : start + _RECEIVERS_PER_BATCH]
        for start in range(0, _PRIMARY_IDENTITIES, _RECEIVERS_PER_BATCH)
    ]
    distractor_ids = [
        example_id
        for example_id, label in zip(ids, canonical_labels, strict=True)
        if label not in selected_set
    ]
    distractor_ids.sort(key=lambda value: (domain_hash(_DISTRACTOR_DOMAIN, value), value))
    needed = len(groups) * _DISTRACTORS_PER_BATCH
    if len(distractor_ids) < needed:
        raise ValueError("not enough nonselected rows for primary distractors")
    distractor_blocks = [
        distractor_ids[start : start + _DISTRACTORS_PER_BATCH]
        for start in range(0, needed, _DISTRACTORS_PER_BATCH)
    ]
    batches = [
        _ordered_batch(receiver_ids[batch * 8 : (batch + 1) * 8], block, batch)
        for batch, block in enumerate(distractor_blocks)
    ]
    support_ids_by_label = {label: roles[label][:2] for label in ordered_eligible}
    return {
        "eligible_labels": ordered_eligible,
        "labels": selected_labels,
        "receiver_ids": receiver_ids,
        "groups": groups,
        "support_ids_by_label": support_ids_by_label,
        "distractor_blocks": distractor_blocks,
        "batches": batches,
        "rows_by_label": rows_by_label,
    }


def select_alternate_panel(
    example_ids: Sequence[str],
    labels: Sequence[int],
    primary_panel: dict[str, Any],
) -> dict[str, Any]:
    """Build the frozen two-batch alternate contexts from a primary panel."""
    ids, canonical_labels = _canonical_inputs(example_ids, labels)
    canonical_primary = select_primary_panel(ids, canonical_labels)
    for name in ("labels", "receiver_ids", "distractor_blocks", "batches"):
        if primary_panel.get(name) != canonical_primary[name]:
            raise ValueError("alternate selection requires the canonical primary panel")
    _, roles = _identity_roles(ids, canonical_labels)
    primary_labels = list(primary_panel.get("labels", []))
    primary_receivers = list(primary_panel.get("receiver_ids", []))
    primary_blocks = list(primary_panel.get("distractor_blocks", []))
    if len(primary_labels) != 64 or len(primary_receivers) != 64 or len(primary_blocks) != 8:
        raise ValueError("alternate selection requires a complete primary panel")
    if len(set(primary_labels)) != 64 or len(set(primary_receivers)) != 64:
        raise ValueError("primary receivers and labels must be unique")
    for label, receiver_id in zip(primary_labels, primary_receivers, strict=True):
        if label not in roles or roles[label][2] != receiver_id:
            raise ValueError("primary receiver roles do not match the supplied rows")
    selected_positions = [index for index in range(64) if index % 8 in (0, 1)]
    alternate_labels = [primary_labels[index] for index in selected_positions]
    alternate_receivers = [primary_receivers[index] for index in selected_positions]
    alternate_set = set(alternate_labels)
    excluded = {value for block in primary_blocks for value in block}
    if len(excluded) != 8 * _DISTRACTORS_PER_BATCH:
        raise ValueError("primary distractors must be complete and nonoverlapping")
    excluded.update(value for role_ids in roles.values() for value in role_ids[:2])
    excluded.update(
        example_id
        for example_id, label in zip(ids, canonical_labels, strict=True)
        if label in alternate_set
    )
    candidates = [value for value in ids if value not in excluded]
    candidates.sort(key=lambda value: (domain_hash(_ALTERNATE_DISTRACTOR_DOMAIN, value), value))
    needed = 2 * _DISTRACTORS_PER_BATCH
    if len(candidates) < needed:
        raise ValueError("not enough rows for alternate distractors")
    blocks = [
        candidates[start : start + _DISTRACTORS_PER_BATCH]
        for start in range(0, needed, _DISTRACTORS_PER_BATCH)
    ]
    groups = [alternate_labels[:8], alternate_labels[8:]]
    batches = [
        _ordered_batch(alternate_receivers[batch * 8 : (batch + 1) * 8], block, batch)
        for batch, block in enumerate(blocks)
    ]
    return {
        "labels": alternate_labels,
        "receiver_ids": alternate_receivers,
        "groups": groups,
        "distractor_blocks": blocks,
        "batches": batches,
    }


def _unit_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite unit vector")
    norm = float(np.linalg.norm(value))
    if norm <= _VECTOR_EPS or abs(norm - 1.0) > 2.0e-5:
        raise ValueError(f"{name} must be a finite unit vector")
    return value / norm


def _unit_matrix(rows: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= _VECTOR_EPS) or np.any(np.abs(norms - 1.0) > 2.0e-5):
        raise ValueError(f"{name} must contain finite unit rows")
    return value / norms[:, None]


def tangent_projection(vector: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Project a vector into the receiver descriptor tangent, rejecting zero."""
    z = _unit_vector(receiver, name="receiver")
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != z.shape or not np.isfinite(value).all():
        raise ValueError("vector must be finite and aligned with receiver")
    projected = value - z * float(np.dot(z, value))
    if float(np.linalg.norm(projected)) <= _VECTOR_EPS:
        raise ValueError("tangent projection has zero norm")
    return projected


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def smooth_margin_gradient(
    receiver: np.ndarray,
    positive_supports: np.ndarray,
    frozen_foreign_supports: np.ndarray,
    *,
    tau: float = 0.05,
) -> np.ndarray:
    """Return the tangent ascent of the registered frozen smooth margin."""
    z = _unit_vector(receiver, name="receiver")
    positives = _unit_matrix(positive_supports, name="positive supports")
    foreign = _unit_matrix(frozen_foreign_supports, name="foreign supports")
    if positives.shape != (2, z.size):
        raise ValueError("smooth margin requires exactly two positive supports")
    if foreign.shape != (32, z.size):
        raise ValueError("smooth margin requires exactly 32 foreign supports")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    ambient = _softmax((positives @ z) / tau) @ positives
    ambient -= _softmax((foreign @ z) / tau) @ foreign
    return tangent_projection(ambient, z)


def select_foreign_supports(
    receiver: np.ndarray,
    *,
    receiver_label: int,
    support_ids: Sequence[str],
    support_labels: Sequence[int],
    support_descriptors: np.ndarray,
    current_batch_ids: set[str] | frozenset[str],
) -> tuple[list[str], np.ndarray]:
    """Freeze the 32 largest foreign receiver-view cosines with registered ties."""
    ids, labels = _canonical_inputs(support_ids, support_labels)
    if len(set(labels)) != len(labels):
        raise ValueError("foreign pool requires exactly one rank-0 support per identity")
    if (
        isinstance(receiver_label, bool)
        or not isinstance(receiver_label, (int, np.integer))
        or int(receiver_label) < 0
    ):
        raise ValueError("receiver label must be an unsigned integer")
    if not isinstance(current_batch_ids, (set, frozenset)) or any(
        not isinstance(value, str) for value in current_batch_ids
    ):
        raise ValueError("current batch IDs must be a set of strings")
    z = _unit_vector(receiver, name="receiver")
    supports = _unit_matrix(support_descriptors, name="foreign support candidates")
    if supports.shape != (len(ids), z.size):
        raise ValueError("foreign support IDs, labels, and descriptors must align")
    eligible = [
        index
        for index, (example_id, label) in enumerate(zip(ids, labels, strict=True))
        if label != int(receiver_label) and example_id not in current_batch_ids
    ]
    if len(eligible) < 32:
        raise ValueError("foreign pool must contain at least 32 eligible supports")
    similarities = supports @ z
    ordered = sorted(
        eligible,
        key=lambda index: (
            -float(similarities[index]),
            domain_hash(_ROLE_DOMAIN, ids[index]),
            ids[index],
        ),
    )[:32]
    return [ids[index] for index in ordered], supports[ordered]


def deranged_tangent_targets(receivers: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Shift targets by +1 within one eight-receiver group and reproject."""
    z = _unit_matrix(receivers, name="receivers")
    values = np.asarray(targets, dtype=np.float64)
    if z.shape[0] != 8 or values.shape != z.shape or not np.isfinite(values).all():
        raise ValueError("derangement requires aligned arrays for eight receivers")
    result = np.empty_like(values)
    for index in range(8):
        projected = tangent_projection(values[(index + 1) % 8], z[index])
        result[index] = projected / float(np.linalg.norm(projected))
    return result


def random_tangent_target(
    receiver: np.ndarray,
    *,
    seed: int,
    example_id: str,
    target_norm: float,
) -> np.ndarray:
    """Draw the registered fresh PCG64 tangent-random negative control."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be an unsigned integer")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("example_id must be a nonempty string")
    if not np.isfinite(target_norm) or target_norm <= _VECTOR_EPS:
        raise ValueError("target_norm must be finite and positive")
    z = _unit_vector(receiver, name="receiver")
    random_seed = domain_seed("rsta-stage-a-v1|random-target|", f"{int(seed)}\0{example_id}")
    random = np.random.Generator(np.random.PCG64(random_seed)).standard_normal(z.size)
    projected = tangent_projection(random, z)
    return projected * (float(target_norm) / float(np.linalg.norm(projected)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return a signed cosine, rejecting nonfinite or zero inputs."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("cosine inputs must be aligned vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("cosine inputs must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _VECTOR_EPS or second_norm <= _VECTOR_EPS:
        raise ValueError("cosine inputs must have nonzero norm")
    denominator = first_norm * second_norm
    return float(np.dot(first, second) / denominator)


def head_only_kernel_motion(
    prehead_features: np.ndarray,
    head_outputs: np.ndarray,
    cotangents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the analytic normalized affine-head kernel to all cotangents."""
    features = np.asarray(prehead_features)
    outputs = np.asarray(head_outputs)
    directions = np.asarray(cotangents)
    if (
        not all(
            np.issubdtype(value.dtype, np.floating) for value in (features, outputs, directions)
        )
        or len({features.dtype, outputs.dtype, directions.dtype}) != 1
    ):
        raise ValueError("head arrays must share one floating model dtype")
    if features.ndim != 2 or outputs.ndim != 2 or directions.shape != outputs.shape:
        raise ValueError("head arrays must be aligned nonempty matrices")
    if features.shape[0] == 0 or features.shape[0] != outputs.shape[0]:
        raise ValueError("head arrays must have the same nonzero batch dimension")
    if not all(np.isfinite(value).all() for value in (features, outputs, directions)):
        raise ValueError("head arrays must be finite")
    raw_norms = np.linalg.norm(outputs, axis=1)
    if np.any(raw_norms <= _VECTOR_EPS):
        raise ValueError("head outputs must have nonzero norms")
    descriptors = outputs / raw_norms[:, None]
    projected_directions = directions - descriptors * np.sum(
        descriptors * directions, axis=1, keepdims=True
    )
    first_actions = projected_directions / raw_norms[:, None]
    coefficients = features @ features.T + 1.0
    batch_motion = np.empty_like(outputs)
    self_motion = np.empty_like(outputs)
    for receiver in range(features.shape[0]):
        after_receiver_projection = (
            first_actions - descriptors[receiver] * (first_actions @ descriptors[receiver])[:, None]
        )
        contributions = after_receiver_projection / raw_norms[receiver]
        batch_motion[receiver] = coefficients[receiver] @ contributions
        self_motion[receiver] = coefficients[receiver, receiver] * contributions[receiver]
        if cosine_similarity(self_motion[receiver], directions[receiver]) < 1.0 - 1.0e-5:
            raise ValueError("head-self motion is not positively collinear with cotangent")
    return batch_motion, self_motion


def construct_rotation(
    dimension: int = 512,
    *,
    seed: int = 200,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    """Construct the registered dense orthogonal descriptor rotation."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("rotation dimension must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("rotation seed must be an unsigned integer")
    matrix = np.ascontiguousarray(
        np.random.Generator(np.random.PCG64(seed)).standard_normal((dimension, dimension))
    )
    rotation, triangular = np.linalg.qr(matrix)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    rotation = rotation * signs
    return rotation.astype(dtype, copy=False)


def check_rotation(
    vectors: dict[str, np.ndarray],
    rotated_vectors: dict[str, np.ndarray],
    statistics: dict[str, float],
    rotated_statistics: dict[str, float],
    rotation: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Check the registered vector and scalar rotation-equivariance gates."""
    raw_rotation = np.asarray(rotation)
    q = np.asarray(raw_rotation, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.isfinite(q).all():
        raise ValueError("rotation must be a finite square matrix")
    orthogonality_tolerance = 5.0e-5 if raw_rotation.dtype.itemsize <= 4 else 1.0e-10
    if not np.allclose(
        q.T @ q,
        np.eye(q.shape[0]),
        atol=orthogonality_tolerance,
        rtol=orthogonality_tolerance,
    ):
        raise ValueError("rotation must be orthogonal")
    if (
        set(vectors) != _ROTATION_VECTOR_NAMES
        or set(rotated_vectors) != _ROTATION_VECTOR_NAMES
        or set(statistics) != _ROTATION_STATISTIC_NAMES
        or set(rotated_statistics) != _ROTATION_STATISTIC_NAMES
    ):
        raise ValueError("rotation check requires all registered names")
    vector_residuals: dict[str, float] = {}
    for name, vector in vectors.items():
        original = np.asarray(vector, dtype=np.float64)
        observed = np.asarray(rotated_vectors[name], dtype=np.float64)
        if original.shape != (q.shape[0],) or observed.shape != original.shape:
            raise ValueError("rotation vectors have incompatible shapes")
        if (
            not np.isfinite(original).all()
            or not np.isfinite(observed).all()
            or float(np.linalg.norm(original)) <= _VECTOR_EPS
            or float(np.linalg.norm(observed)) <= _VECTOR_EPS
        ):
            raise ValueError(f"rotation vectors must be finite and nonzero for {name}")
        residual = float(np.linalg.norm(observed - q @ original)) / max(
            float(np.linalg.norm(original)), _VECTOR_EPS
        )
        if not np.isfinite(residual) or residual > 5.0e-4:
            raise ValueError(f"rotation vector gate failed for {name}")
        vector_residuals[name] = residual
    statistic_differences: dict[str, float] = {}
    for name, value in statistics.items():
        difference = abs(float(rotated_statistics[name]) - float(value))
        if not np.isfinite(difference) or difference > 2.0e-4:
            raise ValueError(f"rotation statistic gate failed for {name}")
        statistic_differences[name] = difference
    return {
        "vector_residuals": vector_residuals,
        "statistic_differences": statistic_differences,
    }


def joint_bootstrap(
    values_by_seed: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 200,
) -> np.ndarray:
    """Jointly resample identity columns, retaining all four seed pairings."""
    values = np.asarray(values_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] == 0:
        raise ValueError("bootstrap values must have shape (4, identity_count)")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be an unsigned integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    identity_count = values.shape[1]
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = generator.integers(0, identity_count, size=identity_count)
        distribution[replicate] = values[:, sampled].mean(axis=1).mean()
    return distribution


def float64_c_order_sha256(values: np.ndarray) -> str:
    """Hash the exact float64 C-order bytes required by the output contract."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("hashed values must be finite")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _panel_matrices(
    rows: Sequence[dict[str, Any]],
    *,
    identity_count: int,
    value_names: tuple[str, ...],
    panel_name: str,
    label_order: Sequence[int] | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    by_seed: dict[int, dict[int, dict[str, Any]]] = {seed: {} for seed in range(4)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{panel_name} rows must be mappings")
        raw_seed = row.get("seed")
        raw_label = row.get("label")
        if (
            isinstance(raw_seed, bool)
            or not isinstance(raw_seed, (int, np.integer))
            or int(raw_seed) not in by_seed
        ):
            raise ValueError(f"{panel_name} seed must be one of 0, 1, 2, 3")
        if (
            isinstance(raw_label, bool)
            or not isinstance(raw_label, (int, np.integer))
            or int(raw_label) < 0
        ):
            raise ValueError(f"{panel_name} labels must be unsigned integers")
        seed = int(raw_seed)
        label = int(raw_label)
        if label in by_seed[seed]:
            raise ValueError(f"duplicate {panel_name} row for seed={seed}, label={label}")
        for name in value_names:
            if name not in row or not np.isfinite(float(row[name])):
                raise ValueError(f"{panel_name} row field {name} must be finite")
        by_seed[seed][label] = row
    label_sets = [set(by_seed[seed]) for seed in range(4)]
    if any(len(labels) != identity_count for labels in label_sets) or any(
        labels != label_sets[0] for labels in label_sets[1:]
    ):
        raise ValueError(
            f"{panel_name} panel must contain a complete shared set of "
            f"{identity_count} identities in every seed"
        )
    if label_order is None:
        labels = sorted(
            label_sets[0],
            key=lambda label: (domain_hash(_IDENTITY_DOMAIN, str(label)), label),
        )
    else:
        labels = list(label_order)
        if len(labels) != identity_count or set(labels) != label_sets[0]:
            raise ValueError(f"{panel_name} labels differ from the registered alternate subset")
    matrices = {
        name: np.asarray(
            [[float(by_seed[seed][label][name]) for label in labels] for seed in range(4)],
            dtype=np.float64,
        )
        for name in value_names
    }
    return labels, matrices


def decide_stage_a(
    rows: Sequence[dict[str, Any]], alternate_rows: Sequence[dict[str, Any]]
) -> dict[str, object]:
    """Aggregate complete panels and apply the frozen Pass200 Stage-A verdict."""
    labels, primary = _panel_matrices(
        rows,
        identity_count=64,
        value_names=("delta", "self_minus_desc", "rho", "log_ratio", "deranged_delta"),
        panel_name="primary",
    )
    registered_alternate_labels = [
        label for index, label in enumerate(labels) if index % 8 in (0, 1)
    ]
    alternate_labels, alternate = _panel_matrices(
        alternate_rows,
        identity_count=16,
        value_names=("delta",),
        panel_name="alternate",
        label_order=registered_alternate_labels,
    )

    seed_deltas_array = primary["delta"].mean(axis=1)
    seed_deltas = {str(seed): float(seed_deltas_array[seed]) for seed in range(4)}
    pooled_delta = float(seed_deltas_array.mean())
    seed_self_desc_array = primary["self_minus_desc"].mean(axis=1)
    seed_self_desc = {str(seed): float(seed_self_desc_array[seed]) for seed in range(4)}
    pooled_self_desc = float(seed_self_desc_array.mean())
    pooled_median_rho = float(np.median(primary["rho"]))
    pooled_median_abs_log_ratio = float(np.median(np.abs(primary["log_ratio"])))
    pooled_deranged_delta = float(primary["deranged_delta"].mean(axis=1).mean())
    alternate_seed_array = alternate["delta"].mean(axis=1)
    alternate_seed_deltas = {str(seed): float(alternate_seed_array[seed]) for seed in range(4)}
    alternate_pooled_delta = float(alternate_seed_array.mean())

    bootstrap_delta = joint_bootstrap(primary["delta"])
    bootstrap_self_desc = joint_bootstrap(primary["self_minus_desc"])
    bootstrap_delta_lower = float(np.percentile(bootstrap_delta, 2.5))
    bootstrap_self_desc_lower = float(np.percentile(bootstrap_self_desc, 2.5))
    primary_seed_ge = int(np.count_nonzero(seed_deltas_array >= 0.02))
    primary_seed_nonpositive = int(np.count_nonzero(seed_deltas_array <= 0.0))
    alternate_seed_positive = int(np.count_nonzero(alternate_seed_array > 0.0))
    alternate_seed_nonpositive = int(np.count_nonzero(alternate_seed_array <= 0.0))
    criteria = {
        "pooled_delta_ge_0_03": pooled_delta >= 0.03,
        "bootstrap_delta_lower_positive": bootstrap_delta_lower > 0.0,
        "three_primary_seed_means_ge_0_02": primary_seed_ge >= 3,
        "pooled_self_minus_desc_positive": pooled_self_desc > 0.0,
        "bootstrap_self_minus_desc_lower_positive": bootstrap_self_desc_lower > 0.0,
        "median_rho_ge_0_20": pooled_median_rho >= 0.20,
        "median_abs_log_ratio_ge_log_1_10": pooled_median_abs_log_ratio >= float(np.log(1.10)),
        "absolute_deranged_delta_le_0_01": abs(pooled_deranged_delta) <= 0.01,
        "alternate_pooled_delta_positive": alternate_pooled_delta > 0.0,
        "three_alternate_seed_means_positive": alternate_seed_positive >= 3,
    }
    fail_clauses = (
        (pooled_delta <= 0.0, "pooled_delta_nonpositive"),
        (primary_seed_nonpositive >= 3, "three_primary_seed_means_nonpositive"),
        (pooled_median_rho < 0.10, "median_rho_below_0_10"),
        (alternate_pooled_delta <= 0.0, "alternate_pooled_delta_nonpositive"),
        (
            alternate_seed_nonpositive >= 3,
            "three_alternate_seed_means_nonpositive",
        ),
    )
    first_fail = next((name for triggered, name in fail_clauses if triggered), None)
    if first_fail is not None:
        stage_a = "FAIL"
        first_decisive_clause = first_fail
    elif all(criteria.values()):
        stage_a = "PASS_ONWARD"
        first_decisive_clause = "all_pass_requirements"
    else:
        stage_a = "UNRESOLVED"
        first_decisive_clause = "no_pass_or_fail_rule"

    return {
        "stage_a": stage_a,
        "first_decisive_clause": first_decisive_clause,
        "criteria": criteria,
        "complete_identity_count": len(labels),
        "alternate_identity_count": len(alternate_labels),
        "complete_labels": labels,
        "alternate_labels": alternate_labels,
        "pooled_delta": pooled_delta,
        "seed_deltas": seed_deltas,
        "seed_deltas_ge_0_02": primary_seed_ge,
        "seed_deltas_nonpositive": primary_seed_nonpositive,
        "pooled_self_minus_desc": pooled_self_desc,
        "seed_self_minus_desc": seed_self_desc,
        "pooled_median_rho": pooled_median_rho,
        "pooled_median_abs_log_ratio": pooled_median_abs_log_ratio,
        "pooled_deranged_delta": pooled_deranged_delta,
        "alternate_pooled_delta": alternate_pooled_delta,
        "alternate_seed_deltas": alternate_seed_deltas,
        "alternate_seed_deltas_positive": alternate_seed_positive,
        "alternate_seed_deltas_nonpositive": alternate_seed_nonpositive,
        "bootstrap_seed": 200,
        "bootstrap_replicates": 10_000,
        "bootstrap_delta_lower_bound": bootstrap_delta_lower,
        "bootstrap_self_desc_lower_bound": bootstrap_self_desc_lower,
        "bootstrap_delta_sha256": float64_c_order_sha256(bootstrap_delta),
        "bootstrap_self_desc_sha256": float64_c_order_sha256(bootstrap_self_desc),
        "numpy_version": np.__version__,
    }
