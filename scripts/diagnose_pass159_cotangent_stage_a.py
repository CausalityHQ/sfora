#!/usr/bin/env python3
"""Preregistered Stage-A diagnostic for Pass159 cotangent transplant.

The CLI and artifact-binding layer are added in later TDD steps.  The functions
below are deliberately NumPy-only so their geometry can be tested in isolation.
"""

from __future__ import annotations

import hashlib

import numpy as np


_VECTOR_EPS = 1.0e-12
_ANTIPODAL_EPS = 1.0e-6
_PARTITION_DOMAIN = "pass159-stage-a-v1|"


def _require_unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite unit vector")
    norm = float(np.linalg.norm(array))
    if norm <= _VECTOR_EPS or not np.isclose(norm, 1.0, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError(f"{name} must be a finite unit vector")
    return array


def _unit_rows(rows: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= _VECTOR_EPS):
        raise ValueError(f"{name} contains a zero vector")
    return array / norms[:, None]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def angular_proxy_anchor_cotangent(
    descriptor: np.ndarray,
    label: int,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """Return the singleton Proxy Anchor cotangent on the descriptor sphere."""
    z = _require_unit(descriptor, name="descriptor")
    normalized_proxies = _unit_rows(proxies, name="proxies")
    labels = np.asarray(proxy_labels, dtype=np.int64)
    if labels.shape != (normalized_proxies.shape[0],):
        raise ValueError("proxy_labels must align with proxies")
    positive = labels == int(label)
    if int(positive.sum()) != 1:
        raise ValueError("singleton diagnostic requires exactly one proxy for the label")
    if normalized_proxies.shape[0] < 2:
        raise ValueError("singleton diagnostic requires at least two classes")
    if alpha <= 0.0 or delta < 0.0:
        raise ValueError("alpha must be positive and delta nonnegative")

    similarities = normalized_proxies @ z
    own_index = int(np.flatnonzero(positive)[0])
    ambient = (
        -float(alpha)
        * _sigmoid(np.asarray([float(alpha) * (float(delta) - similarities[own_index])]))[0]
        * normalized_proxies[own_index]
    )
    foreign = ~positive
    foreign_coefficients = (
        float(alpha)
        * _sigmoid(float(alpha) * (similarities[foreign] + float(delta)))
        / int(foreign.sum())
    )
    ambient = ambient + foreign_coefficients @ normalized_proxies[foreign]
    tangent = ambient - float(np.dot(ambient, z)) * z
    if not np.isfinite(tangent).all():
        raise ValueError("Proxy Anchor cotangent is non-finite")
    return tangent


def parallel_transport(
    tangent: np.ndarray,
    origin: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Parallel-transport a tangent vector along the shortest sphere geodesic."""
    x = _require_unit(origin, name="origin")
    y = _require_unit(target, name="target")
    vector = np.asarray(tangent, dtype=np.float64)
    if vector.shape != x.shape or not np.isfinite(vector).all():
        raise ValueError("tangent must be a finite vector aligned with origin")
    vector = vector - float(np.dot(vector, x)) * x
    norm = float(np.linalg.norm(vector))
    if norm <= _VECTOR_EPS:
        raise ValueError("tangent must have nonzero norm")
    denominator = 1.0 + float(np.dot(x, y))
    if denominator <= _ANTIPODAL_EPS:
        raise ValueError("origin and target are antipodal or numerically unresolved")
    transported = vector - (float(np.dot(vector, y)) / denominator) * (x + y)
    transported = transported - float(np.dot(transported, y)) * y
    if not np.isfinite(transported).all() or np.linalg.norm(transported) <= _VECTOR_EPS:
        raise ValueError("parallel transport produced an invalid vector")
    return transported


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def smooth_margin_gradient(
    receiver: np.ndarray,
    positive_supports: np.ndarray,
    frozen_foreign_supports: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Return the sphere gradient of the frozen smooth retrieval margin."""
    z = _require_unit(receiver, name="receiver")
    positives = _unit_rows(positive_supports, name="positive_supports")
    foreign = _unit_rows(frozen_foreign_supports, name="frozen_foreign_supports")
    if positives.shape[1] != z.shape[0] or foreign.shape[1] != z.shape[0]:
        raise ValueError("supports and receiver must share a descriptor dimension")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    positive_weights = _softmax((positives @ z) / float(tau))
    foreign_weights = _softmax((foreign @ z) / float(tau))
    ambient = positive_weights @ positives - foreign_weights @ foreign
    tangent = ambient - float(np.dot(ambient, z)) * z
    if np.linalg.norm(tangent) <= _VECTOR_EPS:
        raise ValueError("smooth margin gradient has zero norm")
    return tangent


def _partition_hash(example_id: str) -> bytes:
    return hashlib.sha256((_PARTITION_DOMAIN + str(example_id)).encode("utf-8")).digest()


def partition_identity(example_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two support positions and the remaining controller positions."""
    ids = np.asarray(example_ids)
    if ids.ndim != 1 or len(ids) < 5:
        raise ValueError("Pass159 identities require at least five images")
    as_text = [str(value) for value in ids.tolist()]
    if len(set(as_text)) != len(as_text):
        raise ValueError("Pass159 identity contains duplicate example IDs")
    order = sorted(range(len(ids)), key=lambda index: (_partition_hash(as_text[index]), as_text[index]))
    return np.asarray(order[:2], dtype=np.int64), np.asarray(order[2:], dtype=np.int64)
