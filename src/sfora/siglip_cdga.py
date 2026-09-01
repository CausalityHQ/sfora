"""Optimization-only class-disjoint gradient-agreement primitives."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CDGADomainSplit:
    """Two deterministic pseudo-domains within one fold's fit labels."""

    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    domain_a_labels: tuple[int, ...]
    domain_b_labels: tuple[int, ...]
    master_seed_sha256: str
    sha256: str


@dataclass(frozen=True)
class CDGAGradientProjection:
    """One symmetric projection-gradient conflict-removal result."""

    left: torch.Tensor
    right: torch.Tensor
    conflict: bool
    pre_projection_cosine: float


def _label_tuple(value: object) -> bool:
    return (
        type(value) is tuple
        and bool(value)
        and all(type(label) is int and label >= 0 for label in value)
        and tuple(sorted(value)) == value
        and len(set(value)) == len(value)
    )


def _domain_sha256(
    *,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
    domain_a_labels: tuple[int, ...],
    domain_b_labels: tuple[int, ...],
    master_seed_sha256: str,
) -> str:
    payload = bytearray(b"sfora-siglip-cdga-domain-v1\0")
    payload.extend(bytes.fromhex(master_seed_sha256))
    for labels in (fit_labels, validation_labels, domain_a_labels, domain_b_labels):
        payload.extend(len(labels).to_bytes(8, "big"))
        for label in labels:
            payload.extend(struct.pack(">q", label))
    return hashlib.sha256(payload).hexdigest()


def build_cdga_domain_split(
    *,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
    master_seed_sha256: str,
) -> CDGADomainSplit:
    """Split sorted fit labels into two seed-bound alternating pseudo-domains."""

    if (
        not _label_tuple(fit_labels)
        or not _label_tuple(validation_labels)
        or len(fit_labels) < 4
        or not set(fit_labels).isdisjoint(validation_labels)
        or type(master_seed_sha256) is not str
        or len(master_seed_sha256) != 64
        or master_seed_sha256.lower() != master_seed_sha256
    ):
        raise ValueError("CDGA domain authority differs")
    try:
        seed_value = int(master_seed_sha256, 16)
    except ValueError as error:
        raise ValueError("CDGA domain authority differs") from error
    offset = seed_value % len(fit_labels)
    rotated = fit_labels[offset:] + fit_labels[:offset]
    domain_a_labels = tuple(sorted(rotated[::2]))
    domain_b_labels = tuple(sorted(rotated[1::2]))
    if (
        not domain_a_labels
        or not domain_b_labels
        or not set(domain_a_labels).isdisjoint(domain_b_labels)
        or sorted(domain_a_labels + domain_b_labels) != list(fit_labels)
    ):
        raise ValueError("CDGA domain partition differs")
    return CDGADomainSplit(
        fit_labels=fit_labels,
        validation_labels=validation_labels,
        domain_a_labels=domain_a_labels,
        domain_b_labels=domain_b_labels,
        master_seed_sha256=master_seed_sha256,
        sha256=_domain_sha256(
            fit_labels=fit_labels,
            validation_labels=validation_labels,
            domain_a_labels=domain_a_labels,
            domain_b_labels=domain_b_labels,
            master_seed_sha256=master_seed_sha256,
        ),
    )


def symmetric_conflict_projection(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float,
) -> CDGAGradientProjection:
    """Remove each negative projection onto the other gradient symmetrically."""

    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or left.ndim != 1
        or right.shape != left.shape
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.device != right.device
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
        or type(epsilon) is not float
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError("CDGA gradient authority differs")
    dot = torch.dot(left, right)
    left_norm_sq = torch.dot(left, left)
    right_norm_sq = torch.dot(right, right)
    if float(left_norm_sq) == 0.0 or float(right_norm_sq) == 0.0:
        cosine = 0.0
    else:
        cosine = float(dot.double() / torch.sqrt(left_norm_sq.double() * right_norm_sq.double()))
    conflict = float(dot) < 0.0
    if conflict:
        projected_left = left - dot / torch.clamp_min(right_norm_sq, epsilon) * right
        projected_right = right - dot / torch.clamp_min(left_norm_sq, epsilon) * left
    else:
        projected_left = left.clone()
        projected_right = right.clone()
    if (
        not math.isfinite(cosine)
        or not bool(torch.isfinite(projected_left).all())
        or not bool(torch.isfinite(projected_right).all())
    ):
        raise ValueError("CDGA projected gradient differs")
    return CDGAGradientProjection(
        left=projected_left,
        right=projected_right,
        conflict=conflict,
        pre_projection_cosine=cosine,
    )
