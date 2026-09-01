"""Optimization-only class-disjoint gradient-agreement primitives."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_head_screen import FeatureSplitAuthority
from sfora.siglip_sfq import SFQFold, build_sfq_fold_schedule, sfq_label_vector_sha256
from sfora.token_set_proxy_anchor import proxy_anchor_loss


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


@dataclass(frozen=True)
class CDGAFoldTrainingEvidence:
    """Matched learned projections and primitive training evidence for one fold."""

    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    trained_example_count: int
    train_steps: int
    examples_per_class: int
    domain_split_sha256: str
    batch_schedule_sha256: str
    initial_weight: torch.Tensor
    comparator_weight: torch.Tensor
    cdga_weight: torch.Tensor
    comparator_initial_loss: float
    comparator_final_loss: float
    cdga_initial_loss: float
    cdga_final_loss: float
    conflict_count: int
    mean_pre_projection_cosine: float


@dataclass(frozen=True, slots=True)
class CDGAFoldEvidence:
    """Integer retrieval and training evidence for one held-out fold."""

    ordinal: int
    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    fit_count: int
    query_count: int
    raw_hits: int
    spectral_hits: int
    comparator_hits: int
    cdga_hits: int
    spectral_fit_hits: int
    comparator_fit_hits: int
    cdga_minus_comparator_ppm: int
    comparator_initial_loss: float
    comparator_final_loss: float
    cdga_initial_loss: float
    cdga_final_loss: float
    conflict_count: int
    mean_pre_projection_cosine: float
    domain_split_sha256: str
    batch_schedule_sha256: str
    spectral_projection_sha256: str
    comparator_projection_sha256: str
    cdga_projection_sha256: str


@dataclass(frozen=True, slots=True)
class CDGAResult:
    """Canonical aggregate reconstructed from exact fold primitives."""

    schema: str
    claim_eligible: bool
    official_test_access: bool
    source_manifest_sha256: str
    feature_cache_manifest_sha256: str
    ordered_example_ids_sha256: str
    feature_matrix_sha256: str
    label_vector_sha256: str
    master_seed_sha256: str
    input_dimensions: int
    output_dimensions: int
    fold_count: int
    fold_schedule_sha256: str
    train_steps: int
    examples_per_class: int
    query_count: int
    raw_hits: int
    spectral_hits: int
    comparator_hits: int
    cdga_hits: int
    raw_recall_ppm: int
    spectral_recall_ppm: int
    comparator_recall_ppm: int
    cdga_recall_ppm: int
    cdga_minus_comparator_ppm: int
    conflict_count: int
    valid: bool
    passed: bool
    folds: tuple[CDGAFoldEvidence, ...]


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


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("CDGA digest authority differs")
    return value


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _projection_sha256(role: str, value: torch.Tensor) -> str:
    if (
        type(role) is not str
        or not role
        or type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("CDGA projection digest authority differs")
    payload = bytearray(b"sfora-siglip-cdga-projection-v1\0")
    payload.extend(role.encode())
    payload.extend(b"\0")
    payload.extend(value.shape[0].to_bytes(8, "big"))
    payload.extend(value.shape[1].to_bytes(8, "big"))
    payload.extend(value.numpy().tobytes(order="C"))
    return hashlib.sha256(payload).hexdigest()


def _spectral_weight(features: torch.Tensor, output_dimensions: int) -> torch.Tensor:
    normalized = F.normalize(features.double(), dim=1)
    _left, _singular_values, right = torch.linalg.svd(normalized, full_matrices=False)
    weight = right[:output_dimensions].clone()
    for row in range(weight.shape[0]):
        pivot = int(torch.argmax(torch.abs(weight[row])))
        if float(weight[row, pivot]) < 0.0:
            weight[row].neg_()
    result = weight.float().contiguous()
    if result.shape != (output_dimensions, features.shape[1]) or not bool(
        torch.isfinite(result).all()
    ):
        raise ValueError("CDGA spectral projection differs")
    return result


def _batch_rows(
    labels: torch.Tensor,
    domain_labels: tuple[int, ...],
    *,
    step: int,
    examples_per_class: int,
    master_seed_sha256: str,
    fold_ordinal: int,
    domain_ordinal: int,
) -> tuple[int, ...]:
    rows: list[int] = []
    seed_bytes = bytes.fromhex(master_seed_sha256)
    for label in domain_labels:
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
        if len(candidates) < examples_per_class:
            raise ValueError("CDGA class batch population differs")
        payload = bytearray(b"sfora-siglip-cdga-batch-v1\0")
        payload.extend(seed_bytes)
        payload.extend(struct.pack(">QQQQ", fold_ordinal, domain_ordinal, step, label))
        offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(candidates)
        rows.extend(
            candidates[(offset + position) % len(candidates)]
            for position in range(examples_per_class)
        )
    return tuple(rows)


def _batch_schedule(
    labels: torch.Tensor,
    split: CDGADomainSplit,
    *,
    train_steps: int,
    examples_per_class: int,
    fold_ordinal: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    schedule = tuple(
        (
            _batch_rows(
                labels,
                split.domain_a_labels,
                step=step,
                examples_per_class=examples_per_class,
                master_seed_sha256=split.master_seed_sha256,
                fold_ordinal=fold_ordinal,
                domain_ordinal=0,
            ),
            _batch_rows(
                labels,
                split.domain_b_labels,
                step=step,
                examples_per_class=examples_per_class,
                master_seed_sha256=split.master_seed_sha256,
                fold_ordinal=fold_ordinal,
                domain_ordinal=1,
            ),
        )
        for step in range(train_steps)
    )
    if any(set(left).intersection(right) for left, right in schedule):
        raise ValueError("CDGA domain batch overlap differs")
    return schedule


def _batch_schedule_sha256(
    schedule: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    *,
    domain_split_sha256: str,
) -> str:
    payload = bytearray(b"sfora-siglip-cdga-batch-schedule-v1\0")
    payload.extend(bytes.fromhex(domain_split_sha256))
    payload.extend(len(schedule).to_bytes(8, "big"))
    for pair in schedule:
        for rows in pair:
            payload.extend(len(rows).to_bytes(8, "big"))
            for row in rows:
                payload.extend(row.to_bytes(8, "big"))
    return hashlib.sha256(payload).hexdigest()


class _CDGAHead(nn.Module):
    def __init__(self, weight: torch.Tensor, proxies: torch.Tensor) -> None:
        super().__init__()
        self.projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proxies = nn.Parameter(proxies.clone())
        with torch.no_grad():
            self.projection.weight.copy_(weight)

    def loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        alpha: float,
        delta: float,
    ) -> torch.Tensor:
        embeddings = F.normalize(self.projection(features).float(), dim=1)
        scores = embeddings @ F.normalize(self.proxies.float(), dim=1).T
        return proxy_anchor_loss(scores, labels, alpha=alpha, delta=delta)


def _initial_proxies(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    fit_labels: tuple[int, ...],
) -> torch.Tensor:
    with torch.no_grad():
        embeddings = F.normalize(features @ weight.T, dim=1)
        centers = []
        for local_label, _original_label in enumerate(fit_labels):
            center = embeddings[labels == local_label].mean(dim=0)
            if (
                not bool(torch.isfinite(center).all())
                or float(torch.linalg.vector_norm(center)) <= 0
            ):
                raise ValueError("CDGA initial proxy authority differs")
            centers.append(F.normalize(center, dim=0))
    return torch.stack(centers).contiguous()


def train_cdga_fold(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    fold: SFQFold,
    master_seed_sha256: str,
    output_dimensions: int,
    train_steps: int,
    examples_per_class: int,
    projection_learning_rate: float,
    proxy_learning_rate: float,
    weight_decay: float,
    alpha: float,
    delta: float,
    device: torch.device,
) -> CDGAFoldTrainingEvidence:
    """Train matched ordinary and conflict-projected heads on one fold complement."""

    floats = (projection_learning_rate, proxy_learning_rate, weight_decay, alpha, delta)
    if (
        type(features) is not torch.Tensor
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.ndim != 2
        or not bool(torch.isfinite(features).all())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (features.shape[0],)
        or type(fold) is not SFQFold
        or type(output_dimensions) is not int
        or not 1 <= output_dimensions <= min(features.shape)
        or type(train_steps) is not int
        or train_steps < 1
        or type(examples_per_class) is not int
        or examples_per_class < 1
        or any(type(value) is not float or not math.isfinite(value) for value in floats)
        or projection_learning_rate <= 0.0
        or proxy_learning_rate <= 0.0
        or weight_decay < 0.0
        or alpha <= 0.0
        or not 0.0 <= delta < 1.0
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
        or (device.type == "cuda" and not torch.cuda.is_available())
    ):
        raise ValueError("CDGA training authority differs")
    master_seed_sha256 = _hex_digest(master_seed_sha256)
    observed_labels = tuple(sorted(int(value) for value in torch.unique(labels).tolist()))
    if observed_labels != tuple(sorted(fold.fit_labels + fold.validation_labels)):
        raise ValueError("CDGA fold label authority differs")
    fit_mask = torch.isin(labels, torch.tensor(fold.fit_labels, dtype=torch.int64))
    fit_features_cpu = features[fit_mask].contiguous()
    fit_original_labels = labels[fit_mask].contiguous()
    local_by_original = {label: index for index, label in enumerate(fold.fit_labels)}
    fit_labels_cpu = torch.tensor(
        [local_by_original[int(label)] for label in fit_original_labels.tolist()],
        dtype=torch.int64,
    )
    split = build_cdga_domain_split(
        fit_labels=fold.fit_labels,
        validation_labels=fold.validation_labels,
        master_seed_sha256=master_seed_sha256,
    )
    schedule = _batch_schedule(
        fit_original_labels,
        split,
        train_steps=train_steps,
        examples_per_class=examples_per_class,
        fold_ordinal=fold.ordinal,
    )
    initial_weight = _spectral_weight(fit_features_cpu, output_dimensions)
    initial_proxies = _initial_proxies(
        fit_features_cpu,
        fit_labels_cpu,
        initial_weight,
        fold.fit_labels,
    )
    comparator = _CDGAHead(initial_weight, initial_proxies).to(device)
    cdga = _CDGAHead(initial_weight, initial_proxies).to(device)
    execution_features = fit_features_cpu.to(device)
    execution_labels = fit_labels_cpu.to(device)
    comparator_optimizer = torch.optim.AdamW(
        [
            {"params": [comparator.projection.weight], "lr": projection_learning_rate},
            {"params": [comparator.proxies], "lr": proxy_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    cdga_optimizer = torch.optim.AdamW(
        [
            {"params": [cdga.projection.weight], "lr": projection_learning_rate},
            {"params": [cdga.proxies], "lr": proxy_learning_rate},
        ],
        weight_decay=weight_decay,
    )

    def full_loss(model: _CDGAHead) -> torch.Tensor:
        return model.loss(execution_features, execution_labels, alpha=alpha, delta=delta)

    with torch.no_grad():
        comparator_initial_loss = float(full_loss(comparator))
        cdga_initial_loss = float(full_loss(cdga))
    conflict_count = 0
    cosines: list[float] = []
    for left_rows, right_rows in schedule:
        left = torch.tensor(left_rows, dtype=torch.int64, device=device)
        right = torch.tensor(right_rows, dtype=torch.int64, device=device)

        comparator_optimizer.zero_grad(set_to_none=True)
        comparator_loss = (
            comparator.loss(
                execution_features[left], execution_labels[left], alpha=alpha, delta=delta
            )
            + comparator.loss(
                execution_features[right], execution_labels[right], alpha=alpha, delta=delta
            )
        ) / 2.0
        comparator_loss.backward()
        torch.nn.utils.clip_grad_norm_(comparator.parameters(), 10.0)
        comparator_optimizer.step()

        cdga_optimizer.zero_grad(set_to_none=True)
        left_loss = cdga.loss(
            execution_features[left], execution_labels[left], alpha=alpha, delta=delta
        )
        left_gradients = torch.autograd.grad(
            left_loss,
            (cdga.projection.weight, cdga.proxies),
        )
        right_loss = cdga.loss(
            execution_features[right], execution_labels[right], alpha=alpha, delta=delta
        )
        right_gradients = torch.autograd.grad(
            right_loss,
            (cdga.projection.weight, cdga.proxies),
        )
        projected = symmetric_conflict_projection(
            left_gradients[0].reshape(-1),
            right_gradients[0].reshape(-1),
            epsilon=1.0e-12,
        )
        cdga.projection.weight.grad = ((projected.left + projected.right) / 2.0).reshape_as(
            cdga.projection.weight
        )
        cdga.proxies.grad = (left_gradients[1] + right_gradients[1]) / 2.0
        torch.nn.utils.clip_grad_norm_(cdga.parameters(), 10.0)
        cdga_optimizer.step()
        conflict_count += int(projected.conflict)
        cosines.append(projected.pre_projection_cosine)

    with torch.no_grad():
        comparator_final_loss = float(full_loss(comparator))
        cdga_final_loss = float(full_loss(cdga))
    scalars = (
        comparator_initial_loss,
        comparator_final_loss,
        cdga_initial_loss,
        cdga_final_loss,
        *cosines,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise RuntimeError("CDGA training evidence became nonfinite")
    return CDGAFoldTrainingEvidence(
        fit_labels=fold.fit_labels,
        validation_labels=fold.validation_labels,
        trained_example_count=fit_features_cpu.shape[0],
        train_steps=train_steps,
        examples_per_class=examples_per_class,
        domain_split_sha256=split.sha256,
        batch_schedule_sha256=_batch_schedule_sha256(schedule, domain_split_sha256=split.sha256),
        initial_weight=initial_weight,
        comparator_weight=comparator.projection.weight.detach().float().cpu().contiguous(),
        cdga_weight=cdga.projection.weight.detach().float().cpu().contiguous(),
        comparator_initial_loss=comparator_initial_loss,
        comparator_final_loss=comparator_final_loss,
        cdga_initial_loss=cdga_initial_loss,
        cdga_final_loss=cdga_final_loss,
        conflict_count=conflict_count,
        mean_pre_projection_cosine=sum(cosines) / len(cosines),
    )


def _recall_hits(features: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    normalized = F.normalize(features.float(), dim=1)
    scores = normalized @ normalized.T
    scores.fill_diagonal_(-torch.inf)
    nearest = scores.argmax(dim=1)
    hits = int((labels[nearest] == labels).sum())
    return hits, labels.numel()


def _fold_mapping(fold: CDGAFoldEvidence) -> dict[str, object]:
    return {
        "ordinal": fold.ordinal,
        "fit_labels": list(fold.fit_labels),
        "validation_labels": list(fold.validation_labels),
        "fit_count": fold.fit_count,
        "query_count": fold.query_count,
        "raw_hits": fold.raw_hits,
        "spectral_hits": fold.spectral_hits,
        "comparator_hits": fold.comparator_hits,
        "cdga_hits": fold.cdga_hits,
        "spectral_fit_hits": fold.spectral_fit_hits,
        "comparator_fit_hits": fold.comparator_fit_hits,
        "cdga_minus_comparator_ppm": fold.cdga_minus_comparator_ppm,
        "comparator_initial_loss": fold.comparator_initial_loss,
        "comparator_final_loss": fold.comparator_final_loss,
        "cdga_initial_loss": fold.cdga_initial_loss,
        "cdga_final_loss": fold.cdga_final_loss,
        "conflict_count": fold.conflict_count,
        "mean_pre_projection_cosine": fold.mean_pre_projection_cosine,
        "domain_split_sha256": fold.domain_split_sha256,
        "batch_schedule_sha256": fold.batch_schedule_sha256,
        "spectral_projection_sha256": fold.spectral_projection_sha256,
        "comparator_projection_sha256": fold.comparator_projection_sha256,
        "cdga_projection_sha256": fold.cdga_projection_sha256,
    }


def _result_mapping(result: CDGAResult) -> dict[str, object]:
    return {
        "schema": result.schema,
        "claim_eligible": result.claim_eligible,
        "official_test_access": result.official_test_access,
        "source_manifest_sha256": result.source_manifest_sha256,
        "feature_cache_manifest_sha256": result.feature_cache_manifest_sha256,
        "ordered_example_ids_sha256": result.ordered_example_ids_sha256,
        "feature_matrix_sha256": result.feature_matrix_sha256,
        "label_vector_sha256": result.label_vector_sha256,
        "master_seed_sha256": result.master_seed_sha256,
        "input_dimensions": result.input_dimensions,
        "output_dimensions": result.output_dimensions,
        "fold_count": result.fold_count,
        "fold_schedule_sha256": result.fold_schedule_sha256,
        "train_steps": result.train_steps,
        "examples_per_class": result.examples_per_class,
        "query_count": result.query_count,
        "raw_hits": result.raw_hits,
        "spectral_hits": result.spectral_hits,
        "comparator_hits": result.comparator_hits,
        "cdga_hits": result.cdga_hits,
        "raw_recall_ppm": result.raw_recall_ppm,
        "spectral_recall_ppm": result.spectral_recall_ppm,
        "comparator_recall_ppm": result.comparator_recall_ppm,
        "cdga_recall_ppm": result.cdga_recall_ppm,
        "cdga_minus_comparator_ppm": result.cdga_minus_comparator_ppm,
        "conflict_count": result.conflict_count,
        "valid": result.valid,
        "passed": result.passed,
        "folds": [_fold_mapping(fold) for fold in result.folds],
    }


def run_cdga_fold_diagnostic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_authority: FeatureSplitAuthority,
    feature_cache_manifest_sha256: str,
    master_seed_sha256: str,
    output_dimensions: int,
    fold_count: int,
    train_steps: int,
    examples_per_class: int,
    device: str,
) -> bytes:
    """Train matched arms on fold complements and score held optimization classes."""

    if type(split_authority) is not FeatureSplitAuthority or device not in {"cpu", "cuda"}:
        raise ValueError("CDGA diagnostic authority differs")
    split_authority.validated(features=features)
    feature_cache_manifest_sha256 = _hex_digest(feature_cache_manifest_sha256)
    master_seed_sha256 = _hex_digest(master_seed_sha256)
    schedule = build_sfq_fold_schedule(
        features,
        labels,
        split_authority,
        fold_count=fold_count,
    )
    execution_device = torch.device(device)
    fold_results = []
    for fold in schedule.folds:
        trained = train_cdga_fold(
            features,
            labels,
            fold=fold,
            master_seed_sha256=master_seed_sha256,
            output_dimensions=output_dimensions,
            train_steps=train_steps,
            examples_per_class=examples_per_class,
            projection_learning_rate=1.0e-3,
            proxy_learning_rate=1.0e-2,
            weight_decay=1.0e-4,
            alpha=32.0,
            delta=0.1,
            device=execution_device,
        )
        fit_mask = torch.isin(labels, torch.tensor(fold.fit_labels, dtype=torch.int64))
        validation_mask = torch.isin(
            labels, torch.tensor(fold.validation_labels, dtype=torch.int64)
        )
        fit_features = features[fit_mask]
        fit_labels = labels[fit_mask]
        validation_features = features[validation_mask]
        validation_labels = labels[validation_mask]
        raw_hits, query_count = _recall_hits(validation_features, validation_labels)
        spectral_hits, _ = _recall_hits(
            validation_features @ trained.initial_weight.T, validation_labels
        )
        comparator_hits, _ = _recall_hits(
            validation_features @ trained.comparator_weight.T, validation_labels
        )
        cdga_hits, _ = _recall_hits(validation_features @ trained.cdga_weight.T, validation_labels)
        spectral_fit_hits, _ = _recall_hits(fit_features @ trained.initial_weight.T, fit_labels)
        comparator_fit_hits, fit_count = _recall_hits(
            fit_features @ trained.comparator_weight.T, fit_labels
        )
        fold_results.append(
            CDGAFoldEvidence(
                ordinal=fold.ordinal,
                fit_labels=fold.fit_labels,
                validation_labels=fold.validation_labels,
                fit_count=fit_count,
                query_count=query_count,
                raw_hits=raw_hits,
                spectral_hits=spectral_hits,
                comparator_hits=comparator_hits,
                cdga_hits=cdga_hits,
                spectral_fit_hits=spectral_fit_hits,
                comparator_fit_hits=comparator_fit_hits,
                cdga_minus_comparator_ppm=(cdga_hits - comparator_hits) * 1_000_000 // query_count,
                comparator_initial_loss=trained.comparator_initial_loss,
                comparator_final_loss=trained.comparator_final_loss,
                cdga_initial_loss=trained.cdga_initial_loss,
                cdga_final_loss=trained.cdga_final_loss,
                conflict_count=trained.conflict_count,
                mean_pre_projection_cosine=trained.mean_pre_projection_cosine,
                domain_split_sha256=trained.domain_split_sha256,
                batch_schedule_sha256=trained.batch_schedule_sha256,
                spectral_projection_sha256=_projection_sha256("spectral", trained.initial_weight),
                comparator_projection_sha256=_projection_sha256(
                    "comparator", trained.comparator_weight
                ),
                cdga_projection_sha256=_projection_sha256("cdga", trained.cdga_weight),
            )
        )
    folds = tuple(fold_results)
    query_count = sum(fold.query_count for fold in folds)
    hit_totals = tuple(
        sum(getattr(fold, name) for fold in folds)
        for name in ("raw_hits", "spectral_hits", "comparator_hits", "cdga_hits")
    )
    recalls = tuple(hits * 1_000_000 // query_count for hits in hit_totals)
    conflict_count = sum(fold.conflict_count for fold in folds)
    valid = (
        conflict_count > 0
        and all(fold.comparator_final_loss <= fold.comparator_initial_loss for fold in folds)
        and all(fold.cdga_final_loss <= fold.cdga_initial_loss for fold in folds)
        and all(fold.comparator_fit_hits >= fold.spectral_fit_hits for fold in folds)
    )
    passed = (
        valid
        and recalls[3] - recalls[2] >= 2_000
        and hit_totals[3] >= hit_totals[1]
        and all(fold.cdga_minus_comparator_ppm >= -10_000 for fold in folds)
    )
    result = CDGAResult(
        schema="sfora-siglip-cdga-fold-diagnostic-v1",
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=split_authority.source_manifest_sha256,
        feature_cache_manifest_sha256=feature_cache_manifest_sha256,
        ordered_example_ids_sha256=split_authority.ordered_example_ids_sha256,
        feature_matrix_sha256=split_authority.feature_matrix_sha256,
        label_vector_sha256=sfq_label_vector_sha256(labels),
        master_seed_sha256=master_seed_sha256,
        input_dimensions=features.shape[1],
        output_dimensions=output_dimensions,
        fold_count=fold_count,
        fold_schedule_sha256=schedule.sha256,
        train_steps=train_steps,
        examples_per_class=examples_per_class,
        query_count=query_count,
        raw_hits=hit_totals[0],
        spectral_hits=hit_totals[1],
        comparator_hits=hit_totals[2],
        cdga_hits=hit_totals[3],
        raw_recall_ppm=recalls[0],
        spectral_recall_ppm=recalls[1],
        comparator_recall_ppm=recalls[2],
        cdga_recall_ppm=recalls[3],
        cdga_minus_comparator_ppm=recalls[3] - recalls[2],
        conflict_count=conflict_count,
        valid=valid,
        passed=passed,
        folds=folds,
    )
    raw = _canonical_bytes(_result_mapping(result))
    validate_cdga_result_bytes(raw)
    return raw


def _exact_dict(value: object, keys: set[str], *, role: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"CDGA {role} schema differs")
    return cast(dict[str, object], value)


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("CDGA integer evidence differs")
    return value


def _finite(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("CDGA float evidence differs")
    return value


def _labels(value: object) -> tuple[int, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(label) is not int or label < 0 for label in value)
        or value != sorted(value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("CDGA label evidence differs")
    return tuple(value)


_FOLD_KEYS = {
    "ordinal",
    "fit_labels",
    "validation_labels",
    "fit_count",
    "query_count",
    "raw_hits",
    "spectral_hits",
    "comparator_hits",
    "cdga_hits",
    "spectral_fit_hits",
    "comparator_fit_hits",
    "cdga_minus_comparator_ppm",
    "comparator_initial_loss",
    "comparator_final_loss",
    "cdga_initial_loss",
    "cdga_final_loss",
    "conflict_count",
    "mean_pre_projection_cosine",
    "domain_split_sha256",
    "batch_schedule_sha256",
    "spectral_projection_sha256",
    "comparator_projection_sha256",
    "cdga_projection_sha256",
}

_RESULT_KEYS = {
    "schema",
    "claim_eligible",
    "official_test_access",
    "source_manifest_sha256",
    "feature_cache_manifest_sha256",
    "ordered_example_ids_sha256",
    "feature_matrix_sha256",
    "label_vector_sha256",
    "master_seed_sha256",
    "input_dimensions",
    "output_dimensions",
    "fold_count",
    "fold_schedule_sha256",
    "train_steps",
    "examples_per_class",
    "query_count",
    "raw_hits",
    "spectral_hits",
    "comparator_hits",
    "cdga_hits",
    "raw_recall_ppm",
    "spectral_recall_ppm",
    "comparator_recall_ppm",
    "cdga_recall_ppm",
    "cdga_minus_comparator_ppm",
    "conflict_count",
    "valid",
    "passed",
    "folds",
}


def _parse_fold(value: object) -> CDGAFoldEvidence:
    mapping = _exact_dict(value, _FOLD_KEYS, role="fold")
    fit_labels = _labels(mapping["fit_labels"])
    validation_labels = _labels(mapping["validation_labels"])
    fit_count = _integer(mapping["fit_count"], minimum=2)
    query_count = _integer(mapping["query_count"], minimum=2)
    counts = {
        name: _integer(mapping[name])
        for name in (
            "raw_hits",
            "spectral_hits",
            "comparator_hits",
            "cdga_hits",
            "spectral_fit_hits",
            "comparator_fit_hits",
            "conflict_count",
        )
    }
    if (
        not set(fit_labels).isdisjoint(validation_labels)
        or any(
            counts[name] > query_count
            for name in ("raw_hits", "spectral_hits", "comparator_hits", "cdga_hits")
        )
        or any(counts[name] > fit_count for name in ("spectral_fit_hits", "comparator_fit_hits"))
    ):
        raise ValueError("CDGA fold count authority differs")
    cdga_minus = mapping["cdga_minus_comparator_ppm"]
    if (
        type(cdga_minus) is not int
        or cdga_minus
        != (counts["cdga_hits"] - counts["comparator_hits"]) * 1_000_000 // query_count
    ):
        raise ValueError("CDGA fold delta differs")
    return CDGAFoldEvidence(
        ordinal=_integer(mapping["ordinal"]),
        fit_labels=fit_labels,
        validation_labels=validation_labels,
        fit_count=fit_count,
        query_count=query_count,
        raw_hits=counts["raw_hits"],
        spectral_hits=counts["spectral_hits"],
        comparator_hits=counts["comparator_hits"],
        cdga_hits=counts["cdga_hits"],
        spectral_fit_hits=counts["spectral_fit_hits"],
        comparator_fit_hits=counts["comparator_fit_hits"],
        cdga_minus_comparator_ppm=cdga_minus,
        comparator_initial_loss=_finite(mapping["comparator_initial_loss"]),
        comparator_final_loss=_finite(mapping["comparator_final_loss"]),
        cdga_initial_loss=_finite(mapping["cdga_initial_loss"]),
        cdga_final_loss=_finite(mapping["cdga_final_loss"]),
        conflict_count=counts["conflict_count"],
        mean_pre_projection_cosine=_finite(mapping["mean_pre_projection_cosine"]),
        domain_split_sha256=_hex_digest(mapping["domain_split_sha256"]),
        batch_schedule_sha256=_hex_digest(mapping["batch_schedule_sha256"]),
        spectral_projection_sha256=_hex_digest(mapping["spectral_projection_sha256"]),
        comparator_projection_sha256=_hex_digest(mapping["comparator_projection_sha256"]),
        cdga_projection_sha256=_hex_digest(mapping["cdga_projection_sha256"]),
    )


def validate_cdga_result_bytes(raw: bytes) -> CDGAResult:
    """Validate canonical CDGA evidence and recompute all aggregate decisions."""

    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("CDGA result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CDGA result JSON differs") from error
    mapping = _exact_dict(value, _RESULT_KEYS, role="result")
    if _canonical_bytes(mapping) != raw:
        raise ValueError("CDGA result canonical bytes differ")
    if (
        mapping["schema"] != "sfora-siglip-cdga-fold-diagnostic-v1"
        or mapping["claim_eligible"] is not False
        or mapping["official_test_access"] is not False
    ):
        raise ValueError("CDGA result identity differs")
    folds_value = mapping["folds"]
    if type(folds_value) is not list or not folds_value:
        raise ValueError("CDGA fold collection differs")
    folds = tuple(_parse_fold(fold) for fold in folds_value)
    fold_count = _integer(mapping["fold_count"], minimum=2)
    if len(folds) != fold_count or tuple(fold.ordinal for fold in folds) != tuple(
        range(fold_count)
    ):
        raise ValueError("CDGA fold ordinal authority differs")
    validation_labels = sorted(label for fold in folds for label in fold.validation_labels)
    if len(validation_labels) != len(set(validation_labels)):
        raise ValueError("CDGA validation fold overlap differs")
    query_count = sum(fold.query_count for fold in folds)
    hits = tuple(
        sum(getattr(fold, name) for fold in folds)
        for name in ("raw_hits", "spectral_hits", "comparator_hits", "cdga_hits")
    )
    recalls = tuple(count * 1_000_000 // query_count for count in hits)
    conflict_count = sum(fold.conflict_count for fold in folds)
    valid = (
        conflict_count > 0
        and all(fold.comparator_final_loss <= fold.comparator_initial_loss for fold in folds)
        and all(fold.cdga_final_loss <= fold.cdga_initial_loss for fold in folds)
        and all(fold.comparator_fit_hits >= fold.spectral_fit_hits for fold in folds)
    )
    passed = (
        valid
        and recalls[3] - recalls[2] >= 2_000
        and hits[3] >= hits[1]
        and all(fold.cdga_minus_comparator_ppm >= -10_000 for fold in folds)
    )
    expected = {
        "query_count": query_count,
        "raw_hits": hits[0],
        "spectral_hits": hits[1],
        "comparator_hits": hits[2],
        "cdga_hits": hits[3],
        "raw_recall_ppm": recalls[0],
        "spectral_recall_ppm": recalls[1],
        "comparator_recall_ppm": recalls[2],
        "cdga_recall_ppm": recalls[3],
        "cdga_minus_comparator_ppm": recalls[3] - recalls[2],
        "conflict_count": conflict_count,
        "valid": valid,
        "passed": passed,
    }
    if any(
        type(mapping[name]) is not type(expected_value) or mapping[name] != expected_value
        for name, expected_value in expected.items()
    ):
        raise ValueError("CDGA aggregate evidence differs")
    result = CDGAResult(
        schema=cast(str, mapping["schema"]),
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=_hex_digest(mapping["source_manifest_sha256"]),
        feature_cache_manifest_sha256=_hex_digest(mapping["feature_cache_manifest_sha256"]),
        ordered_example_ids_sha256=_hex_digest(mapping["ordered_example_ids_sha256"]),
        feature_matrix_sha256=_hex_digest(mapping["feature_matrix_sha256"]),
        label_vector_sha256=_hex_digest(mapping["label_vector_sha256"]),
        master_seed_sha256=_hex_digest(mapping["master_seed_sha256"]),
        input_dimensions=_integer(mapping["input_dimensions"], minimum=2),
        output_dimensions=_integer(mapping["output_dimensions"], minimum=1),
        fold_count=fold_count,
        fold_schedule_sha256=_hex_digest(mapping["fold_schedule_sha256"]),
        train_steps=_integer(mapping["train_steps"], minimum=1),
        examples_per_class=_integer(mapping["examples_per_class"], minimum=1),
        query_count=query_count,
        raw_hits=hits[0],
        spectral_hits=hits[1],
        comparator_hits=hits[2],
        cdga_hits=hits[3],
        raw_recall_ppm=recalls[0],
        spectral_recall_ppm=recalls[1],
        comparator_recall_ppm=recalls[2],
        cdga_recall_ppm=recalls[3],
        cdga_minus_comparator_ppm=recalls[3] - recalls[2],
        conflict_count=conflict_count,
        valid=valid,
        passed=passed,
        folds=folds,
    )
    if result.output_dimensions > result.input_dimensions:
        raise ValueError("CDGA result dimension authority differs")
    return result
