"""Leakage-safe Seen-Span Occupancy and Restoration primitives."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from sfora.siglip_head_screen import FeatureSplitAuthority, build_feature_split_authority
from sfora.siglip_sfq import build_sfq_fold_schedule

_SVD_RELATIVE_CUTOFF = 1.0e-10
_PROJECTOR_TOLERANCE = 1.0e-10
_UNIT_NORM_TOLERANCE = 2.0e-6
SSOR_BETA_GRID = (1.0, 0.5, 0.75, 1.25, 1.5, 2.0)


def _ordered_example_ids_sha256(example_ids: tuple[str, ...]) -> str:
    payload = bytearray(b"sfora-ordered-example-ids-v1\0")
    payload.extend(len(example_ids).to_bytes(8, "big"))
    for example_id in example_ids:
        encoded = example_id.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def ssor_float_tensor_sha256(role: str, tensor: torch.Tensor) -> str:
    """Hash one finite contiguous CPU floating tensor with explicit framing."""

    if (
        type(tensor) is not torch.Tensor
        or tensor.device.type != "cpu"
        or tensor.dtype not in (torch.float32, torch.float64)
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("SSOR tensor digest authority differs")
    payload = bytearray(b"sfora-ssor-float-tensor-v1\0")
    encoded_role = role.encode("utf-8")
    payload.extend(len(encoded_role).to_bytes(8, "big"))
    payload.extend(encoded_role)
    payload.extend(tensor.ndim.to_bytes(8, "big"))
    for dimension in tensor.shape:
        payload.extend(dimension.to_bytes(8, "big"))
    if tensor.dtype == torch.float32:
        payload.extend(b"f32le\0")
        payload.extend(tensor.numpy().astype("<f4", copy=False).tobytes())
    else:
        payload.extend(b"f64le\0")
        payload.extend(tensor.numpy().astype("<f8", copy=False).tobytes())
    return hashlib.sha256(payload).hexdigest()


def ssor_deployment_head_artifact_bytes(
    control_head_weight: torch.Tensor,
    deployment_projector: SSORProjectorEvidence,
    *,
    beta: float,
) -> bytes:
    """Encode the exact deterministic NPY artifact for one deployed SSOR head."""

    deployed = compose_restored_head(control_head_weight, deployment_projector, beta=beta)
    stream = io.BytesIO()
    np.save(stream, deployed.numpy().astype("<f4", copy=False), allow_pickle=False)
    return stream.getvalue()


@dataclass(frozen=True, slots=True)
class SSORProjectorEvidence:
    """Deterministic seen-class projector and descriptive occupancy evidence."""

    fit_labels: tuple[int, ...]
    rank: int
    dimensions: int
    relative_cutoff: float
    projector: torch.Tensor
    mean_span_energy: float
    mean_complement_energy: float
    class_span_energy: tuple[float, ...]
    class_complement_energy: tuple[float, ...]
    orthogonal_probe_residual: float


@dataclass(frozen=True, slots=True)
class SSORInnerFoldEvidence:
    """One fit-only inner selection fold."""

    ordinal: int
    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    projector_rank: int
    mean_complement_energy: float
    query_count: int
    beta_hits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SSOROuterFoldEvidence:
    """One untouched outer optimization fold and its nested beta evidence."""

    ordinal: int
    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    projector_rank: int
    mean_complement_energy: float
    selected_beta: float
    query_count: int
    identity_hits: int
    scalar_identity_hits: int
    ssor_hits: int
    scalar_ssor_hits: int
    all_beta_hits: tuple[int, ...]
    inner_fold_schedule_sha256: str
    inner_folds: tuple[SSORInnerFoldEvidence, ...]


@dataclass(frozen=True, slots=True)
class SSORDiagnosticEvidence:
    """Primitive nested-fold evidence and reconstructed SSOR gate decision."""

    beta_grid: tuple[float, ...]
    fold_schedule_sha256: str
    folds: tuple[SSOROuterFoldEvidence, ...]
    selected_betas: tuple[float, ...]
    deployment_beta: float | None
    consensus_count: int
    deployment_projector_rank: int
    deployment_mean_complement_energy: float
    query_count: int
    identity_hits: int
    identity_errors: int
    materiality_eligible: bool
    ssor_hits: int
    identity_recall_ppm: int
    ssor_recall_ppm: int
    delta_ppm: int
    fold_wins: int
    minimum_fold_delta_ppm: int
    valid: bool
    passed: bool


def _validated_beta(beta: float) -> float:
    if type(beta) is not float or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("SSOR beta authority differs")
    return beta


def _validated_projector(evidence: SSORProjectorEvidence) -> torch.Tensor:
    if type(evidence) is not SSORProjectorEvidence:
        raise ValueError("SSOR projector authority differs")
    projector = evidence.projector
    if (
        type(projector) is not torch.Tensor
        or projector.device.type != "cpu"
        or projector.dtype != torch.float64
        or projector.shape != (evidence.dimensions, evidence.dimensions)
        or not projector.is_contiguous()
        or not bool(torch.isfinite(projector).all())
    ):
        raise ValueError("SSOR projector authority differs")
    return projector


def _validated_unit_descriptors(descriptors: torch.Tensor) -> torch.Tensor:
    if (
        type(descriptors) is not torch.Tensor
        or descriptors.device.type != "cpu"
        or descriptors.dtype != torch.float32
        or descriptors.ndim != 2
        or descriptors.shape[0] < 2
        or descriptors.shape[1] < 2
        or not descriptors.is_contiguous()
        or not bool(torch.isfinite(descriptors).all())
    ):
        raise ValueError("SSOR descriptor authority differs")
    norms = torch.linalg.vector_norm(descriptors.double(), dim=1)
    if not bool(torch.all(torch.abs(norms - 1.0) <= _UNIT_NORM_TOLERANCE)):
        raise ValueError("SSOR descriptor authority differs")
    return descriptors.double()


def seen_class_projector(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    *,
    fit_labels: tuple[int, ...],
) -> SSORProjectorEvidence:
    """Construct the exact uncentered fit-class mean-span projector."""

    normalized = _validated_unit_descriptors(descriptors)
    if (
        type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (descriptors.shape[0],)
        or not labels.is_contiguous()
    ):
        raise ValueError("SSOR label authority differs")
    if (
        type(fit_labels) is not tuple
        or len(fit_labels) < 2
        or any(type(label) is not int or label < 0 for label in fit_labels)
        or fit_labels != tuple(sorted(set(fit_labels)))
    ):
        raise ValueError("SSOR fit-label authority differs")
    observed = set(int(value) for value in torch.unique(labels).tolist())
    if not set(fit_labels).issubset(observed):
        raise ValueError("SSOR fit-label authority differs")
    if len(fit_labels) > descriptors.shape[1]:
        raise ValueError("SSOR rank authority differs")

    class_means: list[torch.Tensor] = []
    for label in fit_labels:
        members = normalized[labels == label]
        if members.shape[0] < 2:
            raise ValueError("SSOR fit-label authority differs")
        mean = members.mean(dim=0)
        norm = torch.linalg.vector_norm(mean)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise ValueError("SSOR class-mean authority differs")
        class_means.append(mean / norm)
    mean_matrix = torch.stack(class_means)
    _left, singular_values, right = torch.linalg.svd(mean_matrix, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) <= 0.0:
        raise ValueError("SSOR rank authority differs")
    cutoff = float(singular_values[0]) * _SVD_RELATIVE_CUTOFF
    rank = int((singular_values > cutoff).sum())
    expected_rank = len(fit_labels)
    if rank != expected_rank:
        raise ValueError("SSOR rank authority differs")
    retained = right[:rank]
    projector = (retained.T @ retained).contiguous()
    if (
        not torch.equal(projector, projector.T)
        or not torch.allclose(
            projector @ projector,
            projector,
            atol=_PROJECTOR_TOLERANCE,
            rtol=0.0,
        )
        or abs(float(torch.trace(projector)) - expected_rank) > _PROJECTOR_TOLERANCE
        or int(torch.linalg.matrix_rank(projector)) != expected_rank
    ):
        raise ValueError("SSOR projector invariant differs")
    eigenvalues = torch.linalg.eigvalsh(projector)
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or float(eigenvalues.min()) < -_PROJECTOR_TOLERANCE
        or float(eigenvalues.max()) > 1.0 + _PROJECTOR_TOLERANCE
    ):
        raise ValueError("SSOR projector eigenvalue authority differs")
    if not torch.allclose(mean_matrix @ projector, mean_matrix, atol=1e-10, rtol=0.0):
        raise ValueError("SSOR projector mean action differs")
    probe_residual = None
    for dimension in range(descriptors.shape[1]):
        basis = torch.zeros(descriptors.shape[1], dtype=torch.float64)
        basis[dimension] = 1.0
        candidate = basis - basis @ projector
        candidate_norm = torch.linalg.vector_norm(candidate)
        if float(candidate_norm) > _PROJECTOR_TOLERANCE:
            probe = candidate / candidate_norm
            probe_residual = float(torch.linalg.vector_norm(probe @ projector))
            break
    if probe_residual is None or probe_residual > _PROJECTOR_TOLERANCE:
        raise ValueError("SSOR projector orthogonal probe differs")

    fit_mask = torch.zeros(labels.shape, dtype=torch.bool)
    for label in fit_labels:
        fit_mask |= labels == label
    fit_descriptors = normalized[fit_mask]
    span = fit_descriptors @ projector
    complement = fit_descriptors - span
    span_energy = torch.sum(span * span, dim=1)
    complement_energy = torch.sum(complement * complement, dim=1)
    class_span: list[float] = []
    class_complement: list[float] = []
    fit_row_labels = labels[fit_mask]
    for label in fit_labels:
        class_span.append(float(torch.median(span_energy[fit_row_labels == label])))
        class_complement.append(float(torch.median(complement_energy[fit_row_labels == label])))

    return SSORProjectorEvidence(
        fit_labels=fit_labels,
        rank=rank,
        dimensions=descriptors.shape[1],
        relative_cutoff=_SVD_RELATIVE_CUTOFF,
        projector=projector,
        mean_span_energy=float(span_energy.mean()),
        mean_complement_energy=float(complement_energy.mean()),
        class_span_energy=tuple(class_span),
        class_complement_energy=tuple(class_complement),
        orthogonal_probe_residual=probe_residual,
    )


def restore_descriptors(
    descriptors: torch.Tensor,
    evidence: SSORProjectorEvidence,
    *,
    beta: float,
) -> torch.Tensor:
    """Apply the SSOR linear restoration and return unit float64 descriptors."""

    beta = _validated_beta(beta)
    normalized = _validated_unit_descriptors(descriptors)
    projector = _validated_projector(evidence)
    if normalized.shape[1] != evidence.dimensions:
        raise ValueError("SSOR descriptor authority differs")
    span = normalized @ projector
    restored = span + beta * (normalized - span)
    norms = torch.linalg.vector_norm(restored, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool(torch.any(norms <= 0.0)):
        raise ValueError("SSOR restored descriptor authority differs")
    return (restored / norms).contiguous()


def compose_restored_head(
    head_weight: torch.Tensor,
    evidence: SSORProjectorEvidence,
    *,
    beta: float,
) -> torch.Tensor:
    """Fold SSOR into a bias-free ``[output,input]`` linear-head weight."""

    beta = _validated_beta(beta)
    projector = _validated_projector(evidence)
    if (
        type(head_weight) is not torch.Tensor
        or head_weight.device.type != "cpu"
        or head_weight.dtype != torch.float32
        or head_weight.ndim != 2
        or head_weight.shape[0] != evidence.dimensions
        or not head_weight.is_contiguous()
        or not bool(torch.isfinite(head_weight).all())
    ):
        raise ValueError("SSOR head authority differs")
    identity = torch.eye(evidence.dimensions, dtype=torch.float64)
    transform = projector + beta * (identity - projector)
    composed = transform @ head_weight.double()
    if not bool(torch.isfinite(composed).all()):
        raise ValueError("SSOR composed head authority differs")
    return composed.float().contiguous()


def ssor_recall_at_one_hits(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    scalar: bool,
    query_mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Compute exact leave-one-out cosine hits with lowest-row ties."""

    if (
        type(embeddings) is not torch.Tensor
        or embeddings.device.type != "cpu"
        or embeddings.ndim != 2
        or embeddings.shape[0] < 2
        or not embeddings.is_floating_point()
        or not bool(torch.isfinite(embeddings).all())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or type(scalar) is not bool
    ):
        raise ValueError("SSOR retrieval authority differs")
    if query_mask is None:
        query_rows = torch.arange(embeddings.shape[0], dtype=torch.int64)
    elif (
        type(query_mask) is not torch.Tensor
        or query_mask.device.type != "cpu"
        or query_mask.dtype != torch.bool
        or query_mask.shape != (embeddings.shape[0],)
        or not query_mask.is_contiguous()
        or not bool(query_mask.any())
    ):
        raise ValueError("SSOR retrieval query authority differs")
    else:
        query_rows = torch.nonzero(query_mask, as_tuple=False).flatten()
    normalized = embeddings.double()
    norms = torch.linalg.vector_norm(normalized, dim=1, keepdim=True)
    if bool(torch.any(norms <= 0.0)):
        raise ValueError("SSOR retrieval norm differs")
    normalized = normalized / norms
    if not scalar:
        similarities = normalized[query_rows] @ normalized.T
        similarities[torch.arange(query_rows.numel()), query_rows] = -torch.inf
        nearest = torch.argmax(similarities, dim=1)
        return int((labels[nearest] == labels[query_rows]).sum()), query_rows.numel()

    nearest_rows: list[torch.Tensor] = []
    for query_start in range(0, query_rows.numel(), 32):
        block_rows = query_rows[query_start : query_start + 32]
        query_block = normalized[block_rows]
        best_scores = torch.full((block_rows.numel(),), -torch.inf, dtype=torch.float64)
        best_rows = torch.full((block_rows.numel(),), -1, dtype=torch.int64)
        for gallery_start in range(0, normalized.shape[0], 128):
            gallery_end = min(gallery_start + 128, normalized.shape[0])
            gallery = normalized[gallery_start:gallery_end]
            scores = torch.sum(query_block[:, None, :] * gallery[None, :, :], dim=2)
            local_self = block_rows - gallery_start
            contained = (local_self >= 0) & (local_self < gallery.shape[0])
            scores[contained, local_self[contained]] = -torch.inf
            block_scores, block_indexes = torch.max(scores, dim=1)
            improved = block_scores > best_scores
            best_scores[improved] = block_scores[improved]
            best_rows[improved] = block_indexes[improved] + gallery_start
        if bool(torch.any(best_rows < 0)):
            raise ValueError("SSOR scalar replay authority differs")
        nearest_rows.append(best_rows)
    nearest = torch.cat(nearest_rows)
    return int((labels[nearest] == labels[query_rows]).sum()), query_rows.numel()


def _derived_inner_folds(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    ordered_example_ids: tuple[str, ...],
    split_authority: FeatureSplitAuthority,
    *,
    eligible_labels: tuple[int, ...],
    fold_count: int = 3,
) -> tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], str]:
    """Reuse the authenticated SFQ scheduler on one bijectively relabeled subset."""

    local_to_global = tuple(sorted(eligible_labels))
    if len(local_to_global) < 2 * fold_count:
        raise ValueError("SSOR nested fold class authority differs")
    subset_mask = _label_mask(labels, eligible_labels)
    subset_rows = torch.nonzero(subset_mask, as_tuple=False).flatten()
    subset_descriptors = descriptors[subset_rows].contiguous()
    global_to_local = {label: ordinal for ordinal, label in enumerate(local_to_global)}
    subset_labels = torch.tensor(
        [global_to_local[int(labels[row])] for row in subset_rows],
        dtype=torch.int64,
    )
    subset_ids = tuple(ordered_example_ids[int(row)] for row in subset_rows)
    derived_authority = build_feature_split_authority(
        source_manifest_sha256=split_authority.source_manifest_sha256,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=subset_ids,
        features=subset_descriptors,
    )
    schedule = build_sfq_fold_schedule(
        subset_descriptors,
        subset_labels,
        derived_authority,
        fold_count=fold_count,
    )
    partitions = tuple(
        (
            tuple(local_to_global[label] for label in fold.fit_labels),
            tuple(local_to_global[label] for label in fold.validation_labels),
        )
        for fold in schedule.folds
    )
    return partitions, schedule.sha256


def _label_mask(labels: torch.Tensor, selected: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(labels.shape, dtype=torch.bool)
    for label in selected:
        mask |= labels == label
    return mask


def run_ssor_nested_diagnostic(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    *,
    ordered_example_ids: tuple[str, ...],
    split_authority: FeatureSplitAuthority,
) -> SSORDiagnosticEvidence:
    """Select restoration strength without observing each outer validation fold."""

    _validated_unit_descriptors(descriptors)
    if (
        type(split_authority) is not FeatureSplitAuthority
        or type(ordered_example_ids) is not tuple
        or len(ordered_example_ids) != descriptors.shape[0]
        or len(set(ordered_example_ids)) != len(ordered_example_ids)
        or any(type(value) is not str or not value for value in ordered_example_ids)
        or _ordered_example_ids_sha256(ordered_example_ids)
        != split_authority.ordered_example_ids_sha256
    ):
        raise ValueError("SSOR split authority differs")
    split_authority.validated(features=descriptors)
    outer_schedule = build_sfq_fold_schedule(
        descriptors,
        labels,
        split_authority,
        fold_count=4,
    )
    outer_evidence: list[SSOROuterFoldEvidence] = []
    for outer in outer_schedule.folds:
        outer_projector = seen_class_projector(
            descriptors,
            labels,
            fit_labels=outer.fit_labels,
        )
        inner_partitions, inner_schedule_sha256 = _derived_inner_folds(
            descriptors,
            labels,
            ordered_example_ids,
            split_authority,
            eligible_labels=outer.fit_labels,
        )
        outer_fit_mask = _label_mask(labels, outer.fit_labels)
        outer_fit_descriptors = descriptors[outer_fit_mask].contiguous()
        outer_fit_labels = labels[outer_fit_mask].contiguous()
        inner_evidence: list[SSORInnerFoldEvidence] = []
        for inner_ordinal, (inner_fit, inner_validation) in enumerate(inner_partitions):
            projector = seen_class_projector(
                descriptors,
                labels,
                fit_labels=inner_fit,
            )
            validation_mask = _label_mask(outer_fit_labels, inner_validation).contiguous()
            beta_hits = []
            query_count = int(validation_mask.sum())
            for beta in SSOR_BETA_GRID:
                restored = restore_descriptors(
                    outer_fit_descriptors,
                    projector,
                    beta=beta,
                )
                hits, queries = ssor_recall_at_one_hits(
                    restored,
                    outer_fit_labels,
                    scalar=False,
                    query_mask=validation_mask,
                )
                if queries != query_count:
                    raise ValueError("SSOR inner query authority differs")
                beta_hits.append(hits)
            inner_evidence.append(
                SSORInnerFoldEvidence(
                    ordinal=inner_ordinal,
                    fit_labels=inner_fit,
                    validation_labels=inner_validation,
                    projector_rank=projector.rank,
                    mean_complement_energy=projector.mean_complement_energy,
                    query_count=query_count,
                    beta_hits=tuple(beta_hits),
                )
            )
        aggregate_hits = tuple(
            sum(inner.beta_hits[index] for inner in inner_evidence)
            for index in range(len(SSOR_BETA_GRID))
        )
        selected_index = min(
            range(len(SSOR_BETA_GRID)),
            key=lambda index: (-aggregate_hits[index], index),
        )
        selected_beta = SSOR_BETA_GRID[selected_index]
        outer_mask = _label_mask(labels, outer.validation_labels).contiguous()
        all_beta_hits = []
        query_count = int(outer_mask.sum())
        for beta in SSOR_BETA_GRID:
            restored = restore_descriptors(descriptors, outer_projector, beta=beta)
            hits, queries = ssor_recall_at_one_hits(
                restored,
                labels,
                scalar=False,
                query_mask=outer_mask,
            )
            if queries != query_count:
                raise ValueError("SSOR outer query authority differs")
            all_beta_hits.append(hits)
        identity_hits = all_beta_hits[0]
        ssor_hits = all_beta_hits[selected_index]
        outer_evidence.append(
            SSOROuterFoldEvidence(
                ordinal=outer.ordinal,
                fit_labels=outer.fit_labels,
                validation_labels=outer.validation_labels,
                projector_rank=outer_projector.rank,
                mean_complement_energy=outer_projector.mean_complement_energy,
                selected_beta=selected_beta,
                query_count=query_count,
                identity_hits=identity_hits,
                scalar_identity_hits=identity_hits,
                ssor_hits=ssor_hits,
                scalar_ssor_hits=ssor_hits,
                all_beta_hits=tuple(all_beta_hits),
                inner_fold_schedule_sha256=inner_schedule_sha256,
                inner_folds=tuple(inner_evidence),
            )
        )

    preliminary_folds = tuple(outer_evidence)
    selected_betas = tuple(fold.selected_beta for fold in preliminary_folds)
    beta_counts = {beta: selected_betas.count(beta) for beta in SSOR_BETA_GRID}
    deployment_beta = next((beta for beta in SSOR_BETA_GRID if beta_counts[beta] >= 3), None)
    consensus_count = 0 if deployment_beta is None else beta_counts[deployment_beta]
    deployed_beta = 1.0 if deployment_beta is None else deployment_beta
    deployed_index = SSOR_BETA_GRID.index(deployed_beta)
    replayed_folds: list[SSOROuterFoldEvidence] = []
    for fold, outer in zip(preliminary_folds, outer_schedule.folds, strict=True):
        projector = seen_class_projector(descriptors, labels, fit_labels=outer.fit_labels)
        restored = restore_descriptors(descriptors, projector, beta=deployed_beta)
        outer_mask = _label_mask(labels, outer.validation_labels).contiguous()
        scalar_hits, scalar_queries = ssor_recall_at_one_hits(
            restored,
            labels,
            scalar=True,
            query_mask=outer_mask,
        )
        scalar_identity_hits, identity_queries = ssor_recall_at_one_hits(
            descriptors,
            labels,
            scalar=True,
            query_mask=outer_mask,
        )
        if scalar_queries != fold.query_count or identity_queries != fold.query_count:
            raise ValueError("SSOR deployed query authority differs")
        replayed_folds.append(
            replace(
                fold,
                scalar_identity_hits=scalar_identity_hits,
                ssor_hits=fold.all_beta_hits[deployed_index],
                scalar_ssor_hits=scalar_hits,
            )
        )
    folds = tuple(replayed_folds)
    query_count = sum(fold.query_count for fold in folds)
    identity_hits = sum(fold.identity_hits for fold in folds)
    ssor_hits = sum(fold.ssor_hits for fold in folds)
    identity_recall_ppm = identity_hits * 1_000_000 // query_count
    ssor_recall_ppm = ssor_hits * 1_000_000 // query_count
    fold_deltas = tuple(
        fold.ssor_hits * 1_000_000 // fold.query_count
        - fold.identity_hits * 1_000_000 // fold.query_count
        for fold in folds
    )
    fold_wins = sum(fold.ssor_hits > fold.identity_hits for fold in folds)
    valid = all(
        fold.scalar_identity_hits == fold.identity_hits and fold.scalar_ssor_hits == fold.ssor_hits
        for fold in folds
    )
    delta_ppm = ssor_recall_ppm - identity_recall_ppm
    minimum_fold_delta_ppm = min(fold_deltas)
    identity_errors = query_count - identity_hits
    materiality_eligible = identity_errors >= 40
    deployment_projector = seen_class_projector(
        descriptors,
        labels,
        fit_labels=tuple(sorted(int(value) for value in torch.unique(labels).tolist())),
    )
    passed = (
        valid
        and materiality_eligible
        and deployment_beta is not None
        and deployment_beta != 1.0
        and delta_ppm >= 2_000
        and fold_wins >= 3
        and minimum_fold_delta_ppm >= -10_000
    )
    return SSORDiagnosticEvidence(
        beta_grid=SSOR_BETA_GRID,
        fold_schedule_sha256=outer_schedule.sha256,
        folds=folds,
        selected_betas=selected_betas,
        deployment_beta=deployment_beta,
        consensus_count=consensus_count,
        deployment_projector_rank=deployment_projector.rank,
        deployment_mean_complement_energy=deployment_projector.mean_complement_energy,
        query_count=query_count,
        identity_hits=identity_hits,
        identity_errors=identity_errors,
        materiality_eligible=materiality_eligible,
        ssor_hits=ssor_hits,
        identity_recall_ppm=identity_recall_ppm,
        ssor_recall_ppm=ssor_recall_ppm,
        delta_ppm=delta_ppm,
        fold_wins=fold_wins,
        minimum_fold_delta_ppm=minimum_fold_delta_ppm,
        valid=valid,
        passed=passed,
    )


def _hex_digest(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SSOR result digest authority differs")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _inner_mapping(inner: SSORInnerFoldEvidence) -> dict[str, object]:
    return {
        "ordinal": inner.ordinal,
        "fit_labels": list(inner.fit_labels),
        "validation_labels": list(inner.validation_labels),
        "projector_rank": inner.projector_rank,
        "mean_complement_energy": inner.mean_complement_energy,
        "query_count": inner.query_count,
        "beta_hits": list(inner.beta_hits),
    }


def _outer_mapping(outer: SSOROuterFoldEvidence) -> dict[str, object]:
    return {
        "ordinal": outer.ordinal,
        "fit_labels": list(outer.fit_labels),
        "validation_labels": list(outer.validation_labels),
        "projector_rank": outer.projector_rank,
        "mean_complement_energy": outer.mean_complement_energy,
        "selected_beta": outer.selected_beta,
        "query_count": outer.query_count,
        "identity_hits": outer.identity_hits,
        "scalar_identity_hits": outer.scalar_identity_hits,
        "ssor_hits": outer.ssor_hits,
        "scalar_ssor_hits": outer.scalar_ssor_hits,
        "all_beta_hits": list(outer.all_beta_hits),
        "inner_fold_schedule_sha256": outer.inner_fold_schedule_sha256,
        "inner_folds": [_inner_mapping(inner) for inner in outer.inner_folds],
    }


def canonical_ssor_result_bytes(
    evidence: SSORDiagnosticEvidence,
    *,
    source_manifest_sha256: str,
    feature_cache_manifest_sha256: str,
    ordered_example_ids_sha256: str,
    feature_matrix_sha256: str,
    label_vector_sha256: str,
    control_head_weight: torch.Tensor,
    deployment_projector: SSORProjectorEvidence | None,
    deployment_head_artifact: bytes | None,
) -> bytes:
    """Serialize primitive SSOR evidence as a self-authenticating canonical result."""

    if type(evidence) is not SSORDiagnosticEvidence:
        raise ValueError("SSOR result evidence differs")
    control_head_sha256 = ssor_float_tensor_sha256("control-head", control_head_weight)
    deployment_projector_sha256 = None
    deployment_head_sha256 = None
    deployment_head_file_sha256 = None
    if deployment_projector is not None:
        if evidence.deployment_beta is None:
            raise ValueError("SSOR result deployment identity differs")
        projector = _validated_projector(deployment_projector)
        if (
            deployment_projector.rank != evidence.deployment_projector_rank
            or deployment_projector.mean_complement_energy
            != evidence.deployment_mean_complement_energy
        ):
            raise ValueError("SSOR result deployment projector differs")
        deployment_projector_sha256 = ssor_float_tensor_sha256("deployment-projector", projector)
        deployed = compose_restored_head(
            control_head_weight, deployment_projector, beta=evidence.deployment_beta
        )
        deployment_head_sha256 = ssor_float_tensor_sha256("deployment-head", deployed)
        expected_artifact = ssor_deployment_head_artifact_bytes(
            control_head_weight,
            deployment_projector,
            beta=evidence.deployment_beta,
        )
        if (
            type(deployment_head_artifact) is not bytes
            or deployment_head_artifact != expected_artifact
        ):
            raise ValueError("SSOR result deployment artifact differs")
        deployment_head_file_sha256 = hashlib.sha256(deployment_head_artifact).hexdigest()
    elif deployment_head_artifact is not None:
        raise ValueError("SSOR result deployment artifact differs")
    identities = {
        "source_manifest_sha256": _hex_digest(source_manifest_sha256),
        "feature_cache_manifest_sha256": _hex_digest(feature_cache_manifest_sha256),
        "ordered_example_ids_sha256": _hex_digest(ordered_example_ids_sha256),
        "feature_matrix_sha256": _hex_digest(feature_matrix_sha256),
        "label_vector_sha256": _hex_digest(label_vector_sha256),
        "control_head_sha256": control_head_sha256,
        "deployment_projector_sha256": deployment_projector_sha256,
        "deployment_head_sha256": deployment_head_sha256,
        "deployment_head_file_sha256": deployment_head_file_sha256,
    }
    if (evidence.passed and identities["deployment_head_sha256"] is None) or (
        not evidence.passed and identities["deployment_head_sha256"] is not None
    ):
        raise ValueError("SSOR result deployment identity differs")
    payload: dict[str, object] = {
        "schema": "sfora-siglip-ssor-v1",
        "claim_eligible": False,
        "official_test_access": False,
        **identities,
        "beta_grid": list(evidence.beta_grid),
        "fold_schedule_sha256": evidence.fold_schedule_sha256,
        "folds": [_outer_mapping(fold) for fold in evidence.folds],
        "selected_betas": list(evidence.selected_betas),
        "deployment_beta": evidence.deployment_beta,
        "consensus_count": evidence.consensus_count,
        "deployment_projector_rank": evidence.deployment_projector_rank,
        "deployment_mean_complement_energy": evidence.deployment_mean_complement_energy,
        "query_count": evidence.query_count,
        "identity_hits": evidence.identity_hits,
        "identity_errors": evidence.identity_errors,
        "materiality_eligible": evidence.materiality_eligible,
        "ssor_hits": evidence.ssor_hits,
        "identity_recall_ppm": evidence.identity_recall_ppm,
        "ssor_recall_ppm": evidence.ssor_recall_ppm,
        "delta_ppm": evidence.delta_ppm,
        "fold_wins": evidence.fold_wins,
        "minimum_fold_delta_ppm": evidence.minimum_fold_delta_ppm,
        "valid": evidence.valid,
        "passed": evidence.passed,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    raw = _canonical_json_bytes(payload)
    validate_ssor_result_bytes(
        raw,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_feature_cache_manifest_sha256=feature_cache_manifest_sha256,
        expected_ordered_example_ids_sha256=ordered_example_ids_sha256,
        expected_feature_matrix_sha256=feature_matrix_sha256,
        expected_label_vector_sha256=label_vector_sha256,
        expected_control_head_sha256=control_head_sha256,
        expected_deployment_projector_sha256=deployment_projector_sha256,
        expected_deployment_head_sha256=deployment_head_sha256,
        expected_deployment_head_file_sha256=deployment_head_file_sha256,
    )
    return raw


def _exact_integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("SSOR result integer authority differs")
    return value


def _exact_signed_integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("SSOR result signed integer authority differs")
    return value


def _exact_float(value: object, *, minimum: float = 0.0) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise ValueError("SSOR result float authority differs")
    return value


def _integer_labels(value: object) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("SSOR result label authority differs")
    labels = tuple(value)
    if labels != tuple(sorted(set(labels))):
        raise ValueError("SSOR result label authority differs")
    return labels


def validate_ssor_result_bytes(
    raw: bytes,
    *,
    expected_source_manifest_sha256: str,
    expected_feature_cache_manifest_sha256: str,
    expected_ordered_example_ids_sha256: str,
    expected_feature_matrix_sha256: str,
    expected_label_vector_sha256: str,
    expected_control_head_sha256: str,
    expected_deployment_projector_sha256: str | None,
    expected_deployment_head_sha256: str | None,
    expected_deployment_head_file_sha256: str | None,
) -> dict[str, object]:
    """Independently reconstruct all derivable SSOR counts, selections, and gates."""

    try:
        value = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SSOR result JSON differs") from error
    if type(value) is not dict or _canonical_json_bytes(value) != raw:
        raise ValueError("SSOR result canonical bytes differ")
    expected_keys = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "source_manifest_sha256",
        "feature_cache_manifest_sha256",
        "ordered_example_ids_sha256",
        "feature_matrix_sha256",
        "label_vector_sha256",
        "control_head_sha256",
        "deployment_projector_sha256",
        "deployment_head_sha256",
        "deployment_head_file_sha256",
        "beta_grid",
        "fold_schedule_sha256",
        "folds",
        "selected_betas",
        "deployment_beta",
        "consensus_count",
        "deployment_projector_rank",
        "deployment_mean_complement_energy",
        "query_count",
        "identity_hits",
        "identity_errors",
        "materiality_eligible",
        "ssor_hits",
        "identity_recall_ppm",
        "ssor_recall_ppm",
        "delta_ppm",
        "fold_wins",
        "minimum_fold_delta_ppm",
        "valid",
        "passed",
        "result_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("SSOR result schema differs")
    if (
        value["schema"] != "sfora-siglip-ssor-v1"
        or type(value["claim_eligible"]) is not bool
        or value["claim_eligible"]
        or type(value["official_test_access"]) is not bool
        or value["official_test_access"]
    ):
        raise ValueError("SSOR result outer authority differs")
    identity_pairs = (
        ("source_manifest_sha256", expected_source_manifest_sha256),
        ("feature_cache_manifest_sha256", expected_feature_cache_manifest_sha256),
        ("ordered_example_ids_sha256", expected_ordered_example_ids_sha256),
        ("feature_matrix_sha256", expected_feature_matrix_sha256),
        ("label_vector_sha256", expected_label_vector_sha256),
        ("control_head_sha256", expected_control_head_sha256),
    )
    for key, expected in identity_pairs:
        if value[key] != _hex_digest(expected):
            raise ValueError("SSOR result identity differs")
    expected_deployment = _hex_digest(expected_deployment_head_sha256, optional=True)
    expected_projector = _hex_digest(expected_deployment_projector_sha256, optional=True)
    expected_deployment_file = _hex_digest(expected_deployment_head_file_sha256, optional=True)
    if (
        value["deployment_projector_sha256"] != expected_projector
        or value["deployment_head_sha256"] != expected_deployment
        or value["deployment_head_file_sha256"] != expected_deployment_file
    ):
        raise ValueError("SSOR result deployment identity differs")
    result_sha256 = _hex_digest(value["result_sha256"])
    unsigned = {key: item for key, item in value.items() if key != "result_sha256"}
    if result_sha256 != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError("SSOR result checksum differs")
    if (
        type(value["beta_grid"]) is not list
        or any(type(item) is not float for item in value["beta_grid"])
        or value["beta_grid"] != list(SSOR_BETA_GRID)
    ):
        raise ValueError("SSOR result beta grid differs")
    _hex_digest(value["fold_schedule_sha256"])
    folds_value = value["folds"]
    if type(folds_value) is not list or len(folds_value) != 4:
        raise ValueError("SSOR result fold count differs")

    fold_keys = {
        "ordinal",
        "fit_labels",
        "validation_labels",
        "projector_rank",
        "mean_complement_energy",
        "selected_beta",
        "query_count",
        "identity_hits",
        "scalar_identity_hits",
        "ssor_hits",
        "scalar_ssor_hits",
        "all_beta_hits",
        "inner_fold_schedule_sha256",
        "inner_folds",
    }
    inner_keys = {
        "ordinal",
        "fit_labels",
        "validation_labels",
        "projector_rank",
        "mean_complement_energy",
        "query_count",
        "beta_hits",
    }
    parsed_folds: list[dict[str, object]] = []
    validation_coverage: list[int] = []
    selected_betas: list[float] = []
    for ordinal, fold in enumerate(folds_value):
        if (
            type(fold) is not dict
            or set(fold) != fold_keys
            or _exact_integer(fold["ordinal"]) != ordinal
        ):
            raise ValueError("SSOR result fold schema differs")
        fit_labels = _integer_labels(fold["fit_labels"])
        validation_labels = _integer_labels(fold["validation_labels"])
        if len(fit_labels) < 2 or not validation_labels or set(fit_labels) & set(validation_labels):
            raise ValueError("SSOR result fold partition differs")
        validation_coverage.extend(validation_labels)
        if _exact_integer(fold["projector_rank"]) != len(fit_labels):
            raise ValueError("SSOR result fold projector differs")
        _exact_float(fold["mean_complement_energy"])
        selected_beta = _exact_float(fold["selected_beta"])
        if selected_beta not in SSOR_BETA_GRID:
            raise ValueError("SSOR result selected beta differs")
        selected_betas.append(selected_beta)
        query_count = _exact_integer(fold["query_count"], minimum=1)
        all_beta_hits_value = fold["all_beta_hits"]
        if type(all_beta_hits_value) is not list or len(all_beta_hits_value) != len(SSOR_BETA_GRID):
            raise ValueError("SSOR result outer beta evidence differs")
        all_beta_hits = tuple(_exact_integer(item) for item in all_beta_hits_value)
        if any(item > query_count for item in all_beta_hits):
            raise ValueError("SSOR result outer hit authority differs")
        if _exact_integer(fold["identity_hits"]) != all_beta_hits[0]:
            raise ValueError("SSOR result identity fold differs")
        _exact_integer(fold["scalar_identity_hits"])
        _exact_integer(fold["ssor_hits"])
        _exact_integer(fold["scalar_ssor_hits"])
        _hex_digest(fold["inner_fold_schedule_sha256"])
        inner_values = fold["inner_folds"]
        if type(inner_values) is not list or len(inner_values) != 3:
            raise ValueError("SSOR result inner fold count differs")
        inner_coverage: list[int] = []
        aggregate = [0] * len(SSOR_BETA_GRID)
        for inner_ordinal, inner in enumerate(inner_values):
            if (
                type(inner) is not dict
                or set(inner) != inner_keys
                or _exact_integer(inner["ordinal"]) != inner_ordinal
            ):
                raise ValueError("SSOR result inner schema differs")
            inner_fit = _integer_labels(inner["fit_labels"])
            inner_validation = _integer_labels(inner["validation_labels"])
            if (
                len(inner_fit) < 2
                or not inner_validation
                or set(inner_fit) & set(inner_validation)
                or set(inner_fit) | set(inner_validation) != set(fit_labels)
            ):
                raise ValueError("SSOR result inner partition differs")
            inner_coverage.extend(inner_validation)
            if _exact_integer(inner["projector_rank"]) != len(inner_fit):
                raise ValueError("SSOR result inner projector differs")
            _exact_float(inner["mean_complement_energy"])
            inner_queries = _exact_integer(inner["query_count"], minimum=1)
            beta_hits_value = inner["beta_hits"]
            if type(beta_hits_value) is not list or len(beta_hits_value) != len(SSOR_BETA_GRID):
                raise ValueError("SSOR result inner beta evidence differs")
            for index, item in enumerate(beta_hits_value):
                hits = _exact_integer(item)
                if hits > inner_queries:
                    raise ValueError("SSOR result inner hit authority differs")
                aggregate[index] += hits
        if sorted(inner_coverage) != list(fit_labels):
            raise ValueError("SSOR result inner coverage differs")
        selected_index = min(
            range(len(SSOR_BETA_GRID)), key=lambda index: (-aggregate[index], index)
        )
        if selected_beta != SSOR_BETA_GRID[selected_index]:
            raise ValueError("SSOR result inner selection differs")
        parsed_folds.append(
            {
                "fit_labels": fit_labels,
                "validation_labels": validation_labels,
                "query_count": query_count,
                "all_beta_hits": all_beta_hits,
                "identity_hits": fold["identity_hits"],
                "scalar_identity_hits": fold["scalar_identity_hits"],
                "ssor_hits": fold["ssor_hits"],
                "scalar_ssor_hits": fold["scalar_ssor_hits"],
            }
        )
    all_labels = tuple(sorted(validation_coverage))
    if all_labels != tuple(range(len(all_labels))):
        raise ValueError("SSOR result outer coverage differs")
    for fold in parsed_folds:
        if set(fold["fit_labels"]) != set(all_labels).difference(fold["validation_labels"]):
            raise ValueError("SSOR result outer partition differs")
    if (
        type(value["selected_betas"]) is not list
        or any(type(item) is not float for item in value["selected_betas"])
        or value["selected_betas"] != selected_betas
    ):
        raise ValueError("SSOR result selected beta list differs")
    beta_counts = {beta: selected_betas.count(beta) for beta in SSOR_BETA_GRID}
    deployment_beta = next((beta for beta in SSOR_BETA_GRID if beta_counts[beta] >= 3), None)
    consensus_count = 0 if deployment_beta is None else beta_counts[deployment_beta]
    if (
        (value["deployment_beta"] is not None and type(value["deployment_beta"]) is not float)
        or value["deployment_beta"] != deployment_beta
        or _exact_integer(value["consensus_count"]) != consensus_count
    ):
        raise ValueError("SSOR result consensus differs")
    if _exact_integer(value["deployment_projector_rank"]) != len(all_labels):
        raise ValueError("SSOR result deployment projector differs")
    _exact_float(value["deployment_mean_complement_energy"])
    deployed_index = SSOR_BETA_GRID.index(1.0 if deployment_beta is None else deployment_beta)
    query_count = sum(int(fold["query_count"]) for fold in parsed_folds)
    identity_hits = sum(int(fold["identity_hits"]) for fold in parsed_folds)
    ssor_hits = sum(int(fold["all_beta_hits"][deployed_index]) for fold in parsed_folds)
    valid = all(
        fold["scalar_identity_hits"] == fold["identity_hits"]
        and fold["scalar_ssor_hits"] == fold["ssor_hits"]
        and fold["ssor_hits"] == fold["all_beta_hits"][deployed_index]
        for fold in parsed_folds
    )
    identity_errors = query_count - identity_hits
    materiality_eligible = identity_errors >= 40
    identity_recall_ppm = identity_hits * 1_000_000 // query_count
    ssor_recall_ppm = ssor_hits * 1_000_000 // query_count
    fold_deltas = tuple(
        int(fold["ssor_hits"]) * 1_000_000 // int(fold["query_count"])
        - int(fold["identity_hits"]) * 1_000_000 // int(fold["query_count"])
        for fold in parsed_folds
    )
    fold_wins = sum(int(fold["ssor_hits"]) > int(fold["identity_hits"]) for fold in parsed_folds)
    derived = {
        "query_count": query_count,
        "identity_hits": identity_hits,
        "identity_errors": identity_errors,
        "materiality_eligible": materiality_eligible,
        "ssor_hits": ssor_hits,
        "identity_recall_ppm": identity_recall_ppm,
        "ssor_recall_ppm": ssor_recall_ppm,
        "delta_ppm": ssor_recall_ppm - identity_recall_ppm,
        "fold_wins": fold_wins,
        "minimum_fold_delta_ppm": min(fold_deltas),
        "valid": valid,
    }
    for key, item in derived.items():
        if type(item) is bool:
            if type(value[key]) is not bool or value[key] is not item:
                raise ValueError("SSOR result aggregate differs")
        elif key in {"delta_ppm", "minimum_fold_delta_ppm"}:
            if _exact_signed_integer(value[key]) != item:
                raise ValueError("SSOR result aggregate differs")
        elif _exact_integer(value[key]) != item:
            raise ValueError("SSOR result aggregate differs")
    passed = (
        valid
        and materiality_eligible
        and deployment_beta is not None
        and deployment_beta != 1.0
        and derived["delta_ppm"] >= 2_000
        and fold_wins >= 3
        and derived["minimum_fold_delta_ppm"] >= -10_000
    )
    if type(value["passed"]) is not bool or value["passed"] != passed:
        raise ValueError("SSOR result pass gate differs")
    if (
        passed
        and (
            expected_deployment is None
            or expected_projector is None
            or expected_deployment_file is None
        )
    ) or (
        not passed
        and (
            expected_deployment is not None
            or expected_projector is not None
            or expected_deployment_file is not None
        )
    ):
        raise ValueError("SSOR result deployment eligibility differs")
    return value
