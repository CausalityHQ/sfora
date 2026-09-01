"""Optimization-only selection of an intermediate SigLIP vision readout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import FeatureSplitAuthority
from sfora.siglip_sfq import build_sfq_fold_schedule, sfq_label_vector_sha256


@dataclass(frozen=True, slots=True)
class IntermediateReadoutDepthEvidence:
    """Integer retrieval evidence for one one-based encoder depth."""

    depth: int
    descriptor_sha256: str
    query_count: int
    hits: int
    recall_ppm: int
    replay_hits: int
    fold_hits: tuple[int, ...]
    fold_query_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IntermediateReadoutResult:
    """Canonical optimization-only depth-selection result."""

    schema: str
    claim_eligible: bool
    official_test_access: bool
    source_manifest_sha256: str
    feature_manifest_sha256: str
    checkpoint_sha256: str
    ordered_example_ids_sha256: str
    source_feature_matrix_sha256: str
    label_vector_sha256: str
    fold_schedule_sha256: str
    expected_depth_count: int
    output_dimensions: int
    fold_count: int
    query_count: int
    context_fold_hits: tuple[int, ...]
    context_hits: int
    context_recall_ppm: int
    selected_depth: int
    selected_hits: int
    final_depth_hits: int
    selected_minus_final_ppm: int
    fold_wins: int
    replay_equal: bool
    passed: bool
    depths: tuple[IntermediateReadoutDepthEvidence, ...]


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("intermediate readout digest differs")
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("intermediate readout integer differs")
    return value


def _exact_mapping(value: object, keys: set[str], *, error: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(error)
    return cast(dict[str, object], value)


def _descriptor_sha256(descriptors: torch.Tensor) -> str:
    payload = bytearray(b"sfora-siglip-intermediate-readout-descriptors-v1\0")
    payload.extend(descriptors.shape[0].to_bytes(8, "big"))
    payload.extend(descriptors.shape[1].to_bytes(8, "big"))
    payload.extend(descriptors.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return hashlib.sha256(payload).hexdigest()


def _recall_hits(descriptors: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    normalized = F.normalize(descriptors.double(), dim=1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    nearest = torch.argmax(similarity, dim=1)
    return int((labels[nearest] == labels).sum()), labels.numel()


def _scalar_recall_hits(descriptors: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Independently replay leave-one-out cosine scoring with lowest-row ties."""

    normalized = F.normalize(descriptors.double(), dim=1)
    hits = 0
    for query in range(normalized.shape[0]):
        best_row = -1
        best_score = -float("inf")
        for candidate in range(normalized.shape[0]):
            if candidate == query:
                continue
            score = float(torch.dot(normalized[query], normalized[candidate]))
            if score > best_score:
                best_score = score
                best_row = candidate
        if best_row < 0:
            raise ValueError("intermediate scalar replay has no candidate")
        hits += int(labels[best_row] == labels[query])
    return hits, labels.numel()


def _depth_mapping(depth: IntermediateReadoutDepthEvidence) -> dict[str, object]:
    return {
        "depth": depth.depth,
        "descriptor_sha256": depth.descriptor_sha256,
        "query_count": depth.query_count,
        "hits": depth.hits,
        "recall_ppm": depth.recall_ppm,
        "replay_hits": depth.replay_hits,
        "fold_hits": list(depth.fold_hits),
        "fold_query_counts": list(depth.fold_query_counts),
    }


def _result_mapping(result: IntermediateReadoutResult) -> dict[str, object]:
    return {
        "schema": result.schema,
        "claim_eligible": result.claim_eligible,
        "official_test_access": result.official_test_access,
        "source_manifest_sha256": result.source_manifest_sha256,
        "feature_manifest_sha256": result.feature_manifest_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
        "ordered_example_ids_sha256": result.ordered_example_ids_sha256,
        "source_feature_matrix_sha256": result.source_feature_matrix_sha256,
        "label_vector_sha256": result.label_vector_sha256,
        "fold_schedule_sha256": result.fold_schedule_sha256,
        "expected_depth_count": result.expected_depth_count,
        "output_dimensions": result.output_dimensions,
        "fold_count": result.fold_count,
        "query_count": result.query_count,
        "context_fold_hits": list(result.context_fold_hits),
        "context_hits": result.context_hits,
        "context_recall_ppm": result.context_recall_ppm,
        "selected_depth": result.selected_depth,
        "selected_hits": result.selected_hits,
        "final_depth_hits": result.final_depth_hits,
        "selected_minus_final_ppm": result.selected_minus_final_ppm,
        "fold_wins": result.fold_wins,
        "replay_equal": result.replay_equal,
        "passed": result.passed,
        "depths": [_depth_mapping(depth) for depth in result.depths],
    }


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def score_intermediate_readout_depths(
    source_features: torch.Tensor,
    labels: torch.Tensor,
    descriptor_planes: tuple[torch.Tensor, ...],
    *,
    split_authority: FeatureSplitAuthority,
    checkpoint_sha256: str,
    feature_manifest_sha256: str,
    expected_depth_count: int,
    output_dimensions: int,
    fold_count: int = 4,
) -> bytes:
    """Score every registered depth on class-disjoint optimization folds."""

    if (
        type(source_features) is not torch.Tensor
        or source_features.ndim != 2
        or source_features.device.type != "cpu"
        or source_features.dtype != torch.float32
        or not bool(torch.isfinite(source_features).all())
        or type(labels) is not torch.Tensor
        or labels.shape != (source_features.shape[0],)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or not labels.is_contiguous()
        or type(descriptor_planes) is not tuple
        or type(expected_depth_count) is not int
        or expected_depth_count < 1
        or len(descriptor_planes) != expected_depth_count
        or type(output_dimensions) is not int
        or output_dimensions < 2
        or type(fold_count) is not int
        or fold_count != 4
    ):
        raise ValueError("intermediate descriptor authority differs")
    if type(split_authority) is not FeatureSplitAuthority:
        raise ValueError("intermediate split authority differs")
    split_authority.validated(features=source_features)
    checkpoint_sha256 = _hex_digest(checkpoint_sha256)
    feature_manifest_sha256 = _hex_digest(feature_manifest_sha256)

    for descriptors in descriptor_planes:
        if (
            type(descriptors) is not torch.Tensor
            or descriptors.shape != (source_features.shape[0], output_dimensions)
            or descriptors.device.type != "cpu"
            or descriptors.dtype != torch.float32
            or not descriptors.is_contiguous()
            or not bool(torch.isfinite(descriptors).all())
            or not torch.allclose(
                torch.linalg.vector_norm(descriptors, dim=1),
                torch.ones(descriptors.shape[0]),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        ):
            raise ValueError("intermediate descriptor authority differs")

    schedule = build_sfq_fold_schedule(
        source_features,
        labels,
        split_authority,
        fold_count=fold_count,
    )
    context_fold_hits = []
    context_fold_queries = []
    for fold in schedule.folds:
        mask = torch.isin(labels, torch.tensor(fold.validation_labels, dtype=torch.int64))
        hits, queries = _recall_hits(source_features[mask].contiguous(), labels[mask].contiguous())
        context_fold_hits.append(hits)
        context_fold_queries.append(queries)
    depth_evidence = []
    for ordinal, descriptors in enumerate(descriptor_planes, start=1):
        fold_hits = []
        fold_queries = []
        for fold in schedule.folds:
            mask = torch.isin(labels, torch.tensor(fold.validation_labels, dtype=torch.int64))
            hits, queries = _recall_hits(descriptors[mask].contiguous(), labels[mask].contiguous())
            fold_hits.append(hits)
            fold_queries.append(queries)
        hits = sum(fold_hits)
        queries = sum(fold_queries)
        replay_hits = sum(
            _scalar_recall_hits(
                descriptors[
                    torch.isin(
                        labels,
                        torch.tensor(fold.validation_labels, dtype=torch.int64),
                    )
                ].contiguous(),
                labels[
                    torch.isin(
                        labels,
                        torch.tensor(fold.validation_labels, dtype=torch.int64),
                    )
                ].contiguous(),
            )[0]
            for fold in schedule.folds
        )
        depth_evidence.append(
            IntermediateReadoutDepthEvidence(
                depth=ordinal,
                descriptor_sha256=_descriptor_sha256(descriptors),
                query_count=queries,
                hits=hits,
                recall_ppm=hits * 1_000_000 // queries,
                replay_hits=replay_hits,
                fold_hits=tuple(fold_hits),
                fold_query_counts=tuple(fold_queries),
            )
        )
    depths = tuple(depth_evidence)
    selected = min(depths, key=lambda depth: (-depth.hits, depth.depth))
    final = depths[-1]
    fold_wins = sum(
        selected_hits > final_hits
        for selected_hits, final_hits in zip(selected.fold_hits, final.fold_hits, strict=True)
    )
    selected_minus_final_ppm = (selected.hits - final.hits) * 1_000_000 // selected.query_count
    replay_equal = all(depth.hits == depth.replay_hits for depth in depths)
    if not replay_equal:
        raise ValueError("intermediate scalar replay differs")
    result = IntermediateReadoutResult(
        schema="sfora-siglip-intermediate-readout-v1",
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=split_authority.source_manifest_sha256,
        feature_manifest_sha256=feature_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        ordered_example_ids_sha256=split_authority.ordered_example_ids_sha256,
        source_feature_matrix_sha256=split_authority.feature_matrix_sha256,
        label_vector_sha256=sfq_label_vector_sha256(labels),
        fold_schedule_sha256=schedule.sha256,
        expected_depth_count=expected_depth_count,
        output_dimensions=output_dimensions,
        fold_count=fold_count,
        query_count=selected.query_count,
        context_fold_hits=tuple(context_fold_hits),
        context_hits=sum(context_fold_hits),
        context_recall_ppm=sum(context_fold_hits) * 1_000_000 // sum(context_fold_queries),
        selected_depth=selected.depth,
        selected_hits=selected.hits,
        final_depth_hits=final.hits,
        selected_minus_final_ppm=selected_minus_final_ppm,
        fold_wins=fold_wins,
        replay_equal=replay_equal,
        passed=(selected_minus_final_ppm >= 10_000 and fold_wins >= 3 and replay_equal),
        depths=depths,
    )
    raw = _canonical_bytes(_result_mapping(result))
    validate_intermediate_readout_result_bytes(raw)
    return raw


def _parse_depth(value: object, *, fold_count: int) -> IntermediateReadoutDepthEvidence:
    mapping = _exact_mapping(
        value,
        {
            "depth",
            "descriptor_sha256",
            "query_count",
            "hits",
            "recall_ppm",
            "replay_hits",
            "fold_hits",
            "fold_query_counts",
        },
        error="intermediate depth schema differs",
    )
    if type(mapping["fold_hits"]) is not list or type(mapping["fold_query_counts"]) is not list:
        raise ValueError("intermediate fold evidence differs")
    fold_hits = tuple(_integer(value) for value in cast(list[object], mapping["fold_hits"]))
    fold_queries = tuple(
        _integer(value, minimum=2) for value in cast(list[object], mapping["fold_query_counts"])
    )
    query_count = _integer(mapping["query_count"], minimum=2)
    hits = _integer(mapping["hits"])
    replay_hits = _integer(mapping["replay_hits"])
    recall_ppm = _integer(mapping["recall_ppm"])
    if (
        len(fold_hits) != fold_count
        or len(fold_queries) != fold_count
        or any(hit > queries for hit, queries in zip(fold_hits, fold_queries, strict=True))
        or query_count != sum(fold_queries)
        or hits != sum(fold_hits)
        or replay_hits != hits
        or recall_ppm != hits * 1_000_000 // query_count
    ):
        raise ValueError("intermediate depth relation differs")
    return IntermediateReadoutDepthEvidence(
        depth=_integer(mapping["depth"], minimum=1),
        descriptor_sha256=_hex_digest(mapping["descriptor_sha256"]),
        query_count=query_count,
        hits=hits,
        recall_ppm=recall_ppm,
        replay_hits=replay_hits,
        fold_hits=fold_hits,
        fold_query_counts=fold_queries,
    )


def validate_intermediate_readout_result_bytes(raw: bytes) -> IntermediateReadoutResult:
    """Parse canonical bytes and independently reconstruct selection and gates."""

    if type(raw) is not bytes:
        raise ValueError("intermediate result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("intermediate result is not JSON") from error
    if raw != _canonical_bytes(value):
        raise ValueError("intermediate result is not canonical")
    mapping = _exact_mapping(
        value,
        {
            "schema",
            "claim_eligible",
            "official_test_access",
            "source_manifest_sha256",
            "feature_manifest_sha256",
            "checkpoint_sha256",
            "ordered_example_ids_sha256",
            "source_feature_matrix_sha256",
            "label_vector_sha256",
            "fold_schedule_sha256",
            "expected_depth_count",
            "output_dimensions",
            "fold_count",
            "query_count",
            "context_fold_hits",
            "context_hits",
            "context_recall_ppm",
            "selected_depth",
            "selected_hits",
            "final_depth_hits",
            "selected_minus_final_ppm",
            "fold_wins",
            "replay_equal",
            "passed",
            "depths",
        },
        error="intermediate result schema differs",
    )
    expected_depth_count = _integer(mapping["expected_depth_count"], minimum=1)
    output_dimensions = _integer(mapping["output_dimensions"], minimum=2)
    fold_count = _integer(mapping["fold_count"], minimum=2)
    if fold_count != 4:
        raise ValueError("intermediate fold authority differs")
    if type(mapping["depths"]) is not list:
        raise ValueError("intermediate depth bundle differs")
    depths = tuple(
        _parse_depth(depth, fold_count=fold_count)
        for depth in cast(list[object], mapping["depths"])
    )
    if type(mapping["context_fold_hits"]) is not list:
        raise ValueError("intermediate context evidence differs")
    context_fold_hits = tuple(
        _integer(value) for value in cast(list[object], mapping["context_fold_hits"])
    )
    if (
        len(depths) != expected_depth_count
        or tuple(depth.depth for depth in depths) != tuple(range(1, expected_depth_count + 1))
        or any(depth.query_count != depths[0].query_count for depth in depths)
    ):
        raise ValueError("intermediate depth bundle differs")
    selected = min(depths, key=lambda depth: (-depth.hits, depth.depth))
    final = depths[-1]
    if len(context_fold_hits) != fold_count or any(
        hits > queries
        for hits, queries in zip(context_fold_hits, selected.fold_query_counts, strict=True)
    ):
        raise ValueError("intermediate context evidence differs")
    context_hits = sum(context_fold_hits)
    context_recall_ppm = context_hits * 1_000_000 // selected.query_count
    fold_wins = sum(
        left > right for left, right in zip(selected.fold_hits, final.fold_hits, strict=True)
    )
    delta = (selected.hits - final.hits) * 1_000_000 // selected.query_count
    replay_equal = all(depth.hits == depth.replay_hits for depth in depths)
    passed = delta >= 10_000 and fold_wins >= 3 and replay_equal
    if (
        mapping["schema"] != "sfora-siglip-intermediate-readout-v1"
        or mapping["claim_eligible"] is not False
        or mapping["official_test_access"] is not False
        or _integer(mapping["query_count"], minimum=2) != selected.query_count
        or _integer(mapping["context_hits"]) != context_hits
        or _integer(mapping["context_recall_ppm"]) != context_recall_ppm
        or _integer(mapping["selected_depth"], minimum=1) != selected.depth
        or _integer(mapping["selected_hits"]) != selected.hits
        or _integer(mapping["final_depth_hits"]) != final.hits
        or type(mapping["selected_minus_final_ppm"]) is not int
        or mapping["selected_minus_final_ppm"] != delta
        or _integer(mapping["fold_wins"]) != fold_wins
        or mapping["replay_equal"] is not replay_equal
        or mapping["passed"] is not passed
    ):
        raise ValueError("intermediate result relation differs")
    return IntermediateReadoutResult(
        schema=cast(str, mapping["schema"]),
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=_hex_digest(mapping["source_manifest_sha256"]),
        feature_manifest_sha256=_hex_digest(mapping["feature_manifest_sha256"]),
        checkpoint_sha256=_hex_digest(mapping["checkpoint_sha256"]),
        ordered_example_ids_sha256=_hex_digest(mapping["ordered_example_ids_sha256"]),
        source_feature_matrix_sha256=_hex_digest(mapping["source_feature_matrix_sha256"]),
        label_vector_sha256=_hex_digest(mapping["label_vector_sha256"]),
        fold_schedule_sha256=_hex_digest(mapping["fold_schedule_sha256"]),
        expected_depth_count=expected_depth_count,
        output_dimensions=output_dimensions,
        fold_count=fold_count,
        query_count=selected.query_count,
        context_fold_hits=context_fold_hits,
        context_hits=context_hits,
        context_recall_ppm=context_recall_ppm,
        selected_depth=selected.depth,
        selected_hits=selected.hits,
        final_depth_hits=final.hits,
        selected_minus_final_ppm=delta,
        fold_wins=fold_wins,
        replay_equal=replay_equal,
        passed=passed,
        depths=depths,
    )
