"""Leakage-safe frozen SigLIP manufacturer-band audit primitives."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SiglipAuditBand:
    """One exact contiguous Cars training-class band."""

    role: str
    first_label: int
    last_label: int


@dataclass(frozen=True)
class SiglipBandConfusion:
    """One ordered wrong-label nearest-neighbour count."""

    query_label: int
    nearest_label: int
    count: int


@dataclass(frozen=True)
class SiglipBandEvidence:
    """Strict and nameplate-variant-collapsed evidence for one band."""

    role: str
    first_label: int
    last_label: int
    query_count: int
    variant_class_count: int
    variant_query_count: int
    strict_hits: int
    strict_recall_ppm: int
    variant_hits: int
    variant_recall_ppm: int
    variant_rescued_errors: int
    confusions: tuple[SiglipBandConfusion, ...]


@dataclass(frozen=True)
class SiglipBandAuditEvidence:
    """All three band results plus exact query-weighted aggregates."""

    bands: tuple[SiglipBandEvidence, ...]
    query_count: int
    class_query_counts: tuple[int, ...]
    variant_class_count: int
    variant_query_count: int
    strict_hits: int
    strict_recall_ppm: int
    variant_hits: int
    variant_recall_ppm: int
    variant_rescued_errors: int


@dataclass(frozen=True)
class SiglipBandAuditAuthority:
    """Exact source, dataset, model, and tensor identities for one audit."""

    source_commit: str
    source_tree_digest: str
    dataset_revision: str
    dataset_examples_sha256: str
    ordered_example_ids_sha256: str
    descriptor_sha256: str
    label_vector_sha256: str
    class_names_sha256: str
    model_name: str
    model_revision: str
    readout: str
    split: str
    batch_size: int
    query_block: int
    cublas_workspace_config: str


SIGLIP_AUDIT_BANDS = (
    SiglipAuditBand("optimization", 0, 48),
    SiglipAuditBand("clean", 49, 81),
    SiglipAuditBand("burned", 82, 97),
)

SIGLIP_AUDIT_VARIANT_GROUPS = (
    (2, 3),
    (7, 8),
    (9, 10),
    (16, 17),
    (18, 24),
    (20, 21),
    (22, 23),
    (26, 27),
    (28, 29),
    (41, 42),
    (44, 45),
    (53, 68, 69, 73, 74),
    (54, 55, 56),
    (63, 70),
    (66, 72),
    (82, 83),
    (85, 86),
    (89, 90),
    (93, 94),
    (95, 96),
)

_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_CLASS_NAMES_SHA256 = "9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35"
_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_READOUT = "vision_pooler_output"
_RESULT_SCHEMA = "sfora-siglip-band-audit-v2"


def validate_siglip_band_definition(
    bands: Sequence[SiglipAuditBand],
    variant_groups: Sequence[Sequence[int]],
) -> None:
    """Validate one complete partition and disjoint nameplate-variant relation."""

    if tuple(band.role for band in bands) != ("optimization", "clean", "burned"):
        raise ValueError("SigLIP audit band roles differ from the registered partition")
    partition: list[int] = []
    for band in bands:
        if (
            isinstance(band.first_label, bool)
            or isinstance(band.last_label, bool)
            or band.first_label > band.last_label
        ):
            raise ValueError("SigLIP audit band partition is invalid")
        partition.extend(range(band.first_label, band.last_label + 1))
    if partition != list(range(98)):
        raise ValueError("SigLIP audit bands do not form the exact label partition")

    observed: set[int] = set()
    for group in variant_groups:
        values = tuple(group)
        if len(values) < 2 or values != tuple(sorted(values)):
            raise ValueError("SigLIP variant group must be an ordered multi-label group")
        if any(isinstance(label, bool) or not 0 <= label < 98 for label in values):
            raise ValueError("SigLIP variant group label is invalid")
        if observed.intersection(values):
            raise ValueError("SigLIP variant groups overlap")
        containing = [
            band
            for band in bands
            if band.first_label <= values[0] and values[-1] <= band.last_label
        ]
        if len(containing) != 1:
            raise ValueError("SigLIP variant group must remain inside one band")
        observed.update(values)


def nameplate_variant_representative(label: int) -> int:
    """Return the fixed nameplate-variant representative for one train label."""

    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("SigLIP variant label must be an integer")
    if not 0 <= label < 98:
        raise ValueError("SigLIP variant label is outside the train authority")
    for group in SIGLIP_AUDIT_VARIANT_GROUPS:
        if label in group:
            return group[0]
    return label


def validate_siglip_band_inputs(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
) -> None:
    """Require complete finite normalized Cars-train evidence for all three bands."""

    validate_siglip_band_definition(SIGLIP_AUDIT_BANDS, SIGLIP_AUDIT_VARIANT_GROUPS)
    if descriptors.ndim != 2 or labels.ndim != 1:
        raise ValueError("SigLIP audit descriptor and label tensors must have rows")
    if descriptors.shape[0] != labels.shape[0]:
        raise ValueError("SigLIP audit descriptor and label row counts differ")
    if descriptors.dtype != torch.float32:
        raise ValueError("SigLIP audit descriptors must use float32")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("SigLIP audit labels must use an integer dtype")
    if len(class_names) != 196:
        raise ValueError("SigLIP audit requires exactly 196 class names")
    if any(not isinstance(name, str) for name in class_names):
        raise TypeError("SigLIP audit class names must be strings")
    if descriptors.shape[0] < 2 or not bool(torch.isfinite(descriptors).all()):
        raise ValueError("SigLIP audit descriptors must be finite")
    norms = torch.linalg.vector_norm(descriptors, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=0.0):
        raise ValueError("SigLIP audit descriptors must have unit norm")
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    if frozenset(int(label) for label in labels_cpu.tolist()) != frozenset(range(98)):
        raise ValueError("SigLIP audit labels must contain exactly train classes 0 through 97")
    counts = torch.bincount(labels_cpu, minlength=98)[:98]
    if bool((counts < 2).any()):
        raise ValueError("SigLIP audit requires at least two examples per class")


def siglip_band_nearest_rows(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    band: SiglipAuditBand,
    *,
    query_block: int,
) -> tuple[int, ...]:
    """Return exact global nearest rows for one band with lowest-row ties."""

    if query_block < 1:
        raise ValueError("SigLIP audit query block must be positive")
    if band not in SIGLIP_AUDIT_BANDS:
        raise ValueError("SigLIP audit band is not registered")
    if descriptors.ndim != 2 or labels.shape != (descriptors.shape[0],):
        raise ValueError("SigLIP audit descriptor and label row counts differ")
    if descriptors.dtype != torch.float32 or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("SigLIP audit scoring dtypes differ from authority")
    if descriptors.device != labels.device:
        raise ValueError("SigLIP audit scoring tensors must share a device")
    band_mask = (labels >= band.first_label) & (labels <= band.last_label)
    band_rows = torch.nonzero(band_mask, as_tuple=False).flatten()
    if band_rows.numel() < 2:
        raise ValueError("SigLIP audit band has insufficient rows")
    gallery = descriptors[band_rows]
    nearest_rows: list[torch.Tensor] = []
    for start in range(0, int(band_rows.numel()), query_block):
        stop = min(start + query_block, int(band_rows.numel()))
        similarities = descriptors[band_rows[start:stop]] @ gallery.T
        local_queries = torch.arange(stop - start, device=descriptors.device)
        local_columns = torch.arange(start, stop, device=descriptors.device)
        similarities[local_queries, local_columns] = -torch.inf
        nearest_local = similarities.argmax(dim=1)
        nearest_rows.append(band_rows[nearest_local].detach().cpu())
    return tuple(int(row) for row in torch.cat(nearest_rows).tolist())


def _ppm(hits: int, queries: int) -> int:
    if not 0 <= hits <= queries or queries < 1:
        raise ValueError("SigLIP audit hit authority is invalid")
    return hits * 1_000_000 // queries


def score_siglip_frozen_bands(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    *,
    query_block: int,
) -> SiglipBandAuditEvidence:
    """Score strict and fixed nameplate-variant retrieval in each band."""

    if query_block < 1:
        raise ValueError("SigLIP audit query block must be positive")
    validate_siglip_band_inputs(descriptors, labels, class_names)
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    class_query_counts = tuple(int(value) for value in torch.bincount(labels_cpu).tolist())
    band_evidence: list[SiglipBandEvidence] = []
    total_queries = 0
    total_strict = 0
    total_variant = 0
    total_variant_classes = 0
    total_variant_queries = 0
    variant_labels = frozenset(
        label for group in SIGLIP_AUDIT_VARIANT_GROUPS for label in group
    )
    for band in SIGLIP_AUDIT_BANDS:
        query_rows = torch.nonzero(
            (labels_cpu >= band.first_label) & (labels_cpu <= band.last_label),
            as_tuple=False,
        ).flatten()
        nearest_rows = siglip_band_nearest_rows(
            descriptors,
            labels,
            band,
            query_block=query_block,
        )
        if len(nearest_rows) != int(query_rows.numel()):
            raise RuntimeError("SigLIP audit nearest-row cardinality differs")
        strict_hits = 0
        variant_hits = 0
        confusions: Counter[tuple[int, int]] = Counter()
        for query_row_tensor, nearest_row in zip(query_rows, nearest_rows, strict=True):
            query_label = int(labels_cpu[int(query_row_tensor)])
            nearest_label = int(labels_cpu[nearest_row])
            if query_label == nearest_label:
                strict_hits += 1
            else:
                confusions[(query_label, nearest_label)] += 1
            if nameplate_variant_representative(
                query_label
            ) == nameplate_variant_representative(nearest_label):
                variant_hits += 1
        query_count = int(query_rows.numel())
        band_variant_labels = frozenset(
            label
            for label in variant_labels
            if band.first_label <= label <= band.last_label
        )
        variant_query_count = sum(
            1 for row in query_rows if int(labels_cpu[int(row)]) in band_variant_labels
        )
        evidence = SiglipBandEvidence(
            role=band.role,
            first_label=band.first_label,
            last_label=band.last_label,
            query_count=query_count,
            variant_class_count=len(band_variant_labels),
            variant_query_count=variant_query_count,
            strict_hits=strict_hits,
            strict_recall_ppm=_ppm(strict_hits, query_count),
            variant_hits=variant_hits,
            variant_recall_ppm=_ppm(variant_hits, query_count),
            variant_rescued_errors=variant_hits - strict_hits,
            confusions=tuple(
                SiglipBandConfusion(
                    query_label=query_label,
                    nearest_label=nearest_label,
                    count=count,
                )
                for (query_label, nearest_label), count in sorted(confusions.items())
            ),
        )
        band_evidence.append(evidence)
        total_queries += query_count
        total_strict += strict_hits
        total_variant += variant_hits
        total_variant_classes += len(band_variant_labels)
        total_variant_queries += variant_query_count
    return SiglipBandAuditEvidence(
        bands=tuple(band_evidence),
        query_count=total_queries,
        class_query_counts=class_query_counts,
        variant_class_count=total_variant_classes,
        variant_query_count=total_variant_queries,
        strict_hits=total_strict,
        strict_recall_ppm=_ppm(total_strict, total_queries),
        variant_hits=total_variant,
        variant_recall_ppm=_ppm(total_variant, total_queries),
        variant_rescued_errors=total_variant - total_strict,
    )


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _exact_integer(value: object, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"SigLIP audit {field} integer authority differs")
    return value


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"SigLIP audit {field} string authority differs")
    return value


def _hex(value: object, *, field: str, length: int) -> str:
    text = _exact_string(value, field=field)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"SigLIP audit {field} digest authority differs")
    return text


def _validate_authority(authority: SiglipBandAuditAuthority) -> None:
    _hex(authority.source_commit, field="source commit", length=40)
    _hex(authority.source_tree_digest, field="source tree", length=64)
    for field, value in (
        ("dataset examples", authority.dataset_examples_sha256),
        ("ordered example IDs", authority.ordered_example_ids_sha256),
        ("descriptor", authority.descriptor_sha256),
        ("label vector", authority.label_vector_sha256),
        ("class names", authority.class_names_sha256),
    ):
        _hex(value, field=field, length=64)
    _hex(authority.model_revision, field="model revision", length=40)
    if (
        authority.dataset_revision != _DATASET_REVISION
        or authority.class_names_sha256 != _CLASS_NAMES_SHA256
        or authority.model_name != _MODEL_NAME
        or authority.model_revision != _MODEL_REVISION
        or authority.readout != _READOUT
        or authority.split != "train"
        or authority.cublas_workspace_config != ":4096:8"
    ):
        raise ValueError("SigLIP audit registered authority differs")
    _exact_integer(authority.batch_size, field="batch size", positive=True)
    _exact_integer(authority.query_block, field="query block", positive=True)


def _band_mapping(band: SiglipBandEvidence) -> dict[str, Any]:
    return {
        "role": band.role,
        "first_label": band.first_label,
        "last_label": band.last_label,
        "query_count": band.query_count,
        "variant_class_count": band.variant_class_count,
        "variant_query_count": band.variant_query_count,
        "strict_hits": band.strict_hits,
        "strict_recall_ppm": band.strict_recall_ppm,
        "variant_hits": band.variant_hits,
        "variant_recall_ppm": band.variant_recall_ppm,
        "variant_rescued_errors": band.variant_rescued_errors,
        "confusions": [asdict(confusion) for confusion in band.confusions],
    }


def _evidence_mapping(evidence: SiglipBandAuditEvidence) -> dict[str, Any]:
    return {
        "bands": [_band_mapping(band) for band in evidence.bands],
        "query_count": evidence.query_count,
        "class_query_counts": list(evidence.class_query_counts),
        "variant_class_count": evidence.variant_class_count,
        "variant_query_count": evidence.variant_query_count,
        "strict_hits": evidence.strict_hits,
        "strict_recall_ppm": evidence.strict_recall_ppm,
        "variant_hits": evidence.variant_hits,
        "variant_recall_ppm": evidence.variant_recall_ppm,
        "variant_rescued_errors": evidence.variant_rescued_errors,
    }


def _authority_from_mapping(value: object) -> SiglipBandAuditAuthority:
    if type(value) is not dict or set(value) != {
        field.name for field in SiglipBandAuditAuthority.__dataclass_fields__.values()
    }:
        raise ValueError("SigLIP audit authority schema differs")
    mapping = value
    authority = SiglipBandAuditAuthority(
        source_commit=_exact_string(mapping["source_commit"], field="source commit"),
        source_tree_digest=_exact_string(mapping["source_tree_digest"], field="source tree"),
        dataset_revision=_exact_string(mapping["dataset_revision"], field="dataset revision"),
        dataset_examples_sha256=_exact_string(
            mapping["dataset_examples_sha256"], field="dataset examples"
        ),
        ordered_example_ids_sha256=_exact_string(
            mapping["ordered_example_ids_sha256"], field="ordered example IDs"
        ),
        descriptor_sha256=_exact_string(mapping["descriptor_sha256"], field="descriptor"),
        label_vector_sha256=_exact_string(mapping["label_vector_sha256"], field="label vector"),
        class_names_sha256=_exact_string(mapping["class_names_sha256"], field="class names"),
        model_name=_exact_string(mapping["model_name"], field="model name"),
        model_revision=_exact_string(mapping["model_revision"], field="model revision"),
        readout=_exact_string(mapping["readout"], field="readout"),
        split=_exact_string(mapping["split"], field="split"),
        batch_size=_exact_integer(mapping["batch_size"], field="batch size", positive=True),
        query_block=_exact_integer(mapping["query_block"], field="query block", positive=True),
        cublas_workspace_config=_exact_string(
            mapping["cublas_workspace_config"], field="cuBLAS workspace config"
        ),
    )
    _validate_authority(authority)
    return authority


def _validate_band_mapping(value: object, band: SiglipAuditBand) -> SiglipBandEvidence:
    expected_keys = {
        "role",
        "first_label",
        "last_label",
        "query_count",
        "variant_class_count",
        "variant_query_count",
        "strict_hits",
        "strict_recall_ppm",
        "variant_hits",
        "variant_recall_ppm",
        "variant_rescued_errors",
        "confusions",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("SigLIP audit band schema differs")
    mapping = value
    role = _exact_string(mapping["role"], field="band role")
    first_label = _exact_integer(mapping["first_label"], field="band first label")
    last_label = _exact_integer(mapping["last_label"], field="band last label")
    query_count = _exact_integer(mapping["query_count"], field="band query count", positive=True)
    variant_class_count = _exact_integer(
        mapping["variant_class_count"], field="band variant class count", positive=True
    )
    variant_query_count = _exact_integer(
        mapping["variant_query_count"], field="band variant query count", positive=True
    )
    strict_hits = _exact_integer(mapping["strict_hits"], field="band strict hits")
    strict_recall = _exact_integer(mapping["strict_recall_ppm"], field="band strict recall")
    variant_hits = _exact_integer(mapping["variant_hits"], field="band variant hits")
    variant_recall = _exact_integer(
        mapping["variant_recall_ppm"], field="band variant recall"
    )
    variant_rescued = _exact_integer(
        mapping["variant_rescued_errors"], field="band variant rescued errors"
    )
    if (role, first_label, last_label) != (band.role, band.first_label, band.last_label):
        raise ValueError("SigLIP audit band authority differs")
    rows = mapping["confusions"]
    if type(rows) is not list:
        raise ValueError("SigLIP audit confusion schema differs")
    confusions: list[SiglipBandConfusion] = []
    previous: tuple[int, int] | None = None
    confusion_count = 0
    rescued = 0
    for row in rows:
        if type(row) is not dict or set(row) != {"query_label", "nearest_label", "count"}:
            raise ValueError("SigLIP audit confusion schema differs")
        query_label = _exact_integer(row["query_label"], field="confusion query label")
        nearest_label = _exact_integer(row["nearest_label"], field="confusion nearest label")
        count = _exact_integer(row["count"], field="confusion count", positive=True)
        pair = (query_label, nearest_label)
        if (
            previous is not None
            and pair <= previous
            or query_label == nearest_label
            or not band.first_label <= query_label <= band.last_label
            or not band.first_label <= nearest_label <= band.last_label
        ):
            raise ValueError("SigLIP audit confusion authority differs")
        previous = pair
        confusion_count += count
        if nameplate_variant_representative(
            query_label
        ) == nameplate_variant_representative(nearest_label):
            rescued += count
        confusions.append(SiglipBandConfusion(query_label, nearest_label, count))
    expected_variant_classes = len(
        {
            label
            for group in SIGLIP_AUDIT_VARIANT_GROUPS
            for label in group
            if band.first_label <= label <= band.last_label
        }
    )
    if (
        strict_hits > query_count
        or variant_hits > query_count
        or variant_class_count != expected_variant_classes
        or variant_query_count > query_count
        or strict_recall != _ppm(strict_hits, query_count)
        or variant_recall != _ppm(variant_hits, query_count)
        or variant_hits != strict_hits + variant_rescued
    ):
        raise ValueError("SigLIP audit band metric authority differs")
    if confusion_count != query_count - strict_hits or rescued != variant_rescued:
        raise ValueError("SigLIP audit confusion metric authority differs")
    return SiglipBandEvidence(
        role=role,
        first_label=first_label,
        last_label=last_label,
        query_count=query_count,
        variant_class_count=variant_class_count,
        variant_query_count=variant_query_count,
        strict_hits=strict_hits,
        strict_recall_ppm=strict_recall,
        variant_hits=variant_hits,
        variant_recall_ppm=variant_recall,
        variant_rescued_errors=variant_rescued,
        confusions=tuple(confusions),
    )


def _validate_result_mapping(
    value: object,
    *,
    expected_authority: SiglipBandAuditAuthority,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "authority",
        "bands",
        "query_count",
        "class_query_counts",
        "variant_class_count",
        "variant_query_count",
        "strict_hits",
        "strict_recall_ppm",
        "variant_hits",
        "variant_recall_ppm",
        "variant_rescued_errors",
        "result_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("SigLIP audit result schema differs")
    mapping = value
    if mapping["schema"] != _RESULT_SCHEMA:
        raise ValueError("SigLIP audit result schema differs")
    if mapping["claim_eligible"] is not False:
        raise ValueError("SigLIP audit claim authority differs")
    if mapping["official_test_access"] is not False:
        raise ValueError("SigLIP audit official-test authority differs")
    authority = _authority_from_mapping(mapping["authority"])
    if authority != expected_authority:
        raise ValueError("SigLIP audit authority differs")
    rows = mapping["bands"]
    if type(rows) is not list or len(rows) != len(SIGLIP_AUDIT_BANDS):
        raise ValueError("SigLIP audit band schema differs")
    bands = tuple(
        _validate_band_mapping(row, band)
        for row, band in zip(rows, SIGLIP_AUDIT_BANDS, strict=True)
    )
    query_count = _exact_integer(
        mapping["query_count"], field="aggregate query count", positive=True
    )
    raw_class_query_counts = mapping["class_query_counts"]
    if type(raw_class_query_counts) is not list or len(raw_class_query_counts) != 98:
        raise ValueError("SigLIP audit aggregate class query schema differs")
    class_query_counts = tuple(
        _exact_integer(value, field="class query count", positive=True)
        for value in raw_class_query_counts
    )
    variant_class_count = _exact_integer(
        mapping["variant_class_count"], field="aggregate variant class count", positive=True
    )
    variant_query_count = _exact_integer(
        mapping["variant_query_count"], field="aggregate variant query count", positive=True
    )
    strict_hits = _exact_integer(mapping["strict_hits"], field="aggregate strict hits")
    strict_recall = _exact_integer(mapping["strict_recall_ppm"], field="aggregate strict recall")
    variant_hits = _exact_integer(mapping["variant_hits"], field="aggregate variant hits")
    variant_recall = _exact_integer(
        mapping["variant_recall_ppm"], field="aggregate variant recall"
    )
    variant_rescued = _exact_integer(
        mapping["variant_rescued_errors"], field="aggregate variant rescued errors"
    )
    if (
        query_count != sum(band.query_count for band in bands)
        or query_count != sum(class_query_counts)
        or any(
            band.query_count
            != sum(class_query_counts[band.first_label : band.last_label + 1])
            for band in bands
        )
        or variant_class_count != sum(band.variant_class_count for band in bands)
        or variant_query_count != sum(band.variant_query_count for band in bands)
        or any(
            band.variant_query_count
            != sum(
                class_query_counts[label]
                for group in SIGLIP_AUDIT_VARIANT_GROUPS
                for label in group
                if band.first_label <= label <= band.last_label
            )
            for band in bands
        )
        or strict_hits != sum(band.strict_hits for band in bands)
        or variant_hits != sum(band.variant_hits for band in bands)
        or variant_rescued != sum(band.variant_rescued_errors for band in bands)
        or strict_recall != _ppm(strict_hits, query_count)
        or variant_recall != _ppm(variant_hits, query_count)
        or variant_rescued != variant_hits - strict_hits
    ):
        raise ValueError("SigLIP audit aggregate metric authority differs")
    return mapping


def canonical_siglip_band_audit_bytes(
    evidence: SiglipBandAuditEvidence,
    *,
    authority: SiglipBandAuditAuthority,
) -> bytes:
    """Emit one canonical, claim-ineligible frozen-band result."""

    _validate_authority(authority)
    value: dict[str, Any] = {
        "schema": _RESULT_SCHEMA,
        "claim_eligible": False,
        "official_test_access": False,
        "authority": asdict(authority),
        **_evidence_mapping(evidence),
    }
    unsigned = _canonical_bytes(value)
    value["result_sha256"] = hashlib.sha256(unsigned).hexdigest()
    raw = _canonical_bytes(value)
    validate_siglip_band_audit_bytes(raw, expected_authority=authority)
    return raw


def validate_siglip_band_audit_bytes(
    raw: bytes,
    *,
    expected_authority: SiglipBandAuditAuthority,
) -> dict[str, Any]:
    """Independently authenticate and recompute one canonical result."""

    _validate_authority(expected_authority)
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("SigLIP audit canonical result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SigLIP audit canonical result bytes differ") from error
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ValueError("SigLIP audit canonical result bytes differ")
    result_digest = value.get("result_sha256")
    if type(result_digest) is not str:
        raise ValueError("SigLIP audit result digest differs")
    unsigned = dict(value)
    unsigned.pop("result_sha256")
    if result_digest != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest():
        raise ValueError("SigLIP audit result digest differs")
    return _validate_result_mapping(value, expected_authority=expected_authority)


validate_siglip_band_definition(SIGLIP_AUDIT_BANDS, SIGLIP_AUDIT_VARIANT_GROUPS)
