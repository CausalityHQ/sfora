"""Claim-ineligible query-shrinkage ceiling for CESD representation triage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sfora.siglip_band_audit import (
    SIGLIP_AUDIT_VARIANT_GROUPS,
    nameplate_variant_representative,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES

_GAIN_GATE_PPM = 15_000


@dataclass(frozen=True)
class CesdOracleEvidence:
    """Exact frozen-gallery evidence from a binary query-side shrinkage oracle."""

    query_count: int
    baseline_hits: int
    shrinkage_hits: int
    oracle_hits: int
    rescued_query_rows: tuple[int, ...]
    alpha_zero_query_rows: tuple[int, ...]
    baseline_recall_ppm: int
    oracle_recall_ppm: int
    gain_ppm: int
    passed: bool


def _ppm(hits: int, queries: int) -> int:
    return hits * 1_000_000 // queries


def _validate_inputs(descriptors: torch.Tensor, labels: torch.Tensor) -> None:
    if descriptors.ndim != 2 or labels.shape != (descriptors.shape[0],):
        raise ValueError("CESD descriptor and label shapes differ")
    if descriptors.dtype != torch.float32:
        raise ValueError("CESD descriptors must use float32")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("CESD labels must use an integer dtype")
    if descriptors.device != labels.device:
        raise ValueError("CESD descriptors and labels must share a device")
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("CESD descriptors must be finite")
    norms = torch.linalg.vector_norm(descriptors, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=0.0):
        raise ValueError("CESD descriptors must have unit norm")
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    if frozenset(int(value) for value in labels_cpu.tolist()) != SUBSTRATE_F0_CLASSES:
        raise ValueError("CESD oracle requires exactly burned labels 82 through 97")
    counts = torch.bincount(labels_cpu, minlength=98)[82:98]
    if bool((counts < 2).any()):
        raise ValueError("CESD oracle requires at least two rows per burned class")


def _leave_one_out_nameplate_directions(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    representatives = torch.tensor(
        [nameplate_variant_representative(int(label)) for label in labels_cpu],
        device=descriptors.device,
        dtype=torch.int64,
    )
    directions = descriptors.clone()
    eligible_representatives = frozenset(group[0] for group in SIGLIP_AUDIT_VARIANT_GROUPS)
    for representative in torch.unique(representatives, sorted=True):
        if int(representative) not in eligible_representatives:
            continue
        rows = torch.nonzero(representatives == representative, as_tuple=False).flatten()
        group_sum = descriptors[rows].sum(dim=0)
        leave_one_out = group_sum.unsqueeze(0) - descriptors[rows]
        if bool((torch.linalg.vector_norm(leave_one_out, dim=1) == 0).any()):
            raise ValueError("CESD leave-one-out nameplate direction is zero")
        directions[rows] = F.normalize(leave_one_out, dim=1)
    return directions


def _nearest_rows(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    *,
    query_block: int,
) -> tuple[int, ...]:
    nearest: list[torch.Tensor] = []
    count = int(queries.shape[0])
    for start in range(0, count, query_block):
        stop = min(start + query_block, count)
        scores = queries[start:stop] @ gallery.T
        local_rows = torch.arange(stop - start, device=scores.device)
        gallery_rows = torch.arange(start, stop, device=scores.device)
        scores[local_rows, gallery_rows] = -torch.inf
        nearest.append(scores.argmax(dim=1).detach().cpu())
    return tuple(int(row) for row in torch.cat(nearest).tolist())


def score_cesd_query_shrinkage_oracle(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> CesdOracleEvidence:
    """Score an optimistic binary query-side nameplate-shrinkage upper bound."""

    if isinstance(query_block, bool) or not isinstance(query_block, int) or query_block < 1:
        raise ValueError("CESD query block must be a positive integer")
    _validate_inputs(descriptors, labels)
    shrinkage = _leave_one_out_nameplate_directions(descriptors, labels)
    baseline_rows = _nearest_rows(descriptors, descriptors, query_block=query_block)
    shrinkage_rows = _nearest_rows(shrinkage, descriptors, query_block=query_block)
    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    baseline_hits = 0
    shrinkage_hits = 0
    rescued: list[int] = []
    for query_row, (baseline_row, shrinkage_row) in enumerate(
        zip(baseline_rows, shrinkage_rows, strict=True)
    ):
        label = int(labels_cpu[query_row])
        baseline_correct = int(labels_cpu[baseline_row]) == label
        shrinkage_correct = int(labels_cpu[shrinkage_row]) == label
        baseline_hits += baseline_correct
        shrinkage_hits += shrinkage_correct
        if not baseline_correct and shrinkage_correct:
            rescued.append(query_row)
    query_count = int(labels.numel())
    oracle_hits = baseline_hits + len(rescued)
    gain_ppm = _ppm(len(rescued), query_count)
    return CesdOracleEvidence(
        query_count=query_count,
        baseline_hits=baseline_hits,
        shrinkage_hits=shrinkage_hits,
        oracle_hits=oracle_hits,
        rescued_query_rows=tuple(rescued),
        alpha_zero_query_rows=tuple(rescued),
        baseline_recall_ppm=_ppm(baseline_hits, query_count),
        oracle_recall_ppm=_ppm(oracle_hits, query_count),
        gain_ppm=gain_ppm,
        passed=len(rescued) * 1_000_000 >= _GAIN_GATE_PPM * query_count,
    )
