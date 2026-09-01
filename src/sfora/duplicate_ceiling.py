"""Exact pixel-equivalence label-conflict ceiling for retrieval datasets."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ExactDuplicateCeilingEvidence:
    """Conservative ambiguity floor induced by exact RGB-equivalent records."""

    query_count: int
    conflicting_group_count: int
    conflicting_row_count: int
    irreducible_error_floor: int
    strict_ceiling_hits: int
    strict_ceiling_recall_ppm: int
    groups: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]


def score_exact_duplicate_ceiling(
    *,
    labels: tuple[int, ...],
    rgb_sha256: tuple[str, ...],
) -> ExactDuplicateCeilingEvidence:
    """Count minority labels inside exact RGB equivalence classes."""

    if not labels or len(labels) != len(rgb_sha256):
        raise ValueError("duplicate-ceiling row authority differs")
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("duplicate-ceiling labels must be concrete nonnegative integers")
    if any(type(digest) is not str or _SHA256.fullmatch(digest) is None for digest in rgb_sha256):
        raise ValueError("duplicate-ceiling RGB digest authority differs")
    by_digest: dict[str, list[int]] = defaultdict(list)
    for label, digest in zip(labels, rgb_sha256, strict=True):
        by_digest[digest].append(label)
    groups: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    conflicting_rows = 0
    irreducible_errors = 0
    for digest in sorted(by_digest):
        counts = Counter(by_digest[digest])
        if len(counts) < 2:
            continue
        ordered_counts = tuple(sorted(counts.items()))
        group_size = sum(counts.values())
        groups.append((digest, ordered_counts))
        conflicting_rows += group_size
        irreducible_errors += group_size - max(counts.values())
    query_count = len(labels)
    ceiling_hits = query_count - irreducible_errors
    return ExactDuplicateCeilingEvidence(
        query_count=query_count,
        conflicting_group_count=len(groups),
        conflicting_row_count=conflicting_rows,
        irreducible_error_floor=irreducible_errors,
        strict_ceiling_hits=ceiling_hits,
        strict_ceiling_recall_ppm=ceiling_hits * 1_000_000 // query_count,
        groups=tuple(groups),
    )
