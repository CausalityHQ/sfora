"""Exact pixel-equivalence ambiguity bound for deterministic image labelling."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ExactDuplicateCeilingEvidence:
    """Bayes ambiguity bound induced by exact RGB-equivalent records.

    This is not a leave-one-out retrieval ceiling: retrieval removes the query
    from its gallery and may intentionally map other pixels to the same vector.
    """

    query_count: int
    conflicting_group_count: int
    conflicting_row_count: int
    same_label_duplicate_group_count: int
    same_label_duplicate_row_count: int
    deterministic_label_error_floor: int
    deterministic_label_ceiling_hits: int
    deterministic_label_ceiling_recall_ppm: int
    conflicting_row_indices: tuple[int, ...]
    groups: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]


def score_exact_duplicate_ceiling(
    *,
    labels: tuple[int, ...],
    rgb_sha256: tuple[str, ...],
) -> ExactDuplicateCeilingEvidence:
    """Count minority labels inside exact RGB equivalence classes.

    The result bounds deterministic classification from decoded RGB pixels. It
    describes dataset ambiguity and must not be reported as a retrieval bound.
    """

    if not labels or len(labels) != len(rgb_sha256):
        raise ValueError("duplicate-ceiling row authority differs")
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("duplicate-ceiling labels must be concrete nonnegative integers")
    if any(type(digest) is not str or _SHA256.fullmatch(digest) is None for digest in rgb_sha256):
        raise ValueError("duplicate-ceiling RGB digest authority differs")
    by_digest: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row_index, (label, digest) in enumerate(zip(labels, rgb_sha256, strict=True)):
        by_digest[digest].append((row_index, label))
    groups: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    conflicting_rows = 0
    conflicting_indices: list[int] = []
    same_label_groups = 0
    same_label_rows = 0
    irreducible_errors = 0
    for digest in sorted(by_digest):
        rows = by_digest[digest]
        counts = Counter(label for _, label in rows)
        if len(counts) < 2:
            if len(rows) > 1:
                same_label_groups += 1
                same_label_rows += len(rows)
            continue
        ordered_counts = tuple(sorted(counts.items()))
        group_size = sum(counts.values())
        groups.append((digest, ordered_counts))
        conflicting_rows += group_size
        conflicting_indices.extend(row_index for row_index, _ in rows)
        irreducible_errors += group_size - max(counts.values())
    query_count = len(labels)
    ceiling_hits = query_count - irreducible_errors
    return ExactDuplicateCeilingEvidence(
        query_count=query_count,
        conflicting_group_count=len(groups),
        conflicting_row_count=conflicting_rows,
        same_label_duplicate_group_count=same_label_groups,
        same_label_duplicate_row_count=same_label_rows,
        deterministic_label_error_floor=irreducible_errors,
        deterministic_label_ceiling_hits=ceiling_hits,
        deterministic_label_ceiling_recall_ppm=ceiling_hits * 1_000_000 // query_count,
        conflicting_row_indices=tuple(sorted(conflicting_indices)),
        groups=tuple(groups),
    )
