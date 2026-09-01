from __future__ import annotations

from sfora.duplicate_ceiling import score_exact_duplicate_ceiling


def test_duplicate_ceiling_counts_only_minority_labels_as_irreducible() -> None:
    evidence = score_exact_duplicate_ceiling(
        labels=(82, 83, 82, 82, 83, 84),
        rgb_sha256=("a" * 64, "a" * 64, "b" * 64, "b" * 64, "b" * 64, "c" * 64),
    )

    assert evidence.query_count == 6
    assert evidence.conflicting_group_count == 2
    assert evidence.conflicting_row_count == 5
    assert evidence.irreducible_error_floor == 2
    assert evidence.strict_ceiling_hits == 4
    assert evidence.strict_ceiling_recall_ppm == 666_666
    assert evidence.groups == (
        ("a" * 64, ((82, 1), (83, 1))),
        ("b" * 64, ((82, 2), (83, 1))),
    )
