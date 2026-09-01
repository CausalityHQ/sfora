from __future__ import annotations

from sfora.duplicate_ceiling import score_exact_duplicate_ceiling


def test_duplicate_ceiling_counts_only_minority_labels_as_irreducible() -> None:
    evidence = score_exact_duplicate_ceiling(
        labels=(82, 83, 82, 82, 83, 84, 84, 84),
        rgb_sha256=(
            "a" * 64,
            "a" * 64,
            "b" * 64,
            "b" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "d" * 64,
        ),
    )

    assert evidence.query_count == 8
    assert evidence.conflicting_group_count == 2
    assert evidence.conflicting_row_count == 5
    assert evidence.same_label_duplicate_group_count == 1
    assert evidence.same_label_duplicate_row_count == 2
    assert evidence.deterministic_label_error_floor == 2
    assert evidence.deterministic_label_ceiling_hits == 6
    assert evidence.deterministic_label_ceiling_recall_ppm == 750_000
    assert evidence.conflicting_row_indices == (0, 1, 2, 3, 4)
    assert evidence.groups == (
        ("a" * 64, ((82, 1), (83, 1))),
        ("b" * 64, ((82, 2), (83, 1))),
    )
