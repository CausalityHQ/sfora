from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sfora.asymmetric_cohort_residual import (
    ArmComparison,
    DirectComparison,
    Evaluation,
    _assignment_permutations,
    _comparison,
    decide_ahncr,
    evaluate_ahncr,
    exact_mcnemar,
    hard_negative_centroids,
    reporting_shards,
    select_query_and_gallery,
    split_heldout_labels,
    unary_cohort_density,
)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return np.asarray(values / np.linalg.norm(values, axis=1, keepdims=True), dtype=np.float32)


def _label_key(value: np.int64) -> tuple[bytes, int]:
    encoded = np.asarray(value, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).digest(), int(value)


def _id_key(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(value.encode("utf-8")).digest(), value


def test_split_heldout_labels_is_literal_hash_ordered_and_class_disjoint() -> None:
    labels = np.asarray([9, 9, 7, 7, 7, 5, 5, 3, 3, 1, 1, 11], dtype=np.int64)
    split = split_heldout_labels(labels)
    eligible = sorted(
        [np.int64(1), np.int64(3), np.int64(5), np.int64(7), np.int64(9)],
        key=_label_key,
    )
    boundary = int(np.floor(0.8 * len(eligible)))
    assert split.cohort_labels.tolist() == [int(v) for v in eligible[:boundary]] + [11]
    assert split.evaluation_labels.tolist() == [int(v) for v in eligible[boundary:]]
    assert set(split.cohort_labels.tolist()).isdisjoint(split.evaluation_labels)


def test_select_query_and_gallery_uses_smallest_hashed_id_per_label() -> None:
    labels = np.asarray([5, 5, 8, 8, 8, 9], dtype=np.int64)
    ids = np.asarray(["z", "a", "three", "one", "two", "unused"])
    selected = select_query_and_gallery(labels, ids, np.asarray([5, 8], dtype=np.int64))
    expected_queries = [
        min([0, 1], key=lambda i: _id_key(str(ids[i]))),
        min([2, 3, 4], key=lambda i: _id_key(str(ids[i]))),
    ]
    assert selected.query_indices.tolist() == expected_queries
    assert selected.gallery_indices.tolist() == [
        index for index in [0, 1, 2, 3, 4] if index not in expected_queries
    ]
    assert selected.query_labels.tolist() == [5, 8]


def test_reporting_shards_uses_domain_separated_label_hash() -> None:
    labels = np.arange(1000, 1100, dtype=np.int64)
    observed = reporting_shards(labels)
    expected = []
    for value in labels:
        digest = hashlib.sha256(
            b"AHNCR-shard-v1:" + np.asarray(value, dtype="<i8").tobytes()
        ).digest()
        expected.append(digest[0] >> 6)
    assert observed.tolist() == expected
    assert set(observed.tolist()) == {0, 1, 2, 3}


def test_hard_negative_centroids_matches_stable_literal_topk_and_blocks() -> None:
    cohort = _unit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    queries = _unit([[1, 1], [-1, 1]])
    expected = np.asarray(
        [(cohort[0] + cohort[1]) / 2, (cohort[1] + cohort[2]) / 2],
        dtype=np.float32,
    )
    assert hard_negative_centroids(queries, cohort, k=2, block_size=1) == pytest.approx(expected)
    assert hard_negative_centroids(queries, cohort, k=2, block_size=8) == pytest.approx(expected)


def test_hard_negative_centroid_breaks_kth_boundary_tie_by_first_index() -> None:
    cohort = _unit([[0.8, 0.6], [0.8, -0.6], [0, 1]])
    query = _unit([[1, 0]])
    assert hard_negative_centroids(query, cohort, k=1, block_size=1) == pytest.approx(cohort[[0]])


def test_unary_cohort_density_matches_literal_topk_mean() -> None:
    cohort = _unit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    gallery = _unit([[1, 1], [1, -1]])
    root_half = np.float32(1 / np.sqrt(2))
    expected = np.asarray([root_half, root_half], dtype=np.float64)
    assert unary_cohort_density(gallery, cohort, k=2, block_size=1) == pytest.approx(expected)


@pytest.mark.parametrize("bad_k", [True, 0, 5])
def test_cohort_operators_reject_invalid_k(bad_k: object) -> None:
    values = _unit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    with pytest.raises(ValueError, match="k"):
        hard_negative_centroids(values[:1], values, k=bad_k)  # type: ignore[arg-type]


def test_split_rejects_wrong_label_dtype_and_empty_partitions() -> None:
    with pytest.raises(ValueError, match="int64"):
        split_heldout_labels(np.asarray([1, 1, 2, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="eligible"):
        split_heldout_labels(np.asarray([1, 2, 3], dtype=np.int64))


def _literal_comparison(
    raw_indices: np.ndarray,
    corrected_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    shards: np.ndarray,
) -> tuple[float, int, int, float, tuple[float, ...]]:
    raw_correct = gallery_labels[raw_indices] == query_labels
    corrected = gallery_labels[corrected_indices] == query_labels
    wrong_to_right = int(np.sum(~raw_correct & corrected))
    right_to_wrong = int(np.sum(raw_correct & ~corrected))
    gain = float(np.mean(corrected) - np.mean(raw_correct))
    shard_gains = tuple(
        float(np.mean(corrected[shards == shard]) - np.mean(raw_correct[shards == shard]))
        for shard in range(4)
    )
    return (
        gain,
        wrong_to_right,
        right_to_wrong,
        exact_mcnemar(wrong_to_right, right_to_wrong),
        shard_gains,
    )


def test_evaluate_ahncr_matches_literal_scores_and_controls() -> None:
    cohort = _unit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    queries = _unit([[1, 0.2], [0.2, 1], [-1, 0.2], [0.2, -1]])
    gallery = _unit(
        [
            [1, 0.1],
            [0.8, 0.6],
            [0.1, 1],
            [-0.6, 0.8],
            [-1, 0.1],
            [-0.8, -0.6],
            [0.1, -1],
            [0.6, -0.8],
        ]
    )
    query_labels = np.asarray([10, 20, 30, 40], dtype=np.int64)
    gallery_labels = np.asarray([10, 99, 20, 98, 30, 97, 40, 96], dtype=np.int64)
    shards = np.asarray([0, 1, 2, 3], dtype=np.int64)

    result = evaluate_ahncr(
        queries,
        query_labels,
        gallery,
        gallery_labels,
        cohort,
        shards,
        k=1,
        block_size=2,
        permutation_seed=20260814,
    )

    raw = np.asarray(queries @ gallery.T, dtype=np.float32)
    q_centroids = hard_negative_centroids(queries, cohort, k=1)
    g_centroids = hard_negative_centroids(gallery, cohort, k=1)
    global_mean = np.mean(cohort.astype(np.float64), axis=0).astype(np.float32)
    density = unary_cohort_density(gallery, cohort, k=1)
    q_residual = queries - q_centroids
    g_residual = gallery - g_centroids
    symmetric = (q_residual / np.linalg.norm(q_residual, axis=1, keepdims=True)) @ (
        g_residual / np.linalg.norm(g_residual, axis=1, keepdims=True)
    ).T
    score_matrices = {
        "raw": raw,
        "ahncr": 2 * raw - q_centroids @ gallery.T,
        "global_mean": 2 * raw - np.broadcast_to(global_mean @ gallery.T, raw.shape),
        "positive_expansion": 2 * raw + q_centroids @ gallery.T,
        "unary_cohort_density": 2 * raw - density[None, :],
        "symmetric_ad_norm": symmetric,
    }
    raw_indices = np.argmax(score_matrices["raw"], axis=1)
    assert list(result.arms) == list(score_matrices)
    for name, scores in score_matrices.items():
        expected = _literal_comparison(
            raw_indices,
            np.argmax(scores, axis=1),
            query_labels,
            gallery_labels,
            shards,
        )
        arm = result.arms[name]
        assert (
            arm.gain,
            arm.wrong_to_right,
            arm.right_to_wrong,
            arm.p_value,
        ) == pytest.approx(expected[:4])
        assert arm.shard_gains == pytest.approx(expected[4])

    rng = np.random.Generator(np.random.PCG64(20260814))
    expected_null = []
    while len(expected_null) < 20:
        permutation = rng.permutation(len(queries))
        if np.any(permutation == np.arange(len(queries))):
            continue
        indices = np.argmax(2 * raw - q_centroids[permutation] @ gallery.T, axis=1)
        expected_null.append(
            _literal_comparison(raw_indices, indices, query_labels, gallery_labels, shards)[0]
        )
    assert result.shuffled_gains == pytest.approx(expected_null)
    assert result.shuffled_p95 == pytest.approx(np.quantile(expected_null, 0.95, method="linear"))
    ahncr_correct = gallery_labels[np.argmax(score_matrices["ahncr"], axis=1)] == query_labels
    for name in (
        "global_mean",
        "positive_expansion",
        "unary_cohort_density",
        "symmetric_ad_norm",
    ):
        control_correct = gallery_labels[np.argmax(score_matrices[name], axis=1)] == query_labels
        expected_favor = int(np.sum(~control_correct & ahncr_correct))
        expected_against = int(np.sum(control_correct & ~ahncr_correct))
        comparison = result.control_comparisons[name]
        assert comparison.control_wrong_ahncr_right == expected_favor
        assert comparison.control_right_ahncr_wrong == expected_against
        assert comparison.p_value == exact_mcnemar(expected_favor, expected_against)


def test_exact_mcnemar_uses_exact_two_sided_binomial_tail() -> None:
    assert exact_mcnemar(0, 5) == pytest.approx(0.0625)
    assert exact_mcnemar(2, 10) == pytest.approx(158 / 4096)


def test_assignment_null_uses_fixed_derangements() -> None:
    first = _assignment_permutations(11, 20, 20260814)
    second = _assignment_permutations(11, 20, 20260814)
    assert len(first) == 20
    for permutation, repeated in zip(first, second, strict=True):
        assert permutation.tolist() == repeated.tolist()
        assert sorted(permutation.tolist()) == list(range(11))
        assert np.all(permutation != np.arange(11))


def test_shard_gain_uses_one_exact_count_quotient() -> None:
    query_labels = np.zeros(216, dtype=np.int64)
    gallery_labels = np.asarray([0, 1], dtype=np.int64)
    raw_indices = np.ones(216, dtype=np.int64)
    raw_indices[:117] = 0
    corrected = np.ones(216, dtype=np.int64)
    corrected[:112] = 0
    raw_indices[213:] = 0
    corrected[213:] = 0
    shards = np.asarray([0] * 213 + [1, 2, 3], dtype=np.int64)
    value = _comparison(raw_indices, corrected, query_labels, gallery_labels, shards)
    assert value.shard_gains[0] == (112 - 117) / 213


def _passing_evaluation() -> Evaluation:
    raw = ArmComparison(0.70, 0.70, 0.0, 0, 0, 1.0, (0.0, 0.0, 0.0, 0.0))
    candidate = ArmComparison(0.70, 0.71, 0.01, 20, 10, 0.009, (0.01, 0.01, 0.01, -0.001))
    control = ArmComparison(0.70, 0.7089, 0.0089, 10, 1, 0.02, (0, 0, 0, 0))
    direct = DirectComparison(12, 0, exact_mcnemar(12, 0))
    return Evaluation(
        arms={
            "raw": raw,
            "ahncr": candidate,
            "global_mean": control,
            "positive_expansion": control,
            "unary_cohort_density": control,
            "symmetric_ad_norm": control,
        },
        shuffled_gains=(0.001,) * 19 + (0.008,),
        shuffled_p95=0.00135,
        control_comparisons={
            "global_mean": direct,
            "positive_expansion": direct,
            "unary_cohort_density": direct,
            "symmetric_ad_norm": direct,
        },
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "gain",
        "p",
        "shards",
        "global",
        "positive",
        "unary",
        "symmetric",
        "null",
        "direction",
        "direct_control",
    ],
)
def test_decide_ahncr_rejects_every_failed_boundary(mutation: str) -> None:
    value = _passing_evaluation()
    arms = dict(value.arms)
    candidate = arms["ahncr"]
    if mutation == "gain":
        candidate = ArmComparison(0.70, 0.703, 0.003 - 1e-12, 20, 10, 0.009, candidate.shard_gains)
    elif mutation == "p":
        candidate = ArmComparison(0.70, 0.71, 0.01, 20, 10, 0.01, candidate.shard_gains)
    elif mutation == "shards":
        candidate = ArmComparison(0.70, 0.71, 0.01, 20, 10, 0.009, (0.01, 0.01, 0.0, -0.01))
    elif mutation == "direction":
        candidate = ArmComparison(0.70, 0.71, 0.01, 10, 10, 0.009, candidate.shard_gains)
    elif mutation in {"global", "positive", "unary", "symmetric"}:
        key = {
            "global": "global_mean",
            "positive": "positive_expansion",
            "unary": "unary_cohort_density",
            "symmetric": "symmetric_ad_norm",
        }[mutation]
        arms[key] = ArmComparison(0.70, 0.709000000001, 0.009000000001, 10, 1, 0.02, (0, 0, 0, 0))
    arms["ahncr"] = candidate
    if mutation == "null":
        value = Evaluation(arms, value.shuffled_gains, 0.01, value.control_comparisons)
    elif mutation == "direct_control":
        comparisons = dict(value.control_comparisons)
        comparisons["global_mean"] = DirectComparison(5, 5, 1.0)
        value = Evaluation(arms, value.shuffled_gains, value.shuffled_p95, comparisons)
    else:
        value = Evaluation(
            arms,
            value.shuffled_gains,
            value.shuffled_p95,
            value.control_comparisons,
        )
    passed, predicates = decide_ahncr(value)
    assert not passed
    assert not all(predicates.values())


def test_decide_ahncr_accepts_frozen_boundaries() -> None:
    passed, predicates = decide_ahncr(_passing_evaluation())
    assert passed
    assert all(predicates.values())
