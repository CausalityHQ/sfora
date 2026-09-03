from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sfora.verbalizer_margin_f0 import (
    VMD_F0_CALIBER_WINS_GATE,
    VMD_F0_M2_SHA256,
    VMD_F0_M4_QUERY_SHA256,
    VMD_F0_OTHER_WINS_GATE,
    VMD_F0_OVERALL_WINS_GATE,
    VmdF0Observation,
    VmdF0Result,
    build_vmd_f0_candidates,
    canonical_vmd_f0_result_bytes,
)


def _authorities() -> tuple[dict[str, object], dict[str, object], tuple[object, ...]]:
    examples = tuple(
        type(
            "Example",
            (),
            {
                "example_id": f"cars-train-{min(i // 85, 15) + 82}-{i}",
                "label": min(i // 85, 15) + 82,
            },
        )()
        for i in range(1345)
    )
    m4_rows: list[dict[str, object]] = []
    for position, example in enumerate(examples):
        same = next(
            index
            for index, candidate in enumerate(examples)
            if index != position and candidate.label == example.label
        )
        different = next(
            index for index, candidate in enumerate(examples) if candidate.label != example.label
        )
        m4_rows.append(
            {
                "query_position": position,
                "query_example_id": example.example_id,
                "query_label": example.label,
                "nearest_position": different,
                "nearest_example_id": examples[different].example_id,
                "nearest_label": examples[different].label,
                "best_same_position": same,
                "best_different_position": different,
                "nearest_score_bits": 1,
                "best_same_score_bits": 1,
                "best_different_score_bits": 1,
                "margin_bits": 1,
                "correct": False,
            }
        )
    errors = []
    for ordinal in range(103):
        if ordinal < 28:
            position = ordinal
            wrong_label = 83
            wrong = next(
                index for index, example in enumerate(examples) if example.label == wrong_label
            )
        elif ordinal < 63:
            position = 85 + ordinal - 28
            wrong_label = 82
            wrong = next(
                index for index, example in enumerate(examples) if example.label == wrong_label
            )
        else:
            position = 170 + ordinal - 63
            wrong = m4_rows[position]["nearest_position"]
        m4_rows[position]["nearest_position"] = wrong
        m4_rows[position]["nearest_example_id"] = examples[wrong].example_id
        m4_rows[position]["nearest_label"] = examples[wrong].label
        errors.append(
            {
                "query_position": position,
                "query_example_id": examples[position].example_id,
                "query_label": examples[position].label,
                "nearest_position": wrong,
                "nearest_example_id": examples[wrong].example_id,
                "nearest_label": examples[wrong].label,
            }
        )
    errors.sort(key=lambda row: row["query_position"])
    m2 = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "dataset": "cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "dataset_examples_sha256": (
            "83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f"
        ),
        "descriptor_sha256": "4031dc2da90588dcc39005eab92c6c519f3058c581222421ca917501dd3df071",
        "batch_size": 8,
        "query_block": 32,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "class_names": [{"id": value, "name": str(value)} for value in range(82, 98)],
        "cell": "siglip-so400m",
        "model_name": "google/siglip-so400m-patch14-384",
        "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "source_revision": "0" * 40,
        "source_tree_digest": "1" * 64,
        "error_count": 103,
        "errors": errors,
    }
    m4 = {
        "schema": "sfora-pass209-m4-query-evidence-v1",
        "claim_eligible": False,
        "cell": "siglip-so400m",
        "dataset_examples_sha256": m2["dataset_examples_sha256"],
        "dataset_examples_ordered_sha256": "2" * 64,
        "descriptor_file_sha256": (
            "2cb7c25e803ec66fca879f08f77813a4e98c7bb3e52b510e75014c66c203e214"
        ),
        "query_block": 32,
        "rows": m4_rows,
        "historical_cuda_rows": m4_rows,
    }
    return m2, m4, examples


def test_candidates_bind_true_and_wrong_neighbors_before_teacher_scoring() -> None:
    m2, m4, examples = _authorities()
    candidates = build_vmd_f0_candidates(m2, m4, examples)

    assert len(candidates) == 103
    assert tuple(row.ordinal for row in candidates) == tuple(range(103))
    assert sum(row.is_caliber_block for row in candidates) == 63
    assert all(examples[row.true_position].label == row.query_label for row in candidates)
    assert all(examples[row.wrong_position].label != row.query_label for row in candidates)

    broken = json.loads(json.dumps(m4))
    position = candidates[0].query_position
    broken["historical_cuda_rows"][position]["nearest_position"] ^= 1
    with pytest.raises(ValueError, match="wrong neighbor"):
        build_vmd_f0_candidates(m2, broken, examples)


def _observations(*, caliber_wins: int, other_wins: int) -> tuple[VmdF0Observation, ...]:
    rows = []
    for ordinal in range(103):
        block_ordinal = ordinal if ordinal < 63 else ordinal - 63
        win = block_ordinal < (caliber_wins if ordinal < 63 else other_wins)
        rows.append(
            VmdF0Observation(
                source_commit="4" * 40,
                fixture_source_commit="5" * 40,
                model_revision="6" * 40,
                launch_authority_sha256="7" * 64,
                fixture_sha256="8" * 64,
                m2_manifest_sha256=VMD_F0_M2_SHA256,
                m4_query_sha256=VMD_F0_M4_QUERY_SHA256,
                ordinal=ordinal,
                query_position=ordinal,
                query_example_id=f"q-{ordinal}",
                query_label=82 if ordinal < 63 else 84,
                true_position=ordinal + 200,
                true_example_id=f"t-{ordinal}",
                wrong_position=ordinal + 400,
                wrong_example_id=f"w-{ordinal}",
                is_caliber_block=ordinal < 63,
                true_same_score=2.0 if win else 0.0,
                true_different_score=0.0,
                wrong_same_score=0.0 if win else 2.0,
                wrong_different_score=0.0,
                elapsed_ns=1,
                peak_cuda_reserved_bytes=1,
                peak_rss_bytes=1,
            )
        )
    return tuple(rows)


def test_result_requires_overall_and_both_dependence_block_gates() -> None:
    passing = VmdF0Result.from_observations(
        _observations(caliber_wins=38, other_wins=24),
        repeat_checked_ordinals=(0, 102),
        repeat_branch_scores=((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)),
        total_elapsed_ns=1,
    )
    assert passing.overall_wins == VMD_F0_OVERALL_WINS_GATE == 62
    assert passing.caliber_wins == VMD_F0_CALIBER_WINS_GATE == 38
    assert passing.other_wins == VMD_F0_OTHER_WINS_GATE == 24
    assert passing.outcome == "teacher-target-supported"
    assert passing.passed is True

    for rows in (
        _observations(caliber_wins=37, other_wins=25),
        _observations(caliber_wins=39, other_wins=23),
        _observations(caliber_wins=37, other_wins=24),
    ):
        result = VmdF0Result.from_observations(
            rows,
            repeat_checked_ordinals=(0, 102),
            repeat_branch_scores=(
                tuple(
                    getattr(rows[0], name)
                    for name in (
                        "true_same_score",
                        "true_different_score",
                        "wrong_same_score",
                        "wrong_different_score",
                    )
                ),
                tuple(
                    getattr(rows[102], name)
                    for name in (
                        "true_same_score",
                        "true_different_score",
                        "wrong_same_score",
                        "wrong_different_score",
                    )
                ),
            ),
            total_elapsed_ns=1,
        )
        assert result.outcome == "teacher-target-rejected"
        assert result.passed is False


def test_result_recomputes_scores_replays_resources_and_canonical_bytes() -> None:
    base_rows = _observations(caliber_wins=38, other_wins=24)
    rows = (replace(base_rows[0], peak_rss_bytes=49 * 1024**3), *base_rows[1:])
    result = VmdF0Result.from_observations(
        rows,
        repeat_checked_ordinals=(0, 102),
        repeat_branch_scores=(
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0, 0.0),
        ),
        total_elapsed_ns=899_000_000_000,
    )
    raw = canonical_vmd_f0_result_bytes(result)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw)["claim_eligible"] is False
    assert json.loads(raw)["official_test_access"] is False

    with pytest.raises(ValueError, match="replay"):
        VmdF0Result.from_observations(
            rows,
            repeat_checked_ordinals=(0, 102),
            repeat_branch_scores=((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)),
            total_elapsed_ns=1,
        )
    with pytest.raises(ValueError, match="identity"):
        VmdF0Result.from_observations(
            (rows[0], replace(rows[1], model_revision="b" * 40), *rows[2:]),
            repeat_checked_ordinals=(0, 102),
            repeat_branch_scores=((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)),
            total_elapsed_ns=1,
        )
    with pytest.raises(ValueError, match="resource"):
        VmdF0Result.from_observations(
            (replace(rows[0], peak_cuda_reserved_bytes=56 * 1024**3 + 1), *rows[1:]),
            repeat_checked_ordinals=(0, 102),
            repeat_branch_scores=((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)),
            total_elapsed_ns=1,
        )
    with pytest.raises(ValueError, match="resource"):
        VmdF0Result.from_observations(
            (replace(rows[0], peak_rss_bytes=64 * 1024**3 + 1), *rows[1:]),
            repeat_checked_ordinals=(0, 102),
            repeat_branch_scores=((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0, 0.0)),
            total_elapsed_ns=1,
        )
