from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sfora.nested_rank_evaluation as MODULE

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_nested_neighborhood_rank.py"
SPEC = importlib.util.spec_from_file_location("evaluate_sop_nnrl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray(
        (
            (1.0, 0.0),
            (0.9, 0.1),
            (0.0, 1.0),
            (0.1, 0.9),
            (-1.0, 0.0),
            (-0.9, -0.1),
        ),
        dtype=np.float32,
    )
    labels = np.asarray((10, 10, 20, 20, 30, 30), dtype=np.int64)
    sample_ids = np.asarray((101, 102, 201, 202, 301, 302), dtype=np.int64)
    return embeddings, labels, sample_ids


def test_ranked_evidence_excludes_self_and_recomputes_metrics() -> None:
    embeddings, labels, sample_ids = _fixture()

    evidence = MODULE.rank_self_retrieval(embeddings, labels, sample_ids, block_rows=2)
    metrics = MODULE.recompute_self_retrieval(evidence, labels, sample_ids)

    assert len(evidence) == 6
    assert all(row["query_sample_id"] not in row["ranked_sample_ids"] for row in evidence)
    assert all(len(row["ranked_sample_ids"]) == 1 for row in evidence)
    assert metrics == {
        "map_at_r": 1.0,
        "recall_at_1": 1.0,
        "query_count": 6,
    }


def test_metric_recomputation_rejects_membership_rank_and_scalar_drift() -> None:
    embeddings, labels, sample_ids = _fixture()
    evidence = MODULE.rank_self_retrieval(embeddings, labels, sample_ids, block_rows=3)

    for mutate in (
        lambda rows: rows[:-1],
        lambda rows: [{**rows[0], "ranked_sample_ids": [rows[0]["query_sample_id"]]}, *rows[1:]],
        lambda rows: [{**rows[0], "ap_at_r": 0.0}, *rows[1:]],
        lambda rows: [{**rows[0], "query_sample_id": True}, *rows[1:]],
    ):
        with pytest.raises((TypeError, ValueError)):
            MODULE.recompute_self_retrieval(mutate(evidence), labels, sample_ids)


def test_metric_recomputation_accepts_canonical_key_order_and_rejects_duplicate_ids() -> None:
    embeddings, labels, sample_ids = _fixture()
    evidence = MODULE.rank_self_retrieval(embeddings, labels, sample_ids, block_rows=3)
    canonical = json.loads(json.dumps(evidence, sort_keys=True, separators=(",", ":")))

    assert MODULE.recompute_self_retrieval(canonical, labels, sample_ids)["map_at_r"] == 1.0
    duplicate_ids = sample_ids.copy()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="inventory"):
        MODULE.recompute_self_retrieval(evidence, labels, duplicate_ids)


def test_class_bootstrap_and_promotion_are_recomputed() -> None:
    class_ids = np.asarray((10, 10, 20, 20), dtype=np.int64)
    candidate_ap = np.asarray((0.8, 0.9, 0.7, 0.8), dtype=np.float64)
    control_ap = np.asarray((0.7, 0.8, 0.6, 0.7), dtype=np.float64)

    lower = MODULE.class_bootstrap_lower_bound(
        candidate_ap,
        control_ap,
        class_ids,
        seed=17,
        replicates=1_000,
    )
    decision = MODULE.classify_promotion(
        candidate_map_at_r=0.805,
        control_map_at_r=0.800,
        bootstrap_lower_bound=lower,
        candidate_recall_at_1=0.899,
        control_recall_at_1=0.900,
    )

    assert lower > 0.0
    assert decision["status"] == "PROMOTE"
    assert decision["map_at_r_delta"] == pytest.approx(0.005)
    assert decision["recall_at_1_delta"] == pytest.approx(-0.001)
    assert (
        MODULE.classify_promotion(
            candidate_map_at_r=0.8049,
            control_map_at_r=0.800,
            bootstrap_lower_bound=lower,
            candidate_recall_at_1=0.900,
            control_recall_at_1=0.900,
        )["status"]
        == "REJECT"
    )


def test_paired_evaluation_uses_ranked_rows_not_supplied_summaries() -> None:
    embeddings, labels, sample_ids = _fixture()
    control = np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (-0.7, -0.7)),
        dtype=np.float32,
    )
    candidate_rows = MODULE.rank_self_retrieval(embeddings, labels, sample_ids)
    control_rows = MODULE.rank_self_retrieval(control, labels, sample_ids)

    result = MODULE.evaluate_paired_rankings(
        candidate_rows,
        control_rows,
        labels,
        sample_ids,
        seed=17,
        replicates=1_000,
    )

    assert result["candidate"]["map_at_r"] == 1.0
    assert result["control"]["map_at_r"] < 1.0
    assert result["decision"]["status"] == "PROMOTE"
    tampered = [{**candidate_rows[0], "ap_at_r": 0.0}, *candidate_rows[1:]]
    with pytest.raises(ValueError, match="query metric evidence differs"):
        MODULE.evaluate_paired_rankings(
            tampered,
            control_rows,
            labels,
            sample_ids,
            seed=17,
            replicates=1_000,
        )


def test_bounded_top_r_matches_full_stable_oracle_with_ties() -> None:
    rng = np.random.Generator(np.random.PCG64(91))
    sample_ids = rng.permutation(np.arange(100, 300, dtype=np.int64))
    distances = rng.normal(size=200).astype(np.float64)
    distances[20:35] = 0.25

    for retained in (1, 7, 31, 199):
        expected = np.lexsort((sample_ids, distances))[:retained]
        actual = MODULE._bounded_top_indices(distances, sample_ids, retained)
        assert np.array_equal(actual, expected)


def test_int8_self_retrieval_uses_exact_codes_norms_and_sample_id_ties() -> None:
    embeddings, labels, sample_ids = _fixture()
    rows = MODULE.rank_self_retrieval_int8(
        embeddings,
        labels,
        sample_ids,
        block_rows=2,
    )

    normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    codes = np.clip(np.rint(normalized * 127), -127, 127).astype(np.int8)
    norms = np.linalg.norm(codes.astype(np.float64), axis=1)
    for query_index, row in enumerate(rows):
        scores = (codes[query_index].astype(np.int32) @ codes.astype(np.int32).T).astype(
            np.float64
        ) / (norms[query_index] * norms)
        scores[query_index] = -np.inf
        expected = np.lexsort((sample_ids, -scores))[:1]
        assert row["ranked_sample_ids"] == [int(sample_ids[expected[0]])]


def test_query_gallery_ranking_is_generic_and_deterministic() -> None:
    query = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    query_labels = np.asarray((10, 20), dtype=np.int64)
    query_ids = np.asarray((1, 2), dtype=np.int64)
    gallery = np.asarray(((0.9, 0.1), (0.8, 0.2), (0.1, 0.9), (0.2, 0.8)), dtype=np.float32)
    gallery_labels = np.asarray((10, 10, 20, 20), dtype=np.int64)
    gallery_ids = np.asarray((11, 12, 21, 22), dtype=np.int64)

    evidence = MODULE.rank_query_gallery(
        query,
        query_labels,
        query_ids,
        gallery,
        gallery_labels,
        gallery_ids,
        block_rows=1,
    )

    assert [row["ranked_sample_ids"] for row in evidence] == [[11, 12], [21, 22]]
    assert all(row["ap_at_r"] == 1.0 for row in evidence)
    with pytest.raises(ValueError, match="inventory"):
        MODULE.rank_query_gallery(
            query,
            np.asarray((10, 30), dtype=np.int64),
            query_ids,
            gallery,
            gallery_labels,
            gallery_ids,
        )


def test_training_artifact_loader_authenticates_result_and_model(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    torch.save(
        {
            "encoder": {"weight": torch.ones(2)},
            "head": {"weight": torch.ones(3)},
            "raw_proxies": torch.ones(2, 3),
        },
        model,
    )
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    result = {
        "schema": "sfora-nnrl-sop-training-result-v1",
        "claim_eligible": False,
        "status": "COMPLETE",
        "arm": "combined",
        "split_seed": 17,
        "temperature": 0.05,
        "optimization_rows": 80,
        "optimization_classes": 8,
        "epochs": 10,
        "history": [
            {
                "epoch": epoch,
                "steps": 1,
                "attempted_steps": 1,
                "skipped_updates": 0,
                "mean_loss": 0.5,
            }
            for epoch in range(1, 11)
        ],
        "authority": {"ordered_train_record_sha256": "a" * 64},
        "model_artifact": {"path": "model.pt", "sha256": model_sha, "bytes": model.stat().st_size},
        "run_receipt": {"path": "run-receipt.json", "sha256": "b" * 64, "bytes": 10},
    }
    result_path = tmp_path / "RESULT_COMPLETE.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()

    loaded_result, loaded_model = SCRIPT_MODULE.load_training_artifact(
        result_path,
        result_sha,
        model,
    )

    assert loaded_result == result
    assert set(loaded_model) == {"encoder", "head", "raw_proxies"}
    with pytest.raises(ValueError, match="result digest"):
        SCRIPT_MODULE.load_training_artifact(result_path, "0" * 64, model)
    result["model_artifact"]["sha256"] = "0" * 64
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model authority"):
        SCRIPT_MODULE.load_training_artifact(
            result_path,
            hashlib.sha256(result_path.read_bytes()).hexdigest(),
            model,
        )


def test_candidate_state_restores_encoder_and_head_strictly() -> None:
    encoder = nn.Linear(2, 3, bias=False)
    head = nn.Linear(3, 2, bias=False)
    expected_encoder = torch.full_like(encoder.weight, 2.0)
    expected_head = torch.full_like(head.weight, 3.0)
    artifact = {
        "encoder": {"weight": expected_encoder},
        "head": {"weight": expected_head},
        "raw_proxies": torch.ones(4, 2),
    }

    proxies = SCRIPT_MODULE.restore_candidate_state(encoder, head, artifact)

    assert torch.equal(encoder.weight, expected_encoder)
    assert torch.equal(head.weight, expected_head)
    assert torch.equal(proxies, artifact["raw_proxies"])
    with pytest.raises(ValueError, match="model state"):
        SCRIPT_MODULE.restore_candidate_state(
            encoder,
            head,
            {**artifact, "encoder": {"wrong": torch.ones(1)}},
        )


def test_three_way_evaluation_uses_identical_rows_and_int8_candidate() -> None:
    candidate, labels, sample_ids = _fixture()
    source = np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (-0.7, -0.7)),
        dtype=np.float32,
    )
    teacher = candidate.copy()

    result = SCRIPT_MODULE.evaluate_three_way_embeddings(
        candidate,
        source,
        teacher,
        labels,
        sample_ids,
    )

    assert result["candidate_float"]["metrics"]["map_at_r"] == 1.0
    assert result["candidate_int8"]["metrics"]["map_at_r"] == 1.0
    assert result["source"]["metrics"]["map_at_r"] < 1.0
    assert result["teacher"]["metrics"]["map_at_r"] == 1.0
    for arm in result.values():
        assert [row["query_sample_id"] for row in arm["queries"]] == sample_ids.tolist()
