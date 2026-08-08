"""Focused tests for the preregistered Pass159 Stage-A diagnostic."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_pass159_cotangent_stage_a.py"
)
_SPEC = importlib.util.spec_from_file_location("diagnose_pass159_cotangent_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_angular_proxy_anchor_cotangent_matches_singleton_autograd() -> None:
    rng = np.random.default_rng(159)
    z_np = _unit(rng.normal(size=(1, 7)))[0]
    proxies_np = _unit(rng.normal(size=(5, 7)))
    proxy_labels = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)

    analytic = _MODULE.angular_proxy_anchor_cotangent(
        z_np,
        30,
        proxies_np,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
    )

    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    proxies = torch.tensor(proxies_np, dtype=torch.float64)
    similarities = z @ proxies.T
    own = similarities[2]
    positive = torch.nn.functional.softplus(32.0 * (0.1 - own))
    foreign = torch.cat((similarities[:2], similarities[3:]))
    negative = torch.nn.functional.softplus(32.0 * (foreign + 0.1)).mean()
    ambient = torch.autograd.grad(positive + negative, z)[0]
    expected = ambient - torch.dot(ambient, z) * z

    np.testing.assert_allclose(analytic, expected.detach().numpy(), atol=1e-11, rtol=1e-11)
    assert float(np.dot(analytic, z_np)) == pytest.approx(0.0, abs=1e-11)


def test_parallel_transport_preserves_norm_and_target_tangency() -> None:
    origin = np.asarray([1.0, 0.0, 0.0])
    target = np.asarray([0.0, 1.0, 0.0])
    tangent = np.asarray([0.0, 0.6, 0.8])

    transported = _MODULE.parallel_transport(tangent, origin, target)

    assert float(np.dot(transported, target)) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(tangent), abs=1e-12)


def test_parallel_transport_rejects_antipodal_or_zero_inputs() -> None:
    with pytest.raises(ValueError, match="antipodal"):
        _MODULE.parallel_transport(
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="unit"):
        _MODULE.parallel_transport(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
        )


def test_smooth_margin_gradient_matches_autograd_with_frozen_foreign_set() -> None:
    receiver = _unit(np.asarray([[1.0, 0.3, -0.2]]))[0]
    positives = _unit(np.asarray([[0.9, 0.2, 0.1], [0.7, 0.4, -0.1]]))
    foreign = _unit(
        np.asarray(
            [
                [0.2, 0.9, 0.1],
                [-0.1, 0.8, 0.3],
                [0.4, -0.2, 0.9],
            ]
        )
    )

    analytic = _MODULE.smooth_margin_gradient(receiver, positives, foreign, tau=0.05)

    z = torch.tensor(receiver, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(positives, dtype=torch.float64)
    neg = torch.tensor(foreign, dtype=torch.float64)
    margin = 0.05 * torch.logsumexp((z @ pos.T) / 0.05, dim=0)
    margin -= 0.05 * torch.logsumexp((z @ neg.T) / 0.05, dim=0)
    ambient = torch.autograd.grad(margin, z)[0]
    expected = ambient - torch.dot(ambient, z) * z

    np.testing.assert_allclose(analytic, expected.detach().numpy(), atol=1e-11, rtol=1e-11)


def test_partition_identity_is_input_order_invariant_and_disjoint() -> None:
    ids = np.asarray(["img-c", "img-a", "img-e", "img-b", "img-d", "img-f"])
    support, controllers = _MODULE.partition_identity(ids)
    support_ids = set(ids[support])
    controller_ids = set(ids[controllers])

    permutation = np.asarray([4, 2, 0, 5, 1, 3])
    shuffled = ids[permutation]
    support_2, controllers_2 = _MODULE.partition_identity(shuffled)

    assert support_ids == set(shuffled[support_2])
    assert controller_ids == set(shuffled[controllers_2])
    assert len(support_ids) == 2
    assert support_ids.isdisjoint(controller_ids)


def test_partition_identity_requires_two_supports_and_three_controllers() -> None:
    with pytest.raises(ValueError, match="at least five"):
        _MODULE.partition_identity(np.asarray(["a", "b", "c", "d"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_final_pack(
    path: Path,
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    split: str,
    checkpoint_sha256: str,
    report_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(ids),
        source_paths=np.asarray([f"/{split}/{value}.jpg" for value in ids]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        report_sha256=np.asarray(report_sha256),
    )


def _synthetic_bound_artifacts(
    root: Path,
    *,
    prehead_label_mismatch: bool = False,
    reported_r1: float = 1.0,
) -> dict[str, dict[str, str]]:
    seed = 0
    config = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": seed,
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "checkpoint_selection_interval": 0,
        "backbone_name": "bn_inception",
        "head_pooling": "avg_max",
    }
    train = np.asarray(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.1],
            [0.7, 0.3, -0.1],
            [0.6, 0.4, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, -0.1],
            [0.3, 0.7, 0.1],
            [0.4, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([10] * 5 + [20] * 5, dtype=np.int64)
    query = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    query_labels = np.asarray([30, 40], dtype=np.int64)
    gallery = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float32)
    gallery_labels = query_labels.copy()
    weight = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    bias = np.asarray([0.0, 0.0], dtype=np.float32)

    def embed(rows: np.ndarray) -> np.ndarray:
        raw = rows @ weight.T + bias
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    report = {
        "config": config,
        "methods": {
            "proxy_anchor_end_to_end:bn_inception": {
                "dimensions": 2,
                "recall_at_1": reported_r1,
            }
        },
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    checkpoint_path = root / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                "model.embedding.weight": torch.tensor(weight),
                "model.embedding.bias": torch.tensor(bias),
                "metric_proxies": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "metric_proxy_labels": torch.tensor([10, 20]),
            },
            "arch": {"backbone_name": "bn_inception", "head_pooling": "avg_max"},
            "artifact_selection": "final_training_state",
            "evaluation_model_source": "trained_model",
            "training_config": config,
            "training_step": 10,
        },
        checkpoint_path,
    )
    checkpoint_digest = _sha256(checkpoint_path)
    report_digest = _sha256(report_path)

    prehead_path = root / "prehead.npz"
    written_train_labels = train_labels.copy()
    if prehead_label_mismatch:
        written_train_labels[[0, 5]] = written_train_labels[[5, 0]]
    np.savez_compressed(
        prehead_path,
        train=train,
        train_labels=written_train_labels,
        query=query,
        query_labels=query_labels,
        gallery=gallery,
        gallery_labels=gallery_labels,
    )
    paths = {
        "prehead_npz": prehead_path,
        "checkpoint_pt": checkpoint_path,
        "report_json": report_path,
        "train_npz": root / "train.npz",
        "query_npz": root / "query.npz",
        "gallery_npz": root / "gallery.npz",
        "retrieval_json": root / "retrieval.json",
    }
    _write_final_pack(
        paths["train_npz"],
        embeddings=embed(train),
        labels=train_labels,
        ids=[f"train-{index}" for index in range(len(train))],
        split="train",
        checkpoint_sha256=checkpoint_digest,
        report_sha256=report_digest,
    )
    _write_final_pack(
        paths["query_npz"],
        embeddings=embed(query),
        labels=query_labels,
        ids=["query-0", "query-1"],
        split="query",
        checkpoint_sha256=checkpoint_digest,
        report_sha256=report_digest,
    )
    _write_final_pack(
        paths["gallery_npz"],
        embeddings=embed(gallery),
        labels=gallery_labels,
        ids=["gallery-0", "gallery-1"],
        split="gallery",
        checkpoint_sha256=checkpoint_digest,
        report_sha256=report_digest,
    )
    paths["retrieval_json"].write_text(
        json.dumps(
            {
                "artifact_selection": "final_training_state",
                "checkpoint_sha256": checkpoint_digest,
                "report_sha256": report_digest,
                "resolved_training_steps": 10,
                "reported_final_recall_at_1": reported_r1,
                "independent_recall_at_1": 1.0,
                "canonical_float64_euclidean_recall_at_1": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
    }


def test_load_bound_seed_reconstructs_head_and_binds_all_rows(tmp_path: Path) -> None:
    entry = _synthetic_bound_artifacts(tmp_path)

    bound = _MODULE.load_bound_seed(
        entry,
        seed=0,
        expected_partition={"train": (10, 2), "query": (2, 2), "gallery": (2, 2)},
    )

    assert bound.seed == 0
    assert bound.train_embeddings.shape == (10, 2)
    assert bound.train_raw_norms.shape == (10,)
    assert bound.train_example_ids.tolist() == [f"train-{index}" for index in range(10)]
    assert bound.official_recall_at_1 == pytest.approx(1.0)
    assert bound.alpha == 32.0
    assert bound.delta == 0.1


def test_load_bound_seed_fails_closed_on_manifest_digest_mismatch(tmp_path: Path) -> None:
    entry = _synthetic_bound_artifacts(tmp_path)
    entry["prehead_npz"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="manifest SHA-256"):
        _MODULE.load_bound_seed(
            entry,
            seed=0,
            expected_partition={"train": (10, 2), "query": (2, 2), "gallery": (2, 2)},
        )


def test_load_bound_seed_fails_closed_on_prehead_label_order_mismatch(tmp_path: Path) -> None:
    entry = _synthetic_bound_artifacts(tmp_path, prehead_label_mismatch=True)

    with pytest.raises(ValueError, match="train labels differ"):
        _MODULE.load_bound_seed(
            entry,
            seed=0,
            expected_partition={"train": (10, 2), "query": (2, 2), "gallery": (2, 2)},
        )


def test_load_bound_seed_fails_closed_on_reported_r1_mismatch(tmp_path: Path) -> None:
    entry = _synthetic_bound_artifacts(tmp_path, reported_r1=0.5)

    with pytest.raises(ValueError, match="official R@1"):
        _MODULE.load_bound_seed(
            entry,
            seed=0,
            expected_partition={"train": (10, 2), "query": (2, 2), "gallery": (2, 2)},
        )


def _synthetic_bound_seed() -> object:
    rng = np.random.default_rng(1590)
    labels = np.repeat(np.asarray([10, 20, 30], dtype=np.int64), 6)
    ids = np.asarray([f"identity-{label}-image-{index}" for label in (10, 20, 30) for index in range(6)])
    centers = _unit(rng.normal(size=(3, 8)))
    embeddings = []
    for class_index in range(3):
        rows = centers[class_index] + 0.25 * rng.normal(size=(6, 8))
        embeddings.append(_unit(rows))
    descriptors = np.concatenate(embeddings, axis=0)
    raw_norms = np.concatenate(
        [np.asarray([0.7, 0.9, 1.1, 1.3, 1.5, 1.7]) + 0.03 * class_index for class_index in range(3)]
    )
    proxies = _unit(centers + 0.1 * rng.normal(size=centers.shape))
    return _MODULE.BoundSeed(
        seed=0,
        train_embeddings=descriptors,
        train_raw_norms=raw_norms,
        train_labels=labels,
        train_example_ids=ids,
        proxies=proxies,
        proxy_labels=np.asarray([10, 20, 30], dtype=np.int64),
        alpha=32.0,
        delta=0.1,
        official_recall_at_1=1.0,
        artifact_binding={},
    )


def test_compute_seed_rows_keeps_outcomes_disjoint_and_builds_all_controls() -> None:
    bound = _synthetic_bound_seed()

    result = _MODULE.compute_seed_rows(bound, top_k=2)

    assert result["eligible_identities"] == 3
    assert result["excluded_identities"] == 0
    assert len(result["identity_rows"]) == 3
    expected_arms = {
        "candidate",
        "receiver_own",
        "fixed_hash_donor",
        "norm_permuted_donor",
        "cosine_matched_donor",
        "ambient_projection",
        "proxy_only",
    }
    for row in result["identity_rows"]:
        support_ids = set(row["support_ids"])
        controller_ids = set(row["controller_ids"])
        assert support_ids.isdisjoint(controller_ids)
        assert row["receiver_id"] in controller_ids
        assert row["candidate_donor_id"] in controller_ids
        assert set(row["alignments"]) == expected_arms
        assert all(np.isfinite(list(row["alignments"].values())))
        assert len(row["foreign_support_ids"]) == 2
        assert all(not value.startswith(f"identity-{row['label']}-") for value in row["foreign_support_ids"])
        assert 0.0 <= row["orthogonal_fraction"] <= 1.0


def test_compute_seed_rows_is_invariant_to_global_input_order() -> None:
    bound = _synthetic_bound_seed()
    original = _MODULE.compute_seed_rows(bound, top_k=2)
    order = np.random.default_rng(88).permutation(len(bound.train_labels))
    shuffled = _MODULE.BoundSeed(
        seed=bound.seed,
        train_embeddings=bound.train_embeddings[order],
        train_raw_norms=bound.train_raw_norms[order],
        train_labels=bound.train_labels[order],
        train_example_ids=bound.train_example_ids[order],
        proxies=bound.proxies,
        proxy_labels=bound.proxy_labels,
        alpha=bound.alpha,
        delta=bound.delta,
        official_recall_at_1=bound.official_recall_at_1,
        artifact_binding=bound.artifact_binding,
    )
    reordered = _MODULE.compute_seed_rows(shuffled, top_k=2)

    by_label = {row["label"]: row for row in original["identity_rows"]}
    shuffled_by_label = {row["label"]: row for row in reordered["identity_rows"]}
    assert by_label.keys() == shuffled_by_label.keys()
    for label in by_label:
        left = by_label[label]
        right = shuffled_by_label[label]
        assert left["support_ids"] == right["support_ids"]
        assert left["receiver_id"] == right["receiver_id"]
        assert left["candidate_donor_id"] == right["candidate_donor_id"]
        assert left["foreign_support_ids"] == right["foreign_support_ids"]
        assert left["alignments"] == pytest.approx(right["alignments"], abs=1e-12)
