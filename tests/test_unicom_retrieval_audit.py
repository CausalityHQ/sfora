from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import sfora.unicom_retrieval_audit as retrieval_audit
from sfora.unicom_inshop import InshopRecord
from sfora.unicom_retrieval_audit import (
    _stable_top_indices,
    audit_deployment_geometry,
    canonical_logical_record,
    geometry_decision,
    l2_normalize,
    paired_r1_interval,
    query_evidence,
    random_masks,
    recompute_query_metrics,
    retrieval_metrics_from_score_chunks,
    retrieval_view,
    validate_evaluation_evidence,
    write_evaluation_evidence,
)


def _evidence_records(
    dataset_root: Path, *, gallery_count: int = 31
) -> tuple[tuple[InshopRecord, ...], tuple[InshopRecord, ...]]:
    image_root = dataset_root / "Img"
    image_root.mkdir(parents=True)
    query_path = image_root / "WOMEN" / "query.jpg"
    query_path.parent.mkdir()
    query_path.write_bytes(b"query")
    query = (InshopRecord(split="query", image_path=query_path, label="item_a"),)
    gallery = []
    for index in range(gallery_count):
        path = image_root / "MEN" / f"gallery-{index:02d}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"gallery-{index}".encode())
        label = "item_a" if index in {0, 2} else f"other-{index}"
        gallery.append(InshopRecord(split="gallery", image_path=path, label=label))
    return query, tuple(gallery)


def _full_width_evidence_values(gallery_count: int = 31) -> tuple[np.ndarray, np.ndarray]:
    query = np.zeros((1, 768), dtype=np.float32)
    query[0, 0] = 1.0
    angles = np.linspace(0.0, 1.5, gallery_count, dtype=np.float32)
    gallery = np.zeros((gallery_count, 768), dtype=np.float32)
    gallery[:, 0] = np.cos(angles)
    gallery[:, 1] = np.sin(angles)
    return query, gallery


def _ranked_rows(receipt: dict[str, object], evidence_root: Path) -> list[dict[str, object]]:
    binding = receipt["ranked_prefix_evidence"]
    return json.loads((evidence_root / binding["path"]).read_bytes())


def _rewrite_ranked_rows(
    receipt: dict[str, object], evidence_root: Path, rows: list[dict[str, object]]
) -> None:
    payload = (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode()
    path = evidence_root / receipt["ranked_prefix_evidence"]["path"]
    path.write_bytes(payload)
    receipt["ranked_prefix_evidence"]["sha256"] = hashlib.sha256(payload).hexdigest()
    receipt["ranked_prefix_evidence"]["bytes"] = len(payload)


def test_query_evidence_recomputes_all_metrics_and_ranked_prefix(tmp_path: Path) -> None:
    query_records, gallery_records = _evidence_records(tmp_path / "dataset")
    query, gallery = _full_width_evidence_values()

    rows = query_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=tmp_path / "dataset",
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    expected = retrieval_view(
        query,
        gallery,
        np.asarray([record.label for record in query_records]),
        np.asarray([record.label for record in gallery_records]),
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )

    assert type(rows) is tuple
    assert len(rows[0]["ranked_prefix"]) == max(
        30, rows[0]["relevant_gallery_count"]
    )
    assert rows[0]["query_path"] == "WOMEN/query.jpg"
    assert rows[0]["ranked_prefix"][0] == {
        "gallery_index": 0,
        "gallery_path": "MEN/gallery-00.jpg",
        "gallery_label": "item_a",
        "score": 0.0,
        "correct": True,
    }
    metrics = recompute_query_metrics(rows)
    assert metrics == {
        "recall_at_1": expected.recall[1],
        "recall_at_10": expected.recall[10],
        "recall_at_20": expected.recall[20],
        "recall_at_30": expected.recall[30],
        "map_at_r": expected.map_at_r,
    }


def test_query_evidence_recomputes_metrics_for_gallery_smaller_than_recall_k(
    tmp_path: Path,
) -> None:
    query_records, gallery_records = _evidence_records(
        tmp_path / "dataset", gallery_count=5
    )
    query, gallery = _full_width_evidence_values(gallery_count=5)
    rows = query_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=tmp_path / "dataset",
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    expected = retrieval_view(
        query,
        gallery,
        np.asarray(["item_a"]),
        np.asarray([record.label for record in gallery_records]),
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )

    assert len(rows[0]["ranked_prefix"]) == 5
    assert recompute_query_metrics(rows)["map_at_r"] == expected.map_at_r


def test_logical_records_and_query_hashes_are_relocation_stable(tmp_path: Path) -> None:
    query, gallery = _full_width_evidence_values()
    payloads = []
    receipts = []
    for root_name in ("first/absolute/location", "second"):
        root = tmp_path / root_name
        query_records, gallery_records = _evidence_records(root)
        logical = (
            [canonical_logical_record(record, root).__dict__ for record in query_records],
            [canonical_logical_record(record, root).__dict__ for record in gallery_records],
        )
        rows = query_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
        )
        payloads.append(
            json.dumps({"inventory": logical, "rows": rows}, sort_keys=True).encode()
        )
        evidence_root = tmp_path / f"evidence-{len(receipts)}"
        evidence_root.mkdir()
        write_evaluation_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
            epoch=4,
            evidence_root=evidence_root,
        )
        receipts.append((evidence_root / "evaluation-epoch-0004.json").read_bytes())

    assert payloads[0] == payloads[1]
    assert receipts[0] == receipts[1]


def test_canonical_logical_record_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "Img").mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    escaped = InshopRecord(split="query", image_path=outside, label="item_a")
    with pytest.raises(ValueError, match="image path"):
        canonical_logical_record(escaped, root)

    link = root / "Img" / "linked.jpg"
    link.symlink_to(outside)
    linked = InshopRecord(split="query", image_path=link, label="item_a")
    with pytest.raises(ValueError, match="image path"):
        canonical_logical_record(linked, root)


def test_query_evidence_uses_full_normalize_then_prefix_geometry(tmp_path: Path) -> None:
    query_records, gallery_records = _evidence_records(tmp_path / "dataset")
    query = np.zeros((1, 768), dtype=np.float32)
    gallery = np.zeros((31, 768), dtype=np.float32)
    query[0, [0, 1, 512]] = (3.0, 4.0, 12.0)
    gallery[0, [0, 1, 512]] = (3.0, 4.0, 0.0)
    gallery[1, [0, 1, 512]] = (0.0, 5.0, 12.0)
    gallery[2:, 0] = -1.0

    official = query_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=tmp_path / "dataset",
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    prefix_then_normalize = retrieval_view(
        query,
        gallery,
        np.asarray(["item_a"]),
        np.asarray([record.label for record in gallery_records]),
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=False,
    )

    assert official[0]["ranked_prefix"][0]["gallery_index"] != int(
        prefix_then_normalize.top1_indices[0]
    )


def test_query_evidence_ranks_squared_distance_not_dot_product(tmp_path: Path) -> None:
    query_records, gallery_records = _evidence_records(tmp_path / "dataset")
    query = np.zeros((1, 768), dtype=np.float32)
    gallery = np.zeros((31, 768), dtype=np.float32)
    query[0, 0] = 1.0
    gallery[0, [0, 1, 512]] = (0.8, 0.0, 0.6)
    gallery[1, [0, 1, 512]] = (0.9, np.sqrt(0.19), 0.0)
    gallery[2:, 0] = -1.0

    rows = query_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=tmp_path / "dataset",
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    selected_query = l2_normalize(query)[:, :512]
    selected_gallery = l2_normalize(gallery)[:, :512]
    dot_order = np.lexsort(
        (np.arange(gallery.shape[0]), -(selected_query @ selected_gallery.T)[0])
    )

    assert rows[0]["ranked_prefix"][0]["gallery_index"] == 0
    assert int(dot_order[0]) == 1


def _persisted_evaluation_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    dataset_root = tmp_path / "dataset-retry"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    receipt = write_evaluation_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=dataset_root,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
        epoch=4,
        evidence_root=evidence_root,
    )
    return receipt, evidence_root


def test_per_query_evaluation_receipt_strictly_reloads_descriptor_preimages(
    tmp_path: Path,
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)

    validate_evaluation_evidence(receipt, evidence_root)
    persisted = json.loads((evidence_root / "evaluation-epoch-0004.json").read_text())
    validate_evaluation_evidence(persisted, evidence_root)
    query = np.load(evidence_root / persisted["query_descriptors"]["path"])
    gallery = np.load(evidence_root / persisted["gallery_descriptors"]["path"])
    assert query.dtype == gallery.dtype == np.float32
    assert query.shape == (1, 768)
    assert gallery.shape == (31, 768)
    assert query.flags.c_contiguous and gallery.flags.c_contiguous


@pytest.mark.parametrize(
    "mutation",
    (
        "top_label",
        "relevant_count",
        "ap_at_r",
        "gallery_index",
        "score",
        "record_order",
        "record_path",
        "duplicate_path",
        "aggregate",
        "descriptor_hash",
        "evaluation_descriptor_hash",
        "operation",
    ),
)
def test_per_query_evaluation_receipt_rejects_derived_mutation(
    tmp_path: Path, mutation: str
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    changed = copy.deepcopy(receipt)
    ranked = _ranked_rows(receipt, evidence_root)
    if mutation == "top_label":
        ranked[0]["ranked_prefix"][0]["gallery_label"] = "wrong"
    elif mutation == "relevant_count":
        ranked[0]["relevant_gallery_count"] = 3
    elif mutation == "ap_at_r":
        ranked[0]["ap_at_r"] = 0.25
    elif mutation == "gallery_index":
        ranked[0]["ranked_prefix"][0]["gallery_index"] = 1
    elif mutation == "score":
        ranked[0]["ranked_prefix"][0]["score"] = 1.0
    elif mutation == "record_order":
        changed["gallery_records"][0], changed["gallery_records"][1] = (
            changed["gallery_records"][1],
            changed["gallery_records"][0],
        )
    elif mutation == "record_path":
        changed["gallery_records"][0]["image_name"] = "MEN/rebased.jpg"
    elif mutation == "duplicate_path":
        changed["gallery_records"][1]["image_name"] = changed["gallery_records"][0][
            "image_name"
        ]
    elif mutation == "aggregate":
        changed["metrics"]["recall_at_1"] = 0.0
    elif mutation == "descriptor_hash":
        changed["query_descriptors"]["sha256"] = "0" * 64
    elif mutation == "evaluation_descriptor_hash":
        changed["evaluation_signature"]["descriptor_sha256"] = "0" * 64
    else:
        changed["evaluation_signature"]["operations"][-1] = "dot_product"
    if mutation in {"top_label", "relevant_count", "ap_at_r", "gallery_index", "score"}:
        _rewrite_ranked_rows(changed, evidence_root, ranked)

    with pytest.raises(ValueError):
        validate_evaluation_evidence(changed, evidence_root)


@pytest.mark.parametrize(
    "mutation",
    (
        "geometry_dimension_float",
        "coordinate_bool",
        "signature_dimension_float",
        "score_bool",
        "index_bool",
        "correct_int",
        "relevant_count_float",
        "aggregate_bool",
    ),
)
def test_per_query_evaluation_receipt_rejects_type_only_mutation(
    tmp_path: Path, mutation: str
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    changed = copy.deepcopy(receipt)
    ranked = _ranked_rows(receipt, evidence_root)
    if mutation == "geometry_dimension_float":
        changed["geometry"]["input_dimension"] = 768.0
    elif mutation == "coordinate_bool":
        changed["geometry"]["coordinates"][0] = False
    elif mutation == "signature_dimension_float":
        changed["evaluation_signature"]["descriptor_dimension"] = 512.0
    elif mutation == "score_bool":
        ranked[0]["ranked_prefix"][0]["score"] = False
    elif mutation == "index_bool":
        ranked[0]["ranked_prefix"][0]["gallery_index"] = False
    elif mutation == "correct_int":
        ranked[0]["ranked_prefix"][0]["correct"] = 1
    elif mutation == "relevant_count_float":
        ranked[0]["relevant_gallery_count"] = 2.0
    else:
        changed["metrics"]["recall_at_1"] = True
    if mutation in {"score_bool", "index_bool", "correct_int", "relevant_count_float"}:
        _rewrite_ranked_rows(changed, evidence_root, ranked)

    with pytest.raises(ValueError):
        validate_evaluation_evidence(changed, evidence_root)


@pytest.mark.parametrize(
    "alias",
    (
        "MEN/./gallery-00.jpg",
        "MEN//gallery-00.jpg",
        "MEN/gallery-00.jpg/",
        "MEN\\gallery-00.jpg",
        "C:/MEN/gallery-00.jpg",
    ),
)
def test_per_query_evaluation_receipt_rejects_noncanonical_logical_path_alias(
    tmp_path: Path, alias: str
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    receipt["gallery_records"][1]["image_name"] = alias
    rows = _ranked_rows(receipt, evidence_root)
    for ranked in rows[0]["ranked_prefix"]:
        if ranked["gallery_index"] == 1:
            ranked["gallery_path"] = alias
    _rewrite_ranked_rows(receipt, evidence_root, rows)

    with pytest.raises(ValueError, match="inventory|path"):
        validate_evaluation_evidence(receipt, evidence_root)


@pytest.mark.parametrize("field", ("query", "ranked"))
def test_recompute_query_metrics_rejects_noncanonical_path_alias(
    tmp_path: Path, field: str
) -> None:
    query_records, gallery_records = _evidence_records(tmp_path / "dataset")
    query, gallery = _full_width_evidence_values()
    rows = query_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=tmp_path / "dataset",
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    changed = copy.deepcopy(rows)
    if field == "query":
        changed[0]["query_path"] = "WOMEN//query.jpg"
    else:
        changed[0]["ranked_prefix"][0]["gallery_path"] = "MEN/./gallery-00.jpg"

    with pytest.raises(ValueError, match="evidence"):
        recompute_query_metrics(changed)


def test_per_query_evaluation_receipt_rejects_descriptor_byte_mutation(
    tmp_path: Path,
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    descriptor = evidence_root / receipt["gallery_descriptors"]["path"]
    payload = bytearray(descriptor.read_bytes())
    payload[-1] ^= 1
    descriptor.write_bytes(payload)

    with pytest.raises(ValueError, match="descriptor"):
        validate_evaluation_evidence(receipt, evidence_root)


def test_per_query_evaluation_receipt_rejects_fortran_order_descriptor(
    tmp_path: Path,
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    descriptor = evidence_root / receipt["gallery_descriptors"]["path"]
    values = np.asfortranarray(np.load(descriptor, allow_pickle=False))
    with descriptor.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    receipt["gallery_descriptors"]["sha256"] = hashlib.sha256(
        descriptor.read_bytes()
    ).hexdigest()
    receipt["gallery_descriptors"]["bytes"] = descriptor.stat().st_size

    with pytest.raises(ValueError, match="descriptor bytes"):
        validate_evaluation_evidence(receipt, evidence_root)


def test_per_query_evaluation_receipt_rejects_escaping_and_symlink_paths(
    tmp_path: Path,
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    absolute = copy.deepcopy(receipt)
    absolute["query_descriptors"]["path"] = str(
        evidence_root / receipt["query_descriptors"]["path"]
    )
    with pytest.raises(ValueError, match="descriptor path"):
        validate_evaluation_evidence(absolute, evidence_root)

    target = evidence_root / receipt["query_descriptors"]["path"]
    target.rename(evidence_root / "moved.npy")
    target.symlink_to(evidence_root / "moved.npy")
    with pytest.raises(ValueError, match="descriptor path"):
        validate_evaluation_evidence(receipt, evidence_root)


def test_per_query_evaluation_receipt_cross_binds_descriptor_paths_to_epoch(
    tmp_path: Path,
) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    source = evidence_root / receipt["query_descriptors"]["path"]
    substitute = evidence_root / "arbitrary-query.npy"
    substitute.write_bytes(source.read_bytes())
    receipt["query_descriptors"]["path"] = substitute.name

    with pytest.raises(ValueError, match="descriptor path"):
        validate_evaluation_evidence(receipt, evidence_root)


def test_per_query_evaluation_artifacts_are_immutable(tmp_path: Path) -> None:
    receipt, evidence_root = _persisted_evaluation_fixture(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in evidence_root.iterdir()
        if path.is_file()
    }
    dataset_root = tmp_path / "dataset"
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()

    with pytest.raises(FileExistsError):
        write_evaluation_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=dataset_root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
            epoch=4,
            evidence_root=evidence_root,
        )

    assert receipt["epoch"] == 4
    assert {
        path.name: path.read_bytes()
        for path in evidence_root.iterdir()
        if path.is_file()
    } == before


def test_official_and_prefix_unit_use_different_normalization_order() -> None:
    query = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    gallery = np.array(
        [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]],
        dtype=np.float32,
    )
    query_labels = np.array(["a"])
    gallery_labels = np.array(["a", "b"])

    official = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.array([0, 1]),
        normalize_before=True,
    )
    corrected = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert official.top1_indices.tolist() != corrected.top1_indices.tolist()


def test_random_masks_are_sorted_unique_and_seed_exact() -> None:
    masks = random_masks(dimension=8, selected=4, count=2)
    expected = tuple(
        np.sort(np.random.Generator(np.random.PCG64(seed)).choice(8, 4, replace=False))
        for seed in range(2)
    )

    assert len(masks) == len(expected)
    for actual, oracle in zip(masks, expected, strict=True):
        assert np.array_equal(actual, oracle)
        assert np.array_equal(actual, np.unique(actual))


def test_stable_gallery_order_breaks_exact_distance_ties() -> None:
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = retrieval_view(
        query,
        gallery,
        np.array(["right"]),
        np.array(["wrong", "right"]),
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.top1_indices.tolist() == [0]
    assert result.top1_correct.tolist() == [False]


def test_score_chunk_metrics_use_the_same_recall_map_and_tie_contract() -> None:
    chunks = (np.asarray([[1.0, 1.0, 0.0], [0.1, 0.9, 0.8]], dtype=np.float64),)

    result = retrieval_metrics_from_score_chunks(
        chunks,
        np.asarray(["a", "b"]),
        np.asarray(["x", "a", "b"]),
    )

    assert result.top1_indices.tolist() == [0, 1]
    assert result.top1_correct.tolist() == [False, False]
    assert result.recall[1] == 0.0
    assert result.recall[10] == 1.0
    assert result.map_at_r == 0.0
    assert result.average_precision is not None
    assert result.average_precision.tolist() == [0.0, 0.0]


def test_recall_and_map_at_r_match_hand_computed_fixture() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    result = retrieval_view(
        queries,
        gallery,
        np.array(["a", "b"]),
        np.array(["a", "x", "b", "b"]),
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.recall[1] == 1.0
    assert result.recall[10] == 1.0
    assert result.recall[20] == 1.0
    assert result.recall[30] == 1.0
    assert result.map_at_r == 1.0


def test_map_at_r_masks_nonmatches_and_uses_relevant_count_denominator() -> None:
    angles = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    gallery = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)

    result = retrieval_view(
        np.array([[1.0, 0.0]], dtype=np.float32),
        gallery,
        np.array(["match"]),
        np.array(["match", "wrong", "match", "match", "wrong"]),
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.map_at_r == pytest.approx((1.0 + 2.0 / 3.0) / 3.0)


def test_retrieval_only_stably_sorts_the_registered_metric_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    gallery = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    labels = np.asarray(["match", *[f"other-{index}" for index in range(1, 64)]])
    original_lexsort = np.lexsort
    sorted_lengths: list[int] = []

    def bounded_lexsort(keys):
        sorted_lengths.append(len(keys[0]))
        assert len(keys[0]) <= 30
        return original_lexsort(keys)

    monkeypatch.setattr(np, "lexsort", bounded_lexsort)

    result = retrieval_view(
        np.array([[1.0, 0.0]], dtype=np.float32),
        gallery,
        np.array(["match"]),
        labels,
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.recall[1] == 1.0
    assert sorted_lengths == [30]


def test_stable_partial_selection_matches_full_sort_at_boundary_ties() -> None:
    distances = np.asarray([0.0] * 25 + [1.0] * 20 + [2.0] * 19)
    full_order = np.lexsort((np.arange(distances.size), distances))

    assert np.array_equal(_stable_top_indices(distances, 30), full_order[:30])


def test_map_at_r_requests_more_than_thirty_results_when_identity_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery = np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), 64, axis=0)
    labels = np.asarray(["other"] * 64)
    labels[:40] = "match"
    original_lexsort = np.lexsort
    sorted_lengths: list[int] = []

    def measured_lexsort(keys, *args, **kwargs):
        sorted_lengths.append(len(keys[0]))
        return original_lexsort(keys, *args, **kwargs)

    monkeypatch.setattr(np, "lexsort", measured_lexsort)

    result = retrieval_view(
        np.array([[1.0, 0.0]], dtype=np.float32),
        gallery,
        np.array(["match"]),
        labels,
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.map_at_r == 1.0
    assert sorted_lengths == [40]


def test_selection_width_is_derived_from_registered_recall_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery = np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), 64, axis=0)
    labels = np.asarray([f"other-{index}" for index in range(64)])
    labels[40] = "match"
    monkeypatch.setattr(retrieval_audit, "RECALL_AT_K", (1, 50))

    result = retrieval_view(
        np.array([[1.0, 0.0]], dtype=np.float32),
        gallery,
        np.array(["match"]),
        labels,
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.recall == {1: 0.0, 50: 1.0}


@pytest.mark.parametrize(
    (
        "delta_norm",
        "norm_lb",
        "delta_full",
        "full_lb",
        "delta_mask",
        "mask_wins",
        "disagree",
        "primary",
    ),
    [
        (0.002, 1e-9, 0.0, 0.0, 0.002, 24, 0.10, "EVALUATOR_REPAIR"),
        (0.002, 1e-9, 0.002, 1e-9, 0.002, 24, 0.10, "FULL_DIMENSION_CONTROL"),
        (0.0, -1e-9, 0.0, 0.0, 0.002, 24, 0.10, "COORDINATE_NONEXCHANGEABILITY"),
        (0.0, 0.0, 0.0, 0.0, 0.001999999, 32, 1.0, "GEOMETRY_NULL"),
    ],
)
def test_geometry_decision_boundaries(
    delta_norm: float,
    norm_lb: float,
    delta_full: float,
    full_lb: float,
    delta_mask: float,
    mask_wins: int,
    disagree: float,
    primary: str,
) -> None:
    decision = geometry_decision(
        delta_norm=delta_norm,
        norm_lower_bound=norm_lb,
        delta_full=delta_full,
        full_lower_bound=full_lb,
        delta_mask=delta_mask,
        mask_wins=mask_wins,
        disagree=disagree,
    )

    assert decision.primary == primary


def test_geometry_flags_remain_independent_when_full_dimension_has_precedence() -> None:
    decision = geometry_decision(
        delta_norm=0.002,
        norm_lower_bound=1e-9,
        delta_full=0.002,
        full_lower_bound=1e-9,
        delta_mask=0.002,
        mask_wins=24,
        disagree=0.10,
    )

    assert decision.primary == "FULL_DIMENSION_CONTROL"
    assert decision.full_dimension_control is True
    assert decision.evaluator_repair is True
    assert decision.coordinate_nonexchangeability is True


def test_reproduction_gate_uses_published_full_768_view() -> None:
    query = np.array([[3.1690565, 3.7732935, 4.94705]], dtype=np.float32)
    gallery = np.array(
        [[4.7299457, 8.226437, 1.7401735], [8.516344, 8.891572, 0.7644352]],
        dtype=np.float32,
    )

    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["correct"]),
        np.array(["correct", "wrong"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=1.0,
        reproduction_tolerance=0.0,
    )

    assert result.official.recall[1] == 0.0
    assert result.full_unit.recall[1] == 1.0
    assert result.reproduction_passed is True


def test_paired_interval_uses_exact_registered_stream() -> None:
    baseline = np.array([False, True, False, True])
    candidate = np.array([True, True, False, True])
    interval = paired_r1_interval(baseline, candidate, samples=10_000, seed=205)

    generator = np.random.Generator(np.random.PCG64(205))
    indices = generator.integers(0, 4, size=(10_000, 4))
    deltas = candidate[indices].mean(axis=1) - baseline[indices].mean(axis=1)
    oracle = np.percentile(deltas, [2.5, 97.5])

    assert interval == pytest.approx((float(oracle[0]), float(oracle[1])))


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[np.inf, 1.0]], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float64),
    ],
)
def test_l2_normalize_rejects_invalid_embeddings(values: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        l2_normalize(values)


def test_deployment_audit_records_negative_gallery_energy_bias() -> None:
    query = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    gallery = np.array(
        [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]],
        dtype=np.float32,
    )
    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["a"]),
        np.array(["a", "b"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=0.0,
        reproduction_tolerance=0.0,
    )

    assert result.reproduction_passed is True
    assert result.energy_disagreement_count == 1
    assert result.energy_gap_mean is not None
    assert result.energy_gap_mean < 0.0


def test_deployment_audit_marks_reproduction_failure_without_scientific_decision() -> None:
    query = np.array([[1.0, 0.1, 0.1, 0.1]], dtype=np.float32)
    gallery = np.array([[1.0, 0.1, 0.1, 0.1]], dtype=np.float32)
    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["a"]),
        np.array(["a"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=0.746,
        reproduction_tolerance=0.002,
    )

    assert result.reproduction_passed is False
    assert result.decision.primary == "REPRODUCTION_FAILED"


def test_deployment_audit_is_exactly_reproducible() -> None:
    gallery = np.array(
        [
            [1.0, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    query = np.ascontiguousarray(gallery[:2])
    labels = np.array(["a", "b", "c", "d"])
    kwargs = {
        "selected": 2,
        "random_count": 2,
        "bootstrap_samples": 64,
        "expected_official_r1": 1.0,
        "reproduction_tolerance": 0.0,
    }
    first = audit_deployment_geometry(query, gallery, labels[:2], labels, **kwargs)
    second = audit_deployment_geometry(query, gallery, labels[:2], labels, **kwargs)

    assert first == second


def test_review6_evaluation_failure_preserves_same_byte_racer_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset-review6"
    evidence_root = tmp_path / "evidence-review6"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    original = retrieval_audit._write_npy_exclusive
    query_path = evidence_root / "evaluation-epoch-0004-query.npy"
    calls = 0

    def fail_after_racer(path: Path, values: np.ndarray):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(path, values)
        payload = query_path.read_bytes()
        racer = query_path.with_name(f"{query_path.name}.racer")
        racer.write_bytes(payload)
        racer.replace(query_path)
        raise OSError("gallery publication failed")

    monkeypatch.setattr(retrieval_audit, "_write_npy_exclusive", fail_after_racer)
    with pytest.raises(OSError, match="gallery"):
        write_evaluation_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=dataset_root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
            epoch=4,
            evidence_root=evidence_root,
        )
    assert query_path.is_file()


def test_review8_group_rollback_uses_publisher_identity_not_post_return_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset-review8"
    evidence_root = tmp_path / "evidence-review8"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    original = retrieval_audit._write_npy_exclusive
    query_path = evidence_root / "evaluation-epoch-0004-query.npy"

    def replace_before_return(path: Path, values: np.ndarray):
        published = original(path, values)
        if path == query_path:
            payload = path.read_bytes()
            path.unlink()
            path.write_bytes(payload)
        return published

    monkeypatch.setattr(
        retrieval_audit, "_write_npy_exclusive", replace_before_return
    )
    with pytest.raises(OSError, match="gallery guard"):
        write_evaluation_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=dataset_root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
            epoch=4,
            evidence_root=evidence_root,
            publication_guard=lambda component, _destination, _payload: (
                (_ for _ in ()).throw(OSError("gallery guard failed"))
                if component == "gallery"
                else None
            ),
        )
    assert query_path.is_file()


def test_review10_retrieval_guards_header_bearing_npy_bytes_before_publication(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset-review10"
    evidence_root = tmp_path / "evidence-review10"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    raw_query_bytes = query.nbytes
    observed: list[tuple[str, Path, int]] = []

    def guard(component: str, destination: Path, payload: bytes) -> None:
        observed.append((component, destination, len(payload)))
        if component == "query" and len(payload) > raw_query_bytes:
            raise OSError("query NPY payload exceeds raw-only row")

    with pytest.raises(OSError, match="NPY payload"):
        write_evaluation_evidence(
            query_values=query,
            gallery_values=gallery,
            query_records=query_records,
            gallery_records=gallery_records,
            dataset_root=dataset_root,
            coordinates=np.arange(512, dtype=np.int64),
            normalize_before=True,
            epoch=4,
            evidence_root=evidence_root,
            publication_guard=guard,
        )
    assert observed[0][0] == "query"
    assert observed[0][1].name == "evaluation-epoch-0004-query.npy"
    assert observed[0][2] > raw_query_bytes
    assert not any(evidence_root.iterdir())


def test_review11_evaluation_uses_established_retained_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset-review11"
    evidence_root = tmp_path / "evidence-review11"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    npy_calls: list[Path] = []
    json_calls: list[Path] = []
    original_npy = retrieval_audit._write_npy_exclusive
    original_json = retrieval_audit._write_json_exclusive

    def record_npy(path: Path, values: np.ndarray):
        npy_calls.append(path)
        return original_npy(path, values)

    def record_json(path: Path, value: dict[str, object]):
        json_calls.append(path)
        return original_json(path, value)

    monkeypatch.setattr(retrieval_audit, "_write_npy_exclusive", record_npy)
    monkeypatch.setattr(retrieval_audit, "_write_json_exclusive", record_json)
    write_evaluation_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=dataset_root,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
        epoch=4,
        evidence_root=evidence_root,
    )
    assert [path.name for path in npy_calls] == [
        "evaluation-epoch-0004-query.npy",
        "evaluation-epoch-0004-gallery.npy",
    ]
    assert [path.name for path in json_calls] == [
        "evaluation-epoch-0004-ranked-prefix.json",
        "evaluation-epoch-0004.json",
    ]


def test_review12_ranked_prefix_is_a_separate_guarded_publication(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset-review12"
    evidence_root = tmp_path / "evidence-review12"
    evidence_root.mkdir()
    query_records, gallery_records = _evidence_records(dataset_root)
    query, gallery = _full_width_evidence_values()
    guarded: list[tuple[str, str, int]] = []

    receipt = write_evaluation_evidence(
        query_values=query,
        gallery_values=gallery,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=dataset_root,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
        epoch=4,
        evidence_root=evidence_root,
        publication_guard=lambda name, path, payload: guarded.append(
            (name, path.name, len(payload))
        ),
    )
    assert [name for name, _path, _bytes in guarded] == [
        "query", "gallery", "ranked-prefix", "receipt"
    ]
    assert guarded[2][1] == "evaluation-epoch-0004-ranked-prefix.json"
    assert "query_evidence" not in receipt
    assert receipt["ranked_prefix_evidence"]["path"] == guarded[2][1]
