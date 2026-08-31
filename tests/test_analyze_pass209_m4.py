from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from sfora.pass209_m4 import (
    REGISTERED_M4_CELLS,
    M4Cell,
    M4CellPaths,
    M4CellSpec,
    M4DescriptorHeader,
    M4Example,
    M4SourceError,
    QueryEvidence,
    adapt_m3_m4,
    analyze_rescue_evidence,
    audit_exact_rgb_duplicates,
    canonical_json_bytes,
    classify_m3_transfer,
    encode_descriptor_file,
    load_m4_cells,
    load_m4_source_errors,
    m4_receipt_bytes,
    score_descriptor_plane,
)
from sfora.substrate_screen import score_frozen_substrate_evidence


def _bits(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", value))[0])


def _queries(correct_positions: set[int]) -> tuple[QueryEvidence, ...]:
    rows = []
    errors = _errors()
    for position in range(103):
        correct = position in correct_positions
        query_label = errors[position].query_label
        nearest_label = query_label if correct else errors[position].nearest_label
        rows.append(
            QueryEvidence(
                query_position=position,
                query_example_id=f"q{position}",
                query_label=query_label,
                nearest_position=(position + 1) % 103,
                nearest_example_id=f"q{(position + 1) % 103}",
                nearest_label=nearest_label,
                nearest_score_bits=_bits(0.8),
                best_same_position=(position + 2) % 103,
                best_same_score_bits=_bits(0.7 if correct else 0.4),
                best_different_position=(position + 3) % 103,
                best_different_score_bits=_bits(0.3 if correct else 0.7),
                margin_bits=_bits(0.4 if correct else -0.3),
                correct=correct,
            )
        )
    return tuple(rows)


def _errors() -> tuple[M4SourceError, ...]:
    unordered = [(82, 83)] * 63 + [(85, 86)] * 12 + [(89, 90)] * 9
    unordered += [(91, 92)] * 10 + [(93, 94)] * 9
    pairs = [pair if ordinal % 2 == 0 else pair[::-1] for ordinal, pair in enumerate(unordered)]
    rows = []
    for ordinal, (query_label, nearest_label) in enumerate(pairs):
        nearest_position = next(
            index
            for index, (candidate_label, _) in enumerate(pairs)
            if index != ordinal and candidate_label == nearest_label
        )
        rows.append(
            M4SourceError(
                error_ordinal=ordinal,
                query_position=ordinal,
                query_example_id=f"q{ordinal}",
                query_label=query_label,
                nearest_position=nearest_position,
                nearest_example_id=f"q{nearest_position}",
                nearest_label=nearest_label,
            )
        )
    return tuple(rows)


def test_objective_rescue_recomputes_reachability_and_dominant_pair() -> None:
    dino = _queries(set(range(20)))
    siglip2 = _queries(set(range(20, 30)))
    selecting = _queries(set())

    evidence = analyze_rescue_evidence(
        source_errors=_errors(),
        dinov2_queries=dino,
        siglip2_queries=siglip2,
        selecting_queries=selecting,
        selecting_cuda_queries=selecting,
    )

    assert evidence.source_error_count == 103
    assert evidence.reachable_count == 30
    assert evidence.universal_three_device_error_count == 73
    assert evidence.dinov2_rescued == 20
    assert evidence.siglip2_rescued == 10
    assert evidence.selecting_rescued == 0
    assert evidence.selecting_cpu_correct_on_source_errors == 0
    assert evidence.selecting_cpu_cuda_correctness_disagreements == 0
    assert evidence.dominant_pair == (82, 83)
    assert evidence.dominant_pair_count == 63
    assert evidence.dominant_pair_dinov2_rescued == 20
    assert evidence.dominant_pair_siglip2_rescued == 10
    assert evidence.dominant_pair_rescuable is True
    assert evidence.bootstrap.observed_share == pytest.approx(30 / 103)
    assert tuple(panel.pair for panel in evidence.pair_panels) == (
        (82, 83),
        (85, 86),
        (89, 90),
    )
    assert evidence.pair_panels[0].dinov2_rescue_rate == pytest.approx(20 / 63)
    assert evidence.pair_panels[0].siglip2_rescue_rate == pytest.approx(10 / 63)
    assert evidence.pair_panels[0].selecting_rescue_rate == 0.0
    assert evidence.rows[0].reachable is True
    assert evidence.rows[30].universal_three_device_error is True

    selecting_cpu = _queries({0, 1})
    disagreement = analyze_rescue_evidence(
        source_errors=_errors(),
        dinov2_queries=dino,
        siglip2_queries=siglip2,
        selecting_queries=selecting_cpu,
        selecting_cuda_queries=selecting,
    )
    assert disagreement.selecting_cpu_correct_on_source_errors == 2
    assert disagreement.selecting_cpu_cuda_correctness_disagreements == 2


def test_objective_uses_cuda_population_but_preserves_cpu_divergence() -> None:
    selecting_cpu = (replace(_queries(set())[0], correct=True), *_queries(set())[1:])
    evidence = analyze_rescue_evidence(
        source_errors=_errors(),
        dinov2_queries=_queries(set()),
        siglip2_queries=_queries(set()),
        selecting_queries=selecting_cpu,
        selecting_cuda_queries=_queries(set()),
    )

    assert evidence.selecting_rescued == 0
    assert evidence.rows[0].selecting_correct is False
    assert evidence.rows[0].selecting_cpu_correct is True


def test_exact_rgb_duplicate_audit_searches_the_full_different_label_gallery() -> None:
    errors = _errors()
    examples = tuple(
        M4Example(
            position=row.query_position,
            example_id=row.query_example_id,
            label=row.query_label,
        )
        for row in errors
    )
    digests = [hashlib.sha256(f"rgb-{index}".encode()).hexdigest() for index in range(103)]
    digests[63] = digests[0]
    evidence = audit_exact_rgb_duplicates(
        source_errors=errors,
        examples=examples,
        rgb_sha256=tuple(digests),
    )
    assert evidence[0].matching_positions == (63,)
    assert evidence[0].matching_labels == (errors[63].query_label,)
    assert evidence[63].matching_positions == (0,)
    assert all(not row.matching_positions for row in evidence[1:63])

    inconsistent = replace(errors[0], nearest_example_id="wrong-nearest")
    with pytest.raises(ValueError, match="nearest identity"):
        audit_exact_rgb_duplicates(
            source_errors=(inconsistent, *errors[1:]),
            examples=examples,
            rgb_sha256=tuple(digests),
        )


@pytest.mark.parametrize(
    ("ratios", "expected"),
    [
        ((0.50, 0.70, 1.20), "T-high"),
        ((0.10, 0.35, -0.20), "T-low"),
        ((0.20, 0.40, 0.60), "T-mid"),
        ((0.20, None, 0.60), "T-undefined"),
    ],
)
def test_m3_state_and_first_match_adapter(
    ratios: tuple[float | None, float | None, float | None], expected: str
) -> None:
    assert classify_m3_transfer(ratios) == expected


def test_adapter_admits_only_preregistered_family() -> None:
    assert (
        adapt_m3_m4(m3_state="T-low", reachable_p10=0.25, dominant_rescuable=True) == "F4-TRANSFER"
    )
    assert (
        adapt_m3_m4(m3_state="T-high", reachable_p10=0.0, dominant_rescuable=True) == "F4-CAPACITY"
    )
    assert adapt_m3_m4(m3_state="T-mid", reachable_p10=1.0, dominant_rescuable=True) == "F4-NONE"
    with pytest.raises(ValueError, match="state"):
        adapt_m3_m4(m3_state="T-low", reachable_p10=float("nan"), dominant_rescuable=True)


def _cell_paths(tmp_path: Path, cell: str) -> tuple[M4CellPaths, M4CellSpec]:
    spec = replace(
        REGISTERED_M4_CELLS[cell],
        expected_rows=4,
        descriptor_dimensions=2,
        expected_correct=1,
        legacy_descriptor_sha256=None,
    )
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    examples = tuple(
        M4Example(position=index, example_id=f"q{index}", label=label)
        for index, label in enumerate((82, 82, 83, 83))
    )
    raw = descriptors.numpy().astype("<f4", copy=False).tobytes(order="C")
    header = M4DescriptorHeader(
        schema="sfora-pass209-m4-descriptor-v1",
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset="cars",
        dataset_revision="3" * 40,
        dataset_examples_sha256="4" * 64,
        dataset_examples_ordered_sha256="5" * 64,
        split="train",
        holdout_classes=tuple(range(82, 98)),
        compute_dtype="float32",
        cell=cell,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        readout=spec.readout,
        rows=4,
        dimensions=2,
        payload_bytes=len(raw),
        payload_sha256=hashlib.sha256(raw).hexdigest(),
    )
    descriptor_payload = encode_descriptor_file(header, descriptors)
    queries = score_descriptor_plane(descriptors, examples, block_size=32)
    query_payload = canonical_json_bytes(
        {
            "schema": "sfora-pass209-m4-query-evidence-v1",
            "claim_eligible": False,
            "cell": cell,
            "dataset_examples_sha256": "4" * 64,
            "dataset_examples_ordered_sha256": "5" * 64,
            "descriptor_file_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
            "query_block": 32,
            "rows": [asdict(row) for row in queries],
            "historical_cuda_rows": [asdict(row) for row in queries],
        }
    )
    historical = score_frozen_substrate_evidence(
        descriptors,
        torch.tensor((82, 82, 83, 83), dtype=torch.int64),
        query_block=32,
    )
    receipt_payload = canonical_json_bytes(
        {
            "schema": "sfora-pass209-m4-cell-receipt-v1",
            "claim_eligible": False,
            "source_revision": "1" * 40,
            "source_tree_digest": "2" * 64,
            "dataset": "cars",
            "dataset_revision": "3" * 40,
            "dataset_examples_sha256": "4" * 64,
            "dataset_examples_ordered_sha256": "5" * 64,
            "split": "train",
            "holdout_classes": list(range(82, 98)),
            "compute_dtype": "float32",
            "error_manifest_sha256": "6" * 64,
            "cell": cell,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "readout": spec.readout,
            "batch_size": spec.batch_size,
            "query_block": 32,
            "processor_image_shape": list(spec.processor_image_shape),
            "descriptor_shape": [4, 2],
            "descriptor_file_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
            "descriptor_payload_sha256": header.payload_sha256,
            "legacy_descriptor_sha256": "7" * 64,
            "query_evidence_sha256": hashlib.sha256(query_payload).hexdigest(),
            "correct": 1,
            "cpu_reference_correct": 1,
            "expected_historical_correct": 1,
            "historical_cuda_correct": 1,
            "historical_cuda_errors": [asdict(row) for row in historical.errors],
            "legacy_descriptor_passed": True,
            "historical_count_passed": True,
            "historical_errors_passed": True,
            "reproduction_passed": True,
            "queries": 4,
            "prerequisite_sha256": spec.prerequisite_sha256,
            "prerequisite_schema": spec.prerequisite_schema,
            "scorer_environment": {"schema": "fixture-scorer"},
            "cuda_environment": {"schema": "fixture-cuda"},
        }
    )
    root = tmp_path / cell
    root.mkdir()
    receipt = root / "receipt.json"
    descriptor = root / "descriptor.bin"
    query = root / "queries.json"
    receipt.write_bytes(receipt_payload)
    descriptor.write_bytes(descriptor_payload)
    query.write_bytes(query_payload)
    return M4CellPaths(receipt=receipt, descriptor=descriptor, queries=query), spec


def test_three_cell_loader_recomputes_and_cross_binds_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths_and_specs = [_cell_paths(tmp_path, cell) for cell in REGISTERED_M4_CELLS]
    for (path_set, spec), cell in zip(paths_and_specs, REGISTERED_M4_CELLS, strict=True):
        del path_set
        monkeypatch.setitem(REGISTERED_M4_CELLS, cell, spec)
    all_paths = tuple(item[0] for item in paths_and_specs)
    expected_examples = tuple(
        M4Example(position=index, example_id=f"q{index}", label=label)
        for index, label in enumerate((82, 82, 83, 83))
    )

    cells = load_m4_cells(all_paths, expected_examples=expected_examples)
    assert tuple(cell.spec.cell for cell in cells) == tuple(REGISTERED_M4_CELLS)
    assert all(cell.receipt["reproduction_passed"] is True for cell in cells)
    assert all(tuple(cell.descriptors.shape) == (4, 2) for cell in cells)

    first_receipt = all_paths[0].receipt
    value = json.loads(first_receipt.read_bytes())
    value["reproduction_passed"] = False
    first_receipt.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="reproduction"):
        load_m4_cells(all_paths, expected_examples=expected_examples)


def test_three_cell_loader_rejects_self_consistent_but_false_gallery_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths_and_specs = [_cell_paths(tmp_path, cell) for cell in REGISTERED_M4_CELLS]
    for (_, spec), cell in zip(paths_and_specs, REGISTERED_M4_CELLS, strict=True):
        monkeypatch.setitem(REGISTERED_M4_CELLS, cell, spec)
    expected_examples = tuple(
        M4Example(position=index, example_id=f"q{index}", label=label)
        for index, label in enumerate((82, 83, 82, 83))
    )

    with pytest.raises(ValueError, match="dataset example identity"):
        load_m4_cells(
            tuple(item[0] for item in paths_and_specs),
            expected_examples=expected_examples,
        )


def test_source_manifest_loader_requires_exact_digest_schema_and_rows(
    tmp_path: Path,
) -> None:
    source_errors = _errors()
    value = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "source_revision": "1" * 40,
        "source_tree_digest": "2" * 64,
        "dataset": "cars",
        "dataset_revision": "3" * 40,
        "dataset_examples_sha256": "4" * 64,
        "descriptor_sha256": "5" * 64,
        "batch_size": 8,
        "query_block": 32,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "class_names": [{"id": label, "name": f"class-{label}"} for label in range(82, 98)],
        "cell": "siglip-so400m",
        "model_name": REGISTERED_M4_CELLS["siglip-so400m"].model_name,
        "model_revision": REGISTERED_M4_CELLS["siglip-so400m"].model_revision,
        "error_count": 103,
        "errors": [
            {
                "query_position": row.query_position,
                "query_example_id": row.query_example_id,
                "query_label": row.query_label,
                "nearest_position": row.nearest_position,
                "nearest_example_id": row.nearest_example_id,
                "nearest_label": row.nearest_label,
            }
            for row in source_errors
        ],
    }
    path = tmp_path / "errors.json"
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest, loaded = load_m4_source_errors(path, expected_sha256=digest)
    assert manifest["error_count"] == 103
    assert loaded == source_errors

    mutable_errors = value["errors"]
    assert isinstance(mutable_errors, list)
    first_error = mutable_errors[0]
    assert isinstance(first_error, dict)
    first_error["query_position"] = False
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="row"):
        load_m4_source_errors(path, expected_sha256=hashlib.sha256(payload).hexdigest())


def _analysis_cells(error_manifest_sha256: str) -> tuple[M4Cell, M4Cell, M4Cell]:
    query_tables = (_queries(set(range(20))), _queries(set(range(20, 30))), _queries(set()))
    cells = []
    for index, (cell_name, queries) in enumerate(
        zip(REGISTERED_M4_CELLS, query_tables, strict=True)
    ):
        spec = replace(
            REGISTERED_M4_CELLS[cell_name],
            expected_rows=103,
            descriptor_dimensions=2,
        )
        descriptor = torch.nn.functional.normalize(
            torch.arange(206, dtype=torch.float32).reshape(103, 2) + 1,
            dim=1,
        )
        raw = descriptor.numpy().astype("<f4", copy=False).tobytes(order="C")
        header = M4DescriptorHeader(
            schema="sfora-pass209-m4-descriptor-v1",
            source_revision="1" * 40,
            source_tree_digest="2" * 64,
            dataset="cars",
            dataset_revision="3" * 40,
            dataset_examples_sha256="4" * 64,
            dataset_examples_ordered_sha256="5" * 64,
            split="train",
            holdout_classes=tuple(range(82, 98)),
            compute_dtype="float32",
            cell=cell_name,
            model_name=spec.model_name,
            model_revision=spec.model_revision,
            readout=spec.readout,
            rows=103,
            dimensions=2,
            payload_bytes=len(raw),
            payload_sha256=hashlib.sha256(raw).hexdigest(),
        )
        cells.append(
            M4Cell(
                spec=spec,
                receipt={
                    "source_revision": "1" * 40,
                    "source_tree_digest": "2" * 64,
                    "dataset_revision": "3" * 40,
                    "dataset_examples_sha256": "4" * 64,
                    "dataset_examples_ordered_sha256": "5" * 64,
                    "error_manifest_sha256": error_manifest_sha256,
                    "reproduction_passed": True,
                    "historical_cuda_errors": (
                        [
                            {
                                "query_position": row.query_position,
                                "nearest_position": row.nearest_position,
                                "query_label": row.query_label,
                                "nearest_label": row.nearest_label,
                            }
                            for row in _errors()
                        ]
                        if index == 2
                        else []
                    ),
                },
                header=header,
                descriptors=descriptor,
                queries=queries,
                cuda_queries=queries,
                receipt_sha256=str(index + 6) * 64,
                descriptor_sha256=str(index + 1) * 64,
                query_sha256=str(index + 3) * 64,
            )
        )
    return cells[0], cells[1], cells[2]


def test_m4_receipt_is_canonical_and_recomputes_every_derived_result() -> None:
    digest = "9" * 64
    errors = _errors()
    examples = tuple(
        M4Example(
            position=row.query_position,
            example_id=row.query_example_id,
            label=row.query_label,
        )
        for row in errors
    )
    rgb = tuple(hashlib.sha256(f"rgb-{index}".encode()).hexdigest() for index in range(103))
    cells = _analysis_cells(digest)
    payload = m4_receipt_bytes(
        cells=cells,
        source_errors=errors,
        error_manifest_sha256=digest,
        examples=examples,
        rgb_sha256=rgb,
    )
    value = json.loads(payload)
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert value["schema"] == "sfora-pass209-m4-objective-rescue-v1"
    assert value["objective"]["reachable_count"] == 30
    assert value["objective"]["dominant_pair_rescuable"] is True
    assert value["passed"] is True
    assert value["duplicate_query_count"] == 0
    assert [row["cell"] for row in value["cells"]] == list(REGISTERED_M4_CELLS)

    cuda_row = cells[0].cuda_queries[0]
    divergent_cuda_row = replace(
        cuda_row,
        nearest_position=2,
        nearest_example_id="q2",
        nearest_label=errors[2].query_label,
        nearest_score_bits=_bits(0.75),
        margin_bits=_bits(-0.25),
        correct=errors[0].query_label == errors[2].query_label,
    )
    divergent = replace(
        cells[0],
        cuda_queries=(divergent_cuda_row, *cells[0].cuda_queries[1:]),
    )
    divergent_value = json.loads(
        m4_receipt_bytes(
            cells=(divergent, cells[1], cells[2]),
            source_errors=errors,
            error_manifest_sha256=digest,
            examples=examples,
            rgb_sha256=rgb,
        )
    )
    assert divergent_value["cells"][0]["cpu_cuda_nearest_divergence_count"] == 1
    divergence = divergent_value["cells"][0]["cpu_cuda_nearest_divergences"][0]
    assert divergence["query_position"] == 0
    assert divergence["cpu_nearest_position"] == 1
    assert divergence["cuda_nearest_position"] == 2
    assert divergence["cpu_nearest_score_bits"] == _bits(0.8)
    assert divergence["cuda_nearest_score_bits"] == _bits(0.75)

    no_rescue = _queries(set())
    failed_objective = json.loads(
        m4_receipt_bytes(
            cells=(
                replace(cells[0], queries=no_rescue, cuda_queries=no_rescue),
                replace(cells[1], queries=no_rescue, cuda_queries=no_rescue),
                cells[2],
            ),
            source_errors=errors,
            error_manifest_sha256=digest,
            examples=examples,
            rgb_sha256=rgb,
        )
    )
    assert failed_objective["passed"] is False

    failed = replace(cells[0], receipt={**cells[0].receipt, "reproduction_passed": False})
    with pytest.raises(ValueError, match="failed reproduction"):
        m4_receipt_bytes(
            cells=(failed, cells[1], cells[2]),
            source_errors=errors,
            error_manifest_sha256=digest,
            examples=examples,
            rgb_sha256=rgb,
        )
