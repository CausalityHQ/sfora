from __future__ import annotations

import hashlib
import json

import pytest
import torch

from sfora.siglip_coverage_calibration import (
    CoverageAffine,
    CoverageArtifactIdentity,
    CoverageCalibrationRun,
    CoverageMaps,
    analyze_coverage_calibration,
    build_coverage_calibration_result,
    build_coverage_fit_receipt,
    coverage_calibration_classification,
    coverage_retrieval_cells,
    coverage_support_indexes,
    fit_coverage_affine,
    validate_coverage_calibration_result_bytes,
    validate_coverage_execution_receipt_bytes,
    validate_coverage_fit_receipt_bytes,
)


def _support_ids(ids: tuple[str, ...], indexes: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(ids[index] for index in indexes)


def test_coverage_support_indexes_are_exact_balanced_and_permutation_stable() -> None:
    ids = tuple(f"image-{index:02d}" for index in range(20))
    labels = (3,) * 10 + (7,) * 10
    expected = tuple(
        sorted(
            (
                image_id
                for label in (3, 7)
                for _digest, image_id in sorted(
                    (
                        hashlib.sha256(
                            b"sfora-coverage-support-v1\0"
                            + str(label).encode("ascii")
                            + b"\0"
                            + ids[index].encode("utf-8")
                        ).digest(),
                        ids[index],
                    )
                    for index, value in enumerate(labels)
                    if value == label
                )[:8]
            )
        )
    )
    selected = coverage_support_indexes(ids, labels)
    assert _support_ids(ids, selected) == expected
    assert {label: sum(labels[index] == label for index in selected) for label in (3, 7)} == {
        3: 8,
        7: 8,
    }

    permutation = tuple(reversed(range(len(ids))))
    permuted_ids = tuple(ids[index] for index in permutation)
    permuted_labels = tuple(labels[index] for index in permutation)
    permuted = coverage_support_indexes(permuted_ids, permuted_labels)
    assert _support_ids(permuted_ids, permuted) == expected


@pytest.mark.parametrize(
    ("ids", "labels", "support"),
    [
        (("x",) * 20, (0,) * 10 + (1,) * 10, 8),
        (tuple(f"x{i}" for i in range(19)), (0,) * 9 + (1,) * 10, 8),
        (tuple(f"x{i}" for i in range(20)), (0,) * 10 + (1,) * 10, True),
        (tuple(f"x{i}" for i in range(20)), (0,) * 10 + (1,) * 10, 4),
    ],
)
def test_coverage_support_indexes_reject_authority_drift(
    ids: tuple[str, ...], labels: tuple[int, ...], support: object
) -> None:
    with pytest.raises(ValueError, match="support authority"):
        coverage_support_indexes(ids, labels, support_per_class=support)  # type: ignore[arg-type]


def test_fit_coverage_affine_recovers_full_rank_map_and_is_order_invariant() -> None:
    source = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.5],
                [0.25, -0.75],
                [2.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    weight = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    bias = torch.zeros(2, dtype=torch.float64)
    target = (source.double() @ weight).float()
    ids = tuple(f"id-{index}" for index in range(source.shape[0]))

    fitted = fit_coverage_affine(source, target, ids)
    assert isinstance(fitted, CoverageAffine)
    assert fitted.rank == 3
    assert fitted.rcond == 1e-12
    assert fitted.driver == "gelsd"
    # The exact affine targets crossed the registered FP32 descriptor boundary.
    assert torch.allclose(fitted.weight, weight, atol=1e-6, rtol=0)
    assert torch.allclose(fitted.bias, bias, atol=1e-6, rtol=0)
    expected = torch.nn.functional.normalize(target, dim=1)
    assert fitted.apply(source).dtype == torch.float32
    assert fitted.apply(source).is_contiguous()
    assert torch.allclose(fitted.apply(source), expected, atol=1e-6, rtol=0)

    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    reordered = fit_coverage_affine(
        source[permutation],
        target[permutation],
        tuple(ids[index] for index in permutation.tolist()),
    )
    assert torch.equal(fitted.weight, reordered.weight)
    assert torch.equal(fitted.bias, reordered.bias)


@pytest.mark.parametrize("mutation", ("rank", "duplicate", "nonfinite", "dtype"))
def test_fit_coverage_affine_rejects_invalid_inputs(mutation: str) -> None:
    source = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    target = source.clone()
    ids = ("a", "b", "c", "d")
    if mutation == "rank":
        source[:, 1] = source[:, 0]
    elif mutation == "duplicate":
        ids = ("a", "a", "c", "d")
    elif mutation == "nonfinite":
        target[0, 0] = torch.nan
    elif mutation == "dtype":
        source = source.double()
    with pytest.raises(ValueError, match="affine authority"):
        fit_coverage_affine(source, target, ids)


def test_coverage_affine_rejects_zero_mapped_output() -> None:
    mapping = CoverageAffine(
        weight=torch.zeros((2, 2), dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    with pytest.raises(ValueError, match="affine output"):
        mapping.apply(torch.eye(2, dtype=torch.float32))


@pytest.mark.parametrize("scale", (1e-20, 1e200))
def test_coverage_affine_normalizes_extreme_finite_scales_safely(scale: float) -> None:
    mapping = CoverageAffine(
        weight=scale * torch.eye(2, dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    result = mapping.apply(torch.eye(2, dtype=torch.float32))
    assert torch.equal(result, torch.eye(2, dtype=torch.float32))
    assert torch.equal(torch.linalg.vector_norm(result, dim=1), torch.ones(2))


def test_fit_and_apply_reject_unnormalized_descriptors() -> None:
    source = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [-2.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="affine authority"):
        fit_coverage_affine(source, source, ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="affine authority"):
        _identity_affine().apply(source)


def _identity_affine() -> CoverageAffine:
    return CoverageAffine(
        weight=torch.eye(2, dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )


def test_coverage_retrieval_cells_cover_both_deployment_directions() -> None:
    student = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    student = torch.nn.functional.normalize(student, dim=1)
    teacher = student @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    forward = CoverageAffine(
        weight=torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    reverse = CoverageAffine(
        weight=torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    ids = ("a", "b", "c", "d")
    labels = (0, 0, 1, 1)
    cells = coverage_retrieval_cells(
        student,
        teacher,
        ids,
        labels,
        CoverageMaps(forward, reverse),
    )
    assert set(cells) == {
        "identity-forward",
        "identity-reverse",
        "student-self",
        "teacher-self",
        "forward-student-query",
        "forward-teacher-query",
        "offline-student-query",
        "offline-teacher-query",
    }
    passing = {
        "student-self",
        "teacher-self",
        "forward-student-query",
        "forward-teacher-query",
        "offline-student-query",
        "offline-teacher-query",
    }
    assert all(cells[name].micro_r1 == 1.0 for name in passing)
    assert all(cells[name].micro_map_at_r == 1.0 for name in passing)
    assert cells["identity-forward"].micro_r1 < 1.0
    assert cells["identity-forward"].paired_cosines != cells["forward-student-query"].paired_cosines
    assert coverage_calibration_classification(cells) == "coverage-calibration-qualified"


def test_coverage_classification_distinguishes_offline_and_rejection() -> None:
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32
    )
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)
    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ("a", "b", "c", "d"),
        (0, 0, 1, 1),
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    failed = cells["identity-forward"]
    object.__setattr__(failed, "hits", (False,) * 4)
    object.__setattr__(failed, "average_precisions", (0.0,) * 4)
    cells["forward-student-query"] = failed
    assert coverage_calibration_classification(cells) == "offline-gallery-qualified"
    cells["offline-student-query"] = failed
    assert coverage_calibration_classification(cells) == "coverage-calibration-rejected"

    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ("a", "b", "c", "d"),
        (0, 0, 1, 1),
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    cells["offline-student-query"] = failed
    assert coverage_calibration_classification(cells) == "forward-only-qualified"

    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ("a", "b", "c", "d"),
        (0, 0, 1, 1),
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    cells["student-self"] = failed
    assert coverage_calibration_classification(cells) == "student-quality-rejected"


def test_fit_coverage_affine_accepts_the_registered_512_dimension_shape() -> None:
    generator = torch.Generator().manual_seed(20260906)
    source = torch.nn.functional.normalize(torch.randn(520, 512, generator=generator), dim=1)
    mapping = fit_coverage_affine(
        source,
        source.clone(),
        tuple(f"real-shape-{index:03d}" for index in range(520)),
    )
    assert mapping.rank == 513
    assert mapping.weight.shape == (512, 512)
    assert torch.allclose(mapping.apply(source), source, atol=2e-5, rtol=0)


def test_analyze_coverage_calibration_seals_support_and_excludes_it_from_evaluation() -> None:
    base_student = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    transform = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    base_teacher = base_student @ transform
    base_ids = ("base-a", "base-b", "base-c", "base-d")
    offsets = torch.linspace(0.0, 0.09, 10)
    candidate_student = torch.cat(
        (
            torch.stack((torch.ones(10), offsets), dim=1),
            torch.stack((offsets, torch.ones(10)), dim=1),
        ),
        dim=0,
    )
    candidate_student = torch.nn.functional.normalize(candidate_student, dim=1)
    candidate_teacher = candidate_student @ transform
    candidate_ids = tuple(f"candidate-{index:02d}" for index in range(20))
    candidate_labels = (10,) * 10 + (11,) * 10

    run = analyze_coverage_calibration(
        base_student,
        base_teacher,
        base_ids,
        candidate_student,
        candidate_teacher,
        candidate_ids,
        candidate_labels,
    )
    assert len(run.support_ids) == 16
    assert len(run.evaluation_ids) == 4
    assert set(run.support_ids).isdisjoint(run.evaluation_ids)
    assert set(run.support_ids) | set(run.evaluation_ids) == set(candidate_ids)
    assert all(evidence.ids == run.evaluation_ids for evidence in run.cells.values())
    assert run.classification == "coverage-calibration-qualified"


def _result_run() -> CoverageCalibrationRun:
    descriptors = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    candidate_ids = tuple(f"candidate-{index:02d}" for index in range(20))
    candidate_labels = (0,) * 10 + (1,) * 10
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support_set = set(support_indexes)
    evaluation_indexes = tuple(
        index for index in range(len(candidate_ids)) if index not in support_set
    )
    ids = tuple(candidate_ids[index] for index in evaluation_indexes)
    labels = tuple(candidate_labels[index] for index in evaluation_indexes)
    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ids,
        labels,
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    return CoverageCalibrationRun(
        maps=CoverageMaps(_identity_affine(), _identity_affine()),
        support_ids=tuple(candidate_ids[index] for index in support_indexes),
        support_labels=tuple(candidate_labels[index] for index in support_indexes),
        evaluation_ids=ids,
        cells=cells,
        classification="coverage-calibration-qualified",
    )


def _result_bytes() -> bytes:
    identities = {
        role: CoverageArtifactIdentity(
            path=f"/fixture/{role}",
            sha256=f"{index + 1:02x}" * 32,
            byte_length=100 + index,
        )
        for index, role in enumerate(
            (
                "checkpoint",
                "control_binding",
                "evaluation_manifest",
                "fit_receipt",
                "map_artifact",
                "optimization_manifest",
                "spatial_artifact",
            )
        )
    }
    return build_coverage_calibration_result(
        _result_run(),
        artifact_identities=identities,
        model_source_commit="a" * 40,
        execution_source_commit="c" * 40,
        dataset_id="fixture/cars",
        dataset_revision="b" * 40,
        optimization_image_root="/fixture/optimization-images",
        evaluation_image_root="/fixture/evaluation-images",
        optimization_images_sha256="77" * 32,
        support_images_sha256="88" * 32,
        evaluation_images_sha256="99" * 32,
        optimization_rows=49,
        torch_version="2.8.0",
        torch_num_threads=1,
        blas_config="fixture-blas",
        cpu_identity="fixture-cpu",
    )


def _fit_receipt_bytes() -> bytes:
    identities = {
        role: CoverageArtifactIdentity(
            path=f"/fixture/{role}",
            sha256=f"{index + 1:02x}" * 32,
            byte_length=100 + index,
        )
        for index, role in enumerate(
            (
                "checkpoint",
                "control_binding",
                "evaluation_manifest",
                "map_artifact",
                "optimization_manifest",
                "spatial_artifact",
            )
        )
    }
    run = _result_run()
    return build_coverage_fit_receipt(
        maps=run.maps,
        support_ids=run.support_ids,
        support_labels=run.support_labels,
        artifact_identities=identities,
        model_source_commit="a" * 40,
        execution_source_commit="c" * 40,
        dataset_id="fixture/cars",
        dataset_revision="b" * 40,
        optimization_image_root="/fixture/optimization-images",
        support_image_root="/fixture/support-images",
        optimization_images_sha256="77" * 32,
        support_images_sha256="88" * 32,
        optimization_rows=49,
        heldout_rows=4,
        heldout_denied=4,
        heldout_directory_denied=True,
        torch_version="2.8.0",
        torch_num_threads=1,
        blas_config="fixture-blas",
        cpu_identity="fixture-cpu",
    )


def test_coverage_fit_receipt_is_canonical_and_rejects_drift() -> None:
    raw = _fit_receipt_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    receipt = validate_coverage_fit_receipt_bytes(raw)
    assert receipt["schema"] == "sfora-siglip-coverage-fit-receipt-v1"
    assert receipt["claim_eligible"] is False
    assert receipt["heldout_probe"] == {
        "rows": 4,
        "denied": 4,
        "directory_denied": True,
    }
    for mutation in ("map", "support", "denial", "claim"):
        changed = json.loads(raw)
        if mutation == "map":
            changed["inputs"]["map_artifact"]["sha256"] = "bad"
        elif mutation == "support":
            changed["support_ids"][0] = changed["support_ids"][1]
        elif mutation == "denial":
            changed["heldout_probe"]["denied"] -= 1
        else:
            changed["claim_eligible"] = True
        mutated = (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(ValueError, match="coverage fit receipt"):
            validate_coverage_fit_receipt_bytes(mutated)


def test_coverage_execution_receipt_is_canonical_and_rejects_false_success() -> None:
    identity = {"path": "/fixture/result", "sha256": "11" * 32, "byte_length": 10}
    value = {
        "schema": "sfora-siglip-coverage-execution-v1",
        "claim_eligible": False,
        "execution_source_commit": "a" * 40,
        "status": "complete",
        "failed_phase": None,
        "reason_code": "complete",
        "exit_code": 0,
        "group_drained": True,
        "resource_monitor": {
            "samples": [
                {
                    "ordinal": 0,
                    "process_group_rss_bytes": 7_301_447_680,
                    "gpu_memory_mib": 10_912,
                    "memory_psi_full_avg10_percent": 1.25,
                    "swap_growth_kib": 0,
                }
            ],
            "peak_process_group_rss_bytes": 7_301_447_680,
            "peak_gpu_memory_mib": 10_912,
            "peak_memory_psi_full_avg10_percent": 1.25,
            "maximum_swap_growth_kib": 0,
            "terminal_process_group_rss_bytes": 7_301_447_680,
            "terminal_gpu_memory_mib": 10_912,
            "terminal_memory_psi_full_avg10_percent": 1.25,
            "terminal_swap_growth_kib": 0,
            "immediate_psi_percent": 79.0,
            "sustained_psi_percent": 50.0,
            "sustained_samples": 3,
        },
        "verified_artifacts": {
            "map_artifact": identity,
            "fit_receipt": identity,
            "result": identity,
        },
    }
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert validate_coverage_execution_receipt_bytes(raw)["status"] == "complete"
    for mutation in (
        "exit",
        "missing-result",
        "claim",
        "phase",
        "not-drained",
        "monitor-type",
        "monitor-threshold",
        "monitor-terminal-over-peak",
        "monitor-peak-drift",
        "monitor-false-success",
        "monitor-rss-false-success",
        "monitor-gpu-false-success",
        "monitor-swap-false-success",
        "monitor-empty-success",
    ):
        changed = json.loads(raw)
        if mutation == "exit":
            changed["exit_code"] = 1
        elif mutation == "missing-result":
            changed["verified_artifacts"]["result"] = None
        elif mutation == "claim":
            changed["claim_eligible"] = True
        elif mutation == "not-drained":
            changed["group_drained"] = False
        elif mutation == "monitor-type":
            changed["resource_monitor"]["samples"][0]["ordinal"] = True
        elif mutation == "monitor-threshold":
            changed["resource_monitor"]["immediate_psi_percent"] = 0.79
        elif mutation == "monitor-terminal-over-peak":
            changed["resource_monitor"]["terminal_gpu_memory_mib"] = 10_913
        elif mutation == "monitor-peak-drift":
            changed["resource_monitor"]["peak_process_group_rss_bytes"] += 1
        elif mutation == "monitor-false-success":
            changed["resource_monitor"]["samples"][0]["memory_psi_full_avg10_percent"] = 79.0
            changed["resource_monitor"]["peak_memory_psi_full_avg10_percent"] = 79.0
            changed["resource_monitor"]["terminal_memory_psi_full_avg10_percent"] = 79.0
        elif mutation == "monitor-rss-false-success":
            changed["resource_monitor"]["samples"][0]["process_group_rss_bytes"] = 118_111_600_641
            changed["resource_monitor"]["peak_process_group_rss_bytes"] = 118_111_600_641
            changed["resource_monitor"]["terminal_process_group_rss_bytes"] = 118_111_600_641
        elif mutation == "monitor-gpu-false-success":
            changed["resource_monitor"]["samples"][0]["gpu_memory_mib"] = 98_305
            changed["resource_monitor"]["peak_gpu_memory_mib"] = 98_305
            changed["resource_monitor"]["terminal_gpu_memory_mib"] = 98_305
        elif mutation == "monitor-swap-false-success":
            changed["resource_monitor"]["samples"][0]["swap_growth_kib"] = 1
            changed["resource_monitor"]["maximum_swap_growth_kib"] = 1
            changed["resource_monitor"]["terminal_swap_growth_kib"] = 1
        elif mutation == "monitor-empty-success":
            changed["resource_monitor"].update(
                samples=[],
                peak_process_group_rss_bytes=0,
                peak_gpu_memory_mib=0,
                peak_memory_psi_full_avg10_percent=0.0,
                maximum_swap_growth_kib=0,
                terminal_process_group_rss_bytes=0,
                terminal_gpu_memory_mib=0,
                terminal_memory_psi_full_avg10_percent=0.0,
                terminal_swap_growth_kib=0,
            )
        else:
            changed["failed_phase"] = "fit"
        mutated = (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(ValueError, match="coverage execution receipt"):
            validate_coverage_execution_receipt_bytes(mutated)
    failed = json.loads(raw)
    failed.update(
        status="failed",
        failed_phase="preparation",
        reason_code="signal-term",
        exit_code=143,
        group_drained=False,
    )
    failed["verified_artifacts"] = {
        "map_artifact": None,
        "fit_receipt": None,
        "result": None,
    }
    failed_raw = (json.dumps(failed, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert validate_coverage_execution_receipt_bytes(failed_raw)["group_drained"] is False


def test_coverage_result_is_canonical_and_recomputes_all_decisions() -> None:
    raw = _result_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    result = validate_coverage_calibration_result_bytes(raw)
    assert result["schema"] == "sfora-siglip-coverage-calibration-v2"
    assert result["claim_eligible"] is False
    assert result["classification"] == "coverage-calibration-qualified"
    assert result["support_per_class"] == 8
    assert result["support_labels"] == [0] * 8 + [1] * 8
    assert result["offline_query_encoder_extra_ops"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "aggregate",
        "classification",
        "threshold",
        "digest",
        "bool-as-int",
        "overlap",
        "solver-rank",
        "environment",
        "execution-source",
        "image-digest",
        "support-class-drift",
        "support-membership",
    ),
)
def test_coverage_result_rejects_schema_type_and_derived_evidence_drift(
    mutation: str,
) -> None:
    value = json.loads(_result_bytes())
    if mutation == "aggregate":
        value["cells"]["student-self"]["micro_r1"] = 0.5
    elif mutation == "classification":
        value["classification"] = "coverage-calibration-rejected"
    elif mutation == "threshold":
        value["thresholds"]["micro_r1"] = 0.95
    elif mutation == "digest":
        value["inputs"]["checkpoint"]["sha256"] = "not-a-digest"
    elif mutation == "bool-as-int":
        value["support_per_class"] = True
    elif mutation == "overlap":
        value["support_ids"][0] = value["evaluation_ids"][0]
    elif mutation == "solver-rank":
        value["solver"]["student_to_teacher"]["rank"] = 2
    elif mutation == "environment":
        value["environment"]["torch_num_threads"] = True
    elif mutation == "execution-source":
        value["execution_source_commit"] = "bad"
    elif mutation == "image-digest":
        value["image_namespaces"]["evaluation"]["sha256"] = "bad"
    elif mutation == "support-class-drift":
        value["support_labels"][8:] = [2] * 8
    elif mutation == "support-membership":
        support_id = value["support_ids"][0]
        support_label = value["support_labels"][0]
        evaluation_id = next(
            identity
            for identity, label in zip(
                value["evaluation_ids"],
                value["cells"]["student-self"]["labels"],
                strict=True,
            )
            if label == support_label
        )
        support_pairs = list(zip(value["support_ids"], value["support_labels"], strict=True))
        support_pairs[0] = (evaluation_id, support_label)
        support_pairs.sort()
        value["support_ids"] = [identity for identity, _label in support_pairs]
        value["support_labels"] = [label for _identity, label in support_pairs]
        row_fields = (
            "ids",
            "hits",
            "average_precisions",
            "labels",
            "paired_cosines",
            "top10_overlaps",
            "hub_counts",
        )
        for cell in value["cells"].values():
            replacement_ids = [
                support_id if identity == evaluation_id else identity for identity in cell["ids"]
            ]
            order = sorted(range(len(replacement_ids)), key=replacement_ids.__getitem__)
            cell["ids"] = replacement_ids
            for field in row_fields[1:]:
                cell[field] = [cell[field][index] for index in order]
            cell["ids"] = [cell["ids"][index] for index in order]
        value["evaluation_ids"] = value["cells"]["student-self"]["ids"]
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match="coverage calibration result"):
        validate_coverage_calibration_result_bytes(raw)
