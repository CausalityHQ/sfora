from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sfora.siglip_coverage_calibration import CoverageAffine, CoverageMaps

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_siglip_coverage_calibration.py"
_SPEC = importlib.util.spec_from_file_location("probe_siglip_coverage_calibration", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

load_coverage_map_artifact = _MODULE.load_coverage_map_artifact
coverage_image_namespace_sha256 = _MODULE.coverage_image_namespace_sha256
parse_args = _MODULE.parse_args
main = _MODULE.main
seal_and_evaluate_coverage = _MODULE.seal_and_evaluate_coverage
write_coverage_map_artifact = _MODULE.write_coverage_map_artifact


def _maps() -> CoverageMaps:
    forward = CoverageAffine(
        weight=torch.eye(2, dtype=torch.float64),
        bias=torch.tensor([0.1, -0.2], dtype=torch.float64),
        rank=3,
        singular_values=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    reverse = CoverageAffine(
        weight=torch.eye(2, dtype=torch.float64),
        bias=torch.tensor([-0.1, 0.2], dtype=torch.float64),
        rank=3,
        singular_values=torch.tensor([4.0, 2.0, 0.5], dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    return CoverageMaps(forward, reverse)


def _provenance() -> dict[str, str]:
    return {
        "checkpoint_sha256": "11" * 32,
        "control_binding_sha256": "22" * 32,
        "optimization_manifest_sha256": "33" * 32,
        "evaluation_manifest_sha256": "44" * 32,
        "spatial_artifact_sha256": "55" * 32,
        "optimization_images_sha256": "66" * 32,
        "support_images_sha256": "77" * 32,
    }


def test_coverage_image_namespace_digest_binds_ids_lengths_and_bytes(tmp_path: Path) -> None:
    paths = (tmp_path / "a.image", tmp_path / "b.image")
    paths[0].write_bytes(b"alpha")
    paths[1].write_bytes(b"beta")
    before = coverage_image_namespace_sha256(("a", "b"), paths)
    paths[1].write_bytes(b"changed")
    after = coverage_image_namespace_sha256(("a", "b"), paths)
    assert before != after
    assert before != coverage_image_namespace_sha256(("a", "b"), tuple(reversed(paths)))


def test_candidate_manifest_metadata_and_split_roots_are_independently_validated(
    tmp_path: Path,
) -> None:
    ids = tuple(f"candidate-{index:02d}" for index in range(20))
    labels = (49,) * 10 + (50,) * 10
    manifest = tmp_path / "manifest.json"
    raw = (
        json.dumps(
            {
                "schema": "sfora-attention-readout-evaluation-v1",
                "claim_eligible": False,
                "dataset_id": "fixture/cars",
                "dataset_revision": "b" * 40,
                "examples": [
                    {"example_id": identity, "label": label}
                    for identity, label in zip(ids, labels, strict=True)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    manifest.write_bytes(raw)
    observed_ids, observed_labels = _MODULE.load_coverage_candidate_manifest(
        manifest,
        hashlib.sha256(raw).hexdigest(),
        dataset_id="fixture/cars",
        dataset_revision="b" * 40,
        expected_count=20,
        expected_labels=frozenset((49, 50)),
    )
    support_indexes = _MODULE.coverage_support_indexes(observed_ids, observed_labels)
    support_ids = tuple(observed_ids[index] for index in support_indexes)
    heldout_ids = tuple(identity for identity in observed_ids if identity not in set(support_ids))
    support_root = tmp_path / "support"
    heldout_root = tmp_path / "heldout"
    support_root.mkdir()
    heldout_root.mkdir()
    for root, selected in ((support_root, support_ids), (heldout_root, heldout_ids)):
        for identity in selected:
            (root / _MODULE.coverage_image_basename(identity)).write_bytes(b"image")
    assert tuple(path.name for path in _MODULE.coverage_partition_paths(support_root, support_ids))
    assert tuple(path.name for path in _MODULE.coverage_partition_paths(heldout_root, heldout_ids))
    (support_root / "unexpected.image").write_bytes(b"leak")
    with pytest.raises(ValueError, match="partition namespace"):
        _MODULE.coverage_partition_paths(support_root, support_ids)


def test_heldout_denial_probe_requires_eacces_for_every_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heldout = tmp_path / "heldout"
    ids = ("candidate-a", "candidate-b")

    def denied_open(path: object, flags: int) -> int:
        raise PermissionError(13, "Permission denied", os.fspath(path))

    def denied_scandir(path: object) -> object:
        raise PermissionError(13, "Permission denied", os.fspath(path))

    monkeypatch.setattr(os, "open", denied_open)
    monkeypatch.setattr(os, "scandir", denied_scandir)
    assert _MODULE.verify_heldout_denied(heldout, ids) == {
        "rows": 2,
        "denied": 2,
        "directory_denied": True,
    }


def test_coverage_map_artifact_round_trips_exact_maps_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "maps.safetensors"
    support_ids = tuple(f"support-{index:03d}" for index in range(16))
    fitting_ids = tuple(f"fit-{index:03d}" for index in range(20))
    digest = write_coverage_map_artifact(
        path,
        _maps(),
        support_ids=support_ids,
        fitting_ids=fitting_ids,
        **_provenance(),
    )
    assert len(digest) == 64
    restored = load_coverage_map_artifact(
        path,
        expected_sha256=digest,
        support_ids=support_ids,
        fitting_ids=fitting_ids,
        **_provenance(),
    )
    assert torch.equal(restored.student_to_teacher.weight, _maps().student_to_teacher.weight)
    assert torch.equal(restored.student_to_teacher.bias, _maps().student_to_teacher.bias)
    assert torch.equal(restored.teacher_to_student.weight, _maps().teacher_to_student.weight)
    assert torch.equal(restored.teacher_to_student.bias, _maps().teacher_to_student.bias)


def test_coverage_map_artifact_rejects_provenance_and_digest_drift(tmp_path: Path) -> None:
    path = tmp_path / "maps.safetensors"
    support_ids = tuple(f"support-{index:03d}" for index in range(16))
    fitting_ids = tuple(f"fit-{index:03d}" for index in range(20))
    digest = write_coverage_map_artifact(
        path,
        _maps(),
        support_ids=support_ids,
        fitting_ids=fitting_ids,
        **_provenance(),
    )
    with pytest.raises(ValueError, match="artifact digest"):
        load_coverage_map_artifact(
            path,
            expected_sha256="aa" * 32,
            support_ids=support_ids,
            fitting_ids=fitting_ids,
            **_provenance(),
        )
    with pytest.raises(ValueError, match="artifact digest"):
        load_coverage_map_artifact(
            path,
            expected_sha256="invalid",
            support_ids=support_ids,
            fitting_ids=fitting_ids,
            **_provenance(),
        )
    changed = _provenance()
    changed["evaluation_manifest_sha256"] = "66" * 32
    with pytest.raises(ValueError, match="artifact schema"):
        load_coverage_map_artifact(
            path,
            expected_sha256=digest,
            support_ids=support_ids,
            fitting_ids=fitting_ids,
            **changed,
        )
    changed = _provenance()
    changed["support_images_sha256"] = "88" * 32
    with pytest.raises(ValueError, match="artifact schema"):
        load_coverage_map_artifact(
            path,
            expected_sha256=digest,
            support_ids=support_ids,
            fitting_ids=fitting_ids,
            **changed,
        )


def test_coverage_probe_cli_requires_frozen_local_surface(tmp_path: Path) -> None:
    digest = "11" * 32
    paths = [str(tmp_path / f"input-{index}") for index in range(12)]
    common = [
        "--phase",
        "fit",
        "--control-binding",
        paths[0],
        "--control-binding-sha256",
        digest,
        "--checkpoint-seed17",
        paths[1],
        "--optimization-manifest",
        paths[2],
        "--optimization-manifest-sha256",
        digest,
        "--optimization-image-root",
        paths[3],
        "--evaluation-manifest",
        paths[4],
        "--evaluation-manifest-sha256",
        digest,
        "--support-image-root",
        paths[5],
        "--heldout-image-root",
        paths[6],
        "--spatial-artifact",
        paths[7],
        "--spatial-artifact-sha256",
        digest,
        "--execution-source-commit",
        "aa" * 20,
        "--map-artifact",
        paths[8],
        "--fit-receipt",
        paths[9],
        "--execute-coverage-calibration",
    ]
    parsed = parse_args(common)
    assert parsed.phase == "fit"
    assert parsed.execute_coverage_calibration is True
    evaluate = [
        *("evaluate" if value == "fit" else value for value in common),
        "--fit-receipt-sha256",
        digest,
        "--result",
        paths[10],
    ]
    parsed = parse_args(evaluate)
    assert parsed.phase == "evaluate"
    for forbidden in (
        "--support-per-class",
        "--rcond",
        "--class-names",
        "--aws-profile",
        "--storage-uri",
    ):
        with pytest.raises(SystemExit):
            parse_args([*common, forbidden, "x"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--fit-receipt", paths[9]])
    with pytest.raises(SystemExit):
        parse_args([*("evaluate" if value == "fit" else value for value in common)])


def test_seal_and_evaluate_opens_evaluation_only_after_map_reload(tmp_path: Path) -> None:
    path = tmp_path / "maps.safetensors"
    base_student = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    transform = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    base_teacher = base_student @ transform
    offsets = torch.linspace(0.0, 0.09, 10)
    candidate_student = torch.cat(
        (
            torch.stack((torch.ones(10), offsets), dim=1),
            torch.stack((offsets, torch.ones(10)), dim=1),
        )
    )
    candidate_student = torch.nn.functional.normalize(candidate_student, dim=1)
    candidate_teacher = candidate_student @ transform
    candidate_ids = tuple(f"candidate-{index:02d}" for index in range(20))
    candidate_labels = (10,) * 10 + (11,) * 10
    calls: list[tuple[int, ...]] = []
    support_tensors: list[weakref.ReferenceType[torch.Tensor]] = []

    def extract(indexes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if calls:
            assert path.is_file()
            gc.collect()
            assert all(reference() is None for reference in support_tensors)
        else:
            assert not path.exists()
        calls.append(indexes)
        student = candidate_student[list(indexes)].clone()
        teacher = candidate_teacher[list(indexes)].clone()
        if not support_tensors:
            support_tensors.extend((weakref.ref(student), weakref.ref(teacher)))
        return student, teacher

    run, digest = seal_and_evaluate_coverage(
        base_student=base_student,
        base_teacher=base_teacher,
        base_ids=("base-a", "base-b", "base-c", "base-d"),
        candidate_ids=candidate_ids,
        candidate_labels=candidate_labels,
        extract=extract,
        map_artifact=path,
        **_provenance(),
    )
    assert len(digest) == 64
    assert len(calls) == 2
    assert set(calls[0]).isdisjoint(calls[1])
    assert len(calls[0]) == 16
    assert len(calls[1]) == 4
    assert run.classification == "coverage-calibration-qualified"


def test_split_fit_and_evaluate_match_monolithic_reference(tmp_path: Path) -> None:
    base_student = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    transform = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    base_teacher = base_student @ transform
    offsets = torch.linspace(0.0, 0.09, 10)
    candidate_student = torch.nn.functional.normalize(
        torch.cat(
            (
                torch.stack((torch.ones(10), offsets), dim=1),
                torch.stack((offsets, torch.ones(10)), dim=1),
            )
        ),
        dim=1,
    )
    candidate_teacher = candidate_student @ transform
    candidate_ids = tuple(f"candidate-{index:02d}" for index in range(20))
    candidate_labels = (10,) * 10 + (11,) * 10
    support_indexes = _MODULE.coverage_support_indexes(candidate_ids, candidate_labels)
    support_set = set(support_indexes)
    evaluation_indexes = tuple(
        sorted(
            (index for index in range(len(candidate_ids)) if index not in support_set),
            key=candidate_ids.__getitem__,
        )
    )
    support_ids = tuple(candidate_ids[index] for index in support_indexes)
    support_labels = tuple(candidate_labels[index] for index in support_indexes)
    evaluation_ids = tuple(candidate_ids[index] for index in evaluation_indexes)
    evaluation_labels = tuple(candidate_labels[index] for index in evaluation_indexes)
    map_path = tmp_path / "maps.safetensors"

    sealed = _MODULE.fit_and_seal_coverage(
        base_student=base_student,
        base_teacher=base_teacher,
        base_ids=("base-a", "base-b", "base-c", "base-d"),
        support_student=candidate_student[list(support_indexes)],
        support_teacher=candidate_teacher[list(support_indexes)],
        support_ids=support_ids,
        support_labels=support_labels,
        map_artifact=map_path,
        **_provenance(),
    )
    run = _MODULE.evaluate_sealed_coverage(
        sealed=sealed,
        evaluation_student=candidate_student[list(evaluation_indexes)],
        evaluation_teacher=candidate_teacher[list(evaluation_indexes)],
        evaluation_ids=evaluation_ids,
        evaluation_labels=evaluation_labels,
    )

    assert map_path.is_file()
    assert len(sealed.map_sha256) == 64
    assert run.support_ids == tuple(sorted(support_ids))
    assert run.evaluation_ids == tuple(sorted(evaluation_ids))
    assert run.classification == "coverage-calibration-qualified"


def test_seal_and_evaluate_rejects_base_candidate_identity_overlap(tmp_path: Path) -> None:
    descriptors = torch.nn.functional.normalize(torch.randn(4, 2), dim=1)
    candidate_ids = ("base-a", *(f"candidate-{index:02d}" for index in range(19)))
    candidate_labels = (10,) * 10 + (11,) * 10
    with pytest.raises(ValueError, match="identity overlap"):
        seal_and_evaluate_coverage(
            base_student=descriptors,
            base_teacher=descriptors,
            base_ids=("base-a", "base-b", "base-c", "base-d"),
            candidate_ids=candidate_ids,
            candidate_labels=candidate_labels,
            extract=lambda indexes: (_ for _ in ()).throw(AssertionError(indexes)),
            map_artifact=tmp_path / "maps.safetensors",
            **_provenance(),
        )


def test_coverage_probe_main_authenticates_seals_then_publishes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = {}
    for name in ("binding", "checkpoint", "optimization", "evaluation", "spatial"):
        path = tmp_path / name
        path.write_bytes(f"fixture-{name}".encode())
        inputs[name] = path
    optimization_root = tmp_path / "optimization-images"
    evaluation_root = tmp_path / "evaluation-images"
    optimization_root.mkdir()
    evaluation_root.mkdir()
    optimization_ids = tuple(f"base-{index:03d}" for index in range(49))
    optimization_labels = tuple(range(49))
    optimization_paths = tuple(optimization_root / f"{index}.image" for index in range(49))
    evaluation_ids = tuple(f"evaluation-{index:03d}" for index in range(20))
    evaluation_labels = (49,) * 10 + (50,) * 10
    evaluation_paths = tuple(evaluation_root / f"{index}.image" for index in range(20))
    for path in (*optimization_paths, *evaluation_paths):
        path.write_bytes(b"image")
    optimization_sha = hashlib.sha256(inputs["optimization"].read_bytes()).hexdigest()
    evaluation_sha = hashlib.sha256(inputs["evaluation"].read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(inputs["checkpoint"].read_bytes()).hexdigest()
    binding_sha = hashlib.sha256(inputs["binding"].read_bytes()).hexdigest()
    spatial_sha = hashlib.sha256(inputs["spatial"].read_bytes()).hexdigest()
    checkpoint_authority = SimpleNamespace(seed=17, sha256=checkpoint_sha)
    binding = SimpleNamespace(
        control_complete=True,
        optimization_manifest_sha256=optimization_sha,
        checkpoints=(
            checkpoint_authority,
            SimpleNamespace(seed=29),
            SimpleNamespace(seed=43),
        ),
        dataset_id="fixture/cars",
        dataset_revision="b" * 40,
        source_commit="a" * 40,
    )

    class Model:
        def __init__(self) -> None:
            vision = SimpleNamespace(post_layernorm=torch.nn.Identity(), head=torch.nn.Identity())
            self.tower = SimpleNamespace(vision_model=vision)
            self.projection = torch.nn.Linear(2, 2, bias=False)

        def eval(self) -> Model:
            return self

    fake_diagnose = types.ModuleType("diagnose_siglip_rsta_stage_a")
    fake_diagnose._read_regular = lambda path, role: path.read_bytes()
    fake_diagnose._parse_control_binding = lambda raw: binding
    fake_diagnose._load_model_state_checkpoint = lambda path, authority, value: object()
    fake_diagnose._load_optimization_manifest = lambda *args: (
        optimization_ids,
        optimization_labels,
        optimization_paths,
    )
    fake_diagnose.configure_stage_a_determinism = lambda: None
    fake_diagnose.load_stage_a_siglip_runtime = lambda: SimpleNamespace(
        model_factory=object(), disable_checkpointing=lambda model: None, processor=object()
    )
    fake_diagnose.load_stage_a_checkpoint_model = lambda *args, **kwargs: Model()
    fake_diagnose._stage_a_transforms = lambda processor: (object(), object())
    monkeypatch.setitem(sys.modules, "diagnose_siglip_rsta_stage_a", fake_diagnose)

    monkeypatch.setattr(
        _MODULE,
        "load_coverage_candidate_manifest",
        lambda *args, **kwargs: (
            evaluation_ids,
            evaluation_labels,
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "coverage_partition_paths",
        lambda root, ids: tuple(
            evaluation_paths[evaluation_ids.index(identity)] for identity in ids
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "verify_heldout_denied",
        lambda root, ids: {"rows": len(ids), "denied": len(ids), "directory_denied": True},
    )
    fake_attention = types.ModuleType("probe_siglip_attention_readout_recovery")
    fake_attention.load_local_evaluation_manifest = lambda *args, **kwargs: (
        evaluation_ids,
        evaluation_labels,
        evaluation_paths,
    )
    monkeypatch.setitem(sys.modules, "probe_siglip_attention_readout_recovery", fake_attention)

    fake_spatial = types.ModuleType("probe_siglip_spatial_tail_recovery")
    fake_spatial.FrozenTeacherReadout = lambda *args: torch.nn.Identity()
    fake_spatial.TokenwiseTailControl = lambda dimensions: torch.nn.Identity()
    fake_spatial.LatentInteractionTail = lambda dimensions: torch.nn.Identity()
    fake_spatial._load_spatial_tail_artifact = lambda *args: (
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    monkeypatch.setitem(sys.modules, "probe_siglip_spatial_tail_recovery", fake_spatial)

    calls = 0
    support_indexes = _MODULE.coverage_support_indexes(evaluation_ids, evaluation_labels)
    support_set = set(support_indexes)
    evaluation_indexes = tuple(
        index for index in range(len(evaluation_ids)) if index not in support_set
    )
    offsets = torch.linspace(0.01, 0.1, 10)
    candidate_student = torch.nn.functional.normalize(
        torch.cat(
            (
                torch.stack((torch.ones(10), offsets), dim=1),
                torch.stack((offsets, torch.ones(10)), dim=1),
            )
        ),
        dim=1,
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

    def stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        selected = (None, support_indexes, evaluation_indexes)[calls]
        calls += 1
        if selected is None:
            angles = torch.linspace(0.1, 1.4, 49)
            student = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        else:
            student = candidate_student[list(selected)]
        return student, student @ rotation

    fake_alignment = types.ModuleType("probe_siglip_gallery_compatibility_alignment")
    fake_alignment.stream_alignment_descriptor_pairs = stream
    monkeypatch.setitem(sys.modules, "probe_siglip_gallery_compatibility_alignment", fake_alignment)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    result = tmp_path / "result.json"
    maps = tmp_path / "maps.safetensors"
    receipt = tmp_path / "fit-receipt.json"
    common = [
        "--control-binding",
        str(inputs["binding"]),
        "--control-binding-sha256",
        binding_sha,
        "--checkpoint-seed17",
        str(inputs["checkpoint"]),
        "--optimization-manifest",
        str(inputs["optimization"]),
        "--optimization-manifest-sha256",
        optimization_sha,
        "--optimization-image-root",
        str(optimization_root),
        "--evaluation-manifest",
        str(inputs["evaluation"]),
        "--evaluation-manifest-sha256",
        evaluation_sha,
        "--support-image-root",
        str(evaluation_root),
        "--heldout-image-root",
        str(evaluation_root),
        "--spatial-artifact",
        str(inputs["spatial"]),
        "--spatial-artifact-sha256",
        spatial_sha,
        "--execution-source-commit",
        "a" * 40,
        "--map-artifact",
        str(maps),
        "--fit-receipt",
        str(receipt),
        "--execute-coverage-calibration",
    ]
    assert main(["--phase", "fit", *common]) == 0
    assert receipt.is_file()
    receipt_raw = receipt.read_bytes()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    monkeypatch.setattr(
        _MODULE,
        "fit_coverage_affine",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation must not invoke fitting")
        ),
    )
    changed_receipt = json.loads(receipt_raw)
    changed_receipt["inputs"]["control_binding"]["path"] = "/wrong/control-binding"
    changed_raw = (
        json.dumps(changed_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    receipt.write_bytes(changed_raw)
    with pytest.raises(ValueError, match="fit receipt input binding"):
        main(
            [
                "--phase",
                "evaluate",
                *common,
                "--fit-receipt-sha256",
                hashlib.sha256(changed_raw).hexdigest(),
                "--result",
                str(tmp_path / "mutated-result.json"),
            ]
        )
    receipt.write_bytes(receipt_raw)
    changed_receipt = json.loads(receipt_raw)
    changed_receipt["solver"]["student_to_teacher"]["singular_values"][0] *= 2
    changed_raw = (
        json.dumps(changed_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    receipt.write_bytes(changed_raw)
    with pytest.raises(ValueError, match="fit receipt solver binding"):
        main(
            [
                "--phase",
                "evaluate",
                *common,
                "--fit-receipt-sha256",
                hashlib.sha256(changed_raw).hexdigest(),
                "--result",
                str(tmp_path / "mutated-solver-result.json"),
            ]
        )
    receipt.write_bytes(receipt_raw)
    changed_receipt = json.loads(receipt_raw)
    changed_receipt["inputs"]["map_artifact"]["path"] = "/wrong/maps.safetensors"
    changed_raw = (
        json.dumps(changed_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    receipt.write_bytes(changed_raw)
    with pytest.raises(ValueError, match="fit receipt map binding"):
        main(
            [
                "--phase",
                "evaluate",
                *common,
                "--fit-receipt-sha256",
                hashlib.sha256(changed_raw).hexdigest(),
                "--result",
                str(tmp_path / "mutated-map-result.json"),
            ]
        )
    receipt.write_bytes(receipt_raw)
    assert (
        main(
            [
                "--phase",
                "evaluate",
                *common,
                "--fit-receipt-sha256",
                receipt_sha,
                "--result",
                str(result),
            ]
        )
        == 0
    )
    assert calls == 3
    assert maps.is_file()
    assert result.is_file()
    assert (
        _MODULE.validate_coverage_calibration_result_bytes(result.read_bytes())["classification"]
        == "coverage-calibration-qualified"
    )
