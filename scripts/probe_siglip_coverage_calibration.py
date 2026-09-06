#!/usr/bin/env python3
"""Run the local-only SigLIP coverage-calibration experiment."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sfora.siglip_coverage_calibration import (
    CoverageAffine,
    CoverageArtifactIdentity,
    CoverageCalibrationRun,
    CoverageMaps,
    build_coverage_calibration_result,
    build_coverage_fit_receipt,
    coverage_calibration_classification,
    coverage_retrieval_cells,
    coverage_support_indexes,
    fit_coverage_affine,
    validate_coverage_calibration_result_bytes,
    validate_coverage_fit_receipt_bytes,
)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if not _valid_sha256(value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def _git_sha(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("revision must be a lowercase Git SHA")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not any(character not in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("coverage calibration input must be one regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_image_namespace_sha256(ids: tuple[str, ...], paths: tuple[Path, ...]) -> str:
    """Bind an ordered image namespace to exact regular-file lengths and bytes."""

    if (
        type(ids) is not tuple
        or type(paths) is not tuple
        or not ids
        or len(ids) != len(paths)
        or len(set(ids)) != len(ids)
        or tuple(sorted(ids)) != ids
        or any(type(value) is not str or not value for value in ids)
    ):
        raise ValueError("coverage image namespace authority differs")
    digest = hashlib.sha256(b"sfora-coverage-image-namespace-v1\0")
    for identity, path in zip(ids, paths, strict=True):
        if (
            not isinstance(path, Path)
            or not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise ValueError("coverage image namespace authority differs")
        encoded = identity.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def coverage_image_basename(example_id: str) -> str:
    """Return the frozen local image name for one example identity."""

    if type(example_id) is not str or not example_id:
        raise ValueError("coverage candidate identity differs")
    digest = hashlib.sha256(
        b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")
    ).hexdigest()
    return f"{digest}.image"


def load_coverage_candidate_manifest(
    path: Path,
    expected_sha256: str,
    *,
    dataset_id: str,
    dataset_revision: str,
    expected_count: int = 2_746,
    expected_labels: frozenset[int] = frozenset(range(49, 82)),
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Authenticate candidate identities and labels without opening image roots."""

    raw = path.read_bytes() if isinstance(path, Path) and path.is_file() else b""
    try:
        value = json.loads(raw)
        canonical = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("coverage candidate manifest differs") from error
    if (
        not _valid_sha256(expected_sha256)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or type(value) is not dict
        or set(value) != {"schema", "claim_eligible", "dataset_id", "dataset_revision", "examples"}
        or raw != canonical
        or value["schema"] != "sfora-attention-readout-evaluation-v1"
        or value["claim_eligible"] is not False
        or value["dataset_id"] != dataset_id
        or value["dataset_revision"] != dataset_revision
        or type(value["examples"]) is not list
        or len(value["examples"]) != expected_count
        or type(expected_count) is not int
        or expected_count < 1
        or type(expected_labels) is not frozenset
        or not expected_labels
    ):
        raise ValueError("coverage candidate manifest authority differs")
    ids: list[str] = []
    labels: list[int] = []
    for row in value["examples"]:
        if (
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or row["label"] not in expected_labels
        ):
            raise ValueError("coverage candidate manifest row differs")
        ids.append(row["example_id"])
        labels.append(row["label"])
    if (
        tuple(sorted(ids)) != tuple(ids)
        or len(set(ids)) != expected_count
        or set(labels) != set(expected_labels)
    ):
        raise ValueError("coverage candidate manifest authority differs")
    return tuple(ids), tuple(labels)


def coverage_partition_paths(root: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve exactly one support or held-out image namespace."""

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("coverage partition root differs") from error
    if (
        type(ids) is not tuple
        or not ids
        or len(set(ids)) != len(ids)
        or tuple(sorted(ids)) != ids
        or root.is_symlink()
        or not root.is_dir()
        or resolved_root != root
    ):
        raise ValueError("coverage partition namespace differs")
    paths = tuple(root / coverage_image_basename(identity) for identity in ids)
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError("coverage partition image differs") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or resolved != path
            or not resolved.is_relative_to(resolved_root)
        ):
            raise ValueError("coverage partition image differs")
    if {entry.name for entry in root.iterdir()} != {path.name for path in paths}:
        raise ValueError("coverage partition namespace differs")
    return paths


def verify_heldout_denied(root: Path, ids: tuple[str, ...]) -> dict[str, object]:
    """Require the fit sandbox to deny every held-out pixel and directory listing."""

    if type(ids) is not tuple or not ids or tuple(sorted(ids)) != ids:
        raise ValueError("coverage heldout denial authority differs")
    denied = 0
    for identity in ids:
        path = root / coverage_image_basename(identity)
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except PermissionError as error:
            if error.errno != errno.EACCES:
                raise ValueError("coverage heldout denial differs") from error
            denied += 1
        else:
            os.close(descriptor)
            raise ValueError("coverage heldout pixel was readable during fit")
    try:
        iterator = os.scandir(root)
    except PermissionError as error:
        if error.errno != errno.EACCES:
            raise ValueError("coverage heldout denial differs") from error
    else:
        iterator.close()
        raise ValueError("coverage heldout directory was readable during fit")
    return {"rows": len(ids), "denied": denied, "directory_denied": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed local-file-only calibration surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--support-image-root", type=_absolute_path, required=True)
    parser.add_argument("--heldout-image-root", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--execution-source-commit", type=_git_sha, required=True)
    parser.add_argument("--map-artifact", type=_absolute_path, required=True)
    parser.add_argument("--fit-receipt", type=_absolute_path, required=True)
    parser.add_argument("--fit-receipt-sha256", type=_sha256)
    parser.add_argument("--result", type=_absolute_path)
    parser.add_argument("--execute-coverage-calibration", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    parsed = parser.parse_args(effective)
    if parsed.phase == "fit" and (
        parsed.fit_receipt_sha256 is not None or parsed.result is not None
    ):
        parser.error("fit phase forbids evaluation-only arguments")
    if parsed.phase == "evaluate" and (parsed.fit_receipt_sha256 is None or parsed.result is None):
        parser.error("evaluate phase requires receipt digest and result")
    return parsed


def _ids_sha256(ids: tuple[str, ...], *, domain: bytes) -> str:
    if (
        type(ids) is not tuple
        or not ids
        or len(set(ids)) != len(ids)
        or tuple(sorted(ids)) != ids
        or any(type(value) is not str or not value for value in ids)
        or type(domain) is not bytes
        or not domain
    ):
        raise ValueError("coverage map artifact support differs")
    digest = hashlib.sha256(domain)
    for value in ids:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _artifact_metadata(
    *,
    support_ids: tuple[str, ...],
    fitting_ids: tuple[str, ...],
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    spatial_artifact_sha256: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
) -> dict[str, str]:
    digests = {
        "checkpoint_sha256": checkpoint_sha256,
        "control_binding_sha256": control_binding_sha256,
        "optimization_manifest_sha256": optimization_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "spatial_artifact_sha256": spatial_artifact_sha256,
        "optimization_images_sha256": optimization_images_sha256,
        "support_images_sha256": support_images_sha256,
    }
    if any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests.values()
    ):
        raise ValueError("coverage map artifact provenance differs")
    return {
        "schema": "sfora-siglip-coverage-maps-v1",
        **digests,
        "support_ids_sha256": _ids_sha256(support_ids, domain=b"sfora-coverage-map-support-v1\0"),
        "fitting_ids_sha256": _ids_sha256(fitting_ids, domain=b"sfora-coverage-map-fitting-v1\0"),
        "support_rows": str(len(support_ids)),
        "rcond": "1e-12",
        "driver": "gelsd",
    }


def _map_tensors(maps: CoverageMaps) -> dict[str, torch.Tensor]:
    if type(maps) is not CoverageMaps:
        raise ValueError("coverage map artifact authority differs")
    return {
        "student_to_teacher.bias": maps.student_to_teacher.bias,
        "student_to_teacher.singular_values": maps.student_to_teacher.singular_values,
        "student_to_teacher.weight": maps.student_to_teacher.weight,
        "teacher_to_student.bias": maps.teacher_to_student.bias,
        "teacher_to_student.singular_values": maps.teacher_to_student.singular_values,
        "teacher_to_student.weight": maps.teacher_to_student.weight,
    }


def write_coverage_map_artifact(
    path: Path,
    maps: CoverageMaps,
    *,
    support_ids: tuple[str, ...],
    fitting_ids: tuple[str, ...],
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    spatial_artifact_sha256: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
) -> str:
    """Atomically seal both affine directions and their exact authority."""

    if not isinstance(path, Path) or path.exists() or path.is_symlink():
        raise FileExistsError(path)
    metadata = _artifact_metadata(
        support_ids=support_ids,
        fitting_ids=fitting_ids,
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
    )
    tensors = _map_tensors(maps)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(tensors.items())), str(partial), metadata=metadata)
    restored = load_file(str(partial), device="cpu")
    if set(restored) != set(tensors) or any(
        not torch.equal(restored[name], tensor) for name, tensor in tensors.items()
    ):
        partial.unlink(missing_ok=True)
        raise ValueError("coverage map artifact replay differs")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(path)
    return digest


def load_coverage_map_artifact(
    path: Path,
    *,
    expected_sha256: str,
    support_ids: tuple[str, ...],
    fitting_ids: tuple[str, ...],
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    spatial_artifact_sha256: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
) -> CoverageMaps:
    """Authenticate and restore both sealed affine directions."""

    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("coverage map artifact authority differs")
    raw = path.read_bytes()
    if not _valid_sha256(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("coverage map artifact digest differs")
    expected_metadata = _artifact_metadata(
        support_ids=support_ids,
        fitting_ids=fitting_ids,
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
    )
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if stream.metadata() != expected_metadata:
            raise ValueError("coverage map artifact schema differs")
    tensors = load_file(str(path), device="cpu")
    expected_names = {
        f"{direction}.{name}"
        for direction in ("student_to_teacher", "teacher_to_student")
        for name in ("weight", "bias", "singular_values")
    }
    if set(tensors) != expected_names:
        raise ValueError("coverage map artifact schema differs")

    def mapping(direction: str) -> CoverageAffine:
        weight = tensors[f"{direction}.weight"]
        dimensions = int(weight.shape[0]) if weight.ndim == 2 else -1
        return CoverageAffine(
            weight=weight,
            bias=tensors[f"{direction}.bias"],
            rank=dimensions + 1,
            singular_values=tensors[f"{direction}.singular_values"],
            rcond=1e-12,
            driver="gelsd",
        )

    return CoverageMaps(mapping("student_to_teacher"), mapping("teacher_to_student"))


@dataclass(frozen=True, slots=True)
class SealedCoverageFit:
    """Authenticated fit-only output consumed by the evaluation phase."""

    maps: CoverageMaps
    support_ids: tuple[str, ...]
    support_labels: tuple[int, ...]
    map_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.maps) is not CoverageMaps
            or type(self.support_ids) is not tuple
            or type(self.support_labels) is not tuple
            or not self.support_ids
            or tuple(sorted(self.support_ids)) != self.support_ids
            or len(self.support_labels) != len(self.support_ids)
            or any(type(label) is not int or label < 0 for label in self.support_labels)
            or any(self.support_labels.count(label) != 8 for label in set(self.support_labels))
            or not _valid_sha256(self.map_sha256)
        ):
            raise ValueError("coverage sealed fit authority differs")


def fit_and_seal_coverage(
    *,
    base_student: torch.Tensor,
    base_teacher: torch.Tensor,
    base_ids: tuple[str, ...],
    support_student: torch.Tensor,
    support_teacher: torch.Tensor,
    support_ids: tuple[str, ...],
    support_labels: tuple[int, ...],
    map_artifact: Path,
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    spatial_artifact_sha256: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
) -> SealedCoverageFit:
    """Fit and seal maps without accepting evaluation descriptors."""

    if (
        type(base_ids) is not tuple
        or type(support_ids) is not tuple
        or type(support_labels) is not tuple
        or set(base_ids) & set(support_ids)
        or tuple(sorted(support_ids)) != support_ids
        or len(support_labels) != len(support_ids)
    ):
        raise ValueError("coverage fitting authority differs")
    fitting_student = torch.cat((base_student, support_student), dim=0).contiguous()
    fitting_teacher = torch.cat((base_teacher, support_teacher), dim=0).contiguous()
    fitting_ids = (*base_ids, *support_ids)
    sealed_fitting_ids = tuple(sorted(fitting_ids))
    maps = CoverageMaps(
        student_to_teacher=fit_coverage_affine(fitting_student, fitting_teacher, fitting_ids),
        teacher_to_student=fit_coverage_affine(fitting_teacher, fitting_student, fitting_ids),
    )
    digest = write_coverage_map_artifact(
        map_artifact,
        maps,
        support_ids=support_ids,
        fitting_ids=sealed_fitting_ids,
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
    )
    restored = load_coverage_map_artifact(
        map_artifact,
        expected_sha256=digest,
        support_ids=support_ids,
        fitting_ids=sealed_fitting_ids,
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
    )
    del fitting_student, fitting_teacher, maps
    return SealedCoverageFit(restored, support_ids, support_labels, digest)


def evaluate_sealed_coverage(
    *,
    sealed: SealedCoverageFit,
    evaluation_student: torch.Tensor,
    evaluation_teacher: torch.Tensor,
    evaluation_ids: tuple[str, ...],
    evaluation_labels: tuple[int, ...],
) -> CoverageCalibrationRun:
    """Evaluate an authenticated sealed fit without exposing a fitting API."""

    if type(sealed) is not SealedCoverageFit or set(sealed.support_ids) & set(evaluation_ids):
        raise ValueError("coverage evaluation authority differs")
    cells = coverage_retrieval_cells(
        evaluation_student,
        evaluation_teacher,
        evaluation_ids,
        evaluation_labels,
        sealed.maps,
    )
    return CoverageCalibrationRun(
        maps=sealed.maps,
        support_ids=sealed.support_ids,
        support_labels=sealed.support_labels,
        evaluation_ids=evaluation_ids,
        cells=cells,
        classification=coverage_calibration_classification(cells),
    )


def coverage_solver_evidence(maps: CoverageMaps) -> dict[str, dict[str, object]]:
    """Reconstruct exact fit-receipt solver evidence from authenticated tensors."""

    if type(maps) is not CoverageMaps:
        raise ValueError("coverage solver evidence differs")

    def mapping(value: CoverageAffine) -> dict[str, object]:
        return {
            "dimensions": value.weight.shape[0],
            "rank": value.rank,
            "singular_values": [float(item) for item in value.singular_values],
            "rcond": value.rcond,
            "driver": value.driver,
        }

    return {
        "student_to_teacher": mapping(maps.student_to_teacher),
        "teacher_to_student": mapping(maps.teacher_to_student),
    }


def seal_and_evaluate_coverage(
    *,
    base_student: torch.Tensor,
    base_teacher: torch.Tensor,
    base_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    candidate_labels: tuple[int, ...],
    extract: Callable[[tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
    map_artifact: Path,
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    spatial_artifact_sha256: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
) -> tuple[CoverageCalibrationRun, str]:
    """Fit and seal support-only maps before extracting evaluation descriptors."""

    if not callable(extract) or type(base_ids) is not tuple or type(candidate_ids) is not tuple:
        raise ValueError("coverage extractor authority differs")
    if set(base_ids) & set(candidate_ids):
        raise ValueError("coverage calibration identity overlap")
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support_ids = tuple(sorted(candidate_ids[index] for index in support_indexes))
    ordered_support_indexes = tuple(sorted(support_indexes, key=candidate_ids.__getitem__))
    support_labels = tuple(candidate_labels[index] for index in ordered_support_indexes)
    support_student, support_teacher = extract(support_indexes)
    sealed = fit_and_seal_coverage(
        base_student=base_student,
        base_teacher=base_teacher,
        base_ids=base_ids,
        support_student=support_student,
        support_teacher=support_teacher,
        support_ids=support_ids,
        support_labels=support_labels,
        map_artifact=map_artifact,
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
    )
    del support_student, support_teacher
    support = set(support_indexes)
    evaluation_indexes = tuple(
        sorted(
            (index for index in range(len(candidate_ids)) if index not in support),
            key=candidate_ids.__getitem__,
        )
    )
    evaluation_ids = tuple(candidate_ids[index] for index in evaluation_indexes)
    evaluation_labels = tuple(candidate_labels[index] for index in evaluation_indexes)
    evaluation_student, evaluation_teacher = extract(evaluation_indexes)
    run = evaluate_sealed_coverage(
        sealed=sealed,
        evaluation_student=evaluation_student,
        evaluation_teacher=evaluation_teacher,
        evaluation_ids=evaluation_ids,
        evaluation_labels=evaluation_labels,
    )
    return run, sealed.map_sha256


def main(argv: list[str] | None = None) -> int:
    """Run one explicitly selected fit or evaluation phase."""

    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from diagnose_siglip_rsta_stage_a import (
        _load_model_state_checkpoint,
        _load_optimization_manifest,
        _parse_control_binding,
        _stage_a_transforms,
        configure_stage_a_determinism,
        load_stage_a_checkpoint_model,
        load_stage_a_siglip_runtime,
    )
    from diagnose_siglip_rsta_stage_a import _read_regular as read_rsta_regular
    from PIL import Image
    from probe_siglip_gallery_compatibility_alignment import (
        stream_alignment_descriptor_pairs,
    )
    from probe_siglip_spatial_tail_recovery import (
        FrozenTeacherReadout,
        LatentInteractionTail,
        TokenwiseTailControl,
        _load_spatial_tail_artifact,
    )

    arguments = parse_args(argv)
    outputs = (
        (arguments.map_artifact, arguments.fit_receipt)
        if arguments.phase == "fit"
        else (arguments.result,)
    )
    for output in outputs:
        assert output is not None
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    if _file_sha256(arguments.spatial_artifact) != arguments.spatial_artifact_sha256:
        raise ValueError("coverage calibration spatial artifact digest differs")
    binding_raw = read_rsta_regular(arguments.control_binding, role="coverage calibration binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("coverage calibration binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("coverage calibration binding authority differs")
    seed17 = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(arguments.checkpoint_seed17, seed17, binding)
    base_ids, base_labels, base_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.optimization_image_root,
    )
    if set(base_labels) != set(range(49)):
        raise ValueError("coverage calibration optimization partition differs")
    candidate_ids, candidate_labels = load_coverage_candidate_manifest(
        arguments.evaluation_manifest,
        arguments.evaluation_manifest_sha256,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
    )
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support_set = set(support_indexes)
    evaluation_indexes = tuple(
        index for index in range(len(candidate_ids)) if index not in support_set
    )
    support_ids = tuple(candidate_ids[index] for index in support_indexes)
    support_labels = tuple(candidate_labels[index] for index in support_indexes)
    evaluation_ids = tuple(candidate_ids[index] for index in evaluation_indexes)
    evaluation_labels = tuple(candidate_labels[index] for index in evaluation_indexes)
    support_paths = coverage_partition_paths(arguments.support_image_root, support_ids)
    heldout_probe: dict[str, object] | None = None
    evaluation_paths: tuple[Path, ...] | None = None
    if arguments.phase == "fit":
        heldout_probe = verify_heldout_denied(arguments.heldout_image_root, evaluation_ids)
    else:
        evaluation_paths = coverage_partition_paths(arguments.heldout_image_root, evaluation_ids)
    optimization_images_sha256 = coverage_image_namespace_sha256(base_ids, base_paths)
    support_images_sha256 = coverage_image_namespace_sha256(support_ids, support_paths)

    def identity(path: Path, sha256: str) -> CoverageArtifactIdentity:
        return CoverageArtifactIdentity(
            path=str(path), sha256=sha256, byte_length=path.stat().st_size
        )

    common_identities = {
        "checkpoint": identity(arguments.checkpoint_seed17, seed17.sha256),
        "control_binding": identity(arguments.control_binding, arguments.control_binding_sha256),
        "evaluation_manifest": identity(
            arguments.evaluation_manifest, arguments.evaluation_manifest_sha256
        ),
        "optimization_manifest": identity(
            arguments.optimization_manifest, arguments.optimization_manifest_sha256
        ),
        "spatial_artifact": identity(arguments.spatial_artifact, arguments.spatial_artifact_sha256),
    }

    sealed: SealedCoverageFit | None = None
    fit_receipt_sha256: str | None = None
    if arguments.phase == "evaluate":
        assert arguments.fit_receipt_sha256 is not None
        receipt_raw = arguments.fit_receipt.read_bytes()
        fit_receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        if fit_receipt_sha256 != arguments.fit_receipt_sha256:
            raise ValueError("coverage fit receipt digest differs")
        receipt = validate_coverage_fit_receipt_bytes(receipt_raw)
        receipt_inputs = cast(dict[str, dict[str, object]], receipt["inputs"])
        if any(
            receipt_inputs[role] != item.to_mapping() for role, item in common_identities.items()
        ):
            raise ValueError("coverage fit receipt input binding differs")
        receipt_namespaces = cast(dict[str, dict[str, object]], receipt["image_namespaces"])
        if (
            receipt["model_source_commit"] != binding.source_commit
            or receipt["execution_source_commit"] != arguments.execution_source_commit
            or receipt["dataset_id"] != binding.dataset_id
            or receipt["dataset_revision"] != binding.dataset_revision
            or receipt["support_ids"] != list(support_ids)
            or receipt["support_labels"] != list(support_labels)
            or receipt_namespaces["optimization"]
            != {"sha256": optimization_images_sha256, "rows": len(base_ids)}
            or receipt_namespaces["support"]
            != {"sha256": support_images_sha256, "rows": len(support_ids)}
        ):
            raise ValueError("coverage fit receipt authority differs")
        map_identity = receipt_inputs["map_artifact"]
        map_sha256 = cast(str, map_identity["sha256"])
        if map_identity != identity(arguments.map_artifact, map_sha256).to_mapping():
            raise ValueError("coverage fit receipt map binding differs")
        fitting_ids = tuple(sorted((*base_ids, *support_ids)))
        maps = load_coverage_map_artifact(
            arguments.map_artifact,
            expected_sha256=map_sha256,
            support_ids=support_ids,
            fitting_ids=fitting_ids,
            checkpoint_sha256=seed17.sha256,
            control_binding_sha256=arguments.control_binding_sha256,
            optimization_manifest_sha256=arguments.optimization_manifest_sha256,
            evaluation_manifest_sha256=arguments.evaluation_manifest_sha256,
            spatial_artifact_sha256=arguments.spatial_artifact_sha256,
            optimization_images_sha256=optimization_images_sha256,
            support_images_sha256=support_images_sha256,
        )
        if receipt["solver"] != coverage_solver_evidence(maps):
            raise ValueError("coverage fit receipt solver binding differs")
        sealed = SealedCoverageFit(maps, support_ids, support_labels, map_sha256)

    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("coverage calibration requires CUDA bf16")
    device = torch.device("cuda")
    runtime = load_stage_a_siglip_runtime()
    model = load_stage_a_checkpoint_model(
        checkpoint, model_factory=runtime.model_factory, device=device
    )
    runtime.disable_checkpointing(model)
    model.eval()
    vision = model.tower.vision_model
    projection = model.projection
    _graph_transform, evaluation_transform = _stage_a_transforms(runtime.processor)
    readout_template = FrozenTeacherReadout(vision.post_layernorm, vision.head, projection).eval()
    tokenwise, _interaction, readout = _load_spatial_tail_artifact(
        arguments.spatial_artifact,
        TokenwiseTailControl(1152).eval(),
        LatentInteractionTail(1152).eval(),
        readout_template,
    )
    tokenwise = tokenwise.to(device)
    readout = readout.to(device)

    def pixel_batches(paths: tuple[Path, ...]) -> Iterable[torch.Tensor]:
        for start in range(0, len(paths), 32):
            tensors: list[torch.Tensor] = []
            for path in paths[start : start + 32]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("coverage calibration image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    def extract(selected: tuple[Path, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            stream_alignment_descriptor_pairs(
                vision,
                projection,
                tokenwise,
                readout,
                pixel_batches(selected),
                source_depth=18,
                device=device,
            ),
        )

    if arguments.phase == "fit":
        assert heldout_probe is not None
        base_student, base_teacher = extract(base_paths)
        support_student, support_teacher = extract(support_paths)
        sealed = fit_and_seal_coverage(
            base_student=base_student,
            base_teacher=base_teacher,
            base_ids=base_ids,
            support_student=support_student,
            support_teacher=support_teacher,
            support_ids=support_ids,
            support_labels=support_labels,
            map_artifact=arguments.map_artifact,
            checkpoint_sha256=seed17.sha256,
            control_binding_sha256=arguments.control_binding_sha256,
            optimization_manifest_sha256=arguments.optimization_manifest_sha256,
            evaluation_manifest_sha256=arguments.evaluation_manifest_sha256,
            spatial_artifact_sha256=arguments.spatial_artifact_sha256,
            optimization_images_sha256=optimization_images_sha256,
            support_images_sha256=support_images_sha256,
        )
        fit_identities = {
            **common_identities,
            "map_artifact": identity(arguments.map_artifact, sealed.map_sha256),
        }
        raw = build_coverage_fit_receipt(
            maps=sealed.maps,
            support_ids=support_ids,
            support_labels=support_labels,
            artifact_identities=fit_identities,
            model_source_commit=binding.source_commit,
            execution_source_commit=arguments.execution_source_commit,
            dataset_id=binding.dataset_id,
            dataset_revision=binding.dataset_revision,
            optimization_image_root=str(arguments.optimization_image_root),
            support_image_root=str(arguments.support_image_root),
            optimization_images_sha256=optimization_images_sha256,
            support_images_sha256=support_images_sha256,
            optimization_rows=len(base_ids),
            heldout_rows=cast(int, heldout_probe["rows"]),
            heldout_denied=cast(int, heldout_probe["denied"]),
            heldout_directory_denied=cast(bool, heldout_probe["directory_denied"]),
            torch_version=torch.__version__,
            torch_num_threads=torch.get_num_threads(),
            blas_config=torch.__config__.show().strip(),
            cpu_identity=platform.processor() or platform.machine(),
        )
        arguments.fit_receipt.parent.mkdir(parents=True, exist_ok=True)
        partial = arguments.fit_receipt.with_name(f"{arguments.fit_receipt.name}.partial")
        if partial.exists() or partial.is_symlink():
            raise FileExistsError(partial)
        partial.write_bytes(raw)
        partial.replace(arguments.fit_receipt)
        print(
            "coverage-fit:COMPLETE "
            f"map_sha256={sealed.map_sha256} "
            f"receipt_sha256={hashlib.sha256(raw).hexdigest()}",
            flush=True,
        )
        return 0

    assert sealed is not None and evaluation_paths is not None and arguments.result is not None
    evaluation_student, evaluation_teacher = extract(evaluation_paths)
    evaluation_images_sha256 = coverage_image_namespace_sha256(evaluation_ids, evaluation_paths)
    run = evaluate_sealed_coverage(
        sealed=sealed,
        evaluation_student=evaluation_student,
        evaluation_teacher=evaluation_teacher,
        evaluation_ids=evaluation_ids,
        evaluation_labels=evaluation_labels,
    )
    assert fit_receipt_sha256 is not None

    raw = build_coverage_calibration_result(
        run,
        artifact_identities={
            **common_identities,
            "fit_receipt": identity(arguments.fit_receipt, fit_receipt_sha256),
            "map_artifact": identity(arguments.map_artifact, sealed.map_sha256),
        },
        model_source_commit=binding.source_commit,
        execution_source_commit=arguments.execution_source_commit,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        optimization_image_root=str(arguments.optimization_image_root),
        evaluation_image_root=str(arguments.heldout_image_root),
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=support_images_sha256,
        evaluation_images_sha256=evaluation_images_sha256,
        optimization_rows=len(base_ids),
        torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        blas_config=torch.__config__.show().strip(),
        cpu_identity=platform.processor() or platform.machine(),
    )
    validate_coverage_calibration_result_bytes(raw)
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    print(
        f"coverage-calibration:COMPLETE sha256={hashlib.sha256(raw).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
