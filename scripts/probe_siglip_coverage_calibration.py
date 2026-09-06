#!/usr/bin/env python3
"""Run the local-only SigLIP coverage-calibration experiment."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections.abc import Callable, Iterable
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
    coverage_calibration_classification,
    coverage_retrieval_cells,
    coverage_support_indexes,
    fit_coverage_affine,
    validate_coverage_calibration_result_bytes,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed local-file-only calibration surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--evaluation-image-root", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--execution-source-commit", type=_git_sha, required=True)
    parser.add_argument("--map-artifact", type=_absolute_path, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-coverage-calibration", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


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
) -> dict[str, str]:
    digests = {
        "checkpoint_sha256": checkpoint_sha256,
        "control_binding_sha256": control_binding_sha256,
        "optimization_manifest_sha256": optimization_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "spatial_artifact_sha256": spatial_artifact_sha256,
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
) -> tuple[CoverageCalibrationRun, str]:
    """Fit and seal support-only maps before extracting evaluation descriptors."""

    if not callable(extract) or type(base_ids) is not tuple or type(candidate_ids) is not tuple:
        raise ValueError("coverage extractor authority differs")
    if set(base_ids) & set(candidate_ids):
        raise ValueError("coverage calibration identity overlap")
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support_ids = tuple(sorted(candidate_ids[index] for index in support_indexes))
    support_student, support_teacher = extract(support_indexes)
    fitting_student = torch.cat((base_student, support_student), dim=0).contiguous()
    fitting_teacher = torch.cat((base_teacher, support_teacher), dim=0).contiguous()
    fitting_ids = (*base_ids, *(candidate_ids[index] for index in support_indexes))
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
    )
    del support_student, support_teacher, fitting_student, fitting_teacher, maps
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
    cells = coverage_retrieval_cells(
        evaluation_student,
        evaluation_teacher,
        evaluation_ids,
        evaluation_labels,
        restored,
    )
    run = CoverageCalibrationRun(
        maps=restored,
        support_ids=support_ids,
        support_labels=tuple(
            candidate_labels[index]
            for index in sorted(support_indexes, key=candidate_ids.__getitem__)
        ),
        evaluation_ids=evaluation_ids,
        cells=cells,
        classification=coverage_calibration_classification(cells),
    )
    return run, digest


def main(argv: list[str] | None = None) -> int:
    """Run the complete frozen calibration experiment."""

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
    from probe_siglip_attention_readout_recovery import load_local_evaluation_manifest
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
    for output in (arguments.map_artifact, arguments.result):
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
    candidate_ids, candidate_labels, candidate_paths = load_local_evaluation_manifest(
        arguments.evaluation_manifest,
        arguments.evaluation_manifest_sha256,
        arguments.evaluation_image_root,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
    )
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support_set = set(support_indexes)
    evaluation_indexes = tuple(
        sorted(
            (index for index in range(len(candidate_ids)) if index not in support_set),
            key=candidate_ids.__getitem__,
        )
    )
    optimization_images_sha256 = coverage_image_namespace_sha256(base_ids, base_paths)

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

    base_student, base_teacher = stream_alignment_descriptor_pairs(
        vision,
        projection,
        tokenwise,
        readout,
        pixel_batches(base_paths),
        source_depth=18,
        device=device,
    )

    phase_image_digests: dict[str, str] = {}

    def extract(indexes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if indexes == support_indexes:
            phase = "support"
        elif indexes == evaluation_indexes:
            phase = "evaluation"
        else:
            raise ValueError("coverage calibration extraction partition differs")
        selected = tuple(candidate_paths[index] for index in indexes)
        selected_ids = tuple(candidate_ids[index] for index in indexes)
        phase_image_digests[phase] = coverage_image_namespace_sha256(selected_ids, selected)
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

    run, map_sha256 = seal_and_evaluate_coverage(
        base_student=base_student,
        base_teacher=base_teacher,
        base_ids=base_ids,
        candidate_ids=candidate_ids,
        candidate_labels=candidate_labels,
        extract=extract,
        map_artifact=arguments.map_artifact,
        checkpoint_sha256=seed17.sha256,
        control_binding_sha256=arguments.control_binding_sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        evaluation_manifest_sha256=arguments.evaluation_manifest_sha256,
        spatial_artifact_sha256=arguments.spatial_artifact_sha256,
    )
    if set(phase_image_digests) != {"support", "evaluation"}:
        raise ValueError("coverage calibration image evidence differs")

    def identity(path: Path, sha256: str) -> CoverageArtifactIdentity:
        return CoverageArtifactIdentity(
            path=str(path), sha256=sha256, byte_length=path.stat().st_size
        )

    raw = build_coverage_calibration_result(
        run,
        artifact_identities={
            "checkpoint": identity(arguments.checkpoint_seed17, seed17.sha256),
            "control_binding": identity(
                arguments.control_binding, arguments.control_binding_sha256
            ),
            "evaluation_manifest": identity(
                arguments.evaluation_manifest, arguments.evaluation_manifest_sha256
            ),
            "map_artifact": identity(arguments.map_artifact, map_sha256),
            "optimization_manifest": identity(
                arguments.optimization_manifest, arguments.optimization_manifest_sha256
            ),
            "spatial_artifact": identity(
                arguments.spatial_artifact, arguments.spatial_artifact_sha256
            ),
        },
        model_source_commit=binding.source_commit,
        execution_source_commit=arguments.execution_source_commit,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        optimization_image_root=str(arguments.optimization_image_root),
        evaluation_image_root=str(arguments.evaluation_image_root),
        optimization_images_sha256=optimization_images_sha256,
        support_images_sha256=phase_image_digests["support"],
        evaluation_images_sha256=phase_image_digests["evaluation"],
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
