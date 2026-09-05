#!/usr/bin/env python3
"""Fit and evaluate a local-only orthogonal student-to-teacher alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_gallery_compatibility_alignment import (
    apply_orthogonal_alignment,
    build_alignment_result,
    fit_orthogonal_alignment,
)
from sfora.siglip_spatial_tail_recovery import spatial_retrieval_evidence


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict optimization/development-only alignment command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--alignment-artifact", type=_absolute_path, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-gallery-alignment", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("alignment input must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_registered_optimization_images(
    manifest_path: Path,
    output_root: Path,
    *,
    dataset_loader: Callable[..., object] | None = None,
) -> None:
    """Decode only the source rows named by a sealed optimization manifest."""

    from PIL import Image

    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("alignment optimization manifest differs")
    raw = manifest_path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("alignment optimization manifest differs") from error
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if (
        raw != canonical
        or type(value) is not dict
        or set(value)
        != {"claim_eligible", "dataset_id", "dataset_revision", "examples", "schema"}
        or value["schema"] != "rsta-optimization-manifest-v1"
        or value["claim_eligible"] is not False
        or value["dataset_id"] != "tanganke/stanford_cars"
        or value["dataset_revision"]
        != "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
        or type(value["examples"]) is not list
        or not value["examples"]
    ):
        raise ValueError("alignment optimization manifest differs")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    train = dataset_loader(
        value["dataset_id"], revision=value["dataset_revision"], split="train"
    )
    test = dataset_loader(
        value["dataset_id"], revision=value["dataset_revision"], split="test"
    )
    if not hasattr(train, "__len__") or not hasattr(test, "__len__"):
        raise TypeError("alignment Cars datasets differ")
    rows = cast(tuple[Sequence[object], Sequence[object]], (train, test))
    train_length = len(rows[0])
    output_root.mkdir(parents=True)
    seen: set[str] = set()
    try:
        for item in cast(list[object], value["examples"]):
            if type(item) is not dict or set(item) != {"example_id", "label"}:
                raise ValueError("alignment optimization example differs")
            example_id = item["example_id"]
            label = item["label"]
            if (
                type(example_id) is not str
                or not example_id.startswith("cars-train-")
                or type(label) is not int
                or not 0 <= label <= 48
                or example_id in seen
            ):
                raise ValueError("alignment optimization example differs")
            prefix, label_text, index_text = example_id.rsplit("-", 2)
            if prefix != "cars-train" or label_text != str(label) or not index_text.isdigit():
                raise ValueError("alignment optimization example differs")
            global_index = int(index_text)
            source, local_index = (
                (rows[0], global_index)
                if global_index < train_length
                else (rows[1], global_index - train_length)
            )
            if not 0 <= local_index < len(source):
                raise ValueError("alignment optimization source index differs")
            record = source[local_index]
            if type(record) is not dict or record.get("label") != label:
                raise ValueError("alignment optimization source row differs")
            image = record.get("image")
            if not isinstance(image, Image.Image):
                raise ValueError("alignment optimization image differs")
            basename = (
                hashlib.sha256(
                    b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")
                ).hexdigest()
                + ".image"
            )
            with (output_root / basename).open("xb") as stream:
                image.convert("RGB").save(
                    stream, format="PNG", optimize=False, compress_level=0
                )
            seen.add(example_id)
    except BaseException:
        for path in output_root.iterdir():
            path.unlink()
        output_root.rmdir()
        raise


def stream_alignment_descriptor_pairs(
    vision_model: nn.Module,
    projection: nn.Linear,
    tail: nn.Module,
    readout: nn.Module,
    pixel_batches: Iterable[torch.Tensor],
    *,
    source_depth: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream normalized student/teacher pairs without retaining token fields."""

    if (
        vision_model.training
        or projection.training
        or tail.training
        or readout.training
        or projection.bias is not None
        or type(source_depth) is not int
        or source_depth != 18
        or device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("alignment model authority differs")
    students: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    with torch.inference_mode():
        for pixels in pixel_batches:
            if (
                type(pixels) is not torch.Tensor
                or pixels.ndim != 4
                or pixels.shape[0] < 1
                or not pixels.is_floating_point()
                or not bool(torch.isfinite(pixels).all())
            ):
                raise ValueError("alignment pixel authority differs")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = vision_model(
                    pixel_values=pixels.to(device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = getattr(output, "hidden_states", None)
                pooler = getattr(output, "pooler_output", None)
                if (
                    type(hidden_states) not in {tuple, list}
                    or len(hidden_states) != 28
                    or type(hidden_states[source_depth]) is not torch.Tensor
                    or type(pooler) is not torch.Tensor
                ):
                    raise ValueError("alignment hidden-state authority differs")
                student = readout(tail(hidden_states[source_depth]))
                teacher = F.normalize(projection(pooler.float()), dim=1)
            student = F.normalize(student.float(), dim=1)
            teacher = F.normalize(teacher.float(), dim=1)
            if (
                type(student) is not torch.Tensor
                or type(teacher) is not torch.Tensor
                or student.shape != teacher.shape
                or student.ndim != 2
                or not bool(torch.isfinite(student).all())
                or not bool(torch.isfinite(teacher).all())
            ):
                raise ValueError("alignment descriptor authority differs")
            students.append(student.cpu().contiguous())
            teachers.append(teacher.cpu().contiguous())
    if not students:
        raise ValueError("alignment descriptor stream is empty")
    return torch.cat(students).contiguous(), torch.cat(teachers).contiguous()


def write_alignment_artifact(
    path: Path,
    matrix: torch.Tensor,
    development_student: torch.Tensor,
    development_aligned: torch.Tensor,
) -> str:
    """Seal the orthogonal matrix and identity-bound self-geometry evidence."""

    from safetensors.torch import load_file, save_file

    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if (
        type(matrix) is not torch.Tensor
        or matrix.device.type != "cpu"
        or matrix.dtype != torch.float64
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or type(development_student) is not torch.Tensor
        or type(development_aligned) is not torch.Tensor
        or development_student.device.type != "cpu"
        or development_aligned.device.type != "cpu"
        or development_student.dtype != torch.float32
        or development_aligned.dtype != torch.float32
        or development_student.ndim != 2
        or development_student.shape != development_aligned.shape
        or development_student.shape[0] < 2
        or development_student.shape[1] != matrix.shape[0]
        or not bool(torch.isfinite(development_student).all())
        or not bool(torch.isfinite(development_aligned).all())
    ):
        raise ValueError("alignment matrix authority differs")
    if not torch.equal(
        apply_orthogonal_alignment(development_student, matrix), development_aligned
    ):
        raise ValueError("alignment descriptor evidence differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    save_file(
        {
            "development_aligned": development_aligned.contiguous(),
            "development_student": development_student.contiguous(),
            "student_to_teacher": matrix.contiguous(),
        },
        str(partial),
        metadata={"schema": "sfora-siglip-gallery-compatibility-alignment-v1"},
    )
    restored = load_file(str(partial), device="cpu")
    expected = {
        "development_aligned": development_aligned,
        "development_student": development_student,
        "student_to_teacher": matrix,
    }
    if set(restored) != set(expected) or any(
        not torch.equal(restored[name], tensor) for name, tensor in expected.items()
    ):
        partial.unlink(missing_ok=True)
        raise ValueError("alignment artifact replay differs")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(path)
    return digest


def load_alignment_artifact(path: Path) -> torch.Tensor:
    """Load a sealed orthogonal alignment artifact."""

    from safetensors.torch import load_file

    if not path.is_file() or path.is_symlink():
        raise ValueError("alignment artifact authority differs")
    values = load_file(str(path), device="cpu")
    if set(values) != {
        "development_aligned",
        "development_student",
        "student_to_teacher",
    }:
        raise ValueError("alignment artifact schema differs")
    matrix = values["student_to_teacher"]
    student = values["development_student"]
    aligned = values["development_aligned"]
    probe = torch.eye(matrix.shape[0], dtype=torch.float32)
    apply_orthogonal_alignment(probe, matrix)
    if (
        student.dtype != torch.float32
        or aligned.dtype != torch.float32
        or student.ndim != 2
        or student.shape != aligned.shape
        or student.shape[1] != matrix.shape[0]
        or not torch.equal(apply_orthogonal_alignment(student, matrix), aligned)
    ):
        raise ValueError("alignment descriptor evidence differs")
    return matrix


def main(argv: list[str] | None = None) -> int:
    """Run one fitting-only alignment and burned-development evaluation."""

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
    from probe_siglip_spatial_tail_recovery import (
        FrozenTeacherReadout,
        LatentInteractionTail,
        TokenwiseTailControl,
        _load_spatial_tail_artifact,
    )

    from sfora.siglip_spatial_tail_recovery import spatial_tail_class_split

    arguments = parse_args(argv)
    for output in (arguments.alignment_artifact, arguments.result):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    if _file_sha256(arguments.spatial_artifact) != arguments.spatial_artifact_sha256:
        raise ValueError("alignment spatial artifact digest differs")
    binding_raw = read_rsta_regular(arguments.control_binding, role="alignment binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("alignment binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("alignment binding authority differs")
    seed17 = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(arguments.checkpoint_seed17, seed17, binding)
    example_ids, labels, paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.optimization_image_root,
    )
    fit_labels, development_labels = spatial_tail_class_split(tuple(range(49)))
    fit_indexes = tuple(index for index, label in enumerate(labels) if label in fit_labels)
    development_indexes = tuple(
        index for index, label in enumerate(labels) if label in development_labels
    )
    if not fit_indexes or not development_indexes:
        raise ValueError("alignment class partition differs")

    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("gallery compatibility alignment requires CUDA bf16")
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

    def pixel_batches(selected_paths: tuple[Path, ...]) -> Iterable[torch.Tensor]:
        for start in range(0, len(selected_paths), 32):
            tensors: list[torch.Tensor] = []
            for path in selected_paths[start : start + 32]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("alignment image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    fitting_student, fitting_teacher = stream_alignment_descriptor_pairs(
        vision,
        projection,
        tokenwise,
        readout,
        pixel_batches(tuple(paths[index] for index in fit_indexes)),
        source_depth=18,
        device=device,
    )
    matrix = fit_orthogonal_alignment(fitting_student, fitting_teacher)
    fitting_aligned = apply_orthogonal_alignment(fitting_student, matrix)

    development_student, development_teacher = stream_alignment_descriptor_pairs(
        vision,
        projection,
        tokenwise,
        readout,
        pixel_batches(tuple(paths[index] for index in development_indexes)),
        source_depth=18,
        device=device,
    )
    development_aligned = apply_orthogonal_alignment(development_student, matrix)
    artifact_sha = write_alignment_artifact(
        arguments.alignment_artifact,
        matrix,
        development_student,
        development_aligned,
    )
    matrix = load_alignment_artifact(arguments.alignment_artifact)
    fitting_aligned = apply_orthogonal_alignment(fitting_student, matrix)
    development_aligned = apply_orthogonal_alignment(development_student, matrix)
    ids = tuple(example_ids[index] for index in development_indexes)
    development_label_values = tuple(labels[index] for index in development_indexes)

    def evidence(query: torch.Tensor, gallery: torch.Tensor):  # type: ignore[no-untyped-def]
        return spatial_retrieval_evidence(
            query,
            gallery,
            query_ids=ids,
            gallery_ids=ids,
            query_labels=development_label_values,
            gallery_labels=development_label_values,
        )

    cells = {
        "teacher-teacher": evidence(development_teacher, development_teacher),
        "student-student": evidence(development_student, development_student),
        "student-teacher": evidence(development_student, development_teacher),
        "teacher-student": evidence(development_teacher, development_student),
        "aligned-aligned": evidence(development_aligned, development_aligned),
        "aligned-teacher": evidence(development_aligned, development_teacher),
        "teacher-aligned": evidence(development_teacher, development_aligned),
    }

    def paired_cosines(left: torch.Tensor, right: torch.Tensor) -> tuple[float, ...]:
        return tuple(
            float(value) for value in (left.double() * right.double()).sum(dim=1).clamp(-1, 1)
        )

    orthogonality_error = float(
        (matrix.T @ matrix - torch.eye(matrix.shape[0], dtype=torch.float64)).abs().max()
    )
    self_geometry_max_score_delta = float(
        (
            development_student.double() @ development_student.double().T
            - development_aligned.double() @ development_aligned.double().T
        )
        .abs()
        .max()
    )
    raw = build_alignment_result(
        checkpoint_sha256=seed17.sha256,
        spatial_artifact_sha256=arguments.spatial_artifact_sha256,
        alignment_artifact_sha256=artifact_sha,
        fit_labels=tuple(sorted(fit_labels)),
        development_labels=tuple(sorted(development_labels)),
        cells=cells,
        fidelity={
            "fit-before": paired_cosines(fitting_student, fitting_teacher),
            "fit-after": paired_cosines(fitting_aligned, fitting_teacher),
            "development-before": paired_cosines(development_student, development_teacher),
            "development-after": paired_cosines(development_aligned, development_teacher),
        },
        orthogonality_error=orthogonality_error,
        self_geometry_max_score_delta=self_geometry_max_score_delta,
    )
    if not isinstance(json.loads(raw), Mapping):
        raise ValueError("alignment result differs")
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    print(
        f"gallery-alignment:COMPLETE sha256={hashlib.sha256(raw).hexdigest()} ",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
