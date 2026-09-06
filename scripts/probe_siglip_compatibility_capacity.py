#!/usr/bin/env python3
"""Run a local-only SigLIP gallery-compatibility capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from sfora.siglip_compatibility_capacity import (
    CompatibilityRetrievalEvidence,
    ResidualFit,
    build_compatibility_capacity_result,
    combine_compatibility_retrieval_evidence,
    compatibility_folds,
    compatibility_oracle_halves,
    compatibility_retrieval_evidence,
    csls_scores,
    fit_centered_similarity,
    fit_regularized_affine,
    fit_teacher_anchored_residual,
    select_compatibility_finalist,
)
from sfora.siglip_spatial_tail_recovery import spatial_tail_class_split


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
    """Parse the strict burned-data-only capacity diagnostic command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--descriptor-artifact", type=_absolute_path, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-capacity-diagnostic", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


@dataclass(frozen=True, slots=True)
class CapacityDescriptorArtifact:
    """Reloaded identity-bound student and teacher descriptor evidence."""

    student: torch.Tensor
    teacher: torch.Tensor
    ids: tuple[str, ...]
    labels: tuple[int, ...]
    fit_labels: tuple[int, ...]
    development_labels: tuple[int, ...]


def _validated_descriptor_inputs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.device.type != "cpu"
        or teacher.device.type != "cpu"
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] != len(ids)
        or student.shape[1] < 2
        or type(ids) is not tuple
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
        or type(labels) is not tuple
        or len(labels) != len(ids)
        or any(type(value) is not int for value in labels)
        or set(labels) != set(range(49))
        or min(labels.count(label) for label in range(49)) < 2
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or bool((torch.linalg.vector_norm(student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(teacher, dim=1) <= 0).any())
    ):
        raise ValueError("capacity descriptor artifact authority differs")
    fit, development = spatial_tail_class_split(tuple(range(49)))
    return tuple(sorted(fit)), tuple(sorted(development))


def _id_digest_tensor(ids: tuple[str, ...]) -> torch.Tensor:
    return torch.tensor(
        [
            list(
                hashlib.sha256(
                    b"sfora-compatibility-capacity-id-v1\0" + value.encode("utf-8")
                ).digest()
            )
            for value in ids
        ],
        dtype=torch.uint8,
    )


def _descriptor_metadata(
    *,
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    spatial_artifact_sha256: str,
    image_manifest_sha256: str,
    preprocessing: str,
) -> dict[str, str]:
    values = {
        "checkpoint_sha256": checkpoint_sha256,
        "control_binding_sha256": control_binding_sha256,
        "optimization_manifest_sha256": optimization_manifest_sha256,
        "spatial_artifact_sha256": spatial_artifact_sha256,
        "image_manifest_sha256": image_manifest_sha256,
    }
    if (
        any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in values.values()
        )
        or preprocessing != "siglip-evaluation-transform-v1"
    ):
        raise ValueError("capacity descriptor artifact provenance differs")
    return {
        "schema": "sfora-siglip-compatibility-capacity-descriptors-v1",
        **values,
        "preprocessing": preprocessing,
    }


def capacity_image_manifest_sha256(ids: tuple[str, ...], paths: tuple[Path, ...]) -> str:
    """Bind ordered example identities to exact authorized image bytes."""

    if (
        type(ids) is not tuple
        or type(paths) is not tuple
        or len(ids) != len(paths)
        or not ids
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
        or any(
            not isinstance(path, Path) or not path.is_file() or path.is_symlink() for path in paths
        )
    ):
        raise ValueError("capacity image manifest authority differs")
    digest = hashlib.sha256(b"sfora-compatibility-capacity-images-v1\0")
    for identity, path in zip(ids, paths, strict=True):
        encoded = identity.encode("utf-8")
        image_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(image_digest)
    return digest.hexdigest()


def write_capacity_descriptor_artifact(
    path: Path,
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
    *,
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    spatial_artifact_sha256: str,
    image_manifest_sha256: str,
    preprocessing: str,
) -> str:
    """Atomically seal the one-pass descriptor bank and identity evidence."""

    from safetensors.torch import load_file, save_file

    _validated_descriptor_inputs(student, teacher, ids, labels)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    expected = {
        "student": student.contiguous(),
        "teacher": teacher.contiguous(),
        "id_sha256": _id_digest_tensor(ids),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    save_file(
        expected,
        str(partial),
        metadata=_descriptor_metadata(
            checkpoint_sha256=checkpoint_sha256,
            control_binding_sha256=control_binding_sha256,
            optimization_manifest_sha256=optimization_manifest_sha256,
            spatial_artifact_sha256=spatial_artifact_sha256,
            image_manifest_sha256=image_manifest_sha256,
            preprocessing=preprocessing,
        ),
    )
    restored = load_file(str(partial), device="cpu")
    if set(restored) != set(expected) or any(
        not torch.equal(restored[name], value) for name, value in expected.items()
    ):
        partial.unlink(missing_ok=True)
        raise ValueError("capacity descriptor artifact replay differs")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(path)
    return digest


def load_capacity_descriptor_artifact(
    path: Path,
    *,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
    checkpoint_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    spatial_artifact_sha256: str,
    image_manifest_sha256: str,
    preprocessing: str,
) -> CapacityDescriptorArtifact:
    """Authenticate and reload a sealed descriptor bank against manifest identity."""

    from safetensors import safe_open
    from safetensors.torch import load_file

    if not path.is_file() or path.is_symlink():
        raise ValueError("capacity descriptor artifact authority differs")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    if metadata != _descriptor_metadata(
        checkpoint_sha256=checkpoint_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing=preprocessing,
    ):
        raise ValueError("capacity descriptor artifact schema differs")
    values = load_file(str(path), device="cpu")
    if set(values) != {"student", "teacher", "id_sha256", "labels"}:
        raise ValueError("capacity descriptor artifact schema differs")
    stored_labels = tuple(int(value) for value in values["labels"].tolist())
    fit, development = _validated_descriptor_inputs(
        values["student"], values["teacher"], ids, stored_labels
    )
    if stored_labels != labels or not torch.equal(values["id_sha256"], _id_digest_tensor(ids)):
        raise ValueError("capacity descriptor artifact identity differs")
    return CapacityDescriptorArtifact(
        student=values["student"],
        teacher=values["teacher"],
        ids=ids,
        labels=labels,
        fit_labels=fit,
        development_labels=development,
    )


def _indexes_for_labels(labels: tuple[int, ...], selected: set[int]) -> torch.Tensor:
    indexes = [index for index, label in enumerate(labels) if label in selected]
    if len(indexes) < 2:
        raise ValueError("capacity class partition differs")
    return torch.tensor(indexes, dtype=torch.int64)


def _evidence(
    query: torch.Tensor,
    gallery: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> CompatibilityRetrievalEvidence:
    return compatibility_retrieval_evidence(
        query,
        gallery,
        query_ids=ids,
        gallery_ids=ids,
        query_labels=labels,
        gallery_labels=labels,
        reference_query=teacher,
        reference_gallery=teacher,
    )


def _mapped_fold_evidence(
    mapped: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence]:
    forward = _evidence(mapped, teacher, teacher, ids, labels)
    reverse = _evidence(teacher, mapped, teacher, ids, labels)
    return forward, reverse


def _residual_optimization_evidence(fit: ResidualFit) -> dict[str, object]:
    state_dict = fit.state_dict
    dimensions = int(state_dict["up.bias"].shape[0])
    return {
        "relational": fit.relational,
        "seed": fit.seed,
        "dimensions": dimensions,
        "parameter_count": sum(value.numel() for value in state_dict.values()),
        "anchor_ids": list(fit.anchor_ids),
        "losses": {name: list(values) for name, values in fit.losses.items()},
        "device": fit.device,
        "torch_version": fit.torch_version,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }


def _r1_hits_from_scores(scores: torch.Tensor, labels: tuple[int, ...]) -> tuple[bool, ...]:
    if scores.shape != (len(labels), len(labels)):
        raise ValueError("capacity score authority differs")
    scores = scores.clone()
    scores.diagonal().fill_(-torch.inf)
    ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, 0]
    return tuple(labels[int(index)] == labels[row] for row, index in enumerate(ranked))


def cross_fitted_oracle_evidence(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[
    CompatibilityRetrievalEvidence,
    CompatibilityRetrievalEvidence,
    CompatibilityRetrievalEvidence,
    tuple[dict[str, object], ...],
]:
    """Fit on one development half and score only within the disjoint half."""

    oracle_halves = compatibility_oracle_halves(tuple(sorted(set(labels))))
    panels: list[tuple[CompatibilityRetrievalEvidence, ...]] = [(), (), ()]
    optimization: list[dict[str, object]] = []
    for fold_index, (training_labels, validation_labels) in enumerate(
        (oracle_halves, tuple(reversed(oracle_halves)))
    ):
        training_indexes = _indexes_for_labels(labels, set(training_labels))
        validation_indexes = _indexes_for_labels(labels, set(validation_labels))
        training_ids = tuple(ids[index] for index in training_indexes.tolist())
        validation_ids = tuple(ids[index] for index in validation_indexes.tolist())
        validation_label_values = tuple(labels[index] for index in validation_indexes.tolist())
        oracle = fit_teacher_anchored_residual(
            student[training_indexes],
            teacher[training_indexes],
            training_ids,
            relational=True,
            seed=20260909 + fold_index,
        )
        optimization.append(_residual_optimization_evidence(oracle))
        mapped = oracle.apply(student[validation_indexes])
        validation_teacher = teacher[validation_indexes]
        panel = (
            _evidence(
                mapped,
                validation_teacher,
                validation_teacher,
                validation_ids,
                validation_label_values,
            ),
            _evidence(
                validation_teacher,
                mapped,
                validation_teacher,
                validation_ids,
                validation_label_values,
            ),
            _evidence(
                mapped,
                mapped,
                validation_teacher,
                validation_ids,
                validation_label_values,
            ),
        )
        panels = [current + (value,) for current, value in zip(panels, panel, strict=True)]
        print(f"capacity:oracle={fold_index + 1}/2", flush=True)
    combined = tuple(
        combine_compatibility_retrieval_evidence(part, expected_ids=ids) for part in panels
    )
    return combined[0], combined[1], combined[2], tuple(optimization)


def held_out_panel_evidence(
    mapped: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[
    CompatibilityRetrievalEvidence,
    CompatibilityRetrievalEvidence,
    CompatibilityRetrievalEvidence,
]:
    """Score one frozen mapping on the same disjoint panels as the oracle."""

    panels: list[tuple[CompatibilityRetrievalEvidence, ...]] = [(), (), ()]
    for validation_labels in compatibility_oracle_halves(tuple(sorted(set(labels)))):
        indexes = _indexes_for_labels(labels, set(validation_labels))
        panel_ids = tuple(ids[index] for index in indexes.tolist())
        panel_labels = tuple(labels[index] for index in indexes.tolist())
        panel_mapped = mapped[indexes]
        panel_teacher = teacher[indexes]
        evidence = (
            _evidence(panel_mapped, panel_teacher, panel_teacher, panel_ids, panel_labels),
            _evidence(panel_teacher, panel_mapped, panel_teacher, panel_ids, panel_labels),
            _evidence(panel_mapped, panel_mapped, panel_teacher, panel_ids, panel_labels),
        )
        panels = [current + (value,) for current, value in zip(panels, evidence, strict=True)]
    return tuple(
        combine_compatibility_retrieval_evidence(part, expected_ids=ids) for part in panels
    )  # type: ignore[return-value]


def _validate_capacity_anchor_coverage(
    labels: tuple[int, ...],
    fit_labels: tuple[int, ...],
    development_labels: tuple[int, ...],
) -> None:
    fit_folds = compatibility_folds(fit_labels)
    fit_label_set = set(fit_labels)
    development_halves = compatibility_oracle_halves(development_labels)
    anchor_groups = [fit_label_set - set(validation_labels) for validation_labels in fit_folds] + [
        set(training_labels) for training_labels in development_halves
    ]
    if any(len(_indexes_for_labels(labels, group)) < 256 for group in anchor_groups):
        raise ValueError("capacity anchor coverage differs")


def analyze_capacity_descriptors(
    artifact: CapacityDescriptorArtifact,
    *,
    checkpoint_sha256: str,
    descriptor_artifact_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    spatial_artifact_sha256: str,
    image_manifest_sha256: str,
    preprocessing: str,
) -> bytes:
    """Run the registered fitting-fold and burned-development diagnostics."""

    fit_folds = compatibility_folds(artifact.fit_labels)
    fit_label_set = set(artifact.fit_labels)
    _validate_capacity_anchor_coverage(
        artifact.labels, artifact.fit_labels, artifact.development_labels
    )
    fold_evidence: dict[
        str,
        list[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence]],
    ] = {
        "affine-0.0001": [],
        "affine-0.01": [],
        "affine-1": [],
        "teacher-anchored-residual": [],
        "centered-similarity": [],
        "paired-only-residual": [],
    }
    residual_optimization: dict[str, list[dict[str, object]]] = {
        "teacher-anchored-residual": [],
        "paired-only-residual": [],
    }
    for fold_index, validation_labels in enumerate(fit_folds):
        validation = set(validation_labels)
        training_indexes = _indexes_for_labels(artifact.labels, fit_label_set - validation)
        validation_indexes = _indexes_for_labels(artifact.labels, validation)
        training_student = artifact.student[training_indexes]
        training_teacher = artifact.teacher[training_indexes]
        training_ids = tuple(artifact.ids[index] for index in training_indexes.tolist())
        validation_student = artifact.student[validation_indexes]
        validation_teacher = artifact.teacher[validation_indexes]
        validation_ids = tuple(artifact.ids[index] for index in validation_indexes.tolist())
        validation_label_values = tuple(
            artifact.labels[index] for index in validation_indexes.tolist()
        )
        for regularization, name in (
            (1e-4, "affine-0.0001"),
            (1e-2, "affine-0.01"),
            (1.0, "affine-1"),
        ):
            mapping = fit_regularized_affine(training_student, training_teacher, regularization)
            fold_evidence[name].append(
                _mapped_fold_evidence(
                    mapping.apply(validation_student),
                    validation_teacher,
                    validation_ids,
                    validation_label_values,
                )
            )
        centered = fit_centered_similarity(training_student, training_teacher)
        fold_evidence["centered-similarity"].append(
            _mapped_fold_evidence(
                centered.apply(validation_student),
                validation_teacher,
                validation_ids,
                validation_label_values,
            )
        )
        residual = fit_teacher_anchored_residual(
            training_student,
            training_teacher,
            training_ids,
            relational=True,
            seed=20260905 + fold_index,
        )
        fold_evidence["teacher-anchored-residual"].append(
            _mapped_fold_evidence(
                residual.apply(validation_student),
                validation_teacher,
                validation_ids,
                validation_label_values,
            )
        )
        residual_optimization["teacher-anchored-residual"].append(
            _residual_optimization_evidence(residual)
        )
        paired_only = fit_teacher_anchored_residual(
            training_student,
            training_teacher,
            training_ids,
            relational=False,
            seed=20260905 + fold_index,
        )
        fold_evidence["paired-only-residual"].append(
            _mapped_fold_evidence(
                paired_only.apply(validation_student),
                validation_teacher,
                validation_ids,
                validation_label_values,
            )
        )
        residual_optimization["paired-only-residual"].append(
            _residual_optimization_evidence(paired_only)
        )
        print(f"capacity:fold={fold_index + 1}/3", flush=True)
    frozen_fold_evidence = {name: tuple(values) for name, values in fold_evidence.items()}
    finalist = select_compatibility_finalist(
        {
            name: tuple(
                (forward.class_macro_map_at_r, reverse.class_macro_map_at_r)
                for forward, reverse in frozen_fold_evidence[name]
            )
            for name in (
                "affine-0.0001",
                "affine-0.01",
                "affine-1",
                "teacher-anchored-residual",
            )
        }
    )
    fit_indexes = _indexes_for_labels(artifact.labels, fit_label_set)
    development_indexes = _indexes_for_labels(artifact.labels, set(artifact.development_labels))
    fitting_student = artifact.student[fit_indexes]
    fitting_teacher = artifact.teacher[fit_indexes]
    fitting_ids = tuple(artifact.ids[index] for index in fit_indexes.tolist())
    fitting_label_values = tuple(artifact.labels[index] for index in fit_indexes.tolist())
    fitting_identity = _evidence(
        fitting_student,
        fitting_teacher,
        fitting_teacher,
        fitting_ids,
        fitting_label_values,
    )
    if finalist == "teacher-anchored-residual":
        fitted = fit_teacher_anchored_residual(
            fitting_student,
            fitting_teacher,
            fitting_ids,
            relational=True,
            seed=20260908,
        )
        apply_finalist = fitted.apply
        finalist_refit_optimization: dict[str, object] | None = _residual_optimization_evidence(
            fitted
        )
    else:
        regularization = {
            "affine-0.0001": 1e-4,
            "affine-0.01": 1e-2,
            "affine-1": 1.0,
        }[finalist]
        affine = fit_regularized_affine(fitting_student, fitting_teacher, regularization)
        apply_finalist = affine.apply
        finalist_refit_optimization = None
    development_student = artifact.student[development_indexes]
    development_teacher = artifact.teacher[development_indexes]
    development_ids = tuple(artifact.ids[index] for index in development_indexes.tolist())
    development_label_values = tuple(
        artifact.labels[index] for index in development_indexes.tolist()
    )
    finalist_mapped = apply_finalist(development_student)
    identity_forward = _evidence(
        development_student,
        development_teacher,
        development_teacher,
        development_ids,
        development_label_values,
    )
    identity_reverse = _evidence(
        development_teacher,
        development_student,
        development_teacher,
        development_ids,
        development_label_values,
    )
    identity_self = _evidence(
        development_student,
        development_student,
        development_teacher,
        development_ids,
        development_label_values,
    )

    (
        oracle_forward,
        oracle_reverse,
        oracle_self,
        oracle_optimization,
    ) = cross_fitted_oracle_evidence(
        development_student,
        development_teacher,
        development_ids,
        development_label_values,
    )
    identity_panel_forward, identity_panel_reverse, identity_panel_self = held_out_panel_evidence(
        development_student,
        development_teacher,
        development_ids,
        development_label_values,
    )
    finalist_panel_forward, finalist_panel_reverse, finalist_panel_self = held_out_panel_evidence(
        finalist_mapped,
        development_teacher,
        development_ids,
        development_label_values,
    )
    cells = {
        "identity-forward": identity_forward,
        "identity-reverse": identity_reverse,
        "identity-self": identity_self,
        "identity-oracle-panel-forward": identity_panel_forward,
        "identity-oracle-panel-reverse": identity_panel_reverse,
        "identity-oracle-panel-self": identity_panel_self,
        "finalist-forward": _evidence(
            finalist_mapped,
            development_teacher,
            development_teacher,
            development_ids,
            development_label_values,
        ),
        "finalist-reverse": _evidence(
            development_teacher,
            finalist_mapped,
            development_teacher,
            development_ids,
            development_label_values,
        ),
        "finalist-self": _evidence(
            finalist_mapped,
            finalist_mapped,
            development_teacher,
            development_ids,
            development_label_values,
        ),
        "finalist-oracle-panel-forward": finalist_panel_forward,
        "finalist-oracle-panel-reverse": finalist_panel_reverse,
        "finalist-oracle-panel-self": finalist_panel_self,
        "oracle-forward": oracle_forward,
        "oracle-reverse": oracle_reverse,
        "oracle-self": oracle_self,
    }
    csls_forward = csls_scores(finalist_mapped, development_teacher, neighbors=10)
    csls_reverse = csls_scores(development_teacher, finalist_mapped, neighbors=10)
    identity_csls_forward = csls_scores(development_student, development_teacher, neighbors=10)
    identity_csls_reverse = csls_scores(development_teacher, development_student, neighbors=10)
    return build_compatibility_capacity_result(
        checkpoint_sha256=checkpoint_sha256,
        descriptor_artifact_sha256=descriptor_artifact_sha256,
        control_binding_sha256=control_binding_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        spatial_artifact_sha256=spatial_artifact_sha256,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing=preprocessing,
        fold_evidence=frozen_fold_evidence,
        fitting_identity=fitting_identity,
        residual_optimization={
            name: tuple(values) for name, values in residual_optimization.items()
        },
        decision_optimization={
            "oracle-halves": oracle_optimization,
            "finalist-refit": finalist_refit_optimization,
        },
        cells=cells,
        identity_cosine_r1=(identity_forward.micro_r1, identity_reverse.micro_r1),
        identity_csls_hits=(
            _r1_hits_from_scores(identity_csls_forward, development_label_values),
            _r1_hits_from_scores(identity_csls_reverse, development_label_values),
        ),
        finalist_cosine_r1=(
            cells["finalist-forward"].micro_r1,
            cells["finalist-reverse"].micro_r1,
        ),
        finalist_csls_hits=(
            _r1_hits_from_scores(csls_forward, development_label_values),
            _r1_hits_from_scores(csls_reverse, development_label_values),
        ),
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("capacity input must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Extract one descriptor bank and run the registered local diagnostic."""

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
    for output in (arguments.descriptor_artifact, arguments.result):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    if _file_sha256(arguments.spatial_artifact) != arguments.spatial_artifact_sha256:
        raise ValueError("capacity spatial artifact digest differs")
    binding_raw = read_rsta_regular(arguments.control_binding, role="capacity binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("capacity binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("capacity binding authority differs")
    seed17 = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(arguments.checkpoint_seed17, seed17, binding)
    example_ids, labels, paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.optimization_image_root,
    )
    if set(labels) != set(range(49)):
        raise ValueError("capacity class partition differs")
    fitting_labels, development_labels = spatial_tail_class_split(tuple(range(49)))
    _validate_capacity_anchor_coverage(
        tuple(labels), tuple(sorted(fitting_labels)), tuple(sorted(development_labels))
    )
    preprocessing = "siglip-evaluation-transform-v1"
    image_manifest_sha256 = capacity_image_manifest_sha256(tuple(example_ids), tuple(paths))
    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("compatibility capacity diagnostic requires CUDA bf16")
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

    def pixel_batches() -> Iterable[torch.Tensor]:
        for start in range(0, len(paths), 32):
            tensors: list[torch.Tensor] = []
            for path in paths[start : start + 32]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("capacity image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    student, teacher = stream_alignment_descriptor_pairs(
        vision,
        projection,
        tokenwise,
        readout,
        pixel_batches(),
        source_depth=18,
        device=device,
    )
    descriptor_sha = write_capacity_descriptor_artifact(
        arguments.descriptor_artifact,
        student,
        teacher,
        tuple(example_ids),
        tuple(labels),
        checkpoint_sha256=seed17.sha256,
        control_binding_sha256=arguments.control_binding_sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        spatial_artifact_sha256=arguments.spatial_artifact_sha256,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing=preprocessing,
    )
    artifact = load_capacity_descriptor_artifact(
        arguments.descriptor_artifact,
        ids=tuple(example_ids),
        labels=tuple(labels),
        checkpoint_sha256=seed17.sha256,
        control_binding_sha256=arguments.control_binding_sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        spatial_artifact_sha256=arguments.spatial_artifact_sha256,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing=preprocessing,
    )
    del model, vision, projection, tokenwise, readout, student, teacher
    torch.cuda.empty_cache()
    raw = analyze_capacity_descriptors(
        artifact,
        checkpoint_sha256=seed17.sha256,
        descriptor_artifact_sha256=descriptor_sha,
        control_binding_sha256=arguments.control_binding_sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        spatial_artifact_sha256=arguments.spatial_artifact_sha256,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing=preprocessing,
    )
    if not isinstance(json.loads(raw), Mapping):
        raise ValueError("capacity result differs")
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    print(
        f"compatibility-capacity:COMPLETE sha256={hashlib.sha256(raw).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
