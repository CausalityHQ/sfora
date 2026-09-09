#!/usr/bin/env python3
"""Diagnose train-only source, width, and linear representation ceilings."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import export_unicom_sop_embeddings as sop_export_module
import numpy as np
import probe_sop_relational_linear as sop_relational_module
import torch
import train_sop_nested_neighborhood_rank as nnrl_snapshot_module
from probe_sop_relational_linear import score_symmetric

import sfora.representation_ceiling as representation_ceiling_module
from sfora.representation_ceiling import (
    apply_normalized_affine,
    deterministic_class_partition,
    fit_centered_pca,
    fit_ridge_affine,
)

DIMENSIONS = 128
OUTER_SPLIT_SEEDS = (17, 1729, 65537)
RIDGE_PENALTIES = (1e-6, 1e-4, 1e-2)
BOOTSTRAP_SAMPLES = 10_000
RIDGE_GAIN_GATE = 0.005
WIDTH_NONBINDING_LOSS = -0.010
WIDTH_MATERIAL_LOSS = -0.030


class PairedTrainArchives(TypedDict):
    """Authenticated paired train members without materialized test members."""

    source_metadata: dict[str, object]
    source_train: torch.Tensor
    teacher_metadata: dict[str, object]
    teacher_train: torch.Tensor
    train_labels: tuple[int, ...]


@dataclass(frozen=True)
class ReservedOutput:
    """An anonymous staging inode plus an exclusive output-directory lock."""

    path: Path
    descriptor: int
    directory_descriptor: int
    device: int
    inode: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_train_archive_snapshot(
    path: Path,
    sha256: str,
    *,
    model_identifier: str,
    expected_rows: int = 59_551,
    expected_classes: int = 11_318,
    dimensions: int = 768,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        not isinstance(path, Path)
        or type(sha256) is not str
        or len(sha256) != 64
        or set(sha256) - set("0123456789abcdef")
        or type(model_identifier) is not str
        or not model_identifier
    ):
        raise ValueError("representation ceiling archive authority differs")
    snapshot = path.read_bytes()
    if _sha256_bytes(snapshot) != sha256:
        raise ValueError("representation ceiling archive digest differs")
    train_only_required = {
        "metadata_json",
        "train_embeddings",
        "train_labels",
        "train_image_ids",
        "train_relative_paths",
    }
    required = {
        "metadata_json",
        "train_embeddings",
        "train_labels",
        "train_image_ids",
        "train_relative_paths",
        "test_embeddings",
        "test_labels",
        "test_image_ids",
        "test_relative_paths",
    }
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
        members = set(archive.files)
        if members not in (required, train_only_required):
            raise ValueError("representation ceiling archive schema differs")
        if members == train_only_required:
            loaded = nnrl_snapshot_module.load_train_snapshot(path, sha256)
            metadata = loaded.metadata
            if (
                metadata.get("model_identifier") != model_identifier
                or metadata.get("embedding_dimension") != dimensions
                or metadata.get("train_rows") != expected_rows
                or metadata.get("train_classes") != expected_classes
            ):
                raise ValueError("representation ceiling archive metadata differs")
            return (
                metadata,
                loaded.embeddings,
                loaded.labels,
                loaded.image_ids,
                loaded.relative_paths,
            )
        metadata_value = archive["metadata_json"]
        metadata = json.loads(str(metadata_value.item()))
        train_embeddings = archive["train_embeddings"].copy()
        train_labels = archive["train_labels"].copy()
        train_image_ids = archive["train_image_ids"].copy()
        train_paths = archive["train_relative_paths"].copy()
    if (
        type(metadata) is not dict
        or set(metadata)
        != {
            "array_sha256",
            "checkpoint_sha256",
            "embedding_dimension",
            "model_identifier",
            "model_revision",
            "ordered_record_sha256",
            "schema",
            "split_classes",
            "split_counts",
            "transform",
        }
        or metadata.get("schema") != "sfora-unicom-sop-embeddings-v1"
        or metadata.get("model_identifier") != model_identifier
        or metadata.get("embedding_dimension") != dimensions
        or metadata.get("split_counts") != {"test": 60_502, "train": expected_rows}
        or metadata.get("split_classes") != {"test": 11_316, "train": expected_classes}
        or type(metadata.get("array_sha256")) is not dict
        or set(cast(dict[str, object], metadata["array_sha256"])) != required - {"metadata_json"}
    ):
        raise ValueError("representation ceiling archive metadata differs")
    train_arrays = {
        "train_embeddings": train_embeddings,
        "train_image_ids": train_image_ids,
        "train_labels": train_labels,
        "train_relative_paths": train_paths,
    }
    array_digests = cast(dict[str, object], metadata["array_sha256"])
    if any(
        _sha256_bytes(value.tobytes(order="C")) != array_digests.get(name)
        for name, value in train_arrays.items()
    ):
        raise ValueError("representation ceiling train array digest differs")
    if (
        train_embeddings.dtype != np.float32
        or train_embeddings.shape != (expected_rows, dimensions)
        or not train_embeddings.flags.c_contiguous
        or not np.isfinite(train_embeddings).all()
        or np.any(np.linalg.norm(train_embeddings.astype(np.float64), axis=1) == 0.0)
        or train_labels.dtype != np.int64
        or train_labels.shape != (expected_rows,)
        or len(set(train_labels.tolist())) != expected_classes
        or train_image_ids.dtype != np.int64
        or train_image_ids.shape != (expected_rows,)
        or len(set(train_image_ids.tolist())) != expected_rows
        or train_paths.dtype.kind != "U"
        or train_paths.shape != (expected_rows,)
        or len(set(train_paths.tolist())) != expected_rows
    ):
        raise ValueError("representation ceiling train array authority differs")
    return metadata, train_embeddings, train_labels, train_image_ids, train_paths


def load_paired_train_archives(
    source_path: Path,
    source_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
) -> PairedTrainArchives:
    """Authenticate immutable archive snapshots and materialize train members only."""

    source = _load_train_archive_snapshot(
        source_path, source_sha256, model_identifier="UNICOM-ViT-B/16"
    )
    teacher = _load_train_archive_snapshot(
        teacher_path, teacher_sha256, model_identifier="UNICOM-ViT-L/14@336px"
    )
    source_metadata, source_rows, source_labels, source_ids, source_paths = source
    teacher_metadata, teacher_rows, teacher_labels, teacher_ids, teacher_paths = teacher
    if (
        source_metadata.get(
            "ordered_train_record_sha256", source_metadata.get("ordered_record_sha256")
        )
        != teacher_metadata.get(
            "ordered_train_record_sha256", teacher_metadata.get("ordered_record_sha256")
        )
        or source_metadata.get("model_revision") != teacher_metadata.get("model_revision")
        or not np.array_equal(source_labels, teacher_labels)
        or not np.array_equal(source_ids, teacher_ids)
        or not np.array_equal(source_paths, teacher_paths)
    ):
        raise ValueError("representation ceiling paired train identity differs")
    return PairedTrainArchives(
        source_metadata=source_metadata,
        source_train=torch.from_numpy(source_rows).float().contiguous(),
        teacher_metadata=teacher_metadata,
        teacher_train=torch.from_numpy(teacher_rows).float().contiguous(),
        train_labels=tuple(int(value) for value in source_labels.tolist()),
    )


def _unit(value: torch.Tensor) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] < 2
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("representation ceiling tensor authority differs")
    norms = torch.linalg.vector_norm(value.double(), dim=1, keepdim=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError("representation ceiling tensor authority differs")
    return cast(torch.Tensor, (value.double() / norms).float().contiguous())


def class_cluster_lower_bound(
    treatment: tuple[float, ...],
    baseline: tuple[float, ...],
    identities: tuple[int, ...],
    *,
    seed: int,
    samples: int,
) -> float:
    """Return a deterministic paired one-sided 95% class-cluster lower bound."""

    if (
        type(treatment) is not tuple
        or type(baseline) is not tuple
        or type(identities) is not tuple
        or not treatment
        or len(treatment) != len(baseline)
        or len(treatment) != len(identities)
        or any(type(value) is not float or not math.isfinite(value) for value in treatment)
        or any(type(value) is not float or not math.isfinite(value) for value in baseline)
        or any(type(value) is not int for value in identities)
        or type(seed) is not int
        or type(samples) is not int
        or samples < 2
    ):
        raise ValueError("representation ceiling bootstrap authority differs")
    differences = torch.tensor(treatment, dtype=torch.float64) - torch.tensor(
        baseline, dtype=torch.float64
    )
    identity_tensor = torch.tensor(identities, dtype=torch.int64)
    groups = torch.unique(identity_tensor, sorted=True)
    sums = torch.stack([differences[identity_tensor == group].sum() for group in groups])
    counts = torch.tensor(
        [int((identity_tensor == group).sum()) for group in groups], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    blocks: list[torch.Tensor] = []
    for start in range(0, samples, 128):
        block = min(128, samples - start)
        draws = torch.randint(len(groups), (block, len(groups)), generator=generator)
        blocks.append(sums[draws].sum(dim=1) / counts[draws].sum(dim=1))
    values = torch.cat(blocks)
    return float(torch.quantile(values, 0.05, interpolation="lower"))


def select_ridge_penalty(
    source: torch.Tensor,
    teacher: torch.Tensor,
    labels: tuple[int, ...],
    *,
    penalties: tuple[float, ...],
    seed: int,
) -> dict[str, object]:
    """Choose one relative ridge penalty on a class-disjoint inner split."""

    if (
        type(penalties) is not tuple
        or penalties != tuple(sorted(set(penalties)))
        or any(
            type(value) is not float or not math.isfinite(value) or value <= 0
            for value in penalties
        )
        or type(source) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or source.shape[0] != teacher.shape[0]
        or len(labels) != source.shape[0]
    ):
        raise ValueError("representation ceiling ridge selection authority differs")
    source_unit = _unit(source)
    teacher_unit = _unit(teacher)
    partition = deterministic_class_partition(labels, fit_fraction=0.8, seed=seed)
    fit_indexes = list(partition.fit_row_indexes)
    validation_indexes = list(partition.validation_row_indexes)
    losses: dict[str, float] = {}
    for penalty in penalties:
        fitted = fit_ridge_affine(
            source_unit[fit_indexes].contiguous(),
            teacher_unit[fit_indexes].contiguous(),
            penalty=penalty,
        )
        predicted = apply_normalized_affine(source_unit[validation_indexes].contiguous(), fitted)
        loss = float(
            torch.mean(torch.square(predicted.double() - teacher_unit[validation_indexes].double()))
        )
        if not math.isfinite(loss):
            raise ValueError("representation ceiling ridge selection authority differs")
        losses[str(penalty)] = loss
    selected = min(penalties, key=lambda value: (losses[str(value)], value))
    return {
        "fit_class_ids": list(partition.fit_class_ids),
        "losses": losses,
        "penalty": selected,
        "seed": seed,
        "validation_class_ids": list(partition.validation_class_ids),
    }


def _score(value: torch.Tensor, labels: tuple[int, ...]) -> dict[str, object]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    width = max(counts.values()) - 1
    scored = score_symmetric(
        value,
        labels,
        candidate_width=width,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return {
        "map_at_r": scored["map_at_r"],
        "per_query_ap": list(scored["per_query_ap"]),
        "per_query_r1": list(scored["per_query_r1"]),
        "r1": scored["r1"],
    }


def _parameter_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("representation ceiling parameter authority differs")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
        array = value.detach().numpy()
        digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def ceiling_decision(
    *,
    ridge_full_gain: float,
    ridge_full_lower_bound: float,
    teacher_pca_loss: float,
    teacher_pca_lower_bound: float,
) -> dict[str, bool | str]:
    """Apply the frozen diagnostic gates without selecting a deployment method."""

    if any(
        type(value) is not float or not math.isfinite(value)
        for value in (
            ridge_full_gain,
            ridge_full_lower_bound,
            teacher_pca_loss,
            teacher_pca_lower_bound,
        )
    ):
        raise ValueError("representation ceiling decision authority differs")
    sampling = ridge_full_gain >= RIDGE_GAIN_GATE and ridge_full_lower_bound > 0.0
    if teacher_pca_loss < WIDTH_MATERIAL_LOSS:
        width_outcome = "material"
    elif (
        teacher_pca_loss >= WIDTH_NONBINDING_LOSS
        and teacher_pca_lower_bound >= WIDTH_NONBINDING_LOSS
    ):
        width_outcome = "nonbinding"
    else:
        width_outcome = "intermediate"
    return {
        "backbone_quality_work_warranted": not sampling,
        "neighborhood_sampling_warranted": sampling,
        "wider_code_warranted": teacher_pca_loss < WIDTH_MATERIAL_LOSS,
        "width_outcome": width_outcome,
    }


def run_representation_ceiling_train_only(
    source_train: torch.Tensor,
    teacher_train: torch.Tensor,
    train_labels: tuple[int, ...],
    *,
    dimensions: int = DIMENSIONS,
    outer_split_seeds: tuple[int, ...] = OUTER_SPLIT_SEEDS,
    ridge_penalties: tuple[float, ...] = RIDGE_PENALTIES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    """Evaluate the frozen six-arm ceiling using training rows only."""

    source_unit = _unit(source_train)
    teacher_unit = _unit(teacher_train)
    if (
        source_unit.shape[0] != teacher_unit.shape[0]
        or len(train_labels) != source_unit.shape[0]
        or any(type(label) is not int or label < 1 for label in train_labels)
        or type(dimensions) is not int
        or dimensions < 2
        or dimensions >= min(source_unit.shape[1], teacher_unit.shape[1])
        or type(outer_split_seeds) is not tuple
        or len(outer_split_seeds) < 1
        or len(set(outer_split_seeds)) != len(outer_split_seeds)
        or any(type(seed) is not int for seed in outer_split_seeds)
        or type(bootstrap_samples) is not int
        or bootstrap_samples < 2
    ):
        raise ValueError("representation ceiling protocol authority differs")
    splits: list[dict[str, object]] = []
    ridge_treatment: list[float] = []
    source_baseline: list[float] = []
    teacher_pca_treatment: list[float] = []
    teacher_baseline: list[float] = []
    cluster_identities: list[int] = []

    for seed in outer_split_seeds:
        partition = deterministic_class_partition(train_labels, fit_fraction=0.8, seed=seed)
        fit_indexes = list(partition.fit_row_indexes)
        validation_indexes = list(partition.validation_row_indexes)
        fit_labels = tuple(train_labels[index] for index in fit_indexes)
        validation_labels = tuple(train_labels[index] for index in validation_indexes)
        source_fit = source_unit[fit_indexes].contiguous()
        teacher_fit = teacher_unit[fit_indexes].contiguous()
        source_validation = source_unit[validation_indexes].contiguous()
        teacher_validation = teacher_unit[validation_indexes].contiguous()

        selection = select_ridge_penalty(
            source_fit,
            teacher_fit,
            fit_labels,
            penalties=ridge_penalties,
            seed=seed ^ 0x5F3759DF,
        )
        selected_penalty = cast(float, selection["penalty"])
        ridge = fit_ridge_affine(source_fit, teacher_fit, penalty=selected_penalty)
        ridge_fit = apply_normalized_affine(source_fit, ridge)
        ridge_validation = apply_normalized_affine(source_validation, ridge)
        source_pca = fit_centered_pca(source_fit, dimensions=dimensions)
        teacher_pca = fit_centered_pca(teacher_fit, dimensions=dimensions)
        ridge_pca = fit_centered_pca(ridge_fit, dimensions=dimensions)
        arms = {
            "source-full": _score(source_validation, validation_labels),
            "teacher-full": _score(teacher_validation, validation_labels),
            "teacher-pca128": _score(teacher_pca.apply(teacher_validation), validation_labels),
            "ridge-source-teacher-full": _score(ridge_validation, validation_labels),
            "ridge-source-teacher-pca128": _score(
                ridge_pca.apply(ridge_validation), validation_labels
            ),
            "source-pca128": _score(source_pca.apply(source_validation), validation_labels),
        }
        splits.append(
            {
                "arms": arms,
                "fit_class_ids": list(partition.fit_class_ids),
                "fit_row_indexes": list(partition.fit_row_indexes),
                "inner_ridge_selection": selection,
                "selected_ridge_penalty": selected_penalty,
                "seed": seed,
                "transform_sha256": {
                    "ridge": _parameter_sha256(ridge.weight, ridge.bias),
                    "ridge-pca": _parameter_sha256(ridge_pca.mean, ridge_pca.components),
                    "source-pca": _parameter_sha256(source_pca.mean, source_pca.components),
                    "teacher-pca": _parameter_sha256(teacher_pca.mean, teacher_pca.components),
                },
                "validation_class_ids": list(partition.validation_class_ids),
                "validation_row_indexes": list(partition.validation_row_indexes),
            }
        )
        ridge_ap = cast(list[float], arms["ridge-source-teacher-full"]["per_query_ap"])
        source_ap = cast(list[float], arms["source-full"]["per_query_ap"])
        teacher_pca_ap = cast(list[float], arms["teacher-pca128"]["per_query_ap"])
        teacher_ap = cast(list[float], arms["teacher-full"]["per_query_ap"])
        ridge_treatment.extend(ridge_ap)
        source_baseline.extend(source_ap)
        teacher_pca_treatment.extend(teacher_pca_ap)
        teacher_baseline.extend(teacher_ap)
        cluster_identities.extend(validation_labels)

    ridge_gain = math.fsum(ridge_treatment) / len(ridge_treatment) - math.fsum(
        source_baseline
    ) / len(source_baseline)
    teacher_pca_loss = math.fsum(teacher_pca_treatment) / len(teacher_pca_treatment) - math.fsum(
        teacher_baseline
    ) / len(teacher_baseline)
    ridge_lower = class_cluster_lower_bound(
        tuple(ridge_treatment),
        tuple(source_baseline),
        tuple(cluster_identities),
        seed=17,
        samples=bootstrap_samples,
    )
    teacher_pca_lower = class_cluster_lower_bound(
        tuple(teacher_pca_treatment),
        tuple(teacher_baseline),
        tuple(cluster_identities),
        seed=17,
        samples=bootstrap_samples,
    )
    return {
        "bootstrap": {
            "cluster_identity": "original-class-id-across-splits",
            "cluster_ids": sorted(set(cluster_identities)),
            "ridge_seed": 17,
            "width_seed": 17,
        },
        "bootstrap_samples": bootstrap_samples,
        "claim_eligible": False,
        "decisions": ceiling_decision(
            ridge_full_gain=ridge_gain,
            ridge_full_lower_bound=ridge_lower,
            teacher_pca_loss=teacher_pca_loss,
            teacher_pca_lower_bound=teacher_pca_lower,
        ),
        "dimensions": dimensions,
        "gates": {
            "ridge_full_gain": RIDGE_GAIN_GATE,
            "ridge_full_lower_bound": 0.0,
            "teacher_pca_material_width_loss": WIDTH_MATERIAL_LOSS,
            "teacher_pca_nonbinding_width_loss": WIDTH_NONBINDING_LOSS,
        },
        "outer_split_seeds": list(outer_split_seeds),
        "ridge_full_gain": ridge_gain,
        "ridge_full_lower_bound": ridge_lower,
        "ridge_penalties": list(ridge_penalties),
        "schema": "sfora-representation-ceiling-v1",
        "splits": splits,
        "teacher_pca_loss": teacher_pca_loss,
        "teacher_pca_lower_bound": teacher_pca_lower,
        "train_rows": len(source_train),
    }


def _canonical_bytes(receipt: dict[str, object]) -> bytes:
    if (
        type(receipt) is not dict
        or receipt.get("schema")
        not in {
            "sfora-representation-ceiling-v1",
            "sfora-representation-ceiling-failure-v1",
        }
        or receipt.get("claim_eligible") is not False
    ):
        raise ValueError("representation ceiling receipt authority differs")
    return (
        json.dumps(
            receipt, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode()


def _reserve_output(output: Path) -> ReservedOutput:
    if not isinstance(output, Path):
        raise ValueError("representation ceiling output authority differs")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError("representation ceiling output exists")
    directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
            raise FileExistsError("representation ceiling output exists")
        descriptor = os.open(output.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        info = os.fstat(descriptor)
        return ReservedOutput(
            path=partial,
            descriptor=descriptor,
            directory_descriptor=directory_descriptor,
            device=info.st_dev,
            inode=info.st_ino,
        )
    except Exception:
        os.close(directory_descriptor)
        raise


def _reserved_path_is_owned(reserved: ReservedOutput) -> bool:
    try:
        info = os.fstat(reserved.descriptor)
    except OSError:
        return False
    return info.st_dev == reserved.device and info.st_ino == reserved.inode


def _cleanup_reserved_output(reserved: ReservedOutput) -> None:
    try:
        os.close(reserved.descriptor)
    finally:
        os.close(reserved.directory_descriptor)


def _link_descriptor_no_clobber(descriptor: int, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    if linkat(descriptor, b"", -100, os.fsencode(output), 0x1000) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError("representation ceiling output exists")
        raise OSError(error_number, os.strerror(error_number), output)


def _publish_reserved_receipt(
    receipt: dict[str, object], output: Path, reserved: ReservedOutput
) -> None:
    if reserved.path != output.with_name(output.name + ".partial"):
        raise ValueError("representation ceiling output authority differs")
    data = _canonical_bytes(receipt)
    os.ftruncate(reserved.descriptor, 0)
    os.lseek(reserved.descriptor, 0, os.SEEK_SET)
    written = 0
    while written < len(data):
        written += os.write(reserved.descriptor, data[written:])
    os.fsync(reserved.descriptor)
    if not _reserved_path_is_owned(reserved):
        raise FileExistsError("representation ceiling output ownership differs")
    _link_descriptor_no_clobber(reserved.descriptor, output)
    os.fsync(reserved.directory_descriptor)


def publish_receipt(receipt: dict[str, object], output: Path) -> None:
    """Publish one canonical receipt without clobbering final or partial paths."""

    reserved = _reserve_output(output)
    try:
        _publish_reserved_receipt(receipt, output, reserved)
    finally:
        _cleanup_reserved_output(reserved)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _digest(value: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def _commit(value: str) -> str:
    if len(value) != 40 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("commit must be lowercase hexadecimal")
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only train ceiling surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--source-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--teacher-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--teacher-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-representation-ceiling", action="store_true", required=True)
    return parser.parse_args(arguments)


def verify_source_commit(source_commit: str) -> dict[str, str]:
    """Require every loaded repository Python module to match one commit."""

    root = Path(__file__).resolve().parents[1]
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if type(module_file) is not str:
            continue
        path = Path(module_file).resolve()
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            continue
        if relative_path.parts[:2] == ("src", "sfora") or relative_path.parts[:1] == ("scripts",):
            paths.add(path)
    required = {
        Path(__file__).resolve(),
        Path(representation_ceiling_module.__file__).resolve(),
        Path(sop_export_module.__file__).resolve(),
        Path(sop_relational_module.__file__).resolve(),
    }
    if not required <= paths:
        raise ValueError("representation ceiling source closure differs")
    identities: dict[str, str] = {}
    for path in sorted(paths):
        relative_text = path.relative_to(root).as_posix()
        try:
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"{source_commit}:{relative_text}"],
                check=True,
                capture_output=True,
            ).stdout
            current = path.read_bytes()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("representation ceiling source closure differs") from error
        if committed != current:
            raise ValueError("representation ceiling source differs from registered commit")
        identities[relative_text] = _sha256_bytes(current)
    return identities


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate paired train archives, diagnose once, and publish evidence."""

    args = parse_args(arguments)
    reserved = _reserve_output(args.output)
    stage = "source-authentication"
    source_files_sha256 = {
        "scripts/probe_representation_ceiling.py": _sha256_bytes(Path(__file__).read_bytes())
    }
    try:
        source_files_sha256 = verify_source_commit(args.source_commit)
        stage = "archive-authentication"
        pair = load_paired_train_archives(
            args.source_embeddings,
            args.source_embeddings_sha256,
            args.teacher_embeddings,
            args.teacher_embeddings_sha256,
        )
        stage = "evaluation"
        receipt = run_representation_ceiling_train_only(
            pair["source_train"], pair["teacher_train"], pair["train_labels"]
        )
        receipt.update(
            {
                "environment": {
                    "cuda_device": (
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    ),
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "torch": torch.__version__,
                },
                "source_commit": args.source_commit,
                "source_embeddings_sha256": args.source_embeddings_sha256,
                "source_files_sha256": source_files_sha256,
                "source_metadata": pair["source_metadata"],
                "source_sha256": source_files_sha256["scripts/probe_representation_ceiling.py"],
                "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
                "teacher_metadata": pair["teacher_metadata"],
            }
        )
        stage = "publication"
        _publish_reserved_receipt(receipt, args.output, reserved)
    except Exception as error:
        if (
            stage != "publication"
            and not args.output.exists()
            and _reserved_path_is_owned(reserved)
        ):
            failure: dict[str, object] = {
                "claim_eligible": False,
                "failure": {
                    "message": str(error),
                    "stage": stage,
                    "type": type(error).__name__,
                },
                "schema": "sfora-representation-ceiling-failure-v1",
                "source_commit": args.source_commit,
                "source_embeddings_sha256": args.source_embeddings_sha256,
                "source_files_sha256": source_files_sha256,
                "source_sha256": source_files_sha256["scripts/probe_representation_ceiling.py"],
                "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
            }
            _publish_reserved_receipt(failure, args.output, reserved)
        raise
    finally:
        _cleanup_reserved_output(reserved)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
