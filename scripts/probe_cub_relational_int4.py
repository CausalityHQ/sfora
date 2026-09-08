#!/usr/bin/env python3
"""Evaluate equal-byte relational int4 compaction on CUB-200-2011."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import export_unicom_cub_embeddings as cub_export_module
import export_unicom_sop_embeddings as sop_export_module
import numpy as np
import probe_inshop_relational_linear as inshop_probe_module
import probe_sop_relational_linear as sop_probe_module
import torch
from export_unicom_cub_embeddings import load_cub_embedding_archive
from probe_inshop_relational_linear import (
    _encode_floating,
    _familywise_lower_bound,
    _profile_paired_latency,
    _summary,
)
from probe_sop_relational_linear import _lexicographic_candidates, score_symmetric
from torch.nn import functional as F

import sfora.joint_relational_compaction as relational_compaction_module
import sfora.packed_int4 as packed_int4_module
from sfora.joint_relational_compaction import (
    RelationalLinearEncoder,
    RelationalLinearTrainingConfig,
    _fit_uncentered_covariance_basis,
    fit_relational_linear_encoder,
    pack_int8_unit_embeddings,
)
from sfora.packed_int4 import PackedInt4Embeddings, pack_int4_unit_embeddings

RELATIONAL_DIMENSIONS = 128
RIDGE_DIMENSIONS = 128
PCA_INT8_DIMENSIONS = 64
CANDIDATE_WIDTH = 128
RIDGE_RELATIVE_REGULARIZATION = 1e-3
SEEDS = (17, 1729, 65537)
DEPLOYMENT_SEED = 17
BATCH = 1024
EPOCHS = 200
TEMPERATURE = 0.05
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_COMPARISONS = len(SEEDS) * 3 * 2
MAP_GAIN_GATE = 0.01
R1_LOWER_BOUND_GATE = -0.005
LATENCY_WARMUP_PAIRS = 1_000
LATENCY_MEASURED_PAIRS = 10_000
LATENCY_RATIO_GATE = 1.10
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
OFFICIAL_CHECKPOINT_SHA256 = (
    "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
)


class SymmetricScore(TypedDict):
    """Per-query and aggregate leave-self-out retrieval quality."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


class PairedCubArchives(TypedDict):
    """Authenticated aligned source and teacher embedding tensors."""

    source_metadata: dict[str, object]
    teacher_metadata: dict[str, object]
    source_archive_sha256: str
    teacher_archive_sha256: str
    source_train: torch.Tensor
    teacher_train: torch.Tensor
    source_test: torch.Tensor
    teacher_test: torch.Tensor
    train_labels: tuple[int, ...]
    test_labels: tuple[int, ...]


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
    """Parse the strict local-only evaluation surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--source-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--teacher-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--teacher-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--model-output", required=True, type=_absolute_path)
    parser.add_argument("--latency-output", required=True, type=_absolute_path)
    parser.add_argument("--execute-relational-int4", action="store_true", required=True)
    return parser.parse_args(arguments)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paired_cub_archives(
    source_path: Path,
    source_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
    *,
    expected_counts: tuple[int, int] = (5_864, 5_924),
    expected_classes: tuple[int, int] = (100, 100),
    expected_dimension: int = 768,
    expected_identifiers: tuple[str, str] = (
        "UNICOM-ViT-B/16",
        "UNICOM-ViT-L/14@336px",
    ),
    expected_revision: str = UNICOM_REVISION,
    expected_checkpoint_sha256: tuple[str, str] = OFFICIAL_CHECKPOINT_SHA256,
) -> PairedCubArchives:
    """Authenticate both archives and bind every row identity before use."""

    if (
        type(source_sha256) is not str
        or type(teacher_sha256) is not str
        or len(source_sha256) != 64
        or len(teacher_sha256) != 64
        or set(source_sha256 + teacher_sha256) - set("0123456789abcdef")
        or _sha256(source_path) != source_sha256
        or _sha256(teacher_path) != teacher_sha256
    ):
        raise ValueError("CUB archive digest differs")
    source = load_cub_embedding_archive(
        source_path,
        expected_counts=expected_counts,
        expected_classes=expected_classes,
        expected_dimension=expected_dimension,
    )
    teacher = load_cub_embedding_archive(
        teacher_path,
        expected_counts=expected_counts,
        expected_classes=expected_classes,
        expected_dimension=expected_dimension,
    )
    source_metadata = source["metadata"]
    teacher_metadata = teacher["metadata"]
    if (
        type(source_metadata) is not dict
        or type(teacher_metadata) is not dict
        or source_metadata.get("model_identifier") != expected_identifiers[0]
        or teacher_metadata.get("model_identifier") != expected_identifiers[1]
        or source_metadata.get("model_revision") != teacher_metadata.get("model_revision")
        or source_metadata.get("model_revision") != expected_revision
        or source_metadata.get("checkpoint_sha256") != expected_checkpoint_sha256[0]
        or teacher_metadata.get("checkpoint_sha256") != expected_checkpoint_sha256[1]
        or source_metadata.get("cub_exporter_source_sha256")
        != _sha256(Path(cub_export_module.__file__))
        or teacher_metadata.get("cub_exporter_source_sha256")
        != _sha256(Path(cub_export_module.__file__))
        or source_metadata.get("sop_exporter_source_sha256")
        != _sha256(Path(sop_export_module.__file__))
        or teacher_metadata.get("sop_exporter_source_sha256")
        != _sha256(Path(sop_export_module.__file__))
        or source_metadata.get("ordered_record_sha256")
        != teacher_metadata.get("ordered_record_sha256")
        or source_metadata.get("dataset_archive_sha256")
        != teacher_metadata.get("dataset_archive_sha256")
        or source_metadata.get("dataset_content_sha256")
        != teacher_metadata.get("dataset_content_sha256")
    ):
        raise ValueError("CUB paired metadata differs")
    for name in (
        "train_labels",
        "test_labels",
        "train_image_ids",
        "test_image_ids",
        "train_relative_paths",
        "test_relative_paths",
        "train_class_names",
        "test_class_names",
    ):
        left = cast(np.ndarray, source[name])
        right = cast(np.ndarray, teacher[name])
        if left.shape != right.shape or not bool(np.array_equal(left, right)):
            raise ValueError("CUB paired row identity differs")
    return PairedCubArchives(
        source_metadata=source_metadata,
        teacher_metadata=teacher_metadata,
        source_archive_sha256=source_sha256,
        teacher_archive_sha256=teacher_sha256,
        source_train=torch.from_numpy(cast(np.ndarray, source["train_embeddings"]).copy()),
        teacher_train=torch.from_numpy(cast(np.ndarray, teacher["train_embeddings"]).copy()),
        source_test=torch.from_numpy(cast(np.ndarray, source["test_embeddings"]).copy()),
        teacher_test=torch.from_numpy(cast(np.ndarray, teacher["test_embeddings"]).copy()),
        train_labels=tuple(int(x) for x in cast(np.ndarray, source["train_labels"]).tolist()),
        test_labels=tuple(int(x) for x in cast(np.ndarray, source["test_labels"]).tolist()),
    )


def cub_evidence_authority(pair: PairedCubArchives, *, source_commit: str) -> dict[str, object]:
    """Bind a result to its immutable inputs and exact implementation sources."""

    source_metadata = pair["source_metadata"]
    teacher_metadata = pair["teacher_metadata"]
    return {
        "dataset_archive_sha256": source_metadata["dataset_archive_sha256"],
        "dataset_content_sha256": source_metadata["dataset_content_sha256"],
        "evaluator_source_sha256": _sha256(Path(__file__)),
        "exporter_source_sha256": _sha256(Path(cub_export_module.__file__)),
        "archive_cub_exporter_source_sha256": source_metadata["cub_exporter_source_sha256"],
        "archive_sop_exporter_source_sha256": source_metadata["sop_exporter_source_sha256"],
        "inshop_probe_source_sha256": _sha256(Path(inshop_probe_module.__file__)),
        "ordered_record_sha256": source_metadata["ordered_record_sha256"],
        "packed_int4_source_sha256": _sha256(Path(packed_int4_module.__file__)),
        "relational_compaction_source_sha256": _sha256(Path(relational_compaction_module.__file__)),
        "source_checkpoint_sha256": source_metadata["checkpoint_sha256"],
        "source_commit": source_commit,
        "source_embeddings_sha256": pair["source_archive_sha256"],
        "source_model_identifier": source_metadata["model_identifier"],
        "source_model_revision": source_metadata["model_revision"],
        "teacher_checkpoint_sha256": teacher_metadata["checkpoint_sha256"],
        "teacher_embeddings_sha256": pair["teacher_archive_sha256"],
        "teacher_model_identifier": teacher_metadata["model_identifier"],
        "teacher_model_revision": teacher_metadata["model_revision"],
        "sop_probe_source_sha256": _sha256(Path(sop_probe_module.__file__)),
    }


def verify_source_commit(source_commit: str) -> None:
    """Require every executing scientific source to match the registered commit."""

    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        Path(cub_export_module.__file__).resolve(),
        Path(sop_export_module.__file__).resolve(),
        Path(sop_probe_module.__file__).resolve(),
        Path(inshop_probe_module.__file__).resolve(),
        Path(relational_compaction_module.__file__).resolve(),
        Path(packed_int4_module.__file__).resolve(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{source_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise ValueError("CUB executing source differs from registered commit")


def score_int4_symmetric(
    embeddings: PackedInt4Embeddings,
    labels: tuple[int, ...],
    *,
    candidate_width: int,
    device: torch.device,
) -> SymmetricScore:
    """Score exact packed-int4 cosine with deterministic leave-self-out ties."""

    if (
        type(embeddings) is not PackedInt4Embeddings
        or type(labels) is not tuple
        or len(labels) != embeddings.packed_codes.shape[0]
        or any(type(label) is not int or label < 1 for label in labels)
        or type(candidate_width) is not int
        or not 1 <= candidate_width < len(labels)
        or type(device) is not torch.device
    ):
        raise ValueError("CUB int4 scoring authority differs")
    counts = Counter(labels)
    if any(count < 2 for count in counts.values()) or max(counts.values()) - 1 > candidate_width:
        raise ValueError("CUB int4 positive authority differs")
    gallery_codes = embeddings.signed_codes().to(device=device, dtype=torch.float32)
    gallery_inverse = embeddings.inverse_norms.to(device=device, dtype=torch.float32)
    rankings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(labels), 256):
            stop = min(start + 256, len(labels))
            scores = (
                (gallery_codes[start:stop] @ gallery_codes.T)
                * gallery_inverse[start:stop].unsqueeze(1)
                * gallery_inverse.unsqueeze(0)
            )
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, candidate_width).cpu())
    aps: list[float] = []
    hits: list[float] = []
    for ranking, label in zip(torch.cat(rankings).tolist(), labels, strict=True):
        positives = counts[label] - 1
        found = 0
        terms = []
        for rank, index in enumerate(ranking[:positives], start=1):
            if labels[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positives)
        hits.append(float(labels[ranking[0]] == label))
    return SymmetricScore(
        map_at_r=math.fsum(aps) / len(labels),
        per_query_ap=tuple(aps),
        per_query_r1=tuple(hits),
        r1=math.fsum(hits) / len(labels),
    )


def fit_ridge_teacher_control(
    source_train: torch.Tensor,
    teacher_train: torch.Tensor,
    source_test: torch.Tensor,
    *,
    dimensions: int = RIDGE_DIMENSIONS,
) -> torch.Tensor:
    """Fit a closed-form train-only ridge map to compact teacher PCA coordinates."""

    if (
        type(source_train) is not torch.Tensor
        or type(teacher_train) is not torch.Tensor
        or type(source_test) is not torch.Tensor
        or source_train.device.type != "cpu"
        or teacher_train.device.type != "cpu"
        or source_test.device.type != "cpu"
        or source_train.dtype != torch.float32
        or teacher_train.dtype != torch.float32
        or source_test.dtype != torch.float32
        or source_train.ndim != 2
        or teacher_train.ndim != 2
        or source_test.ndim != 2
        or source_train.shape[0] != teacher_train.shape[0]
        or source_train.shape[1] != source_test.shape[1]
        or not 1 < dimensions < min(teacher_train.shape)
        or not all(
            bool(torch.isfinite(x).all()) for x in (source_train, teacher_train, source_test)
        )
    ):
        raise ValueError("CUB ridge authority differs")
    teacher_basis = _fit_uncentered_covariance_basis(teacher_train, dimensions=dimensions)
    targets = F.normalize(teacher_train.double() @ teacher_basis.double().T, dim=1)
    source = source_train.double()
    covariance = source.T @ source
    regularization = (
        RIDGE_RELATIVE_REGULARIZATION * float(torch.trace(covariance)) / source.shape[1]
    )
    weight = torch.linalg.solve(
        covariance + torch.eye(source.shape[1], dtype=torch.float64) * regularization,
        source.T @ targets,
    )
    result = F.normalize(source_test.double() @ weight, dim=1).float().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("CUB ridge geometry differs")
    return result


def fixed_random_rotation(dimensions: int, *, seed: int) -> torch.Tensor:
    """Construct one source-independent Haar-like orthogonal rotation."""

    if type(dimensions) is not int or dimensions < 2 or type(seed) is not int or seed < 0:
        raise ValueError("CUB rotation authority differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((dimensions, dimensions), generator=generator, dtype=torch.float64)
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.where(torch.diagonal(triangular) < 0.0, -1.0, 1.0)
    result = (orthogonal * signs.unsqueeze(0)).float().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("CUB rotation geometry differs")
    return cast(torch.Tensor, result)


def _model_sha256(model: RelationalLinearEncoder) -> str:
    return hashlib.sha256(model.to_bytes()).hexdigest()


def retrieve_int4_candidates(
    model: RelationalLinearEncoder,
    gallery: PackedInt4Embeddings,
    source_row: torch.Tensor,
    *,
    exclude_index: int,
    candidate_width: int,
) -> torch.Tensor:
    """Run the exact single-query packed path used by quality and latency."""

    if (
        type(model) is not RelationalLinearEncoder
        or type(gallery) is not PackedInt4Embeddings
        or type(source_row) is not torch.Tensor
        or source_row.device.type != "cpu"
        or source_row.dtype != torch.float32
        or source_row.shape != (1, model.projection.in_features)
        or type(exclude_index) is not int
        or not 0 <= exclude_index < gallery.packed_codes.shape[0]
        or type(candidate_width) is not int
        or not 1 <= candidate_width < gallery.packed_codes.shape[0]
    ):
        raise ValueError("CUB deployed retrieval authority differs")
    encoded = model(F.normalize(source_row, dim=1))
    query = pack_int4_unit_embeddings(encoded)
    scores = query.cosine_similarity(gallery, device=torch.device("cpu"))
    scores[0, exclude_index] = -torch.inf
    return _lexicographic_candidates(scores, candidate_width)


def run_sealed_cub_evaluation(
    pair: PairedCubArchives,
    *,
    source_commit: str,
) -> tuple[bytes, bytes, bytes]:
    """Run the preregistered equal-byte controls and three relational seeds once."""

    source_train = pair["source_train"]
    teacher_train = pair["teacher_train"]
    source_test = pair["source_test"]
    teacher_test = pair["teacher_test"]
    test_labels = pair["test_labels"]
    basis = _fit_uncentered_covariance_basis(source_train, dimensions=RELATIONAL_DIMENSIONS)
    pca = RelationalLinearEncoder(basis)
    pca_int8 = RelationalLinearEncoder(basis[:PCA_INT8_DIMENSIONS].contiguous())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding_device = torch.device("cpu")
    pca_float = _encode_floating(pca, source_test, device=encoding_device)
    rotation = fixed_random_rotation(RELATIONAL_DIMENSIONS, seed=17)
    rotated_pca = RelationalLinearEncoder((rotation @ basis).contiguous())
    rotated_float = _encode_floating(rotated_pca, source_test, device=encoding_device)
    pca_packed = pack_int4_unit_embeddings(pca_float)
    pca_int8_float = _encode_floating(pca_int8, source_test, device=encoding_device)
    pca_int8_packed = pack_int8_unit_embeddings(pca_int8_float)
    rotated_packed = pack_int4_unit_embeddings(rotated_float)
    ridge_float = fit_ridge_teacher_control(
        source_train,
        teacher_train,
        source_test,
        dimensions=RIDGE_DIMENSIONS,
    )
    ridge_packed = pack_int4_unit_embeddings(ridge_float)
    controls = {
        "pca128_int4": score_int4_symmetric(
            pca_packed, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "pca128_random_rotation_int4": score_int4_symmetric(
            rotated_packed,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "pca64_int8": score_symmetric(
            pca_int8_packed,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "pca128_float": score_symmetric(
            pca_float,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "ridge_teacher128_float": score_symmetric(
            ridge_float,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "ridge_teacher128_int4": score_int4_symmetric(
            ridge_packed,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "source_full_float": score_symmetric(
            F.normalize(source_test, dim=1),
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
        "teacher_full_float": score_symmetric(
            F.normalize(teacher_test, dim=1),
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        ),
    }
    comparison_names = (
        "pca128_int4",
        "pca128_random_rotation_int4",
        "ridge_teacher128_int4",
    )
    clusters = torch.tensor(test_labels, dtype=torch.int64)
    arms: dict[str, object] = {}
    arm_per_query: dict[str, object] = {}
    quality_passes = True
    deployment_model: RelationalLinearEncoder | None = None
    deployment_gallery: PackedInt4Embeddings | None = None
    for seed in SEEDS:
        model, losses = fit_relational_linear_encoder(
            source_train,
            teacher_train,
            basis,
            config=RelationalLinearTrainingConfig(
                batch_size=BATCH,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                seed=seed,
                temperature=TEMPERATURE,
                weight_decay=WEIGHT_DECAY,
            ),
            device=device,
        )
        floating = _encode_floating(model, source_test, device=encoding_device)
        packed = pack_int4_unit_embeddings(floating)
        score = score_int4_symmetric(
            packed,
            test_labels,
            candidate_width=CANDIDATE_WIDTH,
            device=device,
        )
        comparisons: dict[str, object] = {}
        passes = True
        for name in comparison_names:
            control = controls[name]
            map_gain = score["map_at_r"] - control["map_at_r"]
            map_lower = _familywise_lower_bound(
                score["per_query_ap"],
                control["per_query_ap"],
                clusters,
                seed=seed,
                samples=BOOTSTRAP_SAMPLES,
                comparisons=BOOTSTRAP_COMPARISONS,
            )
            r1_lower = _familywise_lower_bound(
                score["per_query_r1"],
                control["per_query_r1"],
                clusters,
                seed=seed,
                samples=BOOTSTRAP_SAMPLES,
                comparisons=BOOTSTRAP_COMPARISONS,
            )
            comparison_passes = (
                map_gain >= MAP_GAIN_GATE and map_lower > 0.0 and r1_lower > R1_LOWER_BOUND_GATE
            )
            passes = passes and comparison_passes
            comparisons[name] = {
                "map_gain": map_gain,
                "map_lower_bound": map_lower,
                "passes": comparison_passes,
                "r1_lower_bound": r1_lower,
            }
        quality_passes = quality_passes and passes
        arms[str(seed)] = {
            "comparisons": comparisons,
            "final_training_loss": losses[-1],
            "int4": _summary(score),
            "float": _summary(
                score_symmetric(
                    floating,
                    test_labels,
                    candidate_width=CANDIDATE_WIDTH,
                    device=device,
                )
            ),
            "model_sha256": _model_sha256(model),
            "passes_quality": passes,
            "training_losses": losses,
        }
        arm_per_query[str(seed)] = {
            "ap": score["per_query_ap"],
            "r1": score["per_query_r1"],
        }
        if seed == DEPLOYMENT_SEED:
            deployment_model = model.cpu().eval()
            deployment_gallery = packed
    if deployment_model is None or deployment_gallery is None:
        raise RuntimeError("CUB deployment arm is absent")
    persistent_bytes_per_item = deployment_gallery.bytes_per_vector
    if (
        persistent_bytes_per_item != RELATIONAL_DIMENSIONS // 2 + 2
        or pca_int8_packed.bytes_per_vector != persistent_bytes_per_item
        or pca_packed.bytes_per_vector != persistent_bytes_per_item
        or rotated_packed.bytes_per_vector != persistent_bytes_per_item
        or ridge_packed.bytes_per_vector != persistent_bytes_per_item
    ):
        raise RuntimeError("CUB packed width differs")
    rotated_pca = rotated_pca.cpu().eval()
    cpu_device = torch.device("cpu")
    cpu_rotated_packed = pack_int4_unit_embeddings(
        _encode_floating(rotated_pca, source_test, device=cpu_device)
    )
    cpu_deployment_gallery = pack_int4_unit_embeddings(
        _encode_floating(deployment_model, source_test, device=cpu_device)
    )
    if (
        cpu_rotated_packed.to_bytes() != rotated_packed.to_bytes()
        or cpu_deployment_gallery.to_bytes() != deployment_gallery.to_bytes()
    ):
        raise RuntimeError("CUB CPU quality code parity differs")
    rotated_packed = cpu_rotated_packed
    deployment_gallery = cpu_deployment_gallery

    def baseline_call(index: int) -> None:
        retrieve_int4_candidates(
            rotated_pca,
            rotated_packed,
            source_test[index : index + 1],
            exclude_index=index,
            candidate_width=CANDIDATE_WIDTH,
        )

    def treatment_call(index: int) -> None:
        retrieve_int4_candidates(
            deployment_model,
            deployment_gallery,
            source_test[index : index + 1],
            exclude_index=index,
            candidate_width=CANDIDATE_WIDTH,
        )

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            latency = _profile_paired_latency(
                baseline_call,
                treatment_call,
                query_count=len(source_test),
                warmup_pairs=LATENCY_WARMUP_PAIRS,
                measured_pairs=LATENCY_MEASURED_PAIRS,
            )
    finally:
        torch.set_num_threads(prior_threads)
    baseline_latency = cast(dict[str, object], latency["baseline"])
    treatment_latency = cast(dict[str, object], latency["treatment"])
    latency_ratio = cast(int, treatment_latency["p95_ns"]) / cast(int, baseline_latency["p95_ns"])
    passes_latency = latency_ratio <= LATENCY_RATIO_GATE
    model_raw = deployment_model.to_bytes()
    model_sha256 = hashlib.sha256(model_raw).hexdigest()
    environment = {
        "cuda_device": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
    }
    latency_result = {
        **latency,
        "baseline_name": "cub-pca128-random-rotation-int4",
        "claim_eligible": False,
        "environment": environment,
        "gallery_rows": len(source_test),
        "model_artifact_sha256": model_sha256,
        "p95_ratio": latency_ratio,
        "passes_latency": passes_latency,
        "persistent_bytes_per_item": persistent_bytes_per_item,
        "ratio_gate": LATENCY_RATIO_GATE,
        "schema": "sfora-cub-relational-int4-latency-v1",
        "selection_policy": "score-descending-ordinal-ascending",
        "threads": 1,
        "treatment_name": "cub-relational128-int4",
    }
    latency_raw = (
        json.dumps(latency_result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    result = {
        "arms": arms,
        "bootstrap_comparisons": BOOTSTRAP_COMPARISONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "candidate_width": CANDIDATE_WIDTH,
        "claim_eligible": False,
        "controls": {name: _summary(value) for name, value in controls.items()},
        "cpu_quality_code_parity": True,
        "deployment_seed": DEPLOYMENT_SEED,
        "dimensions": RELATIONAL_DIMENSIONS,
        "evaluation_protocol": "cub-classes-101-200-symmetric-leave-self-out",
        "evidence_authority": cub_evidence_authority(pair, source_commit=source_commit),
        "examples_processed": EPOCHS * (len(source_train) // BATCH) * BATCH,
        "map_gain_gate": MAP_GAIN_GATE,
        "learning_rate": LEARNING_RATE,
        "model_artifact_bytes": len(model_raw),
        "model_artifact_sha256": model_sha256,
        "passes_latency": passes_latency,
        "passes_quality": quality_passes,
        "per_query_evidence": {
            "arms": arm_per_query,
            "identity_cluster": test_labels,
            "controls": {
                name: {"ap": controls[name]["per_query_ap"], "r1": controls[name]["per_query_r1"]}
                for name in comparison_names
            },
        },
        "optimizer_steps": EPOCHS * (len(source_train) // BATCH),
        "persistent_bytes_per_item": persistent_bytes_per_item,
        "r1_lower_bound_gate": R1_LOWER_BOUND_GATE,
        "schema": "sfora-cub-relational-int4-evaluation-v1",
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "status": "complete" if quality_passes and passes_latency else "quality-or-latency-stopped",
        "test_classes": len(set(test_labels)),
        "test_rows": len(test_labels),
        "training_device": device.type,
        "training_epochs": EPOCHS,
        "training_batch_size": BATCH,
        "weight_decay": WEIGHT_DECAY,
        "train_classes": len(set(pair["train_labels"])),
        "train_rows": len(source_train),
    }
    quality_raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return quality_raw, model_raw, latency_raw


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate inputs, run once, and publish canonical local evidence."""

    args = parse_args(arguments)
    outputs = (args.output, args.model_output, args.latency_output)
    partials = tuple(path.with_name(path.name + ".partial") for path in outputs)
    publication_paths = outputs + partials
    if len({path.resolve(strict=False) for path in publication_paths}) != len(publication_paths):
        raise ValueError("CUB relational outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("CUB relational output exists")
    if any(path.exists() or path.is_symlink() for path in partials):
        raise FileExistsError("CUB relational partial output exists")
    verify_source_commit(args.source_commit)
    owned_partials: list[Path] = []
    published: list[Path] = []
    try:
        for partial in partials:
            with partial.open("xb"):
                pass
            owned_partials.append(partial)
        pair = load_paired_cub_archives(
            args.source_embeddings,
            args.source_embeddings_sha256,
            args.teacher_embeddings,
            args.teacher_embeddings_sha256,
        )
        payloads = run_sealed_cub_evaluation(pair, source_commit=args.source_commit)
        for partial, payload in zip(partials, payloads, strict=True):
            with partial.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for partial, output in zip(partials, outputs, strict=True):
            os.link(partial, output)
            published.append(output)
        for partial in owned_partials:
            partial.unlink()
    except BaseException:
        for partial in owned_partials:
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
        for output in published:
            if output.is_file() and not output.is_symlink():
                output.unlink()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
