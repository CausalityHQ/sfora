#!/usr/bin/env python3
"""Train and evaluate relational linear compaction on official SOP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from probe_inshop_relational_linear import (
    _encode_deployed,
    _encode_floating,
    _familywise_lower_bound,
    _profile_paired_latency,
    _summary,
)
from torch.nn import functional as F

from sfora.joint_relational_compaction import (
    PackedInt8Embeddings,
    RelationalLinearEncoder,
    RelationalLinearTrainingConfig,
    _fit_uncentered_covariance_basis,
    fit_relational_linear_encoder,
    pack_int8_unit_embeddings,
)

DIMENSIONS = 64
PCA_CONTROL_DIMENSIONS = 128
BATCH = 1024
EPOCHS = 20
TEMPERATURE = 0.05
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEEDS = (17, 1729, 65537)
DEPLOYMENT_SEED = 17
CANDIDATE_WIDTH = 256
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_COMPARISONS = len(SEEDS) * 2
MAP_IMPROVEMENT_GATE = 0.005
R1_LOWER_BOUND_GATE = -0.005
PERSISTENT_BYTES_PER_ITEM = 66
LATENCY_WARMUP_PAIRS = 1_000
LATENCY_MEASURED_PAIRS = 10_000
LATENCY_RATIO_GATE = 1.10


class SymmetricScore(TypedDict):
    """Per-row and aggregate official symmetric retrieval evidence."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


class PairedArchives(TypedDict):
    """Authenticated source/teacher rows with shared retrieval identities."""

    source_metadata: dict[str, object]
    teacher_metadata: dict[str, object]
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


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only SOP evaluation surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--source-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--teacher-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--teacher-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--model-output", required=True, type=_absolute_path)
    parser.add_argument("--latency-output", required=True, type=_absolute_path)
    parser.add_argument("--execute-relational-linear", action="store_true", required=True)
    return parser.parse_args(arguments)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paired_archives(
    source_path: Path,
    source_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
    *,
    expected_counts: tuple[int, int] = (59_551, 60_502),
    expected_classes: tuple[int, int] = (11_318, 11_316),
    expected_dimension: int = 768,
    expected_identifiers: tuple[str, str] = (
        "UNICOM-ViT-B/16",
        "UNICOM-ViT-L/14@336px",
    ),
) -> PairedArchives:
    """Authenticate two SOP archives and require identical row identities."""

    if (
        not isinstance(source_path, Path)
        or not isinstance(teacher_path, Path)
        or type(source_sha256) is not str
        or type(teacher_sha256) is not str
        or len(source_sha256) != 64
        or len(teacher_sha256) != 64
        or set(source_sha256 + teacher_sha256) - set("0123456789abcdef")
    ):
        raise ValueError("SOP archive digest authority differs")
    if _sha256(source_path) != source_sha256 or _sha256(teacher_path) != teacher_sha256:
        raise ValueError("SOP archive digest differs")
    source = load_sop_embedding_archive(
        source_path,
        expected_counts=expected_counts,
        expected_classes=expected_classes,
        expected_dimension=expected_dimension,
    )
    teacher = load_sop_embedding_archive(
        teacher_path,
        expected_counts=expected_counts,
        expected_classes=expected_classes,
        expected_dimension=expected_dimension,
    )
    source_metadata = source["metadata"]
    teacher_metadata = teacher["metadata"]
    if type(source_metadata) is not dict or type(teacher_metadata) is not dict:
        raise ValueError("SOP paired metadata differs")
    if (
        source_metadata.get("model_identifier") != expected_identifiers[0]
        or teacher_metadata.get("model_identifier") != expected_identifiers[1]
        or source_metadata.get("ordered_record_sha256")
        != teacher_metadata.get("ordered_record_sha256")
        or source_metadata.get("model_revision") != teacher_metadata.get("model_revision")
    ):
        raise ValueError("SOP paired metadata differs")
    for name in (
        "train_labels",
        "test_labels",
        "train_image_ids",
        "test_image_ids",
        "train_relative_paths",
        "test_relative_paths",
    ):
        source_value = cast("np.ndarray", source[name])
        teacher_value = cast("np.ndarray", teacher[name])
        if source_value.shape != teacher_value.shape or not bool(
            np.array_equal(source_value, teacher_value)
        ):
            raise ValueError("SOP paired row identity differs")
    source_train = cast("np.ndarray", source["train_embeddings"])
    teacher_train = cast("np.ndarray", teacher["train_embeddings"])
    source_test = cast("np.ndarray", source["test_embeddings"])
    teacher_test = cast("np.ndarray", teacher["test_embeddings"])
    train_labels = cast("np.ndarray", source["train_labels"])
    test_labels = cast("np.ndarray", source["test_labels"])
    return PairedArchives(
        source_metadata=source_metadata,
        teacher_metadata=teacher_metadata,
        source_train=torch.from_numpy(source_train.copy()).float(),
        teacher_train=torch.from_numpy(teacher_train.copy()).float(),
        source_test=torch.from_numpy(source_test.copy()).float(),
        teacher_test=torch.from_numpy(teacher_test.copy()).float(),
        train_labels=tuple(int(value) for value in train_labels.tolist()),
        test_labels=tuple(int(value) for value in test_labels.tolist()),
    )


def _lexicographic_candidates(scores: torch.Tensor, width: int) -> torch.Tensor:
    """Select score-descending, ordinal-ascending candidates exactly."""

    retained = min(width + 1, scores.shape[1])
    values, indexes = torch.topk(scores, k=retained, dim=1, largest=True, sorted=False)
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    indexes = indexes.gather(1, score_order)
    values = values.gather(1, score_order)
    if retained > width:
        ambiguous = values[:, width - 1] == values[:, width]
        for row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
            boundary = values[row, width - 1]
            candidates = torch.nonzero(scores[row] >= boundary, as_tuple=False).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = scores[row, candidates]
            order = torch.argsort(candidate_values, descending=True, stable=True)
            indexes[row, :width] = candidates[order[:width]]
    return indexes[:, :width]


def score_symmetric(
    embeddings: torch.Tensor | PackedInt8Embeddings,
    labels: tuple[int, ...],
    *,
    candidate_width: int,
    device: torch.device,
) -> SymmetricScore:
    """Evaluate leave-self-out MAP@R and Recall@1 on one symmetric split."""

    if isinstance(embeddings, PackedInt8Embeddings):
        row_count = embeddings.codes.shape[0]
        packed = True
    elif type(embeddings) is torch.Tensor:
        row_count = len(embeddings)
        packed = False
        norms = torch.linalg.vector_norm(embeddings.detach().double(), dim=1)
        if (
            embeddings.device.type != "cpu"
            or embeddings.dtype != torch.float32
            or embeddings.ndim != 2
            or embeddings.shape[1] < 2
            or not bool(torch.isfinite(embeddings).all())
            or not bool((torch.abs(norms - 1.0) <= 2e-5).all())
        ):
            raise ValueError("SOP floating embedding authority differs")
    else:
        raise ValueError("SOP scoring representation differs")
    if (
        type(labels) is not tuple
        or len(labels) != row_count
        or any(type(label) is not int or label < 1 for label in labels)
        or type(candidate_width) is not int
        or candidate_width < 1
        or candidate_width >= row_count
        or type(device) is not torch.device
    ):
        raise ValueError("SOP candidate authority differs")
    counts = Counter(labels)
    if any(count < 2 for count in counts.values()):
        raise ValueError("SOP singleton class differs")
    if max(counts.values()) - 1 > candidate_width:
        raise ValueError("SOP candidate width is smaller than a positive set")
    rankings = []
    with torch.inference_mode():
        if packed:
            assert isinstance(embeddings, PackedInt8Embeddings)
            gallery_codes = embeddings.codes.to(device=device, dtype=torch.float32)
            gallery_inverse_norms = embeddings.inverse_norms.to(device=device, dtype=torch.float32)
        else:
            assert isinstance(embeddings, torch.Tensor)
            gallery = embeddings.to(device)
        for start in range(0, row_count, 256):
            stop = min(start + 256, row_count)
            if packed:
                assert isinstance(embeddings, PackedInt8Embeddings)
                query_codes = gallery_codes[start:stop]
                query_inverse_norms = gallery_inverse_norms[start:stop]
                scores = (
                    (query_codes @ gallery_codes.T)
                    * query_inverse_norms.unsqueeze(1)
                    * gallery_inverse_norms.unsqueeze(0)
                )
            else:
                scores = gallery[start:stop] @ gallery.T
            rows = torch.arange(stop - start, device=device)
            columns = torch.arange(start, stop, device=device)
            scores[rows, columns] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, candidate_width).cpu())
    ranked = torch.cat(rankings)
    aps = []
    hits = []
    for ranking, label in zip(ranked.tolist(), labels, strict=True):
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
        map_at_r=math.fsum(aps) / row_count,
        per_query_ap=tuple(aps),
        per_query_r1=tuple(hits),
        r1=math.fsum(hits) / row_count,
    )


def _library_sha256() -> str:
    path = Path(__file__).resolve().parents[1] / "src/sfora/joint_relational_compaction.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _sha256(path)


def _model_sha256(model: RelationalLinearEncoder) -> str:
    value = model.projection.weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the frozen three-seed SOP evaluation and publish canonical evidence."""

    args = parse_args(arguments)
    outputs = (args.output, args.model_output, args.latency_output)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("SOP relational output exists")
    pair = load_paired_archives(
        args.source_embeddings,
        args.source_embeddings_sha256,
        args.teacher_embeddings,
        args.teacher_embeddings_sha256,
    )
    source_train = pair["source_train"]
    teacher_train = pair["teacher_train"]
    source_test = pair["source_test"]
    teacher_test = pair["teacher_test"]
    test_labels = pair["test_labels"]
    basis128 = _fit_uncentered_covariance_basis(source_train, dimensions=PCA_CONTROL_DIMENSIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca64 = RelationalLinearEncoder(basis128[:DIMENSIONS].contiguous())
    pca128 = RelationalLinearEncoder(basis128)
    pca64_float, pca64_packed = _encode_deployed(pca64, source_test, device=device)
    pca128_float = _encode_floating(pca128, source_test, device=device)
    pca128_packed = pack_int8_unit_embeddings(pca128_float)
    source_full = F.normalize(source_test.float(), dim=1)
    teacher_full = F.normalize(teacher_test.float(), dim=1)
    controls: dict[str, SymmetricScore] = {
        "pca64_float": score_symmetric(
            pca64_float, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "pca64_int8": score_symmetric(
            pca64_packed, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "pca128_float": score_symmetric(
            pca128_float, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "pca128_int8": score_symmetric(
            pca128_packed, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "source_full_float": score_symmetric(
            source_full, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
        "teacher_full_float": score_symmetric(
            teacher_full, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        ),
    }
    clusters = torch.tensor(test_labels, dtype=torch.int64)
    arms: dict[str, object] = {}
    arm_per_query: dict[str, object] = {}
    quality_passes = True
    deployment_model: RelationalLinearEncoder | None = None
    deployment_gallery: PackedInt8Embeddings | None = None
    deployment_wire: bytes | None = None
    for seed in SEEDS:
        model, losses = fit_relational_linear_encoder(
            source_train,
            teacher_train,
            basis128[:DIMENSIONS].contiguous(),
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
        floating, packed = _encode_deployed(model, source_test, device=device)
        floating_score = score_symmetric(
            floating, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        )
        packed_score = score_symmetric(
            packed, test_labels, candidate_width=CANDIDATE_WIDTH, device=device
        )
        map_lower = _familywise_lower_bound(
            packed_score["per_query_ap"],
            controls["pca64_int8"]["per_query_ap"],
            clusters,
            seed=seed,
            samples=BOOTSTRAP_SAMPLES,
            comparisons=BOOTSTRAP_COMPARISONS,
        )
        r1_lower = _familywise_lower_bound(
            packed_score["per_query_r1"],
            controls["pca64_int8"]["per_query_r1"],
            clusters,
            seed=seed,
            samples=BOOTSTRAP_SAMPLES,
            comparisons=BOOTSTRAP_COMPARISONS,
        )
        improvement = packed_score["map_at_r"] - controls["pca64_int8"]["map_at_r"]
        passes = (
            improvement >= MAP_IMPROVEMENT_GATE
            and map_lower > 0.0
            and r1_lower > R1_LOWER_BOUND_GATE
        )
        quality_passes = quality_passes and passes
        arms[str(seed)] = {
            "bootstrap_map_lower_bound": map_lower,
            "bootstrap_r1_lower_bound": r1_lower,
            "final_training_loss": losses[-1],
            "floating": _summary(floating_score),
            "int8": _summary(packed_score),
            "map_improvement": improvement,
            "model_sha256": _model_sha256(model),
            "passes_quality": passes,
            "training_losses": losses,
        }
        arm_per_query[str(seed)] = {
            "ap": packed_score["per_query_ap"],
            "r1": packed_score["per_query_r1"],
        }
        if seed == DEPLOYMENT_SEED:
            deployment_model = model.cpu().eval()
            deployment_gallery = packed
            deployment_wire = model.to_bytes()
    if deployment_model is None or deployment_gallery is None or deployment_wire is None:
        raise RuntimeError("SOP deployment arm is absent")
    model_artifact_sha256 = hashlib.sha256(deployment_wire).hexdigest()
    pca64 = pca64.cpu().eval()

    def retrieval_call(
        model: RelationalLinearEncoder,
        gallery: PackedInt8Embeddings,
        index: int,
    ) -> None:
        source_row = F.normalize(source_test[index : index + 1], dim=1)
        encoded = model(source_row)
        packed_query = pack_int8_unit_embeddings(encoded)
        scores = packed_query.cosine_similarity(gallery, device=torch.device("cpu"))
        scores[0, index] = -torch.inf
        torch.topk(scores, k=CANDIDATE_WIDTH, dim=1, largest=True, sorted=True)

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            latency = _profile_paired_latency(
                lambda index: retrieval_call(pca64, pca64_packed, index),
                lambda index: retrieval_call(deployment_model, deployment_gallery, index),
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
    environment = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
    }
    latency_result = {
        **latency,
        "baseline_name": "sop-pca64-int8-packed",
        "claim_eligible": False,
        "dimensions": DIMENSIONS,
        "environment": environment,
        "gallery_rows": len(source_test),
        "library_sha256": _library_sha256(),
        "model_artifact_sha256": model_artifact_sha256,
        "p95_ratio": latency_ratio,
        "passes_latency": passes_latency,
        "persistent_bytes_per_item": PERSISTENT_BYTES_PER_ITEM,
        "ratio_gate": LATENCY_RATIO_GATE,
        "schema": "sfora-sop-relational-linear-packed-latency-v1",
        "script_sha256": _sha256(Path(__file__)),
        "source_embeddings_sha256": args.source_embeddings_sha256,
        "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
        "threads": 1,
        "timed_operation": "normalize-project-pack-packed-cosine-top256-exclude-self",
        "treatment_name": "sop-relational-linear64-int8-packed",
    }
    latency_raw = (
        json.dumps(latency_result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    result = {
        "arms": arms,
        "batch": BATCH,
        "bootstrap_comparisons": BOOTSTRAP_COMPARISONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "candidate_width": CANDIDATE_WIDTH,
        "claim_eligible": False,
        "controls": {name: _summary(value) for name, value in controls.items()},
        "deployment_seed": DEPLOYMENT_SEED,
        "dimensions": DIMENSIONS,
        "epochs": EPOCHS,
        "environment": environment,
        "evaluation_protocol": "official-sop-test-symmetric-leave-self-out",
        "library_sha256": _library_sha256(),
        "map_improvement_gate": MAP_IMPROVEMENT_GATE,
        "model_artifact_bytes": len(deployment_wire),
        "model_artifact_sha256": model_artifact_sha256,
        "model_artifact_schema": "SFORA-RL1",
        "passes_latency": passes_latency,
        "passes_quality": quality_passes,
        "per_query_evidence": {
            "arms": arm_per_query,
            "identity_cluster": test_labels,
            "pca64_int8": {
                "ap": controls["pca64_int8"]["per_query_ap"],
                "r1": controls["pca64_int8"]["per_query_r1"],
            },
        },
        "persistent_bytes_per_item": PERSISTENT_BYTES_PER_ITEM,
        "r1_lower_bound_gate": R1_LOWER_BOUND_GATE,
        "schema": "sfora-sop-relational-linear-evaluation-v1",
        "script_sha256": _sha256(Path(__file__)),
        "seeds": SEEDS,
        "source_embeddings_sha256": args.source_embeddings_sha256,
        "status": (
            "complete" if quality_passes and passes_latency else "quality-or-latency-stopped"
        ),
        "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
        "temperature": TEMPERATURE,
        "test_classes": len(set(test_labels)),
        "test_rows": len(test_labels),
        "train_classes": len(set(pair["train_labels"])),
        "train_rows": len(source_train),
        "weight_decay": WEIGHT_DECAY,
    }
    quality_raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    partials = tuple(path.with_name(path.name + ".partial") for path in outputs)
    if any(path.exists() or path.is_symlink() for path in partials):
        raise FileExistsError("SOP relational partial output exists")
    partials[0].write_bytes(quality_raw)
    partials[1].write_bytes(deployment_wire)
    partials[2].write_bytes(latency_raw)
    for partial, output in zip(partials, outputs, strict=True):
        os.replace(partial, output)
    print(f"sop-relational-linear:COMPLETE sha256={hashlib.sha256(quality_raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
