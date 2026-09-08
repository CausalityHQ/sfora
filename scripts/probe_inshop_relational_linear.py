#!/usr/bin/env python3
"""Train and evaluate the fixed shared relational 64D linear compressor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter_ns
from typing import TypedDict, cast

import numpy as np
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import (
    PackedInt8Embeddings,
    RelationalLinearEncoder,
    RelationalLinearTrainingConfig,
    fit_relational_linear_encoder,
    pack_int8_unit_embeddings,
)
from sfora.split_code_anchor import fit_uncentered_covariance_basis

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
MAP_LOWER_BOUND_GATE = 0.005
R1_LOSS_GATE = 0.002
PERSISTENT_BYTES_PER_ITEM = 66
LATENCY_WARMUP_PAIRS = 1_000
LATENCY_MEASURED_PAIRS = 10_000
LATENCY_RATIO_GATE = 1.10


class ArchiveData(TypedDict):
    """Authenticated paired embedding rows used by the frozen evaluator."""

    metadata: dict[str, object]
    train: torch.Tensor
    train_labels: tuple[str, ...]
    query: torch.Tensor
    query_labels: tuple[str, ...]
    gallery: torch.Tensor
    gallery_labels: tuple[str, ...]


class ScoreResult(TypedDict):
    """Per-query and aggregate retrieval evidence for one arm."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


def _path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only evaluation surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--student-embeddings", type=_path, required=True)
    parser.add_argument("--student-embeddings-sha256", type=_digest, required=True)
    parser.add_argument("--teacher-embeddings", type=_path, required=True)
    parser.add_argument("--teacher-embeddings-sha256", type=_digest, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--model-output", type=_path, required=True)
    parser.add_argument("--latency-output", type=_path, required=True)
    parser.add_argument("--execute-relational-linear", action="store_true", required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _library_sha256() -> str:
    """Bind receipts to the exact library implementation used by this script."""

    path = Path(__file__).resolve().parents[1] / "src/sfora/joint_relational_compaction.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _sha256(path)


def _identity_cluster_ids(labels: tuple[str, ...]) -> tuple[int, ...]:
    """Anonymize identity names into stable sorted integer cluster IDs."""

    if (
        type(labels) is not tuple
        or not labels
        or any(type(label) is not str or not label for label in labels)
    ):
        raise ValueError("relational linear identity authority differs")
    identities = {value: index for index, value in enumerate(sorted(set(labels)))}
    return tuple(identities[value] for value in labels)


def _load_archive(path: Path, expected: str) -> ArchiveData:
    if _sha256(path) != expected:
        raise ValueError("relational linear archive digest differs")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "query_embeddings",
            "query_labels",
            "gallery_embeddings",
            "gallery_labels",
        }
        if set(archive.files) != required:
            raise ValueError("relational linear archive differs")
        return ArchiveData(
            metadata=json.loads(str(archive["metadata_json"].item())),
            train=torch.from_numpy(archive["train_embeddings"].copy()).float(),
            train_labels=tuple(str(x) for x in archive["train_labels"].tolist()),
            query=torch.from_numpy(archive["query_embeddings"].copy()).float(),
            query_labels=tuple(str(x) for x in archive["query_labels"].tolist()),
            gallery=torch.from_numpy(archive["gallery_embeddings"].copy()).float(),
            gallery_labels=tuple(str(x) for x in archive["gallery_labels"].tolist()),
        )


def _validate_paired_archive_rows(
    student: Mapping[str, object], teacher: Mapping[str, object]
) -> None:
    """Require a shared ordered row manifest and a disjoint train/evaluation split."""

    archives = (student, teacher)
    for archive in archives:
        metadata = archive.get("metadata")
        if type(metadata) is not dict:
            raise ValueError("relational linear paired row authority differs")
        digest = metadata.get("image_list_sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("relational linear paired row authority differs")
        for split in ("train", "query", "gallery"):
            values = archive.get(split)
            labels = archive.get(f"{split}_labels")
            if (
                type(values) is not torch.Tensor
                or type(labels) is not tuple
                or len(labels) != len(values)
                or any(type(label) is not str or not label for label in labels)
            ):
                raise ValueError("relational linear paired row authority differs")
    student_metadata = cast(dict[str, object], student["metadata"])
    teacher_metadata = cast(dict[str, object], teacher["metadata"])
    if student_metadata["image_list_sha256"] != teacher_metadata["image_list_sha256"]:
        raise ValueError("relational linear paired row authority differs")
    for role in ("train_labels", "query_labels", "gallery_labels"):
        if student[role] != teacher[role]:
            raise ValueError("relational linear paired row authority differs")
    train_labels = set(cast(tuple[str, ...], student["train_labels"]))
    query_labels = set(cast(tuple[str, ...], student["query_labels"]))
    gallery_labels = set(cast(tuple[str, ...], student["gallery_labels"]))
    if train_labels & (query_labels | gallery_labels) or not query_labels <= gallery_labels:
        raise ValueError("relational linear paired row authority differs")


def _familywise_lower_bound(
    treatment: tuple[float, ...],
    baseline: tuple[float, ...],
    identities: torch.Tensor,
    *,
    seed: int,
    samples: int,
    comparisons: int,
) -> float:
    """Return a deterministic Bonferroni-adjusted identity-bootstrap lower bound."""

    if (
        type(treatment) is not tuple
        or type(baseline) is not tuple
        or len(treatment) != len(baseline)
        or type(identities) is not torch.Tensor
        or len(treatment) != identities.numel()
        or identities.dtype != torch.int64
        or identities.ndim != 1
        or any(not math.isfinite(value) for value in treatment + baseline)
        or type(seed) is not int
        or type(samples) is not int
        or samples < 2
        or type(comparisons) is not int
        or comparisons < 1
    ):
        raise ValueError("relational linear bootstrap authority differs")
    differences = torch.tensor(treatment, dtype=torch.float64) - torch.tensor(
        baseline, dtype=torch.float64
    )
    groups = torch.unique(identities, sorted=True)
    sums = torch.stack([differences[identities == group].sum() for group in groups])
    counts = torch.tensor(
        [int((identities == group).sum()) for group in groups], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(groups), (samples, len(groups)), generator=generator)
    values = sums[draws].sum(dim=1) / counts[draws].sum(dim=1)
    quantile = 0.05 / comparisons
    return float(torch.quantile(values, quantile, interpolation="lower"))


def _random_basis(input_dimensions: int, output_dimensions: int, *, seed: int) -> torch.Tensor:
    """Return a seeded row-orthonormal random projection control."""

    if (
        type(input_dimensions) is not int
        or type(output_dimensions) is not int
        or not 1 < output_dimensions < input_dimensions
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("relational linear random basis authority differs")
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(
        (input_dimensions, output_dimensions),
        dtype=torch.float64,
        generator=generator,
    )
    orthogonal, _triangular = torch.linalg.qr(values, mode="reduced")
    return cast(torch.Tensor, orthogonal.T.float().contiguous())


def _encode_floating(
    model: RelationalLinearEncoder, values: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    model = model.to(device)
    values = F.normalize(values.float(), dim=1)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(values), 1024):
            parts.append(model(values[start : start + 1024].to(device)).cpu())
    return torch.cat(parts).contiguous()


def _encode_deployed(
    model: RelationalLinearEncoder, values: torch.Tensor, *, device: torch.device
) -> tuple[torch.Tensor, PackedInt8Embeddings]:
    floating = _encode_floating(model, values, device=device)
    packed = pack_int8_unit_embeddings(floating)
    if packed.bytes_per_vector != PERSISTENT_BYTES_PER_ITEM:
        raise RuntimeError("relational linear storage authority differs")
    return floating, packed


def _score(
    queries: torch.Tensor | PackedInt8Embeddings,
    gallery: torch.Tensor | PackedInt8Embeddings,
    query_labels: tuple[str, ...],
    gallery_labels: tuple[str, ...],
    *,
    device: torch.device,
) -> ScoreResult:
    if isinstance(queries, PackedInt8Embeddings):
        if not isinstance(gallery, PackedInt8Embeddings):
            raise ValueError("relational linear scoring representations differ")
        packed = True
        query_count = queries.codes.shape[0]
    elif isinstance(queries, torch.Tensor) and isinstance(gallery, torch.Tensor):
        packed = False
        query_count = len(queries)
    else:
        raise ValueError("relational linear scoring representations differ")
    parts = []
    with torch.inference_mode():
        for start in range(0, query_count, 512):
            if packed:
                assert isinstance(queries, PackedInt8Embeddings)
                assert isinstance(gallery, PackedInt8Embeddings)
                query_part = PackedInt8Embeddings(
                    codes=queries.codes[start : start + 512].contiguous(),
                    inverse_norms=queries.inverse_norms[start : start + 512].contiguous(),
                )
                scores = query_part.cosine_similarity(gallery, device=device)
            else:
                assert isinstance(queries, torch.Tensor)
                assert isinstance(gallery, torch.Tensor)
                query_device = queries[start : start + 512].float().to(device)
                gallery_device = gallery.float().to(device)
                scores = query_device @ gallery_device.T
            parts.append(
                torch.topk(
                    scores,
                    k=CANDIDATE_WIDTH,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices.cpu()
            )
    rankings = torch.cat(parts)
    aps = []
    hits = []
    positive_counts = Counter(gallery_labels)
    for ranking, label in zip(rankings.tolist(), query_labels, strict=True):
        positives = positive_counts[label]
        if not 1 <= positives <= CANDIDATE_WIDTH:
            raise ValueError("relational linear positive authority differs")
        found = 0
        terms = []
        for rank, index in enumerate(ranking[:positives], 1):
            if gallery_labels[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positives)
        hits.append(float(gallery_labels[ranking[0]] == label))
    return ScoreResult(
        map_at_r=math.fsum(aps) / len(aps),
        per_query_ap=tuple(aps),
        per_query_r1=tuple(hits),
        r1=math.fsum(hits) / len(hits),
    )


def _model_sha256(model: RelationalLinearEncoder) -> str:
    value = model.projection.weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _summary(value: ScoreResult) -> dict[str, float]:
    return {"map_at_r": value["map_at_r"], "r1": value["r1"]}


def _latency_summary(samples: list[int]) -> dict[str, object]:
    if not samples or any(type(value) is not int or value <= 0 for value in samples):
        raise ValueError("relational linear latency samples differ")
    ordered = sorted(samples)

    def percentile(percent: int) -> int:
        return ordered[(percent * len(ordered) + 99) // 100 - 1]

    return {
        "mean_ns": round(math.fsum(samples) / len(samples)),
        "p50_ns": percentile(50),
        "p95_ns": percentile(95),
        "p99_ns": percentile(99),
        "samples": len(samples),
        "samples_ns": samples,
    }


def _profile_paired_latency(
    baseline: Callable[[int], None],
    treatment: Callable[[int], None],
    *,
    query_count: int,
    warmup_pairs: int,
    measured_pairs: int,
    clock: Callable[[], int] = perf_counter_ns,
) -> dict[str, object]:
    """Alternate paired calls and retain every positive post-warmup sample."""

    if (
        query_count < 1
        or warmup_pairs < 0
        or measured_pairs < 1
        or type(query_count) is not int
        or type(warmup_pairs) is not int
        or type(measured_pairs) is not int
    ):
        raise ValueError("relational linear latency config differs")
    samples: dict[str, list[int]] = {"baseline": [], "treatment": []}
    callables = {"baseline": baseline, "treatment": treatment}
    for pair in range(warmup_pairs + measured_pairs):
        index = pair % query_count
        order = ("baseline", "treatment") if pair % 2 == 0 else ("treatment", "baseline")
        for name in order:
            if pair < warmup_pairs:
                callables[name](index)
                continue
            started = clock()
            callables[name](index)
            elapsed = clock() - started
            if elapsed <= 0:
                raise RuntimeError("relational linear latency clock failed")
            samples[name].append(elapsed)
    return {
        "alternating_order": True,
        "baseline": _latency_summary(samples["baseline"]),
        "measured_pairs": measured_pairs,
        "treatment": _latency_summary(samples["treatment"]),
        "warmup_pairs": warmup_pairs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the frozen training recipe and its one sealed evaluation."""

    args = parse_args(argv)
    for path in (args.output, args.model_output, args.latency_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    student = _load_archive(args.student_embeddings, args.student_embeddings_sha256)
    teacher = _load_archive(args.teacher_embeddings, args.teacher_embeddings_sha256)
    _validate_paired_archive_rows(student, teacher)
    if (
        student["metadata"].get("model_identifier") != "UNICOM-ViT-B/16"
        or teacher["metadata"].get("model_identifier") != "UNICOM-ViT-L/14@336px"
        or student["metadata"].get("image_list_sha256")
        != teacher["metadata"].get("image_list_sha256")
        or student["train"].shape != (25_882, 768)
        or student["query"].shape != (14_218, 768)
        or student["gallery"].shape != (12_612, 768)
        or any(
            not bool(torch.isfinite(values).all())
            for archive in (student, teacher)
            for values in (archive["train"], archive["query"], archive["gallery"])
        )
    ):
        raise ValueError("relational linear In-Shop evidence differs")
    basis128 = fit_uncentered_covariance_basis(
        student["train"], dimensions=PCA_CONTROL_DIMENSIONS
    ).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca64 = RelationalLinearEncoder(basis128[:DIMENSIONS].contiguous())
    pca128 = RelationalLinearEncoder(basis128)
    random64 = RelationalLinearEncoder(_random_basis(768, DIMENSIONS, seed=17))
    pca64_float, pca64_int8 = _encode_deployed(pca64, student["query"], device=device)
    pca64_gallery_float, pca64_gallery_int8 = _encode_deployed(
        pca64, student["gallery"], device=device
    )
    pca128_float = _encode_floating(pca128, student["query"], device=device)
    pca128_gallery_float = _encode_floating(pca128, student["gallery"], device=device)
    pca128_int8 = pack_int8_unit_embeddings(pca128_float)
    pca128_gallery_int8 = pack_int8_unit_embeddings(pca128_gallery_float)
    _random_float, random_int8 = _encode_deployed(random64, student["query"], device=device)
    _random_gallery_float, random_gallery_int8 = _encode_deployed(
        random64, student["gallery"], device=device
    )
    controls = {
        "pca64_float": _score(
            pca64_float,
            pca64_gallery_float,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        ),
        "pca64_int8": _score(
            pca64_int8,
            pca64_gallery_int8,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        ),
        "pca128_float": _score(
            pca128_float,
            pca128_gallery_float,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        ),
        "pca128_int8": _score(
            pca128_int8,
            pca128_gallery_int8,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        ),
        "random64_int8": _score(
            random_int8,
            random_gallery_int8,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        ),
    }
    identity_clusters = _identity_cluster_ids(student["query_labels"])
    clusters = torch.tensor(identity_clusters, dtype=torch.int64)
    arms = {}
    arm_per_query = {}
    quality_passes = True
    deployment_model_wire = None
    deployment_model = None
    deployment_gallery = None
    for seed in SEEDS:
        model, losses = fit_relational_linear_encoder(
            student["train"],
            teacher["train"],
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
        query_float, query_int8 = _encode_deployed(model, student["query"], device=device)
        gallery_float, gallery_int8 = _encode_deployed(model, student["gallery"], device=device)
        floating = _score(
            query_float,
            gallery_float,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        )
        deployed = _score(
            query_int8,
            gallery_int8,
            student["query_labels"],
            student["gallery_labels"],
            device=device,
        )
        map_lower = _familywise_lower_bound(
            deployed["per_query_ap"],
            controls["pca64_int8"]["per_query_ap"],
            clusters,
            seed=seed,
            samples=BOOTSTRAP_SAMPLES,
            comparisons=BOOTSTRAP_COMPARISONS,
        )
        r1_lower = _familywise_lower_bound(
            deployed["per_query_r1"],
            controls["pca64_int8"]["per_query_r1"],
            clusters,
            seed=seed,
            samples=BOOTSTRAP_SAMPLES,
            comparisons=BOOTSTRAP_COMPARISONS,
        )
        halfway = 0.5 * (
            float(controls["pca128_int8"]["map_at_r"]) - float(controls["pca64_int8"]["map_at_r"])
        )
        improvement = float(deployed["map_at_r"]) - float(controls["pca64_int8"]["map_at_r"])
        passes = (
            improvement >= halfway
            and map_lower > MAP_LOWER_BOUND_GATE
            and r1_lower >= -R1_LOSS_GATE
        )
        quality_passes = quality_passes and passes
        arms[str(seed)] = {
            "bootstrap_map_lower_bound": map_lower,
            "bootstrap_r1_lower_bound": r1_lower,
            "final_training_loss": losses[-1],
            "floating": _summary(floating),
            "int8": _summary(deployed),
            "map_improvement": improvement,
            "model_sha256": _model_sha256(model),
            "passes_quality": passes,
            "training_losses": losses,
        }
        arm_per_query[str(seed)] = {
            "ap": deployed["per_query_ap"],
            "r1": deployed["per_query_r1"],
        }
        if seed == DEPLOYMENT_SEED:
            deployment_model_wire = model.to_bytes()
            deployment_model = model
            deployment_gallery = gallery_int8
    if deployment_model_wire is None or deployment_model is None or deployment_gallery is None:
        raise RuntimeError("relational linear deployment model is absent")
    model_artifact_sha256 = hashlib.sha256(deployment_model_wire).hexdigest()
    pca64 = pca64.cpu().eval()
    deployment_model = deployment_model.cpu().eval()

    def retrieval_call(
        model: RelationalLinearEncoder,
        gallery: PackedInt8Embeddings,
        index: int,
    ) -> None:
        source_row = F.normalize(student["query"][index : index + 1], dim=1)
        encoded = model(source_row)
        packed_query = pack_int8_unit_embeddings(encoded)
        scores = packed_query.cosine_similarity(gallery, device=torch.device("cpu"))
        torch.topk(scores, k=CANDIDATE_WIDTH, dim=1, largest=True, sorted=True)

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            latency = _profile_paired_latency(
                lambda index: retrieval_call(pca64, pca64_gallery_int8, index),
                lambda index: retrieval_call(deployment_model, deployment_gallery, index),
                query_count=len(student["query"]),
                warmup_pairs=LATENCY_WARMUP_PAIRS,
                measured_pairs=LATENCY_MEASURED_PAIRS,
            )
    finally:
        torch.set_num_threads(prior_threads)
    baseline_latency = cast(dict[str, object], latency["baseline"])
    treatment_latency = cast(dict[str, object], latency["treatment"])
    latency_ratio = cast(int, treatment_latency["p95_ns"]) / cast(int, baseline_latency["p95_ns"])
    passes_latency = latency_ratio <= LATENCY_RATIO_GATE
    latency_result = {
        **latency,
        "baseline_name": "pca64-int8-packed",
        "claim_eligible": False,
        "dimensions": DIMENSIONS,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "gallery_rows": len(student["gallery"]),
        "library_sha256": _library_sha256(),
        "model_artifact_sha256": model_artifact_sha256,
        "p95_ratio": latency_ratio,
        "passes_latency": passes_latency,
        "persistent_bytes_per_item": PERSISTENT_BYTES_PER_ITEM,
        "ratio_gate": LATENCY_RATIO_GATE,
        "schema": "sfora-relational-linear-packed-latency-v2",
        "script_sha256": _sha256(Path(__file__)),
        "student_embeddings_sha256": args.student_embeddings_sha256,
        "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
        "threads": 1,
        "timed_operation": "normalize-project-pack-packed-cosine-top256",
        "treatment_name": "relational-linear64-int8-packed",
    }
    latency_raw = (
        json.dumps(latency_result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    latency_sha256 = hashlib.sha256(latency_raw).hexdigest()
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
        "evaluation_opened": True,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "image_list_sha256": student["metadata"]["image_list_sha256"],
        "library_sha256": _library_sha256(),
        "learning_rate": LEARNING_RATE,
        "latency_receipt_sha256": latency_sha256,
        "passes_latency": passes_latency,
        "map_lower_bound_gate": MAP_LOWER_BOUND_GATE,
        "model_artifact_bytes": len(deployment_model_wire),
        "model_artifact_sha256": model_artifact_sha256,
        "model_artifact_schema": "SFORA-RL1",
        "passes_quality": quality_passes,
        "per_query_evidence": {
            "arms": arm_per_query,
            "identity_cluster": identity_clusters,
            "pca64_int8": {
                "ap": controls["pca64_int8"]["per_query_ap"],
                "r1": controls["pca64_int8"]["per_query_r1"],
            },
        },
        "persistent_bytes_per_item": PERSISTENT_BYTES_PER_ITEM,
        "r1_loss_gate": R1_LOSS_GATE,
        "schema": "sfora-inshop-relational-linear-evaluation-v2",
        "script_sha256": _sha256(Path(__file__)),
        "seeds": SEEDS,
        "status": (
            "complete" if quality_passes and passes_latency else "quality-or-latency-stopped"
        ),
        "student_embeddings_sha256": args.student_embeddings_sha256,
        "teacher_embeddings_sha256": args.teacher_embeddings_sha256,
        "temperature": TEMPERATURE,
        "weight_decay": WEIGHT_DECAY,
    }
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    partial = args.output.with_name(args.output.name + ".partial")
    model_partial = args.model_output.with_name(args.model_output.name + ".partial")
    latency_partial = args.latency_output.with_name(args.latency_output.name + ".partial")
    for path in (partial, model_partial, latency_partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    model_partial.write_bytes(deployment_model_wire)
    latency_partial.write_bytes(latency_raw)
    partial.write_bytes(raw)
    model_partial.replace(args.model_output)
    latency_partial.replace(args.latency_output)
    partial.replace(args.output)
    print(f"relational-linear:COMPLETE sha256={hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
