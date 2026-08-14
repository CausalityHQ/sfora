#!/usr/bin/env python3
"""Evaluate the frozen-teacher relational-anchor falsifier on train identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from sfora.unicom_retrieval_audit import retrieval_metrics_from_score_chunks

REGISTERED_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
SOUP_PROTOCOL = "unicom-train-identity-holdout-suffix-soup-wise-v3"
PROVENANCE_FIELDS = (
    "screen_report",
    "screen_report_sha256",
    "endpoint_state",
    "endpoint_state_sha256",
    "epoch8_checkpoint",
    "epoch8_checkpoint_sha256",
    "initial_checkpoint_sha256",
    "batch_norm_recalibration",
    "training_protocol",
)


def _embedding_matrix(values: np.ndarray, *, name: str) -> None:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.float32
        or values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty C-contiguous FP32 matrix")


def deployment_scores(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    *,
    selected_features: int,
) -> np.ndarray:
    """Return exact deployment scores: full L2 normalization, then prefix distance."""

    _embedding_matrix(query_embeddings, name="query embeddings")
    _embedding_matrix(gallery_embeddings, name="gallery embeddings")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("query/gallery embedding dimensions differ")
    if (
        type(selected_features) is not int
        or selected_features <= 0
        or selected_features > query_embeddings.shape[1]
    ):
        raise ValueError("selected features differ")

    def normalized_prefix(values: np.ndarray) -> np.ndarray:
        values64 = values.astype(np.float64)
        norms = np.sqrt(np.sum(values64 * values64, axis=1, keepdims=True, dtype=np.float64))
        if np.any(norms == 0.0):
            raise ValueError("embedding row has zero norm")
        normalized = np.ascontiguousarray((values64 / norms).astype(np.float32))
        return np.ascontiguousarray(normalized[:, :selected_features].astype(np.float64))

    query = normalized_prefix(query_embeddings)
    gallery = normalized_prefix(gallery_embeddings)
    query_norms = np.sum(query * query, axis=1, dtype=np.float64)
    gallery_norms = np.sum(gallery * gallery, axis=1, dtype=np.float64)
    distances = query_norms[:, None] + gallery_norms[None, :] - 2.0 * (query @ gallery.T)
    return np.ascontiguousarray(-distances, dtype=np.float64)


def _score_matrix(values: np.ndarray, *, name: str) -> None:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.float64
        or values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty C-contiguous FP64 matrix")


def standardize_score_rows(values: np.ndarray) -> np.ndarray:
    """Remove per-query location/scale so blend weights are comparable across arms."""

    _score_matrix(values, name="scores")
    means = np.mean(values, axis=1, keepdims=True, dtype=np.float64)
    centered = values - means
    scales = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float64))
    scales[scales == 0.0] = 1.0
    return np.ascontiguousarray(centered / scales, dtype=np.float64)


def evaluate_blend_grid(
    endpoint_scores: np.ndarray,
    teacher_scores: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    alphas: Sequence[float] = REGISTERED_ALPHAS,
) -> list[dict[str, object]]:
    _score_matrix(endpoint_scores, name="endpoint scores")
    _score_matrix(teacher_scores, name="teacher scores")
    if endpoint_scores.shape != teacher_scores.shape:
        raise ValueError("endpoint/teacher score shapes differ")
    endpoint_scores = standardize_score_rows(endpoint_scores)
    teacher_scores = standardize_score_rows(teacher_scores)
    alphas = tuple(alphas)
    if (
        not alphas
        or tuple(sorted(set(alphas))) != alphas
        or any(
            type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
            for alpha in alphas
        )
    ):
        raise ValueError("blend alphas must be unique increasing finite builtin floats in [0, 1]")
    if alphas[0] != 0.0:
        raise ValueError("blend alpha grid must begin with the endpoint")
    candidates: list[dict[str, object]] = []
    for alpha in alphas:
        scores = np.ascontiguousarray(
            endpoint_scores * (1.0 - alpha) + teacher_scores * alpha,
            dtype=np.float64,
        )
        result = retrieval_metrics_from_score_chunks((scores,), query_labels, gallery_labels)
        candidates.append(
            {
                "alpha": alpha,
                "metrics": {
                    "recall_at_1": result.recall[1],
                    "recall_at_10": result.recall[10],
                    "recall_at_20": result.recall[20],
                    "recall_at_30": result.recall[30],
                    "map_at_r": result.map_at_r,
                },
                "query_evidence": {
                    "top1_correct": result.top1_correct.tolist(),
                    "average_precision": result.average_precision.tolist(),
                },
            }
        )
    return candidates


def select_blend_candidate(candidates: Sequence[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise ValueError("candidate list is empty")
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["map_at_r"],
            row["metrics"]["recall_at_1"],
            -row["alpha"],
        ),
    )


def falsifier_gate(
    endpoint: dict[str, object],
    pretrained: dict[str, object],
    pure_teacher: dict[str, object],
    epoch8: dict[str, object],
    *,
    map_bootstrap_interval: tuple[float, float],
) -> dict[str, object]:
    endpoint_map = endpoint["metrics"]["map_at_r"]
    pretrained_map = pretrained["metrics"]["map_at_r"]
    epoch8_map = epoch8["metrics"]["map_at_r"]
    pure_teacher_map = pure_teacher["metrics"]["map_at_r"]
    endpoint_r1 = endpoint["metrics"]["recall_at_1"]
    pretrained_r1 = pretrained["metrics"]["recall_at_1"]
    values = (
        endpoint_map,
        pretrained_map,
        pure_teacher_map,
        epoch8_map,
        endpoint_r1,
        pretrained_r1,
        *map_bootstrap_interval,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise TypeError("falsifier metrics must be finite builtin floats")
    pretrained_gain = pretrained_map - endpoint_map
    pure_teacher_gain = pretrained_map - pure_teacher_map
    epoch8_gain = max(0.0, epoch8_map - endpoint_map)
    gain_fraction = epoch8_gain / pretrained_gain if pretrained_gain > 0.0 else None
    recall_delta = pretrained_r1 - endpoint_r1
    return {
        "minimum_map_gain": 0.003,
        "maximum_control_gain_fraction": 0.7,
        "recall_at_1_guard": -0.00125,
        "winner_is_interior": 0.0 < pretrained["alpha"] < 1.0,
        "pretrained_map_gain_over_endpoint": pretrained_gain,
        "pretrained_map_gain_over_pure_teacher": pure_teacher_gain,
        "epoch8_map_gain": epoch8_gain,
        "epoch8_gain_fraction": gain_fraction,
        "pretrained_recall_at_1_delta": recall_delta,
        "bootstrap_seed": 0,
        "bootstrap_samples": 10_000,
        "pretrained_map_delta_query_bootstrap_95_interval": list(map_bootstrap_interval),
        "interpretation": "exploratory_seed0_train_holdout_falsifier_only",
        "passed": (
            pretrained_gain >= 0.003
            and pure_teacher_gain >= 0.003
            and 0.0 < pretrained["alpha"] < 1.0
            and gain_fraction is not None
            and gain_fraction < 0.7
            and recall_delta >= -0.00125
            and map_bootstrap_interval[0] > 0.0
        ),
    }


def paired_map_bootstrap_interval(
    endpoint: dict[str, object],
    candidate: dict[str, object],
    *,
    seed: int = 0,
    samples: int = 10_000,
    chunk_size: int = 64,
) -> tuple[float, float]:
    endpoint_values = np.asarray(endpoint["query_evidence"]["average_precision"], dtype=np.float64)
    candidate_values = np.asarray(
        candidate["query_evidence"]["average_precision"], dtype=np.float64
    )
    if endpoint_values.shape != candidate_values.shape or endpoint_values.ndim != 1:
        raise ValueError("paired mAP evidence differs")
    generator = np.random.Generator(np.random.PCG64(seed))
    differences = candidate_values - endpoint_values
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1, dtype=np.float64)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def evaluate_falsifier(
    endpoint_scores: np.ndarray,
    pretrained_scores: np.ndarray,
    epoch8_scores: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    alphas: Sequence[float] = REGISTERED_ALPHAS,
    selected_features: int,
    provenance: dict[str, object],
) -> dict[str, object]:
    pretrained_candidates = evaluate_blend_grid(
        endpoint_scores,
        pretrained_scores,
        query_labels,
        gallery_labels,
        alphas=alphas,
    )
    epoch8_candidates = evaluate_blend_grid(
        endpoint_scores,
        epoch8_scores,
        query_labels,
        gallery_labels,
        alphas=alphas,
    )
    endpoint = pretrained_candidates[0]
    pretrained_best = select_blend_candidate(pretrained_candidates)
    epoch8_best = select_blend_candidate(epoch8_candidates)
    pure_teacher = pretrained_candidates[-1]
    map_interval = paired_map_bootstrap_interval(endpoint, pretrained_best)
    report = {
        "protocol": "unicom-frozen-teacher-relational-anchor-falsifier-v1",
        "alphas": list(alphas),
        "selected_features": selected_features,
        "score_preprocessing": "per-query-fp64-zero-mean-unit-rms",
        "provenance": provenance,
        "endpoint": endpoint,
        "pretrained": {"best": pretrained_best, "candidates": pretrained_candidates},
        "epoch8_control": {"best": epoch8_best, "candidates": epoch8_candidates},
        "gate": falsifier_gate(
            endpoint,
            pretrained_best,
            pure_teacher,
            epoch8_best,
            map_bootstrap_interval=map_interval,
        ),
    }
    validate_falsifier_report(report)
    return report


def _validate_candidate(candidate: object, *, query_count: int | None = None) -> int:
    if type(candidate) is not dict or tuple(candidate) != (
        "alpha",
        "metrics",
        "query_evidence",
    ):
        raise ValueError("blend candidate schema differs")
    alpha = candidate["alpha"]
    metrics = candidate["metrics"]
    evidence = candidate["query_evidence"]
    if type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("blend candidate alpha differs")
    if type(metrics) is not dict or tuple(metrics) != (
        "recall_at_1",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "map_at_r",
    ):
        raise ValueError("blend candidate metrics schema differs")
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in metrics.values()
    ):
        raise ValueError("blend candidate metrics differ")
    if type(evidence) is not dict or tuple(evidence) != (
        "top1_correct",
        "average_precision",
    ):
        raise ValueError("blend candidate evidence schema differs")
    top1 = evidence["top1_correct"]
    average_precision = evidence["average_precision"]
    if (
        type(top1) is not list
        or type(average_precision) is not list
        or not top1
        or len(top1) != len(average_precision)
        or (query_count is not None and len(top1) != query_count)
        or any(type(value) is not bool for value in top1)
        or any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in average_precision
        )
    ):
        raise ValueError("blend candidate evidence differs")
    if metrics["recall_at_1"] != sum(top1) / len(top1):
        raise ValueError("blend candidate recall evidence differs")
    recomputed_map = float(np.mean(np.asarray(average_precision, dtype=np.float64)))
    if metrics["map_at_r"] != recomputed_map:
        raise ValueError("blend candidate mAP evidence differs")
    return len(top1)


def validate_falsifier_report(report: object) -> None:
    if type(report) is not dict or tuple(report) != (
        "protocol",
        "alphas",
        "selected_features",
        "score_preprocessing",
        "provenance",
        "endpoint",
        "pretrained",
        "epoch8_control",
        "gate",
    ):
        raise ValueError("falsifier report schema differs")
    if report["protocol"] != "unicom-frozen-teacher-relational-anchor-falsifier-v1":
        raise ValueError("falsifier report protocol differs")
    alphas = report["alphas"]
    if alphas != list(REGISTERED_ALPHAS):
        raise ValueError("falsifier report alpha grid differs")
    if report["selected_features"] != 512:
        raise ValueError("falsifier selected features differ")
    if report["score_preprocessing"] != "per-query-fp64-zero-mean-unit-rms":
        raise ValueError("falsifier score preprocessing differs")
    provenance = report["provenance"]
    if type(provenance) is not dict or tuple(provenance) != PROVENANCE_FIELDS:
        raise ValueError("falsifier provenance differs")
    if any(
        type(provenance[name]) is not str or not provenance[name]
        for name in ("screen_report", "endpoint_state", "epoch8_checkpoint")
    ):
        raise ValueError("falsifier provenance paths differ")
    hexadecimal = frozenset("0123456789abcdef")
    if any(
        type(provenance[name]) is not str
        or len(provenance[name]) != 64
        or any(character not in hexadecimal for character in provenance[name])
        for name in (
            "screen_report_sha256",
            "endpoint_state_sha256",
            "epoch8_checkpoint_sha256",
            "initial_checkpoint_sha256",
        )
    ):
        raise ValueError("falsifier provenance hashes differ")
    if provenance["batch_norm_recalibration"] != ("full-optimization-cumulative-batches-all-arms"):
        raise ValueError("falsifier provenance recalibration differs")
    if type(provenance["training_protocol"]) is not dict or not provenance["training_protocol"]:
        raise ValueError("falsifier provenance training protocol differs")
    query_count = _validate_candidate(report["endpoint"])
    groups: list[tuple[str, dict[str, object]]] = []
    for name in ("pretrained", "epoch8_control"):
        group = report[name]
        if type(group) is not dict or tuple(group) != ("best", "candidates"):
            raise ValueError(f"{name} group schema differs")
        candidates = group["candidates"]
        if type(candidates) is not list or len(candidates) != len(alphas):
            raise ValueError(f"{name} candidates differ")
        for alpha, candidate in zip(alphas, candidates, strict=True):
            _validate_candidate(candidate, query_count=query_count)
            if candidate["alpha"] != alpha:
                raise ValueError(f"{name} candidate alpha differs")
        groups.append((name, group))
    if report["endpoint"] != report["pretrained"]["candidates"][0]:
        raise ValueError("endpoint candidate differs")
    if report["epoch8_control"]["candidates"][0] != report["endpoint"]:
        raise ValueError("control endpoint candidate differs")
    for name, group in groups:
        if group["best"] != select_blend_candidate(group["candidates"]):
            raise ValueError(f"{name.replace('_control', '')} winner differs")
    expected_gate = falsifier_gate(
        report["endpoint"],
        report["pretrained"]["best"],
        report["pretrained"]["candidates"][-1],
        report["epoch8_control"]["best"],
        map_bootstrap_interval=paired_map_bootstrap_interval(
            report["endpoint"], report["pretrained"]["best"]
        ),
    )
    if report["gate"] != expected_gate:
        raise ValueError("falsifier gate differs")


def write_report_atomic(report: dict[str, object], output: Path) -> None:
    validate_falsifier_report(report)
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    descriptor: int | None = None
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = json.loads(temporary.read_text())
        validate_falsifier_report(persisted)
        os.link(temporary, output)
        temporary.unlink()
        validate_falsifier_report(json.loads(output.read_text()))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if owned is not None and (info.st_dev, info.st_ino) == owned:
                temporary.unlink()


def validate_screen_binding(
    screen: object,
    endpoint_payload: object,
    epoch8_checkpoint: object,
) -> None:
    if type(screen) is not dict or tuple(screen) != (
        "protocol",
        "mode",
        "training_seed",
        "alphas",
        "batch_size",
        "workers",
        "selected_features",
        "screen_report",
        "screen_report_sha256",
        "holdout_seed",
        "holdout_fraction",
        "training_protocol",
        "batch_norm_recalibration",
        "selected",
        "endpoint",
        "gate",
        "candidates",
        "model_path",
    ):
        raise ValueError("screen report schema differs")
    if (
        screen["protocol"] != SOUP_PROTOCOL
        or screen["mode"] != "screen"
        or screen["training_seed"] != 0
        or screen["alphas"] != [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]
        or screen["batch_size"] != 128
        or screen["workers"] != 4
        or screen["selected_features"] != 512
        or screen["screen_report"] is not None
        or screen["screen_report_sha256"] is not None
        or screen["holdout_seed"] != 0
        or screen["holdout_fraction"] != 0.2
        or screen["batch_norm_recalibration"] != "full-optimization-cumulative-batches"
    ):
        raise ValueError("screen report protocol differs")
    candidates = screen["candidates"]
    selected = screen["selected"]
    endpoint = screen["endpoint"]
    expected_specs = [
        (list(epochs), alpha)
        for epochs in ((16,), (12, 16), (8, 12, 16), (4, 8, 12, 16))
        for alpha in (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0)
    ]
    if (
        type(candidates) is not list
        or [
            (candidate.get("epochs"), candidate.get("alpha"))
            for candidate in candidates
            if type(candidate) is dict
        ]
        != expected_specs
    ):
        raise ValueError("screen candidate grid differs")
    if (
        type(selected) is not dict
        or type(endpoint) is not dict
        or selected not in candidates
        or endpoint not in candidates
        or selected
        != max(
            candidates,
            key=lambda row: (row["metrics"]["map_at_r"], row["metrics"]["recall_at_1"]),
        )
        or selected != endpoint
        or endpoint.get("epochs") != [16]
        or endpoint.get("alpha") != 1.0
    ):
        raise ValueError("screen endpoint closure differs")
    gate = screen["gate"]
    if type(gate) is not dict or gate.get("promoted") is not False:
        raise ValueError("screen did not close soup selection")
    training_protocol = screen["training_protocol"]
    if type(training_protocol) is not dict or not training_protocol:
        raise ValueError("screen training protocol differs")
    if (
        type(endpoint_payload) is not dict
        or tuple(endpoint_payload) != ("model", "endpoint", "training_protocol")
        or not isinstance(endpoint_payload["model"], Mapping)
        or endpoint_payload["endpoint"] != endpoint
        or endpoint_payload["training_protocol"] != training_protocol
    ):
        raise ValueError("endpoint state binding differs")
    if (
        type(epoch8_checkpoint) is not dict
        or epoch8_checkpoint.get("epoch") != 8
        or epoch8_checkpoint.get("selection_holdout") != {"seed": 0, "fraction": 0.2}
        or epoch8_checkpoint.get("training_protocol") != training_protocol
        or not isinstance(epoch8_checkpoint.get("model"), Mapping)
    ):
        raise ValueError("epoch-8 control binding differs")


def validate_epoch8_path(screen: dict[str, object], epoch8_path: Path) -> None:
    registered_paths = {
        checkpoint for candidate in screen["candidates"] for checkpoint in candidate["checkpoints"]
    }
    if str(epoch8_path) not in registered_paths:
        raise ValueError("epoch-8 control path is absent from screen")


def validate_endpoint_reproduction(
    screen: dict[str, object], reproduced: dict[str, object]
) -> None:
    expected_metrics = screen["endpoint"]["metrics"]
    actual_metrics = reproduced.get("metrics")
    metric_names = (
        "recall_at_1",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "map_at_r",
    )
    if (
        type(actual_metrics) is not dict
        or tuple(actual_metrics) != metric_names
        or any(
            type(actual_metrics[name]) is not float
            or abs(actual_metrics[name] - expected_metrics[name]) > 1e-6
            for name in metric_names
        )
    ):
        raise ValueError("endpoint metrics do not reproduce the soup screen")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encode_recalibrated_arm(
    soup,
    model,
    state: Mapping[str, object] | None,
    calibration_loader,
    *,
    device,
    encode: Callable[[], object],
):
    if state is not None:
        model.load_state_dict(state, strict=True)
    soup.recalibrate_batch_norm(model, calibration_loader, device=device)
    return encode()


def load_registered_scores(args: argparse.Namespace):
    """Load three registered states serially and return train-holdout score matrices."""

    import torch

    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import identity_holdout

    soup = _load_local_module(
        Path(__file__).resolve().with_name("evaluate_unicom_checkpoint_soup.py"),
        "_unicom_soup_for_ftra",
    )
    trainer = soup._load_trainer()
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if trainer._sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM initial checkpoint SHA-256 differs")
    screen = json.loads(args.screen_report.read_text())
    endpoint_payload = torch.load(
        args.endpoint_state, map_location="cpu", weights_only=False, mmap=True
    )
    epoch8_checkpoint = torch.load(
        args.epoch8_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    validate_screen_binding(screen, endpoint_payload, epoch8_checkpoint)
    validate_epoch8_path(screen, args.epoch8_checkpoint)
    expected_protocol = soup._expected_screen_training_protocol(trainer, args)
    if screen["training_protocol"] != expected_protocol:
        raise ValueError("screen training protocol differs from registered runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FTRA falsifier")

    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, _labels = identity_holdout(train_records, fraction=0.2, seed=0)
    if not query or not gallery:
        raise ValueError("falsifier requires a nonempty train-only holdout")
    model, transform = trainer._load_official_model(args.unicom_checkout, args.initial_checkpoint)
    device = torch.device("cuda")
    model = model.to(device)

    def encode() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_values, query_labels = trainer._encode_records(
            model,
            query,
            transform,
            device=device,
            batch_size=128,
            workers=4,
        )
        gallery_values, gallery_labels = trainer._encode_records(
            model,
            gallery,
            transform,
            device=device,
            batch_size=128,
            workers=4,
        )
        return query_values, gallery_values, query_labels, gallery_labels

    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    pretrained_query, pretrained_gallery, query_labels, gallery_labels = encode_recalibrated_arm(
        soup,
        model,
        None,
        calibration_loader,
        device=device,
        encode=encode,
    )
    pretrained_scores = deployment_scores(
        pretrained_query, pretrained_gallery, selected_features=512
    )

    endpoint_query, endpoint_gallery, endpoint_query_labels, endpoint_gallery_labels = (
        encode_recalibrated_arm(
            soup,
            model,
            endpoint_payload["model"],
            calibration_loader,
            device=device,
            encode=encode,
        )
    )
    if not np.array_equal(query_labels, endpoint_query_labels) or not np.array_equal(
        gallery_labels, endpoint_gallery_labels
    ):
        raise ValueError("endpoint holdout labels differ")
    endpoint_scores = deployment_scores(endpoint_query, endpoint_gallery, selected_features=512)
    endpoint_reproduction = evaluate_blend_grid(
        endpoint_scores,
        endpoint_scores,
        query_labels,
        gallery_labels,
        alphas=(0.0,),
    )[0]
    validate_endpoint_reproduction(screen, endpoint_reproduction)

    epoch8_query, epoch8_gallery, epoch8_query_labels, epoch8_gallery_labels = (
        encode_recalibrated_arm(
            soup,
            model,
            epoch8_checkpoint["model"],
            calibration_loader,
            device=device,
            encode=encode,
        )
    )
    if not np.array_equal(query_labels, epoch8_query_labels) or not np.array_equal(
        gallery_labels, epoch8_gallery_labels
    ):
        raise ValueError("epoch-8 holdout labels differ")
    epoch8_scores = deployment_scores(epoch8_query, epoch8_gallery, selected_features=512)
    provenance = {
        "screen_report": str(args.screen_report),
        "screen_report_sha256": _sha256_file(args.screen_report),
        "endpoint_state": str(args.endpoint_state),
        "endpoint_state_sha256": _sha256_file(args.endpoint_state),
        "epoch8_checkpoint": str(args.epoch8_checkpoint),
        "epoch8_checkpoint_sha256": _sha256_file(args.epoch8_checkpoint),
        "initial_checkpoint_sha256": _sha256_file(args.initial_checkpoint),
        "batch_norm_recalibration": "full-optimization-cumulative-batches-all-arms",
        "training_protocol": screen["training_protocol"],
    }
    return (
        endpoint_scores,
        pretrained_scores,
        epoch8_scores,
        query_labels,
        gallery_labels,
        provenance,
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--screen-report", required=True, type=Path)
    parser.add_argument("--endpoint-state", required=True, type=Path)
    parser.add_argument("--epoch8-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    values = parser.parse_args(arguments)
    values.training_seed = 0
    return values


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        if args.output.exists() or args.output.is_symlink() or temporary.exists():
            raise FileExistsError(args.output)
        endpoint, pretrained, epoch8, query_labels, gallery_labels, provenance = (
            load_registered_scores(args)
        )
        report = evaluate_falsifier(
            endpoint,
            pretrained,
            epoch8,
            query_labels,
            gallery_labels,
            selected_features=512,
            provenance=provenance,
        )
        write_report_atomic(report, args.output)
    except Exception as error:
        print(f"FTRA falsifier failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["gate"], sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
