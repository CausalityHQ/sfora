#!/usr/bin/env python3
"""Strict local evaluator for teacher-anchored SOP experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import stat
import struct
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import torch
from torch import nn

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.teacher_anchored_distillation import teacher_anchored_forward


class TeacherAnchoredEvaluationEvidence(NamedTuple):
    """Exact float32 and deployed symmetric-int8 retrieval evidence."""

    candidate_width: int
    float_map_at_r: float
    float_r1: float
    float_per_query_ap: tuple[float, ...]
    float_per_query_r1: tuple[float, ...]
    packed_map_at_r: float
    packed_r1: float
    packed_per_query_ap: tuple[float, ...]
    packed_per_query_r1: tuple[float, ...]


class TeacherAnchoredCandidateEvidence(NamedTuple):
    """One registered control or trained-arm endpoint."""

    arm: str
    representation: str
    candidate_epoch: int | None
    stopped_reason: str | None
    map_at_r: float | None
    r1: float | None
    per_query_ap: tuple[float, ...] | None


class TeacherAnchoredAdvancementEvidence(NamedTuple):
    """Recomputed gates and one exhaustive seed-level outcome."""

    outcome: str
    complete_step_zero_map_delta: float | None
    complete_base_map_delta: float | None
    complete_step_zero_bootstrap_lower: float | None
    complete_base_bootstrap_lower: float | None
    gates: tuple[bool, ...]


class TeacherAnchoredServingReconstruction(NamedTuple):
    """Fresh-model serving codes and their authenticated state evidence."""

    codes: torch.Tensor
    rows: int
    dimensions: int
    batch_rows: tuple[int, ...]
    checkpoint_sha256: str
    codes_sha256: str
    expected_codes_sha256: str
    maximum_batch_shape_error: float
    minimum_batch_shape_cosine: float


def load_teacher_anchored_evaluation_scorer() -> tuple[Callable[..., object], Path]:
    """Load the registered repository scorer with sibling imports enabled."""

    path = Path(__file__).resolve().parent / "probe_sop_relational_linear.py"
    scripts_directory = str(path.parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    module_name = "sfora_teacher_anchored_evaluation_scorer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored evaluation scorer authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    scorer = getattr(module, "score_symmetric", None)
    if not callable(scorer):
        raise ValueError("teacher-anchored evaluation scorer authority differs")
    return cast(Callable[..., object], scorer), path


def _load_teacher_anchored_bootstrap() -> Callable[..., object]:
    path = Path(__file__).resolve().parent / "probe_representation_ceiling.py"
    scripts_directory = str(path.parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    module_name = "sfora_teacher_anchored_evaluation_bootstrap"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored advancement authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    estimator = getattr(module, "class_cluster_lower_bound", None)
    if not callable(estimator):
        raise ValueError("teacher-anchored advancement authority differs")
    return cast(Callable[..., object], estimator)


def score_teacher_anchored_evaluation(
    codes: torch.Tensor,
    labels: tuple[int, ...],
    *,
    device: torch.device,
) -> TeacherAnchoredEvaluationEvidence:
    """Score one normalized code matrix through float and serving paths."""

    if (
        type(codes) is not torch.Tensor
        or codes.device.type != "cpu"
        or codes.dtype != torch.float32
        or codes.ndim != 2
        or codes.shape[0] < 2
        or codes.shape[1] < 2
        or not codes.is_contiguous()
        or not bool(torch.isfinite(codes).all())
        or type(labels) is not tuple
        or len(labels) != len(codes)
        or any(type(label) is not int or not 1 <= label < 2**63 for label in labels)
        or type(device) is not torch.device
    ):
        raise ValueError("teacher-anchored evaluation authority differs")
    norms = torch.linalg.vector_norm(codes.double(), dim=1)
    counts = Counter(labels)
    if (
        not bool((torch.abs(norms - 1.0) <= 2e-5).all())
        or not counts
        or any(count < 2 for count in counts.values())
    ):
        raise ValueError("teacher-anchored evaluation authority differs")
    candidate_width = max(counts.values()) - 1
    scorer, _path = load_teacher_anchored_evaluation_scorer()
    floating = scorer(codes, labels, candidate_width=candidate_width, device=device)
    packed = scorer(
        pack_int8_unit_embeddings(codes),
        labels,
        candidate_width=candidate_width,
        device=device,
    )
    if type(floating) is not dict or type(packed) is not dict:
        raise ValueError("teacher-anchored evaluation authority differs")
    scalar_keys = ("map_at_r", "r1")
    vector_keys = ("per_query_ap", "per_query_r1")
    if any(
        type(result.get(key)) is not float or not math.isfinite(result[key])
        for result in (floating, packed)
        for key in scalar_keys
    ) or any(
        type(result.get(key)) is not tuple
        or len(result[key]) != len(codes)
        or any(type(value) is not float or not math.isfinite(value) for value in result[key])
        for result in (floating, packed)
        for key in vector_keys
    ):
        raise ValueError("teacher-anchored evaluation authority differs")
    return TeacherAnchoredEvaluationEvidence(
        candidate_width=candidate_width,
        float_map_at_r=cast(float, floating["map_at_r"]),
        float_r1=cast(float, floating["r1"]),
        float_per_query_ap=cast(tuple[float, ...], floating["per_query_ap"]),
        float_per_query_r1=cast(tuple[float, ...], floating["per_query_r1"]),
        packed_map_at_r=cast(float, packed["map_at_r"]),
        packed_r1=cast(float, packed["r1"]),
        packed_per_query_ap=cast(tuple[float, ...], packed["per_query_ap"]),
        packed_per_query_r1=cast(tuple[float, ...], packed["per_query_r1"]),
    )


def classify_teacher_anchored_advancement(
    *,
    source: TeacherAnchoredCandidateEvidence | None,
    teacher_pca: TeacherAnchoredCandidateEvidence | None,
    step_zero: TeacherAnchoredCandidateEvidence | None,
    base: TeacherAnchoredCandidateEvidence | None,
    complete: TeacherAnchoredCandidateEvidence | None,
    labels: tuple[int, ...],
) -> TeacherAnchoredAdvancementEvidence:
    """Recompute every seed-level quality and causal advancement gate."""

    if (
        type(labels) is not tuple
        or len(labels) < 2
        or any(type(label) is not int or not 1 <= label < 2**63 for label in labels)
    ):
        raise ValueError("teacher-anchored advancement authority differs")
    candidates = (source, teacher_pca, step_zero, base, complete)
    roles = ("source", "teacher-pca", "step-zero", "base", "complete")
    representations = (
        "float32",
        "symmetric-int8",
        "symmetric-int8",
        "symmetric-int8",
        "symmetric-int8",
    )
    stop_reasons = {
        "nonfinite-update",
        "epoch-one-fitting-map-regression",
        "epoch-one-effective-rank-collapse",
        "epoch-one-leading-eigenvalue-collapse",
    }
    for candidate, role, representation, expected_epoch in zip(
        candidates,
        roles,
        representations,
        (None, None, 0, 10, 10),
        strict=True,
    ):
        if candidate is None:
            continue
        if (
            type(candidate) is not TeacherAnchoredCandidateEvidence
            or candidate.arm != role
            or candidate.representation != representation
        ):
            raise ValueError("teacher-anchored advancement authority differs")
        stopped = candidate.stopped_reason is not None
        if stopped:
            if (
                role in roles[:3]
                or candidate.stopped_reason not in stop_reasons
                or candidate.candidate_epoch is not None
                or candidate.map_at_r is not None
                or candidate.r1 is not None
                or candidate.per_query_ap is not None
            ):
                raise ValueError("teacher-anchored advancement authority differs")
            continue
        if (
            candidate.candidate_epoch != expected_epoch
            or type(candidate.map_at_r) is not float
            or type(candidate.r1) is not float
            or not math.isfinite(candidate.map_at_r)
            or not math.isfinite(candidate.r1)
            or not 0.0 <= candidate.map_at_r <= 1.0
            or not 0.0 <= candidate.r1 <= 1.0
            or type(candidate.per_query_ap) is not tuple
            or len(candidate.per_query_ap) != len(labels)
            or any(
                type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in candidate.per_query_ap
            )
            or math.fsum(candidate.per_query_ap) / len(labels) != candidate.map_at_r
        ):
            raise ValueError("teacher-anchored advancement authority differs")
    if source is None or teacher_pca is None or step_zero is None or complete is None:
        return TeacherAnchoredAdvancementEvidence(
            "implementation-error", None, None, None, None, ()
        )
    if base is None:
        return TeacherAnchoredAdvancementEvidence("inconclusive", None, None, None, None, ())
    if base.stopped_reason is not None:
        return TeacherAnchoredAdvancementEvidence("inconclusive", None, None, None, None, ())
    if complete.stopped_reason is not None:
        return TeacherAnchoredAdvancementEvidence("stopped", None, None, None, None, ())

    complete_map = cast(float, complete.map_at_r)
    complete_r1 = cast(float, complete.r1)
    complete_ap = cast(tuple[float, ...], complete.per_query_ap)
    source_map = cast(float, source.map_at_r)
    source_r1 = cast(float, source.r1)
    teacher_map = cast(float, teacher_pca.map_at_r)
    step_map = cast(float, step_zero.map_at_r)
    step_ap = cast(tuple[float, ...], step_zero.per_query_ap)
    base_map = cast(float, base.map_at_r)
    base_ap = cast(tuple[float, ...], base.per_query_ap)
    step_delta = complete_map - step_map
    base_delta = complete_map - base_map
    estimator = _load_teacher_anchored_bootstrap()
    step_lower = estimator(
        complete_ap,
        step_ap,
        labels,
        seed=17,
        samples=10_000,
    )
    base_lower = estimator(
        complete_ap,
        base_ap,
        labels,
        seed=17,
        samples=10_000,
    )
    if (
        type(step_lower) is not float
        or type(base_lower) is not float
        or not math.isfinite(step_lower)
        or not math.isfinite(base_lower)
    ):
        raise ValueError("teacher-anchored advancement authority differs")
    gates = (
        step_delta >= 0.015,
        step_lower > 0.0,
        complete_map >= source_map,
        complete_map >= teacher_map - 0.015,
        complete_r1 >= source_r1 - 0.002,
        base_delta >= 0.003,
        base_lower > 0.0,
    )
    if all(gates):
        outcome = "stability-warranted"
    elif all(gates[:5]):
        outcome = "generic-anchored-adaptation"
    else:
        outcome = "quality-rejected"
    return TeacherAnchoredAdvancementEvidence(
        outcome,
        step_delta,
        base_delta,
        step_lower,
        base_lower,
        gates,
    )


def reconstruct_teacher_anchored_serving(
    encoder: nn.Module,
    head: nn.Linear,
    checkpoint: Path,
    batches: tuple[torch.Tensor, ...],
    *,
    expected_checkpoint_sha256: str,
    expected_codes: torch.Tensor,
    expected_input_shape: tuple[int, ...],
    device: torch.device,
) -> TeacherAnchoredServingReconstruction:
    """Load a merged student-only checkpoint into fresh models and encode batches."""

    if (
        not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or head.out_features != 128
        or not isinstance(checkpoint, Path)
        or not checkpoint.is_absolute()
        or not checkpoint.exists()
        or checkpoint.is_symlink()
        or not stat.S_ISREG(checkpoint.stat().st_mode)
        or not _is_sha256(expected_checkpoint_sha256)
        or type(batches) is not tuple
        or not batches
        or type(expected_codes) is not torch.Tensor
        or expected_codes.device.type != "cpu"
        or expected_codes.dtype != torch.float32
        or expected_codes.ndim != 2
        or expected_codes.shape[1] != 128
        or not expected_codes.is_contiguous()
        or not bool(torch.isfinite(expected_codes).all())
        or type(expected_input_shape) is not tuple
        or not expected_input_shape
        or any(type(value) is not int or value < 1 for value in expected_input_shape)
        or type(device) is not torch.device
        or (device.type == "cuda" and not torch.cuda.is_available())
        or device.type not in ("cpu", "cuda")
    ):
        raise ValueError("teacher-anchored serving authority differs")
    trailing_shape: tuple[int, ...] | None = None
    for batch in batches:
        if (
            type(batch) is not torch.Tensor
            or batch.device.type != "cpu"
            or batch.dtype != torch.float32
            or batch.ndim < 2
            or not 1 <= len(batch) <= 256
            or not batch.is_contiguous()
            or not bool(torch.isfinite(batch).all())
        ):
            raise ValueError("teacher-anchored serving authority differs")
        shape = tuple(batch.shape[1:])
        if shape != expected_input_shape:
            raise ValueError("teacher-anchored serving authority differs")
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("teacher-anchored serving authority differs")
    if expected_codes.shape[0] != sum(len(batch) for batch in batches):
        raise ValueError("teacher-anchored serving authority differs")

    checkpoint_digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checkpoint_digest.update(chunk)
    if checkpoint_digest.hexdigest() != expected_checkpoint_sha256:
        raise ValueError("teacher-anchored serving authority differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder_state = encoder.state_dict()
    head_state = head.state_dict()
    expected_keys = {
        *(f"encoder.{name}" for name in encoder_state),
        *(f"head.{name}" for name in head_state),
    }
    if type(state) is not dict or set(state) != expected_keys:
        raise ValueError("teacher-anchored serving authority differs")
    for prefix, expected in (("encoder", encoder_state), ("head", head_state)):
        for name, reference in expected.items():
            value = state[f"{prefix}.{name}"]
            if (
                type(value) is not torch.Tensor
                or value.device.type != "cpu"
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("teacher-anchored serving authority differs")
    encoder.load_state_dict({name: state[f"encoder.{name}"] for name in encoder_state}, strict=True)
    head.load_state_dict({name: state[f"head.{name}"] for name in head_state}, strict=True)
    encoder.to(device).eval()
    head.to(device).eval()
    if any(
        module.training
        for module in encoder.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        or type(module).__name__ == "DropPath"
    ):
        raise ValueError("teacher-anchored serving authority differs")
    chunks: list[torch.Tensor] = []
    single_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in batches:
            _features, raw = teacher_anchored_forward(encoder, head, batch.to(device))
            if (
                raw.ndim != 2
                or raw.shape != (len(batch), 128)
                or not bool(torch.isfinite(raw).all())
            ):
                raise ValueError("teacher-anchored serving authority differs")
            norms = torch.linalg.vector_norm(raw.double(), dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
                raise ValueError("teacher-anchored serving authority differs")
            chunks.append(raw.cpu().contiguous())
            for row in batch:
                _single_features, single_raw = teacher_anchored_forward(
                    encoder, head, row.unsqueeze(0).to(device)
                )
                if not bool(torch.isfinite(single_raw).all()):
                    raise ValueError("teacher-anchored serving authority differs")
                single_chunks.append(single_raw.cpu().contiguous())
    codes = torch.cat(chunks).contiguous()
    singles = torch.cat(single_chunks).contiguous()
    maximum_batch_shape_error = float(torch.max(torch.abs(codes - singles)))
    batch_shape_cosines = torch.sum(codes.double() * singles.double(), dim=1) / (
        torch.linalg.vector_norm(codes.double(), dim=1)
        * torch.linalg.vector_norm(singles.double(), dim=1)
    )
    minimum_batch_shape_cosine = float(torch.min(batch_shape_cosines))
    if (
        not math.isfinite(maximum_batch_shape_error)
        or not math.isfinite(minimum_batch_shape_cosine)
        or maximum_batch_shape_error > 0.002
        or minimum_batch_shape_cosine < 1.0 - 1e-5
        or not torch.equal(codes, expected_codes)
    ):
        raise ValueError("teacher-anchored serving authority differs")
    return TeacherAnchoredServingReconstruction(
        codes=codes,
        rows=len(codes),
        dimensions=codes.shape[1],
        batch_rows=tuple(len(batch) for batch in batches),
        checkpoint_sha256=expected_checkpoint_sha256,
        codes_sha256=_codes_sha256(codes),
        expected_codes_sha256=_codes_sha256(expected_codes),
        maximum_batch_shape_error=maximum_batch_shape_error,
        minimum_batch_shape_cosine=minimum_batch_shape_cosine,
    )


def canonical_teacher_anchored_evaluation_bytes(
    *,
    seed: int,
    source: TeacherAnchoredCandidateEvidence | None,
    teacher_pca: TeacherAnchoredCandidateEvidence | None,
    step_zero: TeacherAnchoredCandidateEvidence | None,
    base: TeacherAnchoredCandidateEvidence | None,
    complete: TeacherAnchoredCandidateEvidence | None,
    labels: tuple[int, ...],
    serving: TeacherAnchoredServingReconstruction | None,
) -> bytes:
    """Recompute gates and encode one claim-ineligible validation receipt."""

    if type(seed) is not int or seed not in (17, 1729, 65537):
        raise ValueError("teacher-anchored result authority differs")
    try:
        advancement = classify_teacher_anchored_advancement(
            source=source,
            teacher_pca=teacher_pca,
            step_zero=step_zero,
            base=base,
            complete=complete,
            labels=labels,
        )
    except ValueError as error:
        raise ValueError("teacher-anchored result authority differs") from error
    if advancement.outcome in ("stopped", "inconclusive", "implementation-error"):
        if serving is not None:
            raise ValueError("teacher-anchored result authority differs")
        serving_value = None
    else:
        if (
            type(serving) is not TeacherAnchoredServingReconstruction
            or type(serving.codes) is not torch.Tensor
            or serving.codes.device.type != "cpu"
            or serving.codes.dtype != torch.float32
            or serving.codes.ndim != 2
            or not serving.codes.is_contiguous()
            or serving.rows != len(serving.codes)
            or serving.rows != len(labels)
            or serving.dimensions != 128
            or serving.codes.shape != (serving.rows, serving.dimensions)
            or type(serving.batch_rows) is not tuple
            or not serving.batch_rows
            or any(type(rows) is not int or not 1 <= rows <= 256 for rows in serving.batch_rows)
            or sum(serving.batch_rows) != serving.rows
            or not _is_sha256(serving.checkpoint_sha256)
            or not _is_sha256(serving.codes_sha256)
            or not _is_sha256(serving.expected_codes_sha256)
            or serving.expected_codes_sha256 != serving.codes_sha256
            or type(serving.maximum_batch_shape_error) is not float
            or not math.isfinite(serving.maximum_batch_shape_error)
            or not 0.0 <= serving.maximum_batch_shape_error <= 0.002
            or type(serving.minimum_batch_shape_cosine) is not float
            or not math.isfinite(serving.minimum_batch_shape_cosine)
            or not 1.0 - 1e-5 <= serving.minimum_batch_shape_cosine <= 1.0
            or not bool(torch.isfinite(serving.codes).all())
        ):
            raise ValueError("teacher-anchored result authority differs")
        norms = torch.linalg.vector_norm(serving.codes.double(), dim=1)
        if (
            not bool((torch.abs(norms - 1.0) <= 2e-5).all())
            or _codes_sha256(serving.codes) != serving.codes_sha256
        ):
            raise ValueError("teacher-anchored result authority differs")
        try:
            serving_score = score_teacher_anchored_evaluation(
                serving.codes,
                labels,
                device=torch.device("cpu"),
            )
        except ValueError as error:
            raise ValueError("teacher-anchored result authority differs") from error
        if (
            complete is None
            or complete.map_at_r != serving_score.packed_map_at_r
            or complete.r1 != serving_score.packed_r1
            or complete.per_query_ap != serving_score.packed_per_query_ap
        ):
            raise ValueError("teacher-anchored result authority differs")
        serving_value = {
            "batch_rows": list(serving.batch_rows),
            "checkpoint_sha256": serving.checkpoint_sha256,
            "codes_sha256": serving.codes_sha256,
            "dimensions": serving.dimensions,
            "expected_codes_sha256": serving.expected_codes_sha256,
            "maximum_batch_shape_error": serving.maximum_batch_shape_error,
            "minimum_batch_shape_cosine": serving.minimum_batch_shape_cosine,
            "rows": serving.rows,
        }
    label_digest = hashlib.sha256()
    label_digest.update(struct.pack(f"<{len(labels)}q", *labels))
    candidates = {
        "source": source,
        "teacher-pca": teacher_pca,
        "step-zero": step_zero,
        "base": base,
        "complete": complete,
    }
    value = {
        "advancement": {
            "complete_base_bootstrap_lower": advancement.complete_base_bootstrap_lower,
            "complete_base_map_delta": advancement.complete_base_map_delta,
            "complete_step_zero_bootstrap_lower": (advancement.complete_step_zero_bootstrap_lower),
            "complete_step_zero_map_delta": advancement.complete_step_zero_map_delta,
            "gates": list(advancement.gates),
            "outcome": advancement.outcome,
        },
        "arms": {
            role: None
            if candidate is None
            else {
                "candidate_epoch": candidate.candidate_epoch,
                "map_at_r": candidate.map_at_r,
                "per_query_ap": (
                    None if candidate.per_query_ap is None else list(candidate.per_query_ap)
                ),
                "r1": candidate.r1,
                "representation": candidate.representation,
                "stopped_reason": candidate.stopped_reason,
            }
            for role, candidate in candidates.items()
        },
        "claim_eligible": False,
        "labels_sha256": label_digest.hexdigest(),
        "partition": "class-disjoint-training-validation",
        "schema": "sfora-teacher-anchored-evaluation-v1",
        "seed": seed,
        "serving": serving_value,
    }
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("teacher-anchored result authority differs") from error


def _codes_sha256(codes: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", codes.ndim))
    digest.update(struct.pack(f"<{codes.ndim}Q", *codes.shape))
    digest.update(codes.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    """Refuse execution until causal gates and serving reconstruction are complete."""

    raise SystemExit("teacher-anchored evaluator is not yet enabled")


if __name__ == "__main__":
    main()
