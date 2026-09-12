#!/usr/bin/env python3
"""Evaluate frozen positive-coverage adapters on the official SOP test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, cast

import torch
from torch.nn import functional as F

SEEDS = (0, 1, 2, 3, 4)


class SopOfficialSeedArtifact(NamedTuple):
    """Local receipt and checkpoints for one frozen training seed."""

    seed: int
    receipt: Path
    receipt_sha256: str
    pooled_checkpoint: Path
    coverage_checkpoint: Path


class FrozenPositiveCoveragePanel(NamedTuple):
    """Authenticated base head and five paired adapter states."""

    seeds: tuple[int, ...]
    base_weight: torch.Tensor
    base_bias: torch.Tensor
    pooled_weights: tuple[torch.Tensor, ...]
    coverage_weights: tuple[torch.Tensor, ...]


class SopOfficialArguments(NamedTuple):
    """Strict local inputs for one frozen official-test evaluation."""

    source_snapshot: Path
    source_sha256: str
    teacher_snapshot: Path
    teacher_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    seed_artifacts: tuple[SopOfficialSeedArtifact, ...]
    source_revision: str
    training_source_revision: str
    driver_sha256: str
    output: Path


class OfficialOutputReservation(NamedTuple):
    """Owned exclusive partial output held before official rows are opened."""

    path: Path
    descriptor: int
    device: int
    inode: int


def _lower_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not (set(value) - set("0123456789abcdef"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def reserved_official_output(output: Path) -> Iterator[OfficialOutputReservation]:
    """Reserve a unique official output and clean only the owned partial inode."""

    partial = output.with_name(f"{output.name}.partial")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    status = os.fstat(descriptor)
    reservation = OfficialOutputReservation(
        path=partial,
        descriptor=descriptor,
        device=status.st_dev,
        inode=status.st_ino,
    )
    try:
        yield reservation
    finally:
        os.close(descriptor)


def publish_reserved_official_output(reservation: OfficialOutputReservation, wire: bytes) -> None:
    """Publish canonical bytes from an already-owned reservation without replacement."""

    if type(reservation) is not OfficialOutputReservation or type(wire) is not bytes or not wire:
        raise ValueError("official output authority differs")
    status = os.fstat(reservation.descriptor)
    if (status.st_dev, status.st_ino) != (reservation.device, reservation.inode):
        raise ValueError("official output authority differs")
    os.lseek(reservation.descriptor, 0, os.SEEK_SET)
    os.ftruncate(reservation.descriptor, 0)
    written = 0
    while written < len(wire):
        written += os.write(reservation.descriptor, wire[written:])
    os.fsync(reservation.descriptor)
    output = reservation.path.with_name(reservation.path.name.removesuffix(".partial"))
    os.link(reservation.path, output)
    reservation.path.unlink()
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def parse_official_args(arguments: Sequence[str] | None = None) -> SopOfficialArguments:
    """Parse a local inference-only surface and reject undeclared capabilities."""

    parser = argparse.ArgumentParser(allow_abbrev=False, exit_on_error=False)
    for name in (
        "source-snapshot",
        "source-sha256",
        "teacher-snapshot",
        "teacher-sha256",
        "base-checkpoint",
        "base-checkpoint-sha256",
        "source-revision",
        "training-source-revision",
        "driver-sha256",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--seed-artifact",
        action="append",
        nargs=5,
        required=True,
        metavar=("SEED", "RECEIPT", "SHA256", "POOLED", "COVERAGE"),
    )
    parser.add_argument("--execute-official-evaluation", action="store_true", required=True)
    try:
        parsed, unknown = parser.parse_known_args(arguments)
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError("official argument authority differs") from error
    raw = list(arguments) if arguments is not None else []
    repeatable = "--seed-artifact"
    flags = [value.split("=", 1)[0] for value in raw if value.startswith("--")]
    if (
        unknown
        or any(flags.count(flag) > 1 for flag in set(flags) - {repeatable})
        or parsed.execute_official_evaluation is not True
    ):
        raise ValueError("official argument authority differs")
    source_snapshot = Path(parsed.source_snapshot)
    teacher_snapshot = Path(parsed.teacher_snapshot)
    base_checkpoint = Path(parsed.base_checkpoint)
    output = Path(parsed.output)
    if (
        any(
            not path.is_absolute() or not path.is_file()
            for path in (source_snapshot, teacher_snapshot, base_checkpoint)
        )
        or not output.is_absolute()
        or not output.parent.is_dir()
        or output.exists()
        or output.with_name(f"{output.name}.partial").exists()
        or any(
            not _lower_sha256(value)
            for value in (
                parsed.source_sha256,
                parsed.teacher_sha256,
                parsed.base_checkpoint_sha256,
                parsed.driver_sha256,
            )
        )
        or type(parsed.source_revision) is not str
        or len(parsed.source_revision) != 40
        or set(parsed.source_revision) - set("0123456789abcdef")
        or type(parsed.training_source_revision) is not str
        or len(parsed.training_source_revision) != 40
        or set(parsed.training_source_revision) - set("0123456789abcdef")
        or len(parsed.seed_artifact) != 5
    ):
        raise ValueError("official argument authority differs")
    seed_artifacts: list[SopOfficialSeedArtifact] = []
    for values in parsed.seed_artifact:
        seed_text, receipt_text, digest, pooled_text, coverage_text = values
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError("official argument authority differs") from error
        receipt = Path(receipt_text)
        pooled = Path(pooled_text)
        coverage = Path(coverage_text)
        if (
            str(seed) != seed_text
            or not _lower_sha256(digest)
            or any(
                not path.is_absolute() or not path.is_file() for path in (receipt, pooled, coverage)
            )
        ):
            raise ValueError("official argument authority differs")
        seed_artifacts.append(SopOfficialSeedArtifact(seed, receipt, digest, pooled, coverage))
    if tuple(artifact.seed for artifact in seed_artifacts) != SEEDS:
        raise ValueError("official argument authority differs")
    return SopOfficialArguments(
        source_snapshot,
        parsed.source_sha256,
        teacher_snapshot,
        parsed.teacher_sha256,
        base_checkpoint,
        parsed.base_checkpoint_sha256,
        tuple(seed_artifacts),
        parsed.source_revision,
        parsed.training_source_revision,
        parsed.driver_sha256,
        output,
    )


def parameter_sha256(*values: torch.Tensor) -> str:
    """Hash float32 parameters with explicit shape and byte order."""

    if not values:
        raise ValueError("official panel authority differs")
    digest = hashlib.sha256()
    for value in values:
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("official panel authority differs")
        array = value.numpy().astype("<f4", copy=False)
        digest.update(struct.pack("<I", array.ndim))
        digest.update(struct.pack(f"<{array.ndim}Q", *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_checkpoint(path: Path, expected: dict[str, object]) -> torch.Tensor:
    if (
        not isinstance(path, Path)
        or not path.is_file()
        or set(expected) != {"bytes", "sha256"}
        or type(expected["bytes"]) is not int
        or expected["bytes"] <= 0
        or not _lower_sha256(expected["sha256"])
        or path.stat().st_size != expected["bytes"]
        or _file_sha256(path) != expected["sha256"]
    ):
        raise ValueError("official panel authority differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if type(state) is not dict or tuple(state) != ("weight",):
        raise ValueError("official panel authority differs")
    weight = state["weight"]
    if (
        type(weight) is not torch.Tensor
        or weight.dtype != torch.float32
        or weight.device.type != "cpu"
        or weight.ndim != 2
        or weight.shape[0] != weight.shape[1]
        or not weight.is_contiguous()
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("official panel authority differs")
    return weight


def load_frozen_positive_coverage_panel(
    artifacts: tuple[SopOfficialSeedArtifact, ...],
    *,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
    source_revision: str,
) -> FrozenPositiveCoveragePanel:
    """Authenticate the frozen five-seed panel without evaluating test rows."""

    if (
        type(artifacts) is not tuple
        or tuple(artifact.seed for artifact in artifacts) != SEEDS
        or not isinstance(base_checkpoint, Path)
        or not base_checkpoint.is_file()
        or any(
            not _lower_sha256(value)
            for value in (
                base_checkpoint_sha256,
                source_snapshot_sha256,
                teacher_snapshot_sha256,
            )
        )
        or type(source_revision) is not str
        or len(source_revision) != 40
        or set(source_revision) - set("0123456789abcdef")
        or _file_sha256(base_checkpoint) != base_checkpoint_sha256
    ):
        raise ValueError("official panel authority differs")
    base_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    if type(base_state) is not dict or tuple(base_state) != ("weight", "bias"):
        raise ValueError("official panel authority differs")
    base_weight = base_state["weight"]
    base_bias = base_state["bias"]
    if (
        type(base_weight) is not torch.Tensor
        or type(base_bias) is not torch.Tensor
        or base_weight.shape != (128, 768)
        or base_bias.shape != (128,)
    ):
        raise ValueError("official panel authority differs")
    base_parameter = parameter_sha256(base_weight, base_bias)
    pooled: list[torch.Tensor] = []
    coverage: list[torch.Tensor] = []
    for expected_seed, artifact in zip(SEEDS, artifacts, strict=True):
        if (
            type(artifact) is not SopOfficialSeedArtifact
            or artifact.seed != expected_seed
            or not artifact.receipt.is_file()
            or not _lower_sha256(artifact.receipt_sha256)
            or _file_sha256(artifact.receipt) != artifact.receipt_sha256
        ):
            raise ValueError("official panel authority differs")
        wire = artifact.receipt.read_bytes()
        try:
            receipt = json.loads(wire)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("official panel authority differs") from error
        if (
            wire != (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            or type(receipt) is not dict
            or receipt.get("schema") != "sfora-positive-coverage-metric-screen-v2"
            or receipt.get("dataset") != "sop-official-train-class-disjoint-validation"
            or receipt.get("seed") != expected_seed
            or receipt.get("claim_eligible") is not False
            or receipt.get("official_test_touched") is not False
            or receipt.get("passes") is not True
            or receipt.get("source") is None
            or receipt["source"].get("source_revision") != source_revision
            or receipt.get("inputs")
            != {
                "source_snapshot_sha256": source_snapshot_sha256,
                "teacher_snapshot_sha256": teacher_snapshot_sha256,
            }
            or type(receipt.get("base")) is not dict
            or receipt["base"].get("checkpoint_sha256") != base_checkpoint_sha256
            or receipt["base"].get("parameter_sha256") != base_parameter
            or type(receipt.get("arms")) is not dict
            or set(receipt["arms"]) != {"pooled", "coverage"}
        ):
            raise ValueError("official panel authority differs")
        arm_values: list[torch.Tensor] = []
        for name, path in (
            ("pooled", artifact.pooled_checkpoint),
            ("coverage", artifact.coverage_checkpoint),
        ):
            arm = receipt["arms"][name]
            if type(arm) is not dict or type(arm.get("checkpoint")) is not dict:
                raise ValueError("official panel authority differs")
            weight = _load_checkpoint(path, arm["checkpoint"])
            if arm.get("parameter_sha256") != parameter_sha256(weight):
                raise ValueError("official panel authority differs")
            arm_values.append(weight)
        if any(weight.shape != (128, 128) for weight in arm_values):
            raise ValueError("official panel authority differs")
        pooled.append(arm_values[0])
        coverage.append(arm_values[1])
    return FrozenPositiveCoveragePanel(
        seeds=SEEDS,
        base_weight=base_weight,
        base_bias=base_bias,
        pooled_weights=tuple(pooled),
        coverage_weights=tuple(coverage),
    )


def official_panel_codes(
    teacher_rows: torch.Tensor, panel: FrozenPositiveCoveragePanel
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Apply the frozen base head and each adapter without training."""

    if (
        type(teacher_rows) is not torch.Tensor
        or teacher_rows.device.type != "cpu"
        or teacher_rows.dtype != torch.float32
        or teacher_rows.ndim != 2
        or teacher_rows.shape[1] != panel.base_weight.shape[1]
        or not teacher_rows.is_contiguous()
        or not bool(torch.isfinite(teacher_rows).all())
    ):
        raise ValueError("official panel authority differs")
    with torch.inference_mode():
        normalized_teacher = F.normalize(teacher_rows, dim=1).contiguous()
        base = F.normalize(
            F.linear(normalized_teacher, panel.base_weight, panel.base_bias), dim=1
        ).contiguous()
        pooled = tuple(
            F.normalize(F.linear(base, weight), dim=1).contiguous()
            for weight in panel.pooled_weights
        )
        coverage = tuple(
            F.normalize(F.linear(base, weight), dim=1).contiguous()
            for weight in panel.coverage_weights
        )
    return base, pooled, coverage


def official_result(
    *,
    seed_rows: Iterable[dict[str, object]],
    pooled_lower_bound: float,
    pooled_r1_lower_bound: float,
    base_score: dict[str, float],
    teacher_score: dict[str, float],
    input_authority: dict[str, str],
    training_source_revision: str,
    source_revision: str,
) -> dict[str, object]:
    """Build a recomputable official-test decision from five seed endpoints."""

    rows: tuple[dict[str, object], ...] = tuple(seed_rows)
    row_keys = {"seed", "pooled", "coverage"}
    score_keys = {"float_map_at_r", "float_r1", "packed_map_at_r", "packed_r1"}
    scores: list[object] = [base_score, teacher_score]
    for row in rows:
        scores.extend((row.get("pooled"), row.get("coverage")))
    if (
        len(rows) != 5
        or tuple(row.get("seed") for row in rows) != SEEDS
        or any(set(row) != row_keys for row in rows)
        or any(
            type(score) is not dict
            or set(score) != score_keys
            or any(
                type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in score.values()
            )
            for score in scores
        )
        or type(pooled_lower_bound) is not float
        or not math.isfinite(pooled_lower_bound)
        or type(pooled_r1_lower_bound) is not float
        or not math.isfinite(pooled_r1_lower_bound)
        or type(input_authority) is not dict
        or not input_authority
        or any(
            type(key) is not str or not key or not _lower_sha256(value)
            for key, value in input_authority.items()
        )
        or type(source_revision) is not str
        or len(source_revision) != 40
        or type(training_source_revision) is not str
        or len(training_source_revision) != 40
    ):
        raise ValueError("official result authority differs")
    pooled_mean = (
        math.fsum(cast(dict[str, float], row["pooled"])["packed_map_at_r"] for row in rows) / 5
    )
    coverage_mean = (
        math.fsum(cast(dict[str, float], row["coverage"])["packed_map_at_r"] for row in rows) / 5
    )
    pooled_r1 = math.fsum(cast(dict[str, float], row["pooled"])["packed_r1"] for row in rows) / 5
    coverage_r1 = (
        math.fsum(cast(dict[str, float], row["coverage"])["packed_r1"] for row in rows) / 5
    )
    passes = (
        coverage_mean > pooled_mean and pooled_lower_bound > 0.0 and pooled_r1_lower_bound >= -0.002
    )
    return {
        "base": base_score,
        "claim_eligible": False,
        "coverage_mean_packed_map_at_r": coverage_mean,
        "coverage_mean_packed_r1": coverage_r1,
        "coverage_minus_pooled_packed_map": coverage_mean - pooled_mean,
        "dataset": "stanford-online-products-official-test",
        "gates": {
            "coverage_minus_pooled_packed_map": 0.0,
            "coverage_minus_pooled_packed_r1": -0.002,
            "pooled_class_cluster_lower_bound": 0.0,
        },
        "inputs": input_authority,
        "official_test_touched": True,
        "passes": passes,
        "pooled_class_cluster_lower_bound": pooled_lower_bound,
        "pooled_class_cluster_r1_lower_bound": pooled_r1_lower_bound,
        "pooled_mean_packed_map_at_r": pooled_mean,
        "pooled_mean_packed_r1": pooled_r1,
        "schema": "sfora-positive-coverage-sop-official-v1",
        "seed_results": list(rows),
        "source_revision": source_revision,
        "teacher": teacher_score,
        "training_source_revision": training_source_revision,
    }


def canonical_official_result_bytes(result: dict[str, object]) -> bytes:
    """Validate and encode the official-test result as canonical JSON."""

    if type(result) is not dict:
        raise ValueError("official result authority differs")
    try:
        recomputed = official_result(
            seed_rows=cast(Iterable[dict[str, object]], result["seed_results"]),
            pooled_lower_bound=cast(float, result["pooled_class_cluster_lower_bound"]),
            pooled_r1_lower_bound=cast(float, result["pooled_class_cluster_r1_lower_bound"]),
            base_score=cast(dict[str, float], result["base"]),
            teacher_score=cast(dict[str, float], result["teacher"]),
            input_authority=cast(dict[str, str], result["inputs"]),
            training_source_revision=cast(str, result["training_source_revision"]),
            source_revision=cast(str, result["source_revision"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("official result authority differs") from error
    if result != recomputed:
        raise ValueError("official result authority differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _scalar_score(score: dict[str, object]) -> dict[str, float]:
    keys = ("float_map_at_r", "float_r1", "packed_map_at_r", "packed_r1")
    if any(type(score.get(key)) is not float for key in keys):
        raise ValueError("official score authority differs")
    result = {key: cast(float, score[key]) for key in keys}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("official score authority differs")
    return result


def run_official(args: SopOfficialArguments) -> dict[str, object]:
    """Authenticate the panel, evaluate official rows once, and return evidence."""

    from positive_coverage_artifacts import positive_coverage_source_identity
    from probe_representation_ceiling import class_cluster_lower_bound
    from probe_sop_relational_linear import load_paired_archives, score_symmetric

    from sfora.deterministic_similarity_runtime import (
        configure_deterministic_similarity_runtime,
    )
    from sfora.joint_relational_compaction import pack_int8_unit_embeddings

    positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pair = load_paired_archives(
        args.source_snapshot,
        args.source_sha256,
        args.teacher_snapshot,
        args.teacher_sha256,
    )
    panel = load_frozen_positive_coverage_panel(
        args.seed_artifacts,
        base_checkpoint=args.base_checkpoint,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
        source_snapshot_sha256=args.source_sha256,
        teacher_snapshot_sha256=args.teacher_sha256,
        source_revision=args.training_source_revision,
    )
    teacher_test = pair["teacher_test"].float().contiguous()
    labels = pair["test_labels"]
    base_codes, pooled_codes, coverage_codes = official_panel_codes(teacher_test, panel)
    device = torch.device("cuda")

    def score(codes: torch.Tensor) -> dict[str, object]:
        candidate_width = max(Counter(labels).values()) - 1
        floating = score_symmetric(codes, labels, candidate_width=candidate_width, device=device)
        packed = score_symmetric(
            pack_int8_unit_embeddings(codes),
            labels,
            candidate_width=candidate_width,
            device=device,
        )
        return {
            "float_map_at_r": float(floating["map_at_r"]),
            "float_r1": float(floating["r1"]),
            "packed_map_at_r": float(packed["map_at_r"]),
            "packed_per_query_ap": [float(value) for value in packed["per_query_ap"]],
            "packed_per_query_r1": [float(value) for value in packed["per_query_r1"]],
            "packed_r1": float(packed["r1"]),
        }

    base_full = score(base_codes)
    teacher_full = score(F.normalize(teacher_test, dim=1).contiguous())
    seed_rows: list[dict[str, object]] = []
    pooled_ap: list[tuple[float, ...]] = []
    coverage_ap: list[tuple[float, ...]] = []
    pooled_r1_rows: list[tuple[float, ...]] = []
    coverage_r1_rows: list[tuple[float, ...]] = []
    for seed, pooled_codes_seed, coverage_codes_seed in zip(
        panel.seeds, pooled_codes, coverage_codes, strict=True
    ):
        pooled_full = score(pooled_codes_seed)
        coverage_full = score(coverage_codes_seed)
        pooled_ap.append(tuple(cast(list[float], pooled_full["packed_per_query_ap"])))
        coverage_ap.append(tuple(cast(list[float], coverage_full["packed_per_query_ap"])))
        pooled_r1_rows.append(tuple(cast(list[float], pooled_full["packed_per_query_r1"])))
        coverage_r1_rows.append(tuple(cast(list[float], coverage_full["packed_per_query_r1"])))
        seed_rows.append(
            {
                "seed": seed,
                "pooled": _scalar_score(pooled_full),
                "coverage": _scalar_score(coverage_full),
            }
        )
    pooled_mean_ap = tuple(
        math.fsum(values) / len(values) for values in zip(*pooled_ap, strict=True)
    )
    coverage_mean_ap = tuple(
        math.fsum(values) / len(values) for values in zip(*coverage_ap, strict=True)
    )
    pooled_mean_r1 = tuple(
        math.fsum(values) / len(values) for values in zip(*pooled_r1_rows, strict=True)
    )
    coverage_mean_r1 = tuple(
        math.fsum(values) / len(values) for values in zip(*coverage_r1_rows, strict=True)
    )
    lower_bound = class_cluster_lower_bound(
        coverage_mean_ap,
        pooled_mean_ap,
        labels,
        seed=17,
        samples=10_000,
    )
    r1_lower_bound = class_cluster_lower_bound(
        coverage_mean_r1,
        pooled_mean_r1,
        labels,
        seed=17,
        samples=10_000,
    )
    input_authority = {
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "driver_sha256": args.driver_sha256,
        "source_snapshot_sha256": args.source_sha256,
        "teacher_snapshot_sha256": args.teacher_sha256,
    }
    for artifact in args.seed_artifacts:
        receipt = json.loads(artifact.receipt.read_bytes())
        input_authority[f"seed{artifact.seed}_receipt_sha256"] = artifact.receipt_sha256
        for name, path in (
            ("pooled", artifact.pooled_checkpoint),
            ("coverage", artifact.coverage_checkpoint),
        ):
            input_authority[f"seed{artifact.seed}_{name}_checkpoint_sha256"] = receipt["arms"][
                name
            ]["checkpoint"]["sha256"]
            if (
                _file_sha256(path)
                != input_authority[f"seed{artifact.seed}_{name}_checkpoint_sha256"]
            ):
                raise ValueError("official panel authority differs")
    return official_result(
        seed_rows=seed_rows,
        pooled_lower_bound=lower_bound,
        pooled_r1_lower_bound=r1_lower_bound,
        base_score=_scalar_score(base_full),
        teacher_score=_scalar_score(teacher_full),
        input_authority=input_authority,
        training_source_revision=args.training_source_revision,
        source_revision=args.source_revision,
    )


def main() -> None:
    try:
        args = parse_official_args(sys.argv[1:])
    except ValueError as error:
        print(
            f"official evaluation refused: {error}; --execute-official-evaluation is required",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    with reserved_official_output(args.output) as reservation:
        result = run_official(args)
        publish_reserved_official_output(reservation, canonical_official_result_bytes(result))


if __name__ == "__main__":
    main()
