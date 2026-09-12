#!/usr/bin/env python3
"""Strict local evaluator for teacher-anchored SOP experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
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

_UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"


class TeacherAnchoredEvaluationArguments(NamedTuple):
    """Authenticated local-only inputs for one seed-level panel evaluation."""

    panel_receipt: Path
    panel_receipt_sha256: str
    source_checkpoint: Path
    source_checkpoint_sha256: str
    source_snapshot: Path
    source_snapshot_sha256: str
    teacher_snapshot: Path
    teacher_snapshot_sha256: str
    unicom_checkout: Path
    image_root: Path
    image_tree_sha256: str
    ceiling_receipt: Path
    ceiling_receipt_sha256: str
    teacher_pca_sha256: str
    source_revision: str
    seed: int
    output: Path
    execute_teacher_anchored_evaluation: bool


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
    retained_batch_rows: tuple[int, ...]
    scoring_device: str
    checkpoint_sha256: str
    codes_sha256: str
    expected_codes_sha256: str
    maximum_batch_shape_error: float
    minimum_batch_shape_cosine: float


def parse_teacher_anchored_evaluation_args(
    arguments: list[str],
) -> TeacherAnchoredEvaluationArguments:
    """Parse the fail-closed local evaluation capability without running models."""

    if type(arguments) is not list or any(type(argument) is not str for argument in arguments):
        raise ValueError("unsupported argument")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    input_names = (
        "panel-receipt",
        "source-checkpoint",
        "source-snapshot",
        "teacher-snapshot",
        "unicom-checkout",
        "image-root",
        "ceiling-receipt",
    )
    for name in input_names:
        parser.add_argument(f"--{name}")
    parser.add_argument("--output")
    digest_names = (
        "panel-receipt-sha256",
        "source-checkpoint-sha256",
        "source-snapshot-sha256",
        "teacher-snapshot-sha256",
        "image-tree-sha256",
        "ceiling-receipt-sha256",
        "teacher-pca-sha256",
    )
    for name in (*digest_names, "source-revision", "seed"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--execute-teacher-anchored-evaluation", action="store_true")
    known_flags = {action.option_strings[0] for action in parser._actions if action.option_strings}
    presented = [argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")]
    if any(presented.count(flag) > 1 for flag in known_flags):
        raise ValueError("unsupported argument")
    try:
        parsed, unknown = parser.parse_known_args(arguments)
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError("unsupported argument") from error
    if unknown:
        raise ValueError("unsupported argument")
    paths: dict[str, Path] = {}
    for name in input_names:
        attribute = name.replace("-", "_")
        raw = getattr(parsed, attribute)
        path = Path(raw) if type(raw) is str else Path()
        directory = name in ("unicom-checkout", "image-root")
        if (
            type(raw) is not str
            or not path.is_absolute()
            or not path.exists()
            or (directory and not path.is_dir())
            or (not directory and not path.is_file())
            or (
                name in ("panel-receipt", "ceiling-receipt", "unicom-checkout")
                and path.is_symlink()
            )
        ):
            raise ValueError("teacher-anchored evaluation input authority differs")
        paths[attribute] = path
    raw_output = parsed.output
    output = Path(raw_output) if type(raw_output) is str else Path()
    if (
        type(raw_output) is not str
        or not output.is_absolute()
        or not output.parent.is_dir()
        or output.exists()
    ):
        raise ValueError("teacher-anchored evaluation output authority differs")
    digests: dict[str, str] = {}
    for name in digest_names:
        attribute = name.replace("-", "_")
        value = getattr(parsed, attribute)
        if not _is_sha256(value):
            raise ValueError("teacher-anchored evaluation SHA-256 authority differs")
        digests[attribute] = value
    if parsed.source_revision != _UNICOM_REVISION:
        raise ValueError("teacher-anchored evaluation revision authority differs")
    try:
        seed = int(parsed.seed)
    except (TypeError, ValueError) as error:
        raise ValueError("teacher-anchored evaluation seed authority differs") from error
    if type(parsed.seed) is not str or str(seed) != parsed.seed or seed not in (17, 1729, 65537):
        raise ValueError("teacher-anchored evaluation seed authority differs")
    if parsed.execute_teacher_anchored_evaluation is not True:
        raise ValueError("teacher-anchored evaluation flag is required")
    return TeacherAnchoredEvaluationArguments(
        panel_receipt=paths["panel_receipt"],
        panel_receipt_sha256=digests["panel_receipt_sha256"],
        source_checkpoint=paths["source_checkpoint"],
        source_checkpoint_sha256=digests["source_checkpoint_sha256"],
        source_snapshot=paths["source_snapshot"],
        source_snapshot_sha256=digests["source_snapshot_sha256"],
        teacher_snapshot=paths["teacher_snapshot"],
        teacher_snapshot_sha256=digests["teacher_snapshot_sha256"],
        unicom_checkout=paths["unicom_checkout"],
        image_root=paths["image_root"],
        image_tree_sha256=digests["image_tree_sha256"],
        ceiling_receipt=paths["ceiling_receipt"],
        ceiling_receipt_sha256=digests["ceiling_receipt_sha256"],
        teacher_pca_sha256=digests["teacher_pca_sha256"],
        source_revision=parsed.source_revision,
        seed=seed,
        output=output,
        execute_teacher_anchored_evaluation=True,
    )


def validate_teacher_anchored_cross_arm_authority(
    results: dict[str, dict[str, object]],
) -> None:
    """Require all completed causal arms to share one experiment authority."""

    arms = ("head-only", "base", "anchor", "symmetric", "complete")
    authority_keys = {
        "anchor_schedule_sha256",
        "arm",
        "batch_schedule_sha256",
        "fitting_probe_sha256",
        "head_replay_sha256",
        "inputs_sha256",
        "model_mode_sha256",
        "objective_sha256",
        "ridge_sha256",
        "runtime_sha256",
        "seed",
        "snapshot_replay_sha256",
        "source_revision",
        "split_sha256",
        "teacher_pca_sha256",
        "trainable_inventory_sha256",
    }
    input_keys = {
        "ceiling_receipt",
        "image_tree",
        "launch_receipt",
        "schedule",
        "source_checkpoint",
        "source_snapshot",
        "teacher_checkpoint",
        "teacher_snapshot",
    }
    common_authority_keys = (
        "anchor_schedule_sha256",
        "batch_schedule_sha256",
        "fitting_probe_sha256",
        "ridge_sha256",
        "runtime_sha256",
        "seed",
        "snapshot_replay_sha256",
        "source_revision",
        "split_sha256",
        "teacher_pca_sha256",
    )
    common_result_keys = (
        "attempted_updates",
        "initial_encoder_sha256",
        "initial_frozen_sha256",
        "initial_head_sha256",
        "schedule_sha256",
        "successful_updates",
    )
    if type(results) is not dict or set(results) != set(arms):
        raise ValueError("teacher-anchored cross-arm authority differs")
    baseline_result: dict[str, object] | None = None
    baseline_authority: dict[str, object] | None = None
    baseline_inputs: dict[str, object] | None = None
    launch_digests: set[str] = set()
    for arm in arms:
        result = results[arm]
        authority = result.get("authority") if type(result) is dict else None
        inputs = authority.get("inputs_sha256") if type(authority) is dict else None
        if (
            type(result) is not dict
            or result.get("arm") != arm
            or type(authority) is not dict
            or set(authority) != authority_keys
            or authority.get("arm") != arm
            or type(inputs) is not dict
            or set(inputs) != input_keys
            or any(not _is_sha256(inputs[key]) for key in input_keys)
            or any(
                not _is_sha256(authority[key])
                for key in authority_keys
                - {"arm", "inputs_sha256", "seed", "source_revision"}
            )
            or type(authority["seed"]) is not int
            or authority["seed"] not in (17, 1729, 65537)
            or type(authority["source_revision"]) is not str
            or len(authority["source_revision"]) != 40
            or set(authority["source_revision"]) - set("0123456789abcdef")
            or type(result.get("attempted_updates")) is not int
            or cast(int, result["attempted_updates"]) <= 0
            or type(result.get("successful_updates")) is not int
            or result["successful_updates"] != result["attempted_updates"]
            or any(
                not _is_sha256(result.get(key))
                for key in (
                    "final_encoder_sha256",
                    "initial_encoder_sha256",
                    "initial_frozen_sha256",
                    "initial_head_sha256",
                    "schedule_sha256",
                )
            )
            or result["schedule_sha256"] != authority["batch_schedule_sha256"]
        ):
            raise ValueError("teacher-anchored cross-arm authority differs")
        launch_digest = cast(str, inputs["launch_receipt"])
        if launch_digest in launch_digests:
            raise ValueError("teacher-anchored cross-arm authority differs")
        launch_digests.add(launch_digest)
        if baseline_result is None:
            baseline_result = result
            baseline_authority = authority
            baseline_inputs = inputs
        else:
            assert baseline_authority is not None and baseline_inputs is not None
            if any(
                type(authority[key]) is not type(baseline_authority[key])
                or authority[key] != baseline_authority[key]
                for key in common_authority_keys
            ) or any(
                type(result[key]) is not type(baseline_result[key])
                or result[key] != baseline_result[key]
                for key in common_result_keys
            ):
                raise ValueError("teacher-anchored cross-arm authority differs")
            for key in input_keys - {"launch_receipt"}:
                if inputs[key] != baseline_inputs[key]:
                    raise ValueError("teacher-anchored cross-arm authority differs")
    head_only = results["head-only"]
    if head_only["final_encoder_sha256"] != head_only["initial_encoder_sha256"]:
        raise ValueError("teacher-anchored cross-arm authority differs")


def _load_authenticated_canonical_json(
    path: Path, expected_sha256: str, *, maximum_bytes: int
) -> dict[str, object]:
    """Read one stable regular file, authenticate it, and reject ambiguous JSON."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not _is_sha256(expected_sha256)
        or type(maximum_bytes) is not int
        or maximum_bytes < 2
    ):
        raise ValueError("teacher-anchored panel authority differs")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not 1 <= opened.st_size <= maximum_bytes:
                raise ValueError("teacher-anchored panel authority differs")
            payload = stream.read(maximum_bytes + 1)
            final_opened = os.fstat(stream.fileno())
        final_path = path.lstat()
    except OSError as error:
        raise ValueError("teacher-anchored panel authority differs") from error
    if (
        len(payload) != opened.st_size
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (final_opened.st_dev, final_opened.st_ino, final_opened.st_size)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (final_path.st_dev, final_path.st_ino, final_path.st_size)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("teacher-anchored panel authority differs")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if type(key) is not str or key in value:
                raise ValueError("teacher-anchored panel authority differs")
            value[key] = item
        return value

    try:
        parsed = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher-anchored panel authority differs") from error
    if (
        type(parsed) is not dict
        or payload
        != json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        + b"\n"
    ):
        raise ValueError("teacher-anchored panel authority differs")
    return parsed


def load_teacher_anchored_completed_panel(
    arguments: TeacherAnchoredEvaluationArguments,
) -> dict[str, dict[str, object]]:
    """Authenticate a complete panel's receipt, launches, results, and checkpoints."""

    arms = ("head-only", "base", "anchor", "symmetric", "complete")
    panel_keys = {"arms", "claim_eligible", "inputs_sha256", "schema", "seed", "source_revision"}
    launch_keys = {
        "arm",
        "claim_eligible",
        "inputs_sha256",
        "output",
        "schema",
        "seed",
        "source_revision",
    }
    result_keys = {
        "arm",
        "attempted_updates",
        "authority",
        "candidate_epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "claim_eligible",
        "completed_epochs",
        "diagnostics",
        "final_encoder_sha256",
        "final_frozen_sha256",
        "final_head_sha256",
        "initial_encoder_sha256",
        "initial_frozen_sha256",
        "initial_head_sha256",
        "optimizer_reset_epochs",
        "schedule_sha256",
        "schema",
        "stopped_reason",
        "successful_updates",
    }
    if type(arguments) is not TeacherAnchoredEvaluationArguments:
        raise ValueError("teacher-anchored panel authority differs")
    panel = _load_authenticated_canonical_json(
        arguments.panel_receipt, arguments.panel_receipt_sha256, maximum_bytes=1024 * 1024
    )
    panel_inputs = panel.get("inputs_sha256")
    panel_arms = panel.get("arms")
    expected_inputs = {
        "ceiling_receipt": arguments.ceiling_receipt_sha256,
        "image_tree": arguments.image_tree_sha256,
        "source_checkpoint": arguments.source_checkpoint_sha256,
        "source_snapshot": arguments.source_snapshot_sha256,
        "teacher_snapshot": arguments.teacher_snapshot_sha256,
    }
    if (
        set(panel) != panel_keys
        or panel["schema"] != "sfora-teacher-anchored-panel-v1"
        or panel["claim_eligible"] is not False
        or panel["seed"] != arguments.seed
        or panel["source_revision"] != arguments.source_revision
        or type(panel_inputs) is not dict
        or set(panel_inputs)
        != {
            "ceiling_receipt",
            "image_tree",
            "schedule",
            "source_checkpoint",
            "source_snapshot",
            "teacher_checkpoint",
            "teacher_snapshot",
        }
        or any(not _is_sha256(value) for value in panel_inputs.values())
        or any(panel_inputs[key] != value for key, value in expected_inputs.items())
        or type(panel_arms) is not dict
        or set(panel_arms) != set(arms)
    ):
        raise ValueError("teacher-anchored panel authority differs")
    directory = arguments.panel_receipt.parent
    results: dict[str, dict[str, object]] = {}
    for arm in arms:
        reference = panel_arms[arm]
        if (
            type(reference) is not dict
            or set(reference) != {"launch_receipt_sha256", "result_sha256"}
            or any(not _is_sha256(value) for value in reference.values())
        ):
            raise ValueError("teacher-anchored panel authority differs")
        launch_path = (directory / f"{arm}.launch.json").resolve()
        result_path = (directory / f"{arm}.result.json").resolve()
        checkpoint_path = result_path.with_suffix(".pt")
        if any(path.parent != directory for path in (launch_path, result_path, checkpoint_path)):
            raise ValueError("teacher-anchored panel authority differs")
        launch = _load_authenticated_canonical_json(
            launch_path,
            cast(str, reference["launch_receipt_sha256"]),
            maximum_bytes=1024 * 1024,
        )
        result = _load_authenticated_canonical_json(
            result_path,
            cast(str, reference["result_sha256"]),
            maximum_bytes=16 * 1024 * 1024,
        )
        authority = result.get("authority")
        authority_inputs = authority.get("inputs_sha256") if type(authority) is dict else None
        if type(authority) is not dict:
            raise ValueError("teacher-anchored panel authority differs")
        if (
            set(launch) != launch_keys
            or launch["schema"] != "sfora-teacher-anchored-launch-v1"
            or launch["claim_eligible"] is not False
            or launch["arm"] != arm
            or launch["seed"] != arguments.seed
            or launch["source_revision"] != arguments.source_revision
            or launch["inputs_sha256"] != panel_inputs
            or launch["output"] != str(result_path)
            or set(result) != result_keys
            or result["schema"] != "sfora-teacher-anchored-arm-v1"
            or result["claim_eligible"] is not False
            or result["arm"] != arm
            or result["candidate_epoch"] != 10
            or result["stopped_reason"] is not None
            or result["completed_epochs"] != list(range(1, 11))
            or result["optimizer_reset_epochs"] != [1, 2]
            or type(result["diagnostics"]) is not list
            or len(cast(list[object], result["diagnostics"])) != 11
            or result["checkpoint_path"] != checkpoint_path.name
            or not _is_sha256(result["checkpoint_sha256"])
            or type(authority_inputs) is not dict
            or authority_inputs.get("launch_receipt") != reference["launch_receipt_sha256"]
            or any(authority_inputs.get(key) != value for key, value in panel_inputs.items())
            or authority.get("seed") != arguments.seed
            or authority.get("source_revision") != arguments.source_revision
            or authority.get("teacher_pca_sha256") != arguments.teacher_pca_sha256
        ):
            raise ValueError("teacher-anchored panel authority differs")
        checkpoint_sha256 = cast(str, result["checkpoint_sha256"])
        try:
            checkpoint_info = checkpoint_path.lstat()
            checkpoint_payload = checkpoint_path.read_bytes()
        except OSError as error:
            raise ValueError("teacher-anchored panel authority differs") from error
        if (
            checkpoint_path.is_symlink()
            or not stat.S_ISREG(checkpoint_info.st_mode)
            or hashlib.sha256(checkpoint_payload).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("teacher-anchored panel authority differs")
        results[arm] = result
    validate_teacher_anchored_cross_arm_authority(results)
    return results


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
    retained_batch_rows: tuple[int, ...],
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
        or type(retained_batch_rows) is not tuple
        or len(retained_batch_rows) != len(batches)
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
    for batch_index, batch in enumerate(batches):
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
        retained = retained_batch_rows[batch_index]
        if (
            type(retained) is not int
            or not 1 <= retained <= len(batch)
            or (batch_index < len(batches) - 1 and retained != len(batch))
            or (
                retained < len(batch)
                and not torch.equal(
                    batch[retained:],
                    batch[retained - 1]
                    .unsqueeze(0)
                    .expand(len(batch) - retained, *batch.shape[1:]),
                )
            )
        ):
            raise ValueError("teacher-anchored serving authority differs")
        shape = tuple(batch.shape[1:])
        if shape != expected_input_shape:
            raise ValueError("teacher-anchored serving authority differs")
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("teacher-anchored serving authority differs")
    if expected_codes.shape[0] != sum(retained_batch_rows):
        raise ValueError("teacher-anchored serving authority differs")

    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(checkpoint, flags)
    with os.fdopen(descriptor, "rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        checkpoint_bytes = stream.read()
        final_opened = os.fstat(stream.fileno())
    final_path = checkpoint.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (final_opened.st_dev, final_opened.st_ino, final_opened.st_size)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (final_path.st_dev, final_path.st_ino, final_path.st_size)
        or hashlib.sha256(checkpoint_bytes).hexdigest() != expected_checkpoint_sha256
    ):
        raise ValueError("teacher-anchored serving authority differs")
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
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
        for batch, retained in zip(batches, retained_batch_rows, strict=True):
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
            chunks.append(raw[:retained].cpu().contiguous())
            for row in batch[:retained]:
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
        retained_batch_rows=retained_batch_rows,
        scoring_device=(
            "cpu"
            if device.type == "cpu"
            else f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
        ),
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
        if type(serving) is not TeacherAnchoredServingReconstruction:
            raise ValueError("teacher-anchored result authority differs")
        try:
            scoring_device = torch.device(serving.scoring_device)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("teacher-anchored result authority differs") from error
        if (
            type(serving.codes) is not torch.Tensor
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
            or type(serving.retained_batch_rows) is not tuple
            or len(serving.retained_batch_rows) != len(serving.batch_rows)
            or any(
                type(retained) is not int or not 1 <= retained <= physical
                for physical, retained in zip(
                    serving.batch_rows, serving.retained_batch_rows, strict=True
                )
            )
            or any(
                retained != physical
                for physical, retained in zip(
                    serving.batch_rows[:-1], serving.retained_batch_rows[:-1], strict=True
                )
            )
            or sum(serving.retained_batch_rows) != serving.rows
            or type(serving.scoring_device) is not str
            or scoring_device.type not in ("cpu", "cuda")
            or str(scoring_device) != serving.scoring_device
            or (scoring_device.type == "cuda" and not torch.cuda.is_available())
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
                device=scoring_device,
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
            "retained_batch_rows": list(serving.retained_batch_rows),
            "rows": serving.rows,
            "scoring_device": serving.scoring_device,
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
