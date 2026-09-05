#!/usr/bin/env python3
"""Evaluate sealed recovery students as queries against the full teacher gallery."""

from __future__ import annotations

import argparse
import math
import resource
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import evaluate_siglip_recovery_pair as recovery  # noqa: E402
import probe_siglip_depth_recovery as probe  # noqa: E402
import run_siglip_proxy_control as control  # noqa: E402
import run_siglip_recovery_pair as pair  # noqa: E402

import sfora.qwen_geometry_control as retrieval_core  # noqa: E402
import sfora.siglip_asymmetric_recovery as asymmetric_core  # noqa: E402
import sfora.siglip_recovery_evaluation as evaluation_core  # noqa: E402
from sfora.siglip_depth_recovery import prune_siglip_student  # noqa: E402

PRIOR_EVALUATION_SHA256 = "d61dbf622609b03ad6adce48ecff2428567b018a5aa6b7ff9c1d71f2b522ca8b"
PRIOR_EVALUATION_MONITOR_SHA256 = "a2ba883c87c552b40833ad22022b2503e65c3e3913cf7b588b598992ffbd68b7"
WHOLE_PROCESS_CAP_SECONDS = 1800.0


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "pair-directory",
        "smoke-result",
        "audit-result",
        "pair-monitor",
        "prior-evaluation",
        "prior-evaluation-monitor",
        "control-root",
        "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--monitor-sha256", required=True)
    parser.add_argument("--prior-evaluation-sha256", required=True)
    parser.add_argument("--prior-evaluation-monitor-sha256", required=True)
    parser.add_argument("--execute-asymmetric-evaluation", action="store_true", required=True)
    return parser.parse_args(arguments)


def read_prior_result(
    path: Path, digest: str, pair_digest: str, monitor_digest: str
) -> dict[str, Any]:
    """Authenticate the frozen within-model result used as the control."""
    value = recovery._read_json(path, digest)
    if (
        value.get("schema") != "sfora-siglip-depth-recovery-evaluation-v1"
        or value.get("claim_eligible") is not False
        or value.get("quality_measured") is not True
        or value.get("pair_sha256") != pair_digest
        or value.get("monitor_sha256") != monitor_digest
        or not isinstance(value.get("cells"), dict)
        or set(value["cells"]) != {"teacher", "pa", "relational"}
        or any(cell.get("queries") != 2746 for cell in value["cells"].values())
        or not isinstance(value.get("resources"), dict)
        or not isinstance(value.get("source_sha256"), dict)
        or value["source_sha256"].get("runner") != probe._file_sha(Path(str(recovery.__file__)))
        or value["source_sha256"].get("retrieval_core")
        != probe._file_sha(Path(str(retrieval_core.__file__)))
    ):
        raise ValueError("prior recovery evaluation authority differs")
    return value


def asymmetric_budget_seconds(training_remaining: float, prior: dict[str, Any]) -> float:
    """Deduct the completed control evaluation and apply the diagnostic cap."""
    elapsed = prior.get("elapsed_s")
    if (
        type(training_remaining) is not float
        or not math.isfinite(training_remaining)
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or elapsed >= training_remaining
    ):
        raise ValueError("asymmetric evaluation budget authority differs")
    return min(WHOLE_PROCESS_CAP_SECONDS, training_remaining - elapsed)


def read_prior_monitor(path: Path, digest: str) -> dict[str, Any]:
    """Authenticate successful whole-process time for the frozen control evaluation."""
    value = recovery._read_json(path, digest)
    elapsed = value.get("elapsed_s")
    if (
        value.get("schema") != "sfora-recovery-evaluation-monitor-v1"
        or value.get("claim_eligible") is not False
        or type(value.get("exit_code")) is not int
        or value["exit_code"] != 0
        or value.get("stop_reason") is not None
        or value.get("result_sha256") != PRIOR_EVALUATION_SHA256
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("prior recovery evaluation monitor authority differs")
    return value


def _cross_cell(
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    labels: tuple[int, ...],
    example_ids: tuple[str, ...],
    check_time: Callable[[], None] | None = None,
) -> dict[str, Any]:
    evidence = asymmetric_core.asymmetric_retrieval_evidence(
        query,
        gallery,
        query_ids=example_ids,
        gallery_ids=example_ids,
        query_labels=labels,
        gallery_labels=labels,
        check_time=check_time,
    )
    return {
        "queries": len(labels),
        "correct": sum(evidence.correct),
        "recall_at_one": evidence.recall_at_one,
        "map_at_r": evidence.map_at_r,
        "retrieval": asdict(evidence),
    }


def cross_cells(
    descriptors: dict[str, torch.Tensor],
    *,
    labels: tuple[int, ...],
    example_ids: tuple[str, ...],
    prior: dict[str, Any],
    check_time: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Reproduce all within-model controls, then compute both cross-model cells."""
    if set(descriptors) != {"teacher", "pa", "relational"}:
        raise ValueError("asymmetric descriptor arms differ")
    controls = {
        name: recovery._retrieval_cell(vectors, labels) for name, vectors in descriptors.items()
    }
    for name in ("teacher", "pa", "relational"):
        if probe._canonical(controls[name]) != probe._canonical(prior["cells"][name]):
            raise ValueError(f"{name} within-model control differs")
    return {
        name: _cross_cell(
            descriptors[name],
            descriptors["teacher"],
            labels=labels,
            example_ids=example_ids,
            check_time=check_time,
        )
        for name in ("pa", "relational")
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    began = perf_counter_ns()
    receipt = recovery._read_json(args.pair_directory / "pair-complete.json", args.pair_sha256)
    smoke = pair.read_smoke_authority(args.smoke_result, recovery.SMOKE_SHA256)
    recovery.validate_pair_receipt(receipt, smoke)
    recovery.verify_inference_dependencies(receipt)
    monitor = recovery._read_json(args.pair_monitor, args.monitor_sha256)
    remaining = recovery.evaluation_budget_seconds(receipt, monitor, args.pair_sha256)
    prior = read_prior_result(
        args.prior_evaluation,
        args.prior_evaluation_sha256,
        args.pair_sha256,
        args.monitor_sha256,
    )
    prior_monitor = read_prior_monitor(
        args.prior_evaluation_monitor, args.prior_evaluation_monitor_sha256
    )
    cap_seconds = asymmetric_budget_seconds(remaining, prior_monitor)
    prior_evaluation_internal_seconds = prior["resources"]["elapsed_seconds"]
    prior_evaluation_seconds = prior_monitor["elapsed_s"]
    prior_campaign_seconds = 21600.0 - remaining
    audit = recovery._read_json(args.audit_result, recovery.AUDIT_SHA256)
    paths = recovery.authenticate_checkpoint_files(args.pair_directory, receipt)
    payloads = {}
    for arm, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        recovery.validate_student_payload(payload, receipt, arm)
        payloads[arm] = payload
    device = recovery.evaluation_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def check_time() -> None:
        elapsed = (perf_counter_ns() - began) / 1e9
        if elapsed >= cap_seconds:
            raise RuntimeError("asymmetric evaluation exceeds frozen wall-clock cap")

    check_time()
    teacher, processor = recovery.load_teacher_and_processor(args.control_root)
    teacher = teacher.to(device).eval().requires_grad_(False)
    if control._model_state_sha256(teacher) != receipt["teacher_state_sha256"]:
        raise ValueError("asymmetric evaluation teacher state differs")
    check_time()
    examples = evaluation_core.load_recovery_evaluation_images()
    if (
        len(examples) != 2746
        or [example.example_id for example in examples] != audit["common_image_ids"]
        or len({example.example_id for example in examples}) != 2746
        or recovery.decoded_native_digest(examples, audit["common_to_native"])
        != audit["decoded_native_sha256"]
    ):
        raise ValueError("asymmetric evaluation image authority differs")
    labels = tuple(example.label for example in examples)
    example_ids = tuple(example.example_id for example in examples)
    descriptors = {}
    for name in ("teacher", "pa", "relational"):
        check_time()
        if name == "teacher":
            model = teacher
        else:
            model = prune_siglip_student(teacher).eval().requires_grad_(False)
            model.load_state_dict(payloads[name]["model_state"], strict=True)
            if control._model_state_sha256(model) != receipt["arms"][name]["final_state_sha256"]:
                raise ValueError("asymmetric evaluation restored student differs")
        print(f"asymmetric-recovery-eval: arm={name}", flush=True)
        vectors = recovery.embed_recovery_model(model, examples, processor, device, check_time)
        if (
            vectors.device.type != "cpu"
            or vectors.dtype != torch.float32
            or tuple(vectors.shape) != (2746, 512)
            or not bool(torch.isfinite(vectors).all())
            or not torch.allclose(
                torch.linalg.vector_norm(vectors, dim=1),
                torch.ones(2746),
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("asymmetric evaluation descriptors differ")
        descriptors[name] = vectors
        if name != "teacher":
            del model
            del payloads[name]
            probe._release_cpu_pages()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if control._model_state_sha256(teacher) != receipt["teacher_state_sha256"]:
        raise ValueError("asymmetric evaluation teacher changed")
    cells = cross_cells(
        descriptors,
        labels=labels,
        example_ids=example_ids,
        prior=prior,
        check_time=check_time,
    )
    decision = asymmetric_core.asymmetric_recovery_decision(cells)
    check_time()
    return {
        "schema": "sfora-siglip-asymmetric-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "surface": "exploratory-reuse-49..81",
        "pair_sha256": args.pair_sha256,
        "monitor_sha256": args.monitor_sha256,
        "prior_evaluation_sha256": args.prior_evaluation_sha256,
        "prior_evaluation_monitor_sha256": args.prior_evaluation_monitor_sha256,
        "teacher_checkpoint_sha256": receipt["teacher_checkpoint_sha256"],
        "checkpoint_seals": receipt["checkpoints"],
        "cells": cells,
        "decision": decision,
        "common_image_ids": list(example_ids),
        "decoded_native_sha256": audit["decoded_native_sha256"],
        "resources": {
            "elapsed_seconds": (perf_counter_ns() - began) / 1e9,
            "internal_work_cap_seconds": cap_seconds,
            "prior_campaign_seconds": prior_campaign_seconds,
            "prior_evaluation_internal_seconds": prior_evaluation_internal_seconds,
            "prior_evaluation_seconds": prior_evaluation_seconds,
            "cumulative_prior_seconds": prior_campaign_seconds + prior_evaluation_seconds,
            "remaining_campaign_seconds_at_start": remaining - prior_evaluation_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if device.type == "cuda"
            else 0,
        },
        "runner_sha256": probe._file_sha(Path(__file__)),
        "source_sha256": {
            "runner": probe._file_sha(Path(__file__)),
            "asymmetric_core": probe._file_sha(Path(str(asymmetric_core.__file__))),
            "evaluation_core": probe._file_sha(Path(str(evaluation_core.__file__))),
            "retrieval_core": probe._file_sha(Path(str(retrieval_core.__file__))),
            "recovery_evaluator": probe._file_sha(Path(str(recovery.__file__))),
        },
    }


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.prior_evaluation_sha256 != PRIOR_EVALUATION_SHA256:
        raise ValueError("asymmetric evaluation prior result digest differs")
    if args.prior_evaluation_monitor_sha256 != PRIOR_EVALUATION_MONITOR_SHA256:
        raise ValueError("asymmetric evaluation prior monitor digest differs")
    result = run_evaluation(args)
    control._write_new(args.output, probe._canonical(result))
    print("asymmetric-recovery-eval: COMPLETE " + probe._file_sha(args.output), flush=True)


if __name__ == "__main__":
    main()
