#!/usr/bin/env python3
"""Evaluate only the two sealed final students from the fixed recovery pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from PIL import Image

import sfora.qwen_geometry_control as retrieval_core
import sfora.siglip_recovery_evaluation as evaluation_core
from sfora.qwen_geometry_control import (
    GeometryRetrievalEvidence,
    geometry_retrieval_evidence,
    validate_geometry_retrieval_evidence,
)
from sfora.siglip_depth_recovery import RETAINED_BLOCKS, prune_siglip_student, recovery_multiplier
from sfora.siglip_proxy_control import SiglipProxyControlConfig
from sfora.siglip_recovery_evaluation import (
    load_recovery_evaluation_images,
    profile_recovery_search,
    recovery_decision,
)

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import probe_siglip_depth_recovery as probe  # noqa: E402
import run_siglip_proxy_control as control  # noqa: E402
import run_siglip_recovery_pair as pair  # noqa: E402
import run_siglip_recovery_smoke as smoke_runner  # noqa: E402

SMOKE_SHA256 = "0481b835f594cbc9f910c40259a5d40c1958f236f51bbea49104cfdcaffd0344"
PAIR_RUNNER_SHA256 = "7b630dc3f15fec64729114e9a4a5edf70e570eb6ba4e21ae01951eaaf10fe6bb"
AUDIT_SHA256 = "4ad592f0514bbb77515fe92d8b207c06d14c16271fd1c1bc0286d190718976cf"
QUALITY_BATCH_SIZE = 32


def retrieval_cells(
    descriptors: dict[str, torch.Tensor], labels: tuple[int, ...]
) -> dict[str, Any]:
    """Retain complete per-query rankings and independently recomputed hit/AP evidence."""
    if set(descriptors) != {"teacher", "pa", "relational"}:
        raise ValueError("evaluation descriptor arms differ")
    return {name: _retrieval_cell(vectors, labels) for name, vectors in descriptors.items()}


def _retrieval_cell(vectors: torch.Tensor, labels: tuple[int, ...]) -> dict[str, Any]:
    evidence = geometry_retrieval_evidence(
        vectors, labels=labels, ordinals=tuple(range(len(labels)))
    )
    validate_geometry_retrieval_evidence(evidence)
    return {
        "queries": len(labels),
        "correct": sum(evidence.correct),
        "recall_at_one": evidence.recall_at_one,
        "map_at_r": evidence.map_at_r,
        "descriptor_bytes": vectors.numel() * vectors.element_size(),
        "retrieval": asdict(evidence),
    }


def require_teacher_reproduction(cell: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Require the fixed aggregate gate; disclose ranking-level differences separately."""
    evidence = []
    for item in (cell["retrieval"], baseline):
        if set(item) != {
            "ordinals",
            "labels",
            "nearest_ordinals",
            "top_r_ordinals",
            "correct",
            "average_precisions",
        }:
            raise ValueError("teacher retrieval schema differs")
        value = GeometryRetrievalEvidence(
            ordinals=tuple(item["ordinals"]),
            labels=tuple(item["labels"]),
            nearest_ordinals=tuple(item["nearest_ordinals"]),
            top_r_ordinals=tuple(tuple(row) for row in item["top_r_ordinals"]),
            correct=tuple(item["correct"]),
            average_precisions=tuple(item["average_precisions"]),
        )
        validate_geometry_retrieval_evidence(value)
        evidence.append(value)
    actual, old = evidence
    if (
        actual.ordinals != old.ordinals
        or actual.labels != old.labels
        or cell["correct"] != sum(actual.correct)
        or cell["map_at_r"] != actual.map_at_r
        or sum(actual.correct) != sum(old.correct)
        or actual.map_at_r != old.map_at_r
    ):
        raise ValueError("teacher aggregate or ordinal reproduction differs")
    exact = probe._canonical(cell["retrieval"]) == probe._canonical(baseline)
    first = next(
        (
            ordinal
            for i, ordinal in enumerate(actual.ordinals)
            if any(
                getattr(actual, key)[i] != getattr(old, key)[i]
                for key in ("nearest_ordinals", "top_r_ordinals", "correct", "average_precisions")
            )
        ),
        None,
    )
    return {
        "aggregate_reproduced": True,
        "per_query_bitwise_reproduced": exact,
        "first_differing_ordinal": first,
    }


def quality_batch_size(receipt: dict[str, Any]) -> int:
    size = receipt["environment"]["evaluation_batch_size"]
    if type(size) is not int or size != QUALITY_BATCH_SIZE or 128 % size:
        raise ValueError("quality batch partition differs from authenticated teacher audit")
    return size


def decoded_native_digest(examples: tuple[Any, ...], common_to_native: list[int]) -> str:
    """Reproduce the audit's RGB-byte digest in its native, not common, ordinal order."""
    n = len(examples)
    if (
        len(common_to_native) != n
        or any(type(i) is not int for i in common_to_native)
        or sorted(common_to_native) != list(range(n))
    ):
        raise ValueError("native/common pixel permutation differs")
    common_order = sorted(range(n), key=lambda i: common_to_native[i])
    digest = hashlib.sha256()
    for ordinal, common in enumerate(common_order):
        example = examples[common]
        if not isinstance(example.image, Image.Image):
            raise ValueError("evaluation pixel evidence is not decoded RGB")
        rgb = example.image.convert("RGB")
        digest.update(
            probe._canonical(
                {
                    "ordinal": ordinal,
                    "label": example.label,
                    "example_id": example.example_id,
                    "size": list(rgb.size),
                }
            )
        )
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def paired_discordances(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, int]:
    a, b = teacher["retrieval"]["correct"], student["retrieval"]["correct"]
    if len(a) != len(b) or any(type(v) is not bool for v in (*a, *b)):
        raise ValueError("paired correctness evidence differs")
    return {
        "both_correct": sum(x and y for x, y in zip(a, b, strict=True)),
        "teacher_only": sum(x and not y for x, y in zip(a, b, strict=True)),
        "student_only": sum(not x and y for x, y in zip(a, b, strict=True)),
        "both_wrong": sum(not x and not y for x, y in zip(a, b, strict=True)),
    }


def evaluation_budget_seconds(
    receipt: dict[str, Any], monitor: dict[str, Any], pair_sha: str
) -> float:
    """Use original whole-process time, never refund setup/serialization to the budget."""
    fixed = {
        "schema": "sfora-recovery-pair-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "prior_seconds": 1355,
        "result_sha256": pair_sha,
    }
    if any(not _exact(monitor.get(k), v) for k, v in fixed.items()):
        raise ValueError("training monitor is not the successful original")
    elapsed = monitor["elapsed_s"]
    if (
        type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < receipt["resources"]["elapsed_seconds"]
        or monitor["prior_seconds"] < receipt["resources"]["prior_gpu_seconds"]
        or 1355 + elapsed >= 21600
    ):
        raise ValueError("training monitor elapsed/budget differs")
    return 21600 - 1355 - elapsed


def _exact(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_pair_receipt(value: dict[str, Any], smoke: dict[str, Any]) -> None:
    """Recompute final-only ordering, pair identities, numerical and budget evidence."""
    expected = {
        "schema": "sfora-siglip-depth-recovery-pair-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "updates_per_arm": 198,
        "status": "complete",
        "smoke_sha256": SMOKE_SHA256,
        "teacher_checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "runner_sha256": PAIR_RUNNER_SHA256,
        "dependencies": smoke["source_sha256"],
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "teacher_unchanged": True,
        "teacher_state_sha256": smoke["teacher_state_sha256"],
    }
    try:
        if any(not _exact(value.get(k), v) for k, v in expected.items()):
            raise ValueError("pair fixed authority differs")
        if set(value["arms"]) != {"pa", "relational"} or set(value["checkpoints"]) != {
            "pa",
            "relational",
        }:
            raise ValueError("pair is not two sealed final arms")
        total_steps_ns = 0
        for name in ("pa", "relational"):
            arm, seal = value["arms"][name], value["checkpoints"][name]
            if (
                not _exact(arm["completed_updates"], 198)
                or arm["arm"] != name
                or not _exact(seal["completed_updates"], 198)
                or seal["arm"] != name
                or seal["basename"] != f"{name}-final.pt"
                or type(seal["bytes"]) is not int
                or seal["bytes"] <= 0
                or not _digest(seal["sha256"])
                or arm["initial_state_sha256"] != smoke["arms"][name]["initial_state_sha256"]
                or not _digest(arm["final_state_sha256"])
                or arm["final_state_sha256"] == arm["initial_state_sha256"]
                or type(arm["steps"]) is not list
                or len(arm["steps"]) != 198
                or type(arm["input_sha256"]) is not list
                or len(arm["input_sha256"]) != 198
                or not all(_digest(h) for h in arm["input_sha256"])
                or arm["input_sha256"][:10] != smoke["arms"][name]["input_sha256"]
            ):
                raise ValueError("pair final arm/checkpoint authority differs")
            for index, step in enumerate(arm["steps"], 1):
                numeric = (
                    "loss",
                    "proxy_loss",
                    "relational_loss",
                    "gradient_norm",
                    "maximum_descriptor_disagreement",
                    "lr_multiplier",
                )
                if (
                    not _exact(step["update"], index)
                    or step["arm"] != name
                    or type(step["elapsed_ns"]) is not int
                    or step["elapsed_ns"] <= 0
                    or any(
                        type(step[k]) is not float or not math.isfinite(step[k]) for k in numeric
                    )
                    or step["gradient_norm"] <= 0
                    or not 0 <= step["maximum_descriptor_disagreement"] <= 2e-5
                    or step["lr_multiplier"] != recovery_multiplier(index)
                    or (name == "pa" and step["relational_loss"] != 0.0)
                ):
                    raise ValueError("pair update numerical authority differs")
                total_steps_ns += step["elapsed_ns"]
        a, b = value["arms"]["pa"], value["arms"]["relational"]
        if (
            a["input_sha256"] != b["input_sha256"]
            or a["initial_state_sha256"] != b["initial_state_sha256"]
        ):
            raise ValueError("pair inputs or initial states differ")
        resource = value["resources"]
        prior = smoke["resources"]["elapsed_seconds"] + smoke_runner.PREFLIGHT_SECONDS
        for key in ("elapsed_seconds", "prior_gpu_seconds", "remaining_campaign_seconds"):
            if type(resource[key]) is not float or not math.isfinite(resource[key]):
                raise ValueError("pair budget numbers differ")
        if (
            resource["within_campaign_cap"] is not True
            or resource["elapsed_seconds"] < total_steps_ns / 1e9
            or resource["prior_gpu_seconds"] != prior
            or resource["remaining_campaign_seconds"] != 21600 - prior - resource["elapsed_seconds"]
            or resource["remaining_campaign_seconds"] <= 0
        ):
            raise ValueError("pair campaign budget differs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("pair authority is incomplete") from error


def authenticate_checkpoint_files(directory: Path, receipt: dict[str, Any]) -> dict[str, Path]:
    """Check both byte streams before any tensor parser or evaluation image access."""
    paths = {}
    for arm in ("pa", "relational"):
        seal = receipt["checkpoints"][arm]
        path = directory / f"{arm}-final.pt"
        if (
            seal["basename"] != path.name
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != seal["bytes"]
            or probe._file_sha(path) != seal["sha256"]
        ):
            raise ValueError("checkpoint byte identity differs")
        paths[arm] = path
    return paths


def validate_student_payload(payload: Mapping[str, Any], receipt: dict[str, Any], arm: str) -> None:
    """Bind a finite FP32 state to its exact terminal arm; strict topology loads follow."""
    evidence = receipt["arms"][arm]
    expected = {
        "schema": "sfora-siglip-depth-recovery-student-v1",
        "claim_eligible": False,
        "seed": 17,
        "completed_updates": 198,
        "arm": arm,
        "teacher_checkpoint_sha256": receipt["teacher_checkpoint_sha256"],
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "input_dimensions": 1152,
        "embedding_dimensions": 512,
        "initial_state_sha256": evidence["initial_state_sha256"],
        "final_state_sha256": evidence["final_state_sha256"],
        "input_sha256": evidence["input_sha256"],
    }
    if set(payload) != {*expected, "model_state"} or any(
        not _exact(payload.get(k), v) for k, v in expected.items()
    ):
        raise ValueError("student payload bindings differ")
    state = payload["model_state"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("student state absent")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if (
            type(name) is not str
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("student state must be finite FP32")
        meta = probe._canonical(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(meta).to_bytes(8, "little"))
        digest.update(meta)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    if digest.hexdigest() != evidence["final_state_sha256"]:
        raise ValueError("student actual state digest differs")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "pair-directory",
        "smoke-result",
        "audit-result",
        "pair-monitor",
        "control-root",
        "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--monitor-sha256", required=True)
    parser.add_argument("--execute-recovery-evaluation", action="store_true", required=True)
    return parser.parse_args(arguments)


def _read_json(path: Path, digest: str) -> dict[str, Any]:
    if not _digest(digest) or path.is_symlink() or probe._file_sha(path) != digest:
        raise ValueError("evaluation input digest differs")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or probe._canonical(value) != raw:
        raise ValueError("evaluation input is not finite canonical JSON")
    return value


def verify_inference_dependencies(receipt: dict[str, Any]) -> None:
    import sfora.siglip_depth_recovery as depth
    import sfora.siglip_proxy_control as control_core
    import sfora.siglip_recovery_inputs as inputs

    modules = {
        "runner": smoke_runner,
        "probe": probe,
        "control_runner": control,
        "control_core": control_core,
        "depth_core": depth,
        "input_core": inputs,
    }
    if (
        probe._file_sha(Path(pair.__file__)) != PAIR_RUNNER_SHA256
        or {k: probe._file_sha(Path(str(v.__file__))) for k, v in modules.items()}
        != receipt["dependencies"]
    ):
        raise ValueError("inference dependencies differ from sealed training")


def evaluation_device() -> torch.device:
    import transformers

    if not torch.cuda.is_available() or transformers.__version__ != "5.12.1":
        raise RuntimeError("scientific evaluation requires the pinned CUDA environment")
    device = torch.device("cuda")
    control.require_control_determinism(device)
    return device


def load_teacher_and_processor(root: Path) -> tuple[Any, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor

    quality_batch_size(_read_json(root / "seed-017.receipt.json", probe.SEED_RECEIPT_SHA256))
    teacher = pair.load_teacher(root)
    config = SiglipProxyControlConfig()
    snapshot = Path(
        snapshot_download(config.model_name, revision=config.model_revision, local_files_only=True)
    ).resolve(strict=True)
    if snapshot.name != config.model_revision:
        raise ValueError("evaluation processor revision differs")
    processor_class: Any = AutoImageProcessor
    processor = processor_class.from_pretrained(str(snapshot), local_files_only=True)
    return teacher, processor


def embed_recovery_model(
    model: Any,
    examples: tuple[Any, ...],
    processor: Any,
    device: torch.device,
    check_time: Callable[[], None],
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(examples), 128):
        check_time()
        check_cuda_evaluation_budget(device)
        print(f"recovery-eval: embedding {start}/{len(examples)}", flush=True)
        _, projected, _ = control.embed_control_examples(
            model=model,
            examples=examples[start : start + 128],
            processor=processor,
            device=device,
            batch_size=QUALITY_BATCH_SIZE,
        )
        chunks.append(projected)
        check_cuda_evaluation_budget(device)
    return torch.cat(chunks)


def check_cuda_evaluation_budget(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
        raise RuntimeError("evaluation CUDA memory limit")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    began = perf_counter_ns()
    receipt = _read_json(args.pair_directory / "pair-complete.json", args.pair_sha256)
    smoke = pair.read_smoke_authority(args.smoke_result, SMOKE_SHA256)
    validate_pair_receipt(receipt, smoke)
    verify_inference_dependencies(receipt)
    monitor = _read_json(args.pair_monitor, args.monitor_sha256)
    remaining = evaluation_budget_seconds(receipt, monitor, args.pair_sha256)
    audit = _read_json(args.audit_result, AUDIT_SHA256)
    paths = authenticate_checkpoint_files(args.pair_directory, receipt)
    # Authenticate both tensor payloads before reading any evaluation image.
    payloads = {}
    for arm, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        validate_student_payload(payload, receipt, arm)
        payloads[arm] = payload
    device = evaluation_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    teacher, processor = load_teacher_and_processor(args.control_root)
    teacher = teacher.to(device).eval().requires_grad_(False)
    if control._model_state_sha256(teacher) != receipt["teacher_state_sha256"]:
        raise ValueError("evaluation teacher state differs")

    def check_time() -> None:
        if (perf_counter_ns() - began) / 1e9 >= remaining:
            raise RuntimeError("evaluation exceeds total six-hour campaign cap")

    check_time()
    examples = load_recovery_evaluation_images()
    if (
        len(examples) != 2746
        or [e.example_id for e in examples] != audit["common_image_ids"]
        or len(set(e.example_id for e in examples)) != 2746
    ):
        raise ValueError("evaluation image ordinal authority differs")
    labels = tuple(e.label for e in examples)
    pixel_digest = decoded_native_digest(examples, audit["common_to_native"])
    if pixel_digest != audit["decoded_native_sha256"]:
        raise ValueError("evaluation decoded RGB bytes differ from authenticated audit")
    baseline = audit["reproductions"]["siglip-projected-17"]["retrieval"]
    if list(labels) != baseline["labels"]:
        raise ValueError("evaluation label authority differs")
    descriptors = {}
    cells = {}
    reproduction = None
    for name in ("teacher", "pa", "relational"):
        check_time()
        if name == "teacher":
            model = teacher
        else:
            model = prune_siglip_student(teacher).eval().requires_grad_(False)
            model.load_state_dict(payloads[name]["model_state"], strict=True)
            if control._model_state_sha256(model) != receipt["arms"][name]["final_state_sha256"]:
                raise ValueError("restored final student differs")
        print(f"recovery-eval: arm={name}", flush=True)
        vectors = embed_recovery_model(model, examples, processor, device, check_time)
        if (
            vectors.device.type != "cpu"
            or vectors.dtype != torch.float32
            or tuple(vectors.shape) != (2746, 512)
            or not bool(torch.isfinite(vectors).all())
            or not torch.allclose(
                torch.linalg.vector_norm(vectors, dim=1), torch.ones(2746), atol=1e-6, rtol=0
            )
        ):
            raise ValueError("evaluation descriptors differ from2746x512unitFP32")
        descriptors[name] = vectors
        cells[name] = _retrieval_cell(vectors, labels)
        if name == "teacher":
            reproduction = require_teacher_reproduction(cells[name], baseline)
        if name != "teacher":
            del model
            del payloads[name]
            probe._release_cpu_pages()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if control._model_state_sha256(teacher) != receipt["teacher_state_sha256"]:
        raise ValueError("teacher changed during evaluation")
    check_time()
    search = profile_recovery_search(descriptors)
    decision = recovery_decision(
        cells["teacher"], {k: cells[k] for k in ("pa", "relational")}, search["samples_ns"]
    )
    check_time()
    return {
        "schema": "sfora-siglip-depth-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "seed": 17,
        "surface": "exploratory-reuse-49..81",
        "pair_sha256": args.pair_sha256,
        "monitor_sha256": args.monitor_sha256,
        "smoke_sha256": SMOKE_SHA256,
        "audit_sha256": AUDIT_SHA256,
        "teacher_checkpoint_sha256": receipt["teacher_checkpoint_sha256"],
        "checkpoint_seals": receipt["checkpoints"],
        "teacher_reproduction": reproduction,
        "quality_batch_size": QUALITY_BATCH_SIZE,
        "cells": cells,
        "search_profile": search,
        "search_scope": "identical-width-CPU-search-regression-guard;not-a-search-cost-saving",
        "paired_discordances": {
            k: paired_discordances(cells["teacher"], cells[k]) for k in ("pa", "relational")
        },
        "decision": decision,
        "common_image_ids": [e.example_id for e in examples],
        "decoded_native_sha256": pixel_digest,
        "speed_evidence": {
            "sha256": smoke_runner.SPEED_SHA256,
            "scope": "pre-recovery-fixed-architecture;not-remeasured-after-training",
        },
        "resources": {
            "elapsed_seconds": (perf_counter_ns() - began) / 1e9,
            "prior_campaign_seconds": 21600 - remaining,
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
            "evaluation_core": probe._file_sha(Path(evaluation_core.__file__)),
            "retrieval_core": probe._file_sha(Path(retrieval_core.__file__)),
        },
    }


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = run_evaluation(args)
    control._write_new(args.output, probe._canonical(result))
    print("recovery-eval: COMPLETE " + probe._file_sha(args.output), flush=True)


if __name__ == "__main__":
    main()
