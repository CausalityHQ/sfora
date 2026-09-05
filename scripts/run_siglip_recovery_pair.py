#!/usr/bin/env python3
"""Final-only fixed recovery training engine; no quality-driven checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from sfora.siglip_depth_recovery import RETAINED_BLOCKS, prune_siglip_student
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.siglip_recovery_inputs import load_optimization_images

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import probe_siglip_depth_recovery as probe  # noqa: E402
import run_siglip_proxy_control as control  # noqa: E402
import run_siglip_recovery_smoke as smoke  # noqa: E402

SMOKE_RUNNER_SHA256 = "4182deddf5be7af0fd538bb0fe197914e18f2f015e59fd2eb238dc64196690e7"


def remaining_training_seconds(prior_seconds: float) -> float:
    """Keep evaluation1800s and checkpoint300s outside the update deadline."""
    if (
        type(prior_seconds) is not float
        or not math.isfinite(prior_seconds)
        or not 0 <= prior_seconds < 19500
    ):
        raise ValueError("campaign has no valid remaining training budget")
    return 21600 - prior_seconds - 2100


def read_smoke_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate the original smoke and independently recompute its feasibility."""
    if probe._file_sha(path) != expected_sha256:
        raise ValueError("smoke result SHA differs")
    value = json.loads(path.read_bytes())
    if probe._canonical(value) != path.read_bytes():
        raise ValueError("smoke result is not finite canonical JSON")
    expected = {
        "schema": "sfora-siglip-depth-recovery-smoke-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "trained_checkpoint_retained": False,
        "seed": 17,
        "logical_batch": 120,
        "microbatch": 120,
        "checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "speed_sha256": smoke.SPEED_SHA256,
        "input_proof_sha256": smoke.INPUT_PROOF_SHA256,
        "teacher_unchanged": True,
    }
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError("smoke fixed authority differs")
    if value["source_sha256"]["runner"] != SMOKE_RUNNER_SHA256:
        raise ValueError("smoke source differs")
    if set(value["arms"]) != {"pa", "relational"}:
        raise ValueError("smoke arms differ")
    times: dict[str, list[float]] = {}
    for name, arm in value["arms"].items():
        if (
            type(arm["optimizer_steps"]) is not int
            or arm["optimizer_steps"] != 10
            or arm["state_changed"] is not True
            or arm["initial_state_sha256"] == arm["final_state_sha256"]
            or type(arm["steps"]) is not list
            or len(arm["steps"]) != 10
            or type(arm["input_sha256"]) is not list
            or len(arm["input_sha256"]) != 10
        ):
            raise ValueError("smoke did not complete ten finite moving updates")
        times[name] = []
        for index, step in enumerate(arm["steps"], 1):
            if type(step["update"]) is not int or step["update"] != index:
                raise ValueError("smoke update ordering differs")
            if type(step["elapsed_ns"]) is not int or step["elapsed_ns"] <= 0:
                raise ValueError("smoke timing invalid")
            numeric = (
                "loss",
                "proxy_loss",
                "relational_loss",
                "gradient_norm",
                "maximum_descriptor_disagreement",
                "lr_multiplier",
            )
            if any(type(step[k]) is not float or not math.isfinite(step[k]) for k in numeric):
                raise ValueError("smoke numerical evidence invalid")
            if (
                not 0 <= step["maximum_descriptor_disagreement"] <= 2e-5
                or step["gradient_norm"] <= 0
                or step["lr_multiplier"] != index / 10
            ):
                raise ValueError("smoke numerical gate failed")
            times[name].append(step["elapsed_ns"] / 1e9)
    a, b = value["arms"]["pa"], value["arms"]["relational"]
    if (
        a["initial_state_sha256"] != b["initial_state_sha256"]
        or a["input_sha256"] != b["input_sha256"]
    ):
        raise ValueError("smoke pair identities differ")
    elapsed = value["resources"]["elapsed_seconds"]
    if (
        type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < sum(sum(t) for t in times.values())
        or value["resources"]["prior_gpu_seconds"] != smoke.PREFLIGHT_SECONDS
    ):
        raise ValueError("smoke elapsed authority differs")
    budget = smoke.project_recovery_budget(
        times,
        elapsed_seconds=elapsed + smoke.PREFLIGHT_SECONDS,
        startup_seconds=elapsed - sum(sum(t) for t in times.values()),
    )
    if (
        probe._canonical(budget) != probe._canonical(value["budget"])
        or not budget["within_six_hours"]
    ):
        raise ValueError("smoke feasibility recomputation failed")
    return dict(value)


def train_recovery_arm(
    student: PooledProxyAnchorModel,
    teacher: PooledProxyAnchorModel,
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    arm: str,
    expected_input_hashes: list[str] | None,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Execute all198 fixed updates from fresh optimizer state, retaining raw evidence."""
    if arm not in ("pa", "relational"):
        raise ValueError("recovery arm differs")
    if arm == "pa" and expected_input_hashes is not None:
        raise ValueError("first arm must establish the paired input authority")
    if arm == "relational" and (
        type(expected_input_hashes) is not list
        or len(expected_input_hashes) != 198
        or any(
            type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in expected_input_hashes
        )
    ):
        raise ValueError("relational arm needs all198 paired input hashes")
    device = next(student.parameters()).device
    initial_sha = control._model_state_sha256(student)
    optimizer = smoke.new_recovery_optimizer(student)
    steps, hashes = [], []
    for update in range(1, 199):
        synchronize()
        began = perf_counter_ns()
        pixels, labels = batch(update)
        input_sha = smoke._batch_sha(pixels, labels)
        if expected_input_hashes is not None and input_sha != expected_input_hashes[update - 1]:
            raise ValueError("paired input bytes differ before update")
        hashes.append(input_sha)
        evidence = smoke.recovery_update(
            student,
            optimizer,
            pixels.to(device),
            labels.to(device),
            update=update,
            teacher=teacher if arm == "relational" else None,
            microbatch_size=microbatch_size,
        )
        synchronize()
        elapsed_ns = perf_counter_ns() - began
        if elapsed_ns <= 0:
            raise RuntimeError("training clock failed to advance")
        if device.type == "cuda" and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
            raise RuntimeError("recovery CUDA memory limit")
        event = {"arm": arm, "update": update, "elapsed_ns": elapsed_ns, **evidence}
        steps.append(event)
        progress(event)
        del pixels, labels
    if any(int(optimizer.state[p]["step"]) != 198 for p in student.parameters()):
        raise RuntimeError("not every parameter completed198 optimizer steps")
    final_sha = control._model_state_sha256(student)
    if final_sha == initial_sha:
        raise RuntimeError("recovery student did not change")
    optimizer.zero_grad(set_to_none=True)
    return {
        "arm": arm,
        "completed_updates": 198,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
        "steps": steps,
        "input_sha256": hashes,
    }


def write_terminal_student(
    path: Path,
    student: PooledProxyAnchorModel,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Create/fsync/hash a terminal student; no intermediate or teacher checkpoint."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if (
        evidence.get("arm") not in ("pa", "relational")
        or type(evidence.get("completed_updates")) is not int
        or evidence["completed_updates"] != 198
        or type(evidence.get("steps")) is not list
        or len(evidence["steps"]) != 198
        or any(
            type(s.get("update")) is not int or s["update"] != i
            for i, s in enumerate(evidence["steps"], 1)
        )
        or type(evidence.get("input_sha256")) is not list
        or len(evidence["input_sha256"]) != 198
    ):
        raise ValueError("checkpoint requires complete final-only198-update evidence")
    hashes = [
        evidence.get("initial_state_sha256"),
        evidence.get("final_state_sha256"),
        *evidence["input_sha256"],
    ]
    if any(
        type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in hashes
    ):
        raise ValueError("checkpoint digest authority differs")
    vision = getattr(student.tower, "vision_model", None)
    layers = getattr(getattr(vision, "encoder", None), "layers", None)
    if (
        not isinstance(layers, torch.nn.ModuleList)
        or len(layers) != 18
        or getattr(getattr(vision, "config", None), "num_hidden_layers", None) != 18
        or student.projection.out_features != 512
        or student.class_count != 49
        or any(
            p.dtype != torch.float32 or not bool(torch.isfinite(p).all())
            for p in student.parameters()
        )
    ):
        raise ValueError("terminal student topology or finite FP32 state differs")
    if control._model_state_sha256(student) != evidence["final_state_sha256"]:
        raise ValueError("terminal state does not match final update evidence")
    payload = {
        "schema": "sfora-siglip-depth-recovery-student-v1",
        "claim_eligible": False,
        "seed": 17,
        "completed_updates": 198,
        "arm": evidence["arm"],
        "teacher_checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "input_dimensions": student.projection.in_features,
        "embedding_dimensions": 512,
        "initial_state_sha256": evidence["initial_state_sha256"],
        "final_state_sha256": evidence["final_state_sha256"],
        "input_sha256": evidence["input_sha256"],
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in student.state_dict().items()
        },
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    control._fsync_directory(path.parent)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": probe._file_sha(path),
        "arm": evidence["arm"],
        "completed_updates": 198,
    }


def execute_pair(
    teacher: PooledProxyAnchorModel,
    student_factory: Callable[[], PooledProxyAnchorModel],
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    output_dir: Path,
    initial_sha256: str,
    teacher_sha256: str,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
    writer: Callable[
        [Path, PooledProxyAnchorModel, dict[str, Any]], dict[str, Any]
    ] = write_terminal_student,
) -> dict[str, Any]:
    """Train and seal both final arms, with no evaluation or warm-state reuse."""
    if control._model_state_sha256(teacher) != teacher_sha256:
        raise ValueError("teacher differs from authenticated smoke state")
    if any(m.training for m in teacher.modules()) or any(
        p.requires_grad for p in teacher.parameters()
    ):
        raise ValueError("teacher must be frozen/eval")
    arms: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for arm in ("pa", "relational"):
        student = student_factory()
        if control._model_state_sha256(student) != initial_sha256:
            raise ValueError("student differs from authenticated initial pruned state")
        evidence = train_recovery_arm(
            student,
            teacher,
            batch,
            arm=arm,
            expected_input_hashes=None if arm == "pa" else arms["pa"]["input_sha256"],
            microbatch_size=microbatch_size,
            progress=progress,
            synchronize=synchronize,
        )
        checkpoints[arm] = writer(output_dir / f"{arm}-final.pt", student, evidence)
        arms[arm] = evidence
        del student
        probe._release_cpu_pages()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if control._model_state_sha256(teacher) != teacher_sha256:
        raise RuntimeError("teacher changed during recovery pair")
    return {
        "arms": arms,
        "checkpoints": checkpoints,
        "teacher_unchanged": True,
        "teacher_state_sha256": teacher_sha256,
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-root", "smoke-result", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke-sha256", required=True)
    parser.add_argument("--execute-recovery-pair", action="store_true", required=True)
    return parser.parse_args(arguments)


def load_teacher(root: Path) -> PooledProxyAnchorModel:
    """Strictly restore the complete fixed teacher before any student surgery."""
    receipt, checkpoint = probe.authenticate_control(root)
    tower, processor = control.load_siglip_control_components(config=SiglipProxyControlConfig())
    teacher = PooledProxyAnchorModel(
        tower=tower, input_dimensions=1152, embedding_dimensions=512, class_count=49
    ).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload.get("schema") != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload.get("claim_eligible") is not False
        or type(payload.get("seed")) is not int
        or payload["seed"] != 17
        or type(payload.get("completed_epoch")) is not int
        or payload["completed_epoch"] != 60
        or payload.get("config_sha256") != receipt["config_sha256"]
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError("recovery teacher payload differs")
    teacher.load_state_dict(payload["model_state"], strict=True)
    del payload, tower, processor
    return teacher.eval().requires_grad_(False)


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    began = perf_counter_ns()
    proof = read_smoke_authority(args.smoke_result, args.smoke_sha256)
    import sfora.siglip_depth_recovery as depth_core
    import sfora.siglip_proxy_control as control_core
    import sfora.siglip_recovery_inputs as input_core

    dependencies = {
        "runner": Path(smoke.__file__),
        "probe": Path(probe.__file__),
        "control_runner": Path(control.__file__),
        "control_core": Path(str(control_core.__file__)),
        "depth_core": Path(str(depth_core.__file__)),
        "input_core": Path(str(input_core.__file__)),
    }
    if {k: probe._file_sha(p) for k, p in dependencies.items()} != proof["source_sha256"]:
        raise ValueError("training dependencies differ from the measured smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("scientific recovery pair requires CUDA")
    import transformers

    if transformers.__version__ != "5.12.1":
        raise RuntimeError("recovery Transformers version differs")
    device = torch.device("cuda")
    control.require_control_determinism(device)
    print("recovery-pair: smoke and dependencies authenticated", flush=True)
    teacher = load_teacher(args.control_root).to(device)
    if control._model_state_sha256(teacher) != proof["teacher_state_sha256"]:
        raise ValueError("teacher state differs before input acquisition")
    examples = load_optimization_images()
    batches = smoke.recovery_batches(examples)
    transform = control.build_control_train_transform()
    probe._release_cpu_pages()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    prior = proof["resources"]["elapsed_seconds"] + smoke.PREFLIGHT_SECONDS
    remaining = remaining_training_seconds(prior)

    def batch(update: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (perf_counter_ns() - began) / 1e9 >= remaining:
            raise RuntimeError("total recovery campaign wall cap")
        pixels, labels = smoke.paired_training_batch(
            examples, batches[update - 1], transform, update=update
        )
        if (
            update <= 10
            and smoke._batch_sha(pixels, labels) != proof["arms"]["pa"]["input_sha256"][update - 1]
        ):
            raise ValueError("recovery crops differ from measured smoke")
        return pixels, labels

    result = execute_pair(
        teacher,
        lambda: prune_siglip_student(teacher).train().requires_grad_(True),
        batch,
        output_dir=args.output_dir,
        initial_sha256=proof["arms"]["pa"]["initial_state_sha256"],
        teacher_sha256=proof["teacher_state_sha256"],
        microbatch_size=120,
        synchronize=torch.cuda.synchronize,
        progress=lambda event: print(
            "recovery-pair:" + probe._canonical(event).decode().strip(), flush=True
        ),
    )
    elapsed = (perf_counter_ns() - began) / 1e9
    within_cap = prior + elapsed < 21600
    return {
        "schema": "sfora-siglip-depth-recovery-pair-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "updates_per_arm": 198,
        "status": "complete" if within_cap else "completed-outside-budget",
        "smoke_sha256": args.smoke_sha256,
        "teacher_checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "runner_sha256": probe._file_sha(Path(__file__)),
        "dependencies": proof["source_sha256"],
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "resources": {
            "within_campaign_cap": within_cap,
            "elapsed_seconds": elapsed,
            "prior_gpu_seconds": prior,
            "remaining_campaign_seconds": 21600 - prior - elapsed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        **result,
    }


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    result = run_pair(args)
    path = args.output_dir / "pair-complete.json"
    control._write_new(path, probe._canonical(result))
    print("recovery-pair: TERMINAL " + probe._file_sha(path), flush=True)
    if not result["resources"]["within_campaign_cap"]:
        raise RuntimeError("completed checkpoint evidence retained; campaign budget failed")


if __name__ == "__main__":
    main()
