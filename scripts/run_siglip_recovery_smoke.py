#!/usr/bin/env python3
"""Fixed paired recovery smoke; no evaluation or retained trained checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from sfora.data import ImageExample
from sfora.siglip_depth_recovery import (
    RETAINED_BLOCKS,
    prune_siglip_student,
    recomputed_recovery_backward,
    recovery_multiplier,
    speed_gate,
)
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.siglip_recovery_inputs import load_optimization_images

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import probe_siglip_depth_recovery as probe  # noqa: E402
import run_siglip_proxy_control as control  # noqa: E402

SPEED_SHA256 = "90bc6846060e9df54ef344857eb7cc0d52433d4613865320cc53ea89abedadda"
INPUT_PROOF_SHA256 = "effeabc19451897d2cb5b75de1347b671c9071694aad7e23a8b92472d7197d45"
PREFLIGHT_SECONDS = 761.4893234689953


def new_recovery_optimizer(model: PooledProxyAnchorModel) -> torch.optim.AdamW:
    """Allocate fresh FP32 AdamW moments; never restore incumbent optimizer state."""
    if any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("all student parameters must train in FP32")
    return torch.optim.AdamW(
        control._optimizer_groups(model, SiglipProxyControlConfig()),
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )


def recovery_update(
    student: PooledProxyAnchorModel,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    update: int,
    teacher: PooledProxyAnchorModel | None = None,
    microbatch_size: int,
) -> dict[str, float]:
    """Apply one fixed scheduled/clipped full-logical-batch recovery update."""
    multiplier = recovery_multiplier(update)
    if any(not p.requires_grad for p in student.parameters()):
        raise ValueError("all surviving student parameters must train")
    if teacher is not None:
        if any(m.training for m in teacher.modules()) or any(
            p.requires_grad or p.grad is not None for p in teacher.parameters()
        ):
            raise ValueError("teacher must be frozen/eval with no gradients")
        if {p.data_ptr() for p in teacher.parameters()} & {
            p.data_ptr() for p in student.parameters()
        }:
            raise ValueError("teacher and student must have independent storage")
    expected_ids = {id(p) for p in student.parameters()}
    actual_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("optimizer does not own exactly this student")
    student.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
    target = None
    if teacher is not None:
        with torch.no_grad():
            target = torch.cat(
                [
                    teacher.encode(inputs[start : start + microbatch_size])
                    for start in range(0, len(inputs), microbatch_size)
                ]
            )
    evidence = recomputed_recovery_backward(
        student,
        inputs,
        labels,
        teacher_descriptors=target,
        microbatch_size=microbatch_size,
    )
    norm = torch.nn.utils.clip_grad_norm_(
        student.parameters(),
        10.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in student.parameters()):
        raise RuntimeError("nonfinite recovery parameter after AdamW")
    return {
        "loss": float(evidence.loss),
        "proxy_loss": float(evidence.proxy_loss),
        "relational_loss": float(evidence.relational_loss),
        "gradient_norm": float(norm),
        "maximum_descriptor_disagreement": evidence.maximum_descriptor_disagreement,
        "lr_multiplier": multiplier,
    }


def paired_training_batch(
    examples: tuple[ImageExample, ...],
    positions: tuple[int, ...],
    transform: Callable[[object], torch.Tensor],
    *,
    update: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce crops by update, independent of arm/model RNG consumption."""
    recovery_multiplier(update)  # Validate the registered 1..198 coordinate.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(
            control._seed64("siglip-depth-recovery-crops-v1", 17, update)
        )
        torch.set_rng_state(generator.get_state())
        return control.materialize_control_training_batch(
            examples=examples,
            positions=positions,
            transform=transform,
        )


def project_recovery_budget(
    step_seconds: dict[str, list[float]],
    *,
    elapsed_seconds: float,
    startup_seconds: float,
) -> dict[str, Any]:
    """Conservative two-arm198-update projection within total six GPU-hours."""
    if set(step_seconds) != {"pa", "relational"} or any(
        type(values) is not list
        or len(values) != 10
        or any(type(v) is not float or not math.isfinite(v) or v <= 0 for v in values)
        for values in step_seconds.values()
    ):
        raise ValueError("budget requires ten finite positive measured steps per arm")
    if (
        type(elapsed_seconds) is not float
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("elapsed cap usage must be finite and nonnegative")
    if (
        type(startup_seconds) is not float
        or not math.isfinite(startup_seconds)
        or startup_seconds < 0
    ):
        raise ValueError("future startup allowance must be finite and nonnegative")
    projected = (
        elapsed_seconds
        + startup_seconds
        + sum(max(v) for v in step_seconds.values()) * 198 * 1.25
        + 1800
        + 300
    )
    return {
        "projected_total_seconds": projected,
        "within_six_hours": projected <= 21600,
        "future_startup_seconds": startup_seconds,
        "checkpoint_allowance_seconds": 300,
        "evaluation_allowance_seconds": 1800,
    }


def authenticate_preflights(speed_path: Path, proof_path: Path) -> None:
    """Check immutable speed and exact optimization-only input-equivalence evidence."""
    if (
        probe._file_sha(speed_path) != SPEED_SHA256
        or probe._file_sha(proof_path) != INPUT_PROOF_SHA256
    ):
        raise ValueError("registered preflight bytes differ")
    speed, proof = json.loads(speed_path.read_bytes()), json.loads(proof_path.read_bytes())
    if not speed_gate(speed["timing"]["windows"]):
        raise ValueError("registered speed gate failed")
    if (
        proof["prior_speed_sha256"] != SPEED_SHA256
        or proof["original_rgb_sha256"] != speed["original_rgb_sha256"]
        or proof["matches_prior_speed_inputs"] is not True
        or proof["only_optimization_image_rows_fetched"] is not True
        or type(proof["selected_count"]) is not int
        or proof["selected_count"] != 128
        or type(proof["pixel_access_count"]) is not int
        or proof["pixel_access_count"] != 128
        or proof["quality_measured"] is not False
        or proof["claim_eligible"] is not False
    ):
        raise ValueError("registered optimization input proof differs")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("control-root", "speed-result", "input-proof", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--execute-recovery-smoke", action="store_true", required=True)
    return parser.parse_args(arguments)


def _batch_sha(inputs: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256(b"sfora-recovery-batch-v1\0")
    for tensor in (inputs, labels):
        digest.update(probe._canonical({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure_smoke_pair(
    teacher: PooledProxyAnchorModel,
    student_factory: Callable[[], PooledProxyAnchorModel],
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    microbatch_size: int,
    synchronize: Callable[[], None],
    progress: Callable[[dict[str, Any]], None],
    clock: Callable[[], int] = perf_counter_ns,
) -> dict[str, Any]:
    """Measure ten disposable updates per fresh arm; bind crops, state and teacher."""
    teacher_sha = control._model_state_sha256(teacher)
    arms: dict[str, Any] = {}
    initial_sha = None
    for name in ("pa", "relational"):
        student = student_factory()
        state_sha = control._model_state_sha256(student)
        if initial_sha is not None and initial_sha != state_sha:
            raise ValueError("paired student initialization differs")
        initial_sha = state_sha
        optimizer = new_recovery_optimizer(student)
        steps, input_hashes = [], []
        for update in range(1, 11):
            synchronize()
            began = clock()
            pixels, labels = batch(update)
            input_sha = _batch_sha(pixels, labels)
            if name == "relational" and input_sha != arms["pa"]["input_sha256"][update - 1]:
                raise ValueError("paired input bytes differ")
            input_hashes.append(input_sha)
            evidence = recovery_update(
                student,
                optimizer,
                pixels.to(device),
                labels.to(device),
                update=update,
                teacher=teacher if name == "relational" else None,
                microbatch_size=microbatch_size,
            )
            synchronize()
            elapsed_ns = clock() - began
            if type(elapsed_ns) is not int or elapsed_ns <= 0:
                raise RuntimeError("smoke clock failed to advance")
            if device.type == "cuda" and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
                raise RuntimeError("recovery CUDA memory limit")
            steps.append({"update": update, "elapsed_ns": elapsed_ns, **evidence})
            progress({"arm": name, **steps[-1]})
            del pixels, labels
        final_sha = control._model_state_sha256(student)
        if final_sha == state_sha:
            raise RuntimeError("student did not move during recovery")
        step_counts = [int(optimizer.state[p]["step"]) for p in student.parameters()]
        if any(count != 10 for count in step_counts):
            raise RuntimeError("some student parameter missed optimizer updates")
        arms[name] = {
            "initial_state_sha256": state_sha,
            "final_state_sha256": final_sha,
            "input_sha256": input_hashes,
            "steps": steps,
            "optimizer_steps": 10,
            "state_changed": True,
        }
        del optimizer, student
        probe._release_cpu_pages()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if control._model_state_sha256(teacher) != teacher_sha:
        raise RuntimeError("frozen teacher state changed")
    return {"arms": arms, "teacher_state_sha256": teacher_sha, "teacher_unchanged": True}


def recovery_batches(examples: tuple[ImageExample, ...]) -> tuple[tuple[int, ...], ...]:
    """Freeze the original30x4 sampler over six33-update recovery epochs."""
    state = control.SamplerState.initial()
    batches: list[tuple[int, ...]] = []
    for epoch in range(6):
        epoch_batches, state = control._build_epoch_batches(
            example_ids=tuple(e.example_id for e in examples),
            labels=torch.tensor([e.label for e in examples], dtype=torch.int64),
            seed=17,
            epoch=epoch,
            steps_per_epoch=33,
            state=state,
        )
        batches.extend(epoch_batches)
    return tuple(batches)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Execute only the authenticated, fixed, disposable two-arm smoke."""
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter_ns()
    authenticate_preflights(args.speed_result, args.input_proof)
    receipt, checkpoint = probe.authenticate_control(args.control_root)
    if not torch.cuda.is_available():
        raise RuntimeError("scientific recovery smoke requires CUDA")
    device = torch.device("cuda")
    control.require_control_determinism(device)
    import transformers

    if transformers.__version__ != "5.12.1":
        raise RuntimeError("recovery Transformers version differs from5.12.1")
    print("recovery-smoke: checkpoint and speed/input authorities authenticated", flush=True)
    examples = load_optimization_images()
    if len(examples) != 3963:
        raise ValueError("recovery optimization population differs")
    batches = recovery_batches(examples)
    transform = control.build_control_train_transform()
    probe._release_cpu_pages()
    print("recovery-smoke:3963 optimization images; no evaluation pixel access", flush=True)
    tower, processor = control.load_siglip_control_components(config=SiglipProxyControlConfig())
    teacher = PooledProxyAnchorModel(
        tower=tower,
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
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
        raise ValueError("recovery incumbent payload differs")
    teacher.load_state_dict(payload["model_state"], strict=True)
    del payload, tower, processor
    teacher = teacher.to(device).eval().requires_grad_(False)
    probe._release_cpu_pages()
    torch.cuda.reset_peak_memory_stats()

    def student_factory() -> PooledProxyAnchorModel:
        return prune_siglip_student(teacher).train().requires_grad_(True)

    def batch(update: int) -> tuple[torch.Tensor, torch.Tensor]:
        return paired_training_batch(examples, batches[update - 1], transform, update=update)

    evidence = measure_smoke_pair(
        teacher,
        student_factory,
        batch,
        device=device,
        microbatch_size=120,
        synchronize=torch.cuda.synchronize,
        progress=lambda event: print(
            "recovery-smoke:" + probe._canonical(event).decode().strip(), flush=True
        ),
    )
    elapsed = (perf_counter_ns() - began) / 1e9
    times = {
        name: [float(step["elapsed_ns"] / 1e9) for step in arm["steps"]]
        for name, arm in evidence["arms"].items()
    }
    import sfora.siglip_depth_recovery as depth_core
    import sfora.siglip_proxy_control as control_core
    import sfora.siglip_recovery_inputs as input_core

    sources = {
        "runner": Path(__file__),
        "depth_core": Path(str(depth_core.__file__)),
        "input_core": Path(str(input_core.__file__)),
        "probe": Path(probe.__file__),
        "control_runner": Path(control.__file__),
        "control_core": Path(str(control_core.__file__)),
    }
    return {
        "schema": "sfora-siglip-depth-recovery-smoke-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "trained_checkpoint_retained": False,
        "seed": 17,
        "logical_batch": 120,
        "microbatch": 120,
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "speed_sha256": SPEED_SHA256,
        "input_proof_sha256": INPUT_PROOF_SHA256,
        "source_sha256": {key: probe._file_sha(path) for key, path in sources.items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "threads": torch.get_num_threads(),
        },
        "resources": {
            "elapsed_seconds": elapsed,
            "prior_gpu_seconds": PREFLIGHT_SECONDS,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "budget": project_recovery_budget(
            times,
            elapsed_seconds=elapsed + PREFLIGHT_SECONDS,
            startup_seconds=max(0.0, elapsed - sum(sum(t) for t in times.values())),
        ),
        **evidence,
    }


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    result = run_smoke(args)
    control._write_new(args.output, probe._canonical(result))
    print("recovery-smoke: COMPLETE " + probe._file_sha(args.output), flush=True)


if __name__ == "__main__":
    main()
