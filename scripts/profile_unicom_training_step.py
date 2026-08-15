#!/usr/bin/env python3
"""Profile the registered UniCOM training step without mutating its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WARMUP_STEPS = 20
MEASURE_STEPS = 50
PROFILER_STEPS = 10
BOOTSTRAP_SEED = 20_016
BOOTSTRAP_REPLICATES = 10_000
KERNEL_GATE_THRESHOLD = 0.1
OBJECTIVE_MARKER = "unicom_objective_profile_step"
COMPONENT_KEYS = (
    "h2d_seconds",
    "zero_grad_seconds",
    "backbone_forward_seconds",
    "objective_forward_seconds",
    "head_backward_seconds",
    "backbone_backward_seconds",
    "update_seconds",
    "tail_seconds",
)
TIMING_KEYS = (
    "step_wall_seconds",
    "cuda_step_seconds",
    *COMPONENT_KEYS,
)
_GEMM_NAMES = {
    "aten::addmm",
    "aten::baddbmm",
    "aten::bmm",
    "aten::linear",
    "aten::matmul",
    "aten::mm",
}


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_timing_sample(sample: object, index: int) -> dict[str, float]:
    if type(sample) is not dict or tuple(sample) != TIMING_KEYS:
        raise ValueError(f"timing sample {index} schema differs")
    validated = {
        key: _finite_float(sample[key], f"timing sample {index} {key}", nonnegative=True)
        for key in TIMING_KEYS
    }
    if validated["step_wall_seconds"] <= 0.0 or validated["cuda_step_seconds"] <= 0.0:
        raise ValueError("timing step durations must be positive")
    contiguous = math.fsum(validated[key] for key in COMPONENT_KEYS)
    if not math.isclose(
        contiguous,
        validated["cuda_step_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("timing contiguous component sum differs")
    if validated["cuda_step_seconds"] > validated["step_wall_seconds"] * (1.0 + 1e-9):
        raise ValueError("CUDA step exceeds wall step")
    return validated


def summarize_timing_samples(samples: object) -> dict[str, object]:
    """Validate and aggregate unprofiled contiguous CUDA-event samples."""

    if type(samples) is not tuple or not samples:
        raise ValueError("timing samples must be a nonempty tuple")
    rows = tuple(_validate_timing_sample(sample, index) for index, sample in enumerate(samples))
    wall = np.asarray([row["step_wall_seconds"] for row in rows], dtype=np.float64)
    component_means = {
        key: math.fsum(row[key] for row in rows) / len(rows) for key in COMPONENT_KEYS
    }
    objective = (
        component_means["objective_forward_seconds"]
        + component_means["head_backward_seconds"]
    )
    wall_mean = float(wall.mean(dtype=np.float64))
    return {
        "step_wall_seconds": float(np.median(wall)),
        "step_wall_p10_seconds": float(np.quantile(wall, 0.1)),
        "step_wall_p90_seconds": float(np.quantile(wall, 0.9)),
        "objective_ceiling_seconds": objective,
        "objective_ceiling_fraction": objective / wall_mean,
        "component_mean_seconds": component_means,
    }


def _event_children(event: object) -> tuple[object, ...]:
    children = getattr(event, "cpu_children", None)
    if type(children) is not list:
        raise TypeError("profiler event children differ")
    return tuple(children)


def _descendants(event: object):
    for child in _event_children(event):
        yield child
        yield from _descendants(child)


def fusible_nonbackbone_seconds(events: object) -> float:
    """Sum objective-local CUDA self time except irreducible classifier GEMMs."""

    if type(events) is not tuple:
        raise TypeError("profiler events must be a tuple")
    markers = tuple(event for event in events if getattr(event, "name", None) == OBJECTIVE_MARKER)
    if len(markers) != 1:
        raise ValueError("profiler objective marker count differs")
    total_microseconds = 0.0
    for event in _descendants(markers[0]):
        name = getattr(event, "name", None)
        value = getattr(event, "self_device_time_total", None)
        if type(name) is not str or type(value) not in (float, int):
            raise TypeError("profiler event fields differ")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("profiler event device time differs")
        if name not in _GEMM_NAMES:
            total_microseconds += duration
    return total_microseconds / 1_000_000.0


def summarize_profile(timing_samples: object, fusible_samples: object) -> dict[str, object]:
    """Combine unprofiled wall samples with separate objective-only profiler samples."""

    timing = summarize_timing_samples(timing_samples)
    if type(fusible_samples) is not tuple or not fusible_samples:
        raise ValueError("fusible samples must be a nonempty tuple")
    fusible = np.asarray(
        [
            _finite_float(value, f"fusible sample {index}", nonnegative=True)
            for index, value in enumerate(fusible_samples)
        ],
        dtype=np.float64,
    )
    wall = np.asarray(
        [
            _validate_timing_sample(sample, index)["step_wall_seconds"]
            for index, sample in enumerate(timing_samples)
        ],
        dtype=np.float64,
    )
    if float(fusible.max()) > float(wall.max()) * (1.0 + 1e-9):
        raise ValueError("fusible objective time exceeds the measured step")
    fusible_mean = float(fusible.mean(dtype=np.float64))
    wall_mean = float(wall.mean(dtype=np.float64))
    fraction = fusible_mean / wall_mean
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    wall_draws = wall[generator.integers(0, wall.size, size=(BOOTSTRAP_REPLICATES, wall.size))]
    fusible_draws = fusible[
        generator.integers(
            0,
            fusible.size,
            size=(BOOTSTRAP_REPLICATES, fusible.size),
        )
    ]
    bootstrap = fusible_draws.mean(axis=1) / wall_draws.mean(axis=1)
    lower = float(np.quantile(bootstrap, 0.025))
    return {
        **timing,
        "fusible_non_backbone_seconds": fusible_mean,
        "fusible_non_backbone_fraction": fraction,
        "fusible_fraction_bootstrap_lower_95": lower,
        "kernel_gate_threshold": KERNEL_GATE_THRESHOLD,
        "kernel_gate_passed": bool(lower >= KERNEL_GATE_THRESHOLD),
    }


def _profile_samples(payload: object, expected_init: str) -> tuple[tuple, tuple]:
    if type(payload) is not dict:
        raise TypeError("ABBA profile must be an object")
    required = (
        "schema_version",
        "classifier_init",
        "warmup_steps",
        "measure_steps",
        "profiler_steps",
        "timing_samples",
        "fusible_samples",
        "summary",
    )
    if any(key not in payload for key in required):
        raise ValueError("ABBA profile schema differs")
    if (
        payload["schema_version"] != "unicom-training-step-profile-v1"
        or payload["classifier_init"] != expected_init
        or payload["warmup_steps"] != WARMUP_STEPS
        or payload["measure_steps"] != MEASURE_STEPS
        or payload["profiler_steps"] != PROFILER_STEPS
    ):
        raise ValueError("ABBA profile binding differs")
    timing = payload["timing_samples"]
    fusible = payload["fusible_samples"]
    if (
        type(timing) is not list
        or len(timing) != MEASURE_STEPS
        or type(fusible) is not list
        or len(fusible) != PROFILER_STEPS
    ):
        raise ValueError("ABBA profile sample counts differ")
    timing_tuple = tuple(timing)
    fusible_tuple = tuple(fusible)
    recomputed = summarize_profile(timing_tuple, fusible_tuple)
    if payload["summary"] != recomputed:
        raise ValueError("ABBA profile summary differs")
    return timing_tuple, fusible_tuple


def aggregate_abba_profiles(profiles: object) -> dict[str, dict[str, object]]:
    """Pool two fresh checkpoint reloads per arm in fixed A-B-B-A order."""

    if type(profiles) is not tuple or len(profiles) != 4:
        raise ValueError("ABBA requires exactly four profiles")
    expected = ("random", "imprinted", "imprinted", "random")
    validated = tuple(
        _profile_samples(profile, classifier_init)
        for profile, classifier_init in zip(profiles, expected, strict=True)
    )
    result: dict[str, dict[str, object]] = {}
    for classifier_init, indices in (("random", (0, 3)), ("imprinted", (1, 2))):
        timing = validated[indices[0]][0] + validated[indices[1]][0]
        fusible = validated[indices[0]][1] + validated[indices[1]][1]
        result[classifier_init] = summarize_profile(timing, fusible)
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-checkpoint", required=True, type=Path)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=MEASURE_STEPS)
    parser.add_argument("--profiler-steps", type=int, default=PROFILER_STEPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(arguments)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trainer(source: Path):
    source = source.resolve()
    spec = importlib.util.spec_from_file_location("_unicom_profile_trainer", source)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the UniCOM trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source:
        raise ValueError("loaded trainer source differs")
    return module


def _validate_counts(args: argparse.Namespace) -> None:
    for name in ("warmup_steps", "measure_steps", "profiler_steps"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(args.bootstrap_seed) is not int or args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed differs from the registered profiler")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = (
        "epoch",
        "model",
        "classifier",
        "ema",
        "optimizer",
        "scheduler",
        "scaler",
        "mask_generator",
        "torch_rng_state",
        "cuda_rng_states",
        "selection_holdout",
        "training_protocol",
        "history",
    )
    if type(payload) is not dict or tuple(payload) != expected:
        raise ValueError("training checkpoint schema differs")
    if type(payload["training_protocol"]) is not dict:
        raise ValueError("training checkpoint protocol differs")
    if type(payload["selection_holdout"]) is not dict:
        raise ValueError("training checkpoint holdout differs")
    return payload


def _restore_checkpoint_payload(payload: dict[str, Any], state: dict[str, Any]) -> int:
    import torch

    protocol = state["protocol"]
    raw_model = state["raw_model"]
    classifier = state["classifier"]
    step_ema = state["step_ema"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    mask_generator = state["mask_generator"]
    if payload["selection_holdout"] != state["holdout"]:
        raise ValueError("training checkpoint holdout differs")
    if payload["training_protocol"] != protocol:
        raise ValueError("training checkpoint protocol differs")
    raw_model.load_state_dict(payload["model"], strict=True)
    value = payload["classifier"]
    if type(value) is not torch.Tensor or value.shape != classifier.shape:
        raise ValueError("training checkpoint classifier differs")
    with torch.no_grad():
        classifier.copy_(value)
    step_ema.load_state_dict(payload["ema"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is None:
        if payload["scaler"] is not None:
            raise ValueError("training checkpoint scaler differs")
    else:
        scaler.load_state_dict(payload["scaler"])
    mask_generator.set_state(payload["mask_generator"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload["cuda_rng_states"]
    if type(cuda_states) is not list or not cuda_states:
        raise ValueError("training checkpoint CUDA RNG state differs")
    torch.cuda.set_rng_state_all(cuda_states)
    epoch = payload["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("training checkpoint epoch differs")
    return epoch


def _build_replay_state(args: argparse.Namespace, trainer):
    import torch

    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import experiment_stream_seed, identity_holdout

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniCOM step replay")
    checkpoint = _load_checkpoint(args.run_checkpoint)
    protocol = dict(checkpoint["training_protocol"])
    holdout = dict(checkpoint["selection_holdout"])
    trainer_source = Path(trainer.__file__).resolve()
    if protocol.get("trainer_sha256") != _sha256_file(trainer_source):
        raise ValueError("checkpoint trainer source differs")
    if protocol.get("initial_checkpoint_sha256") != _sha256_file(args.initial_checkpoint):
        raise ValueError("initial UniCOM checkpoint differs")
    if protocol.get("unicom_revision") != trainer._git_revision(args.unicom_checkout):
        raise ValueError("UniCOM checkout revision differs")

    seed = protocol["seed"]
    if type(seed) is not int:
        raise TypeError("training seed differs")
    trainer._seed_process(seed)
    device = torch.device("cuda")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, _query, _gallery, labels = identity_holdout(
        train_records,
        fraction=protocol["holdout_fraction"],
        seed=protocol["holdout_seed"],
    )
    raw_model, _eval_transform = trainer._load_official_model(
        args.unicom_checkout,
        args.initial_checkpoint,
    )
    raw_model = raw_model.to(device)
    train_model = (
        torch.compile(raw_model, mode="reduce-overhead")
        if protocol["compile"]
        else raw_model
    )

    checkpoint_shape = checkpoint["classifier"].shape
    classifier = torch.nn.Parameter(
        torch.empty(checkpoint_shape, device=device, dtype=torch.float32)
    )
    step_ema = trainer.StepEMA(raw_model, classifier)
    optimizer = trainer.build_optimizer(
        raw_model,
        classifier,
        learning_rate=protocol["learning_rate"],
        classifier_learning_rate=protocol["classifier_learning_rate"],
        fused=protocol["fused"],
    )
    sampler = trainer.PaddedEpochSampler(
        size=len(optimization),
        batch_size=protocol["batch_size"],
        seed=seed,
    )
    data_generator = torch.Generator().manual_seed(experiment_stream_seed(seed, 2_000))
    loader = torch.utils.data.DataLoader(
        trainer.InshopTrainDataset(
            optimization,
            labels,
            trainer.build_train_transform(336),
        ),
        batch_size=protocol["batch_size"],
        sampler=sampler,
        num_workers=protocol["workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=trainer._seed_worker,
        generator=data_generator,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[protocol["learning_rate"], protocol["classifier_learning_rate"]],
        steps_per_epoch=len(loader),
        epochs=protocol["epochs"],
        pct_start=0.1,
    )
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(seed, 3_000)
    )
    scaler = (
        None
        if protocol["bf16"]
        else torch.amp.GradScaler("cuda", growth_interval=200)
    )
    state = {
        "raw_model": raw_model,
        "train_model": train_model,
        "classifier": classifier,
        "step_ema": step_ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loader": loader,
        "mask_generator": mask_generator,
        "scaler": scaler,
        "device": device,
        "protocol": protocol,
        "holdout": holdout,
    }
    start_epoch = _restore_checkpoint_payload(checkpoint, state)
    del checkpoint
    sampler.set_epoch(start_epoch)
    trainer._seed_training_loader(loader, seed=seed, epoch=start_epoch)
    step_ema.register_step_hook(optimizer)
    raw_model.train()
    state["checkpoint_epoch"] = start_epoch
    return state


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _cuda_event(torch):
    return torch.cuda.Event(enable_timing=True)


def _training_step(state: dict[str, Any], batch, *, measured: bool) -> dict[str, float] | None:
    import torch

    protocol = state["protocol"]
    train_model = state["train_model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    device = state["device"]
    boundaries = [_cuda_event(torch) for _ in range(len(COMPONENT_KEYS) + 1)]
    wall_start = time.perf_counter()
    boundaries[0].record()
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device=device, dtype=torch.int64, non_blocking=True)
    boundaries[1].record()
    optimizer.zero_grad(set_to_none=True)
    boundaries[2].record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=protocol["bf16"]):
        embeddings = train_model(images)
    boundaries[3].record()
    embeddings = embeddings.float()
    masks = trainer_objective_masks(
        state,
        dimension=embeddings.shape[1],
    )
    loss = trainer_loss(state, embeddings, labels, masks)
    boundaries[4].record()

    head_boundary_recorded = False

    def mark_head_backward(gradient):
        nonlocal head_boundary_recorded
        if not head_boundary_recorded:
            boundaries[5].record()
            head_boundary_recorded = True
        return gradient

    hook = embeddings.register_hook(mark_head_backward)
    try:
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
    finally:
        hook.remove()
    if not head_boundary_recorded:
        raise RuntimeError("embedding-gradient boundary was not observed")
    boundaries[6].record()
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    boundaries[7].record()
    float(loss.detach())
    boundaries[8].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    if not measured:
        return None
    components = {
        key: float(boundaries[index].elapsed_time(boundaries[index + 1])) / 1_000.0
        for index, key in enumerate(COMPONENT_KEYS)
    }
    return {
        "step_wall_seconds": float(wall),
        "cuda_step_seconds": math.fsum(components.values()),
        **components,
    }


def trainer_objective_masks(state: dict[str, Any], *, dimension: int):
    trainer = state["trainer"]
    protocol = state["protocol"]
    return trainer.objective_masks(
        protocol["objective"],
        dimension=dimension,
        selected=protocol["selected_features"],
        generator=state["mask_generator"],
        device=state["device"],
    )


def trainer_loss(state: dict[str, Any], embeddings, labels, masks):
    trainer = state["trainer"]
    protocol = state["protocol"]
    return trainer.sharded_mask_arcface_loss(
        embeddings,
        state["classifier"],
        labels,
        masks,
        margin=protocol["margin"],
        scale=protocol["scale"],
    )


def _profile_objective(state: dict[str, Any], embeddings, labels) -> float:
    import torch

    detached = embeddings.detach().requires_grad_(True)
    classifier = state["classifier"].detach().requires_grad_(True)
    profile_state = dict(state)
    profile_state["classifier"] = classifier
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof,
        torch.profiler.record_function(OBJECTIVE_MARKER),
    ):
        masks = trainer_objective_masks(profile_state, dimension=detached.shape[1])
        loss = trainer_loss(profile_state, detached, labels, masks)
        loss.backward()
    torch.cuda.synchronize()
    return fusible_nonbackbone_seconds(tuple(prof.events()))


def replay_profile(args: argparse.Namespace) -> dict[str, object]:
    import torch

    _validate_counts(args)
    trainer_source = Path(__file__).with_name("train_unicom_inshop.py")
    trainer = _load_trainer(trainer_source)
    state = _build_replay_state(args, trainer)
    state["trainer"] = trainer
    iterator = iter(state["loader"])
    timing_rows: list[dict[str, float]] = []
    try:
        for index in range(args.warmup_steps + args.measure_steps):
            batch, iterator = _next_batch(iterator, state["loader"])
            row = _training_step(state, batch, measured=index >= args.warmup_steps)
            if row is not None:
                timing_rows.append(row)

        fusible: list[float] = []
        for _index in range(args.profiler_steps):
            batch, iterator = _next_batch(iterator, state["loader"])
            images, labels = batch
            images = images.to(state["device"], non_blocking=True)
            labels = labels.to(state["device"], dtype=torch.int64, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=state["protocol"]["bf16"],
            ):
                embeddings = state["train_model"](images).float()
            fusible.append(_profile_objective(state, embeddings, labels))
    finally:
        state["step_ema"].release_step_hook()
    summary = summarize_profile(tuple(timing_rows), tuple(fusible))
    return {
        "schema_version": "unicom-training-step-profile-v1",
        "run_checkpoint": str(args.run_checkpoint.resolve()),
        "run_checkpoint_sha256": _sha256_file(args.run_checkpoint),
        "trainer_sha256": _sha256_file(trainer_source),
        "checkpoint_epoch": state["checkpoint_epoch"],
        "classifier_init": state["protocol"]["classifier_init"],
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "profiler_steps": args.profiler_steps,
        "timing_samples": timing_rows,
        "fusible_samples": fusible,
        "summary": summary,
        "runtime": {
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "cuda_version": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(state["device"]),
        },
    }


def write_json_atomic(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"profile output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"profile temporary already exists: {temporary}")
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        payload = replay_profile(args)
        write_json_atomic(args.output, payload)
    except Exception as error:
        print(f"profiling failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "kernel_gate_passed": payload["summary"]["kernel_gate_passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
