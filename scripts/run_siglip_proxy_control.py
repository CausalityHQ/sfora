#!/usr/bin/env python3
"""Run the authenticated SigLIP-so400m pooled Proxy Anchor control."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.token_set_screen import F1_TRAIN_CLASSES


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON refuses nonfinite floats")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be concrete strings")
            _validate_json_value(item)
        return
    raise TypeError(f"canonical JSON refuses {type(value).__name__}")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Return strict sorted compact JSON with exactly one trailing newline."""

    if type(payload) is not dict:
        raise TypeError("canonical payload must be a concrete object")
    _validate_json_value(payload)
    if "claim_eligible" in payload and payload["claim_eligible"] is not False:
        raise ValueError("control receipts must be concretely claim-ineligible")
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _write_new(path: Path, payload: bytes) -> None:
    """Publish bytes without overwriting an existing authority path."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)


def _learning_rate_multiplier(
    config: SiglipProxyControlConfig,
    *,
    step: int,
    steps_per_epoch: int,
) -> float:
    """Resolve the exact warmup-inclusive, epoch-bound schedule multiplier."""

    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative concrete integer")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be a positive concrete integer")
    warmup_steps = config.warmup_epochs * steps_per_epoch
    warmup = min(1.0, (step + 1) / warmup_steps)
    completed_epoch = step // steps_per_epoch
    decays = sum(completed_epoch >= epoch for epoch in config.decay_epochs)
    return warmup * config.decay_gamma**decays


def _seed64(*parts: object) -> int:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


@dataclass(frozen=True)
class SamplerState:
    """Persistent per-class permutation cycles and cursors."""

    cycles: tuple[int, ...]
    positions: tuple[int, ...]

    @classmethod
    def initial(cls) -> SamplerState:
        return cls(cycles=(0,) * 49, positions=(0,) * 49)


def _permutation(values: list[int], *, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[int(position)] for position in order]


def _build_epoch_batches(
    *,
    example_ids: tuple[str, ...],
    labels: torch.Tensor,
    seed: int,
    epoch: int,
    steps_per_epoch: int,
    state: SamplerState,
) -> tuple[tuple[tuple[int, ...], ...], SamplerState]:
    """Build exact 30-class/4-image logical batches and advance sampler cursors."""

    if len(example_ids) != labels.numel() or labels.ndim != 1:
        raise ValueError("example IDs and labels differ")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("sampler labels must use an integer dtype")
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("sampler seed and epoch must be concrete integers")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("sampler example IDs must be unique")
    if len(state.cycles) != 49 or len(state.positions) != 49:
        raise ValueError("sampler state must bind all 49 optimization classes")
    grouped: dict[int, list[int]] = {label: [] for label in sorted(F1_TRAIN_CLASSES)}
    for index, label_tensor in enumerate(labels.detach().cpu().to(torch.int64)):
        label = int(label_tensor)
        if label not in grouped:
            raise ValueError("sampler labels differ from the optimization band")
        grouped[label].append(index)
    for label in grouped:
        grouped[label].sort(key=example_ids.__getitem__)
        if len(grouped[label]) < 4:
            raise ValueError("every optimization class requires at least four examples")

    cycles = list(state.cycles)
    positions = list(state.positions)
    batches: list[tuple[int, ...]] = []
    classes = sorted(F1_TRAIN_CLASSES)
    for step in range(steps_per_epoch):
        selected_classes = _permutation(
            classes,
            seed=_seed64("classes", seed, epoch, step),
        )[:30]
        batch: list[int] = []
        for label in selected_classes:
            candidates = grouped[label]
            if positions[label] + 4 > len(candidates):
                cycles[label] += 1
                positions[label] = 0
            ordered = _permutation(
                candidates,
                seed=_seed64("examples", seed, label, cycles[label]),
            )
            start = positions[label]
            batch.extend(ordered[start : start + 4])
            positions[label] += 4
        if len(batch) != 120 or len(set(batch)) != 120:
            raise RuntimeError("sampler failed to construct a unique logical batch")
        batches.append(tuple(batch))
    return tuple(batches), SamplerState(cycles=tuple(cycles), positions=tuple(positions))


def _optimizer_groups(
    model: PooledProxyAnchorModel,
    config: SiglipProxyControlConfig,
) -> list[dict[str, Any]]:
    """Partition every trainable parameter once by family and decay authority."""

    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "proxies":
            learning_rate = config.proxy_learning_rate
            decay = 0.0
        elif name.startswith("projection."):
            learning_rate = config.projection_learning_rate
            decay = 0.0 if name.endswith(".bias") else config.weight_decay
        elif name.startswith("tower."):
            learning_rate = config.tower_learning_rate
            decay = (
                0.0
                if name.endswith(".bias") or ".norm." in name.lower()
                else config.weight_decay
            )
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
        grouped.setdefault((learning_rate, decay), []).append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), parameters in sorted(grouped.items())
    ]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    duplicated = len(parameter_ids) != len(set(parameter_ids))
    incomplete = sorted(parameter_ids) != sorted(expected_ids)
    if duplicated or incomplete:
        raise RuntimeError("optimizer groups do not partition trainable parameters exactly once")
    return groups
