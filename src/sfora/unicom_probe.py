"""Training-data-only classifier-probe screen for the UniCOM recipe."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_training import (
    experiment_stream_seed,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_logits,
    sharded_mask_arcface_loss,
)

PROBE_STEPS = 512
PROBE_SPLIT_SEED = 23_000
PROBE_BATCH_SIZE = 128
PROBE_BATCH_SEED = 23_001
PROBE_MASK_SEED = 23_002
PROBE_SHARDS = 8
PROBE_SELECTED_FEATURES = 512


@dataclass(frozen=True)
class ProbeFit:
    head: torch.Tensor
    initial_loss: float
    final_loss: float
    steps: int


@dataclass(frozen=True)
class ProbeMetrics:
    mean_loss: float
    accuracy: float
    correct_count: int
    observation_count: int


@dataclass(frozen=True)
class ProbeDecision:
    status: str
    relative_validation_loss_reduction: float
    accuracy_delta: float
    predicates: dict[str, bool]


@dataclass(frozen=True)
class ProbeSplit:
    fitting: tuple[InshopRecord, ...]
    validation: tuple[InshopRecord, ...]
    validation_group_represented: tuple[bool, ...]
    validation_class_count: int
    singleton_class_count: int


_INSHOP_FILENAME = re.compile(r"^(?P<series>[^_]+)_[0-9]+_[^.]+\.jpg$")


def _acquisition_series(row: InshopRecord) -> str:
    match = _INSHOP_FILENAME.fullmatch(row.image_path.name)
    if match is None:
        raise ValueError("probe In-Shop filename differs")
    return match.group("series")


def split_probe_records(
    records: tuple[InshopRecord, ...],
    labels: Mapping[str, int],
    *,
    seed: int = PROBE_SPLIT_SEED,
) -> ProbeSplit:
    """Draw one validation row per nonsingleton class and retain singletons in fit."""

    if (
        type(records) is not tuple
        or not records
        or type(labels) is not dict
        or not labels
        or tuple(labels.values()) != tuple(range(len(labels)))
        or any(type(label) is not str or not label for label in labels)
        or any(type(row) is not InshopRecord or row.split != "train" for row in records)
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("probe record inventory differs")
    grouped: dict[str, list[InshopRecord]] = defaultdict(list)
    for row in records:
        if row.label not in labels:
            raise ValueError("probe record label differs")
        grouped[row.label].append(row)
    if set(grouped) != set(labels):
        raise ValueError("probe classes differ")

    fitting: list[InshopRecord] = []
    validation: list[InshopRecord] = []
    represented: list[bool] = []
    singleton_count = 0
    generator = np.random.Generator(
        np.random.PCG64(experiment_stream_seed(0, seed))
    )
    for label in labels:
        ordered = sorted(grouped[label], key=lambda row: str(row.image_path))
        if len(ordered) == 1:
            singleton_count += 1
            fitting.extend(ordered)
            continue
        validation_index = int(generator.integers(0, len(ordered)))
        validation_row = ordered[validation_index]
        fitting_rows = [
            row for index, row in enumerate(ordered) if index != validation_index
        ]
        fitting.extend(fitting_rows)
        validation.append(validation_row)
        validation_series = _acquisition_series(validation_row)
        represented.append(
            any(_acquisition_series(row) == validation_series for row in fitting_rows)
        )
    return ProbeSplit(
        fitting=tuple(fitting),
        validation=tuple(validation),
        validation_group_represented=tuple(represented),
        validation_class_count=len(validation),
        singleton_class_count=singleton_count,
    )


def class_mean_head(
    features: torch.Tensor, labels: torch.Tensor, class_count: int
) -> torch.Tensor:
    """Return normalized FP64 class means at the registered random-head norm."""

    if (
        type(features) is not torch.Tensor
        or features.dtype != torch.float32
        or features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] == 0
        or not features.is_contiguous()
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.ndim != 1
        or labels.shape[0] != features.shape[0]
        or labels.device != features.device
        or type(class_count) is not int
        or class_count <= 0
    ):
        raise ValueError("probe feature inventory differs")
    if not torch.isfinite(features).all():
        raise ValueError("probe features must be finite")
    feature_norms = torch.linalg.vector_norm(features, dim=1)
    if torch.any(feature_norms == 0.0):
        raise ValueError("probe feature has zero norm")
    if torch.any(labels < 0) or torch.any(labels >= class_count):
        raise ValueError("probe label is outside the class range")

    normalized = F.normalize(features, dim=1).double()
    sums = torch.zeros(class_count, features.shape[1], dtype=torch.float64, device=features.device)
    sums.index_add_(0, labels, normalized)
    counts = torch.bincount(labels, minlength=class_count)
    if counts.shape[0] != class_count or torch.any(counts == 0):
        raise ValueError("probe class is empty")
    means = sums / counts.to(dtype=torch.float64)[:, None]
    mean_norms = torch.linalg.vector_norm(means, dim=1)
    if not torch.isfinite(mean_norms).all() or torch.any(mean_norms == 0.0):
        raise ValueError("probe class mean differs")
    return F.normalize(means, dim=1).float() * (0.01 * math.sqrt(features.shape[1]))


def _validate_probe_tensors(
    features: torch.Tensor, labels: torch.Tensor, initial: torch.Tensor
) -> float:
    if (
        type(features) is not torch.Tensor
        or features.dtype != torch.float32
        or features.ndim != 2
        or not features.is_contiguous()
        or features.shape[0] == 0
        or features.shape[1] < PROBE_SELECTED_FEATURES
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.ndim != 1
        or labels.shape[0] != features.shape[0]
        or labels.device != features.device
        or type(initial) is not torch.Tensor
        or initial.dtype != torch.float32
        or initial.ndim != 2
        or initial.shape[1] != features.shape[1]
        or initial.device != features.device
        or not initial.is_contiguous()
        or initial.shape[0] < PROBE_SHARDS
    ):
        raise ValueError("spherical probe tensor contract differs")
    if not torch.isfinite(features).all() or not torch.isfinite(initial).all():
        raise ValueError("spherical probe tensors must be finite")
    if torch.any(labels < 0) or torch.any(labels >= initial.shape[0]):
        raise ValueError("spherical probe label is outside the class range")
    counts = torch.bincount(labels, minlength=initial.shape[0])
    if counts.shape[0] != initial.shape[0] or torch.any(counts == 0):
        raise ValueError("spherical probe class is empty")
    row_norms = torch.linalg.vector_norm(initial, dim=1)
    if torch.any(row_norms == 0.0) or not torch.isfinite(row_norms).all():
        raise ValueError("spherical probe initial row norm differs")
    target_norm = float(row_norms[0])
    if not torch.allclose(
        row_norms,
        torch.full_like(row_norms, target_norm),
        rtol=2e-6,
        atol=2e-7,
    ):
        raise ValueError("spherical probe initial row norms differ")
    return target_norm


def _diagnostic_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
    *,
    batch_size: int,
    batch_seed: int,
    mask_seed: int,
) -> float:
    indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=batch_size,
        epoch=0,
        seed=batch_seed,
        shards=PROBE_SHARDS,
    )[:batch_size]
    batch_indices = torch.tensor(indices, dtype=torch.int64, device=features.device)
    generator = torch.Generator(device=features.device).manual_seed(mask_seed)
    masks = sample_shard_masks(
        dimension=features.shape[1],
        selected=PROBE_SELECTED_FEATURES,
        shards=PROBE_SHARDS,
        generator=generator,
        device=features.device,
    )
    with torch.no_grad():
        loss = sharded_mask_arcface_loss(
            features.index_select(0, batch_indices),
            head,
            labels.index_select(0, batch_indices),
            masks,
        )
    return float(loss)


def fit_spherical_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
) -> ProbeFit:
    """Optimize ArcFace directions while projecting every row to its initial norm."""

    target_norm = _validate_probe_tensors(features, labels, initial)
    if (
        type(steps) is not int
        or steps <= 0
        or type(batch_size) is not int
        or batch_size <= 0
        or batch_size % PROBE_SHARDS != 0
        or any(type(seed) is not int or seed < 0 for seed in (batch_seed, mask_seed))
    ):
        raise ValueError("spherical probe schedule differs")

    initial_loss = _diagnostic_loss(
        features,
        labels,
        initial,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
    )
    head = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW(
        [head],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    mask_generator = torch.Generator(device=features.device).manual_seed(mask_seed)
    completed = 0
    epoch = 0
    while completed < steps:
        epoch_indices = padded_epoch_indices(
            size=features.shape[0],
            global_batch=batch_size,
            epoch=epoch,
            seed=batch_seed,
            shards=PROBE_SHARDS,
        )
        for start in range(0, len(epoch_indices), batch_size):
            indices = torch.tensor(
                epoch_indices[start : start + batch_size],
                dtype=torch.int64,
                device=features.device,
            )
            masks = sample_shard_masks(
                dimension=features.shape[1],
                selected=PROBE_SELECTED_FEATURES,
                shards=PROBE_SHARDS,
                generator=mask_generator,
                device=features.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = sharded_mask_arcface_loss(
                features.index_select(0, indices),
                head,
                labels.index_select(0, indices),
                masks,
            )
            if not torch.isfinite(loss):
                raise ValueError("spherical probe loss is nonfinite")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norms = torch.linalg.vector_norm(head, dim=1)
                if not torch.isfinite(norms).all() or torch.any(norms == 0.0):
                    raise ValueError("spherical probe updated row norm differs")
                head.mul_((target_norm / norms)[:, None])
            completed += 1
            if completed == steps:
                break
        epoch += 1

    result = head.detach().contiguous()
    final_loss = _diagnostic_loss(
        features,
        labels,
        result,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
    )
    return ProbeFit(
        head=result,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=completed,
    )


def evaluate_probe_heads(
    features: torch.Tensor,
    labels: torch.Tensor,
    heads: Mapping[str, torch.Tensor],
    *,
    mask_sets: int = 64,
    mask_seed: int = 23_003,
) -> dict[str, ProbeMetrics]:
    """Evaluate registered heads over fixed masked ArcFace observations."""

    if type(heads) is not dict or tuple(heads) != ("class_mean", "spherical_probe"):
        raise ValueError("probe evaluation head order differs")
    _validate_probe_tensors(features, labels, heads["class_mean"])
    _validate_probe_tensors(features, labels, heads["spherical_probe"])
    if (
        type(mask_sets) is not int
        or mask_sets <= 0
        or type(mask_seed) is not int
        or mask_seed < 0
    ):
        raise ValueError("probe evaluation schedule differs")

    loss_totals = {name: [] for name in heads}
    correct = {name: 0 for name in heads}
    generator = torch.Generator(device=features.device).manual_seed(mask_seed)
    for _index in range(mask_sets):
        masks = sample_shard_masks(
            dimension=features.shape[1],
            selected=PROBE_SELECTED_FEATURES,
            shards=PROBE_SHARDS,
            generator=generator,
            device=features.device,
        )
        for name, head in heads.items():
            with torch.no_grad():
                logits = sharded_mask_arcface_logits(features, head, labels, masks)
                losses = F.cross_entropy(logits, labels, reduction="none")
                loss_totals[name].append(float(losses.double().sum()))
                correct[name] += int(torch.count_nonzero(logits.argmax(dim=1) == labels))

    observations = labels.numel() * mask_sets
    return {
        name: ProbeMetrics(
            mean_loss=math.fsum(loss_totals[name]) / observations,
            accuracy=correct[name] / observations,
            correct_count=correct[name],
            observation_count=observations,
        )
        for name in heads
    }


def _validate_metrics(value: ProbeMetrics, name: str) -> None:
    if (
        type(value) is not ProbeMetrics
        or type(value.mean_loss) is not float
        or not math.isfinite(value.mean_loss)
        or value.mean_loss <= 0.0
        or type(value.accuracy) is not float
        or not math.isfinite(value.accuracy)
        or not 0.0 <= value.accuracy <= 1.0
        or type(value.correct_count) is not int
        or type(value.observation_count) is not int
        or value.observation_count <= 0
        or not 0 <= value.correct_count <= value.observation_count
        or value.accuracy != value.correct_count / value.observation_count
    ):
        raise ValueError(f"{name} probe metrics differ")


def probe_decision(
    *,
    initial_fit_loss: float,
    final_fit_loss: float,
    class_mean: ProbeMetrics,
    spherical_probe: ProbeMetrics,
    row_norm_min: float,
    row_norm_max: float,
) -> ProbeDecision:
    """Apply the prospective direction-screen promotion rule."""

    scalar_values = (initial_fit_loss, final_fit_loss, row_norm_min, row_norm_max)
    if any(type(value) is not float or not math.isfinite(value) for value in scalar_values):
        raise ValueError("probe decision scalar differs")
    if initial_fit_loss <= 0.0 or final_fit_loss <= 0.0 or row_norm_min <= 0.0:
        raise ValueError("probe decision scalar must be positive")
    if row_norm_max < row_norm_min:
        raise ValueError("probe row-norm extrema differ")
    _validate_metrics(class_mean, "class_mean")
    _validate_metrics(spherical_probe, "spherical_probe")
    if class_mean.observation_count != spherical_probe.observation_count:
        raise ValueError("probe metric observation counts differ")

    relative_reduction = (class_mean.mean_loss - spherical_probe.mean_loss) / class_mean.mean_loss
    accuracy_delta = spherical_probe.accuracy - class_mean.accuracy
    target_norm = 0.01 * math.sqrt(768.0)
    predicates = {
        "fit_loss_decreased": final_fit_loss < initial_fit_loss,
        "validation_loss_reduction": relative_reduction >= 0.01,
        "validation_accuracy_noninferior": accuracy_delta >= 0.0,
        "row_norms_match": math.isclose(
            row_norm_min, target_norm, rel_tol=2e-6, abs_tol=2e-7
        )
        and math.isclose(row_norm_max, target_norm, rel_tol=2e-6, abs_tol=2e-7),
    }
    return ProbeDecision(
        status="PROMOTE" if all(predicates.values()) else "CLOSE_DIRECTION",
        relative_validation_loss_reduction=relative_reduction,
        accuracy_delta=accuracy_delta,
        predicates=predicates,
    )
