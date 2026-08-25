"""Training-data-only classifier-probe screen for the UniCOM recipe."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_proxy_muon import (
    MuonTrace,
    PrecisionMuon,
    build_head_optimizer,
    trace_builtin_muon_step,
    trace_precision_muon_step,
)
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
PROBE_DIAGNOSTIC_SEED = 23_004
PROBE_GRADIENT_SEED = 23_005
PROBE_LEARNING_RATE = 1e-4
PROBE_SHARDS = 8
PROBE_SELECTED_FEATURES = 512
PROBE_VALIDATION_IMAGES = 3_188
PROBE_IDENTITY_T_CRITICAL_DF3187 = 1.9607086212236648


@dataclass(frozen=True)
class ProbeFit:
    head: torch.Tensor
    initial_loss: float
    final_loss: float
    steps: int
    row_cosine_min: float
    row_cosine_p05: float
    row_cosine_median: float
    row_cosine_mean: float


@dataclass(frozen=True)
class ProbeMetrics:
    mean_loss: float
    accuracy: float
    correct_count: int
    observation_count: int
    per_mask_mean_losses: tuple[float, ...]
    per_mask_represented_mean_losses: tuple[float, ...]
    per_mask_unrepresented_mean_losses: tuple[float, ...]
    per_image_mean_losses: tuple[float, ...]
    represented_mean_loss: float
    unrepresented_mean_loss: float


@dataclass(frozen=True)
class ProbeGradientMetrics:
    class_mean_l2: float
    spherical_probe_l2: float
    relative_l2_difference: float
    spherical_to_class_mean_l2_ratio: float
    cosine_min: float
    cosine_p05: float
    cosine_median: float
    cosine_mean: float
    zero_gradient_row_count: int


@dataclass(frozen=True)
class ProbeSeedStatistics:
    mean_paired_loss_delta: float
    paired_95_lower_bound: float
    identity_95_lower_bound: float
    positive_mask_count: int
    unrepresented_paired_95_lower_bound: float
    unrepresented_positive_mask_count: int
    accuracy_delta: float
    represented_loss_delta: float
    unrepresented_loss_delta: float


@dataclass(frozen=True)
class ProbeDecision:
    status: str
    per_seed_statistics: dict[int, ProbeSeedStatistics]
    per_seed_predicates: dict[int, dict[str, bool]]


@dataclass(frozen=True)
class ProbeSplit:
    fitting: tuple[InshopRecord, ...]
    validation: tuple[InshopRecord, ...]
    validation_group_represented: tuple[bool, ...]
    validation_class_count: int
    singleton_class_count: int
    excluded_same_series_count: int


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
    excluded_same_series_count = 0
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
        validation_series = _acquisition_series(validation_row)
        series = tuple(_acquisition_series(row) for row in ordered)
        if len(set(series)) > 1:
            fitting_rows = [
                row for row, row_series in zip(ordered, series, strict=True)
                if row_series != validation_series
            ]
            excluded_same_series_count += sum(
                row_series == validation_series for row_series in series
            ) - 1
        else:
            fitting_rows = [
                row for index, row in enumerate(ordered) if index != validation_index
            ]
        fitting.extend(fitting_rows)
        validation.append(validation_row)
        represented.append(
            any(_acquisition_series(row) == validation_series for row in fitting_rows)
        )
    return ProbeSplit(
        fitting=tuple(fitting),
        validation=tuple(validation),
        validation_group_represented=tuple(represented),
        validation_class_count=len(validation),
        singleton_class_count=singleton_count,
        excluded_same_series_count=excluded_same_series_count,
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
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    require_all_classes: bool = True,
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
        or type(require_all_classes) is not bool
    ):
        raise ValueError("spherical probe tensor contract differs")
    if not torch.isfinite(features).all() or not torch.isfinite(initial).all():
        raise ValueError("spherical probe tensors must be finite")
    if torch.any(labels < 0) or torch.any(labels >= initial.shape[0]):
        raise ValueError("spherical probe label is outside the class range")
    counts = torch.bincount(labels, minlength=initial.shape[0])
    if counts.shape[0] != initial.shape[0] or (
        require_all_classes and torch.any(counts == 0)
    ):
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


def diagnostic_panel_losses(
    features: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
    *,
    fit_seed: int,
    batch_size: int = PROBE_BATCH_SIZE,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
) -> tuple[float, ...]:
    """Return the registered 4-batch by 4-mask diagnostic panel."""

    _validate_probe_tensors(features, labels, head)
    if (
        type(fit_seed) is not int
        or fit_seed < 0
        or type(batch_size) is not int
        or batch_size <= 0
        or batch_size % PROBE_SHARDS != 0
        or type(diagnostic_seed) is not int
        or diagnostic_seed < 0
    ):
        raise ValueError("probe diagnostic panel schedule differs")
    seed = experiment_stream_seed(fit_seed, diagnostic_seed)
    epoch_indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=batch_size,
        epoch=0,
        seed=seed,
        shards=PROBE_SHARDS,
    )
    if len(epoch_indices) < 4 * batch_size:
        raise ValueError("probe diagnostic panel inventory differs")
    batches = tuple(
        torch.tensor(
            epoch_indices[start : start + batch_size],
            dtype=torch.int64,
            device=features.device,
        )
        for start in range(0, 4 * batch_size, batch_size)
    )
    generator = torch.Generator(device=features.device).manual_seed(seed)
    mask_sets = tuple(
        sample_shard_masks(
            dimension=features.shape[1],
            selected=PROBE_SELECTED_FEATURES,
            shards=PROBE_SHARDS,
            generator=generator,
            device=features.device,
        )
        for _ in range(4)
    )
    with torch.no_grad():
        losses = tuple(
            float(
                sharded_mask_arcface_loss(
                    features.index_select(0, batch),
                    head,
                    labels.index_select(0, batch),
                    masks,
                )
            )
            for batch in batches
            for masks in mask_sets
        )
    if len(losses) != 16 or any(not math.isfinite(loss) for loss in losses):
        raise ValueError("probe diagnostic panel differs")
    return losses


def _fit_probe_trajectory_with_optimizer(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    snapshot_steps: tuple[int, ...],
    optimizer_factory: Callable[[torch.nn.Parameter], torch.optim.Optimizer],
    trace_factory: Callable[[torch.nn.Parameter, torch.optim.Optimizer], object]
    | None,
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
    fit_seed: int = 0,
) -> tuple[ProbeFit, dict[int, torch.Tensor], dict[int, object | None]]:
    target_norm = _validate_probe_tensors(features, labels, initial)
    if (
        type(steps) is not int
        or steps <= 0
        or type(batch_size) is not int
        or batch_size <= 0
        or batch_size % PROBE_SHARDS != 0
        or any(
            type(seed) is not int or seed < 0
            for seed in (batch_seed, mask_seed, diagnostic_seed, fit_seed)
        )
    ):
        raise ValueError("spherical probe schedule differs")
    snapshots: dict[int, torch.Tensor] = {}
    traces: dict[int, object | None] = {}
    if 0 in snapshot_steps:
        snapshots[0] = initial.detach().clone().contiguous()
        traces[0] = None
    initial_loss = _diagnostic_loss(
        features,
        labels,
        initial,
        batch_size=batch_size,
        batch_seed=experiment_stream_seed(fit_seed, diagnostic_seed),
        mask_seed=experiment_stream_seed(fit_seed, diagnostic_seed),
    )
    head = torch.nn.Parameter(initial.detach().clone())
    if not callable(optimizer_factory) or (
        trace_factory is not None and not callable(trace_factory)
    ):
        raise ValueError("spherical probe optimizer callback differs")
    optimizer = optimizer_factory(head)
    if (
        not isinstance(optimizer, torch.optim.Optimizer)
        or len(optimizer.param_groups) != 1
        or len(optimizer.param_groups[0]["params"]) != 1
        or optimizer.param_groups[0]["params"][0] is not head
    ):
        raise ValueError("spherical probe optimizer differs")
    mask_generator = torch.Generator(device=features.device).manual_seed(
        experiment_stream_seed(fit_seed, mask_seed)
    )
    completed = 0
    epoch = 0
    while completed < steps:
        epoch_indices = padded_epoch_indices(
            size=features.shape[0],
            global_batch=batch_size,
            epoch=epoch,
            seed=experiment_stream_seed(fit_seed, batch_seed),
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
            producing_step = completed + 1
            trace = (
                trace_factory(head, optimizer)
                if trace_factory is not None and producing_step in snapshot_steps
                else None
            )
            optimizer.step()
            with torch.no_grad():
                norms = torch.linalg.vector_norm(head, dim=1)
                if not torch.isfinite(norms).all() or torch.any(norms == 0.0):
                    raise ValueError("spherical probe updated row norm differs")
                head.mul_((target_norm / norms)[:, None])
            completed += 1
            if completed in snapshot_steps and completed != steps:
                snapshots[completed] = head.detach().clone().contiguous()
                traces[completed] = trace
            if completed == steps:
                break
        epoch += 1

    result = head.detach().contiguous()
    if steps in snapshot_steps:
        snapshots[steps] = result
        traces[steps] = trace
    final_loss = _diagnostic_loss(
        features,
        labels,
        result,
        batch_size=batch_size,
        batch_seed=experiment_stream_seed(fit_seed, diagnostic_seed),
        mask_seed=experiment_stream_seed(fit_seed, diagnostic_seed),
    )
    cosine = F.cosine_similarity(result, initial, dim=1).clamp(-1.0, 1.0).double()
    fit = ProbeFit(
        head=result,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=completed,
        row_cosine_min=float(cosine.min()),
        row_cosine_p05=float(torch.quantile(cosine, 0.05)),
        row_cosine_median=float(torch.quantile(cosine, 0.5)),
        row_cosine_mean=float(cosine.mean()),
    )
    return fit, snapshots, traces


def _fit_spherical_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    snapshot_steps: tuple[int, ...],
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
    fit_seed: int = 0,
) -> tuple[ProbeFit, dict[int, torch.Tensor]]:
    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [head],
            lr=PROBE_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

    fit, snapshots, _traces = _fit_probe_trajectory_with_optimizer(
        features,
        labels,
        initial,
        snapshot_steps=snapshot_steps,
        optimizer_factory=optimizer_factory,
        trace_factory=None,
        steps=steps,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
        diagnostic_seed=diagnostic_seed,
        fit_seed=fit_seed,
    )
    return fit, snapshots


def fit_spherical_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
    fit_seed: int = 0,
) -> ProbeFit:
    """Optimize ArcFace directions while projecting every row to its initial norm."""

    fit, _snapshots = _fit_spherical_probe(
        features,
        labels,
        initial,
        snapshot_steps=(),
        steps=steps,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
        diagnostic_seed=diagnostic_seed,
        fit_seed=fit_seed,
    )
    return fit


def fit_spherical_probe_trajectory(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    snapshot_steps: tuple[int, ...],
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
    fit_seed: int = 0,
) -> tuple[ProbeFit, dict[int, torch.Tensor]]:
    """Fit one probe and retain registered optimizer-step snapshots."""

    return _fit_spherical_probe(
        features,
        labels,
        initial,
        snapshot_steps=snapshot_steps,
        steps=steps,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
        diagnostic_seed=diagnostic_seed,
        fit_seed=fit_seed,
    )


def fit_proxy_muon_trajectory(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    *,
    learning_rate: float,
    ns_dtype: torch.dtype = torch.bfloat16,
    snapshot_steps: tuple[int, ...],
    steps: int = PROBE_STEPS,
    batch_size: int = PROBE_BATCH_SIZE,
    batch_seed: int = PROBE_BATCH_SEED,
    mask_seed: int = PROBE_MASK_SEED,
    diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED,
    fit_seed: int = 0,
) -> tuple[ProbeFit, dict[int, torch.Tensor], dict[int, MuonTrace | None]]:
    """Fit one built-in ProxyMuon trajectory with producing-step traces."""

    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        if ns_dtype == torch.bfloat16:
            return build_head_optimizer(head, "proxy_muon", learning_rate)
        if ns_dtype == torch.float32:
            return PrecisionMuon(head, lr=learning_rate, ns_dtype=ns_dtype)
        raise ValueError("ProxyMuon trajectory precision differs")

    trace_factory = (
        trace_builtin_muon_step
        if ns_dtype == torch.bfloat16
        else trace_precision_muon_step
    )

    fit, snapshots, raw_traces = _fit_probe_trajectory_with_optimizer(
        features,
        labels,
        initial,
        snapshot_steps=snapshot_steps,
        optimizer_factory=optimizer_factory,
        trace_factory=trace_factory,
        steps=steps,
        batch_size=batch_size,
        batch_seed=batch_seed,
        mask_seed=mask_seed,
        diagnostic_seed=diagnostic_seed,
        fit_seed=fit_seed,
    )
    traces: dict[int, MuonTrace | None] = {}
    for step, trace in raw_traces.items():
        if trace is not None and type(trace) is not MuonTrace:
            raise ValueError("ProxyMuon trajectory trace differs")
        traces[step] = trace
    cpu_snapshots = {
        step: snapshot.detach().cpu().contiguous()
        for step, snapshot in snapshots.items()
    }
    cpu_head = (
        cpu_snapshots[steps]
        if steps in cpu_snapshots
        else fit.head.detach().cpu().contiguous()
    )
    return replace(fit, head=cpu_head), cpu_snapshots, traces


def _evaluate_probe_head_mapping(
    features: torch.Tensor,
    labels: torch.Tensor,
    heads: Mapping[str, torch.Tensor],
    *,
    validation_group_represented: tuple[bool, ...],
    mask_sets: int = 64,
    mask_seed: int = 23_003,
) -> dict[str, ProbeMetrics]:
    if type(heads) is not dict or not heads:
        raise ValueError("probe evaluation head order differs")
    for head in heads.values():
        _validate_probe_tensors(features, labels, head, require_all_classes=False)
    if (
        type(mask_sets) is not int
        or mask_sets <= 0
        or type(mask_seed) is not int
        or mask_seed < 0
        or type(validation_group_represented) is not tuple
        or len(validation_group_represented) != labels.numel()
        or any(type(value) is not bool for value in validation_group_represented)
        or not any(validation_group_represented)
        or all(validation_group_represented)
    ):
        raise ValueError("probe evaluation schedule differs")

    loss_totals = {name: [] for name in heads}
    represented_mask_losses = {name: [] for name in heads}
    unrepresented_mask_losses = {name: [] for name in heads}
    per_image_loss = {
        name: torch.zeros(labels.numel(), dtype=torch.float64, device=features.device)
        for name in heads
    }
    correct = {name: 0 for name in heads}
    represented = torch.tensor(
        validation_group_represented, dtype=torch.bool, device=features.device
    )
    generator = torch.Generator(device=features.device).manual_seed(
        experiment_stream_seed(0, mask_seed)
    )
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
                loss_totals[name].append(float(losses.double().mean()))
                represented_mask_losses[name].append(
                    float(losses[represented].double().mean())
                )
                unrepresented_mask_losses[name].append(
                    float(losses[~represented].double().mean())
                )
                per_image_loss[name].add_(losses.double())
                margin_free = sharded_mask_arcface_logits(
                    features, head, labels, masks, margin=0.0
                )
                correct[name] += int(
                    torch.count_nonzero(margin_free.argmax(dim=1) == labels)
                )

    observations = labels.numel() * mask_sets
    return {
        name: ProbeMetrics(
            mean_loss=math.fsum(loss_totals[name]) / mask_sets,
            accuracy=correct[name] / observations,
            correct_count=correct[name],
            observation_count=observations,
            per_mask_mean_losses=tuple(loss_totals[name]),
            per_mask_represented_mean_losses=tuple(represented_mask_losses[name]),
            per_mask_unrepresented_mean_losses=tuple(unrepresented_mask_losses[name]),
            per_image_mean_losses=tuple(
                float(value) for value in (per_image_loss[name] / mask_sets).cpu()
            ),
            represented_mean_loss=(
                math.fsum(represented_mask_losses[name]) / mask_sets
            ),
            unrepresented_mean_loss=(
                math.fsum(unrepresented_mask_losses[name]) / mask_sets
            ),
        )
        for name in heads
    }


def evaluate_probe_heads(
    features: torch.Tensor,
    labels: torch.Tensor,
    heads: Mapping[str, torch.Tensor],
    *,
    validation_group_represented: tuple[bool, ...],
    mask_sets: int = 64,
    mask_seed: int = 23_003,
) -> dict[str, ProbeMetrics]:
    """Evaluate the registered baseline/candidate pair over fixed masks."""

    if type(heads) is not dict or tuple(heads) != ("class_mean", "spherical_probe"):
        raise ValueError("probe evaluation head order differs")
    return _evaluate_probe_head_mapping(
        features,
        labels,
        heads,
        validation_group_represented=validation_group_represented,
        mask_sets=mask_sets,
        mask_seed=mask_seed,
    )


def evaluate_probe_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
    *,
    validation_group_represented: tuple[bool, ...],
    mask_sets: int = 64,
    mask_seed: int = 23_003,
) -> ProbeMetrics:
    """Evaluate one candidate head without replaying a stored baseline."""

    return _evaluate_probe_head_mapping(
        features,
        labels,
        {"candidate": head},
        validation_group_represented=validation_group_represented,
        mask_sets=mask_sets,
        mask_seed=mask_seed,
    )["candidate"]


def compare_probe_gradients(
    features: torch.Tensor,
    labels: torch.Tensor,
    heads: Mapping[str, torch.Tensor],
    *,
    fit_seed: int,
    batch_size: int = PROBE_BATCH_SIZE,
    diagnostic_seed: int = PROBE_GRADIENT_SEED,
) -> ProbeGradientMetrics:
    """Compare the cached-feature gradient fields induced by two classifier heads."""

    if type(heads) is not dict or tuple(heads) != ("class_mean", "spherical_probe"):
        raise ValueError("probe gradient head order differs")
    _validate_probe_tensors(features, labels, heads["class_mean"])
    _validate_probe_tensors(features, labels, heads["spherical_probe"])
    if (
        type(fit_seed) is not int
        or fit_seed < 0
        or type(batch_size) is not int
        or batch_size <= 0
        or batch_size % PROBE_SHARDS != 0
        or type(diagnostic_seed) is not int
        or diagnostic_seed < 0
    ):
        raise ValueError("probe gradient schedule differs")
    stream_seed = experiment_stream_seed(fit_seed, diagnostic_seed)
    indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=batch_size,
        epoch=0,
        seed=stream_seed,
        shards=PROBE_SHARDS,
    )[:batch_size]
    batch_indices = torch.tensor(indices, dtype=torch.int64, device=features.device)
    batch_labels = labels.index_select(0, batch_indices)
    generator = torch.Generator(device=features.device).manual_seed(stream_seed)
    masks = sample_shard_masks(
        dimension=features.shape[1],
        selected=PROBE_SELECTED_FEATURES,
        shards=PROBE_SHARDS,
        generator=generator,
        device=features.device,
    )
    gradients: list[torch.Tensor] = []
    for head in heads.values():
        batch = features.index_select(0, batch_indices).detach().clone().requires_grad_(True)
        loss = sharded_mask_arcface_loss(batch, head, batch_labels, masks)
        gradient = torch.autograd.grad(loss, batch)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("probe gradient is nonfinite")
        gradients.append(gradient)
    class_mean_gradient, probe_gradient = gradients
    class_mean_norm = torch.linalg.vector_norm(class_mean_gradient)
    probe_norm = torch.linalg.vector_norm(probe_gradient)
    if class_mean_norm == 0.0 or probe_norm == 0.0:
        raise ValueError("probe gradient field has zero norm")
    class_mean_row_norms = torch.linalg.vector_norm(class_mean_gradient, dim=1)
    probe_row_norms = torch.linalg.vector_norm(probe_gradient, dim=1)
    zero_mean = class_mean_row_norms == 0.0
    zero_probe = probe_row_norms == 0.0
    if not torch.equal(zero_mean, zero_probe):
        raise ValueError("probe per-sample gradient zero pattern differs")
    nonzero = ~zero_mean
    cosine = F.cosine_similarity(
        class_mean_gradient[nonzero], probe_gradient[nonzero], dim=1
    ).clamp(-1.0, 1.0).double()
    difference = torch.linalg.vector_norm(probe_gradient - class_mean_gradient)
    return ProbeGradientMetrics(
        class_mean_l2=float(class_mean_norm.double()),
        spherical_probe_l2=float(probe_norm.double()),
        relative_l2_difference=float(difference.double() / class_mean_norm.double()),
        spherical_to_class_mean_l2_ratio=float(
            probe_norm.double() / class_mean_norm.double()
        ),
        cosine_min=float(cosine.min()),
        cosine_p05=float(torch.quantile(cosine, 0.05)),
        cosine_median=float(torch.quantile(cosine, 0.5)),
        cosine_mean=float(cosine.mean()),
        zero_gradient_row_count=int(torch.count_nonzero(zero_mean)),
    )


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
        or type(value.per_mask_mean_losses) is not tuple
        or len(value.per_mask_mean_losses) != 64
        or type(value.per_mask_represented_mean_losses) is not tuple
        or len(value.per_mask_represented_mean_losses) != 64
        or type(value.per_mask_unrepresented_mean_losses) is not tuple
        or len(value.per_mask_unrepresented_mean_losses) != 64
        or type(value.per_image_mean_losses) is not tuple
        or len(value.per_image_mean_losses) != PROBE_VALIDATION_IMAGES
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in (
                value.per_mask_mean_losses
                + value.per_mask_represented_mean_losses
                + value.per_mask_unrepresented_mean_losses
                + value.per_image_mean_losses
            )
        )
        or value.mean_loss != math.fsum(value.per_mask_mean_losses) / 64
        or value.represented_mean_loss
        != math.fsum(value.per_mask_represented_mean_losses) / 64
        or value.unrepresented_mean_loss
        != math.fsum(value.per_mask_unrepresented_mean_losses) / 64
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in (value.represented_mean_loss, value.unrepresented_mean_loss)
        )
    ):
        raise ValueError(f"{name} probe metrics differ")


def probe_decision(
    *,
    class_mean: ProbeMetrics,
    probe_fits: Mapping[int, ProbeFit],
    probe_metrics: Mapping[int, ProbeMetrics],
    probe_gradients: Mapping[int, ProbeGradientMetrics],
) -> ProbeDecision:
    """Apply the prospective direction-screen promotion rule."""

    _validate_metrics(class_mean, "class_mean")
    if any(
        type(values) is not dict or tuple(values) != (0, 1, 2)
        for values in (probe_fits, probe_metrics, probe_gradients)
    ):
        raise ValueError("probe decision seed order differs")
    per_seed: dict[int, dict[str, bool]] = {}
    per_seed_statistics: dict[int, ProbeSeedStatistics] = {}
    t_critical_df63 = 1.998340542520741
    for seed in range(3):
        fit = probe_fits[seed]
        metrics = probe_metrics[seed]
        gradient = probe_gradients[seed]
        if type(fit) is not ProbeFit or type(gradient) is not ProbeGradientMetrics:
            raise ValueError("probe seed evidence type differs")
        fit_scalars = (
            fit.initial_loss,
            fit.final_loss,
            fit.row_cosine_min,
            fit.row_cosine_p05,
            fit.row_cosine_median,
            fit.row_cosine_mean,
        )
        gradient_scalars = tuple(
            value for value in vars(gradient).values() if type(value) is float
        )
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in fit_scalars + gradient_scalars
        ):
            raise ValueError("probe seed evidence scalar differs")
        if (
            fit.initial_loss <= 0.0
            or fit.final_loss <= 0.0
            or not -1.0
            <= fit.row_cosine_min
            <= fit.row_cosine_p05
            <= fit.row_cosine_median
            <= 1.0
            or not -1.0 <= fit.row_cosine_mean <= 1.0
            or gradient.class_mean_l2 <= 0.0
            or gradient.spherical_probe_l2 <= 0.0
            or gradient.relative_l2_difference < 0.0
            or gradient.spherical_to_class_mean_l2_ratio <= 0.0
            or type(gradient.zero_gradient_row_count) is not int
            or not 0 <= gradient.zero_gradient_row_count < 128
            or not -1.0
            <= gradient.cosine_min
            <= gradient.cosine_p05
            <= gradient.cosine_median
            <= 1.0
            or not -1.0 <= gradient.cosine_mean <= 1.0
        ):
            raise ValueError("probe seed evidence range differs")
        _validate_metrics(metrics, f"spherical_probe_seed_{seed}")
        if class_mean.observation_count != metrics.observation_count:
            raise ValueError("probe metric observation counts differ")
        deltas = tuple(
            class_loss - probe_loss
            for class_loss, probe_loss in zip(
                class_mean.per_mask_mean_losses,
                metrics.per_mask_mean_losses,
                strict=True,
            )
        )
        mean_delta = math.fsum(deltas) / len(deltas)
        variance = math.fsum((value - mean_delta) ** 2 for value in deltas) / (
            len(deltas) - 1
        )
        lower = mean_delta - t_critical_df63 * math.sqrt(variance / len(deltas))
        identity_deltas = tuple(
            class_loss - probe_loss
            for class_loss, probe_loss in zip(
                class_mean.per_image_mean_losses,
                metrics.per_image_mean_losses,
                strict=True,
            )
        )
        identity_mean = math.fsum(identity_deltas) / len(identity_deltas)
        identity_variance = math.fsum(
            (value - identity_mean) ** 2 for value in identity_deltas
        ) / (len(identity_deltas) - 1)
        identity_lower = identity_mean - PROBE_IDENTITY_T_CRITICAL_DF3187 * math.sqrt(
            identity_variance / len(identity_deltas)
        )
        unrepresented_deltas = tuple(
            class_loss - probe_loss
            for class_loss, probe_loss in zip(
                class_mean.per_mask_unrepresented_mean_losses,
                metrics.per_mask_unrepresented_mean_losses,
                strict=True,
            )
        )
        unrepresented_mean = math.fsum(unrepresented_deltas) / 64
        unrepresented_variance = math.fsum(
            (value - unrepresented_mean) ** 2 for value in unrepresented_deltas
        ) / 63
        unrepresented_lower = unrepresented_mean - t_critical_df63 * math.sqrt(
            unrepresented_variance / 64
        )
        per_seed_statistics[seed] = ProbeSeedStatistics(
            mean_paired_loss_delta=mean_delta,
            paired_95_lower_bound=lower,
            identity_95_lower_bound=identity_lower,
            positive_mask_count=sum(value > 0.0 for value in deltas),
            unrepresented_paired_95_lower_bound=unrepresented_lower,
            unrepresented_positive_mask_count=sum(
                value > 0.0 for value in unrepresented_deltas
            ),
            accuracy_delta=metrics.accuracy - class_mean.accuracy,
            represented_loss_delta=(
                class_mean.represented_mean_loss - metrics.represented_mean_loss
            ),
            unrepresented_loss_delta=(
                class_mean.unrepresented_mean_loss - metrics.unrepresented_mean_loss
            ),
        )
        per_seed[seed] = {
            "fit_loss_decreased": fit.final_loss < fit.initial_loss,
            "head_cosine_minimum": fit.row_cosine_min >= 0.80,
            "head_cosine_mean": fit.row_cosine_mean >= 0.95,
            "validation_loss_positive": mean_delta > 0.0,
            "paired_95_lower_positive": lower > 0.0,
            "identity_95_lower_positive": identity_lower > 0.0,
            "positive_mask_majority": sum(value > 0.0 for value in deltas) >= 48,
            "validation_accuracy_noninferior": metrics.accuracy >= class_mean.accuracy,
            "represented_loss_positive": (
                metrics.represented_mean_loss < class_mean.represented_mean_loss
            ),
            "unrepresented_loss_positive": (
                metrics.unrepresented_mean_loss < class_mean.unrepresented_mean_loss
            ),
            "unrepresented_paired_95_lower_positive": unrepresented_lower > 0.0,
            "unrepresented_positive_mask_majority": sum(
                value > 0.0 for value in unrepresented_deltas
            )
            >= 48,
            "gradient_median_cosine": gradient.cosine_median <= 0.995,
        }
    return ProbeDecision(
        status=(
            "PROMOTE"
            if all(all(predicates.values()) for predicates in per_seed.values())
            else "CLOSE_DIRECTION"
        ),
        per_seed_statistics=per_seed_statistics,
        per_seed_predicates=per_seed,
    )
