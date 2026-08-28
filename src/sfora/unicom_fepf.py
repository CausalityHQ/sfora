"""Deterministic primitives for UniCOM frozen-embedding proxy fitting."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from sfora.unicom_training import (
    experiment_stream_seed,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_loss,
)

FEPF_ROW_NORM_RTOL = 2e-6
FEPF_ROW_NORM_ATOL = 2e-7


@dataclass(frozen=True)
class FepfCache:
    features: torch.Tensor
    labels: torch.Tensor
    record_inventory: tuple[tuple[str, str], ...]
    label_map_inventory: tuple[tuple[str, int], ...]
    class_count: int
    feature_sha256: str
    label_sha256: str
    inventory_sha256: str
    label_map_sha256: str


@dataclass(frozen=True)
class FepfFitResult:
    head: torch.Tensor
    initial_loss: float
    final_loss: float
    completed_steps: int
    batch_root_seed: int
    mask_root_seed: int
    mask_generator_initial_sha256: str
    mask_generator_final_sha256: str
    diagnostic_indices: tuple[int, ...]
    diagnostic_feature_sha256: str
    diagnostic_label_sha256: str
    diagnostic_mask_sha256: str
    start_head_sha256: str
    final_head_sha256: str
    fit_seconds: float


@dataclass(frozen=True)
class InitializationRngAudit:
    python_rng_entry_sha256: str
    python_rng_post_draw_sha256: str
    python_rng_restored_sha256: str
    numpy_rng_entry_sha256: str
    numpy_rng_post_draw_sha256: str
    numpy_rng_restored_sha256: str
    torch_cpu_rng_entry_sha256: str
    torch_cpu_rng_post_draw_sha256: str
    torch_cpu_rng_restored_sha256: str
    torch_cuda_rng_entry_sha256: tuple[str, ...]
    torch_cuda_rng_post_draw_sha256: tuple[str, ...]
    torch_cuda_rng_restored_sha256: tuple[str, ...]


@dataclass(frozen=True)
class FepfDiagnostic:
    loss: float
    indices: tuple[int, ...]
    feature_sha256: str
    label_sha256: str
    mask_sha256: str


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact canonical CPU tensor bytes."""
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
    ):
        raise ValueError("FEPF tensor bytes differ")
    if tensor.dtype not in (torch.float32, torch.int64, torch.uint8):
        raise ValueError("FEPF tensor dtype differs")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _inventory_sha256(value: tuple[tuple[object, object], ...]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_cache(cache: FepfCache) -> None:
    if not isinstance(cache, FepfCache):
        raise ValueError("FEPF cache differs")
    label_map = dict(cache.label_map_inventory)
    if (
        cache.features.device.type != "cpu"
        or cache.features.dtype != torch.float32
        or cache.features.ndim != 2
        or not cache.features.is_contiguous()
        or cache.labels.device.type != "cpu"
        or cache.labels.dtype != torch.int64
        or cache.labels.ndim != 1
        or not cache.labels.is_contiguous()
        or cache.features.shape[0] != cache.labels.numel()
        or not cache.features.numel()
        or type(cache.class_count) is not int
        or cache.class_count <= 0
        or cache.features.shape[1] <= 0
        or len(cache.record_inventory) != cache.features.shape[0]
        or len(cache.label_map_inventory) != cache.class_count
        or any(
            type(label) is not str
            or not label
            or type(path) is not str
            or not path
            for label, path in cache.record_inventory
        )
        or tuple(index for _, index in cache.label_map_inventory) != tuple(range(cache.class_count))
        or any(
            type(label) is not str or not label or type(index) is not int
            for label, index in cache.label_map_inventory
        )
        or any(label not in label_map for label, _ in cache.record_inventory)
        or tuple(cache.labels.tolist())
        != tuple(label_map[label] for label, _ in cache.record_inventory)
        or not torch.isfinite(cache.features).all()
        or torch.any(torch.linalg.vector_norm(cache.features, dim=1) == 0)
        or torch.any(cache.labels < 0)
        or torch.any(cache.labels >= cache.class_count)
        or torch.any(torch.bincount(cache.labels, minlength=cache.class_count) == 0)
        or tensor_sha256(cache.features) != cache.feature_sha256
        or tensor_sha256(cache.labels) != cache.label_sha256
        or _inventory_sha256(cache.record_inventory) != cache.inventory_sha256
        or _inventory_sha256(cache.label_map_inventory) != cache.label_map_sha256
    ):
        raise ValueError("FEPF cache differs")


def build_fepf_cache(
    records: tuple[tuple[str, str], ...], features: torch.Tensor, label_map: Mapping[str, int]
) -> FepfCache:
    """Bind ordered optimization rows to their canonical CPU feature cache."""
    if (
        type(records) is not tuple
        or not records
        or not isinstance(label_map, Mapping)
        or type(label_map) is not dict
        or not label_map
        or any(type(row) is not tuple or len(row) != 2 for row in records)
        or any(
            type(label) is not str or not label or type(path) is not str or not path
            for label, path in records
        )
        or any(
            type(label) is not str or not label or type(index) is not int
            for label, index in label_map.items()
        )
        or tuple(label_map.values()) != tuple(range(len(label_map)))
        or any(label not in label_map for label, _ in records)
    ):
        raise ValueError("FEPF record inventory differs")
    if (
        not isinstance(features, torch.Tensor)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.ndim != 2
        or not features.is_contiguous()
        or features.shape[0] != len(records)
        or features.shape[1] <= 0
        or not torch.isfinite(features).all()
        or torch.any(torch.linalg.vector_norm(features, dim=1) == 0)
    ):
        raise ValueError("FEPF feature cache differs")
    labels = torch.tensor([label_map[label] for label, _ in records], dtype=torch.int64)
    if torch.any(torch.bincount(labels, minlength=len(label_map)) == 0):
        raise ValueError("FEPF class is empty")
    inventory = tuple((label, path) for label, path in records)
    label_inventory = tuple(label_map.items())
    return FepfCache(
        features=features,
        labels=labels,
        record_inventory=inventory,
        label_map_inventory=label_inventory,
        class_count=len(label_map),
        feature_sha256=tensor_sha256(features),
        label_sha256=tensor_sha256(labels),
        inventory_sha256=_inventory_sha256(inventory),
        label_map_sha256=_inventory_sha256(label_inventory),
    )


def validate_projected_head(head: torch.Tensor) -> None:
    if (
        not isinstance(head, torch.Tensor)
        or head.dtype != torch.float32
        or head.ndim != 2
        or not head.is_contiguous()
        or head.shape[0] <= 0
        or head.shape[1] <= 0
    ):
        raise ValueError("FEPF projected head differs")
    observed = torch.linalg.vector_norm(head, dim=1)
    target = torch.full_like(observed, 0.01 * math.sqrt(head.shape[1]))
    if not torch.isfinite(observed).all() or not torch.allclose(
        observed, target, rtol=FEPF_ROW_NORM_RTOL, atol=FEPF_ROW_NORM_ATOL
    ):
        raise ValueError("FEPF projected row norm differs")


def canonical_class_means(cache: FepfCache, *, dimension: int = 768) -> torch.Tensor:
    """Return sequential-FP64 normalized and projected class means."""
    _validate_cache(cache)
    if type(dimension) is not int or dimension <= 0 or cache.features.shape[1] != dimension:
        raise ValueError("FEPF class-mean dimension differs")
    rows: list[torch.Tensor] = []
    for class_index in range(cache.class_count):
        total = torch.zeros(dimension, dtype=torch.float64)
        count = 0
        for feature, label in zip(cache.features, cache.labels, strict=True):
            if int(label) == class_index:
                norm = torch.linalg.vector_norm(feature)
                if not torch.isfinite(norm) or norm == 0:
                    raise ValueError("FEPF embedding norm differs")
                total.add_((feature / norm).double())
                count += 1
        if count == 0:
            raise ValueError("FEPF class is empty")
        mean = total / count
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("FEPF class mean differs")
        rows.append((mean / norm).float())
    result = torch.stack(rows).mul_(0.01 * math.sqrt(dimension)).contiguous()
    validate_projected_head(result)
    return result


def prepare_fepf_start_head(
    random_head: torch.Tensor, class_means: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Select the immutable mean or normalized official random start head."""
    if mode == "fepf_mean":
        validate_projected_head(class_means)
        return class_means.detach().clone().contiguous()
    if mode != "fepf_random":
        raise ValueError("FEPF mode differs")
    if (
        not isinstance(random_head, torch.Tensor)
        or random_head.dtype != torch.float32
        or random_head.ndim != 2
        or random_head.shape != class_means.shape
    ):
        raise ValueError("FEPF random head differs")
    values = random_head.detach().clone().contiguous()
    norms = torch.linalg.vector_norm(values, dim=1)
    if not torch.isfinite(norms).all() or torch.any(norms == 0):
        raise ValueError("FEPF random row norm differs")
    values.mul_(((0.01 * math.sqrt(values.shape[1])) / norms)[:, None])
    validate_projected_head(values)
    return values


def project_and_validate_head_(head: torch.Tensor) -> None:
    """Project one fitted head and fail closed if the resulting norm drifts."""
    norms = torch.linalg.vector_norm(head, dim=1)
    if not torch.isfinite(norms).all() or torch.any(norms == 0):
        raise ValueError("FEPF updated row norm differs")
    head.mul_(((0.01 * math.sqrt(head.shape[1])) / norms)[:, None])
    validate_projected_head(head)


def registered_diagnostic(
    features: torch.Tensor, labels: torch.Tensor, head: torch.Tensor, *, training_seed: int
) -> FepfDiagnostic:
    """Evaluate the one fixed pre/post-fit diagnostic batch and mask stream."""
    if type(training_seed) is not int or training_seed < 0 or features.shape[1] != 768:
        raise ValueError("FEPF diagnostic schedule differs")
    seed = experiment_stream_seed(training_seed, 23_004)
    order = padded_epoch_indices(
        size=len(labels), global_batch=128, epoch=0, seed=seed, shards=8
    )[:128]
    if len(order) != 128:
        raise ValueError("FEPF diagnostic inventory differs")
    indices = torch.tensor(order, dtype=torch.int64, device=features.device)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    masks = sample_shard_masks(
        dimension=768,
        selected=512,
        shards=8,
        generator=generator,
        device=features.device,
    )
    with torch.no_grad():
        loss = sharded_mask_arcface_loss(
            features.index_select(0, indices),
            head,
            labels.index_select(0, indices),
            masks,
            margin=0.25,
            scale=32.0,
        )
    if not torch.isfinite(loss):
        raise ValueError("FEPF diagnostic loss is nonfinite")
    diagnostic_features = features.index_select(0, indices).detach().cpu().contiguous()
    diagnostic_labels = labels.index_select(0, indices).detach().cpu().contiguous()
    return FepfDiagnostic(
        loss=float(loss),
        indices=tuple(order),
        feature_sha256=tensor_sha256(diagnostic_features),
        label_sha256=tensor_sha256(diagnostic_labels),
        mask_sha256=tensor_sha256(masks.detach().cpu().contiguous()),
    )


def _fit_fepf_head_core(
    cache: FepfCache,
    start_head: torch.Tensor,
    *,
    training_seed: int,
    device: torch.device,
    steps: int = 512,
    monotonic: Callable[[], float] = time.perf_counter,
) -> FepfFitResult:
    """Injected fit runner for unit tests; it is not a registered initializer."""
    _validate_cache(cache)
    if (
        not isinstance(device, torch.device)
        or type(training_seed) is not int
        or training_seed < 0
        or type(steps) is not int
        or not 0 < steps <= 512
        or not callable(monotonic)
        or cache.features.shape[1] != 768
        or start_head.shape != (cache.class_count, 768)
        or start_head.dtype != torch.float32
    ):
        raise ValueError("FEPF fit schedule differs")
    fit_started = monotonic()
    features = cache.features.to(device=device, dtype=torch.float32)
    labels = cache.labels.to(device=device, dtype=torch.int64)
    batch_root = experiment_stream_seed(training_seed, 23_001)
    mask_root = experiment_stream_seed(training_seed, 23_002)
    mask_generator = torch.Generator(device=device).manual_seed(mask_root)
    mask_generator_initial_sha256 = tensor_sha256(mask_generator.get_state().cpu().contiguous())
    head = torch.nn.Parameter(start_head.detach().to(device=device).clone().contiguous())
    validate_projected_head(head)
    start_head_sha256 = tensor_sha256(head.detach().cpu().contiguous())
    initial = registered_diagnostic(features, labels, head, training_seed=training_seed)
    optimizer = torch.optim.AdamW(
        [head], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    completed = 0
    pseudoepoch = 0
    while completed < steps:
        order = padded_epoch_indices(
            size=len(cache.labels),
            global_batch=128,
            epoch=pseudoepoch,
            seed=batch_root,
            shards=8,
        )
        if len(order) % 128:
            raise ValueError("FEPF batch inventory differs")
        for start in range(0, len(order), 128):
            indices = torch.tensor(order[start : start + 128], dtype=torch.int64, device=device)
            shard_masks = sample_shard_masks(
                dimension=768,
                selected=512,
                shards=8,
                generator=mask_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = sharded_mask_arcface_loss(
                features.index_select(0, indices),
                head,
                labels.index_select(0, indices),
                shard_masks,
                margin=0.25,
                scale=32.0,
            )
            if not torch.isfinite(loss):
                raise ValueError("FEPF loss is nonfinite")
            loss.backward()
            if head.grad is None or not torch.isfinite(head.grad).all():
                raise ValueError("FEPF gradient is nonfinite")
            optimizer.step()
            with torch.no_grad():
                project_and_validate_head_(head)
            completed += 1
            if completed == steps:
                break
        pseudoepoch += 1
    diagnostic = registered_diagnostic(features, labels, head, training_seed=training_seed)
    fit_seconds = monotonic() - fit_started
    if not math.isfinite(fit_seconds) or fit_seconds <= 0:
        raise ValueError("FEPF fit duration differs")
    result = FepfFitResult(
        head=head.detach().clone().contiguous(),
        initial_loss=initial.loss,
        final_loss=diagnostic.loss,
        completed_steps=completed,
        batch_root_seed=batch_root,
        mask_root_seed=mask_root,
        mask_generator_initial_sha256=mask_generator_initial_sha256,
        mask_generator_final_sha256=tensor_sha256(mask_generator.get_state().cpu().contiguous()),
        diagnostic_indices=diagnostic.indices,
        diagnostic_feature_sha256=diagnostic.feature_sha256,
        diagnostic_label_sha256=diagnostic.label_sha256,
        diagnostic_mask_sha256=diagnostic.mask_sha256,
        start_head_sha256=start_head_sha256,
        final_head_sha256=tensor_sha256(head.detach().cpu().contiguous()),
        fit_seconds=fit_seconds,
    )
    return result


def fit_fepf_head(
    cache: FepfCache,
    start_head: torch.Tensor,
    *,
    training_seed: int,
    device: torch.device,
    steps: int = 512,
    monotonic: Callable[[], float] = time.perf_counter,
) -> FepfFitResult:
    """Run the sole registered CUDA, 512-step FEPF fit schedule."""
    if not isinstance(device, torch.device) or device.type != "cuda" or steps != 512:
        raise ValueError("FEPF registered fit schedule differs")
    return _fit_fepf_head_core(
        cache,
        start_head,
        training_seed=training_seed,
        device=device,
        steps=steps,
        monotonic=monotonic,
    )


_RECEIPT_KEYS = (
    "schema",
    "mode",
    "training_seed",
    "holdout_fraction",
    "holdout_seed",
    "source_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "schedule_sha256",
    "row_norm_rtol",
    "row_norm_atol",
    "python_rng_entry_sha256",
    "python_rng_post_draw_sha256",
    "python_rng_restored_sha256",
    "numpy_rng_entry_sha256",
    "numpy_rng_post_draw_sha256",
    "numpy_rng_restored_sha256",
    "torch_cpu_rng_entry_sha256",
    "torch_cpu_rng_post_draw_sha256",
    "torch_cpu_rng_restored_sha256",
    "torch_cuda_rng_entry_sha256",
    "torch_cuda_rng_post_draw_sha256",
    "torch_cuda_rng_restored_sha256",
    "official_random_head_sha256",
    "prepared_start_head_sha256",
    "final_head_sha256",
    "initialization_seconds",
    "fit_seconds",
    "diagnostic_indices",
    "diagnostic_feature_sha256",
    "diagnostic_label_sha256",
    "diagnostic_mask_sha256",
    "feature_sha256",
    "label_sha256",
    "inventory_sha256",
    "label_map_sha256",
    "class_count",
    "classifier_shape",
    "initial_loss",
    "final_loss",
    "mask_generator_initial_sha256",
    "mask_generator_final_sha256",
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mask_state_sha256(*, training_seed: int, device: torch.device, draws: int) -> str:
    generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(training_seed, 23_002)
    )
    for _ in range(draws):
        sample_shard_masks(
            dimension=768,
            selected=512,
            shards=8,
            generator=generator,
            device=device,
        )
    return tensor_sha256(generator.get_state().cpu().contiguous())


def _validate_rng_audit(rng_audit: InitializationRngAudit) -> None:
    if not isinstance(rng_audit, InitializationRngAudit):
        raise ValueError("FEPF initialization RNG audit differs")
    scalar_names = (
        "python_rng_entry_sha256",
        "python_rng_post_draw_sha256",
        "python_rng_restored_sha256",
        "numpy_rng_entry_sha256",
        "numpy_rng_post_draw_sha256",
        "numpy_rng_restored_sha256",
        "torch_cpu_rng_entry_sha256",
        "torch_cpu_rng_post_draw_sha256",
        "torch_cpu_rng_restored_sha256",
    )
    if not all(_is_sha256(getattr(rng_audit, name)) for name in scalar_names):
        raise ValueError("FEPF initialization RNG hash differs")
    cuda_states = (
        rng_audit.torch_cuda_rng_entry_sha256,
        rng_audit.torch_cuda_rng_post_draw_sha256,
        rng_audit.torch_cuda_rng_restored_sha256,
    )
    if any(
        type(states) is not tuple
        or len(states) != torch.cuda.device_count()
        or not all(_is_sha256(value) for value in states)
        for states in cuda_states
    ):
        raise ValueError("FEPF initialization CUDA RNG audit differs")
    if (
        rng_audit.python_rng_entry_sha256
        != rng_audit.python_rng_post_draw_sha256
        != rng_audit.python_rng_restored_sha256
        or rng_audit.numpy_rng_entry_sha256
        != rng_audit.numpy_rng_post_draw_sha256
        != rng_audit.numpy_rng_restored_sha256
        or rng_audit.torch_cuda_rng_entry_sha256
        != rng_audit.torch_cuda_rng_post_draw_sha256
        != rng_audit.torch_cuda_rng_restored_sha256
        or rng_audit.torch_cpu_rng_post_draw_sha256
        != rng_audit.torch_cpu_rng_restored_sha256
    ):
        raise ValueError("FEPF initialization RNG restoration differs")


def _head_sha256(head: torch.Tensor) -> str:
    return tensor_sha256(head.detach().cpu().contiguous())


def _initialization_receipt_v2_core(
    *,
    mode: str,
    training_seed: int,
    holdout_fraction: float,
    holdout_seed: int,
    source_sha256: str,
    checkpoint_sha256: str,
    config_sha256: str,
    schedule_sha256: str,
    official_random_head: torch.Tensor,
    prepared_start_head: torch.Tensor,
    initialization_seconds: float,
    cache: FepfCache,
    rng_audit: InitializationRngAudit,
    fit: FepfFitResult | None,
    device: torch.device,
    allow_test_device: bool,
) -> dict[str, object]:
    """Derive every receipt relation; CPU use is restricted to test injection."""
    _validate_cache(cache)
    if (
        mode not in {"imprinted", "fepf_mean", "fepf_random"}
        or type(training_seed) is not int
        or training_seed < 0
        or type(holdout_fraction) is not float
        or not math.isfinite(holdout_fraction)
        or not 0.0 <= holdout_fraction < 1.0
        or type(holdout_seed) is not int
        or holdout_seed < 0
        or not isinstance(device, torch.device)
        or (not allow_test_device and device.type != "cuda")
        or type(initialization_seconds) is not float
        or not math.isfinite(initialization_seconds)
        or initialization_seconds <= 0
        or not all(
            _is_sha256(value)
            for value in (source_sha256, checkpoint_sha256, config_sha256, schedule_sha256)
        )
    ):
        raise ValueError("FEPF initialization receipt differs")
    _validate_rng_audit(rng_audit)
    expected_shape = (cache.class_count, 768)
    if (
        not isinstance(official_random_head, torch.Tensor)
        or official_random_head.device.type != "cpu"
        or official_random_head.dtype != torch.float32
        or official_random_head.shape != expected_shape
        or not official_random_head.is_contiguous()
        or not torch.isfinite(official_random_head).all()
        or not isinstance(prepared_start_head, torch.Tensor)
        or prepared_start_head.device.type != "cpu"
        or prepared_start_head.dtype != torch.float32
        or prepared_start_head.shape != expected_shape
        or not prepared_start_head.is_contiguous()
        or not torch.isfinite(prepared_start_head).all()
    ):
        raise ValueError("FEPF initialization head differs")
    class_means = canonical_class_means(cache)
    if mode in {"imprinted", "fepf_mean"}:
        expected_start = class_means
    else:
        expected_start = prepare_fepf_start_head(official_random_head, class_means, mode=mode)
    if not torch.equal(prepared_start_head, expected_start):
        raise ValueError("FEPF prepared head differs")
    prepared_hash = _head_sha256(prepared_start_head)
    device_features = cache.features.to(device)
    device_labels = cache.labels.to(device)
    initial = registered_diagnostic(
        device_features, device_labels, prepared_start_head.to(device), training_seed=training_seed
    )
    initial_mask_hash = _mask_state_sha256(training_seed=training_seed, device=device, draws=0)
    if mode == "imprinted":
        if fit is not None:
            raise ValueError("FEPF imprinted receipt differs")
        final = initial
        final_head_hash = prepared_hash
        fit_seconds = 0.0
        final_mask_hash = initial_mask_hash
    else:
        if (
            not isinstance(fit, FepfFitResult)
            or fit.completed_steps != 512
            or fit.head.device != device
            or fit.head.dtype != torch.float32
            or fit.head.shape != expected_shape
            or not fit.head.is_contiguous()
        ):
            raise ValueError("FEPF fitted receipt differs")
        validate_projected_head(fit.head)
        final = registered_diagnostic(
            device_features, device_labels, fit.head, training_seed=training_seed
        )
        final_head_hash = _head_sha256(fit.head)
        final_mask_hash = _mask_state_sha256(training_seed=training_seed, device=device, draws=512)
        if (
            fit.batch_root_seed != experiment_stream_seed(training_seed, 23_001)
            or fit.mask_root_seed != experiment_stream_seed(training_seed, 23_002)
            or fit.start_head_sha256 != prepared_hash
            or fit.final_head_sha256 != final_head_hash
            or fit.initial_loss != initial.loss
            or fit.final_loss != final.loss
            or fit.diagnostic_indices != final.indices
            or fit.diagnostic_feature_sha256 != final.feature_sha256
            or fit.diagnostic_label_sha256 != final.label_sha256
            or fit.diagnostic_mask_sha256 != final.mask_sha256
            or fit.mask_generator_initial_sha256 != initial_mask_hash
            or fit.mask_generator_final_sha256 != final_mask_hash
            or type(fit.fit_seconds) is not float
            or not math.isfinite(fit.fit_seconds)
            or fit.fit_seconds <= 0
        ):
            raise ValueError("FEPF fitted receipt differs")
        fit_seconds = fit.fit_seconds
    return {
        "schema": "initialization-receipt-v2",
        "mode": mode,
        "training_seed": training_seed,
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "schedule_sha256": schedule_sha256,
        "row_norm_rtol": FEPF_ROW_NORM_RTOL,
        "row_norm_atol": FEPF_ROW_NORM_ATOL,
        "python_rng_entry_sha256": rng_audit.python_rng_entry_sha256,
        "python_rng_post_draw_sha256": rng_audit.python_rng_post_draw_sha256,
        "python_rng_restored_sha256": rng_audit.python_rng_restored_sha256,
        "numpy_rng_entry_sha256": rng_audit.numpy_rng_entry_sha256,
        "numpy_rng_post_draw_sha256": rng_audit.numpy_rng_post_draw_sha256,
        "numpy_rng_restored_sha256": rng_audit.numpy_rng_restored_sha256,
        "torch_cpu_rng_entry_sha256": rng_audit.torch_cpu_rng_entry_sha256,
        "torch_cpu_rng_post_draw_sha256": rng_audit.torch_cpu_rng_post_draw_sha256,
        "torch_cpu_rng_restored_sha256": rng_audit.torch_cpu_rng_restored_sha256,
        "torch_cuda_rng_entry_sha256": list(rng_audit.torch_cuda_rng_entry_sha256),
        "torch_cuda_rng_post_draw_sha256": list(rng_audit.torch_cuda_rng_post_draw_sha256),
        "torch_cuda_rng_restored_sha256": list(rng_audit.torch_cuda_rng_restored_sha256),
        "official_random_head_sha256": _head_sha256(official_random_head),
        "prepared_start_head_sha256": prepared_hash,
        "final_head_sha256": final_head_hash,
        "initialization_seconds": initialization_seconds,
        "fit_seconds": fit_seconds,
        "diagnostic_indices": list(final.indices),
        "diagnostic_feature_sha256": final.feature_sha256,
        "diagnostic_label_sha256": final.label_sha256,
        "diagnostic_mask_sha256": final.mask_sha256,
        "feature_sha256": cache.feature_sha256,
        "label_sha256": cache.label_sha256,
        "inventory_sha256": cache.inventory_sha256,
        "label_map_sha256": cache.label_map_sha256,
        "class_count": cache.class_count,
        "classifier_shape": list(expected_shape),
        "initial_loss": initial.loss,
        "final_loss": final.loss,
        "mask_generator_initial_sha256": initial_mask_hash,
        "mask_generator_final_sha256": final_mask_hash,
    }


def initialization_receipt_v2(**kwargs: object) -> dict[str, object]:
    """Build a receipt only for the registered CUDA initialization path."""
    return _initialization_receipt_v2_core(**kwargs, allow_test_device=False)


_PROVENANCE_KEYS = (
    "mode",
    "training_seed",
    "holdout_fraction",
    "holdout_seed",
    "source_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "schedule_sha256",
)


def _validate_initialization_receipt_v2_core(
    receipt: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    device: torch.device,
    allow_test_device: bool,
) -> None:
    """Validate a receipt against the caller's authenticated run context."""
    if type(receipt) is not dict or tuple(receipt) != _RECEIPT_KEYS:
        raise ValueError("FEPF initialization receipt schema differs")
    if (
        type(expected) is not dict
        or tuple(expected) not in (_PROVENANCE_KEYS, _RECEIPT_KEYS)
        or not isinstance(device, torch.device)
        or (not allow_test_device and device.type != "cuda")
    ):
        raise ValueError("FEPF initialization receipt device differs")
    if receipt["schema"] != "initialization-receipt-v2":
        raise ValueError("FEPF initialization receipt schema differs")
    mode = receipt["mode"]
    training_seed = receipt["training_seed"]
    holdout_fraction = receipt["holdout_fraction"]
    holdout_seed = receipt["holdout_seed"]
    if (
        mode not in {"imprinted", "fepf_mean", "fepf_random"}
        or type(training_seed) is not int
        or training_seed < 0
        or type(holdout_fraction) is not float
        or not math.isfinite(holdout_fraction)
        or not 0.0 <= holdout_fraction < 1.0
        or type(holdout_seed) is not int
        or holdout_seed < 0
        or receipt["row_norm_rtol"] != FEPF_ROW_NORM_RTOL
        or receipt["row_norm_atol"] != FEPF_ROW_NORM_ATOL
    ):
        raise ValueError("FEPF initialization receipt scalar differs")
    for key in expected:
        if receipt[key] != expected[key]:
            raise ValueError("FEPF initialization receipt provenance differs")
    hash_keys = tuple(key for key in _RECEIPT_KEYS if key.endswith("_sha256"))
    for key in hash_keys:
        value = receipt[key]
        if key.startswith("torch_cuda_rng_"):
            if type(value) is not list or not all(_is_sha256(item) for item in value):
                raise ValueError("FEPF initialization receipt CUDA RNG differs")
        elif not _is_sha256(value):
            raise ValueError("FEPF initialization receipt hash differs")
    cuda_states = tuple(
        receipt[f"torch_cuda_rng_{phase}_sha256"] for phase in ("entry", "post_draw", "restored")
    )
    if (
        any(len(states) != torch.cuda.device_count() for states in cuda_states)
        or receipt["python_rng_entry_sha256"]
        != receipt["python_rng_post_draw_sha256"]
        != receipt["python_rng_restored_sha256"]
        or receipt["numpy_rng_entry_sha256"]
        != receipt["numpy_rng_post_draw_sha256"]
        != receipt["numpy_rng_restored_sha256"]
        or cuda_states[0] != cuda_states[1] != cuda_states[2]
        or receipt["torch_cpu_rng_post_draw_sha256"]
        != receipt["torch_cpu_rng_restored_sha256"]
    ):
        raise ValueError("FEPF initialization receipt RNG restoration differs")
    classifier_shape = receipt["classifier_shape"]
    if (
        type(receipt["class_count"]) is not int
        or receipt["class_count"] <= 0
        or type(classifier_shape) is not list
        or len(classifier_shape) != 2
        or any(type(value) is not int for value in classifier_shape)
        or classifier_shape != [receipt["class_count"], 768]
    ):
        raise ValueError("FEPF initialization receipt classifier differs")
    if (
        type(receipt["initialization_seconds"]) is not float
        or not math.isfinite(receipt["initialization_seconds"])
        or receipt["initialization_seconds"] <= 0
        or type(receipt["fit_seconds"]) is not float
        or not math.isfinite(receipt["fit_seconds"])
        or type(receipt["initial_loss"]) is not float
        or not math.isfinite(receipt["initial_loss"])
        or type(receipt["final_loss"]) is not float
        or not math.isfinite(receipt["final_loss"])
        or type(receipt["diagnostic_indices"]) is not list
        or len(receipt["diagnostic_indices"]) != 128
        or any(type(index) is not int or index < 0 for index in receipt["diagnostic_indices"])
    ):
        raise ValueError("FEPF initialization receipt diagnostic differs")
    if mode == "imprinted":
        if (
            receipt["fit_seconds"] != 0.0
            or receipt["initial_loss"] != receipt["final_loss"]
            or receipt["prepared_start_head_sha256"] != receipt["final_head_sha256"]
        ):
            raise ValueError("FEPF imprinted receipt differs")
        expected_initial = _mask_state_sha256(training_seed=training_seed, device=device, draws=0)
        expected_final = expected_initial
    else:
        if receipt["fit_seconds"] <= 0:
            raise ValueError("FEPF fitted receipt duration differs")
        expected_initial = _mask_state_sha256(training_seed=training_seed, device=device, draws=0)
        expected_final = _mask_state_sha256(training_seed=training_seed, device=device, draws=512)
    if (
        receipt["mask_generator_initial_sha256"] != expected_initial
        or receipt["mask_generator_final_sha256"] != expected_final
    ):
        raise ValueError("FEPF initialization receipt mask stream differs")


def validate_initialization_receipt_v2(
    receipt: Mapping[str, object], *, expected: Mapping[str, object], device: torch.device
) -> None:
    """Validate a registered CUDA receipt against expected run provenance."""
    _validate_initialization_receipt_v2_core(
        receipt, expected=expected, device=device, allow_test_device=False
    )
