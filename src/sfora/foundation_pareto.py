"""Reproducible foundation-encoder screening primitives."""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from sfora.data import ImageExample
from sfora.image_recipes import (
    RecipeSelectionSplit,
    class_disjoint_recipe_selection_split,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
_REMOTE_ALLOW_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
)


def _require_nonempty(name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty builtin string")


def _require_sha256(name: str, value: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class F1Decision:
    """Prospective foundation-transfer gate decision."""

    status: Literal["CONTINUE", "CLOSE_FOUNDATION_TRANSFER", "UNAVAILABLE_COMPARATOR"]
    quality_gap_points: float | None
    quality_within_one_point: bool
    quality_within_point_four: bool
    cost_pareto_dominant: bool
    cost_status: Literal["available", "unavailable"]
    continuation_kind: Literal["quality_margin", "pareto_equivalent", "both", "none"]
    authorized_followup: Literal[
        "matryoshka_adapter_lane",
        "dada_vptsp_fidelity_comparator_only",
        "resolve_local_comparator",
    ]
    fidelity_only: bool


def build_identity_disjoint_validation_split(
    examples: Sequence[ImageExample],
    *,
    fraction: float,
    seed: int,
) -> RecipeSelectionSplit:
    """Build the registered train-only, identity-disjoint probe-selection split."""

    return class_disjoint_recipe_selection_split(examples, fraction=fraction, seed=seed)


def decide_f1(
    *,
    candidate_probe: BiasFreeProbeResult | None,
    comparator_probe: BiasFreeProbeResult | None,
    candidate_encoder_p95_ms: float | None,
    comparator_encoder_p95_ms: float | None,
    candidate_descriptor_bytes_per_image: int | None,
    comparator_descriptor_bytes_per_image: int | None,
) -> F1Decision:
    """Apply the preregistered F1 quality-or-efficiency continuation gate."""

    if comparator_probe is None:
        return F1Decision(
            status="UNAVAILABLE_COMPARATOR",
            quality_gap_points=None,
            quality_within_one_point=False,
            quality_within_point_four=False,
            cost_pareto_dominant=False,
            cost_status="unavailable",
            continuation_kind="none",
            authorized_followup="resolve_local_comparator",
            fidelity_only=False,
        )
    if candidate_probe is None:
        raise ValueError("available comparator requires a candidate probe")
    if (
        type(candidate_probe) is not BiasFreeProbeResult
        or type(comparator_probe) is not BiasFreeProbeResult
    ):
        raise ValueError("F1 decision requires exact BiasFreeProbeResult objects")
    if (
        candidate_probe.config != comparator_probe.config
        or candidate_probe.dataset != comparator_probe.dataset
        or candidate_probe.split_sha256 != comparator_probe.split_sha256
        or candidate_probe.device_type != comparator_probe.device_type
    ):
        raise ValueError("candidate and comparator must use the same probe protocol and split")
    candidate_validation_recall_at_1_points = candidate_probe.validation_recall_at_1_points
    comparator_validation_recall_at_1_points = comparator_probe.validation_recall_at_1_points
    _require_points(
        "candidate_validation_recall_at_1_points",
        candidate_validation_recall_at_1_points,
    )
    _require_points(
        "comparator_validation_recall_at_1_points",
        comparator_validation_recall_at_1_points,
    )
    gap = round(
        candidate_validation_recall_at_1_points - comparator_validation_recall_at_1_points,
        9,
    )
    within_one = gap >= -1.0
    within_point_four = abs(gap) <= 0.40
    costs = (
        candidate_encoder_p95_ms,
        comparator_encoder_p95_ms,
        candidate_descriptor_bytes_per_image,
        comparator_descriptor_bytes_per_image,
    )
    cost_status: Literal["available", "unavailable"] = (
        "available" if all(value is not None for value in costs) else "unavailable"
    )
    for name, value, validator in (
        ("candidate_encoder_p95_ms", candidate_encoder_p95_ms, _require_positive_float),
        ("comparator_encoder_p95_ms", comparator_encoder_p95_ms, _require_positive_float),
        (
            "candidate_descriptor_bytes_per_image",
            candidate_descriptor_bytes_per_image,
            _require_positive_int,
        ),
        (
            "comparator_descriptor_bytes_per_image",
            comparator_descriptor_bytes_per_image,
            _require_positive_int,
        ),
    ):
        if value is not None:
            validator(name, value)  # type: ignore[arg-type]
    pareto = False
    if cost_status == "available":
        assert candidate_encoder_p95_ms is not None
        assert comparator_encoder_p95_ms is not None
        assert candidate_descriptor_bytes_per_image is not None
        assert comparator_descriptor_bytes_per_image is not None
        pareto = (
            candidate_encoder_p95_ms <= comparator_encoder_p95_ms
            and candidate_descriptor_bytes_per_image <= comparator_descriptor_bytes_per_image
            and (
                candidate_encoder_p95_ms < comparator_encoder_p95_ms
                or candidate_descriptor_bytes_per_image < comparator_descriptor_bytes_per_image
            )
        )
    pareto_equivalent = within_point_four and pareto
    status: Literal["CONTINUE", "CLOSE_FOUNDATION_TRANSFER"] = (
        "CONTINUE" if within_one or pareto_equivalent else "CLOSE_FOUNDATION_TRANSFER"
    )
    if within_one and pareto_equivalent:
        continuation_kind: Literal["quality_margin", "pareto_equivalent", "both", "none"] = "both"
    elif within_one:
        continuation_kind = "quality_margin"
    elif pareto_equivalent:
        continuation_kind = "pareto_equivalent"
    else:
        continuation_kind = "none"
    return F1Decision(
        status=status,
        quality_gap_points=gap,
        quality_within_one_point=within_one,
        quality_within_point_four=within_point_four,
        cost_pareto_dominant=pareto,
        cost_status=cost_status,
        continuation_kind=continuation_kind,
        authorized_followup=(
            "matryoshka_adapter_lane"
            if status == "CONTINUE"
            else "dada_vptsp_fidelity_comparator_only"
        ),
        fidelity_only=status == "CLOSE_FOUNDATION_TRANSFER",
    )


def _require_points(name: str, value: float) -> None:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be a finite builtin float in [0, 100]")


def _require_positive_float(name: str, value: float) -> None:
    if type(value) is not float or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite builtin float")


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive builtin integer")


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Registered bias-free supervised-contrastive probe protocol."""

    output_dim: int = 512
    identities_per_batch: int = 32
    images_per_identity: int = 2
    epochs: int = 20
    learning_rates: tuple[float, ...] = (0.001, 0.003, 0.01)
    temperatures: tuple[float, ...] = (0.05, 0.10)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    protocol_id: str = "f1-bias-free-512-supcon-v1"

    def __post_init__(self) -> None:
        if self.output_dim != 512:
            raise ValueError("bias-free probe output_dim must be exactly 512")
        _require_positive_int("identities_per_batch", self.identities_per_batch)
        if self.images_per_identity != 2:
            raise ValueError("bias-free probe images_per_identity must be exactly 2")
        _require_positive_int("epochs", self.epochs)
        _validate_positive_float_tuple("learning_rates", self.learning_rates)
        _validate_positive_float_tuple("temperatures", self.temperatures)
        if (
            type(self.adam_betas) is not tuple
            or len(self.adam_betas) != 2
            or any(type(value) is not float or not 0.0 < value < 1.0 for value in self.adam_betas)
        ):
            raise ValueError("adam_betas must be two builtin floats in (0, 1)")
        _require_positive_float("adam_eps", self.adam_eps)
        if type(self.weight_decay) is not float or self.weight_decay != 0.0:
            raise ValueError("bias-free probe weight_decay must be exactly 0.0")
        _require_nonempty("protocol_id", self.protocol_id)
        registered_values = (
            self.output_dim == 512
            and self.identities_per_batch == 32
            and self.images_per_identity == 2
            and self.epochs == 20
            and self.learning_rates == (0.001, 0.003, 0.01)
            and self.temperatures == (0.05, 0.10)
            and self.adam_betas == (0.9, 0.999)
            and self.adam_eps == 1e-8
            and self.weight_decay == 0.0
        )
        if self.protocol_id == "f1-bias-free-512-supcon-v1" and not registered_values:
            raise ValueError("custom probe config requires a distinct protocol_id")


@dataclass(frozen=True)
class ProbeEpochEvaluation:
    epoch: int
    validation_recall_at_1: float


@dataclass(frozen=True)
class ProbeGridEvaluation:
    learning_rate: float
    temperature: float
    sampler_seed: int
    epochs: tuple[ProbeEpochEvaluation, ...]


@dataclass(frozen=True)
class BiasFreeProbeResult:
    """Selected probe and complete validation-selection audit."""

    arm_key: str
    dataset: str
    input_dim: int
    output_dim: int
    split_seed: int
    split_sha256: str
    initialization_seed: int
    excluded_singleton_identities: int
    dropped_tail_identities: int
    selection_evaluations: int
    grid_evaluations: tuple[ProbeGridEvaluation, ...]
    selected_learning_rate: float
    selected_temperature: float
    selected_epoch: int
    validation_recall_at_1: float
    validation_recall_at_1_points: float
    validation_metrics: Any
    selected_weight_bytes: bytes
    weight_sha256: str
    parameter_count: int
    fitted: bool
    bias: bool
    input_normalized: bool
    output_normalized: bool
    loss: Literal["supervised_infonce"]
    protocol_id: str
    config: ProbeTrainingConfig
    cublas_workspace_config: str
    torch_version: str
    device_type: str
    device_name: str


def _validate_positive_float_tuple(name: str, values: tuple[float, ...]) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a nonempty builtin tuple")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")
    for value in values:
        _require_positive_float(name, value)


_DEFAULT_PROBE_TRAINING_CONFIG = ProbeTrainingConfig()


def _probe_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _probe_vector(value: Any, *, expected_dim: int | None) -> Any:
    import numpy as np

    row = np.asarray(value, dtype=np.float32)
    if row.ndim != 1 or row.shape[0] == 0:
        raise ValueError("probe embeddings must be nonempty rank-1 rows")
    if expected_dim is not None and row.shape[0] != expected_dim:
        raise ValueError("probe embedding widths differ")
    if not bool(np.isfinite(row).all()):
        raise ValueError("probe embeddings must contain only finite values")
    norm = float(np.linalg.norm(row.astype(np.float64)))
    if norm == 0.0:
        raise ValueError("probe embeddings cannot contain zero-norm rows")
    return row


def _supervised_infonce_loss(output: Any, labels: Any, *, temperature: float) -> Any:
    import torch

    logits = (output @ output.T) / temperature
    identity = torch.eye(output.shape[0], dtype=torch.bool, device=output.device)
    logits = logits.masked_fill(identity, -torch.inf)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every supervised InfoNCE anchor requires a positive")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob))
    return -(positive_log_prob.sum(dim=1) / positive_count).mean()


def _probe_weight_bytes(weight: Any) -> bytes:
    value = weight.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    if type(value) is not bytes:
        raise ValueError("probe weight serialization did not produce builtin bytes")
    return value


def _probe_validation_metrics(
    model: Any,
    query: Any,
    query_labels: Any,
    gallery: Any,
    gallery_labels: Any,
) -> Any:
    import torch

    with torch.no_grad():
        query_output = torch.nn.functional.normalize(model(query), p=2, dim=1)
        gallery_output = torch.nn.functional.normalize(model(gallery), p=2, dim=1)
    query_array = _normalize_rows(query_output.detach().cpu().numpy())
    gallery_array = _normalize_rows(gallery_output.detach().cpu().numpy())
    relevant = max(int((gallery_labels == label).sum()) for label in query_labels)
    order = _gallery_order(
        query_array,
        gallery_array,
        geometry="cosine",
        depth=max(100, relevant),
    )
    return _metrics_from_gallery_order(order, query_labels, gallery_labels)


def fit_bias_free_probe_512(
    embeddings_by_example_id: Mapping[str, Any],
    split: RecipeSelectionSplit,
    *,
    arm_key: str,
    dataset: str,
    split_seed: int,
    config: ProbeTrainingConfig = _DEFAULT_PROBE_TRAINING_CONFIG,
) -> BiasFreeProbeResult:
    """Fit the registered train-only supervised-contrastive linear probe."""

    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from registered :4096:8")
    import torch

    _require_nonempty("arm_key", arm_key)
    _require_nonempty("dataset", dataset)
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative builtin integer")
    if not isinstance(split, RecipeSelectionSplit):
        raise ValueError("split must be a registered RecipeSelectionSplit")
    role_rows = (split.optimization, split.query, split.gallery)
    role_ids = [[row.example_id for row in rows] for rows in role_rows]
    flattened_ids = [example_id for ids in role_ids for example_id in ids]
    if len(flattened_ids) != len(set(flattened_ids)):
        raise ValueError("probe split example IDs must be unique across roles")
    optimization_labels = {int(row.label) for row in split.optimization}
    validation_labels = {int(row.label) for row in split.query + split.gallery}
    if not optimization_labels.isdisjoint(validation_labels):
        raise ValueError("probe optimization and validation identities must be identity-disjoint")
    if {int(row.label) for row in split.query} != {int(row.label) for row in split.gallery}:
        raise ValueError("probe validation query/gallery identities must match")
    split_sha256 = sha256(
        _canonical_json_bytes(
            {
                "config": asdict(config),
                "dataset": dataset,
                "split_seed": split_seed,
                "roles": [
                    [{"example_id": row.example_id, "label": int(row.label)} for row in rows]
                    for rows in role_rows
                ],
            }
        )
    ).hexdigest()

    expected_dim: int | None = None
    rows_by_id: dict[str, Any] = {}
    for example_id in flattened_ids:
        if example_id not in embeddings_by_example_id:
            raise ValueError(f"probe cache lacks example ID {example_id}")
        row = _probe_vector(embeddings_by_example_id[example_id], expected_dim=expected_dim)
        expected_dim = int(row.shape[0]) if expected_dim is None else expected_dim
        rows_by_id[example_id] = row
    assert expected_dim is not None

    grouped: dict[int, list[ImageExample]] = {}
    for example in sorted(split.optimization, key=lambda row: row.example_id):
        grouped.setdefault(int(example.label), []).append(example)
    eligible_labels = sorted(label for label, rows in grouped.items() if len(rows) >= 2)
    excluded_singletons = len(grouped) - len(eligible_labels)
    if len(eligible_labels) < config.identities_per_batch:
        raise ValueError("probe has fewer eligible identities than identities_per_batch")
    dropped_tail = len(eligible_labels) % config.identities_per_batch

    canonical_optimization = [
        row
        for label in eligible_labels
        for row in sorted(grouped[label], key=lambda item: item.example_id)
    ]
    optimization_index = {row.example_id: index for index, row in enumerate(canonical_optimization)}
    optimization_array = np.stack([rows_by_id[row.example_id] for row in canonical_optimization])
    optimization_array /= np.linalg.norm(optimization_array, axis=1, keepdims=True)
    query_array = np.stack([rows_by_id[row.example_id] for row in split.query])
    query_array /= np.linalg.norm(query_array, axis=1, keepdims=True)
    gallery_array = np.stack([rows_by_id[row.example_id] for row in split.gallery])
    gallery_array /= np.linalg.norm(gallery_array, axis=1, keepdims=True)
    query_labels = np.asarray([int(row.label) for row in split.query], dtype=np.int64)
    gallery_labels = np.asarray([int(row.label) for row in split.gallery], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.from_numpy(optimization_array).to(device=device, dtype=torch.float32)
    query_tensor = torch.from_numpy(query_array).to(device=device, dtype=torch.float32)
    gallery_tensor = torch.from_numpy(gallery_array).to(device=device, dtype=torch.float32)
    initialization_seed = _probe_seed(
        config.protocol_id,
        arm_key,
        dataset,
        split_seed,
        "initialization",
    )
    initialization_generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
    initial_weight = torch.empty((config.output_dim, expected_dim), dtype=torch.float32)
    torch.nn.init.orthogonal_(initial_weight, generator=initialization_generator)

    grid_evaluations: list[ProbeGridEvaluation] = []
    best_candidate: tuple[float, int, float, float, bytes, Any] | None = None
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for learning_rate in sorted(config.learning_rates):
            for temperature in sorted(config.temperatures):
                with torch.random.fork_rng(devices=[]):
                    model = torch.nn.Linear(expected_dim, config.output_dim, bias=False)
                model.weight.data.copy_(initial_weight)
                model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    betas=config.adam_betas,
                    eps=config.adam_eps,
                    weight_decay=config.weight_decay,
                )
                sampler_seed = _probe_seed(
                    config.protocol_id,
                    arm_key,
                    dataset,
                    split_seed,
                    "sampler",
                )
                generator = torch.Generator(device="cpu").manual_seed(sampler_seed)
                epoch_rows: list[ProbeEpochEvaluation] = []
                for epoch in range(config.epochs + 1):
                    metrics = _probe_validation_metrics(
                        model,
                        query_tensor,
                        query_labels,
                        gallery_tensor,
                        gallery_labels,
                    )
                    recall = float(metrics.recall_at_1)
                    epoch_rows.append(
                        ProbeEpochEvaluation(epoch=epoch, validation_recall_at_1=recall)
                    )
                    candidate_key = (-recall, epoch, learning_rate, temperature)
                    if epoch > 0 and (best_candidate is None or candidate_key < best_candidate[:4]):
                        best_candidate = (
                            *candidate_key,
                            _probe_weight_bytes(model.weight),
                            metrics,
                        )
                    if epoch == config.epochs:
                        continue
                    identity_order = torch.randperm(
                        len(eligible_labels), generator=generator
                    ).tolist()
                    usable = len(eligible_labels) - dropped_tail
                    for start in range(0, usable, config.identities_per_batch):
                        labels = [
                            eligible_labels[index]
                            for index in identity_order[start : start + config.identities_per_batch]
                        ]
                        batch_indexes: list[int] = []
                        batch_labels: list[int] = []
                        for label in labels:
                            examples = sorted(grouped[label], key=lambda row: row.example_id)
                            example_order = torch.randperm(
                                len(examples), generator=generator
                            ).tolist()[: config.images_per_identity]
                            for index in example_order:
                                batch_indexes.append(optimization_index[examples[index].example_id])
                                batch_labels.append(label)
                        batch_index_tensor = torch.tensor(
                            batch_indexes, dtype=torch.long, device=device
                        )
                        batch_label_tensor = torch.tensor(
                            batch_labels, dtype=torch.long, device=device
                        )
                        output = torch.nn.functional.normalize(
                            model(input_tensor[batch_index_tensor]), p=2, dim=1
                        )
                        loss = _supervised_infonce_loss(
                            output, batch_label_tensor, temperature=temperature
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                grid_evaluations.append(
                    ProbeGridEvaluation(
                        learning_rate=learning_rate,
                        temperature=temperature,
                        sampler_seed=sampler_seed,
                        epochs=tuple(epoch_rows),
                    )
                )
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn

    assert best_candidate is not None
    _, selected_epoch, selected_lr, selected_temperature, weight_bytes, metrics = best_candidate
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else _cpu_device_identity()
    )
    return BiasFreeProbeResult(
        arm_key=arm_key,
        dataset=dataset,
        input_dim=expected_dim,
        output_dim=config.output_dim,
        split_seed=split_seed,
        split_sha256=split_sha256,
        initialization_seed=initialization_seed,
        excluded_singleton_identities=excluded_singletons,
        dropped_tail_identities=dropped_tail,
        selection_evaluations=len(config.learning_rates) * len(config.temperatures) * config.epochs,
        grid_evaluations=tuple(grid_evaluations),
        selected_learning_rate=selected_lr,
        selected_temperature=selected_temperature,
        selected_epoch=selected_epoch,
        validation_recall_at_1=float(metrics.recall_at_1),
        validation_recall_at_1_points=float(metrics.recall_at_1) * 100.0,
        validation_metrics=metrics,
        selected_weight_bytes=weight_bytes,
        weight_sha256=sha256(weight_bytes).hexdigest(),
        parameter_count=config.output_dim * expected_dim,
        fitted=True,
        bias=False,
        input_normalized=True,
        output_normalized=True,
        loss="supervised_infonce",
        protocol_id=config.protocol_id,
        config=config,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        torch_version=str(torch.__version__),
        device_type=device.type,
        device_name=str(device_name),
    )


@dataclass(frozen=True)
class RemoteFoundationModelSpec:
    """Immutable authority for one remote foundation encoder."""

    arm: str
    model_id: str
    revision: str
    weight_sha256: str
    processor_sha256: str
    config_sha256: str
    pooling: Literal["image_features", "pooler", "cls"]
    resolution: int
    embedding_width: int
    license: str
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("model_id", self.model_id)
        if type(self.revision) is not str or _GIT_REVISION.fullmatch(self.revision) is None:
            raise ValueError("revision must be an immutable lowercase Git object ID")
        _require_sha256("weight_sha256", self.weight_sha256)
        _require_sha256("processor_sha256", self.processor_sha256)
        _require_sha256("config_sha256", self.config_sha256)
        if self.pooling not in {"image_features", "pooler", "cls"}:
            raise ValueError("unsupported pooling rule")
        if type(self.resolution) is not int or self.resolution <= 0:
            raise ValueError("resolution must be a positive builtin integer")
        if type(self.embedding_width) is not int or self.embedding_width <= 0:
            raise ValueError("embedding_width must be a positive builtin integer")
        _require_nonempty("license", self.license)
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be a builtin boolean")


@dataclass(frozen=True)
class LocalCheckpointFoundationSpec:
    """Immutable authority for one trained local retrieval encoder."""

    arm: str
    checkpoint_path: Path
    pretrained_backbone_path: Path
    checkpoint_sha256: str
    resolved_config_sha256: str
    pretrained_backbone_sha256: str
    transform_id: str
    embedding_width: int
    pooling: Literal["embedding"]
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        for name in ("checkpoint_path", "pretrained_backbone_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a concrete pathlib.Path")
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("resolved_config_sha256", self.resolved_config_sha256)
        _require_sha256("pretrained_backbone_sha256", self.pretrained_backbone_sha256)
        _require_nonempty("transform_id", self.transform_id)
        if type(self.embedding_width) is not int or self.embedding_width <= 0:
            raise ValueError("embedding_width must be a positive builtin integer")
        if self.pooling != "embedding":
            raise ValueError("local checkpoint pooling must be embedding")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be a builtin boolean")
        if self.normalize is not True:
            raise ValueError("local BN-Inception comparator requires normalized embeddings")


@dataclass(frozen=True)
class FoundationEncoderAudit:
    """Observed source authority for a foundation encoder."""

    status: Literal["available", "unavailable"]
    model_id: str
    revision: str
    weight_sha256: str | None
    processor_sha256: str | None
    config_sha256: str | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("audit status differs from exact schema")
        _require_nonempty("audit model_id", self.model_id)
        if type(self.revision) is not str or _GIT_REVISION.fullmatch(self.revision) is None:
            raise ValueError("audit revision must be an immutable lowercase Git object ID")
        if self.status == "available":
            _require_sha256("audit weight_sha256", self.weight_sha256)  # type: ignore[arg-type]
            _require_sha256("audit processor_sha256", self.processor_sha256)  # type: ignore[arg-type]
            _require_sha256("audit config_sha256", self.config_sha256)  # type: ignore[arg-type]
            if self.reason is not None:
                raise ValueError("available audit reason must be null")
        else:
            for name in ("weight_sha256", "processor_sha256", "config_sha256"):
                digest = getattr(self, name)
                if digest is not None:
                    _require_sha256(f"audit {name}", digest)
            if type(self.reason) is not str or not self.reason:
                raise ValueError("unavailable audit reason must be a nonempty builtin string")


@dataclass(frozen=True)
class TransformersFoundationEncoder:
    """Loaded remote components bound to their authenticated audit."""

    spec: RemoteFoundationModelSpec
    processor: Any
    model: Any
    device: Any
    audit: FoundationEncoderAudit

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        import numpy as np
        import torch

        _validate_encode_request(images, batch_size, normalize_embeddings)
        if normalize_embeddings is not self.spec.normalize:
            raise ValueError("requested normalization differs from registered normalization")
        dtype = torch.float32 if self.spec.dtype == "float32" else torch.bfloat16
        batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                values = self.processor(
                    images=images[start : start + batch_size],
                    return_tensors="pt",
                    size={"height": self.spec.resolution, "width": self.spec.resolution},
                )
                values = {
                    name: (
                        value.to(device=self.device, dtype=dtype)
                        if torch.is_tensor(value) and value.is_floating_point()
                        else value.to(self.device)
                        if hasattr(value, "to")
                        else value
                    )
                    for name, value in values.items()
                }
                if self.spec.pooling == "image_features":
                    output = self.model.get_image_features(**values)
                else:
                    result = self.model(**values)
                    output = (
                        result.pooler_output
                        if self.spec.pooling == "pooler"
                        else result.last_hidden_state[:, 0, :]
                    )
                _validate_embedding_output(
                    output,
                    batch_count=len(images[start : start + batch_size]),
                    embedding_width=self.spec.embedding_width,
                )
                if normalize_embeddings:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                batches.append(output.detach().cpu().float().numpy())
        return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class LocalFoundationEncoderAudit:
    """Observed byte authority for a trained local retrieval encoder."""

    checkpoint_sha256: str
    resolved_config_sha256: str
    pretrained_backbone_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("local audit checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("local audit resolved_config_sha256", self.resolved_config_sha256)
        _require_sha256(
            "local audit pretrained_backbone_sha256",
            self.pretrained_backbone_sha256,
        )


@dataclass(frozen=True)
class LocalCheckpointFoundationEncoder:
    """Loaded local model bound to all of its authenticated inputs."""

    spec: LocalCheckpointFoundationSpec
    model: Any
    transform: Callable[[object], Any]
    device: Any
    audit: LocalFoundationEncoderAudit

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        import numpy as np
        import torch

        _validate_encode_request(images, batch_size, normalize_embeddings)
        if normalize_embeddings is not self.spec.normalize:
            raise ValueError("requested normalization differs from registered normalization")
        dtype = torch.float32 if self.spec.dtype == "float32" else torch.bfloat16
        batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                values = torch.stack(
                    [self.transform(image) for image in images[start : start + batch_size]]
                ).to(device=self.device, dtype=dtype)
                output = self.model(values)
                _validate_embedding_output(
                    output,
                    batch_count=len(images[start : start + batch_size]),
                    embedding_width=self.spec.embedding_width,
                )
                if normalize_embeddings:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                batches.append(output.detach().cpu().float().numpy())
        return np.concatenate(batches, axis=0)


def _validate_encode_request(
    images: list[object],
    batch_size: int,
    normalize_embeddings: bool,
) -> None:
    if type(images) is not list or not images:
        raise ValueError("images must be a nonempty builtin list")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive builtin integer")
    if type(normalize_embeddings) is not bool:
        raise ValueError("normalize_embeddings must be a builtin boolean")


def _validate_embedding_output(
    output: Any,
    *,
    batch_count: int,
    embedding_width: int,
) -> None:
    import torch

    if not torch.is_tensor(output) or not output.is_floating_point():
        raise ValueError("encoder output must be a floating-point torch tensor")
    if output.ndim != 2:
        raise ValueError("encoder output must be a rank-2 embedding tensor")
    if tuple(output.shape) != (batch_count, embedding_width):
        raise ValueError("encoder output shape differs from registered embedding width")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("encoder output must contain only finite values")


@dataclass(frozen=True)
class NativeFixtureRecord:
    arm: str
    metric: str
    native_value: float | None
    input_sha256: str
    source_sha256: str
    native_cross_check: Literal["available", "unavailable"]
    reason: str | None

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        _require_sha256("input_sha256", self.input_sha256)
        _require_sha256("source_sha256", self.source_sha256)
        if self.native_cross_check == "available":
            if type(self.native_value) is not float or not isfinite(self.native_value):
                raise ValueError("available native fixture value must be a finite builtin float")
            if self.reason is not None:
                raise ValueError("available native fixture cannot have an unavailable reason")
        elif self.native_cross_check == "unavailable":
            if self.native_value is not None:
                raise ValueError("unavailable native fixture cannot carry a value")
            _require_nonempty("reason", self.reason)  # type: ignore[arg-type]
        else:
            raise ValueError("unsupported native cross-check state")


@dataclass(frozen=True)
class MetricToleranceRecord:
    arm: str
    metric: str
    tolerance: float
    frozen_before_execution: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        if type(self.tolerance) is not float or not isfinite(self.tolerance):
            raise ValueError("tolerance must be a finite builtin float")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if self.frozen_before_execution is not True:
            raise ValueError("tolerance must be frozen before execution")


@dataclass(frozen=True)
class FoundationFidelityAudit:
    arm: str
    metric: str
    native_value: float | None
    repository_value: float
    tolerance: float
    provenance: Literal["native_cross_check", "unavailable"]
    passed: bool | None


@dataclass(frozen=True)
class PublishedMetricRecord:
    arm: str
    metric: str
    native_value: float | None
    tolerance: float | None
    source: str
    provenance: Literal["native_cross_check", "repository_only"]

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        _require_nonempty("source", self.source)
        if self.provenance == "native_cross_check":
            if type(self.native_value) is not float or not isfinite(self.native_value):
                raise ValueError("published native value must be a finite builtin float")
            if type(self.tolerance) is not float or not isfinite(self.tolerance):
                raise ValueError("published tolerance must be a finite builtin float")
            if self.tolerance < 0.0:
                raise ValueError("published tolerance must be nonnegative")
        elif self.provenance == "repository_only":
            if self.native_value is not None or self.tolerance is not None:
                raise ValueError("repository-only metric cannot carry native comparison values")
        else:
            raise ValueError("unsupported published metric provenance")


@dataclass(frozen=True)
class PublishedMetricAudit:
    arm: str
    metric: str
    native_value: float | None
    repository_value: float
    tolerance: float | None
    provenance: Literal["native_cross_check", "repository_only"]
    passed: bool | None
    invalidates_confirmatory_claim: bool


@dataclass(frozen=True)
class FoundationGeometryEvaluation:
    geometry: Literal[
        "normalized_cosine",
        "normalized_euclidean",
        "native_unnormalized_euclidean",
    ]
    gallery_order: tuple[tuple[int, ...], ...]
    metrics: Any


@dataclass(frozen=True)
class EncoderBatchCost:
    batch_size: int
    latency_samples_ms: tuple[float, ...]
    latency_p50_ms: float
    latency_p95_ms: float
    peak_memory_bytes: int | None
    mac_status: Literal["available", "unavailable"]
    macs: int | None


@dataclass(frozen=True)
class EncoderCostProfile:
    batches: tuple[EncoderBatchCost, ...]
    parameter_count: int
    warmup_iterations: int
    measured_iterations: int
    descriptor_rows: int
    descriptor_width: int
    descriptor_dtype: str
    descriptor_bytes: int
    python_version: str
    torch_version: str
    numpy_version: str
    transformers_version: str | None
    cuda_version: str | None
    device_type: str
    device_name: str


def _cpu_device_identity() -> str:
    import platform

    fields: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            normalized = key.strip()
            if separator and normalized in {
                "model name",
                "Hardware",
                "CPU implementer",
                "CPU architecture",
                "CPU variant",
                "CPU part",
                "CPU revision",
            }:
                fields.setdefault(normalized, value.strip())
    except OSError:
        pass
    if fields:
        return ";".join(f"{key}={value}" for key, value in fields.items())
    return platform.processor() or platform.machine() or "unknown-cpu"


def profile_foundation_encoder(
    encoder: Any,
    fixtures: Sequence[object],
    batch_sizes: Sequence[int] = (1, 8, 32),
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 50,
    clock_ns: Callable[[], int] | None = None,
    synchronize: Callable[[], None] | None = None,
    reset_peak_memory: Callable[[], None] | None = None,
    read_peak_memory_bytes: Callable[[], int] | None = None,
    mac_counter: Callable[[Any, Sequence[object]], int | None] | None = None,
) -> EncoderCostProfile:
    """Measure registered end-to-end encoder costs without timing warm-ups."""

    import platform
    import time
    from importlib.metadata import PackageNotFoundError, version

    import numpy as np
    import torch

    if not fixtures:
        raise ValueError("profiling fixtures must be nonempty")
    if not batch_sizes or any(type(value) is not int or value <= 0 for value in batch_sizes):
        raise ValueError("profiling batch sizes must be positive builtin integers")
    if max(batch_sizes) > len(fixtures):
        raise ValueError("profiling fixtures must cover the largest batch size")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a nonnegative builtin integer")
    if type(measured_iterations) is not int or measured_iterations <= 0:
        raise ValueError("measured_iterations must be a positive builtin integer")
    if (reset_peak_memory is None) != (read_peak_memory_bytes is None):
        raise ValueError("peak-memory reset and reader must be supplied together")

    device = getattr(encoder, "device", torch.device("cpu"))
    device_type = str(getattr(device, "type", device))
    is_cuda = device_type == "cuda"
    if clock_ns is None:
        clock_ns = time.perf_counter_ns
    if synchronize is None:
        synchronize = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)
    if reset_peak_memory is None:
        reset_peak_memory = (
            (lambda: torch.cuda.reset_peak_memory_stats(device)) if is_cuda else (lambda: None)
        )
    if read_peak_memory_bytes is None:
        memory_reader: Callable[[], int | None] = (
            (lambda: int(torch.cuda.max_memory_allocated(device))) if is_cuda else (lambda: None)
        )
    else:
        memory_reader = read_peak_memory_bytes

    model = getattr(encoder, "model", None)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("profiled encoder must expose model.parameters()")
    parameter_count = sum(int(parameter.numel()) for parameter in parameters())
    normalize = bool(getattr(getattr(encoder, "spec", None), "normalize", True))
    batch_rows: list[EncoderBatchCost] = []
    descriptor_width: int | None = None
    descriptor_dtype: str | None = None
    for batch_size in batch_sizes:
        batch = list(fixtures[:batch_size])
        for _ in range(warmup_iterations):
            encoder.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )
        reset_peak_memory()
        samples: list[float] = []
        for _ in range(measured_iterations):
            synchronize()
            started = clock_ns()
            output = encoder.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )
            synchronize()
            elapsed_ns = clock_ns() - started
            if type(elapsed_ns) is not int or elapsed_ns < 0:
                raise ValueError("profiling clock must return monotonic integer nanoseconds")
            samples.append(elapsed_ns / 1_000_000.0)
            output_array = np.asarray(output)
            if (
                output_array.ndim != 2
                or output_array.shape[0] != batch_size
                or not np.issubdtype(output_array.dtype, np.floating)
            ):
                raise ValueError("profiled encoder output must be a floating rank-2 batch")
            current_width = int(output_array.shape[1])
            current_dtype = output_array.dtype.name
            if descriptor_width is None:
                descriptor_width = current_width
                descriptor_dtype = current_dtype
            elif current_width != descriptor_width or current_dtype != descriptor_dtype:
                raise ValueError("profiled descriptor shape/dtype changed across batch sizes")
        p50, p95 = np.percentile(np.asarray(samples, dtype=np.float64), (50, 95))
        peak_memory = memory_reader()
        if peak_memory is not None and (type(peak_memory) is not int or peak_memory < 0):
            raise ValueError("peak memory must be a nonnegative builtin integer or null")
        macs = mac_counter(encoder, batch) if mac_counter is not None else None
        if macs is not None and (type(macs) is not int or macs < 0):
            raise ValueError("MAC counter must return a nonnegative builtin integer or null")
        batch_rows.append(
            EncoderBatchCost(
                batch_size=batch_size,
                latency_samples_ms=tuple(samples),
                latency_p50_ms=float(p50),
                latency_p95_ms=float(p95),
                peak_memory_bytes=peak_memory,
                mac_status="available" if macs is not None else "unavailable",
                macs=macs,
            )
        )
    assert descriptor_width is not None and descriptor_dtype is not None
    descriptor_itemsize = int(np.dtype(descriptor_dtype).itemsize)
    device_name = torch.cuda.get_device_name(device) if is_cuda else _cpu_device_identity()
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    return EncoderCostProfile(
        batches=tuple(batch_rows),
        parameter_count=parameter_count,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        descriptor_rows=len(fixtures),
        descriptor_width=descriptor_width,
        descriptor_dtype=descriptor_dtype,
        descriptor_bytes=len(fixtures) * descriptor_width * descriptor_itemsize,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        transformers_version=transformers_version,
        cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
        device_type=device_type,
        device_name=str(device_name),
    )


def _normalize_rows(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("geometry embeddings must be nonempty rank-2 arrays")
    if not bool(np.isfinite(array).all()):
        raise ValueError("geometry embeddings must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool((norms == 0.0).any()):
        raise ValueError("normalized geometry cannot contain zero-norm rows")
    return array / norms


def _stable_top_indices(values: Any, *, depth: int) -> Any:
    import numpy as np

    if depth >= values.shape[0]:
        return np.argsort(values, kind="stable")
    candidates = np.argpartition(values, kth=depth - 1)[:depth]
    boundary = values[candidates].max()
    below = np.flatnonzero(values < boundary)
    ties = np.flatnonzero(values == boundary)[: depth - below.shape[0]]
    selected = np.concatenate((below, ties))
    return selected[np.lexsort((selected, values[selected]))]


def _gallery_order(
    query: Any,
    gallery: Any,
    *,
    geometry: Literal["cosine", "euclidean"],
    depth: int,
) -> tuple[tuple[int, ...], ...]:
    import numpy as np

    if type(depth) is not int or depth <= 0:
        raise ValueError("geometry ranking depth must be a positive builtin integer")
    depth = min(depth, gallery.shape[0])
    gallery_norms = np.sum(gallery * gallery, axis=1)
    rows: list[tuple[int, ...]] = []
    for start in range(0, query.shape[0], 256):
        chunk = query[start : start + 256]
        if geometry == "cosine":
            values = -(chunk @ gallery.T)
        else:
            values = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + gallery_norms[np.newaxis, :]
                - (2.0 * chunk @ gallery.T)
            )
            values = np.maximum(values, 0.0)
        if not bool(np.isfinite(values).all()):
            raise ValueError("geometry ranking produced non-finite values")
        rows.extend(
            tuple(int(index) for index in _stable_top_indices(row, depth=depth)) for row in values
        )
    return tuple(rows)


def _metrics_from_gallery_order(
    gallery_order: tuple[tuple[int, ...], ...],
    query_labels: Any,
    gallery_labels: Any,
) -> Any:
    import numpy as np

    from sfora.image_benchmark import ImageRetrievalMetrics

    precision_at_1: list[float] = []
    recalls: dict[int, list[float]] = {cutoff: [] for cutoff in (1, 2, 4, 8, 10, 20, 30, 100)}
    average_precisions: list[float] = []
    relevant_counts: list[int] = []
    for query_index, order in enumerate(gallery_order):
        matches = gallery_labels == query_labels[query_index]
        relevant_count = int(matches.sum())
        if relevant_count == 0:
            continue
        ordered_matches = matches[np.asarray(order, dtype=np.int64)]
        precision_at_1.append(float(ordered_matches[0]))
        for cutoff in recalls:
            recalls[cutoff].append(float(bool(ordered_matches[:cutoff].any())))
        top_r = ordered_matches[:relevant_count]
        relevant_ranks = np.flatnonzero(top_r) + 1
        average_precisions.append(
            sum(float(top_r[:rank].sum() / rank) for rank in relevant_ranks) / relevant_count
        )
        relevant_counts.append(relevant_count)
    if not average_precisions:
        raise ValueError("no query shares an identity label with any gallery item")
    return ImageRetrievalMetrics(
        precision_at_1=float(np.mean(precision_at_1)),
        recall_at_1=float(np.mean(recalls[1])),
        recall_at_2=float(np.mean(recalls[2])),
        recall_at_4=float(np.mean(recalls[4])),
        recall_at_8=float(np.mean(recalls[8])),
        map_at_r=float(np.mean(average_precisions)),
        mean_relevant_items=float(np.mean(relevant_counts)),
        evaluated_queries=len(average_precisions),
        total_queries=len(gallery_order),
        recall_at_10=float(np.mean(recalls[10])),
        recall_at_20=float(np.mean(recalls[20])),
        recall_at_30=float(np.mean(recalls[30])),
        recall_at_100=float(np.mean(recalls[100])),
    )


def evaluate_foundation_geometries(
    query_embeddings: Any,
    query_labels: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
) -> tuple[FoundationGeometryEvaluation, ...]:
    """Evaluate every preregistered geometry without selecting a winner."""

    import numpy as np

    query = np.asarray(query_embeddings, dtype=np.float64)
    gallery = np.asarray(gallery_embeddings, dtype=np.float64)
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("query/gallery geometry arrays must share a rank-2 feature shape")
    normalized_query = _normalize_rows(query)
    normalized_gallery = _normalize_rows(gallery)
    query_label_array = np.asarray(query_labels, dtype=np.int64)
    gallery_label_array = np.asarray(gallery_labels, dtype=np.int64)
    if query_label_array.ndim != 1 or gallery_label_array.ndim != 1:
        raise ValueError("query/gallery labels must be rank-1 arrays")
    if (
        query.shape[0] != query_label_array.shape[0]
        or gallery.shape[0] != gallery_label_array.shape[0]
    ):
        raise ValueError("geometry embeddings and labels must have matching row counts")
    gallery_label_counts = {
        int(label): int(count)
        for label, count in zip(
            *np.unique(gallery_label_array, return_counts=True),
            strict=True,
        )
    }
    max_relevant_count = max(
        (gallery_label_counts.get(int(label), 0) for label in query_label_array),
        default=0,
    )
    depth = max(100, max_relevant_count)
    cosine_order = _gallery_order(
        normalized_query,
        normalized_gallery,
        geometry="cosine",
        depth=depth,
    )
    normalized_euclidean_order = _gallery_order(
        normalized_query,
        normalized_gallery,
        geometry="euclidean",
        depth=depth,
    )
    native_order = _gallery_order(
        query,
        gallery,
        geometry="euclidean",
        depth=depth,
    )
    return (
        FoundationGeometryEvaluation(
            geometry="normalized_cosine",
            gallery_order=cosine_order,
            metrics=_metrics_from_gallery_order(
                cosine_order,
                query_label_array,
                gallery_label_array,
            ),
        ),
        FoundationGeometryEvaluation(
            geometry="normalized_euclidean",
            gallery_order=normalized_euclidean_order,
            metrics=_metrics_from_gallery_order(
                normalized_euclidean_order,
                query_label_array,
                gallery_label_array,
            ),
        ),
        FoundationGeometryEvaluation(
            geometry="native_unnormalized_euclidean",
            gallery_order=native_order,
            metrics=_metrics_from_gallery_order(
                native_order,
                query_label_array,
                gallery_label_array,
            ),
        ),
    )


@dataclass(frozen=True)
class EmbeddingCacheKeyV2:
    """Complete content identity for one stable foundation embedding export."""

    arm: str
    model_revision: str
    weight_sha256: str
    processor_sha256: str
    transform_id: str
    resolution: int
    dtype: Literal["float32", "bfloat16"]
    storage_dtype: Literal["float32"]
    normalize: bool
    dataset_rows_sha256: str
    split: str

    def __post_init__(self) -> None:
        _require_nonempty("cache arm", self.arm)
        if (
            type(self.model_revision) is not str
            or _GIT_REVISION.fullmatch(self.model_revision) is None
        ):
            raise ValueError("cache model_revision must be an immutable lowercase object ID")
        _require_sha256("cache weight_sha256", self.weight_sha256)
        _require_sha256("cache processor_sha256", self.processor_sha256)
        _require_nonempty("cache transform_id", self.transform_id)
        if type(self.resolution) is not int or self.resolution <= 0:
            raise ValueError("cache resolution must be a positive builtin integer")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("cache dtype differs from registered choices")
        if self.storage_dtype != "float32":
            raise ValueError("cache storage_dtype must be float32")
        if type(self.normalize) is not bool:
            raise ValueError("cache normalize must be a builtin boolean")
        _require_sha256("cache dataset_rows_sha256", self.dataset_rows_sha256)
        _require_nonempty("cache split", self.split)

    @classmethod
    def from_model_spec(
        cls,
        spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec,
        *,
        dataset_rows_sha256: str,
        split: str,
        resolution: int | None = None,
    ) -> EmbeddingCacheKeyV2:
        if isinstance(spec, RemoteFoundationModelSpec):
            if resolution is not None and resolution != spec.resolution:
                raise ValueError("explicit remote resolution differs from model spec")
            return cls(
                arm=spec.arm,
                model_revision=spec.revision,
                weight_sha256=spec.weight_sha256,
                processor_sha256=spec.processor_sha256,
                transform_id=f"{spec.model_id}:{spec.pooling}",
                resolution=spec.resolution,
                dtype=spec.dtype,
                storage_dtype="float32",
                normalize=spec.normalize,
                dataset_rows_sha256=dataset_rows_sha256,
                split=split,
            )
        if resolution is None:
            raise ValueError("local cache identity requires an explicit resolution")
        return cls(
            arm=spec.arm,
            model_revision=spec.checkpoint_sha256,
            weight_sha256=spec.pretrained_backbone_sha256,
            processor_sha256=spec.resolved_config_sha256,
            transform_id=spec.transform_id,
            resolution=resolution,
            dtype=spec.dtype,
            storage_dtype="float32",
            normalize=spec.normalize,
            dataset_rows_sha256=dataset_rows_sha256,
            split=split,
        )

    def _json_object(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "model_revision": self.model_revision,
            "weight_sha256": self.weight_sha256,
            "processor_sha256": self.processor_sha256,
            "transform_id": self.transform_id,
            "resolution": self.resolution,
            "dtype": self.dtype,
            "storage_dtype": self.storage_dtype,
            "normalize": self.normalize,
            "dataset_rows_sha256": self.dataset_rows_sha256,
            "split": self.split,
        }

    def cache_path(self, root: Path) -> Path:
        if not isinstance(root, Path):
            raise ValueError("cache root must be a concrete pathlib.Path")
        digest = sha256(_canonical_json_bytes(self._json_object())).hexdigest()
        return root / f"foundation-cache-v2-{digest}.npz"


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical JSON data") from error


def _ordered_nonempty_strings(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a nonempty builtin tuple")
    for value in values:
        _require_nonempty(name, value)
    return tuple(values)


def _embedding_bytes(embeddings: Any, *, storage_dtype: str) -> bytes:
    import numpy as np

    if type(embeddings) is not np.ndarray or embeddings.ndim != 2:
        raise ValueError("embeddings must be a rank-2 NumPy array")
    if storage_dtype != "float32" or embeddings.dtype != np.dtype("float32"):
        raise ValueError("embedding array dtype differs from registered export dtype")
    if not bool(np.isfinite(embeddings).all()):
        raise ValueError("embeddings must contain only finite values")
    return np.ascontiguousarray(embeddings).tobytes(order="C")


def _cache_metadata(
    *,
    key: EmbeddingCacheKeyV2,
    embeddings: Any,
    ids: tuple[str, ...],
    labels: tuple[str, ...],
) -> dict[str, object]:
    embedding_bytes = _embedding_bytes(embeddings, storage_dtype=key.storage_dtype)
    return {
        "schema_version": "foundation-embedding-cache-v2",
        "key": key._json_object(),
        "ids": list(ids),
        "labels": list(labels),
        "shape": list(embeddings.shape),
        "dtype": key.storage_dtype,
        "embedding_sha256": sha256(embedding_bytes).hexdigest(),
    }


def load_embeddings_v2(
    path: Path,
    *,
    key: EmbeddingCacheKeyV2,
    expected_ids: tuple[str, ...],
    expected_labels: tuple[str, ...],
) -> Any:
    import numpy as np

    ids = _ordered_nonempty_strings("expected row IDs", expected_ids)
    labels = _ordered_nonempty_strings("expected labels", expected_labels)
    if path != key.cache_path(path.parent):
        raise ValueError("cache-v2 path differs from content-addressed key")
    if path.is_symlink() or not path.is_file():
        raise ValueError("cache-v2 path must be a regular non-symlink file")
    try:
        with np.load(path, allow_pickle=False) as value:
            if set(value.files) != {"embeddings", "metadata_json"}:
                raise ValueError("cache-v2 members differ from exact schema")
            embeddings = value["embeddings"]
            metadata_array = value["metadata_json"]
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "cache-v2" in str(error):
            raise
        raise ValueError("cache-v2 archive is invalid") from error
    if metadata_array.dtype != np.dtype("uint8") or metadata_array.ndim != 1:
        raise ValueError("cache-v2 metadata bytes differ from exact schema")
    if embeddings.ndim != 2:
        raise ValueError("cache-v2 embeddings must be rank-2")
    if embeddings.shape[0] != len(ids) or len(ids) != len(labels):
        raise ValueError("cache-v2 embedding, ID, and label row counts differ")
    expected = _cache_metadata(key=key, embeddings=embeddings, ids=ids, labels=labels)
    expected_metadata_bytes = _canonical_json_bytes(expected)
    try:
        metadata = json.loads(
            bytes(metadata_array).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cache-v2 metadata is not strict UTF-8 JSON") from error
    if type(metadata) is not dict:
        raise ValueError("cache-v2 metadata root must be a JSON object")
    if metadata.get("schema_version") != "foundation-embedding-cache-v2":
        raise ValueError("cache-v2 schema version differs")
    if metadata.get("ids") != list(ids):
        raise ValueError("cache-v2 row IDs differ")
    if metadata.get("labels") != list(labels):
        raise ValueError("cache-v2 labels differ")
    if bytes(metadata_array) != expected_metadata_bytes:
        raise ValueError("cache-v2 metadata bytes are not canonical or differ")
    if metadata != expected:
        raise ValueError("cache-v2 metadata or embedding digest differs")
    return embeddings


def export_embeddings_v2(
    path: Path,
    *,
    key: EmbeddingCacheKeyV2,
    embeddings: Any,
    ids: tuple[str, ...],
    labels: tuple[str, ...],
) -> Path:
    import numpy as np

    ordered_ids = _ordered_nonempty_strings("row IDs", ids)
    ordered_labels = _ordered_nonempty_strings("labels", labels)
    if path != key.cache_path(path.parent):
        raise ValueError("cache-v2 path differs from content-addressed key")
    if len(ordered_ids) != len(ordered_labels) or len(ordered_ids) != embeddings.shape[0]:
        raise ValueError("embedding, ID, and label row counts differ")
    metadata = _cache_metadata(
        key=key,
        embeddings=embeddings,
        ids=ordered_ids,
        labels=ordered_labels,
    )
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("cache-v2 parent must be a real directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    published_identity: tuple[int, int] | None = None
    try:
        with temp.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            np.savez_compressed(
                handle,
                embeddings=np.ascontiguousarray(embeddings),
                metadata_json=np.frombuffer(_canonical_json_bytes(metadata), dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temp_stat = temp.stat()
        published_identity = (temp_stat.st_dev, temp_stat.st_ino)
        os.link(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            temp.unlink()
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=ordered_ids,
            expected_labels=ordered_labels,
        )
    except BaseException:
        if published_identity is not None and path.is_file() and not path.is_symlink():
            path_stat = path.stat()
            if (path_stat.st_dev, path_stat.st_ino) == published_identity:
                path.unlink()
        if temp.is_file() and not temp.is_symlink():
            temp.unlink()
        raise
    return path


def _record_keys(records: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    return tuple((record.arm, record.metric) for record in records)


def validate_native_fixture_authority(
    fixtures: Sequence[NativeFixtureRecord],
    tolerances: Sequence[MetricToleranceRecord],
    *,
    registered_pairs: Sequence[tuple[str, str]],
) -> None:
    """Require complete, exact, ordered fixture and tolerance authorities."""

    expected = tuple(registered_pairs)
    fixture_keys = _record_keys(fixtures)
    tolerance_keys = _record_keys(tolerances)
    if set(fixture_keys) != set(expected):
        raise ValueError("fixture key set differs from registered arm/metric pairs")
    if set(tolerance_keys) != set(expected):
        raise ValueError("tolerance key set differs from registered arm/metric pairs")
    if fixture_keys != expected or tolerance_keys != expected:
        raise ValueError("fixture and tolerance ordered keys differ from registered order")


def _finite_repository_value(values: Mapping[str, float], metric: str) -> float:
    try:
        value = values[metric]
    except KeyError as error:
        raise ValueError(f"repository value missing for metric {metric}") from error
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"repository value for {metric} must be a finite builtin float")
    return value


def verify_native_fixture(
    *,
    arm: str,
    encoder: object,
    fixture_inputs: Mapping[str, Path],
    native_sources: Mapping[str, Path],
    repository_metric: Callable[[object, Path, Path, str], float],
    fixtures: Sequence[NativeFixtureRecord],
    tolerances: Sequence[MetricToleranceRecord],
    registered_pairs: Sequence[tuple[str, str]],
) -> tuple[FoundationFidelityAudit, ...]:
    encoder_spec = getattr(encoder, "spec", None)
    if getattr(encoder_spec, "arm", None) != arm:
        raise ValueError("encoder arm differs from requested fixture arm")
    validate_native_fixture_authority(
        fixtures,
        tolerances,
        registered_pairs=registered_pairs,
    )
    expected_arm_pairs = tuple(pair for pair in registered_pairs if pair[0] == arm)
    if not expected_arm_pairs:
        raise ValueError("requested arm has no registered native fixture pairs")
    fixture_rows = tuple(row for row in fixtures if row.arm == arm)
    tolerance_rows = tuple(row for row in tolerances if row.arm == arm)
    if _record_keys(fixture_rows) != expected_arm_pairs:
        raise ValueError("requested arm fixture order differs from registered order")
    if _record_keys(tolerance_rows) != expected_arm_pairs:
        raise ValueError("requested arm tolerance order differs from registered order")
    audits: list[FoundationFidelityAudit] = []
    for fixture, tolerance in zip(fixture_rows, tolerance_rows, strict=True):
        try:
            input_path = fixture_inputs[fixture.metric]
            source_path = native_sources[fixture.metric]
        except KeyError as error:
            raise ValueError(
                f"registered fixture path missing for metric {fixture.metric}"
            ) from error
        if _sha256_file(input_path, field="fixture input") != fixture.input_sha256:
            raise ValueError(f"fixture input digest differs for metric {fixture.metric}")
        if _sha256_file(source_path, field="native source") != fixture.source_sha256:
            raise ValueError(f"native source digest differs for metric {fixture.metric}")
        repository_value = repository_metric(
            encoder,
            input_path,
            source_path,
            fixture.metric,
        )
        if type(repository_value) is not float or not isfinite(repository_value):
            raise ValueError(
                f"repository value for {fixture.metric} must be a finite builtin float"
            )
        available = fixture.native_cross_check == "available"
        passed = (
            abs(repository_value - fixture.native_value) <= tolerance.tolerance
            if available and fixture.native_value is not None
            else None
        )
        audits.append(
            FoundationFidelityAudit(
                arm=arm,
                metric=fixture.metric,
                native_value=fixture.native_value,
                repository_value=repository_value,
                tolerance=tolerance.tolerance,
                provenance="native_cross_check" if available else "unavailable",
                passed=passed,
            )
        )
    return tuple(audits)


def cross_check_published_metrics(
    *,
    arm: str,
    repository_values: Mapping[str, float],
    records: Sequence[PublishedMetricRecord],
    registered_pairs: Sequence[tuple[str, str]],
) -> tuple[PublishedMetricAudit, ...]:
    expected = tuple(registered_pairs)
    if _record_keys(records) != expected:
        raise ValueError("published metric ordered keys differ from registered order")
    expected_arm_pairs = tuple(pair for pair in expected if pair[0] == arm)
    if not expected_arm_pairs:
        raise ValueError("requested arm has no registered published metric pairs")
    audits: list[PublishedMetricAudit] = []
    for record in records:
        if record.arm != arm:
            continue
        repository_value = _finite_repository_value(repository_values, record.metric)
        available = record.provenance == "native_cross_check"
        passed = (
            abs(repository_value - record.native_value) <= record.tolerance
            if available and record.native_value is not None and record.tolerance is not None
            else None
        )
        audits.append(
            PublishedMetricAudit(
                arm=arm,
                metric=record.metric,
                native_value=record.native_value,
                repository_value=repository_value,
                tolerance=record.tolerance,
                provenance=record.provenance,
                passed=passed,
                invalidates_confirmatory_claim=passed is False,
            )
        )
    return tuple(audits)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON authority must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authority is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("authority root must be a JSON object")
    return value


def _require_ordered_keys(value: dict[str, Any], expected: tuple[str, ...], *, name: str) -> None:
    if tuple(value) != expected:
        raise ValueError(f"{name} keys differ from exact schema")


def load_native_fixture_authority(
    fixture_path: Path,
    tolerance_path: Path,
    *,
    registered_pairs: Sequence[tuple[str, str]],
    require_frozen: bool = True,
) -> tuple[tuple[NativeFixtureRecord, ...], tuple[MetricToleranceRecord, ...]]:
    fixture_value = _load_strict_json(fixture_path)
    tolerance_value = _load_strict_json(tolerance_path)
    _require_ordered_keys(
        fixture_value,
        ("schema_version", "status", "records"),
        name="fixture authority",
    )
    _require_ordered_keys(
        tolerance_value,
        ("schema_version", "status", "records"),
        name="tolerance authority",
    )
    if fixture_value["schema_version"] != "foundation-native-fixtures-v1":
        raise ValueError("fixture schema version differs")
    if tolerance_value["schema_version"] != "foundation-metric-tolerances-v1":
        raise ValueError("tolerance schema version differs")
    for name, value in (("fixture", fixture_value), ("tolerance", tolerance_value)):
        if value["status"] not in {"prospective_unfrozen", "frozen"}:
            raise ValueError(f"{name} authority status differs")
        if require_frozen and value["status"] != "frozen":
            raise ValueError(f"{name} authority is not frozen")
    if fixture_value["status"] != tolerance_value["status"]:
        raise ValueError("fixture and tolerance authority statuses differ")
    if fixture_value["status"] == "prospective_unfrozen" and (
        fixture_value["records"] != [] or tolerance_value["records"] != []
    ):
        raise ValueError("prospective unfrozen fixture authorities must be empty")
    if type(fixture_value["records"]) is not list:
        raise ValueError("fixture records must be a JSON array")
    if type(tolerance_value["records"]) is not list:
        raise ValueError("tolerance records must be a JSON array")
    fixtures: list[NativeFixtureRecord] = []
    for record in fixture_value["records"]:
        if type(record) is not dict:
            raise ValueError("fixture record must be a JSON object")
        _require_ordered_keys(
            record,
            (
                "arm",
                "metric",
                "native_value",
                "input_sha256",
                "source_sha256",
                "native_cross_check",
                "reason",
            ),
            name="fixture record",
        )
        fixtures.append(NativeFixtureRecord(**record))
    tolerances: list[MetricToleranceRecord] = []
    for record in tolerance_value["records"]:
        if type(record) is not dict:
            raise ValueError("tolerance record must be a JSON object")
        _require_ordered_keys(
            record,
            ("arm", "metric", "tolerance", "frozen_before_execution"),
            name="tolerance record",
        )
        tolerances.append(MetricToleranceRecord(**record))
    validate_native_fixture_authority(
        fixtures,
        tolerances,
        registered_pairs=registered_pairs,
    )
    return tuple(fixtures), tuple(tolerances)


def load_published_metric_register(
    path: Path,
    *,
    require_frozen: bool = True,
) -> tuple[PublishedMetricRecord, ...]:
    value = _load_strict_json(path)
    _require_ordered_keys(
        value,
        ("schema_version", "status", "records"),
        name="published metric authority",
    )
    if value["schema_version"] != "foundation-published-metrics-v1":
        raise ValueError("published metric schema version differs")
    if value["status"] not in {"prospective_unfrozen", "frozen"}:
        raise ValueError("published metric authority status differs")
    if require_frozen and value["status"] != "frozen":
        raise ValueError("published metric authority is not frozen")
    if value["status"] == "prospective_unfrozen" and value["records"] != []:
        raise ValueError("prospective unfrozen published authority must be empty")
    if type(value["records"]) is not list:
        raise ValueError("published metric records must be a JSON array")
    records: list[PublishedMetricRecord] = []
    for record in value["records"]:
        if type(record) is not dict:
            raise ValueError("published metric record must be a JSON object")
        _require_ordered_keys(
            record,
            ("arm", "metric", "native_value", "tolerance", "source", "provenance"),
            name="published metric record",
        )
        records.append(PublishedMetricRecord(**record))
    return tuple(records)


def _load_transformers_dependencies() -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("install the research extra to load foundation encoders") from error
    return AutoImageProcessor, AutoModel


def _snapshot_download(
    repo_id: str,
    *,
    revision: str,
    allow_patterns: tuple[str, ...],
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install the research extra to authenticate remote artifacts") from error
    return str(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(allow_patterns),
        )
    )


def _artifact_set_sha256(root: Path, paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return sha256(_remote_artifact_bytes(root, paths[0])).hexdigest()
    digest = sha256(b"foundation-artifact-set-v1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        payload = _remote_artifact_bytes(root, path)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _remote_artifact_bytes(root: Path, path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"remote artifact is not a file: {path}")
    resolved_root = root.resolve(strict=True)
    cache_scope = (
        resolved_root.parent.parent if resolved_root.parent.name == "snapshots" else resolved_root
    )
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(cache_scope):
        raise ValueError("remote artifact symlink escapes authenticated cache scope")
    return resolved_path.read_bytes()


def _observe_remote_snapshot(
    spec: RemoteFoundationModelSpec,
) -> tuple[Path | None, FoundationEncoderAudit]:
    try:
        root = Path(
            _snapshot_download(
                spec.model_id,
                revision=spec.revision,
                allow_patterns=_REMOTE_ALLOW_PATTERNS,
            )
        )
    except OSError as error:
        return None, FoundationEncoderAudit(
            status="unavailable",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=None,
            processor_sha256=None,
            config_sha256=None,
            reason=f"snapshot unavailable: {error}",
        )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot root must be a real directory")
    config_path = root / "config.json"
    processor_paths = tuple(
        path
        for name in ("preprocessor_config.json", "processor_config.json")
        if (path := root / name).is_file()
    )
    weight_paths = tuple(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".safetensors.index.json")
            or (path.name.startswith("pytorch_model") and path.name.endswith(".bin"))
            or (path.name.startswith("pytorch_model") and path.name.endswith(".bin.index.json"))
        )
    )
    missing = []
    if not config_path.is_file():
        missing.append("config.json")
    if not processor_paths:
        missing.append("processor configuration")
    if not weight_paths:
        missing.append("model weights")
    if missing:
        return None, FoundationEncoderAudit(
            status="unavailable",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=None,
            processor_sha256=None,
            config_sha256=None,
            reason=f"snapshot lacks {', '.join(missing)}",
        )
    return root, FoundationEncoderAudit(
        status="available",
        model_id=spec.model_id,
        revision=spec.revision,
        weight_sha256=_artifact_set_sha256(root, weight_paths),
        processor_sha256=_artifact_set_sha256(root, processor_paths),
        config_sha256=sha256(_remote_artifact_bytes(root, config_path)).hexdigest(),
        reason=None,
    )


def _observe_remote_artifacts(spec: RemoteFoundationModelSpec) -> FoundationEncoderAudit:
    """Return the structured availability audit without loading model code."""

    return _observe_remote_snapshot(spec)[1]


def _sha256_file(path: Path, *, field: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} path must be a regular non-symlink file")
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("resolved training config is not canonical JSON data") from error
    return sha256(payload).hexdigest()


def _observe_local_artifacts(
    spec: LocalCheckpointFoundationSpec,
    checkpoint: Mapping[str, Any],
) -> LocalFoundationEncoderAudit:
    if type(checkpoint) is not dict or type(checkpoint.get("training_config")) is not dict:
        raise ValueError("local checkpoint lacks builtin training_config authority")
    return LocalFoundationEncoderAudit(
        checkpoint_sha256=_sha256_file(spec.checkpoint_path, field="checkpoint"),
        resolved_config_sha256=_canonical_json_sha256(checkpoint["training_config"]),
        pretrained_backbone_sha256=_sha256_file(
            spec.pretrained_backbone_path,
            field="pretrained backbone",
        ),
    )


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the research extra to load the local comparator") from error
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict:
        raise ValueError("local checkpoint root must be a builtin dictionary")
    return value


def _build_local_bn_inception(*, embedding_size: int, add_gmp: bool) -> Any:
    from sfora.bn_inception import build_bn_inception

    return build_bn_inception(
        embedding_size=embedding_size,
        pretrained=False,
        add_gmp=add_gmp,
    )


def _build_local_eval_transform(spec: LocalCheckpointFoundationSpec) -> Callable[[object], Any]:
    if spec.transform_id != "proxy-anchor-eval-224-v1":
        raise ValueError("unsupported local evaluation transform")
    from sfora.image_end_to_end import ImageEndToEndConfig, _default_transform_factory

    return _default_transform_factory(
        ImageEndToEndConfig(
            backbone_name="bn_inception",
            input_size=224,
        ),
        False,
    )


def _load_local_checkpoint_model(
    spec: LocalCheckpointFoundationSpec,
    checkpoint: dict[str, Any] | None = None,
) -> Any:
    if checkpoint is None:
        checkpoint = _torch_load_checkpoint(spec.checkpoint_path)
    required = ("state_dict", "arch", "training_config")
    if any(name not in checkpoint for name in required):
        raise ValueError("local checkpoint lacks required trained-model fields")
    arch = checkpoint["arch"]
    training_config = checkpoint["training_config"]
    if type(arch) is not dict or type(training_config) is not dict:
        raise ValueError("local checkpoint architecture/config must be builtin dictionaries")
    if _canonical_json_sha256(training_config) != spec.resolved_config_sha256:
        raise ValueError("checkpoint training_config digest differs from resolved authority")
    arch_fields = (
        "backbone_name",
        "pretrained_weights",
        "head_pooling",
        "embedding_dimensions",
        "embedding_head_init",
        "embedding_layer_norm",
    )
    if set(arch) != set(arch_fields):
        raise ValueError("checkpoint architecture key set differs")
    expected_arch = {field: training_config.get(field) for field in arch_fields}
    if arch != expected_arch:
        raise ValueError("checkpoint architecture differs from resolved training config")
    supported_arch = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "head_pooling": "avg_max",
        "embedding_dimensions": spec.embedding_width,
        "embedding_head_init": "default",
        "embedding_layer_norm": False,
    }
    if expected_arch != supported_arch:
        raise ValueError("checkpoint architecture is unsupported by the local comparator")
    model = _build_local_bn_inception(
        embedding_size=spec.embedding_width,
        add_gmp=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _require_matching_remote_audit(
    spec: RemoteFoundationModelSpec,
    audit: FoundationEncoderAudit,
) -> None:
    if audit.status != "available" or audit.reason is not None:
        raise ValueError("remote encoder authority is unavailable")
    expected = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "weight_sha256": spec.weight_sha256,
        "processor_sha256": spec.processor_sha256,
        "config_sha256": spec.config_sha256,
    }
    for field, value in expected.items():
        if getattr(audit, field) != value:
            raise ValueError(f"observed {field} differs from registered authority")


def load_foundation_encoder(
    spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec,
) -> TransformersFoundationEncoder | LocalCheckpointFoundationEncoder:
    """Load one encoder only after its immutable source authority matches."""

    if isinstance(spec, LocalCheckpointFoundationSpec):
        import torch

        if _sha256_file(spec.checkpoint_path, field="checkpoint") != spec.checkpoint_sha256:
            raise ValueError("observed checkpoint_sha256 differs from registered authority")
        if (
            _sha256_file(spec.pretrained_backbone_path, field="pretrained backbone")
            != spec.pretrained_backbone_sha256
        ):
            raise ValueError(
                "observed pretrained_backbone_sha256 differs from registered authority"
            )
        checkpoint = _torch_load_checkpoint(spec.checkpoint_path)
        local_audit = _observe_local_artifacts(spec, checkpoint)
        for field in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "pretrained_backbone_sha256",
        ):
            if getattr(local_audit, field) != getattr(spec, field):
                raise ValueError(f"observed {field} differs from registered authority")
        dtype = torch.float32 if spec.dtype == "float32" else torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_local_checkpoint_model(spec, checkpoint)
        model.to(device=device, dtype=dtype)
        model.eval()
        return LocalCheckpointFoundationEncoder(
            spec=spec,
            model=model,
            transform=_build_local_eval_transform(spec),
            device=device,
            audit=local_audit,
        )
    root, remote_audit = _observe_remote_snapshot(spec)
    _require_matching_remote_audit(spec, remote_audit)
    if root is None:
        raise ValueError("available remote audit lacks an authenticated snapshot root")
    processor_class, model_class = _load_transformers_dependencies()
    import torch

    dtype = torch.float32 if spec.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = processor_class.from_pretrained(
        str(root),
        revision=spec.revision,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        str(root),
        revision=spec.revision,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return TransformersFoundationEncoder(
        spec=spec,
        processor=processor,
        model=model,
        device=device,
        audit=remote_audit,
    )
