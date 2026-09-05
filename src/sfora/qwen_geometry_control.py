"""Frozen pooling boundary for the paired Qwen geometry-control experiment."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _registered_protocol_values() -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "arms": ("mean", "attention"),
        "seeds": (17, 29, 43),
        "optimization_classes": tuple(range(49)),
        "clean_development_classes": tuple(range(49, 82)),
        "burned_diagnostic_classes": tuple(range(82, 98)),
        "image_size": 224,
        "embedding_dimensions": 4096,
        "logical_batch_size": 64,
        "classes_per_batch": 16,
        "images_per_class": 4,
        "epochs": 3,
        "steps_per_epoch": 61,
        "optimizer_updates": 183,
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "tower_learning_rate": 2.0e-5,
        "pooler_learning_rate": 1.0e-4,
        "proxy_learning_rate": 1.0e-2,
        "adamw_betas": (0.9, 0.999),
        "adamw_epsilon": 1.0e-8,
        "weight_decay": 1.0e-4,
        "warmup_updates": 10,
        "gradient_clip_norm": 1.0,
    }


@dataclass(frozen=True)
class QwenGeometryProtocol:
    """Scientific constants that must remain identical across paired arms."""

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_revision: str = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    arms: tuple[str, ...] = ("mean", "attention")
    seeds: tuple[int, ...] = (17, 29, 43)
    optimization_classes: tuple[int, ...] = tuple(range(49))
    clean_development_classes: tuple[int, ...] = tuple(range(49, 82))
    burned_diagnostic_classes: tuple[int, ...] = tuple(range(82, 98))
    image_size: int = 224
    embedding_dimensions: int = 4096
    logical_batch_size: int = 64
    classes_per_batch: int = 16
    images_per_class: int = 4
    epochs: int = 3
    steps_per_epoch: int = 61
    optimizer_updates: int = 183
    proxy_anchor_alpha: float = 32.0
    proxy_anchor_delta: float = 0.1
    tower_learning_rate: float = 2.0e-5
    pooler_learning_rate: float = 1.0e-4
    proxy_learning_rate: float = 1.0e-2
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-4
    warmup_updates: int = 10
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        registered = _registered_protocol_values()
        for field in fields(self):
            actual = getattr(self, field.name)
            expected = registered[field.name]
            if type(actual) is not type(expected) or actual != expected:
                name = field.name.replace("_", " ")
                raise ValueError(f"{name} differs from the registered protocol")


@dataclass(frozen=True)
class GeometryBatchPlan:
    """One deterministic epoch of class-balanced logical batches."""

    seed: int
    epoch: int
    batches: tuple[tuple[int, ...], ...]
    batch_digests: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class GeometryParameterRoleManifest:
    """Complete qualified parameter-name to scientific-role assignment."""

    roles: tuple[tuple[str, str], ...]


def _counter_digest(domain: str, *values: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"sfora-qwen-geometry-v1\0")
    digest.update(domain.encode("ascii"))
    for value in values:
        digest.update(struct.pack("<q", value))
    return digest.digest()


def derive_epoch_batches(
    class_members: Mapping[int, Sequence[int]], *, seed: int, epoch: int
) -> GeometryBatchPlan:
    """Derive the registered 61 class-balanced batches without arm input."""

    protocol = QwenGeometryProtocol()
    valid_header = (
        type(seed) is int
        and seed in protocol.seeds
        and type(epoch) is int
        and 0 <= epoch < protocol.epochs
        and type(class_members) is dict
        and set(class_members) == set(protocol.optimization_classes)
    )
    canonical: dict[int, tuple[int, ...]] = {}
    if valid_header:
        seen: set[int] = set()
        for label in protocol.optimization_classes:
            incoming = class_members[label]
            values = tuple(incoming)
            if (
                len(values) < protocol.images_per_class
                or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != len(values)
                or any(value in seen for value in values)
            ):
                valid_header = False
                break
            seen.update(values)
            canonical[label] = values
    if not valid_header:
        raise ValueError("optimization class members differ from the registered partition")

    batches: list[tuple[int, ...]] = []
    digests: list[str] = []
    for step in range(protocol.steps_per_epoch):
        ranked_labels = sorted(
            protocol.optimization_classes,
            key=lambda label: (_counter_digest("class", seed, epoch, step, label), label),
        )[: protocol.classes_per_batch]
        batch: list[int] = []
        for label in ranked_labels:
            selected = sorted(
                canonical[label],
                key=lambda index: (
                    _counter_digest("image", seed, epoch, step, label, index),
                    index,
                ),
            )[: protocol.images_per_class]
            batch.extend(selected)
        frozen_batch = tuple(batch)
        encoded = b"".join(struct.pack("<Q", index) for index in frozen_batch)
        batches.append(frozen_batch)
        digests.append(hashlib.sha256(encoded).hexdigest())
    frozen_digests = tuple(digests)
    return GeometryBatchPlan(
        seed=seed,
        epoch=epoch,
        batches=tuple(batches),
        batch_digests=frozen_digests,
        digest=hashlib.sha256("".join(frozen_digests).encode("ascii")).hexdigest(),
    )


def learning_rate_multiplier(update_index: int) -> float:
    """Return the registered ten-update warm-up and cosine multiplier."""

    protocol = QwenGeometryProtocol()
    if type(update_index) is not int or not 0 <= update_index < protocol.optimizer_updates:
        raise ValueError("update index is outside the registered schedule")
    if update_index < protocol.warmup_updates:
        return (update_index + 1) / protocol.warmup_updates
    decay_updates = protocol.optimizer_updates - protocol.warmup_updates - 1
    progress = (update_index - protocol.warmup_updates) / decay_updates
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _role_entries(
    *,
    tower: nn.Module,
    pooler: nn.Module,
    proxies: nn.Parameter,
    allow_frozen: bool = False,
) -> list[tuple[str, str, nn.Parameter]]:
    entries = [
        (f"tower.{name}", "tower", parameter)
        for name, parameter in tower.named_parameters(remove_duplicate=False)
    ]
    entries.extend(
        (f"pooler.{name}", "pooler", parameter)
        for name, parameter in pooler.named_parameters(remove_duplicate=False)
    )
    entries.append(("proxies", "proxies", proxies))
    has_forbidden_frozen = not allow_frozen and any(
        not parameter.requires_grad for _, _, parameter in entries
    )
    if not entries or has_forbidden_frozen:
        raise ValueError("every registered parameter must be trainable")
    if allow_frozen:
        entries = [entry for entry in entries if entry[2].requires_grad]
    if not entries:
        raise ValueError("registered trainable parameter set is empty")
    identities = [id(parameter) for _, _, parameter in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("parameter roles contain a duplicated parameter")
    return entries


def parameter_role_manifest(
    *, tower: nn.Module, pooler: nn.Module, proxies: nn.Parameter
) -> GeometryParameterRoleManifest:
    """Bind every trainable parameter to exactly one scientific role."""

    entries = _role_entries(tower=tower, pooler=pooler, proxies=proxies)
    return GeometryParameterRoleManifest(
        roles=tuple((name, role) for name, role, _ in entries)
    )


def optimizer_groups(
    *,
    tower: nn.Module,
    pooler: nn.Module,
    proxies: nn.Parameter,
    allow_frozen: bool = False,
) -> list[dict[str, Any]]:
    """Build disjoint AdamW groups with registered learning rates and decay rules."""

    protocol = QwenGeometryProtocol()
    learning_rates = {
        "tower": protocol.tower_learning_rate,
        "pooler": protocol.pooler_learning_rate,
        "proxies": protocol.proxy_learning_rate,
    }
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, role, parameter in _role_entries(
        tower=tower,
        pooler=pooler,
        proxies=proxies,
        allow_frozen=allow_frozen,
    ):
        decay = role != "proxies" and not name.endswith(".bias") and parameter.ndim > 1
        grouped.setdefault((role, decay), []).append(parameter)
    return [
        {
            "role": role,
            "decay": decay,
            "params": parameters,
            "lr": learning_rates[role],
            "weight_decay": protocol.weight_decay if decay else 0.0,
        }
        for (role, decay), parameters in grouped.items()
    ]


class MeanProjectionPooler(nn.Module):
    """Mean-pool patch tokens and project to the registered descriptor width."""

    def __init__(self, token_dimensions: int) -> None:
        super().__init__()
        if type(token_dimensions) is not int or token_dimensions <= 0:
            raise ValueError("token dimensions must be a positive integer")
        self.token_dimensions = token_dimensions
        self.output = nn.Linear(
            token_dimensions, QwenGeometryProtocol().embedding_dimensions, bias=False
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, None]:
        return self.output(tokens.mean(dim=1)), None


class SingleQueryAttentionPooler(nn.Module):
    """Pool patch tokens using one learned query and a learned key projection."""

    def __init__(self, token_dimensions: int) -> None:
        super().__init__()
        if type(token_dimensions) is not int or token_dimensions <= 0:
            raise ValueError("token dimensions must be a positive integer")
        self.token_dimensions = token_dimensions
        self.query = nn.Parameter(torch.empty(token_dimensions))
        self.key = nn.Linear(token_dimensions, token_dimensions, bias=False)
        self.output = nn.Linear(
            token_dimensions, QwenGeometryProtocol().embedding_dimensions, bias=False
        )
        nn.init.normal_(self.query, std=token_dimensions**-0.5)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        keys = self.key(tokens)
        logits = torch.einsum("d,bpd->bp", self.query, keys) / math.sqrt(
            self.token_dimensions
        )
        weights = logits.softmax(dim=-1)
        pooled = torch.einsum("bp,bpd->bd", weights, tokens)
        return self.output(pooled), weights


def build_geometry_pooler(
    arm: str, *, token_dimensions: int
) -> MeanProjectionPooler | SingleQueryAttentionPooler:
    """Construct exactly one of the two preregistered pooling arms."""

    if arm == "mean":
        return MeanProjectionPooler(token_dimensions)
    if arm == "attention":
        return SingleQueryAttentionPooler(token_dimensions)
    raise ValueError(f"{arm!r} is not a registered geometry arm")


def _role_seed(seed: int, role: str) -> int:
    raw = _counter_digest(f"initialize-{role}", seed)
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def _copy_normal_(parameter: Tensor, *, seed: int, std: float) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
    value.normal_(mean=0.0, std=std, generator=generator)
    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def initialize_geometry_pooler(
    pooler: MeanProjectionPooler | SingleQueryAttentionPooler, *, seed: int
) -> None:
    """Initialize each parameter role from an independent deterministic stream."""

    if type(seed) is not int or seed not in QwenGeometryProtocol().seeds:
        raise ValueError("seed differs from the registered geometry seeds")
    _copy_normal_(
        pooler.output.weight,
        seed=_role_seed(seed, "output"),
        std=math.sqrt(2.0 / pooler.output.out_features),
    )
    if isinstance(pooler, SingleQueryAttentionPooler):
        _copy_normal_(
            pooler.query,
            seed=_role_seed(seed, "query"),
            std=pooler.token_dimensions**-0.5,
        )
        _copy_normal_(
            pooler.key.weight,
            seed=_role_seed(seed, "key"),
            std=math.sqrt(2.0 / pooler.key.out_features),
        )


def initialize_geometry_proxies(proxies: nn.Parameter, *, seed: int) -> None:
    """Initialize the 49 registered proxies from their own deterministic stream."""

    protocol = QwenGeometryProtocol()
    if type(seed) is not int or seed not in protocol.seeds:
        raise ValueError("seed differs from the registered geometry seeds")
    if tuple(proxies.shape) != (
        len(protocol.optimization_classes),
        protocol.embedding_dimensions,
    ):
        raise ValueError("proxy shape differs from the registered geometry protocol")
    _copy_normal_(
        proxies,
        seed=_role_seed(seed, "proxies"),
        std=math.sqrt(2.0 / protocol.embedding_dimensions),
    )


def pool_patch_tokens(
    pooler: MeanProjectionPooler | SingleQueryAttentionPooler, tokens: Tensor
) -> tuple[Tensor, Tensor | None]:
    """Validate patch tokens and return a finite unit-normalized descriptor."""

    if (
        type(tokens) is not Tensor
        or tokens.ndim != 3
        or tokens.shape[0] == 0
        or tokens.shape[1] == 0
        or tokens.shape[2] != pooler.token_dimensions
        or not torch.isfinite(tokens).all().item()
    ):
        raise ValueError("patch tokens must be finite nonempty [batch, patch, dimension]")

    descriptor, weights = pooler(tokens.float())
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.isfinite(descriptor).all().item() or not torch.all(norms > 0).item():
        raise ValueError("pooled descriptors must be finite and nonzero")
    return F.normalize(descriptor, dim=-1), weights
