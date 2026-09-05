#!/usr/bin/env python3
"""Run the local-only paired Qwen geometry-control experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from sfora.data import load_image_retrieval_examples, materialize_image
from sfora.qwen_geometry_control import (
    QwenGeometryProtocol,
    build_geometry_pooler,
    derive_epoch_batches,
    initialize_geometry_pooler,
    initialize_geometry_proxies,
    learning_rate_multiplier,
    optimizer_groups,
    pool_patch_tokens,
)
from sfora.token_set_proxy_anchor import proxy_anchor_loss


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}".encode("ascii")
    payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return _frame(header) + _frame(payload)


def state_sha256(parameters: Iterable[Tensor]) -> str:
    """Hash ordered tensor state with dtype and shape framing."""

    digest = hashlib.sha256()
    count = 0
    for ordinal, parameter in enumerate(parameters):
        digest.update(_frame(str(ordinal).encode("ascii")))
        digest.update(_tensor_bytes(parameter))
        count += 1
    if count == 0:
        raise ValueError("parameter state is empty")
    return digest.hexdigest()


def _hash_value(digest: object, value: object) -> None:
    update = digest.update  # type: ignore[attr-defined]
    if isinstance(value, Tensor):
        update(b"tensor")
        update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            update(_frame(repr(key).encode("utf-8")))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        update(b"sequence")
        for item in value:
            _hash_value(digest, item)
    else:
        update(_frame(repr(value).encode("utf-8")))


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, optimizer.state_dict())
    return digest.hexdigest()


@dataclass(frozen=True)
class GeometryStepEvidence:
    """Auditable evidence from exactly one successful logical-batch update."""

    update_index: int
    loss: float
    scores: Tensor
    score_gradients: Tensor
    parameter_gradients: tuple[Tensor, ...]
    gradient_norm: float
    maximum_score_disagreement: float
    learning_rate_multiplier: float
    updated_state_sha256: str
    optimizer_state_sha256: str


class QwenVisionGeometryModel(nn.Module):
    """Expose only Qwen vision features followed by one registered pooling arm."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: object,
        token_dimensions: int,
        arm: str,
    ) -> None:
        super().__init__()
        try:
            visual = model.model.visual  # type: ignore[attr-defined]
            language = model.model.language_model  # type: ignore[attr-defined]
            language_head = model.lm_head  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ValueError("Qwen vision-only model structure differs") from error
        if not isinstance(visual, nn.Module):
            raise ValueError("Qwen visual tower differs")
        if not callable(getattr(processor, "image_processor", None)):
            raise ValueError("Qwen image processor differs")
        self.visual = visual
        self.pooler = build_geometry_pooler(arm, token_dimensions=token_dimensions)
        self.__dict__["_qwen_model"] = model
        self.__dict__["_image_processor"] = processor.image_processor
        for parameter in (*language.parameters(), *language_head.parameters()):
            parameter.requires_grad_(False)
        for parameter in self.visual.parameters():
            parameter.requires_grad_(True)
        deepstack = getattr(self.visual, "deepstack_merger_list", None)
        if isinstance(deepstack, nn.Module):
            for parameter in deepstack.parameters():
                parameter.requires_grad_(False)

    def forward(self, images: Sequence[object]) -> Tensor:
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or not images:
            raise ValueError("Qwen image batch differs")
        raw = self._image_processor(images=list(images), return_tensors="pt")
        if not isinstance(raw, Mapping) or set(raw) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("Qwen image processor output differs")
        pixel_values = raw["pixel_values"]
        image_grid_thw = raw["image_grid_thw"]
        if not isinstance(pixel_values, Tensor) or not isinstance(image_grid_thw, Tensor):
            raise ValueError("Qwen image processor tensors differ")
        try:
            device = next(self.visual.parameters()).device
        except StopIteration as error:
            raise ValueError("Qwen visual tower has no parameters") from error
        output = self._qwen_model.get_image_features(
            pixel_values.to(device), image_grid_thw.to(device)
        )
        features = getattr(output, "pooler_output", None)
        if (
            type(features) not in {tuple, list}
            or len(features) != len(images)
            or any(not isinstance(feature, Tensor) or feature.ndim != 2 for feature in features)
            or any(feature.shape != features[0].shape for feature in features)
        ):
            raise ValueError("Qwen patch feature authority differs")
        descriptors, _ = pool_patch_tokens(self.pooler, torch.stack(tuple(features)))
        return descriptors


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def _write_new(path: Path, raw: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("Qwen geometry output already exists")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _rgb_224(image: object) -> np.ndarray:
    converted = cast(Image.Image, materialize_image(image)).convert("RGB")
    resized = converted.resize((224, 224), resample=Image.Resampling.BICUBIC)
    value = np.asarray(resized, dtype=np.uint8)
    if value.shape != (224, 224, 3):
        raise ValueError("Cars image differs from RGB 224 authority")
    return value.copy(order="C")


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only real-model smoke boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("smoke", nargs="?")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=QwenGeometryProtocol().arms, required=True)
    parser.add_argument("--seed", type=int, choices=QwenGeometryProtocol().seeds, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--execute-smoke", action="store_true", required=True)
    values = parser.parse_args(argv)
    if values.smoke != "smoke":
        parser.error("only the smoke phase is currently available")
    if len(values.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in values.source_commit
    ):
        parser.error("source commit must be 40 lowercase hex")
    if (
        values.microbatch_size < 1
        or QwenGeometryProtocol().logical_batch_size % values.microbatch_size != 0
    ):
        parser.error("microbatch size must divide the logical batch")
    invalid_output = (
        values.output.exists()
        or not values.output.parent.is_dir()
        or values.output.parent.is_symlink()
    )
    if invalid_output:
        parser.error("output must be absent beneath an existing regular directory")
    return values


def _load_real_geometry_model(args: argparse.Namespace) -> QwenVisionGeometryModel:
    from scripts.diagnose_saga_gb10_feasibility import (
        LoadedAuthority,
        TransformersFactory,
        load_qwen_adapter,
    )
    from sfora.saga_feasibility import load_fixture_authority, load_snapshot_authority

    snapshot = load_snapshot_authority(
        root=args.model_root, manifest_path=args.snapshot_manifest
    )
    fixture = load_fixture_authority(args.fixture)
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture), factory=TransformersFactory()
    )
    model = QwenVisionGeometryModel(
        model=adapter._model,
        processor=adapter._processor,
        token_dimensions=adapter.pooler_token_dim,
        arm=args.arm,
    )
    device = next(model.visual.parameters()).device
    model.pooler.to(device)
    initialize_geometry_pooler(model.pooler, seed=args.seed)
    return model


def run_smoke(args: argparse.Namespace) -> bytes:
    """Run exactly three authenticated real-data updates and return a receipt."""

    protocol = QwenGeometryProtocol()
    _configure_determinism()
    torch.manual_seed(args.seed)
    examples = tuple(
        row
        for row in load_image_retrieval_examples(dataset_name="cars", split="train")
        if row.label in protocol.optimization_classes
    )
    members = {
        label: tuple(index for index, row in enumerate(examples) if row.label == label)
        for label in protocol.optimization_classes
    }
    plan = derive_epoch_batches(members, seed=args.seed, epoch=0)
    model = _load_real_geometry_model(args)
    device = next(model.visual.parameters()).device
    proxies = nn.Parameter(
        torch.empty(
            len(protocol.optimization_classes),
            protocol.embedding_dimensions,
            device=device,
            dtype=torch.float32,
        )
    )
    initialize_geometry_proxies(proxies, seed=args.seed)
    groups = optimizer_groups(
        tower=model.visual,
        pooler=model.pooler,
        proxies=proxies,
        allow_frozen=True,
    )
    for group in groups:
        group["base_lr"] = group["lr"]
        group["schedule_update"] = 0
    optimizer = torch.optim.AdamW(
        groups,
        betas=protocol.adamw_betas,
        eps=protocol.adamw_epsilon,
    )
    model.train()
    started = perf_counter_ns()
    updates: list[dict[str, object]] = []
    for update_index, batch in enumerate(plan.batches[:3]):
        images = tuple(_rgb_224(examples[index].image) for index in batch)
        labels = torch.tensor(
            [examples[index].label for index in batch], dtype=torch.int64, device=device
        )
        update_started = perf_counter_ns()
        evidence = replayed_proxy_anchor_step(
            model=model,
            proxies=proxies,
            inputs=images,
            labels=labels,
            optimizer=optimizer,
            microbatch_size=args.microbatch_size,
            update_index=update_index,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        updates.append(
            {
                "batch_sha256": plan.batch_digests[update_index],
                "elapsed_ns": perf_counter_ns() - update_started,
                "gradient_norm": evidence.gradient_norm,
                "loss": evidence.loss,
                "maximum_score_disagreement": evidence.maximum_score_disagreement,
                "optimizer_state_sha256": evidence.optimizer_state_sha256,
                "state_sha256": evidence.updated_state_sha256,
                "update": update_index,
            }
        )
    payload = {
        "arm": args.arm,
        "claim_eligible": False,
        "elapsed_ns": perf_counter_ns() - started,
        "language_forward_calls": 0,
        "microbatch_size": args.microbatch_size,
        "optimizer_updates": 3,
        "peak_cuda_bytes": (
            max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1
        ),
        "peak_rss_bytes": max(1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "protocol_batch_plan_sha256": plan.digest,
        "schema": "sfora-qwen-geometry-smoke-v1",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "updates": updates,
    }
    return _canonical_bytes(payload)


def _validate_replay_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("logical replay refuses training batch normalization")
        if isinstance(module, nn.Dropout) and module.training and module.p > 0.0:
            raise ValueError("logical replay refuses active dropout")


def replayed_proxy_anchor_step(
    *,
    model: nn.Module,
    proxies: nn.Parameter,
    inputs: Tensor | Sequence[object],
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    microbatch_size: int,
    update_index: int,
) -> GeometryStepEvidence:
    """Apply Proxy Anchor once while replaying a logical batch in bounded slices."""

    protocol = QwenGeometryProtocol()
    tensor_inputs = isinstance(inputs, Tensor)
    sequence_inputs = isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes))
    batch_size = (
        int(inputs.shape[0])
        if tensor_inputs and inputs.ndim >= 1
        else len(inputs)
        if sequence_inputs
        else 0
    )
    parameters = (*filter(lambda parameter: parameter.requires_grad, model.parameters()), proxies)
    if (
        not (tensor_inputs or sequence_inputs)
        or (tensor_inputs and (not inputs.is_floating_point() or inputs.ndim < 2))
        or batch_size < 1
        or labels.shape != (batch_size,)
        or labels.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("logical batch inputs and labels differ")
    if tensor_inputs and not torch.isfinite(inputs).all().item():
        raise ValueError("logical batch inputs must be finite")
    if (
        type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > batch_size
        or batch_size % microbatch_size != 0
    ):
        raise ValueError("microbatch size must be a positive logical-batch divisor")
    if type(update_index) is not int or not 0 <= update_index < protocol.optimizer_updates:
        raise ValueError("update index differs from the registered schedule")
    if any(group.get("schedule_update") != update_index for group in optimizer.param_groups):
        raise ValueError("optimizer schedule position differs")
    if any("base_lr" not in group for group in optimizer.param_groups):
        raise ValueError("optimizer base learning-rate authority is absent")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("every logical replay parameter must be trainable")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("logical replay parameters are duplicated")
    if not torch.isfinite(proxies).all().item() or bool(
        (torch.linalg.vector_norm(proxies, dim=-1) <= 0).any()
    ):
        raise ValueError("class proxies must be finite and nonzero")
    if bool((labels < 0).any()) or bool((labels >= proxies.shape[0]).any()):
        raise ValueError("logical batch labels exceed the proxy authority")
    _validate_replay_model(model)

    optimizer.zero_grad(set_to_none=True)
    score_chunks: list[Tensor] = []
    with torch.no_grad():
        normalized_proxies = F.normalize(proxies, dim=-1)
        for start in range(0, batch_size, microbatch_size):
            descriptors = model(inputs[start : start + microbatch_size])
            if descriptors.ndim != 2 or descriptors.shape[1] != proxies.shape[1]:
                raise ValueError("descriptor and proxy shapes differ")
            score_chunks.append(descriptors @ normalized_proxies.T)
    scores = torch.cat(score_chunks)
    if not torch.isfinite(scores).all().item():
        raise ValueError("logical batch scores must be finite")
    score_leaf = scores.detach().requires_grad_(True)
    loss = proxy_anchor_loss(
        score_leaf,
        labels,
        alpha=protocol.proxy_anchor_alpha,
        delta=protocol.proxy_anchor_delta,
    )
    (score_gradients,) = torch.autograd.grad(loss, score_leaf)
    if not torch.isfinite(loss).item() or not torch.isfinite(score_gradients).all().item():
        raise ValueError("Proxy Anchor loss and cotangent must be finite")

    maximum_disagreement = 0.0
    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size
        replay_scores = model(inputs[start:stop]) @ F.normalize(proxies, dim=-1).T
        disagreement = float((replay_scores.detach() - scores[start:stop]).abs().max())
        maximum_disagreement = max(maximum_disagreement, disagreement)
        if not math.isfinite(disagreement) or disagreement > 1.0e-10:
            raise RuntimeError("logical replay score disagreement exceeds tolerance")
        torch.autograd.backward(replay_scores, score_gradients[start:stop])

    gradients: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("every logical replay parameter must receive a finite gradient")
        gradients.append(parameter.grad.detach().clone())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, protocol.gradient_clip_norm, error_if_nonfinite=True
    )
    if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0.0:
        raise RuntimeError("logical replay gradient norm must be finite and positive")

    multiplier = learning_rate_multiplier(update_index)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier
    optimizer.step()
    for group in optimizer.param_groups:
        group["schedule_update"] = update_index + 1

    return GeometryStepEvidence(
        update_index=update_index,
        loss=float(loss.detach()),
        scores=scores.detach(),
        score_gradients=score_gradients.detach(),
        parameter_gradients=tuple(gradients),
        gradient_norm=float(gradient_norm),
        maximum_score_disagreement=maximum_disagreement,
        learning_rate_multiplier=multiplier,
        updated_state_sha256=state_sha256(parameters),
        optimizer_state_sha256=_optimizer_state_sha256(optimizer),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = run_smoke(args)
    _write_new(args.output, raw)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
