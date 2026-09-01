#!/usr/bin/env python3
"""Run the local, quality-blind SAGA GB10 feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import resource
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as torch_functional

from sfora.asgcv_protocol import AsgcvCompletionGroup
from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import (
    FeasibilityEvidence,
    FeasibilityOutcome,
    FixtureAuthority,
    ObjectAuthority,
    PhaseMeasurement,
    ResourceEnvelope,
    ScientificEvidence,
    SnapshotAuthority,
    canonical_feasibility_result_bytes,
    generated_fixture_image_bytes,
    load_fixture_authority,
    load_snapshot_authority,
)


class FeasibilityFailure(Exception):
    """Expected, canonically classifiable diagnostic failure."""

    def __init__(self, outcome: FeasibilityOutcome, clause: str) -> None:
        super().__init__(clause)
        self.outcome = outcome
        self.clause = clause


class AttentionUnavailable(Exception):
    """The sealed backend cannot expose exact layer attention."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Result identities and controller envelope unavailable from model state."""

    source_commit: str
    controller_commit: str
    binary_sha256: str
    environment_sha256: str
    host: str
    model_object: ObjectAuthority
    fixture_object: ObjectAuthority
    envelope: ResourceEnvelope


@dataclass(frozen=True, slots=True)
class LoadedAuthority:
    """Cross-bound snapshot and synthetic fixture authorities."""

    snapshot: SnapshotAuthority
    fixture: FixtureAuthority
    result_identity: RunIdentity | None = None


class ModelFactory(Protocol):
    """Offline-only factory boundary for the concrete Transformers adapter."""

    def load_model(self, root: Path, **kwargs: object) -> object: ...

    def load_processor(self, root: Path, **kwargs: object) -> object: ...


class TransformersFactory:
    """Lazy local-only Transformers construction boundary."""

    def load_model(self, root: Path, **kwargs: object) -> object:
        from transformers import Qwen3VLForConditionalGeneration

        values = dict(kwargs)
        if values.get("dtype") == "bfloat16":
            values["dtype"] = torch.bfloat16
        return Qwen3VLForConditionalGeneration.from_pretrained(root, **values)

    def load_processor(self, root: Path, **kwargs: object) -> object:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(root, **kwargs)


class SagaModelAdapter(Protocol):
    """Backend-independent scientific operations required by the diagnostic."""

    def prepare_pair(self, fixture: FixtureAuthority) -> object: ...

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]: ...

    def replay(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        advantages: tuple[float, ...],
        *,
        output_attentions: bool,
    ) -> ReplayOutput: ...

    def assert_gradient_roles(self) -> GradientEvidence: ...

    def clear_graphs(self) -> None: ...

    def attention_observation(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        *,
        layer: int,
    ) -> AttentionOutput: ...

    def prepare_microbatch(self, fixture: FixtureAuthority) -> object: ...

    def vision_pool(self, microbatch: object) -> torch.Tensor: ...

    def validate_structure(self, authority: LoadedAuthority) -> None: ...


@dataclass(frozen=True, slots=True)
class SealedRollouts:
    """Content-addressed completion IDs from one exact rollout group."""

    completion_ids: tuple[tuple[int, ...], ...]
    token_counts: tuple[int, ...]
    completion_sha256: tuple[str, ...]
    elapsed_ns: int

    @property
    def group_size(self) -> int:
        return len(self.completion_ids)


@dataclass(frozen=True, slots=True)
class ReplayOutput:
    """Backend replay scalar and token count before gradient validation."""

    loss: float
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class PatchGradientTarget:
    """Stopped merged vision tokens and their exact eight-branch replay gradient."""

    replay: ReplayOutput
    patch_tokens: torch.Tensor
    exact_gradient: torch.Tensor
    replay_branch_count: int
    attention_kl: float
    teacher_gradient_parameters: int


@dataclass(frozen=True, slots=True)
class GradientEvidence:
    """Exact trainable/frozen gradient-role evidence."""

    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int
    finite: bool
    gradient_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayEvidence:
    """Validated replay measurement."""

    elapsed_ns: int
    loss: float
    generated_tokens: int
    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int
    gradient_sha256: str


@dataclass(frozen=True, slots=True)
class AttentionOutput:
    """Exact layer attention and patch tokens returned by the adapter."""

    teacher_maps: torch.Tensor
    patch_tokens: torch.Tensor
    head_count: int


@dataclass(frozen=True, slots=True)
class AttentionEvidence:
    """Validated detached-teacher KL evidence."""

    layer: int
    head_count: int
    teacher_shape: tuple[int, int]
    patch_token_shape: tuple[int, int, int]
    elapsed_ns: int
    kl: float
    teacher_unit_mass: bool
    teacher_gradient_parameters: int
    pooler_nonzero_gradient_parameters: int


@dataclass(frozen=True, slots=True)
class DmlFloorEvidence:
    """Validated 64-image vision/pooler activation-floor evidence."""

    batch_size: int
    embedding_shape: tuple[int, int]
    elapsed_ns: int
    loss: float
    maximum_norm_delta_ppm: int
    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int


@dataclass(frozen=True, slots=True)
class PreparedPair:
    """Processor-authenticated two-image model inputs."""

    inputs: dict[str, torch.Tensor]
    input_length: int
    image_token_ranges: tuple[tuple[int, int], tuple[int, int]]
    attribute_token_span: tuple[int, int]
    patch_tokens_per_image: int


@dataclass(frozen=True, slots=True)
class PreparedMicrobatch:
    """Processor-authenticated 64-image vision inputs."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class SingleQueryPooler(torch.nn.Module):
    """One learned query attention pooler with a 4096-dimensional output."""

    def __init__(self, token_dim: int, embedding_dim: int = 4096) -> None:
        super().__init__()
        if type(token_dim) is not int or token_dim <= 0:
            raise ValueError("SAGA pooler token dimension differs")
        if type(embedding_dim) is not int or embedding_dim != 4096:
            raise ValueError("SAGA pooler embedding dimension differs")
        self.query = torch.nn.Parameter(torch.empty(token_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=token_dim**-0.5)
        self.key = torch.nn.Linear(token_dim, token_dim, bias=False)
        self.output = torch.nn.Linear(token_dim, embedding_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.query.numel():
            raise ValueError("SAGA pooler token shape differs")
        logits = torch.einsum("d,bpd->bp", self.query, self.key(tokens)) / math.sqrt(
            tokens.shape[-1]
        )
        weights = logits.softmax(dim=-1)
        pooled = torch.einsum("bp,bpd->bd", weights, tokens)
        return torch_functional.normalize(self.output(pooled), dim=-1), weights


def pooler_state_sha256(pooler: SingleQueryPooler) -> str:
    """Hash the exact initialized pooler tensors in stable name order."""

    if type(pooler) is not SingleQueryPooler:
        raise ValueError("SAGA pooler authority differs")
    digest = hashlib.sha256()
    for name, tensor in sorted(pooler.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(canonical_json_bytes({"shape": list(contiguous.shape)}))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _source_bound_pooler(
    *, token_dim: int, source_commit: str, model_revision: str
) -> SingleQueryPooler:
    seed_material = canonical_json_bytes(
        {
            "model_revision": model_revision,
            "role": "single-query-pooler-v1",
            "source_commit": source_commit,
        }
    )
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        return SingleQueryPooler(token_dim=token_dim)
    finally:
        torch.random.set_rng_state(cpu_rng_state)


class QwenSagaAdapter:
    """Validated Qwen model/processor pair with frozen gradient roles."""

    def __init__(
        self,
        model: object,
        processor: object,
        *,
        vision_parameters: tuple[object, ...],
        language_parameters: tuple[object, ...],
        token_dim: int,
        load_measurement: PhaseMeasurement,
        source_commit: str,
        model_revision: str,
    ) -> None:
        self._model = model
        self._processor = processor
        self._vision_parameters = vision_parameters
        self._language_parameters = language_parameters
        self.architecture = "Qwen3VLForConditionalGeneration"
        self.pooler_token_dim = token_dim
        self.load_measurement = load_measurement
        operational_device = next(
            (
                parameter.device
                for parameter in vision_parameters
                if isinstance(parameter, torch.Tensor)
            ),
            torch.device("cpu"),
        )
        self.pooler = _source_bound_pooler(
            token_dim=token_dim,
            source_commit=source_commit,
            model_revision=model_revision,
        ).to(operational_device)
        self.pooler_sha256 = pooler_state_sha256(self.pooler)

    def trainable_parameter_roles(self) -> tuple[str, ...]:
        return ("vision",)

    def frozen_parameter_roles(self) -> tuple[str, ...]:
        return ("language",)

    def validate_structure(self, authority: LoadedAuthority) -> None:
        if authority.snapshot.architecture != self.architecture:
            raise FeasibilityFailure(FeasibilityOutcome.BACKEND_INVALID, "model-architecture")
        if not self._vision_parameters or not self._language_parameters:
            raise FeasibilityFailure(FeasibilityOutcome.BACKEND_INVALID, "parameter-roles")

    @staticmethod
    def _images(fixture: FixtureAuthority, ordinals: tuple[int, ...]) -> list[np.ndarray]:
        return [
            np.frombuffer(
                generated_fixture_image_bytes(fixture.source_commit, ordinal),
                dtype=np.uint8,
            )
            .reshape(224, 224, 3)
            .copy()
            for ordinal in ordinals
        ]

    @staticmethod
    def _tensor_device(inputs: dict[str, torch.Tensor]) -> torch.device:
        return inputs["input_ids"].device

    def _operational_device(self) -> torch.device:
        for parameter in self._vision_parameters:
            if isinstance(parameter, torch.Tensor):
                return parameter.device
        return torch.device("cpu")

    def prepare_pair(self, fixture: FixtureAuthority) -> PreparedPair:
        images = self._images(fixture, fixture.pair_ordinals)
        return self.prepare_image_pair(
            tuple(images),
            fixture.prompt_utf8,
            fixture.attribute_token_span,
            fixture.patch_tokens_per_image,
        )

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> PreparedPair:
        """Prepare two authenticated RGB arrays without accepting split or label data."""

        if (
            type(images) is not tuple
            or len(images) != 2
            or any(
                type(image) is not np.ndarray
                or image.dtype != np.dtype(np.uint8)
                or image.ndim != 3
                or image.shape[-1] != 3
                or any(size <= 0 for size in image.shape)
                for image in images
            )
            or type(prompt_utf8) is not str
            or not prompt_utf8
            or type(attribute_token_span) is not tuple
            or len(attribute_token_span) != 2
            or any(type(value) is not int for value in attribute_token_span)
            or not 0 <= attribute_token_span[0] < attribute_token_span[1]
            or type(patch_tokens_per_image) is not int
            or patch_tokens_per_image <= 0
        ):
            raise ValueError("SAGA image-pair authority differs")
        copied_images = [image.copy(order="C") for image in images]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": copied_images[0]},
                    {"type": "image", "image": copied_images[1]},
                    {"type": "text", "text": prompt_utf8},
                ],
            }
        ]
        raw = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        expected = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("SAGA processor output authority differs")
        if any(type(value) is not torch.Tensor for value in raw.values()):
            raise ValueError("SAGA processor tensor authority differs")
        inputs = {name: value.to(self._operational_device()) for name, value in raw.items()}
        input_ids = inputs["input_ids"]
        token_types = inputs["mm_token_type_ids"]
        if input_ids.shape[0] != 1 or token_types.shape != input_ids.shape:
            raise ValueError("SAGA processor sequence authority differs")
        image_positions = torch.nonzero(token_types[0] == 1, as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise ValueError("SAGA image token authority differs")
        groups: list[tuple[int, int]] = []
        start = int(image_positions[0])
        previous = start
        for position_tensor in image_positions[1:]:
            position = int(position_tensor)
            if position != previous + 1:
                groups.append((start, previous + 1))
                start = position
            previous = position
        groups.append((start, previous + 1))
        if len(groups) != 2:
            raise ValueError("SAGA image token ranges differ")
        if any(end - start != patch_tokens_per_image for start, end in groups):
            raise ValueError("SAGA image patch span authority differs")
        return PreparedPair(
            inputs=inputs,
            input_length=input_ids.shape[1],
            image_token_ranges=(groups[0], groups[1]),
            attribute_token_span=attribute_token_span,
            patch_tokens_per_image=patch_tokens_per_image,
        )

    def prepare_microbatch(self, fixture: FixtureAuthority) -> PreparedMicrobatch:
        images = self._images(fixture, fixture.microbatch_ordinals)
        raw = self._processor.image_processor(images=images, return_tensors="pt")
        if type(raw) is not dict or set(raw) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("SAGA microbatch processor authority differs")
        pixel_values = raw["pixel_values"]
        image_grid_thw = raw["image_grid_thw"]
        if type(pixel_values) is not torch.Tensor or type(image_grid_thw) is not torch.Tensor:
            raise ValueError("SAGA microbatch tensor authority differs")
        if tuple(image_grid_thw.shape) != (64, 3):
            raise ValueError("SAGA microbatch grid authority differs")
        device = self._operational_device()
        return PreparedMicrobatch(
            pixel_values=pixel_values.to(device),
            image_grid_thw=image_grid_thw.to(device),
        )

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        if type(pair) is not PreparedPair or type(seed) is not int:
            raise ValueError("SAGA generation authority differs")
        device = self._tensor_device(pair.inputs)
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                generated = self._model.generate(
                    **pair.inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        if type(generated) is not torch.Tensor or generated.shape[0] != 1:
            raise ValueError("SAGA generated token authority differs")
        completion = generated[0, pair.input_length :].detach().cpu().tolist()
        if not completion:
            raise ValueError("SAGA generated completion is empty")
        return tuple(int(token) for token in completion)

    @staticmethod
    def _completed_inputs(
        pair: PreparedPair, completion: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        device = pair.inputs["input_ids"].device
        completion_tensor = torch.tensor([completion], dtype=torch.long, device=device)
        inputs = dict(pair.inputs)
        inputs["input_ids"] = torch.cat((pair.inputs["input_ids"], completion_tensor), dim=1)
        extension = torch.ones_like(completion_tensor)
        inputs["attention_mask"] = torch.cat((pair.inputs["attention_mask"], extension), dim=1)
        inputs["mm_token_type_ids"] = torch.cat(
            (pair.inputs["mm_token_type_ids"], torch.zeros_like(completion_tensor)),
            dim=1,
        )
        return inputs

    def replay(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        advantages: tuple[float, ...],
        *,
        output_attentions: bool,
    ) -> ReplayOutput:
        if type(pair) is not PreparedPair or len(completion_ids) != 8:
            raise ValueError("SAGA replay authority differs")
        losses: list[torch.Tensor] = []
        generated_tokens = 0
        for completion, advantage in zip(completion_ids, advantages, strict=True):
            inputs = self._completed_inputs(pair, completion)
            outputs = self._model.forward(
                **inputs,
                output_attentions=output_attentions,
                use_cache=False,
            )
            logits = outputs.logits
            start = pair.input_length - 1
            completion_logits = logits[:, start : start + len(completion), :].float()
            target = torch.tensor([completion], dtype=torch.long, device=completion_logits.device)
            token_log_probabilities = (
                torch_functional.log_softmax(completion_logits, dim=-1)
                .gather(-1, target.unsqueeze(-1))
                .squeeze(-1)
            )
            losses.append(-float(advantage) * token_log_probabilities.mean())
            generated_tokens += len(completion)
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("SAGA replay loss differs")
        loss.backward()
        return ReplayOutput(loss=float(loss.detach()), generated_tokens=generated_tokens)

    def replay_patch_gradient(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        advantages: tuple[float, ...],
        *,
        correct_rollouts: tuple[bool, ...],
        attribute_spans: tuple[tuple[int, int] | None, ...],
        attention_layer: int,
    ) -> PatchGradientTarget:
        """Capture combined GRPO and attention-KL gradient at the vision boundary."""

        if (
            type(pair) is not PreparedPair
            or len(completion_ids) != len(advantages)
            or type(correct_rollouts) is not tuple
            or len(correct_rollouts) != len(completion_ids)
            or any(type(value) is not bool for value in correct_rollouts)
            or type(attribute_spans) is not tuple
            or len(attribute_spans) != len(completion_ids)
            or type(attention_layer) is not int
            or attention_layer != 26
        ):
            raise ValueError("SAGA patch gradient replay authority differs")
        if not any(correct_rollouts):
            raise ValueError("SAGA correct rollout authority differs")
        if all(correct_rollouts):
            raise ValueError("SAGA reward variance authority differs")
        for correct, span in zip(correct_rollouts, attribute_spans, strict=True):
            if span is not None and (
                type(span) is not tuple
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1]
            ):
                raise ValueError("SAGA attribute token span differs")
            if correct and span is None:
                raise ValueError("SAGA correct rollout attribute span differs")
        model = getattr(self._model, "model", None)
        visual = getattr(model, "visual", None)
        merger = getattr(visual, "merger", None)
        if not isinstance(merger, torch.nn.Module):
            raise ValueError("SAGA vision merger authority differs")
        captured: list[torch.Tensor] = []

        def capture_output(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if type(output) is not torch.Tensor or output.ndim != 2:
                raise ValueError("SAGA merged patch token authority differs")
            output.retain_grad()
            captured.append(output)

        handle = merger.register_forward_hook(capture_output)
        try:
            losses: list[torch.Tensor] = []
            teacher_maps: list[torch.Tensor] = []
            generated_tokens = 0
            head_count: int | None = None
            for completion, advantage, correct, attribute_span in zip(
                completion_ids,
                advantages,
                correct_rollouts,
                attribute_spans,
                strict=True,
            ):
                inputs = self._completed_inputs(pair, completion)
                outputs = self._model.forward(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                )
                logits = outputs.logits
                start = pair.input_length - 1
                completion_logits = logits[:, start : start + len(completion), :].float()
                target = torch.tensor(
                    [completion],
                    dtype=torch.long,
                    device=completion_logits.device,
                )
                token_log_probabilities = (
                    torch_functional.log_softmax(completion_logits, dim=-1)
                    .gather(-1, target.unsqueeze(-1))
                    .squeeze(-1)
                )
                losses.append(-float(advantage) * token_log_probabilities.mean())
                generated_tokens += len(completion)
                if correct:
                    if attribute_span is None:
                        raise ValueError("SAGA correct rollout attribute span differs")
                    teacher_map, completion_head_count = self._completion_teacher_map(
                        pair,
                        outputs.attentions,
                        layer=attention_layer,
                        completion_length=len(completion),
                        attribute_span=attribute_span,
                    )
                    if head_count is None:
                        head_count = completion_head_count
                    elif head_count != completion_head_count:
                        raise ValueError("SAGA attention head authority differs")
                    teacher_maps.append(teacher_map)
            if len(captured) != len(completion_ids) or not captured:
                raise ValueError("SAGA patch gradient replay count differs")
            expected_rows = 2 * pair.patch_tokens_per_image
            shape = captured[0].shape
            if shape[0] != expected_rows or any(value.shape != shape for value in captured):
                raise ValueError("SAGA merged patch token shape differs")
            reference = captured[0].detach()
            if any(not torch.equal(value.detach(), reference) for value in captured[1:]):
                raise ValueError("SAGA repeated patch token values differ")
            target_shape = (2, pair.patch_tokens_per_image, shape[1])
            live_patch_tokens = captured[0].reshape(target_shape)
            teacher = torch.stack(teacher_maps).mean(dim=0).detach()
            _, pooler_weights = self.pooler(live_patch_tokens)
            if teacher.shape != pooler_weights.shape or head_count is None:
                raise ValueError("SAGA attention/feature alignment differs")
            epsilon = torch.finfo(pooler_weights.dtype).tiny
            attention_kl = (
                (
                    teacher
                    * (teacher.clamp_min(epsilon).log() - pooler_weights.clamp_min(epsilon).log())
                )
                .sum(dim=-1)
                .mean()
            )
            grpo_loss = torch.stack(losses).mean()
            combined_loss = grpo_loss + attention_kl
            if not bool(torch.isfinite(combined_loss)):
                raise ValueError("SAGA semantic patch loss differs")
            combined_loss.backward()
            gradients = [value.grad for value in captured]
            if any(value is None for value in gradients):
                raise ValueError("SAGA merged patch gradient is absent")
            exact_gradient = torch.stack(
                [value.detach().float() for value in gradients if value is not None]
            ).sum(dim=0)
            patch_tokens = reference.float()
            if not bool(torch.isfinite(patch_tokens).all()) or not bool(
                torch.isfinite(exact_gradient).all()
            ):
                raise ValueError("SAGA merged patch gradient is non-finite")
            return PatchGradientTarget(
                replay=ReplayOutput(
                    loss=float(grpo_loss.detach()),
                    generated_tokens=generated_tokens,
                ),
                patch_tokens=patch_tokens.reshape(target_shape).detach(),
                exact_gradient=exact_gradient.reshape(target_shape).detach(),
                replay_branch_count=len(captured),
                attention_kl=float(attention_kl.detach()),
                teacher_gradient_parameters=int(teacher.grad is not None),
            )
        finally:
            handle.remove()
            self.clear_graphs()

    @staticmethod
    def _completion_teacher_map(
        pair: PreparedPair,
        attentions: object,
        *,
        layer: int,
        completion_length: int,
        attribute_span: tuple[int, int],
    ) -> tuple[torch.Tensor, int]:
        if type(attentions) not in {tuple, list} or len(attentions) <= layer:
            raise AttentionUnavailable("layer-26-attention-unavailable")
        attention = attentions[layer]
        if type(attention) is not torch.Tensor or attention.ndim != 4:
            raise AttentionUnavailable("layer-26-attention-unavailable")
        span_start, span_end = attribute_span
        if type(completion_length) is not int or span_end > completion_length:
            raise ValueError("SAGA attribute token span differs")
        query_start = pair.input_length + span_start
        query_end = pair.input_length + span_end
        query_attention = attention[0, :, query_start:query_end, :].mean(dim=(0, 1))
        teacher_rows = []
        for start, end in pair.image_token_ranges:
            row = query_attention[start:end].float()
            if row.numel() == 0 or not bool(torch.isfinite(row).all()):
                raise ValueError("SAGA attention patch authority differs")
            row_sum = row.sum()
            if not bool(torch.isfinite(row_sum)) or float(row_sum) <= 0.0:
                raise ValueError("SAGA attention patch authority differs")
            teacher_rows.append(row / row_sum)
        return torch.stack(teacher_rows), attention.shape[1]

    def _image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        output = self._model.get_image_features(pixel_values, image_grid_thw)
        features = output.pooler_output
        if type(features) not in {tuple, list} or not features:
            raise ValueError("SAGA vision feature authority differs")
        if any(type(feature) is not torch.Tensor for feature in features):
            raise ValueError("SAGA vision feature authority differs")
        return tuple(features)

    def attention_observation(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        *,
        layer: int,
    ) -> AttentionOutput:
        if type(pair) is not PreparedPair or layer != 26:
            raise ValueError("SAGA attention authority differs")
        if len(completion_ids) != 8:
            raise ValueError("SAGA attention completion authority differs")
        completion_maps: list[torch.Tensor] = []
        head_count: int | None = None
        for completion in completion_ids:
            inputs = self._completed_inputs(pair, completion)
            outputs = self._model.forward(**inputs, output_attentions=True, use_cache=False)
            teacher_map, completion_head_count = self._completion_teacher_map(
                pair,
                outputs.attentions,
                layer=layer,
                completion_length=len(completion),
                attribute_span=pair.attribute_token_span,
            )
            if head_count is None:
                head_count = completion_head_count
            elif completion_head_count != head_count:
                raise ValueError("SAGA attention head authority differs")
            completion_maps.append(teacher_map.detach())
        teacher_maps = torch.stack(completion_maps).mean(dim=0)
        features = self._image_features(pair.inputs["pixel_values"], pair.inputs["image_grid_thw"])
        if len(features) != 2 or features[0].shape != features[1].shape:
            raise ValueError("SAGA pair vision feature authority differs")
        patch_tokens = torch.stack(features)
        if patch_tokens.shape[1] != pair.patch_tokens_per_image:
            raise ValueError("SAGA vision patch span authority differs")
        if teacher_maps.shape[:2] != patch_tokens.shape[:2]:
            raise ValueError("SAGA attention/feature alignment differs")
        return AttentionOutput(
            teacher_maps=teacher_maps,
            patch_tokens=patch_tokens,
            head_count=head_count or 0,
        )

    def vision_pool(self, microbatch: object) -> torch.Tensor:
        if type(microbatch) is not PreparedMicrobatch:
            raise ValueError("SAGA DML microbatch authority differs")
        features = self._image_features(microbatch.pixel_values, microbatch.image_grid_thw)
        if len(features) != 64 or any(feature.shape != features[0].shape for feature in features):
            raise ValueError("SAGA DML vision feature shape differs")
        embeddings, _ = self.pooler(torch.stack(features))
        return embeddings

    def assert_gradient_roles(self) -> GradientEvidence:
        vision_gradients = []
        for parameter in (*self._vision_parameters, *tuple(self.pooler.parameters())):
            gradient = getattr(parameter, "grad", None)
            if gradient is not None and bool(torch.count_nonzero(gradient)):
                vision_gradients.append(gradient)
        language_gradient_parameters = sum(
            int(getattr(parameter, "grad", None) is not None)
            for parameter in self._language_parameters
        )
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in vision_gradients)
        digest = hashlib.sha256()
        for gradient in vision_gradients[:4]:
            digest.update(gradient.detach().float().flatten()[:256].cpu().numpy().tobytes())
        return GradientEvidence(
            vision_nonzero_gradient_parameters=len(vision_gradients),
            language_gradient_parameters=language_gradient_parameters,
            finite=finite,
            gradient_sha256=digest.hexdigest(),
        )

    def clear_graphs(self) -> None:
        for parameter in (*self._vision_parameters, *self._language_parameters):
            parameter.grad = None
        self.pooler.zero_grad(set_to_none=True)


def _model_attribute(model: object, name: str) -> object:
    try:
        return getattr(model, name)
    except AttributeError as error:
        raise ValueError("SAGA model authority differs") from error


def load_qwen_adapter(authority: LoadedAuthority, *, factory: ModelFactory) -> QwenSagaAdapter:
    """Load one registered local model and freeze exact parameter roles."""

    if type(authority) is not LoadedAuthority:
        raise ValueError("SAGA loaded authority differs")
    snapshot = authority.snapshot
    fixture = authority.fixture
    if fixture.model_revision != snapshot.model_revision:
        raise ValueError("SAGA model authority differs")
    load_started = _begin_runtime_measurement()
    model = factory.load_model(
        snapshot.root,
        local_files_only=True,
        trust_remote_code=False,
        dtype="bfloat16",
        attn_implementation=snapshot.attention_backend,
    )
    processor = factory.load_processor(
        snapshot.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model, "config"):
        config = model.config
        architectures = getattr(config, "architectures", None)
        architecture = (
            architectures[0]
            if type(architectures) is list and len(architectures) == 1
            else type(model).__name__
        )
        dtype = str(getattr(model, "dtype", "")).removeprefix("torch.")
        device = str(getattr(model, "device", ""))
        attention_backend = getattr(config, "_attn_implementation", None)
        layer_count = getattr(config.text_config, "num_hidden_layers", None)
        token_dim = getattr(config.vision_config, "out_hidden_size", None)
        visual = model.model.visual
        language_model = model.model.language_model
        vision_parameters = tuple(visual.parameters())
        language_parameters = tuple(language_model.parameters()) + tuple(model.lm_head.parameters())
        processor_keys = tuple(getattr(processor, "model_input_names", ()))
    else:
        architecture = _model_attribute(model, "architecture")
        dtype = _model_attribute(model, "dtype")
        device = _model_attribute(model, "device")
        attention_backend = _model_attribute(model, "attention_backend")
        layer_count = _model_attribute(model, "layer_count")
        token_dim = 16
        roles = _model_attribute(model, "parameters_by_role")
        if type(roles) is not dict or set(roles) != {"vision", "language"}:
            raise ValueError("SAGA model authority differs")
        vision_parameters = tuple(roles["vision"])
        language_parameters = tuple(roles["language"])
        processor_keys = tuple(_model_attribute(processor, "output_keys"))
    if (
        architecture != snapshot.architecture
        or dtype != snapshot.dtype
        or device != "cuda"
        or attention_backend != snapshot.attention_backend
    ):
        raise ValueError("SAGA model authority differs")
    if type(layer_count) is not int or layer_count <= fixture.attention_layer:
        raise ValueError("SAGA model authority differs")
    if type(token_dim) is not int or token_dim <= 0:
        raise ValueError("SAGA model authority differs")
    expected_processor_keys = (
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "mm_token_type_ids",
        "pixel_values",
    )
    if tuple(sorted(processor_keys)) != expected_processor_keys:
        raise ValueError("SAGA model authority differs")
    if not vision_parameters or not language_parameters:
        raise ValueError("SAGA model authority differs")
    for parameter in vision_parameters:
        parameter.requires_grad = True
    for parameter in language_parameters:
        parameter.requires_grad = False
    load_measurement = _end_runtime_measurement("load", load_started)
    return QwenSagaAdapter(
        model,
        processor,
        vision_parameters=vision_parameters,
        language_parameters=language_parameters,
        token_dim=token_dim,
        load_measurement=load_measurement,
        source_commit=fixture.source_commit,
        model_revision=fixture.model_revision,
    )


def group_normalized_advantages(rewards: tuple[int, ...]) -> tuple[float, ...]:
    """Compute the exact population-normalized synthetic group advantages."""

    if (
        type(rewards) is not tuple
        or len(rewards) != 8
        or any(type(reward) is not int or reward not in {0, 1} for reward in rewards)
    ):
        raise ValueError("SAGA synthetic reward authority differs")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance <= 0.0:
        raise ValueError("SAGA synthetic rewards have zero group variance")
    scale = math.sqrt(variance)
    return tuple((reward - mean) / scale for reward in rewards)


def capture_asgcv_patch_gradient(
    adapter: QwenSagaAdapter,
    pair: PreparedPair,
    group: AsgcvCompletionGroup,
    *,
    attention_layer: int,
) -> PatchGradientTarget:
    """Capture one exact gradient from a sealed eligible completion group."""

    if (
        type(adapter) is not QwenSagaAdapter
        or type(pair) is not PreparedPair
        or type(group) is not AsgcvCompletionGroup
    ):
        raise ValueError("ASG-CV patch gradient context differs")
    group.validated()
    if group.nonzero_reward_variance is not True:
        raise ValueError("ASG-CV patch gradient eligibility differs")
    return adapter.replay_patch_gradient(
        pair,
        group.completion_ids,
        group_normalized_advantages(group.rewards),
        correct_rollouts=group.correct_rollouts,
        attribute_spans=group.attribute_spans,
        attention_layer=attention_layer,
    )


def _completion_digest(token_ids: tuple[int, ...]) -> str:
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("SAGA completion token authority differs")
    return hashlib.sha256(canonical_json_bytes({"token_ids": list(token_ids)})).hexdigest()


def run_rollout_phase(adapter: SagaModelAdapter, fixture: FixtureAuthority) -> SealedRollouts:
    """Generate and seal one exact eight-completion rollout group."""

    if fixture.group_size != 8 or fixture.generation_seeds != tuple(range(8)):
        raise ValueError("SAGA rollout fixture authority differs")
    pair = adapter.prepare_pair(fixture)
    started = perf_counter_ns()
    completions = tuple(
        adapter.generate(
            pair,
            seed,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=1024,
        )
        for seed in fixture.generation_seeds
    )
    elapsed_ns = max(1, perf_counter_ns() - started)
    if len(completions) != 8 or any(type(tokens) is not tuple for tokens in completions):
        raise ValueError("SAGA rollout completion evidence differs")
    digests = tuple(_completion_digest(tokens) for tokens in completions)
    if len(set(digests)) != 8:
        raise ValueError("SAGA rollout generators are not distinct")
    return SealedRollouts(
        completion_ids=completions,
        token_counts=tuple(len(tokens) for tokens in completions),
        completion_sha256=digests,
        elapsed_ns=elapsed_ns,
    )


def _validate_gradient_evidence(gradient: GradientEvidence) -> None:
    if type(gradient) is not GradientEvidence:
        raise ValueError("SAGA gradient evidence differs")
    if (
        type(gradient.vision_nonzero_gradient_parameters) is not int
        or gradient.vision_nonzero_gradient_parameters <= 0
    ):
        raise ValueError("SAGA vision gradient evidence differs")
    if (
        type(gradient.language_gradient_parameters) is not int
        or gradient.language_gradient_parameters != 0
    ):
        raise ValueError("SAGA language gradient evidence differs")
    if gradient.finite is not True:
        raise ValueError("SAGA finite gradient evidence differs")
    if (
        type(gradient.gradient_sha256) is not str
        or len(gradient.gradient_sha256) != 64
        or any(character not in "0123456789abcdef" for character in gradient.gradient_sha256)
    ):
        raise ValueError("SAGA gradient digest differs")


def run_replay_phase(
    adapter: SagaModelAdapter,
    fixture: FixtureAuthority,
    rollouts: SealedRollouts,
) -> ReplayEvidence:
    """Replay sealed completions and prove exact gradient-role separation."""

    if type(rollouts) is not SealedRollouts or rollouts.group_size != fixture.group_size:
        raise ValueError("SAGA replay rollout authority differs")
    pair = adapter.prepare_pair(fixture)
    advantages = group_normalized_advantages(fixture.synthetic_rewards)
    started = perf_counter_ns()
    try:
        output = adapter.replay(
            pair,
            rollouts.completion_ids,
            advantages,
            output_attentions=False,
        )
        elapsed_ns = max(1, perf_counter_ns() - started)
        if (
            type(output) is not ReplayOutput
            or type(output.loss) is not float
            or not math.isfinite(output.loss)
            or type(output.generated_tokens) is not int
            or output.generated_tokens != sum(rollouts.token_counts)
        ):
            raise ValueError("SAGA replay output evidence differs")
        gradient = adapter.assert_gradient_roles()
        _validate_gradient_evidence(gradient)
        return ReplayEvidence(
            elapsed_ns=elapsed_ns,
            loss=output.loss,
            generated_tokens=output.generated_tokens,
            vision_nonzero_gradient_parameters=(gradient.vision_nonzero_gradient_parameters),
            language_gradient_parameters=gradient.language_gradient_parameters,
            gradient_sha256=gradient.gradient_sha256,
        )
    finally:
        adapter.clear_graphs()


def _nonzero_finite_gradient_count(parameters: object) -> int:
    count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("SAGA finite pooler gradient evidence differs")
        if bool(torch.count_nonzero(gradient)):
            count += 1
    return count


def run_attention_phase(
    adapter: SagaModelAdapter,
    pooler: SingleQueryPooler,
    fixture: FixtureAuthority,
    rollouts: SealedRollouts,
) -> AttentionEvidence:
    """Measure exact layer-26 detached-teacher attention/pooler KL."""

    if fixture.attention_layer != 26 or rollouts.group_size != fixture.group_size:
        raise ValueError("SAGA attention authority differs")
    pair = adapter.prepare_pair(fixture)
    started = perf_counter_ns()
    try:
        output = adapter.attention_observation(
            pair,
            rollouts.completion_ids,
            layer=fixture.attention_layer,
        )
        if type(output) is not AttentionOutput:
            raise ValueError("SAGA attention output differs")
        teacher = output.teacher_maps
        tokens = output.patch_tokens
        if (
            teacher.ndim != 2
            or tokens.ndim != 3
            or teacher.shape != tokens.shape[:2]
            or type(output.head_count) is not int
            or output.head_count <= 0
        ):
            raise ValueError("SAGA teacher attention shape differs")
        if (
            not bool(torch.isfinite(teacher).all())
            or bool((teacher < 0).any())
            or not torch.allclose(
                teacher.sum(dim=-1),
                torch.ones(teacher.shape[0], device=teacher.device),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise ValueError("SAGA teacher attention differs")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("SAGA patch token evidence differs")
        if teacher.requires_grad:
            teacher.retain_grad()
        if tokens.requires_grad:
            tokens.retain_grad()
        detached_teacher = teacher.detach()
        detached_tokens = tokens.detach()
        _, pooler_weights = pooler(detached_tokens)
        epsilon = torch.finfo(pooler_weights.dtype).tiny
        kl = (
            (
                detached_teacher
                * (
                    detached_teacher.clamp_min(epsilon).log()
                    - pooler_weights.clamp_min(epsilon).log()
                )
            )
            .sum(dim=-1)
            .mean()
        )
        if not bool(torch.isfinite(kl)):
            raise ValueError("SAGA attention KL differs")
        kl.backward()
        pooler_gradient_count = _nonzero_finite_gradient_count(pooler.parameters())
        if pooler_gradient_count <= 0:
            raise ValueError("SAGA pooler gradient evidence differs")
        elapsed_ns = max(1, perf_counter_ns() - started)
        return AttentionEvidence(
            layer=26,
            head_count=output.head_count,
            teacher_shape=(teacher.shape[0], teacher.shape[1]),
            patch_token_shape=(tokens.shape[0], tokens.shape[1], tokens.shape[2]),
            elapsed_ns=elapsed_ns,
            kl=float(kl.detach()),
            teacher_unit_mass=True,
            teacher_gradient_parameters=int(teacher.grad is not None)
            + int(tokens.grad is not None),
            pooler_nonzero_gradient_parameters=pooler_gradient_count,
        )
    finally:
        adapter.clear_graphs()
        pooler.zero_grad(set_to_none=True)


def fixture_pairwise_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Backend-independent pairwise activation-floor loss."""

    distances = torch.cdist(embeddings.float(), embeddings.float()).square()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive = distances[same & ~eye]
    negative = distances[~same]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("SAGA DML pseudo-label evidence differs")
    return positive.mean() + torch_functional.relu(1.0 - negative).mean()


def run_dml_floor_phase(adapter: SagaModelAdapter, fixture: FixtureAuthority) -> DmlFloorEvidence:
    """Measure the 64-image, 4096-dimensional activation and gradient floor."""

    if fixture.image_count != 64:
        raise ValueError("SAGA DML fixture authority differs")
    started = perf_counter_ns()
    try:
        embeddings = adapter.vision_pool(adapter.prepare_microbatch(fixture))
        if type(embeddings) is not torch.Tensor or tuple(embeddings.shape) != (64, 4096):
            raise ValueError("SAGA DML embedding shape differs")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("SAGA DML embeddings are not finite")
        norm_delta = (embeddings.float().norm(dim=-1) - 1.0).abs()
        maximum_norm_delta_ppm = int(math.ceil(float(norm_delta.max().detach()) * 1_000_000))
        if maximum_norm_delta_ppm > 10:
            raise ValueError("SAGA DML embedding norms differ")
        labels = torch.tensor([ordinal % 2 for ordinal in range(64)], device=embeddings.device)
        loss = fixture_pairwise_loss(embeddings, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("SAGA DML loss differs")
        loss.backward()
        gradient = adapter.assert_gradient_roles()
        _validate_gradient_evidence(gradient)
        elapsed_ns = max(1, perf_counter_ns() - started)
        return DmlFloorEvidence(
            batch_size=64,
            embedding_shape=(64, 4096),
            elapsed_ns=elapsed_ns,
            loss=float(loss.detach()),
            maximum_norm_delta_ppm=maximum_norm_delta_ppm,
            vision_nonzero_gradient_parameters=(gradient.vision_nonzero_gradient_parameters),
            language_gradient_parameters=gradient.language_gradient_parameters,
        )
    finally:
        adapter.clear_graphs()


@dataclass(frozen=True, slots=True)
class _ScientificRun:
    rollout: SealedRollouts
    replay: ReplayEvidence
    attention: AttentionEvidence
    dml: DmlFloorEvidence


def _process_peak_rss_bytes() -> int:
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max(1, int(peak_kib) * 1024)


def _begin_runtime_measurement() -> int:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return perf_counter_ns()


def _end_runtime_measurement(name: str, started_ns: int) -> PhaseMeasurement:
    peak_cuda_reserved_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved())
    return PhaseMeasurement(
        name=name,
        completed=True,
        elapsed_ns=max(1, perf_counter_ns() - started_ns),
        peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
        peak_rss_bytes=_process_peak_rss_bytes(),
    )


def _measure_runtime[MeasuredValue](
    name: str, operation: Callable[[], MeasuredValue]
) -> tuple[MeasuredValue, PhaseMeasurement]:
    started_ns = _begin_runtime_measurement()
    value = operation()
    return value, _end_runtime_measurement(name, started_ns)


def _combine_measurements(name: str, *measurements: PhaseMeasurement) -> PhaseMeasurement:
    if not measurements or any(
        measurement.name != name or not measurement.completed for measurement in measurements
    ):
        raise ValueError("SAGA runtime measurement differs")
    return PhaseMeasurement(
        name=name,
        completed=True,
        elapsed_ns=sum(measurement.elapsed_ns for measurement in measurements),
        peak_cuda_reserved_bytes=max(
            measurement.peak_cuda_reserved_bytes for measurement in measurements
        ),
        peak_rss_bytes=max(measurement.peak_rss_bytes for measurement in measurements),
    )


def _phase_measurement(name: str, elapsed_ns: int = 0) -> PhaseMeasurement:
    return PhaseMeasurement(
        name=name,
        completed=elapsed_ns > 0,
        elapsed_ns=elapsed_ns,
        peak_cuda_reserved_bytes=0,
        peak_rss_bytes=0,
    )


def _run_scientific_phases(
    authority: LoadedAuthority,
    adapter: SagaModelAdapter,
    pooler: SingleQueryPooler,
    measurements: dict[str, PhaseMeasurement] | None,
    progress: Callable[[str], None] | None = None,
) -> _ScientificRun:
    rollout, rollout_measurement = _measure_runtime(
        "rollout", lambda: run_rollout_phase(adapter, authority.fixture)
    )
    if measurements is not None:
        measurements["rollout"] = rollout_measurement
        if progress is not None:
            progress("rollout")
    replay, replay_measurement = _measure_runtime(
        "replay", lambda: run_replay_phase(adapter, authority.fixture, rollout)
    )
    if measurements is not None:
        measurements["replay"] = replay_measurement
        if progress is not None:
            progress("replay")
    attention, attention_measurement = _measure_runtime(
        "attention",
        lambda: run_attention_phase(adapter, pooler, authority.fixture, rollout),
    )
    if measurements is not None:
        measurements["attention"] = attention_measurement
        if progress is not None:
            progress("attention")
    dml, dml_measurement = _measure_runtime(
        "dml", lambda: run_dml_floor_phase(adapter, authority.fixture)
    )
    if measurements is not None:
        measurements["dml"] = dml_measurement
        if progress is not None:
            progress("dml")
    return _ScientificRun(
        rollout=rollout,
        replay=replay,
        attention=attention,
        dml=dml,
    )


def _repeatability_signature(run: _ScientificRun) -> tuple[object, ...]:
    return (
        run.rollout.token_counts,
        run.rollout.completion_sha256,
        run.replay.loss,
        run.replay.generated_tokens,
        run.replay.vision_nonzero_gradient_parameters,
        run.replay.language_gradient_parameters,
        run.replay.gradient_sha256,
        run.attention.layer,
        run.attention.head_count,
        run.attention.teacher_shape,
        run.attention.patch_token_shape,
        run.attention.kl,
        run.attention.teacher_unit_mass,
        run.attention.teacher_gradient_parameters,
        run.attention.pooler_nonzero_gradient_parameters,
        run.dml.embedding_shape,
        run.dml.loss,
        run.dml.maximum_norm_delta_ppm,
        run.dml.vision_nonzero_gradient_parameters,
        run.dml.language_gradient_parameters,
    )


def _f64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _scientific_evidence(run: _ScientificRun, *, pooler_sha256: str) -> ScientificEvidence:
    return ScientificEvidence(
        pooler_sha256=pooler_sha256,
        rollout_group_size=run.rollout.group_size,
        rollout_token_counts=run.rollout.token_counts,
        rollout_completion_sha256=run.rollout.completion_sha256,
        replay_loss_f64_bits=_f64_bits(run.replay.loss),
        replay_generated_tokens=run.replay.generated_tokens,
        replay_vision_nonzero_gradient_parameters=(run.replay.vision_nonzero_gradient_parameters),
        replay_language_gradient_parameters=run.replay.language_gradient_parameters,
        replay_gradient_sha256=run.replay.gradient_sha256,
        attention_layer=run.attention.layer,
        attention_head_count=run.attention.head_count,
        attention_teacher_shape=run.attention.teacher_shape,
        attention_patch_token_shape=run.attention.patch_token_shape,
        attention_kl_f64_bits=_f64_bits(run.attention.kl),
        attention_teacher_unit_mass=run.attention.teacher_unit_mass,
        attention_teacher_gradient_parameters=(run.attention.teacher_gradient_parameters),
        attention_pooler_nonzero_gradient_parameters=(
            run.attention.pooler_nonzero_gradient_parameters
        ),
        dml_batch_size=run.dml.batch_size,
        dml_embedding_shape=run.dml.embedding_shape,
        dml_loss_f64_bits=_f64_bits(run.dml.loss),
        dml_maximum_norm_delta_ppm=run.dml.maximum_norm_delta_ppm,
        dml_vision_nonzero_gradient_parameters=(run.dml.vision_nonzero_gradient_parameters),
        dml_language_gradient_parameters=run.dml.language_gradient_parameters,
    ).validated()


def _failure_flags(outcome: FeasibilityOutcome) -> dict[str, bool]:
    flags = {
        "authority_valid": True,
        "backend_valid": True,
        "deterministic": True,
        "memory_within_envelope": True,
        "attention_available": True,
        "time_within_envelope": True,
    }
    field = {
        FeasibilityOutcome.AUTHORITY_INVALID: "authority_valid",
        FeasibilityOutcome.BACKEND_INVALID: "backend_valid",
        FeasibilityOutcome.DETERMINISM_FAIL: "deterministic",
        FeasibilityOutcome.MEMORY_FAIL: "memory_within_envelope",
        FeasibilityOutcome.ATTENTION_UNAVAILABLE: "attention_available",
        FeasibilityOutcome.TIME_BUDGET_FAIL: "time_within_envelope",
    }.get(outcome)
    if field is not None:
        flags[field] = False
    return flags


def _canonical_preflight_failure(identity: RunIdentity, outcome: FeasibilityOutcome) -> bytes:
    measurements = {
        name: _phase_measurement(name) for name in ("load", "rollout", "replay", "attention", "dml")
    }
    return canonical_feasibility_result_bytes(
        FeasibilityEvidence(
            source_commit=identity.source_commit,
            controller_commit=identity.controller_commit,
            binary_sha256=identity.binary_sha256,
            environment_sha256=identity.environment_sha256,
            host=identity.host,
            model=identity.model_object,
            fixture=identity.fixture_object,
            envelope=identity.envelope,
            load=measurements["load"],
            rollout=measurements["rollout"],
            replay=measurements["replay"],
            attention=measurements["attention"],
            dml=measurements["dml"],
            dataset_reads=0,
            label_reads=0,
            evaluation_reads=0,
            optimizer_steps=0,
            **_failure_flags(outcome),
        )
    )


def _validate_run_identity(authority: LoadedAuthority) -> RunIdentity:
    identity = authority.result_identity
    if type(identity) is not RunIdentity:
        raise ValueError("SAGA result identity differs")
    fixture = authority.fixture
    if (
        identity.source_commit != fixture.source_commit
        or identity.controller_commit != fixture.controller_commit
        or identity.binary_sha256 != fixture.binary_sha256
        or identity.environment_sha256 != fixture.environment_sha256
        or identity.host != fixture.host
        or authority.snapshot.model_revision != fixture.model_revision
    ):
        raise ValueError("SAGA result identity differs")
    identity.model_object.validated()
    identity.fixture_object.validated()
    identity.envelope.to_mapping()
    return identity


def run_feasibility(
    authority: LoadedAuthority,
    adapter: SagaModelAdapter,
    *,
    progress: Callable[[str], None] | None = None,
) -> bytes:
    """Run one complete repeatability diagnostic and emit canonical evidence."""

    identity = _validate_run_identity(authority)
    measurements = {
        name: _phase_measurement(name) for name in ("load", "rollout", "replay", "attention", "dml")
    }
    outcome = FeasibilityOutcome.FITS
    pooler_sha256: str | None = None
    scientific: ScientificEvidence | None = None
    try:
        _, validation_measurement = _measure_runtime(
            "load", lambda: adapter.validate_structure(authority)
        )
        load_measurement = getattr(adapter, "load_measurement", None)
        measurements["load"] = (
            _combine_measurements("load", load_measurement, validation_measurement)
            if type(load_measurement) is PhaseMeasurement
            else validation_measurement
        )
        if progress is not None:
            progress("load")
        token_dim = getattr(adapter, "pooler_token_dim", 16)
        pooler = getattr(adapter, "pooler", None)
        if type(pooler) is not SingleQueryPooler:
            pooler = SingleQueryPooler(token_dim=token_dim)
        pooler_sha256 = pooler_state_sha256(pooler)
        first = _run_scientific_phases(authority, adapter, pooler, measurements, progress)
        scientific = _scientific_evidence(first, pooler_sha256=pooler_sha256)
        second = _run_scientific_phases(authority, adapter, pooler, None)
        if _repeatability_signature(first) != _repeatability_signature(second):
            raise FeasibilityFailure(FeasibilityOutcome.DETERMINISM_FAIL, "repeatability")
    except FeasibilityFailure as failure:
        outcome = failure.outcome
    except AttentionUnavailable:
        outcome = FeasibilityOutcome.ATTENTION_UNAVAILABLE
    except torch.OutOfMemoryError:
        outcome = FeasibilityOutcome.MEMORY_FAIL

    flags = _failure_flags(outcome)
    evidence = FeasibilityEvidence(
        source_commit=identity.source_commit,
        controller_commit=identity.controller_commit,
        binary_sha256=identity.binary_sha256,
        environment_sha256=identity.environment_sha256,
        host=identity.host,
        model=identity.model_object,
        fixture=identity.fixture_object,
        envelope=identity.envelope,
        load=measurements["load"],
        rollout=measurements["rollout"],
        replay=measurements["replay"],
        attention=measurements["attention"],
        dml=measurements["dml"],
        dataset_reads=0,
        label_reads=0,
        evaluation_reads=0,
        optimizer_steps=0,
        pooler_sha256=pooler_sha256,
        scientific=scientific,
        **flags,
    )
    return canonical_feasibility_result_bytes(evidence)


def _reject_duplicate_options(argv: list[str]) -> None:
    options = [token for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate SAGA feasibility option")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse only local scientific capabilities and fail closed otherwise."""

    values = list(argv) if argv is not None else None
    if values is not None:
        _reject_duplicate_options(values)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--result-output", required=True, type=Path)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--execute-feasibility", required=True, action="store_true")
    parsed = parser.parse_args(values)
    for name in ("source_commit", "controller_commit"):
        value = getattr(parsed, name)
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name.replace('_', ' ')} must be 40 lowercase hex")
    for name in ("binary_sha256", "environment_sha256"):
        value = getattr(parsed, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name.replace('_', ' ')} must be 64 lowercase hex")
    if not parsed.host:
        parser.error("host must be nonempty")
    if not parsed.model_root.is_dir():
        parser.error("model root must be an existing directory")
    for name in ("snapshot_manifest", "fixture"):
        if not getattr(parsed, name).is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing file")
    if parsed.result_output.exists():
        parser.error("result output must not already exist")
    if parsed.progress_output is not None and parsed.progress_output.exists():
        parser.error("progress output must not already exist")
    return parsed


def _path_authority(path: Path, *, role: str) -> ObjectAuthority:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SAGA {role} path authority differs")
    payload = path.read_bytes()
    return ObjectAuthority(
        role=role,
        relative_path=path.name,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    ).validated()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.resolve(strict=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_progress(path: Path, phase: str) -> None:
    payload = canonical_json_bytes(
        {
            "schema": "sfora-saga-gb10-feasibility-progress-v1",
            "completed_phase": phase,
        }
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs, execute once, and publish canonical evidence."""

    args = parse_args(argv)
    snapshot = load_snapshot_authority(root=args.model_root, manifest_path=args.snapshot_manifest)
    fixture = load_fixture_authority(args.fixture)
    identity = RunIdentity(
        source_commit=args.source_commit,
        controller_commit=args.controller_commit,
        binary_sha256=args.binary_sha256,
        environment_sha256=args.environment_sha256,
        host=args.host,
        model_object=_path_authority(args.snapshot_manifest, role="model-snapshot-manifest"),
        fixture_object=_path_authority(args.fixture, role="synthetic-fixture"),
        envelope=ResourceEnvelope(
            cuda_reserved_limit_bytes=103_079_215_104,
            rss_limit_bytes=118_111_600_640,
            wall_limit_ns=7_200_000_000_000,
            progress_limit_ns=300_000_000_000,
        ),
    )
    authority = LoadedAuthority(
        snapshot=snapshot,
        fixture=fixture,
        result_identity=identity,
    )
    progress = (
        (lambda phase: _write_progress(args.progress_output, phase))
        if args.progress_output is not None
        else None
    )
    try:
        adapter = load_qwen_adapter(authority, factory=TransformersFactory())
    except torch.OutOfMemoryError:
        raw = _canonical_preflight_failure(identity, FeasibilityOutcome.MEMORY_FAIL)
    except ValueError:
        raw = _canonical_preflight_failure(identity, FeasibilityOutcome.BACKEND_INVALID)
    else:
        raw = run_feasibility(authority, adapter, progress=progress)
    _write_new(args.result_output, raw)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
