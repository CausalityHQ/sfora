#!/usr/bin/env python3
"""Run the local-only FVCG-Direct Phase-A combined-step falsifier."""

from __future__ import annotations

import hashlib
import math
import resource
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from sfora.fvcg_direct import (  # noqa: E402
    FvcgStepAuthority,
    FvcgStepEvidence,
    select_stratum_pair,
)
from sfora.pfml import pfml_potential_loss  # noqa: E402


class CombinedStepAdapter(Protocol):
    pooler: torch.nn.Module

    def vision_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...

    def language_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...

    def vision_pool(self, microbatch: object) -> torch.Tensor: ...

    def direct_collapsed_verdict_backward(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def parameter_state_sha256(parameters: Iterable[torch.Tensor]) -> str:
    """Hash ordered tensor state with exact dtype, shape, and little-endian bytes."""

    digest = hashlib.sha256()
    seen = 0
    for ordinal, parameter in enumerate(parameters):
        if not isinstance(parameter, torch.Tensor):
            raise ValueError("FVCG parameter state differs")
        value = parameter.detach().to(device="cpu").contiguous()
        header = f"{ordinal}:{value.dtype}:{tuple(value.shape)}".encode()
        digest.update(_frame(header))
        digest.update(_frame(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")))
        seen += 1
    if seen == 0:
        raise ValueError("FVCG parameter state is empty")
    return digest.hexdigest()


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    groups = state.get("param_groups")
    states = state.get("state")
    if type(groups) is not list or type(states) is not dict:
        raise ValueError("FVCG optimizer state differs")
    for group in groups:
        if type(group) is not dict:
            raise ValueError("FVCG optimizer group differs")
        for key in sorted(group):
            value = group[key]
            if key == "params":
                digest.update(_frame(f"params:{len(value)}".encode()))
            else:
                digest.update(_frame(f"{key}:{value!r}".encode()))
    for parameter_ordinal in sorted(states):
        row = states[parameter_ordinal]
        if type(row) is not dict:
            raise ValueError("FVCG optimizer tensor state differs")
        digest.update(_frame(str(parameter_ordinal).encode()))
        for key in sorted(row):
            value = row[key]
            digest.update(_frame(str(key).encode()))
            if isinstance(value, torch.Tensor):
                digest.update(bytes.fromhex(parameter_state_sha256((value,))))
            else:
                digest.update(_frame(repr(value).encode()))
    return digest.hexdigest()


def _gradient_vector(parameters: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    values = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            values.append(torch.zeros_like(parameter, dtype=torch.float32).flatten())
        else:
            values.append(gradient.detach().float().flatten())
    if not values:
        raise ValueError("FVCG gradient field is empty")
    return torch.cat(values)


def _nonzero_finite_count(parameters: Iterable[torch.nn.Parameter]) -> tuple[int, bool]:
    count = 0
    finite = True
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = finite and bool(torch.isfinite(gradient).all())
        count += int(bool(torch.count_nonzero(gradient)))
    return count, finite


def _peak_cuda_reserved_bytes() -> int:
    return max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1


def _peak_rss_bytes() -> int:
    return max(1, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_combined_step(
    adapter: CombinedStepAdapter,
    *,
    authority: FvcgStepAuthority,
    optimizer: torch.optim.Optimizer,
    proxies: torch.nn.Parameter,
    proxy_labels: torch.Tensor,
    dml_microbatch: object,
    dml_labels: torch.Tensor,
    semantic_pair: object,
    correct_completion_ids: tuple[int, ...],
    incorrect_completion_ids: tuple[int, ...],
    ordinal: int,
    direct_vjp_errors: tuple[float, float],
    memory_psi_full_avg10_ppm: int,
) -> FvcgStepEvidence:
    """Accumulate PFML and weighted FVCG, clip once, and take one step."""

    authority.validated()
    if (
        type(ordinal) is not int
        or ordinal < 0
        or type(direct_vjp_errors) is not tuple
        or len(direct_vjp_errors) != 2
        or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in direct_vjp_errors
        )
        or type(memory_psi_full_avg10_ppm) is not int
        or memory_psi_full_avg10_ppm < 0
    ):
        raise ValueError("FVCG combined-step authority differs")
    vision = adapter.vision_parameters()
    pooler = tuple(adapter.pooler.parameters())
    language = adapter.language_parameters()
    if not vision or not pooler or proxies.ndim != 2:
        raise ValueError("FVCG trainable role authority differs")
    if any(parameter.grad is not None for parameter in language):
        raise ValueError("FVCG language gradient differs")

    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _synchronize()
    started = perf_counter_ns()
    embeddings = adapter.vision_pool(dml_microbatch)
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.ndim != 2
        or embeddings.shape[0] != dml_labels.shape[0]
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("FVCG DML embedding authority differs")
    dml_loss = pfml_potential_loss(
        embeddings,
        dml_labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.2,
        alpha=2.0,
        torch_module=torch,
    )
    dml_loss.backward()
    dml_gradients = tuple(
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in vision
    )
    dml_vector = _gradient_vector(vision)
    dml_norm = float(torch.linalg.vector_norm(dml_vector))
    if not math.isfinite(dml_norm) or dml_norm <= 0.0:
        raise ValueError("FVCG DML gradient differs")

    _synchronize()
    semantic_started = perf_counter_ns()
    semantic = adapter.direct_collapsed_verdict_backward(
        semantic_pair,
        correct_completion_ids=correct_completion_ids,
        incorrect_completion_ids=incorrect_completion_ids,
    )
    _synchronize()
    semantic_elapsed_ns = max(1, perf_counter_ns() - semantic_started)
    semantic_parts = []
    for parameter, dml_gradient in zip(vision, dml_gradients, strict=True):
        if parameter.grad is None:
            raise ValueError("FVCG semantic gradient is absent")
        semantic_gradient = parameter.grad.detach() - dml_gradient
        semantic_parts.append(semantic_gradient.float().flatten())
        parameter.grad.copy_(dml_gradient + authority.semantic_weight * semantic_gradient)
    semantic_vector = torch.cat(semantic_parts)
    semantic_norm = float(torch.linalg.vector_norm(semantic_vector))
    combined_vector = _gradient_vector(vision)
    combined_norm = float(torch.linalg.vector_norm(combined_vector))
    if not math.isfinite(semantic_norm) or semantic_norm <= 0.0 or combined_norm <= 0.0:
        raise ValueError("FVCG semantic gradient differs")
    cosine = float(torch.dot(dml_vector, combined_vector) / (dml_norm * combined_norm))
    cosine_distance_ppm = max(0, round((1.0 - max(-1.0, min(1.0, cosine))) * 1_000_000))

    vision_count, vision_finite = _nonzero_finite_count(vision)
    pooler_count, pooler_finite = _nonzero_finite_count(pooler)
    proxy_count, proxy_finite = _nonzero_finite_count((proxies,))
    language_count, language_finite = _nonzero_finite_count(language)
    if language_count != 0:
        raise ValueError("FVCG language gradient differs")
    all_trainable = (*vision, *pooler, proxies)
    preclip_norm = float(torch.linalg.vector_norm(_gradient_vector(all_trainable)))
    gradient_sha256 = hashlib.sha256(
        _gradient_vector(all_trainable).detach().cpu().numpy().astype("<f4").tobytes()
    ).hexdigest()
    torch.nn.utils.clip_grad_norm_(all_trainable, authority.gradient_clip_norm)
    optimizer.step()
    optimizer_state_sha256 = _optimizer_state_sha256(optimizer)
    updated_state_sha256 = hashlib.sha256(
        bytes.fromhex(parameter_state_sha256(vision))
        + bytes.fromhex(parameter_state_sha256(pooler))
        + bytes.fromhex(parameter_state_sha256((proxies,)))
    ).hexdigest()
    language_state_sha256 = parameter_state_sha256(language)
    _synchronize()
    combined_elapsed_ns = max(1, perf_counter_ns() - started)

    evidence = FvcgStepEvidence(
        ordinal=ordinal,
        selected_pair=select_stratum_pair(
            tuple(range(8)), seed_sha256=authority.selection_seed_sha256, step=ordinal
        ),
        correct_score=float(semantic.branch_scores[0]),
        incorrect_score=float(semantic.branch_scores[1]),
        correct_probability_ppm=round(float(semantic.correct_probability) * 1_000_000),
        coefficient_ppm=round(float(semantic.coefficient) * 1_000_000),
        loss=float(semantic.loss),
        generated_tokens=int(semantic.generated_tokens),
        vision_nonzero_gradient_parameters=vision_count,
        pooler_nonzero_gradient_parameters=pooler_count,
        proxy_nonzero_gradient_parameters=proxy_count,
        language_gradient_parameters=language_count,
        gradients_finite=(
            vision_finite and pooler_finite and proxy_finite and language_finite
        ),
        dml_gradient_norm=dml_norm,
        semantic_gradient_norm=semantic_norm,
        combined_gradient_cosine_distance_ppm=cosine_distance_ppm,
        clip_activated=preclip_norm > authority.gradient_clip_norm,
        combined_elapsed_ns=combined_elapsed_ns,
        semantic_elapsed_ns=semantic_elapsed_ns,
        peak_cuda_reserved_bytes=_peak_cuda_reserved_bytes(),
        peak_rss_bytes=_peak_rss_bytes(),
        memory_psi_full_avg10_ppm=memory_psi_full_avg10_ppm,
        direct_vjp_max_abs_error=direct_vjp_errors[0],
        direct_vjp_max_rel_error=direct_vjp_errors[1],
        gradient_sha256=gradient_sha256,
        updated_state_sha256=updated_state_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        language_state_sha256=language_state_sha256,
    )
    return evidence.validated(authority)
