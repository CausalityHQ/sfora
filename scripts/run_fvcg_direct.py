#!/usr/bin/env python3
"""Run the local-only FVCG-Direct Phase-A combined-step falsifier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import resource
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.diagnose_saga_gb10_feasibility import (  # noqa: E402
    LoadedAuthority,
    TransformersFactory,
    load_qwen_adapter,
)
from scripts.prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from scripts.run_asgcv_p32 import load_p32_local_authority  # noqa: E402
from sfora.fvcg_direct import (  # noqa: E402
    FvcgPhaseAResult,
    FvcgStepAuthority,
    FvcgStepEvidence,
    canonical_fvcg_phase_a_result_bytes,
    select_stratum_pair,
    validate_fvcg_phase_a_result_bytes,
)
from sfora.pfml import pfml_potential_loss  # noqa: E402
from sfora.saga_feasibility import (  # noqa: E402
    load_fixture_authority,
    load_snapshot_authority,
)

FVCG_PFML_ALPHA = 3.0
FVCG_ADAMW_LR = 1.0e-3
FVCG_ADAMW_WEIGHT_DECAY = 1.0e-4


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


class Fp32MasterAdamW:
    """AdamW over FP32 master weights with explicit model-gradient transfer."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
    ) -> None:
        self.model_parameters = tuple(parameters)
        if not self.model_parameters:
            raise ValueError("FVCG optimizer parameters are empty")
        self.master_parameters = tuple(
            torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=False)
            for parameter in self.model_parameters
        )
        self.optimizer = torch.optim.AdamW(
            self.master_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.model_parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for model, master in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            master.grad = (
                None
                if model.grad is None
                else model.grad.detach().to(device=master.device, dtype=torch.float32)
            )
        self.optimizer.step()
        with torch.no_grad():
            for model, master in zip(
                self.model_parameters, self.master_parameters, strict=True
            ):
                model.copy_(master.to(device=model.device, dtype=model.dtype))

    def state_dict(self) -> dict[str, object]:
        return self.optimizer.state_dict()


@dataclass(frozen=True, slots=True)
class PhaseAContext:
    """One restored combined-step context with no mutable state shared across runs."""

    adapter: CombinedStepAdapter
    optimizer: torch.optim.Optimizer
    proxies: torch.nn.Parameter
    proxy_labels: torch.Tensor
    dml_microbatch: object
    dml_labels: torch.Tensor
    semantic_pair: object
    correct_completion_ids: tuple[int, ...]
    incorrect_completion_ids: tuple[int, ...]
    direct_vjp_errors: tuple[float, float]
    memory_psi_full_avg10_ppm: int
    direct_vjp_reference: tuple[torch.Tensor | None, ...] | None = None


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


def _memory_psi_full_avg10_ppm(path: Path = Path("/proc/pressure/memory")) -> int:
    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise ValueError("FVCG memory PSI authority is unavailable") from error
    for row in rows:
        fields = row.split()
        if fields and fields[0] == "full":
            values = dict(field.split("=", 1) for field in fields[1:])
            try:
                avg10 = float(values["avg10"])
            except (KeyError, ValueError) as error:
                raise ValueError("FVCG memory PSI authority differs") from error
            if not math.isfinite(avg10) or avg10 < 0.0:
                raise ValueError("FVCG memory PSI authority differs")
            return round(avg10 * 10_000)
    raise ValueError("FVCG memory PSI full row is absent")


def _direct_vjp_errors(
    actual_gradients: tuple[torch.Tensor, ...],
    reference_gradients: tuple[torch.Tensor | None, ...],
    *,
    absolute_floor: float,
) -> tuple[float, float]:
    """Return max-absolute and field-L2-relative BF16 replay errors."""

    if (
        len(actual_gradients) != len(reference_gradients)
        or type(absolute_floor) is not float
        or not math.isfinite(absolute_floor)
        or absolute_floor <= 0.0
    ):
        raise ValueError("FVCG direct VJP parameter authority differs")
    maximum_absolute = 0.0
    delta_squared = 0.0
    reference_squared = 0.0
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        if expected is None:
            if bool(torch.count_nonzero(actual)):
                raise ValueError("FVCG direct VJP sparsity differs")
            continue
        reference = expected.to(device=actual.device, dtype=torch.float32)
        delta = actual.float() - reference
        maximum_absolute = max(maximum_absolute, float(delta.abs().max()))
        delta_squared += float(torch.sum(delta * delta))
        reference_squared += float(torch.sum(reference * reference))
    relative_l2 = math.sqrt(delta_squared) / max(
        math.sqrt(reference_squared), absolute_floor
    )
    return maximum_absolute, relative_l2


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
    direct_vjp_reference: tuple[torch.Tensor | None, ...] | None = None,
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

    initial_vision_state_sha256 = parameter_state_sha256(vision)
    initial_pooler_state_sha256 = parameter_state_sha256(pooler)
    initial_proxy_state_sha256 = parameter_state_sha256((proxies,))
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
        alpha=FVCG_PFML_ALPHA,
        torch_module=torch,
    )
    dml_loss.backward()
    dml_gradients = tuple(
        torch.zeros_like(parameter, dtype=torch.float32)
        if parameter.grad is None
        else parameter.grad.detach().float().clone()
        for parameter in vision
    )
    dml_vector = torch.cat(tuple(gradient.flatten() for gradient in dml_gradients))
    dml_norm = float(torch.linalg.vector_norm(dml_vector))
    if not math.isfinite(dml_norm) or dml_norm <= 0.0:
        raise ValueError("FVCG DML gradient differs")

    for parameter in vision:
        parameter.grad = None

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
    semantic_gradients = []
    for parameter, dml_gradient in zip(vision, dml_gradients, strict=True):
        if parameter.grad is None:
            raise ValueError("FVCG semantic gradient is absent")
        semantic_gradient = parameter.grad.detach().float().clone()
        semantic_gradients.append(semantic_gradient)
        semantic_parts.append(semantic_gradient.flatten())
        parameter.grad.copy_(
            (dml_gradient + authority.semantic_weight * semantic_gradient).to(
                dtype=parameter.dtype
            )
        )
    all_trainable = (*vision, *pooler, proxies)
    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(all_trainable, authority.gradient_clip_norm)
    )
    optimizer.step()
    _synchronize()
    combined_elapsed_ns = max(1, perf_counter_ns() - started)

    semantic_vector = torch.cat(semantic_parts)
    combined_vector = torch.cat(
        tuple(
            (dml_gradient + authority.semantic_weight * semantic_gradient).flatten()
            for dml_gradient, semantic_gradient in zip(
                dml_gradients, semantic_gradients, strict=True
            )
        )
    )
    semantic_norm = float(torch.linalg.vector_norm(semantic_vector))
    combined_norm = float(torch.linalg.vector_norm(combined_vector))
    if not math.isfinite(semantic_norm) or semantic_norm <= 0.0 or combined_norm <= 0.0:
        raise ValueError("FVCG semantic gradient differs")
    cosine = float(torch.dot(dml_vector, combined_vector) / (dml_norm * combined_norm))
    cosine_distance_ppm = max(
        0, round((1.0 - max(-1.0, min(1.0, cosine))) * 1_000_000)
    )
    if direct_vjp_reference is not None:
        direct_vjp_errors = _direct_vjp_errors(
            tuple(semantic_gradients),
            direct_vjp_reference,
            absolute_floor=authority.direct_vjp_atol,
        )

    vision_count, vision_finite = _nonzero_finite_count(vision)
    pooler_count, pooler_finite = _nonzero_finite_count(pooler)
    proxy_count, proxy_finite = _nonzero_finite_count((proxies,))
    language_count, language_finite = _nonzero_finite_count(language)
    if language_count != 0:
        raise ValueError("FVCG language gradient differs")
    gradient_sha256 = hashlib.sha256(
        _gradient_vector(all_trainable).detach().cpu().numpy().astype("<f4").tobytes()
    ).hexdigest()
    optimizer_state_sha256 = _optimizer_state_sha256(optimizer)
    vision_state_sha256 = parameter_state_sha256(vision)
    pooler_state_sha256 = parameter_state_sha256(pooler)
    proxy_state_sha256 = parameter_state_sha256((proxies,))
    updated_state_sha256 = hashlib.sha256(
        bytes.fromhex(vision_state_sha256)
        + bytes.fromhex(pooler_state_sha256)
        + bytes.fromhex(proxy_state_sha256)
    ).hexdigest()
    language_state_sha256 = parameter_state_sha256(language)

    observed_psi = max(memory_psi_full_avg10_ppm, _memory_psi_full_avg10_ppm())
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
        vision_state_changed=vision_state_sha256 != initial_vision_state_sha256,
        pooler_state_changed=pooler_state_sha256 != initial_pooler_state_sha256,
        proxy_state_changed=proxy_state_sha256 != initial_proxy_state_sha256,
        combined_elapsed_ns=combined_elapsed_ns,
        semantic_elapsed_ns=semantic_elapsed_ns,
        peak_cuda_reserved_bytes=_peak_cuda_reserved_bytes(),
        peak_rss_bytes=_peak_rss_bytes(),
        memory_psi_full_avg10_ppm=observed_psi,
        direct_vjp_max_abs_error=direct_vjp_errors[0],
        direct_vjp_max_rel_error=direct_vjp_errors[1],
        gradient_sha256=gradient_sha256,
        updated_state_sha256=updated_state_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        language_state_sha256=language_state_sha256,
    )
    return evidence.validated(authority)


def _run_context(
    context: PhaseAContext, *, authority: FvcgStepAuthority, ordinal: int
) -> FvcgStepEvidence:
    if type(context) is not PhaseAContext:
        raise ValueError("FVCG Phase A context differs")
    return run_combined_step(
        context.adapter,
        authority=authority,
        optimizer=context.optimizer,
        proxies=context.proxies,
        proxy_labels=context.proxy_labels,
        dml_microbatch=context.dml_microbatch,
        dml_labels=context.dml_labels,
        semantic_pair=context.semantic_pair,
        correct_completion_ids=context.correct_completion_ids,
        incorrect_completion_ids=context.incorrect_completion_ids,
        ordinal=ordinal,
        direct_vjp_errors=context.direct_vjp_errors,
        memory_psi_full_avg10_ppm=context.memory_psi_full_avg10_ppm,
        direct_vjp_reference=context.direct_vjp_reference,
    )


def _write_new_atomic(path: Path, raw: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(partial, path)


def run_phase_a(
    *,
    authority: FvcgStepAuthority,
    context_factory: Callable[[int], PhaseAContext],
    output_directory: Path,
) -> bytes:
    """Run one warm-up, three restored measured steps, and a restored repeat."""

    authority.validated()
    if (
        not callable(context_factory)
        or not isinstance(output_directory, Path)
        or output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise ValueError("FVCG Phase A execution authority differs")
    result_path = output_directory / "result.json"
    if result_path.exists():
        return canonical_fvcg_phase_a_result_bytes(
            validate_fvcg_phase_a_result_bytes(result_path.read_bytes())
        )

    def release_context() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    warmup = context_factory(0)
    _run_context(warmup, authority=authority, ordinal=0)
    del warmup
    release_context()
    initial_language_state: str | None = None
    measured: list[FvcgStepEvidence] = []
    for ordinal in range(3):
        context = context_factory(ordinal)
        observed_language_state = parameter_state_sha256(context.adapter.language_parameters())
        if initial_language_state is None:
            initial_language_state = observed_language_state
        elif observed_language_state != initial_language_state:
            raise ValueError("FVCG restored language state differs")
        measured.append(_run_context(context, authority=authority, ordinal=ordinal))
        del context
        release_context()
    if initial_language_state is None:
        raise ValueError("FVCG initial language state differs")
    steps = tuple(measured)
    repeat_context = context_factory(0)
    if (
        parameter_state_sha256(repeat_context.adapter.language_parameters())
        != initial_language_state
    ):
        raise ValueError("FVCG repeated language state differs")
    repeated_step_zero = _run_context(repeat_context, authority=authority, ordinal=0)
    result = FvcgPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=repeated_step_zero,
        initial_language_state_sha256=initial_language_state,
    )
    raw = canonical_fvcg_phase_a_result_bytes(result)
    _write_new_atomic(result_path, raw)
    if validate_fvcg_phase_a_result_bytes(result_path.read_bytes()) != result:
        raise ValueError("FVCG Phase A persisted result differs")
    return raw


def _reject_duplicate_options(argv: list[str]) -> None:
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate FVCG option")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only Phase-A execution boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    _reject_duplicate_options(values)
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--selection-seed-sha256", required=True)
    parser.add_argument("--execute-phase-a", required=True, action="store_true")
    parsed = parser.parse_args(values)
    for name, length in (("source_commit", 40), ("selection_seed_sha256", 64)):
        value = getattr(parsed, name)
        if (
            type(value) is not str
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"{name.replace('_', ' ')} must be {length} lowercase hex")
    if parsed.model_root.is_symlink() or not parsed.model_root.is_dir():
        parser.error("model root must be an existing regular directory")
    for name in ("snapshot_manifest", "fixture", "p32_authority", "train_manifest"):
        path = getattr(parsed, name)
        if path.is_symlink() or not path.is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing regular file")
    if parsed.output_directory.is_symlink() or not parsed.output_directory.is_dir():
        parser.error("output directory must be an existing regular directory")
    return parsed


def _executing_source_commit() -> str:
    return _authenticated_source_commit(_REPOSITORY_ROOT)


def _configure_determinism() -> None:
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in {None, ":4096:8"}:
        raise ValueError("FVCG CUBLAS determinism authority differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _real_context_factory(
    *,
    snapshot: object,
    fixture: object,
    local: object,
    authority: FvcgStepAuthority,
) -> Callable[[int], PhaseAContext]:
    def make_context(ordinal: int) -> PhaseAContext:
        adapter = load_qwen_adapter(
            LoadedAuthority(snapshot=snapshot, fixture=fixture),
            factory=TransformersFactory(),
        )
        selected = select_stratum_pair(
            tuple(range(8)), seed_sha256=authority.selection_seed_sha256, step=ordinal
        )
        row = local.pilot_schedule.pairs[selected]
        pair = adapter.prepare_image_pair(
            (local.images[row.left_index], local.images[row.right_index]),
            local.prompt_utf8,
            local.attribute_token_span,
            local.patch_tokens_per_image,
        )
        protocol = local.completion_protocol
        correct_ids, incorrect_ids = (
            (protocol.same_prefix_ids, protocol.different_prefix_ids)
            if row.relation_sign == 1
            else (protocol.different_prefix_ids, protocol.same_prefix_ids)
        )
        device = adapter.vision_parameters()[0].device
        labels = torch.tensor(fixture.pseudo_labels, dtype=torch.long, device=device)
        cpu_state = torch.random.get_rng_state()
        try:
            seed = int.from_bytes(
                hashlib.sha256(
                    b"fvcg-proxies-v1\0" + bytes.fromhex(authority.selection_seed_sha256)
                ).digest()[:8],
                "little",
            )
            torch.random.default_generator.manual_seed(seed)
            proxy_labels = torch.arange(
                len(set(fixture.pseudo_labels)), dtype=torch.long
            ).repeat_interleave(15)
            proxies = torch.nn.Parameter(
                torch.randn(
                    proxy_labels.numel(),
                    adapter.pooler_token_dim,
                    dtype=torch.float32,
                ).to(device)
            )
        finally:
            torch.random.set_rng_state(cpu_state)
        proxy_labels = proxy_labels.to(device)
        optimizer = Fp32MasterAdamW(
            [*adapter.vision_parameters(), *adapter.pooler.parameters(), proxies],
            lr=FVCG_ADAMW_LR,
            weight_decay=FVCG_ADAMW_WEIGHT_DECAY,
        )
        captured = adapter.collapsed_verdict_patch_gradient(
            pair,
            correct_completion_ids=correct_ids,
            incorrect_completion_ids=incorrect_ids,
        )
        adapter.boundary_verdict_vjp_backward(
            pair,
            completion_ids=correct_ids,
            boundary_names=captured.boundary_names,
            boundary_patch_tokens=captured.boundary_patch_tokens,
            boundary_gradient=captured.boundary_predicted_gradient,
        )
        vjp_reference = tuple(
            None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for parameter in adapter.vision_parameters()
        )
        adapter.clear_graphs()
        return PhaseAContext(
            adapter=adapter,
            optimizer=optimizer,
            proxies=proxies,
            proxy_labels=proxy_labels,
            dml_microbatch=adapter.prepare_microbatch(fixture),
            dml_labels=labels,
            semantic_pair=pair,
            correct_completion_ids=correct_ids,
            incorrect_completion_ids=incorrect_ids,
            direct_vjp_errors=(0.0, 0.0),
            memory_psi_full_avg10_ppm=_memory_psi_full_avg10_ppm(),
            direct_vjp_reference=vjp_reference,
        )

    return make_context


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs and run exactly one Phase-A campaign."""

    args = parse_args(argv)
    if _executing_source_commit() != args.source_commit:
        raise ValueError("FVCG executing source commit differs")
    _configure_determinism()
    local = load_p32_local_authority(
        args.p32_authority,
        args.train_manifest,
        source_commit=args.source_commit,
    )
    snapshot = load_snapshot_authority(root=args.model_root, manifest_path=args.snapshot_manifest)
    fixture = load_fixture_authority(args.fixture)
    if (
        fixture.source_commit != args.source_commit
        or getattr(snapshot, "model_revision", None) != local.rollout_authority.model_revision
    ):
        raise ValueError("FVCG local authority binding differs")
    authority = FvcgStepAuthority(
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        model_revision=local.rollout_authority.model_revision,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        selection_seed_sha256=args.selection_seed_sha256,
        semantic_weight=1.0,
        gradient_clip_norm=10.0,
        direct_vjp_atol=0.05,
        direct_vjp_rtol=0.01,
    ).validated()
    raw = run_phase_a(
        authority=authority,
        context_factory=_real_context_factory(
            snapshot=snapshot,
            fixture=fixture,
            local=local,
            authority=authority,
        ),
        output_directory=args.output_directory,
    )
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
