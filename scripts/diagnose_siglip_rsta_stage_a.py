#!/usr/bin/env python3
"""Authenticated local-only SigLIP RSTA Stage-A scientific diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import struct
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sfora.siglip_rsta_stage_a import (
        RstaAggregate,
        RstaCheckpointBinding,
        RstaControlBinding,
        RstaJvpBackendEvidence,
        RstaReceiverEvidence,
        RstaRolePanel,
    )


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("scientific input paths must be normalized absolute paths")
    return path


def parse_stage_a_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the closed local capability surface without touching any input."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--control-binding", required=True, type=_absolute_path)
    parser.add_argument("--control-binding-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--checkpoint-seed17", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-seed29", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-seed43", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--image-root", required=True, type=_absolute_path)
    parser.add_argument("--execute-stage-a", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


@dataclass(frozen=True)
class LoadedStageACheckpoint:
    """One authenticated final checkpoint projected to model state only."""

    seed: int
    model_state: Mapping[str, object]


@dataclass(frozen=True)
class LoadedStageAAuthority:
    """Outcome-blind local authority available to the scientific loop."""

    binding: RstaControlBinding
    checkpoints: tuple[LoadedStageACheckpoint, ...]
    example_ids: tuple[str, ...]
    labels: tuple[int, ...]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class StageATensorCache:
    """Exactly-once deterministic tensors for every selected scientific role."""

    example_ids: tuple[str, ...]
    tensors: Mapping[str, object]
    tensor_sha256: Mapping[str, str]

    def batch(self, example_ids: tuple[str, ...]):
        import torch

        if (
            type(example_ids) is not tuple
            or not example_ids
            or len(set(example_ids)) != len(example_ids)
            or any(example_id not in self.tensors for example_id in example_ids)
        ):
            raise ValueError("RSTA tensor-cache batch authority differs")
        return torch.stack([self.tensors[example_id] for example_id in example_ids])


class PreScienceInvalid(ValueError):
    """A registered authority failure before any scientific receiver row opens."""


class PostScienceFailure(RuntimeError):
    """A terminal scientific failure for which no candidate result may be emitted."""


_INVALID_CLAUSES = frozenset(
    {
        "authority-mismatch",
        "backend-unavailable",
        "fixture-failure",
        "throughput-budget",
        "determinism-failure",
    }
)


def pre_science_invalid_result_bytes(clause: str) -> bytes:
    """Serialize the only candidate result allowed before scientific rows open."""

    if type(clause) is not str or clause not in _INVALID_CLAUSES:
        raise ValueError("RSTA pre-science INVALID clause differs from authority")
    return _canonical_json(
        {
            "schema": "siglip-rsta-stage-a-result-v1",
            "claim_eligible": False,
            "verdict": "INVALID",
            "first_decisive_clause": clause,
        }
    )


@dataclass(frozen=True)
class StageASeedExecution:
    """One complete seed execution plus its bitwise repeatability witness."""

    receiver_evidence: tuple[RstaReceiverEvidence, ...]
    first_receiver_first_sha256: str
    first_receiver_repeat_sha256: str
    parameter_names: tuple[str, ...]
    parameter_numels: tuple[int, ...]
    logical_batch_replays: int
    receiver_actions: int
    autocast_device_type: str
    autocast_dtype: str
    autocast_enabled: bool
    support_replays: int
    module_training: bool
    gradient_checkpointing_enabled: bool
    torch_compile_enabled: bool
    attention_implementation: str


@dataclass(frozen=True)
class CompletedStageAScience:
    """Complete in-memory Stage-A evidence; no partial result is representable."""

    authority: LoadedStageAAuthority
    panel: RstaRolePanel
    tensor_sha256_by_id: Mapping[str, str]
    backend: RstaJvpBackendEvidence
    seeds: tuple[StageASeedExecution, ...]
    aggregate: RstaAggregate
    aggregate_bytes: bytes


@dataclass(frozen=True)
class StageAExecutionAudit:
    """Outcome-independent execution and resource evidence for one complete run."""

    parameter_names: tuple[str, ...]
    parameter_numels: tuple[int, ...]
    checkpointing_max_relative_disagreement: float
    fixture_sha256: str
    module_training: bool
    gradient_checkpointing_enabled: bool
    torch_compile_enabled: bool
    attention_implementation: str
    autocast_device_type: str
    autocast_dtype: str
    autocast_enabled: bool
    cublas_workspace_config: str
    deterministic_algorithms_enabled: bool
    deterministic_algorithms_warn_only: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    elapsed_ns: int
    peak_rss_bytes: int
    peak_cuda_bytes: int
    memory_psi_growth_ppm: int
    swap_growth_bytes: int


@dataclass(frozen=True)
class StageAModelPreflight:
    """Sealed checkpointing/backend/parameter evidence before scientific rows."""

    backend: RstaJvpBackendEvidence
    checkpointing_max_relative_disagreement: float
    fixture_sha256: str
    parameter_names: tuple[str, ...]
    parameter_numels: tuple[int, ...]


@dataclass(frozen=True)
class StageADeterminismEvidence:
    """Observed deterministic-kernel policy established before CUDA science."""

    cublas_workspace_config: str
    deterministic_algorithms_enabled: bool
    deterministic_algorithms_warn_only: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


@dataclass(frozen=True)
class StageAModelCampaign:
    """A complete fixed-runner campaign and its observed pre-science evidence."""

    completed: CompletedStageAScience
    preflights: tuple[StageAModelPreflight, ...]
    determinism: StageADeterminismEvidence


def configure_stage_a_determinism() -> StageADeterminismEvidence:
    """Establish and return the exact deterministic CUDA execution policy."""

    import torch

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace != ":4096:8":
        raise ValueError("RSTA CUBLAS workspace authority differs")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    evidence = StageADeterminismEvidence(
        cublas_workspace_config=workspace,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        deterministic_algorithms_warn_only=(torch.is_deterministic_algorithms_warn_only_enabled()),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    if evidence != StageADeterminismEvidence(
        cublas_workspace_config=":4096:8",
        deterministic_algorithms_enabled=True,
        deterministic_algorithms_warn_only=False,
        cudnn_benchmark=False,
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
    ):
        raise ValueError("RSTA deterministic execution policy differs")
    return evidence


def _read_regular(path: Path, *, role: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def _canonical_json(value: dict[str, object]) -> bytes:
    from sfora.pass209_m4 import canonical_json_bytes

    return canonical_json_bytes(value)


def _image_basename(example_id: str) -> str:
    digest = hashlib.sha256(
        b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")
    ).hexdigest()
    return f"{digest}.image"


def _parse_control_binding(raw: bytes) -> RstaControlBinding:
    from sfora.siglip_rsta_stage_a import RstaCheckpointBinding, RstaControlBinding

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RSTA control binding is not valid JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "control_complete",
        "source_commit",
        "config_sha256",
        "run_authority_sha256",
        "dataset_id",
        "dataset_revision",
        "environment_sha256",
        "optimization_manifest_sha256",
        "selected_microbatch_size",
        "checkpoints",
    }
    if type(value) is not dict or set(value) != expected or raw != _canonical_json(value):
        raise ValueError("RSTA control binding authority differs")
    checkpoints = value["checkpoints"]
    if type(checkpoints) is not list or any(
        type(item) is not dict or set(item) != {"seed", "sha256", "byte_length"}
        for item in checkpoints
    ):
        raise ValueError("RSTA control binding checkpoint schema differs")
    try:
        return RstaControlBinding(
            schema=value["schema"],
            claim_eligible=value["claim_eligible"],
            control_complete=value["control_complete"],
            source_commit=value["source_commit"],
            config_sha256=value["config_sha256"],
            run_authority_sha256=value["run_authority_sha256"],
            dataset_id=value["dataset_id"],
            dataset_revision=value["dataset_revision"],
            environment_sha256=value["environment_sha256"],
            optimization_manifest_sha256=value["optimization_manifest_sha256"],
            selected_microbatch_size=value["selected_microbatch_size"],
            checkpoints=tuple(
                RstaCheckpointBinding(
                    seed=item["seed"],
                    sha256=item["sha256"],
                    byte_length=item["byte_length"],
                )
                for item in checkpoints
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RSTA control binding authority differs") from error


def _load_model_state_checkpoint(
    path: Path,
    authority: RstaCheckpointBinding,
    binding: RstaControlBinding,
) -> LoadedStageACheckpoint:
    import numpy as np
    import torch

    raw = _read_regular(path, role="RSTA checkpoint")
    if len(raw) != authority.byte_length or hashlib.sha256(raw).hexdigest() != authority.sha256:
        raise ValueError("RSTA checkpoint digest or length differs")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))

    expected = {
        "claim_eligible",
        "completed_epoch",
        "config_sha256",
        "cpu_rng_state",
        "cuda_rng_states",
        "final_objective",
        "initial_snapshot_sha256",
        "maximum_score_disagreement",
        "model_state",
        "optimizer_state",
        "run_authority_sha256",
        "sampler_cycles",
        "sampler_positions",
        "schema",
        "seed",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("RSTA checkpoint payload schema differs")
    model_state = payload["model_state"]
    cycles = payload["sampler_cycles"]
    positions = payload["sampler_positions"]
    objective = payload["final_objective"]
    disagreement = payload["maximum_score_disagreement"]
    cpu_rng_state = payload["cpu_rng_state"]
    checkpoint_cuda_states = payload["cuda_rng_states"]
    if (
        payload["schema"] != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload["claim_eligible"] is not False
        or type(payload["seed"]) is not int
        or payload["seed"] != authority.seed
        or type(payload["completed_epoch"]) is not int
        or payload["completed_epoch"] != 60
        or payload["config_sha256"] != binding.config_sha256
        or payload["run_authority_sha256"] != binding.run_authority_sha256
        or type(objective) is not float
        or not math.isfinite(objective)
        or type(disagreement) is not float
        or not math.isfinite(disagreement)
        or disagreement < 0.0
        or type(payload["initial_snapshot_sha256"]) is not str
        or len(payload["initial_snapshot_sha256"]) != 64
        or any(
            character not in "0123456789abcdef" for character in payload["initial_snapshot_sha256"]
        )
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or len(cycles) != 49
        or len(positions) != 49
        or any(type(value) is not int or value < 0 for value in cycles + positions)
        or not isinstance(cpu_rng_state, torch.Tensor)
        or cpu_rng_state.dtype != torch.uint8
        or type(checkpoint_cuda_states) is not tuple
        or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
            for value in checkpoint_cuda_states
        )
        or type(payload["optimizer_state"]) is not dict
        or type(model_state) is not OrderedDict
        or not model_state
        or any(type(name) is not str or not name for name in model_state)
        or any(not isinstance(tensor, torch.Tensor) for tensor in model_state.values())
    ):
        raise ValueError("RSTA checkpoint authority differs")
    return LoadedStageACheckpoint(
        seed=authority.seed,
        model_state=MappingProxyType(dict(model_state)),
    )


def _load_optimization_manifest(
    path: Path,
    expected_sha256: str,
    binding: RstaControlBinding,
    image_root: Path,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[Path, ...]]:
    raw = _read_regular(path, role="RSTA optimization manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("RSTA optimization manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RSTA optimization manifest is not valid JSON") from error
    if (
        type(value) is not dict
        or set(value) != {"schema", "claim_eligible", "dataset_id", "dataset_revision", "examples"}
        or raw != _canonical_json(value)
        or value["schema"] != "rsta-optimization-manifest-v1"
        or value["claim_eligible"] is not False
        or value["dataset_id"] != binding.dataset_id
        or value["dataset_revision"] != binding.dataset_revision
        or type(value["examples"]) is not list
        or not value["examples"]
    ):
        raise ValueError("RSTA optimization manifest authority differs")
    try:
        resolved_root = image_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("RSTA image root must be one real directory") from error
    if image_root.is_symlink() or not image_root.is_dir() or resolved_root != image_root:
        raise ValueError("RSTA image root must be one real directory")
    example_ids: list[str] = []
    labels: list[int] = []
    image_paths: list[Path] = []
    expected_basenames: set[str] = set()
    for row in value["examples"]:
        if (
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or not 0 <= row["label"] < 49
        ):
            raise ValueError("RSTA optimization manifest row differs")
        basename = _image_basename(row["example_id"])
        image_path = image_root / basename
        try:
            resolved_image = image_path.resolve(strict=True)
        except OSError as error:
            raise ValueError("RSTA optimization image must be one regular file") from error
        if not resolved_image.is_relative_to(resolved_root) or resolved_image != image_path:
            raise ValueError("RSTA optimization image path escapes authority")
        if image_path.is_symlink() or not image_path.is_file():
            raise ValueError("RSTA optimization image must be one regular file")
        example_ids.append(row["example_id"])
        labels.append(row["label"])
        image_paths.append(image_path)
        expected_basenames.add(basename)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("RSTA optimization example identities are duplicated")
    observed_basenames = {path.name for path in image_root.iterdir()}
    if observed_basenames != expected_basenames:
        raise ValueError("RSTA image namespace differs from optimization authority")
    return tuple(example_ids), tuple(labels), tuple(image_paths)


def load_stage_a_authority(arguments: argparse.Namespace) -> LoadedStageAAuthority:
    """Authenticate local evidence and retain no optimizer or outcome artifacts."""

    binding_raw = _read_regular(arguments.control_binding, role="RSTA control binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("RSTA control binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if arguments.optimization_manifest_sha256 != binding.optimization_manifest_sha256:
        raise ValueError("RSTA optimization manifest digest differs")
    checkpoint_paths = (
        arguments.checkpoint_seed17,
        arguments.checkpoint_seed29,
        arguments.checkpoint_seed43,
    )
    if len(set(checkpoint_paths)) != 3:
        raise ValueError("RSTA checkpoint paths must be distinct")
    checkpoints = tuple(
        _load_model_state_checkpoint(path, authority, binding)
        for path, authority in zip(checkpoint_paths, binding.checkpoints, strict=True)
    )
    example_ids, labels, image_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.image_root,
    )
    return LoadedStageAAuthority(
        binding=binding,
        checkpoints=checkpoints,
        example_ids=example_ids,
        labels=labels,
        image_paths=image_paths,
    )


def stage_a_tensor_example_ids(panel: RstaRolePanel) -> tuple[str, ...]:
    """Return the exact selected support/graph tensor identities in frozen order."""

    from sfora.siglip_rsta_stage_a import RstaRolePanel

    if type(panel) is not RstaRolePanel:
        raise ValueError("RSTA tensor roles require one registered panel")
    ordered: list[str] = []
    seen: set[str] = set()

    def append(example_id: str) -> None:
        if example_id not in seen:
            seen.add(example_id)
            ordered.append(example_id)

    for support_pair in panel.support_ids_by_label:
        for example_id in support_pair:
            append(example_id)
    for batch in (*panel.primary_batches, *panel.alternate_batches):
        for row in batch.rows:
            append(row.example_id)
    return tuple(ordered)


def _scientific_tensor_sha256(tensor: object) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("RSTA transform must return one torch tensor")
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(str(tuple(int(value) for value in tensor.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def cache_stage_a_tensors(
    authority: LoadedStageAAuthority,
    panel: RstaRolePanel,
    *,
    graph_transform: Callable[[object], object],
    evaluation_transform: Callable[[object], object],
    materialize: Callable[[Path], object],
) -> StageATensorCache:
    """Apply each role transform once under ID-derived isolated RNG states."""

    import numpy as np
    import torch

    if type(authority) is not LoadedStageAAuthority:
        raise ValueError("RSTA tensor cache authority differs")
    paths = dict(zip(authority.example_ids, authority.image_paths, strict=True))
    required_ids = stage_a_tensor_example_ids(panel)
    if any(example_id not in paths for example_id in required_ids):
        raise ValueError("RSTA tensor role is absent from image authority")
    support_ids = {example_id for pair in panel.support_ids_by_label for example_id in pair}
    tensors: dict[str, torch.Tensor] = {}
    digests: dict[str, str] = {}
    for example_id in required_ids:
        source = materialize(paths[example_id])
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        try:
            random.seed(
                int.from_bytes(
                    hashlib.sha256(b"augment-python\0" + example_id.encode()).digest()[:8],
                    "big",
                )
            )
            np.random.seed(
                int.from_bytes(
                    hashlib.sha256(b"augment-numpy\0" + example_id.encode()).digest()[:8],
                    "big",
                )
                % (2**32)
            )
            torch.manual_seed(
                int.from_bytes(
                    hashlib.sha256(b"augment-torch\0" + example_id.encode()).digest()[:8],
                    "big",
                )
            )
            transform = evaluation_transform if example_id in support_ids else graph_transform
            value = transform(source)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("RSTA transform returned an invalid tensor")
        cached = value.detach().cpu().contiguous().clone()
        tensors[example_id] = cached
        digests[example_id] = _scientific_tensor_sha256(cached)
    return StageATensorCache(
        example_ids=required_ids,
        tensors=MappingProxyType(tensors),
        tensor_sha256=MappingProxyType(digests),
    )


def load_stage_a_checkpoint_model(
    checkpoint: LoadedStageACheckpoint,
    *,
    model_factory: Callable[[], object],
    device: object,
):
    """Strictly restore one model-only checkpoint without leaking RNG state."""

    import numpy as np
    import torch

    from sfora.siglip_proxy_control import PooledProxyAnchorModel

    if (
        type(checkpoint) is not LoadedStageACheckpoint
        or not callable(model_factory)
        or not isinstance(device, torch.device)
    ):
        raise ValueError("RSTA checkpoint model authority differs")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    try:
        model = model_factory()
        if type(model) is not PooledProxyAnchorModel:
            raise ValueError("RSTA model factory returned the wrong concrete type")
        try:
            model.load_state_dict(dict(checkpoint.model_state), strict=True)
        except RuntimeError as error:
            raise ValueError("RSTA checkpoint model state differs") from error
        model.to(device).train()
        for parameter in model.parameters():
            parameter.grad = None
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))
    observed = model.state_dict()
    if set(observed) != set(checkpoint.model_state) or any(
        not torch.equal(observed[name].detach().cpu(), value.detach().cpu())
        for name, value in checkpoint.model_state.items()
    ):
        raise ValueError("RSTA checkpoint model state differs")
    return model


def preflight_stage_a_model(
    model: object,
    binding: RstaControlBinding,
    *,
    input_shape: tuple[int, ...],
    checkpointing_enabled: Callable[[], bool],
    disable_checkpointing: Callable[[], None],
) -> StageAModelPreflight:
    """Seal checkpointing equivalence and one JVP backend on synthetic inputs."""

    import numpy as np
    import torch

    from sfora.siglip_proxy_control import PooledProxyAnchorModel
    from sfora.siglip_rsta_stage_a import (
        RstaControlBinding,
        contextual_rsta_direction,
        preflight_rsta_jvp_backend,
    )

    if (
        type(model) is not PooledProxyAnchorModel
        or type(binding) is not RstaControlBinding
        or type(input_shape) is not tuple
        or not input_shape
        or any(type(value) is not int or value <= 0 for value in input_shape)
        or not callable(checkpointing_enabled)
        or not callable(disable_checkpointing)
        or checkpointing_enabled() is not True
        or model.training is not True
    ):
        raise ValueError("RSTA checkpointing preflight authority differs")
    parameters = tuple(model.parameters())
    device = parameters[0].device
    elements = math.prod(input_shape)
    columns = torch.arange(elements, dtype=torch.int64, device=device).unsqueeze(0)
    rows = torch.arange(120, dtype=torch.int64, device=device).unsqueeze(1)
    fixture = (((columns + rows * 131) % 251).float() / 125.0 - 1.0).reshape((120, *input_shape))
    labels = (torch.arange(120, dtype=torch.int64, device=device) % 49).contiguous()
    enabled_direction = contextual_rsta_direction(
        model,
        fixture,
        labels,
        binding=binding,
        alpha=32.0,
        delta=0.1,
    )
    disable_checkpointing()
    if checkpointing_enabled() is not False:
        raise ValueError("RSTA checkpointing preflight did not disable checkpointing")
    disabled_direction = contextual_rsta_direction(
        model,
        fixture,
        labels,
        binding=binding,
        alpha=32.0,
        delta=0.1,
    )
    if enabled_direction.parameter_names != disabled_direction.parameter_names:
        raise ValueError("RSTA checkpointing preflight parameter order differs")

    def relative(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = max(
            float(torch.linalg.vector_norm(left)),
            float(torch.linalg.vector_norm(right)),
            1.0e-12,
        )
        return float(torch.linalg.vector_norm(left - right)) / denominator

    disagreements = [relative(enabled_direction.dbar, disabled_direction.dbar)]
    disagreements.extend(
        relative(left, right)
        for left, right in zip(
            enabled_direction.parameter_direction,
            disabled_direction.parameter_direction,
            strict=True,
        )
    )
    maximum = max(disagreements)
    if not math.isfinite(maximum) or maximum > 1.0e-5:
        raise ValueError("RSTA checkpointing preflight disagreement exceeds authority")
    backend = preflight_rsta_jvp_backend(
        model,
        fixture[0:1],
        disabled_direction.dbar[0],
        parameter_names=disabled_direction.parameter_names,
        parameter_direction=disabled_direction.parameter_direction,
    )
    digest = hashlib.sha256()
    for value in (fixture, labels):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    for name in disabled_direction.parameter_names:
        digest.update(name.encode("utf-8") + b"\0")
    parameter_by_name = dict(model.named_parameters())
    parameter_numels = tuple(
        int(parameter_by_name[name].numel()) for name in disabled_direction.parameter_names
    )
    for parameter in model.parameters():
        parameter.grad = None
    return StageAModelPreflight(
        backend=backend,
        checkpointing_max_relative_disagreement=maximum,
        fixture_sha256=digest.hexdigest(),
        parameter_names=disabled_direction.parameter_names,
        parameter_numels=parameter_numels,
    )


def _receiver_execution_sha256(
    tensors: tuple[object, ...],
    score: RstaReceiverEvidence | object,
) -> str:
    import torch

    digest = hashlib.sha256()
    for value in tensors:
        if not isinstance(value, torch.Tensor):
            raise ValueError("RSTA repeatability tensor differs from authority")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    values = vars(score)
    for name, value in values.items():
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("RSTA repeatability score differs from authority")
        digest.update(name.encode("ascii") + b"\0" + struct.pack(">d", value))
    return digest.hexdigest()


def _validate_checkpoint_model_state(checkpoint: LoadedStageACheckpoint, model: object) -> None:
    import torch

    observed = model.state_dict()
    if set(observed) != set(checkpoint.model_state) or any(
        not isinstance(expected, torch.Tensor)
        or not torch.equal(observed[name].detach().cpu(), expected.detach().cpu())
        for name, expected in checkpoint.model_state.items()
    ):
        raise ValueError("RSTA checkpoint model state differs")


def run_stage_a_seed_model(
    checkpoint: LoadedStageACheckpoint,
    panel: RstaRolePanel,
    binding: RstaControlBinding,
    cache: StageATensorCache,
    backend: RstaJvpBackendEvidence,
    model: object,
) -> StageASeedExecution:
    """Execute both registered panels for one already-authenticated model state."""

    import torch

    from sfora.siglip_proxy_control import PooledProxyAnchorModel
    from sfora.siglip_rsta_stage_a import (
        RstaControlBinding,
        RstaJvpBackendEvidence,
        RstaReceiverEvidence,
        contextual_rsta_direction,
        proxy_free_margin_direction,
        receiver_rsta_fields,
        rsta_control_directions,
        score_rsta_receiver,
    )

    if (
        type(checkpoint) is not LoadedStageACheckpoint
        or type(binding) is not RstaControlBinding
        or type(cache) is not StageATensorCache
        or type(backend) is not RstaJvpBackendEvidence
        or type(model) is not PooledProxyAnchorModel
        or model.training is not True
        or cache.example_ids != stage_a_tensor_example_ids(panel)
        or set(cache.tensor_sha256) != set(cache.example_ids)
    ):
        raise ValueError("RSTA seed model authority differs")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("RSTA seed model has no parameters")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("RSTA seed model spans multiple devices")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("RSTA seed model gradients are not cleared")
    _validate_checkpoint_model_state(checkpoint, model)

    support_ids = tuple(example_id for pair in panel.support_ids_by_label for example_id in pair)
    support_replays = 0

    def encode_supports() -> dict[str, object]:
        nonlocal support_replays
        support_replays += 1
        model.eval()
        with torch.no_grad():
            values = model.encode(cache.batch(support_ids).to(device))
        model.train()
        return dict(zip(support_ids, values, strict=True))

    support_by_id = encode_supports()
    foreign_ids_by_label = tuple(pair[0] for pair in panel.support_ids_by_label)
    observed_parameter_names: tuple[str, ...] | None = None
    logical_batch_replays = 0
    receiver_actions = 0

    def run_batch_autocast(batch) -> tuple[dict[str, object], dict[str, str]]:
        nonlocal logical_batch_replays, observed_parameter_names, receiver_actions
        logical_batch_replays += 1
        row_ids = tuple(row.example_id for row in batch.rows)
        inputs = cache.batch(row_ids).to(device)
        labels = torch.tensor([row.label for row in batch.rows], dtype=torch.int64, device=device)
        direction = contextual_rsta_direction(
            model,
            inputs,
            labels,
            binding=binding,
            alpha=32.0,
            delta=0.1,
        )
        if observed_parameter_names is None:
            observed_parameter_names = direction.parameter_names
        elif direction.parameter_names != observed_parameter_names:
            raise ValueError("RSTA seed parameter authority changed between batches")
        receiver_positions = tuple(
            index for index, row in enumerate(batch.rows) if row.role == "receiver"
        )
        receiver_ids = tuple(batch.rows[index].example_id for index in receiver_positions)
        receiver_actions += len(receiver_ids)
        fields_by_id: dict[str, object] = {}
        outcomes_by_id: dict[str, object] = {}
        for position in receiver_positions:
            row = batch.rows[position]
            fields = receiver_rsta_fields(
                model,
                inputs[position : position + 1],
                direction.dbar[position],
                parameter_names=direction.parameter_names,
                parameter_direction=direction.parameter_direction,
                backend=backend.backend,
            )
            positive_descriptors = torch.stack(
                [support_by_id[value] for value in panel.support_ids_by_label[row.label]]
            )
            foreign_ids = tuple(
                foreign_ids_by_label[label] for label in range(49) if label != row.label
            )
            foreign_descriptors = torch.stack([support_by_id[value] for value in foreign_ids])
            role_digests = tuple(
                hashlib.sha256(b"rsta-siglip-a-v1|role|\0" + value.encode()).hexdigest()
                for value in foreign_ids
            )
            outcome = proxy_free_margin_direction(
                fields.descriptor,
                positive_descriptors,
                foreign_descriptors,
                foreign_example_ids=foreign_ids,
                foreign_role_digests=role_digests,
            )
            fields_by_id[row.example_id] = fields
            outcomes_by_id[row.example_id] = outcome
        descriptor_rows = torch.stack(
            [fields_by_id[example_id].descriptor for example_id in receiver_ids]
        )
        outcome_rows = torch.stack(
            [outcomes_by_id[example_id].direction for example_id in receiver_ids]
        )
        controls = rsta_control_directions(
            descriptor_rows,
            outcome_rows,
            receiver_ids=receiver_ids,
        )
        scores: dict[str, object] = {}
        digests: dict[str, str] = {}
        for index, example_id in enumerate(receiver_ids):
            position = receiver_positions[index]
            fields = fields_by_id[example_id]
            outcome = outcomes_by_id[example_id]
            score = score_rsta_receiver(
                fields=fields,
                dbar=direction.dbar[position],
                outcome_direction=outcome.direction,
                random_target=controls.random_targets[index],
                deranged_direction=controls.deranged_directions[index],
            )
            scores[example_id] = score
            digests[example_id] = _receiver_execution_sha256(
                (
                    fields.descriptor,
                    fields.batch_motion,
                    fields.self_motion,
                    direction.dbar[position],
                    outcome.direction,
                    controls.random_targets[index],
                    controls.deranged_directions[index],
                ),
                score,
            )
        del direction, inputs, labels, fields_by_id, outcomes_by_id, controls
        for parameter in model.parameters():
            parameter.grad = None
        return scores, digests

    def run_batch(batch) -> tuple[dict[str, object], dict[str, str]]:
        return run_batch_autocast(batch)

    def run_panel(batches) -> tuple[dict[str, object], dict[str, str]]:
        scores: dict[str, object] = {}
        digests: dict[str, str] = {}
        for batch in batches:
            batch_scores, batch_digests = run_batch(batch)
            if set(scores) & set(batch_scores):
                raise ValueError("RSTA receiver appears in multiple panel batches")
            scores.update(batch_scores)
            digests.update(batch_digests)
        return scores, digests

    primary_scores, primary_digests = run_panel(panel.primary_batches)
    alternate_scores, alternate_digests = run_panel(panel.alternate_batches)
    first = panel.receivers[0]
    support_by_id = encode_supports()
    primary_repeat_scores, primary_repeat_digests = run_batch(
        panel.primary_batches[first.primary_batch]
    )
    alternate_repeat_scores, alternate_repeat_digests = run_batch(
        panel.alternate_batches[first.alternate_batch]
    )

    def combined_digest(primary_digest: str, alternate_digest: str) -> str:
        return hashlib.sha256(
            bytes.fromhex(primary_digest) + bytes.fromhex(alternate_digest)
        ).hexdigest()

    first_digest = combined_digest(
        primary_digests[first.example_id], alternate_digests[first.example_id]
    )
    repeat_digest = combined_digest(
        primary_repeat_digests[first.example_id],
        alternate_repeat_digests[first.example_id],
    )
    if (
        primary_repeat_scores[first.example_id] != primary_scores[first.example_id]
        or alternate_repeat_scores[first.example_id] != alternate_scores[first.example_id]
    ):
        raise ValueError("RSTA first receiver repeat score differs")
    rows = tuple(
        RstaReceiverEvidence(
            seed=checkpoint.seed,
            label=receiver.label,
            receiver_id=receiver.example_id,
            primary=primary_scores[receiver.example_id],
            alternate=alternate_scores[receiver.example_id],
        )
        for receiver in panel.receivers
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("RSTA seed model retained gradients")
    if device.type == "cuda":
        evidence_reader = getattr(model.tower, "rsta_autocast_evidence", None)
        if not callable(evidence_reader):
            raise ValueError("RSTA seed tower lacks autocast evidence")
        autocast_device_type, autocast_dtype, autocast_enabled = evidence_reader()
    else:
        autocast_device_type, autocast_dtype, autocast_enabled = "cpu", "float32", False
    vision_model = getattr(model.tower, "vision_model", None)
    gradient_checkpointing_enabled = bool(getattr(vision_model, "is_gradient_checkpointing", False))
    attention_implementation = getattr(
        getattr(vision_model, "config", None),
        "_attn_implementation",
        "eager",
    )
    if observed_parameter_names is None:
        raise ValueError("RSTA seed parameter authority is empty")
    parameter_by_name = dict(model.named_parameters())
    return StageASeedExecution(
        receiver_evidence=rows,
        first_receiver_first_sha256=first_digest,
        first_receiver_repeat_sha256=repeat_digest,
        parameter_names=observed_parameter_names,
        parameter_numels=tuple(
            int(parameter_by_name[name].numel()) for name in observed_parameter_names
        ),
        logical_batch_replays=logical_batch_replays,
        receiver_actions=receiver_actions,
        autocast_device_type=autocast_device_type,
        autocast_dtype=autocast_dtype,
        autocast_enabled=autocast_enabled,
        support_replays=support_replays,
        module_training=model.training,
        gradient_checkpointing_enabled=gradient_checkpointing_enabled,
        torch_compile_enabled=hasattr(model, "_orig_mod"),
        attention_implementation=attention_implementation,
    )


def execute_stage_a_scientific_loop(
    authority: LoadedStageAAuthority,
    *,
    backend: RstaJvpBackendEvidence,
    tensor_cache: StageATensorCache,
    seed_runner: Callable[
        [
            LoadedStageACheckpoint,
            RstaRolePanel,
            RstaControlBinding,
            StageATensorCache,
            RstaJvpBackendEvidence,
        ],
        StageASeedExecution,
    ],
) -> CompletedStageAScience:
    """Run the three sealed seeds in registered order and publish only complete evidence."""

    from sfora.siglip_rsta_stage_a import (
        RstaJvpBackendEvidence,
        RstaReceiverEvidence,
        RstaStageAConfig,
        rsta_stage_a_result_bytes,
        select_rsta_roles,
        summarize_rsta_stage_a,
    )

    config = RstaStageAConfig()
    if (
        type(authority) is not LoadedStageAAuthority
        or type(authority.example_ids) is not tuple
        or type(authority.labels) is not tuple
        or type(authority.image_paths) is not tuple
        or not (len(authority.example_ids) == len(authority.labels) == len(authority.image_paths))
        or len(set(authority.example_ids)) != len(authority.example_ids)
        or len(set(authority.image_paths)) != len(authority.image_paths)
        or tuple(checkpoint.seed for checkpoint in authority.checkpoints) != config.seeds
        or tuple(checkpoint.seed for checkpoint in authority.binding.checkpoints) != config.seeds
        or not callable(seed_runner)
    ):
        raise PreScienceInvalid("RSTA scientific authority or seed runner differs")
    if type(backend) is not RstaJvpBackendEvidence:
        raise PreScienceInvalid("RSTA sealed backend authority differs")
    if backend.backend == "forward-mode":
        backend_valid = (
            backend.comparison_available is True
            and type(backend.maximum_relative_disagreement) is float
            and math.isfinite(backend.maximum_relative_disagreement)
            and 0.0 <= backend.maximum_relative_disagreement <= 1.0e-5
            and backend.forward_error is None
        )
    else:
        backend_valid = (
            backend.backend == "double-backward"
            and backend.comparison_available is False
            and backend.maximum_relative_disagreement == 0.0
            and type(backend.forward_error) is str
            and bool(backend.forward_error)
        )
    if not backend_valid:
        raise PreScienceInvalid("RSTA sealed backend authority differs")
    try:
        panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))
    except ValueError as error:
        raise PreScienceInvalid("RSTA role panel authority differs") from error
    required_tensor_ids = stage_a_tensor_example_ids(panel)
    if (
        type(tensor_cache) is not StageATensorCache
        or tensor_cache.example_ids != required_tensor_ids
        or set(tensor_cache.tensors) != set(required_tensor_ids)
        or set(tensor_cache.tensor_sha256) != set(required_tensor_ids)
        or any(
            type(example_id) is not str
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or _scientific_tensor_sha256(tensor_cache.tensors[example_id]) != digest
            for example_id, digest in tensor_cache.tensor_sha256.items()
        )
    ):
        raise PreScienceInvalid("RSTA tensor digest authority differs")
    expected_receivers = tuple((row.label, row.example_id) for row in panel.receivers)
    first_receiver = panel.receivers[0]
    expected_batch_replays = len(panel.primary_batches) + len(panel.alternate_batches) + 2
    expected_receiver_actions = (
        2 * len(panel.receivers)
        + sum(
            row.role == "receiver"
            for row in panel.primary_batches[first_receiver.primary_batch].rows
        )
        + sum(
            row.role == "receiver"
            for row in panel.alternate_batches[first_receiver.alternate_batch].rows
        )
    )
    executions: list[StageASeedExecution] = []
    all_rows: list[RstaReceiverEvidence] = []
    for checkpoint in authority.checkpoints:
        try:
            execution = seed_runner(
                checkpoint,
                panel,
                authority.binding,
                tensor_cache,
                backend,
            )
        except Exception as error:
            raise PostScienceFailure("RSTA seed execution failed") from error
        if type(execution) is not StageASeedExecution:
            raise PostScienceFailure("RSTA seed execution has the wrong concrete type")
        rows = execution.receiver_evidence
        parameter_authority_invalid = (
            type(execution.parameter_names) is not tuple
            or not execution.parameter_names
            or execution.parameter_names != tuple(sorted(execution.parameter_names))
            or type(execution.parameter_numels) is not tuple
            or len(execution.parameter_numels) != len(execution.parameter_names)
            or any(type(value) is not int or value <= 0 for value in execution.parameter_numels)
            or bool(executions)
            and (
                execution.parameter_names != executions[0].parameter_names
                or execution.parameter_numels != executions[0].parameter_numels
            )
        )
        if (
            type(rows) is not tuple
            or len(rows) != len(expected_receivers)
            or any(type(row) is not RstaReceiverEvidence for row in rows)
            or tuple((row.label, row.receiver_id) for row in rows) != expected_receivers
            or any(row.seed != checkpoint.seed for row in rows)
            or type(execution.first_receiver_first_sha256) is not str
            or len(execution.first_receiver_first_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in execution.first_receiver_first_sha256
            )
            or execution.first_receiver_repeat_sha256 != execution.first_receiver_first_sha256
            or parameter_authority_invalid
            or execution.logical_batch_replays != expected_batch_replays
            or execution.receiver_actions != expected_receiver_actions
            or type(execution.autocast_device_type) is not str
            or not execution.autocast_device_type
            or type(execution.autocast_dtype) is not str
            or not execution.autocast_dtype
            or type(execution.autocast_enabled) is not bool
            or bool(executions)
            and (
                execution.autocast_device_type != executions[0].autocast_device_type
                or execution.autocast_dtype != executions[0].autocast_dtype
                or execution.autocast_enabled is not executions[0].autocast_enabled
            )
            or execution.support_replays != 2
            or execution.module_training is not True
            or execution.gradient_checkpointing_enabled is not False
            or execution.torch_compile_enabled is not False
            or execution.attention_implementation != "eager"
        ):
            raise PostScienceFailure("RSTA seed execution row or repeatability authority differs")
        executions.append(execution)
        all_rows.extend(rows)
    try:
        aggregate = summarize_rsta_stage_a(tuple(all_rows), config)
        aggregate_bytes = rsta_stage_a_result_bytes(aggregate)
    except ValueError as error:
        raise PostScienceFailure("RSTA seed execution evidence differs") from error
    return CompletedStageAScience(
        authority=authority,
        panel=panel,
        tensor_sha256_by_id=MappingProxyType(dict(tensor_cache.tensor_sha256)),
        backend=backend,
        seeds=tuple(executions),
        aggregate=aggregate,
        aggregate_bytes=aggregate_bytes,
    )


def execute_stage_a_model_campaign(
    authority: LoadedStageAAuthority,
    *,
    tensor_cache: StageATensorCache,
    model_factory: Callable[[], object],
    device: object,
    input_shape: tuple[int, ...],
    checkpointing_enabled: Callable[[object], bool],
    disable_checkpointing: Callable[[object], None],
) -> StageAModelCampaign:
    """Load, preflight, and execute every bound seed through the fixed real runner."""

    import torch

    if (
        type(authority) is not LoadedStageAAuthority
        or not isinstance(device, torch.device)
        or not callable(model_factory)
        or not callable(checkpointing_enabled)
        or not callable(disable_checkpointing)
    ):
        raise PreScienceInvalid("RSTA model campaign authority differs")
    determinism = configure_stage_a_determinism()
    preflights: list[StageAModelPreflight] = []
    first_checkpoint = authority.checkpoints[0]
    first_model = load_stage_a_checkpoint_model(
        first_checkpoint,
        model_factory=model_factory,
        device=device,
    )

    def preflight(model: object) -> StageAModelPreflight:
        evidence = preflight_stage_a_model(
            model,
            authority.binding,
            input_shape=input_shape,
            checkpointing_enabled=lambda: checkpointing_enabled(model),
            disable_checkpointing=lambda: disable_checkpointing(model),
        )
        if preflights and (
            evidence.backend.backend != preflights[0].backend.backend
            or evidence.parameter_names != preflights[0].parameter_names
            or evidence.parameter_numels != preflights[0].parameter_numels
            or evidence.fixture_sha256 != preflights[0].fixture_sha256
        ):
            raise PreScienceInvalid("RSTA model preflight differs between seeds")
        preflights.append(evidence)
        return evidence

    first_preflight = preflight(first_model)
    pending_first_model: list[object] = [first_model]

    def fixed_runner(checkpoint, panel, binding, cache, backend):
        if checkpoint.seed == first_checkpoint.seed:
            if not pending_first_model:
                raise ValueError("RSTA first seed model was reused")
            model = pending_first_model.pop()
            evidence = first_preflight
        else:
            model = load_stage_a_checkpoint_model(
                checkpoint,
                model_factory=model_factory,
                device=device,
            )
            evidence = preflight(model)
        if evidence.backend.backend != backend.backend:
            raise ValueError("RSTA seed backend differs from sealed backend")
        execution = run_stage_a_seed_model(
            checkpoint,
            panel,
            binding,
            cache,
            backend,
            model,
        )
        if (
            execution.parameter_names != evidence.parameter_names
            or execution.parameter_numels != evidence.parameter_numels
        ):
            raise ValueError("RSTA seed parameters differ from preflight")
        return execution

    completed = execute_stage_a_scientific_loop(
        authority,
        backend=first_preflight.backend,
        tensor_cache=tensor_cache,
        seed_runner=fixed_runner,
    )
    if len(preflights) != len(authority.checkpoints):
        raise PostScienceFailure("RSTA model campaign preflight count differs")
    return StageAModelCampaign(
        completed=completed,
        preflights=tuple(preflights),
        determinism=determinism,
    )


def _role_panel_payload(panel: RstaRolePanel) -> dict[str, object]:
    def batches(values):
        return [
            {
                "index": batch.index,
                "rows": [
                    {"example_id": row.example_id, "label": row.label, "role": row.role}
                    for row in batch.rows
                ],
            }
            for batch in values
        ]

    return {
        "primary_class_order": panel.primary_class_order,
        "alternate_class_order": panel.alternate_class_order,
        "ranked_ids_by_label": panel.ranked_ids_by_label,
        "support_ids_by_label": panel.support_ids_by_label,
        "primary_batches": batches(panel.primary_batches),
        "alternate_batches": batches(panel.alternate_batches),
        "receiver_count": len(panel.receivers),
        "receivers": [
            {
                "example_id": row.example_id,
                "label": row.label,
                "support_ids": row.support_ids,
                "primary_batch": row.primary_batch,
                "primary_row": row.primary_row,
                "alternate_batch": row.alternate_batch,
                "alternate_row": row.alternate_row,
                "primary_peer_id": row.primary_peer_id,
                "alternate_peer_id": row.alternate_peer_id,
                "primary_foreign_labels": row.primary_foreign_labels,
                "alternate_foreign_labels": row.alternate_foreign_labels,
            }
            for row in panel.receivers
        ],
    }


def _validate_execution_audit(audit: StageAExecutionAudit) -> None:
    if (
        type(audit) is not StageAExecutionAudit
        or type(audit.parameter_names) is not tuple
        or not audit.parameter_names
        or audit.parameter_names != tuple(sorted(audit.parameter_names))
        or len(set(audit.parameter_names)) != len(audit.parameter_names)
        or any(
            type(name) is not str
            or not (name.startswith("tower.") or name.startswith("projection."))
            for name in audit.parameter_names
        )
        or not any(name.startswith("tower.") for name in audit.parameter_names)
        or not any(name.startswith("projection.") for name in audit.parameter_names)
        or type(audit.parameter_numels) is not tuple
        or len(audit.parameter_numels) != len(audit.parameter_names)
        or any(type(value) is not int or value <= 0 for value in audit.parameter_numels)
        or type(audit.checkpointing_max_relative_disagreement) is not float
        or not math.isfinite(audit.checkpointing_max_relative_disagreement)
        or not 0.0 <= audit.checkpointing_max_relative_disagreement <= 1.0e-5
        or type(audit.fixture_sha256) is not str
        or len(audit.fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in audit.fixture_sha256)
        or audit.module_training is not True
        or audit.gradient_checkpointing_enabled is not False
        or audit.torch_compile_enabled is not False
        or audit.attention_implementation != "eager"
        or audit.autocast_device_type != "cuda"
        or audit.autocast_dtype != "bfloat16"
        or audit.autocast_enabled is not True
        or audit.cublas_workspace_config != ":4096:8"
        or audit.deterministic_algorithms_enabled is not True
        or audit.deterministic_algorithms_warn_only is not False
        or audit.cudnn_benchmark is not False
        or audit.cuda_matmul_allow_tf32 is not False
        or audit.cudnn_allow_tf32 is not False
        or type(audit.elapsed_ns) is not int
        or audit.elapsed_ns <= 0
        or type(audit.peak_rss_bytes) is not int
        or audit.peak_rss_bytes <= 0
        or type(audit.peak_cuda_bytes) is not int
        or audit.peak_cuda_bytes <= 0
        or type(audit.memory_psi_growth_ppm) is not int
        or audit.memory_psi_growth_ppm != 0
        or type(audit.swap_growth_bytes) is not int
        or audit.swap_growth_bytes != 0
    ):
        raise ValueError("RSTA execution audit differs from authority")


def build_stage_a_execution_audit(
    campaign: StageAModelCampaign,
    *,
    elapsed_ns: int,
    peak_rss_bytes: int,
    peak_cuda_bytes: int,
    memory_psi_growth_ppm: int,
    swap_growth_bytes: int,
) -> StageAExecutionAudit:
    """Derive policy evidence from the completed campaign plus measured resources."""

    if (
        type(campaign) is not StageAModelCampaign
        or not campaign.preflights
        or not campaign.completed.seeds
    ):
        raise ValueError("RSTA model campaign audit authority differs")
    preflight = campaign.preflights[0]
    execution = campaign.completed.seeds[0]
    determinism = campaign.determinism
    audit = StageAExecutionAudit(
        parameter_names=preflight.parameter_names,
        parameter_numels=preflight.parameter_numels,
        checkpointing_max_relative_disagreement=(preflight.checkpointing_max_relative_disagreement),
        fixture_sha256=preflight.fixture_sha256,
        module_training=execution.module_training,
        gradient_checkpointing_enabled=execution.gradient_checkpointing_enabled,
        torch_compile_enabled=execution.torch_compile_enabled,
        attention_implementation=execution.attention_implementation,
        autocast_device_type=execution.autocast_device_type,
        autocast_dtype=execution.autocast_dtype,
        autocast_enabled=execution.autocast_enabled,
        cublas_workspace_config=determinism.cublas_workspace_config,
        deterministic_algorithms_enabled=(determinism.deterministic_algorithms_enabled),
        deterministic_algorithms_warn_only=(determinism.deterministic_algorithms_warn_only),
        cudnn_benchmark=determinism.cudnn_benchmark,
        cuda_matmul_allow_tf32=determinism.cuda_matmul_allow_tf32,
        cudnn_allow_tf32=determinism.cudnn_allow_tf32,
        elapsed_ns=elapsed_ns,
        peak_rss_bytes=peak_rss_bytes,
        peak_cuda_bytes=peak_cuda_bytes,
        memory_psi_growth_ppm=memory_psi_growth_ppm,
        swap_growth_bytes=swap_growth_bytes,
    )
    _validate_execution_audit(audit)
    return audit


def stage_a_scientific_result_bytes(
    campaign: StageAModelCampaign,
    audit: StageAExecutionAudit,
) -> bytes:
    """Bind complete scientific, identity, determinism, and resource evidence."""

    from sfora.siglip_rsta_stage_a import (
        RstaJvpBackendEvidence,
        rsta_control_binding_bytes,
        rsta_stage_a_result_bytes,
        select_rsta_roles,
    )

    if type(campaign) is not StageAModelCampaign:
        raise ValueError("RSTA model campaign has the wrong concrete type")
    completed = campaign.completed
    if type(completed) is not CompletedStageAScience:
        raise ValueError("RSTA completed science has the wrong concrete type")
    _validate_execution_audit(audit)
    if campaign.determinism != StageADeterminismEvidence(
        cublas_workspace_config=audit.cublas_workspace_config,
        deterministic_algorithms_enabled=audit.deterministic_algorithms_enabled,
        deterministic_algorithms_warn_only=audit.deterministic_algorithms_warn_only,
        cudnn_benchmark=audit.cudnn_benchmark,
        cuda_matmul_allow_tf32=audit.cuda_matmul_allow_tf32,
        cudnn_allow_tf32=audit.cudnn_allow_tf32,
    ):
        raise ValueError("RSTA model campaign determinism differs from audit")
    if (
        type(campaign.preflights) is not tuple
        or len(campaign.preflights) != len(completed.authority.checkpoints)
        or any(type(value) is not StageAModelPreflight for value in campaign.preflights)
        or any(
            value.backend != completed.backend
            or value.parameter_names != audit.parameter_names
            or value.parameter_numels != audit.parameter_numels
            or value.fixture_sha256 != audit.fixture_sha256
            or value.checkpointing_max_relative_disagreement
            != audit.checkpointing_max_relative_disagreement
            for value in campaign.preflights
        )
    ):
        raise ValueError("RSTA model campaign preflight differs from audit")
    recomputed_panel = select_rsta_roles(
        tuple(zip(completed.authority.example_ids, completed.authority.labels, strict=True))
    )
    if recomputed_panel != completed.panel:
        raise ValueError("RSTA completed role panel differs from authority")
    required_ids = stage_a_tensor_example_ids(completed.panel)
    if (
        not isinstance(completed.tensor_sha256_by_id, Mapping)
        or set(completed.tensor_sha256_by_id) != set(required_ids)
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in completed.tensor_sha256_by_id.values()
        )
    ):
        raise ValueError("RSTA completed tensor authority differs")
    if rsta_stage_a_result_bytes(completed.aggregate) != completed.aggregate_bytes:
        raise ValueError("RSTA completed aggregate bytes differ")
    if (
        type(completed.seeds) is not tuple
        or tuple(row.seed for row in completed.authority.checkpoints) != (17, 29, 43)
        or len(completed.seeds) != 3
        or any(type(item) is not StageASeedExecution for item in completed.seeds)
        or any(
            type(item.first_receiver_first_sha256) is not str
            or len(item.first_receiver_first_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in item.first_receiver_first_sha256
            )
            or item.first_receiver_repeat_sha256 != item.first_receiver_first_sha256
            for item in completed.seeds
        )
    ):
        raise ValueError("RSTA completed seed authority differs")
    expected_receivers = tuple(
        (receiver.label, receiver.example_id) for receiver in recomputed_panel.receivers
    )
    first_receiver = recomputed_panel.receivers[0]
    expected_batch_replays = (
        len(recomputed_panel.primary_batches) + len(recomputed_panel.alternate_batches) + 2
    )
    expected_receiver_actions = (
        2 * len(recomputed_panel.receivers)
        + sum(
            row.role == "receiver"
            for row in recomputed_panel.primary_batches[first_receiver.primary_batch].rows
        )
        + sum(
            row.role == "receiver"
            for row in recomputed_panel.alternate_batches[first_receiver.alternate_batch].rows
        )
    )
    expected_seeds = tuple(checkpoint.seed for checkpoint in completed.authority.checkpoints)
    if any(
        seed_execution.parameter_names != audit.parameter_names
        or seed_execution.parameter_numels != audit.parameter_numels
        or seed_execution.logical_batch_replays != expected_batch_replays
        or seed_execution.receiver_actions != expected_receiver_actions
        or seed_execution.autocast_device_type != audit.autocast_device_type
        or seed_execution.autocast_dtype != audit.autocast_dtype
        or seed_execution.autocast_enabled is not audit.autocast_enabled
        or seed_execution.support_replays != 2
        or seed_execution.module_training is not audit.module_training
        or seed_execution.gradient_checkpointing_enabled is not audit.gradient_checkpointing_enabled
        or seed_execution.torch_compile_enabled is not audit.torch_compile_enabled
        or seed_execution.attention_implementation != audit.attention_implementation
        or tuple((row.label, row.receiver_id) for row in seed_execution.receiver_evidence)
        != expected_receivers
        or any(row.seed != seed for row in seed_execution.receiver_evidence)
        for seed, seed_execution in zip(expected_seeds, completed.seeds, strict=True)
    ):
        raise ValueError("RSTA completed seed parameter or role authority differs")
    flattened_receiver_evidence = tuple(
        row for seed_execution in completed.seeds for row in seed_execution.receiver_evidence
    )
    if flattened_receiver_evidence != completed.aggregate.receiver_evidence:
        raise ValueError("RSTA completed aggregate receiver evidence differs")
    logical_batch_replays = sum(item.logical_batch_replays for item in completed.seeds)
    receiver_actions = sum(item.receiver_actions for item in completed.seeds)
    support_replays = sum(item.support_replays for item in completed.seeds)

    backend = completed.backend
    if type(backend) is not RstaJvpBackendEvidence:
        raise ValueError("RSTA completed backend authority differs")
    if backend.backend == "forward-mode":
        valid_backend = (
            backend.comparison_available is True
            and type(backend.maximum_relative_disagreement) is float
            and math.isfinite(backend.maximum_relative_disagreement)
            and 0.0 <= backend.maximum_relative_disagreement <= 1.0e-5
            and backend.forward_error is None
        )
    else:
        valid_backend = (
            backend.backend == "double-backward"
            and backend.comparison_available is False
            and backend.maximum_relative_disagreement == 0.0
            and type(backend.forward_error) is str
            and bool(backend.forward_error)
        )
    if not valid_backend:
        raise ValueError("RSTA completed backend authority differs")

    binding_bytes = rsta_control_binding_bytes(completed.authority.binding)
    image_rows = [
        {
            "example_id": example_id,
            "label": label,
            "basename": path.name,
        }
        for example_id, label, path in zip(
            completed.authority.example_ids,
            completed.authority.labels,
            completed.authority.image_paths,
            strict=True,
        )
    ]
    image_namespace_sha256 = hashlib.sha256(_canonical_json({"images": image_rows})).hexdigest()
    panel_payload = _role_panel_payload(completed.panel)
    panel_payload["sha256"] = hashlib.sha256(_canonical_json(panel_payload)).hexdigest()
    parameter_rows = [
        {"name": name, "numel": numel}
        for name, numel in zip(audit.parameter_names, audit.parameter_numels, strict=True)
    ]
    parameter_sha256 = hashlib.sha256(_canonical_json({"parameters": parameter_rows})).hexdigest()
    payload = {
        "schema": "siglip-rsta-stage-a-scientific-result-v1",
        "claim_eligible": False,
        "authority": {
            "control_binding": json.loads(binding_bytes),
            "control_binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "optimization_manifest_sha256": (
                completed.authority.binding.optimization_manifest_sha256
            ),
            "image_namespace_sha256": image_namespace_sha256,
            "selected_microbatch_size": completed.authority.binding.selected_microbatch_size,
        },
        "role_panel": panel_payload,
        "tensor_sha256_by_id": [
            {"example_id": example_id, "sha256": completed.tensor_sha256_by_id[example_id]}
            for example_id in required_ids
        ],
        "parameter_authority": {
            "parameters": parameter_rows,
            "parameter_count": len(parameter_rows),
            "parameter_numel": sum(audit.parameter_numels),
            "sha256": parameter_sha256,
        },
        "backend_preflight": {
            "backend": backend.backend,
            "comparison_available": backend.comparison_available,
            "maximum_relative_disagreement": backend.maximum_relative_disagreement,
            "forward_error": backend.forward_error,
            "checkpointing_max_relative_disagreement": (
                audit.checkpointing_max_relative_disagreement
            ),
            "fixture_sha256": audit.fixture_sha256,
        },
        "execution": {
            "module_training": audit.module_training,
            "gradient_checkpointing_enabled": audit.gradient_checkpointing_enabled,
            "torch_compile_enabled": audit.torch_compile_enabled,
            "attention_implementation": audit.attention_implementation,
            "autocast_device_type": audit.autocast_device_type,
            "autocast_dtype": audit.autocast_dtype,
            "autocast_enabled": audit.autocast_enabled,
            "cublas_workspace_config": audit.cublas_workspace_config,
            "deterministic_algorithms_enabled": audit.deterministic_algorithms_enabled,
            "deterministic_algorithms_warn_only": (audit.deterministic_algorithms_warn_only),
            "cudnn_benchmark": audit.cudnn_benchmark,
            "cuda_matmul_allow_tf32": audit.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": audit.cudnn_allow_tf32,
            "logical_batch_replays": logical_batch_replays,
            "receiver_actions": receiver_actions,
            "receiver_vjps": receiver_actions,
            "receiver_jvps": 2 * receiver_actions,
            "support_forward_replays": support_replays,
            "elapsed_ns": audit.elapsed_ns,
            "peak_rss_bytes": audit.peak_rss_bytes,
            "peak_cuda_bytes": audit.peak_cuda_bytes,
            "memory_psi_growth_ppm": audit.memory_psi_growth_ppm,
            "swap_growth_bytes": audit.swap_growth_bytes,
        },
        "repeatability": [
            {
                "seed": checkpoint.seed,
                "first_receiver_first_sha256": execution.first_receiver_first_sha256,
                "first_receiver_repeat_sha256": execution.first_receiver_repeat_sha256,
            }
            for checkpoint, execution in zip(
                completed.authority.checkpoints, completed.seeds, strict=True
            )
        ],
        "result": json.loads(completed.aggregate_bytes),
    }
    return _canonical_json(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments; scientific execution is added behind this boundary."""

    parse_stage_a_args(argv)
    raise RuntimeError("SigLIP RSTA Stage-A scientific runner is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
