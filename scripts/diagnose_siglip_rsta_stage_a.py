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


def execute_stage_a_scientific_loop(
    authority: LoadedStageAAuthority,
    *,
    backend: RstaJvpBackendEvidence,
    tensor_sha256_by_id: Mapping[str, str],
    seed_runner: Callable[
        [
            LoadedStageACheckpoint,
            RstaRolePanel,
            RstaControlBinding,
            Mapping[str, Path],
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
        not isinstance(tensor_sha256_by_id, Mapping)
        or set(tensor_sha256_by_id) != set(required_tensor_ids)
        or any(
            type(example_id) is not str
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for example_id, digest in tensor_sha256_by_id.items()
        )
    ):
        raise PreScienceInvalid("RSTA tensor digest authority differs")
    image_paths = MappingProxyType(
        dict(zip(authority.example_ids, authority.image_paths, strict=True))
    )
    expected_receivers = tuple((row.label, row.example_id) for row in panel.receivers)
    executions: list[StageASeedExecution] = []
    all_rows: list[RstaReceiverEvidence] = []
    for checkpoint in authority.checkpoints:
        try:
            execution = seed_runner(
                checkpoint,
                panel,
                authority.binding,
                image_paths,
                backend,
            )
        except Exception as error:
            raise PostScienceFailure("RSTA seed execution failed") from error
        if type(execution) is not StageASeedExecution:
            raise PostScienceFailure("RSTA seed execution has the wrong concrete type")
        rows = execution.receiver_evidence
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
        tensor_sha256_by_id=MappingProxyType(dict(tensor_sha256_by_id)),
        backend=backend,
        seeds=tuple(executions),
        aggregate=aggregate,
        aggregate_bytes=aggregate_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments; scientific execution is added behind this boundary."""

    parse_stage_a_args(argv)
    raise RuntimeError("SigLIP RSTA Stage-A scientific runner is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
