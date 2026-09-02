#!/usr/bin/env python3
"""Authenticate three endpoints and publish outcome-free cross-seed model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch

from scripts.diagnose_weight_space_transfer import (
    load_seed_endpoint_authority,
    load_transfer_checkpoint,
    reconstruct_initial_model,
)
from scripts.run_siglip_proxy_control import load_siglip_control_components
from sfora.cross_seed_denoising import write_tensor_artifact
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.token_set_screen import F1_TRAIN_CLASSES
from sfora.weight_space_transfer import model_state_sha256

_SEEDS = (17, 29, 43)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _lower_hex(value: str, length: int, role: str) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(f"{role} must be lowercase hexadecimal")
    return value


def _sha256(value: str) -> str:
    return _lower_hex(value, 64, "digest")


def _commit(value: str) -> str:
    return _lower_hex(value, 40, "commit")


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("byte length must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("byte length must be positive")
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the exact local checkpoint-preparation capability."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--source-tree-digest", required=True, type=_sha256)
    parser.add_argument("--seed-result", required=True, action="append", type=_absolute_path)
    parser.add_argument("--seed-result-sha256", required=True, action="append", type=_sha256)
    parser.add_argument("--seed-result-bytes", required=True, action="append", type=_positive)
    parser.add_argument("--checkpoint", required=True, action="append", type=_absolute_path)
    parser.add_argument("--checkpoint-sha256", required=True, action="append", type=_sha256)
    parser.add_argument("--checkpoint-bytes", required=True, action="append", type=_positive)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument(
        "--execute-cross-seed-preparation", required=True, action="store_true"
    )
    arguments = parser.parse_args(argv)
    cardinalities = (
        len(arguments.seed_result),
        len(arguments.seed_result_sha256),
        len(arguments.seed_result_bytes),
        len(arguments.checkpoint),
        len(arguments.checkpoint_sha256),
        len(arguments.checkpoint_bytes),
    )
    if cardinalities != (3, 3, 3, 3, 3, 3):
        parser.error("exactly three seed results and checkpoints are required")
    return arguments


def _validate_bindings(value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ValueError("preparation bindings differ")
    typed = cast(dict[object, object], value)
    if any(type(key) is not str or type(item) is not str for key, item in typed.items()):
        raise ValueError("preparation bindings differ")
    return {cast(str, key): cast(str, item) for key, item in sorted(typed.items())}


def _normalize_states(
    value: object,
    *,
    role: str,
) -> dict[int, OrderedDict[str, torch.Tensor]]:
    if type(value) is not dict or set(value) != set(_SEEDS):
        raise ValueError(f"{role} states must contain exactly the registered seeds")
    result: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for seed in _SEEDS:
        state = cast(dict[int, object], value)[seed]
        if not isinstance(state, (OrderedDict, Mapping)) or not state:
            raise ValueError(f"{role} state schema differs")
        copied: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, tensor in sorted(cast(Mapping[object, object], state).items()):
            if type(name) is not str or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{role} state schema differs")
            if name not in ("projection.weight", "proxies") and not name.startswith("tower."):
                raise ValueError(f"{role} state names differ")
            if tensor.layout != torch.strided or (
                tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"{role} tensor authority differs")
            copied[name] = tensor.detach().cpu().contiguous().clone()
        if "projection.weight" not in copied or "proxies" not in copied:
            raise ValueError(f"{role} state lacks the registered head")
        if not any(name.startswith("tower.") for name in copied):
            raise ValueError(f"{role} state lacks the registered tower")
        result[seed] = copied
    return result


def _partition(
    state: OrderedDict[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    tower = OrderedDict(
        (name, tensor.clone()) for name, tensor in state.items() if name.startswith("tower.")
    )
    head = OrderedDict(
        (name, state[name].clone()) for name in ("projection.weight", "proxies")
    )
    if set(tower) | set(head) != set(state):
        raise ValueError("state names differ from tower/head authority")
    return tower, head


def _equal_state(
    left: OrderedDict[str, torch.Tensor], right: OrderedDict[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and torch.equal(left[name], right[name])
        for name in left
    )


def _artifact_row(directory: str, raw: bytes) -> dict[str, object]:
    value = json.loads(raw)
    return {
        "directory": directory,
        "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "state_sha256": value["state_sha256"],
    }


def prepare_cross_seed_artifacts(
    *,
    initial_states: object,
    trained_states: object,
    bindings: object,
    output: Path,
) -> bytes:
    """Publish one atomic, outcome-free three-seed construction namespace."""

    if not isinstance(output, Path):
        raise TypeError("preparation output path differs")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    normalized_bindings = _validate_bindings(bindings)
    initial = _normalize_states(initial_states, role="initial")
    trained = _normalize_states(trained_states, role="trained")
    initial_towers: dict[int, OrderedDict[str, torch.Tensor]] = {}
    trained_towers: dict[int, OrderedDict[str, torch.Tensor]] = {}
    trained_heads: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for seed in _SEEDS:
        if tuple(initial[seed]) != tuple(trained[seed]):
            raise ValueError("initial and trained state names differ")
        initial_towers[seed], _ = _partition(initial[seed])
        trained_towers[seed], trained_heads[seed] = _partition(trained[seed])
    if any(not _equal_state(initial_towers[17], initial_towers[seed]) for seed in (29, 43)):
        raise ValueError("initial tower differs across seeds")

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.mkdir()
    try:
        initial_directory = "initial-tower"
        initial_raw = write_tensor_artifact(
            partial / initial_directory,
            initial_towers[17],
            role="initial-tower",
            bindings=normalized_bindings,
        )
        seed_rows: list[dict[str, object]] = []
        for seed in _SEEDS:
            tower_directory = f"seed-{seed:03d}-tower"
            head_directory = f"seed-{seed:03d}-head"
            seed_bindings = {**normalized_bindings, "seed": str(seed)}
            tower_raw = write_tensor_artifact(
                partial / tower_directory,
                trained_towers[seed],
                role="trained-tower",
                bindings=seed_bindings,
            )
            head_raw = write_tensor_artifact(
                partial / head_directory,
                trained_heads[seed],
                role="trained-head",
                bindings=seed_bindings,
            )
            tower_row = _artifact_row(tower_directory, tower_raw)
            head_row = _artifact_row(head_directory, head_raw)
            seed_rows.append(
                {
                    "head_directory": head_directory,
                    "head_manifest_bytes": head_row["manifest_bytes"],
                    "head_manifest_sha256": head_row["manifest_sha256"],
                    "head_state_sha256": head_row["state_sha256"],
                    "initial_state_sha256": model_state_sha256(initial[seed]),
                    "seed": seed,
                    "tower_directory": tower_directory,
                    "tower_manifest_bytes": tower_row["manifest_bytes"],
                    "tower_manifest_sha256": tower_row["manifest_sha256"],
                    "tower_state_sha256": tower_row["state_sha256"],
                }
            )
        manifest = {
            "bindings": normalized_bindings,
            "claim_eligible": False,
            "initial_tower": _artifact_row(initial_directory, initial_raw),
            "schema": "sfora-cross-seed-prepared-inputs-v1",
            "seeds": seed_rows,
        }
        raw = _canonical(manifest)
        with (partial / "manifest.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.rename(output)
        return raw
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise


def _reconstruct_initial_state(seed: int, expected_sha256: str) -> OrderedDict[str, torch.Tensor]:
    config = SiglipProxyControlConfig()

    def tower_loader() -> torch.nn.Module:
        tower, _processor = load_siglip_control_components(config=config)
        return tower

    def model_builder(tower: torch.nn.Module) -> torch.nn.Module:
        return PooledProxyAnchorModel(
            tower=tower,
            input_dimensions=config.input_dimensions,
            embedding_dimensions=config.embedding_dimensions,
            class_count=len(F1_TRAIN_CLASSES),
            projection_initialization=config.projection_initialization,
            proxy_initialization=config.proxy_initialization,
        )

    reconstructed = reconstruct_initial_model(
        seed=seed,
        expected_sha256=expected_sha256,
        device=torch.device("cpu"),
        tower_loader=tower_loader,
        model_builder=model_builder,
    )
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in reconstructed.model.state_dict().items()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate three complete endpoints, then publish only model tensors."""

    arguments = parse_arguments(argv)
    authorities = []
    checkpoints = []
    for index, seed in enumerate(_SEEDS):
        authority = load_seed_endpoint_authority(
            path=arguments.seed_result[index],
            expected_sha256=arguments.seed_result_sha256[index],
            expected_bytes=arguments.seed_result_bytes[index],
            expected_seed=seed,
        )
        if (
            authority.source_revision != arguments.source_commit
            or authority.source_tree_digest != arguments.source_tree_digest
            or authority.checkpoint_sha256 != arguments.checkpoint_sha256[index]
            or authority.checkpoint_bytes != arguments.checkpoint_bytes[index]
        ):
            raise ValueError("endpoint source or checkpoint binding differs")
        checkpoint = load_transfer_checkpoint(
            path=arguments.checkpoint[index],
            expected_sha256=arguments.checkpoint_sha256[index],
            expected_bytes=arguments.checkpoint_bytes[index],
            expected_seed=seed,
            expected_config_sha256=authority.config_sha256,
            expected_run_authority_sha256=authority.run_authority_sha256,
        )
        if checkpoint.initial_snapshot_sha256 != authority.initial_state_sha256:
            raise ValueError("checkpoint initial binding differs")
        authorities.append(authority)
        checkpoints.append(checkpoint)
    if len({authority.manifest_sha256 for authority in authorities}) != 1:
        raise ValueError("endpoint dataset manifest differs")
    if (
        len({authority.evaluation_batch_size for authority in authorities}) != 1
        or len({authority.query_block for authority in authorities}) != 1
    ):
        raise ValueError("endpoint evaluation protocol differs")
    bindings = {
        "dataset_manifest_sha256": authorities[0].manifest_sha256,
        "evaluation_batch_size": str(authorities[0].evaluation_batch_size),
        "query_block": str(authorities[0].query_block),
        "source_commit": arguments.source_commit,
        "source_tree_digest": arguments.source_tree_digest,
        **{
            f"seed_{seed}_checkpoint_sha256": arguments.checkpoint_sha256[index]
            for index, seed in enumerate(_SEEDS)
        },
        **{
            f"seed_{seed}_result_sha256": arguments.seed_result_sha256[index]
            for index, seed in enumerate(_SEEDS)
        },
    }
    raw = prepare_cross_seed_artifacts(
        initial_states={
            seed: _reconstruct_initial_state(seed, authority.initial_state_sha256)
            for seed, authority in zip(_SEEDS, authorities, strict=True)
        },
        trained_states={
            seed: OrderedDict(checkpoint.model_state)
            for seed, checkpoint in zip(_SEEDS, checkpoints, strict=True)
        },
        bindings=bindings,
        output=arguments.output,
    )
    sys.stdout.buffer.write(
        _canonical(
            {
                "artifact": str(arguments.output / "manifest.json"),
                "artifact_bytes": len(raw),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "claim_eligible": False,
                "schema": "sfora-cross-seed-preparation-receipt-v1",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
