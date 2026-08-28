#!/usr/bin/env python3
"""Build and authenticate the immutable UniCOM FEPF campaign config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
INSHOP_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
PARENT_TRAINER_COMMIT = "70c760e57e6c27dec1473eecd4765e0a8cd4cf6b"
PARENT_TRAINER_PATH = "scripts/train_unicom_inshop.py"
PARENT_TRAINER_SHA256 = "6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170"
RUNTIME_ORDER = ("current", "composed", "composed", "current") * 2
CONFIRMATION_PAIRS = (
    (7, 20_260_828),
    (8, 271_828),
    (9, 314_159),
    (10, 1_618_033),
    (11, 57_721),
)
REGISTERED_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/sfora/unicom_fepf.py",
    "src/sfora/unicom_retrieval_audit.py",
    "scripts/train_unicom_inshop.py",
    "scripts/profile_unicom_training_step.py",
    "scripts/evaluate_unicom_fepf.py",
    "scripts/build_unicom_fepf_run_config.py",
    "scripts/run_unicom_fepf_campaign.py",
    "scripts/run_unicom_fepf_cuda_canary.py",
)
CONFIG_KEYS = (
    "schema",
    "source_commit",
    "source_files",
    "source",
    "handoff",
    "model",
    "inputs",
    "parent_trainer_commit",
    "parent_trainer_path",
    "parent_trainer_sha256",
    "live_trainer_sha256",
    "profiler_sha256",
    "fepf_inference_structure",
    "checkout_root_template",
    "artifact_root",
    "runtime_order",
    "exploratory",
    "confirmation_pairs",
    "thresholds",
    "artifact_inventory",
    "artifact_budget_inputs",
    "artifact_budget_bytes",
    "artifact_budget_inodes",
    "cuda_canary_authority",
    "cuda_canary_command",
    "cuda_canary_receipt",
    "commands",
)
_LOWER_COMMIT = re.compile(r"[0-9a-f]{40}").fullmatch

# Frozen conservative sizing authorities. The eightfold multiplier covers the
# complete checkpoint payload rather than pretending state_dict tensor bytes are
# the serialized checkpoint size.
RAW_BACKBONE_STATE_BYTES = 1_218_000_000
CLASSIFIER_STATE_BYTES = 3_997 * 768 * 4
QUALITY_CHECKPOINTS = 13 * 4
ATOMIC_COPY_FACTOR = 2


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True,
        text=not binary,
    )
    return result.stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_absolute_plain(path: Path, name: str) -> Path:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(f"{name} must be a normalized absolute path")
    return path


def _validate_template(template: str) -> None:
    if type(template) is not str or template.count("{config_commit}") != 1:
        raise ValueError("checkout root template differs")
    residual = template.replace("{config_commit}", "")
    if "{" in residual or "}" in residual:
        raise ValueError("checkout root template differs")
    expanded = Path(template.replace("{config_commit}", "0" * 40))
    _require_absolute_plain(expanded, "checkout root template")


def _paths_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _source_inventory(repo: Path, commit: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in REGISTERED_SOURCE_PATHS:
        payload = _git(repo, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(payload, bytes)
        rows.append({"path": relative, "sha256": _sha256(payload), "bytes": len(payload)})
    return rows


def _partition_inventory(path: Path) -> dict[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered partition differs")
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("registered partition differs")
    queries = [row for row in rows if row[2] == "query"]
    galleries = [row for row in rows if row[2] == "gallery"]
    gallery_counts: dict[str, int] = {}
    for _image, label, _split in galleries:
        gallery_counts[label] = gallery_counts.get(label, 0) + 1
    relevant = [gallery_counts.get(label, 0) for _image, label, _split in queries]
    if not queries or not galleries or not relevant or min(relevant) <= 0:
        raise ValueError("registered partition relevance differs")
    return {
        "query_rows": len(queries), "gallery_rows": len(galleries),
        "maximum_relevant_count": max(relevant),
        "maximum_path_bytes": max(len(row[0].encode()) for row in rows),
    }


def _budget(
    partition: dict[str, int]
) -> tuple[dict[str, int], int, int, list[dict[str, object]]]:
    required = ("query_rows", "gallery_rows", "maximum_relevant_count", "maximum_path_bytes")
    if tuple(partition) != required or any(type(partition[key]) is not int or partition[key] <= 0
                                            for key in required):
        raise ValueError("artifact partition inventory differs")
    checkpoint_bound = 8 * (RAW_BACKBONE_STATE_BYTES + CLASSIFIER_STATE_BYTES) + 64 * 1024**2
    inventory = _artifact_inventory_rows(partition, checkpoint_bound)
    subtotal = sum(row["count"] * row["bytes_each"] for row in inventory)
    planned_files = sum(row["inodes"] for row in inventory)
    inputs = {
        "quality_checkpoints": QUALITY_CHECKPOINTS,
        "raw_backbone_state_bytes": RAW_BACKBONE_STATE_BYTES,
        "classifier_state_bytes": CLASSIFIER_STATE_BYTES,
        **partition,
        "descriptor_dimension": 768,
        "atomic_copy_factor": ATOMIC_COPY_FACTOR,
        "planned_file_inodes": planned_files,
    }
    return (
        inputs,
        math.ceil(1.25 * subtotal),
        math.ceil(1.25 * planned_files),
        inventory,
    )


def _artifact_inventory_rows(
    partition: dict[str, int], checkpoint_bound: int
) -> list[dict[str, object]]:
    query_descriptor = partition["query_rows"] * 768 * 4
    gallery_descriptor = partition["gallery_rows"] * 768 * 4
    ranked_evidence = (
        partition["query_rows"] * partition["maximum_relevant_count"]
        * (2 * partition["maximum_path_bytes"] + 128)
    )
    rows = (
        ("quality_checkpoint", 104, checkpoint_bound, 104),
        ("query_descriptor", 104, query_descriptor, 104),
        ("gallery_descriptor", 104, gallery_descriptor, 104),
        ("ranked_prefix_evidence", 104, ranked_evidence, 104),
        ("training_terminal_chain", 104, 256 * 1024, 104),
        ("runtime_profile", 16, 2 * 1024**2, 16),
        ("quality_profile", 48, 2 * 1024**2, 48),
        ("evaluation_sources_and_result", 12, 2 * 1024**2, 12),
        ("canary_and_controller", 4, 2 * 1024**2, 4),
        ("stage_directories", 48, 0, 48),
    )
    return [
        {"role": role, "count": count, "bytes_each": bytes_each, "inodes": inodes}
        for role, count, bytes_each, inodes in rows
    ]


def registered_artifact_inventory(config: object) -> tuple[dict[str, object], ...]:
    value = _strict_shape(config)
    _validate_config_values(value)
    return tuple(dict(row) for row in value["artifact_inventory"])


def _inputs() -> dict[str, str]:
    return {
        "unicom_checkout": "/home/riomus/unicom-d71992e",
        "checkpoint": "/home/riomus/.cache/unicom/FP16-ViT-L-14-336px.pt",
        "dataset_root": "/home/riomus/datasets/inshop_official_standard",
        "partition": "/home/riomus/datasets/inshop_official_standard/Eval/list_eval_partition.txt",
        "runtime_checkpoint": "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/epoch-0016.pt",
        "runtime_run_receipt": (
            "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/run-receipt.json"
        ),
    }


def _commands() -> dict[str, object]:
    python = ".venv/bin/python"
    inputs = _inputs()
    return {
        "runtime": [
            [python, "-I", "-B", "scripts/profile_unicom_training_step.py",
             "--profile-kind", "runtime", "--runtime-mode", mode,
             "--run-checkpoint", inputs["runtime_checkpoint"], "--run-receipt",
             inputs["runtime_run_receipt"], "--config",
             "docs/unicom_fepf_run_config.json", "--unicom-checkout",
             inputs["unicom_checkout"], "--initial-checkpoint", inputs["checkpoint"],
             "--dataset-root", inputs["dataset_root"], "--parent-trainer-source",
             f"{PARENT_TRAINER_COMMIT}:{PARENT_TRAINER_PATH}",
             "--output", "{output}"]
            for mode in RUNTIME_ORDER
        ],
        "train": [
            python, "-I", "-B", "scripts/train_unicom_inshop.py",
            "--unicom-checkout", inputs["unicom_checkout"],
            "--checkpoint", inputs["checkpoint"],
            "--dataset-root", inputs["dataset_root"],
            "--run-config", "docs/unicom_fepf_run_config.json",
        ],
        "profile_quality": [
            python, "-I", "-B", "scripts/profile_unicom_training_step.py",
            "--profile-kind", "quality", "--unicom-checkout", inputs["unicom_checkout"],
            "--initial-checkpoint", inputs["checkpoint"], "--dataset-root",
            inputs["dataset_root"], "--parent-trainer-source",
            f"{PARENT_TRAINER_COMMIT}:{PARENT_TRAINER_PATH}",
        ],
        "evaluate": [python, "-I", "-B", "scripts/evaluate_unicom_fepf.py"],
        "quality_profile_order": ["imprinted", "fepf_mean", "fepf_mean", "imprinted"],
    }


def _checkpoint_inference_structure(path: Path) -> dict[str, object]:
    import torch

    if path.is_symlink() or not path.is_file():
        raise ValueError("registered runtime checkpoint differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict or type(checkpoint.get("model")) is not dict:
        raise ValueError("registered runtime checkpoint differs")
    state = checkpoint["model"]
    ema = checkpoint.get("ema")
    parameter_names = (
        set(ema["backbone"])
        if type(ema) is dict and type(ema.get("backbone")) is dict
        else set(state)
    )
    rows = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("registered runtime checkpoint tensor differs")
        rows.append({
            "name": name,
            "kind": "parameter" if name in parameter_names else "buffer",
            "shape": list(value.shape), "dtype": str(value.dtype),
            "numel": value.numel(), "element_size": value.element_size(),
            "bytes": value.numel() * value.element_size(),
        })
    classifier = checkpoint.get("classifier")
    if not rows or not isinstance(classifier, torch.Tensor):
        raise ValueError("registered runtime checkpoint structure differs")
    return {
        "schema": "unicom-fepf-structure-v1", "tensors": rows,
        "classifier": {
            "shape": list(classifier.shape), "dtype": str(classifier.dtype),
            "numel": classifier.numel(), "element_size": classifier.element_size(),
            "bytes": classifier.numel() * classifier.element_size(),
        },
        "operations": [
            "official_forward", "full768_l2", "prefix512", "squared_euclidean"
        ],
    }


def build_run_config(
    *, repo: Path, checkout_root_template: str, artifact_root: Path,
    inference_structure: dict[str, object], partition_inventory: dict[str, int],
    cuda_canary_authority: dict[str, str],
) -> dict[str, object]:
    repo = _require_absolute_plain(repo.resolve(), "repository")
    artifact_root = _require_absolute_plain(artifact_root, "artifact root")
    _validate_template(checkout_root_template)
    expanded = Path(checkout_root_template.replace("{config_commit}", "0" * 40))
    if _paths_nested(expanded, artifact_root):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")
    source_commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    if _LOWER_COMMIT(source_commit) is None:
        raise ValueError("source commit differs")
    budget_inputs, budget_bytes, budget_inodes, artifact_inventory = _budget(
        partition_inventory
    )
    source_files = _source_inventory(repo, source_commit)
    source_hashes = {row["path"]: row["sha256"] for row in source_files}
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": source_commit,
        "source_files": source_files,
        "source": {"commit": source_commit},
        "handoff": {
            "config_parent": source_commit,
            "config_commit_paths": ["docs/unicom_fepf_run_config.json"],
            "execution_checkout": "config_commit_detached_clean",
        },
        "model": {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        },
        "inputs": _inputs(),
        "parent_trainer_commit": PARENT_TRAINER_COMMIT,
        "parent_trainer_path": PARENT_TRAINER_PATH,
        "parent_trainer_sha256": PARENT_TRAINER_SHA256,
        "live_trainer_sha256": source_hashes["scripts/train_unicom_inshop.py"],
        "profiler_sha256": source_hashes["scripts/profile_unicom_training_step.py"],
        "fepf_inference_structure": inference_structure,
        "checkout_root_template": checkout_root_template,
        "artifact_root": str(artifact_root),
        "runtime_order": list(RUNTIME_ORDER),
        "exploratory": {
            "training_seed": 0,
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "arms": ["imprinted", "fepf_mean", "fepf_random"],
            "evaluation_epochs": [4, 8, 12, 16],
        },
        "confirmation_pairs": [list(pair) for pair in CONFIRMATION_PAIRS],
        "thresholds": {
            "epoch4_map_delta_min": 0.003,
            "exploratory_map_delta_min": 0.010,
            "confirmation_map_delta_min": 0.010,
            "confirmation_r1_delta_min": 0.005,
            "compute_ratio_max": 1.02,
            "row_norm_rtol": 2e-6,
            "row_norm_atol": 2e-7,
        },
        "artifact_inventory": artifact_inventory,
        "artifact_budget_inputs": budget_inputs,
        "artifact_budget_bytes": budget_bytes,
        "artifact_budget_inodes": budget_inodes,
        "cuda_canary_authority": dict(cuda_canary_authority),
        "cuda_canary_command": [
            ".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json",
        ],
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
        "commands": _commands(),
    }


def _strict_shape(config: object) -> dict[str, object]:
    if type(config) is not dict or tuple(config) != CONFIG_KEYS:
        raise ValueError("FEPF config schema differs")
    return config


def _validate_config_values(config: dict[str, object]) -> None:
    structure = config.get("fepf_inference_structure")
    structure_valid = (
        type(structure) is dict
        and tuple(structure) == ("schema", "tensors", "classifier", "operations")
        and structure["schema"] == "unicom-fepf-structure-v1"
        and type(structure["tensors"]) is list
        and bool(structure["tensors"])
        and type(structure["classifier"]) is dict
        and structure["operations"]
        == ["official_forward", "full768_l2", "prefix512", "squared_euclidean"]
    )
    if (
        config["schema"] != "unicom-fepf-run-config-v1"
        or _LOWER_COMMIT(config["source_commit"]) is None
        or config["model"] != {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        }
        or config["inputs"] != _inputs()
        or config["source"] != {"commit": config["source_commit"]}
        or config["handoff"]
        != {
            "config_parent": config["source_commit"],
            "config_commit_paths": ["docs/unicom_fepf_run_config.json"],
            "execution_checkout": "config_commit_detached_clean",
        }
        or config["parent_trainer_commit"] != PARENT_TRAINER_COMMIT
        or config["parent_trainer_path"] != PARENT_TRAINER_PATH
        or config["parent_trainer_sha256"] != PARENT_TRAINER_SHA256
        or config["live_trainer_sha256"]
        != {row["path"]: row["sha256"] for row in config["source_files"]}.get(
            "scripts/train_unicom_inshop.py"
        )
        or config["profiler_sha256"]
        != {row["path"]: row["sha256"] for row in config["source_files"]}.get(
            "scripts/profile_unicom_training_step.py"
        )
        or not structure_valid
        or config["runtime_order"] != list(RUNTIME_ORDER)
        or config["confirmation_pairs"] != [list(pair) for pair in CONFIRMATION_PAIRS]
        or type(config["artifact_budget_bytes"]) is not int
        or config["artifact_budget_bytes"] <= 0
        or type(config["artifact_budget_inodes"]) is not int
        or config["artifact_budget_inodes"] <= 0
    ):
        raise ValueError("FEPF config protocol differs")
    threshold = config["thresholds"]
    if type(threshold) is not dict or threshold != {
        "epoch4_map_delta_min": 0.003,
        "exploratory_map_delta_min": 0.010,
        "confirmation_map_delta_min": 0.010,
        "confirmation_r1_delta_min": 0.005,
        "compute_ratio_max": 1.02,
        "row_norm_rtol": 2e-6,
        "row_norm_atol": 2e-7,
    }:
        raise ValueError("FEPF config threshold differs")
    if config["exploratory"] != {
        "training_seed": 0, "holdout_seed": 0, "holdout_fraction": 0.2,
        "arms": ["imprinted", "fepf_mean", "fepf_random"],
        "evaluation_epochs": [4, 8, 12, 16],
    } or config["commands"] != _commands():
        raise ValueError("FEPF config commands differ")
    partition = {
        key: config["artifact_budget_inputs"][key]
        for key in ("query_rows", "gallery_rows", "maximum_relevant_count", "maximum_path_bytes")
    }
    inputs, budget_bytes, budget_inodes, artifact_inventory = _budget(partition)
    if (
        config["artifact_inventory"] != artifact_inventory
        or config["artifact_budget_inputs"] != inputs
        or config["artifact_budget_bytes"] != budget_bytes
        or config["artifact_budget_inodes"] != budget_inodes
        or config["cuda_canary_command"]
        != [".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json"]
        or config["cuda_canary_receipt"] != "preflight/cuda_canary_v1.json"
        or type(config["cuda_canary_authority"]) is not dict
        or tuple(config["cuda_canary_authority"]) != ("device_uuid", "environment_sha256")
        or type(config["cuda_canary_authority"]["device_uuid"]) is not str
        or not config["cuda_canary_authority"]["device_uuid"].startswith("GPU-")
        or type(config["cuda_canary_authority"]["environment_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", config["cuda_canary_authority"]["environment_sha256"]
        ) is None
    ):
        raise ValueError("FEPF config budget differs")
    _validate_template(config["checkout_root_template"])
    artifact = _require_absolute_plain(Path(config["artifact_root"]), "artifact root")
    checkout = Path(config["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if _paths_nested(artifact, checkout):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")


def _require_clean(repo: Path) -> None:
    status_text = str(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    if status_text:
        raise ValueError("source checkout is not clean")


def validate_config_build(config: object, repo: Path) -> None:
    value = _strict_shape(config)
    _validate_config_values(value)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != value["source_commit"]:
        raise ValueError("source commit differs")
    if value["source_files"] != _source_inventory(repo, head):
        raise ValueError("source files differ")
    artifact = Path(value["artifact_root"])
    checkout = Path(value["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if os.path.lexists(artifact) or os.path.lexists(checkout):
        raise FileExistsError("campaign destination already exists")


def build_and_write(
    *, repo: Path, checkout_root_template: str, artifact_root: Path, output: Path,
    inference_structure: dict[str, object] | None = None,
    partition_inventory: dict[str, int] | None = None,
    cuda_canary_authority: dict[str, str],
) -> dict[str, object]:
    if os.path.lexists(output):
        raise FileExistsError(output)
    config = build_run_config(
        repo=repo, checkout_root_template=checkout_root_template,
        artifact_root=artifact_root,
        inference_structure=(
            inference_structure
            if inference_structure is not None
            else _checkpoint_inference_structure(Path(_inputs()["runtime_checkpoint"]))
        ),
        partition_inventory=(
            partition_inventory
            if partition_inventory is not None
            else _partition_inventory(Path(_inputs()["partition"]))
        ),
        cuda_canary_authority=cuda_canary_authority,
    )
    validate_config_build(config, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(config))
        handle.flush()
        os.fsync(handle.fileno())
    reloaded = json.loads(output.read_bytes())
    if canonical_json_bytes(reloaded) != output.read_bytes() or reloaded != config:
        raise RuntimeError("persisted FEPF config differs")
    return config


def validate_config_handoff(config_path: Path, repo: Path) -> dict[str, str]:
    raw = config_path.read_bytes()
    config = _strict_shape(json.loads(raw))
    if canonical_json_bytes(config) != raw:
        raise ValueError("FEPF config is not canonical")
    _validate_config_values(config)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo,
        check=False, capture_output=True, text=True,
    ).stdout.strip()
    if _LOWER_COMMIT(head) is None or symbolic:
        raise ValueError("handoff checkout must be detached")
    parent = str(_git(repo, "rev-parse", f"{head}^")).strip()
    relative = config_path.resolve().relative_to(repo).as_posix()
    changed = str(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head)
    ).splitlines()
    committed = _git(repo, "show", f"{head}:{relative}", binary=True)
    if (
        parent != config["source_commit"]
        or changed != [relative]
        or committed != raw
        or config["source_files"] != _source_inventory(repo, parent)
    ):
        raise ValueError("config handoff commit differs")
    checkout = config["checkout_root_template"].replace("{config_commit}", head)
    if "{" in checkout or "}" in checkout or os.path.lexists(checkout):
        raise ValueError("resolved checkout root differs")
    if os.path.lexists(config["artifact_root"]):
        raise FileExistsError(config["artifact_root"])
    return {"config_commit": head, "checkout_root": checkout}


def _available(stat_result: Any) -> tuple[int, int]:
    return stat_result.f_bavail * stat_result.f_frsize, stat_result.f_favail


def _plain_directory(path: Path, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{name} is not a plain directory")
    return info


def _require_capacity(
    path: Path, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any],
) -> None:
    available_bytes, available_inodes = _available(statvfs(path))
    if available_bytes < required_bytes or available_inodes < required_inodes:
        raise OSError("FEPF artifact capacity is insufficient")


def prepare_artifact_root(
    root: Path, *, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    root = _require_absolute_plain(root, "artifact root")
    if type(required_bytes) is not int or type(required_inodes) is not int:
        raise ValueError("artifact budget differs")
    if os.path.lexists(root):
        raise FileExistsError(root)
    parent = root.parent
    try:
        parent_info = _plain_directory(parent, "artifact parent")
    except FileNotFoundError as error:
        raise ValueError("artifact parent is absent") from error
    _require_capacity(parent, required_bytes, required_inodes, statvfs)
    root.mkdir(mode=0o700)
    root_info = _plain_directory(root, "artifact root")
    owned = (root_info.st_dev, root_info.st_ino)
    try:
        if root_info.st_dev != parent_info.st_dev:
            raise OSError("artifact root device differs")
        _require_capacity(root, required_bytes, required_inodes, statvfs)
    except Exception:
        current = root.lstat()
        if (current.st_dev, current.st_ino) == owned and not any(root.iterdir()):
            root.rmdir()
        raise


def require_remaining_capacity(
    root: Path, *, total_budget_bytes: int, total_budget_inodes: int,
    consumed_bytes: int, consumed_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    _plain_directory(root, "artifact root")
    values = (total_budget_bytes, total_budget_inodes, consumed_bytes, consumed_inodes)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("artifact budget differs")
    _require_capacity(
        root, max(total_budget_bytes - consumed_bytes, 0),
        max(total_budget_inodes - consumed_inodes, 0), statvfs,
    )


def validate_campaign_resume(
    config: object, run_root: Path,
    *, terminal_validator: Callable[[Path], None],
) -> tuple[Path, ...]:
    value = _strict_shape(config)
    _validate_config_values(value)
    root = run_root.resolve()
    _plain_directory(root, "campaign root")
    inventory = set(registered_stage_inventory(value))
    allowed_root_files = {"controller-status.json"}
    root_file_names = {
        child.name for child in root.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    for name in tuple(root_file_names):
        if name.endswith("-sources.json"):
            result = f"{name[:-len('-sources.json')]}-result.json"
            if result not in root_file_names:
                raise ValueError("campaign resume publication is incomplete")
        elif name.endswith("-result.json"):
            sources = f"{name[:-len('-result.json')]}-sources.json"
            if sources not in root_file_names:
                raise ValueError("campaign resume publication is incomplete")
    for child in root.iterdir():
        if child.is_symlink():
            raise ValueError("campaign resume symlink differs")
        if child.is_file():
            if (
                child.name not in allowed_root_files
                and not child.name.endswith("-sources.json")
                and not child.name.endswith("-result.json")
            ):
                raise ValueError("campaign resume registered path differs")
            continue
        if child.name == "preflight":
            unexpected = [path for path in child.iterdir() if path.name != "cuda_canary_v1.json"]
            if unexpected:
                raise ValueError("campaign resume registered path differs")
            continue
        if child.name not in inventory:
            raise ValueError("campaign resume registered path differs")
        terminal_name = (
            "run-receipt.json"
            if (
                ("control" in child.name or "candidate" in child.name or "random" in child.name)
                and "profile" not in child.name
                and "decision" not in child.name
            )
            else "terminal.json"
        )
        terminal = child / terminal_name
        if not terminal.is_file() or terminal.is_symlink():
            raise ValueError("campaign resume terminal differs")
    terminals: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign resume symlink differs")
        if path.is_file():
            if path.name.endswith(".tmp") or path.name.endswith(".partial"):
                raise ValueError("campaign partial artifact differs")
            if path.name in {"run-receipt.json", "profile-receipt.json", "result.json",
                             "cuda_canary_v1.json"} or path.name.endswith("-result.json"):
                terminal_validator(path)
                terminals.append(path)
    return tuple(terminals)


def registered_stage_inventory(config: object) -> tuple[str, ...]:
    _validate_config_values(_strict_shape(config))
    return (
        "cuda-canary", *(f"runtime-{index:02d}" for index in range(8)),
        "exploratory-control-stage4",
        "exploratory-candidate-stage4", "exploratory-epoch4-decision",
        "exploratory-control-stage16", "exploratory-candidate-stage16",
        "exploratory-profile-control-0", "exploratory-profile-candidate-0",
        "exploratory-profile-candidate-1", "exploratory-profile-control-1",
        "exploratory-decision", "exploratory-random-stage16",
        *(f"confirmation-{index}-{suffix}" for index in range(5) for suffix in (
            "control", "candidate", "profile-control-0", "profile-candidate-0",
            "profile-candidate-1", "profile-control-1",
        )), "confirmation-decision",
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkout-root-template", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-device-uuid")
    parser.add_argument("--cuda-environment-sha256")
    parser.add_argument("--validate-handoff", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.validate_handoff:
            validate_config_handoff(args.output, args.repo)
        else:
            if args.cuda_device_uuid is None or args.cuda_environment_sha256 is None:
                raise ValueError("CUDA canary external authority is required")
            build_and_write(
                repo=args.repo, checkout_root_template=args.checkout_root_template,
                artifact_root=args.artifact_root, output=args.output,
                cuda_canary_authority={
                    "device_uuid": args.cuda_device_uuid,
                    "environment_sha256": args.cuda_environment_sha256,
                },
            )
    except Exception as error:
        print(f"FEPF config failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
