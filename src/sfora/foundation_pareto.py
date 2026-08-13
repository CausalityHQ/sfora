"""Reproducible foundation-encoder screening primitives."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import secrets
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal, cast

from sfora.data import (
    ImageDatasetName,
    ImageExample,
    load_image_retrieval_examples,
    materialize_image,
    preflight_official_image_retrieval_split,
)
from sfora.image_end_to_end import run_image_end_to_end_benchmark
from sfora.image_recipes import (
    RecipeSelectionSplit,
    class_disjoint_recipe_selection_split,
    config_for_recipe,
    recipe_digest,
    reference_recipe,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
FOUNDATION_FIXTURE_METRICS = ("embedding_cosine",)
FOUNDATION_PUBLISHED_METRICS = (
    "recall_at_1",
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
    "recall_at_100",
    "map_at_r",
)
FOUNDATION_DATASETS = ("cub", "cars", "sop", "inshop", "inat2018")
FOUNDATION_TEST_READ_PURPOSE = "registered_f1_quality_evaluation"

_IDENTITY_DISJOINT_REQUEST_SCHEMA = "foundation-identity-disjoint-comparator-request-v1"
_IDENTITY_DISJOINT_RECEIPT_SCHEMA = "foundation-identity-disjoint-comparator-receipt-v1"


def _require_foundation_dataset(value: object) -> str:
    if type(value) is not str or value not in FOUNDATION_DATASETS:
        raise ValueError("foundation dataset differs from registered choices")
    return value


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

    status: Literal[
        "CONTINUE",
        "CLOSE_FOUNDATION_TRANSFER",
        "UNAVAILABLE_COMPARATOR",
        "INVALID_SPLIT_POWER",
        "BOUNDARY_REPLICATION_REQUIRED",
    ]
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
        "redesign_split_power_gate",
        "replicate_comparator_seed_zero",
    ]
    fidelity_only: bool


@dataclass(frozen=True)
class IdentityDisjointComparatorRequest:
    """Prospective authority for one identity-disjoint Proxy Anchor training run."""

    schema_version: str
    dataset: str
    dataset_root: Path
    source_commit: str
    recipe_id: str
    recipe_digest: str
    outer_seed: int
    outer_fraction: float
    training_seed: int
    epochs: int
    checkpoint_path: Path
    receipt_path: Path
    pretrained_backbone_path: Path
    wall_clock_ceiling_seconds: int

    def __post_init__(self) -> None:
        if self.schema_version != _IDENTITY_DISJOINT_REQUEST_SCHEMA:
            raise ValueError("identity-disjoint request schema differs")
        if self.dataset != "inshop":
            raise ValueError("identity-disjoint comparator is registered only for inshop")
        if (
            type(self.source_commit) is not str
            or _GIT_REVISION.fullmatch(self.source_commit) is None
        ):
            raise ValueError("identity-disjoint source commit differs")
        if self.recipe_id != "proxy_anchor.inshop.official-51db570":
            raise ValueError("identity-disjoint recipe differs")
        _require_sha256("identity-disjoint recipe digest", self.recipe_digest)
        if type(self.outer_seed) is not int or self.outer_seed != 0:
            raise ValueError("identity-disjoint outer seed differs")
        if type(self.outer_fraction) is not float or self.outer_fraction != 0.2:
            raise ValueError("identity-disjoint outer fraction differs")
        if type(self.training_seed) is not int or self.training_seed != 2:
            raise ValueError("identity-disjoint training seed differs")
        if type(self.epochs) is not int or self.epochs != 60:
            raise ValueError("identity-disjoint epoch count differs")
        if (
            type(self.wall_clock_ceiling_seconds) is not int
            or self.wall_clock_ceiling_seconds != 9000
        ):
            raise ValueError("identity-disjoint wall-clock ceiling differs")
        for name in (
            "dataset_root",
            "checkpoint_path",
            "receipt_path",
            "pretrained_backbone_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute() or value != value.resolve():
                raise ValueError(f"identity-disjoint {name} must be an absolute normalized Path")
        if self.checkpoint_path == self.receipt_path:
            raise ValueError("identity-disjoint checkpoint and receipt paths must differ")


def _identity_disjoint_request_payload(
    request: IdentityDisjointComparatorRequest,
) -> dict[str, object]:
    return {
        "schema_version": request.schema_version,
        "dataset": request.dataset,
        "dataset_root": str(request.dataset_root),
        "source_commit": request.source_commit,
        "recipe_id": request.recipe_id,
        "recipe_digest": request.recipe_digest,
        "outer_seed": request.outer_seed,
        "outer_fraction": request.outer_fraction,
        "training_seed": request.training_seed,
        "epochs": request.epochs,
        "checkpoint_path": str(request.checkpoint_path),
        "receipt_path": str(request.receipt_path),
        "pretrained_backbone_path": str(request.pretrained_backbone_path),
        "wall_clock_ceiling_seconds": request.wall_clock_ceiling_seconds,
    }


def identity_disjoint_role_digests(split: RecipeSelectionSplit) -> dict[str, object]:
    """Bind exact outer-role order, IDs, labels, and identity separation."""

    if not isinstance(split, RecipeSelectionSplit):
        raise ValueError("identity-disjoint split must be a RecipeSelectionSplit")
    roles = {
        "optimization": [
            {"example_id": row.example_id, "label": int(row.label)} for row in split.optimization
        ],
        "query": [{"example_id": row.example_id, "label": int(row.label)} for row in split.query],
        "gallery": [
            {"example_id": row.example_id, "label": int(row.label)} for row in split.gallery
        ],
    }
    flattened = [row["example_id"] for rows in roles.values() for row in rows]
    if any(type(value) is not str or not value for value in flattened):
        raise ValueError("identity-disjoint example IDs must be nonempty builtin strings")
    if len(flattened) != len(set(flattened)):
        raise ValueError("identity-disjoint example IDs must be unique across roles")
    optimization_labels = {int(row.label) for row in split.optimization}
    query_labels = {int(row.label) for row in split.query}
    gallery_labels = {int(row.label) for row in split.gallery}
    if not optimization_labels.isdisjoint(query_labels | gallery_labels):
        raise ValueError("identity-disjoint optimization and heldout labels overlap")
    if query_labels != gallery_labels or not query_labels:
        raise ValueError("identity-disjoint query/gallery labels differ")
    if not split.optimization:
        raise ValueError("identity-disjoint optimization role is empty")
    return {
        "split_sha256": sha256(_identity_disjoint_json_bytes(roles)).hexdigest(),
        "optimization_example_ids_sha256": sha256(
            _canonical_json_bytes([row["example_id"] for row in roles["optimization"]])
        ).hexdigest(),
        "query_example_ids_sha256": sha256(
            _canonical_json_bytes([row["example_id"] for row in roles["query"]])
        ).hexdigest(),
        "gallery_example_ids_sha256": sha256(
            _canonical_json_bytes([row["example_id"] for row in roles["gallery"]])
        ).hexdigest(),
        "optimization_count": len(split.optimization),
        "query_count": len(split.query),
        "gallery_count": len(split.gallery),
        "optimization_label_count": len(optimization_labels),
        "heldout_label_count": len(query_labels),
        "identity_disjoint": True,
    }


def _require_exact_builtin_float(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) is not float or not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite builtin float in [{minimum}, {maximum}]")
    return value


def _identity_disjoint_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("identity-disjoint value is not canonical JSON data") from error


def _require_exact_typed_json(name: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected):
        raise TypeError(f"{name} has a non-builtin or incorrect JSON type")
    if type(expected) is dict:
        observed_dict = cast(dict[str, object], observed)
        expected_dict = cast(dict[str, object], expected)
        if tuple(observed_dict) != tuple(expected_dict):
            raise ValueError(f"{name} key order differs")
        for key in expected_dict:
            _require_exact_typed_json(
                f"{name}.{key}",
                observed_dict[key],
                expected_dict[key],
            )
        return
    if type(expected) is list:
        observed_list = cast(list[object], observed)
        expected_list = cast(list[object], expected)
        if len(observed_list) != len(expected_list):
            raise ValueError(f"{name} list length differs")
        for index, (observed_item, expected_item) in enumerate(
            zip(observed_list, expected_list, strict=True)
        ):
            _require_exact_typed_json(
                f"{name}[{index}]",
                observed_item,
                expected_item,
            )
        return
    if observed != expected:
        raise ValueError(f"{name} value differs")


def validate_identity_disjoint_comparator_receipt(
    value: object,
    *,
    request: IdentityDisjointComparatorRequest,
) -> dict[str, object]:
    """Strictly validate one persisted identity-disjoint comparator receipt."""

    if type(value) is not dict:
        raise TypeError("identity-disjoint receipt must be a builtin dictionary")
    _require_ordered_keys(
        value,
        (
            "schema_version",
            "status",
            "request",
            "source",
            "recipe",
            "split",
            "training",
            "environment",
            "checkpoint",
            "diagnostic",
            "official_test",
            "process",
        ),
        name="identity-disjoint receipt",
    )
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"] != _IDENTITY_DISJOINT_RECEIPT_SCHEMA
    ):
        raise ValueError("identity-disjoint receipt schema differs")
    if type(value["status"]) is not str or value["status"] != "VALID":
        raise ValueError("identity-disjoint receipt status differs")
    request_value = value["request"]
    _require_exact_typed_json(
        "identity-disjoint receipt request",
        request_value,
        _identity_disjoint_request_payload(request),
    )

    source = value["source"]
    if type(source) is not dict:
        raise TypeError("identity-disjoint source audit must be a builtin dictionary")
    _require_ordered_keys(source, ("commit", "files"), name="identity-disjoint source audit")
    if source["commit"] != request.source_commit:
        raise ValueError("identity-disjoint source commit differs")
    files = source["files"]
    if type(files) is not list or not files:
        raise ValueError("identity-disjoint source files differ")
    observed_paths: set[str] = set()
    for row in files:
        if type(row) is not dict:
            raise TypeError("identity-disjoint source row must be a builtin dictionary")
        _require_ordered_keys(row, ("path", "sha256"), name="identity-disjoint source row")
        _require_nonempty("identity-disjoint source path", row["path"])
        _require_sha256("identity-disjoint source sha256", row["sha256"])
        if row["path"] in observed_paths:
            raise ValueError("identity-disjoint source paths must be unique")
        observed_paths.add(row["path"])

    recipe = value["recipe"]
    if type(recipe) is not dict:
        raise TypeError("identity-disjoint recipe audit must be a builtin dictionary")
    _require_ordered_keys(
        recipe,
        ("id", "digest", "resolved_config", "resolved_config_sha256"),
        name="identity-disjoint recipe audit",
    )
    if recipe["id"] != request.recipe_id or recipe["digest"] != request.recipe_digest:
        raise ValueError("identity-disjoint recipe authority differs")
    if type(recipe["resolved_config"]) is not dict:
        raise TypeError("identity-disjoint resolved config must be a builtin dictionary")
    expected_config_sha = sha256(
        _identity_disjoint_json_bytes(recipe["resolved_config"])
    ).hexdigest()
    if recipe["resolved_config_sha256"] != expected_config_sha:
        raise ValueError("identity-disjoint resolved config digest differs")

    split = value["split"]
    if type(split) is not dict:
        raise TypeError("identity-disjoint split audit must be a builtin dictionary")
    split_keys = (
        "split_sha256",
        "optimization_example_ids_sha256",
        "query_example_ids_sha256",
        "gallery_example_ids_sha256",
        "optimization_count",
        "query_count",
        "gallery_count",
        "optimization_label_count",
        "heldout_label_count",
        "identity_disjoint",
    )
    _require_ordered_keys(split, split_keys, name="identity-disjoint split audit")
    for key in split_keys[:4]:
        _require_sha256(f"identity-disjoint {key}", split[key])
    for key in split_keys[4:9]:
        if type(split[key]) is not int or split[key] <= 0:
            raise ValueError(f"identity-disjoint {key} must be a positive builtin integer")
    if split["identity_disjoint"] is not True:
        raise ValueError("identity-disjoint split relation differs")

    training = value["training"]
    if type(training) is not dict:
        raise TypeError("identity-disjoint training audit must be a builtin dictionary")
    _require_ordered_keys(
        training,
        (
            "seed",
            "epochs",
            "steps",
            "artifact_selection",
            "checkpoint_selection_interval",
            "eval_test_interval_epochs",
        ),
        name="identity-disjoint training audit",
    )
    if training["seed"] != request.training_seed or type(training["seed"]) is not int:
        raise ValueError("identity-disjoint observed training seed differs")
    if training["epochs"] != request.epochs or type(training["epochs"]) is not int:
        raise ValueError("identity-disjoint observed epochs differ")
    if type(training["steps"]) is not int or training["steps"] <= 0:
        raise ValueError("identity-disjoint training steps differ")
    if (
        type(training["artifact_selection"]) is not str
        or training["artifact_selection"] != "final_training_state"
    ):
        raise ValueError("identity-disjoint artifact selection differs")
    if (
        type(training["checkpoint_selection_interval"]) is not int
        or training["checkpoint_selection_interval"] != 0
    ):
        raise ValueError("identity-disjoint checkpoint selection must be disabled")
    if (
        type(training["eval_test_interval_epochs"]) is not int
        or training["eval_test_interval_epochs"] != 0
    ):
        raise ValueError("identity-disjoint periodic heldout evaluation must be disabled")

    environment = value["environment"]
    if type(environment) is not dict:
        raise TypeError("identity-disjoint environment must be a builtin dictionary")
    _require_ordered_keys(
        environment,
        (
            "python_version",
            "torch_version",
            "numpy_version",
            "device_type",
            "device_name",
            "cublas_workspace_config",
            "deterministic_algorithms",
        ),
        name="identity-disjoint environment",
    )
    for key in ("python_version", "torch_version", "numpy_version", "device_name"):
        _require_nonempty(f"identity-disjoint {key}", environment[key])
    if type(environment["device_type"]) is not str or environment["device_type"] != "cuda":
        raise ValueError("identity-disjoint device type differs")
    if (
        type(environment["cublas_workspace_config"]) is not str
        or environment["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("identity-disjoint CUBLAS workspace differs")
    if environment["deterministic_algorithms"] is not True:
        raise ValueError("identity-disjoint deterministic algorithms differ")

    checkpoint = value["checkpoint"]
    if type(checkpoint) is not dict:
        raise TypeError("identity-disjoint checkpoint audit must be a builtin dictionary")
    _require_ordered_keys(
        checkpoint,
        ("path", "sha256", "mode", "size_bytes", "resolved_config_sha256"),
        name="identity-disjoint checkpoint audit",
    )
    if checkpoint["path"] != str(request.checkpoint_path):
        raise ValueError("identity-disjoint checkpoint path differs")
    _require_sha256("identity-disjoint checkpoint sha256", checkpoint["sha256"])
    if type(checkpoint["mode"]) is not int or checkpoint["mode"] != 0o600:
        raise ValueError("identity-disjoint checkpoint mode differs")
    if type(checkpoint["size_bytes"]) is not int or checkpoint["size_bytes"] <= 0:
        raise ValueError("identity-disjoint checkpoint size differs")
    if checkpoint["resolved_config_sha256"] != expected_config_sha:
        raise ValueError("identity-disjoint checkpoint config digest differs")

    diagnostic = value["diagnostic"]
    if type(diagnostic) is not dict:
        raise TypeError("identity-disjoint diagnostic must be a builtin dictionary")
    _require_ordered_keys(
        diagnostic,
        ("heldout_recall_at_1",),
        name="identity-disjoint diagnostic",
    )
    _require_exact_builtin_float(
        "identity-disjoint heldout R@1",
        diagnostic["heldout_recall_at_1"],
        minimum=0.0,
        maximum=1.0,
    )

    official = value["official_test"]
    if type(official) is not dict:
        raise TypeError("identity-disjoint official-test audit must be a builtin dictionary")
    _require_ordered_keys(
        official,
        ("consumed", "receipts"),
        name="identity-disjoint official-test audit",
    )
    if official["consumed"] is not False or official["receipts"] != []:
        raise ValueError("identity-disjoint training consumed official-test capability")

    process = value["process"]
    if type(process) is not dict:
        raise TypeError("identity-disjoint process audit must be a builtin dictionary")
    _require_ordered_keys(
        process,
        ("pid", "started_at_utc", "finished_at_utc", "elapsed_seconds", "exit_code"),
        name="identity-disjoint process audit",
    )
    if type(process["pid"]) is not int or process["pid"] <= 0:
        raise ValueError("identity-disjoint process PID differs")
    for key in ("started_at_utc", "finished_at_utc"):
        _require_nonempty(f"identity-disjoint {key}", process[key])
    _require_exact_builtin_float(
        "identity-disjoint elapsed seconds",
        process["elapsed_seconds"],
        minimum=0.0,
        maximum=float(request.wall_clock_ceiling_seconds),
    )
    if process["exit_code"] != 0 or type(process["exit_code"]) is not int:
        raise ValueError("identity-disjoint process exit code differs")
    return value


def _validate_identity_disjoint_backbone(path: Path) -> Path:
    import torch

    expected = "52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353"
    if path.is_symlink() or not path.is_file():
        raise ValueError("identity-disjoint pretrained backbone must be a regular file")
    consumed_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / "bn_inception-52deb4733.pth"
    ).resolve()
    if path.resolve() != consumed_path:
        raise ValueError("identity-disjoint pretrained backbone differs from Torch cache input")
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("identity-disjoint pretrained backbone digest differs")
    return path


def _load_identity_disjoint_checkpoint(path: Path) -> dict[str, object]:
    return cast(dict[str, object], _torch_load_checkpoint(path))


def _identity_disjoint_environment() -> dict[str, object]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("identity-disjoint comparator training requires CUDA")
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device_type": "cuda",
        "device_name": str(torch.cuda.get_device_name(0)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _identity_disjoint_source_files() -> list[dict[str, object]]:
    root = Path(__file__).resolve().parents[2]
    paths = ("src/sfora/foundation_pareto.py", "src/sfora/cli.py")
    return [
        {"path": relative, "sha256": sha256((root / relative).read_bytes()).hexdigest()}
        for relative in paths
    ]


def _authenticate_identity_disjoint_source(reviewed_revision: str) -> None:
    if type(reviewed_revision) is not str or _GIT_REVISION.fullmatch(reviewed_revision) is None:
        raise ValueError("identity-disjoint reviewed source revision differs")
    root = Path(__file__).resolve().parents[2]
    executing_revision = _source_commit()
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                reviewed_revision,
                executing_revision,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--quiet",
                reviewed_revision,
                executing_revision,
                "--",
                "src/sfora",
            ],
            check=True,
            capture_output=True,
        )
        for relative in ("src/sfora/foundation_pareto.py", "src/sfora/cli.py"):
            reviewed_bytes = subprocess.run(
                ["git", "-C", str(root), "show", f"{reviewed_revision}:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
            if reviewed_bytes != (root / relative).read_bytes():
                raise ValueError("identity-disjoint reviewed source bytes differ")
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "identity-disjoint reviewed source ancestry or production tree differs"
        ) from error


def _validate_identity_disjoint_checkpoint(
    checkpoint: object,
    *,
    config: Any,
) -> dict[str, object]:
    if type(checkpoint) is not dict:
        raise ValueError("identity-disjoint checkpoint root differs")
    expected_keys = (
        "state_dict",
        "arch",
        "artifact_selection",
        "training_step",
        "evaluation_model_source",
        "training_config",
    )
    if tuple(checkpoint) != expected_keys:
        raise ValueError("identity-disjoint checkpoint schema differs")
    if not isinstance(checkpoint["state_dict"], Mapping) or not checkpoint["state_dict"]:
        raise ValueError("identity-disjoint checkpoint state differs")
    expected_arch = {
        "backbone_name": config.backbone_name,
        "pretrained_weights": config.pretrained_weights,
        "head_pooling": config.head_pooling,
        "embedding_dimensions": config.embedding_dimensions,
        "embedding_head_init": config.embedding_head_init,
        "embedding_layer_norm": config.embedding_layer_norm,
    }
    _require_exact_typed_json(
        "identity-disjoint checkpoint architecture",
        checkpoint["arch"],
        expected_arch,
    )
    if checkpoint["artifact_selection"] != "final_training_state":
        raise ValueError("identity-disjoint checkpoint is not the final training state")
    if type(checkpoint["training_step"]) is not int or checkpoint["training_step"] <= 0:
        raise ValueError("identity-disjoint checkpoint step differs")
    if checkpoint["evaluation_model_source"] != "student":
        raise ValueError("identity-disjoint checkpoint evaluation model differs")
    expected_config = config.model_dump(mode="json")
    _require_exact_typed_json(
        "identity-disjoint checkpoint training config",
        checkpoint["training_config"],
        expected_config,
    )
    return checkpoint


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run_identity_disjoint_comparator_training(
    request: IdentityDisjointComparatorRequest,
) -> Path:
    """Train and atomically publish one final-state identity-disjoint comparator."""

    started = time.monotonic()
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not isinstance(request, IdentityDisjointComparatorRequest):
        raise TypeError("identity-disjoint training request differs")
    _authenticate_identity_disjoint_source(request.source_commit)
    for output in (request.checkpoint_path, request.receipt_path):
        if output.parent.is_symlink() or not output.parent.is_dir():
            raise ValueError("identity-disjoint output parent must be a real directory")
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("identity-disjoint CUBLAS workspace must be :4096:8 before launch")
    _validate_identity_disjoint_backbone(request.pretrained_backbone_path)
    recipe = reference_recipe("proxy_anchor", "inshop")
    if recipe.recipe_id != request.recipe_id or recipe_digest(recipe) != request.recipe_digest:
        raise ValueError("identity-disjoint recipe authority differs")

    examples = load_image_retrieval_examples(
        dataset_name="inshop",
        split="train",
        dataset_root=request.dataset_root,
    )
    split = build_identity_disjoint_validation_split(
        examples,
        fraction=request.outer_fraction,
        seed=request.outer_seed,
    )
    role_digests = identity_disjoint_role_digests(split)
    checkpoint_temp = request.checkpoint_path.with_name(
        f".{request.checkpoint_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    if checkpoint_temp.exists() or checkpoint_temp.is_symlink():
        raise FileExistsError(checkpoint_temp)
    config = config_for_recipe(recipe).model_copy(
        update={
            "dataset_root": request.dataset_root,
            "seed": request.training_seed,
            "train_epochs": request.epochs,
            "checkpoint_selection_interval": 0,
            "eval_test_interval_epochs": 0,
            "deterministic": True,
            "save_model_path": str(checkpoint_temp),
        }
    )

    published_identity: tuple[int, int] | None = None
    receipt_identity: tuple[int, int] | None = None
    try:
        result = run_image_end_to_end_benchmark(
            train_examples=split.optimization,
            test_examples=split.query,
            gallery_examples=split.gallery,
            config=config,
        )
        if checkpoint_temp.is_symlink() or not checkpoint_temp.is_file():
            raise ValueError("identity-disjoint training did not create a checkpoint")
        os.chmod(checkpoint_temp, 0o600)
        with checkpoint_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        checkpoint = _validate_identity_disjoint_checkpoint(
            _load_identity_disjoint_checkpoint(checkpoint_temp),
            config=config,
        )
        temporary_stat = checkpoint_temp.stat()
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(checkpoint_temp, request.checkpoint_path)
        _fsync_directory(request.checkpoint_path.parent)

        methods = list(result.methods.values())
        if len(methods) != 1:
            raise ValueError("identity-disjoint training method result differs")
        heldout_recall = float(methods[0].recall_at_1)
        finished_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        elapsed = float(time.monotonic() - started)
        resolved_config = config.model_dump(mode="json")
        resolved_config_sha256 = sha256(_identity_disjoint_json_bytes(resolved_config)).hexdigest()
        payload: dict[str, object] = {
            "schema_version": _IDENTITY_DISJOINT_RECEIPT_SCHEMA,
            "status": "VALID",
            "request": _identity_disjoint_request_payload(request),
            "source": {
                "commit": request.source_commit,
                "files": _identity_disjoint_source_files(),
            },
            "recipe": {
                "id": request.recipe_id,
                "digest": request.recipe_digest,
                "resolved_config": resolved_config,
                "resolved_config_sha256": resolved_config_sha256,
            },
            "split": role_digests,
            "training": {
                "seed": request.training_seed,
                "epochs": request.epochs,
                "steps": checkpoint["training_step"],
                "artifact_selection": checkpoint["artifact_selection"],
                "checkpoint_selection_interval": config.checkpoint_selection_interval,
                "eval_test_interval_epochs": config.eval_test_interval_epochs,
            },
            "environment": _identity_disjoint_environment(),
            "checkpoint": {
                "path": str(request.checkpoint_path),
                "sha256": sha256(request.checkpoint_path.read_bytes()).hexdigest(),
                "mode": request.checkpoint_path.stat().st_mode & 0o777,
                "size_bytes": request.checkpoint_path.stat().st_size,
                "resolved_config_sha256": resolved_config_sha256,
            },
            "diagnostic": {"heldout_recall_at_1": heldout_recall},
            "official_test": {"consumed": False, "receipts": []},
            "process": {
                "pid": os.getpid(),
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "elapsed_seconds": elapsed,
                "exit_code": 0,
            },
        }
        validate_identity_disjoint_comparator_receipt(payload, request=request)
        _publish_json_no_clobber(request.receipt_path, payload)
        receipt_stat = request.receipt_path.stat()
        receipt_identity = (receipt_stat.st_dev, receipt_stat.st_ino)
        persisted = _load_strict_json(request.receipt_path)
        validate_identity_disjoint_comparator_receipt(persisted, request=request)
        if persisted != payload:
            raise ValueError("persisted identity-disjoint receipt differs")
        checkpoint_temp.unlink()
        _fsync_directory(request.checkpoint_path.parent)
        return request.receipt_path
    except BaseException:
        if (
            receipt_identity is not None
            and request.receipt_path.is_file()
            and not request.receipt_path.is_symlink()
        ):
            receipt_stat = request.receipt_path.stat()
            if (receipt_stat.st_dev, receipt_stat.st_ino) == receipt_identity:
                request.receipt_path.unlink()
        if (
            published_identity is not None
            and request.checkpoint_path.is_file()
            and not request.checkpoint_path.is_symlink()
        ):
            output_stat = request.checkpoint_path.stat()
            if (output_stat.st_dev, output_stat.st_ino) == published_identity:
                request.checkpoint_path.unlink()
        if checkpoint_temp.is_file() and not checkpoint_temp.is_symlink():
            checkpoint_temp.unlink()
        raise


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
    if comparator_validation_recall_at_1_points >= 99.5:
        status: Literal[
            "CONTINUE",
            "CLOSE_FOUNDATION_TRANSFER",
            "INVALID_SPLIT_POWER",
            "BOUNDARY_REPLICATION_REQUIRED",
        ] = "INVALID_SPLIT_POWER"
    elif -1.5 <= gap <= -0.5:
        status = "BOUNDARY_REPLICATION_REQUIRED"
    else:
        status = "CONTINUE" if within_one or pareto_equivalent else "CLOSE_FOUNDATION_TRANSFER"
    continuation_kind: Literal["quality_margin", "pareto_equivalent", "both", "none"]
    if status != "CONTINUE":
        continuation_kind = "none"
    elif within_one and pareto_equivalent:
        continuation_kind = "both"
    elif within_one:
        continuation_kind = "quality_margin"
    elif pareto_equivalent:
        continuation_kind = "pareto_equivalent"
    else:
        continuation_kind = "none"
    authorized_followup: Literal[
        "matryoshka_adapter_lane",
        "dada_vptsp_fidelity_comparator_only",
        "redesign_split_power_gate",
        "replicate_comparator_seed_zero",
    ]
    if status == "CONTINUE":
        authorized_followup = "matryoshka_adapter_lane"
    elif status == "CLOSE_FOUNDATION_TRANSFER":
        authorized_followup = "dada_vptsp_fidelity_comparator_only"
    elif status == "INVALID_SPLIT_POWER":
        authorized_followup = "redesign_split_power_gate"
    else:
        authorized_followup = "replicate_comparator_seed_zero"
    return F1Decision(
        status=status,
        quality_gap_points=gap,
        quality_within_one_point=within_one,
        quality_within_point_four=within_point_four,
        cost_pareto_dominant=pareto,
        cost_status=cost_status,
        continuation_kind=continuation_kind,
        authorized_followup=authorized_followup,
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
class FoundationScreenArmSpec:
    kind: Literal["remote", "local"]
    spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec
    cache_resolution: int
    role: Literal["candidate", "comparator", "contaminated_control"]

    def __post_init__(self) -> None:
        if self.kind == "remote":
            if type(self.spec) is not RemoteFoundationModelSpec:
                raise ValueError("remote screen arm requires a remote model spec")
            if self.cache_resolution != self.spec.resolution:
                raise ValueError("remote cache resolution differs from model spec")
        elif self.kind == "local":
            if type(self.spec) is not LocalCheckpointFoundationSpec:
                raise ValueError("local screen arm requires a local checkpoint spec")
            _require_positive_int("local cache_resolution", self.cache_resolution)
        else:
            raise ValueError("foundation screen arm kind differs from exact choices")
        if self.role not in {"candidate", "comparator", "contaminated_control"}:
            raise ValueError("foundation screen arm role differs from exact choices")


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
                output: Any
                if self.spec.pooling == "image_features":
                    result = self.model.get_image_features(**values)
                    output = (
                        result
                        if torch.is_tensor(result)
                        else getattr(result, "pooler_output", None)
                    )
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
    provenance: Literal["native_cross_check", "repository_replay", "unavailable"]
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
class FoundationTestReadRecord:
    """Prospective authority for one arm's single official-test evaluation."""

    dataset: str
    arm: str
    model_revision: str
    checkpoint_sha256: str
    metrics: tuple[str, ...]
    purpose: Literal["registered_f1_quality_evaluation"]
    permitted_evaluations: Literal[1]

    def __post_init__(self) -> None:
        _require_foundation_dataset(self.dataset)
        _require_nonempty("test-read arm", self.arm)
        if (
            type(self.model_revision) is not str
            or _GIT_REVISION.fullmatch(self.model_revision) is None
        ):
            raise ValueError("test-read model_revision must be an immutable object ID")
        _require_sha256("test-read checkpoint_sha256", self.checkpoint_sha256)
        if type(self.metrics) is not tuple or not self.metrics:
            raise ValueError("test-read metrics must be a nonempty builtin tuple")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("test-read metrics must be unique and ordered")
        for metric in self.metrics:
            _require_nonempty("test-read metric", metric)
        if self.purpose != FOUNDATION_TEST_READ_PURPOSE:
            raise ValueError("test-read purpose differs from registered purpose")
        if type(self.permitted_evaluations) is not int or self.permitted_evaluations != 1:
            raise ValueError("test-read permitted_evaluations must be exactly one")


@dataclass(frozen=True)
class OfficialTestReadAudit:
    dataset: str
    arm: str
    model_revision: str
    checkpoint_sha256: str
    metrics: tuple[str, ...]
    purpose: Literal["registered_f1_quality_evaluation"]
    evaluation_number: Literal[1]


@dataclass(frozen=True)
class OfficialTestReadReceipt:
    dataset: str
    arm: str
    model_revision: str
    checkpoint_sha256: str
    metrics: tuple[str, ...]
    purpose: Literal["registered_f1_quality_evaluation"]
    evaluation_number: Literal[1]
    decision_sha256: str
    receipt_path: Path


@dataclass(frozen=True)
class FoundationOfficialReadBinding:
    """Reviewed source and train-only evidence required before official pixels."""

    reviewed_source_commit: str
    addendum_path: str
    addendum_sha256: str
    train_report_path: str
    train_report_sha256: str
    train_decision_sha256: str
    train_split_sha256: str

    def __post_init__(self) -> None:
        if _GIT_REVISION.fullmatch(self.reviewed_source_commit) is None:
            raise ValueError("official reviewed_source_commit differs")
        for name, value in (
            ("addendum_path", self.addendum_path),
            ("train_report_path", self.train_report_path),
        ):
            if type(value) is not str or not value or Path(value).is_absolute():
                raise ValueError(f"official {name} must be a nonempty repository path")
        for name, value in (
            ("addendum_sha256", self.addendum_sha256),
            ("train_report_sha256", self.train_report_sha256),
            ("train_decision_sha256", self.train_decision_sha256),
            ("train_split_sha256", self.train_split_sha256),
        ):
            _require_sha256(f"official {name}", value)


class FoundationTestReadLedger:
    """In-process one-shot capability over a frozen test-read register."""

    def __init__(
        self,
        records: tuple[FoundationTestReadRecord, ...],
        *,
        receipt_root: Path,
    ) -> None:
        if type(records) is not tuple:
            raise ValueError("test-read records must be a builtin tuple")
        identities = tuple((record.dataset, record.arm) for record in records)
        if len(set(identities)) != len(identities):
            raise ValueError("test-read register contains duplicate dataset/arm identities")
        if not isinstance(receipt_root, Path) or not receipt_root.is_absolute():
            raise ValueError("test-read receipt_root must be an absolute Path")
        self.records = records
        self.receipt_root = receipt_root
        self._consumed: set[tuple[str, str]] = set()

    def consume(
        self,
        *,
        dataset: str,
        arm: str,
        model_revision: str,
        checkpoint_sha256: str,
        metrics: tuple[str, ...],
        purpose: str,
    ) -> OfficialTestReadAudit:
        identity = (dataset, arm)
        record = next(
            (value for value in self.records if (value.dataset, value.arm) == identity),
            None,
        )
        if record is None:
            raise ValueError("official evaluation is not a registered test read")
        if identity in self._consumed:
            raise ValueError("registered test read was already consumed")
        request = (
            model_revision,
            checkpoint_sha256,
            metrics,
            purpose,
        )
        expected = (
            record.model_revision,
            record.checkpoint_sha256,
            record.metrics,
            record.purpose,
        )
        if request != expected:
            raise ValueError("official evaluation differs from registered test read")
        self._consumed.add(identity)
        return OfficialTestReadAudit(
            dataset=dataset,
            arm=arm,
            model_revision=model_revision,
            checkpoint_sha256=checkpoint_sha256,
            metrics=metrics,
            purpose="registered_f1_quality_evaluation",
            evaluation_number=1,
        )


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


@dataclass(frozen=True)
class FoundationTrainCacheResult:
    train_embeddings: Mapping[str, Any]
    records: tuple[dict[str, object], ...]


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
    exclude_self: bool = False,
) -> tuple[tuple[int, ...], ...]:
    import numpy as np

    if type(depth) is not int or depth <= 0:
        raise ValueError("geometry ranking depth must be a positive builtin integer")
    if type(exclude_self) is not bool:
        raise ValueError("exclude_self must be a builtin boolean")
    if exclude_self and query.shape[0] != gallery.shape[0]:
        raise ValueError("self-excluding geometry requires equal query/gallery rows")
    depth = min(depth, gallery.shape[0] - int(exclude_self))
    if depth <= 0:
        raise ValueError("geometry ranking has no eligible gallery rows")
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
        if exclude_self:
            local_rows = np.arange(chunk.shape[0])
            values[local_rows, start + local_rows] = np.inf
        rows.extend(
            tuple(int(index) for index in _stable_top_indices(row, depth=depth)) for row in values
        )
    return tuple(rows)


def _metrics_from_gallery_order(
    gallery_order: tuple[tuple[int, ...], ...],
    query_labels: Any,
    gallery_labels: Any,
    *,
    exclude_self: bool = False,
) -> Any:
    import numpy as np

    from sfora.image_benchmark import ImageRetrievalMetrics

    precision_at_1: list[float] = []
    recalls: dict[int, list[float]] = {cutoff: [] for cutoff in (1, 2, 4, 8, 10, 20, 30, 100)}
    average_precisions: list[float] = []
    relevant_counts: list[int] = []
    for query_index, order in enumerate(gallery_order):
        matches = gallery_labels == query_labels[query_index]
        if exclude_self:
            matches = matches.copy()
            matches[query_index] = False
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
    *,
    exclude_self: bool = False,
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
        exclude_self=exclude_self,
    )
    normalized_euclidean_order = _gallery_order(
        normalized_query,
        normalized_gallery,
        geometry="euclidean",
        depth=depth,
        exclude_self=exclude_self,
    )
    native_order = _gallery_order(
        query,
        gallery,
        geometry="euclidean",
        depth=depth,
        exclude_self=exclude_self,
    )
    return (
        FoundationGeometryEvaluation(
            geometry="normalized_cosine",
            gallery_order=cosine_order,
            metrics=_metrics_from_gallery_order(
                cosine_order,
                query_label_array,
                gallery_label_array,
                exclude_self=exclude_self,
            ),
        ),
        FoundationGeometryEvaluation(
            geometry="normalized_euclidean",
            gallery_order=normalized_euclidean_order,
            metrics=_metrics_from_gallery_order(
                normalized_euclidean_order,
                query_label_array,
                gallery_label_array,
                exclude_self=exclude_self,
            ),
        ),
        FoundationGeometryEvaluation(
            geometry="native_unnormalized_euclidean",
            gallery_order=native_order,
            metrics=_metrics_from_gallery_order(
                native_order,
                query_label_array,
                gallery_label_array,
                exclude_self=exclude_self,
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
        normalize: bool | None = None,
    ) -> EmbeddingCacheKeyV2:
        resolved_normalize = spec.normalize if normalize is None else normalize
        if type(resolved_normalize) is not bool:
            raise ValueError("cache normalize override must be a builtin boolean")
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
                normalize=resolved_normalize,
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
            normalize=resolved_normalize,
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


def _publish_json_no_clobber(path: Path, payload: dict[str, object]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("JSON output parent must be a real directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = _canonical_json_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    published_identity: tuple[int, int] | None = None
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = temporary.stat()
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            temporary.unlink()
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        persisted = _load_strict_json(path)
        if persisted != payload or path.read_bytes() != encoded:
            raise ValueError("persisted JSON output differs from canonical input")
    except BaseException:
        if published_identity is not None and path.is_file() and not path.is_symlink():
            path_stat = path.stat()
            if (path_stat.st_dev, path_stat.st_ino) == published_identity:
                path.unlink()
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _official_test_receipt_payload(
    audit: OfficialTestReadAudit,
    *,
    decision_sha256: str,
) -> dict[str, object]:
    if type(audit) is not OfficialTestReadAudit:
        raise ValueError("official test read requires an exact consumed audit")
    _require_sha256("decision_sha256", decision_sha256)
    return {
        "schema_version": "foundation-official-test-read-v2",
        "dataset": audit.dataset,
        "arm": audit.arm,
        "model_revision": audit.model_revision,
        "checkpoint_sha256": audit.checkpoint_sha256,
        "metrics": list(audit.metrics),
        "purpose": audit.purpose,
        "evaluation_number": audit.evaluation_number,
        "decision_sha256": decision_sha256,
    }


def publish_official_test_read_receipt(
    root: Path,
    audit: OfficialTestReadAudit,
    *,
    decision_sha256: str,
) -> OfficialTestReadReceipt:
    payload = _official_test_receipt_payload(audit, decision_sha256=decision_sha256)
    identity_payload = {key: payload[key] for key in tuple(payload)[1:-1]}
    receipt_id = sha256(_canonical_json_bytes(identity_payload)).hexdigest()
    path = root / f"official-test-read-{receipt_id}.json"
    _publish_json_no_clobber(path, payload)
    return OfficialTestReadReceipt(
        dataset=audit.dataset,
        arm=audit.arm,
        model_revision=audit.model_revision,
        checkpoint_sha256=audit.checkpoint_sha256,
        metrics=audit.metrics,
        purpose=audit.purpose,
        evaluation_number=1,
        decision_sha256=decision_sha256,
        receipt_path=path,
    )


def load_registered_official_test(
    receipt: OfficialTestReadReceipt,
    *,
    dataset: str,
    arm: str,
    metrics: tuple[str, ...],
    loader: Callable[[], Any],
) -> Any:
    """Call an official-test loader only after its durable receipt matches."""

    if type(receipt) is not OfficialTestReadReceipt:
        raise ValueError("official test receipt differs from exact type")
    if receipt.dataset != dataset or receipt.arm != arm or receipt.metrics != metrics:
        raise ValueError("official test receipt differs from requested read")
    persisted = _load_strict_json(receipt.receipt_path)
    audit = OfficialTestReadAudit(
        dataset=receipt.dataset,
        arm=receipt.arm,
        model_revision=receipt.model_revision,
        checkpoint_sha256=receipt.checkpoint_sha256,
        metrics=receipt.metrics,
        purpose=receipt.purpose,
        evaluation_number=1,
    )
    expected = _official_test_receipt_payload(
        audit,
        decision_sha256=receipt.decision_sha256,
    )
    if persisted != expected:
        raise ValueError("official test receipt differs from persisted authority")
    return loader()


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
        repository_replay = not available and fixture.metric == "embedding_cosine"
        if available and fixture.native_value is not None:
            passed: bool | None = (
                abs(repository_value - fixture.native_value) <= tolerance.tolerance
            )
            provenance: Literal["native_cross_check", "repository_replay", "unavailable"] = (
                "native_cross_check"
            )
        elif repository_replay:
            passed = abs(repository_value - 1.0) <= tolerance.tolerance
            provenance = "repository_replay"
        else:
            passed = None
            provenance = "unavailable"
        audits.append(
            FoundationFidelityAudit(
                arm=arm,
                metric=fixture.metric,
                native_value=fixture.native_value,
                repository_value=repository_value,
                tolerance=tolerance.tolerance,
                provenance=provenance,
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
    schema_version = value.get("schema_version") if type(value) is dict else None
    authority_keys = (
        ("schema_version", "status", *_OFFICIAL_BINDING_KEYS, "records")
        if schema_version == "foundation-published-metrics-v4"
        else ("schema_version", "status", "records")
    )
    _require_ordered_keys(
        value,
        authority_keys,
        name="published metric authority",
    )
    if value["schema_version"] not in {
        "foundation-published-metrics-v1",
        "foundation-published-metrics-v4",
    }:
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


def load_test_read_register(
    path: Path,
    *,
    require_frozen: bool = True,
) -> FoundationTestReadLedger:
    """Load the exact prospective one-read authority for official test data."""

    value = _load_strict_json(path)
    schema_version = value.get("schema_version") if type(value) is dict else None
    authority_keys = (
        (
            "schema_version",
            "status",
            "receipt_root",
            *_OFFICIAL_BINDING_KEYS,
            "records",
        )
        if schema_version == "foundation-test-read-register-v4"
        else ("schema_version", "status", "receipt_root", "records")
    )
    _require_ordered_keys(
        value,
        authority_keys,
        name="test-read authority",
    )
    if value["schema_version"] not in {
        "foundation-test-read-register-v3",
        "foundation-test-read-register-v4",
    }:
        raise ValueError("test-read schema version differs")
    if value["status"] not in {"prospective_unfrozen", "frozen"}:
        raise ValueError("test-read authority status differs")
    if require_frozen and value["status"] != "frozen":
        raise ValueError("test-read authority is not frozen")
    if value["status"] == "prospective_unfrozen" and value["records"] != []:
        raise ValueError("prospective unfrozen test-read authority must be empty")
    if type(value["receipt_root"]) is not str or not value["receipt_root"]:
        raise ValueError("test-read receipt_root must be a nonempty string")
    receipt_root = Path(value["receipt_root"])
    if not receipt_root.is_absolute():
        raise ValueError("test-read receipt_root must be absolute")
    if type(value["records"]) is not list:
        raise ValueError("test-read records must be a JSON array")
    records: list[FoundationTestReadRecord] = []
    for raw in value["records"]:
        if type(raw) is not dict:
            raise ValueError("test-read record must be a JSON object")
        _require_ordered_keys(
            raw,
            (
                "dataset",
                "arm",
                "model_revision",
                "checkpoint_sha256",
                "metrics",
                "purpose",
                "permitted_evaluations",
            ),
            name="test-read record",
        )
        metrics = raw["metrics"]
        if type(metrics) is not list:
            raise ValueError("test-read metrics must be a JSON array")
        records.append(
            FoundationTestReadRecord(
                dataset=raw["dataset"],
                arm=raw["arm"],
                model_revision=raw["model_revision"],
                checkpoint_sha256=raw["checkpoint_sha256"],
                metrics=tuple(metrics),
                purpose=raw["purpose"],
                permitted_evaluations=raw["permitted_evaluations"],
            )
        )
    return FoundationTestReadLedger(tuple(records), receipt_root=receipt_root)


_OFFICIAL_BINDING_KEYS = (
    "reviewed_source_commit",
    "addendum_path",
    "addendum_sha256",
    "train_report_path",
    "train_report_sha256",
    "train_decision_sha256",
    "train_split_sha256",
)

_OFFICIAL_CORRECTION_PATH = (
    "docs/superpowers/specs/2026-08-13-foundation-f1-official-read-review-correction.md"
)
_OFFICIAL_CORRECTION_SHA256 = "d5e3f8601d62105a3c7a7559c6f150b52b08081bd7037e31b066c938325bbcaf"
_OFFICIAL_TRAIN_REPORT_PATH = "docs/foundation_f1_identity_disjoint_inshop_train_only_report.json"
_OFFICIAL_TRAIN_REPORT_SHA256 = "791cd1499327bd95abb5093d993a68c7192d44965af8669073bf35ad5b6ae066"
_OFFICIAL_TRAIN_DECISION_SHA256 = "a3400169c6b94dbde2a2ecbb329a839ffba9edd43679587877133c5f3c83a9c8"
_OFFICIAL_TRAIN_SPLIT_SHA256 = "7b075d601dbfa0b3f3587f80af169a378621cd4ba93aca35c2c9be745eac1f45"
_OFFICIAL_RECEIPT_ROOT = Path(
    "/home/riomus/group-learning/reports/generated/foundation_test_read_receipts"
)


def _validate_official_binding_constants(binding: FoundationOfficialReadBinding) -> None:
    if type(binding) is not FoundationOfficialReadBinding:
        raise ValueError("official binding differs from exact type")
    observed = (
        binding.addendum_path,
        binding.addendum_sha256,
        binding.train_report_path,
        binding.train_report_sha256,
        binding.train_decision_sha256,
        binding.train_split_sha256,
    )
    expected = (
        _OFFICIAL_CORRECTION_PATH,
        _OFFICIAL_CORRECTION_SHA256,
        _OFFICIAL_TRAIN_REPORT_PATH,
        _OFFICIAL_TRAIN_REPORT_SHA256,
        _OFFICIAL_TRAIN_DECISION_SHA256,
        _OFFICIAL_TRAIN_SPLIT_SHA256,
    )
    if observed != expected:
        raise ValueError("official binding differs from frozen evidence")


def _validate_official_receipt_root(receipt_root: Path) -> None:
    if not isinstance(receipt_root, Path) or receipt_root != _OFFICIAL_RECEIPT_ROOT:
        raise ValueError("official receipt root differs from frozen durable directory")


_OFFICIAL_AUTHORITY_PATHS = {
    "model_specs_path": "docs/foundation_model_specs.json",
    "fixture_authority_path": "docs/foundation_native_fixtures.json",
    "tolerance_authority_path": "docs/foundation_metric_tolerances.json",
    "published_register_path": "docs/foundation_published_metric_register.json",
    "test_read_register_path": "docs/foundation_test_read_register.json",
}


def _authenticate_official_authority_paths(
    root: Path,
    *,
    model_specs_path: Path,
    fixture_authority_path: Path,
    tolerance_authority_path: Path,
    published_register_path: Path,
    test_read_register_path: Path,
) -> None:
    supplied = {
        "model_specs_path": model_specs_path,
        "fixture_authority_path": fixture_authority_path,
        "tolerance_authority_path": tolerance_authority_path,
        "published_register_path": published_register_path,
        "test_read_register_path": test_read_register_path,
    }
    labels = {
        "model_specs_path": "model-spec authority",
        "fixture_authority_path": "fixture authority",
        "tolerance_authority_path": "tolerance authority",
        "published_register_path": "published-metric authority",
        "test_read_register_path": "test-read authority",
    }
    for name, relative in _OFFICIAL_AUTHORITY_PATHS.items():
        path = supplied[name]
        expected = root / relative
        if path.is_symlink() or not path.is_file() or path.resolve() != expected.resolve():
            raise ValueError(f"official {labels[name]} path differs")


def _authenticate_official_executing_sources(
    root: Path,
    *,
    executing_revision: str,
    foundation_source_path: Path | None = None,
    cli_source_path: Path | None = None,
) -> None:
    if _GIT_REVISION.fullmatch(executing_revision) is None:
        raise ValueError("official executing revision differs")
    cli_spec = importlib.util.find_spec("sfora.cli")
    if cli_spec is None or cli_spec.origin is None:
        raise ValueError("official executing src/sfora/cli.py path differs")
    raw_actual_paths = {
        "src/sfora/foundation_pareto.py": (
            Path(__file__) if foundation_source_path is None else foundation_source_path
        ),
        "src/sfora/cli.py": (Path(cli_spec.origin) if cli_source_path is None else cli_source_path),
    }
    for relative, raw_actual in raw_actual_paths.items():
        if raw_actual.is_symlink() or not raw_actual.is_file():
            raise ValueError(f"official executing {relative} must be a regular file")
        actual = raw_actual.resolve()
        worktree = root / relative
        if worktree.is_symlink() or not worktree.is_file():
            raise ValueError(f"official {relative} worktree must be a regular file")
        blob = cast(
            bytes,
            _git_capture(root, "show", f"{executing_revision}:{relative}", binary=True),
        )
        if actual.read_bytes() != blob:
            raise ValueError(f"official executing {relative} differs from HEAD")
        if worktree.read_bytes() != blob:
            raise ValueError(f"official {relative} worktree differs from HEAD")


def _load_official_read_binding(
    test_read_register_path: Path,
    published_register_path: Path,
) -> FoundationOfficialReadBinding:
    test_read = _load_strict_json(test_read_register_path)
    published = _load_strict_json(published_register_path)
    _require_ordered_keys(
        test_read,
        (
            "schema_version",
            "status",
            "receipt_root",
            *_OFFICIAL_BINDING_KEYS,
            "records",
        ),
        name="official test-read authority",
    )
    _require_ordered_keys(
        published,
        ("schema_version", "status", *_OFFICIAL_BINDING_KEYS, "records"),
        name="official published-metric authority",
    )
    if test_read["schema_version"] != "foundation-test-read-register-v4":
        raise ValueError("official test-read authority is not v4")
    if published["schema_version"] != "foundation-published-metrics-v4":
        raise ValueError("official published-metric authority is not v4")
    if test_read["status"] != "frozen" or published["status"] != "frozen":
        raise ValueError("official authorities are not frozen")
    test_binding = tuple(test_read[key] for key in _OFFICIAL_BINDING_KEYS)
    published_binding = tuple(published[key] for key in _OFFICIAL_BINDING_KEYS)
    if test_binding != published_binding:
        raise ValueError("official authority bindings differ")
    return FoundationOfficialReadBinding(**{key: test_read[key] for key in _OFFICIAL_BINDING_KEYS})


def _validate_official_pre_read_gate(
    binding: FoundationOfficialReadBinding,
    *,
    validation_seed: int,
    validation_fraction: float,
    decision_sha256: str,
    probe_split_sha256s: Sequence[str],
) -> None:
    if type(binding) is not FoundationOfficialReadBinding:
        raise ValueError("official binding differs from exact type")
    if type(validation_seed) is not int or validation_seed != 0:
        raise ValueError("official validation seed differs from reviewed zero")
    if type(validation_fraction) is not float or validation_fraction != 0.2:
        raise ValueError("official validation fraction differs from reviewed 0.2")
    if decision_sha256 != binding.train_decision_sha256:
        raise ValueError("official train-only decision differs from reviewed decision")
    if len(probe_split_sha256s) != 3:
        raise ValueError("official pre-read gate requires exactly three probe splits")
    if any(value != binding.train_split_sha256 for value in probe_split_sha256s):
        raise ValueError("official train-only split differs from reviewed split")


_OFFICIAL_HANDOFF_PATHS = (
    "docs/foundation_metric_tolerances.json",
    "docs/foundation_published_metric_register.json",
    "docs/foundation_test_read_register.json",
)


def _git_capture(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("official handoff Git authentication failed") from error
    return cast(bytes, completed.stdout) if binary else cast(str, completed.stdout)


def _official_repository_root(source_path: Path) -> Path:
    start = source_path if source_path.is_dir() else source_path.parent
    observed = cast(str, _git_capture(start, "rev-parse", "--show-toplevel")).strip()
    root = Path(observed).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("official repository must be a real directory")
    return root


def _authenticate_official_read_handoff(
    binding: FoundationOfficialReadBinding,
    *,
    executing_revision: str,
    repository: Path | None = None,
) -> dict[str, Any]:
    if type(binding) is not FoundationOfficialReadBinding:
        raise ValueError("official binding differs from exact type")
    if _GIT_REVISION.fullmatch(executing_revision) is None:
        raise ValueError("official executing revision differs")
    root = _official_repository_root(Path(__file__).resolve()) if repository is None else repository
    if root.is_symlink() or not root.is_dir():
        raise ValueError("official repository must be a real directory")
    observed_head = cast(str, _git_capture(root, "rev-parse", "HEAD")).strip()
    if observed_head != executing_revision:
        raise ValueError("official executing revision differs from HEAD")
    attached = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if attached.returncode == 0:
        raise ValueError("official handoff requires a detached HEAD")
    if attached.returncode != 1:
        raise ValueError("official detached HEAD state is not authenticatable")
    parents = cast(
        str, _git_capture(root, "rev-list", "--parents", "-n", "1", executing_revision)
    ).split()
    if parents != [executing_revision, binding.reviewed_source_commit]:
        raise ValueError("official handoff is not a direct reviewed-source child")
    changed_paths = tuple(
        sorted(
            cast(
                str,
                _git_capture(
                    root,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    executing_revision,
                ),
            ).splitlines()
        )
    )
    if changed_paths != _OFFICIAL_HANDOFF_PATHS:
        raise ValueError("official handoff path set differs")
    for name, relative, expected_digest in (
        ("addendum", binding.addendum_path, binding.addendum_sha256),
        ("train report", binding.train_report_path, binding.train_report_sha256),
    ):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"official {name} must be a regular file")
        worktree_digest = sha256(path.read_bytes()).hexdigest()
        if worktree_digest != expected_digest:
            raise ValueError(f"official {name} worktree digest differs")
        blob = cast(
            bytes,
            _git_capture(root, "show", f"{executing_revision}:{relative}", binary=True),
        )
        if sha256(blob).hexdigest() != expected_digest:
            raise ValueError(f"official {name} Git blob digest differs")
    dirty = cast(
        str,
        _git_capture(root, "status", "--porcelain", "--untracked-files=no"),
    )
    if dirty:
        raise ValueError("official handoff tracked worktree is dirty")
    report = load_foundation_screen_report(root / binding.train_report_path)
    if report["overall_status"] != "CONTINUE":
        raise ValueError("official train-only report did not continue")
    if report["decision_sha256"] != binding.train_decision_sha256:
        raise ValueError("official train-only report decision differs")
    if report["official_test_reads"] != [] or report["published_metric_audits"] != []:
        raise ValueError("official train-only report already contains official evidence")
    splits = tuple(row["split_sha256"] for row in report["probe_audits"])
    if not splits or any(value != binding.train_split_sha256 for value in splits):
        raise ValueError("official train-only report split differs")
    return report


def load_foundation_model_specs(path: Path) -> tuple[FoundationScreenArmSpec, ...]:
    """Load the frozen ordered candidate/comparator authority for one F0/F1 screen."""

    value = _load_strict_json(path)
    _require_ordered_keys(
        value,
        ("schema_version", "status", "arms"),
        name="foundation model authority",
    )
    if value["schema_version"] != "foundation-model-specs-v1":
        raise ValueError("foundation model schema version differs")
    if value["status"] != "frozen":
        raise ValueError("foundation model authority is not frozen")
    if type(value["arms"]) is not list or not value["arms"]:
        raise ValueError("foundation model arms must be a nonempty JSON array")
    remote_keys = (
        "arm",
        "model_id",
        "revision",
        "weight_sha256",
        "processor_sha256",
        "config_sha256",
        "pooling",
        "resolution",
        "embedding_width",
        "license",
        "dtype",
        "normalize",
    )
    local_keys = (
        "arm",
        "checkpoint_path",
        "pretrained_backbone_path",
        "checkpoint_sha256",
        "resolved_config_sha256",
        "pretrained_backbone_sha256",
        "transform_id",
        "embedding_width",
        "pooling",
        "dtype",
        "normalize",
    )
    arms: list[FoundationScreenArmSpec] = []
    for raw in value["arms"]:
        if type(raw) is not dict:
            raise ValueError("foundation model arm must be a JSON object")
        _require_ordered_keys(
            raw,
            ("kind", "spec", "cache_resolution", "role"),
            name="foundation model arm",
        )
        raw_spec = raw["spec"]
        if type(raw_spec) is not dict:
            raise ValueError("foundation model spec must be a JSON object")
        if raw["kind"] == "remote":
            _require_ordered_keys(raw_spec, remote_keys, name="remote model spec")
            spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec = (
                RemoteFoundationModelSpec(**raw_spec)
            )
        elif raw["kind"] == "local":
            _require_ordered_keys(raw_spec, local_keys, name="local model spec")
            local_value = dict(raw_spec)
            local_value["checkpoint_path"] = Path(local_value["checkpoint_path"])
            local_value["pretrained_backbone_path"] = Path(local_value["pretrained_backbone_path"])
            spec = LocalCheckpointFoundationSpec(**local_value)
        else:
            raise ValueError("foundation model arm kind differs from exact choices")
        arms.append(
            FoundationScreenArmSpec(
                kind=raw["kind"],
                spec=spec,
                cache_resolution=raw["cache_resolution"],
                role=raw["role"],
            )
        )
    arm_names = tuple(arm.spec.arm for arm in arms)
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("foundation model arm names must be unique")
    if sum(arm.role == "comparator" for arm in arms) != 1:
        raise ValueError("foundation model authority requires exactly one comparator")
    if next(arm for arm in arms if arm.role == "comparator").kind != "local":
        raise ValueError("foundation model comparator must be the local anchor")
    if not any(arm.role == "candidate" for arm in arms):
        raise ValueError("foundation model authority requires at least one candidate")
    roles = tuple(arm.role for arm in arms)
    if "contaminated_control" in roles:
        if roles != ("candidate", "comparator", "contaminated_control"):
            raise ValueError("foundation model arm role order differs")
        if arms[2].kind != "local":
            raise ValueError("foundation contaminated control must be local")
    return tuple(arms)


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
    import torch

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
        "embedding_head_init": "kaiming_normal",
        "embedding_layer_norm": False,
    }
    if expected_arch != supported_arch:
        raise ValueError("checkpoint architecture is unsupported by the local comparator")
    model = _build_local_bn_inception(
        embedding_size=spec.embedding_width,
        add_gmp=True,
    )
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("local checkpoint state_dict must be a mapping")
    training_only = ("metric_proxies", "metric_proxy_labels")
    if any(name not in state_dict for name in training_only):
        raise ValueError("local checkpoint lacks registered proxy training state")
    proxies = state_dict["metric_proxies"]
    proxy_labels = state_dict["metric_proxy_labels"]
    if (
        not torch.is_tensor(proxies)
        or proxies.dtype != torch.float32
        or proxies.ndim != 2
        or proxies.shape[1] != spec.embedding_width
        or not bool(torch.isfinite(proxies).all())
    ):
        raise ValueError("local checkpoint metric proxies differ from registered shape/type")
    if (
        not torch.is_tensor(proxy_labels)
        or proxy_labels.dtype != torch.int64
        or proxy_labels.ndim != 1
        or proxy_labels.shape[0] != proxies.shape[0]
    ):
        raise ValueError("local checkpoint proxy labels differ from registered shape/type")
    encoder_state = {name: value for name, value in state_dict.items() if name not in training_only}
    model.load_state_dict(encoder_state, strict=True)
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


_FOUNDATION_REPORT_KEYS = (
    "schema_version",
    "source_commit",
    "dataset",
    "registered_arms",
    "stage_order",
    "encoder_audits",
    "fixture_fidelity_audits",
    "cache_records",
    "cost_profiles",
    "probe_audits",
    "f1_decisions",
    "overall_status",
    "decision_sha256",
    "official_test_reads",
    "published_metric_audits",
)
_FOUNDATION_STAGE_ORDER = (
    "authenticate",
    "fixture",
    "cache",
    "profile",
    "probe",
    "decision",
    "published",
)
_FOUNDATION_FORBIDDEN_REPORT_KEYS = (
    "student",
    "kernel",
    "distill",
    "compression",
    "adapter_weights",
)


def _source_commit() -> str:
    repository = _official_repository_root(Path(__file__).resolve())
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("foundation source checkout is not authenticatable") from error
    if dirty:
        raise ValueError("foundation source checkout has dirty tracked bytes")
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("source revision is not an immutable Git object ID")
    return revision


def _dataset_rows_sha256(examples: Sequence[ImageExample]) -> str:
    return sha256(
        _canonical_json_bytes(
            [{"example_id": row.example_id, "label": int(row.label)} for row in examples]
        )
    ).hexdigest()


def _prepare_foundation_train_cache(
    *,
    arm_spec: FoundationScreenArmSpec,
    encoder: Any,
    train_examples: Sequence[ImageExample],
    cache_dir: Path,
    dataset: str,
) -> FoundationTrainCacheResult:
    import numpy as np

    ids = tuple(row.example_id for row in train_examples)
    labels = tuple(str(int(row.label)) for row in train_examples)
    key = EmbeddingCacheKeyV2.from_model_spec(
        arm_spec.spec,
        dataset_rows_sha256=_dataset_rows_sha256(train_examples),
        split=f"{dataset}:train",
        resolution=(arm_spec.cache_resolution if arm_spec.kind == "local" else None),
    )
    path = key.cache_path(cache_dir)
    if path.exists() or path.is_symlink():
        embeddings = load_embeddings_v2(
            path,
            key=key,
            expected_ids=ids,
            expected_labels=labels,
        )
        status = "reused"
    else:
        images = [materialize_image(row.image) for row in train_examples]
        embeddings = np.asarray(
            encoder.encode(
                images,
                batch_size=min(32, len(images)),
                normalize_embeddings=arm_spec.spec.normalize,
            ),
            dtype=np.float32,
        )
        export_embeddings_v2(
            path,
            key=key,
            embeddings=embeddings,
            ids=ids,
            labels=labels,
        )
        status = "exported"
    embedding_bytes = _embedding_bytes(embeddings, storage_dtype="float32")
    return FoundationTrainCacheResult(
        train_embeddings={example_id: embeddings[index] for index, example_id in enumerate(ids)},
        records=(
            {
                "arm": arm_spec.spec.arm,
                "split": "train",
                "status": status,
                "path": str(path),
                "rows": len(ids),
                "embedding_sha256": sha256(embedding_bytes).hexdigest(),
            },
        ),
    )


def _repository_fixture_metric(
    encoder: Any,
    input_path: Path,
    source_path: Path,
    metric: str,
) -> float:
    import numpy as np

    del source_path
    if metric != "embedding_cosine":
        raise ValueError(f"unsupported foundation fixture metric {metric}")
    value = _load_strict_json(input_path)
    _require_ordered_keys(
        value,
        ("schema_version", "image_paths", "reference_embedding"),
        name="embedding fixture",
    )
    if value["schema_version"] != "foundation-embedding-fixture-v1":
        raise ValueError("embedding fixture schema version differs")
    if type(value["image_paths"]) is not list or not value["image_paths"]:
        raise ValueError("embedding fixture image_paths must be a nonempty array")
    reference = np.asarray(value["reference_embedding"], dtype=np.float64)
    images = [materialize_image(input_path.parent / Path(name)) for name in value["image_paths"]]
    observed = np.asarray(
        encoder.encode(
            images,
            batch_size=len(images),
            normalize_embeddings=encoder.spec.normalize,
        ),
        dtype=np.float64,
    )
    if reference.shape != observed.shape or reference.ndim != 2:
        raise ValueError("embedding fixture reference shape differs")
    denominator = float(np.linalg.norm(reference.ravel()) * np.linalg.norm(observed.ravel()))
    if denominator == 0.0:
        raise ValueError("embedding fixture cosine denominator is zero")
    result = float(np.dot(reference.ravel(), observed.ravel()) / denominator)
    if not isfinite(result):
        raise ValueError("embedding fixture cosine is nonfinite")
    return result


def _serialize_probe(probe: BiasFreeProbeResult) -> dict[str, object]:
    return {
        "arm": probe.arm_key,
        "dataset": probe.dataset,
        "input_dim": probe.input_dim,
        "output_dim": probe.output_dim,
        "split_seed": probe.split_seed,
        "split_sha256": probe.split_sha256,
        "protocol_id": probe.protocol_id,
        "selection_evaluations": probe.selection_evaluations,
        "selected_learning_rate": probe.selected_learning_rate,
        "selected_temperature": probe.selected_temperature,
        "selected_epoch": probe.selected_epoch,
        "validation_recall_at_1_points": probe.validation_recall_at_1_points,
        "weight_sha256": probe.weight_sha256,
        "parameter_count": probe.parameter_count,
        "device_type": probe.device_type,
        "device_name": probe.device_name,
    }


def _decision_payload(
    decisions: Sequence[dict[str, object]],
    overall_status: str,
) -> dict[str, object]:
    return {"f1_decisions": list(decisions), "overall_status": overall_status}


def _decision_sha256(decisions: Sequence[dict[str, object]], overall_status: str) -> str:
    return sha256(_canonical_json_bytes(_decision_payload(decisions, overall_status))).hexdigest()


def _walk_report_keys(value: object) -> Sequence[str]:
    keys: list[str] = []
    if type(value) is dict:
        for key, item in value.items():
            keys.append(key)
            keys.extend(_walk_report_keys(item))
    elif type(value) is list:
        for item in value:
            keys.extend(_walk_report_keys(item))
    return keys


def _json_ready(value: object) -> object:
    if type(value) is dict:
        mapping = cast(Mapping[str, object], value)
        return {key: _json_ready(item) for key, item in mapping.items()}
    if type(value) in {list, tuple}:
        return [_json_ready(item) for item in cast(Sequence[object], value)]
    if isinstance(value, Path):
        return str(value)
    return value


def validate_foundation_screen_report(value: object) -> None:
    if type(value) is not dict:
        raise ValueError("foundation screen report must be a JSON object")
    _require_ordered_keys(value, _FOUNDATION_REPORT_KEYS, name="foundation screen report")
    if value["schema_version"] != "foundation-screen-report-v2":
        raise ValueError("foundation screen report schema version differs")
    if (
        type(value["source_commit"]) is not str
        or _GIT_REVISION.fullmatch(value["source_commit"]) is None
    ):
        raise ValueError("foundation screen source_commit differs")
    _require_foundation_dataset(value["dataset"])
    arms = value["registered_arms"]
    if (
        type(arms) is not list
        or not arms
        or any(type(arm) is not str or not arm for arm in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError("foundation screen registered_arms differ")
    if value["stage_order"] != list(_FOUNDATION_STAGE_ORDER):
        raise ValueError("foundation screen stage order differs")
    for name in (
        "encoder_audits",
        "fixture_fidelity_audits",
        "cache_records",
        "cost_profiles",
        "probe_audits",
        "f1_decisions",
        "official_test_reads",
        "published_metric_audits",
    ):
        if type(value[name]) is not list:
            raise ValueError(f"foundation screen {name} must be a JSON array")
    if value["overall_status"] not in {
        "CONTINUE",
        "CLOSE_FOUNDATION_TRANSFER",
        "UNAVAILABLE_COMPARATOR",
        "INVALID_SPLIT_POWER",
        "BOUNDARY_REPLICATION_REQUIRED",
    }:
        raise ValueError("foundation screen overall_status differs")
    expected_digest = _decision_sha256(value["f1_decisions"], value["overall_status"])
    if value["decision_sha256"] != expected_digest:
        raise ValueError("foundation screen decision digest differs")
    for row in value["official_test_reads"]:
        if type(row) is not dict or row.get("decision_sha256") != expected_digest:
            raise ValueError("official test read is not bound to the frozen decision")
    for key in _walk_report_keys(value):
        normalized = key.lower()
        if any(token in normalized for token in _FOUNDATION_FORBIDDEN_REPORT_KEYS):
            raise ValueError(f"foundation screen contains forbidden report key {key}")
    arm_set = set(arms)
    encoder_arms: list[str] = []
    for row in value["encoder_audits"]:
        _require_report_row(
            row,
            ("arm", "status", "audit", "reason"),
            name="encoder audit",
        )
        arm = row["arm"]
        if arm not in arm_set:
            raise ValueError("encoder audit arm differs from registered arms")
        encoder_arms.append(arm)
        if row["status"] == "available":
            if type(row["audit"]) is not dict or row["reason"] is not None:
                raise ValueError("available encoder audit relation differs")
            audit = row["audit"]
            remote_keys = tuple(FoundationEncoderAudit.__dataclass_fields__)
            local_keys = tuple(LocalFoundationEncoderAudit.__dataclass_fields__)
            if tuple(audit) == remote_keys:
                if audit["status"] != "available":
                    raise ValueError("available remote encoder audit status differs")
                _require_nonempty("encoder audit model_id", audit["model_id"])
                if (
                    type(audit["revision"]) is not str
                    or _GIT_REVISION.fullmatch(audit["revision"]) is None
                    or audit["reason"] is not None
                ):
                    raise ValueError("available remote encoder audit relation differs")
                for name in ("weight_sha256", "processor_sha256", "config_sha256"):
                    _require_sha256(f"encoder audit {name}", audit[name])
            elif tuple(audit) == local_keys:
                for name in local_keys:
                    _require_sha256(f"encoder audit {name}", audit[name])
            else:
                raise ValueError("available encoder audit schema differs")
        elif row["status"] == "unavailable":
            if row["audit"] is not None or type(row["reason"]) is not str or not row["reason"]:
                raise ValueError("unavailable encoder audit relation differs")
        else:
            raise ValueError("encoder audit status differs")
    if encoder_arms != arms:
        raise ValueError("encoder audit order differs from registered arms")
    for row in value["fixture_fidelity_audits"]:
        _require_report_row(
            row,
            (
                "arm",
                "metric",
                "native_value",
                "repository_value",
                "tolerance",
                "provenance",
                "passed",
            ),
            name="fixture fidelity audit",
        )
        _require_report_arm(row, arm_set, name="fixture fidelity audit")
        repository_value = _report_float(row["repository_value"], "fixture repository_value")
        tolerance = _report_float(row["tolerance"], "fixture tolerance")
        if tolerance < 0.0:
            raise ValueError("fixture tolerance must be nonnegative")
        if row["provenance"] == "native_cross_check":
            native_value = _report_float(row["native_value"], "fixture native_value")
            expected_passed = abs(repository_value - native_value) <= tolerance
            if type(row["passed"]) is not bool or row["passed"] is not expected_passed:
                raise ValueError("fixture fidelity decision differs")
        elif row["provenance"] == "repository_replay":
            if row["native_value"] is not None:
                raise ValueError("repository replay cannot carry a native value")
            expected_passed = abs(repository_value - 1.0) <= tolerance
            if type(row["passed"]) is not bool or row["passed"] is not expected_passed:
                raise ValueError("repository replay decision differs")
        elif row["provenance"] == "unavailable":
            if row["native_value"] is not None or row["passed"] is not None:
                raise ValueError("unavailable fixture fidelity relation differs")
        else:
            raise ValueError("fixture fidelity provenance differs")
    _require_registered_arm_order(
        value["fixture_fidelity_audits"], arms, name="fixture fidelity audit"
    )
    for row in value["cache_records"]:
        _require_report_row(
            row,
            ("arm", "split", "status", "path", "rows", "embedding_sha256"),
            name="cache record",
        )
        _require_report_arm(row, arm_set, name="cache record")
        if row["status"] not in {"exported", "reused"}:
            raise ValueError("cache status differs")
        if type(row["split"]) is not str or not row["split"]:
            raise ValueError("cache split differs")
        if type(row["path"]) is not str or not row["path"]:
            raise ValueError("cache path differs")
        _require_positive_int("cache rows", row["rows"])
        _require_sha256("cache embedding_sha256", row["embedding_sha256"])
    _require_registered_arm_order(value["cache_records"], arms, name="cache record")
    for row in value["cost_profiles"]:
        _require_report_row(row, ("arm", "profile"), name="cost profile")
        _require_report_arm(row, arm_set, name="cost profile")
        profile = row["profile"]
        _require_report_row(
            profile,
            tuple(EncoderCostProfile.__dataclass_fields__),
            name="cost profile value",
        )
        if type(profile["batches"]) is not list:
            raise ValueError("cost profile batches must be a JSON array")
        for batch in profile["batches"]:
            _require_report_row(
                batch,
                tuple(EncoderBatchCost.__dataclass_fields__),
                name="cost profile batch",
            )
    probe_arms: list[str] = []
    probe_keys = (
        "arm",
        "dataset",
        "input_dim",
        "output_dim",
        "split_seed",
        "split_sha256",
        "protocol_id",
        "selection_evaluations",
        "selected_learning_rate",
        "selected_temperature",
        "selected_epoch",
        "validation_recall_at_1_points",
        "weight_sha256",
        "parameter_count",
        "device_type",
        "device_name",
    )
    for row in value["probe_audits"]:
        _require_report_row(row, probe_keys, name="probe audit")
        _require_report_arm(row, arm_set, name="probe audit")
        probe_arms.append(row["arm"])
        _require_sha256("probe split_sha256", row["split_sha256"])
        _require_sha256("probe weight_sha256", row["weight_sha256"])
        _require_points("probe validation_recall_at_1_points", row["validation_recall_at_1_points"])
    if len(set(probe_arms)) != len(probe_arms):
        raise ValueError("probe audit arms must be unique")
    _require_registered_arm_order(value["probe_audits"], arms, name="probe audit")
    decision_keys = ("arm",) + tuple(F1Decision.__dataclass_fields__)
    for row in value["f1_decisions"]:
        _require_report_row(row, decision_keys, name="F1 decision")
        _require_report_arm(row, arm_set, name="F1 decision")
    if value["overall_status"] == "UNAVAILABLE_COMPARATOR":
        if value["f1_decisions"]:
            raise ValueError("unavailable comparator cannot carry F1 decisions")
        expected_overall = "UNAVAILABLE_COMPARATOR"
    else:
        if not value["f1_decisions"]:
            raise ValueError("available comparator requires a candidate decision")
        decision_statuses = {row["status"] for row in value["f1_decisions"]}
        if "INVALID_SPLIT_POWER" in decision_statuses:
            expected_overall = "INVALID_SPLIT_POWER"
        elif "BOUNDARY_REPLICATION_REQUIRED" in decision_statuses:
            expected_overall = "BOUNDARY_REPLICATION_REQUIRED"
        elif "CONTINUE" in decision_statuses:
            expected_overall = "CONTINUE"
        else:
            expected_overall = "CLOSE_FOUNDATION_TRANSFER"
    if value["overall_status"] != expected_overall:
        raise ValueError("foundation screen aggregate decision differs")
    if value["overall_status"] != "CONTINUE" and (
        value["official_test_reads"] or value["published_metric_audits"]
    ):
        raise ValueError("non-continuing foundation screen cannot carry official test evidence")
    official_keys = (
        "dataset",
        "arm",
        "model_revision",
        "checkpoint_sha256",
        "metrics",
        "purpose",
        "evaluation_number",
        "decision_sha256",
        "receipt_path",
        "selected_geometry",
        "geometry_evaluations",
    )
    for row in value["official_test_reads"]:
        _require_report_row(row, official_keys, name="official test read")
        if row["dataset"] != value["dataset"]:
            raise ValueError("official test read dataset differs from report")
        _require_report_arm(row, arm_set, name="official test read")
        if (
            type(row["model_revision"]) is not str
            or _GIT_REVISION.fullmatch(row["model_revision"]) is None
        ):
            raise ValueError("official model_revision differs")
        _require_sha256("official checkpoint_sha256", row["checkpoint_sha256"])
        if row["metrics"] != list(FOUNDATION_PUBLISHED_METRICS):
            raise ValueError("official metrics differ from registered order")
        if row["purpose"] != FOUNDATION_TEST_READ_PURPOSE:
            raise ValueError("official purpose differs")
        if type(row["evaluation_number"]) is not int or row["evaluation_number"] != 1:
            raise ValueError("official evaluation number differs")
        if type(row["receipt_path"]) is not str or not row["receipt_path"]:
            raise ValueError("official receipt path differs")
        if row["selected_geometry"] not in {
            "normalized_cosine",
            "native_unnormalized_euclidean",
        }:
            raise ValueError("official selected geometry differs")
        geometries = row["geometry_evaluations"]
        expected_geometries = (
            "normalized_cosine",
            "normalized_euclidean",
            "native_unnormalized_euclidean",
        )
        if type(geometries) is not list or len(geometries) != 3:
            raise ValueError("official geometry evaluations differ")
        for geometry, expected_geometry in zip(geometries, expected_geometries, strict=True):
            _require_report_row(geometry, ("geometry", "metrics"), name="geometry evaluation")
            if geometry["geometry"] != expected_geometry or type(geometry["metrics"]) is not dict:
                raise ValueError("official geometry evaluation differs")
            metric_keys = (
                "precision_at_1",
                "recall_at_1",
                "recall_at_2",
                "recall_at_4",
                "recall_at_8",
                "map_at_r",
                "mean_relevant_items",
                "evaluated_queries",
                "total_queries",
                "recall_at_10",
                "recall_at_20",
                "recall_at_30",
                "recall_at_100",
            )
            _require_ordered_keys(geometry["metrics"], metric_keys, name="geometry metrics")
    published_keys = tuple(PublishedMetricAudit.__dataclass_fields__)
    for row in value["published_metric_audits"]:
        _require_report_row(row, published_keys, name="published metric audit")
        _require_report_arm(row, arm_set, name="published metric audit")
        repository_value = _report_float(row["repository_value"], "published repository_value")
        if row["provenance"] == "native_cross_check":
            native_value = _report_float(row["native_value"], "published native_value")
            tolerance = _report_float(row["tolerance"], "published tolerance")
            expected_passed = abs(repository_value - native_value) <= tolerance
            if type(row["passed"]) is not bool or row["passed"] is not expected_passed:
                raise ValueError("published metric decision differs")
        elif row["provenance"] == "repository_only":
            if row["native_value"] is not None or row["tolerance"] is not None:
                raise ValueError("repository-only published relation differs")
            if row["passed"] is not None:
                raise ValueError("repository-only published decision must be null")
        else:
            raise ValueError("published metric provenance differs")
        if row["invalidates_confirmatory_claim"] is not (row["passed"] is False):
            raise ValueError("published metric invalidation relation differs")


def _require_report_row(value: object, keys: tuple[str, ...], *, name: str) -> None:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    _require_ordered_keys(value, keys, name=name)


def _require_report_arm(value: dict[str, Any], arms: set[str], *, name: str) -> None:
    if value["arm"] not in arms:
        raise ValueError(f"{name} arm differs from registered arms")


def _require_registered_arm_order(
    rows: Sequence[dict[str, Any]], registered_arms: Sequence[str], *, name: str
) -> None:
    positions = {arm: index for index, arm in enumerate(registered_arms)}
    observed = [positions[row["arm"]] for row in rows]
    if observed != sorted(observed):
        raise ValueError(f"{name} order differs from registered arms")


def _report_float(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite builtin float")
    return value


def load_foundation_screen_report(path: Path) -> dict[str, Any]:
    value = _load_strict_json(path)
    validate_foundation_screen_report(value)
    return value


def publish_foundation_screen_report(path: Path, payload: dict[str, object]) -> Path:
    validate_foundation_screen_report(payload)
    _publish_json_no_clobber(path, payload)
    persisted = load_foundation_screen_report(path)
    if persisted != payload:
        raise ValueError("persisted foundation screen report differs")
    return path


def _foundation_execution_arms(
    arms: Sequence[FoundationScreenArmSpec],
) -> tuple[FoundationScreenArmSpec, ...]:
    priority = {"comparator": 0, "contaminated_control": 1, "candidate": 2}
    return tuple(sorted(arms, key=lambda arm_spec: priority[arm_spec.role]))


def _foundation_official_read_arms(
    arms: Sequence[FoundationScreenArmSpec],
) -> tuple[FoundationScreenArmSpec, ...]:
    return tuple(arm for arm in arms if arm.role != "contaminated_control")


def run_foundation_screen(
    *,
    dataset: str,
    dataset_root: Path,
    model_specs_path: Path,
    cache_dir: Path,
    report_path: Path,
    fixture_authority_path: Path,
    tolerance_authority_path: Path,
    published_register_path: Path,
    test_read_register_path: Path,
    validation_seed: int,
    validation_fraction: float,
    allow_registered_test_read: bool,
) -> Path:
    """Run F0/F1 using train identities only until a durable official-read receipt exists."""

    _require_foundation_dataset(dataset)
    if type(validation_seed) is not int or validation_seed < 0:
        raise ValueError("validation_seed must be a nonnegative builtin integer")
    if type(validation_fraction) is not float or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be a builtin float in (0, 1)")
    if type(allow_registered_test_read) is not bool:
        raise ValueError("allow_registered_test_read must be a builtin boolean")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from registered :4096:8")
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError("foundation cache directory must be a real directory")
    if report_path.parent.is_symlink() or not report_path.parent.is_dir():
        raise ValueError("foundation report parent must be a real directory")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(report_path)
    source_commit = _source_commit()
    official_binding: FoundationOfficialReadBinding | None = None
    if allow_registered_test_read:
        official_root = _official_repository_root(Path(__file__).resolve())
        _authenticate_official_executing_sources(
            official_root,
            executing_revision=source_commit,
        )
        _authenticate_official_authority_paths(
            official_root,
            model_specs_path=model_specs_path,
            fixture_authority_path=fixture_authority_path,
            tolerance_authority_path=tolerance_authority_path,
            published_register_path=published_register_path,
            test_read_register_path=test_read_register_path,
        )
        official_binding = _load_official_read_binding(
            test_read_register_path,
            published_register_path,
        )
        _validate_official_binding_constants(official_binding)
        _authenticate_official_read_handoff(
            official_binding,
            executing_revision=source_commit,
        )
        if validation_seed != 0:
            raise ValueError("official validation seed differs from reviewed zero")
        if validation_fraction != 0.2:
            raise ValueError("official validation fraction differs from reviewed 0.2")
    arms = load_foundation_model_specs(model_specs_path)
    registered_arms = tuple(arm.spec.arm for arm in arms)
    fixture_pairs = tuple(
        (arm, metric) for arm in registered_arms for metric in FOUNDATION_FIXTURE_METRICS
    )
    fixtures, tolerances = load_native_fixture_authority(
        fixture_authority_path,
        tolerance_authority_path,
        registered_pairs=fixture_pairs,
    )
    test_reads = load_test_read_register(test_read_register_path)
    if allow_registered_test_read:
        _validate_official_receipt_root(test_reads.receipt_root)
        _validate_registered_receipt_root(
            receipt_root=test_reads.receipt_root,
            ledger=test_reads,
            cache_dir=cache_dir,
        )
    train_examples = load_image_retrieval_examples(
        dataset_name=cast(ImageDatasetName, dataset),
        split="train",
        seed=validation_seed,
        dataset_root=dataset_root,
    )
    split = build_identity_disjoint_validation_split(
        train_examples,
        fraction=validation_fraction,
        seed=validation_seed,
    )
    config = ProbeTrainingConfig()
    encoders: dict[str, Any] = {}
    probes: dict[str, BiasFreeProbeResult] = {}
    profiles: dict[str, EncoderCostProfile] = {}
    encoder_rows_by_arm: dict[str, dict[str, object]] = {}
    fixture_rows_by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in registered_arms}
    cache_rows_by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in registered_arms}
    probe_rows_by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in registered_arms}
    execution_arms = _foundation_execution_arms(arms)
    comparator_unavailable = False
    for arm_spec in execution_arms:
        arm = arm_spec.spec.arm
        try:
            encoder = load_foundation_encoder(arm_spec.spec)
        except (OSError, RuntimeError, ValueError) as error:
            encoder_rows_by_arm[arm] = {
                "arm": arm,
                "status": "unavailable",
                "audit": None,
                "reason": str(error),
            }
            if arm_spec.role == "comparator":
                comparator_unavailable = True
                break
            continue
        encoders[arm] = encoder
        audit = getattr(encoder, "audit", None)
        if not isinstance(audit, FoundationEncoderAudit | LocalFoundationEncoderAudit):
            raise ValueError("foundation encoder lacks an exact source audit")
        encoder_rows_by_arm[arm] = {
            "arm": arm,
            "status": "available",
            "audit": asdict(audit),
            "reason": None,
        }
        fixture_inputs = {
            metric: fixture_authority_path.parent
            / "foundation_native_inputs"
            / f"{arm}__{metric}.json"
            for metric in FOUNDATION_FIXTURE_METRICS
        }
        native_sources = {
            metric: fixture_authority_path.parent
            / "foundation_native_sources"
            / f"{arm}__{metric}.py"
            for metric in FOUNDATION_FIXTURE_METRICS
        }
        fidelity = verify_native_fixture(
            arm=arm,
            encoder=encoder,
            fixture_inputs=fixture_inputs,
            native_sources=native_sources,
            repository_metric=_repository_fixture_metric,
            fixtures=fixtures,
            tolerances=tolerances,
            registered_pairs=fixture_pairs,
        )
        fixture_rows_by_arm[arm].extend(asdict(row) for row in fidelity)
        if any(row.passed is False for row in fidelity):
            raise ValueError("foundation fixture fidelity replay failed")
        cached = _prepare_foundation_train_cache(
            arm_spec=arm_spec,
            encoder=encoder,
            train_examples=train_examples,
            cache_dir=cache_dir,
            dataset=dataset,
        )
        cache_rows_by_arm[arm].extend(cached.records)
        profile = profile_foundation_encoder(
            encoder,
            [materialize_image(row.image) for row in train_examples[:32]],
        )
        profiles[arm] = profile
        probe = fit_bias_free_probe_512(
            cached.train_embeddings,
            split,
            arm_key=arm,
            dataset=dataset,
            split_seed=validation_seed,
            config=config,
        )
        probes[arm] = probe
        probe_rows_by_arm[arm].append(_serialize_probe(probe))
    if comparator_unavailable:
        for arm_spec in arms:
            arm = arm_spec.spec.arm
            encoder_rows_by_arm.setdefault(
                arm,
                {
                    "arm": arm,
                    "status": "unavailable",
                    "audit": None,
                    "reason": "not attempted because the registered comparator is unavailable",
                },
            )
    encoder_rows = [encoder_rows_by_arm[arm] for arm in registered_arms]
    comparator_arm = next(arm.spec.arm for arm in arms if arm.role == "comparator")
    comparator_probe = probes.get(comparator_arm)
    decisions: list[dict[str, object]] = []
    if comparator_probe is None:
        overall_status = "UNAVAILABLE_COMPARATOR"
    else:
        for arm_spec in arms:
            if arm_spec.role != "candidate" or arm_spec.spec.arm not in probes:
                continue
            arm = arm_spec.spec.arm
            decision = decide_f1(
                candidate_probe=probes[arm],
                comparator_probe=comparator_probe,
                candidate_encoder_p95_ms=_profile_batch_one_p95(profiles[arm]),
                comparator_encoder_p95_ms=_profile_batch_one_p95(profiles[comparator_arm]),
                candidate_descriptor_bytes_per_image=probes[arm].output_dim * 4,
                comparator_descriptor_bytes_per_image=comparator_probe.output_dim * 4,
            )
            decisions.append({"arm": arm, **asdict(decision)})
        if not decisions:
            raise ValueError("available comparator requires at least one candidate decision")
        decision_statuses = {row["status"] for row in decisions}
        if "INVALID_SPLIT_POWER" in decision_statuses:
            overall_status = "INVALID_SPLIT_POWER"
        elif "BOUNDARY_REPLICATION_REQUIRED" in decision_statuses:
            overall_status = "BOUNDARY_REPLICATION_REQUIRED"
        elif "CONTINUE" in decision_statuses:
            overall_status = "CONTINUE"
        else:
            overall_status = "CLOSE_FOUNDATION_TRANSFER"
    decision_digest = _decision_sha256(decisions, overall_status)
    if official_binding is not None:
        _validate_official_pre_read_gate(
            official_binding,
            validation_seed=validation_seed,
            validation_fraction=validation_fraction,
            decision_sha256=decision_digest,
            probe_split_sha256s=tuple(
                probes[arm].split_sha256 for arm in registered_arms if arm in probes
            ),
        )
    official_rows: list[dict[str, object]] = []
    published_rows: list[dict[str, object]] = []
    if allow_registered_test_read and overall_status == "CONTINUE":
        receipt_root = test_reads.receipt_root
        if receipt_root.resolve() == cache_dir.resolve():
            raise ValueError("official test receipts cannot use the rebuildable cache directory")
        official_arms = _foundation_official_read_arms(arms)
        official_rows, published_rows, official_cache_rows = _run_registered_official_reads(
            dataset=dataset,
            dataset_root=dataset_root,
            validation_seed=validation_seed,
            arms=official_arms,
            encoders={
                arm.spec.arm: encoders[arm.spec.arm]
                for arm in official_arms
                if arm.spec.arm in probes
            },
            cache_dir=cache_dir,
            receipt_root=receipt_root,
            decision_sha256=decision_digest,
            ledger=test_reads,
            published_register_path=published_register_path,
        )
        for row in official_cache_rows:
            cache_rows_by_arm[cast(str, row["arm"])].append(row)
    fixture_rows = [row for arm in registered_arms for row in fixture_rows_by_arm[arm]]
    cache_rows = [row for arm in registered_arms for row in cache_rows_by_arm[arm]]
    probe_rows = [row for arm in registered_arms for row in probe_rows_by_arm[arm]]
    payload: dict[str, object] = {
        "schema_version": "foundation-screen-report-v2",
        "source_commit": source_commit,
        "dataset": dataset,
        "registered_arms": list(registered_arms),
        "stage_order": list(_FOUNDATION_STAGE_ORDER),
        "encoder_audits": encoder_rows,
        "fixture_fidelity_audits": fixture_rows,
        "cache_records": cache_rows,
        "cost_profiles": [
            {"arm": arm, "profile": _json_ready(asdict(profiles[arm]))}
            for arm in registered_arms
            if arm in profiles
        ],
        "probe_audits": probe_rows,
        "f1_decisions": decisions,
        "overall_status": overall_status,
        "decision_sha256": decision_digest,
        "official_test_reads": official_rows,
        "published_metric_audits": published_rows,
    }
    return publish_foundation_screen_report(report_path, payload)


def _profile_batch_one_p95(profile: EncoderCostProfile) -> float | None:
    row = next((batch for batch in profile.batches if batch.batch_size == 1), None)
    return row.latency_p95_ms if row is not None else None


def _run_registered_official_reads(
    *,
    dataset: str,
    dataset_root: Path,
    validation_seed: int,
    arms: Sequence[FoundationScreenArmSpec],
    encoders: Mapping[str, Any],
    cache_dir: Path,
    receipt_root: Path,
    decision_sha256: str,
    ledger: FoundationTestReadLedger,
    published_register_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    _validate_registered_receipt_root(
        receipt_root=receipt_root,
        ledger=ledger,
        cache_dir=cache_dir,
    )
    registered_pairs = tuple(
        (arm.spec.arm, metric) for arm in arms for metric in FOUNDATION_PUBLISHED_METRICS
    )
    published_records = load_published_metric_register(published_register_path)
    if _record_keys(published_records) != registered_pairs:
        raise ValueError("published metric register differs from screen arm/metric order")
    official_bundle = preflight_official_image_retrieval_split(
        dataset_name=cast(ImageDatasetName, dataset),
        dataset_root=dataset_root,
    )
    reads: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    evaluated: list[tuple[str, dict[str, float]]] = []
    for arm_spec in arms:
        arm = arm_spec.spec.arm
        if arm not in encoders:
            continue
        model_revision, checkpoint_sha256 = _test_read_identity(arm_spec.spec)
        consumed = ledger.consume(
            dataset=dataset,
            arm=arm,
            model_revision=model_revision,
            checkpoint_sha256=checkpoint_sha256,
            metrics=FOUNDATION_PUBLISHED_METRICS,
            purpose=FOUNDATION_TEST_READ_PURPOSE,
        )
        receipt = publish_official_test_read_receipt(
            receipt_root,
            consumed,
            decision_sha256=decision_sha256,
        )
        bundle = load_registered_official_test(
            receipt,
            dataset=dataset,
            arm=arm,
            metrics=FOUNDATION_PUBLISHED_METRICS,
            loader=lambda: official_bundle,
        )
        repository_values, arm_cache_rows, geometry_rows = _evaluate_official_foundation_arm(
            arm_spec=arm_spec,
            encoder=encoders[arm],
            bundle=bundle,
            cache_dir=cache_dir,
            dataset=dataset,
        )
        reads.append(
            {
                "dataset": dataset,
                "arm": arm,
                "model_revision": model_revision,
                "checkpoint_sha256": checkpoint_sha256,
                "metrics": list(FOUNDATION_PUBLISHED_METRICS),
                "purpose": consumed.purpose,
                "evaluation_number": 1,
                "decision_sha256": decision_sha256,
                "receipt_path": str(receipt.receipt_path),
                "selected_geometry": (
                    "normalized_cosine"
                    if arm_spec.spec.normalize
                    else "native_unnormalized_euclidean"
                ),
                "geometry_evaluations": geometry_rows,
            }
        )
        cache_rows.extend(arm_cache_rows)
        evaluated.append((arm, repository_values))
    for arm, repository_values in evaluated:
        arm_audits = cross_check_published_metrics(
            arm=arm,
            repository_values=repository_values,
            records=published_records,
            registered_pairs=registered_pairs,
        )
        audits.extend(asdict(row) for row in arm_audits)
    return reads, audits, cache_rows


def _validate_registered_receipt_root(
    *,
    receipt_root: Path,
    ledger: FoundationTestReadLedger,
    cache_dir: Path,
) -> None:
    if receipt_root.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError("official test receipts cannot use the rebuildable cache directory")
    if receipt_root != ledger.receipt_root:
        raise ValueError("official test receipt root differs from frozen authority")
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise ValueError("official test receipt root must be a real directory")


def _test_read_identity(
    spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec,
) -> tuple[str, str]:
    if type(spec) is RemoteFoundationModelSpec:
        return spec.revision, spec.weight_sha256
    if type(spec) is LocalCheckpointFoundationSpec:
        return spec.checkpoint_sha256, spec.checkpoint_sha256
    raise ValueError("test-read identity requires an exact foundation model spec")


def _official_split_cache(
    *,
    arm_spec: FoundationScreenArmSpec,
    encoder: Any,
    examples: Sequence[ImageExample],
    cache_dir: Path,
    dataset: str,
    split_name: str,
) -> tuple[Any, dict[str, object]]:
    import numpy as np

    ids = tuple(row.example_id for row in examples)
    labels = tuple(str(int(row.label)) for row in examples)
    key = EmbeddingCacheKeyV2.from_model_spec(
        arm_spec.spec,
        dataset_rows_sha256=_dataset_rows_sha256(examples),
        split=f"{dataset}:{split_name}",
        resolution=(arm_spec.cache_resolution if arm_spec.kind == "local" else None),
        normalize=False,
    )
    path = key.cache_path(cache_dir)
    if path.exists() or path.is_symlink():
        embeddings = load_embeddings_v2(
            path,
            key=key,
            expected_ids=ids,
            expected_labels=labels,
        )
        status = "reused"
    else:
        images = [materialize_image(row.image) for row in examples]
        embeddings = np.asarray(
            encoder.encode(
                images,
                batch_size=min(32, len(images)),
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        export_embeddings_v2(
            path,
            key=key,
            embeddings=embeddings,
            ids=ids,
            labels=labels,
        )
        status = "exported"
    return embeddings, {
        "arm": arm_spec.spec.arm,
        "split": split_name,
        "status": status,
        "path": str(path),
        "rows": len(ids),
        "embedding_sha256": sha256(
            _embedding_bytes(embeddings, storage_dtype="float32")
        ).hexdigest(),
    }


def _evaluate_official_foundation_arm(
    *,
    arm_spec: FoundationScreenArmSpec,
    encoder: Any,
    bundle: Any,
    cache_dir: Path,
    dataset: str,
) -> tuple[dict[str, float], tuple[dict[str, object], ...], list[dict[str, object]]]:
    import numpy as np

    query_embeddings, query_record = _official_split_cache(
        arm_spec=arm_spec,
        encoder=encoder,
        examples=bundle.query,
        cache_dir=cache_dir,
        dataset=dataset,
        split_name="query",
    )
    records: tuple[dict[str, object], ...]
    if bundle.gallery is None:
        gallery_embeddings = query_embeddings
        gallery_examples = bundle.query
        records = (query_record,)
        exclude_self = True
    else:
        gallery_examples = bundle.gallery
        gallery_embeddings, gallery_record = _official_split_cache(
            arm_spec=arm_spec,
            encoder=encoder,
            examples=gallery_examples,
            cache_dir=cache_dir,
            dataset=dataset,
            split_name="gallery",
        )
        records = (query_record, gallery_record)
        exclude_self = False
    geometries = evaluate_foundation_geometries(
        query_embeddings,
        np.asarray([int(row.label) for row in bundle.query], dtype=np.int64),
        gallery_embeddings,
        np.asarray([int(row.label) for row in gallery_examples], dtype=np.int64),
        exclude_self=exclude_self,
    )
    selected_geometry = (
        "normalized_cosine" if arm_spec.spec.normalize else "native_unnormalized_euclidean"
    )
    selected = next(row for row in geometries if row.geometry == selected_geometry)
    metrics = selected.metrics
    return (
        {
            "recall_at_1": float(metrics.recall_at_1),
            "recall_at_10": float(metrics.recall_at_10),
            "recall_at_20": float(metrics.recall_at_20),
            "recall_at_30": float(metrics.recall_at_30),
            "recall_at_100": float(metrics.recall_at_100),
            "map_at_r": float(metrics.map_at_r),
        },
        records,
        [
            {"geometry": row.geometry, "metrics": _json_ready(asdict(row.metrics))}
            for row in geometries
        ],
    )
