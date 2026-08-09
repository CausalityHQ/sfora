"""Capture and derive deterministic authority for the Pass201 ordinary-PA source."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

from pass201_pa_source_v2_contract import (
    TRAIN_MANIFEST_CALL_GRAPH,
    BoundCheckpointMetadata,
    CheckpointMetadata,
    ExternalFileBinding,
    PrelaunchAuthority,
    PrivateChildFrame,
    canonical_json_bytes,
    decode_checkpoint_binding_response,
    decode_checkpoint_metadata_response,
    decode_private_child_frame,
    encode_checkpoint_binding_request,
    encode_checkpoint_metadata_request,
    encode_private_child_frame,
    load_strict_json_bytes,
    load_strict_json_value_bytes,
    validate_prelaunch,
    validate_train_manifest,
)
from typer.main import get_command

import sfora.cli
import sfora.image_end_to_end as image_end_to_end
from sfora.data import ImageExample
from sfora.image_end_to_end import ImageEndToEndConfig

RECIPE_ID = "proxy_anchor.inshop.official-51db570"
RECIPE_DIGEST = "97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361"
EXPECTED_REPORT_KEYS = frozenset(
    {
        "name",
        "dataset_name",
        "protocol",
        "config",
        "train_examples",
        "test_examples",
        "methods",
    }
)
SIDECAR_FRAME_MAGIC = b"pass201-sidecars-v1\0"


@dataclass(frozen=True)
class CapturedAuthority:
    config_bytes: bytes
    recipe_id: str
    recipe_digest: str
    train_count: int
    query_count: int
    gallery_count: int
    protocol: str
    protocol_name: str
    rows: tuple[tuple[int, str, int], ...]
    resolved_membership_sha256: str
    resolved_train_steps: int
    steps_per_epoch: int
    total_epochs: int


@dataclass(frozen=True)
class SidecarFrame:
    pid: int
    config_bytes: bytes
    manifest_bytes: bytes
    config_sha256: str
    manifest_sha256: str


class _CaptureComplete(BaseException):
    """Controller-only unwind that bypasses the production CLI exception handlers."""


class _CaptureRejected(BaseException):
    """Carry a fail-closed capture error through the production CLI handlers."""


def _require(predicate: bool, message: str) -> None:
    if not predicate:
        raise ValueError(message)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_bound_image(
    path: Path, declared_root: Path, resolved_root: Path
) -> dict[str, object]:
    _require(isinstance(path, Path), "optimization image must be a Path")
    lexical = Path(os.path.abspath(path))
    try:
        declared_relative = lexical.relative_to(declared_root)
    except ValueError as exc:
        raise ValueError("optimization image escapes declared image root") from exc
    current = declared_root
    for component in declared_relative.parts:
        current /= component
        _require(not stat.S_ISLNK(os.lstat(current).st_mode), "optimization image is a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("optimization image escapes resolved image root") from exc
    _require(relative == declared_relative, "optimization image path resolution drift")
    _require(relative.parts and ".." not in relative.parts, "invalid optimization image path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), "optimization image is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    _require(
        _stat_identity(before) == _stat_identity(after),
        "optimization image changed during read",
    )
    data = b"".join(chunks)
    _require(len(data) == before.st_size, "optimization image size drift")
    return {
        "relative_path": relative.as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ordered_hash(records: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = canonical_json_bytes(record)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _capture_boundary(
    captured: list[CapturedAuthority], dataset_root: Path
) -> Callable[..., NoReturn]:
    bound_root = dataset_root.resolve(strict=True)
    declared_image_root = bound_root / "Img" / "img"
    physical_image_root = (bound_root / "img" / "img").resolve(strict=True)
    _require((bound_root / "Img").is_symlink(), "In-Shop Img must be a symlink")
    _require(os.readlink(bound_root / "Img") == "img", "In-Shop Img symlink drift")
    _require(
        declared_image_root.resolve(strict=True) == physical_image_root,
        "In-Shop resolved image root drift",
    )

    def capture_impl(**kwargs: Any) -> NoReturn:
        _require(not captured, "production boundary count")
        _require(
            set(kwargs)
            == {
                "train_examples",
                "test_examples",
                "gallery_examples",
                "config",
                "progress_callback",
            },
            "production boundary arguments",
        )
        train = kwargs["train_examples"]
        query = kwargs["test_examples"]
        gallery = kwargs.get("gallery_examples")
        config = kwargs["config"]
        _require(type(train) is list and type(query) is list, "production bundle types")
        _require(type(gallery) is list, "official In-Shop gallery")
        _require(type(config) is ImageEndToEndConfig, "final ImageEndToEndConfig")
        _require(
            config.dataset_root is not None
            and Path(config.dataset_root).resolve(strict=True) == bound_root,
            "dataset root drift",
        )
        _require(config.dataset_name == "inshop", "dataset drift")
        _require(config.objectives == ("proxy_anchor",), "objective drift")
        _require(type(config.seed) is int and config.seed == 0, "seed drift")
        _require(config.recipe_id == RECIPE_ID, "recipe ID drift")
        _require(config.recipe_digest == RECIPE_DIGEST, "recipe digest drift")
        _require(config.proxy_count_per_class == 1, "proxy count drift")
        _require(config.backbone_name == "bn_inception", "backbone drift")
        _require(config.pretrained_weights == "bn_inception_52deb4733", "weights drift")
        _require(config.batch_size == 180, "batch size drift")
        _require(config.drop_last_train_batch is True, "drop-last drift")
        _require(config.train_epochs == 60, "epoch drift")
        _require(config.checkpoint_selection_interval == 0, "checkpoint split drift")
        _require(config.label_noise_fraction == 0.0, "label noise drift")
        _require(config.limit_per_class is None, "class limit drift")
        _require(config.max_classes is None, "class cap drift")
        _require(config.num_workers == 8, "worker count drift")
        _require(config.recipe_modified_fields == {}, "modified recipe drift")
        _require(
            config.dataset_selection_policy == "full_official_partition",
            "selection policy drift",
        )
        fraction = (
            config.checkpoint_selection_validation_fraction
            if config.checkpoint_selection_interval > 0
            else 0.0
        )
        optimization, _checkpoint = image_end_to_end._checkpoint_train_validation_split(
            train, fraction=fraction, seed=config.seed
        )
        optimization = image_end_to_end._apply_training_label_noise(
            optimization, fraction=config.label_noise_fraction, seed=config.seed
        )
        schedule = image_end_to_end._resolve_training_schedule(
            config,
            optimization_example_count=len(optimization),
            optimization_labels=[int(example.label) for example in optimization],
        )
        _require(
            type(schedule) is tuple
            and len(schedule) == 3
            and all(type(value) is int and value > 0 for value in schedule),
            "schedule type drift",
        )
        membership = [
            _read_bound_image(example.image, declared_image_root, physical_image_root)
            for example in optimization
        ]
        rows = tuple(
            (index, example.example_id, int(example.label))
            for index, example in enumerate(optimization)
        )
        _require(
            all(type(example) is ImageExample for example in (*train, *query, *gallery)),
            "production example type drift",
        )
        captured.append(
            CapturedAuthority(
                canonical_json_bytes(config.model_dump(mode="json")),
                config.recipe_id,
                config.recipe_digest,
                len(train),
                len(query),
                len(gallery),
                "query_gallery",
                "deepfashion-inshop-official",
                rows,
                _ordered_hash(membership),
                schedule[0],
                schedule[1],
                schedule[2],
            )
        )
        raise _CaptureComplete

    def capture_then_raise(**kwargs: Any) -> NoReturn:
        try:
            capture_impl(**kwargs)
        except ValueError as exc:
            raise _CaptureRejected(str(exc)) from exc

    return capture_then_raise


def _invoke_real_typer(argv: Sequence[str], dataset_root: Path) -> None:
    _require(type(argv) in (list, tuple), "argv sequence type")
    values = list(argv)
    _require(
        len(values) == 20
        and values[1:7]
        == [
            "-m",
            "sfora.cli",
            "image-end-to-end",
            "--dataset-name",
            "inshop",
            "--dataset-root",
        ]
        and values[8:17]
        == [
            "--objectives",
            "proxy_anchor",
            "--recipe",
            "auto",
            "--num-workers",
            "8",
            "--seed",
            "0",
            "--save-model-path",
        ]
        and values[18] == "--output",
        "frozen argv drift",
    )
    _require(
        Path(values[7]).resolve(strict=True) == dataset_root.resolve(strict=True),
        "frozen dataset root drift",
    )
    _require(bool(values[0]) and bool(values[17]) and bool(values[19]), "frozen argv path drift")
    command = get_command(sfora.cli.app)
    with contextlib.redirect_stdout(io.StringIO()):
        command.main(args=values[3:], prog_name=values[0], standalone_mode=False)


def capture_authority(argv: Sequence[str], dataset_root: Path) -> CapturedAuthority:
    captured: list[CapturedAuthority] = []
    try:
        with patch.object(
            sfora.cli,
            "run_image_end_to_end_benchmark",
            _capture_boundary(captured, dataset_root),
        ):
            _invoke_real_typer(argv, dataset_root)
    except _CaptureComplete:
        pass
    except _CaptureRejected as exc:
        raise ValueError(str(exc)) from None
    _require(len(captured) == 1, "production boundary count")
    return captured[0]


def encode_capture_request(argv: Sequence[str], dataset_root: Path) -> bytes:
    _require(type(argv) in (list, tuple), "capture request argv type")
    values = list(argv)
    _require(all(type(value) is str for value in values), "capture request argument type")
    payload = canonical_json_bytes(
        {"argv": values, "dataset_root": dataset_root.absolute().as_posix()}
    )
    return encode_private_child_frame(PrivateChildFrame("capture-request", os.getpid(), payload))


def _decode_capture_request(data: bytes) -> tuple[list[str], Path]:
    frame = decode_private_child_frame(data)
    _require(frame.role == "capture-request", "capture child request role")
    payload = load_strict_json_bytes(frame.payload)
    _require(set(payload) == {"argv", "dataset_root"}, "capture request keys")
    argv = payload["argv"]
    dataset_root = payload["dataset_root"]
    _require(
        type(argv) is list and all(type(value) is str for value in argv),
        "capture request argv type",
    )
    _require(type(dataset_root) is str and Path(dataset_root).is_absolute(), "dataset root type")
    return argv, Path(dataset_root)


def _capture_payload(capture: CapturedAuthority) -> bytes:
    return canonical_json_bytes(
        {
            "config_json": capture.config_bytes.decode("utf-8"),
            "recipe_id": capture.recipe_id,
            "recipe_digest": capture.recipe_digest,
            "train_count": capture.train_count,
            "query_count": capture.query_count,
            "gallery_count": capture.gallery_count,
            "protocol": capture.protocol,
            "protocol_name": capture.protocol_name,
            "rows": [list(row) for row in capture.rows],
            "resolved_membership_sha256": capture.resolved_membership_sha256,
            "resolved_train_steps": capture.resolved_train_steps,
            "steps_per_epoch": capture.steps_per_epoch,
            "total_epochs": capture.total_epochs,
        }
    )


def decode_capture_response(data: bytes) -> CapturedAuthority:
    frame = decode_private_child_frame(data)
    _require(frame.role == "capture-response", "capture child response role")
    payload = load_strict_json_bytes(frame.payload)
    expected_keys = {
        "config_json",
        "recipe_id",
        "recipe_digest",
        "train_count",
        "query_count",
        "gallery_count",
        "protocol",
        "protocol_name",
        "rows",
        "resolved_membership_sha256",
        "resolved_train_steps",
        "steps_per_epoch",
        "total_epochs",
    }
    _require(set(payload) == expected_keys, "capture response keys")
    config_text = payload["config_json"]
    _require(type(config_text) is str, "capture response config type")
    config_bytes = config_text.encode("utf-8")
    _require(
        canonical_json_bytes(load_strict_json_bytes(config_bytes)) == config_bytes,
        "capture response config is not canonical",
    )
    for key in (
        "recipe_id",
        "recipe_digest",
        "protocol",
        "protocol_name",
        "resolved_membership_sha256",
    ):
        _require(type(payload[key]) is str and bool(payload[key]), f"capture response {key} type")
    for key in (
        "train_count",
        "query_count",
        "gallery_count",
        "resolved_train_steps",
        "steps_per_epoch",
        "total_epochs",
    ):
        _require(type(payload[key]) is int and payload[key] >= 0, f"capture response {key} type")
    raw_rows = payload["rows"]
    _require(type(raw_rows) is list, "capture response rows type")
    rows: list[tuple[int, str, int]] = []
    for index, row in enumerate(raw_rows):
        _require(
            type(row) is list
            and len(row) == 3
            and type(row[0]) is int
            and row[0] == index
            and type(row[1]) is str
            and bool(row[1])
            and type(row[2]) is int,
            "capture response row type",
        )
        rows.append((row[0], row[1], row[2]))
    return CapturedAuthority(
        config_bytes,
        payload["recipe_id"],
        payload["recipe_digest"],
        payload["train_count"],
        payload["query_count"],
        payload["gallery_count"],
        payload["protocol"],
        payload["protocol_name"],
        tuple(rows),
        payload["resolved_membership_sha256"],
        payload["resolved_train_steps"],
        payload["steps_per_epoch"],
        payload["total_epochs"],
    )


def _capture_child_output() -> bytes:
    argv, dataset_root = _decode_capture_request(sys.stdin.buffer.read())
    capture = capture_authority(argv, dataset_root)
    return encode_private_child_frame(
        PrivateChildFrame("capture-response", os.getpid(), _capture_payload(capture))
    )


def _validate_operating_config(config: object) -> dict[str, Any]:
    _require(type(config) is dict, "resolved config must be an object")
    result = config
    _require(result.get("objectives") == ["proxy_anchor"], "resolved objective drift")
    _require(type(result.get("seed")) is int and result["seed"] == 0, "resolved seed drift")
    _require(result.get("recipe_id") == RECIPE_ID, "resolved recipe ID drift")
    _require(result.get("recipe_digest") == RECIPE_DIGEST, "resolved recipe digest drift")
    return result


def _skip_json_whitespace(data: bytes, offset: int) -> int:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    return offset


def _scan_json_string(data: bytes, offset: int) -> int:
    _require(offset < len(data) and data[offset] == ord('"'), "expected JSON string")
    offset += 1
    while offset < len(data):
        character = data[offset]
        if character == ord('"'):
            return offset + 1
        if character == ord("\\"):
            offset += 2
            continue
        _require(character >= 0x20, "control byte in JSON string")
        offset += 1
    raise ValueError("unterminated JSON string")


def _scan_opaque_json_value(data: bytes, offset: int) -> int:
    offset = _skip_json_whitespace(data, offset)
    _require(offset < len(data), "missing JSON value")
    if data[offset] == ord('"'):
        return _scan_json_string(data, offset)
    if data[offset] in b"{[":
        closers = [ord("}") if data[offset] == ord("{") else ord("]")]
        offset += 1
        while offset < len(data) and closers:
            character = data[offset]
            if character == ord('"'):
                offset = _scan_json_string(data, offset)
                continue
            if character in b"{[":
                closers.append(ord("}") if character == ord("{") else ord("]"))
            elif character in b"}]":
                _require(character == closers.pop(), "mismatched JSON container")
            offset += 1
        _require(not closers, "unterminated JSON container")
        return offset
    end = offset
    while end < len(data) and data[end] not in b",}":
        end += 1
    _require(bool(data[offset:end].strip()), "missing JSON scalar")
    return end


def _extract_report_config(report_bytes: bytes) -> object:
    _require(type(report_bytes) is bytes, "report bytes type")
    offset = _skip_json_whitespace(report_bytes, 0)
    _require(
        offset < len(report_bytes) and report_bytes[offset] == ord("{"),
        "report must be an object",
    )
    offset += 1
    seen_keys: set[str] = set()
    raw_values: dict[str, bytes] = {}
    while True:
        offset = _skip_json_whitespace(report_bytes, offset)
        _require(offset < len(report_bytes), "unterminated report object")
        if report_bytes[offset] == ord("}"):
            offset += 1
            break
        key_start = offset
        key_end = _scan_json_string(report_bytes, key_start)
        key = load_strict_json_value_bytes(report_bytes[key_start:key_end])
        _require(type(key) is str, "report key type")
        _require(key not in seen_keys, f"duplicate report key: {key}")
        seen_keys.add(key)
        offset = _skip_json_whitespace(report_bytes, key_end)
        _require(
            offset < len(report_bytes) and report_bytes[offset] == ord(":"),
            "report key missing colon",
        )
        value_start = _skip_json_whitespace(report_bytes, offset + 1)
        value_end = _scan_opaque_json_value(report_bytes, value_start)
        if key == "methods":
            _require(
                report_bytes[value_start] == ord("{")
                and report_bytes[value_end - 1] == ord("}"),
                "report methods must be an opaque object",
            )
        else:
            raw_values[key] = report_bytes[value_start:value_end]
        offset = _skip_json_whitespace(report_bytes, value_end)
        _require(offset < len(report_bytes), "unterminated report object")
        if report_bytes[offset] == ord(","):
            offset += 1
            continue
        _require(report_bytes[offset] == ord("}"), "invalid report member delimiter")
        offset += 1
        break
    _require(
        not report_bytes[_skip_json_whitespace(report_bytes, offset) :],
        "trailing report bytes",
    )
    _require(seen_keys == EXPECTED_REPORT_KEYS, "report keys drift")
    parsed = {key: load_strict_json_value_bytes(raw) for key, raw in raw_values.items()}
    for key in ("name", "dataset_name", "protocol"):
        _require(type(parsed[key]) is str, f"report {key} type drift")
    for key in ("train_examples", "test_examples"):
        _require(type(parsed[key]) is int and parsed[key] >= 0, f"report {key} type drift")
    return parsed["config"]


def derive_resolved_config(
    report_bytes: bytes,
    checkpoint: CheckpointMetadata,
    authority: PrelaunchAuthority,
) -> bytes:
    config = canonical_json_bytes(_validate_operating_config(_extract_report_config(report_bytes)))
    _require(config == authority.expected_config_bytes, "report config drift")
    _require(
        hashlib.sha256(config).hexdigest() == checkpoint.training_config_sha256,
        "checkpoint config drift",
    )
    return config


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(child) for key, child in value.items()}
    if type(value) is tuple:
        return [_plain_json(child) for child in value]
    if type(value) is list:
        return [_plain_json(child) for child in value]
    return value


def _capture_rows(capture: CapturedAuthority) -> list[dict[str, object]]:
    return [
        {"sample_index": sample_index, "example_id": example_id, "label": label}
        for sample_index, example_id, label in capture.rows
    ]


def _validate_capture_authority(
    capture: CapturedAuthority, authority: PrelaunchAuthority
) -> list[dict[str, object]]:
    _require(type(capture.config_bytes) is bytes, "capture config type drift")
    for field, value in (
        ("recipe ID", capture.recipe_id),
        ("recipe digest", capture.recipe_digest),
        ("protocol", capture.protocol),
        ("protocol name", capture.protocol_name),
        ("membership", capture.resolved_membership_sha256),
    ):
        _require(type(value) is str and bool(value), f"capture {field} type drift")
    for field, value in (
        ("train count", capture.train_count),
        ("query count", capture.query_count),
        ("gallery count", capture.gallery_count),
        ("resolved training steps", capture.resolved_train_steps),
        ("steps per epoch", capture.steps_per_epoch),
        ("total epochs", capture.total_epochs),
    ):
        _require(type(value) is int and value >= 0, f"capture {field} type drift")
    _require(type(capture.rows) is tuple, "capture rows type drift")
    for index, row in enumerate(capture.rows):
        _require(
            type(row) is tuple
            and len(row) == 3
            and type(row[0]) is int
            and row[0] == index
            and type(row[1]) is str
            and bool(row[1])
            and type(row[2]) is int,
            "capture row type drift",
        )
    expected_dataset = authority.payload["dataset"]
    expected_bundle = expected_dataset["bundle"]
    expected_optimization = expected_dataset["optimization_authority"]
    expected_execution = authority.payload["execution"]
    _require(capture.config_bytes == authority.expected_config_bytes, "config drift")
    _require(capture.recipe_id == RECIPE_ID, "recipe ID drift")
    _require(capture.recipe_digest == RECIPE_DIGEST, "recipe digest drift")
    _require(
        capture.recipe_id == expected_execution["recipe_id"],
        "authority recipe ID drift",
    )
    _require(
        capture.recipe_digest == expected_execution["recipe_digest"],
        "authority recipe digest drift",
    )
    _require(capture.train_count == expected_bundle["train"], "train count drift")
    _require(capture.query_count == expected_bundle["query"], "query count drift")
    _require(capture.gallery_count == expected_bundle["gallery"], "gallery count drift")
    _require(capture.protocol == expected_bundle["protocol"], "protocol drift")
    _require(capture.protocol_name == expected_bundle["protocol_name"], "protocol name drift")
    rows = _capture_rows(capture)
    _require(len(rows) == expected_optimization["row_count"], "row count drift")
    _require(
        len({row["label"] for row in rows}) == expected_optimization["identity_count"],
        "identity count drift",
    )
    _require(
        _ordered_hash(rows) == expected_optimization["ordered_row_sha256"],
        "ordered row drift",
    )
    _require(
        capture.resolved_membership_sha256
        == expected_optimization["resolved_membership_sha256"],
        "resolved membership drift",
    )
    _require(capture.resolved_train_steps == authority.expected_train_steps, "schedule drift")
    _require(capture.steps_per_epoch == authority.steps_per_epoch, "schedule drift")
    _require(capture.total_epochs == authority.total_epochs, "schedule drift")
    return rows


def derive_train_manifest(capture: CapturedAuthority, authority: PrelaunchAuthority) -> bytes:
    rows = _validate_capture_authority(capture, authority)
    dataset = authority.payload["dataset"]
    files = sorted(
        (_plain_json(value) for value in authority.payload["source"]["files"]),
        key=lambda value: value["path"].encode("utf-8"),
    )
    payload = {
        "schema_version": "pass201-train-manifest-v1",
        "algorithm_id": "pass201-inshop-benchmark-row-suffix-v2",
        "source_commit": authority.source_commit,
        "dataset_authority": {
            "root": dataset["root"],
            "partition_sha256": dataset["partition"]["sha256"],
            "resolved_image_root": dataset["resolved_image_root"],
            "image_tree_sha256": dataset["image_tree"]["root_sha256"],
            "bundle": _plain_json(dataset["bundle"]),
            "selection_policy": dataset["selection_policy"],
        },
        "rows": rows,
        "derivation": {
            "call_graph": list(TRAIN_MANIFEST_CALL_GRAPH),
            "source_files": files,
            "resolved_config_sha256": authority.expected_config_sha256,
            "row_count": len(rows),
            "identity_count": len({row["label"] for row in rows}),
            "ordered_row_sha256": _ordered_hash(rows),
            "resolved_membership_count": len(rows),
            "resolved_membership_sha256": capture.resolved_membership_sha256,
        },
    }
    validate_train_manifest(payload, authority)
    return canonical_json_bytes(payload)


def validate_completed_epoch(
    capture: CapturedAuthority,
    checkpoint_step: int,
    optimization_count: int,
    batch_size: int,
) -> int:
    _require(type(checkpoint_step) is int and checkpoint_step > 0, "invalid checkpoint step")
    _require(type(optimization_count) is int and optimization_count > 0, "invalid row count")
    _require(type(batch_size) is int and batch_size > 0, "invalid batch size")
    _require(
        type(capture.resolved_train_steps) is int and capture.resolved_train_steps > 0,
        "invalid resolved training steps",
    )
    _require(
        type(capture.steps_per_epoch) is int and capture.steps_per_epoch > 0,
        "invalid steps per epoch",
    )
    _require(
        type(capture.total_epochs) is int and capture.total_epochs > 0,
        "invalid total epochs",
    )
    _require(
        capture.steps_per_epoch == max(1, optimization_count // batch_size),
        "drop-last schedule drift",
    )
    _require(checkpoint_step == capture.resolved_train_steps, "training step drift")
    _require(checkpoint_step % capture.steps_per_epoch == 0, "partial epoch")
    completed = checkpoint_step // capture.steps_per_epoch
    _require(completed == capture.total_epochs, "completed epoch drift")
    return completed


def encode_sidecar_frame(frame: SidecarFrame) -> bytes:
    _require(type(frame.pid) is int and frame.pid > 0, "invalid child PID")
    _require(type(frame.config_bytes) is bytes, "invalid config bytes")
    _require(type(frame.manifest_bytes) is bytes, "invalid manifest bytes")
    config_hash = hashlib.sha256(frame.config_bytes).hexdigest()
    manifest_hash = hashlib.sha256(frame.manifest_bytes).hexdigest()
    _require(frame.config_sha256 == config_hash, "config frame hash drift")
    _require(frame.manifest_sha256 == manifest_hash, "manifest frame hash drift")
    return b"".join(
        (
            SIDECAR_FRAME_MAGIC,
            struct.pack(">Q", frame.pid),
            struct.pack(">Q", len(frame.config_bytes)),
            frame.config_bytes,
            bytes.fromhex(frame.config_sha256),
            struct.pack(">Q", len(frame.manifest_bytes)),
            frame.manifest_bytes,
            bytes.fromhex(frame.manifest_sha256),
        )
    )


def decode_sidecar_frame(data: bytes) -> SidecarFrame:
    _require(type(data) is bytes and data.startswith(SIDECAR_FRAME_MAGIC), "sidecar frame magic")
    offset = len(SIDECAR_FRAME_MAGIC)

    def take(count: int) -> bytes:
        nonlocal offset
        _require(count >= 0 and count <= len(data) - offset, "truncated sidecar frame")
        result = data[offset : offset + count]
        offset += count
        return result

    pid = struct.unpack(">Q", take(8))[0]
    config_size = struct.unpack(">Q", take(8))[0]
    config = take(config_size)
    config_hash = take(32).hex()
    manifest_size = struct.unpack(">Q", take(8))[0]
    manifest = take(manifest_size)
    manifest_hash = take(32).hex()
    _require(offset == len(data), "trailing sidecar frame bytes")
    frame = SidecarFrame(pid, config, manifest, config_hash, manifest_hash)
    _require(encode_sidecar_frame(frame) == data, "noncanonical sidecar frame")
    return frame


def validate_sidecar_identity(
    first: SidecarFrame, second: SidecarFrame
) -> tuple[bytes, bytes]:
    encode_sidecar_frame(first)
    encode_sidecar_frame(second)
    _require(first.pid != second.pid, "sidecar child PIDs must be distinct")
    _require(first.config_bytes == second.config_bytes, "sidecar config identity drift")
    _require(first.manifest_bytes == second.manifest_bytes, "sidecar manifest identity drift")
    _require(first.config_sha256 == second.config_sha256, "sidecar config hash drift")
    _require(first.manifest_sha256 == second.manifest_sha256, "sidecar manifest hash drift")
    return first.config_bytes, first.manifest_bytes


def _read_immutable_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), "sidecar input is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    _require(
        _stat_identity(before) == _stat_identity(after),
        "sidecar input changed during read",
    )
    data = b"".join(chunks)
    _require(len(data) == before.st_size, "sidecar input size drift")
    return data


def _exact_existing_path(path: Path, *, directory: bool) -> Path:
    candidate = Path(os.path.abspath(path))
    try:
        mode = os.lstat(candidate).st_mode
    except OSError as exc:
        raise ValueError(f"sidecar path does not exist: {candidate}") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"sidecar path must not be a symlink: {candidate}")
    expected_kind = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    _require(expected_kind, f"invalid sidecar path type: {candidate}")
    resolved = candidate.resolve(strict=True)
    _require(resolved == candidate, f"sidecar path contains a symlink: {candidate}")
    return candidate


def _exact_regular_path(path: Path) -> Path:
    return _exact_existing_path(path, directory=False)


def _validate_bound_environment(authority: PrelaunchAuthority) -> None:
    expected = dict(authority.payload["execution"]["environment"])
    _require(dict(os.environ) == expected, "sidecar environment drift")


def _run_private_child(
    authority: PrelaunchAuthority,
    script_name: str,
    command: Sequence[str],
    request: bytes,
) -> bytes:
    execution = authority.payload["execution"]
    interpreter = str(execution["python"]["path"])
    script = authority.checkout_root / "scripts" / script_name
    try:
        result = subprocess.run(
            [interpreter, str(script), *command],
            cwd=authority.checkout_root,
            env=dict(execution["environment"]),
            input=request,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"{command[0]} failed to start") from exc
    _require(result.returncode == 0, f"{command[0]} failed")
    _require(not result.stderr, f"{command[0]} emitted stderr")
    return result.stdout


def _run_capture_child(authority: PrelaunchAuthority) -> CapturedAuthority:
    request = encode_capture_request(
        tuple(authority.payload["execution"]["argv"]),
        Path(authority.payload["dataset"]["root"]),
    )
    response = _run_private_child(
        authority, "run_pass201_pa_source_v2.py", ("capture-authority-child",), request
    )
    return decode_capture_response(response)


def _run_metadata_child(
    authority: PrelaunchAuthority, checkpoint_path: Path
) -> BoundCheckpointMetadata:
    response = _run_private_child(
        authority,
        "pass201_pa_source_v2_contract.py",
        ("restricted-metadata-child", "--checkpoint", str(checkpoint_path)),
        encode_checkpoint_metadata_request(authority),
    )
    return decode_checkpoint_metadata_response(response, authority, checkpoint_path)


def _run_binding_child(
    authority: PrelaunchAuthority,
    checkpoint_path: Path,
    expected: ExternalFileBinding,
) -> ExternalFileBinding:
    response = _run_private_child(
        authority,
        "pass201_pa_source_v2_contract.py",
        ("restricted-binding-child", "--checkpoint", str(checkpoint_path)),
        encode_checkpoint_binding_request(expected),
    )
    return decode_checkpoint_binding_response(response, expected, checkpoint_path)


def derive_sidecars_from_files(
    manifest_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> SidecarFrame:
    manifest_path = _exact_regular_path(manifest_path)
    report_path = _exact_regular_path(report_path)
    checkpoint_path = _exact_regular_path(checkpoint_path)
    output_dir = _exact_existing_path(output_dir, directory=True)
    manifest_bytes = _read_immutable_regular(manifest_path)
    authority = validate_prelaunch(load_strict_json_bytes(manifest_bytes))
    _validate_bound_environment(authority)
    expected_manifest = (
        authority.checkout_root / authority.payload["authorization"]["manifest_path"]
    ).resolve(strict=True)
    expected_output_dir = (
        authority.checkout_root / authority.payload["outputs"]["run_directory"]
    ).resolve(strict=True)
    _require(manifest_path == expected_manifest, "alternative manifest path")
    _require(output_dir == expected_output_dir, "alternative output directory")
    expected_report = (
        authority.checkout_root / authority.payload["outputs"]["report"]["path"]
    ).resolve(strict=True)
    expected_checkpoint = (
        authority.checkout_root / authority.payload["outputs"]["checkpoint"]["path"]
    ).resolve(strict=True)
    _require(
        report_path == expected_report,
        "alternative report path",
    )
    _require(
        checkpoint_path == expected_checkpoint,
        "alternative checkpoint path",
    )
    report_bytes = _read_immutable_regular(report_path)
    capture = _run_capture_child(authority)
    _validate_capture_authority(capture, authority)
    bound_checkpoint = _run_metadata_child(authority, checkpoint_path)
    checkpoint = bound_checkpoint.metadata
    expected_config = load_strict_json_bytes(authority.expected_config_bytes)
    _require(expected_config.get("drop_last_train_batch") is True, "drop-last config drift")
    batch_size = expected_config.get("batch_size")
    _require(type(batch_size) is int, "batch size config drift")
    validate_completed_epoch(capture, checkpoint.training_step, len(capture.rows), batch_size)
    config = derive_resolved_config(report_bytes, checkpoint, authority)
    manifest = derive_train_manifest(capture, authority)
    frame = SidecarFrame(
        os.getpid(),
        config,
        manifest,
        hashlib.sha256(config).hexdigest(),
        hashlib.sha256(manifest).hexdigest(),
    )
    _require(_read_immutable_regular(manifest_path) == manifest_bytes, "manifest input drift")
    _require(_read_immutable_regular(report_path) == report_bytes, "report input drift")
    _run_binding_child(authority, checkpoint_path, bound_checkpoint.binding)
    return frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_pass201_pa_source_v2.py")
    commands = parser.add_subparsers(dest="command", required=True)
    sidecars = commands.add_parser("derive-sidecars")
    sidecars.add_argument("--manifest", required=True, type=Path)
    sidecars.add_argument("--report", required=True, type=Path)
    sidecars.add_argument("--checkpoint", required=True, type=Path)
    sidecars.add_argument("--output-dir", required=True, type=Path)
    commands.add_parser("capture-authority-child")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "capture-authority-child":
        sys.stdout.buffer.write(_capture_child_output())
        sys.stdout.buffer.flush()
        return 0
    _require(args.command == "derive-sidecars", "private command drift")
    frame = derive_sidecars_from_files(
        args.manifest,
        args.report,
        args.checkpoint,
        args.output_dir,
    )
    sys.stdout.buffer.write(encode_sidecar_frame(frame))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
