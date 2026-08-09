from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import NoReturn

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pass201_pa_source_v2_contract as contract  # noqa: E402
import run_pass201_pa_source_v2 as controller  # noqa: E402
from pass201_pa_source_v2_contract import (  # noqa: E402
    BoundCheckpointMetadata,
    CheckpointArch,
    CheckpointMetadata,
    ExternalFileBinding,
    OutputEvidence,
    PrelaunchAuthority,
    bind_external_file,
    canonical_json_bytes,
    decode_checkpoint_binding_response,
    encode_checkpoint_binding_request,
    load_strict_json_bytes,
    load_strict_json_value_bytes,
    validate_complete_receipt,
)
from run_pass201_pa_source_v2 import (  # noqa: E402
    CapturedAuthority,
    SidecarFrame,
    capture_authority,
    decode_sidecar_frame,
    derive_resolved_config,
    derive_train_manifest,
    encode_sidecar_frame,
    main,
    validate_completed_epoch,
    validate_sidecar_identity,
)

from sfora import image_end_to_end  # noqa: E402
from sfora.data import load_image_retrieval_bundle  # noqa: E402
from sfora.image_end_to_end import (  # noqa: E402
    _apply_training_label_noise,
    _checkpoint_train_validation_split,
)


@pytest.fixture
def tiny_inshop(tmp_path: Path) -> Path:
    root = tmp_path / "inshop"
    resolved_images = root / "img" / "img"
    resolved_images.mkdir(parents=True)
    (root / "Img").symlink_to("img", target_is_directory=True)
    rows = [
        ("img/WOMEN/Dresses/id_0001/01_1_front.jpg", "id_0001", "train"),
        ("img/WOMEN/Dresses/id_0002/01_1_front.jpg", "id_0002", "train"),
        ("img/WOMEN/Tees/id_0003/01_1_front.jpg", "id_0003", "query"),
        ("img/WOMEN/Tees/id_0003/02_1_front.jpg", "id_0003", "gallery"),
        ("img/MEN/Tees/id_0004/01_1_front.jpg", "id_0004", "query"),
        ("img/MEN/Tees/id_0004/02_1_front.jpg", "id_0004", "gallery"),
    ]
    for index, (name, _item_id, _status) in enumerate(rows):
        path = root / "img" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"tiny-image-{index}".encode())
    partition = root / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir()
    partition.write_text(
        f"{len(rows)}\nimage_name item_id evaluation_status\n"
        + "".join(f"{name} {item_id} {status}\n" for name, item_id, status in rows),
        encoding="utf-8",
    )
    return root


def frozen_test_argv(root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sfora.cli",
        "image-end-to-end",
        "--dataset-name",
        "inshop",
        "--dataset-root",
        str(root),
        "--objectives",
        "proxy_anchor",
        "--recipe",
        "auto",
        "--num-workers",
        "8",
        "--seed",
        "0",
        "--save-model-path",
        str(root / "out" / "checkpoint.pt"),
        "--output",
        str(root / "out" / "report.json"),
    ]


def expected_optimization_rows(root: Path):
    bundle = load_image_retrieval_bundle(
        dataset_name="inshop",
        dataset_root=root,
        limit_per_class=None,
        train_min_per_class=None,
        evaluation_min_per_class=None,
        max_classes=None,
        seed=0,
    )
    optimization, _ = _checkpoint_train_validation_split(bundle.train, fraction=0.0, seed=0)
    return _apply_training_label_noise(optimization, fraction=0.0, seed=0)


def forbidden(name: str):
    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError(f"forbidden {name}")

    return fail


def ordered_row_hash(rows: tuple[tuple[int, str, int], ...]) -> str:
    digest = hashlib.sha256()
    for sample_index, example_id, label in rows:
        encoded = canonical_json_bytes(
            {"sample_index": sample_index, "example_id": example_id, "label": label}
        )
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


@pytest.fixture
def sidecar_inputs() -> tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata]:
    config = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "protocol": "proxy-anchor-resnet50-512",
        "seed": 0,
        "recipe_id": controller.RECIPE_ID,
        "recipe_digest": controller.RECIPE_DIGEST,
    }
    config_bytes = canonical_json_bytes(config)
    rows = (
        (0, "inshop-train-img/WOMEN/id_0001/a.jpg", 1),
        (1, "inshop-train-img/WOMEN/id_0002/b.jpg", 2),
    )
    row_hash = ordered_row_hash(rows)
    membership_hash = "c" * 64
    file_a = {
        "path": "src/sfora/data.py",
        "git_mode": "100644",
        "bytes": 10,
        "sha256": "a" * 64,
        "git_blob": "b" * 40,
    }
    file_b = {
        "path": "src/sfora/image_end_to_end.py",
        "git_mode": "100644",
        "bytes": 20,
        "sha256": "d" * 64,
        "git_blob": "e" * 40,
    }
    payload = {
        "execution": {
            "recipe_id": controller.RECIPE_ID,
            "recipe_digest": controller.RECIPE_DIGEST,
        },
        "source": {"files": [file_a, file_b]},
        "dataset": {
            "root": "/home/riomus/datasets/inshop_official_standard",
            "partition": {"sha256": "f" * 64},
            "resolved_image_root": "/home/riomus/datasets/inshop_official_standard/img/img",
            "image_tree": {"root_sha256": "1" * 64},
            "bundle": {
                "train": 2,
                "query": 2,
                "gallery": 2,
                "protocol": "query_gallery",
                "protocol_name": "deepfashion-inshop-official",
            },
            "selection_policy": "full_official_partition",
            "optimization_authority": {
                "algorithm_id": "pass201-production-invocation-capture-v1",
                "row_count": 2,
                "identity_count": 2,
                "ordered_row_sha256": row_hash,
                "resolved_membership_sha256": membership_hash,
            },
        },
        "sidecars": {
            "config_algorithm": "pass201-resolved-config-v2",
            "manifest_algorithm": "pass201-inshop-benchmark-row-suffix-v2",
            "schedule_algorithm": "pass201-inshop-completed-epoch-v1",
        },
    }
    authority = PrelaunchAuthority(
        payload=payload,
        source_commit="9" * 40,
        checkout_root=Path("/checkout"),
        expected_config_bytes=config_bytes,
        expected_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        expected_train_steps=120,
        steps_per_epoch=2,
        total_epochs=60,
    )
    capture = CapturedAuthority(
        config_bytes,
        controller.RECIPE_ID,
        controller.RECIPE_DIGEST,
        2,
        2,
        2,
        "query_gallery",
        "deepfashion-inshop-official",
        rows,
        membership_hash,
        120,
        2,
        60,
    )
    arch = CheckpointArch(
        "bn_inception",
        "bn_inception_52deb4733",
        "avg_max",
        512,
        "kaiming_normal",
        False,
    )
    checkpoint = CheckpointMetadata(
        "2" * 64,
        (
            "arch",
            "artifact_selection",
            "evaluation_model_source",
            "state_dict",
            "training_config",
            "training_step",
        ),
        "final_training_state",
        "student",
        120,
        arch,
        "3" * 64,
        authority.expected_config_sha256,
        1,
    )
    return capture, authority, checkpoint


def test_production_capture_uses_real_cli_boundary_without_training(
    tiny_inshop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_end_to_end, "_torchvision_model_factory", forbidden("model"))

    captured = capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)

    assert captured.protocol == "query_gallery"
    assert captured.protocol_name == "deepfashion-inshop-official"
    assert captured.rows == tuple(
        (index, row.example_id, int(row.label))
        for index, row in enumerate(expected_optimization_rows(tiny_inshop))
    )
    config = load_strict_json_bytes(captured.config_bytes)
    assert config["objectives"] == ["proxy_anchor"]
    assert type(config["seed"]) is int and config["seed"] == 0
    assert captured.recipe_id == controller.RECIPE_ID
    assert captured.recipe_digest == controller.RECIPE_DIGEST
    assert (captured.train_count, captured.query_count, captured.gallery_count) == (2, 2, 2)
    assert (captured.resolved_train_steps, captured.steps_per_epoch, captured.total_epochs) == (
        60,
        1,
        60,
    )
    assert not (tiny_inshop / "out").exists()


def _capture_child(root: Path, hash_seed: str) -> dict[str, object]:
    program = """
import dataclasses, json, os, sys
from pathlib import Path
from run_pass201_pa_source_v2 import capture_authority
capture = capture_authority(json.loads(sys.argv[1]), Path(sys.argv[2]))
payload = dataclasses.asdict(capture)
payload["config_bytes"] = capture.config_bytes.decode("utf-8")
print(json.dumps({"pid": os.getpid(), "capture": payload}, sort_keys=True, separators=(",", ":")))
"""
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'scripts'}"
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", program, json.dumps(frozen_test_argv(root)), str(root)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_production_capture_is_identical_in_two_fresh_processes(tiny_inshop: Path) -> None:
    first = _capture_child(tiny_inshop, "1")
    second = _capture_child(tiny_inshop, "987654")
    assert first["pid"] != second["pid"]
    assert first["capture"] == second["capture"]


def test_capture_child_never_opens_produced_checkpoint(tiny_inshop: Path, tmp_path: Path) -> None:
    checkpoint = tiny_inshop / "out" / "checkpoint.pt"
    guard_dir = tmp_path / "open-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        """
import os, sys
target = os.environ["PASS201_TEST_FORBIDDEN_OPEN"]
def guard(event, args):
    if event != "open" or not args:
        return
    try:
        candidate = os.path.abspath(os.fspath(args[0]))
    except TypeError:
        return
    if candidate == target:
        raise RuntimeError("capture child opened produced checkpoint")
sys.addaudithook(guard)
""",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{guard_dir}:{repo / 'src'}:{repo / 'scripts'}"
    env["PASS201_TEST_FORBIDDEN_OPEN"] = str(checkpoint.absolute())
    request = controller.encode_capture_request(frozen_test_argv(tiny_inshop), tiny_inshop)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_pass201_pa_source_v2.py"),
            "capture-authority-child",
        ],
        cwd=repo,
        env=env,
        input=request,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    captured = controller.decode_capture_response(result.stdout)
    assert captured.rows == tuple(
        (index, row.example_id, int(row.label))
        for index, row in enumerate(expected_optimization_rows(tiny_inshop))
    )
    assert not checkpoint.exists()


def test_production_capture_calls_exact_split_noise_schedule_suffix(
    tiny_inshop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[object, ...]] = []
    real_split = image_end_to_end._checkpoint_train_validation_split
    real_noise = image_end_to_end._apply_training_label_noise
    real_schedule = image_end_to_end._resolve_training_schedule

    def split(examples, *, fraction: float, seed: int):
        events.append(("split", fraction, seed, len(examples)))
        return real_split(examples, fraction=fraction, seed=seed)

    def noise(examples, *, fraction: float, seed: int):
        events.append(("noise", fraction, seed, len(examples)))
        return real_noise(examples, fraction=fraction, seed=seed)

    def schedule(config, *, optimization_example_count: int, optimization_labels):
        events.append(
            ("schedule", optimization_example_count, tuple(optimization_labels), config.batch_size)
        )
        return real_schedule(
            config,
            optimization_example_count=optimization_example_count,
            optimization_labels=optimization_labels,
        )

    monkeypatch.setattr(image_end_to_end, "_checkpoint_train_validation_split", split)
    monkeypatch.setattr(image_end_to_end, "_apply_training_label_noise", noise)
    monkeypatch.setattr(image_end_to_end, "_resolve_training_schedule", schedule)

    capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)

    assert events == [
        ("split", 0.0, 0, 2),
        ("noise", 0.0, 0, 2),
        ("schedule", 2, (0, 1), 180),
    ]


def test_production_capture_membership_hashes_resolved_path_content(tiny_inshop: Path) -> None:
    first = capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)
    image_path = tiny_inshop / "img" / "img" / "WOMEN" / "Dresses" / "id_0001" / "01_1_front.jpg"
    image_path.write_bytes(b"mutated-image-content")

    second = capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)

    assert first.rows == second.rows
    assert first.resolved_membership_sha256 != second.resolved_membership_sha256


def test_production_capture_membership_uses_normative_resolved_records(tiny_inshop: Path) -> None:
    captured = capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)
    records = []
    physical_root = tiny_inshop / "img" / "img"
    for relative in (
        Path("WOMEN/Dresses/id_0001/01_1_front.jpg"),
        Path("WOMEN/Dresses/id_0002/01_1_front.jpg"),
    ):
        data = (physical_root / relative).read_bytes()
        records.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    framed = b"".join(
        struct.pack(">Q", len(encoded)) + encoded
        for encoded in (
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for record in records
        )
    )
    assert captured.resolved_membership_sha256 == hashlib.sha256(framed).hexdigest()


def test_production_capture_rejects_symlinked_optimization_image(tiny_inshop: Path) -> None:
    physical_root = tiny_inshop / "img" / "img"
    image = physical_root / "WOMEN" / "Dresses" / "id_0001" / "01_1_front.jpg"
    target = physical_root / "WOMEN" / "Dresses" / "id_0002" / "01_1_front.jpg"
    image.unlink()
    image.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)


@pytest.mark.parametrize(
    "extra",
    [
        ["--label-noise-fraction", "0.5"],
        ["--train-epochs", "59"],
        ["--max-classes", "2"],
    ],
)
def test_production_capture_rejects_nonfrozen_cli_parameters(
    tiny_inshop: Path, extra: list[str]
) -> None:
    argv = frozen_test_argv(tiny_inshop)
    argv[4:4] = extra
    with pytest.raises(ValueError, match="drift"):
        capture_authority(argv, tiny_inshop)


def test_resolved_config_reads_only_exact_report_config(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    config_text = authority.expected_config_bytes.decode().strip()
    report = (
        '{"name":"image-end-to-end-benchmark","dataset_name":"inshop",'
        f'"protocol":"proxy-anchor-resnet50-512","config":{config_text},'
        '"train_examples":2,"test_examples":2,'
        '"methods":{"score":NaN,"score":2,"payload":"}],\\"config\\":false"}}\n'
    ).encode()
    real_load = load_strict_json_value_bytes
    parsed_slices: list[bytes] = []

    def reject_metric_parse(raw: bytes) -> object:
        if b'"score"' in raw or b"NaN" in raw:
            raise AssertionError("report methods are forbidden before activation")
        parsed_slices.append(raw)
        return real_load(raw)

    monkeypatch.setattr(controller, "load_strict_json_value_bytes", reject_metric_parse)

    assert derive_resolved_config(report, checkpoint, authority) == authority.expected_config_bytes
    assert authority.expected_config_bytes.rstrip() in parsed_slices


@pytest.mark.parametrize(
    "methods_raw",
    [
        b'{"score":NaN}',
        b'{"score":1,"score":2}',
        b'{"payload":"}],\\"config\\":false","nested":{"x":Infinity}}',
        b'{"payload":"\xff"}',
    ],
)
def test_resolved_config_treats_method_values_as_opaque_raw_bytes(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    methods_raw: bytes,
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    config = authority.expected_config_bytes.rstrip()
    report = b"".join(
        (
            b'{"name":"image-end-to-end-benchmark","dataset_name":"inshop",',
            b'"protocol":"proxy-anchor-resnet50-512","config":',
            config,
            b',"train_examples":2,"test_examples":2,"methods":',
            methods_raw,
            b"}\n",
        )
    )
    assert derive_resolved_config(report, checkpoint, authority) == authority.expected_config_bytes


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("name", "image-end-to-end"),
        ("dataset_name", "cub"),
        ("protocol", "hpl-resnet50-512"),
        ("train_examples", 3),
        ("test_examples", 3),
    ],
)
def test_resolved_config_rejects_each_report_authority_scalar_drift(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    field: str,
    drifted: object,
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    report = {
        "name": "image-end-to-end-benchmark",
        "dataset_name": "inshop",
        "protocol": "proxy-anchor-resnet50-512",
        "config": load_strict_json_bytes(authority.expected_config_bytes),
        "train_examples": 2,
        "test_examples": 2,
        "methods": {},
    }
    report[field] = drifted

    with pytest.raises(ValueError, match=f"report {field} drift"):
        derive_resolved_config(canonical_json_bytes(report), checkpoint, authority)


@pytest.mark.parametrize(
    "report",
    [
        b'{"name":"x","name":"y","dataset_name":"inshop","protocol":"query_gallery",'
        b'"config":{},"train_examples":2,"test_examples":2,"methods":{}}',
        b'{"name":"x","dataset_name":"inshop","protocol":"query_gallery",'
        b'"config":{"seed":0,"seed":1},"train_examples":2,"test_examples":2,'
        b'"methods":{}}',
        b'{"name":"x","dataset_name":"inshop","protocol":"query_gallery",'
        b'"config":[],"train_examples":2,"test_examples":2,"methods":{}}',
        b'{"name":"x","dataset_name":"inshop","protocol":"query_gallery",'
        b'"config":{},"train_examples":true,"test_examples":2,"methods":{}}',
        b'{"name":"x","dataset_name":"inshop","protocol":"query_gallery",'
        b'"config":{},"train_examples":2,"test_examples":2,"methods":[]}',
    ],
)
def test_resolved_config_rejects_top_level_or_config_structure_drift(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    report: bytes,
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    with pytest.raises(ValueError):
        derive_resolved_config(report, checkpoint, authority)


@pytest.mark.parametrize("drift", ["report", "checkpoint"])
def test_resolved_config_rejects_report_or_checkpoint_drift(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    drift: str,
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    config = json.loads(authority.expected_config_bytes)
    if drift == "report":
        config["seed"] = 1
    else:
        checkpoint = replace(checkpoint, training_config_sha256="0" * 64)
    report = canonical_json_bytes(
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "inshop",
            "protocol": "proxy-anchor-resnet50-512",
            "config": config,
            "train_examples": 2,
            "test_examples": 2,
            "methods": {},
        }
    )
    with pytest.raises(ValueError, match="drift"):
        derive_resolved_config(report, checkpoint, authority)


def test_resolved_config_rejects_equal_cross_type_seed(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    _capture, authority, checkpoint = sidecar_inputs
    config = json.loads(authority.expected_config_bytes)
    config["seed"] = False
    report = canonical_json_bytes(
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "inshop",
            "protocol": "proxy-anchor-resnet50-512",
            "config": config,
            "train_examples": 2,
            "test_examples": 2,
            "methods": {},
        }
    )
    with pytest.raises(ValueError, match="seed drift"):
        derive_resolved_config(report, checkpoint, authority)


def test_private_sidecar_environment_must_equal_authority(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture, authority, _checkpoint = sidecar_inputs
    bound = {"PATH": "/bound/bin", "PYTHONHASHSEED": "0"}
    authority = replace(
        authority,
        payload={**authority.payload, "execution": {"environment": bound}},
    )
    monkeypatch.setattr(controller.os, "environ", dict(bound))
    controller._validate_bound_environment(authority)

    monkeypatch.setattr(controller.os, "environ", {**bound, "EXTRA": "drift"})
    with pytest.raises(ValueError, match="environment drift"):
        controller._validate_bound_environment(authority)


def test_train_manifest_is_exact_and_bound(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    capture, authority, _checkpoint = sidecar_inputs

    payload = load_strict_json_bytes(derive_train_manifest(capture, authority))

    assert set(payload) == {
        "schema_version",
        "algorithm_id",
        "source_commit",
        "dataset_authority",
        "rows",
        "derivation",
    }
    assert payload["rows"] == [
        {"sample_index": 0, "example_id": capture.rows[0][1], "label": 1},
        {"sample_index": 1, "example_id": capture.rows[1][1], "label": 2},
    ]
    assert payload["dataset_authority"]["resolved_image_root"].endswith("/img/img")
    assert payload["derivation"]["source_files"] == authority.payload["source"]["files"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("recipe_digest", "0" * 64),
        ("train_count", 3),
        ("protocol", "self"),
        ("resolved_membership_sha256", "0" * 64),
        ("resolved_train_steps", 119),
        ("resolved_train_steps", 120.0),
        ("steps_per_epoch", 1),
        ("steps_per_epoch", 2.0),
        ("steps_per_epoch", True),
        ("total_epochs", 59),
        ("total_epochs", 60.0),
    ],
)
def test_train_manifest_rejects_capture_drift(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    field: str,
    replacement: object,
) -> None:
    capture, authority, _checkpoint = sidecar_inputs
    with pytest.raises(ValueError, match="drift"):
        derive_train_manifest(replace(capture, **{field: replacement}), authority)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("recipe_id", "different-recipe"), ("recipe_digest", "0" * 64)],
)
def test_train_manifest_rejects_recipe_authority_drift(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    field: str,
    replacement: str,
) -> None:
    capture, authority, _checkpoint = sidecar_inputs
    execution = {**authority.payload["execution"], field: replacement}
    authority = replace(authority, payload={**authority.payload, "execution": execution})

    with pytest.raises(ValueError, match="authority recipe"):
        derive_train_manifest(capture, authority)


def test_private_sidecar_rejects_recipe_drift_before_checkpoint_child(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, authority, _checkpoint_metadata = sidecar_inputs
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "authority.json"
    report = run_dir / "report.json"
    checkpoint = run_dir / "checkpoint.pt"
    manifest.write_bytes(b"{}\n")
    report.write_bytes(b"{}\n")
    checkpoint.write_bytes(b"checkpoint")
    execution = {
        **authority.payload["execution"],
        "environment": dict(os.environ),
        "argv": [],
        "recipe_id": "different-recipe",
    }
    payload = {
        **authority.payload,
        "authorization": {"manifest_path": "authority.json"},
        "execution": execution,
        "outputs": {
            "run_directory": "run",
            "report": {"path": "run/report.json"},
            "checkpoint": {"path": "run/checkpoint.pt"},
        },
    }
    authority = replace(authority, payload=payload, checkout_root=tmp_path)
    monkeypatch.setattr(controller, "validate_prelaunch", lambda _payload: authority)
    monkeypatch.setattr(controller, "_run_capture_child", lambda _authority: capture)
    monkeypatch.setattr(
        controller,
        "_run_metadata_child",
        forbidden("checkpoint metadata before recipe validation"),
    )

    with pytest.raises(ValueError, match="authority recipe ID drift"):
        controller.derive_sidecars_from_files(manifest, report, checkpoint, run_dir)


def _derive_with_synchronized_checkpoint_handoff(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
    *,
    replace_checkpoint: bool,
) -> SidecarFrame:
    capture, authority, checkpoint_metadata = sidecar_inputs
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = tmp_path / "authority.json"
    report_path = run_dir / "report.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    manifest_path.write_bytes(b"{}\n")
    checkpoint_path.write_bytes(b"original-checkpoint")
    config = {
        "batch_size": 180,
        "dataset_name": "inshop",
        "drop_last_train_batch": True,
        "objectives": ["proxy_anchor"],
        "protocol": "proxy-anchor-resnet50-512",
        "recipe_digest": controller.RECIPE_DIGEST,
        "recipe_id": controller.RECIPE_ID,
        "seed": 0,
    }
    config_bytes = canonical_json_bytes(config)
    report_path.write_bytes(
        canonical_json_bytes(
            {
                "name": "image-end-to-end-benchmark",
                "dataset_name": "inshop",
                "protocol": "proxy-anchor-resnet50-512",
                "config": config,
                "train_examples": 2,
                "test_examples": 2,
                "methods": {},
            }
        )
    )
    capture = replace(
        capture,
        config_bytes=config_bytes,
        resolved_train_steps=60,
        steps_per_epoch=1,
        total_epochs=60,
    )
    checkpoint_metadata = replace(
        checkpoint_metadata,
        training_step=60,
        training_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    execution = {
        **authority.payload["execution"],
        "environment": dict(os.environ),
        "argv": [],
    }
    payload = {
        **authority.payload,
        "authorization": {"manifest_path": "authority.json"},
        "execution": execution,
        "outputs": {
            "run_directory": "run",
            "report": {"path": "run/report.json"},
            "checkpoint": {"path": "run/checkpoint.pt"},
        },
    }
    authority = replace(
        authority,
        payload=payload,
        checkout_root=tmp_path,
        expected_config_bytes=config_bytes,
        expected_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        expected_train_steps=60,
        steps_per_epoch=1,
        total_epochs=60,
    )
    metadata_complete = threading.Event()

    def metadata_child(
        _authority: PrelaunchAuthority, _checkpoint_path: Path
    ) -> BoundCheckpointMetadata:
        result = BoundCheckpointMetadata(bind_external_file(checkpoint_path), checkpoint_metadata)
        metadata_complete.set()
        return result

    real_validate_completed_epoch = controller.validate_completed_epoch

    def after_metadata(*args: object, **kwargs: object) -> int:
        assert metadata_complete.wait(timeout=1)
        if replace_checkpoint:
            replacement = run_dir / "replacement.pt"
            replacement.write_bytes(b"replacement-checkpoint")
            os.replace(replacement, checkpoint_path)
        return real_validate_completed_epoch(*args, **kwargs)  # type: ignore[arg-type]

    def binding_child(
        _authority: PrelaunchAuthority, path: Path, expected: ExternalFileBinding
    ) -> ExternalFileBinding:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "pass201_pa_source_v2_contract.py"),
                "restricted-binding-child",
                "--checkpoint",
                str(path),
            ],
            cwd=repo,
            env=dict(os.environ),
            input=encode_checkpoint_binding_request(expected),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("checkpoint input drift")
        return decode_checkpoint_binding_response(result.stdout, expected, path)

    monkeypatch.setattr(controller, "validate_prelaunch", lambda _payload: authority)
    monkeypatch.setattr(controller, "_run_capture_child", lambda _authority: capture)
    monkeypatch.setattr(controller, "_run_metadata_child", metadata_child)
    monkeypatch.setattr(controller, "_run_binding_child", binding_child)
    monkeypatch.setattr(controller, "validate_completed_epoch", after_metadata)
    return controller.derive_sidecars_from_files(
        manifest_path, report_path, checkpoint_path, run_dir
    )


def test_private_sidecar_rejects_checkpoint_replaced_after_metadata_child(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="checkpoint input drift"):
        _derive_with_synchronized_checkpoint_handoff(
            tmp_path, sidecar_inputs, monkeypatch, replace_checkpoint=True
        )


def test_private_sidecar_accepts_unchanged_checkpoint_after_metadata_child(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _derive_with_synchronized_checkpoint_handoff(
        tmp_path, sidecar_inputs, monkeypatch, replace_checkpoint=False
    )
    assert frame.config_bytes
    assert frame.manifest_bytes


def test_schedule_validates_completed_epoch(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    capture, _authority, checkpoint = sidecar_inputs
    assert validate_completed_epoch(capture, checkpoint.training_step, 360, 180) == 60


@pytest.mark.parametrize(
    ("capture_update", "checkpoint_step", "optimization_count", "batch_size"),
    [
        ({"steps_per_epoch": 0}, 120, 360, 180),
        ({"steps_per_epoch": 3}, 120, 360, 180),
        ({"steps_per_epoch": 2.0}, 120, 360, 180),
        ({"resolved_train_steps": 120.0}, 120, 360, 180),
        ({"total_epochs": 60.0}, 120, 360, 180),
        ({"steps_per_epoch": True}, 120, 360, 180),
        ({}, 119, 360, 180),
        ({"resolved_train_steps": 119}, 119, 360, 180),
        ({"total_epochs": 59}, 120, 360, 180),
    ],
)
def test_schedule_rejects_partial_or_drifted_epoch(
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    capture_update: dict[str, object],
    checkpoint_step: int,
    optimization_count: int,
    batch_size: int,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    with pytest.raises(ValueError):
        validate_completed_epoch(
            replace(capture, **capture_update),
            checkpoint_step,
            optimization_count,
            batch_size,
        )


def test_sidecar_frame_round_trips_exact_bytes() -> None:
    frame = SidecarFrame(
        pid=123,
        config_bytes=b'{"x":1}\n',
        manifest_bytes=b'{"rows":[]}\n',
        config_sha256=hashlib.sha256(b'{"x":1}\n').hexdigest(),
        manifest_sha256=hashlib.sha256(b'{"rows":[]}\n').hexdigest(),
    )
    assert decode_sidecar_frame(encode_sidecar_frame(frame)) == frame


@pytest.mark.parametrize("mutation", ["trailing", "hash", "length"])
def test_sidecar_frame_rejects_ambiguous_or_corrupt_bytes(mutation: str) -> None:
    config = b"{}\n"
    manifest = b'{"rows":[]}\n'
    encoded = bytearray(
        encode_sidecar_frame(
            SidecarFrame(
                7,
                config,
                manifest,
                hashlib.sha256(config).hexdigest(),
                hashlib.sha256(manifest).hexdigest(),
            )
        )
    )
    if mutation == "trailing":
        encoded.extend(b"x")
    elif mutation == "hash":
        encoded[-1] ^= 1
    else:
        start = len(controller.SIDECAR_FRAME_MAGIC) + 8
        encoded[start : start + 8] = struct.pack(">Q", 2**63)
    with pytest.raises(ValueError):
        decode_sidecar_frame(bytes(encoded))


def test_private_derive_sidecars_cli_emits_frame_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    manifest = tmp_path / "authority.json"
    report = tmp_path / "run" / "report.json"
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    output_dir = tmp_path / "run"
    expected = SidecarFrame(
        os.getpid(),
        b"{}\n",
        b'{"rows":[]}\n',
        hashlib.sha256(b"{}\n").hexdigest(),
        hashlib.sha256(b'{"rows":[]}\n').hexdigest(),
    )
    monkeypatch.setattr(controller, "derive_sidecars_from_files", lambda *args: expected)

    assert (
        main(
            [
                "derive-sidecars",
                "--manifest",
                str(manifest),
                "--report",
                str(report),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert decode_sidecar_frame(capsysbinary.readouterr().out) == expected
    assert not output_dir.exists()


def test_sidecar_input_path_rejects_symlink_alias(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_bytes(b"{}\n")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        controller._exact_regular_path(alias)


def test_private_sidecar_rejects_alternative_report_before_reading_it(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture, authority, _checkpoint = sidecar_inputs
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "authority.json"
    manifest.write_bytes(b"{}\n")
    expected_report = run_dir / "report.json"
    expected_report.write_bytes(b"{}\n")
    alternative_report = run_dir / "alternative-report.json"
    alternative_report.write_bytes(b"forbidden-report-bytes")
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        **authority.payload,
        "authorization": {"manifest_path": "authority.json"},
        "execution": {"environment": dict(os.environ), "argv": []},
        "outputs": {
            "run_directory": "run",
            "report": {"path": "run/report.json"},
            "checkpoint": {"path": "run/checkpoint.pt"},
        },
    }
    authority = replace(authority, payload=payload, checkout_root=tmp_path)
    monkeypatch.setattr(controller, "validate_prelaunch", lambda _payload: authority)
    reads: list[Path] = []
    real_read = controller._read_immutable_regular

    def record_read(path: Path) -> bytes:
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(controller, "_read_immutable_regular", record_read)

    with pytest.raises(ValueError, match="alternative report path"):
        controller.derive_sidecars_from_files(manifest, alternative_report, checkpoint, run_dir)

    assert reads == [manifest]


def test_two_sidecar_children_must_be_distinct_and_byte_identical() -> None:
    config = b"{}\n"
    manifest = b'{"rows":[]}\n'
    first = SidecarFrame(
        10,
        config,
        manifest,
        hashlib.sha256(config).hexdigest(),
        hashlib.sha256(manifest).hexdigest(),
    )
    second = replace(first, pid=11)
    assert validate_sidecar_identity(first, second) == (config, manifest)

    for bad in (
        replace(second, pid=10),
        replace(
            second,
            config_bytes=b'{"x":1}\n',
            config_sha256=hashlib.sha256(b'{"x":1}\n').hexdigest(),
        ),
        replace(
            second,
            manifest_bytes=b'{"rows":[1]}\n',
            manifest_sha256=hashlib.sha256(b'{"rows":[1]}\n').hexdigest(),
        ),
    ):
        with pytest.raises(ValueError):
            validate_sidecar_identity(first, bad)


def _run_mocked_external_sidecar_child(checkout: Path) -> subprocess.CompletedProcess[bytes]:
    program = r"""
import hashlib, json, os, struct, sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint_path = (root / "run" / "checkpoint.pt").absolute()
def forbid_checkpoint_open(event, args):
    if event != "open" or not args:
        return
    try:
        opened = Path(os.path.abspath(os.fspath(args[0])))
    except TypeError:
        return
    if opened == checkpoint_path:
        raise RuntimeError("orchestrator opened produced checkpoint")
sys.addaudithook(forbid_checkpoint_open)

import run_pass201_pa_source_v2 as c
from pass201_pa_source_v2_contract import (
    BoundCheckpointMetadata, CheckpointArch, CheckpointMetadata, PrelaunchAuthority,
    ExternalFileBinding,
)

config = {
    "batch_size": 180,
    "dataset_name": "inshop",
    "drop_last_train_batch": True,
    "objectives": ["proxy_anchor"],
    "protocol": "proxy-anchor-resnet50-512",
    "recipe_digest": c.RECIPE_DIGEST,
    "recipe_id": c.RECIPE_ID,
    "seed": 0,
}
config_bytes = c.canonical_json_bytes(config)
rows = ((0, "inshop-train-img/a.jpg", 1), (1, "inshop-train-img/b.jpg", 2))
row_objects = [{"sample_index":i,"example_id":e,"label":label} for i,e,label in rows]
digest = hashlib.sha256()
for row in row_objects:
    encoded = c.canonical_json_bytes(row)
    digest.update(struct.pack(">Q", len(encoded)) + encoded)
row_hash = digest.hexdigest()
payload = {
    "authorization": {"manifest_path": "authority.json"},
    "outputs": {
        "run_directory": "run",
        "report": {"path": "run/report.json"},
        "checkpoint": {"path": "run/checkpoint.pt"},
    },
    "execution": {
        "environment": dict(os.environ),
        "argv": [],
        "recipe_id": c.RECIPE_ID,
        "recipe_digest": c.RECIPE_DIGEST,
    },
    "dataset": {
        "root": str(root / "dataset"),
        "partition": {"sha256": "f" * 64},
        "resolved_image_root": str(root / "dataset" / "img" / "img"),
        "image_tree": {"root_sha256": "1" * 64},
        "bundle": {
            "train": 2,
            "query": 2,
            "gallery": 2,
            "protocol": "query_gallery",
            "protocol_name": "deepfashion-inshop-official",
        },
        "selection_policy": "full_official_partition",
        "optimization_authority": {
            "row_count": 2,
            "identity_count": 2,
            "ordered_row_sha256": row_hash,
            "resolved_membership_sha256": "c" * 64,
        },
    },
    "source": {"files": [
        {"path":"src/sfora/data.py","git_mode":"100644","bytes":1,"sha256":"a"*64,"git_blob":"b"*40}
    ]},
}
authority = PrelaunchAuthority(
    payload, "9" * 40, root, config_bytes,
    hashlib.sha256(config_bytes).hexdigest(), 60, 1, 60,
)
capture = c.CapturedAuthority(
    config_bytes, c.RECIPE_ID, c.RECIPE_DIGEST, 2, 2, 2,
    "query_gallery", "deepfashion-inshop-official", rows, "c" * 64, 60, 1, 60,
)
arch = CheckpointArch(
    "bn_inception", "bn_inception_52deb4733", "avg_max", 512, "kaiming_normal", False,
)
checkpoint = CheckpointMetadata(
    "2" * 64,
    (
        "arch", "artifact_selection", "evaluation_model_source",
        "state_dict", "training_config", "training_step",
    ),
    "final_training_state", "student", 60, arch, "3" * 64,
    authority.expected_config_sha256, 1,
)
def validate_manifest(value):
    if value != {"bound": True}:
        raise ValueError("manifest algorithm drift")
    return authority
c.validate_prelaunch = validate_manifest
c._run_capture_child = lambda *_args: capture
checkpoint_stat = os.stat(checkpoint_path, follow_symlinks=False)
checkpoint_binding = ExternalFileBinding(
    checkpoint_path, checkpoint_stat.st_mode, checkpoint_stat.st_dev,
    checkpoint_stat.st_ino, len(b"restricted-checkpoint-placeholder"),
    hashlib.sha256(b"restricted-checkpoint-placeholder").hexdigest(),
)
c._run_metadata_child = lambda *_args: BoundCheckpointMetadata(
    checkpoint_binding, checkpoint
)
c._run_binding_child = lambda _authority, _path, binding: binding
raise SystemExit(c.main([
    "derive-sidecars", "--manifest", str(root / "authority.json"),
    "--report", str(root / "run" / "report.json"),
    "--checkpoint", str(root / "run" / "checkpoint.pt"),
    "--output-dir", str(root / "run"),
]))
"""
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'scripts'}"
    return subprocess.run(
        [sys.executable, "-c", program, str(checkout)],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
    )


def test_private_sidecar_orchestrator_never_opens_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "authority.json").write_bytes(b'{"bound":true}\n')
    config = {
        "batch_size": 180,
        "dataset_name": "inshop",
        "drop_last_train_batch": True,
        "objectives": ["proxy_anchor"],
        "protocol": "proxy-anchor-resnet50-512",
        "recipe_digest": controller.RECIPE_DIGEST,
        "recipe_id": controller.RECIPE_ID,
        "seed": 0,
    }
    (run_dir / "report.json").write_bytes(
        canonical_json_bytes(
            {
                "name": "image-end-to-end-benchmark",
                "dataset_name": "inshop",
                "protocol": "proxy-anchor-resnet50-512",
                "config": config,
                "train_examples": 2,
                "test_examples": 2,
                "methods": {},
            }
        )
    )
    (run_dir / "checkpoint.pt").write_bytes(b"restricted-checkpoint-placeholder")

    result = _run_mocked_external_sidecar_child(tmp_path)

    assert result.returncode == 0, result.stderr.decode()


def test_two_fresh_private_sidecar_children_are_identical_and_fail_on_input_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "authority.json").write_bytes(b'{"bound":true}\n')
    config = {
        "batch_size": 180,
        "dataset_name": "inshop",
        "drop_last_train_batch": True,
        "objectives": ["proxy_anchor"],
        "protocol": "proxy-anchor-resnet50-512",
        "recipe_digest": controller.RECIPE_DIGEST,
        "recipe_id": controller.RECIPE_ID,
        "seed": 0,
    }
    (run_dir / "report.json").write_bytes(
        canonical_json_bytes(
            {
                "name": "image-end-to-end-benchmark",
                "dataset_name": "inshop",
                "protocol": "proxy-anchor-resnet50-512",
                "config": config,
                "train_examples": 2,
                "test_examples": 2,
                "methods": {"forbidden": {"executed_train_steps": 120}},
            }
        )
    )
    (run_dir / "checkpoint.pt").write_bytes(b"restricted-checkpoint-placeholder")

    first_result = _run_mocked_external_sidecar_child(tmp_path)
    second_result = _run_mocked_external_sidecar_child(tmp_path)
    assert first_result.returncode == second_result.returncode == 0
    first = decode_sidecar_frame(first_result.stdout)
    second = decode_sidecar_frame(second_result.stdout)
    validate_sidecar_identity(first, second)

    (tmp_path / "authority.json").write_bytes(b'{"bound":true,"algorithm":"drift"}\n')
    rejected = _run_mocked_external_sidecar_child(tmp_path)
    assert rejected.returncode != 0
    assert rejected.stdout == b""


EXPECTED_CONTROLLER_ORDER = [
    "strict_manifest",
    "detached_exact_git_topology",
    "replacement_runtime_bindings",
    "frozen_preflight_absence",
    "private_run_directory_lock",
    "one_training_child",
    "training_exit_zero",
    "postflight_equality",
    "freeze_scientific_outputs",
    "restricted_checkpoint_metadata",
    "sidecar_child_1",
    "sidecar_child_2",
    "publish_resolved_config",
    "publish_train_manifest",
    "publish_receipt",
]


def _freeze_test_environment(checkout: Path) -> dict[str, str]:
    return {
        "HOME": "/operator",
        "PATH": "/venv/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{checkout}/src",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": "/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XDG_CACHE_HOME": "/operator/.cache",
        "TORCH_HOME": "/operator/.cache/torch",
    }


def _repo_binding(path: str, fill: str) -> dict[str, object]:
    return {
        "path": path,
        "git_mode": "100644",
        "bytes": 10,
        "sha256": fill * 64,
        "git_blob": fill * 40,
    }


def _external_binding(path: str, fill: str) -> dict[str, object]:
    return {
        "path": path,
        "mode": stat.S_IFREG | 0o755,
        "device": 1,
        "inode": 2,
        "bytes": 10,
        "sha256": fill * 64,
    }


def _fake_freeze_runtime(checkout: Path) -> dict[str, object]:
    environment = _freeze_test_environment(checkout)
    return {
        "source_commit": "a" * 40,
        "controller": _repo_binding("scripts/run_pass201_pa_source_v2.py", "a"),
        "source_files": [
            _repo_binding("scripts/pass201_pa_source_v2_contract.py", "b"),
            _repo_binding("src/sfora/cli.py", "c"),
        ],
        "python_tree": {
            "root": "src/sfora",
            "algorithm": "pass201-length-framed-merkle-v1",
            "count": 2,
            "bytes": 20,
            "root_sha256": "d" * 64,
        },
        "pyproject": _repo_binding("pyproject.toml", "e"),
        "lockfile": _repo_binding("uv.lock", "f"),
        "python": _external_binding("/venv/bin/python", "1"),
        "python_realpath": "/usr/bin/python3",
        "python_version": "3.12.11",
        "git": _external_binding("/usr/bin/git", "2"),
        "python_packages": {"bytes": 12, "sha256": "3" * 64},
        "python_import_roots": [{"entry": "/missing", "status": "nonexistent"}],
        "environment": environment,
        "pretrained_checkpoint": _external_binding("/models/pretrained.pt", "4"),
        "partition": _external_binding(
            "/home/riomus/datasets/inshop_official_standard/Eval/list_eval_partition.txt",
            "5",
        ),
        "partition_lines": 7,
        "image_root_link": {
            "path": "/home/riomus/datasets/inshop_official_standard/Img",
            "target": "img",
            "lstat_mode": stat.S_IFLNK | 0o777,
        },
        "image_tree": {
            "root": "/home/riomus/datasets/inshop_official_standard/img/img",
            "algorithm": "pass201-length-framed-merkle-v1",
            "count": 6,
            "bytes": 60,
            "root_sha256": "6" * 64,
        },
    }


def _prelaunch_freeze_output(checkout: Path, ordinal: int) -> Path:
    return checkout.parent / f"{checkout.name}.pass201-prelaunch-freeze-{ordinal}.tmp"


def _freeze_args(checkout: Path, output: Path) -> controller.FreezeArgs:
    return controller.FreezeArgs(
        checkout_root=checkout,
        dataset_root=controller.DATASET_ROOT,
        python_path=Path("/venv/bin/python"),
        frozen_absence_checked_utc="2026-08-09T00:00:00Z",
        output_path=output,
    )


@pytest.mark.parametrize("ordinal", [1, 2])
def test_prelaunch_output_accepts_only_exact_normalized_sibling(
    tmp_path: Path,
    ordinal: int,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)

    controller._require_frozen_absence(
        _freeze_args(checkout, _prelaunch_freeze_output(checkout, ordinal))
    )


@pytest.mark.parametrize(
    "case",
    [
        "canonical_manifest",
        "relative",
        "parent_alias",
        "symlink_alias",
        "other_sibling",
        "inside_checkout",
        "inside_dataset",
        "inside_run_directory",
        "inside_import_tree",
    ],
)
def test_prelaunch_output_rejects_every_other_path_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    selected = _prelaunch_freeze_output(checkout, 1)
    if case == "canonical_manifest":
        output = checkout / controller.PRELAUNCH_PATH
    elif case == "relative":
        output = Path(selected.name)
    elif case == "parent_alias":
        alias_component = checkout.parent / "existing-alias-component"
        alias_component.mkdir()
        output = alias_component / ".." / selected.name
    elif case == "symlink_alias":
        alias = tmp_path / "parent-link"
        alias.symlink_to(checkout.parent, target_is_directory=True)
        output = alias / selected.name
    elif case == "other_sibling":
        output = checkout.parent / f"{checkout.name}.pass201-prelaunch-freeze-3.tmp"
    elif case == "inside_checkout":
        output = checkout / selected.name
    elif case == "inside_dataset":
        output = controller.DATASET_ROOT / selected.name
    elif case == "inside_run_directory":
        output = checkout / controller.RUN_DIRECTORY / selected.name
    elif case == "inside_import_tree":
        output = Path(sys.base_prefix).resolve(strict=True) / "lib" / selected.name
    else:  # pragma: no cover - parameter set is deliberately closed
        raise AssertionError(case)
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        forbidden("freeze capture preparation"),
    )

    with pytest.raises(ValueError, match="prelaunch output path"):
        controller.freeze_authority(_freeze_args(checkout, output))


@pytest.mark.parametrize("existing_kind", ["regular", "symlink"])
def test_prelaunch_output_rejects_existing_selected_sink_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    selected = _prelaunch_freeze_output(checkout, 1)
    if existing_kind == "regular":
        selected.write_bytes(b"occupied\n")
    else:
        selected.symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        forbidden("freeze capture preparation"),
    )

    with pytest.raises(ValueError, match="selected prelaunch output already exists"):
        controller.freeze_authority(_freeze_args(checkout, selected))


def test_prelaunch_output_requires_canonical_manifest_absent_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    canonical = checkout / controller.PRELAUNCH_PATH
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"already published\n")
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        forbidden("freeze capture preparation"),
    )

    with pytest.raises(ValueError, match="canonical prelaunch manifest already exists"):
        controller.freeze_authority(
            _freeze_args(checkout, _prelaunch_freeze_output(checkout, 1))
        )


def test_prelaunch_output_ignores_other_permitted_sink(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    first = _prelaunch_freeze_output(checkout, 1)
    first.write_bytes(b"first authority bytes\n")

    controller._require_frozen_absence(
        _freeze_args(checkout, _prelaunch_freeze_output(checkout, 2))
    )


@pytest.mark.parametrize("parent_state", ["missing", "symlink"])
def test_prelaunch_output_rejects_wrong_canonical_parent_semantics_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_state: str,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if parent_state == "symlink":
        external_docs = tmp_path / "external-docs"
        external_docs.mkdir()
        (checkout / "docs").symlink_to(external_docs, target_is_directory=True)
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        forbidden("freeze capture preparation"),
    )

    with pytest.raises(ValueError, match="canonical manifest parent"):
        controller.freeze_authority(
            _freeze_args(checkout, _prelaunch_freeze_output(checkout, 1))
        )


def test_prelaunch_output_requires_same_filesystem_as_canonical_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    canonical_parent = checkout / "docs"
    canonical_parent.mkdir(parents=True)
    real_stat = os.stat

    def mismatched_device(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if (
            kwargs.get("follow_symlinks") is False
            and isinstance(path, (str, os.PathLike))
            and Path(path) == canonical_parent
        ):
            fields = list(result)
            fields[2] = result.st_dev + 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(controller.os, "stat", mismatched_device)

    with pytest.raises(ValueError, match="same filesystem"):
        controller._require_frozen_absence(
            _freeze_args(checkout, _prelaunch_freeze_output(checkout, 1))
        )


def _run_tiny_freeze_process(
    checkout: Path,
    inputs_path: Path,
    output: Path,
) -> subprocess.CompletedProcess[bytes]:
    program = r"""
import json
import os
import sys
from pathlib import Path

import run_pass201_pa_source_v2 as controller

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = payload["runtime"]
capture_payload = payload["capture"]
capture_payload["config_bytes"] = capture_payload["config_bytes"].encode("utf-8")
capture_payload["rows"] = tuple(tuple(row) for row in capture_payload["rows"])
capture = controller.CapturedAuthority(**capture_payload)
controller._build_replacement_environment = lambda _args: runtime["environment"]
controller._bind_freeze_runtime = lambda _args, _environment: runtime
controller._run_freeze_capture_child = lambda _args, _argv, _environment: capture
status = controller.main(
    [
        "freeze-authority",
        "--frozen-absence-checked-utc",
        "2026-08-09T00:00:00Z",
        "--output",
        sys.argv[2],
    ]
)
print(os.getpid())
raise SystemExit(status)
"""
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{repo / 'scripts'}:{repo / 'src'}"
    return subprocess.run(
        [sys.executable, "-c", program, str(inputs_path), str(output)],
        cwd=checkout,
        env=environment,
        capture_output=True,
        check=False,
    )


def test_prelaunch_output_two_fresh_top_level_processes_are_byte_identical(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    checkout = tmp_path / "tiny-checkout"
    (checkout / "docs").mkdir(parents=True)
    (checkout / "tracked.txt").write_text("source checkout\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.name", "Pass201 Test")
    _git(checkout, "config", "user.email", "pass201@example.invalid")
    _git(checkout, "add", "tracked.txt")
    _git(checkout, "commit", "-q", "-m", "source")
    runtime = _fake_freeze_runtime(checkout)
    capture_payload = {
        "config_bytes": capture.config_bytes.decode("utf-8"),
        "recipe_id": capture.recipe_id,
        "recipe_digest": capture.recipe_digest,
        "train_count": capture.train_count,
        "query_count": capture.query_count,
        "gallery_count": capture.gallery_count,
        "protocol": capture.protocol,
        "protocol_name": capture.protocol_name,
        "rows": capture.rows,
        "resolved_membership_sha256": capture.resolved_membership_sha256,
        "resolved_train_steps": capture.resolved_train_steps,
        "steps_per_epoch": capture.steps_per_epoch,
        "total_epochs": capture.total_epochs,
    }
    inputs_path = tmp_path / "tiny-freeze-inputs.json"
    inputs_path.write_text(
        json.dumps({"runtime": runtime, "capture": capture_payload}),
        encoding="utf-8",
    )
    first_output = _prelaunch_freeze_output(checkout, 1)
    second_output = _prelaunch_freeze_output(checkout, 2)

    first = _run_tiny_freeze_process(checkout, inputs_path, first_output)
    assert first.returncode == 0, first.stderr.decode()
    assert first_output.exists()
    assert not (checkout / controller.PRELAUNCH_PATH).exists()
    assert _git(checkout, "status", "--porcelain=v1") == ""
    second = _run_tiny_freeze_process(checkout, inputs_path, second_output)
    assert second.returncode == 0, second.stderr.decode()

    assert int(first.stdout) != int(second.stdout)
    assert first_output.read_bytes() == second_output.read_bytes()
    assert not (checkout / controller.PRELAUNCH_PATH).exists()
    assert _git(checkout, "status", "--porcelain=v1") == ""


def test_freeze_authority_captures_twice_and_emits_canonical_strict_manifest(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = _prelaunch_freeze_output(checkout, 1)
    runtime = _fake_freeze_runtime(checkout)
    capture_calls: list[int] = []
    absence_calls: list[Path] = []

    monkeypatch.setattr(
        controller,
        "_bind_freeze_runtime",
        lambda _args, _environment: runtime,
        raising=False,
    )
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        lambda _args: runtime["environment"],
        raising=False,
    )

    def capture_child(_args: object, argv: list[str], environment: dict[str, str]):
        capture_calls.append(len(capture_calls) + 1)
        assert argv[0] == "/venv/bin/python"
        assert environment == runtime["environment"]
        return capture

    monkeypatch.setattr(controller, "_run_freeze_capture_child", capture_child, raising=False)

    def absence(args: object) -> None:
        absence_calls.append(args.checkout_root)

    monkeypatch.setattr(controller, "_require_frozen_absence", absence, raising=False)
    args = controller.FreezeArgs(
        checkout_root=checkout,
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        python_path=Path("/venv/bin/python"),
        frozen_absence_checked_utc="2026-08-09T00:00:00Z",
        output_path=output,
    )

    manifest_bytes = controller.freeze_authority(args)

    payload = load_strict_json_bytes(manifest_bytes)
    authority = controller.validate_prelaunch(payload)
    assert canonical_json_bytes(payload) == manifest_bytes
    assert authority.source_commit == "a" * 40
    assert payload["execution"]["environment"] == runtime["environment"]
    assert payload["execution"]["recipe_id"] == capture.recipe_id
    assert payload["execution"]["recipe_digest"] == capture.recipe_digest
    assert payload["dataset"]["bundle"] == {
        "train": 2,
        "query": 2,
        "gallery": 2,
        "protocol": "query_gallery",
        "protocol_name": "deepfashion-inshop-official",
    }
    assert payload["authorization"]["frozen_absence_checked_utc"] == ("2026-08-09T00:00:00Z")
    assert set(payload["authorization"]["frozen_absence"].values()) == {"ENOENT"}
    assert capture_calls == [1, 2]
    assert absence_calls == [checkout, checkout, checkout]
    assert not output.exists()


def test_freeze_authority_checks_first_capture_absence_before_second_child(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    args = _freeze_args(checkout, _prelaunch_freeze_output(checkout, 1))
    runtime = _fake_freeze_runtime(checkout)
    capture_calls = 0
    monkeypatch.setattr(
        controller,
        "_bind_freeze_runtime",
        lambda _args, _environment: runtime,
    )
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        lambda _args: runtime["environment"],
    )

    def dirty_capture(*_args: object) -> CapturedAuthority:
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 1:
            (checkout / controller.RUN_DIRECTORY).mkdir(parents=True)
        return capture

    monkeypatch.setattr(controller, "_run_freeze_capture_child", dirty_capture)

    with pytest.raises(ValueError, match="private run directory already exists"):
        controller.freeze_authority(args)

    assert capture_calls == 1


def test_freeze_authority_rejects_capture_child_disagreement(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    args = _freeze_args(checkout, _prelaunch_freeze_output(checkout, 1))
    runtime = _fake_freeze_runtime(checkout)
    captures = iter((capture, replace(capture, recipe_digest="0" * 64)))
    monkeypatch.setattr(
        controller,
        "_bind_freeze_runtime",
        lambda _args, _environment: runtime,
        raising=False,
    )
    monkeypatch.setattr(
        controller,
        "_build_replacement_environment",
        lambda _args: runtime["environment"],
        raising=False,
    )
    monkeypatch.setattr(
        controller,
        "_run_freeze_capture_child",
        lambda *_args: next(captures),
        raising=False,
    )
    monkeypatch.setattr(controller, "_require_frozen_absence", lambda _args: None, raising=False)

    with pytest.raises(ValueError, match="capture children disagree"):
        controller.freeze_authority(args)

    assert not args.output_path.exists()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_authorized_git_fixture(
    tmp_path: Path,
    capture: CapturedAuthority,
    *,
    topology: str = "valid",
    payload_mutation: tuple[str, object] | None = None,
) -> SimpleNamespace:
    repo = tmp_path / "authorized-checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Pass201 Test")
    _git(repo, "config", "user.email", "pass201@example.invalid")
    (repo / ".gitignore").write_text("reports/generated/\n", encoding="utf-8")
    (repo / "source-marker.txt").write_text("source C\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "source-marker.txt")
    _git(repo, "commit", "-q", "-m", "source C")
    source_commit = _git(repo, "rev-parse", "HEAD")

    environment = _freeze_test_environment(repo)
    runtime = _fake_freeze_runtime(repo)
    runtime["source_commit"] = source_commit
    git_executable = Path(shutil.which("git") or "").resolve(strict=True)
    runtime["git"] = controller._external_file_payload(bind_external_file(git_executable))
    args = controller.FreezeArgs(
        checkout_root=repo,
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        python_path=Path("/venv/bin/python"),
        frozen_absence_checked_utc="2026-08-09T00:00:00Z",
        output_path=repo / "docs" / "pass201_pa_source_v2_prelaunch.json",
    )
    payload = controller._build_prelaunch_payload(args, capture, runtime)
    if payload_mutation is not None:
        mutation, value = payload_mutation
        if mutation == "query_count":
            payload["dataset"]["bundle"]["query"] = value
        elif mutation == "recipe_id":
            payload["execution"]["recipe_id"] = value
        else:  # pragma: no cover - test fixture is deliberately closed
            raise AssertionError(mutation)
    controller.validate_prelaunch(payload)

    if topology == "extra_parent":
        (repo / "intermediate.txt").write_text("not authorized\n", encoding="utf-8")
        _git(repo, "add", "intermediate.txt")
        _git(repo, "commit", "-q", "-m", "intermediate")
    manifest = args.output_path
    manifest.parent.mkdir()
    manifest.write_bytes(canonical_json_bytes(payload))
    _git(repo, "add", manifest.relative_to(repo).as_posix())
    if topology == "extra_diff":
        (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
        _git(repo, "add", "extra.txt")
    _git(repo, "commit", "-q", "-m", "authorization A")
    authorization_commit = _git(repo, "rev-parse", "HEAD")
    if topology != "branch_head":
        _git(repo, "checkout", "-q", "--detach", authorization_commit)
    authority = controller.validate_prelaunch(payload)
    return SimpleNamespace(
        repo=repo,
        manifest=manifest,
        manifest_bytes=canonical_json_bytes(payload),
        authority=authority,
        runtime=runtime,
        environment=environment,
        source_commit=source_commit,
        authorization_commit=authorization_commit,
    )


def test_run_preflight_accepts_only_detached_sole_manifest_addition(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture)
    monkeypatch.setattr(controller, "_ambient_environment", lambda: fixture.environment)
    monkeypatch.setattr(controller, "_bind_runtime_after", lambda _authority: fixture.runtime)

    authorized = controller.validate_runtime_preflight(fixture.manifest)

    assert authorized.authorization_commit == fixture.authorization_commit
    assert authorized.manifest_bytes == fixture.manifest_bytes
    assert authorized.authority.source_commit == fixture.source_commit


def test_run_preflight_rejects_alternate_compatible_controller_runtime_before_side_effect(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture)
    repo = Path(__file__).resolve().parents[1]
    expected_python = Path(sys.executable).absolute()
    alternate_python = expected_python.with_name("python3")
    assert alternate_python.exists() and alternate_python != expected_python
    payload = load_strict_json_bytes(fixture.manifest_bytes)
    python_binding = controller._external_file_payload(
        bind_external_file(expected_python.resolve(strict=True))
    )
    python_binding["path"] = expected_python.as_posix()
    payload["execution"]["python"] = python_binding
    payload["execution"]["python_realpath"] = expected_python.resolve(strict=True).as_posix()
    payload["execution"]["argv"][0] = expected_python.as_posix()
    payload["execution"]["environment"]["PATH"] = (
        f"{expected_python.parent.as_posix()}:/usr/bin:/bin"
    )
    authority_bytes = canonical_json_bytes(payload)
    authority_path = tmp_path / "alternate-runtime-authority.json"
    authority_path.write_bytes(authority_bytes)
    side_effect = tmp_path / "preflight-side-effect"
    program = """
import copy
import sys
from pathlib import Path

import run_pass201_pa_source_v2 as controller

authority_path = Path(sys.argv[1])
side_effect = Path(sys.argv[2])
authority_bytes = authority_path.read_bytes()
authority = controller.validate_prelaunch(controller.load_strict_json_bytes(authority_bytes))
expected = controller._expected_runtime_bindings(authority)
expected_python = authority.payload["execution"]["python"]["path"]

def bind_measured_runtime(args, _environment):
    current = copy.deepcopy(expected)
    actual_python = args.python_path.absolute().as_posix()
    if actual_python != expected_python:
        current["python"]["path"] = actual_python
        current["python_realpath"] = args.python_path.resolve(strict=True).as_posix()
        current["python_packages"]["sha256"] = "0" * 64
        current["python_import_roots"] = [
            *current["python_import_roots"],
            {"entry": "/alternate-runtime", "status": "nonexistent"},
        ]
    return current

def forbidden_side_effect(_authority):
    side_effect.write_text("reached", encoding="utf-8")
    raise RuntimeError("preflight reached a side effect")

controller._load_manifest_authority = lambda _path: (
    authority,
    authority_path,
    authority_bytes,
)
controller.validate_authorization_topology = lambda _checkout, _authority: "b" * 40
controller._require_authority_scope = lambda _authority: None
controller._require_replacement_environment = lambda _authority: None
controller._bind_freeze_runtime = bind_measured_runtime
controller._record_preflight_absence = forbidden_side_effect
try:
    controller.validate_runtime_preflight(authority_path)
except ValueError as error:
    if "executing controller Python" not in str(error):
        raise
    print("alternate controller runtime rejected")
    raise SystemExit(0)
except RuntimeError:
    print("side effect reached")
    raise SystemExit(8)
raise SystemExit(9)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{repo / 'scripts'}{os.pathsep}{repo / 'src'}"

    result = subprocess.run(
        [str(alternate_python), "-c", program, str(authority_path), str(side_effect)],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout == "alternate controller runtime rejected\n"
    assert not side_effect.exists()
    assert not (fixture.repo / controller.RUN_DIRECTORY).exists()


@pytest.mark.parametrize("topology", ["branch_head", "extra_diff", "extra_parent"])
def test_run_preflight_rejects_branch_extra_diff_or_parent(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture, topology=topology)
    monkeypatch.setattr(controller, "_ambient_environment", lambda: fixture.environment)
    monkeypatch.setattr(
        controller,
        "_bind_runtime_after",
        forbidden("runtime binding after invalid Git topology"),
    )

    with pytest.raises(ValueError):
        controller.validate_runtime_preflight(fixture.manifest)


def test_replacement_environment_rejects_ambient_leak_before_runtime_binding(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture)
    monkeypatch.setattr(
        controller,
        "_ambient_environment",
        lambda: {**fixture.environment, "AMBIENT_SECRET": "must-not-leak"},
    )
    monkeypatch.setattr(
        controller,
        "_bind_runtime_after",
        forbidden("runtime binding after ambient environment leak"),
    )

    with pytest.raises(ValueError, match="environment"):
        controller.validate_runtime_preflight(fixture.manifest)


@pytest.mark.parametrize(
    "binding_name",
    [
        "controller",
        "source_files",
        "python_tree",
        "python",
        "python_realpath",
        "python_version",
        "git",
        "python_packages",
        "python_import_roots",
        "pretrained_checkpoint",
        "partition",
        "partition_lines",
        "image_root_link",
        "image_tree",
    ],
)
def test_run_preflight_rejects_changed_runtime_source_pretrained_or_data_binding(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
    binding_name: str,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture)
    drifted = copy.deepcopy(fixture.runtime)
    value = drifted[binding_name]
    if type(value) is str:
        drifted[binding_name] = f"{value}-drift"
    elif type(value) is int:
        drifted[binding_name] = value + 1
    elif type(value) is list:
        drifted[binding_name] = [*value, {"status": "nonexistent", "entry": "/drift"}]
    else:
        value["sha256" if "sha256" in value else "root_sha256"] = "0" * 64
    monkeypatch.setattr(controller, "_ambient_environment", lambda: fixture.environment)
    monkeypatch.setattr(controller, "_bind_runtime_after", lambda _authority: drifted)

    with pytest.raises(ValueError, match="runtime binding"):
        controller.validate_runtime_preflight(fixture.manifest)


@pytest.mark.parametrize(
    ("payload_mutation", "match"),
    [
        (("query_count", 0), "query/gallery"),
        (("recipe_id", "drifted-recipe"), "recipe"),
    ],
)
def test_run_preflight_rejects_false_scope_or_recipe_authority_drift(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
    payload_mutation: tuple[str, object],
    match: str,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(
        tmp_path,
        capture,
        payload_mutation=payload_mutation,
    )
    monkeypatch.setattr(controller, "_ambient_environment", lambda: fixture.environment)
    monkeypatch.setattr(
        controller,
        "_bind_runtime_after",
        forbidden("runtime binding after invalid scientific scope"),
    )

    with pytest.raises(ValueError, match=match):
        controller.validate_runtime_preflight(fixture.manifest)


@pytest.mark.parametrize("collision", ["run_directory", "report", "report_temp"])
def test_run_preflight_rejects_existing_run_directory_output_or_temporary(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    capture, _authority, _checkpoint = sidecar_inputs
    fixture = _make_authorized_git_fixture(tmp_path, capture)
    run_directory = fixture.repo / controller.RUN_DIRECTORY
    run_directory.mkdir(parents=True)
    if collision == "report":
        (run_directory / "report.json").write_bytes(b"collision")
    elif collision == "report_temp":
        (run_directory / "report.json.tmp").write_bytes(b"collision")
    monkeypatch.setattr(controller, "_ambient_environment", lambda: fixture.environment)
    monkeypatch.setattr(controller, "_bind_runtime_after", lambda _authority: fixture.runtime)

    with pytest.raises(ValueError, match="run directory|output"):
        controller.validate_runtime_preflight(fixture.manifest)


def test_replacement_environment_launches_exact_argv_once_without_ambient_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "run-v2"
    run_directory.mkdir(mode=0o700)
    child = tmp_path / "bound-python"
    ledger = tmp_path / "child-ledger.json"
    child.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['PASS201_TEST_LEDGER'])\n"
        "path.write_text(json.dumps({'argv': sys.argv, 'environment': dict(os.environ)}, "
        "sort_keys=True), encoding='utf-8')\n"
        "print('ordinary child complete')\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    environment = {
        **_freeze_test_environment(tmp_path),
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}/src",
        "TORCH_HOME": str(ledger),
    }
    child.write_text(
        child.read_text(encoding="utf-8").replace(
            "os.environ['PASS201_TEST_LEDGER']", "os.environ['TORCH_HOME']"
        ),
        encoding="utf-8",
    )
    argv = [str(child), "-m", "sfora.cli", "image-end-to-end", "--seed", "0"]
    authority = SimpleNamespace(
        checkout_root=tmp_path,
        payload={
            "execution": {"python": {"path": str(child)}, "argv": argv, "environment": environment},
            "outputs": {"log": {"path": "run-v2/training.log"}},
        },
    )
    authorized = SimpleNamespace(authority=authority)
    monkeypatch.setenv("AMBIENT_SECRET", "must-not-leak")

    running = controller.launch_once(authorized, run_directory)
    completed = controller._complete_child(running)

    assert completed.returncode == 0
    observed = json.loads(ledger.read_text(encoding="utf-8"))
    assert observed["argv"] == argv
    assert observed["environment"] == environment
    assert "AMBIENT_SECRET" not in observed["environment"]
    assert (run_directory / "training.log").read_text(encoding="utf-8") == (
        "ordinary child complete\n"
    )
    with pytest.raises(ValueError, match="log|exists"):
        controller.launch_once(authorized, run_directory)


def test_restricted_checkpoint_child_skips_python_startup_import_hooks(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    guard = tmp_path / "import-guard"
    guard.mkdir()
    torch_imported = tmp_path / "torch-imported"
    (guard / "torch.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(torch_imported)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (guard / "sitecustomize.py").write_text("import torch\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"binding-only checkpoint")
    expected = bind_external_file(checkpoint)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(guard)
    authority = SimpleNamespace(
        checkout_root=repo,
        payload={
            "execution": {
                "python": {"path": sys.executable},
                "environment": environment,
            }
        },
    )

    assert controller._run_binding_child(authority, checkpoint, expected) == expected
    assert not torch_imported.exists()


def _immutable_output_evidence(path: Path, data: bytes) -> OutputEvidence:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)
    current = os.stat(path, follow_symlinks=False)
    return OutputEvidence(
        PurePosixPath(path.as_posix()),
        "regular",
        current.st_mode,
        len(data),
        hashlib.sha256(data).hexdigest(),
    )


def _make_complete_receipt_inputs(
    tmp_path: Path,
    capture: CapturedAuthority,
    checkpoint_metadata: CheckpointMetadata,
) -> SimpleNamespace:
    checkout = tmp_path / "receipt-checkout"
    checkout.mkdir()
    runtime = _fake_freeze_runtime(checkout)
    args = controller.FreezeArgs(
        checkout_root=checkout,
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        python_path=Path("/venv/bin/python"),
        frozen_absence_checked_utc="2026-08-09T00:00:00Z",
        output_path=checkout / "docs" / "pass201_pa_source_v2_prelaunch.json",
    )
    payload = controller._build_prelaunch_payload(args, capture, runtime)
    authority = controller.validate_prelaunch(payload)
    manifest_bytes = canonical_json_bytes(payload)
    run_directory = checkout / controller.RUN_DIRECTORY
    report = _immutable_output_evidence(
        run_directory / "report.json",
        b'{"methods":{"opaque":NaN}}\n',
    )
    checkpoint = _immutable_output_evidence(
        run_directory / "checkpoint.pt",
        b"tiny-restricted-checkpoint",
    )
    log = _immutable_output_evidence(
        run_directory / "training.log",
        b"ordinary child complete\n",
    )
    config = _immutable_output_evidence(
        run_directory / "resolved_config.json",
        capture.config_bytes,
    )
    manifest = _immutable_output_evidence(
        run_directory / "train_manifest.json",
        b'{"rows":[]}\n',
    )
    run_directory.chmod(0o700)
    checkpoint_binding = bind_external_file(run_directory / "checkpoint.pt")
    metadata = BoundCheckpointMetadata(checkpoint_binding, checkpoint_metadata)
    first = SidecarFrame(
        101,
        capture.config_bytes,
        b'{"rows":[]}\n',
        config.sha256,
        manifest.sha256,
    )
    second = replace(first, pid=102)
    authorized = SimpleNamespace(
        authority=authority,
        authorization_commit="b" * 40,
        manifest_path=args.output_path,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_git_blob=hashlib.sha1(
            b"blob " + str(len(manifest_bytes)).encode("ascii") + b"\0" + manifest_bytes
        ).hexdigest(),
        runtime_bindings=runtime,
        preflight_started_utc="2026-08-09T00:00:01Z",
    )
    process = controller.CompletedChild(
        77,
        "2026-08-09T00:00:02Z",
        "2026-08-09T00:00:03Z",
        0,
    )
    postflight = SimpleNamespace(
        bindings=runtime,
        ended_utc="2026-08-09T00:00:04Z",
    )
    scientific = SimpleNamespace(report=report, checkpoint=checkpoint, log=log)
    return SimpleNamespace(
        checkout=checkout,
        run_directory=run_directory,
        authority=authority,
        authorized=authorized,
        process=process,
        postflight=postflight,
        scientific=scientific,
        metadata=metadata,
        frames=(first, second),
        config=config,
        manifest=manifest,
    )


def test_complete_receipt_builds_exact_valid_authority(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)

    receipt_bytes = controller._build_complete_receipt(
        inputs.authorized,
        inputs.process,
        inputs.postflight,
        inputs.scientific,
        inputs.metadata,
        inputs.frames,
        inputs.config,
        inputs.manifest,
    )

    payload = load_strict_json_bytes(receipt_bytes)
    complete = validate_complete_receipt(payload, inputs.authority)
    assert canonical_json_bytes(payload) == receipt_bytes
    assert complete.authorization_commit == "b" * 40
    assert payload["status"] == "complete"
    assert payload["candidate_values_computed"] is False
    assert "receipt" not in payload["outputs"]
    assert "metric" not in payload and "methods" not in payload


def test_receipt_publication_is_terminal_after_final_directory_validation(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    receipt_bytes = controller._build_complete_receipt(
        inputs.authorized,
        inputs.process,
        inputs.postflight,
        inputs.scientific,
        inputs.metadata,
        inputs.frames,
        inputs.config,
        inputs.manifest,
    )
    receipt = inputs.run_directory / "receipt.json"
    real_publish = controller.publish_new_file
    real_require_entries = controller._require_run_entries
    receipt_published = False

    def observed_publish(path: Path, data: bytes, *, mode: int = 0o444) -> OutputEvidence:
        nonlocal receipt_published
        evidence = real_publish(path, data, mode=mode)
        if path == receipt:
            receipt_published = True
        return evidence

    def reject_post_publication_validation(run_dir: Path, expected: set[str]) -> None:
        if receipt_published:
            raise AssertionError("fallible validation ran after receipt publication")
        real_require_entries(run_dir, expected)

    monkeypatch.setattr(controller, "publish_new_file", observed_publish)
    monkeypatch.setattr(controller, "_require_run_entries", reject_post_publication_validation)

    controller._publish_complete_receipt(
        inputs.authorized,
        inputs.run_directory,
        receipt_bytes,
    )

    validate_complete_receipt(load_strict_json_bytes(receipt.read_bytes()), inputs.authority)


def test_receipt_post_link_failure_rolls_back_and_attempt_stays_nonreusable(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    receipt_bytes = controller._build_complete_receipt(
        inputs.authorized,
        inputs.process,
        inputs.postflight,
        inputs.scientific,
        inputs.metadata,
        inputs.frames,
        inputs.config,
        inputs.manifest,
    )
    receipt = inputs.run_directory / "receipt.json"
    linked = threading.Event()
    tampered = threading.Event()
    real_link = contract.os.link
    raced = False

    def synchronized_link(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal raced
        real_link(source, destination, *args, **kwargs)
        if destination == receipt.name and not raced:
            raced = True
            linked.set()
            if not tampered.wait(timeout=5):
                raise AssertionError("receipt tamper did not complete")

    def tamper_linked_receipt() -> None:
        if not linked.wait(timeout=5):
            return
        receipt.write_bytes(b"tampered after link\n")
        tampered.set()

    monkeypatch.setattr(contract.os, "link", synchronized_link)
    racer = threading.Thread(target=tamper_linked_receipt)
    racer.start()
    try:
        with pytest.raises(ValueError, match="publish"):
            controller._publish_complete_receipt(
                inputs.authorized,
                inputs.run_directory,
                receipt_bytes,
            )
    finally:
        racer.join(timeout=5)

    assert raced
    assert not racer.is_alive()
    assert not receipt.exists()
    assert {path.name for path in inputs.run_directory.iterdir()} == {
        "report.json",
        "checkpoint.pt",
        "training.log",
        "resolved_config.json",
        "train_manifest.json",
    }
    with pytest.raises(ValueError, match="private run directory already exists"):
        controller._record_preflight_absence(inputs.authority)


def test_terminal_receipt_is_not_invalidated_by_directory_lock_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = SimpleNamespace(
        checkout_root=tmp_path,
        payload={"outputs": {"run_directory": "run-v2"}},
    )
    authorized = SimpleNamespace(authority=authority)
    receipt = tmp_path / "run-v2" / "receipt.json"
    real_close = controller.os.close
    close_failed = False

    def fail_closed_directory_descriptor(fd: int) -> None:
        nonlocal close_failed
        is_directory = stat.S_ISDIR(controller.os.fstat(fd).st_mode)
        real_close(fd)
        if is_directory and not close_failed:
            close_failed = True
            raise OSError("directory close failed after receipt")

    monkeypatch.setattr(controller.os, "close", fail_closed_directory_descriptor)

    with controller.create_and_lock_private_run_directory(authorized) as run_dir:
        receipt.write_bytes(b'{"status":"complete"}\n')
        assert run_dir == receipt.parent

    assert close_failed
    assert receipt.read_bytes() == b'{"status":"complete"}\n'


@pytest.mark.parametrize(
    "mutation",
    [
        "source_binding",
        "data_binding",
        "runtime_binding",
        "pretrained_binding",
        "report_hash",
        "config_hash",
        "checkpoint_hash",
        "output_type",
        "checkpoint_scalar",
        "child_sidecar",
        "membership",
        "algorithm",
        "success_flag",
        "candidate_flag",
        "scope_flag",
    ],
)
def test_complete_receipt_rejects_every_mutated_success_predicate(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    mutation: str,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    receipt = load_strict_json_bytes(
        controller._build_complete_receipt(
            inputs.authorized,
            inputs.process,
            inputs.postflight,
            inputs.scientific,
            inputs.metadata,
            inputs.frames,
            inputs.config,
            inputs.manifest,
        )
    )
    if mutation == "source_binding":
        receipt["controller"]["source_tree"]["root_sha256"] = "0" * 64
    elif mutation == "data_binding":
        receipt["postflight"]["partition"]["sha256"] = "0" * 64
    elif mutation == "runtime_binding":
        receipt["controller"]["python"]["sha256"] = "0" * 64
    elif mutation == "pretrained_binding":
        receipt["preflight"]["pretrained_checkpoint"]["sha256"] = "0" * 64
    elif mutation == "report_hash":
        receipt["outputs"]["report"]["sha256"] = "0" * 64
    elif mutation == "config_hash":
        receipt["outputs"]["resolved_config"]["sha256"] = "0" * 64
    elif mutation == "checkpoint_hash":
        receipt["outputs"]["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "output_type":
        receipt["outputs"]["log"]["file_type"] = "directory"
    elif mutation == "checkpoint_scalar":
        receipt["checkpoint_metadata"]["training_step"] += 1
    elif mutation == "child_sidecar":
        receipt["sidecar_derivation"]["child_processes"][1]["pid"] = 101
    elif mutation == "membership":
        receipt["sidecar_derivation"]["membership_covered_by_postflight"] = False
    elif mutation == "algorithm":
        receipt["sidecar_derivation"]["config_algorithm"] = "drift"
    elif mutation == "success_flag":
        receipt["status"] = "incomplete"
    elif mutation == "candidate_flag":
        receipt["candidate_values_computed"] = True
    elif mutation == "scope_flag":
        receipt["scope"]["pass201_candidate_paths_read"] = True
    else:  # pragma: no cover - closed mutation table
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        validate_complete_receipt(receipt, inputs.authority)


def test_postflight_rejects_runtime_drift_before_hashing_outputs(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    drifted = copy.deepcopy(inputs.authorized.runtime_bindings)
    drifted["partition"]["sha256"] = "0" * 64
    monkeypatch.setattr(controller, "_ambient_environment", lambda: drifted["environment"])
    monkeypatch.setattr(controller, "_bind_runtime_after", lambda _authority: drifted)
    monkeypatch.setattr(
        controller,
        "_freeze_scientific_outputs",
        forbidden("output hashing after postflight drift"),
    )

    with pytest.raises(ValueError, match="postflight|runtime binding"):
        controller.publish_postflight(inputs.authorized, inputs.process, inputs.run_directory)

    assert not (inputs.run_directory / "receipt.json").exists()


def test_postflight_publishes_immutable_complete_receipt_without_methods_access(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    (inputs.run_directory / "resolved_config.json").unlink()
    (inputs.run_directory / "train_manifest.json").unlink()
    frames = iter(inputs.frames)
    method_accesses = 0

    def forbidden_methods_access(_data: bytes) -> object:
        nonlocal method_accesses
        method_accesses += 1
        raise AssertionError("ordinary report methods were parsed")

    def metadata_child(_authorized: object, scientific: object) -> BoundCheckpointMetadata:
        return BoundCheckpointMetadata(
            bind_external_file(Path(scientific.checkpoint.path)),
            checkpoint_metadata,
        )

    monkeypatch.setattr(controller, "_require_pre_post_identity", lambda _a: inputs.postflight)
    monkeypatch.setattr(controller, "_read_restricted_metadata", metadata_child)
    monkeypatch.setattr(controller, "_run_sidecar_child", lambda *_args: next(frames))
    monkeypatch.setattr(controller, "load_strict_json_value_bytes", forbidden_methods_access)

    controller.publish_postflight(inputs.authorized, inputs.process, inputs.run_directory)

    receipt_path = inputs.run_directory / "receipt.json"
    receipt = load_strict_json_bytes(receipt_path.read_bytes())
    validate_complete_receipt(receipt, inputs.authority)
    assert method_accesses == 0
    assert sorted(path.name for path in inputs.run_directory.iterdir()) == sorted(
        controller.OUTPUT_FILENAMES.values()
    )
    assert stat.S_IMODE(inputs.run_directory.stat().st_mode) == 0o700
    for filename in controller.OUTPUT_FILENAMES.values():
        assert stat.S_IMODE((inputs.run_directory / filename).stat().st_mode) == 0o444


def test_postflight_sidecar_disagreement_leaves_receipt_absent(
    tmp_path: Path,
    sidecar_inputs: tuple[CapturedAuthority, PrelaunchAuthority, CheckpointMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _authority, checkpoint_metadata = sidecar_inputs
    inputs = _make_complete_receipt_inputs(tmp_path, capture, checkpoint_metadata)
    (inputs.run_directory / "resolved_config.json").unlink()
    (inputs.run_directory / "train_manifest.json").unlink()
    first, second = inputs.frames
    drifted = replace(
        second,
        config_bytes=b'{"drift":true}\n',
        config_sha256=hashlib.sha256(b'{"drift":true}\n').hexdigest(),
    )
    frames = iter((first, drifted))
    monkeypatch.setattr(controller, "_require_pre_post_identity", lambda _a: inputs.postflight)
    monkeypatch.setattr(
        controller,
        "_read_restricted_metadata",
        lambda *_args: inputs.metadata,
    )
    monkeypatch.setattr(controller, "_run_sidecar_child", lambda *_args: next(frames))

    with pytest.raises(ValueError, match="sidecar config"):
        controller.publish_postflight(inputs.authorized, inputs.process, inputs.run_directory)

    assert not (inputs.run_directory / "receipt.json").exists()
    assert not (inputs.run_directory / "resolved_config.json").exists()
    assert not (inputs.run_directory / "train_manifest.json").exists()


def test_public_cli_freeze_authority_publishes_only_selected_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    output = _prelaunch_freeze_output(checkout, 1)
    monkeypatch.chdir(checkout)
    expected = b'{"authority":true}\n'
    observed: list[object] = []

    def freeze(args: object) -> bytes:
        observed.append(args)
        return expected

    monkeypatch.setattr(controller, "freeze_authority", freeze)

    assert (
        main(
            [
                "freeze-authority",
                "--frozen-absence-checked-utc",
                "2026-08-09T00:00:00Z",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert len(observed) == 1
    args = observed[0]
    assert args.checkout_root == checkout
    assert args.dataset_root == controller.DATASET_ROOT
    assert args.python_path == Path(sys.executable).absolute()
    assert args.frozen_absence_checked_utc == "2026-08-09T00:00:00Z"
    assert args.output_path == output
    assert output.read_bytes() == expected
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not (checkout / controller.PRELAUNCH_PATH).exists()


@pytest.mark.parametrize("spelling", ["relative", "dot_alias"])
def test_public_cli_freeze_authority_rejects_non_normalized_output_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    output = _prelaunch_freeze_output(checkout, 1)
    monkeypatch.chdir(checkout)
    if spelling == "relative":
        output_argument = f"../{output.name}"
    else:
        output_argument = f"{output.parent}/./{output.name}"
    monkeypatch.setattr(controller, "freeze_authority", forbidden("freeze capture"))

    with pytest.raises(ValueError, match="normalized absolute prelaunch output path"):
        main(
            [
                "freeze-authority",
                "--frozen-absence-checked-utc",
                "2026-08-09T00:00:00Z",
                "--output",
                output_argument,
            ]
        )

    assert not output.exists()


def test_public_cli_run_dispatches_exact_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    observed: list[Path] = []
    monkeypatch.setattr(controller, "run_authorized_source", observed.append)

    assert main(["run", "--manifest", "docs/pass201_pa_source_v2_prelaunch.json"]) == 0

    assert observed == [tmp_path / "docs" / "pass201_pa_source_v2_prelaunch.json"]


class ControllerOrderFixture:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.events: list[str] = []
        self.publish_order: list[str] = []
        self.failure_event: str | None = None
        self.run_directory = tmp_path / "run-v2"
        self.receipt = self.run_directory / "receipt.json"
        self.manifest = tmp_path / "prelaunch.json"
        self.manifest.write_bytes(b"{}\n")
        child = tmp_path / "ordinary_child.py"
        child.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.child = child
        self.authority = SimpleNamespace(
            checkout_root=tmp_path,
            payload={
                "outputs": {
                    "run_directory": "run-v2",
                    "report": {"path": "run-v2/report.json"},
                    "checkpoint": {"path": "run-v2/checkpoint.pt"},
                    "log": {"path": "run-v2/training.log"},
                    "resolved_config": {"path": "run-v2/resolved_config.json"},
                    "train_manifest": {"path": "run-v2/train_manifest.json"},
                    "receipt": {"path": "run-v2/receipt.json"},
                }
            },
        )

        def event(name: str) -> None:
            self.events.append(name)
            if self.failure_event == name:
                raise RuntimeError(f"injected failure at {name}")

        def load_manifest(_path: Path):
            event("strict_manifest")
            return self.authority, self.manifest, b"{}\n"

        def topology(_root: Path, _authority: object) -> str:
            event("detached_exact_git_topology")
            return "a" * 40

        def runtime_bindings(_authority: object) -> object:
            event("replacement_runtime_bindings")
            return {"bound": True}

        def require_bindings(_authority: object, bindings: object) -> None:
            assert bindings == {"bound": True}

        def absence(_authority: object) -> None:
            event("frozen_preflight_absence")

        @contextmanager
        def private_directory(_authorized: object):
            event("private_run_directory_lock")
            self.run_directory.mkdir(mode=0o700)
            yield self.run_directory

        def launch(_authorized: object, _run_dir: Path):
            event("one_training_child")
            process = subprocess.Popen(
                [sys.executable, str(self.child)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return SimpleNamespace(
                process=process,
                pid=process.pid,
                started_utc="2026-08-09T00:00:01Z",
            )

        def complete(running: object):
            stdout, stderr = running.process.communicate(timeout=5)
            assert stdout == stderr == b""
            assert running.process.returncode == 0
            event("training_exit_zero")
            return SimpleNamespace(
                pid=running.pid,
                started_utc=running.started_utc,
                ended_utc="2026-08-09T00:00:02Z",
                returncode=0,
            )

        def post_identity(_authorized: object) -> object:
            event("postflight_equality")
            return {"bound": True}

        def freeze_outputs(_authorized: object, _run_dir: Path) -> object:
            event("freeze_scientific_outputs")
            return SimpleNamespace()

        def restricted_metadata(_authorized: object, _scientific: object) -> object:
            event("restricted_checkpoint_metadata")
            return SimpleNamespace()

        sidecar_calls = 0

        def sidecar_child(
            _authorized: object,
            _scientific: object,
            _run_dir: Path,
        ) -> SidecarFrame:
            nonlocal sidecar_calls
            sidecar_calls += 1
            event(f"sidecar_child_{sidecar_calls}")
            config = b"{}\n"
            manifest = b'{"rows":[]}\n'
            return SidecarFrame(
                100 + sidecar_calls,
                config,
                manifest,
                hashlib.sha256(config).hexdigest(),
                hashlib.sha256(manifest).hexdigest(),
            )

        def publish_sidecar(
            _authorized: object,
            _run_dir: Path,
            output_name: str,
            data: bytes,
        ) -> object:
            event(f"publish_{output_name}")
            filename = {
                "resolved_config": "resolved_config.json",
                "train_manifest": "train_manifest.json",
            }[output_name]
            self.publish_order.append(filename)
            path = self.run_directory / filename
            path.write_bytes(data)
            return SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())

        def build_receipt(*_args: object) -> bytes:
            return b'{"status":"complete"}\n'

        def publish_receipt(_authorized: object, _run_dir: Path, data: bytes) -> object:
            event("publish_receipt")
            self.publish_order.append("receipt.json")
            self.receipt.write_bytes(data)
            return SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())

        monkeypatch.setattr(controller, "_load_manifest_authority", load_manifest, raising=False)
        monkeypatch.setattr(controller, "validate_authorization_topology", topology, raising=False)
        monkeypatch.setattr(
            controller, "_require_authority_scope", lambda _authority: None, raising=False
        )
        monkeypatch.setattr(
            controller,
            "_require_replacement_environment",
            lambda _authority: None,
            raising=False,
        )
        monkeypatch.setattr(controller, "_bind_runtime_after", runtime_bindings, raising=False)
        monkeypatch.setattr(
            controller,
            "_require_runtime_matches_authority",
            require_bindings,
            raising=False,
        )
        monkeypatch.setattr(controller, "_record_preflight_absence", absence, raising=False)
        monkeypatch.setattr(
            controller,
            "create_and_lock_private_run_directory",
            private_directory,
            raising=False,
        )
        monkeypatch.setattr(controller, "launch_once", launch, raising=False)
        monkeypatch.setattr(controller, "_complete_child", complete, raising=False)
        monkeypatch.setattr(controller, "_require_pre_post_identity", post_identity, raising=False)
        monkeypatch.setattr(controller, "_freeze_scientific_outputs", freeze_outputs, raising=False)
        monkeypatch.setattr(
            controller, "_read_restricted_metadata", restricted_metadata, raising=False
        )
        monkeypatch.setattr(controller, "_run_sidecar_child", sidecar_child, raising=False)
        monkeypatch.setattr(controller, "_publish_sidecar_output", publish_sidecar, raising=False)
        monkeypatch.setattr(controller, "_build_complete_receipt", build_receipt, raising=False)
        monkeypatch.setattr(controller, "_publish_complete_receipt", publish_receipt, raising=False)

    def fail_at(self, event: str) -> None:
        self.failure_event = event

    def run(self) -> None:
        controller.run_authorized_source(self.manifest)


@pytest.fixture
def controller_order_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ControllerOrderFixture:
    return ControllerOrderFixture(tmp_path, monkeypatch)


def test_controller_order_publishes_receipt_last(
    controller_order_fixture: ControllerOrderFixture,
) -> None:
    controller_order_fixture.run()

    assert controller_order_fixture.events == EXPECTED_CONTROLLER_ORDER
    assert controller_order_fixture.publish_order == [
        "resolved_config.json",
        "train_manifest.json",
        "receipt.json",
    ]


@pytest.mark.parametrize("failure_event", EXPECTED_CONTROLLER_ORDER[:-1])
def test_controller_order_failure_never_publishes_receipt(
    controller_order_fixture: ControllerOrderFixture,
    failure_event: str,
) -> None:
    controller_order_fixture.fail_at(failure_event)

    with pytest.raises(RuntimeError, match="injected failure"):
        controller_order_fixture.run()

    failure_index = EXPECTED_CONTROLLER_ORDER.index(failure_event)
    assert controller_order_fixture.events == EXPECTED_CONTROLLER_ORDER[: failure_index + 1]
    assert not controller_order_fixture.receipt.exists()
