from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_pass201_pa_source_v2 as controller  # noqa: E402
from pass201_pa_source_v2_contract import (  # noqa: E402
    CheckpointArch,
    CheckpointMetadata,
    PrelaunchAuthority,
    canonical_json_bytes,
    load_strict_json_bytes,
    load_strict_json_value_bytes,
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
        "objectives": ["proxy_anchor"],
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


def test_capture_child_never_opens_produced_checkpoint(
    tiny_inshop: Path, tmp_path: Path
) -> None:
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
        '{"name":"image-end-to-end","dataset_name":"inshop",'
        f'"protocol":"query_gallery","config":{config_text},'
        '"train_examples":2,"test_examples":2,'
        '"methods":{"score":NaN,"score":2,"payload":"}],\\\"config\\\":false"}}\n'
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
        b'{"payload":"}],\\\"config\\\":false","nested":{"x":Infinity}}',
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
            b'{"name":"image-end-to-end","dataset_name":"inshop",',
            b'"protocol":"query_gallery","config":',
            config,
            b',"train_examples":2,"test_examples":2,"methods":',
            methods_raw,
            b"}\n",
        )
    )
    assert derive_resolved_config(report, checkpoint, authority) == authority.expected_config_bytes


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
            "name": "image-end-to-end",
            "dataset_name": "inshop",
            "protocol": "query_gallery",
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
            "name": "image-end-to-end",
            "dataset_name": "inshop",
            "protocol": "query_gallery",
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
        controller.derive_sidecars_from_files(
            manifest, alternative_report, checkpoint, run_dir
        )

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
    program = r'''
import hashlib, json, os, struct, sys
from pathlib import Path

import run_pass201_pa_source_v2 as c
from pass201_pa_source_v2_contract import CheckpointArch, CheckpointMetadata, PrelaunchAuthority

root = Path(sys.argv[1])
config = {
    "batch_size": 180,
    "drop_last_train_batch": True,
    "objectives": ["proxy_anchor"],
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
c._run_metadata_child = lambda *_args: checkpoint
raise SystemExit(c.main([
    "derive-sidecars", "--manifest", str(root / "authority.json"),
    "--report", str(root / "run" / "report.json"),
    "--checkpoint", str(root / "run" / "checkpoint.pt"),
    "--output-dir", str(root / "run"),
]))
'''
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


def test_two_fresh_private_sidecar_children_are_identical_and_fail_on_input_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "authority.json").write_bytes(b'{"bound":true}\n')
    config = {
        "batch_size": 180,
        "drop_last_train_batch": True,
        "objectives": ["proxy_anchor"],
        "recipe_digest": controller.RECIPE_DIGEST,
        "recipe_id": controller.RECIPE_ID,
        "seed": 0,
    }
    (run_dir / "report.json").write_bytes(
        canonical_json_bytes(
            {
                "name": "image-end-to-end",
                "dataset_name": "inshop",
                "protocol": "query_gallery",
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
