from __future__ import annotations

import errno
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_unicom_ema_imprint_replication.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_unicom_ema_imprint_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol(seed: int, classifier_init: str) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "1" * 64,
        "unicom_revision": "2" * 40,
        "initial_checkpoint_sha256": "3" * 64,
        "partition_sha256": "4" * 64,
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
        "classifier_init": classifier_init,
        "ema_decay": 0.999,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }


def _rows(seed: int) -> list[dict[str, object]]:
    rows = []
    for arm_index, cell in enumerate(("random_raw", "imprinted_raw")):
        for epoch_index, epoch in enumerate((4, 8, 12, 16)):
            map_at_r = 0.80 + 0.02 * epoch_index + 0.01 * arm_index
            top1 = [True] * (970 + 2 * arm_index) + [False] * (30 - 2 * arm_index)
            rows.append(
                {
                    "cell": cell,
                    "epoch": epoch,
                    "checkpoint_sha256": f"{seed * 8 + arm_index * 4 + epoch_index:064x}",
                    "training_history_sha256": (
                        f"{100 + seed * 8 + arm_index * 4 + epoch_index:064x}"
                    ),
                    "metrics": {"map_at_r": map_at_r, "recall_at_1": sum(top1) / len(top1)},
                    "query_evidence": {
                        "top1_correct": top1,
                        "average_precision": [map_at_r] * len(top1),
                    },
                }
            )
    return rows


def _build(module, seed: int = 1) -> dict[str, object]:
    return module.build_replication_pair(
        seed=seed,
        rows=_rows(seed),
        random_protocol=_protocol(seed, "random"),
        imprinted_protocol=_protocol(seed, "imprinted"),
        random_training_seconds=15000.0,
        imprinted_training_seconds=14000.0,
        random_peak_gpu_mib=87000,
        imprinted_peak_gpu_mib=86900,
        random_checkpoint_storage_bytes=1000,
        imprinted_checkpoint_storage_bytes=1001,
        deployment_storage_bytes=500,
        inference_latency_ms_per_image=11.8,
        random_measurement_receipt_sha256="5" * 64,
        imprinted_measurement_receipt_sha256="6" * 64,
        random_profile=(1.0, 0.05),
        imprinted_profile=(1.0, 0.04),
    )


def _initialization_receipt(seed: int, classifier_init: str) -> dict[str, object]:
    return {
        "schema_version": "unicom-classifier-initialization-v1",
        "seed": seed,
        "classifier_init": classifier_init,
        "trainer_sha256": "1" * 64,
        "algorithm": {
            "random": "torch-normal-std-0.01-rng-balanced",
            "imprinted": "normalized-class-means-norm-matched-rng-restored",
        }[classifier_init],
        "classifier_tensor_sha256": ("7" if classifier_init == "random" else "8") * 64,
        "classifier_shape": [3200, 768],
        "classifier_dtype": "torch.float32",
        "optimizer_steps_per_epoch": 161,
        "initialization_seconds": 1.25 if classifier_init == "random" else 2.5,
        "post_initialization_rng": {
            "python_sha256": "9" * 64,
            "numpy_sha256": "a" * 64,
            "torch_cpu_sha256": "b" * 64,
            "torch_cuda_sha256_by_device": ["c" * 64],
        },
    }


def _future_build(
    module,
    *,
    seed: int = 2,
    random_initialization: dict[str, object] | None = None,
    imprinted_initialization: dict[str, object] | None = None,
) -> dict[str, object]:
    random_initialization = random_initialization or _initialization_receipt(seed, "random")
    imprinted_initialization = imprinted_initialization or _initialization_receipt(
        seed, "imprinted"
    )
    return module.build_replication_pair(
        seed=seed,
        rows=_rows(seed),
        random_protocol=_protocol(seed, "random"),
        imprinted_protocol=_protocol(seed, "imprinted"),
        random_training_seconds=15000.0,
        imprinted_training_seconds=14000.0,
        random_peak_gpu_mib=87000,
        imprinted_peak_gpu_mib=86900,
        random_checkpoint_storage_bytes=1000,
        imprinted_checkpoint_storage_bytes=1001,
        deployment_storage_bytes=500,
        inference_latency_ms_per_image=11.8,
        random_measurement_receipt_sha256="5" * 64,
        imprinted_measurement_receipt_sha256="6" * 64,
        random_profile=(1.0, 0.05),
        imprinted_profile=(1.0, 0.04),
        random_initialization_receipt=random_initialization,
        imprinted_initialization_receipt=imprinted_initialization,
        random_initialization_receipt_sha256="d" * 64,
        imprinted_initialization_receipt_sha256="e" * 64,
    )


def test_future_pair_v2_binds_initialization_evidence_and_rng() -> None:
    module = _load_script()

    report = _future_build(module)

    module.validate_replication_pair(report)
    assert report["schema_version"] == "unicom-ema-imprint-replication-pair-v2"
    assert tuple(report["random_raw"])[-4:] == (
        "optimizer_steps_per_epoch",
        "initialization_seconds",
        "initialization_receipt_sha256",
        "post_initialization_rng_sha256",
    )
    assert report["random_raw"]["optimizer_steps_per_epoch"] == 161
    assert report["random_raw"]["initialization_seconds"] == 1.25
    assert report["random_raw"]["initialization_receipt_sha256"] == "d" * 64
    assert report["random_raw"]["post_initialization_rng_sha256"] == module._json_sha256(
        _initialization_receipt(2, "random")["post_initialization_rng"]
    )


def test_future_pair_rejects_cross_arm_rng_or_historical_schema_routing() -> None:
    module = _load_script()
    changed = _initialization_receipt(2, "imprinted")
    changed["post_initialization_rng"]["torch_cpu_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="post-initialization RNG"):
        _future_build(module, imprinted_initialization=changed)
    changed = _initialization_receipt(2, "imprinted")
    changed["optimizer_steps_per_epoch"] = 162
    with pytest.raises(ValueError, match="optimizer steps differ"):
        _future_build(module, imprinted_initialization=changed)
    with pytest.raises(ValueError, match="initialization"):
        module.build_replication_pair(
            seed=1,
            rows=_rows(1),
            random_protocol=_protocol(1, "random"),
            imprinted_protocol=_protocol(1, "imprinted"),
            random_training_seconds=1.0,
            imprinted_training_seconds=1.0,
            random_peak_gpu_mib=1,
            imprinted_peak_gpu_mib=1,
            random_checkpoint_storage_bytes=1,
            imprinted_checkpoint_storage_bytes=1,
            deployment_storage_bytes=1,
            inference_latency_ms_per_image=1.0,
            random_measurement_receipt_sha256="5" * 64,
            imprinted_measurement_receipt_sha256="6" * 64,
            random_profile=(1.0, 0.0),
            imprinted_profile=(1.0, 0.0),
            random_initialization_receipt=_initialization_receipt(1, "random"),
            imprinted_initialization_receipt=_initialization_receipt(1, "imprinted"),
            random_initialization_receipt_sha256="d" * 64,
            imprinted_initialization_receipt_sha256="e" * 64,
        )


def test_build_pair_report_binds_protocol_rows_evidence_and_costs() -> None:
    module = _load_script()

    report = _build(module)

    module.validate_replication_pair(report)
    assert report["seed"] == 1
    endpoint = report["random_raw"]["epoch_metrics"][-1]
    assert endpoint["epoch"] == 16
    assert endpoint["map_at_r"] == pytest.approx(0.86)
    assert endpoint["recall_at_1"] == 0.97
    assert report["imprinted_raw"]["checkpoint_sha256_by_epoch"][-1] == f"{15:064x}"
    assert report["evidence"]["random_raw"][-1]["average_precision"] == [
        0.8600000000000001
    ] * 1000


def test_validator_recomputes_metrics_and_rejects_protocol_or_evidence_drift() -> None:
    module = _load_script()
    kwargs = {
        "seed": 1,
        "rows": _rows(1),
        "random_protocol": _protocol(1, "random"),
        "imprinted_protocol": _protocol(1, "imprinted"),
        "random_training_seconds": 1.0,
        "imprinted_training_seconds": 1.0,
        "random_peak_gpu_mib": 1,
        "imprinted_peak_gpu_mib": 1,
        "random_checkpoint_storage_bytes": 1,
        "imprinted_checkpoint_storage_bytes": 1,
        "deployment_storage_bytes": 1,
        "inference_latency_ms_per_image": 1.0,
        "random_profile": (1.0, 0.0),
        "imprinted_profile": (1.0, 0.0),
        "random_measurement_receipt_sha256": "5" * 64,
        "imprinted_measurement_receipt_sha256": "6" * 64,
    }
    report = module.build_replication_pair(**kwargs)

    report["evidence"]["random_raw"][0]["average_precision"][0] += 0.1
    with pytest.raises(ValueError, match="metric|evidence"):
        module.validate_replication_pair(report)

    report = module.build_replication_pair(**kwargs)
    report["imprinted_training_protocol"]["seed"] = 3
    with pytest.raises(ValueError, match="protocol|seed"):
        module.validate_replication_pair(report)


def test_builder_rejects_wrong_row_order_or_reused_checkpoint() -> None:
    module = _load_script()
    rows = _rows(3)
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="row order"):
        module.build_replication_pair(
            seed=3,
            rows=rows,
            random_protocol=_protocol(3, "random"),
            imprinted_protocol=_protocol(3, "imprinted"),
            random_training_seconds=1.0,
            imprinted_training_seconds=1.0,
            random_peak_gpu_mib=1,
            imprinted_peak_gpu_mib=1,
            random_checkpoint_storage_bytes=1,
            imprinted_checkpoint_storage_bytes=1,
            deployment_storage_bytes=1,
            inference_latency_ms_per_image=1.0,
            random_measurement_receipt_sha256="5" * 64,
            imprinted_measurement_receipt_sha256="6" * 64,
            random_profile=(1.0, 0.0),
            imprinted_profile=(1.0, 0.0),
        )


def test_atomic_report_publication_strict_reloads_and_never_clobbers(tmp_path: Path) -> None:
    module = _load_script()
    report = _build(module)
    output = tmp_path / "pair.json"

    module.write_report_atomic(report, output)

    module.validate_replication_pair(module.strict_json_object(output.read_bytes()))
    assert list(tmp_path.iterdir()) == [output]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_report_atomic(report, output)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_validator_requires_common_queries_and_distinct_histories() -> None:
    module = _load_script()
    report = _build(module)
    report["evidence"]["imprinted_raw"][0]["top1_correct"].pop()
    report["evidence"]["imprinted_raw"][0]["average_precision"].pop()
    metric = module._metric_from_evidence(report["evidence"]["imprinted_raw"][0])
    report["imprinted_raw"]["epoch_metrics"][0] = {"epoch": 4, **metric}
    with pytest.raises(ValueError, match="query"):
        module.validate_replication_pair(report)

    report = _build(module)
    duplicate = report["random_raw"]["training_history_sha256_by_epoch"][0]
    report["imprinted_raw"]["training_history_sha256_by_epoch"][3] = duplicate
    with pytest.raises(ValueError, match="history"):
        module.validate_replication_pair(report)


def test_runtime_binding_requires_exact_checkpoint_claims() -> None:
    module = _load_script()
    protocols = {
        "random_raw": _protocol(1, "random"),
        "imprinted_raw": _protocol(1, "imprinted"),
    }
    module.validate_runtime_bindings(
        protocols,
        trainer_sha256="1" * 64,
        unicom_revision="2" * 40,
        initial_checkpoint_sha256="3" * 64,
        partition_sha256="4" * 64,
    )
    protocols["random_raw"]["partition_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="runtime|partition"):
        module.validate_runtime_bindings(
            protocols,
            trainer_sha256="1" * 64,
            unicom_revision="2" * 40,
            initial_checkpoint_sha256="3" * 64,
            partition_sha256="4" * 64,
        )


def test_measurement_receipt_binds_log_protocol_and_run_evidence(tmp_path: Path) -> None:
    module = _load_script()
    protocol = _protocol(1, "random")
    checkpoints = [f"{index:064x}" for index in range(1, 5)]
    histories = [f"{index:064x}" for index in range(11, 15)]
    log = tmp_path / "train.log"
    log.write_text("completed seed 1 random\n", encoding="utf-8")
    receipt = {
        "schema_version": "unicom-training-measurement-v1",
        "seed": 1,
        "classifier_init": "random",
        "training_seconds": 10.0,
        "peak_gpu_mib": 100,
        "step_wall_seconds": 1.0,
        "fusible_non_backbone_seconds": 0.05,
        "training_log_sha256": module._sha256_file(log),
        "training_protocol_sha256": module._json_sha256(protocol),
        "checkpoint_sha256_by_epoch": checkpoints,
        "training_history_sha256_by_epoch": histories,
    }
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    loaded, digest = module.load_measurement_receipt(
        path,
        log_path=log,
        seed=1,
        classifier_init="random",
        training_protocol=protocol,
        checkpoint_sha256_by_epoch=checkpoints,
        training_history_sha256_by_epoch=histories,
    )

    assert loaded == receipt
    assert digest == module._sha256_file(path)
    expected_checkpoints = list(checkpoints)
    receipt["checkpoint_sha256_by_epoch"][0] = "f" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement"):
        module.load_measurement_receipt(
            path,
            log_path=log,
            seed=1,
            classifier_init="random",
            training_protocol=protocol,
            checkpoint_sha256_by_epoch=expected_checkpoints,
            training_history_sha256_by_epoch=histories,
        )


def test_measurement_v2_authenticates_initialization_receipt_and_transitive_fields(
    tmp_path: Path,
) -> None:
    module = _load_script()
    seed = 2
    protocol = _protocol(seed, "random")
    checkpoints = [f"{index:064x}" for index in range(1, 5)]
    histories = [f"{index:064x}" for index in range(11, 15)]
    log = tmp_path / "train.log"
    log.write_text("completed seed 2 random\n", encoding="utf-8")
    initialization = _initialization_receipt(seed, "random")
    initialization_path = tmp_path / "initialization-receipt.json"
    initialization_path.write_text(json.dumps(initialization), encoding="utf-8")
    receipt = {
        "schema_version": "unicom-training-measurement-v2",
        "seed": seed,
        "classifier_init": "random",
        "training_seconds": 10.0,
        "peak_gpu_mib": 100,
        "step_wall_seconds": 1.0,
        "fusible_non_backbone_seconds": 0.05,
        "training_log_sha256": module._sha256_file(log),
        "training_protocol_sha256": module._json_sha256(protocol),
        "checkpoint_sha256_by_epoch": checkpoints,
        "training_history_sha256_by_epoch": histories,
        "optimizer_steps_per_epoch": 161,
        "initialization_seconds": 1.25,
        "initialization_receipt_sha256": module._sha256_file(initialization_path),
        "post_initialization_rng_sha256": module._json_sha256(
            initialization["post_initialization_rng"]
        ),
    }
    measurement_path = tmp_path / "measurement.json"
    measurement_path.write_text(json.dumps(receipt), encoding="utf-8")

    loaded, digest, loaded_initialization, initialization_digest = (
        module.load_measurement_receipt(
            measurement_path,
            log_path=log,
            seed=seed,
            classifier_init="random",
            training_protocol=protocol,
            checkpoint_sha256_by_epoch=checkpoints,
            training_history_sha256_by_epoch=histories,
            initialization_receipt_path=initialization_path,
        )
    )

    assert loaded == receipt
    assert digest == module._sha256_file(measurement_path)
    assert loaded_initialization == initialization
    assert initialization_digest == module._sha256_file(initialization_path)

    receipt["initialization_receipt_sha256"] = "f" * 64
    measurement_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization"):
        module.load_measurement_receipt(
            measurement_path,
            log_path=log,
            seed=seed,
            classifier_init="random",
            training_protocol=protocol,
            checkpoint_sha256_by_epoch=checkpoints,
            training_history_sha256_by_epoch=histories,
            initialization_receipt_path=initialization_path,
        )


def test_measurement_schema_routes_v1_only_to_seed1_and_v2_only_to_future(
    tmp_path: Path,
) -> None:
    module = _load_script()
    protocol = _protocol(2, "random")
    checkpoints = [f"{index:064x}" for index in range(1, 5)]
    histories = [f"{index:064x}" for index in range(11, 15)]
    log = tmp_path / "train.log"
    log.write_text("completed\n", encoding="utf-8")
    v1 = {
        "schema_version": "unicom-training-measurement-v1",
        "seed": 2,
        "classifier_init": "random",
        "training_seconds": 10.0,
        "peak_gpu_mib": 100,
        "step_wall_seconds": 1.0,
        "fusible_non_backbone_seconds": 0.05,
        "training_log_sha256": module._sha256_file(log),
        "training_protocol_sha256": module._json_sha256(protocol),
        "checkpoint_sha256_by_epoch": checkpoints,
        "training_history_sha256_by_epoch": histories,
    }
    measurement = tmp_path / "measurement.json"
    measurement.write_text(json.dumps(v1), encoding="utf-8")

    with pytest.raises(ValueError, match="measurement.*schema|schema.*measurement"):
        module.load_measurement_receipt(
            measurement,
            log_path=log,
            seed=2,
            classifier_init="random",
            training_protocol=protocol,
            checkpoint_sha256_by_epoch=checkpoints,
            training_history_sha256_by_epoch=histories,
            initialization_receipt_path=None,
        )


@pytest.mark.parametrize(
    ("seed", "random_initialization", "imprinted_initialization"),
    (
        (2, None, Path("imprinted.json")),
        (2, Path("random.json"), None),
        (1, Path("random.json"), Path("imprinted.json")),
    ),
)
def test_run_rejects_initialization_argument_routing_before_gpu_evaluation(
    seed: int,
    random_initialization: Path | None,
    imprinted_initialization: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def forbidden_load(_args):
        raise AssertionError("GPU/checkpoint evaluation ran before argument preflight")

    monkeypatch.setattr(module, "load_replication_rows", forbidden_load)
    args = SimpleNamespace(
        seed=seed,
        random_initialization_receipt=random_initialization,
        imprinted_initialization_receipt=imprinted_initialization,
    )

    with pytest.raises(ValueError, match="initialization receipt arguments"):
        module.run(args)


def test_measurement_receipt_hashes_the_exact_bytes_it_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    protocol = _protocol(1, "random")
    checkpoints = [f"{index:064x}" for index in range(1, 5)]
    histories = [f"{index:064x}" for index in range(11, 15)]
    log = tmp_path / "train.log"
    log.write_text("completed seed 1 random\n", encoding="utf-8")
    receipt = {
        "schema_version": "unicom-training-measurement-v1",
        "seed": 1,
        "classifier_init": "random",
        "training_seconds": 10.0,
        "peak_gpu_mib": 100,
        "step_wall_seconds": 1.0,
        "fusible_non_backbone_seconds": 0.05,
        "training_log_sha256": module._sha256_file(log),
        "training_protocol_sha256": module._json_sha256(protocol),
        "checkpoint_sha256_by_epoch": checkpoints,
        "training_history_sha256_by_epoch": histories,
    }
    path = tmp_path / "measurement.json"
    original = json.dumps(receipt).encode()
    replacement = original + b" "
    path.write_bytes(original)
    real_strict = module.strict_json_object

    def replace_after_parse(payload: bytes):
        parsed = real_strict(payload)
        path.write_bytes(replacement)
        return parsed

    monkeypatch.setattr(module, "strict_json_object", replace_after_parse)
    _loaded, digest = module.load_measurement_receipt(
        path,
        log_path=log,
        seed=1,
        classifier_init="random",
        training_protocol=protocol,
        checkpoint_sha256_by_epoch=checkpoints,
        training_history_sha256_by_epoch=histories,
    )

    assert digest == module.hashlib.sha256(original).hexdigest()
    assert digest != module.hashlib.sha256(replacement).hexdigest()


def test_checkpoint_hashes_the_exact_bytes_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    path = tmp_path / "epoch-0001.pt"
    original = b"checkpoint-a"
    path.write_bytes(original)
    protocol = _protocol(1, "random")
    checkpoint = {
        "epoch": 1,
        "model": {},
        "classifier": {},
        "ema": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "mask_generator": {},
        "torch_rng_state": {},
        "cuda_rng_states": {},
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": protocol,
        "history": [{"epoch": 1}],
    }

    def fake_load(source, **kwargs):
        assert isinstance(source, Path)
        assert source != path
        assert source.read_bytes() == original
        assert kwargs == {"map_location": "cpu", "weights_only": False, "mmap": True}
        path.write_bytes(b"checkpoint-b")
        return checkpoint

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=fake_load))
    loaded, digest, size = module._load_checkpoint(
        path, seed=1, epoch=1, classifier_init="random"
    )

    assert loaded is checkpoint
    assert digest == module.hashlib.sha256(original).hexdigest()
    assert size == len(original)


def test_initial_model_loads_the_exact_checkpoint_bytes_it_hashes(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    original = b"initial-checkpoint"
    checkpoint.write_bytes(original)

    class Trainer:
        @staticmethod
        def _load_official_model(actual_checkout: Path, snapshot: Path):
            assert actual_checkout == checkout
            assert snapshot.name == checkpoint.name
            assert snapshot.read_bytes() == original
            checkpoint.write_bytes(b"replacement")
            return "model", "transform"

    loaded, digest = module._load_official_model_snapshot(
        Trainer, checkout=checkout, checkpoint=checkpoint
    )

    assert loaded == ("model", "transform")
    assert digest == module.hashlib.sha256(original).hexdigest()


def test_snapshot_streams_when_reflink_and_copy_file_range_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "source.pt"
    destination = tmp_path / "snapshot.pt"
    payload = b"streamed-snapshot" * 1000
    source.write_bytes(payload)

    def unsupported_reflink(*_args) -> None:
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(module.fcntl, "ioctl", unsupported_reflink)
    monkeypatch.delattr(module.os, "copy_file_range", raising=False)
    digest, size = module._snapshot_file(source, destination)

    assert destination.read_bytes() == payload
    assert digest == module.hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_partition_parser_consumes_the_exact_bytes_it_hashes(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "inshop"
    image = root / "Img" / "img.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    payload = (
        b"2\nimage_name item_id evaluation_status\n"
        b"img.jpg item_1 query\nimg.jpg item_1 gallery\n"
    )

    records, digest = module._parse_partition_bytes(
        root, payload, expected_counts=None
    )

    assert [(row.label, row.split) for row in records] == [
        ("item_1", "query"),
        ("item_1", "gallery"),
    ]
    assert digest == module.hashlib.sha256(payload).hexdigest()


def test_atomic_report_rolls_back_owned_destination_after_post_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    calls = 0
    real_fsync = module.os.fsync

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_second_fsync)
    with pytest.raises(OSError, match="directory fsync"):
        module.write_report_atomic(_build(module), output)
    assert not output.exists()
    rollback = list(tmp_path.glob(".*.rollback"))
    assert len(rollback) == 1
    module.validate_replication_pair(module.strict_json_object(rollback[0].read_bytes()))


def test_rollback_preserves_destination_replaced_by_foreign_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    output.write_bytes(b"owned")
    info = output.stat()
    owned = (info.st_dev, info.st_ino)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign")
    real_rename = module.os.rename
    real_rename_noreplace = module._rename_noreplace
    calls = 0

    def replace_then_rename(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            source.unlink()
            real_rename(foreign, source)
        real_rename_noreplace(source, destination)

    monkeypatch.setattr(module, "_rename_noreplace", replace_then_rename)
    directory_descriptor = module.os.open(tmp_path, module.os.O_RDONLY | module.os.O_DIRECTORY)
    try:
        module._rollback_published_link(
            output, owned=owned, directory_descriptor=directory_descriptor
        )
    finally:
        module.os.close(directory_descriptor)
    assert output.read_bytes() == b"foreign"
    assert list(tmp_path.iterdir()) == [output]


def test_rollback_never_clobbers_a_foreign_quarantine_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    output.write_bytes(b"owned")
    info = output.stat()
    owned = (info.st_dev, info.st_ino)
    real_rename_noreplace = module._rename_noreplace
    collision: Path | None = None
    calls = 0

    def collide_once(source: Path, destination: Path) -> None:
        nonlocal calls, collision
        calls += 1
        if calls == 1:
            collision = destination
            destination.write_bytes(b"foreign quarantine")
        real_rename_noreplace(source, destination)

    monkeypatch.setattr(module, "_rename_noreplace", collide_once)
    directory_descriptor = module.os.open(tmp_path, module.os.O_RDONLY | module.os.O_DIRECTORY)
    try:
        module._rollback_published_link(
            output, owned=owned, directory_descriptor=directory_descriptor
        )
    finally:
        module.os.close(directory_descriptor)

    assert not output.exists()
    assert collision is not None
    assert collision.read_bytes() == b"foreign quarantine"
    entries = list(tmp_path.iterdir())
    assert collision in entries
    owned_quarantine = [path for path in entries if path != collision]
    assert len(owned_quarantine) == 1
    assert owned_quarantine[0].read_bytes() == b"owned"


def test_rollback_never_unlinks_a_quarantine_replaced_after_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    output.write_bytes(b"owned")
    info = output.stat()
    owned = (info.st_dev, info.st_ino)
    real_lstat = module.Path.lstat
    replaced: Path | None = None

    def replace_after_lstat(path: Path):
        nonlocal replaced
        result = real_lstat(path)
        if path.name.endswith(".rollback") and replaced is None:
            path.unlink()
            path.write_bytes(b"foreign after lstat")
            replaced = path
        return result

    monkeypatch.setattr(module.Path, "lstat", replace_after_lstat)
    directory_descriptor = module.os.open(tmp_path, module.os.O_RDONLY | module.os.O_DIRECTORY)
    try:
        module._rollback_published_link(
            output, owned=owned, directory_descriptor=directory_descriptor
        )
    finally:
        module.os.close(directory_descriptor)

    assert replaced is not None
    assert replaced.read_bytes() == b"foreign after lstat"


def test_atomic_report_uses_an_unnamed_inode_and_ignores_foreign_temp(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    temporary = output.with_name(f".{output.name}.{module.os.getpid()}.tmp")
    temporary.write_bytes(b"foreign temp")

    module.write_report_atomic(_build(module), output)

    module.validate_replication_pair(module.strict_json_object(output.read_bytes()))
    assert temporary.read_bytes() == b"foreign temp"


def test_atomic_report_reloads_the_published_inode_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "pair.json"
    real_link_fd = module._link_fd_noreplace

    def corrupt_after_link(
        descriptor: int, destination: Path, directory_descriptor: int
    ) -> None:
        real_link_fd(descriptor, destination, directory_descriptor)
        module.os.lseek(descriptor, 0, module.os.SEEK_SET)
        module.os.write(descriptor, b"!")
        module.os.fsync(descriptor)

    monkeypatch.setattr(module, "_link_fd_noreplace", corrupt_after_link)
    with pytest.raises((ValueError, RuntimeError)):
        module.write_report_atomic(_build(module), output)

    assert not output.exists()
