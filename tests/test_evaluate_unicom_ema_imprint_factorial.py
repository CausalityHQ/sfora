from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_unicom_ema_imprint_factorial.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_unicom_ema_imprint_factorial", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(cell: str, epoch: int, map_at_r: float, recall_at_1: float) -> dict[str, object]:
    top1 = [True] * 775 + [False] * 22
    return {
        "cell": cell,
        "epoch": epoch,
        "metrics": {"map_at_r": map_at_r, "recall_at_1": recall_at_1},
        "query_evidence": {
            "average_precision": [map_at_r] * len(top1),
            "top1_correct": top1,
        },
    }


def _registered_rows() -> list[dict[str, object]]:
    values = {
        "random_raw": (0.40, 0.50, 0.60, 0.8716329439260202),
        "random_ema": (0.45, 0.55, 0.65, 0.8730),
        "imprinted_raw": (0.60, 0.75, 0.86, 0.8740),
        "imprinted_ema": (0.62, 0.77, 0.87, 0.8752),
    }
    rows = []
    for cell in ("random_raw", "random_ema", "imprinted_raw", "imprinted_ema"):
        for epoch, map_at_r in zip((4, 8, 12, 16), values[cell], strict=True):
            rows.append(_row(cell, epoch, map_at_r, 0.972396486825596))
    return rows


def test_select_candidate_uses_registered_tie_order() -> None:
    module = _load_script()
    rows = _registered_rows()
    for row in rows:
        if row["epoch"] == 16 and row["cell"] != "random_raw":
            row["metrics"] = {"map_at_r": 0.9, "recall_at_1": 0.95}

    selected = module.select_candidate(rows)

    assert selected["cell"] == "random_ema"


@pytest.mark.parametrize(
    ("map_gain", "recall_delta", "bootstrap_lower", "expected"),
    (
        (0.003, -0.00125, 1e-12, True),
        (0.002999, 0.0, 1e-3, False),
        (0.004, -0.001251, 1e-3, False),
        (0.004, 0.0, 0.0, False),
    ),
)
def test_factorial_gate_uses_exact_promotion_boundaries(
    map_gain: float, recall_delta: float, bootstrap_lower: float, expected: bool
) -> None:
    module = _load_script()
    rows = _registered_rows()
    baseline = next(row for row in rows if row["cell"] == "random_raw" and row["epoch"] == 16)
    candidate = next(row for row in rows if row["cell"] == "imprinted_ema" and row["epoch"] == 16)
    candidate["metrics"] = {
        "map_at_r": baseline["metrics"]["map_at_r"] + map_gain,
        "recall_at_1": baseline["metrics"]["recall_at_1"] + recall_delta,
    }
    for row in rows:
        if row["epoch"] == 16 and row["cell"] in ("random_ema", "imprinted_raw"):
            row["metrics"]["map_at_r"] = baseline["metrics"]["map_at_r"]

    gate = module.factorial_gate(rows, bootstrap_interval=(bootstrap_lower, 0.01))

    assert gate["promoted"] is expected
    assert gate["selected_cell"] == "imprinted_ema"


def test_factorial_gate_stops_when_control_does_not_reproduce_endpoint() -> None:
    module = _load_script()
    rows = _registered_rows()
    baseline = next(row for row in rows if row["cell"] == "random_raw" and row["epoch"] == 16)
    baseline["metrics"]["map_at_r"] += 0.0020001

    gate = module.factorial_gate(rows, bootstrap_interval=(0.001, 0.01))

    assert gate["instrument_reproduced"] is False
    assert gate["promoted"] is False
    assert gate["decision"] == "INVALID"


def test_time_to_quality_uses_only_registered_epochs() -> None:
    module = _load_script()
    rows = _registered_rows()

    reached = module.time_to_quality(rows, cell="imprinted_raw", target=0.8716329439260202)
    missed = module.time_to_quality(rows, cell="random_ema", target=0.99)

    assert reached == {"target_map_at_r": 0.8716329439260202, "first_epoch": 16, "speedup": 1.0}
    assert missed == {"target_map_at_r": 0.99, "first_epoch": None, "speedup": None}


def test_paired_bootstrap_is_deterministic_and_rejects_unpaired_evidence() -> None:
    module = _load_script()
    baseline = {"query_evidence": {"average_precision": [0.1, 0.2, 0.3, 0.4]}}
    candidate = {"query_evidence": {"average_precision": [0.2, 0.3, 0.4, 0.5]}}

    first = module.paired_map_bootstrap_interval(baseline, candidate)
    second = module.paired_map_bootstrap_interval(baseline, candidate)

    assert first == second
    assert first[0] > 0.0
    candidate["query_evidence"]["average_precision"].pop()
    with pytest.raises(ValueError, match="paired"):
        module.paired_map_bootstrap_interval(baseline, candidate)


def test_materialize_checkpoint_state_uses_ema_parameters_and_raw_buffers() -> None:
    module = _load_script()
    import torch

    checkpoint = {
        "model": OrderedDict(
            (
                ("weight", torch.tensor([[1.0]], dtype=torch.float32)),
                ("running", torch.tensor([2.0], dtype=torch.float32)),
                ("counter", torch.tensor(3, dtype=torch.int64)),
            )
        ),
        "ema": {
            "decay": 0.999,
            "updates": 4,
            "backbone": {"weight": torch.tensor([[9.0]], dtype=torch.float32)},
            "classifier": torch.tensor([[7.0]], dtype=torch.float32),
        },
    }

    raw = module.materialize_checkpoint_state(checkpoint, use_ema=False)
    ema = module.materialize_checkpoint_state(checkpoint, use_ema=True)

    assert torch.equal(raw["weight"], torch.tensor([[1.0]]))
    assert torch.equal(ema["weight"], torch.tensor([[9.0]]))
    assert torch.equal(ema["running"], torch.tensor([2.0]))
    assert torch.equal(ema["counter"], torch.tensor(3))
    checkpoint["model"]["running"].zero_()
    assert torch.equal(ema["running"], torch.tensor([2.0]))


def test_materialize_checkpoint_state_rejects_shadow_identical_to_raw() -> None:
    module = _load_script()
    import torch

    checkpoint = {
        "model": OrderedDict((("weight", torch.ones(2, dtype=torch.float32)),)),
        "ema": {
            "decay": 0.999,
            "updates": 10,
            "backbone": {"weight": torch.ones(2, dtype=torch.float32)},
            "classifier": torch.ones(1, dtype=torch.float32),
        },
    }

    with pytest.raises(ValueError, match="matches raw"):
        module.materialize_checkpoint_state(checkpoint, use_ema=True)


def test_trainer_checkpoint_roundtrips_through_registered_evaluator(tmp_path: Path) -> None:
    module = _load_script()
    import torch

    trainer_spec = importlib.util.spec_from_file_location(
        "train_unicom_inshop_roundtrip",
        SCRIPT.with_name("train_unicom_inshop.py"),
    )
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name] = trainer
    trainer_spec.loader.exec_module(trainer)
    model = torch.nn.Linear(2, 2)
    classifier = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.SGD((*model.parameters(), classifier), lr=0.1)
    ema = trainer.StepEMA(model, classifier)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update()
    history = [
        {"epoch": index, "train": {"loss": float(index)}, "metrics": None} for index in range(1, 5)
    ]
    path = tmp_path / "epoch-0004.pt"
    trainer.save_training_checkpoint(
        path,
        epoch=4,
        raw_model=model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(7),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=_protocol("random"),
        history=history,
        step_ema=ema,
    )

    checkpoint = module._load_registered_checkpoint(
        path,
        epoch=4,
        classifier_init="random",
        expected_ema_names=tuple(name for name, _parameter in model.named_parameters()),
    )
    raw = module.materialize_checkpoint_state(checkpoint, use_ema=False)
    shadow = module.materialize_checkpoint_state(checkpoint, use_ema=True)

    assert tuple(raw) == tuple(model.state_dict())
    assert tuple(shadow) == tuple(model.state_dict())
    assert checkpoint["history"] == history


def test_registered_checkpoint_rejects_incomplete_history_and_missing_ema(tmp_path: Path) -> None:
    module = _load_script()
    import torch

    path = tmp_path / "epoch-0004.pt"
    payload = {
        "epoch": 4,
        "model": OrderedDict((("weight", torch.ones(1)),)),
        "classifier": torch.ones(1),
        "ema": None,
        "optimizer": {},
        "scheduler": None,
        "scaler": None,
        "mask_generator": torch.Generator().get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": None,
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": _protocol("random"),
        "history": [{"epoch": 1}],
    }
    torch.save(payload, path)

    with pytest.raises(ValueError, match="history"):
        module._load_registered_checkpoint(path, epoch=4, classifier_init="random")
    payload["history"] = [{"epoch": index} for index in range(1, 5)]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="EMA"):
        module._load_registered_checkpoint(path, epoch=4, classifier_init="random")


def test_control_gate_stops_before_imprinted_run_on_bad_reproduction() -> None:
    module = _load_script()
    row = _row("random_raw", 16, module.ARCHIVED_MAP_AT_R, module.ARCHIVED_RECALL_AT_1)

    assert module.control_gate(row) == {
        "instrument_map_tolerance": 0.002,
        "instrument_recall_at_1_tolerance": 0.002,
        "instrument_reproduced": True,
        "decision": "CONTINUE",
    }
    row["metrics"]["map_at_r"] += 0.0020001
    assert module.control_gate(row)["decision"] == "INVALID"


def test_control_cli_does_not_require_imprinted_run() -> None:
    module = _load_script()

    args = module.parse_args(
        [
            "--mode",
            "control",
            "--unicom-checkout",
            "/unicom",
            "--initial-checkpoint",
            "/initial.pt",
            "--dataset-root",
            "/dataset",
            "--random-run",
            "/random",
            "--random-training-seconds",
            "1.0",
            "--random-peak-gpu-mib",
            "2",
            "--output",
            "/control.json",
        ]
    )

    assert args.mode == "control"
    assert args.imprinted_run is None


def _protocol(classifier_init: str) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "d" * 64,
        "unicom_revision": "a" * 40,
        "initial_checkpoint_sha256": "b" * 64,
        "partition_sha256": "c" * 64,
        "seed": 0,
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


def _history_sha256(cell: str, epoch: int) -> str:
    history = [{"epoch": index, "cell": cell} for index in range(1, epoch + 1)]
    payload = json.dumps(history, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_report(module) -> dict[str, object]:

    rows = []
    maps = {
        "random_raw": (0.6, 0.7, 0.8, module.ARCHIVED_MAP_AT_R),
        "random_ema": (0.61, 0.71, 0.81, module.ARCHIVED_MAP_AT_R + 0.001),
        "imprinted_raw": (0.7, 0.8, 0.86, module.ARCHIVED_MAP_AT_R + 0.002),
        "imprinted_ema": (0.72, 0.82, 0.87, module.ARCHIVED_MAP_AT_R + 0.004),
    }
    for cell in module.REGISTERED_CELLS:
        for epoch, map_at_r in zip(module.REGISTERED_EPOCHS, maps[cell], strict=True):
            rows.append(
                {
                    "cell": cell,
                    "epoch": epoch,
                    "checkpoint": f"/{cell}/epoch-{epoch:04d}.pt",
                    "checkpoint_sha256": f"{len(rows) + 1:064x}",
                    "training_history_sha256": _history_sha256(cell, epoch),
                    "ema_updates": None if cell.endswith("_raw") else epoch * 162,
                    "ema_initial_weight": None if cell.endswith("_raw") else 0.999 ** (epoch * 162),
                    "ema_parameter_l2_distance": None if cell.endswith("_raw") else float(epoch),
                    "metrics": {
                        "recall_at_1": module.ARCHIVED_RECALL_AT_1,
                        "recall_at_10": 1.0,
                        "recall_at_20": 1.0,
                        "recall_at_30": 1.0,
                        "map_at_r": map_at_r,
                    },
                    "query_evidence": {
                        "top1_correct": [True] * 775 + [False] * 22,
                        "average_precision": [map_at_r] * 797,
                    },
                }
            )
    selected = module.select_candidate(rows)
    baseline = next(row for row in rows if row["cell"] == "random_raw" and row["epoch"] == 16)
    interval = module.paired_map_bootstrap_interval(baseline, selected)
    return {
        "schema_version": "unicom-ema-imprint-factorial-v1",
        "provenance": {
            "unicom_revision": "a" * 40,
            "initial_checkpoint_sha256": "b" * 64,
            "partition_sha256": "c" * 64,
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "batch_norm_recalibration": "full-optimization-cumulative-batches-all-arms",
            "query_chunk_size": 256,
            "training_history_metrics": "instrument-only-unhardened-not-used-for-selection",
            "control_report": "/control.json",
            "control_report_sha256": "e" * 64,
            "random_run": "/random",
            "imprinted_run": "/imprinted",
            "random_training_protocol": _protocol("random"),
            "imprinted_training_protocol": _protocol("imprinted"),
        },
        "costs": {
            "random_training_seconds": 1.0,
            "imprinted_training_seconds": 2.0,
            "random_peak_gpu_mib": 3,
            "imprinted_peak_gpu_mib": 4,
            "checkpoint_storage_bytes": 5,
            "architecture_inference_latency_ms_per_image": 0.1,
            "evaluator_seconds": 6.0,
        },
        "rows": rows,
        "gate": module.factorial_gate(rows, bootstrap_interval=interval),
    }


def _valid_control_report(module) -> dict[str, object]:
    factorial = _valid_report(module)
    row = next(
        row for row in factorial["rows"] if row["cell"] == "random_raw" and row["epoch"] == 16
    )
    return {
        "schema_version": "unicom-ema-imprint-control-v1",
        "provenance": {
            "unicom_revision": "a" * 40,
            "initial_checkpoint_sha256": "b" * 64,
            "partition_sha256": "c" * 64,
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "batch_norm_recalibration": "full-optimization-cumulative-batches",
            "query_chunk_size": 256,
            "training_history_metrics": "instrument-only-unhardened-not-used-for-selection",
            "random_run": "/random",
            "random_training_protocol": _protocol("random"),
        },
        "costs": {
            "random_training_seconds": 1.0,
            "random_peak_gpu_mib": 2,
            "checkpoint_storage_bytes": 3,
            "architecture_inference_latency_ms_per_image": 0.1,
            "evaluator_seconds": 4.0,
        },
        "row": row,
        "gate": module.control_gate(row),
    }


def test_control_report_validates_and_main_publishes_without_imprinted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    report = _valid_control_report(module)
    module.validate_control_report(report)
    monkeypatch.setattr(module, "run_control", lambda _args: report)
    output = tmp_path / "control.json"

    exit_code = module.main(
        [
            "--mode",
            "control",
            "--unicom-checkout",
            "/unicom",
            "--initial-checkpoint",
            "/initial.pt",
            "--dataset-root",
            "/dataset",
            "--random-run",
            "/random",
            "--random-training-seconds",
            "1.0",
            "--random-peak-gpu-mib",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text()) == report


def test_factorial_binding_rejects_control_from_other_random_checkpoint() -> None:
    module = _load_script()
    factorial = _valid_report(module)
    control = _valid_control_report(module)
    rows = factorial["rows"]
    protocol = factorial["provenance"]["random_training_protocol"]

    module.validate_control_binding(control, rows, protocol)
    control["row"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="control report binding"):
        module.validate_control_binding(control, rows, protocol)


def test_report_validation_recomputes_metrics_gate_and_recursive_schema() -> None:
    module = _load_script()
    report = _valid_report(module)
    module.validate_factorial_report(report)

    report["gate"]["promoted"] = False
    with pytest.raises(ValueError, match="gate"):
        module.validate_factorial_report(report)


def test_atomic_writer_strict_reloads_and_does_not_clobber(tmp_path: Path) -> None:
    module = _load_script()
    report = _valid_report(module)
    output = tmp_path / "factorial.json"

    module.write_report_atomic(report, output)

    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text()) == report
    assert not list(tmp_path.glob("*.tmp"))
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_report_atomic(report, output)
    assert output.read_bytes() == original
