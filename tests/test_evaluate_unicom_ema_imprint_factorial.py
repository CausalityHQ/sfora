from __future__ import annotations

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
    return {
        "cell": cell,
        "epoch": epoch,
        "metrics": {"map_at_r": map_at_r, "recall_at_1": recall_at_1},
        "query_evidence": {
            "average_precision": [map_at_r - 0.01, map_at_r + 0.01],
            "top1_correct": [recall_at_1, recall_at_1],
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


def _valid_report(module) -> dict[str, object]:
    def protocol(classifier_init: str) -> dict[str, object]:
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
            "random_run": "/random",
            "imprinted_run": "/imprinted",
            "random_training_protocol": protocol("random"),
            "imprinted_training_protocol": protocol("imprinted"),
        },
        "costs": {
            "random_training_seconds": 1.0,
            "imprinted_training_seconds": 2.0,
            "random_peak_gpu_mib": 3,
            "imprinted_peak_gpu_mib": 4,
            "checkpoint_storage_bytes": 5,
            "inference_latency_ms_per_image": 0.1,
            "evaluator_seconds": 6.0,
        },
        "rows": rows,
        "gate": module.factorial_gate(rows, bootstrap_interval=interval),
    }


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
