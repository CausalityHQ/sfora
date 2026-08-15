from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/summarize_unicom_ema_imprint_replication.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_unicom_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol(seed: int, classifier_init: str) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "a" * 64,
        "unicom_revision": "b" * 40,
        "initial_checkpoint_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
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


def _report(seed: int, map_delta: float, recall_delta: float) -> dict[str, object]:
    baseline_map = 0.89 + seed * 0.001
    baseline_recall = 0.97 + seed * 0.001
    epochs = [4, 8, 12, 16]

    def arm(*, imprinted: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
        endpoint_map = baseline_map + (map_delta if imprinted else 0.0)
        endpoint_recall = baseline_recall + (recall_delta if imprinted else 0.0)
        start = endpoint_map - (0.005 if imprinted else 0.03)
        map_values = [
            endpoint_map if epoch == 16 else start + index * 0.01
            for index, epoch in enumerate(epochs)
        ]
        true_count = round(endpoint_recall * 1000)
        top1 = [True] * true_count + [False] * (1000 - true_count)
        evidence = [
            {"top1_correct": list(top1), "average_precision": [value] * 1000}
            for value in map_values
        ]
        return {
            "checkpoint_sha256_by_epoch": [
                f"{seed * 8 + (4 if imprinted else 0) + index:064x}" for index in range(4)
            ],
            "training_history_sha256_by_epoch": [
                f"{1000 + seed * 8 + (4 if imprinted else 0) + index:064x}"
                for index in range(4)
            ],
            "epoch_metrics": [
                {
                    "epoch": epoch,
                    "map_at_r": float(np.mean(np.asarray(evidence[index]["average_precision"]))),
                    "recall_at_1": float(np.mean(np.asarray(top1, dtype=np.float64))),
                }
                for index, epoch in enumerate(epochs)
            ],
            "training_seconds": (14000.0 if imprinted else 15000.0) + seed,
            "peak_gpu_mib": (86900 if imprinted else 87000) + seed,
            "checkpoint_storage_bytes": (14499999900 if imprinted else 14500000000) + seed,
            "deployment_storage_bytes": 1820000000 + seed,
            "measurement_receipt_sha256": (
                f"{5000 + seed * 2 + (1 if imprinted else 0):064x}"
            ),
            "profile": {
                "step_wall_seconds": 1.0,
                "fusible_non_backbone_seconds": 0.05,
            },
        }, evidence

    random_arm, random_evidence = arm(imprinted=False)
    imprinted_arm, imprinted_evidence = arm(imprinted=True)

    return {
        "schema_version": "unicom-ema-imprint-replication-pair-v1",
        "seed": seed,
        "selected_cell": "imprinted_raw",
        "registered_epochs": epochs,
        "random_training_protocol": _protocol(seed, "random"),
        "imprinted_training_protocol": _protocol(seed, "imprinted"),
        "random_raw": random_arm,
        "imprinted_raw": imprinted_arm,
        "inference_latency": {
            "warmup_repetitions": 10,
            "measured_repetitions": 50,
            "batch_size": 128,
            "milliseconds_per_image": 11.8,
        },
        "evidence": {"random_raw": random_evidence, "imprinted_raw": imprinted_evidence},
    }


def test_summary_requires_exact_seeds_and_frozen_cell() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.002) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["training_seeds"] == [1, 2, 3, 4, 5, 6]
    assert summary["selected_cell"] == "imprinted_raw"
    assert summary["claim_supported"] is True
    reports[3]["seed"] = 7
    with pytest.raises(ValueError, match="seed|registration"):
        module.summarize_replications(reports)


def test_summary_uses_paired_student_t_sign_and_recall_gates() -> None:
    module = _load_script()
    reports = [_report(seed, 0.010 + seed * 0.001, -0.001) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["map_deltas"] == pytest.approx([0.011, 0.012, 0.013, 0.014, 0.015, 0.016])
    assert summary["map_delta_sample_standard_deviation"] > 0.0
    assert summary["map_delta_paired_student_t_95_interval"][0] > 0.0
    assert summary["exact_two_sided_sign_p_value"] == 0.03125
    assert summary["all_map_deltas_positive"] is True
    assert summary["all_recall_at_1_deltas_above_guard"] is True
    reports[5]["evidence"]["imprinted_raw"][-1]["top1_correct"][0] = False
    reports[5]["imprinted_raw"]["epoch_metrics"][-1]["recall_at_1"] = float(
        np.mean(reports[5]["evidence"]["imprinted_raw"][-1]["top1_correct"])
    )
    assert module.summarize_replications(reports)["claim_supported"] is False


def test_summary_recomputes_time_to_quality_and_cost_pareto_fields() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["first_quality_epochs"] == [
        {"seed": seed, "random_raw": 16, "imprinted_raw": 4, "speedup": 4.0}
        for seed in range(1, 7)
    ]
    assert summary["costs"]["training_seconds"][0] == {
        "seed": 1,
        "random_raw": 15001.0,
        "imprinted_raw": 14001.0,
    }
    assert summary["costs"]["inference_latency_protocol"] == {
        "warmup_repetitions": 10,
        "measured_repetitions": 50,
        "batch_size": 128,
    }
    assert summary["costs"]["kernel_profile_threshold"] == 0.1
    assert summary["costs"]["kernel_eligible"] is False
    assert summary["pareto_cost_noninferior"] is True
    assert summary["pareto_nondominated_against_random_raw"] is True

    dominated = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    for report in dominated:
        report["imprinted_raw"]["training_seconds"] = 20000.0
    dominated_summary = module.summarize_replications(dominated)
    assert dominated_summary["quality_claim_supported"] is True
    assert dominated_summary["pareto_cost_noninferior"] is False
    assert dominated_summary["pareto_nondominated_against_random_raw"] is False
    assert dominated_summary["claim_supported"] is False

    checkpoint_dominated = [
        _report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)
    ]
    for report in checkpoint_dominated:
        report["imprinted_raw"]["checkpoint_storage_bytes"] = 10**15
    assert module.summarize_replications(checkpoint_dominated)["pareto_cost_noninferior"] is False

    reports[0]["evidence"]["imprinted_raw"][0]["average_precision"] = [0.0] * 1000
    reports[0]["imprinted_raw"]["epoch_metrics"][0]["map_at_r"] = 0.0
    assert module.summarize_replications(reports)["first_quality_epochs"][0] == {
        "seed": 1,
        "random_raw": 16,
        "imprinted_raw": 8,
        "speedup": 2.0,
    }


def test_summary_rejects_degenerate_or_reused_checkpoint_evidence() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01, 0.0) for seed in range(1, 7)]
    for report in reports:
        for cell, value in (("random_raw", 0.5), ("imprinted_raw", 0.51)):
            report["evidence"][cell][-1]["average_precision"] = [value] * 1000
            report[cell]["epoch_metrics"][-1]["map_at_r"] = float(
                np.mean(report["evidence"][cell][-1]["average_precision"])
            )

    degenerate = module.summarize_replications(reports)

    assert degenerate["nondegenerate_training_seed_variation"] is False
    assert degenerate["claim_supported"] is False
    reports[5]["imprinted_raw"]["checkpoint_sha256_by_epoch"][3] = reports[0][
        "random_raw"
    ]["checkpoint_sha256_by_epoch"][0]
    with pytest.raises(ValueError, match="checkpoint"):
        module.summarize_replications(reports)


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    (
        ("random_raw", "training_seconds", None),
        ("imprinted_raw", "peak_gpu_mib", -1),
        ("random_raw", "checkpoint_storage_bytes", 0.5),
        ("imprinted_raw", "deployment_storage_bytes", 0),
    ),
)
def test_summary_requires_complete_registered_costs(arm: str, field: str, value: object) -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    reports[2][arm][field] = value

    with pytest.raises((TypeError, ValueError), match="cost|epoch"):
        module.summarize_replications(reports)


def test_atomic_publication_strict_reloads_and_never_clobbers(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    summary = module.summarize_replications(reports)
    output = tmp_path / "summary.json"

    module.write_summary_atomic(summary, output)

    persisted = module.strict_json_object(output.read_bytes())
    module.validate_summary(persisted)
    assert persisted == summary
    assert list(tmp_path.iterdir()) == [output]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_summary_atomic(summary, output)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_summary_rejects_cross_seed_protocol_or_query_count_drift() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    reports[1]["random_training_protocol"]["trainer_sha256"] = "e" * 64
    reports[1]["imprinted_training_protocol"]["trainer_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="protocol"):
        module.summarize_replications(reports)

    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    for cell in ("random_raw", "imprinted_raw"):
        for evidence, metric in zip(
            reports[1]["evidence"][cell], reports[1][cell]["epoch_metrics"], strict=True
        ):
            evidence["top1_correct"].pop()
            evidence["average_precision"].pop()
            metric["map_at_r"] = float(np.mean(evidence["average_precision"]))
            metric["recall_at_1"] = float(np.mean(evidence["top1_correct"]))
    with pytest.raises(ValueError, match="query"):
        module.summarize_replications(reports)


def test_summary_atomic_rolls_back_after_post_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
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
        module.write_summary_atomic(module.summarize_replications(reports), output)
    assert not output.exists()
    rollback = list(tmp_path.glob(".*.rollback"))
    assert len(rollback) == 1
    module.validate_summary(module.strict_json_object(rollback[0].read_bytes()))


def test_summary_rollback_never_clobbers_a_foreign_quarantine_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
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


def test_atomic_summary_uses_an_unnamed_inode_and_ignores_foreign_temp(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    temporary = output.with_name(f".{output.name}.{module.os.getpid()}.tmp")
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    temporary.write_bytes(b"foreign temp")

    module.write_summary_atomic(module.summarize_replications(reports), output)

    module.validate_summary(module.strict_json_object(output.read_bytes()))
    assert temporary.read_bytes() == b"foreign temp"


def test_atomic_summary_reloads_the_published_inode_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
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
        module.write_summary_atomic(module.summarize_replications(reports), output)

    assert not output.exists()
