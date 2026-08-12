from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sfora.lorentz_rider_io import publish_lorentz_report, validate_lorentz_report

SHA = "1" * 64


def _valid_report() -> dict[str, object]:
    replicates = [0.0] * 10_000
    correctness = {
        "lorentz": [True, True, True, False],
        "pca_euclidean": [True, True, False, False],
        "pca_cosine": [True, False, True, False],
        "spatial_only": [True, True, False, False],
        "power_1": [True, False, True, False],
        "power_3": [False, True, True, False],
    }
    controls = {
        name: {
            "recall_at_1": float(np.mean(values)),
            "correct": values,
            "correct_sha256": hashlib.sha256(
                np.asarray(values, dtype=np.bool_).tobytes()
            ).hexdigest(),
        }
        for name, values in correctness.items()
    }
    return {
        "schema_version": 1,
        "input": {
            "bundle_path": "/tmp/bundle.npz",
            "bundle_sha256": SHA,
            "metadata_sha256": "2" * 64,
            "array_sha256": {
                "train_embeddings": "3" * 64,
                "train_labels": "4" * 64,
                "query_embeddings": "5" * 64,
                "query_labels": "6" * 64,
                "gallery_embeddings": "7" * 64,
                "gallery_labels": "8" * 64,
            },
            "embedding_dimension": 32,
            "train_count": 2000,
            "query_count": 4,
            "gallery_count": 8,
            "identical_query_gallery": False,
            "dataset_index": 0,
        },
        "constants": {
            "dimensions": [8, 16, 32, 64],
            "target_median_radii": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            "clip": 2.5,
            "curvature": -1.0,
            "l0_subsample_count": 10,
            "l0_subsample_size": 2000,
            "l0_seed_formula": "7000 + 100 * dataset_index + replicate",
            "l0_null_seed_formula": "7500 + 100 * dataset_index + replicate",
            "l0_relative_convention": "2 * delta / diameter",
            "l0_close_threshold": 0.3,
            "mips_query_transform": "(-q0, qs)",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 205,
            "norm_correlation_threshold": 0.05,
        },
        "l0": {
            "replicates": [
                {
                    "replicate": index,
                    "seed": 7000 + index,
                    "null_seed": 7500 + index,
                    "indices_sha256": f"{index:x}" * 64,
                    "base": 0,
                    "diameter": 2.0,
                    "delta": 0.2,
                    "delta_rel": 0.2,
                    "permutation_delta_rel": 0.3,
                    "spectrum_delta_rel": 0.25,
                }
                for index in range(10)
            ],
            "observed_mean": float(np.mean([0.2] * 10)),
            "permutation_mean": float(np.mean([0.3] * 10)),
            "spectrum_mean": float(np.mean([0.25] * 10)),
        },
        "l1": {
            "probe": {
                "dimension": 32,
                "norm_identity_frequency_spearman": 0.1,
                "norm_cosine_margin_spearman": 0.2,
                "status": "NORM_CHANNEL_INFORMATIVE",
            },
            "dimensions": [
                {
                    "dimension": 32,
                    "pca_mean_sha256": "a" * 64,
                    "pca_components_sha256": "b" * 64,
                    "train_median_radius": 2.0,
                    "scales": [
                        {
                            "target_median_radius": target,
                            "scale": target / 2.0,
                            "controls": controls,
                        }
                        for target in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
                    ],
                    "baseline_name": "pca_euclidean",
                    "bootstrap": {
                        "point": 0.25,
                        "lower": 0.0,
                        "upper": 0.0,
                        "standard_errors": 0.0,
                        "samples": 10000,
                        "seed": 205,
                        "replicates": replicates,
                        "replicates_sha256": hashlib.sha256(
                            np.asarray(replicates, dtype=np.float64).tobytes()
                        ).hexdigest(),
                    },
                    "best_target_median_radius": 0.125,
                    "best_endpoint_gain": 0.25,
                    "best_spatial_gain": 0.25,
                    "best_power_gain": 0.25,
                    "decision": {
                        "status": "CLOSE_GEOMETRY",
                        "endpoint_passed": False,
                        "spatial_passed": True,
                        "power_passed": True,
                    },
                }
            ],
        },
        "runtime": {"python": "3.12.3", "numpy": "2.5.0"},
        "decision": {
            "status": "CLOSE_GEOMETRY_DATASET",
            "l0_mean_delta_rel": float(np.mean([0.2] * 10)),
            "l0_above_close_threshold": False,
            "evaluated_dimensions": [32],
        },
        "timing": {"l0_seconds": 1.0, "l1_seconds": 2.0, "total_seconds": 3.0},
    }


def test_report_roundtrips_with_exact_recursive_schema(tmp_path: Path) -> None:
    report = _valid_report()
    validate_lorentz_report(report)
    output = tmp_path / "report.json"
    publish_lorentz_report(output, report)
    assert output.stat().st_mode & 0o777 == 0o600
    persisted = json.loads(output.read_text())
    assert persisted == report
    validate_lorentz_report(persisted)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.__setitem__("schema_version", True),
        lambda value: value["constants"].__setitem__("l0_relative_convention", "delta/diameter"),
        lambda value: value["constants"].__setitem__("mips_query_transform", "(q0, -qs)"),
        lambda value: value["l0"]["replicates"][0].__setitem__("delta_rel", 0.3),
        lambda value: value["decision"].__setitem__("l0_above_close_threshold", True),
        lambda value: value["l1"]["dimensions"][0]["scales"][0].__setitem__("scale", 123.0),
        lambda value: value["l1"]["dimensions"][0]["bootstrap"].__setitem__("lower", 0.01),
        lambda value: value["l1"]["dimensions"][0]["scales"][0]["controls"]["lorentz"].__setitem__(
            "correct_sha256", "f" * 64
        ),
        lambda value: value["l1"]["dimensions"][0].__setitem__("baseline_name", "pca_cosine"),
        lambda value: value["l1"]["dimensions"][0].__setitem__("best_endpoint_gain", 0.06),
    ],
)
def test_validator_rejects_registered_mutations(mutate) -> None:
    report = copy.deepcopy(_valid_report())
    mutate(report)
    with pytest.raises((TypeError, ValueError)):
        validate_lorentz_report(report)


def test_validator_records_negative_endpoint_gain_and_standard_errors() -> None:
    report = _valid_report()
    lorentz_correct = [True, False, False, False]
    for scale in report["l1"]["dimensions"][0]["scales"]:
        lorentz = scale["controls"]["lorentz"]
        lorentz["recall_at_1"] = 0.25
        lorentz["correct"] = lorentz_correct
        lorentz["correct_sha256"] = hashlib.sha256(
            np.asarray(lorentz_correct, dtype=np.bool_).tobytes()
        ).hexdigest()
    dimension = report["l1"]["dimensions"][0]
    dimension["best_endpoint_gain"] = -0.25
    dimension["best_spatial_gain"] = -0.25
    dimension["best_power_gain"] = -0.25
    replicates = [-0.5, 0.0] * 5000
    values = np.asarray(replicates, dtype=np.float64)
    deviation = float(np.std(values, ddof=1))
    bootstrap = dimension["bootstrap"]
    bootstrap["point"] = -0.25
    bootstrap["lower"] = -0.5
    bootstrap["upper"] = 0.0
    bootstrap["standard_errors"] = -0.25 / deviation
    bootstrap["replicates"] = replicates
    bootstrap["replicates_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    dimension["decision"] = {
        "status": "CLOSE_GEOMETRY",
        "endpoint_passed": False,
        "spatial_passed": False,
        "power_passed": False,
    }
    validate_lorentz_report(report)


def test_publication_never_clobbers_or_removes_foreign_temp(tmp_path: Path) -> None:
    report = _valid_report()
    output = tmp_path / "report.json"
    output.write_bytes(b"sentinel")
    foreign = tmp_path / f".{output.name}.foreign.tmp"
    foreign.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        publish_lorentz_report(output, report)
    assert output.read_bytes() == b"sentinel"
    assert foreign.read_bytes() == b"foreign"


def test_published_bytes_are_stable_and_lf_terminated(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    publish_lorentz_report(output, _valid_report())
    data = output.read_bytes()
    assert data.endswith(b"\n")
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(output.read_bytes()).hexdigest()
    assert os.stat(output).st_nlink == 1


def test_publication_race_preserves_foreign_destination(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    def raced_link(_source, destination):
        Path(destination).write_bytes(b"foreign-race-winner")
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        publish_lorentz_report(output, _valid_report())
    assert output.read_bytes() == b"foreign-race-winner"
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_failed_strict_reload_rolls_back_only_owned_destination(
    monkeypatch, tmp_path: Path
) -> None:
    import sfora.lorentz_rider_io as module

    output = tmp_path / "report.json"
    calls = 0
    real_validate = module.validate_lorentz_report

    def fail_second(value):
        nonlocal calls
        calls += 1
        real_validate(value)
        if calls == 2:
            raise ValueError("injected strict reload failure")

    monkeypatch.setattr(module, "validate_lorentz_report", fail_second)
    with pytest.raises(ValueError, match="strict reload"):
        module.publish_lorentz_report(output, _valid_report())
    assert not output.exists()
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))
