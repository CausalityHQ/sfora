"""Tests for the strict local-only CDGA fold CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sfora.siglip_cdga import validate_cdga_result_bytes

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_cdga.py"
_SPEC = importlib.util.spec_from_file_location("scripts.diagnose_siglip_cdga", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_band(root: Path, role: str, labels: list[int]) -> dict[str, object]:
    rows = []
    for row, label in enumerate(labels):
        values = np.zeros(4, dtype=np.float32)
        values[label % 2] = 1.0
        values[2] = (row % 3 - 1) * 0.02
        values[3] = ((label // 2) % 3 - 1) * 0.01
        rows.append(values)
    features = np.stack(rows)
    path = root / f"{role}.npy"
    with path.open("xb") as stream:
        np.save(stream, features, allow_pickle=False)
    return {
        "role": role,
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": [len(labels), 4],
        "example_ids": [f"{role}-{row:04d}" for row in range(len(labels))],
        "labels": labels,
    }


def _write_fixture(root: Path) -> Path:
    bands = {
        "optimization": _write_band(
            root, "optimization-train", [label for label in range(49) for _ in range(4)]
        ),
        "clean_validation": _write_band(
            root, "clean-validation", [label for label in range(49, 82) for _ in range(2)]
        ),
        "burned_diagnostic": _write_band(
            root, "burned-diagnostic", [label for label in range(82, 98) for _ in range(2)]
        ),
    }
    control = {
        "schema": "sfora-siglip-proxy-control-manifest-v1",
        "claim_eligible": False,
        "dataset_id": "tanganke/stanford_cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "examples": sorted(
            [
                {"example_id": example_id, "label": label}
                for band in bands.values()
                for example_id, label in zip(
                    cast(list[str], band["example_ids"]),
                    cast(list[int], band["labels"]),
                    strict=True,
                )
            ],
            key=lambda row: cast(str, row["example_id"]),
        ),
    }
    control_raw = _canonical(control)
    (root / "control-manifest.json").write_bytes(control_raw)
    manifest = {
        "schema": "sfora-siglip-head-feature-cache-v1",
        "claim_eligible": False,
        "source_manifest_sha256": hashlib.sha256(control_raw).hexdigest(),
        "control_manifest_file": "control-manifest.json",
        "source_commit": "3" * 40,
        "model_name": "google/siglip-so400m-patch14-384",
        "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "bands": bands,
    }
    path = root / "features.json"
    path.write_bytes(_canonical(manifest))
    return path


def _valid_args(*, manifest: str = "/cache/features.json") -> list[str]:
    return [
        "--feature-manifest",
        manifest,
        "--feature-manifest-sha256",
        "11" * 32,
        "--feature-source-commit",
        "3" * 40,
        "--result",
        "/out/cdga.json",
        "--execute-cdga-folds",
    ]


def test_cdga_cli_refuses_evaluation_network_and_tuning_flags() -> None:
    parsed = _MODULE.parse_args(_valid_args())
    assert parsed.execute_cdga_folds is True

    for forbidden in (
        "--clean",
        "--burned",
        "--test",
        "--url",
        "--aws-profile",
        "--output-dimensions",
        "--train-steps",
        "--learning-rate",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*_valid_args(), forbidden, "value"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*_valid_args(), "--result", "/out/duplicate.json"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args(_valid_args()[:-1])


def test_cdga_cli_authenticates_cache_and_writes_canonical_result(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    result = tmp_path / "cdga.json"
    arguments = [
        "--feature-manifest",
        str(manifest),
        "--feature-manifest-sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "--feature-source-commit",
        "3" * 40,
        "--result",
        str(result),
        "--execute-cdga-folds",
    ]

    assert _MODULE.main(arguments) == 0
    parsed = validate_cdga_result_bytes(result.read_bytes())
    manifest_value = json.loads(manifest.read_bytes())
    assert parsed.claim_eligible is False
    assert parsed.official_test_access is False
    assert parsed.source_manifest_sha256 == manifest_value["source_manifest_sha256"]
    assert parsed.feature_cache_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert parsed.output_dimensions == 4
    assert parsed.fold_count == 4
    with pytest.raises(FileExistsError):
        _MODULE.main(arguments)


def test_cdga_cli_never_opens_evaluation_feature_rows(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    (tmp_path / "clean-validation.npy").unlink()
    (tmp_path / "burned-diagnostic.npy").unlink()

    assert (
        _MODULE.main(
            [
                "--feature-manifest",
                str(manifest),
                "--feature-manifest-sha256",
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "--feature-source-commit",
                "3" * 40,
                "--result",
                str(tmp_path / "cdga.json"),
                "--execute-cdga-folds",
            ]
        )
        == 0
    )


def test_cdga_cli_rejects_manifest_source_and_optimization_band_drift(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    base = [
        "--feature-manifest",
        str(manifest),
        "--feature-manifest-sha256",
        digest,
        "--feature-source-commit",
        "3" * 40,
        "--result",
        str(tmp_path / "cdga.json"),
        "--execute-cdga-folds",
    ]
    with pytest.raises(ValueError, match="manifest digest"):
        _MODULE.main([*base[:3], "4" * 64, *base[4:]])
    with pytest.raises(ValueError, match="source revision"):
        _MODULE.main([*base[:5], "4" * 40, *base[6:]])

    (tmp_path / "optimization-train.npy").unlink()
    with pytest.raises(ValueError, match="feature matrix path"):
        _MODULE.main(base)
