"""Tests for the strict local-only SFQ fold CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sfora.siglip_sfq import validate_sfq_result_bytes

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_sfq.py"
_SPEC = importlib.util.spec_from_file_location("scripts.diagnose_siglip_sfq", _SCRIPT)
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
            root,
            "optimization-train",
            [label for label in range(49) for _ in range(4)],
        ),
        "clean_validation": _write_band(
            root,
            "clean-validation",
            [label for label in range(49, 82) for _ in range(2)],
        ),
        "burned_diagnostic": _write_band(
            root,
            "burned-diagnostic",
            [label for label in range(82, 98) for _ in range(2)],
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


def test_sfq_cli_refuses_evaluation_network_and_implicit_execution_flags() -> None:
    """Adding an evaluation selector, network flag, or implicit run must fail this test."""

    valid = [
        "--feature-manifest",
        "/cache/features.json",
        "--feature-manifest-sha256",
        "11" * 32,
        "--feature-source-commit",
        "3" * 40,
        "--result",
        "/out/sfq.json",
        "--execute-sfq-folds",
    ]

    parsed = _MODULE.parse_args(valid)

    assert parsed.execute_sfq_folds is True
    for forbidden in (
        "--clean",
        "--burned",
        "--test",
        "--url",
        "--aws-profile",
        "--output-dimensions",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*valid, forbidden, "value"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*valid, "--result", "/out/duplicate.json"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*valid, "--feature-man=/cache/features.json"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*valid[:1], "relative/features.json", *valid[2:]])
    with pytest.raises(SystemExit):
        _MODULE.parse_args(valid[:-1])


def test_sfq_cli_authenticates_cache_and_writes_one_canonical_result(tmp_path: Path) -> None:
    """Bypassing the real cache loader or emitting noncanonical output must fail this test."""

    manifest = _write_fixture(tmp_path)
    result = tmp_path / "sfq.json"

    status = _MODULE.main(
        [
            "--feature-manifest",
            str(manifest),
            "--feature-manifest-sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "--feature-source-commit",
            "3" * 40,
            "--result",
            str(result),
            "--execute-sfq-folds",
        ]
    )

    assert status == 0
    parsed = validate_sfq_result_bytes(result.read_bytes())
    assert parsed.claim_eligible is False
    assert parsed.official_test_access is False
    manifest_value = json.loads(manifest.read_bytes())
    assert parsed.source_manifest_sha256 == manifest_value["source_manifest_sha256"]
    assert parsed.feature_cache_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert parsed.output_dimensions == 4
    with pytest.raises(FileExistsError):
        _MODULE.main(
            [
                "--feature-manifest",
                str(manifest),
                "--feature-manifest-sha256",
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "--feature-source-commit",
                "3" * 40,
                "--result",
                str(result),
                "--execute-sfq-folds",
            ]
        )
    with pytest.raises(ValueError, match="source revision"):
        _MODULE.main(
            [
                "--feature-manifest",
                str(manifest),
                "--feature-manifest-sha256",
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "--feature-source-commit",
                "4" * 40,
                "--result",
                str(tmp_path / "wrong-source.json"),
                "--execute-sfq-folds",
            ]
        )


def test_sfq_cli_never_opens_evaluation_feature_rows(tmp_path: Path) -> None:
    """Clean and burned feature files must not be capabilities of the SFQ process."""

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
                str(tmp_path / "sfq.json"),
                "--execute-sfq-folds",
            ]
        )
        == 0
    )
