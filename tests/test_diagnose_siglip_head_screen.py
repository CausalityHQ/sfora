"""Tests for the authenticated cached-feature SigLIP head screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_head_screen.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_siglip_head_screen", _SCRIPT)
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
        values[3] = 0.1
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
                for example_id, label in zip(band["example_ids"], band["labels"], strict=True)
            ],
            key=lambda row: row["example_id"],
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


def test_cli_is_local_only_and_requires_explicit_execution() -> None:
    valid = [
        "--feature-manifest",
        "/authority/features.json",
        "--feature-manifest-sha256",
        "11" * 32,
        "--result",
        "/output/result.json",
        "--device",
        "cuda",
        "--execute-head-screen",
    ]
    parsed = _MODULE.parse_args(valid)
    assert parsed.feature_manifest == Path("/authority/features.json")
    for forbidden in ("--dataset-split", "--test-manifest", "--url", "--aws-profile"):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*valid, forbidden, "value"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args(valid[:-1])


def test_feature_cache_authenticates_all_bands_and_rejects_drift(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    cache = _MODULE.load_feature_cache(manifest, expected_sha256=digest)

    assert cache.optimization.features.shape == (196, 4)
    assert cache.clean_validation.role == "clean-validation"
    assert cache.burned_diagnostic.role == "burned-diagnostic"
    cache.clean_validation.features[0, 0] += 0.5
    with pytest.raises(ValueError, match="changed after authentication"):
        _MODULE.run_head_screen(
            cache,
            master_seed_sha256="2" * 64,
            output_dimensions=2,
            subclasses_per_class=2,
            cluster_iterations=3,
            train_steps=2,
            device="cpu",
        )
    (tmp_path / "optimization-train.npy").write_bytes(b"drift")
    with pytest.raises(ValueError, match="feature matrix digest"):
        _MODULE.load_feature_cache(manifest, expected_sha256=digest)


def test_head_screen_emits_deterministic_claim_ineligible_result(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache = _MODULE.load_feature_cache(manifest, expected_sha256=digest)

    first = _MODULE.run_head_screen(
        cache,
        master_seed_sha256="2" * 64,
        output_dimensions=2,
        subclasses_per_class=2,
        cluster_iterations=3,
        train_steps=6,
        device="cpu",
    )
    second = _MODULE.run_head_screen(
        cache,
        master_seed_sha256="2" * 64,
        output_dimensions=2,
        subclasses_per_class=2,
        cluster_iterations=3,
        train_steps=6,
        device="cpu",
    )

    assert first == second
    value = json.loads(first)
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")
    assert value["schema"] == "sfora-siglip-cached-head-screen-v1"
    assert value["claim_eligible"] is False
    assert value["official_test_access"] is False
    assert (
        value["source_manifest_sha256"]
        == hashlib.sha256((tmp_path / "control-manifest.json").read_bytes()).hexdigest()
    )
    assert value["source_commit"] == "3" * 40
    assert value["clean_recall_at_1_gate"] == 0.974
    assert type(value["passed"]) is bool
    assert value["procedure"]["logical_batch_size"] == 120
    assert value["procedure"]["projection_learning_rate"] == 0.01
    assert value["procedure"]["proxy_learning_rate"] == 0.02
    assert value["procedure"]["weight_decay"] == 0.0
    assert value["procedure"]["alpha"] == 32.0
    assert value["procedure"]["delta"] == 0.1
    assert value["procedure"]["spectral_cut_ratio"] is not None
    assert value["procedure"]["control_cotangent_rank_evidence"]["class_count"] == 49
    assert value["procedure"]["subclass_cotangent_rank_evidence"]["class_count"] == 98
    assert len(value["training"]["loss_trajectory"]) == 7
    assert len(value["training"]["projection_sha256"]) == 64
    assert len(value["training"]["subclass_proxies_sha256"]) == 64
    assert value["training"]["initial_loss"] > value["training"]["final_loss"]
    assert set(value["bands"]) == {"optimization", "clean_validation", "burned_diagnostic"}
    assert set(value["bands"]["clean_validation"]) == {"raw", "spectral", "trained"}
