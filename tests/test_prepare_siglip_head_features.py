"""Tests for the optimization-safe SigLIP feature-cache preparer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.data import ImageExample

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_siglip_head_features.py"
_SPEC = importlib.util.spec_from_file_location("prepare_siglip_head_features", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _bands():
    def rows(classes: range, count: int) -> tuple[ImageExample, ...]:
        return tuple(
            ImageExample(f"image-{label:03d}-{row:02d}", object(), label)
            for label in classes
            for row in range(count)
        )

    optimization = rows(range(49), 4)
    clean = rows(range(49, 82), 2)
    burned = rows(range(82, 98), 2)
    ordered = tuple(sorted((*optimization, *clean, *burned), key=lambda item: item.example_id))
    return _MODULE.ControlExampleBands(optimization, clean, burned, ordered)


def test_preparer_seals_exact_disjoint_feature_roles(tmp_path: Path) -> None:
    bands = _bands()
    control = _MODULE.control_manifest_artifact_bytes(bands)

    def embed(examples: tuple[ImageExample, ...]) -> torch.Tensor:
        return torch.tensor(
            [[float(example.label), float(row), 1.0] for row, example in enumerate(examples)],
            dtype=torch.float32,
        )

    output = tmp_path / "cache"
    manifest = _MODULE.prepare_feature_cache(
        bands=bands,
        control_manifest_raw=control,
        output=output,
        embed_band=embed,
        source_commit="3" * 40,
    )

    value = json.loads(manifest)
    assert manifest.endswith(b"\n")
    assert value["claim_eligible"] is False
    assert value["source_manifest_sha256"] == hashlib.sha256(control).hexdigest()
    assert set(value["bands"]) == {"optimization", "clean_validation", "burned_diagnostic"}
    assert value["bands"]["optimization"]["role"] == "optimization-train"
    assert value["bands"]["clean_validation"]["labels"][0] == 49
    assert value["bands"]["burned_diagnostic"]["labels"][0] == 82
    assert (output / "features.json").read_bytes() == manifest
    for role in ("optimization-train", "clean-validation", "burned-diagnostic"):
        matrix = np.load(output / f"{role}.npy", allow_pickle=False)
        assert matrix.dtype == np.float32
        assert matrix.shape[1] == 3


def test_preparer_rejects_manifest_drift_and_cleans_partial_output(tmp_path: Path) -> None:
    bands = _bands()
    control = _MODULE.control_manifest_artifact_bytes(bands)
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="control manifest"):
        _MODULE.prepare_feature_cache(
            bands=bands,
            control_manifest_raw=control + b" ",
            output=output,
            embed_band=lambda examples: torch.ones((len(examples), 3)),
            source_commit="3" * 40,
        )
    assert not output.exists()

    calls = 0

    def interrupted(examples: tuple[ImageExample, ...]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return torch.ones((len(examples), 3))

    with pytest.raises(KeyboardInterrupt):
        _MODULE.prepare_feature_cache(
            bands=bands,
            control_manifest_raw=control,
            output=output,
            embed_band=interrupted,
            source_commit="3" * 40,
        )
    assert not output.exists()
