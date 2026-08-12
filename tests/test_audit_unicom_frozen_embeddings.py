from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sfora.unicom_audit_io import EmbeddingBundle
from sfora.unicom_retrieval_audit import (
    GeometryAudit,
    GeometryConfig,
    GeometryDecision,
    RetrievalView,
)
from sfora.unicom_shard_audit import ShardAudit, ShardConfig

SCRIPT = Path(__file__).parents[1] / "scripts/audit_unicom_frozen_embeddings.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_unicom_frozen_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path) -> EmbeddingBundle:
    empty = np.empty((0, 1), dtype=np.float32)
    labels = np.asarray([], dtype="<U1")
    return EmbeddingBundle(
        path=(tmp_path / "in.npz").resolve(),
        sha256="a" * 64,
        metadata={
            "schema_version": 1,
            "model_identifier": "UNICOM-ViT-B/16",
            "model_revision": "b" * 40,
            "checkpoint_sha256": "c" * 64,
            "image_list_sha256": "d" * 64,
            "transform": "official",
            "embedding_dimension": 768,
            "split_counts": {"train": 25_882, "query": 14_218, "gallery": 12_612},
            "array_sha256": {
                "train_embeddings": "e" * 64,
                "train_labels": "f" * 64,
                "query_embeddings": "1" * 64,
                "query_labels": "2" * 64,
                "gallery_embeddings": "3" * 64,
                "gallery_labels": "4" * 64,
            },
        },
        train_embeddings=empty,
        train_labels=labels,
        query_embeddings=empty,
        query_labels=labels,
        gallery_embeddings=empty,
        gallery_labels=labels,
    )


def _geometry(*, reproduction_passed: bool) -> GeometryAudit:
    top1 = np.asarray([0], dtype=np.int64)
    correct = np.asarray([reproduction_passed], dtype=np.bool_)
    view = RetrievalView(
        recall={1: 0.746 if reproduction_passed else 1.0, 10: 1.0, 20: 1.0, 30: 1.0},
        map_at_r=1.0,
        top1_indices=top1,
        top1_correct=correct,
    )
    primary = "GEOMETRY_NULL" if reproduction_passed else "REPRODUCTION_FAILED"
    return GeometryAudit(
        config=GeometryConfig(
            selected_coordinates=512,
            random_mask_count=32,
            bootstrap_samples=10_000,
            expected_official_r1=0.746,
            reproduction_tolerance=0.002,
        ),
        official=view,
        prefix_unit=view,
        full_unit=view,
        random_units=(view,) * 32,
        reproduction_passed=reproduction_passed,
        delta_norm=0.0,
        norm_interval=(0.0, 0.0),
        delta_full=0.0,
        full_interval=(0.0, 0.0),
        delta_mask=0.0,
        mask_wins=0,
        disagree=0.0,
        energy_disagreement_count=0,
        energy_gap_mean=None,
        energy_gap_median=None,
        energy_gap_negative_fraction=None,
        energy_gap_interval=None,
        error_energy_point_biserial=None,
        decision=GeometryDecision(
            primary=primary,
            full_dimension_control=False,
            evaluator_repair=False,
            coordinate_nonexchangeability=False,
        ),
    )


def _sharding() -> ShardAudit:
    return ShardAudit(
        config=ShardConfig(
            panel_classes=64,
            examples_per_class=4,
            selected_coordinates=512,
            trials=32,
            permutations_per_trial=16,
        ),
        independent_loss_range=0.0,
        independent_loss_std=0.0,
        independent_gradient_mse=0.0,
        coherent_gradient_mse=0.0,
        independent_gradient_cosine_distance=0.0,
        coherent_gradient_cosine_distance=0.0,
        coherent_placement_control_error=0.0,
        prediction_change_rate=0.0,
        mask_union_coverage=1.0,
        all_finite=True,
        decision="SHARD_NULL",
    )


def test_cli_accepts_exact_arguments_and_never_imports_torch() -> None:
    module = _load_script()
    assert module.parse_args(["--bundle", "in.npz", "--output", "out.json"]).bundle == Path(
        "in.npz"
    )
    assert "torch" not in module.__dict__
    with pytest.raises(SystemExit):
        module.parse_args(["--bundle", "in.npz"])


def test_cli_runs_e1_then_e2_publishes_once_and_no_clobbers(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    bundle = _bundle(tmp_path)
    calls: list[str] = []
    output = tmp_path / "report.json"
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: bundle)
    monkeypatch.setattr(
        module,
        "run_geometry",
        lambda value: calls.append("E1") or _geometry(reproduction_passed=True),
    )
    monkeypatch.setattr(module, "run_sharding", lambda value: calls.append("E2") or _sharding())

    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 0
    assert calls == ["E1", "E2"]
    original = output.read_bytes()
    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 2
    assert output.read_bytes() == original


def test_cli_publishes_reproduction_failure_but_returns_nonzero(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script()
    output = tmp_path / "report.json"
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: _bundle(tmp_path))
    monkeypatch.setattr(module, "run_geometry", lambda value: _geometry(reproduction_passed=False))
    monkeypatch.setattr(module, "run_sharding", lambda value: _sharding())

    exit_code = module.run(["--bundle", "in.npz", "--output", str(output)])

    assert exit_code != 0
    assert output.is_file()
    captured = capsys.readouterr()
    assert "reproduction failed" in captured.err
    assert "audit complete" not in captured.out


def test_cli_rejects_underpowered_official_count_constants(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script()
    output = tmp_path / "report.json"
    underpowered = _geometry(reproduction_passed=True)
    underpowered = replace(
        underpowered,
        config=replace(underpowered.config, random_mask_count=31),
        random_units=underpowered.random_units[:31],
    )
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: _bundle(tmp_path))
    monkeypatch.setattr(module, "run_geometry", lambda value: underpowered)
    monkeypatch.setattr(module, "run_sharding", lambda value: _sharding())

    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 2
    assert not output.exists()
    assert "official count constants" in capsys.readouterr().err
