from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sfora.lorentz_rider_io import (
    _norm_probe,
    _score_view,
    evaluate_l0_l1,
    validate_lorentz_report,
)
from sfora.unicom_audit_io import EmbeddingBundle

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_lorentz_rider.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_lorentz_rider", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_relative_paths_before_bundle_open(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    opened = False

    def forbidden(_path):
        nonlocal opened
        opened = True
        raise AssertionError("bundle opened")

    monkeypatch.setattr(module, "load_embedding_bundle", forbidden)
    assert (
        module.run(
            [
                "--bundle",
                "relative.npz",
                "--output",
                str(tmp_path / "x.json"),
                "--dataset-index",
                "0",
            ]
        )
        == 2
    )
    assert opened is False


@pytest.mark.parametrize("kind", ["existing", "symlink", "temp", "missing_parent"])
def test_cli_rejects_output_hazards_before_bundle_open(
    monkeypatch, tmp_path: Path, kind: str
) -> None:
    module = _load_script()
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"bundle")
    output = tmp_path / "report.json"
    if kind == "existing":
        output.write_bytes(b"sentinel")
    elif kind == "symlink":
        output.symlink_to(tmp_path / "missing")
    elif kind == "temp":
        (tmp_path / f".{output.name}.123.tmp").write_bytes(b"temp")
    else:
        output = tmp_path / "missing" / "report.json"
    monkeypatch.setattr(module, "load_embedding_bundle", lambda _path: pytest.fail("bundle opened"))
    assert (
        module.run(["--bundle", str(bundle), "--output", str(output), "--dataset-index", "0"]) == 2
    )


def _registered_l0() -> dict[str, object]:
    rows = [
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
    ]
    return {
        "replicates": rows,
        "observed_mean": float(np.mean([0.2] * 10)),
        "permutation_mean": float(np.mean([0.3] * 10)),
        "spectrum_mean": float(np.mean([0.25] * 10)),
    }


def test_evaluator_runs_l0_before_real_dimension_32_and_validates(
    monkeypatch, tmp_path: Path
) -> None:
    import sfora.lorentz_rider_io as module

    generator = np.random.Generator(np.random.PCG64(12))
    train = np.ascontiguousarray(generator.standard_normal((2000, 32)).astype(np.float32))
    query = np.ascontiguousarray(generator.standard_normal((4, 32)).astype(np.float32))
    gallery = np.ascontiguousarray(generator.standard_normal((8, 32)).astype(np.float32))
    train_labels = np.asarray([f"id-{index % 4}" for index in range(2000)])
    query_labels = np.asarray([f"id-{index}" for index in range(4)])
    gallery_labels = np.asarray([f"id-{index % 4}" for index in range(8)])
    bundle = EmbeddingBundle(
        path=tmp_path / "bundle.npz",
        sha256="a" * 64,
        metadata={"schema_version": 1},
        train_embeddings=train,
        train_labels=train_labels,
        query_embeddings=query,
        query_labels=query_labels,
        gallery_embeddings=gallery,
        gallery_labels=gallery_labels,
    )
    events: list[str] = []
    real_dimension = module._evaluate_dimension

    def fake_l0(_train, *, dataset_index):
        assert dataset_index == 0
        events.append("l0")
        return _registered_l0()

    def observed_dimension(value, dimension):
        events.append(f"d{dimension}")
        return real_dimension(value, dimension)

    monkeypatch.setattr(module, "_evaluate_l0", fake_l0)
    monkeypatch.setattr(module, "_evaluate_dimension", observed_dimension)
    report = evaluate_l0_l1(bundle, dataset_index=0)
    validate_lorentz_report(report)
    assert events[0:2] == ["l0", "d32"]
    assert report["l1"]["dimensions"][0]["dimension"] == 32


def test_cli_success_orders_load_evaluate_publish(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    bundle_path = tmp_path / "bundle.npz"
    bundle_path.write_bytes(b"bundle")
    output = tmp_path / "report.json"
    events: list[object] = []
    bundle = object()
    report = {"report": True}

    def load(path):
        events.append(("load", path))
        return bundle

    def evaluate(value, *, dataset_index):
        events.append(("evaluate", value, dataset_index))
        return report

    def publish(path, value):
        events.append(("publish", path, value))

    monkeypatch.setattr(module, "load_embedding_bundle", load)
    monkeypatch.setattr(module, "evaluate_l0_l1", evaluate)
    monkeypatch.setattr(module, "publish_lorentz_report", publish)
    assert (
        module.run(
            [
                "--bundle",
                str(bundle_path),
                "--output",
                str(output),
                "--dataset-index",
                "2",
            ]
        )
        == 0
    )
    assert events == [
        ("load", bundle_path),
        ("evaluate", bundle, 2),
        ("publish", output, report),
    ]


def test_identical_archive_scoring_excludes_only_same_row() -> None:
    labels = np.asarray(["a", "a", "b"])
    chunks = [
        np.asarray(
            [
                [10.0, 9.0, 0.0],
                [9.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np.float32,
        )
    ]
    view = _score_view(chunks, labels, labels, exclude_same_row=True)
    assert view.top1_indices.tolist() == [1, 0, 0]
    assert view.top1_correct.tolist() == [True, True, False]


def test_norm_probe_chunks_top_two_and_excludes_identical_self(monkeypatch) -> None:
    import sfora.lorentz_rider_io as module

    generator = np.random.Generator(np.random.PCG64(4))
    values = np.ascontiguousarray(generator.normal(size=(513, 32)).astype(np.float32))
    labels = np.asarray([f"id-{index % 64}" for index in range(513)])
    shapes: list[tuple[int, ...]] = []
    real_partition = np.partition

    def observed_partition(array, kth, *, axis):
        shapes.append(array.shape)
        return real_partition(array, kth, axis=axis)

    monkeypatch.setattr(module.np, "partition", observed_partition)
    probe = _norm_probe(labels, labels, values, values.copy())
    assert probe["dimension"] == 32
    assert shapes == [(256, 513), (256, 513), (1, 513)]


def test_norm_probe_identical_archive_margin_omits_diagonal(monkeypatch) -> None:
    import sfora.lorentz_rider_io as module

    values = np.ascontiguousarray(np.eye(4, dtype=np.float32))
    labels = np.asarray(["a", "a", "b", "b"])
    observed: list[np.ndarray] = []

    def capture(_left, right):
        observed.append(right.copy())
        return 0.0

    monkeypatch.setattr(module, "_spearman", capture)
    _norm_probe(labels, labels, values, values.copy())
    assert observed[1].tolist() == [0.0, 0.0, 0.0, 0.0]
