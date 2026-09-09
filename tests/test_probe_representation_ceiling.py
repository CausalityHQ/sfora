from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.nested_rank_protocol import ordered_training_records_sha256

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts/probe_representation_ceiling.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("probe_representation_ceiling", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _synthetic_train() -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    generator = torch.Generator().manual_seed(123)
    centers = F.normalize(torch.randn((8, 4), generator=generator), dim=1)
    source_rows = []
    teacher_rows = []
    labels = []
    transform = torch.tensor(
        [[1.0, 0.2, -0.1, 0.0], [0.1, 0.9, 0.0, 0.3], [0.2, 0.0, 1.1, -0.2], [0.0, -0.1, 0.2, 1.0]],
        dtype=torch.float32,
    )
    for class_id, center in enumerate(centers, start=1):
        for member in range(3):
            noise = torch.randn(4, generator=generator) * 0.02
            source = F.normalize(center + noise, dim=0)
            teacher = F.normalize(source @ transform.T + 0.01 * member, dim=0)
            source_rows.append(source)
            teacher_rows.append(teacher)
            labels.append(class_id)
    return (
        torch.stack(source_rows).contiguous(),
        torch.stack(teacher_rows).contiguous(),
        tuple(labels),
    )


def test_train_archive_loader_never_accesses_test_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
        dtype=np.float32,
    )
    arrays = {
        "train_embeddings": train_embeddings,
        "train_labels": np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        "train_image_ids": np.arange(1, 7, dtype=np.int64),
        "train_relative_paths": np.asarray([f"train/{index}" for index in range(6)]),
        "test_embeddings": np.asarray([[9.0, 9.0]], dtype=np.float32),
        "test_labels": np.asarray([3], dtype=np.int64),
        "test_image_ids": np.asarray([7], dtype=np.int64),
        "test_relative_paths": np.asarray(["test/7"]),
    }
    metadata = {
        "array_sha256": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in arrays.items()
        },
        "checkpoint_sha256": "1" * 64,
        "embedding_dimension": 2,
        "model_identifier": "fixture-source",
        "model_revision": "2" * 40,
        "ordered_record_sha256": "3" * 64,
        "schema": "sfora-unicom-sop-embeddings-v1",
        "split_classes": {"test": 11_316, "train": 2},
        "split_counts": {"test": 60_502, "train": 6},
        "transform": "fixture",
    }
    path = tmp_path / "fixture.npz"
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original_load = _MODULE.np.load
    accessed: list[str] = []

    class RejectTestMembers:
        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.files = inner.files

        def __enter__(self) -> RejectTestMembers:
            self.inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.inner.__exit__(*args)

        def __getitem__(self, name: str) -> object:
            accessed.append(name)
            if name.startswith("test_"):
                raise AssertionError(name)
            return self.inner[name]

    monkeypatch.setattr(
        _MODULE.np,
        "load",
        lambda *args, **kwargs: RejectTestMembers(original_load(*args, **kwargs)),
    )

    _metadata, embeddings, labels, image_ids, paths = _MODULE._load_train_archive_snapshot(
        path,
        digest,
        model_identifier="fixture-source",
        expected_rows=6,
        expected_classes=2,
        dimensions=2,
    )

    assert embeddings.shape == (6, 2)
    assert labels.tolist() == [1, 1, 1, 2, 2, 2]
    assert image_ids.tolist() == list(range(1, 7))
    assert len(paths) == 6
    assert not any(name.startswith("test_") for name in accessed)


def test_train_archive_loader_accepts_strict_runtime_bound_train_only_snapshot(
    tmp_path: Path,
) -> None:
    arrays = {
        "train_embeddings": np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
            dtype=np.float32,
        ),
        "train_labels": np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        "train_image_ids": np.arange(1, 7, dtype=np.int64),
        "train_relative_paths": np.asarray([f"train/{index}" for index in range(6)]),
    }
    metadata = {
        "batch_size": 64,
        "checkpoint_sha256": "1" * 64,
        "embedding_dimension": 2,
        "excluded_test_array_sha256": {
            "test_embeddings": "4" * 64,
            "test_image_ids": "5" * 64,
            "test_labels": "6" * 64,
            "test_relative_paths": "7" * 64,
        },
        "model_identifier": "fixture-source",
        "model_revision": "2" * 40,
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(value) for value in arrays["train_relative_paths"]),
        ),
        "runtime": {
            "blas_threads": 2,
            "cpu_threads": 2,
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_sdp_enabled": False,
            "cudnn_tf32": False,
            "cudnn_version": 92000,
            "cuda_device_capability": "12.1",
            "cuda_device_name": "fixture-gpu",
            "cuda_version": "13.0",
            "deterministic_algorithms": True,
            "flash_sdp_enabled": False,
            "float32_matmul_precision": "highest",
            "math_sdp_enabled": True,
            "memory_efficient_sdp_enabled": False,
            "seed": 17,
            "torch_version": "2.12.1+cu130",
        },
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": "8" * 64,
        "train_array_sha256": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in arrays.items()
        },
        "train_classes": 2,
        "train_rows": 6,
        "transform": "fixture",
    }
    path = tmp_path / "train-only.npz"
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )

    loaded_metadata, embeddings, labels, image_ids, paths = (
        _MODULE._load_train_archive_snapshot(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            model_identifier="fixture-source",
            expected_rows=6,
            expected_classes=2,
            dimensions=2,
        )
    )

    assert loaded_metadata == metadata
    assert embeddings.shape == (6, 2)
    assert labels.tolist() == [1, 1, 1, 2, 2, 2]
    assert image_ids.tolist() == list(range(1, 7))
    assert tuple(paths) == tuple(f"train/{index}" for index in range(6))


def test_train_only_ceiling_emits_all_fixed_arms_and_partitions() -> None:
    source, teacher, labels = _synthetic_train()

    receipt = _MODULE.run_representation_ceiling_train_only(
        source,
        teacher,
        labels,
        dimensions=2,
        outer_split_seeds=(17, 1729),
        ridge_penalties=(1e-6, 1e-4, 1e-2),
        bootstrap_samples=100,
    )

    assert receipt["schema"] == "sfora-representation-ceiling-v1"
    assert receipt["claim_eligible"] is False
    assert receipt["outer_split_seeds"] == [17, 1729]
    assert receipt["dimensions"] == 2
    assert len(receipt["splits"]) == 2
    expected_arms = {
        "source-full",
        "teacher-full",
        "teacher-pca128",
        "ridge-source-teacher-full",
        "ridge-source-teacher-pca128",
        "source-pca128",
    }
    for split in receipt["splits"]:
        assert set(split["arms"]) == expected_arms
        assert set(split["fit_class_ids"]).isdisjoint(split["validation_class_ids"])
        assert split["selected_ridge_penalty"] in (1e-6, 1e-4, 1e-2)
        assert set(split["transform_sha256"]) == {
            "ridge",
            "ridge-pca",
            "source-pca",
            "teacher-pca",
        }
        assert all(len(value) == 64 for value in split["transform_sha256"].values())
        for score in split["arms"].values():
            assert 0.0 <= score["map_at_r"] <= 1.0
            assert 0.0 <= score["r1"] <= 1.0
            assert len(score["per_query_ap"]) == len(split["validation_row_indexes"])
    assert set(receipt["decisions"]) == {
        "backbone_quality_work_warranted",
        "neighborhood_sampling_warranted",
        "wider_code_warranted",
        "width_outcome",
    }
    assert receipt["bootstrap"]["cluster_identity"] == "original-class-id-across-splits"
    assert isinstance(receipt["teacher_pca_lower_bound"], float)
    assert "test" not in json.dumps(receipt)


def test_train_only_ceiling_pins_schedule_cpu_threads_before_each_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, teacher, labels = _synthetic_train()
    original_fit = _MODULE.fit_centered_pca
    observed_threads: list[int] = []

    def observe_fit(value: torch.Tensor, *, dimensions: int) -> object:
        observed_threads.append(torch.get_num_threads())
        return original_fit(value, dimensions=dimensions)

    monkeypatch.setattr(_MODULE, "fit_centered_pca", observe_fit)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _MODULE.run_representation_ceiling_train_only(
            source,
            teacher,
            labels,
            dimensions=2,
            outer_split_seeds=(17, 1729),
            ridge_penalties=(1e-6, 1e-4, 1e-2),
            bootstrap_samples=20,
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert observed_threads == [2, 2, 2, 2, 2, 2]


def test_ridge_selection_uses_only_outer_fit_and_inner_class_disjoint_rows() -> None:
    source, teacher, labels = _synthetic_train()
    first_receipt = _MODULE.run_representation_ceiling_train_only(
        source,
        teacher,
        labels,
        dimensions=2,
        outer_split_seeds=(17,),
        ridge_penalties=(1e-6, 1e-4, 1e-2),
        bootstrap_samples=20,
    )
    changed_validation = teacher.clone()
    outer = _MODULE.deterministic_class_partition(labels, fit_fraction=0.8, seed=17)
    changed_validation[list(outer.validation_row_indexes)] *= -1.0
    second_receipt = _MODULE.run_representation_ceiling_train_only(
        source,
        changed_validation,
        labels,
        dimensions=2,
        outer_split_seeds=(17,),
        ridge_penalties=(1e-6, 1e-4, 1e-2),
        bootstrap_samples=20,
    )

    first = first_receipt["splits"][0]["inner_ridge_selection"]
    second = second_receipt["splits"][0]["inner_ridge_selection"]
    assert first == second
    assert set(first["validation_class_ids"]).isdisjoint(first["fit_class_ids"])
    assert set(first["losses"]) == {"1e-06", "0.0001", "0.01"}


def test_ceiling_decision_uses_literal_numeric_gates() -> None:
    passed = _MODULE.ceiling_decision(
        ridge_full_gain=0.006,
        ridge_full_lower_bound=0.001,
        teacher_pca_loss=-0.035,
        teacher_pca_lower_bound=-0.04,
    )
    stopped = _MODULE.ceiling_decision(
        ridge_full_gain=0.004,
        ridge_full_lower_bound=0.001,
        teacher_pca_loss=-0.005,
        teacher_pca_lower_bound=-0.008,
    )

    assert passed == {
        "backbone_quality_work_warranted": False,
        "neighborhood_sampling_warranted": True,
        "wider_code_warranted": True,
        "width_outcome": "material",
    }
    assert stopped == {
        "backbone_quality_work_warranted": True,
        "neighborhood_sampling_warranted": False,
        "wider_code_warranted": False,
        "width_outcome": "nonbinding",
    }
    with pytest.raises(ValueError, match="ceiling decision authority"):
        _MODULE.ceiling_decision(
            ridge_full_gain=float("nan"),
            ridge_full_lower_bound=0.0,
            teacher_pca_loss=0.0,
            teacher_pca_lower_bound=0.0,
        )


def test_class_cluster_lower_bound_is_deterministic_and_paired() -> None:
    treatment = (0.8, 0.7, 0.9, 0.6)
    baseline = (0.7, 0.6, 0.8, 0.5)
    identities = (11, 11, 22, 22)

    first = _MODULE.class_cluster_lower_bound(treatment, baseline, identities, seed=17, samples=100)
    second = _MODULE.class_cluster_lower_bound(
        treatment, baseline, identities, seed=17, samples=100
    )

    assert first == pytest.approx(0.1)
    assert first == second
    with pytest.raises(ValueError, match="bootstrap authority"):
        _MODULE.class_cluster_lower_bound(
            treatment, baseline[:-1], identities, seed=17, samples=100
        )


def test_class_bootstrap_reuses_original_class_identity_across_outer_splits() -> None:
    per_class_delta = (0.1,) * 7 + (-0.1,) * 3
    treatment = per_class_delta * 3
    baseline = (0.0,) * len(treatment)
    original_class_ids = tuple(range(10)) * 3
    incorrectly_split_local_ids = tuple(
        class_id + split * 10 for split in range(3) for class_id in range(10)
    )

    original_lower = _MODULE.class_cluster_lower_bound(
        treatment, baseline, original_class_ids, seed=17, samples=10_000
    )
    split_local_lower = _MODULE.class_cluster_lower_bound(
        treatment, baseline, incorrectly_split_local_ids, seed=17, samples=10_000
    )

    assert original_lower == pytest.approx(0.0)
    assert split_local_lower > 0.01


def test_cli_requires_exact_local_authority_and_execution_flag(tmp_path: Path) -> None:
    valid = [
        "--source-embeddings",
        str(tmp_path / "source.npz"),
        "--source-embeddings-sha256",
        "1" * 64,
        "--teacher-embeddings",
        str(tmp_path / "teacher.npz"),
        "--teacher-embeddings-sha256",
        "2" * 64,
        "--source-commit",
        "3" * 40,
        "--output",
        str(tmp_path / "receipt.json"),
        "--execute-representation-ceiling",
    ]

    parsed = _MODULE.parse_args(valid)
    assert parsed.execute_representation_ceiling is True
    assert parsed.output.is_absolute()
    with pytest.raises(SystemExit):
        _MODULE.parse_args(valid[:-1])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*valid, "--test-embeddings", str(tmp_path / "test.npz")])


def test_source_verifier_binds_the_loaded_repository_module_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def committed_bytes(
        command: list[str], *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        assert capture_output is True
        relative = command[-1].split(":", 1)[1]
        return subprocess.CompletedProcess(command, 0, stdout=(_ROOT / relative).read_bytes())

    monkeypatch.setattr(_MODULE.subprocess, "run", committed_bytes)
    identities = _MODULE.verify_source_commit("1" * 40)

    assert {
        "scripts/export_unicom_sop_embeddings.py",
        "scripts/probe_inshop_relational_linear.py",
        "scripts/probe_representation_ceiling.py",
        "scripts/probe_sop_relational_linear.py",
        "src/sfora/__init__.py",
        "src/sfora/joint_relational_compaction.py",
        "src/sfora/representation_ceiling.py",
    } <= set(identities)
    assert all(len(value) == 64 for value in identities.values())


def test_publication_is_canonical_no_clobber_and_cleans_owned_partial(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    receipt = {"schema": "sfora-representation-ceiling-v1", "claim_eligible": False}

    _MODULE.publish_receipt(receipt, output)

    assert output.read_bytes() == (
        b'{"claim_eligible":false,"schema":"sfora-representation-ceiling-v1"}\n'
    )
    assert not output.with_name(output.name + ".partial").exists()
    with pytest.raises(FileExistsError, match="output exists"):
        _MODULE.publish_receipt(receipt, output)


def test_main_reserves_partial_before_loading_and_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt.json"
    partial = output.with_name(output.name + ".partial")
    observed: list[bool] = []

    monkeypatch.setattr(
        _MODULE,
        "verify_source_commit",
        lambda _commit: {"scripts/probe_representation_ceiling.py": "4" * 64},
    )

    def fail_load(*_args: object) -> None:
        observed.append(not partial.exists())
        raise RuntimeError("fixture stop")

    monkeypatch.setattr(_MODULE, "load_paired_train_archives", fail_load)
    arguments = [
        "--source-embeddings",
        str(tmp_path / "source.npz"),
        "--source-embeddings-sha256",
        "1" * 64,
        "--teacher-embeddings",
        str(tmp_path / "teacher.npz"),
        "--teacher-embeddings-sha256",
        "2" * 64,
        "--source-commit",
        "3" * 40,
        "--output",
        str(output),
        "--execute-representation-ceiling",
    ]

    with pytest.raises(RuntimeError, match="fixture stop"):
        _MODULE.main(arguments)

    assert observed == [True]
    assert json.loads(output.read_bytes())["failure"]["stage"] == "archive-authentication"
    assert not partial.exists()


@pytest.mark.parametrize("replacement", ["regular", "symlink", "final"])
def test_reserved_publication_preserves_foreign_replacements(
    tmp_path: Path, replacement: str
) -> None:
    output = tmp_path / "receipt.json"
    foreign = tmp_path / "foreign"
    foreign.write_text("foreign")
    receipt = {"schema": "sfora-representation-ceiling-v1", "claim_eligible": False}
    reserved = _MODULE._reserve_output(output)
    if replacement == "regular":
        reserved.path.write_text("replacement")
    elif replacement == "symlink":
        reserved.path.symlink_to(foreign)
    else:
        output.write_text("competitor")

    try:
        if replacement == "final":
            with pytest.raises(FileExistsError):
                _MODULE._publish_reserved_receipt(receipt, output, reserved)
        else:
            _MODULE._publish_reserved_receipt(receipt, output, reserved)
    finally:
        _MODULE._cleanup_reserved_output(reserved)

    assert foreign.read_text() == "foreign"
    if replacement == "regular":
        assert reserved.path.read_text() == "replacement"
        assert json.loads(output.read_bytes()) == receipt
    elif replacement == "symlink":
        assert reserved.path.is_symlink()
        assert json.loads(output.read_bytes()) == receipt
    else:
        assert output.read_text() == "competitor"


def test_main_publishes_durable_failure_receipt_without_success_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        _MODULE,
        "verify_source_commit",
        lambda _commit: {"scripts/probe_representation_ceiling.py": "4" * 64},
    )
    monkeypatch.setattr(
        _MODULE,
        "load_paired_train_archives",
        lambda *_args: (_ for _ in ()).throw(ValueError("injected archive failure")),
    )
    arguments = [
        "--source-embeddings",
        str(tmp_path / "source.npz"),
        "--source-embeddings-sha256",
        "1" * 64,
        "--teacher-embeddings",
        str(tmp_path / "teacher.npz"),
        "--teacher-embeddings-sha256",
        "2" * 64,
        "--source-commit",
        "3" * 40,
        "--output",
        str(output),
        "--execute-representation-ceiling",
    ]

    with pytest.raises(ValueError, match="injected archive failure"):
        _MODULE.main(arguments)

    failure = json.loads(output.read_bytes())
    assert failure["schema"] == "sfora-representation-ceiling-failure-v1"
    assert failure["claim_eligible"] is False
    assert failure["failure"] == {
        "message": "injected archive failure",
        "stage": "archive-authentication",
        "type": "ValueError",
    }
    assert "decisions" not in failure
    assert output.read_bytes().endswith(b"\n")
