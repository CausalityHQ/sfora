from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sfora.nested_neighborhood_rank import (
    NestedRankConfig,
    NestedRankHead,
    asymmetric_neighborhood_loss,
    nested_proxy_anchor_loss,
)
from sfora.nested_rank_protocol import ordered_training_records_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_sop_nested_neighborhood_rank.py"
SPEC = importlib.util.spec_from_file_location("train_sop_nested_neighborhood_rank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot(path: Path, *, bad_digest: bool = False) -> str:
    arrays = {
        "train_embeddings": np.arange(48, dtype=np.float32).reshape(6, 8) + 1,
        "train_labels": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        "train_image_ids": np.arange(10, 16, dtype=np.int64),
        "train_relative_paths": np.asarray([f"train/{index}.jpg" for index in range(6)]),
    }
    train_digests = {name: _sha256(value.tobytes(order="C")) for name, value in arrays.items()}
    if bad_digest:
        train_digests["train_embeddings"] = "0" * 64
    metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": "1" * 64,
        "model_identifier": "fixture-teacher",
        "model_revision": "2" * 40,
        "checkpoint_sha256": "3" * 64,
        "embedding_dimension": 8,
        "train_rows": 6,
        "train_classes": 3,
        "train_array_sha256": train_digests,
        "excluded_test_array_sha256": {
            "test_embeddings": "4" * 64,
            "test_labels": "5" * 64,
            "test_image_ids": "6" * 64,
            "test_relative_paths": "7" * 64,
        },
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(path) for path in arrays["train_relative_paths"]),
        ),
        "transform": "fixture transform",
    }
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return _sha256(payload)


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--dataset-root",
        str((tmp_path / "dataset").resolve()),
        "--train-snapshot",
        str((tmp_path / "snapshot.npz").resolve()),
        "--train-snapshot-sha256",
        "1" * 64,
        "--unicom-checkout",
        str((tmp_path / "unicom").resolve()),
        "--unicom-revision",
        "2" * 40,
        "--unicom-checkpoint",
        str((tmp_path / "FP16-ViT-B-16.pt").resolve()),
        "--unicom-checkpoint-sha256",
        "3" * 64,
        "--source-commit",
        "4" * 40,
        "--arm",
        "combined",
        "--split-seed",
        "17",
        "--output-dir",
        str((tmp_path / "output").resolve()),
        "--execute-nnrl",
    ]


def test_args_are_explicit_local_only_and_fail_closed(tmp_path: Path) -> None:
    args = MODULE.parse_args(_arguments(tmp_path))
    assert args.arm == "combined"
    assert args.split_seed == 17
    assert args.dataset_root.is_absolute()

    for forbidden in ("--test-archive", "--url", "--s3", "--class-names", "--early-stop"):
        with pytest.raises(SystemExit):
            MODULE.parse_args([*_arguments(tmp_path), forbidden, "value"])
    with pytest.raises(SystemExit):
        MODULE.parse_args([*_arguments(tmp_path)[:-1]])
    duplicate = _arguments(tmp_path)
    with pytest.raises(SystemExit):
        MODULE.parse_args([*duplicate, "--arm", "proxy-anchor"])
    with pytest.raises(SystemExit):
        MODULE.parse_args([*duplicate, "--execute-nnrl"])


def test_train_snapshot_authenticates_exact_schema_and_arrays(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.npz"
    digest = _snapshot(path)
    loaded = MODULE.load_train_snapshot(path, digest)
    assert loaded.embeddings.shape == (6, 8)
    assert loaded.labels.tolist() == [0, 0, 1, 1, 2, 2]
    assert loaded.image_ids.tolist() == [10, 11, 12, 13, 14, 15]
    assert tuple(loaded.relative_paths) == tuple(f"train/{index}.jpg" for index in range(6))
    assert loaded.metadata["excluded_test_array_sha256"]["test_embeddings"] == "4" * 64

    with pytest.raises(ValueError, match="snapshot digest"):
        MODULE.load_train_snapshot(path, "0" * 64)
    bad = tmp_path / "bad.npz"
    bad_digest = _snapshot(bad, bad_digest=True)
    with pytest.raises(ValueError, match="train array digest"):
        MODULE.load_train_snapshot(bad, bad_digest)


def test_snapshot_loader_rejects_symlinks_and_test_members(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.npz"
    digest = _snapshot(path)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="snapshot authority"):
        MODULE.load_train_snapshot(link, digest)

    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["test_embeddings"] = np.ones((1, 8), dtype=np.float32)
    bad = io.BytesIO()
    np.savez(bad, **values)
    path.write_bytes(bad.getvalue())
    with pytest.raises(ValueError, match="snapshot schema"):
        MODULE.load_train_snapshot(path, _sha256(path.read_bytes()))


def test_preflight_selects_smallest_nonredundant_temperature() -> None:
    teacher = np.eye(8, dtype=np.float32)
    labels = np.arange(8, dtype=np.int64)
    sample_ids = np.arange(100, 108, dtype=np.int64)
    batches = (tuple(range(8)),) * 1_000

    result = MODULE.preflight_neighborhood_temperature(
        teacher, labels, sample_ids, batches, optimization_rows=tuple(range(8))
    )

    assert result.temperature == 0.05
    assert result.redundant is False
    assert result.batch_count == 1_000
    assert result.metrics[0.05]["median_effective_support"] >= 4.0
    assert result.metrics[0.05]["p05_effective_support"] >= 2.0


def test_preflight_masks_same_sample_and_same_label_and_fails_closed() -> None:
    teacher = np.eye(8, dtype=np.float32)
    labels = np.asarray([0, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    sample_ids = np.asarray([10, 11, 12, 13, 14, 15, 16, 16], dtype=np.int64)
    batches = (tuple(range(8)),) * 1_000
    result = MODULE.preflight_neighborhood_temperature(
        teacher, labels, sample_ids, batches, optimization_rows=tuple(range(8))
    )
    assert result.metrics[0.05]["minimum_valid_keys"] == 6

    with pytest.raises(ValueError, match="preflight batch inventory"):
        MODULE.preflight_neighborhood_temperature(
            teacher[:2],
            np.asarray([0, 0], dtype=np.int64),
            sample_ids[:2],
            ((0, 1),) * 1_000,
            optimization_rows=(0, 1),
        )
    with pytest.raises(ValueError, match="preflight schedule"):
        MODULE.preflight_neighborhood_temperature(
            teacher,
            labels,
            sample_ids,
            batches[:-1],
            optimization_rows=tuple(range(8)),
        )
    with pytest.raises(ValueError, match="preflight schedule"):
        MODULE.preflight_neighborhood_temperature(
            teacher,
            labels,
            sample_ids,
            batches,
            optimization_rows=tuple(range(7)),
        )


def _arm_fixture() -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(17)
    dense = F.normalize(torch.randn(8, 768, generator=generator), dim=1).requires_grad_()
    projection = nn.Linear(768, 128, bias=False)
    torch.nn.init.normal_(projection.weight, std=0.01, generator=generator)
    projected = projection(dense)
    embeddings = {
        32: F.normalize(projected[:, :32], dim=1),
        128: F.normalize(projected, dim=1),
    }
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    sample_ids = torch.arange(8, dtype=torch.int64)
    teacher = F.normalize(torch.randn(8, 768, generator=generator), dim=1)
    proxies = torch.randn(4, 128, generator=generator, requires_grad=True)
    return dense, embeddings, labels, sample_ids, teacher, proxies


def test_arm_objectives_match_registered_equations_and_gradients() -> None:
    dense, embeddings, labels, sample_ids, teacher, proxies = _arm_fixture()
    proxy = nested_proxy_anchor_loss(
        embeddings, labels, proxies, width_weights={32: 0.25, 128: 1.0}
    )
    neighborhood = asymmetric_neighborhood_loss(
        embeddings[128], embeddings[128], teacher, labels, sample_ids, temperature=0.1
    )
    assert torch.allclose(
        MODULE.arm_loss(
            "proxy-anchor", embeddings, dense, labels, sample_ids, teacher, proxies, 0.1
        ),
        proxy,
    )
    assert torch.allclose(
        MODULE.arm_loss(
            "neighborhood", embeddings, dense, labels, sample_ids, teacher, proxies, 0.1
        ),
        neighborhood,
    )
    ignored_proxies = torch.full_like(proxies, torch.nan)
    assert torch.allclose(
        MODULE.arm_loss(
            "neighborhood",
            embeddings,
            dense,
            labels,
            sample_ids,
            teacher,
            ignored_proxies,
            0.1,
        ),
        neighborhood,
    )
    combined = MODULE.arm_loss(
        "combined", embeddings, dense, labels, sample_ids, teacher, proxies, 0.1
    )
    assert torch.allclose(combined, proxy + neighborhood)

    self_distilled = MODULE.arm_loss(
        "s2sd-768-to-128", embeddings, dense, labels, sample_ids, teacher, proxies, 0.1
    )
    self_distilled.backward()
    assert dense.grad is not None and torch.count_nonzero(dense.grad) > 0


def test_proxy_anchor_768_uses_its_trainable_head() -> None:
    torch.manual_seed(1729)
    dense = F.normalize(torch.randn(8, 768), dim=1).requires_grad_()
    head = NestedRankHead(
        NestedRankConfig(
            input_dim=768,
            hidden_dim=1024,
            output_dim=768,
            widths=(768,),
            class_count=4,
        )
    )
    embeddings = head(dense)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    proxies = nn.Parameter(torch.randn(4, 768))
    expected = nested_proxy_anchor_loss({768: embeddings[768]}, labels, proxies + 0.0)

    actual = MODULE.arm_loss(
        "proxy-anchor-768",
        embeddings,
        dense,
        labels,
        torch.arange(8, dtype=torch.int64),
        torch.randn(8, 768),
        proxies,
        0.1,
    )

    assert torch.allclose(actual, expected)
    actual.backward()
    assert head.skip.weight.grad is not None
    assert torch.count_nonzero(head.skip.weight.grad) > 0


def test_optimizer_groups_apply_exact_rates_and_decay_exclusions() -> None:
    encoder = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 8, bias=False))
    head = NestedRankHead(NestedRankConfig(input_dim=8, hidden_dim=16, class_count=3))
    proxies = nn.Parameter(torch.randn(3, 128))
    optimizer = MODULE.build_optimizer(encoder, head, proxies, fused=False)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        "encoder-decay",
        "encoder-no-decay",
        "head-decay",
        "head-no-decay",
        "proxies",
    }
    assert groups["encoder-decay"]["lr"] == 1e-5
    assert groups["head-decay"]["lr"] == groups["proxies"]["lr"] == 1e-3
    assert groups["encoder-decay"]["weight_decay"] == 1e-4
    assert groups["head-decay"]["weight_decay"] == 1e-4
    assert groups["encoder-no-decay"]["weight_decay"] == 0.0
    assert groups["head-no-decay"]["weight_decay"] == 0.0
    assert groups["proxies"]["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in (*encoder.parameters(), *head.parameters(), proxies)
    }


class _CountingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8, bias=False)
        self.norm = nn.BatchNorm1d(8)
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.norm(self.projection(images))


def test_training_epoch_uses_one_forward_and_fixed_step_inventory() -> None:
    torch.manual_seed(17)
    encoder = _CountingEncoder()
    head = NestedRankHead(NestedRankConfig(input_dim=8, hidden_dim=16, class_count=4))
    proxies = nn.Parameter(torch.randn(4, 128))
    optimizer = MODULE.build_optimizer(encoder, head, proxies, fused=False)
    batches = []
    for _ in range(2):
        batches.append(
            (
                torch.randn(8, 4),
                torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64),
                torch.arange(8, dtype=torch.int64),
                F.normalize(torch.randn(8, 16), dim=1),
            )
        )

    result = MODULE.run_training_epoch(
        encoder,
        head,
        proxies,
        batches,
        optimizer,
        arm="combined",
        temperature=0.1,
        device=torch.device("cpu"),
        fp16=False,
        expected_steps=2,
        expected_batch_size=8,
    )

    assert result["steps"] == 2
    assert result["attempted_steps"] == 2
    assert result["skipped_updates"] == 0
    assert result["mean_loss"] > 0.0
    assert encoder.calls == 2
    assert encoder.training is True
    assert encoder.norm.training is False
    assert encoder.norm.weight.grad is not None
    assert all(group["lr"] in (1e-5, 1e-3) for group in optimizer.param_groups)

    with pytest.raises(ValueError, match="step inventory"):
        MODULE.run_training_epoch(
            encoder,
            head,
            proxies,
            batches[:1],
            optimizer,
            arm="combined",
            temperature=0.1,
            device=torch.device("cpu"),
            fp16=False,
            expected_steps=2,
            expected_batch_size=8,
        )

    encoder.calls = 0
    with pytest.raises(ValueError, match="step inventory"):
        MODULE.run_training_epoch(
            encoder,
            head,
            proxies,
            (batch for batch in [*batches, batches[0]]),
            optimizer,
            arm="combined",
            temperature=0.1,
            device=torch.device("cpu"),
            fp16=False,
            expected_steps=2,
            expected_batch_size=8,
        )
    assert encoder.calls == 2


def test_training_epoch_rejects_nonfinite_gradients_before_optimizer_step() -> None:
    torch.manual_seed(18)
    encoder = _CountingEncoder()
    head = NestedRankHead(NestedRankConfig(input_dim=8, hidden_dim=16, class_count=4))
    proxies = nn.Parameter(torch.randn(4, 128))
    optimizer = MODULE.build_optimizer(encoder, head, proxies, fused=False)
    encoder.projection.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    batch = (
        torch.randn(8, 4),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64),
        torch.arange(8, dtype=torch.int64),
        F.normalize(torch.randn(8, 16), dim=1),
    )

    with pytest.raises(ValueError, match="gradient is nonfinite"):
        MODULE.run_training_epoch(
            encoder,
            head,
            proxies,
            [batch],
            optimizer,
            arm="combined",
            temperature=0.1,
            device=torch.device("cpu"),
            fp16=False,
            expected_steps=1,
            expected_batch_size=8,
        )


def test_training_record_binding_is_exact_and_ordered(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.npz"
    snapshot = MODULE.load_train_snapshot(snapshot_path, _snapshot(snapshot_path))
    bound = MODULE.bind_training_records(
        snapshot,
        np.arange(10, 16, dtype=np.int64),
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        tuple(f"train/{index}.jpg" for index in range(6)),
    )
    assert bound == {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5}

    for image_ids, labels, paths in (
        (
            np.asarray([11, 10, 12, 13, 14, 15], dtype=np.int64),
            snapshot.labels,
            tuple(snapshot.relative_paths),
        ),
        (
            snapshot.image_ids,
            np.asarray([1, 0, 1, 1, 2, 2], dtype=np.int64),
            tuple(snapshot.relative_paths),
        ),
        (
            snapshot.image_ids,
            snapshot.labels,
            ("wrong.jpg", *tuple(snapshot.relative_paths[1:])),
        ),
    ):
        with pytest.raises(ValueError, match="training record binding"):
            MODULE.bind_training_records(snapshot, image_ids, labels, paths)


def test_execution_authority_binds_source_model_and_output_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "dataset").mkdir()
    (tmp_path / "unicom").mkdir()
    checkpoint = tmp_path / "FP16-ViT-B-16.pt"
    checkpoint.write_bytes(b"student checkpoint")
    arguments = _arguments(tmp_path)
    digest_index = arguments.index("--unicom-checkpoint-sha256") + 1
    arguments[digest_index] = _sha256(checkpoint.read_bytes())
    args = MODULE.parse_args(arguments)
    args.output_dir.parent.mkdir(exist_ok=True)
    snapshot_path = tmp_path / "snapshot.npz"
    snapshot = MODULE.load_train_snapshot(snapshot_path, _snapshot(snapshot_path))
    monkeypatch.setattr(
        MODULE,
        "_git_head",
        lambda path: (
            args.source_commit if path == MODULE._REPOSITORY_ROOT else args.unicom_revision
        ),
    )
    monkeypatch.setattr(MODULE, "_git_status_porcelain", lambda path: "")

    result = MODULE.validate_execution_authority(args, snapshot)
    assert result["source_commit"] == "4" * 40
    assert result["unicom_revision"] == "2" * 40

    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.validate_execution_authority(args, snapshot)

    args.output_dir.rmdir()
    monkeypatch.setattr(MODULE, "_git_status_porcelain", lambda path: "dirty")
    with pytest.raises(ValueError, match="source revision"):
        MODULE.validate_execution_authority(args, snapshot)


def test_training_record_loader_never_opens_official_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = SimpleNamespace(
        split="train",
        image_id=1,
        label=1,
        relative_path="train/1.jpg",
        image_path=tmp_path / "train" / "1.jpg",
    )
    calls: list[tuple[Path, str]] = []

    def parse_split(root: Path, split: str) -> tuple[object, ...]:
        calls.append((root, split))
        return (record,) * 59_551

    monkeypatch.setattr(
        MODULE.importlib,
        "import_module",
        lambda name: SimpleNamespace(_parse_split=parse_split),
    )

    records = MODULE._load_sop_training_records(tmp_path)

    assert len(records) == 59_551
    assert calls == [(tmp_path, "train")]


def test_training_transform_is_retrieval_safe_and_explicit() -> None:
    transform = MODULE._build_train_transform()

    assert [type(item).__name__ for item in transform.transforms] == [
        "Resize",
        "RandomCrop",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]
    assert transform.transforms[0].size == 256
    assert transform.transforms[1].size == (224, 224)


def test_training_artifacts_are_immutable_canonical_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "run"
    state = {
        "encoder": {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        "head": {"bias": torch.arange(2, dtype=torch.float32)},
        "raw_proxies": torch.ones(3, 2, dtype=torch.float32),
    }
    authority = {
        "source_commit": "1" * 40,
        "unicom_revision": "2" * 40,
        "student_checkpoint_sha256": "3" * 64,
        "teacher_checkpoint_sha256": "4" * 64,
        "train_snapshot_sha256": "5" * 64,
    }
    result = MODULE.publish_training_artifacts(
        output,
        state,
        authority=authority,
        arm="combined",
        split_seed=17,
        temperature=0.1,
        optimization_rows=4,
        optimization_classes=2,
        history=[{"epoch": 1, "steps": 2, "mean_loss": 1.25}],
    )

    assert result["schema"] == "sfora-nnrl-sop-training-result-v1"
    assert result["claim_eligible"] is False
    assert result["status"] == "COMPLETE"
    model = output / "model.pt"
    receipt = output / "run-receipt.json"
    terminal = output / "RESULT_COMPLETE.json"
    assert result["model_artifact"]["sha256"] == _sha256(model.read_bytes())
    assert result["model_artifact"]["bytes"] == model.stat().st_size
    assert receipt.read_bytes().endswith(b"\n")
    assert terminal.read_bytes().endswith(b"\n")
    assert json.loads(terminal.read_bytes()) == result
    with pytest.raises(FileExistsError):
        MODULE.publish_training_artifacts(
            output,
            state,
            authority=authority,
            arm="combined",
            split_seed=17,
            temperature=0.1,
            optimization_rows=4,
            optimization_classes=2,
            history=[{"epoch": 1, "steps": 2, "mean_loss": 1.25}],
        )


def test_phase_one_runs_exactly_ten_epochs_with_constant_schedule() -> None:
    torch.manual_seed(65537)
    encoder = _CountingEncoder()
    head = NestedRankHead(NestedRankConfig(input_dim=8, hidden_dim=16, class_count=4))
    proxies = nn.Parameter(torch.randn(4, 128))
    batch = (
        torch.randn(8, 4),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64),
        torch.arange(8, dtype=torch.int64),
        F.normalize(torch.randn(8, 16), dim=1),
    )
    epochs = tuple((batch, batch) for _ in range(10))

    history = MODULE.run_phase_one(
        encoder,
        head,
        proxies,
        epochs,
        arm="combined",
        temperature=0.1,
        device=torch.device("cpu"),
        fp16=False,
        fused=False,
        expected_batch_size=8,
    )

    assert [row["epoch"] for row in history] == list(range(1, 11))
    assert all(row["steps"] == 2 for row in history)
    assert encoder.calls == 20
