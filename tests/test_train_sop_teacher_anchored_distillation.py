from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "train_sop_teacher_anchored_distillation.py"
    spec = importlib.util.spec_from_file_location("train_sop_teacher_anchored_distillation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()


class DropPath(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(4, 4), DropPath()) for _ in range(12))
        self.norm = nn.LayerNorm(4)
        self.projection = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))


class BatchSensitiveEncoder(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        output[:, 1] += 0.0015 if len(value) == 1 else -0.0015
        return output


def _normalized(rows: int, dimensions: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, dimensions, generator=generator, dtype=torch.float32)
    return (
        (values.double() / torch.linalg.vector_norm(values.double(), dim=1, keepdim=True))
        .float()
        .contiguous()
    )


def _parameter_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
        digest.update(value.detach().numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _registered_cli(tmp_path: Path) -> list[str]:
    inputs = {
        "source-checkpoint": tmp_path / "source.pt",
        "teacher-checkpoint": tmp_path / "teacher.pt",
        "source-snapshot": tmp_path / "source.npz",
        "teacher-snapshot": tmp_path / "teacher.npz",
    }
    for path in inputs.values():
        path.write_bytes(b"fixture")
    image_root = tmp_path / "train-images"
    image_root.mkdir(exist_ok=True)
    arguments: list[str] = []
    for name, path in inputs.items():
        arguments.extend((f"--{name}", str(path.resolve()), f"--{name}-sha256", "1" * 64))
    arguments.extend(
        (
            "--image-root",
            str(image_root.resolve()),
            "--image-tree-sha256",
            "2" * 64,
            "--source-revision",
            "3" * 40,
            "--seed",
            "17",
            "--arm",
            "complete",
            "--output",
            str((tmp_path / "result.json").resolve()),
            "--execute-teacher-anchored",
        )
    )
    return arguments


def test_cli_accepts_only_registered_local_capability(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_args(_registered_cli(tmp_path))

    assert parsed.seed == 17
    assert parsed.arm == "complete"
    assert parsed.execute_teacher_anchored is True
    assert parsed.output == (tmp_path / "result.json").resolve()
    assert parsed.source_revision == "3" * 40
    assert parsed.source_checkpoint_sha256 == "1" * 64
    assert parsed.image_tree_sha256 == "2" * 64


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("--seed", "18"), "seed"),
        (("--arm", "unknown"), "arm"),
        (("--source-checkpoint-sha256", "0"), "SHA-256"),
        (("--source-revision", "main"), "revision"),
    ],
)
def test_cli_rejects_authority_value_drift(
    tmp_path: Path, replacement: tuple[str, str], message: str
) -> None:
    arguments = _registered_cli(tmp_path)
    flag, value = replacement
    arguments[arguments.index(flag) + 1] = value
    with pytest.raises(ValueError, match=message):
        SUBJECT.parse_teacher_anchored_args(arguments)


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--official-test", "/tmp/test"),
        ("--dataset-url", "https://example.invalid/sop"),
        ("--s3-uri", "s3://fixture/key"),
        ("--class-name-file", "/tmp/classes"),
        ("--sweep", "lr"),
        ("--resume", "/tmp/checkpoint"),
        ("--learning-rate", "0.1"),
    ],
)
def test_cli_refuses_forbidden_capabilities(tmp_path: Path, forbidden: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="unsupported argument"):
        SUBJECT.parse_teacher_anchored_args([*_registered_cli(tmp_path), *forbidden])


@pytest.mark.parametrize("duplicate", [("--arm", "base"), ("--arm=base",)])
def test_cli_rejects_duplicate_and_abbreviated_flags(
    tmp_path: Path, duplicate: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="unsupported argument"):
        SUBJECT.parse_teacher_anchored_args([*_registered_cli(tmp_path), *duplicate])
    arguments = _registered_cli(tmp_path)
    arguments[arguments.index("--seed")] = "--see"
    with pytest.raises(ValueError, match="unsupported argument"):
        SUBJECT.parse_teacher_anchored_args(arguments)


def test_cli_requires_absolute_existing_inputs_no_clobber_and_execution_flag(
    tmp_path: Path,
) -> None:
    arguments = _registered_cli(tmp_path)
    source_flag = arguments.index("--source-checkpoint")
    arguments[source_flag + 1] = "relative.pt"
    with pytest.raises(ValueError, match="absolute local input"):
        SUBJECT.parse_teacher_anchored_args(arguments)

    arguments = _registered_cli(tmp_path)
    Path(arguments[arguments.index("--output") + 1]).write_bytes(b"occupied")
    with pytest.raises(ValueError, match="output already exists"):
        SUBJECT.parse_teacher_anchored_args(arguments)
    Path(arguments[arguments.index("--output") + 1]).unlink()

    arguments = _registered_cli(tmp_path)
    arguments.remove("--execute-teacher-anchored")
    with pytest.raises(ValueError, match="execution flag"):
        SUBJECT.parse_teacher_anchored_args(arguments)


def test_initializer_replays_exact_split_local_teacher_projection() -> None:
    source = _normalized(132, 128, seed=1)
    teacher = _normalized(132, 128, seed=2)
    projection = SUBJECT.fit_teacher_guided_projection(
        source, teacher, dimensions=128, penalty=1e-6
    )
    expected_digest = _parameter_sha256(
        projection.teacher_projection.mean, projection.teacher_projection.components
    )

    initialized = SUBJECT.initialize_teacher_anchored_student(
        source,
        teacher,
        expected_teacher_pca_sha256=expected_digest,
    )

    assert initialized.teacher_pca_sha256 == expected_digest
    assert initialized.head.in_features == 128
    assert initialized.head.out_features == 128
    assert torch.equal(initialized.head.weight, projection.source_projection.weight)
    assert torch.equal(initialized.head.bias, projection.source_projection.bias)
    with pytest.raises(ValueError, match="teacher PCA authority differs"):
        SUBJECT.initialize_teacher_anchored_student(
            source,
            teacher,
            expected_teacher_pca_sha256="0" * 64,
        )


def test_initializer_normalizes_raw_rows_with_float64_recipe() -> None:
    source_raw = torch.randn(132, 128, generator=torch.Generator().manual_seed(11))
    teacher_raw = torch.randn(132, 128, generator=torch.Generator().manual_seed(12))
    source_unit = (
        source_raw.double() / torch.linalg.vector_norm(source_raw.double(), dim=1)[:, None]
    )
    teacher_unit = (
        teacher_raw.double() / torch.linalg.vector_norm(teacher_raw.double(), dim=1)[:, None]
    )
    expected = SUBJECT.fit_teacher_guided_projection(
        source_unit.float().contiguous(),
        teacher_unit.float().contiguous(),
        dimensions=128,
        penalty=1e-6,
    )
    initialized = SUBJECT.initialize_teacher_anchored_student(
        source_raw,
        teacher_raw,
        expected_teacher_pca_sha256=_parameter_sha256(
            expected.teacher_projection.mean, expected.teacher_projection.components
        ),
    )
    assert torch.equal(
        initialized.projection.teacher_projection.components,
        expected.teacher_projection.components,
    )
    with pytest.raises(ValueError, match="normalization authority"):
        bad = source_unit.clone()
        bad[0].zero_()
        SUBJECT.initialize_teacher_anchored_student(
            bad.float().contiguous(),
            teacher_unit.float().contiguous(),
            expected_teacher_pca_sha256="0" * 64,
        )


def test_epoch_modes_freeze_exact_inventory_and_disable_batchnorm_droppath() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 128)

    epoch_one = SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=1, head_only=False)
    assert epoch_one == ("head.weight", "head.bias")
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    epoch_two = SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=2, head_only=False)
    assert epoch_two == (
        "blocks.10.0.weight",
        "blocks.10.0.bias",
        "blocks.11.0.weight",
        "blocks.11.0.bias",
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
    )
    assert all(
        not module.training for module in encoder.modules() if isinstance(module, nn.BatchNorm1d)
    )
    assert all(
        not module.training for module in encoder.modules() if type(module).__name__ == "DropPath"
    )

    head_only = SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=10, head_only=True)
    assert head_only == ("head.weight", "head.bias")
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_optimizer_groups_reset_and_registered_lr_schedule() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 128)
    SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=2, head_only=False)
    optimizer = SUBJECT.build_teacher_anchored_optimizer(encoder, head, epoch=2)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.state == {}
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["head-decay"]["lr"] == 1e-4
    assert groups["head-no-decay"]["weight_decay"] == 0.0
    assert groups["backbone-decay"]["lr"] == 1e-6
    assert groups["backbone-no-decay"]["weight_decay"] == 0.0
    assert any(parameter is encoder.norm.weight for parameter in groups["backbone-decay"]["params"])
    assert any(
        parameter is encoder.norm.bias for parameter in groups["backbone-no-decay"]["params"]
    )
    assert SUBJECT.teacher_anchored_lr_factor(0, total_updates=1000) == pytest.approx(0.01)
    assert SUBJECT.teacher_anchored_lr_factor(99, total_updates=1000) == pytest.approx(1.0)
    assert SUBJECT.teacher_anchored_lr_factor(999, total_updates=1000) == pytest.approx(0.0)

    SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=2, head_only=True)
    head_only = SUBJECT.build_teacher_anchored_optimizer(encoder, head, epoch=2, head_only=True)
    assert {group["group_name"] for group in head_only.param_groups} == {
        "head-decay",
        "head-no-decay",
    }


@pytest.mark.parametrize("epoch", [0, 11, True])
def test_model_authority_rejects_epoch_drift(epoch: int) -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 128)
    with pytest.raises(ValueError, match="teacher-anchored epoch authority"):
        SUBJECT.configure_teacher_anchored_epoch(encoder, head, epoch=epoch, head_only=False)


@pytest.mark.parametrize("total_updates", [100, 99, True])
def test_schedule_authority_rejects_drift(total_updates: int) -> None:
    with pytest.raises(ValueError, match="teacher-anchored schedule authority"):
        SUBJECT.teacher_anchored_lr_factor(0, total_updates=total_updates)


def test_runtime_authority_pins_determinism_precision_and_rngs() -> None:
    receipt = SUBJECT.configure_teacher_anchored_runtime(17)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    replay = SUBJECT.configure_teacher_anchored_runtime(17)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert receipt == replay
    assert first == second
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert receipt.seed == 17
    assert receipt.math_sdp_enabled is True
    assert receipt.flash_sdp_enabled is False
    assert receipt.memory_efficient_sdp_enabled is False
    assert receipt.cudnn_sdp_enabled is False


def test_runtime_authority_rejects_conflicting_cublas_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="runtime authority"):
        SUBJECT.configure_teacher_anchored_runtime(17)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="runtime authority"):
        SUBJECT.configure_teacher_anchored_runtime(18)


def test_grad_scaler_and_update_are_exact_and_fail_closed() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scaler = SUBJECT.build_teacher_anchored_grad_scaler("cpu")

    norm = SUBJECT.teacher_anchored_optimizer_step(
        parameter.square().sum(), optimizer, scaler, (parameter,)
    )

    assert math.isfinite(norm)
    assert scaler.get_scale() == 1024.0
    assert optimizer.state[parameter]["step"] == 1
    before = parameter.detach().clone()
    with pytest.raises(ValueError, match="nonfinite update"):
        SUBJECT.teacher_anchored_optimizer_step(
            parameter.sum() * torch.tensor(float("nan")), optimizer, scaler, (parameter,)
        )
    assert torch.equal(parameter, before)


def test_module_state_digest_covers_named_parameters_and_buffers() -> None:
    encoder = FakeEncoder()
    baseline = SUBJECT.module_state_sha256(encoder)
    assert baseline == SUBJECT.module_state_sha256(encoder)
    with torch.no_grad():
        encoder.blocks[0][0].weight[0, 0] += 1
    assert SUBJECT.module_state_sha256(encoder) != baseline


def test_local_file_authentication_rejects_digest_and_symlink_drift(tmp_path: Path) -> None:
    arguments = _registered_cli(tmp_path)
    expected = hashlib.sha256(b"fixture").hexdigest()
    for flag in (
        "--source-checkpoint-sha256",
        "--teacher-checkpoint-sha256",
        "--source-snapshot-sha256",
        "--teacher-snapshot-sha256",
    ):
        arguments[arguments.index(flag) + 1] = expected
    parsed = SUBJECT.parse_teacher_anchored_args(arguments)
    receipt = SUBJECT.authenticate_teacher_anchored_files(parsed)
    assert receipt == (expected, expected, expected, expected)

    Path(parsed.source_checkpoint).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="input digest"):
        SUBJECT.authenticate_teacher_anchored_files(parsed)

    target = tmp_path / "target.pt"
    target.write_bytes(b"fixture")
    parsed.source_checkpoint.unlink()
    parsed.source_checkpoint.symlink_to(target)
    with pytest.raises(ValueError, match="regular local file"):
        SUBJECT.authenticate_teacher_anchored_files(parsed)


def test_initialized_head_replays_float64_map_with_registered_chunking() -> None:
    source = _normalized(132, 128, seed=31)
    teacher = _normalized(132, 128, seed=32)
    projection = SUBJECT.fit_teacher_guided_projection(
        source, teacher, dimensions=128, penalty=1e-6
    )
    initialized = SUBJECT.initialize_teacher_anchored_student(
        source,
        teacher,
        expected_teacher_pca_sha256=_parameter_sha256(
            projection.teacher_projection.mean, projection.teacher_projection.components
        ),
    )

    receipt = SUBJECT.validate_teacher_anchored_head_replay(initialized, source, device="cpu")

    assert receipt.rows == 132
    assert receipt.chunk_rows == 256
    assert receipt.maximum_absolute_error <= 1e-5
    assert receipt.minimum_cosine >= 1 - 1e-7
    assert receipt.device_type == "cpu"
    assert len(receipt.input_sha256) == 64
    assert len(receipt.head_state_sha256) == 64
    assert len(receipt.runtime_codes_sha256) == 64
    assert len(receipt.reference_codes_sha256) == 64
    assert (
        SUBJECT.validate_teacher_anchored_head_replay(
            initialized,
            source,
            device="cpu",
            expected_receipt=receipt,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="receipt differs"):
        SUBJECT.validate_teacher_anchored_head_replay(
            initialized,
            source,
            device="cpu",
            expected_receipt=receipt._replace(runtime_codes_sha256="0" * 64),
        )


def test_snapshot_feature_replay_binds_images_models_and_batch_shape() -> None:
    pristine = nn.Linear(4, 4, bias=False)
    training = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        pristine.weight.copy_(torch.eye(4))
        training.weight.copy_(torch.eye(4))
    images = _normalized(5, 4, seed=41)
    expected = images.clone()

    receipt = SUBJECT.validate_teacher_anchored_snapshot_feature_replay(
        pristine,
        training,
        images,
        expected,
        image_ids=(11, 12, 13, 14, 15),
        device="cpu",
    )

    assert receipt.rows == 5
    assert (
        receipt.image_ids_sha256
        == hashlib.sha256(struct.pack("<5q", 11, 12, 13, 14, 15)).hexdigest()
    )
    assert receipt.maximum_single_error <= 0.002
    assert receipt.maximum_batch_error <= 0.002
    assert receipt.minimum_single_cosine >= 1 - 1e-5
    assert receipt.minimum_batch_cosine >= 1 - 1e-5
    assert receipt.maximum_shape_error <= 0.002
    assert receipt.minimum_shape_cosine >= 1 - 1e-5
    with torch.no_grad():
        training.weight[0, 0] = -1
    with pytest.raises(ValueError, match="snapshot feature replay differs"):
        SUBJECT.validate_teacher_anchored_snapshot_feature_replay(
            pristine,
            training,
            images,
            expected,
            image_ids=(11, 12, 13, 14, 15),
            device="cpu",
        )


def test_snapshot_feature_replay_directly_rejects_batch_shape_drift() -> None:
    images = torch.zeros((4, 4), dtype=torch.float32)
    images[:, 0] = 1.0
    with pytest.raises(ValueError, match="snapshot feature replay differs"):
        SUBJECT.validate_teacher_anchored_snapshot_feature_replay(
            BatchSensitiveEncoder(),
            BatchSensitiveEncoder(),
            images,
            images.clone(),
            image_ids=(1, 2, 3, 4),
            device="cpu",
        )
