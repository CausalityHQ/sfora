from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn

from sfora.teacher_anchored_distillation import (
    TeacherAnchoredConfig,
    teacher_anchored_loss,
)


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
    unicom_checkout = tmp_path / "unicom-checkout"
    unicom_checkout.mkdir(exist_ok=True)
    arguments: list[str] = []
    for name, path in inputs.items():
        arguments.extend((f"--{name}", str(path.resolve()), f"--{name}-sha256", "1" * 64))
    arguments.extend(
        (
            "--image-root",
            str(image_root.resolve()),
            "--unicom-checkout",
            str(unicom_checkout.resolve()),
            "--image-tree-sha256",
            "2" * 64,
            "--source-revision",
            "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
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


def _training_schedules() -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                (
                    *range(update * 128, (update + 1) * 128),
                    *range(((update + 1) % 12) * 128, (((update + 1) % 12) + 1) * 128),
                )
            )
            for update in range(12)
        )
        for epoch in range(1, 11)
    )


def _artifact_authority(receipt: object) -> dict[str, object]:
    return {
        "anchor_schedule_sha256": "4" * 64,
        "arm": receipt.arm,
        "batch_schedule_sha256": receipt.schedule_sha256,
        "inputs_sha256": {
            "image_tree": "5" * 64,
            "source_checkpoint": "6" * 64,
            "source_snapshot": "7" * 64,
            "teacher_checkpoint": "8" * 64,
            "teacher_snapshot": "9" * 64,
        },
        "model_mode_sha256": "a" * 64,
        "objective_sha256": SUBJECT.teacher_anchored_objective_sha256(receipt.arm),
        "ridge_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "seed": 17,
        "source_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "split_sha256": "d" * 64,
        "teacher_pca_sha256": "e" * 64,
        "trainable_inventory_sha256": "f" * 64,
    }


def _run_synthetic_training(
    arm: str,
    progress: object | None = None,
) -> tuple[object, list[tuple[int, int, tuple[str, ...]]], FakeEncoder, nn.Linear]:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)
    updates: list[tuple[int, int, tuple[str, ...]]] = []

    def compute_loss(
        epoch: int,
        update_index: int,
        _rows: tuple[int, ...],
        active: tuple[str, ...],
    ) -> object:
        updates.append((epoch, update_index, active))
        parameters = tuple(
            parameter
            for parameter in (*encoder.parameters(), *head.parameters())
            if parameter.requires_grad
        )
        value = sum((parameter.square().mean() for parameter in parameters), torch.tensor(0.0))
        return SUBJECT.TeacherAnchoredLoss(
            total=value,
            anchor=value,
            point=value,
            symmetric=value,
            drift=value,
            covariance=value,
        )

    def diagnose(epoch: int) -> object:
        return SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch=epoch,
            fitting_map_at_r=0.5 + epoch / 1000,
            fitting_effective_rank=64.0,
            fitting_leading_eigenvalue_share=0.04,
            validation_map_at_r=0.4 + epoch / 1000,
        )

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm=arm,
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=64.0,
        initialization_leading_eigenvalue_share=0.04,
        compute_loss=compute_loss,
        diagnose=diagnose,
        progress=progress if callable(progress) else lambda _stage, _epoch, _update: None,
        device_type="cpu",
    )
    return receipt, updates, encoder, head


def test_cli_accepts_only_registered_local_capability(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_args(_registered_cli(tmp_path))

    assert parsed.seed == 17
    assert parsed.arm == "complete"
    assert parsed.execute_teacher_anchored is True
    assert parsed.output == (tmp_path / "result.json").resolve()
    assert parsed.source_revision == "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
    assert parsed.unicom_checkout == (tmp_path / "unicom-checkout").resolve()
    assert parsed.source_checkpoint_sha256 == "1" * 64
    assert parsed.image_tree_sha256 == "2" * 64


def test_split_boundary_maps_every_training_decision_through_fitting_rows_only() -> None:
    labels = np.repeat(np.arange(1, 801, dtype=np.int64), 2)
    image_ids = np.arange(10_000, 11_600, dtype=np.int64)

    split = SUBJECT.build_teacher_anchored_split(labels, image_ids, seed=17)
    fitting_labels = np.asarray(split.fitting_labels, dtype=np.int64)
    probe = SUBJECT.select_teacher_anchored_fitting_probe(fitting_labels, seed=17)

    assert len(split.fitting_rows) + len(split.validation_rows) == len(labels)
    assert set(split.fitting_rows).isdisjoint(split.validation_rows)
    assert set(split.fitting_class_ids).isdisjoint(split.validation_class_ids)
    assert tuple(int(labels[row]) for row in split.fitting_rows) == split.fitting_labels
    assert tuple(int(image_ids[row]) for row in split.fitting_rows) == split.fitting_image_ids
    assert set(probe.class_ids).issubset(split.fitting_class_ids)
    assert all(0 <= row < len(split.fitting_rows) for row in probe.row_indexes)
    assert len(split.sha256) == 64


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
    checkout_flag = arguments.index("--unicom-checkout")
    arguments[checkout_flag + 1] = str((tmp_path / "missing-checkout").resolve())
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


def test_update_contract_failures_are_not_misclassified_as_nonfinite() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scaler = SUBJECT.build_teacher_anchored_grad_scaler("cpu")

    with pytest.raises(ValueError, match="update authority") as wrong_loss:
        SUBJECT.teacher_anchored_optimizer_step(1.0, optimizer, scaler, (parameter,))
    assert type(wrong_loss.value) is ValueError

    detached = nn.Parameter(torch.tensor([3.0], dtype=torch.float32))
    with pytest.raises(ValueError, match="update authority") as missing_gradient:
        SUBJECT.teacher_anchored_optimizer_step(
            parameter.square().sum(), optimizer, scaler, (parameter, detached)
        )
    assert type(missing_gradient.value) is ValueError


def test_gradient_clipping_resource_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scaler = SUBJECT.build_teacher_anchored_grad_scaler("cpu")

    def out_of_memory(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", out_of_memory)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        SUBJECT.teacher_anchored_optimizer_step(
            parameter.sum() * 1e20, optimizer, scaler, (parameter,)
        )


def test_finite_gradient_norm_overflow_returns_terminal_receipt() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)

    def compute_loss(*_args: object) -> object:
        value = (head.weight.sum() + head.bias.sum()) * 1e20
        return SUBJECT.TeacherAnchoredLoss(value, value, value, value, value, value)

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="head-only",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=compute_loss,
        diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch, 0.5, 10.0, 0.1, 0.4
        ),
        progress=lambda *_args: None,
        device_type="cpu",
    )

    assert receipt.attempted_updates == 1
    assert receipt.successful_updates == 0
    assert receipt.stopped_reason == "nonfinite-update"


def test_module_state_digest_covers_named_parameters_and_buffers() -> None:
    encoder = FakeEncoder()
    baseline = SUBJECT.module_state_sha256(encoder)
    assert baseline == SUBJECT.module_state_sha256(encoder)
    with torch.no_grad():
        encoder.blocks[0][0].weight[0, 0] += 1
    assert SUBJECT.module_state_sha256(encoder) != baseline
    encoder = FakeEncoder()
    baseline = SUBJECT.module_state_sha256(encoder)
    with torch.no_grad():
        encoder.projection[0].running_mean[0] += 1
    assert SUBJECT.module_state_sha256(encoder) != baseline


def test_local_file_authentication_rejects_digest_and_symlink_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(
        SUBJECT,
        "authenticate_teacher_anchored_checkout",
        lambda checkout, revision: revision,
    )
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


def test_unicom_checkout_authentication_requires_exact_clean_git_revision(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "model.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (checkout / ".git/info/exclude").write_text("checkpoints/\n", encoding="utf-8")
    ignored_checkpoint = checkout / "checkpoints/model.pt"
    ignored_checkpoint.parent.mkdir()
    ignored_checkpoint.write_bytes(b"external checkpoint")

    assert SUBJECT.authenticate_teacher_anchored_checkout(checkout, revision) == revision
    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(checkout, "0" * 40)
    (checkout / "model.py").write_text("MODEL = 'mutated'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(checkout, revision)


def test_unicom_checkout_authentication_rejects_ambient_git_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "model.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("GIT_DIR", str(checkout / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(checkout))

    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(unrelated, revision)


def test_unicom_checkout_authentication_rejects_repository_subdirectory(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "model.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    nested = checkout / "nested"
    nested.mkdir()

    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(nested, revision)


def test_unicom_checkout_authentication_rejects_hidden_index_mutation(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    model = checkout / "model.py"
    model.write_text("MODEL = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(checkout), "update-index", "--assume-unchanged", "model.py"],
        check=True,
    )
    model.write_text("MODEL = 'mutated'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(checkout, revision)


def test_unicom_checkout_authentication_rejects_ignored_python_shadow(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    model = checkout / "model.py"
    model.write_text("MODEL = 'fixture'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (checkout / ".git/info/exclude").write_text("model/\n", encoding="utf-8")
    shadow = checkout / "model"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("MODEL = 'mutated'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkout authority differs"):
        SUBJECT.authenticate_teacher_anchored_checkout(checkout, revision)


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


def test_training_loop_runs_exact_modes_resets_and_epoch_ten_candidate() -> None:
    receipt, updates, _encoder, _head = _run_synthetic_training("complete")

    assert receipt.arm == "complete"
    assert receipt.attempted_updates == 120
    assert receipt.successful_updates == 120
    assert receipt.optimizer_reset_epochs == (1, 2)
    assert receipt.completed_epochs == tuple(range(1, 11))
    assert receipt.stopped_reason is None
    assert receipt.candidate_epoch == 10
    assert [epoch for epoch, _update, _active in updates] == [
        epoch for epoch in range(1, 11) for _update in range(12)
    ]
    assert all(
        active == ("head.weight", "head.bias") for epoch, _update, active in updates if epoch == 1
    )
    assert all(
        any(name.startswith("blocks.10.") for name in active)
        and any(name.startswith("blocks.11.") for name in active)
        and any(name.startswith("norm.") for name in active)
        for epoch, _update, active in updates
        if epoch >= 2
    )


def test_head_only_complete_control_preserves_encoder_and_resets_head_optimizer() -> None:
    encoder = FakeEncoder()
    initial = SUBJECT.module_state_sha256(encoder)
    head = nn.Linear(4, 4)

    def compute_loss(
        _epoch: int,
        _update_index: int,
        _rows: tuple[int, ...],
        _active: tuple[str, ...],
    ) -> object:
        value = head.weight.square().mean() + head.bias.square().mean()
        return SUBJECT.TeacherAnchoredLoss(value, value, value, value, value, value)

    def diagnose(epoch: int) -> object:
        return SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch=epoch,
            fitting_map_at_r=0.5 if epoch == 0 else 0.51,
            fitting_effective_rank=32.0,
            fitting_leading_eigenvalue_share=0.1,
            validation_map_at_r=0.4,
        )

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="head-only",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=32.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=compute_loss,
        diagnose=diagnose,
        progress=lambda _stage, _epoch, _update: None,
        device_type="cpu",
    )
    assert receipt.optimizer_reset_epochs == (1, 2)
    assert receipt.initial_encoder_sha256 == initial == receipt.final_encoder_sha256
    assert receipt.candidate_epoch == 10


@pytest.mark.parametrize("arm", ["head-only", "base", "anchor", "symmetric", "complete"])
def test_all_arms_consume_identical_sealed_schedules(arm: str) -> None:
    receipt, _updates, _encoder, _head = _run_synthetic_training(arm)
    assert receipt.schedule_sha256 == SUBJECT.teacher_anchored_schedule_sha256(
        _training_schedules(), fitting_row_count=1536
    )


def test_schedule_requires_every_scheduled_fitting_row_once_as_an_epoch_seed() -> None:
    schedules = _training_schedules()
    assert len(SUBJECT.teacher_anchored_schedule_sha256(schedules, fitting_row_count=1537)) == 64
    first_epoch = list(schedules[0])
    first_epoch[1] = (*first_epoch[0][:128], *first_epoch[1][128:])
    mutated = (tuple(first_epoch), *schedules[1:])
    with pytest.raises(ValueError, match="training schedule differs"):
        SUBJECT.teacher_anchored_schedule_sha256(mutated, fitting_row_count=1536)


def test_epoch_one_fitting_stop_is_binding_and_has_no_candidate() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)

    def compute_loss(
        _epoch: int,
        _update_index: int,
        _rows: tuple[int, ...],
        _active: tuple[str, ...],
    ) -> object:
        value = head.weight.square().mean() + head.bias.square().mean()
        return SUBJECT.TeacherAnchoredLoss(value, value, value, value, value, value)

    def diagnose(epoch: int) -> object:
        return SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch=epoch,
            fitting_map_at_r=0.5 if epoch == 0 else (0.497 if epoch == 1 else 0.6),
            fitting_effective_rank=10.0,
            fitting_leading_eigenvalue_share=0.1,
            validation_map_at_r=0.99,
        )

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="base",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=compute_loss,
        diagnose=diagnose,
        progress=lambda _stage, _epoch, _update: None,
        device_type="cpu",
    )
    assert receipt.completed_epochs == (1,)
    assert tuple(diagnostic.epoch for diagnostic in receipt.diagnostics) == (0, 1)
    assert receipt.diagnostics[0] == diagnose(0)
    assert receipt.stopped_reason == "epoch-one-fitting-map-regression"
    assert receipt.candidate_epoch is None


def test_adapted_arms_reject_frozen_parameter_and_buffer_drift() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)

    def compute_loss(
        epoch: int, _update: int, _rows: tuple[int, ...], _active: tuple[str, ...]
    ) -> object:
        if epoch == 2:
            with torch.no_grad():
                encoder.blocks[0][0].weight[0, 0].add_(1.0)
                encoder.projection[0].running_mean[0].add_(1.0)
        value = sum(
            (
                parameter.square().mean()
                for parameter in (*encoder.parameters(), *head.parameters())
                if parameter.requires_grad
            ),
            torch.tensor(0.0),
        )
        return SUBJECT.TeacherAnchoredLoss(value, value, value, value, value, value)

    with pytest.raises(ValueError, match="frozen state differs"):
        SUBJECT.run_teacher_anchored_training(
            encoder,
            head,
            arm="complete",
            schedules=_training_schedules(),
            fitting_row_count=1536,
            expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
                _training_schedules(), fitting_row_count=1536
            ),
            initialization_fitting_map_at_r=0.5,
            initialization_effective_rank=10.0,
            initialization_leading_eigenvalue_share=0.1,
            compute_loss=compute_loss,
            diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
                epoch, 0.5, 10.0, 0.1, 0.4
            ),
            progress=lambda _stage, _epoch, _update: None,
            device_type="cpu",
        )


def test_nonfinite_update_returns_terminal_receipt_and_counts_attempt() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)

    def compute_loss(
        epoch: int, update: int, _rows: tuple[int, ...], _active: tuple[str, ...]
    ) -> object:
        value = head.weight.square().mean() + head.bias.square().mean()
        if epoch == 3 and update == 5:
            value = value * torch.tensor(float("nan"))
        return SUBJECT.TeacherAnchoredLoss(value, value, value, value, value, value)

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="head-only",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=compute_loss,
        diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(epoch, 0.5, 10.0, 0.1, 0.4),
        progress=lambda _stage, _epoch, _update: None,
        device_type="cpu",
    )
    assert receipt.attempted_updates == 30
    assert receipt.successful_updates == 29
    assert receipt.completed_epochs == (1, 2)
    assert receipt.stopped_reason == "nonfinite-update"
    assert receipt.candidate_epoch is None


def test_real_core_forward_nonfinite_returns_terminal_receipt() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)
    student = _normalized(2, 4, seed=71).requires_grad_()
    teacher = _normalized(2, 4, seed=72)
    anchors = _normalized(2, 4, seed=73).reshape(2, 1, 4)
    anchors[0, 0, 0] = torch.nan
    adapted = _normalized(2, 4, seed=74).requires_grad_()
    original = _normalized(2, 4, seed=75)

    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="complete",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=lambda *_args: teacher_anchored_loss(
            student,
            teacher,
            anchors,
            adapted,
            original,
            TeacherAnchoredConfig(),
        ),
        diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch, 0.5, 10.0, 0.1, 0.4
        ),
        progress=lambda *_args: None,
        device_type="cpu",
    )

    assert receipt.attempted_updates == 1
    assert receipt.successful_updates == 0
    assert receipt.stopped_reason == "nonfinite-update"


def test_nonfinite_parameter_state_still_returns_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)

    def corrupting_step(*_args: object) -> float:
        with torch.no_grad():
            head.weight[0, 0] = float("nan")
        raise SUBJECT.TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")

    monkeypatch.setattr(SUBJECT, "teacher_anchored_optimizer_step", corrupting_step)
    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="head-only",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=lambda *_args: SUBJECT.TeacherAnchoredLoss(
            *(head.weight.square().mean() for _ in range(6))
        ),
        diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch, 0.5, 10.0, 0.1, 0.4
        ),
        progress=lambda *_args: None,
        device_type="cpu",
    )
    assert receipt.stopped_reason == "nonfinite-update"
    assert receipt.attempted_updates == 1
    assert receipt.successful_updates == 0
    assert len(receipt.final_head_sha256) == 64
    assert all(bool(torch.isfinite(value).all()) for value in head.state_dict().values())
    published = SUBJECT.publish_teacher_anchored_artifacts(
        tmp_path / "nonfinite-stop.json",
        encoder,
        head,
        receipt,
        authority=_artifact_authority(receipt),
    )
    assert published.receipt.is_file()
    assert published.checkpoint.is_file()


def test_nonfinite_encoder_update_restores_publishable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)
    calls = 0

    def corrupting_step(*_args: object) -> float:
        nonlocal calls
        calls += 1
        if calls == 13:
            with torch.no_grad():
                encoder.blocks[10][0].weight[0, 0] = float("nan")
            raise SUBJECT.TeacherAnchoredNonfiniteUpdate(
                "teacher-anchored nonfinite update"
            )
        return 1.0

    monkeypatch.setattr(SUBJECT, "teacher_anchored_optimizer_step", corrupting_step)
    receipt = SUBJECT.run_teacher_anchored_training(
        encoder,
        head,
        arm="complete",
        schedules=_training_schedules(),
        fitting_row_count=1536,
        expected_schedule_sha256=SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        initialization_fitting_map_at_r=0.5,
        initialization_effective_rank=10.0,
        initialization_leading_eigenvalue_share=0.1,
        compute_loss=lambda *_args: SUBJECT.TeacherAnchoredLoss(
            *(head.weight.square().mean() for _ in range(6))
        ),
        diagnose=lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch, 0.5, 10.0, 0.1, 0.4
        ),
        progress=lambda *_args: None,
        device_type="cpu",
    )

    assert receipt.stopped_reason == "nonfinite-update"
    assert receipt.attempted_updates == 13
    assert receipt.successful_updates == 12
    assert all(bool(torch.isfinite(value).all()) for value in encoder.state_dict().values())
    published = SUBJECT.publish_teacher_anchored_artifacts(
        tmp_path / "encoder-nonfinite-stop.json",
        encoder,
        head,
        receipt,
        authority=_artifact_authority(receipt),
    )
    assert published.receipt.is_file()
    assert published.checkpoint.is_file()


def test_training_does_not_misclassify_runtime_or_authority_failures() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)
    common = {
        "arm": "head-only",
        "schedules": _training_schedules(),
        "fitting_row_count": 1536,
        "expected_schedule_sha256": SUBJECT.teacher_anchored_schedule_sha256(
            _training_schedules(), fitting_row_count=1536
        ),
        "initialization_fitting_map_at_r": 0.5,
        "initialization_effective_rank": 10.0,
        "initialization_leading_eigenvalue_share": 0.1,
        "diagnose": lambda epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
            epoch, 0.5, 10.0, 0.1, 0.4
        ),
        "progress": lambda *_args: None,
        "device_type": "cpu",
    }
    for error in (RuntimeError("CUDA out of memory"), ValueError("loss authority differs")):
        with pytest.raises(type(error), match=str(error)):
            SUBJECT.run_teacher_anchored_training(
                encoder,
                head,
                compute_loss=lambda *_args, error=error: (_ for _ in ()).throw(error),
                **common,
            )
    malformed = torch.ones(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="objective authority") as objective_error:
        SUBJECT.run_teacher_anchored_training(
            encoder,
            head,
            compute_loss=lambda *_args: SUBJECT.TeacherAnchoredLoss(
                malformed,
                malformed,
                malformed,
                malformed,
                malformed,
                malformed,
            ),
            **common,
        )
    assert type(objective_error.value) is ValueError


def test_training_requires_live_step_zero_and_exact_schedule_authority() -> None:
    encoder = FakeEncoder()
    head = nn.Linear(4, 4)
    common = {
        "arm": "base",
        "schedules": _training_schedules(),
        "fitting_row_count": 1536,
        "expected_schedule_sha256": "0" * 64,
        "initialization_effective_rank": 10.0,
        "initialization_leading_eigenvalue_share": 0.1,
        "compute_loss": lambda *_args: None,
        "diagnose": lambda _epoch: None,
        "progress": lambda *_args: None,
        "device_type": "cpu",
    }
    with pytest.raises(TypeError):
        SUBJECT.run_teacher_anchored_training(encoder, head, **common)
    with pytest.raises(ValueError, match="schedule"):
        SUBJECT.run_teacher_anchored_training(
            encoder,
            head,
            initialization_fitting_map_at_r=0.5,
            **common,
        )
    common["expected_schedule_sha256"] = SUBJECT.teacher_anchored_schedule_sha256(
        _training_schedules(), fitting_row_count=1536
    )
    with pytest.raises(ValueError, match="step-zero"):
        SUBJECT.run_teacher_anchored_training(
            encoder,
            head,
            initialization_fitting_map_at_r=0.5,
            **{
                **common,
                "diagnose": lambda _epoch: SUBJECT.TeacherAnchoredEpochDiagnostic(
                    0, 0.49, 10.0, 0.1, 0.4
                ),
            },
        )


def test_probe_scorer_is_loaded_from_exact_repository_script() -> None:
    scorer, module_path = SUBJECT.load_teacher_anchored_probe_scorer()
    assert callable(scorer)
    assert module_path == (Path(__file__).parents[1] / "scripts" / "probe_sop_relational_linear.py")
    assert str(module_path.parent) in sys.path


def test_live_probe_scores_float_and_deployed_int8_with_exact_candidate_width() -> None:
    codes = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=torch.float32
    )
    codes = torch.nn.functional.normalize(codes, dim=1).contiguous()
    score = SUBJECT.score_teacher_anchored_probe(
        codes,
        (1, 1, 2, 2),
        device=torch.device("cpu"),
    )
    assert score.candidate_width == 1
    assert score.float_map_at_r == 1.0
    assert score.packed_map_at_r == 1.0
    assert score.float_r1 == 1.0
    assert score.packed_r1 == 1.0


def test_fitting_probe_selects_all_rows_of_first_512_hashed_classes() -> None:
    labels = np.repeat(np.arange(-7, 506, dtype=np.int64), 2)
    selection = SUBJECT.select_teacher_anchored_fitting_probe(labels, seed=17)
    expected_classes = tuple(
        sorted(
            (int(value) for value in np.unique(labels)),
            key=lambda value: (
                hashlib.sha256(struct.pack("<Qq", 17, value)).digest(),
                value,
            ),
        )[:512]
    )
    assert selection.class_ids == expected_classes
    assert selection.row_indexes == tuple(
        row for row, label in enumerate(labels) if int(label) in set(expected_classes)
    )
    assert len(selection.row_indexes) == 1024
    assert len(selection.sha256) == 64


def test_fitting_probe_and_bootstrap_reject_incomplete_or_wrong_authority() -> None:
    with pytest.raises(ValueError, match="fitting probe authority"):
        SUBJECT.select_teacher_anchored_fitting_probe(np.arange(512, dtype=np.int32), seed=17)
    with pytest.raises(ValueError, match="fitting probe authority"):
        SUBJECT.select_teacher_anchored_fitting_probe(np.arange(512, dtype=np.int64), seed=18)

    treatment = (0.5, 0.7, 0.2, 0.4)
    baseline = (0.4, 0.4, 0.1, 0.3)
    identities = (1, 1, 2, 2)
    lower = 0.10000000000000002
    assert SUBJECT.teacher_anchored_bootstrap_lower_bound(
        treatment,
        baseline,
        identities,
        expected_lower_bound=lower,
    ) == lower
    with pytest.raises(ValueError, match="bootstrap replay differs"):
        SUBJECT.teacher_anchored_bootstrap_lower_bound(
            treatment,
            baseline,
            identities,
            expected_lower_bound=lower + 1e-12,
        )


def test_authenticated_three_split_bootstrap_replay_maps_original_labels() -> None:
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype="<i8")
    digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    splits = (
        SUBJECT.TeacherAnchoredBootstrapSplit(17, (2, 3), (0.5, 0.7), (0.4, 0.4)),
        SUBJECT.TeacherAnchoredBootstrapSplit(1729, (2, 3), (0.2, 0.4), (0.1, 0.3)),
        SUBJECT.TeacherAnchoredBootstrapSplit(65537, (4, 5), (0.8, 0.9), (0.7, 0.8)),
    )
    lower = SUBJECT.replay_teacher_anchored_bootstrap(
        labels,
        labels_sha256=digest,
        splits=splits,
        expected_lower_bound=0.10000000000000003,
    )
    assert lower == 0.10000000000000003
    with pytest.raises(ValueError, match="bootstrap authority differs"):
        SUBJECT.replay_teacher_anchored_bootstrap(
            labels,
            labels_sha256="0" * 64,
            splits=splits,
            expected_lower_bound=lower,
        )
    with pytest.raises(ValueError, match="bootstrap authority differs"):
        SUBJECT.replay_teacher_anchored_bootstrap(
            labels,
            labels_sha256=digest,
            splits=(splits[0]._replace(row_indexes=(0, 0)), *splits[1:]),
            expected_lower_bound=lower,
        )
    with pytest.raises(ValueError, match="bootstrap authority differs"):
        SUBJECT.replay_teacher_anchored_bootstrap(
            labels,
            labels_sha256=digest,
            splits=(splits[0]._replace(row_indexes=(0,)), *splits[1:]),
            expected_lower_bound=lower,
        )


def test_authenticated_bootstrap_allows_rows_shared_by_different_seed_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype="<i8")
    digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    splits = (
        SUBJECT.TeacherAnchoredBootstrapSplit(17, (2, 3), (0.6, 0.7), (0.4, 0.5)),
        SUBJECT.TeacherAnchoredBootstrapSplit(1729, (2, 3), (0.8, 0.9), (0.5, 0.6)),
        SUBJECT.TeacherAnchoredBootstrapSplit(65537, (4, 5), (0.4, 0.5), (0.3, 0.4)),
    )
    observed: list[tuple[object, ...]] = []

    def replay(
        treatment: tuple[float, ...],
        baseline: tuple[float, ...],
        identities: tuple[int, ...],
        *,
        expected_lower_bound: float,
    ) -> float:
        observed.extend((treatment, baseline, identities))
        assert expected_lower_bound == 0.125
        return expected_lower_bound

    monkeypatch.setattr(SUBJECT, "teacher_anchored_bootstrap_lower_bound", replay)

    assert (
        SUBJECT.replay_teacher_anchored_bootstrap(
            labels,
            labels_sha256=digest,
            splits=splits,
            expected_lower_bound=0.125,
        )
        == 0.125
    )
    assert observed == [
        (0.6, 0.7, 0.8, 0.9, 0.4, 0.5),
        (0.4, 0.5, 0.5, 0.6, 0.3, 0.4),
        (2, 2, 2, 2, 3, 3),
    ]


def test_authenticated_bootstrap_rejects_three_identical_splits() -> None:
    labels = np.asarray([11, 11, 22, 22, 33, 33], dtype="<i8")
    digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    repeated = SUBJECT.TeacherAnchoredBootstrapSplit(17, (0, 1), (0.6, 0.7), (0.4, 0.5))

    with pytest.raises(ValueError, match="bootstrap authority differs"):
        SUBJECT.replay_teacher_anchored_bootstrap(
            labels,
            labels_sha256=digest,
            splits=(
                repeated,
                repeated._replace(seed=1729),
                repeated._replace(seed=65537),
            ),
            expected_lower_bound=0.1,
        )


def test_progress_chain_accepts_only_exact_monotone_canonical_events() -> None:
    launch = "a" * 64
    first = SUBJECT.teacher_anchored_progress_bytes(
        launch_receipt_sha256=launch,
        sequence=1,
        arm="complete",
        epoch=0,
        update=0,
        previous_line_sha256="0" * 64,
        stage="initialized",
    )
    second = SUBJECT.teacher_anchored_progress_bytes(
        launch_receipt_sha256=launch,
        sequence=2,
        arm="complete",
        epoch=1,
        update=1,
        previous_line_sha256=hashlib.sha256(first).hexdigest(),
        stage="update",
    )
    state = SUBJECT.validate_teacher_anchored_progress_chain((first, second), launch)
    assert state.sequence == 2
    assert state.epoch == 1
    assert state.update == 1
    for mutated in (
        (first, first),
        (first, second.replace(launch.encode(), ("b" * 64).encode())),
        (first, second.replace(hashlib.sha256(first).hexdigest().encode(), ("f" * 64).encode())),
        (first, second[:-1]),
    ):
        with pytest.raises(ValueError, match="progress chain"):
            SUBJECT.validate_teacher_anchored_progress_chain(mutated, launch)
    invalid = json.loads(first)
    for key, value in (
        ("arm", "foreign"),
        ("epoch", -1),
        ("update", -1),
        ("stage", "quality-result"),
    ):
        mutated = {**invalid, key: value}
        raw = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with pytest.raises(ValueError, match="progress chain"):
            SUBJECT.validate_teacher_anchored_progress_chain((raw,), launch)
    invalid_transition = SUBJECT.teacher_anchored_progress_bytes(
        launch_receipt_sha256=launch,
        sequence=2,
        arm="complete",
        epoch=2,
        update=1,
        previous_line_sha256=hashlib.sha256(first).hexdigest(),
        stage="update",
    )
    with pytest.raises(ValueError, match="progress chain"):
        SUBJECT.validate_teacher_anchored_progress_chain((first, invalid_transition), launch)
    epoch_complete = SUBJECT.teacher_anchored_progress_bytes(
        launch_receipt_sha256=launch,
        sequence=3,
        arm="complete",
        epoch=1,
        update=1,
        previous_line_sha256=hashlib.sha256(second).hexdigest(),
        stage="epoch-complete",
    )
    stopped_next_epoch = SUBJECT.teacher_anchored_progress_bytes(
        launch_receipt_sha256=launch,
        sequence=4,
        arm="complete",
        epoch=2,
        update=1,
        previous_line_sha256=hashlib.sha256(epoch_complete).hexdigest(),
        stage="stopped",
    )
    terminal = SUBJECT.validate_teacher_anchored_progress_chain(
        (first, second, epoch_complete, stopped_next_epoch), launch
    )
    assert (terminal.epoch, terminal.update) == (2, 1)


def test_real_training_progress_emissions_form_a_valid_chain() -> None:
    launch = "a" * 64
    lines: list[bytes] = []

    def progress(stage: str, epoch: int, update: int) -> None:
        lines.append(
            SUBJECT.teacher_anchored_progress_bytes(
                launch_receipt_sha256=launch,
                sequence=len(lines) + 1,
                arm="complete",
                epoch=epoch,
                update=update,
                previous_line_sha256=(
                    hashlib.sha256(lines[-1]).hexdigest() if lines else "0" * 64
                ),
                stage=stage,
            )
        )

    receipt, _updates, _encoder, _head = _run_synthetic_training("complete", progress)
    terminal = SUBJECT.validate_teacher_anchored_progress_chain(tuple(lines), launch)

    assert receipt.candidate_epoch == 10
    assert (terminal.sequence, terminal.arm, terminal.epoch, terminal.update) == (
        len(lines),
        "complete",
        10,
        receipt.successful_updates,
    )


def test_canonical_receipt_and_complete_merged_checkpoint_are_no_clobber(tmp_path: Path) -> None:
    receipt, _updates, encoder, head = _run_synthetic_training("complete")
    output = tmp_path / "complete.json"
    published = SUBJECT.publish_teacher_anchored_artifacts(
        output,
        encoder,
        head,
        receipt,
        authority=_artifact_authority(receipt),
    )
    raw = output.read_bytes()
    value = json.loads(raw)
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["claim_eligible"] is False
    assert value["candidate_epoch"] == 10
    assert (
        value["checkpoint_sha256"] == hashlib.sha256(published.checkpoint.read_bytes()).hexdigest()
    )
    state = torch.load(published.checkpoint, map_location="cpu", weights_only=True)
    assert set(state) == {
        *(f"encoder.{name}" for name in encoder.state_dict()),
        *(f"head.{name}" for name in head.state_dict()),
    }
    with pytest.raises(ValueError, match="already exists"):
        SUBJECT.publish_teacher_anchored_artifacts(
            output,
            encoder,
            head,
            receipt,
            authority=_artifact_authority(receipt),
        )
    unrelated = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="model state differs"):
        SUBJECT.publish_teacher_anchored_artifacts(
            tmp_path / "unrelated.json",
            encoder,
            unrelated,
            receipt,
            authority=_artifact_authority(receipt),
        )
    with pytest.raises(ValueError, match="artifact authority differs"):
        SUBJECT.publish_teacher_anchored_artifacts(
            tmp_path / "missing-authority.json",
            encoder,
            head,
            receipt,
            authority={},
        )
    with pytest.raises(ValueError, match="path alias"):
        SUBJECT.publish_teacher_anchored_artifacts(
            tmp_path / "alias.pt",
            encoder,
            head,
            receipt,
            authority=_artifact_authority(receipt),
        )


def test_publication_does_not_clobber_a_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _updates, encoder, head = _run_synthetic_training("complete")
    output = tmp_path / "race.json"
    checkpoint = output.with_suffix(".pt")
    original_link = os.link
    calls = 0

    def racing_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            output.write_bytes(b"racing-writer\n")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        SUBJECT.publish_teacher_anchored_artifacts(
            output,
            encoder,
            head,
            receipt,
            authority=_artifact_authority(receipt),
        )
    assert output.read_bytes() == b"racing-writer\n"
    assert not checkpoint.exists()


def test_publication_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _updates, encoder, head = _run_synthetic_training("complete")
    original_fsync = os.fsync
    modes: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    SUBJECT.publish_teacher_anchored_artifacts(
        tmp_path / "durable.json",
        encoder,
        head,
        receipt,
        authority=_artifact_authority(receipt),
    )
    assert any(stat.S_ISDIR(mode) for mode in modes)


def test_publication_requires_complete_nested_authority(tmp_path: Path) -> None:
    receipt, _updates, encoder, head = _run_synthetic_training("complete")
    baseline = _artifact_authority(receipt)
    mutations = []
    for key in baseline:
        mutated = {**baseline, key: False}
        mutations.append(mutated)
    mutations.extend(
        (
            {**baseline, "extra": "0" * 64},
            {**baseline, "inputs_sha256": {}},
            {
                **baseline,
                "inputs_sha256": {**baseline["inputs_sha256"], "source_checkpoint": False},
            },
        )
    )
    for index, authority in enumerate(mutations):
        with pytest.raises(ValueError, match="artifact authority differs"):
            SUBJECT.publish_teacher_anchored_artifacts(
                tmp_path / f"invalid-{index}.json",
                encoder,
                head,
                receipt,
                authority=authority,
            )


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("base", 0.1 * 2 + 0.05 * 4 + 0.01 * 5),
        ("anchor", 1 + 0.1 * 2 + 0.05 * 4 + 0.01 * 5),
        ("symmetric", 0.1 * 2 + 0.5 * 3 + 0.05 * 4 + 0.01 * 5),
        ("complete", 1 + 0.1 * 2 + 0.5 * 3 + 0.05 * 4 + 0.01 * 5),
        ("head-only", 1 + 0.1 * 2 + 0.5 * 3 + 0.05 * 4 + 0.01 * 5),
    ],
)
def test_arm_objectives_are_exact_nested_controls(arm: str, expected: float) -> None:
    terms = SUBJECT.TeacherAnchoredLoss(
        total=torch.tensor(999.0),
        anchor=torch.tensor(1.0),
        point=torch.tensor(2.0),
        symmetric=torch.tensor(3.0),
        drift=torch.tensor(4.0),
        covariance=torch.tensor(5.0),
    )
    value = SUBJECT.teacher_anchored_arm_objective(terms, arm=arm)
    assert value.dtype == torch.float32
    assert float(value) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "value",
    (torch.ones(2, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float64)),
)
def test_arm_objective_shape_and_dtype_drift_are_authority_failures(
    value: torch.Tensor,
) -> None:
    terms = SUBJECT.TeacherAnchoredLoss(
        total=value,
        anchor=value,
        point=value,
        symmetric=value,
        drift=value,
        covariance=value,
    )

    with pytest.raises(ValueError, match="objective authority") as error:
        SUBJECT.teacher_anchored_arm_objective(terms, arm="complete")

    assert type(error.value) is ValueError
