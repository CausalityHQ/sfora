from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_training import sharded_mask_arcface_logits

SCRIPT = Path(__file__).parents[1] / "scripts/train_unicom_inshop.py"
INITIALIZATION_KEYS = (
    "schema_version",
    "seed",
    "classifier_init",
    "trainer_sha256",
    "algorithm",
    "classifier_tensor_sha256",
    "classifier_shape",
    "classifier_dtype",
    "optimizer_steps_per_epoch",
    "initialization_seconds",
    "post_initialization_rng",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("train_unicom_inshop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + pickle.dumps(value, protocol=5)).hexdigest()


def test_registered_source_commit_comes_from_config_only_parent(tmp_path: Path) -> None:
    module = _load_script()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", checkout], check=True)
    subprocess.run(["git", "-C", checkout, "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", checkout, "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    (checkout / "source.txt").write_text("reviewed source\n", encoding="utf-8")
    subprocess.run(["git", "-C", checkout, "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", checkout, "commit", "-qm", "source"], check=True)
    source_commit = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_path = checkout / "docs" / "unicom_full_width_objective_run_config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "source": {"commit": source_commit},
                "handoff": {
                    "config_parent": source_commit,
                    "config_commit_paths": [
                        "docs/unicom_full_width_objective_run_config.json"
                    ],
                    "execution_checkout": "config_commit_detached_clean",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", checkout, "add", str(config_path)], check=True)
    subprocess.run(["git", "-C", checkout, "commit", "-qm", "config"], check=True)
    config_commit = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", checkout, "checkout", "-q", "--detach", config_commit], check=True)

    assert module.registered_source_commit(config_path, checkout) == source_commit

    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="config-only handoff differs"):
        module.registered_source_commit(config_path, checkout)


def test_training_run_receipt_binds_widths_costs_and_checkpoint_bytes(tmp_path: Path) -> None:
    module = _load_script()
    history = tmp_path / "history.json"
    history.write_bytes(b"[]\n")
    checkpoints = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"checkpoint-{epoch}".encode())
        checkpoints.append(path)

    receipt = module.training_run_receipt(
        source_commit="a" * 40,
        config_path="docs/unicom_full_width_objective_run_config.json",
        config_sha256="b" * 64,
        seed=0,
        arm="sampled_512",
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=768,
        command=[".venv/bin/python", "-I", "-B", "scripts/train_unicom_inshop.py"],
        started_unix_ns=100,
        finished_unix_ns=200,
        elapsed_seconds=1.25,
        peak_allocated_bytes=123,
        peak_reserved_bytes=456,
        exit_status=0,
        history_path=history,
        checkpoint_paths=tuple(checkpoints),
        runtime={"python": "3.12.3", "torch": "2.12.1", "cuda": "13.0"},
    )

    module.validate_training_run_receipt(receipt)
    assert receipt["protocol"] == {
        "objective": "official-eight-mask",
        "selected_features": 512,
        "evaluation_features": 768,
    }
    assert receipt["history"]["sha256"] == hashlib.sha256(b"[]\n").hexdigest()
    assert [row["epoch"] for row in receipt["checkpoints"]] == [4, 8, 12, 16]
    assert [row["bytes"] for row in receipt["checkpoints"]] == [12, 12, 13, 13]
    assert receipt["exit_status"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", True),
        ("arm", "full_768"),
        ("elapsed_seconds", float("nan")),
        ("peak_allocated_bytes", -1),
        ("checkpoints", []),
    ),
)
def test_training_run_receipt_rejects_relational_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    module = _load_script()
    history = tmp_path / "history.json"
    history.write_bytes(b"[]\n")
    checkpoints = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"checkpoint-{epoch}".encode())
        checkpoints.append(path)
    receipt = module.training_run_receipt(
        source_commit="a" * 40,
        config_path="docs/unicom_full_width_objective_run_config.json",
        config_sha256="b" * 64,
        seed=0,
        arm="sampled_512",
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=768,
        command=["python"],
        started_unix_ns=100,
        finished_unix_ns=200,
        elapsed_seconds=1.25,
        peak_allocated_bytes=123,
        peak_reserved_bytes=456,
        exit_status=0,
        history_path=history,
        checkpoint_paths=tuple(checkpoints),
        runtime={"python": "3.12.3", "torch": "2.12.1", "cuda": "13.0"},
    )
    receipt[field] = replacement

    with pytest.raises((TypeError, ValueError)):
        module.validate_training_run_receipt(receipt)


def test_main_publishes_one_training_run_receipt_after_history_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "run"
    config = tmp_path / "unicom-full-width.json"
    config.write_bytes(b'{"schema":"fixture"}\n')
    receipt = tmp_path / "sampled-512-receipt.json"

    call_order: list[str] = []

    def fake_registered_source_commit(_config: Path, _checkout: Path) -> str:
        call_order.append("source")
        return "a" * 40

    def fake_run(args) -> list[dict[str, object]]:
        assert call_order == ["source"]
        call_order.append("run")
        module.torch.cuda.reset_peak_memory_stats()
        args.output_dir.mkdir()
        for epoch in (4, 8, 12, 16):
            (args.output_dir / f"epoch-{epoch:04d}.pt").write_bytes(
                f"checkpoint-{epoch}".encode()
            )
        args._training_run_measurement = {
            "started_unix_ns": 1_000_000_000,
            "finished_unix_ns": 3_000_000_000,
            "elapsed_seconds": 2.0,
            "peak_allocated_bytes": 123,
            "peak_reserved_bytes": 456,
        }
        return [{"epoch": 16, "train": {"steps": 1, "mean_loss": 1.0}}]

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(
        module, "registered_source_commit", fake_registered_source_commit
    )
    monkeypatch.setattr(
        module,
        "_git_revision",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("main must bind the reviewed source parent")
        ),
    )
    reset_calls: list[None] = []
    monkeypatch.setattr(
        module.torch.cuda,
        "reset_peak_memory_stats",
        lambda: reset_calls.append(None),
    )
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(module.torch.version, "cuda", "13.0")

    exit_code = module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(output),
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
            "--seed",
            "0",
            "--epochs",
            "16",
            "--classifier-init",
            "imprinted",
            "--run-config",
            str(config),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(receipt),
        ]
    )

    assert exit_code == 0
    assert call_order == ["source", "run"]
    assert reset_calls == [None]
    persisted = module.strict_json_object(receipt.read_bytes())
    module.validate_training_run_receipt(persisted)
    assert persisted["source_commit"] == "a" * 40
    assert persisted["config_path"] == str(config)
    assert persisted["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert persisted["seed"] == 0
    assert persisted["arm"] == "sampled_512"
    assert persisted["started_unix_ns"] == 1_000_000_000
    assert persisted["finished_unix_ns"] == 3_000_000_000
    assert persisted["elapsed_seconds"] == 2.0
    assert persisted["exit_status"] == 0
    assert persisted["peak_allocated_bytes"] == 123
    assert persisted["peak_reserved_bytes"] == 456
    assert persisted["history"]["sha256"] == hashlib.sha256(
        (output / "history.json").read_bytes()
    ).hexdigest()

    before = receipt.read_bytes()
    assert module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(output),
            "--epochs",
            "16",
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
            "--classifier-init",
            "imprinted",
            "--run-config",
            str(config),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(receipt),
        ]
    ) == 2
    assert receipt.read_bytes() == before


def test_main_authenticates_registered_source_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    receipt = tmp_path / "receipt.json"
    config = tmp_path / "config.json"
    config.write_bytes(b"{}\n")
    run_called = False

    def fail_authentication(_config: Path, _checkout: Path) -> str:
        raise ValueError("config-only handoff differs")

    def forbidden_run(_args) -> list[dict[str, object]]:
        nonlocal run_called
        run_called = True
        raise AssertionError("training must not start before source authentication")

    monkeypatch.setattr(module, "registered_source_commit", fail_authentication)
    monkeypatch.setattr(module, "run", forbidden_run)

    assert module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "16",
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
            "--classifier-init",
            "imprinted",
            "--run-config",
            str(config),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(receipt),
        ]
    ) == 2
    assert not run_called
    assert not receipt.exists()


def test_main_failure_never_publishes_training_run_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    receipt = tmp_path / "receipt.json"
    config = tmp_path / "config.json"
    config.write_bytes(b"{}\n")
    monkeypatch.setattr(module, "run", lambda _args: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)

    assert module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(tmp_path / "output"),
            "--epochs",
            "16",
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
            "--classifier-init",
            "imprinted",
            "--run-config",
            str(config),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(receipt),
        ]
    ) == 2
    assert not receipt.exists()


def test_registered_run_rejects_a_partial_output_directory_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "run"
    output.mkdir()
    (output / "epoch-0004.pt").write_bytes(b"partial-first-attempt")
    config = tmp_path / "config.json"
    config.write_bytes(b"{}\n")
    receipt = tmp_path / "receipt.json"
    called = False

    def fake_run(_args):
        nonlocal called
        called = True
        raise AssertionError("training must not start")

    monkeypatch.setattr(module, "run", fake_run)

    assert module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(output),
            "--epochs",
            "16",
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
            "--classifier-init",
            "imprinted",
            "--run-config",
            str(config),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(receipt),
        ]
    ) == 2
    assert called is False
    assert (output / "epoch-0004.pt").read_bytes() == b"partial-first-attempt"
    assert not receipt.exists()


def test_full_width_protocol_change_preserves_checkpoint_byte_count(
    tmp_path: Path,
) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    labels = {f"item_{index:04d}": index for index in range(3_200)}
    sampled_shape = module.classifier_shape_for_run(
        labels,
        record_initialization=False,
        selected_features=512,
        evaluation_features=768,
    )
    full_shape = module.classifier_shape_for_run(
        labels,
        record_initialization=False,
        selected_features=768,
        evaluation_features=768,
    )
    assert sampled_shape == full_shape == [3_200, 768]
    classifier = torch.nn.Parameter(torch.randn(*sampled_shape))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    common = {
        "epoch": 4,
        "raw_model": raw_model,
        "classifier": classifier,
        "optimizer": optimizer,
        "scheduler": None,
        "scaler": None,
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "history": [],
    }
    sampled = tmp_path / "sampled.pt"
    full = tmp_path / "full.pt"
    sampled_generator = torch.Generator().manual_seed(7)
    full_generator = torch.Generator().manual_seed(7)

    module.save_training_checkpoint(
        sampled,
        mask_generator=sampled_generator,
        training_protocol={
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        **common,
    )
    module.save_training_checkpoint(
        full,
        mask_generator=full_generator,
        training_protocol={
            "objective": "official-eight-mask",
            "selected_features": 768,
            "evaluation_features": 768,
        },
        **common,
    )

    assert sampled.stat().st_size == full.stat().st_size


def test_registered_identity_count_fails_before_output_directory_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "seed-0" / "sampled_512"
    output.parent.mkdir()
    records: tuple[object, ...] = ()
    wrong_labels = {f"item_{index:04d}": index for index in range(3_199)}
    monkeypatch.setattr(module, "_git_revision", lambda _path: module.UNICOM_REVISION)
    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda _path: module.UNICOM_L14_336_SHA256,
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "_seed_process", lambda _seed: None)
    monkeypatch.setattr(
        sys.modules["sfora.unicom_inshop"],
        "parse_inshop_partition",
        lambda _root: records,
    )
    monkeypatch.setattr(
        module,
        "identity_holdout",
        lambda *_args, **_kwargs: ((), (), (), wrong_labels),
    )
    args = module.parse_args(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "FP16-ViT-L-14-336px.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(output),
            "--evaluation-features",
            "768",
            "--run-config",
            str(tmp_path / "config.json"),
            "--run-arm",
            "sampled_512",
            "--run-receipt",
            str(tmp_path / "receipt.json"),
        ]
    )

    with pytest.raises(ValueError, match="registered full-width class count"):
        module.run(args)
    assert not output.exists()


def test_initialization_receipt_binds_exact_classifier_bytes_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing classifier bytes or consuming RNG must change/falsify the receipt."""
    module = _load_script()
    classifier = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cuda_states = [torch.tensor([5, 6, 7], dtype=torch.uint8)]
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_rng_state_all", lambda: cuda_states)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=classifier,
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.25,
        trainer_sha256="a" * 64,
    )

    assert tuple(receipt) == INITIALIZATION_KEYS
    assert receipt["classifier_tensor_sha256"] == hashlib.sha256(
        classifier.numpy().tobytes(order="C")
    ).hexdigest()
    assert receipt["classifier_shape"] == [3, 4]
    assert receipt["classifier_dtype"] == "torch.float32"
    assert receipt["post_initialization_rng"] == {
        "python_sha256": _state_digest(b"python-random-v1", python_state),
        "numpy_sha256": _state_digest(b"numpy-random-v1", numpy_state),
        "torch_cpu_sha256": hashlib.sha256(
            b"torch-cpu-random-v1\0" + bytes(torch_state.tolist())
        ).hexdigest(),
        "torch_cuda_sha256_by_device": [
            hashlib.sha256(b"torch-cuda-random-v1:0\0" + bytes(cuda_states[0].tolist())).hexdigest()
        ],
    }
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    module.validate_initialization_receipt(receipt, expected_shape=[3, 4])


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", True),
        ("classifier_shape", [3200, 767]),
        ("classifier_dtype", "float32"),
        ("optimizer_steps_per_epoch", 0),
        ("initialization_seconds", float("nan")),
    ),
)
def test_initialization_receipt_rejects_schema_or_scalar_drift(
    field: str, replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed receipt scalars must fail before training can use their evidence."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3200, 768),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    receipt[field] = replacement
    with pytest.raises((TypeError, ValueError)):
        module.validate_initialization_receipt(receipt, expected_shape=[3200, 768])


def test_initialization_receipt_atomic_publication_reloads_and_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published receipt must be durable, mode-0600, strict, and immutable."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3, 4),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    output = tmp_path / "initialization-receipt.json"

    module.write_initialization_receipt_atomic(
        receipt, output, expected_shape=[3, 4]
    )

    persisted = module.strict_json_object(output.read_bytes())
    module.validate_initialization_receipt(persisted, expected_shape=[3, 4])
    assert persisted == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_initialization_receipt_atomic(
            receipt, output, expected_shape=[3, 4]
        )
    assert output.read_bytes() == original
    assert json.loads(original) == receipt


def test_initialization_receipt_atomic_publication_completes_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legal short write must be retried until every registered byte is durable."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3, 4),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    real_write = module.os.write

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        limit = max(1, len(payload) // 2)
        return real_write(descriptor, payload[:limit])

    monkeypatch.setattr(module.os, "write", short_write)
    output = tmp_path / "initialization-receipt.json"

    module.write_initialization_receipt_atomic(
        receipt, output, expected_shape=[3, 4]
    )

    assert module.strict_json_object(output.read_bytes()) == receipt


def test_resume_reauthenticates_initializer_bytes_without_retiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must reuse duration but reject a changed deterministic initializer."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    classifier = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = tmp_path / "initialization-receipt.json"
    fresh = module.bind_initialization_receipt(
        output=output,
        resume=False,
        seed=2,
        classifier_init="random",
        classifier=classifier,
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.25,
        trainer_sha256="a" * 64,
        expected_shape=[3, 4],
    )
    original = output.read_bytes()

    resumed = module.bind_initialization_receipt(
        output=output,
        resume=True,
        seed=2,
        classifier_init="random",
        classifier=classifier.clone(),
        optimizer_steps_per_epoch=161,
        initialization_seconds=None,
        trainer_sha256="a" * 64,
        expected_shape=[3, 4],
    )

    assert resumed == fresh
    assert resumed["initialization_seconds"] == 1.25
    assert output.read_bytes() == original
    changed = classifier.clone()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="resume initialization receipt"):
        module.bind_initialization_receipt(
            output=output,
            resume=True,
            seed=2,
            classifier_init="random",
            classifier=changed,
            optimizer_steps_per_epoch=161,
            initialization_seconds=None,
            trainer_sha256="a" * 64,
            expected_shape=[3, 4],
        )


def test_initialization_binding_rejects_wrong_registered_classifier_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production shape authority must not be inferred from a wrong live tensor."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )

    with pytest.raises(ValueError, match="classifier shape"):
        module.bind_initialization_receipt(
            output=tmp_path / "initialization-receipt.json",
            resume=False,
            seed=2,
            classifier_init="random",
            classifier=torch.ones(3, 4),
            optimizer_steps_per_epoch=161,
            initialization_seconds=1.0,
            trainer_sha256="a" * 64,
            expected_shape=[3200, 768],
        )


def test_registered_classifier_shape_is_derived_and_cross_checked() -> None:
    """The live holdout mapping, not a receipt-local shape, determines the rows."""
    module = _load_script()
    labels = {f"item_{index:04d}": index for index in range(3200)}

    assert module.registered_classifier_shape(labels) == [len(labels), 768]

    labels.pop("item_3199")
    with pytest.raises(ValueError, match="registered classifier shape"):
        module.registered_classifier_shape(labels)

    reordered = {
        "b": 1,
        "a": 0,
        **{f"item_{index:04d}": index + 2 for index in range(3198)},
    }
    with pytest.raises(ValueError, match="registered classifier shape"):
        module.registered_classifier_shape(reordered)
    with pytest.raises(ValueError, match="registered classifier shape"):
        module.registered_classifier_shape(list(range(3200)))


def test_classifier_shape_only_enforces_replication_contract_for_receipt_runs() -> None:
    """Full-train mode stays usable while prospective replication rows stay exact."""
    module = _load_script()
    full_train_labels = {f"item_{index:04d}": index for index in range(3997)}

    assert module.classifier_shape_for_run(
        full_train_labels,
        record_initialization=False,
        selected_features=512,
        evaluation_features=512,
    ) == [3997, 768]
    with pytest.raises(ValueError, match="registered classifier shape"):
        module.classifier_shape_for_run(full_train_labels, record_initialization=True)


def test_epoch_sampler_matches_padded_global_order() -> None:
    module = _load_script()
    sampler = module.PaddedEpochSampler(size=10, batch_size=8, seed=0)
    sampler.set_epoch(3)

    generator = torch.Generator().manual_seed(1_003)
    shuffled = torch.randperm(10, generator=generator).tolist()
    assert list(sampler) == (shuffled * 2)[:16]
    assert len(sampler) == 16


def test_objective_masks_bind_official_eight_shards_and_prefix_controls() -> None:
    module = _load_script()
    generator = torch.Generator().manual_seed(7)

    eight = module.objective_masks(
        "official-eight-mask",
        dimension=8,
        selected=4,
        generator=generator,
        device=torch.device("cpu"),
    )
    one = module.objective_masks(
        "official-one-mask",
        dimension=8,
        selected=4,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    prefix = module.objective_masks(
        "prefix-512",
        dimension=8,
        selected=4,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )

    assert eight.shape == (8, 4)
    assert one.shape == (1, 4)
    assert torch.equal(eight[0], one[0])
    assert torch.equal(prefix, torch.arange(4)[None])


def test_full_width_mask_state_matches_sampled_mask_state() -> None:
    module = _load_script()
    control_generator = torch.Generator().manual_seed(20_768)
    candidate_generator = torch.Generator().manual_seed(20_768)

    sampled = module.objective_masks(
        "official-eight-mask",
        dimension=768,
        selected=512,
        generator=control_generator,
        device=torch.device("cpu"),
    )
    full = module.objective_masks(
        "official-eight-mask",
        dimension=768,
        selected=768,
        generator=candidate_generator,
        device=torch.device("cpu"),
    )

    assert sampled.shape == (8, 512)
    assert full.shape == (8, 768)
    expected = torch.arange(768)
    assert all(torch.equal(torch.sort(row).values, expected) for row in full)
    assert torch.equal(control_generator.get_state(), candidate_generator.get_state())


def test_full_width_loss_matches_exact_permuted_shard_reference() -> None:
    module = _load_script()
    torch.manual_seed(768)
    embeddings = torch.randn(3, 768, dtype=torch.float32)
    weights = torch.randn(11, 768, dtype=torch.float32)
    labels = torch.tensor([0, 5, 10], dtype=torch.int64)
    masks = module.objective_masks(
        "official-eight-mask",
        dimension=768,
        selected=768,
        generator=torch.Generator().manual_seed(23),
        device=torch.device("cpu"),
    )

    actual = sharded_mask_arcface_logits(
        embeddings,
        weights,
        labels,
        masks,
        margin=0.25,
        scale=32.0,
    )
    shard_logits = []
    quotient, remainder = divmod(weights.shape[0], masks.shape[0])
    start = 0
    for shard, coordinates in enumerate(masks):
        width = quotient + int(shard < remainder)
        stop = start + width
        selected_embeddings = torch.nn.functional.normalize(
            embeddings.index_select(1, coordinates), dim=1
        )
        selected_weights = torch.nn.functional.normalize(
            weights[start:stop].index_select(1, coordinates), dim=1
        )
        shard_logits.append(torch.nn.functional.linear(selected_embeddings, selected_weights))
        start = stop
    expected = torch.cat(shard_logits, dim=1).clamp(-1.0, 1.0)
    rows = torch.arange(labels.numel())
    expected[rows, labels] = torch.cos(torch.acos(expected[rows, labels]) + 0.25)
    expected = expected * 32.0

    assert torch.equal(actual, expected)


def test_cli_defaults_match_official_336_recipe() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--unicom-checkout",
            "/tmp/unicom",
            "--checkpoint",
            "/tmp/FP16-ViT-L-14-336px.pt",
            "--dataset-root",
            "/tmp/inshop",
            "--output-dir",
            "/tmp/output",
        ]
    )

    assert args.epochs == 128
    assert args.batch_size == 128
    assert args.learning_rate == 1e-5
    assert args.classifier_learning_rate == 1e-4
    assert args.margin == 0.25
    assert args.scale == 32.0
    assert args.objective == "official-eight-mask"
    assert args.selected_features == 512
    assert args.evaluation_features is None
    assert args.workers == 4
    assert args.seed == 1024
    assert args.holdout_seed == 0
    assert args.holdout_fraction == 0.2
    assert args.eval_every == 4
    assert args.checkpoint_every == 4
    assert args.max_steps is None
    assert args.resume is None
    assert not args.bf16
    assert not args.compile
    assert not args.fused
    assert args.classifier_init == "random"


def _required_cli(tmp_path: Path) -> list[str]:
    return [
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--checkpoint",
        str(tmp_path / "FP16-ViT-L-14-336px.pt"),
        "--dataset-root",
        str(tmp_path / "inshop"),
        "--output-dir",
        str(tmp_path / "run"),
    ]


def test_fepf_cli_freezes_recipe_and_stop_boundary(tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(
        _required_cli(tmp_path)
        + [
            "--classifier-init",
            "fepf_mean",
            "--epochs",
            "16",
            "--evaluation-features",
            "512",
            "--stop-after-epoch",
            "4",
        ]
    )

    module.validate_fepf_recipe(args)

    assert args.stop_after_epoch == 4
    assert args.parent_initialization_receipt is None
    assert args.parent_run_receipt is None


@pytest.mark.parametrize(
    ("override", "message"),
    (
        (("--epochs", "15"), "recipe differs"),
        (("--stop-after-epoch", "3"), "stop boundary differs"),
        (("--stop-after-epoch", "5"), "stop boundary differs"),
        (("--stop-after-epoch", "15"), "stop boundary differs"),
        (("--stop-after-epoch", "17"), "stop boundary differs"),
        (("--bf16",), "recipe differs"),
        (("--selected-features", "768"), "recipe differs"),
        (("--evaluation-features", "768"), "recipe differs"),
        (("--objective", "official-one-mask"), "recipe differs"),
        (("--learning-rate", "2e-5"), "recipe differs"),
        (("--classifier-learning-rate", "2e-4"), "recipe differs"),
        (("--batch-size", "64"), "recipe differs"),
        (("--workers", "3"), "recipe differs"),
        (("--eval-every", "8"), "recipe differs"),
        (("--checkpoint-every", "8"), "recipe differs"),
    ),
)
def test_fepf_recipe_rejects_protocol_drift(
    tmp_path: Path, override: tuple[str, ...], message: str
) -> None:
    module = _load_script()
    arguments = (
        _required_cli(tmp_path)
        + [
            "--classifier-init",
            "fepf_random",
            "--epochs",
            "16",
            "--evaluation-features",
            "512",
        ]
        + list(override)
    )

    with pytest.raises(ValueError, match=message):
        module.validate_fepf_recipe(module.parse_args(arguments))


def test_fepf_resume_requires_authenticated_parent_receipts_and_fresh_output(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "parent" / "epoch-0004.pt"
    base = _required_cli(tmp_path) + [
        "--classifier-init",
        "fepf_mean",
        "--epochs",
        "16",
        "--evaluation-features",
        "512",
        "--resume",
        str(checkpoint),
    ]
    for suffix in (
        [],
        ["--parent-initialization-receipt", str(tmp_path / "initialization.json")],
        ["--parent-run-receipt", str(tmp_path / "run.json")],
    ):
        with pytest.raises(ValueError, match="parent receipts differ"):
            module.validate_fepf_recipe(module.parse_args(base + suffix))

    same_output = base.copy()
    same_output[same_output.index("--output-dir") + 1] = str(checkpoint.parent)
    same_output += [
        "--parent-initialization-receipt",
        str(tmp_path / "initialization.json"),
        "--parent-run-receipt",
        str(tmp_path / "run.json"),
    ]
    with pytest.raises(ValueError, match="continuation output differs"):
        module.validate_fepf_recipe(module.parse_args(same_output))


def test_evaluation_width_is_independent_and_legacy_default_is_preserved() -> None:
    module = _load_script()

    assert module.resolve_evaluation_features(512, None) == 512
    assert module.resolve_evaluation_features(512, 768) == 768
    assert module.resolve_evaluation_features(768, 768) == 768
    for value in (True, 0, 769):
        with pytest.raises((TypeError, ValueError)):
            module.resolve_evaluation_features(512, value)

    args = module.parse_args(
        [
            "--unicom-checkout",
            "/tmp/unicom",
            "--checkpoint",
            "/tmp/FP16-ViT-L-14-336px.pt",
            "--dataset-root",
            "/tmp/inshop",
            "--output-dir",
            "/tmp/output",
            "--selected-features",
            "512",
            "--evaluation-features",
            "768",
        ]
    )
    assert args.selected_features == 512
    assert args.evaluation_features == 768


def test_holdout_evaluation_uses_evaluation_width_not_training_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    query = np.arange(12, dtype=np.float32).reshape(2, 6)
    gallery = np.arange(18, dtype=np.float32).reshape(3, 6)
    labels = (np.asarray(["a", "b"]), np.asarray(["a", "b", "c"]))
    encoded = iter(((query, labels[0]), (gallery, labels[1])))
    seen: list[np.ndarray] = []

    monkeypatch.setattr(module, "_encode_records", lambda *_args, **_kwargs: next(encoded))

    def retrieval_view(
        _query,
        _gallery,
        _query_labels,
        _gallery_labels,
        *,
        coordinates,
        normalize_before,
    ):
        seen.append(coordinates.copy())
        assert normalize_before is True
        return SimpleNamespace(
            recall={1: 1.0, 10: 1.0, 20: 1.0, 30: 1.0}, map_at_r=1.0
        )

    monkeypatch.setattr(module, "retrieval_view", retrieval_view)
    result = module.evaluate_holdout(
        torch.nn.Identity(),
        (),
        (),
        lambda image: image,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
        evaluation_features=6,
    )

    assert result["map_at_r"] == 1.0
    assert len(seen) == 1
    assert np.array_equal(seen[0], np.arange(6))


def test_worker_seed_preserves_the_epoch_varying_dataloader_seed(monkeypatch) -> None:
    module = _load_script()
    python_seeds: list[int] = []
    numpy_seeds: list[int] = []
    initial_seeds = iter((2**32 + 123, 2**32 + 456))
    monkeypatch.setattr(module.torch, "initial_seed", lambda: next(initial_seeds))
    monkeypatch.setattr(
        module.torch,
        "manual_seed",
        lambda _seed: pytest.fail("worker init must not overwrite PyTorch's worker seed"),
    )
    monkeypatch.setattr(module.random, "seed", python_seeds.append)
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)

    module._seed_worker(0)
    module._seed_worker(0)

    assert python_seeds == [123, 456]
    assert numpy_seeds == [123, 456]


def test_training_loader_seed_is_epoch_derived_and_global_rng_independent() -> None:
    module = _load_script()
    loader = SimpleNamespace(generator=torch.Generator())
    global_state = torch.get_rng_state().clone()

    module._seed_training_loader(loader, seed=7, epoch=3)

    assert loader.generator.initial_seed() == module.experiment_stream_seed(7, 2_003)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_train_dataset_uses_optimization_label_mapping(tmp_path: Path) -> None:
    module = _load_script()
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image_path)
    records = (InshopRecord(split="train", image_path=image_path, label="item_b"),)

    dataset = module.InshopTrainDataset(
        records,
        {"item_b": 7},
        lambda image: torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()),
    )

    image, label = dataset[0]
    assert image.shape == (2, 3, 3)
    assert label == 7
    assert len(dataset) == 1


def test_optimizer_binds_separate_backbone_and_classifier_rates() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(3, 2)
    classifier = torch.nn.Parameter(torch.ones(4, 2))

    optimizer = module.build_optimizer(
        backbone,
        classifier,
        learning_rate=1e-5,
        classifier_learning_rate=1e-4,
        fused=False,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert optimizer.defaults["weight_decay"] == 0.0


def test_step_ema_tracks_only_parameters_on_their_live_device() -> None:
    module = _load_script()

    class BufferedLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 2, bias=False)
            self.register_buffer("running", torch.tensor([3.0], dtype=torch.float32))
            self.register_buffer("counter", torch.tensor(4, dtype=torch.int64))

    backbone = BufferedLinear()
    classifier = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    with torch.no_grad():
        backbone.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    ema = module.StepEMA(backbone, classifier)

    initial = ema.state_dict()
    assert tuple(initial) == ("decay", "updates", "backbone", "classifier")
    assert initial["decay"] == 0.999
    assert initial["updates"] == 0
    assert tuple(initial["backbone"]) == ("weight",)
    assert initial["backbone"]["weight"].device.type == "cpu"

    with torch.no_grad():
        backbone.weight.add_(10.0)
        classifier.add_(20.0)
        backbone.running.fill_(99.0)
        backbone.counter.fill_(7)
    ema.update()

    state = ema.state_dict()
    assert state["updates"] == 1
    assert torch.equal(state["backbone"]["weight"], initial["backbone"]["weight"] + 0.01)
    assert torch.equal(state["classifier"], initial["classifier"] + 0.02)
    materialized = ema.materialize_backbone_state()
    assert torch.equal(materialized["running"], torch.tensor([99.0]))
    assert torch.equal(materialized["counter"], torch.tensor(7))
    assert torch.equal(materialized["weight"], state["backbone"]["weight"])

    with torch.no_grad():
        backbone.weight.zero_()
        classifier.zero_()
    assert not torch.equal(state["backbone"]["weight"], backbone.weight)
    assert not torch.equal(state["classifier"], classifier)


def test_step_ema_optimizer_hook_runs_only_for_executed_steps() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.SGD([*backbone.parameters(), classifier], lr=0.1)
    ema = module.StepEMA(backbone, classifier)

    hook = ema.register_step_hook(optimizer)
    with pytest.raises(RuntimeError, match="already registered"):
        ema.register_step_hook(optimizer)

    optimizer.zero_grad(set_to_none=True)
    (backbone(torch.ones(1, 2)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1

    # This models GradScaler's overflow branch: it does not call optimizer.step().
    optimizer.zero_grad(set_to_none=True)
    assert ema.state_dict()["updates"] == 1

    hook.remove()
    ema.release_step_hook()
    optimizer.zero_grad(set_to_none=True)
    (backbone(torch.ones(1, 2)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1


def test_step_ema_hook_does_not_run_for_grad_scaler_overflow() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.SGD([*backbone.parameters(), classifier], lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    ema = module.StepEMA(backbone, classifier)
    ema.register_step_hook(optimizer)

    optimizer.zero_grad(set_to_none=True)
    loss = (backbone(torch.ones(1, 2)).sum() + classifier.sum()) * torch.tensor(float("inf"))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert ema.state_dict()["updates"] == 0
    ema.release_step_hook()


@pytest.mark.parametrize(
    "mutation",
    ("decay", "updates", "backbone_keys", "backbone_dtype", "classifier_shape"),
)
def test_step_ema_rejects_invalid_serialized_state(mutation: str) -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    ema = module.StepEMA(backbone, classifier)
    state = ema.state_dict()
    if mutation == "decay":
        state["decay"] = 0.9
    elif mutation == "updates":
        state["updates"] = True
    elif mutation == "backbone_keys":
        state["backbone"]["extra"] = torch.ones(1)
    elif mutation == "backbone_dtype":
        state["backbone"]["weight"] = state["backbone"]["weight"].double()
    else:
        state["classifier"] = torch.ones(2, 1)

    with pytest.raises((TypeError, ValueError)):
        ema.load_state_dict(state)


def test_imprinted_classifier_matches_independent_class_mean_formula_and_preserves_state(
    tmp_path: Path,
) -> None:
    module = _load_script()

    def record(name: str, label: str, color: tuple[int, int, int]) -> InshopRecord:
        path = tmp_path / name
        Image.new("RGB", (2, 2), color).save(path)
        return InshopRecord(split="train", image_path=path, label=label)

    records = (
        record("b0.png", "b", (0, 255, 0)),
        record("a0.png", "a", (255, 0, 0)),
        record("b1.png", "b", (0, 0, 255)),
        record("a1.png", "a", (255, 255, 0)),
    )

    class RepeatedMean(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("running", torch.tensor([5.0]))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            rgb = images.mean(dim=(2, 3)) * self.scale
            return rgb.repeat(1, 256)

    def transform(image: Image.Image) -> torch.Tensor:
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values).permute(2, 0, 1)

    model = RepeatedMean().train()
    model_before = {name: value.clone() for name, value in model.state_dict().items()}
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    values = module.imprinted_classifier_values(
        model,
        records,
        {"a": 0, "b": 1},
        transform,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
    )

    per_record = []
    for row in records:
        with Image.open(row.image_path) as image:
            embedding = model(transform(image.convert("RGB"))[None])[0]
        per_record.append(embedding / torch.linalg.vector_norm(embedding))
    expected_rows = []
    for label in ("a", "b"):
        mean = (
            torch.stack(
                [
                    value.double()
                    for row, value in zip(records, per_record, strict=True)
                    if row.label == label
                ]
            ).sum(dim=0, dtype=torch.float64)
            / 2.0
        )
        expected_rows.append((mean / torch.linalg.vector_norm(mean)).float())
    expected = torch.stack(expected_rows) * (0.01 * np.sqrt(768.0))

    assert values.dtype == torch.float32
    assert values.device.type == "cpu"
    assert torch.equal(values, expected)
    assert model.training
    assert all(torch.equal(model.state_dict()[name], value) for name, value in model_before.items())
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    assert np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "mutation",
    ("missing_class", "noncontiguous_labels", "zero_embedding", "wrong_dimension"),
)
def test_imprinted_classifier_rejects_invalid_inputs(tmp_path: Path, mutation: str) -> None:
    module = _load_script()
    path = tmp_path / "row.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    records = (InshopRecord(split="train", image_path=path, label="a"),)
    labels = {"a": 0}

    class Output(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            dimension = 767 if mutation == "wrong_dimension" else 768
            if mutation == "zero_embedding":
                return torch.zeros(images.shape[0], dimension)
            return torch.ones(images.shape[0], dimension)

    if mutation == "missing_class":
        labels = {"a": 0, "b": 1}
    elif mutation == "noncontiguous_labels":
        labels = {"a": 1}

    with pytest.raises(ValueError):
        module.imprinted_classifier_values(
            Output(),
            records,
            labels,
            lambda _image: torch.ones(3, 2, 2),
            device=torch.device("cpu"),
            batch_size=1,
            workers=0,
        )


def test_classifier_initialization_consumes_identical_rng_for_both_arms() -> None:
    module = _load_script()

    torch.manual_seed(37)
    random_values = module.initialize_classifier_values(
        labels=3,
        mode="random",
        imprinted=lambda: pytest.fail("random arm must not build imprints"),
    )
    random_next = torch.rand(4)

    torch.manual_seed(37)
    imprint = torch.full((3, 768), 0.25, dtype=torch.float32)
    imprinted_values = module.initialize_classifier_values(
        labels=3,
        mode="imprinted",
        imprinted=lambda: imprint,
    )
    imprinted_next = torch.rand(4)

    assert not torch.equal(random_values, imprinted_values)
    assert torch.equal(imprinted_values, imprint)
    assert torch.equal(random_next, imprinted_next)


def test_initialization_v2_all_modes_share_one_draw_and_registered_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2)
    labels = {"a": 0, "b": 1}
    cache = SimpleNamespace(class_count=2)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        module,
        "build_registered_fepf_cache",
        lambda **_kwargs: calls.append(("cache", None)) or cache,
    )

    def prepare(_cache, random_head, *, mode, training_seed, device):
        calls.append(("prepare", mode))
        canonical = torch.full_like(random_head, 0.125)
        prepared = canonical if mode != "fepf_random" else random_head / 2
        return SimpleNamespace(
            official_random_head=random_head.detach().clone(),
            canonical_head=canonical,
            prepared_start_head=prepared,
            initial_diagnostic=SimpleNamespace(loss=3.0),
        )

    monkeypatch.setattr(module, "prepare_registered_fepf_evidence", prepare)

    def fit(_cache, evidence, *, training_seed, device):
        calls.append(("fit", evidence.prepared_start_head.detach().clone()))
        return SimpleNamespace(head=evidence.prepared_start_head + 1, fit_seconds=2.0)

    monkeypatch.setattr(module, "fit_fepf_head", fit)

    def receipt(**kwargs):
        calls.append(("receipt", kwargs["mode"]))
        return {
            "classifier_shape": list(kwargs["evidence"].prepared_start_head.shape),
            "official_random_head_sha256": hashlib.sha256(
                kwargs["official_random_head"].numpy().tobytes(order="C")
            ).hexdigest(),
            "prepared_start_head_sha256": hashlib.sha256(
                kwargs["evidence"].prepared_start_head.numpy().tobytes(order="C")
            ).hexdigest(),
            "initial_loss": kwargs["evidence"].initial_diagnostic.loss,
            "final_loss": (
                kwargs["evidence"].initial_diagnostic.loss
                if kwargs["fit"] is None
                else 2.0
            ),
        }

    monkeypatch.setattr(module, "initialization_receipt_v2", receipt)
    args = SimpleNamespace(
        seed=7,
        holdout_fraction=0.2,
        holdout_seed=20_260_828,
        checkpoint=Path("FP16-ViT-L-14-336px.pt"),
        run_config=None,
        epochs=16,
        batch_size=128,
        workers=4,
        learning_rate=1e-5,
        classifier_learning_rate=1e-4,
        margin=0.25,
        scale=32.0,
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=512,
        eval_every=4,
        checkpoint_every=4,
        max_steps=None,
        bf16=False,
    )
    results = {}
    for mode in ("imprinted", "fepf_mean", "fepf_random"):
        torch.manual_seed(991)
        args.classifier_init = mode
        values, result = module.initialize_registered_classifier(
            args=args,
            raw_model=raw_model,
            optimization=(),
            labels=labels,
            eval_transform=lambda image: image,
            device=torch.device("cpu"),
        )
        results[mode] = (values, result, torch.get_rng_state().clone())

    assert len([call for call in calls if call[0] == "cache"]) == 3
    assert [call[1] for call in calls if call[0] == "prepare"] == [
        "imprinted",
        "fepf_mean",
        "fepf_random",
    ]
    assert len([call for call in calls if call[0] == "fit"]) == 2
    official_hashes = {
        result["official_random_head_sha256"] for _values, result, _state in results.values()
    }
    assert len(official_hashes) == 1
    assert torch.equal(results["imprinted"][0], results["fepf_mean"][0] - 1)
    assert results["imprinted"][1]["initial_loss"] == results["imprinted"][1]["final_loss"]
    assert results["fepf_mean"][1]["initial_loss"] != results["fepf_mean"][1]["final_loss"]
    assert all(
        torch.equal(results["imprinted"][2], results[mode][2])
        for mode in ("fepf_mean", "fepf_random")
    )


def test_initialization_v2_rng_restored_to_post_draw_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2)
    raw_model.train(False)
    args = SimpleNamespace(
        seed=7,
        classifier_init="fepf_mean",
        holdout_fraction=0.2,
        holdout_seed=20_260_828,
        checkpoint=Path("FP16-ViT-L-14-336px.pt"),
        run_config=None,
        batch_size=128,
        workers=4,
    )
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)

    def fail_cache(**_kwargs):
        raw_model.train(True)
        random.random()
        np.random.random()
        torch.rand(9)
        raise RuntimeError("injected cache failure")

    monkeypatch.setattr(module, "build_registered_fepf_cache", fail_cache)
    torch.manual_seed(551)
    expected = torch.empty(2, 768)
    torch.nn.init.normal_(expected, std=0.01)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(551)

    with pytest.raises(RuntimeError, match="injected cache failure"):
        module.initialize_registered_classifier(
            args=args,
            raw_model=raw_model,
            optimization=(),
            labels={"a": 0, "b": 1},
            eval_transform=lambda image: image,
            device=torch.device("cpu"),
        )

    assert torch.equal(torch.get_rng_state(), expected_state)
    assert raw_model.training is False


def test_fepf_resume_classifier_path_never_recomputes_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    args = SimpleNamespace(
        resume=tmp_path / "parent" / "epoch-0004.pt",
        parent_run_receipt=tmp_path / "parent-run.json",
        parent_initialization_receipt=tmp_path / "initialization.json",
    )
    parent = {"initialization_receipt_sha256": "a" * 64}
    receipt = {"classifier_shape": [2, 768]}
    monkeypatch.setattr(module, "load_and_validate_parent_run_receipt", lambda **_kwargs: parent)
    observed: list[str] = []

    def load_initialization(**kwargs):
        assert kwargs["expected_sha256"] == "a" * 64
        observed.append("load")
        return receipt

    monkeypatch.setattr(
        module,
        "load_and_validate_parent_initialization_receipt",
        load_initialization,
    )
    monkeypatch.setattr(
        module,
        "initialize_registered_classifier",
        lambda **_kwargs: pytest.fail("resume must not initialize"),
    )

    values, loaded, loaded_parent = module.resolve_registered_classifier_initialization(
        args=args,
        fresh=lambda: pytest.fail("resume must not build cache or fit")
    )

    assert values.shape == (2, 768)
    assert values.dtype == torch.float32
    assert loaded is receipt
    assert loaded_parent is parent
    assert observed == ["load"]


def test_official_train_transform_emits_336_fp32_tensor() -> None:
    module = _load_script()
    transform = module.build_train_transform(336)
    torch.manual_seed(0)

    result = transform(Image.new("RGB", (400, 500), (10, 20, 30)))

    assert result.shape == (3, 336, 336)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_training_epoch_updates_backbone_with_official_mask_objective() -> None:
    module = _load_script()
    torch.manual_seed(4)
    backbone = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = module.build_optimizer(
        backbone,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )
    before = backbone.weight.detach().clone()
    scaler = torch.amp.GradScaler("cpu")

    result = module.run_training_epoch(
        backbone,
        classifier,
        loader,
        optimizer,
        scheduler=None,
        mask_generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=scaler,
    )

    assert result["steps"] == 1
    assert np.isfinite(result["mean_loss"])
    assert not torch.equal(backbone.weight, before)
    assert scaler.state_dict()["_growth_tracker"] == 1


def test_holdout_evaluation_uses_official_normalize_then_prefix_geometry(
    tmp_path: Path,
) -> None:
    module = _load_script()

    def record(name: str, split: str, label: str, color: tuple[int, int, int]):
        path = tmp_path / name
        Image.new("RGB", (2, 2), color).save(path)
        return InshopRecord(split=split, image_path=path, label=label)

    query = (
        record("qa.png", "query", "a", (255, 0, 0)),
        record("qb.png", "query", "b", (0, 255, 0)),
    )
    gallery = (
        record("ga.png", "gallery", "a", (255, 0, 0)),
        record("gb.png", "gallery", "b", (0, 255, 0)),
    )

    class MeanColor(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    def transform(image: Image.Image) -> torch.Tensor:
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values).permute(2, 0, 1)

    result = module.evaluate_holdout(
        MeanColor(),
        query,
        gallery,
        transform,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
        evaluation_features=3,
    )

    assert result == {
        "recall_at_1": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_30": 1.0,
        "map_at_r": 1.0,
    }


def test_fit_writes_sparse_raw_model_checkpoint_and_metrics(tmp_path: Path) -> None:
    module = _load_script()
    torch.manual_seed(5)
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )
    evaluations: list[int] = []
    data_seeds: list[int] = []

    def evaluate(epoch: int) -> dict[str, float]:
        evaluations.append(epoch)
        data_seeds.append(loader.generator.initial_seed())
        return {"recall_at_1": epoch / 2}

    history = module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=2,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=1,
        checkpoint_every=1,
        output_dir=tmp_path,
        evaluate=evaluate,
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert evaluations == [1, 2]
    assert data_seeds == [
        module.experiment_stream_seed(0, 2_000),
        module.experiment_stream_seed(0, 2_001),
    ]
    assert [row["epoch"] for row in history] == [1, 2]
    assert [row["metrics"] for row in history] == [
        {"recall_at_1": 0.5},
        {"recall_at_1": 1.0},
    ]
    checkpoint = torch.load(tmp_path / "epoch-0002.pt", weights_only=False)
    assert tuple(checkpoint) == (
        "epoch",
        "model",
        "classifier",
        "ema",
        "optimizer",
        "scheduler",
        "scaler",
        "mask_generator",
        "torch_rng_state",
        "cuda_rng_states",
        "selection_holdout",
        "training_protocol",
        "history",
    )
    assert checkpoint["epoch"] == 2
    assert set(checkpoint["model"]) == set(raw_model.state_dict())
    assert checkpoint["history"] == history
    assert not list(tmp_path.glob("*.tmp"))


def test_ema_checkpoint_roundtrip_restores_shadow_and_update_count(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    ema = module.StepEMA(raw_model, classifier)
    with torch.no_grad():
        raw_model.weight.add_(2.0)
        classifier.add_(3.0)
    ema.update()
    expected_ema = ema.state_dict()
    protocol = {
        "seed": 0,
        "objective": "official-eight-mask",
        "ema_decay": 0.999,
        "classifier_init": "random",
    }
    path = tmp_path / "ema.pt"

    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        step_ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
        history=[],
    )
    with torch.no_grad():
        raw_model.weight.zero_()
        classifier.zero_()
    ema.update()

    epoch, history = module.restore_training_checkpoint(
        path,
        raw_model=raw_model,
        classifier=classifier,
        step_ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(2),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
    )

    assert epoch == 1
    assert history == []
    restored = ema.state_dict()
    assert restored["updates"] == expected_ema["updates"]
    assert torch.equal(restored["classifier"], expected_ema["classifier"])
    assert all(
        torch.equal(restored["backbone"][name], value)
        for name, value in expected_ema["backbone"].items()
    )


def test_fit_always_checkpoints_final_and_evaluated_epochs(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )

    module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=3,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=3,
        checkpoint_every=2,
        output_dir=tmp_path,
        evaluate=lambda epoch: {"recall_at_1": epoch / 3},
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert sorted(path.name for path in tmp_path.glob("epoch-*.pt")) == [
        "epoch-0002.pt",
        "epoch-0003.pt",
    ]


def test_stop_after_epoch_preserves_checkpoint_protocol_and_real_restore(
    tmp_path: Path,
) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )
    protocol = {
        "seed": 0,
        "epochs": 16,
        "objective": "official-eight-mask",
        "initialization_receipt_sha256": "a" * 64,
    }
    parent = tmp_path / "parent"
    history = module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=16,
        stop_after_epoch=4,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=4,
        checkpoint_every=4,
        output_dir=parent,
        evaluate=lambda epoch: {"recall_at_1": epoch / 16},
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
    )
    checkpoint = torch.load(parent / "epoch-0004.pt", weights_only=False)

    assert [row["epoch"] for row in history] == [1, 2, 3, 4]
    assert checkpoint["training_protocol"] == protocol
    assert "stop_after_epoch" not in checkpoint["training_protocol"]

    restored_model = torch.nn.Linear(3, 8, bias=False)
    restored_classifier = torch.nn.Parameter(torch.randn(4, 8))
    restored_optimizer = module.build_optimizer(
        restored_model,
        restored_classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    epoch, restored_history = module.restore_training_checkpoint(
        parent / "epoch-0004.pt",
        raw_model=restored_model,
        classifier=restored_classifier,
        optimizer=restored_optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(99),
        device=torch.device("cpu"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=dict(protocol),
    )
    assert epoch == 4
    assert restored_history == history


def test_inference_signature_authenticity_and_cross_arm_structure() -> None:
    module = _load_script()

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 4, bias=False)
            self.register_buffer("position_ids", torch.arange(4, dtype=torch.int64))

    first = Backbone()
    second = Backbone()
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.projection.weight.add_(1.0)
    descriptor_a = torch.arange(1024, dtype=torch.float32).reshape(2, 512)
    descriptor_b = descriptor_a + 1

    signature_a = module.build_inference_signature(first, descriptor=descriptor_a)
    signature_b = module.build_inference_signature(second, descriptor=descriptor_b)

    assert [row["name"] for row in signature_a["tensors"]] == [
        "position_ids",
        "projection.weight",
    ]
    assert [row["kind"] for row in signature_a["tensors"]] == ["buffer", "parameter"]
    assert signature_a["operations"] == [
        "official_forward",
        "full768_l2",
        "prefix512",
        "squared_euclidean",
    ]
    module.validate_inference_signature(signature_a, raw_model=first, descriptor=descriptor_a)
    module.require_cross_arm_inference_equality(signature_a, signature_b)
    with pytest.raises(ValueError, match="authenticity differs"):
        module.validate_inference_signature(signature_a, raw_model=second, descriptor=descriptor_b)


@pytest.mark.parametrize(
    "mutation",
    ("reorder", "missing_buffer", "classifier", "operation_order", "tensor_bytes"),
)
def test_inference_signature_rejects_inventory_mutations(mutation: str) -> None:
    module = _load_script()

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 4, bias=False)
            self.register_buffer("position_ids", torch.arange(4, dtype=torch.int64))

    model = Backbone()
    descriptor = torch.arange(1024, dtype=torch.float32).reshape(2, 512)
    signature = module.build_inference_signature(model, descriptor=descriptor)
    changed = json.loads(json.dumps(signature))
    if mutation == "reorder":
        changed["tensors"].reverse()
    elif mutation == "missing_buffer":
        changed["tensors"].pop(0)
    elif mutation == "classifier":
        changed["tensors"].append(
            {
                **changed["tensors"][-1],
                "name": "classifier",
            }
        )
    elif mutation == "operation_order":
        changed["operations"].reverse()
    else:
        changed["tensors"][-1]["sha256"] = "f" * 64

    with pytest.raises(ValueError):
        module.validate_inference_signature(changed, raw_model=model, descriptor=descriptor)


def _fepf_run_receipt_fixture(module, tmp_path: Path):
    parent = tmp_path / "parent"
    continuation = tmp_path / "continuation"
    parent.mkdir()
    continuation.mkdir()
    initialization = parent / "initialization-receipt.json"
    initialization_object = {"schema": "initialization-receipt-v2", "fixture": True}
    initialization.write_text(json.dumps(initialization_object) + "\n", encoding="utf-8")
    initialization_sha256 = module.canonical_initialization_receipt_v2_sha256(
        initialization_object
    )
    parent_history = parent / "history.json"
    parent_history.write_text("[]\n", encoding="utf-8")
    epoch4 = parent / "epoch-0004.pt"
    epoch4.write_bytes(b"epoch-four")
    model = torch.nn.Linear(3, 4, bias=False)
    signature = module.build_inference_signature(
        model, descriptor=torch.arange(512, dtype=torch.float32)[None]
    )
    protocol = {
        "seed": 7,
        "epochs": 16,
        "objective": "official-eight-mask",
        "initialization_receipt_sha256": initialization_sha256,
    }
    parent_receipt = module.training_run_receipt_v2(
        output_dir=parent,
        mode="fepf_mean",
        training_seed=7,
        holdout_fraction=0.2,
        holdout_seed=20_260_828,
        training_protocol=protocol,
        stop_after_epoch=4,
        initialization_receipt_path=initialization,
        history_path=parent_history,
        checkpoint_paths={4: epoch4},
        raw_backbone_pre_initialization_sha256="a" * 64,
        raw_backbone_pre_training_sha256="a" * 64,
        inference_signature=signature,
    )
    parent_run = parent / "run-receipt.json"
    parent_run.write_text(json.dumps(parent_receipt) + "\n", encoding="utf-8")
    continuation_history = continuation / "history.json"
    continuation_history.write_text("[]\n", encoding="utf-8")
    checkpoints = {4: epoch4}
    for epoch in (8, 12, 16):
        path = continuation / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"epoch-{epoch}".encode())
        checkpoints[epoch] = path
    receipt = module.training_run_receipt_v2(
        output_dir=continuation,
        mode="fepf_mean",
        training_seed=7,
        holdout_fraction=0.2,
        holdout_seed=20_260_828,
        training_protocol=dict(protocol),
        stop_after_epoch=16,
        initialization_receipt_path=initialization,
        history_path=continuation_history,
        checkpoint_paths=checkpoints,
        raw_backbone_pre_initialization_sha256="a" * 64,
        raw_backbone_pre_training_sha256="a" * 64,
        inference_signature=signature,
        parent_run_receipt_path=parent_run,
        parent_checkpoint_path=epoch4,
    )
    return receipt, continuation, initialization_sha256, epoch4


def test_run_receipt_v2_authenticates_parent_root_and_original_initialization(
    tmp_path: Path,
) -> None:
    module = _load_script()
    receipt, continuation, initialization_sha256, _epoch4 = _fepf_run_receipt_fixture(
        module, tmp_path
    )

    module.validate_training_run_receipt_v2(receipt, evidence_root=continuation)

    assert receipt["initialization_receipt_sha256"] == initialization_sha256
    assert receipt["training_protocol"]["initialization_receipt_sha256"] == initialization_sha256
    assert receipt["parent_evidence_root"] == {"kind": "relative", "path": "../parent"}
    assert receipt["checkpoints"][0]["root"] == "parent"
    assert [row["root"] for row in receipt["checkpoints"][1:]] == [
        "current",
        "current",
        "current",
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "cross_run_substitution",
        "missing_parent_artifact",
        "path_escape",
        "wrong_root",
        "wrong_initialization_digest",
    ),
)
def test_run_receipt_v2_rejects_parent_link_mutations(
    tmp_path: Path, mutation: str
) -> None:
    module = _load_script()
    receipt, continuation, _initialization_sha256, epoch4 = _fepf_run_receipt_fixture(
        module, tmp_path
    )
    changed = json.loads(json.dumps(receipt))
    if mutation == "cross_run_substitution":
        changed["parent_run_receipt"]["sha256"] = "f" * 64
    elif mutation == "missing_parent_artifact":
        epoch4.unlink()
    elif mutation == "path_escape":
        changed["initialization_receipt"]["path"] = "../initialization-receipt.json"
    elif mutation == "wrong_root":
        changed["checkpoints"][0]["root"] = "current"
    else:
        changed["initialization_receipt_sha256"] = "f" * 64

    with pytest.raises(ValueError):
        module.validate_training_run_receipt_v2(changed, evidence_root=continuation)


def test_parent_initialization_validation_uses_trusted_run_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    receipt_path = tmp_path / "initialization.json"
    receipt_path.write_text('{"receipt":"untrusted-bytes"}\n', encoding="utf-8")
    args = module.parse_args(
        _required_cli(tmp_path)
        + [
            "--classifier-init",
            "fepf_mean",
            "--epochs",
            "16",
            "--evaluation-features",
            "512",
        ]
    )
    observed = []

    def validate(_receipt, *, expected, device):
        observed.append(expected.receipt_sha256)

    monkeypatch.setattr(module, "validate_initialization_receipt_v2", validate)
    trusted = "b" * 64
    module.load_and_validate_parent_initialization_receipt(
        path=receipt_path,
        args=args,
        resume_checkpoint=tmp_path / "epoch-0004.pt",
        expected_sha256=trusted,
    )

    assert observed == [trusted]
    assert trusted != module.canonical_initialization_receipt_v2_sha256(
        {"receipt": "untrusted-bytes"}
    )


def test_fit_registers_ema_for_training_and_releases_hook_afterward(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    ema = module.StepEMA(raw_model, classifier)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )

    module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=1,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=0,
        checkpoint_every=1,
        output_dir=tmp_path,
        evaluate=lambda _epoch: pytest.fail("evaluation is disabled"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={
            "seed": 0,
            "objective": "official-eight-mask",
            "ema_decay": 0.999,
            "classifier_init": "random",
        },
        step_ema=ema,
    )

    assert ema.state_dict()["updates"] == 1
    checkpoint = torch.load(tmp_path / "epoch-0001.pt", weights_only=False)
    assert checkpoint["ema"]["updates"] == 1
    optimizer.zero_grad(set_to_none=True)
    (raw_model(torch.ones(1, 3)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1


def test_restore_checkpoint_recovers_training_state_and_history(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    mask_generator = torch.Generator().manual_seed(41)
    scaler = torch.amp.GradScaler("cpu")
    expected_model = {
        name: value.detach().clone() for name, value in raw_model.state_dict().items()
    }
    expected_classifier = classifier.detach().clone()
    expected_mask_state = mask_generator.get_state().clone()
    expected_next_random = torch.rand(3, generator=torch.Generator().manual_seed(123))
    torch.manual_seed(123)
    path = tmp_path / "resume.pt"
    module.save_training_checkpoint(
        path,
        epoch=7,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        mask_generator=mask_generator,
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
        history=[{"epoch": 7, "train": {"steps": 1, "mean_loss": 2.0}, "metrics": None}],
    )
    with torch.no_grad():
        raw_model.weight.zero_()
        classifier.zero_()
    mask_generator.manual_seed(99)
    torch.rand(19)

    epoch, history = module.restore_training_checkpoint(
        path,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        mask_generator=mask_generator,
        device=torch.device("cuda" if torch.cuda.is_available() else "meta"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert epoch == 7
    assert history == [{"epoch": 7, "train": {"steps": 1, "mean_loss": 2.0}, "metrics": None}]
    for name, value in raw_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    assert torch.equal(classifier, expected_classifier)
    assert torch.equal(mask_generator.get_state(), expected_mask_state)
    assert torch.equal(torch.rand(3), expected_next_random)


def test_restore_checkpoint_rejects_training_protocol_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    path = tmp_path / "resume.pt"
    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
        history=[],
    )

    with pytest.raises(ValueError, match="training protocol differs"):
        module.restore_training_checkpoint(
            path,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            mask_generator=torch.Generator().manual_seed(2),
            device=torch.device("cpu"),
            selection_holdout={"seed": 0, "fraction": 0.2},
            training_protocol={"seed": 1, "objective": "official-eight-mask"},
        )


def test_restore_checkpoint_rejects_selection_holdout_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    path = tmp_path / "resume.pt"
    protocol = {"seed": 0, "objective": "official-eight-mask"}
    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
        history=[],
    )

    with pytest.raises(ValueError, match="selection holdout differs"):
        module.restore_training_checkpoint(
            path,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            mask_generator=torch.Generator().manual_seed(2),
            device=torch.device("cpu"),
            selection_holdout={"seed": 1, "fraction": 0.2},
            training_protocol=protocol,
        )


def test_main_fails_before_training_when_inputs_are_missing(tmp_path: Path, capsys) -> None:
    module = _load_script()

    exit_code = module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert "training failed:" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
