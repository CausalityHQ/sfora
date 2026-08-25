from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_proxy_muon_f0.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_proxy_muon_f0_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_run_config_loader_api_exists() -> None:
    assert callable(module.load_run_config)
    assert callable(module.authenticate_source_and_inputs)
    assert callable(module.observe_runtime)


@pytest.mark.parametrize(
    ("label", "module_name"),
    (
        ("runner", None),
        ("decision", "sfora.unicom_proxy_muon"),
        ("probe", "sfora.unicom_probe"),
        ("training", "sfora.unicom_training"),
        ("inshop", "sfora.unicom_inshop"),
    ),
)
def test_loaded_source_origins_bind_the_executing_modules_to_the_checkout(
    monkeypatch: pytest.MonkeyPatch, label: str, module_name: str | None
) -> None:
    repo_root = SCRIPT.parents[1]
    module.authenticate_loaded_source_origins(repo_root)

    loaded = module if module_name is None else sys.modules[module_name]
    monkeypatch.setattr(loaded, "__file__", f"/tmp/foreign/{label}.py")
    with pytest.raises(ValueError, match=f"loaded source origin differs: {label}"):
        module.authenticate_loaded_source_origins(repo_root)


def test_run_config_rejects_nonobject_and_unknown_version(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("[]\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        module.load_run_config(config)
    config.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_run_config(config)


def _git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=path, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _commit_all(path: Path, message: str) -> str:
    _git(path, "add", "-A")
    _git(
        path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return _git(path, "rev-parse", "HEAD")


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _runtime() -> dict[str, object]:
    return {
        "python_version": "3.12.3",
        "torch_version": "2.12.1",
        "numpy_version": "2.5.0",
        "sklearn_version": "1.7.2",
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_memory_allocated_bytes": 0,
        "cuda_memory_reserved_bytes": 0,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "muon_signature": "pinned",
        "observed_update_dtype": "torch.bfloat16",
    }


def _protocol() -> dict[str, object]:
    return {
        "learning_rate_grid": [0.000025, 0.00005, 0.0001, 0.0002, 0.0004],
        "phase1_seeds": [0, 1, 2],
        "phase2_seeds": [3, 4, 5],
        "phase1_steps": 64,
        "phase2_steps": 512,
        "retained_steps": [0, 64, 128, 192, 256, 307, 384, 435, 512],
        "validation_steps": [307, 435, 512],
        "batch_size": 128,
        "diagnostic_batches": 4,
        "diagnostic_masks": 4,
        "elapsed_limit_seconds": 2700.0,
        "peak_limit_bytes": 8 * 1024**3,
    }


def _authenticated_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, Path]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    source_files = []
    for index, relative in enumerate(module.CONFIG_SOURCE_PATHS):
        digest = _write(repo / relative, f"source-{index}\n".encode())
        source_files.append({"path": relative, "sha256": digest})
    _write(repo / ".gitignore", b"reports/generated/\n")
    source_commit = _commit_all(repo, "source")

    unicom = tmp_path / "unicom"
    unicom.mkdir()
    _git(unicom, "init", "-q")
    _write(unicom / "README", b"unicom\n")
    unicom_revision = _commit_all(unicom, "unicom")
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition_sha = _write(partition, b"0\nimage_name item_id evaluation_status\n")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint_sha = _write(checkpoint, b"checkpoint\n")
    external: dict[str, Path] = {
        "final_report": tmp_path / "final.md",
        "spherical_parent_result": tmp_path / "spherical.json",
        "cap_closure_receipt": tmp_path / "cap.json",
    }
    digests = {
        name: _write(path, f"{name}\n".encode()) for name, path in external.items()
    }

    def reference(name: str) -> dict[str, object]:
        return {"path": str(external[name]), "sha256": digests[name]}

    config: dict[str, object] = {
        "schema_version": module.RUN_CONFIG_SCHEMA_VERSION,
        "source": {"commit": source_commit, "files": source_files},
        "handoff": {
            "parent_commit": source_commit,
            "sole_path": "docs/unicom_proxy_muon_f0_run_config.json",
            "detached_clean": True,
        },
        "environment": _runtime(),
        "inputs": {
            "unicom_checkout": str(unicom),
            "unicom_revision": unicom_revision,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset_root": str(dataset),
            "partition": str(partition),
            "partition_sha256": partition_sha,
        },
        "parent": {
            "final_report": reference("final_report"),
            "spherical_parent_result": reference("spherical_parent_result"),
            "cap_closure_receipt": reference("cap_closure_receipt"),
            "parent_tensors": {
                name: f"{index + 40:064x}"
                for index, name in enumerate(module.CONFIG_PARENT_TENSOR_KEYS)
            },
        },
        "protocol": _protocol(),
        "result": {
            "relative_path": "reports/generated/proxy-muon.json",
            "failure_relative_path": "reports/generated/proxy-muon-failure.json",
        },
    }
    config_path = repo / config["handoff"]["sole_path"]
    _write(config_path, module.canonical_json_bytes(config))
    handoff_commit = _commit_all(repo, "config")
    _git(repo, "checkout", "--detach", "-q", handoff_commit)
    return repo, config_path, config, external


def test_source_and_input_authentication_uses_two_commit_detached_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path, expected, _external = _authenticated_fixture(tmp_path)
    config = module.load_run_config(config_path)
    authenticated_origins: list[Path] = []
    monkeypatch.setattr(
        module,
        "authenticate_loaded_source_origins",
        lambda root: authenticated_origins.append(root),
    )

    authority = module.authenticate_source_and_inputs(config, repo)

    assert config == expected
    assert authority["source_commit"] == config["source"]["commit"]
    assert authority["handoff_commit"] == _git(repo, "rev-parse", "HEAD")
    assert tuple(authority["sources"]) == module.SOURCE_HASH_KEYS
    assert tuple(authority["inputs"]) == module.INPUT_HASH_KEYS
    assert authority["inputs"]["imprinted_head"] == f"{40:064x}"
    assert "fitting_features" not in authority["inputs"]
    assert authenticated_origins == [repo]


def test_authentication_rejects_dirty_source_and_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path, config, _external = _authenticated_fixture(tmp_path)
    monkeypatch.setattr(module, "authenticate_loaded_source_origins", lambda _root: None)
    source = repo / module.CONFIG_SOURCE_PATHS[0]
    original = source.read_bytes()
    source.write_bytes(b"dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        module.authenticate_source_and_inputs(config, repo)
    source.write_bytes(original)
    output = repo / config["result"]["relative_path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        module.authenticate_source_and_inputs(config, repo)


def test_authentication_rejects_attached_head_and_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _config_path, config, external = _authenticated_fixture(tmp_path)
    monkeypatch.setattr(module, "authenticate_loaded_source_origins", lambda _root: None)
    _git(repo, "switch", "-qc", "attached")
    with pytest.raises(ValueError, match="detached"):
        module.authenticate_source_and_inputs(config, repo)
    _git(repo, "checkout", "--detach", "-q", "HEAD")
    report = external["final_report"]
    replacement = report.with_name("replacement.md")
    replacement.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(replacement)
    with pytest.raises(ValueError, match="parent final_report"):
        module.authenticate_source_and_inputs(config, repo)


def test_observed_runtime_has_exact_complete_concrete_shape() -> None:
    runtime = module.observe_runtime()
    assert tuple(runtime) == module.RUNTIME_KEYS
    assert all(type(runtime[key]) is str for key in module.RUNTIME_KEYS[:6])
    assert type(runtime["cuda_available"]) is bool
    assert all(
        type(runtime[key]) is int
        for key in (
            "cuda_device_count",
            "cuda_memory_allocated_bytes",
            "cuda_memory_reserved_bytes",
        )
    )
    assert all(
        type(runtime[key]) is bool
        for key in (
            "deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
        )
    )
    assert type(runtime["muon_signature"]) is str
    assert runtime["observed_update_dtype"] == "torch.bfloat16"


def test_runtime_binding_requires_exact_single_idle_cuda_device() -> None:
    expected = _runtime()
    assert module.validate_observed_runtime(expected, dict(expected)) == expected
    for key, value in (
        ("cuda_available", False),
        ("cuda_device_count", 2),
        ("cuda_memory_allocated_bytes", 1),
        ("cuda_memory_reserved_bytes", 1),
        ("torch_version", "different"),
    ):
        observed = dict(expected)
        observed[key] = value
        with pytest.raises(ValueError):
            module.validate_observed_runtime(expected, observed)
