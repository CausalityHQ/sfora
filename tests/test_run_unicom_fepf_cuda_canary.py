from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_unicom_fepf_cuda_canary", ROOT / "scripts/run_unicom_fepf_cuda_canary.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _environment() -> dict[str, object]:
    return {
        "python_vv": "Python 3.12.3",
        "torch": "2.6.0", "torchvision": "0.21.0", "timm": "1.0.0",
        "numpy": "2.1.3", "cuda": "12.4", "cudnn": "90100",
        "compile": {"available": "True", "inductor": "registered"},
        "device_uuid": "GPU-registered",
        "gpu_inventory": ["H100, GPU-registered, 550.54"],
        "pyproject_sha256": "1" * 64, "uv_lock_sha256": "2" * 64,
        "deterministic_execution": {
            "deterministic_algorithms": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
    }


def _environment_sha256() -> str:
    return MODULE._sha256(MODULE._canonical_json(_environment()))


def _minimal_canary_objects() -> dict[str, object]:
    preimages: dict[str, object] = {}
    initialization: dict[str, object] = {"schema": "initialization-receipt-v2"}
    for index, phase in enumerate(("entry", "post_draw", "restored"), start=1):
        python_hex = "80044e2e"
        numpy_hex = "80044e2e"
        torch_hex = f"{index:02x}"
        preimages[phase] = {
            "python_pickle_v5_hex": python_hex,
            "numpy_pickle_v5_hex": numpy_hex,
            "torch_cpu_hex": torch_hex,
            "torch_cuda_hex": [],
        }
        initialization[f"python_rng_{phase}_sha256"] = hashlib.sha256(
            b"python-random-v1\0" + bytes.fromhex(python_hex)
        ).hexdigest()
        initialization[f"numpy_rng_{phase}_sha256"] = hashlib.sha256(
            b"numpy-random-v1\0" + bytes.fromhex(numpy_hex)
        ).hexdigest()
        initialization[f"torch_cpu_rng_{phase}_sha256"] = hashlib.sha256(
            b"torch-cpu-random-v1\0" + bytes.fromhex(torch_hex)
        ).hexdigest()
        initialization[f"torch_cuda_rng_{phase}_sha256"] = []
    return {
        "initialization_receipt": initialization,
        "cache_inventory": {
            "schema": "unicom-fepf-canary-cache-v1", "tensors": [{}]
        },
        "model_inventory": {
            "schema": "unicom-fepf-canary-model-v1", "tensors": [{}]
        },
        "rng_audit": {"preimages": preimages},
        "model_modes": {"before": False, "after": False, "restored": False},
        "environment": _environment(),
    }


def _publish_complete_canary_family_fixture(
    evidence_root: Path,
    *,
    canary_objects: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    objects = {"observation": observation, **canary_objects}
    bindings: dict[str, object] = {}
    for name, value in objects.items():
        path = evidence_root / f"{name.replace('_', '-')}.json"
        payload = MODULE._canonical_json(value)
        path.write_bytes(payload)
        bindings[name] = {
            "path": path.name,
            "sha256": MODULE._sha256(payload),
            "bytes": len(payload),
        }
    manifest_path = evidence_root / "manifest.json"
    manifest_payload = MODULE._canonical_json(
        {"schema": "unicom-fepf-canary-evidence-v1", "objects": bindings}
    )
    manifest_path.write_bytes(manifest_payload)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": MODULE._sha256(manifest_payload),
        "bytes": len(manifest_payload),
    }


def test_review5_canary_publishes_one_complete_cross_task_environment(
    tmp_path: Path,
) -> None:
    environment = _environment()
    expected = tmp_path / "cuda-environment.json"
    authority = MODULE.publish_canary_environment(expected, environment)
    assert authority == {
        "path": str(expected.resolve()),
        "sha256": _environment_sha256(),
        "bytes": len(MODULE._canonical_json(environment)),
    }
    assert MODULE.validate_canary_environment_payload(
        environment, authority["sha256"]
    )
    with pytest.raises(FileExistsError):
        MODULE.publish_canary_environment(expected, {**environment, "torch": "other"})


def test_review6_canary_uses_downstream_canonical_bytes_and_external_manifest(
    tmp_path: Path,
) -> None:
    expected = (json.dumps(_environment(), indent=2, allow_nan=False) + "\n").encode()
    assert MODULE._canonical_json(_environment()) == expected
    evidence_root = tmp_path / "evidence"
    canary_objects = _minimal_canary_objects()
    manifest = _publish_complete_canary_family_fixture(
        evidence_root,
        canary_objects=canary_objects,
        observation={"cuda": True, "canary_objects": canary_objects, **_observation()},
    )
    MODULE.validate_canary_evidence_manifest(manifest, evidence_root=evidence_root)
    manifest_path = evidence_root / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest|authority"):
        MODULE.validate_canary_evidence_manifest(manifest, evidence_root=evidence_root)


def test_review7_canary_manifest_rejects_self_rooted_empty_evidence(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    canary_objects = {
        "initialization_receipt": {"schema_version": "legacy-self-oracle"},
        "cache_inventory": {"records": []},
        "model_inventory": {"tensors": []},
        "rng_audit": {"arbitrary": True},
        "model_modes": {"before": False, "after": True},
        "environment": _environment(),
    }
    manifest = _publish_complete_canary_family_fixture(
        evidence_root,
        canary_objects=canary_objects,
        observation={"cuda": True, "canary_objects": canary_objects, **_observation()},
    )
    with pytest.raises(ValueError, match="Task 1|cache|model|RNG|mode"):
        MODULE.validate_canary_evidence_manifest(
            manifest, evidence_root=evidence_root
        )


def test_review7_canary_exposes_committed_handoff_validator() -> None:
    assert callable(MODULE.validate_canary_handoff)


def test_review8_canary_main_requires_exact_budget_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config["publication_budget"] = {
        "schema": "unicom-fepf-publication-budget-v1", "publications": []
    }
    config["publication_budget_path"] = "preflight/publication-budget.json"
    config["publication_budget_sha256"] = hashlib.sha256(
        MODULE._canonical_json(config["publication_budget"])
    ).hexdigest()
    path = tmp_path / "config.json"
    path.write_bytes(MODULE._canonical_json(config))
    reached_backend = False

    def forbidden(_config: object) -> Path:
        nonlocal reached_backend
        reached_backend = True
        return tmp_path / "unexpected"

    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(MODULE, "run_cuda_canary", forbidden)
    assert MODULE.main([
        "--config", str(path),
        "--publication-stage", "cuda-canary",
        "--campaign-root", str(Path(config["artifact_root"])),
        "--authority-preflight-only",
    ]) == 2
    assert reached_backend is False


def test_review9_standalone_canary_materializes_root_and_budget_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    budget = config["publication_budget"]
    path = tmp_path / "config.json"
    path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda _config, **_kwargs: {}
    )
    runs: list[tuple[object, object]] = []

    def run_canary(_value, *, backend, crash_after_publication):
        runs.append((backend, crash_after_publication))
        return root / "terminal.json"

    monkeypatch.setattr(MODULE, "run_cuda_canary", run_canary)
    assert MODULE.main(["--config", str(path)]) == 0
    assert len(runs) == 1 and callable(runs[0][0]) and runs[0][1] is None
    assert (root / "preflight/publication-budget.json").read_bytes() == (
        MODULE._canonical_json(budget)
    )


def test_review9_coherent_canary_family_replacement_is_not_resume_authority(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["cuda_canary_authority"] = {}
    root = Path(config["artifact_root"])
    evidence = root / "preflight/canary-evidence"
    evidence.mkdir(parents=True)
    forged_environment = {**_environment(), "torch": "forged-coherent"}
    environment_path = Path(config["cuda_canary_environment"]["path"])
    environment_path.parent.mkdir(exist_ok=True)
    environment_path.write_bytes(MODULE._canonical_json(forged_environment))
    empty_pickle = "80044e2e"
    empty_tensor = "00"
    initialization = {"schema": "initialization-receipt-v2"}
    preimages = {}
    for phase in ("entry", "post_draw", "restored"):
        preimages[phase] = {
            "python_pickle_v5_hex": empty_pickle,
            "numpy_pickle_v5_hex": empty_pickle,
            "torch_cpu_hex": empty_tensor,
        }
        suffix = phase
        initialization[f"python_rng_{suffix}_sha256"] = hashlib.sha256(
            b"python-random-v1\0" + bytes.fromhex(empty_pickle)
        ).hexdigest()
        initialization[f"numpy_rng_{suffix}_sha256"] = hashlib.sha256(
            b"numpy-random-v1\0" + bytes.fromhex(empty_pickle)
        ).hexdigest()
        initialization[f"torch_cpu_rng_{suffix}_sha256"] = hashlib.sha256(
            b"torch-cpu-random-v1\0" + bytes.fromhex(empty_tensor)
        ).hexdigest()
    canary_objects = {
        "initialization_receipt": initialization,
        "cache_inventory": {
            "schema": "unicom-fepf-canary-cache-v1", "tensors": [{}]
        },
        "model_inventory": {
            "schema": "unicom-fepf-canary-model-v1", "tensors": [{}]
        },
        "rng_audit": {
            "entry": "2" * 64,
            "post_draw": "3" * 64,
            "restored": "3" * 64,
            "preimages": preimages,
        },
        "model_modes": {"before": False, "after": False, "restored": False},
        "environment": forged_environment,
    }
    observation = {
        **_observation(),
        "environment": forged_environment,
        "environment_sha256": MODULE._sha256(
            MODULE._canonical_json(forged_environment)
        ),
    }
    manifest = _publish_complete_canary_family_fixture(
        evidence,
        canary_objects=canary_objects,
        observation={"cuda": True, "canary_objects": canary_objects, **observation},
    )
    observation["evidence_manifest_sha256"] = manifest["sha256"]
    terminal = MODULE.build_cuda_canary_receipt(
        config,
        observation,
        expected_device_uuid="GPU-registered",
        expected_environment_sha256=observation["environment_sha256"],
    )
    campaign_spec = importlib.util.spec_from_file_location(
        "review9_campaign_validator", ROOT / "scripts/run_unicom_fepf_campaign.py"
    )
    assert campaign_spec is not None and campaign_spec.loader is not None
    campaign = importlib.util.module_from_spec(campaign_spec)
    campaign_spec.loader.exec_module(campaign)
    validator = campaign.RegisteredTerminalValidator(checkout_root=ROOT, config=config)
    with pytest.raises(ValueError, match="provenance|cache|model|environment|authority"):
        validator({"name": "cuda-canary"}, terminal)


def _review10_complete_forged_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Build a fully shaped but externally false family, not a shallow invalid one."""

    config = _config(tmp_path)
    config["model"]["revision"] = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
    root = Path(config["artifact_root"])
    evidence = root / "preflight/canary-evidence"
    evidence.mkdir(parents=True)
    environment = _environment()
    fixture_spec = importlib.util.spec_from_file_location(
        "review10_task1_receipt_fixture", ROOT / "tests/test_unicom_fepf.py"
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    receipt_fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(receipt_fixture)
    monkeypatch.setattr(receipt_fixture.module.torch.cuda, "device_count", lambda: 2)
    initialization = receipt_fixture._receipt_core("fepf_mean", monkeypatch)
    initialization["training_seed"] = 0
    initialization["holdout_seed"] = 0
    initialization["mask_generator_initial_sha256"] = (
        receipt_fixture.module._mask_state_sha256(
            training_seed=0, device=receipt_fixture.module.torch.device("cpu"), draws=0
        )
    )
    initialization["mask_generator_final_sha256"] = (
        receipt_fixture.module._mask_state_sha256(
            training_seed=0,
            device=receipt_fixture.module.torch.device("cpu"),
            draws=512,
        )
    )
    initialization["config_sha256"] = MODULE._sha256(MODULE._canonical_json(config))
    initialization["checkpoint_sha256"] = config["model"]["checkpoint_sha256"]
    initialization["source_sha256"] = MODULE._sha256(
        (ROOT / "src/sfora/unicom_fepf.py").read_bytes()
    )
    preimages: dict[str, object] = {}
    for phase in ("entry", "post_draw", "restored"):
        python_hex = "80044e2e"
        numpy_hex = "80044e2e"
        cpu_hex = "01" if phase == "entry" else "02"
        cuda_hex = ["03", "04"]
        preimages[phase] = {
            "python_pickle_v5_hex": python_hex,
            "numpy_pickle_v5_hex": numpy_hex,
            "torch_cpu_hex": cpu_hex,
            "torch_cuda_hex": cuda_hex,
        }
        initialization[f"python_rng_{phase}_sha256"] = hashlib.sha256(
            b"python-random-v1\0" + bytes.fromhex(python_hex)
        ).hexdigest()
        initialization[f"numpy_rng_{phase}_sha256"] = hashlib.sha256(
            b"numpy-random-v1\0" + bytes.fromhex(numpy_hex)
        ).hexdigest()
        initialization[f"torch_cpu_rng_{phase}_sha256"] = hashlib.sha256(
            b"torch-cpu-random-v1\0" + bytes.fromhex(cpu_hex)
        ).hexdigest()
        initialization[f"torch_cuda_rng_{phase}_sha256"] = [
            hashlib.sha256(
                f"torch-cuda-random-v1:{index}".encode()
                + b"\0"
                + bytes.fromhex(value)
            ).hexdigest()
            for index, value in enumerate(cuda_hex)
        ]
    cache = {
        "schema": "unicom-fepf-canary-cache-v1",
        "tensors": [{
            "name": "optimization_features", "kind": "cache",
            "shape": [2, 768], "dtype": "torch.float32", "sha256": "4" * 64,
            "values_sha256": "5" * 64,
        }],
        "authorities": {"labels": [0, 1]},
    }
    model = {
        "schema": "unicom-fepf-canary-model-v1",
        "revision": config["model"]["revision"],
        "tensors": [
            {
                "name": "weight", "kind": "parameter", "shape": [2, 2],
                "dtype": "torch.float32", "sha256": "6" * 64,
                "values_sha256": "7" * 64,
            },
            {
                "name": "running_mean", "kind": "buffer", "shape": [2],
                "dtype": "torch.float32", "sha256": "8" * 64,
                "values_sha256": "9" * 64,
            },
        ],
    }
    rng = {
        "entry": initialization["torch_cpu_rng_entry_sha256"],
        "post_draw": initialization["torch_cpu_rng_post_draw_sha256"],
        "restored": initialization["torch_cpu_rng_restored_sha256"],
        "preimages": preimages,
    }
    canary_objects = {
        "initialization_receipt": initialization,
        "cache_inventory": cache,
        "model_inventory": model,
        "rng_audit": rng,
        "model_modes": {"before": True, "after": False, "restored": True},
        "environment": environment,
    }
    observation = {
        **_observation(),
        "cuda": True,
        "canary_objects": canary_objects,
        "environment": environment,
        "environment_sha256": _environment_sha256(),
        "rng_entry_sha256": rng["entry"],
        "rng_post_draw_sha256": rng["post_draw"],
        "rng_restored_sha256": rng["restored"],
    }
    manifest = _publish_complete_canary_family_fixture(
        evidence,
        canary_objects=canary_objects,
        observation=observation,
    )
    observation = dict(observation)
    observation.pop("canary_objects")
    observation.pop("cuda")
    observation["evidence_manifest_sha256"] = manifest["sha256"]
    terminal = MODULE.build_cuda_canary_receipt(
        config,
        observation,
        expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    return config, evidence / "manifest.json", terminal


def test_review10_complete_coherent_canary_family_is_rebuilt_from_external_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest, terminal = _review10_complete_forged_family(tmp_path, monkeypatch)
    # This named non-authentic seam invokes Task 1's CPU core while independently
    # supplying the live cache/model/environment reconstruction authorities.
    with pytest.raises(ValueError, match="external reconstruction"):
        MODULE.validate_non_authentic_cpu_canary_family(
            config,
            manifest,
            terminal,
            live_model_inventory={"schema": "live-model-v1"},
            live_cache_inventory={"schema": "live-cache-v1"},
            live_environment=_environment(),
        )


def test_review10_target_public_cuda_validator_consumes_authentic_family() -> None:
    config_path_value = os.environ.get("SFORA_TASK7_FEPF_CONFIG")
    if config_path_value is None:
        pytest.skip("Task 7 supplies the target-authentic CUDA family")
    config_path = Path(config_path_value)
    config = json.loads(config_path.read_bytes())
    root = Path(config["artifact_root"])
    receipt = json.loads((root / config["cuda_canary_receipt"]).read_bytes())
    MODULE.validate_registered_canary_family(
        config,
        root / "preflight/canary-evidence/manifest.json",
        receipt,
    )


def test_review10_canary_timing_rng_and_mode_authority_cover_the_whole_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    events: list[str] = []
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: (events.append("allocate"), original_empty(*args, **kwargs))[1],
    )
    ticks = iter((10.0, 12.0, 20.0, 22.0))
    monkeypatch.setattr(
        MODULE.time,
        "perf_counter",
        lambda: (events.append("timer"), next(ticks))[1],
    )
    snapshots = [
        (None, None, torch.tensor([1], dtype=torch.uint8), (torch.tensor([2], dtype=torch.uint8),)),
        (None, None, torch.tensor([3], dtype=torch.uint8), (torch.tensor([4], dtype=torch.uint8),)),
        (None, None, torch.tensor([1], dtype=torch.uint8), (torch.tensor([2], dtype=torch.uint8),)),
        (None, None, torch.tensor([5], dtype=torch.uint8), (torch.tensor([6], dtype=torch.uint8),)),
        (None, None, torch.tensor([7], dtype=torch.uint8), (torch.tensor([8], dtype=torch.uint8),)),
        (None, None, torch.tensor([5], dtype=torch.uint8), (torch.tensor([6], dtype=torch.uint8),)),
    ]

    class RawModel:
        def __init__(self) -> None:
            self.training = True
            self.restorations: list[bool] = []

        def train(self, mode: bool = True):
            self.training = mode
            self.restorations.append(mode)
            return self

    raw_model = RawModel()

    class Trainer:
        @staticmethod
        def _global_rng_snapshot():
            return snapshots.pop(0)

        @staticmethod
        def _restore_global_rng_snapshot(_snapshot) -> None:
            return None

        @staticmethod
        def _fepf_rng_audit(_entry, _post):
            return types.SimpleNamespace()

        @staticmethod
        def build_registered_fepf_cache(**kwargs):
            kwargs["raw_model"].training = False
            return types.SimpleNamespace(values=torch.ones((2, 2)))

    fit = types.SimpleNamespace(
        head=torch.ones((2, 768)), final_head_sha256="f" * 64,
        diagnostic_feature_sha256="d" * 64, diagnostic_mask_sha256="e" * 64,
        initial_loss=2.0, final_loss=1.0,
    )
    fepf_path = tmp_path / "task1.py"
    fepf_path.write_text("# authority\n")
    fepf = types.SimpleNamespace(
        __file__=str(fepf_path),
        prepare_registered_fepf_evidence=lambda *_args, **_kwargs: types.SimpleNamespace(
            prepared_start_head_sha256="a" * 64
        ),
        fit_fepf_head=lambda *_args, **_kwargs: fit,
        initialization_receipt_v2=lambda **_kwargs: {
            "torch_cpu_rng_entry_sha256": "1" * 64,
            "torch_cpu_rng_post_draw_sha256": "2" * 64,
            "torch_cpu_rng_restored_sha256": "1" * 64,
        },
        canonical_initialization_receipt_v2_sha256=lambda _value: "3" * 64,
        FepfExpectedProvenance=lambda **kwargs: kwargs,
        validate_initialization_receipt_v2=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    result = MODULE.run_registered_fepf_canary(
        config={"model": {"checkpoint_sha256": "c" * 64}},
        device=torch.device("cpu"), torch=torch, fepf=fepf, trainer=Trainer,
        raw_model=raw_model, eval_transform=None,
        optimization=tuple(), labels={"a": 0, "b": 1},
    )
    assert events[0] == "timer"
    assert tuple(result["rng_preimages"]) == ("entry", "post_draw", "restored")
    assert all("torch_cuda_hex" in value for value in result["rng_preimages"].values())
    assert raw_model.training is True
    assert raw_model.restorations[-1] is True

    raw_model.restorations.clear()
    fepf.fit_fepf_head = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected fit failure")
    )
    with pytest.raises(RuntimeError, match="injected fit failure"):
        MODULE.run_registered_fepf_canary(
            config={"model": {"checkpoint_sha256": "c" * 64}},
            device=torch.device("cpu"), torch=torch, fepf=fepf, trainer=Trainer,
            raw_model=raw_model, eval_transform=None,
            optimization=tuple(), labels={"a": 0, "b": 1},
        )
    assert raw_model.training is True
    assert raw_model.restorations[-1] is True


def test_review10_normal_main_adopts_real_observation_after_every_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda _config, **_kwargs: {}
    )
    monkeypatch.setattr(
        MODULE, "validate_registered_canary_family", lambda *_args, **_kwargs: {}
    )
    for crash_after in range(1, 12):
        attempt = tmp_path / f"attempt-{crash_after}"
        value = json.loads(json.dumps(config))
        value["artifact_root"] = str(attempt)
        value["cuda_canary_environment"]["path"] = str(
            (attempt / "preflight/cuda-environment.json").resolve()
        )
        attempt_config = tmp_path / f"config-{crash_after}.json"
        attempt_config.write_bytes(MODULE._canonical_json(value))
        calls = 0

        def backend(_config):
            nonlocal calls
            calls += 1
            return _backend_observation()

        assert MODULE.main(
            ["--config", str(attempt_config)],
            backend=backend,
            crash_after_publication=crash_after,
        ) == 2
        calls_before_resume = calls
        assert MODULE.main(["--config", str(attempt_config)], backend=backend) == 0
        assert calls == calls_before_resume


def test_review10_standalone_canary_skips_valid_terminal_and_rejects_foreign_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest, terminal = _review10_complete_forged_family(tmp_path, monkeypatch)
    monkeypatch.setattr(
        MODULE,
        "validate_registered_canary_family",
        lambda *_args, **_kwargs: {"valid": True},
        raising=False,
    )
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda _config, **_kwargs: {}
    )
    receipt = Path(config["artifact_root"]) / config["cuda_canary_receipt"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(MODULE._canonical_json(terminal))
    path = tmp_path / "config.json"
    path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    called = False

    def forbidden(_config):
        nonlocal called
        called = True
        raise AssertionError("valid canary terminal reran CUDA")

    assert MODULE.main(["--config", str(path)], backend=forbidden) == 0
    assert called is False
    assert manifest.is_file()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unregistered.json").write_text("{}\n")
    foreign_config = json.loads(json.dumps(config))
    foreign_config["artifact_root"] = str(foreign)
    foreign_config["cuda_canary_environment"]["path"] = str(
        (foreign / "preflight/cuda-environment.json").resolve()
    )
    foreign_path = tmp_path / "foreign-config.json"
    foreign_path.write_bytes(MODULE._canonical_json(foreign_config))
    assert MODULE.main(["--config", str(foreign_path)], backend=forbidden) == 2
    assert not (foreign / "preflight").exists()


def test_review10_canary_first_performs_campaign_wide_physical_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config["artifact_budget_bytes"] = 1 << 30
    config["artifact_budget_inodes"] = 10_000
    statistics = types.SimpleNamespace(
        f_bavail=1, f_frsize=4096, f_favail=1,
    )
    monkeypatch.setattr(MODULE.os, "statvfs", lambda _path: statistics)
    with pytest.raises(OSError, match="capacity|space|inode"):
        MODULE.ensure_campaign_root(config)
    assert not Path(config["artifact_root"]).exists()


def test_review10_canary_execution_envelope_makes_cache_reconstruction_deterministic(
) -> None:
    enabled: list[bool] = []
    fake_torch = types.SimpleNamespace(
        use_deterministic_algorithms=lambda value: enabled.append(value),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=True)
            ),
            cudnn=types.SimpleNamespace(
                allow_tf32=True, benchmark=True, deterministic=False
            ),
        ),
    )
    environment: dict[str, str] = {}
    authority = MODULE.configure_deterministic_canary_execution(
        fake_torch, environment=environment
    )
    assert enabled == [True]
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.backends.cudnn.deterministic is True
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert authority == {
        "deterministic_algorithms": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }


def test_review6_canary_receipt_rejects_same_byte_temp_inode_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _materialize_direct_canary_authorities(config)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    attempted = False

    def substitute(source: Path, destination: Path) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError((source, destination))

    monkeypatch.setattr(MODULE.os, "link", substitute)
    output = MODULE.publish_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    assert output.is_file()
    assert attempted is False


def _config(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint.bin"
    partition = tmp_path / "partition.txt"
    checkpoint.write_bytes(b"checkpoint\n")
    partition.write_bytes(b"partition\n")
    publications = [
        {
            "name": f"cuda-canary:{name}",
            "path": path,
            "persistent_bytes": 16 * 1024**2,
            "temporary_bytes": 16 * 1024**2,
            "persistent_inodes": 1,
            "temporary_inodes": 1,
        }
        for name, path in (
            ("environment", "preflight/cuda-environment.json"),
            ("manifest", "preflight/canary-evidence/manifest.json"),
            ("receipt", "preflight/cuda_canary_v1.json"),
            ("evidence-observation", "preflight/canary-evidence/observation.json"),
            ("evidence-initialization-receipt",
             "preflight/canary-evidence/initialization-receipt.json"),
            ("evidence-cache-inventory", "preflight/canary-evidence/cache-inventory.json"),
            ("evidence-model-inventory", "preflight/canary-evidence/model-inventory.json"),
            ("evidence-rng-audit", "preflight/canary-evidence/rng-audit.json"),
            ("evidence-model-modes", "preflight/canary-evidence/model-modes.json"),
            ("evidence-environment", "preflight/canary-evidence/environment.json"),
        )
    ]
    for name in (
        "observation", "initialization-receipt", "cache-inventory",
        "model-inventory", "rng-audit", "model-modes", "environment",
        "manifest",
    ):
        publications.append({
            "name": f"cuda-canary:staging-{name}",
            "path": f"preflight/canary-evidence.staging/{name}.json",
            "persistent_bytes": 0,
            "temporary_bytes": 16 * 1024**2,
            "persistent_inodes": 0,
            "temporary_inodes": 1,
        })
    budget = {
        "schema": "unicom-fepf-publication-budget-v1",
        "publications": publications,
    }
    self_row = {
        "name": "campaign:publication-budget",
        "path": "preflight/publication-budget.json",
        "persistent_bytes": 0,
        "temporary_bytes": 0,
        "persistent_inodes": 1,
        "temporary_inodes": 1,
    }
    publications.append(self_row)
    for _iteration in range(32):
        size = len(MODULE._canonical_json(budget))
        if self_row["persistent_bytes"] == size:
            break
        self_row["persistent_bytes"] = size
        self_row["temporary_bytes"] = size
    else:
        raise AssertionError("fixture publication budget did not converge")
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": "a" * 40,
        "model": {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        },
        "artifact_root": str(tmp_path / "artifacts"),
        "artifact_budget_bytes": 1 << 30,
        "artifact_budget_inodes": 10_000,
        "inputs": {
            "checkpoint": str(checkpoint),
            "partition": str(partition),
        },
        "cuda_canary_authority": {
            "device_uuid": "GPU-registered", "environment_sha256": _environment_sha256(),
        },
        "cuda_canary_environment": {
            "path": str(
                (tmp_path / "artifacts/preflight/cuda-environment.json").resolve()
            ),
            "sha256": _environment_sha256(),
            "bytes": len(MODULE._canonical_json(_environment())),
        },
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
        "publication_budget": budget,
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": hashlib.sha256(
            MODULE._canonical_json(budget)
        ).hexdigest(),
    }


def _observation() -> dict[str, object]:
    environment = _environment()
    return {
        "environment": environment,
        "environment_sha256": _environment_sha256(),
        "device_uuid": "GPU-registered",
        "completed_steps": 512,
        "initial_head_sha256": "e" * 64,
        "final_head_sha256": "f" * 64,
        "diagnostic_sha256": "1" * 64,
        "rng_entry_sha256": "2" * 64,
        "rng_post_draw_sha256": "3" * 64,
        "rng_restored_sha256": "3" * 64,
        "raw_backbone_pre_sha256": "4" * 64,
        "raw_backbone_post_sha256": "4" * 64,
        "evidence_manifest_sha256": "6" * 64,
        "initial_loss": 3.0,
        "final_loss": 2.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }


def _backend_observation() -> dict[str, object]:
    return {
        "cuda": True,
        "canary_objects": _minimal_canary_objects(),
        **_observation(),
    }


def _materialize_direct_canary_authorities(config: dict[str, object]) -> Path:
    root = Path(config["artifact_root"])
    preflight = root / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    budget_path = root / config["publication_budget_path"]
    payload = MODULE._canonical_json(config["publication_budget"])
    if budget_path.exists():
        assert budget_path.read_bytes() == payload
    else:
        budget_path.write_bytes(payload)
    return root


def test_review12_real_backend_observation_order_builds_public_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal backend's insertion order must survive manifest completion."""

    config = _config(tmp_path)
    _materialize_direct_canary_authorities(config)
    observed = _backend_observation()
    observed.pop("evidence_manifest_sha256")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"schema":"unicom-fepf-canary-evidence-v1"}\n')
    monkeypatch.setattr(
        MODULE,
        "complete_canary_evidence_transaction",
        lambda *_args, **_kwargs: (observed, manifest),
    )
    monkeypatch.setattr(
        MODULE, "validate_registered_canary_family", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        MODULE, "publish_cuda_canary_receipt", lambda receipt, *_args, **_kwargs: receipt
    )

    receipt = MODULE.run_cuda_canary(config, backend=lambda _config: observed)
    assert tuple(receipt) == MODULE.RECEIPT_KEYS
    assert receipt["evidence_manifest_sha256"] == MODULE._sha256(manifest.read_bytes())


def test_review13_recompute_uses_the_already_seeded_model_without_reseeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: events.append(("seed", seed)),
        cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda _device: events.append("reset"),
            max_memory_allocated=lambda _device: 10,
            max_memory_reserved=lambda _device: 20,
        ),
    )
    trainer = types.SimpleNamespace(
        raw_backbone_state_sha256=lambda _model: "a" * 64
    )
    monkeypatch.setattr(
        MODULE,
        "run_registered_fepf_canary",
        lambda **_kwargs: (
            events.append("fit")
            or {
                "initial_head_sha256": "1" * 64,
                "final_head_sha256": "2" * 64,
                "diagnostic_sha256": "3" * 64,
                "rng_entry_sha256": "4" * 64,
                "rng_post_draw_sha256": "5" * 64,
                "rng_restored_sha256": "4" * 64,
                "initial_loss": 2.0,
                "final_loss": 1.0,
                "initialization_receipt": {},
                "rng_preimages": {},
                "cache_inventory": {},
                "model_modes": {},
            }
        ),
    )
    MODULE.recompute_registered_canary_dynamic_authority(
        config={}, device=object(), torch=fake_torch, fepf=object(),
        trainer=trainer, raw_model=object(), eval_transform=object(),
        optimization=(), labels={},
    )
    assert events == ["fit"]


def test_review14_cpu_canary_restores_the_post_draw_rng_for_task1_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official draw is retained while fitting remains globally RNG-neutral."""

    import torch

    from sfora import unicom_fepf as task1

    labels = torch.arange(2, dtype=torch.int64).repeat_interleave(64)
    features = torch.zeros((128, 768), dtype=torch.float32)
    features[torch.arange(128), labels] = 1.0
    cache = task1.build_fepf_cache(
        tuple((f"c{int(label)}", f"{index}.jpg") for index, label in enumerate(labels)),
        features,
        {"c0": 0, "c1": 1},
    )
    trainer = MODULE._load_script(
        ROOT / "scripts/train_unicom_inshop.py", "review14_rng_trainer"
    )
    trainer.build_registered_fepf_cache = lambda **_kwargs: cache
    cpu = torch.device("cpu")
    fepf = types.SimpleNamespace(
        __file__=task1.__file__,
        FepfExpectedProvenance=task1.FepfExpectedProvenance,
        canonical_initialization_receipt_v2_sha256=(
            task1.canonical_initialization_receipt_v2_sha256
        ),
        prepare_registered_fepf_evidence=lambda *args, **kwargs: (
            task1._prepare_registered_fepf_evidence_core(
                *args, **kwargs, allow_test_device=True
            )
        ),
        fit_fepf_head=lambda cache, evidence, **kwargs: task1._seal_registered_fit(
            task1._fit_fepf_head_core(
                cache, evidence.prepared_start_head,
                training_seed=kwargs["training_seed"], device=cpu,
                steps=kwargs["steps"],
                initial_diagnostic=evidence.initial_diagnostic,
            ),
            evidence,
        ),
        initialization_receipt_v2=lambda **kwargs: (
            task1._initialization_receipt_v2_core(
                **kwargs, allow_test_device=True
            )
        ),
        validate_initialization_receipt_v2=lambda receipt, **kwargs: (
            task1._validate_initialization_receipt_v2_core(
                receipt, expected=kwargs["expected"], device=cpu,
                allow_test_device=True,
            )
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    torch.manual_seed(23_001)
    result = MODULE.run_registered_fepf_canary(
        config={"model": {"checkpoint_sha256": "c" * 64}},
        device=cpu, torch=torch, fepf=fepf, trainer=trainer,
        raw_model=types.SimpleNamespace(
            training=True,
            train=lambda _mode=True: None,
        ),
        eval_transform=None, optimization=tuple(), labels={"c0": 0, "c1": 1},
    )
    assert result["rng_entry_sha256"] != result["rng_post_draw_sha256"]
    assert result["rng_post_draw_sha256"] == result["rng_restored_sha256"]


def test_review14_fresh_and_recompute_share_one_seed_model_cache_fit_prologue() -> None:
    events: list[object] = []
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: events.append(("seed", seed)),
        cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda _device: events.append("reset")
        ),
    )

    def loader() -> tuple[str, str, str]:
        events.append("model-rng-draw")
        return "model", "transform", "digest"

    for _phase in ("fresh", "recompute"):
        assert MODULE.prepare_registered_canary_seeded_model(
            torch=fake_torch, device="cuda:0", loader=loader
        ) == ("model", "transform", "digest")
    assert events == [
        ("seed", 23_001), "reset", "model-rng-draw",
        ("seed", 23_001), "reset", "model-rng-draw",
    ]


def test_review14_scientific_projection_requires_every_registered_field() -> None:
    scientific = {
        "completed_steps": 512,
        "initial_head_sha256": "1" * 64,
        "final_head_sha256": "2" * 64,
        "diagnostic_sha256": "3" * 64,
        "rng_entry_sha256": "4" * 64,
        "rng_post_draw_sha256": "5" * 64,
        "rng_restored_sha256": "5" * 64,
        "raw_backbone_pre_sha256": "6" * 64,
        "raw_backbone_post_sha256": "6" * 64,
        "initial_loss": 2.0,
        "final_loss": 1.0,
    }
    for missing in tuple(scientific):
        observed = dict(scientific)
        observed.pop(missing)
        with pytest.raises(ValueError, match="scientific|field|schema"):
            MODULE.validate_registered_canary_scientific_projection(
                observed, scientific
            )


def test_review13_fresh_canary_runs_one_fitted_validation_before_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    validations: list[tuple[Path, object]] = []

    def validate(_config, manifest, terminal, **_kwargs):
        validations.append((manifest, terminal))
        return {}

    monkeypatch.setattr(MODULE, "validate_registered_canary_family", validate)
    output = MODULE.run_cuda_canary(
        config, backend=lambda _config: _backend_observation()
    )
    assert output == root / "preflight/cuda_canary_v1.json"
    assert len(validations) == 1
    assert validations[0][1] is None


def test_review13_failed_fitted_validation_never_publishes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    monkeypatch.setattr(
        MODULE,
        "validate_registered_canary_family",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("independent fitted validation failed")
        ),
    )
    with pytest.raises(ValueError, match="fitted validation failed"):
        MODULE.run_cuda_canary(config, backend=lambda _config: _backend_observation())
    assert not (root / "preflight/cuda_canary_v1.json").exists()


@pytest.mark.parametrize(
    ("field", "replacement", "accepted"),
    (
        ("initialization_seconds", 999.0, True),
        ("fit_seconds", 777.0, True),
        ("peak_allocated_bytes", 999_999, True),
        ("peak_reserved_bytes", 1_999_999, True),
        ("final_head_sha256", "0" * 64, False),
    ),
)
def test_review13_canary_recompute_compares_science_not_observational_noise(
    field: str, replacement: object, accepted: bool
) -> None:
    observed = {
        "completed_steps": 512,
        "initial_head_sha256": "1" * 64,
        "final_head_sha256": "2" * 64,
        "diagnostic_sha256": "3" * 64,
        "rng_entry_sha256": "4" * 64,
        "rng_post_draw_sha256": "5" * 64,
        "rng_restored_sha256": "5" * 64,
        "raw_backbone_pre_sha256": "6" * 64,
        "raw_backbone_post_sha256": "6" * 64,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "initialization_seconds": 10.0,
        "fit_seconds": 9.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    recomputed = dict(observed)
    recomputed[field] = replacement
    if accepted:
        MODULE.validate_registered_canary_scientific_projection(observed, recomputed)
    else:
        with pytest.raises(ValueError, match="scientific|fitted"):
            MODULE.validate_registered_canary_scientific_projection(observed, recomputed)


def test_review12_real_backend_cache_serialization_round_trips_tuple_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest, terminal = _review10_complete_forged_family(tmp_path, monkeypatch)
    root = manifest.parent
    cache_path = root / "cache-inventory.json"
    live_cache = json.loads(cache_path.read_bytes())
    live_cache["authorities"]["labels"] = (0, 1)
    cache_payload = MODULE._canonical_json(live_cache)
    cache_path.write_bytes(cache_payload)
    manifest_object = json.loads(manifest.read_bytes())
    manifest_object["objects"]["cache_inventory"].update(
        sha256=MODULE._sha256(cache_payload), bytes=len(cache_payload)
    )
    manifest.write_bytes(MODULE._canonical_json(manifest_object))
    terminal["evidence_manifest_sha256"] = MODULE._sha256(manifest.read_bytes())
    model = json.loads((root / "model-inventory.json").read_bytes())

    MODULE.validate_non_authentic_cpu_canary_family(
        config, manifest, terminal,
        live_model_inventory=model,
        live_cache_inventory=live_cache,
        live_environment=_environment(),
    )


def test_review12_public_family_rejects_coherent_dynamic_observation_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Family-owned digests cannot authorize changed fitted observations."""

    config, manifest, terminal = _review10_complete_forged_family(tmp_path, monkeypatch)
    evidence = manifest.parent
    observation_path = evidence / "observation.json"
    observation = json.loads(observation_path.read_bytes())
    initialization_path = evidence / "initialization-receipt.json"
    initialization = json.loads(initialization_path.read_bytes())
    authentic_dynamic = {
        key: terminal[key]
        for key in (
            "completed_steps", "initial_head_sha256", "final_head_sha256",
            "diagnostic_sha256", "rng_entry_sha256", "rng_post_draw_sha256",
            "rng_restored_sha256", "raw_backbone_pre_sha256",
            "raw_backbone_post_sha256", "initial_loss", "final_loss",
            "peak_allocated_bytes", "peak_reserved_bytes",
        )
    }
    authentic_dynamic["initialization_receipt"] = copy.deepcopy(initialization)
    authentic_dynamic["rng_audit"] = json.loads(
        (evidence / "rng-audit.json").read_bytes()
    )
    registered_revision = MODULE.subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    import numpy as np
    import timm
    import torch
    import torchvision

    from sfora import unicom_fepf as task1
    from sfora import unicom_inshop

    config["inputs"]["unicom_checkout"] = str(ROOT)
    config["inputs"]["dataset_root"] = str(tmp_path / "dataset")
    config["model"]["revision"] = registered_revision
    task1_source = ROOT / "src/sfora/unicom_fepf.py"
    config["source_files"] = [{
        "path": "src/sfora/unicom_fepf.py",
        "sha256": MODULE._sha256(task1_source.read_bytes()),
        "bytes": task1_source.stat().st_size,
    }]
    records = tuple(
        types.SimpleNamespace(
            split="train", label=f"identity-{index % 2}",
            image_path=Path(f"image-{index:04d}.jpg"),
        )
        for index in range(128)
    )
    monkeypatch.setattr(unicom_inshop, "parse_inshop_partition", lambda _path: records)

    class LiveModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).view(2, 2))
            self.register_buffer("running_mean", torch.tensor([1.0, 2.0]))

    raw_model = LiveModel()
    cache = types.SimpleNamespace(
        optimization_features=torch.zeros((2, 768), dtype=torch.float32),
        labels=[0, 1],
    )

    class LiveTrainer:
        @staticmethod
        def _git_revision(_checkout: Path) -> str:
            return registered_revision

        @staticmethod
        def build_registered_fepf_cache(**kwargs):
            kwargs["raw_model"].train(False)
            return cache

    monkeypatch.setattr(MODULE, "_load_script", lambda *_args, **_kwargs: LiveTrainer)
    monkeypatch.setattr(
        MODULE, "load_registered_canary_model",
        lambda *_args, **_kwargs: (raw_model, object(), "0" * 64),
    )
    model_path = evidence / "model-inventory.json"
    model = {
        "schema": "unicom-fepf-canary-model-v1",
        "revision": registered_revision,
        "tensors": sorted([
            {
                "name": name, "kind": kind, "shape": list(tensor.shape),
                "dtype": str(tensor.dtype), "sha256": MODULE._tensor_hash(tensor),
            }
            for kind, values in (
                ("parameter", list(raw_model.named_parameters())),
                ("buffer", list(raw_model.named_buffers())),
            )
            for name, tensor in values
        ], key=lambda row: (row["name"], row["kind"])),
    }
    model_path.write_bytes(MODULE._canonical_json(model))
    observation["canary_objects"]["model_inventory"] = model
    cache_path = evidence / "cache-inventory.json"
    live_cache = {
        "schema": "unicom-fepf-canary-cache-v1",
        "tensors": [{
            "name": "optimization_features", "kind": "cache",
            "shape": [2, 768], "dtype": "torch.float32",
            "sha256": MODULE._tensor_hash(cache.optimization_features),
        }],
        "authorities": {"labels": [0, 1]},
    }
    cache_path.write_bytes(MODULE._canonical_json(live_cache))
    observation["canary_objects"]["cache_inventory"] = live_cache
    python_vv = MODULE.subprocess.run(
        [MODULE.sys.executable, "-VV"], check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _device: types.SimpleNamespace(
            uuid="00000000-0000-0000-0000-000000000001",
            name="H100", major=9, minor=0,
            total_memory=80 * 1024**3, multi_processor_count=120,
        ),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90100)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda command, **_kwargs: types.SimpleNamespace(
            stdout=(
                python_vv
                if "-VV" in command
                else "H100, GPU-00000000-0000-0000-0000-000000000001, 550.54\n"
            )
        ),
    )
    environment = {
        "python_vv": python_vv,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "numpy": np.__version__,
        "cuda": str(torch.version.cuda), "cudnn": str(torch.backends.cudnn.version()),
        "compile": {
            "available": str(hasattr(torch, "compile")),
            "inductor": str(getattr(torch.version, "git_version", "unknown")),
        },
        "device_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "gpu_inventory": [
            "H100, GPU-00000000-0000-0000-0000-000000000001, 550.54"
        ],
        "pyproject_sha256": MODULE._sha256((ROOT / "pyproject.toml").read_bytes()),
        "uv_lock_sha256": MODULE._sha256((ROOT / "uv.lock").read_bytes()),
        "deterministic_execution": _environment()["deterministic_execution"],
    }
    environment_path = evidence / "environment.json"
    environment_path.write_bytes(MODULE._canonical_json(environment))
    observation["environment"] = environment
    observation["environment_sha256"] = MODULE._sha256(
        MODULE._canonical_json(environment)
    )
    observation["canary_objects"]["environment"] = environment
    observation["device_uuid"] = environment["device_uuid"]
    terminal["device_uuid"] = environment["device_uuid"]
    config["cuda_canary_authority"]["device_uuid"] = environment["device_uuid"]
    config["cuda_canary_authority"]["environment_sha256"] = observation[
        "environment_sha256"
    ]
    config["cuda_canary_environment"].update(
        sha256=observation["environment_sha256"],
        bytes=len(MODULE._canonical_json(environment)),
    )
    for key, value in (
        ("initial_head_sha256", "a" * 64),
        ("final_head_sha256", "b" * 64),
        ("diagnostic_sha256", "c" * 64),
        ("raw_backbone_pre_sha256", "d" * 64),
        ("raw_backbone_post_sha256", "d" * 64),
        ("initial_loss", 9.0),
        ("final_loss", 8.0),
    ):
        observation[key] = value
        terminal[key] = value
    initialization["prepared_start_head_sha256"] = observation["initial_head_sha256"]
    initialization["final_head_sha256"] = observation["final_head_sha256"]
    initialization["initial_loss"] = observation["initial_loss"]
    initialization["final_loss"] = observation["final_loss"]
    initialization["source_sha256"] = config["source_files"][0]["sha256"]
    initialization["schedule_sha256"] = MODULE._sha256(MODULE._canonical_json({
        "steps": 512, "training_seed": 0,
        "records": [[record.label, str(record.image_path)] for record in records],
    }))
    initialization["config_sha256"] = MODULE._sha256(MODULE._canonical_json(config))
    monkeypatch.setattr(
        task1,
        "_mask_state_sha256",
        lambda *, draws, **_kwargs: initialization[
            "mask_generator_initial_sha256"
            if draws == 0
            else "mask_generator_final_sha256"
        ],
    )
    observation["canary_objects"]["initialization_receipt"] = initialization
    initialization_path.write_bytes(MODULE._canonical_json(initialization))
    observation_path.write_bytes(MODULE._canonical_json(observation))
    manifest_object = json.loads(manifest.read_bytes())
    for name, path in (
        ("initialization_receipt", initialization_path),
        ("cache_inventory", cache_path),
        ("model_inventory", model_path),
        ("environment", environment_path),
        ("observation", observation_path),
    ):
        payload = path.read_bytes()
        manifest_object["objects"][name].update(
            sha256=MODULE._sha256(payload), bytes=len(payload)
        )
    manifest.write_bytes(MODULE._canonical_json(manifest_object))
    terminal["config_sha256"] = MODULE._sha256(MODULE._canonical_json(config))
    terminal["source_commit"] = config["source_commit"]
    terminal["environment"] = environment
    terminal["environment_sha256"] = observation["environment_sha256"]
    terminal["evidence_manifest_sha256"] = MODULE._sha256(manifest.read_bytes())
    authentic_dynamic["cache_inventory"] = live_cache
    authentic_dynamic["model_modes"] = json.loads(
        (evidence / "model-modes.json").read_bytes()
    )
    monkeypatch.setattr(
        MODULE,
        "recompute_registered_canary_dynamic_authority",
        lambda **_kwargs: authentic_dynamic,
    )

    with pytest.raises(ValueError, match="external|reconstruct|fitted|observation"):
        MODULE.validate_registered_canary_family(config, manifest, terminal)


@pytest.mark.parametrize(("record_count", "label_count"), ((127, 2), (128, 1)))
def test_review15_public_family_rejects_unregistered_partition_fixture_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_count: int,
    label_count: int,
) -> None:
    config, manifest, terminal = _review10_complete_forged_family(
        tmp_path, monkeypatch
    )
    config["inputs"]["dataset_root"] = str(tmp_path / "registered-partition.txt")
    import torch

    from sfora import unicom_inshop

    records = tuple(
        types.SimpleNamespace(
            split="train",
            label=f"identity-{index % label_count}",
            image_path=Path(f"image-{index:04d}.jpg"),
        )
        for index in range(record_count)
    )

    class Trainer:
        @staticmethod
        def _git_revision(_checkout: Path) -> str:
            return config["model"]["revision"]

    monkeypatch.setattr(MODULE, "authenticate_canary_inputs", lambda _config: {})
    monkeypatch.setattr(
        MODULE, "reconstruct_canary_authority", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(MODULE, "_load_script", lambda *_args, **_kwargs: Trainer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        MODULE, "configure_deterministic_canary_execution",
        lambda *_args, **_kwargs: _environment()["deterministic_execution"],
    )
    monkeypatch.setattr(unicom_inshop, "parse_inshop_partition", lambda _path: records)
    monkeypatch.setattr(
        MODULE,
        "load_registered_canary_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model continuation ran before partition fixture guard")
        ),
    )
    with pytest.raises(ValueError, match="128|labels|partition|fixture"):
        MODULE.validate_registered_canary_family(config, manifest, terminal)


@pytest.mark.parametrize("prefix_length", range(1, 9))
def test_review12_resume_accepts_every_registered_observation_first_prefix(
    tmp_path: Path, prefix_length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    staging = root / "preflight/canary-evidence.staging"
    staging.mkdir()
    observation = _backend_observation()
    names = (
        "observation", "initialization-receipt", "cache-inventory",
        "model-inventory", "rng-audit", "model-modes", "environment", "manifest",
    )
    objects = observation["canary_objects"]
    values = (
        observation, objects["initialization_receipt"], objects["cache_inventory"],
        objects["model_inventory"], objects["rng_audit"], objects["model_modes"],
        objects["environment"], None,
    )
    bindings: dict[str, object] = {}
    object_names = (
        "observation", "initialization_receipt", "cache_inventory",
        "model_inventory", "rng_audit", "model_modes", "environment",
    )
    for name, value in zip(names[:min(prefix_length, 7)], values, strict=False):
        path = staging / f"{name}.json"
        payload = MODULE._canonical_json(value)
        path.write_bytes(payload)
        bindings[object_names[names.index(name)]] = {
            "path": path.name, "sha256": MODULE._sha256(payload), "bytes": len(payload)
        }
    if prefix_length == 8:
        (staging / "manifest.json").write_bytes(MODULE._canonical_json({
            "schema": "unicom-fepf-canary-evidence-v1", "objects": bindings,
        }))
    config_path = tmp_path / "config.json"
    config_path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        MODULE, "validate_registered_canary_family", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(MODULE, "build_cuda_canary_receipt", lambda *_a, **_k: {})
    monkeypatch.setattr(
        MODULE,
        "publish_cuda_canary_receipt",
        lambda *_a, **_k: root / "preflight/cuda_canary_v1.json",
    )
    calls = 0

    def forbidden_backend(_config):
        nonlocal calls
        calls += 1
        raise AssertionError("post-observation resume reran backend")

    assert MODULE.main(["--config", str(config_path)], backend=forbidden_backend) == 0
    assert calls == 0


def test_review12_completed_evidence_with_observation_is_resume_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    observation = _backend_observation()
    _publish_complete_canary_family_fixture(
        root / "preflight/canary-evidence",
        canary_objects=observation["canary_objects"],
        observation=observation,
    )
    config_path = tmp_path / "config.json"
    config_path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        MODULE, "validate_registered_canary_family", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(MODULE, "build_cuda_canary_receipt", lambda *_a, **_k: {})
    monkeypatch.setattr(
        MODULE,
        "publish_cuda_canary_receipt",
        lambda *_a, **_k: root / "preflight/cuda_canary_v1.json",
    )
    called = False

    def forbidden_backend(_config):
        nonlocal called
        called = True
        raise AssertionError("completed evidence reran backend")

    assert MODULE.main(["--config", str(config_path)], backend=forbidden_backend) == 0
    assert called is False


def test_review12_invalid_staging_prefix_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    staging = root / "preflight/canary-evidence.staging"
    staging.mkdir()
    (staging / "cache-inventory.json").write_text("{}\n")
    config_path = tmp_path / "config.json"
    config_path.write_bytes(MODULE._canonical_json(config))
    monkeypatch.setattr(MODULE, "validate_canary_handoff", lambda *_args: {})
    monkeypatch.setattr(
        MODULE, "validate_canary_exact_budget", lambda *_args, **_kwargs: {}
    )
    assert MODULE.main(["--config", str(config_path)]) == 2


def test_cpu_fake_canary_builds_and_strictly_validates_terminal_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    MODULE.validate_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    assert receipt["status"] == "PASS"
    assert receipt["completed_steps"] == 512
    assert receipt["rng_post_draw_sha256"] == receipt["rng_restored_sha256"]
    assert receipt["raw_backbone_pre_sha256"] == receipt["raw_backbone_post_sha256"]


def test_cuda_canary_normalizes_bare_torch_uuid_to_nvidia_authority() -> None:
    bare = "20253fc3-16c0-a26a-579e-ee0adf958974"

    assert MODULE.canonical_cuda_device_uuid(bare) == f"GPU-{bare}"
    assert MODULE.canonical_cuda_device_uuid(f"GPU-{bare}") == f"GPU-{bare}"
    with pytest.raises(ValueError, match="UUID differs"):
        MODULE.canonical_cuda_device_uuid("NVIDIA GB10")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "SKIP"),
        ("completed_steps", 511),
        ("device_uuid", "GPU-wrong"),
        ("environment_sha256", "0" * 64),
        ("final_loss", float("nan")),
        ("peak_reserved_bytes", True),
    ],
)
def test_canary_rejects_terminal_mutations(
    tmp_path: Path, key: str, value: object
) -> None:
    config = _config(tmp_path)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    receipt[key] = value
    with pytest.raises(ValueError):
        MODULE.validate_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )


def test_canary_publication_is_config_derived_no_replace_and_strict_reload(
    tmp_path: Path
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    output = MODULE.publish_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    assert output == root / "preflight" / "cuda_canary_v1.json"
    original = output.stat()
    original_payload = output.read_bytes()
    assert json.loads(original_payload) == receipt
    adopted = MODULE.publish_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    assert adopted == output
    assert output.read_bytes() == original_payload
    assert output.stat().st_ino == original.st_ino


def test_canary_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    real = tmp_path / "real"
    real.mkdir()
    root.mkdir()
    (root / "preflight").symlink_to(real, target_is_directory=True)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    with pytest.raises(ValueError, match="path"):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )


def test_canary_backend_must_report_real_cuda_and_never_skips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _materialize_direct_canary_authorities(config)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE.run_cuda_canary(config, backend=lambda _config: {"cuda": False})


def test_canary_racing_destination_is_never_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    output = root / "preflight" / "cuda_canary_v1.json"
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    publication = importlib.import_module("sfora.atomic_publication")
    original_link = publication._link_fd_noreplace

    def racing_link(descriptor: int, directory: int, name: str) -> None:
        output.write_bytes(b"racer")
        original_link(descriptor, directory, name)

    monkeypatch.setattr(publication, "_link_fd_noreplace", racing_link)
    with pytest.raises(FileExistsError):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )
    assert output.read_bytes() == b"racer"


def test_review2_canary_racing_temporary_is_never_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    temporary = root / "preflight" / ".cuda_canary_v1.json.tmp"
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    temporary.write_bytes(b"racer-temporary")
    with pytest.raises(ValueError, match="unregistered"):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )
    assert not (root / "preflight/cuda_canary_v1.json").exists()
    assert temporary.read_bytes() == b"racer-temporary"


def test_canary_requires_external_input_device_environment_and_task1_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(MODULE, "authenticate_canary_inputs")
    assert hasattr(MODULE, "run_registered_fepf_canary")
    config = _config(tmp_path)
    root = _materialize_direct_canary_authorities(config)
    monkeypatch.setattr(
        MODULE, "validate_registered_canary_family", lambda *_args, **_kwargs: {}
    )
    called = False

    def backend(_config: dict[str, object]) -> dict[str, object]:
        nonlocal called
        called = True
        return _backend_observation()

    assert MODULE.run_cuda_canary(config, backend=backend).is_file()
    assert called
    output = root / "preflight" / "cuda_canary_v1.json"
    output.unlink()
    Path(config["inputs"]["checkpoint"]).write_bytes(b"substituted\n")
    called = False
    with pytest.raises(ValueError, match="checkpoint authority"):
        MODULE.run_cuda_canary(config, backend=backend)
    assert called is False


def test_review2_canary_requires_registered_model_and_observed_provenance() -> None:
    calls: list[object] = []

    class Model:
        def to(self, device: object) -> Model:
            calls.append(("device", device))
            return self

    class Trainer:
        @staticmethod
        def _git_revision(checkout: Path) -> str:
            calls.append(("revision", checkout))
            return "d" * 40

        @staticmethod
        def _load_official_model(checkout: Path, checkpoint: Path):
            calls.append(("load", checkout, checkpoint))
            return Model(), "registered-transform"

        @staticmethod
        def raw_backbone_state_sha256(model: object) -> str:
            calls.append(("hash", model))
            return "a" * 64

        @staticmethod
        def _restore_global_rng_snapshot(snapshot: object) -> None:
            calls.append(("restore", snapshot))

        @staticmethod
        def _fepf_rng_audit(entry: object, post_draw: object) -> str:
            calls.append(("audit", entry, post_draw))
            return "registered-audit"

    config = {
        "model": {"revision": "d" * 40},
        "inputs": {
            "unicom_checkout": "/registered/unicom",
            "checkpoint": "/registered/checkpoint.bin",
        }
    }
    model, transform, digest = MODULE.load_registered_canary_model(
        config, trainer=Trainer, device="cuda:registered"
    )
    assert transform == "registered-transform"
    assert digest == "a" * 64
    assert calls[:4] == [
        ("revision", Path("/registered/unicom")),
        (
            "load",
            Path("/registered/unicom"),
            Path("/registered/checkpoint.bin"),
        ),
        ("device", "cuda:registered"),
        ("hash", model),
    ]
    assert MODULE.capture_canary_rng_audit(Trainer, "entry", "post") == (
        "registered-audit"
    )
    assert calls[-2:] == [("restore", "post"), ("audit", "entry", "post")]
    assert MODULE.validate_canary_environment_payload(
        _environment(), _environment_sha256()
    )
    mutated = _environment()
    mutated["device_uuid"] = "GPU-substituted"
    with pytest.raises(ValueError, match="environment"):
        MODULE.validate_canary_environment_payload(mutated, _environment_sha256())


def test_review11_existing_preflight_does_not_mask_foreign_namespace(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    (root / "foreign.bin").write_bytes(b"foreign")

    with pytest.raises(ValueError, match="foreign|namespace"):
        MODULE.ensure_campaign_root(config, physical_admission=False)
