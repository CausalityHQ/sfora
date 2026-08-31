"""Tests for the authenticated local SigLIP RSTA Stage-A CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from torch import nn

from sfora.pass209_m4 import canonical_json_bytes
from sfora.siglip_proxy_control import PooledProxyAnchorModel
from sfora.siglip_rsta_stage_a import (
    RstaCheckpointBinding,
    RstaControlBinding,
    RstaJvpBackendEvidence,
    RstaReceiverEvidence,
    RstaReceiverScore,
    rsta_control_binding_bytes,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_rsta_stage_a.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_siglip_rsta_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_observed_siglip_tower_confines_autocast_and_reports_policy() -> None:
    class FakeVision(nn.Module):
        def forward(self, *, pixel_values: torch.Tensor, return_dict: bool):
            assert return_dict is True
            return type("VisionOutput", (), {"pooler_output": pixel_values[:, :2]})()

    tower = _MODULE.StageASiglipPooledTower(FakeVision())
    output = tower(torch.ones(3, 4, dtype=torch.float32))

    assert output.dtype == torch.float32
    assert tower.rsta_autocast_evidence() == ("cpu", "float32", False)


def test_siglip_runtime_loads_only_pinned_local_eager_components(tmp_path: Path) -> None:
    from sfora.siglip_proxy_control import SiglipProxyControlConfig

    config = SiglipProxyControlConfig()
    snapshot = tmp_path / config.model_revision
    snapshot.mkdir()
    calls: list[tuple[str, str, object]] = []

    class FakeVision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"_attn_implementation": "eager"})()
            self.is_gradient_checkpointing = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls.append(("vision", path, kwargs))
            return cls()

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            assert gradient_checkpointing_kwargs == {"use_reentrant": False}
            self.is_gradient_checkpointing = True

        def gradient_checkpointing_disable(self):
            self.is_gradient_checkpointing = False

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls.append(("processor", path, kwargs))
            return cls()

    runtime = _MODULE.load_stage_a_siglip_runtime(
        snapshot_resolver=lambda *_args, **_kwargs: str(snapshot),
        vision_model_cls=FakeVision,
        processor_cls=FakeProcessor,
    )
    model = runtime.model_factory()

    assert type(model) is PooledProxyAnchorModel
    assert model.tower.vision_model.is_gradient_checkpointing is True
    assert runtime.checkpointing_enabled(model) is True
    runtime.disable_checkpointing(model)
    assert runtime.checkpointing_enabled(model) is False
    assert calls == [
        ("processor", str(snapshot), {"local_files_only": True}),
        (
            "vision",
            str(snapshot),
            {"local_files_only": True, "attn_implementation": "eager"},
        ),
    ]


def test_resource_sampling_parses_named_psi_field_and_returns_endpoint_growth(monkeypatch) -> None:
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path: (
            "some avg10=0.00 avg60=0.00 total=0\nfull avg60=0.10 avg10=0.25 avg300=0.05 total=10\n"
        ),
    )
    assert _MODULE._memory_psi_full_avg10() == 0.25

    samples = iter([(0.0, 100), (0.25, 110), (0.0, 105)])
    monitor = _MODULE.StageAResourceMonitor(lambda: next(samples))
    monitor.observe()
    monitor.observe()
    observation = monitor.finish()

    assert observation.memory_psi_growth_ppm == 0
    assert observation.swap_growth_bytes == 5


def test_resource_monitor_samples_during_science_until_explicit_finish() -> None:
    reached_peak = threading.Event()
    calls = 0

    def sample() -> tuple[float, int]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            reached_peak.set()
            return 0.5, 256
        return 0.0, 0

    monitor = _MODULE.StageAResourceMonitor(sample)
    monitor.start(interval_seconds=0.001)
    assert reached_peak.wait(1.0)
    observation = monitor.finish()

    assert observation.memory_psi_growth_ppm == 500_000
    assert observation.swap_growth_bytes == 256
    observed_calls = calls
    assert reached_peak.wait(0.01)
    assert calls == observed_calls


def _valid_argv() -> list[str]:
    return [
        "--control-binding",
        "/authority/control-binding.json",
        "--control-binding-sha256",
        "11" * 32,
        "--checkpoint-seed17",
        "/authority/seed17.pt",
        "--checkpoint-seed29",
        "/authority/seed29.pt",
        "--checkpoint-seed43",
        "/authority/seed43.pt",
        "--optimization-manifest",
        "/authority/optimization.json",
        "--optimization-manifest-sha256",
        "22" * 32,
        "--image-root",
        "/authority/images",
        "--execute-stage-a",
    ]


def _fixture_image_basename(example_id: str = "optimization-000") -> str:
    return (
        hashlib.sha256(b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")).hexdigest()
        + ".image"
    )


def _scientific_examples() -> tuple[tuple[str, int], ...]:
    return tuple(
        (f"class-{label:02d}-row-{row:02d}", label) for label in range(49) for row in range(15)
    )


def _scientific_score(delta: float) -> RstaReceiverScore:
    rho = 0.3
    cosine = math.sqrt(1.0 - rho * rho)
    norm_b = math.exp(0.2)
    cross_norm = math.sqrt(norm_b**2 + 1.0 - 2.0 * norm_b * cosine)
    return RstaReceiverScore(
        a_self=delta,
        a_batch=0.0,
        delta=delta,
        a_desc=delta - 0.02,
        self_minus_desc=0.02,
        cos_batch_self=cosine,
        rho=rho,
        log_ratio=0.2,
        cross_contribution=-delta / cross_norm,
        random_a_self=0.0,
        random_a_batch=0.0,
        random_delta=0.0,
        deranged_a_self=0.0,
        deranged_a_batch=0.0,
        deranged_delta=0.0,
        norm_z=1.0,
        norm_dbar=1.0,
        norm_b=norm_b,
        norm_s=1.0,
        norm_q=1.0,
        norm_random_target=1.0,
        norm_deranged_target=1.0,
        batch_radial_fraction=0.0,
        self_radial_fraction=0.0,
        dbar_radial_fraction=0.0,
    )


def test_parser_accepts_only_complete_local_authority() -> None:
    parsed = _MODULE.parse_stage_a_args(_valid_argv())

    assert parsed.control_binding == Path("/authority/control-binding.json")
    assert parsed.checkpoint_seed17 == Path("/authority/seed17.pt")
    assert parsed.checkpoint_seed29 == Path("/authority/seed29.pt")
    assert parsed.checkpoint_seed43 == Path("/authority/seed43.pt")
    assert parsed.optimization_manifest == Path("/authority/optimization.json")
    assert parsed.image_root == Path("/authority/images")
    assert parsed.execute_stage_a is True


def test_main_writes_only_complete_canonical_cli_result(monkeypatch, capsys) -> None:
    observed = []

    def run(arguments):
        observed.append(arguments)
        return b'{"claim_eligible":false}\n'

    monkeypatch.setattr(_MODULE, "run_stage_a_cli", run, raising=False)
    assert _MODULE.main(_valid_argv()) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"claim_eligible":false}\n'
    assert captured.err == ""
    assert len(observed) == 1
    assert observed[0].execute_stage_a is True


@pytest.mark.parametrize(
    "clause",
    [
        "authority-mismatch",
        "backend-unavailable",
        "fixture-failure",
        "throughput-budget",
        "determinism-failure",
    ],
)
def test_main_emits_registered_invalid_receipt_for_pre_science_failure(
    clause: str, monkeypatch, capsys
) -> None:
    def fail(_arguments):
        raise _MODULE.PreScienceInvalid(clause, "registered pre-science failure")

    monkeypatch.setattr(_MODULE, "run_stage_a_cli", fail)

    assert _MODULE.main(_valid_argv()) == 0
    captured = capsys.readouterr()
    value = json.loads(captured.out)
    assert captured.out.encode() == canonical_json_bytes(value)
    assert value == {
        "claim_eligible": False,
        "first_decisive_clause": clause,
        "schema": "siglip-rsta-stage-a-result-v1",
        "verdict": "INVALID",
    }
    assert captured.err == ""


def test_main_never_converts_post_science_failure_into_candidate_result(
    monkeypatch, capsys
) -> None:
    def fail(_arguments):
        raise _MODULE.PostScienceFailure("scientific row failed")

    monkeypatch.setattr(_MODULE, "run_stage_a_cli", fail)

    with pytest.raises(_MODULE.PostScienceFailure, match="scientific row failed"):
        _MODULE.main(_valid_argv())
    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_classifies_local_authority_failure_before_runtime_construction(capsys) -> None:
    assert _MODULE.main(_valid_argv()) == 0
    captured = capsys.readouterr()
    value = json.loads(captured.out)
    assert captured.out.encode() == canonical_json_bytes(value)
    assert value["verdict"] == "INVALID"
    assert value["first_decisive_clause"] == "authority-mismatch"
    assert captured.err == ""


def test_run_cli_executes_fixed_cuda_campaign_and_binds_measured_resources(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path / "authority")
    campaign = _campaign(tmp_path / "campaign")
    tensor_cache = _tensor_cache(authority)
    captured: dict[str, object] = {}

    class FakeMonitor:
        def __init__(self, sampler) -> None:
            assert callable(sampler)
            self.started = False

        def start(self) -> None:
            self.started = True

        def observe(self) -> None:
            assert self.started is True

        def finish(self):
            assert self.started is True
            return _MODULE.StageAResourceObservation(0, 0)

    runtime = _MODULE.StageASiglipRuntime(
        processor=object(),
        model_factory=lambda: object(),
        checkpointing_enabled=lambda _model: True,
        disable_checkpointing=lambda _model: None,
    )
    monkeypatch.setattr(_MODULE, "load_stage_a_authority", lambda _arguments: authority)
    monkeypatch.setattr(_MODULE, "load_stage_a_siglip_runtime", lambda: runtime)
    monkeypatch.setattr(_MODULE, "_stage_a_transforms", lambda _processor: (object(), object()))
    monkeypatch.setattr(_MODULE, "cache_stage_a_tensors", lambda *_args, **_kwargs: tensor_cache)
    monkeypatch.setattr(_MODULE, "StageAResourceMonitor", FakeMonitor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2_000_000)

    def execute(observed_authority, **kwargs):
        captured["authority"] = observed_authority
        captured["campaign_kwargs"] = kwargs
        return campaign

    def build(observed_campaign, **kwargs):
        captured["audit_campaign"] = observed_campaign
        captured["audit_kwargs"] = kwargs
        return _execution_audit()

    monkeypatch.setattr(_MODULE, "execute_stage_a_model_campaign", execute)
    monkeypatch.setattr(_MODULE, "build_stage_a_execution_audit", build)
    monkeypatch.setattr(
        _MODULE,
        "stage_a_scientific_result_bytes",
        lambda observed_campaign, audit: (
            b'{"claim_eligible":false}\n'
            if observed_campaign is campaign and audit == _execution_audit()
            else pytest.fail("scientific result inputs differ")
        ),
    )

    result = _MODULE.run_stage_a_cli(_MODULE.parse_stage_a_args(_valid_argv()))

    assert result == b'{"claim_eligible":false}\n'
    assert captured["authority"] is authority
    assert captured["audit_campaign"] is campaign
    assert captured["campaign_kwargs"]["tensor_cache"] is tensor_cache
    assert captured["campaign_kwargs"]["device"] == torch.device("cuda")
    assert captured["campaign_kwargs"]["input_shape"] == (3, 384, 384)
    assert captured["audit_kwargs"]["peak_cuda_bytes"] == 2_000_000
    assert captured["audit_kwargs"]["memory_psi_growth_ppm"] == 0
    assert captured["audit_kwargs"]["swap_growth_bytes"] == 0


def test_run_cli_classifies_resource_monitor_setup_failure_before_science(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path)
    tensor_cache = _tensor_cache(authority)
    runtime = _MODULE.StageASiglipRuntime(
        processor=object(),
        model_factory=lambda: object(),
        checkpointing_enabled=lambda _model: True,
        disable_checkpointing=lambda _model: None,
    )
    monkeypatch.setattr(_MODULE, "load_stage_a_authority", lambda _arguments: authority)
    monkeypatch.setattr(_MODULE, "load_stage_a_siglip_runtime", lambda: runtime)
    monkeypatch.setattr(_MODULE, "_stage_a_transforms", lambda _processor: (object(), object()))
    monkeypatch.setattr(
        _MODULE,
        "cache_stage_a_tensors",
        lambda *_args, **_kwargs: tensor_cache,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        _MODULE,
        "StageAResourceMonitor",
        lambda _sampler: (_ for _ in ()).throw(RuntimeError("PSI unavailable")),
    )

    with pytest.raises(_MODULE.PreScienceInvalid) as captured:
        _MODULE.run_stage_a_cli(_MODULE.parse_stage_a_args(_valid_argv()))
    assert captured.value.clause == "authority-mismatch"
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "PSI unavailable"


def test_run_cli_classifies_resource_monitor_failure_after_science(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path / "authority")
    tensor_cache = _tensor_cache(authority)
    campaign = _campaign(tmp_path / "campaign")
    state = {"finished": 0}

    class FakeMonitor:
        def __init__(self, sampler) -> None:
            assert callable(sampler)

        def start(self) -> None:
            pass

        def observe(self) -> None:
            raise RuntimeError("PSI read failed")

        def finish(self):
            state["finished"] += 1
            return _MODULE.StageAResourceObservation(0, 0)

    runtime = _MODULE.StageASiglipRuntime(
        processor=object(),
        model_factory=lambda: object(),
        checkpointing_enabled=lambda _model: True,
        disable_checkpointing=lambda _model: None,
    )
    monkeypatch.setattr(_MODULE, "load_stage_a_authority", lambda _arguments: authority)
    monkeypatch.setattr(_MODULE, "load_stage_a_siglip_runtime", lambda: runtime)
    monkeypatch.setattr(_MODULE, "_stage_a_transforms", lambda _processor: (object(), object()))
    monkeypatch.setattr(
        _MODULE,
        "cache_stage_a_tensors",
        lambda *_args, **_kwargs: tensor_cache,
    )
    monkeypatch.setattr(_MODULE, "StageAResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        _MODULE, "execute_stage_a_model_campaign", lambda *_args, **_kwargs: campaign
    )
    monkeypatch.setattr(
        _MODULE,
        "build_stage_a_execution_audit",
        lambda *_args, **_kwargs: pytest.fail("post-science monitor failure reached audit"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)

    with pytest.raises(_MODULE.PostScienceFailure) as captured:
        _MODULE.run_stage_a_cli(_MODULE.parse_stage_a_args(_valid_argv()))
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "PSI read failed"
    assert state["finished"] == 1


def test_run_cli_rejects_nonzero_endpoint_resource_growth_after_science(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path / "authority")
    tensor_cache = _tensor_cache(authority)
    campaign = _campaign(tmp_path / "campaign")

    class FakeMonitor:
        def __init__(self, sampler) -> None:
            assert callable(sampler)

        def start(self) -> None:
            pass

        def observe(self) -> None:
            pass

        def finish(self):
            return _MODULE.StageAResourceObservation(1, 0)

    runtime = _MODULE.StageASiglipRuntime(
        processor=object(),
        model_factory=lambda: object(),
        checkpointing_enabled=lambda _model: True,
        disable_checkpointing=lambda _model: None,
    )
    monkeypatch.setattr(_MODULE, "load_stage_a_authority", lambda _arguments: authority)
    monkeypatch.setattr(_MODULE, "load_stage_a_siglip_runtime", lambda: runtime)
    monkeypatch.setattr(_MODULE, "_stage_a_transforms", lambda _processor: (object(), object()))
    monkeypatch.setattr(
        _MODULE,
        "cache_stage_a_tensors",
        lambda *_args, **_kwargs: tensor_cache,
    )
    monkeypatch.setattr(_MODULE, "StageAResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        _MODULE, "execute_stage_a_model_campaign", lambda *_args, **_kwargs: campaign
    )
    monkeypatch.setattr(
        _MODULE,
        "build_stage_a_execution_audit",
        lambda *_args, **_kwargs: pytest.fail("resource growth reached audit"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)

    with pytest.raises(_MODULE.PostScienceFailure, match="resource growth"):
        _MODULE.run_stage_a_cli(_MODULE.parse_stage_a_args(_valid_argv()))


@pytest.mark.parametrize(
    "flag",
    [
        "--control-binding",
        "--control-binding-sha256",
        "--checkpoint-seed17",
        "--checkpoint-seed29",
        "--checkpoint-seed43",
        "--optimization-manifest",
        "--optimization-manifest-sha256",
        "--image-root",
        "--execute-stage-a",
    ],
)
def test_parser_rejects_every_missing_authority(flag: str) -> None:
    argv = _valid_argv()
    index = argv.index(flag)
    del argv[index : index + (1 if flag == "--execute-stage-a" else 2)]

    with pytest.raises(SystemExit):
        _MODULE.parse_stage_a_args(argv)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--s3-uri", "s3://forbidden"),
        ("--url", "https://forbidden.invalid"),
        ("--clean-root", "/forbidden/clean"),
        ("--burned-root", "/forbidden/burned"),
        ("--test-root", "/forbidden/test"),
        ("--alternate-checkpoint", "/forbidden/alternate.pt"),
        ("--threshold", "0.1"),
        ("--seed", "17"),
        ("--backend", "double-backward"),
    ],
)
def test_parser_rejects_forbidden_override_flags(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _MODULE.parse_stage_a_args([*_valid_argv(), flag, value])


def test_parser_rejects_duplicate_and_noncanonical_digest() -> None:
    with pytest.raises(SystemExit):
        _MODULE.parse_stage_a_args([*_valid_argv(), "--image-root", "/authority/other-images"])

    argv = _valid_argv()
    argv[argv.index("--control-binding-sha256") + 1] = "AA" * 32
    with pytest.raises(SystemExit):
        _MODULE.parse_stage_a_args(argv)


def _write_authority_bundle(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
    checkpoint_authorities: list[RstaCheckpointBinding] = []
    config_sha256 = "33" * 32
    run_authority_sha256 = "44" * 32
    for seed in (17, 29, 43):
        path = tmp_path / f"seed-{seed}.pt"
        torch.save(
            {
                "claim_eligible": False,
                "completed_epoch": 60,
                "config_sha256": config_sha256,
                "cpu_rng_state": torch.random.get_rng_state(),
                "cuda_rng_states": (),
                "final_objective": 1.0,
                "initial_snapshot_sha256": "55" * 32,
                "maximum_score_disagreement": 0.0,
                "model_state": OrderedDict((("weight", torch.tensor([float(seed)])),)),
                "optimizer_state": {"state": {}, "param_groups": []},
                "run_authority_sha256": run_authority_sha256,
                "sampler_cycles": (0,) * 49,
                "sampler_positions": (0,) * 49,
                "schema": "sfora-siglip-proxy-checkpoint-payload-v1",
                "seed": seed,
            },
            path,
        )
        raw = path.read_bytes()
        checkpoint_paths.append(path)
        checkpoint_authorities.append(
            RstaCheckpointBinding(
                seed=seed,
                sha256=hashlib.sha256(raw).hexdigest(),
                byte_length=len(raw),
            )
        )

    manifest = canonical_json_bytes(
        {
            "schema": "rsta-optimization-manifest-v1",
            "claim_eligible": False,
            "dataset_id": "tanganke/stanford_cars",
            "dataset_revision": "66" * 20,
            "examples": [
                {
                    "example_id": "optimization-000",
                    "label": 0,
                }
            ],
        }
    )
    manifest_path = tmp_path / "optimization.json"
    manifest_path.write_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    binding = RstaControlBinding(
        schema="rsta-control-binding-v1",
        claim_eligible=False,
        control_complete=True,
        source_commit="77" * 20,
        config_sha256=config_sha256,
        run_authority_sha256=run_authority_sha256,
        dataset_id="tanganke/stanford_cars",
        dataset_revision="66" * 20,
        environment_sha256="88" * 32,
        optimization_manifest_sha256=manifest_sha256,
        selected_microbatch_size=120,
        checkpoints=tuple(checkpoint_authorities),
    )
    binding_bytes = rsta_control_binding_bytes(binding)
    binding_path = tmp_path / "control-binding.json"
    binding_path.write_bytes(binding_bytes)
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True)
    image_basename = _fixture_image_basename()
    (image_root / image_basename).write_bytes(b"fixture-image")
    argv = [
        "--control-binding",
        str(binding_path),
        "--control-binding-sha256",
        hashlib.sha256(binding_bytes).hexdigest(),
        "--checkpoint-seed17",
        str(checkpoint_paths[0]),
        "--checkpoint-seed29",
        str(checkpoint_paths[1]),
        "--checkpoint-seed43",
        str(checkpoint_paths[2]),
        "--optimization-manifest",
        str(manifest_path),
        "--optimization-manifest-sha256",
        manifest_sha256,
        "--image-root",
        str(image_root),
        "--execute-stage-a",
    ]
    return _MODULE.parse_stage_a_args(argv), binding


def test_authority_loader_returns_only_bound_model_states_and_preserves_rng(
    tmp_path: Path,
) -> None:
    arguments, binding = _write_authority_bundle(tmp_path)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    loaded = _MODULE.load_stage_a_authority(arguments)

    assert loaded.binding == binding
    assert tuple(item.seed for item in loaded.checkpoints) == (17, 29, 43)
    assert [float(item.model_state["weight"][0]) for item in loaded.checkpoints] == [
        17.0,
        29.0,
        43.0,
    ]
    assert all(not hasattr(item, "optimizer_state") for item in loaded.checkpoints)
    assert all(not hasattr(item, "final_objective") for item in loaded.checkpoints)
    assert all(not hasattr(item, "maximum_score_disagreement") for item in loaded.checkpoints)
    assert loaded.example_ids == ("optimization-000",)
    assert loaded.labels == (0,)
    assert loaded.image_paths == (tmp_path / "images" / _fixture_image_basename(),)
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_parser_rejects_production_and_equals_form_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_siglip_rsta_stage_a.py",
            *_valid_argv(),
            "--image-root=/authority/other-images",
        ],
    )
    with pytest.raises(SystemExit) as production:
        _MODULE.parse_stage_a_args()
    assert production.value.code == 2

    with pytest.raises(SystemExit) as explicit:
        _MODULE.parse_stage_a_args([*_valid_argv(), "--image-root=/authority/other-images"])
    assert explicit.value.code == 2


def test_checkpoint_deserializes_the_authenticated_bytes_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _binding = _write_authority_bundle(tmp_path)
    original_load = torch.load
    observed: list[object] = []

    def recording_load(source, *args, **kwargs):
        observed.append(source)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    _MODULE.load_stage_a_authority(arguments)

    assert len(observed) == 3
    assert all(hasattr(source, "read") and not isinstance(source, Path) for source in observed)


def test_authority_loader_rejects_intermediate_image_symlink_escape(
    tmp_path: Path,
) -> None:
    arguments, _binding = _write_authority_bundle(tmp_path)
    image_path = next(arguments.image_root.iterdir())
    image_path.unlink()
    forbidden = tmp_path / "clean-validation"
    forbidden.mkdir()
    forbidden_image = forbidden / "image.jpg"
    forbidden_image.write_bytes(b"forbidden-image")
    image_path.symlink_to(forbidden_image)

    with pytest.raises(ValueError, match="image path escapes authority"):
        _MODULE.load_stage_a_authority(arguments)


def test_parser_rejects_unnormalized_absolute_path() -> None:
    argv = _valid_argv()
    argv[argv.index("--image-root") + 1] = "/authority/../clean-validation"
    with pytest.raises(SystemExit):
        _MODULE.parse_stage_a_args(argv)


def test_authority_loader_rejects_checkpoint_and_manifest_drift(tmp_path: Path) -> None:
    arguments, _binding = _write_authority_bundle(tmp_path)
    arguments.checkpoint_seed29.write_bytes(arguments.checkpoint_seed29.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="checkpoint digest or length"):
        _MODULE.load_stage_a_authority(arguments)

    arguments, _binding = _write_authority_bundle(tmp_path / "second")
    arguments.optimization_manifest.write_bytes(
        arguments.optimization_manifest.read_bytes().replace(
            b"optimization-000", b"clean-validation"
        )
    )
    with pytest.raises(ValueError, match="optimization manifest digest"):
        _MODULE.load_stage_a_authority(arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_epoch", 59),
        ("seed", 17),
        ("claim_eligible", True),
        ("config_sha256", "99" * 32),
        ("model_state", {}),
        ("optimizer_state", []),
    ],
)
def test_authority_loader_rejects_rebound_checkpoint_semantic_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    arguments, binding = _write_authority_bundle(tmp_path)
    path = arguments.checkpoint_seed29
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    raw = path.read_bytes()
    authorities = list(binding.checkpoints)
    authorities[1] = RstaCheckpointBinding(
        seed=29,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
    )
    rebound = replace(binding, checkpoints=tuple(authorities))
    rebound_bytes = rsta_control_binding_bytes(rebound)
    arguments.control_binding.write_bytes(rebound_bytes)
    arguments.control_binding_sha256 = hashlib.sha256(rebound_bytes).hexdigest()

    with pytest.raises(ValueError, match="checkpoint authority"):
        _MODULE.load_stage_a_authority(arguments)


@pytest.mark.parametrize("mutation", ["forbidden-role", "path-field", "bool-label"])
def test_authority_loader_rejects_rebound_manifest_semantic_drift(
    tmp_path: Path, mutation: str
) -> None:
    arguments, binding = _write_authority_bundle(tmp_path)
    manifest = json.loads(arguments.optimization_manifest.read_bytes())
    if mutation == "forbidden-role":
        manifest["clean_validation"] = []
    elif mutation == "path-field":
        manifest["examples"][0]["relative_path"] = "../clean/image.jpg"
    else:
        manifest["examples"][0]["label"] = False
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    arguments.optimization_manifest.write_bytes(manifest_bytes)
    arguments.optimization_manifest_sha256 = manifest_sha256
    rebound = replace(binding, optimization_manifest_sha256=manifest_sha256)
    rebound_bytes = rsta_control_binding_bytes(rebound)
    arguments.control_binding.write_bytes(rebound_bytes)
    arguments.control_binding_sha256 = hashlib.sha256(rebound_bytes).hexdigest()

    with pytest.raises(ValueError, match=r"optimization (manifest|image)"):
        _MODULE.load_stage_a_authority(arguments)


def _scientific_authority(tmp_path: Path):
    arguments, binding = _write_authority_bundle(tmp_path)
    examples = _scientific_examples()
    for path in arguments.image_root.iterdir():
        path.unlink()
    for example_id, _label in examples:
        (arguments.image_root / _fixture_image_basename(example_id)).write_bytes(b"fixture-image")
    manifest = canonical_json_bytes(
        {
            "schema": "rsta-optimization-manifest-v1",
            "claim_eligible": False,
            "dataset_id": binding.dataset_id,
            "dataset_revision": binding.dataset_revision,
            "examples": [
                {"example_id": example_id, "label": label} for example_id, label in examples
            ],
        }
    )
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    arguments.optimization_manifest.write_bytes(manifest)
    arguments.optimization_manifest_sha256 = manifest_sha256
    rebound = replace(binding, optimization_manifest_sha256=manifest_sha256)
    rebound_bytes = rsta_control_binding_bytes(rebound)
    arguments.control_binding.write_bytes(rebound_bytes)
    arguments.control_binding_sha256 = hashlib.sha256(rebound_bytes).hexdigest()
    return _MODULE.load_stage_a_authority(arguments)


def _seed_execution(checkpoint, panel):
    rows = tuple(
        RstaReceiverEvidence(
            seed=checkpoint.seed,
            label=receiver.label,
            receiver_id=receiver.example_id,
            primary=_scientific_score(0.05),
            alternate=_scientific_score(0.02),
        )
        for receiver in panel.receivers
    )
    return _MODULE.StageASeedExecution(
        receiver_evidence=rows,
        first_receiver_first_sha256="91" * 32,
        first_receiver_repeat_sha256="91" * 32,
        parameter_names=("projection.weight", "tower.weight"),
        parameter_numels=(16, 64),
        logical_batch_replays=6,
        receiver_actions=474,
        autocast_device_type="cuda",
        autocast_dtype="bfloat16",
        autocast_enabled=True,
        support_replays=2,
        module_training=True,
        gradient_checkpointing_enabled=False,
        torch_compile_enabled=False,
        attention_implementation="eager",
    )


def _forward_backend() -> RstaJvpBackendEvidence:
    return RstaJvpBackendEvidence(
        backend="forward-mode",
        comparison_available=True,
        maximum_relative_disagreement=0.0,
        forward_error=None,
    )


def _tensor_cache(authority):
    from sfora.siglip_rsta_stage_a import select_rsta_roles

    panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))

    def transform(source: str) -> torch.Tensor:
        raw = hashlib.sha256(source.encode()).digest()
        return torch.tensor([(raw[index] - 127.5) / 127.5 for index in range(4)])

    return _MODULE.cache_stage_a_tensors(
        authority,
        panel,
        graph_transform=transform,
        evaluation_transform=transform,
        materialize=lambda path: path.name,
    )


def test_reduced_scientific_loop_runs_exact_seed_panel_order_and_aggregates(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    calls: list[int] = []

    cache = _tensor_cache(authority)

    def runner(checkpoint, panel, binding, observed_cache, backend):
        calls.append(checkpoint.seed)
        assert binding is authority.binding
        assert observed_cache is cache
        assert backend is sealed_backend
        return _seed_execution(checkpoint, panel)

    sealed_backend = _forward_backend()
    completed = _MODULE.execute_stage_a_scientific_loop(
        authority,
        backend=sealed_backend,
        tensor_cache=cache,
        seed_runner=runner,
    )

    assert calls == [17, 29, 43]
    assert completed.aggregate.receiver_count == 3 * 49 * 3
    assert completed.aggregate.verdict == "PASS_ONWARD"
    assert completed.backend is sealed_backend
    assert completed.authority is authority
    assert completed.aggregate_bytes.endswith(b"\n")


@pytest.mark.parametrize("mutation", ["short", "duplicate", "reordered", "repeat"])
def test_reduced_scientific_loop_rejects_partial_duplicate_order_and_repeatability_drift(
    tmp_path: Path, mutation: str
) -> None:
    authority = _scientific_authority(tmp_path)

    calls: list[int] = []

    def runner(checkpoint, panel, _binding, _cache, _backend):
        calls.append(checkpoint.seed)
        execution = _seed_execution(checkpoint, panel)
        rows = execution.receiver_evidence
        if mutation == "short":
            rows = rows[:-1]
        elif mutation == "duplicate":
            rows = (rows[0], rows[0], *rows[2:])
        elif mutation == "reordered":
            rows = (rows[1], rows[0], *rows[2:])
        elif mutation == "repeat":
            return replace(
                execution,
                first_receiver_repeat_sha256="92" * 32,
            )
        return replace(execution, receiver_evidence=tuple(rows))

    with pytest.raises(_MODULE.PostScienceFailure, match="seed execution"):
        _MODULE.execute_stage_a_scientific_loop(
            authority,
            backend=_forward_backend(),
            tensor_cache=_tensor_cache(authority),
            seed_runner=runner,
        )
    if mutation in {"short", "duplicate", "reordered", "repeat"}:
        assert calls == [17]


@pytest.mark.parametrize("mutation", ["seed-order", "image-count", "tensor-digest", "backend"])
def test_reduced_scientific_loop_rejects_pre_science_authority_drift(
    tmp_path: Path, mutation: str
) -> None:
    authority = _scientific_authority(tmp_path)
    backend = _forward_backend()
    cache = _tensor_cache(authority)
    if mutation == "seed-order":
        authority = replace(authority, checkpoints=tuple(reversed(authority.checkpoints)))
    elif mutation == "image-count":
        authority = replace(authority, image_paths=authority.image_paths[:-1])
    elif mutation == "tensor-digest":
        values = dict(cache.tensor_sha256)
        values.pop(next(iter(values)))
        cache = replace(cache, tensor_sha256=values)
    else:
        backend = RstaJvpBackendEvidence(
            backend="double-backward",
            comparison_available=False,
            maximum_relative_disagreement=1.0e-6,
            forward_error="NotImplementedError",
        )

    with pytest.raises(_MODULE.PreScienceInvalid):
        _MODULE.execute_stage_a_scientific_loop(
            authority,
            backend=backend,
            tensor_cache=cache,
            seed_runner=_seed_execution,
        )


def test_reduced_scientific_loop_stops_after_first_mid_campaign_failure(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    calls: list[int] = []

    def runner(checkpoint, panel, _binding, _cache, _backend):
        calls.append(checkpoint.seed)
        if checkpoint.seed == 29:
            raise RuntimeError("interrupted receiver row")
        return _seed_execution(checkpoint, panel)

    with pytest.raises(_MODULE.PostScienceFailure):
        _MODULE.execute_stage_a_scientific_loop(
            authority,
            backend=_forward_backend(),
            tensor_cache=_tensor_cache(authority),
            seed_runner=runner,
        )
    assert calls == [17, 29]


@pytest.mark.parametrize(
    "backend",
    [
        RstaJvpBackendEvidence("unknown", True, 0.0, None),
        RstaJvpBackendEvidence("forward-mode", False, 0.0, None),
        RstaJvpBackendEvidence("forward-mode", True, 1.1e-5, None),
        RstaJvpBackendEvidence("forward-mode", True, 0.0, "RuntimeError"),
        RstaJvpBackendEvidence("double-backward", False, 0.0, None),
        RstaJvpBackendEvidence("double-backward", False, 1.0e-12, "NotImplementedError"),
    ],
)
def test_reduced_scientific_loop_rejects_every_unsealed_backend_shape(
    tmp_path: Path, backend: RstaJvpBackendEvidence
) -> None:
    authority = _scientific_authority(tmp_path)
    with pytest.raises(_MODULE.PreScienceInvalid):
        _MODULE.execute_stage_a_scientific_loop(
            authority,
            backend=backend,
            tensor_cache=_tensor_cache(authority),
            seed_runner=_seed_execution,
        )


def test_pre_science_invalid_result_is_canonical_and_has_no_scientific_evidence() -> None:
    raw = _MODULE.pre_science_invalid_result_bytes("backend-unavailable")
    value = json.loads(raw)

    assert raw == canonical_json_bytes(value)
    assert value == {
        "schema": "siglip-rsta-stage-a-result-v1",
        "claim_eligible": False,
        "verdict": "INVALID",
        "first_decisive_clause": "backend-unavailable",
    }
    assert "metrics" not in value
    assert "receiver_evidence" not in value

    with pytest.raises(ValueError, match="INVALID clause"):
        _MODULE.pre_science_invalid_result_bytes("adaptive-retry")


def test_tensor_cache_materializes_each_selected_role_once_and_preserves_rng(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    from sfora.siglip_rsta_stage_a import select_rsta_roles

    panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))
    support_ids = {example_id for pair in panel.support_ids_by_label for example_id in pair}
    calls: list[tuple[str, str]] = []

    def materialize(path: Path) -> str:
        return path.name

    def transform(role: str):
        def apply(source: str) -> torch.Tensor:
            calls.append((role, source))
            return torch.tensor(
                [random.random(), float(np.random.random()), float(torch.rand(()))],
                dtype=torch.float32,
            )

        return apply

    random.seed(501)
    np.random.seed(502)
    torch.manual_seed(503)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    cache = _MODULE.cache_stage_a_tensors(
        authority,
        panel,
        graph_transform=transform("graph"),
        evaluation_transform=transform("evaluation"),
        materialize=materialize,
    )

    required_ids = _MODULE.stage_a_tensor_example_ids(panel)
    assert cache.example_ids == required_ids
    assert set(cache.tensors) == set(required_ids)
    assert set(cache.tensor_sha256) == set(required_ids)
    assert len(calls) == len(required_ids)
    assert len({source for _role, source in calls}) == len(calls)
    assert all(
        role == ("evaluation" if example_id in support_ids else "graph")
        for (role, _source), example_id in zip(calls, required_ids, strict=True)
    )
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    first_receiver = panel.receivers[0].example_id
    assert sum(source == _fixture_image_basename(first_receiver) for _role, source in calls) == 1
    assert torch.equal(cache.batch((first_receiver,))[0], cache.tensors[first_receiver])


def _completed_science(tmp_path: Path):
    authority = _scientific_authority(tmp_path)

    def runner(checkpoint, panel, _binding, _cache, _backend):
        return _seed_execution(checkpoint, panel)

    return _MODULE.execute_stage_a_scientific_loop(
        authority,
        backend=_forward_backend(),
        tensor_cache=_tensor_cache(authority),
        seed_runner=runner,
    )


def _execution_audit():
    return _MODULE.StageAExecutionAudit(
        parameter_names=("projection.weight", "tower.weight"),
        parameter_numels=(16, 64),
        checkpointing_max_relative_disagreement=4.0e-7,
        fixture_sha256="a1" * 32,
        module_training=True,
        gradient_checkpointing_enabled=False,
        torch_compile_enabled=False,
        attention_implementation="eager",
        autocast_device_type="cuda",
        autocast_dtype="bfloat16",
        autocast_enabled=True,
        cublas_workspace_config=":4096:8",
        deterministic_algorithms_enabled=True,
        deterministic_algorithms_warn_only=False,
        cudnn_benchmark=False,
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        elapsed_ns=123_456,
        peak_rss_bytes=1_000_000,
        peak_cuda_bytes=2_000_000,
        memory_psi_growth_ppm=0,
        swap_growth_bytes=0,
    )


def _campaign(tmp_path: Path):
    completed = _completed_science(tmp_path)
    audit = _execution_audit()
    preflight = _MODULE.StageAModelPreflight(
        backend=completed.backend,
        checkpointing_max_relative_disagreement=0.0,
        fixture_sha256=audit.fixture_sha256,
        parameter_names=audit.parameter_names,
        parameter_numels=audit.parameter_numels,
        logical_batch_replay_ns=100,
        receiver_action_ns=10,
        support_forward_ns=20,
        projected_science_elapsed_ns=5_380,
        model_load_ns=50,
        preflight_elapsed_ns=1_000,
        projected_attempt_elapsed_ns=6_480,
    )
    second = replace(
        preflight,
        backend=replace(preflight.backend, maximum_relative_disagreement=1.0e-7),
        checkpointing_max_relative_disagreement=3.0e-7,
    )
    third = replace(
        preflight,
        backend=replace(preflight.backend, maximum_relative_disagreement=2.0e-7),
        checkpointing_max_relative_disagreement=4.0e-7,
    )
    return _MODULE.StageAModelCampaign(
        completed=completed,
        preflights=(preflight, second, third),
        determinism=_MODULE.StageADeterminismEvidence(
            cublas_workspace_config=audit.cublas_workspace_config,
            deterministic_algorithms_enabled=audit.deterministic_algorithms_enabled,
            deterministic_algorithms_warn_only=audit.deterministic_algorithms_warn_only,
            cudnn_benchmark=audit.cudnn_benchmark,
            cuda_matmul_allow_tf32=audit.cuda_matmul_allow_tf32,
            cudnn_allow_tf32=audit.cudnn_allow_tf32,
        ),
    )


def test_scientific_result_requires_observed_model_campaign(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model campaign"):
        _MODULE.stage_a_scientific_result_bytes(
            _completed_science(tmp_path),
            _execution_audit(),
        )


def test_execution_audit_is_derived_from_campaign_observations(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    audit = _MODULE.build_stage_a_execution_audit(
        campaign,
        elapsed_ns=123_456,
        peak_rss_bytes=1_000_000,
        peak_cuda_bytes=2_000_000,
        memory_psi_growth_ppm=0,
        swap_growth_bytes=0,
    )

    assert audit == _execution_audit()


def test_scientific_result_envelope_binds_authority_panel_tensors_backend_and_resources(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    completed = campaign.completed
    raw = _MODULE.stage_a_scientific_result_bytes(campaign, _execution_audit())
    value = json.loads(raw)

    assert raw == canonical_json_bytes(value)
    assert set(value) == {
        "schema",
        "claim_eligible",
        "authority",
        "role_panel",
        "tensor_sha256_by_id",
        "parameter_authority",
        "backend_preflight",
        "execution",
        "repeatability",
        "result",
    }
    assert value["schema"] == "siglip-rsta-stage-a-scientific-result-v1"
    assert value["claim_eligible"] is False
    assert (
        value["authority"]["control_binding_sha256"]
        == hashlib.sha256(rsta_control_binding_bytes(completed.authority.binding)).hexdigest()
    )
    assert value["authority"]["selected_microbatch_size"] == 120
    assert value["role_panel"]["receiver_count"] == 147
    assert len(value["tensor_sha256_by_id"]) == len(
        _MODULE.stage_a_tensor_example_ids(completed.panel)
    )
    assert value["parameter_authority"]["parameter_count"] == 2
    assert value["parameter_authority"]["parameter_numel"] == 80
    assert value["backend_preflight"]["backend"] == "forward-mode"
    assert value["backend_preflight"]["maximum_relative_disagreement"] == 2.0e-7
    assert value["backend_preflight"]["checkpointing_max_relative_disagreement"] == 4.0e-7
    assert value["backend_preflight"]["one_dgx_hour_budget_ns"] == 3_600_000_000_000
    assert value["backend_preflight"]["projected_science_elapsed_ns"] == 16_140
    assert value["backend_preflight"]["projected_attempt_elapsed_ns"] == 19_440
    assert [row["seed"] for row in value["backend_preflight"]["seed_timings"]] == [17, 29, 43]
    assert value["execution"]["logical_batch_replays"] == 18
    assert value["execution"]["receiver_actions"] == 1422
    assert value["execution"]["receiver_vjps"] == 1422
    assert value["execution"]["receiver_jvps"] == 2844
    assert value["execution"]["cublas_workspace_config"] == ":4096:8"
    assert value["execution"]["deterministic_algorithms_enabled"] is True
    assert value["execution"]["deterministic_algorithms_warn_only"] is False
    assert value["execution"]["cudnn_benchmark"] is False
    assert value["execution"]["cuda_matmul_allow_tf32"] is False
    assert value["execution"]["cudnn_allow_tf32"] is False
    assert value["result"] == json.loads(completed.aggregate_bytes)
    assert len(value["repeatability"]) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: replace(value, parameter_names=("projection.weight",)),
        lambda value: replace(value, checkpointing_max_relative_disagreement=1.1e-5),
        lambda value: replace(value, gradient_checkpointing_enabled=True),
        lambda value: replace(value, torch_compile_enabled=True),
        lambda value: replace(value, attention_implementation="sdpa"),
        lambda value: replace(value, autocast_enabled=False),
        lambda value: replace(value, deterministic_algorithms_enabled=False),
        lambda value: replace(value, cublas_workspace_config=":16:8"),
        lambda value: replace(value, elapsed_ns=0),
        lambda value: replace(value, elapsed_ns=3_600_000_000_001),
        lambda value: replace(value, peak_rss_bytes=0),
        lambda value: replace(value, memory_psi_growth_ppm=1),
        lambda value: replace(value, swap_growth_bytes=1),
    ],
)
def test_scientific_result_envelope_rejects_execution_authority_drift(
    tmp_path: Path, mutation
) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(ValueError, match="execution audit"):
        _MODULE.stage_a_scientific_result_bytes(campaign, mutation(_execution_audit()))


@pytest.mark.parametrize(
    "mutation", ["tensor", "backend", "repeat", "parameters", "roles", "aggregate"]
)
def test_scientific_result_envelope_revalidates_completed_science(
    tmp_path: Path, mutation: str
) -> None:
    campaign = _campaign(tmp_path)
    completed = campaign.completed
    if mutation == "tensor":
        values = dict(completed.tensor_sha256_by_id)
        values[next(iter(values))] = "not-a-digest"
        completed = replace(completed, tensor_sha256_by_id=values)
    elif mutation == "backend":
        completed = replace(
            completed,
            backend=RstaJvpBackendEvidence("unknown", True, 0.0, None),
        )
    elif mutation == "repeat":
        seeds = list(completed.seeds)
        seeds[0] = replace(seeds[0], first_receiver_repeat_sha256="93" * 32)
        completed = replace(completed, seeds=tuple(seeds))
    elif mutation == "parameters":
        seeds = list(completed.seeds)
        seeds[0] = replace(seeds[0], parameter_names=("projection.weight",))
        completed = replace(completed, seeds=tuple(seeds))
    elif mutation == "roles":
        seeds = list(completed.seeds)
        rows = list(seeds[0].receiver_evidence)
        rows[0] = replace(rows[0], receiver_id="unregistered-receiver")
        seeds[0] = replace(seeds[0], receiver_evidence=tuple(rows))
        completed = replace(completed, seeds=tuple(seeds))
    else:
        completed = replace(completed, aggregate_bytes=completed.aggregate_bytes + b" ")

    campaign = replace(campaign, completed=completed)
    with pytest.raises(ValueError, match=r"RSTA (completed|model campaign)"):
        _MODULE.stage_a_scientific_result_bytes(campaign, _execution_audit())


def test_real_toy_seed_runner_executes_both_panels_and_repeats_first_receiver(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    from sfora.siglip_rsta_stage_a import select_rsta_roles

    panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))

    def toy_transform(source: str) -> torch.Tensor:
        raw = hashlib.sha256(source.encode()).digest()
        return torch.tensor(
            [(raw[index] - 127.5) / 127.5 for index in range(4)], dtype=torch.float32
        )

    cache = _MODULE.cache_stage_a_tensors(
        authority,
        panel,
        graph_transform=toy_transform,
        evaluation_transform=toy_transform,
        materialize=lambda path: path.name,
    )
    torch.manual_seed(730)
    model = PooledProxyAnchorModel(
        tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
        input_dimensions=6,
        embedding_dimensions=4,
        class_count=49,
    ).train()
    checkpoint = replace(
        authority.checkpoints[0],
        model_state=MappingProxyType(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        ),
    )
    backend = RstaJvpBackendEvidence(
        backend="double-backward",
        comparison_available=False,
        maximum_relative_disagreement=0.0,
        forward_error="NotImplementedError",
    )

    execution = _MODULE.run_stage_a_seed_model(
        checkpoint,
        panel,
        authority.binding,
        cache,
        backend,
        model,
    )

    assert len(execution.receiver_evidence) == 147
    assert tuple(row.receiver_id for row in execution.receiver_evidence) == tuple(
        row.example_id for row in panel.receivers
    )
    assert all(row.seed == 17 for row in execution.receiver_evidence)
    assert execution.first_receiver_first_sha256 == execution.first_receiver_repeat_sha256
    assert execution.parameter_names == tuple(sorted(execution.parameter_names))
    assert execution.parameter_names == (
        "projection.weight",
        "tower.0.bias",
        "tower.0.weight",
    )
    assert execution.parameter_numels == (24, 6, 24)
    assert execution.logical_batch_replays == 6
    assert execution.receiver_actions == 474
    assert execution.autocast_device_type == "cpu"
    assert execution.autocast_dtype == "float32"
    assert execution.autocast_enabled is False
    assert execution.support_replays == 2
    assert execution.module_training is True
    assert execution.gradient_checkpointing_enabled is False
    assert execution.torch_compile_enabled is False
    assert execution.attention_implementation == "eager"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_real_seed_runner_rejects_model_not_loaded_from_bound_checkpoint(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    from sfora.siglip_rsta_stage_a import select_rsta_roles

    panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))
    cache = _tensor_cache(authority)
    torch.manual_seed(731)
    model = PooledProxyAnchorModel(
        tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
        input_dimensions=6,
        embedding_dimensions=4,
        class_count=49,
    ).train()

    with pytest.raises(ValueError, match="checkpoint model state"):
        _MODULE.run_stage_a_seed_model(
            authority.checkpoints[0],
            panel,
            authority.binding,
            cache,
            RstaJvpBackendEvidence(
                backend="double-backward",
                comparison_available=False,
                maximum_relative_disagreement=0.0,
                forward_error="NotImplementedError",
            ),
            model,
        )


def test_real_seed_runner_repeat_reexecutes_support_forward(tmp_path: Path) -> None:
    authority = _scientific_authority(tmp_path)
    from sfora.siglip_rsta_stage_a import select_rsta_roles

    class DriftingEvalTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 6)
            self.eval_calls = 0

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            values = self.linear(inputs)
            if not self.training:
                self.eval_calls += 1
                values = values + float(self.eval_calls - 1)
            return torch.tanh(values)

    panel = select_rsta_roles(tuple(zip(authority.example_ids, authority.labels, strict=True)))
    cache = _tensor_cache(authority)
    torch.manual_seed(732)
    model = PooledProxyAnchorModel(
        tower=DriftingEvalTower(),
        input_dimensions=6,
        embedding_dimensions=4,
        class_count=49,
    ).train()
    checkpoint = replace(
        authority.checkpoints[0],
        model_state=MappingProxyType(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        ),
    )

    with pytest.raises(ValueError, match="repeat"):
        _MODULE.run_stage_a_seed_model(
            checkpoint,
            panel,
            authority.binding,
            cache,
            RstaJvpBackendEvidence(
                backend="double-backward",
                comparison_available=False,
                maximum_relative_disagreement=0.0,
                forward_error="NotImplementedError",
            ),
            model,
        )


def test_checkpoint_model_loader_strictly_restores_state_and_preserves_rng() -> None:
    torch.manual_seed(811)
    source = PooledProxyAnchorModel(
        tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
        input_dimensions=6,
        embedding_dimensions=4,
        class_count=49,
    )
    checkpoint = _MODULE.LoadedStageACheckpoint(
        seed=17,
        model_state=MappingProxyType(
            {name: value.detach().clone() for name, value in source.state_dict().items()}
        ),
    )

    def factory() -> PooledProxyAnchorModel:
        return PooledProxyAnchorModel(
            tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
            input_dimensions=6,
            embedding_dimensions=4,
            class_count=49,
        )

    random.seed(812)
    np.random.seed(813)
    torch.manual_seed(814)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    loaded = _MODULE.load_stage_a_checkpoint_model(
        checkpoint,
        model_factory=factory,
        device=torch.device("cpu"),
    )

    assert loaded.training is True
    assert all(parameter.grad is None for parameter in loaded.parameters())
    assert set(loaded.state_dict()) == set(checkpoint.model_state)
    assert all(
        torch.equal(loaded.state_dict()[name], value)
        for name, value in checkpoint.model_state.items()
    )
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    broken = replace(
        checkpoint,
        model_state=MappingProxyType(dict(list(checkpoint.model_state.items())[1:])),
    )
    with pytest.raises(ValueError, match="model state"):
        _MODULE.load_stage_a_checkpoint_model(
            broken,
            model_factory=factory,
            device=torch.device("cpu"),
        )


def test_model_preflight_disables_checkpointing_and_seals_backend_before_science(
    tmp_path: Path, monkeypatch
) -> None:
    import sfora.siglip_rsta_stage_a as rsta_stage_a

    _arguments, binding = _write_authority_bundle(tmp_path)
    torch.manual_seed(901)
    model = PooledProxyAnchorModel(
        tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
        input_dimensions=6,
        embedding_dimensions=4,
        class_count=49,
    ).train()
    state = {"enabled": True}
    receiver_backends: list[str] = []
    original_receiver_rsta_fields = rsta_stage_a.receiver_rsta_fields

    def observed_receiver_rsta_fields(*args, backend: str, **kwargs):
        receiver_backends.append(backend)
        return original_receiver_rsta_fields(*args, backend=backend, **kwargs)

    monkeypatch.setattr(rsta_stage_a, "receiver_rsta_fields", observed_receiver_rsta_fields)

    evidence = _MODULE.preflight_stage_a_model(
        model,
        binding,
        input_shape=(4,),
        checkpointing_enabled=lambda: state["enabled"],
        disable_checkpointing=lambda: state.__setitem__("enabled", False),
    )

    assert state["enabled"] is False
    assert evidence.backend.backend in {"forward-mode", "double-backward"}
    assert receiver_backends == [
        "forward-mode",
        "double-backward",
        evidence.backend.backend,
    ]
    assert evidence.checkpointing_max_relative_disagreement <= 1.0e-5
    assert len(evidence.fixture_sha256) == 64
    assert evidence.parameter_names == tuple(sorted(evidence.parameter_names))
    assert len(evidence.parameter_numels) == len(evidence.parameter_names)
    assert evidence.logical_batch_replay_ns > 0
    assert evidence.receiver_action_ns > 0
    assert evidence.support_forward_ns > 0
    assert evidence.projected_science_elapsed_ns == (
        6 * evidence.logical_batch_replay_ns
        + 474 * evidence.receiver_action_ns
        + 2 * evidence.support_forward_ns
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    with pytest.raises(ValueError, match="checkpointing preflight"):
        _MODULE.preflight_stage_a_model(
            model,
            binding,
            input_shape=(4,),
            checkpointing_enabled=lambda: False,
            disable_checkpointing=lambda: None,
        )


def test_determinism_policy_is_established_and_observed(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_benchmark = torch.backends.cudnn.benchmark
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        evidence = _MODULE.configure_stage_a_determinism()
        assert evidence.cublas_workspace_config == ":4096:8"
        assert evidence.deterministic_algorithms_enabled is True
        assert evidence.deterministic_algorithms_warn_only is False
        assert evidence.cudnn_benchmark is False
        assert evidence.cuda_matmul_allow_tf32 is False
        assert evidence.cudnn_allow_tf32 is False
    finally:
        torch.use_deterministic_algorithms(
            prior_deterministic,
            warn_only=prior_warn_only,
        )
        torch.backends.cudnn.benchmark = prior_benchmark
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


def test_model_campaign_loads_preflights_and_executes_each_bound_seed(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path)
    states = []
    for seed in (17, 29, 43):
        torch.manual_seed(seed)
        source = PooledProxyAnchorModel(
            tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
            input_dimensions=6,
            embedding_dimensions=4,
            class_count=49,
        )
        states.append(
            _MODULE.LoadedStageACheckpoint(
                seed=seed,
                model_state=MappingProxyType(
                    {name: value.detach().clone() for name, value in source.state_dict().items()}
                ),
            )
        )
    authority = replace(authority, checkpoints=tuple(states))
    created: list[PooledProxyAnchorModel] = []
    checkpointing: dict[int, bool] = {}
    events: list[str] = []

    def factory() -> PooledProxyAnchorModel:
        model = PooledProxyAnchorModel(
            tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
            input_dimensions=6,
            embedding_dimensions=4,
            class_count=49,
        )
        created.append(model)
        checkpointing[id(model)] = True
        return model

    real_preflight = _MODULE.preflight_stage_a_model
    real_seed_runner = _MODULE.run_stage_a_seed_model

    def observed_preflight(*args, **kwargs):
        events.append("preflight")
        return real_preflight(*args, **kwargs)

    def observed_seed_runner(checkpoint, *args, **kwargs):
        events.append(f"science-{checkpoint.seed}")
        return real_seed_runner(checkpoint, *args, **kwargs)

    monkeypatch.setattr(_MODULE, "preflight_stage_a_model", observed_preflight)
    monkeypatch.setattr(_MODULE, "run_stage_a_seed_model", observed_seed_runner)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    observed = _MODULE.execute_stage_a_model_campaign(
        authority,
        tensor_cache=_tensor_cache(authority),
        model_factory=factory,
        device=torch.device("cpu"),
        input_shape=(4,),
        checkpointing_enabled=lambda model: checkpointing[id(model)],
        disable_checkpointing=lambda model: checkpointing.__setitem__(id(model), False),
    )

    assert len(created) == 6
    assert events == [
        "preflight",
        "preflight",
        "preflight",
        "science-17",
        "science-29",
        "science-43",
    ]
    assert tuple(item.seed for item in observed.completed.authority.checkpoints) == (17, 29, 43)
    assert (
        tuple(item.backend.backend for item in observed.preflights)
        == (observed.completed.backend.backend,) * 3
    )
    assert all(checkpointing[id(model)] is False for model in created)
    assert observed.determinism.deterministic_algorithms_enabled is True


@pytest.mark.parametrize(
    ("failing_boundary", "error", "clause"),
    [
        ("determinism", RuntimeError("determinism unavailable"), "determinism-failure"),
        ("checkpoint", ValueError("checkpoint differs"), "authority-mismatch"),
        ("preflight", NotImplementedError("no JVP backend"), "backend-unavailable"),
        (
            "preflight",
            ValueError("RSTA JVP backends disagree above the registered tolerance"),
            "backend-unavailable",
        ),
        ("preflight", ValueError("fixture differs"), "fixture-failure"),
    ],
)
def test_model_campaign_classifies_every_pre_science_boundary(
    tmp_path: Path,
    monkeypatch,
    failing_boundary: str,
    error: Exception,
    clause: str,
) -> None:
    authority = _scientific_authority(tmp_path)

    if failing_boundary == "determinism":
        monkeypatch.setattr(
            _MODULE,
            "configure_stage_a_determinism",
            lambda: (_ for _ in ()).throw(error),
        )
    else:
        monkeypatch.setattr(
            _MODULE,
            "configure_stage_a_determinism",
            lambda: _MODULE.StageADeterminismEvidence(
                cublas_workspace_config=":4096:8",
                deterministic_algorithms_enabled=True,
                deterministic_algorithms_warn_only=False,
                cudnn_benchmark=False,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
            ),
        )
        if failing_boundary == "checkpoint":
            monkeypatch.setattr(
                _MODULE,
                "load_stage_a_checkpoint_model",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
            )
        else:
            monkeypatch.setattr(
                _MODULE,
                "load_stage_a_checkpoint_model",
                lambda *_args, **_kwargs: object(),
            )
            monkeypatch.setattr(
                _MODULE,
                "preflight_stage_a_model",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
            )

    with pytest.raises(_MODULE.PreScienceInvalid) as captured:
        _MODULE.execute_stage_a_model_campaign(
            authority,
            tensor_cache=_tensor_cache(authority),
            model_factory=lambda: object(),
            device=torch.device("cpu"),
            input_shape=(4,),
            checkpointing_enabled=lambda _model: True,
            disable_checkpointing=lambda _model: None,
        )
    assert captured.value.clause == clause


def test_model_campaign_rejects_projected_runtime_above_one_dgx_hour_before_science(
    tmp_path: Path, monkeypatch
) -> None:
    authority = _scientific_authority(tmp_path)
    backend = RstaJvpBackendEvidence("forward-mode", True, 0.0, None)
    preflight = _MODULE.StageAModelPreflight(
        backend=backend,
        checkpointing_max_relative_disagreement=0.0,
        fixture_sha256="a1" * 32,
        parameter_names=("projection.weight", "tower.weight"),
        parameter_numels=(16, 64),
        logical_batch_replay_ns=1,
        receiver_action_ns=1,
        support_forward_ns=1,
        projected_science_elapsed_ns=1_200_000_000_001,
        model_load_ns=1,
        preflight_elapsed_ns=1,
        projected_attempt_elapsed_ns=1_200_000_000_004,
    )
    monkeypatch.setattr(
        _MODULE,
        "configure_stage_a_determinism",
        lambda: _MODULE.StageADeterminismEvidence(
            cublas_workspace_config=":4096:8",
            deterministic_algorithms_enabled=True,
            deterministic_algorithms_warn_only=False,
            cudnn_benchmark=False,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "load_stage_a_checkpoint_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        _MODULE,
        "preflight_stage_a_model",
        lambda *_args, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        _MODULE,
        "execute_stage_a_scientific_loop",
        lambda *_args, **_kwargs: pytest.fail("science opened above the runtime budget"),
    )

    with pytest.raises(_MODULE.PreScienceInvalid) as captured:
        _MODULE.execute_stage_a_model_campaign(
            authority,
            tensor_cache=_tensor_cache(authority),
            model_factory=lambda: object(),
            device=torch.device("cpu"),
            input_shape=(4,),
            checkpointing_enabled=lambda _model: True,
            disable_checkpointing=lambda _model: None,
        )
    assert captured.value.clause == "throughput-budget"
