"""Tests for the authenticated local SigLIP RSTA Stage-A CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.pass209_m4 import canonical_json_bytes
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
    )


def _forward_backend() -> RstaJvpBackendEvidence:
    return RstaJvpBackendEvidence(
        backend="forward-mode",
        comparison_available=True,
        maximum_relative_disagreement=0.0,
        forward_error=None,
    )


def _tensor_digests(authority) -> dict[str, str]:
    return {
        example_id: hashlib.sha256(example_id.encode()).hexdigest()
        for example_id in authority.example_ids
    }


def test_reduced_scientific_loop_runs_exact_seed_panel_order_and_aggregates(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    calls: list[int] = []

    def runner(checkpoint, panel, binding, image_paths, backend):
        calls.append(checkpoint.seed)
        assert binding is authority.binding
        assert tuple(image_paths) == authority.example_ids
        assert backend is sealed_backend
        return _seed_execution(checkpoint, panel)

    sealed_backend = _forward_backend()
    completed = _MODULE.execute_stage_a_scientific_loop(
        authority,
        backend=sealed_backend,
        tensor_sha256_by_id=_tensor_digests(authority),
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

    def runner(checkpoint, panel, _binding, _image_paths, _backend):
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
            tensor_sha256_by_id=_tensor_digests(authority),
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
    digests = _tensor_digests(authority)
    if mutation == "seed-order":
        authority = replace(authority, checkpoints=tuple(reversed(authority.checkpoints)))
    elif mutation == "image-count":
        authority = replace(authority, image_paths=authority.image_paths[:-1])
    elif mutation == "tensor-digest":
        digests.pop(next(iter(digests)))
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
            tensor_sha256_by_id=digests,
            seed_runner=_seed_execution,
        )


def test_reduced_scientific_loop_stops_after_first_mid_campaign_failure(
    tmp_path: Path,
) -> None:
    authority = _scientific_authority(tmp_path)
    calls: list[int] = []

    def runner(checkpoint, panel, _binding, _image_paths, _backend):
        calls.append(checkpoint.seed)
        if checkpoint.seed == 29:
            raise RuntimeError("interrupted receiver row")
        return _seed_execution(checkpoint, panel)

    with pytest.raises(_MODULE.PostScienceFailure):
        _MODULE.execute_stage_a_scientific_loop(
            authority,
            backend=_forward_backend(),
            tensor_sha256_by_id=_tensor_digests(authority),
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
            tensor_sha256_by_id=_tensor_digests(authority),
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
