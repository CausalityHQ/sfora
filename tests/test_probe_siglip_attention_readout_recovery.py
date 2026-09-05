"""Tests for exact SigLIP attention-readout extraction and fitting."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "probe_siglip_attention_readout_recovery.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "scripts.probe_siglip_attention_readout_recovery", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
stream_attention_readout_inputs = _MODULE.stream_attention_readout_inputs
fit_learned_attention_readout = _MODULE.fit_learned_attention_readout
parse_args = _MODULE.parse_args
load_local_evaluation_manifest = _MODULE.load_local_evaluation_manifest
image_basename = _MODULE._image_basename
enforce_cuda_memory_cap = _MODULE._enforce_cuda_memory_cap
write_readout_artifact = _MODULE._write_readout_artifact
main = _MODULE.main


class _MeanHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=1)


def test_cuda_memory_cap_is_enforced_inside_the_scientific_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DGX nvidia-smi reports N/A, so external telemetry cannot enforce this cap."""

    device = torch.device("cuda")
    monkeypatch.setattr(_MODULE.torch.cuda, "memory_reserved", lambda _device: 96 * 1024**3)
    enforce_cuda_memory_cap(device)
    monkeypatch.setattr(
        _MODULE.torch.cuda,
        "memory_reserved",
        lambda _device: 96 * 1024**3 + 1,
    )
    with pytest.raises(RuntimeError, match="CUDA memory cap"):
        enforce_cuda_memory_cap(device)


def test_stream_keeps_manual_pooling_inside_teacher_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA pooling outside teacher autocast would drift or fail only on the DGX."""

    active = False

    @contextlib.contextmanager
    def observed_autocast(**_kwargs: object):
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    class ObservedHead(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            assert active
            return hidden.mean(dim=1)

    class ObservedVision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_layernorm = nn.Identity()
            self.head = ObservedHead()

        def forward(self, *, pixel_values: torch.Tensor, **_kwargs: object) -> object:
            assert active
            hidden = pixel_values.flatten(2).transpose(1, 2)
            return type(
                "Output",
                (),
                {"hidden_states": (hidden, hidden), "pooler_output": self.head(hidden)},
            )()

    monkeypatch.setattr(_MODULE.torch, "autocast", observed_autocast)
    model = ObservedVision().eval()
    projection = nn.Linear(3, 2, bias=False).eval()
    streamed = stream_attention_readout_inputs(
        model,
        projection,
        (torch.ones((2, 3, 1, 1)),),
        depths=(1,),
        decisive_depth=1,
        device=torch.device("cpu"),
    )
    assert streamed.control_planes[0].shape == (2, 3)


def _tiny_model() -> tuple[SiglipVisionModel, nn.Linear]:
    torch.manual_seed(9)
    model = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
        )
    )
    model.eval()  # type: ignore[no-untyped-call]
    projection = nn.Linear(16, 8, bias=False).eval()
    return model, projection


def test_stream_uses_native_attention_head_and_preserves_decisive_tokens() -> None:
    """Mean pooling or the wrong hidden-state index would invalidate the causal probe."""

    model, projection = _tiny_model()
    pixels = torch.arange(6 * 3 * 16 * 16, dtype=torch.float32).reshape(6, 3, 16, 16) / 4096
    batches = (pixels[:2], pixels[2:5], pixels[5:])
    calls = 0
    original_forward = model.forward

    def counted_forward(self: nn.Module, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_forward(**kwargs)

    model.forward = MethodType(counted_forward, model)

    streamed = stream_attention_readout_inputs(
        model,
        projection,
        batches,
        depths=(1, 2, 3),
        decisive_depth=2,
        device=torch.device("cpu"),
    )

    model.forward = original_forward
    with torch.inference_mode():
        direct_batches = [
            model(pixel_values=batch, output_hidden_states=True, return_dict=True)
            for batch in batches
        ]
        expected_controls = tuple(
            torch.cat(
                [
                    model.head(model.post_layernorm(output.hidden_states[depth]))
                    for output in direct_batches
                ]
            )
            for depth in (1, 2, 3)
        )
        expected_tokens = torch.cat([output.hidden_states[2].half() for output in direct_batches])
        expected_pooler = torch.cat([output.pooler_output.float() for output in direct_batches])
        expected_targets = torch.cat(
            [
                F.normalize(projection(output.pooler_output.float()), dim=1)
                for output in direct_batches
            ]
        )

    assert calls == len(batches)
    assert len(streamed.control_planes) == 3
    for actual, expected in zip(streamed.control_planes, expected_controls, strict=True):
        assert actual.dtype == torch.float32
        assert actual.device.type == "cpu"
        assert torch.allclose(actual, expected.float(), atol=0.0, rtol=0.0)
    assert streamed.decisive_tokens.dtype == torch.float16
    assert torch.equal(streamed.decisive_tokens, expected_tokens)
    assert torch.allclose(streamed.teacher_targets, expected_targets, atol=0.0, rtol=0.0)
    assert torch.allclose(streamed.control_planes[-1], expected_pooler, atol=0.0, rtol=0.0)
    expected_bytes = sum(value.numel() * value.element_size() for value in streamed.control_planes)
    expected_bytes += streamed.decisive_tokens.numel() * streamed.decisive_tokens.element_size()
    expected_bytes += streamed.teacher_targets.numel() * streamed.teacher_targets.element_size()
    expected_bytes += streamed.teacher_outputs.numel() * streamed.teacher_outputs.element_size()
    assert streamed.cache_bytes == expected_bytes


def test_stream_rejects_training_mode_depth_and_nonfinite_input() -> None:
    """Mutable teacher state or malformed pixels would make cached planes non-authoritative."""

    model, projection = _tiny_model()
    pixels = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    for mutation, match in (
        (lambda: model.train(), "model authority differs"),
        (lambda: None, "depth authority differs"),
        (lambda: pixels.fill_(float("nan")), "pixel authority differs"),
    ):
        model.eval()  # type: ignore[no-untyped-call]
        pixels.zero_()
        mutation()
        depths = (1, 2, 3) if match != "depth authority differs" else (1, 1, 3)
        try:
            stream_attention_readout_inputs(
                model,
                projection,
                (pixels,),
                depths=depths,
                decisive_depth=2,
                device=torch.device("cpu"),
            )
        except ValueError as error:
            assert match in str(error)
        else:
            raise AssertionError("malformed extraction authority was accepted")


def test_learned_attention_readout_is_teacher_initialized_deterministic_and_rng_isolated() -> None:
    """A mutable initialization or stochastic seal would make the decisive cell irreproducible."""

    tokens = torch.tensor(
        [
            [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]],
            [[0.5, 1.0, 0.0, -0.5], [1.0, 0.5, -0.5, 0.0]],
            [[-1.0, 0.0, 0.5, 0.5], [0.0, -1.0, 0.5, 0.5]],
            [[-0.5, -1.0, 0.0, 0.5], [-1.0, -0.5, 0.5, 0.0]],
        ],
        dtype=torch.float16,
    ).contiguous()
    targets = F.normalize(
        torch.tensor([[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [0.0, -1.0]], dtype=torch.float32),
        dim=1,
    ).contiguous()
    layernorm = nn.LayerNorm(4).eval()
    head = _MeanHead().eval()
    projection = nn.Linear(4, 2, bias=False).eval()
    initial = {key: value.detach().clone() for key, value in projection.state_dict().items()}
    rng = torch.random.get_rng_state().clone()

    left = fit_learned_attention_readout(tokens, targets, layernorm, head, projection)
    right = fit_learned_attention_readout(tokens, targets, layernorm, head, projection)

    assert torch.equal(torch.random.get_rng_state(), rng)
    assert left.initial_loss == right.initial_loss
    assert left.final_loss == right.final_loss
    assert left.final_loss < left.initial_loss
    assert len(left.final_200_losses) == 200
    for name, value in left.readout.state_dict().items():
        assert torch.equal(value, right.readout.state_dict()[name])
    for name, value in projection.state_dict().items():
        assert torch.equal(value, initial[name])


def test_learned_attention_readout_rejects_wrong_token_and_target_authority() -> None:
    """The learned readout must not silently accept an incompatible cached representation."""

    layernorm = nn.LayerNorm(4).eval()
    head = _MeanHead().eval()
    projection = nn.Linear(4, 2, bias=False).eval()
    for tokens, targets in (
        (torch.ones((2, 2, 4)), F.normalize(torch.ones((2, 2)), dim=1)),
        (
            torch.ones((2, 2, 4), dtype=torch.float16),
            torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]),
        ),
    ):
        try:
            fit_learned_attention_readout(tokens, targets, layernorm, head, projection)
        except ValueError as error:
            assert "learned attention authority differs" in str(error)
        else:
            raise AssertionError("malformed learned-attention authority was accepted")


def test_readout_artifact_seals_every_fitted_weight_before_evaluation(tmp_path: Path) -> None:
    """Hashes without retained states cannot reproduce or advance a positive result."""

    layernorm = nn.LayerNorm(4).eval()
    head = _MeanHead().eval()
    projection = nn.Linear(4, 2, bias=False).eval()
    learned = _MODULE.LearnedAttentionReadout(layernorm, head, projection).eval()
    linear = [
        (6, "ridge", torch.arange(8, dtype=torch.float32).reshape(2, 4), False),
        (6, "refined", torch.ones((2, 4)), False),
    ]
    output = tmp_path / "readouts.safetensors"

    digest = write_readout_artifact(output, linear, learned)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    from safetensors.torch import load_file

    state = load_file(output)
    assert torch.equal(state["linear.depth-06.ridge.weight"], linear[0][2])
    assert torch.equal(state["linear.depth-06.refined.weight"], linear[1][2])
    assert set(
        name.removeprefix("learned.") for name in state if name.startswith("learned.")
    ) == set(learned.state_dict())
    with pytest.raises(FileExistsError):
        write_readout_artifact(output, linear, learned)


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "--control-binding",
        str(tmp_path / "binding.json"),
        "--control-binding-sha256",
        "11" * 32,
        "--checkpoint-seed17",
        str(tmp_path / "teacher.pt"),
        "--optimization-manifest",
        str(tmp_path / "optimization.json"),
        "--optimization-manifest-sha256",
        "22" * 32,
        "--optimization-image-root",
        str(tmp_path / "optimization-images"),
        "--evaluation-manifest",
        str(tmp_path / "evaluation.json"),
        "--evaluation-manifest-sha256",
        "33" * 32,
        "--evaluation-image-root",
        str(tmp_path / "evaluation-images"),
        "--readout-artifact",
        str(tmp_path / "readouts.safetensors"),
        "--result",
        str(tmp_path / "result.json"),
        "--execute-attention-readout",
    ]


def test_cli_requires_explicit_local_roles_and_execution(tmp_path: Path) -> None:
    """The scientific executable must expose only its pre-authenticated local capabilities."""

    parsed = parse_args(_cli_args(tmp_path))
    assert parsed.checkpoint_seed17 == tmp_path / "teacher.pt"
    assert parsed.execute_attention_readout is True


def test_cli_rejects_missing_duplicate_relative_and_network_or_tuning_flags(tmp_path: Path) -> None:
    """A hidden model, network, student, text, or tuning input would invalidate the experiment."""

    baseline = _cli_args(tmp_path)
    mutations = (
        baseline[:-1],
        [*baseline, "--result", str(tmp_path / "other.json")],
        [
            "relative.json" if value == str(tmp_path / "result.json") else value
            for value in baseline
        ],
        [*baseline, "--model", "remote/model"],
        [*baseline, "--student-checkpoint", str(tmp_path / "student.pt")],
        [*baseline, "--class-names", str(tmp_path / "classes.json")],
        [*baseline, "--learning-rate", "0.001"],
        [*baseline, "--official-test"],
    )
    for arguments in mutations:
        try:
            parse_args(arguments)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("forbidden attention-readout capability was accepted")


def test_local_evaluation_manifest_binds_exact_rows_and_flat_image_namespace(
    tmp_path: Path,
) -> None:
    """Evaluation pixels must be pre-materialized and unavailable during readout fitting."""

    root = tmp_path / "evaluation-images"
    root.mkdir()
    examples = [
        {"example_id": "cars-train-49-1", "label": 49},
        {"example_id": "cars-train-50-2", "label": 50},
    ]
    for row in examples:
        (root / image_basename(row["example_id"])).write_bytes(b"image")
    raw = (
        json.dumps(
            {
                "claim_eligible": False,
                "dataset_id": "cars",
                "dataset_revision": "revision",
                "examples": examples,
                "schema": "sfora-attention-readout-evaluation-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    manifest = tmp_path / "evaluation.json"
    manifest.write_bytes(raw)

    ids, labels, paths = load_local_evaluation_manifest(
        manifest,
        hashlib.sha256(raw).hexdigest(),
        root,
        dataset_id="cars",
        dataset_revision="revision",
        expected_count=2,
        expected_labels=frozenset({49, 50}),
    )

    assert ids == ("cars-train-49-1", "cars-train-50-2")
    assert labels == (49, 50)
    assert tuple(path.name for path in paths) == tuple(
        image_basename(row["example_id"]) for row in examples
    )


def test_local_evaluation_manifest_rejects_digest_schema_and_namespace_drift(
    tmp_path: Path,
) -> None:
    """Missing or extra local files and metadata drift must fail before pixel decoding."""

    root = tmp_path / "evaluation-images"
    root.mkdir()
    example = {"example_id": "cars-train-49-1", "label": 49}
    (root / image_basename(example["example_id"])).write_bytes(b"image")
    value = {
        "claim_eligible": False,
        "dataset_id": "cars",
        "dataset_revision": "revision",
        "examples": [example],
        "schema": "sfora-attention-readout-evaluation-v1",
    }
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest = tmp_path / "evaluation.json"
    manifest.write_bytes(raw)
    for digest, extra_file in (("00" * 32, False), (hashlib.sha256(raw).hexdigest(), True)):
        extra = root / "extra.image"
        if extra_file:
            extra.write_bytes(b"extra")
        try:
            load_local_evaluation_manifest(
                manifest,
                digest,
                root,
                dataset_id="cars",
                dataset_revision="revision",
                expected_count=1,
                expected_labels=frozenset({49}),
            )
        except ValueError as error:
            assert "evaluation" in str(error)
        else:
            raise AssertionError("malformed evaluation manifest was accepted")
        if extra.exists():
            extra.unlink()


def test_direct_script_executes_and_evaluation_access_follows_readout_sealing() -> None:
    """Direct execution and phase order must survive the deployed file path."""

    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--execute-attention-readout" in completed.stdout
    script_source = _SCRIPT.read_text()
    assert script_source.index("def stream_attention_readout_inputs(") < script_source.index(
        'if __name__ == "__main__":'
    )
    source = inspect.getsource(main)
    assert source.index("fit_learned_attention_readout(") < source.index(
        "load_local_evaluation_manifest("
    )
