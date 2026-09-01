from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sfora.saga_feasibility import (
    FeasibilityOutcome,
    FixtureAuthority,
    ObjectAuthority,
    ResourceEnvelope,
    SnapshotAuthority,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_saga_gb10_feasibility.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_saga_gb10_feasibility", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _valid_cli_args(tmp_path: Path) -> list[str]:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    snapshot = tmp_path / "snapshot.json"
    fixture = tmp_path / "fixture.json"
    snapshot.write_bytes(b"{}\n")
    fixture.write_bytes(b"{}\n")
    return [
        "--model-root",
        str(model_root),
        "--snapshot-manifest",
        str(snapshot),
        "--fixture",
        str(fixture),
        "--result-output",
        str(tmp_path / "result.json"),
        "--source-commit",
        "5" * 40,
        "--controller-commit",
        "b" * 40,
        "--binary-sha256",
        "6" * 64,
        "--environment-sha256",
        "7" * 64,
        "--host",
        "spark-fixture",
        "--execute-feasibility",
    ]


def test_cli_accepts_only_local_scientific_capabilities(tmp_path: Path) -> None:
    args = _MODULE.parse_args(_valid_cli_args(tmp_path))
    assert args.execute_feasibility is True
    assert args.model_root == tmp_path / "model"
    for forbidden in (
        "--dataset",
        "--labels",
        "--checkpoint",
        "--hub-token",
        "--model-uri",
        "--aws-profile",
        "--train",
        "--evaluate",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*_valid_cli_args(tmp_path), forbidden, "x"])


@pytest.mark.parametrize("missing_index", range(0, 12, 2))
def test_cli_rejects_missing_required_flags(tmp_path: Path, missing_index: int) -> None:
    argv = _valid_cli_args(tmp_path)
    del argv[missing_index : missing_index + 2]
    with pytest.raises(SystemExit):
        _MODULE.parse_args(argv)


def test_cli_rejects_duplicate_flags(tmp_path: Path) -> None:
    argv = _valid_cli_args(tmp_path)
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*argv, "--fixture", str(tmp_path / "fixture.json")])


@dataclass
class _Parameter:
    requires_grad: bool


class _FakeModel:
    architecture = "Qwen3VLForConditionalGeneration"
    dtype = "bfloat16"
    device = "cuda"
    attention_backend = "eager"
    layer_count = 36

    def __init__(self) -> None:
        self.parameters_by_role = {
            "vision": [_Parameter(False)],
            "language": [_Parameter(True), _Parameter(True)],
        }


class _FakeProcessor:
    output_keys = (
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "mm_token_type_ids",
        "pixel_values",
    )


class _FakeFactory:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.model_kwargs: dict[str, object] | None = None
        self.processor_kwargs: dict[str, object] | None = None

    def load_model(self, root: Path, **kwargs: object) -> _FakeModel:
        assert root.is_dir()
        self.model_kwargs = kwargs
        return self.model

    def load_processor(self, root: Path, **kwargs: object) -> _FakeProcessor:
        assert root.is_dir()
        self.processor_kwargs = kwargs
        return _FakeProcessor()


def _loaded_authority(tmp_path: Path) -> object:
    root = tmp_path / "model"
    root.mkdir()
    snapshot = SnapshotAuthority(
        root=root,
        repository_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision="1" * 40,
        processor_revision="2" * 40,
        tokenizer_revision="3" * 40,
        snapshot_tree_sha256="4" * 64,
        architecture="Qwen3VLForConditionalGeneration",
        dtype="bfloat16",
        attention_backend="eager",
        files=(),
    )
    fixture = FixtureAuthority(
        source_commit="5" * 40,
        controller_commit="b" * 40,
        model_revision="1" * 40,
        binary_sha256="6" * 64,
        environment_sha256="7" * 64,
        host="spark-fixture",
        group_size=8,
        image_count=64,
        generation_seeds=tuple(range(8)),
        synthetic_rewards=(0, 1, 0, 1, 0, 1, 0, 1),
        attention_layer=26,
        attribute_token_span=(0, 1),
        patch_tokens_per_image=4,
        prompt_sha256="8" * 64,
        message_serialization_sha256="9" * 64,
        prompt_utf8="List the visible car attributes and relations.",
        image_sha256=tuple("a" * 64 for _ in range(64)),
        pair_ordinals=(0, 1),
        microbatch_ordinals=tuple(range(64)),
        pseudo_labels=tuple(ordinal % 2 for ordinal in range(64)),
    )
    return _MODULE.LoadedAuthority(snapshot=snapshot, fixture=fixture)


def test_model_load_freezes_language_and_leaves_only_vision_trainable(
    tmp_path: Path,
) -> None:
    factory = _FakeFactory()
    adapter = _MODULE.load_qwen_adapter(_loaded_authority(tmp_path), factory=factory)
    assert adapter.architecture == "Qwen3VLForConditionalGeneration"
    assert adapter.trainable_parameter_roles() == ("vision",)
    assert adapter.frozen_parameter_roles() == ("language",)
    assert all(parameter.requires_grad for parameter in factory.model.parameters_by_role["vision"])
    assert all(
        not parameter.requires_grad for parameter in factory.model.parameters_by_role["language"]
    )
    assert factory.model_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
    }
    assert factory.processor_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("architecture", "WrongModel"),
        ("dtype", "float32"),
        ("device", "cpu"),
        ("attention_backend", "flash_attention_2"),
        ("layer_count", 26),
    ],
)
def test_model_load_rejects_structural_drift(tmp_path: Path, attribute: str, value: object) -> None:
    factory = _FakeFactory()
    setattr(factory.model, attribute, value)
    with pytest.raises(ValueError, match="model authority"):
        _MODULE.load_qwen_adapter(_loaded_authority(tmp_path), factory=factory)


def _fixture_authority() -> FixtureAuthority:
    return FixtureAuthority(
        source_commit="5" * 40,
        controller_commit="b" * 40,
        model_revision="1" * 40,
        binary_sha256="6" * 64,
        environment_sha256="7" * 64,
        host="spark-fixture",
        group_size=8,
        image_count=64,
        generation_seeds=tuple(range(8)),
        synthetic_rewards=(0, 1, 0, 1, 0, 1, 0, 1),
        attention_layer=26,
        attribute_token_span=(0, 1),
        patch_tokens_per_image=4,
        prompt_sha256="8" * 64,
        message_serialization_sha256="9" * 64,
        prompt_utf8="List the visible car attributes and relations.",
        image_sha256=tuple("a" * 64 for _ in range(64)),
        pair_ordinals=(0, 1),
        microbatch_ordinals=tuple(range(64)),
        pseudo_labels=tuple(ordinal % 2 for ordinal in range(64)),
    )


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generation_calls: list[tuple[int, float, float, int]] = []
        self.replay_calls: list[tuple[tuple[float, ...], bool]] = []
        self.cleared = 0

    def prepare_pair(self, fixture: FixtureAuthority) -> object:
        assert fixture == _FIXTURE
        return "prepared-pair"

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        assert pair == "prepared-pair"
        self.generation_calls.append((seed, temperature, top_p, max_new_tokens))
        return (seed + 10, seed + 20)

    def replay(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        advantages: tuple[float, ...],
        *,
        output_attentions: bool,
    ) -> object:
        assert pair == "prepared-pair"
        assert len(completion_ids) == 8
        self.replay_calls.append((advantages, output_attentions))
        return _MODULE.ReplayOutput(loss=0.25, generated_tokens=16)

    def assert_gradient_roles(self) -> object:
        return _MODULE.GradientEvidence(
            vision_nonzero_gradient_parameters=2,
            language_gradient_parameters=0,
            finite=True,
            gradient_sha256="a" * 64,
        )

    def clear_graphs(self) -> None:
        self.cleared += 1


_FIXTURE = _fixture_authority()


def test_rollout_uses_exact_group_sampling_and_distinct_generators() -> None:
    adapter = _RecordingAdapter()
    evidence = _MODULE.run_rollout_phase(adapter, _FIXTURE)
    assert adapter.generation_calls == [
        (seed, 0.7, 0.95, 1024) for seed in _FIXTURE.generation_seeds
    ]
    assert evidence.group_size == 8
    assert evidence.token_counts == (2,) * 8
    assert len(set(evidence.completion_sha256)) == 8


def test_group_normalized_advantages_are_zero_mean_and_unit_scale() -> None:
    advantages = _MODULE.group_normalized_advantages(_FIXTURE.synthetic_rewards)
    assert advantages == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_replay_reaches_vision_and_never_language_parameters() -> None:
    adapter = _RecordingAdapter()
    rollouts = _MODULE.run_rollout_phase(adapter, _FIXTURE)
    evidence = _MODULE.run_replay_phase(adapter, _FIXTURE, rollouts)
    assert evidence.vision_nonzero_gradient_parameters == 2
    assert evidence.language_gradient_parameters == 0
    assert evidence.generated_tokens == sum(rollouts.token_counts)
    assert adapter.replay_calls == [((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0), False)]
    assert adapter.cleared == 1


@pytest.mark.parametrize(
    ("gradient", "message"),
    [
        (
            {
                "vision_nonzero_gradient_parameters": 0,
                "language_gradient_parameters": 0,
                "finite": True,
            },
            "vision gradient",
        ),
        (
            {
                "vision_nonzero_gradient_parameters": 1,
                "language_gradient_parameters": 1,
                "finite": True,
            },
            "language gradient",
        ),
        (
            {
                "vision_nonzero_gradient_parameters": 1,
                "language_gradient_parameters": 0,
                "finite": False,
            },
            "finite gradient",
        ),
    ],
)
def test_replay_rejects_gradient_role_drift(gradient: dict[str, object], message: str) -> None:
    adapter = _RecordingAdapter()
    adapter.assert_gradient_roles = lambda: _MODULE.GradientEvidence(  # type: ignore[method-assign]
        gradient_sha256="a" * 64, **gradient
    )
    with pytest.raises(ValueError, match=message):
        _MODULE.run_replay_phase(adapter, _FIXTURE, _MODULE.run_rollout_phase(adapter, _FIXTURE))
    assert adapter.cleared == 1


class _AttentionDmlAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.teacher = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self.patch_tokens = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
        self.patch_tokens.requires_grad_(True)
        self.microbatch_embeddings: torch.Tensor | None = None

    def validate_structure(self, _authority: object) -> None:
        return None

    def attention_observation(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        *,
        layer: int,
    ) -> object:
        assert pair == "prepared-pair"
        assert len(completion_ids) == 8
        assert layer == 26
        return _MODULE.AttentionOutput(
            teacher_maps=self.teacher,
            patch_tokens=self.patch_tokens,
            head_count=16,
        )

    def prepare_microbatch(self, fixture: FixtureAuthority) -> object:
        assert fixture == _FIXTURE
        return "prepared-microbatch"

    def vision_pool(self, microbatch: object) -> torch.Tensor:
        assert microbatch == "prepared-microbatch"
        embeddings = torch.eye(64, 4096, dtype=torch.float32)
        embeddings.requires_grad_(True)
        self.microbatch_embeddings = embeddings
        return embeddings


def test_attention_phase_detaches_teacher_and_updates_only_pooler() -> None:
    adapter = _AttentionDmlAdapter()
    pooler = _MODULE.SingleQueryPooler(token_dim=16, embedding_dim=4096)
    rollouts = _MODULE.run_rollout_phase(adapter, _FIXTURE)
    evidence = _MODULE.run_attention_phase(adapter, pooler, _FIXTURE, rollouts)
    assert evidence.layer == 26
    assert evidence.teacher_unit_mass is True
    assert evidence.teacher_gradient_parameters == 0
    assert evidence.pooler_nonzero_gradient_parameters > 0
    assert adapter.teacher.grad is None
    assert adapter.patch_tokens.grad is None
    assert adapter.cleared == 1


def test_source_bound_pooler_sends_attention_kl_gradient_to_patch_tokens() -> None:
    pooler = _MODULE._source_bound_pooler(
        token_dim=16,
        source_commit="a" * 40,
        model_revision="b" * 40,
    )
    patch_tokens = torch.linspace(
        -1.0,
        1.0,
        steps=2 * 4 * 16,
        dtype=torch.float32,
    ).reshape(2, 4, 16)
    patch_tokens.requires_grad_(True)
    teacher = torch.tensor(
        [[0.55, 0.25, 0.15, 0.05], [0.05, 0.15, 0.25, 0.55]],
        dtype=torch.float32,
    )

    _, student = pooler(patch_tokens)
    attention_kl = torch.sum(teacher * (teacher.log() - student.log()), dim=-1).mean()
    attention_kl.backward()

    assert attention_kl.item() > 0.0
    assert patch_tokens.grad is not None
    assert torch.isfinite(patch_tokens.grad).all()
    assert torch.count_nonzero(patch_tokens.grad).item() > 0


def test_attention_phase_rejects_nonunit_or_nonfinite_teacher() -> None:
    for teacher in (
        torch.tensor([[0.4, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]]),
        torch.tensor([[float("nan"), 0.3, 0.2, 0.5], [0.1, 0.2, 0.3, 0.4]]),
    ):
        adapter = _AttentionDmlAdapter()
        adapter.teacher = teacher
        with pytest.raises(ValueError, match="teacher attention"):
            _MODULE.run_attention_phase(
                adapter,
                _MODULE.SingleQueryPooler(token_dim=16),
                _FIXTURE,
                _MODULE.run_rollout_phase(adapter, _FIXTURE),
            )
        assert adapter.cleared == 1


def test_dml_floor_emits_64_unit_4096d_embeddings() -> None:
    adapter = _AttentionDmlAdapter()
    evidence = _MODULE.run_dml_floor_phase(adapter, _FIXTURE)
    assert evidence.batch_size == 64
    assert evidence.embedding_shape == (64, 4096)
    assert evidence.maximum_norm_delta_ppm <= 10
    assert evidence.vision_nonzero_gradient_parameters == 2
    assert adapter.microbatch_embeddings is not None
    assert adapter.microbatch_embeddings.grad is not None
    assert adapter.cleared == 1


def test_dml_floor_rejects_embedding_shape_and_norm_drift() -> None:
    adapter = _AttentionDmlAdapter()
    adapter.vision_pool = lambda _microbatch: torch.ones(64, 4095)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="embedding shape"):
        _MODULE.run_dml_floor_phase(adapter, _FIXTURE)
    assert adapter.cleared == 1

    adapter = _AttentionDmlAdapter()
    adapter.vision_pool = lambda _microbatch: torch.ones(64, 4096)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="embedding norms"):
        _MODULE.run_dml_floor_phase(adapter, _FIXTURE)
    assert adapter.cleared == 1


def _complete_authority(tmp_path: Path) -> object:
    authority = _loaded_authority(tmp_path)
    return _MODULE.LoadedAuthority(
        snapshot=authority.snapshot,
        fixture=authority.fixture,
        result_identity=_MODULE.RunIdentity(
            source_commit="5" * 40,
            controller_commit="b" * 40,
            binary_sha256="6" * 64,
            environment_sha256="7" * 64,
            host="spark-fixture",
            model_object=ObjectAuthority(
                role="model-snapshot-manifest",
                relative_path="snapshot.json",
                byte_length=10,
                sha256="e" * 64,
            ),
            fixture_object=ObjectAuthority(
                role="synthetic-fixture",
                relative_path="fixture.json",
                byte_length=20,
                sha256="f" * 64,
            ),
            envelope=ResourceEnvelope(
                cuda_reserved_limit_bytes=103_079_215_104,
                rss_limit_bytes=118_111_600_640,
                wall_limit_ns=7_200_000_000_000,
                progress_limit_ns=300_000_000_000,
            ),
        ),
    )


def test_complete_fake_run_emits_one_claim_ineligible_fits_result(
    tmp_path: Path,
) -> None:
    raw = _MODULE.run_feasibility(_complete_authority(tmp_path), _AttentionDmlAdapter())
    value = json.loads(raw)
    assert value["outcome"] == "FITS"
    assert value["dataset_reads"] == 0
    assert value["optimizer_steps"] == 0
    assert value["quality_metrics"] == []
    assert [phase["name"] for phase in value["phases"]] == [
        "load",
        "rollout",
        "replay",
        "attention",
        "dml",
    ]


class _FaultingAdapter(_AttentionDmlAdapter):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault
        self.generation_count = 0

    def validate_structure(self, _authority: object) -> None:
        if self.fault == "authority":
            raise _MODULE.FeasibilityFailure(
                FeasibilityOutcome.AUTHORITY_INVALID, "fixture-authority"
            )
        if self.fault == "backend":
            raise _MODULE.FeasibilityFailure(FeasibilityOutcome.BACKEND_INVALID, "model-backend")
        if self.fault == "timeout":
            raise _MODULE.FeasibilityFailure(FeasibilityOutcome.TIME_BUDGET_FAIL, "phase-progress")

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        tokens = super().generate(
            pair,
            seed,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self.generation_count += 1
        if self.fault == "repeat" and self.generation_count > 8:
            return (tokens[0] + 999, tokens[1])
        return tokens

    def attention_observation(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        *,
        layer: int,
    ) -> object:
        if self.fault == "attention":
            raise _MODULE.AttentionUnavailable("layer-26-unavailable")
        return super().attention_observation(pair, completion_ids, layer=layer)

    def vision_pool(self, microbatch: object) -> torch.Tensor:
        if self.fault == "oom":
            raise torch.OutOfMemoryError("fixture OOM")
        return super().vision_pool(microbatch)


@pytest.mark.parametrize(
    ("fault", "outcome"),
    [
        ("authority", "AUTHORITY_INVALID"),
        ("backend", "BACKEND_INVALID"),
        ("repeat", "DETERMINISM_FAIL"),
        ("oom", "MEMORY_FAIL"),
        ("attention", "ATTENTION_UNAVAILABLE"),
        ("timeout", "TIME_BUDGET_FAIL"),
    ],
)
def test_failure_precedence_is_exhaustive(tmp_path: Path, fault: str, outcome: str) -> None:
    raw = _MODULE.run_feasibility(_complete_authority(tmp_path), _FaultingAdapter(fault))
    assert json.loads(raw)["outcome"] == outcome


def test_direct_script_main_writes_one_exclusive_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded = _complete_authority(tmp_path)
    adapter = _AttentionDmlAdapter()
    monkeypatch.setattr(_MODULE, "load_snapshot_authority", lambda **_kwargs: loaded.snapshot)
    monkeypatch.setattr(_MODULE, "load_fixture_authority", lambda _path: loaded.fixture)
    monkeypatch.setattr(_MODULE, "TransformersFactory", lambda: object())
    monkeypatch.setattr(_MODULE, "load_qwen_adapter", lambda _authority, factory: adapter)
    argv = _valid_cli_args(tmp_path)
    assert _MODULE.main(argv) == 0
    output = tmp_path / "result.json"
    raw = output.read_bytes()
    assert capsys.readouterr().out.encode() == raw
    assert raw.endswith(b"\n")
    assert json.loads(raw)["outcome"] == "FITS"
    with pytest.raises(SystemExit):
        _MODULE.main(argv)


def test_direct_script_classifies_model_load_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _complete_authority(tmp_path)
    monkeypatch.setattr(_MODULE, "load_snapshot_authority", lambda **_kwargs: loaded.snapshot)
    monkeypatch.setattr(_MODULE, "load_fixture_authority", lambda _path: loaded.fixture)
    monkeypatch.setattr(_MODULE, "TransformersFactory", lambda: object())

    def raise_oom(_authority: object, *, factory: object) -> object:
        del factory
        raise torch.OutOfMemoryError("model-load-oom")

    monkeypatch.setattr(_MODULE, "load_qwen_adapter", raise_oom)
    assert _MODULE.main(_valid_cli_args(tmp_path)) == 0
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["outcome"] == "MEMORY_FAIL"
    assert result["phases"][0]["completed"] is False
    assert result["scientific_evidence"] is None


def test_direct_script_classifies_model_load_structure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _complete_authority(tmp_path)
    monkeypatch.setattr(_MODULE, "load_snapshot_authority", lambda **_kwargs: loaded.snapshot)
    monkeypatch.setattr(_MODULE, "load_fixture_authority", lambda _path: loaded.fixture)
    monkeypatch.setattr(_MODULE, "TransformersFactory", lambda: object())

    def raise_invalid(_authority: object, *, factory: object) -> object:
        del factory
        raise ValueError("model-load-structure")

    monkeypatch.setattr(_MODULE, "load_qwen_adapter", raise_invalid)
    assert _MODULE.main(_valid_cli_args(tmp_path)) == 0
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["outcome"] == "BACKEND_INVALID"


class _HfLikeProcessor:
    model_input_names = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    )

    def apply_chat_template(self, _messages: object, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": torch.tensor([[1, 99, 99, 2, 99, 99, 3, 4, 5, 6]]),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0, 0, 0]]),
            "pixel_values": torch.ones(16, 1),
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 4]]),
        }

    def image_processor(self, *, images: object, return_tensors: str) -> dict[str, torch.Tensor]:
        assert len(images) == 64
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.ones(64 * 8, 1),
            "image_grid_thw": torch.tensor([[1, 2, 4]] * 64),
        }


class _HfLikeVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(1, 1, bias=False)
        self.merger = torch.nn.Linear(1, 16, bias=False)
        self.deepstack_merger_list = torch.nn.ModuleList(
            torch.nn.Linear(1, 16, bias=False) for _ in range(3)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.merger(self.stem(values))

    def deepstack(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(module(self.stem(values)) for module in self.deepstack_merger_list)


class _HfLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.device = "cuda"
        self.dtype = "bfloat16"
        visual = _HfLikeVisual()
        language = torch.nn.Linear(1, 1, bias=False)
        self.model = SimpleNamespace(visual=visual, language_model=language)
        self.lm_head = torch.nn.Linear(1, 128, bias=False)
        self.config = SimpleNamespace(
            architectures=["Qwen3VLForConditionalGeneration"],
            _attn_implementation="eager",
            image_token_id=99,
            text_config=SimpleNamespace(num_hidden_layers=32),
            vision_config=SimpleNamespace(out_hidden_size=16, spatial_merge_size=2),
        )

    def generate(self, **inputs: object) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert inputs["do_sample"] is True
        generated = torch.randint(10, 90, (1, 2), device=input_ids.device)
        return torch.cat((input_ids, generated), dim=1)

    def forward(self, **inputs: object) -> object:
        self.forward_calls += 1
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        sequence = input_ids.shape[1]
        visual_input = torch.ones(4, 1)
        patch_tokens = self.model.visual(visual_input)
        deepstack_tokens = self.model.visual.deepstack(visual_input)
        scale = patch_tokens.mean() + sum(value.mean() for value in deepstack_tokens)
        vocabulary = torch.arange(128, dtype=scale.dtype, device=scale.device)
        logits = scale * vocabulary.view(1, 1, -1).expand(1, sequence, -1)
        positions = torch.arange(sequence, dtype=scale.dtype, device=scale.device)
        attention = positions.view(1, 1, 1, -1).expand(1, 2, sequence, -1) + 1
        attentions = tuple(attention for _ in range(32))
        return SimpleNamespace(logits=logits, attentions=attentions)

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> object:
        del pixel_values
        token = self.model.visual.merger.weight[:, 0]
        features = []
        for ordinal in range(image_grid_thw.shape[0]):
            class_offset = torch.zeros_like(token)
            class_offset[ordinal % 2] = 0.01
            features.append((token + class_offset).expand(2, -1))
        return SimpleNamespace(pooler_output=features)


def test_qwen_replay_captures_exact_merged_patch_gradient_target(tmp_path: Path) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    pair = adapter.prepare_pair(authority.fixture)
    rollouts = _MODULE.run_rollout_phase(adapter, authority.fixture)
    cleared_boundary_gradients: list[tuple[torch.Tensor, ...]] = []
    cleared_stem_gradients: list[torch.Tensor] = []
    original_clear_graphs = adapter.clear_graphs

    def record_then_clear() -> None:
        modules = (
            adapter._model.model.visual.merger,
            *tuple(adapter._model.model.visual.deepstack_merger_list),
        )
        gradients = tuple(module.weight.grad for module in modules)
        assert all(gradient is not None for gradient in gradients)
        stem_gradient = adapter._model.model.visual.stem.weight.grad
        assert stem_gradient is not None
        cleared_stem_gradients.append(stem_gradient.detach().float().clone())
        cleared_boundary_gradients.append(
            tuple(
                gradient.detach().float().clone() for gradient in gradients if gradient is not None
            )
        )
        original_clear_graphs()

    adapter.clear_graphs = record_then_clear  # type: ignore[method-assign]

    rewards = tuple(int(value) for value in authority.fixture.synthetic_rewards)
    group = _MODULE.AsgcvCompletionGroup(
        completion_ids=rollouts.completion_ids,
        expected_relation_sign=1,
        protocol_sha256="a" * 64,
        rollout_authority_sha256="b" * 64,
        candidate_pair_ordinal=0,
        generation_seeds=tuple(range(8)),
        rewards=rewards,
        correct_rollouts=tuple(bool(value) for value in rewards),
        attribute_spans=tuple(pair.attribute_token_span if value else None for value in rewards),
        nonzero_reward_variance=True,
    ).validated()

    target = _MODULE.capture_asgcv_patch_gradient(
        adapter,
        pair,
        group,
        attention_layer=authority.fixture.attention_layer,
    )

    assert target.patch_tokens.shape == (2, 8, 16)
    assert target.exact_gradient.shape == target.patch_tokens.shape
    assert target.boundary_names == ("merger", "deepstack-0", "deepstack-1", "deepstack-2")
    assert target.boundary_patch_tokens.shape == (4, 2, 2, 16)
    assert target.boundary_exact_gradient.shape == target.boundary_patch_tokens.shape
    torch.testing.assert_close(
        target.boundary_patch_tokens.permute(1, 0, 2, 3).reshape(2, 8, 16),
        target.patch_tokens,
    )
    torch.testing.assert_close(
        target.boundary_exact_gradient.permute(1, 0, 2, 3).reshape(2, 8, 16),
        target.exact_gradient,
    )
    assert target.patch_tokens.dtype == torch.float32
    assert target.exact_gradient.dtype == torch.float32
    assert torch.isfinite(target.patch_tokens).all()
    assert torch.isfinite(target.exact_gradient).all()
    assert torch.count_nonzero(target.exact_gradient) > 0
    assert len(cleared_boundary_gradients) == 1
    visual = adapter._model.model.visual
    recomputed = (
        visual.merger(visual.stem(torch.ones(4, 1))),
        *(module(visual.stem(torch.ones(4, 1))) for module in visual.deepstack_merger_list),
    )
    torch.autograd.backward(
        recomputed,
        tuple(target.boundary_exact_gradient[index].reshape(4, 16) for index in range(4)),
    )
    torch.testing.assert_close(
        visual.stem.weight.grad,
        cleared_stem_gradients[0],
        rtol=1e-6,
        atol=1e-6,
    )
    for boundary_ordinal, module in enumerate(
        (visual.merger, *tuple(visual.deepstack_merger_list))
    ):
        torch.testing.assert_close(
            module.weight.grad,
            cleared_boundary_gradients[0][boundary_ordinal],
            rtol=1e-6,
            atol=1e-6,
        )
    original_clear_graphs()
    assert target.replay.generated_tokens == sum(rollouts.token_counts)
    assert target.attention_kl > 0.0
    assert target.teacher_gradient_parameters == 0
    assert target.replay_branch_count == authority.fixture.group_size
    assert all(parameter.grad is None for parameter in adapter._vision_parameters)


def test_qwen_collapsed_verdict_control_uses_two_forced_branches_without_generation(
    tmp_path: Path,
) -> None:
    authority = _hf_authority(tmp_path)
    factory = _HfLikeFactory()
    adapter = _MODULE.load_qwen_adapter(authority, factory=factory)
    pair = adapter.prepare_pair(authority.fixture)
    target = adapter.collapsed_verdict_patch_gradient(
        pair,
        correct_completion_ids=(10, 11, 12),
        incorrect_completion_ids=(20, 21, 22),
    )

    assert factory.model.forward_calls == 2
    assert target.branch_count == 2
    assert target.generated_tokens == 0
    assert target.boundary_names == ("merger", "deepstack-0", "deepstack-1", "deepstack-2")
    assert target.patch_tokens.shape == (2, 8, 16)
    assert target.predicted_gradient.shape == target.patch_tokens.shape
    assert target.boundary_patch_tokens.shape == (4, 2, 2, 16)
    assert target.boundary_predicted_gradient.shape == target.boundary_patch_tokens.shape
    assert 0.0 < target.correct_probability < 1.0
    assert target.coefficient > 0.0
    assert torch.isfinite(target.predicted_gradient).all()
    assert torch.count_nonzero(target.predicted_gradient) > 0
    assert all(parameter.grad is None for parameter in adapter._vision_parameters)

    with pytest.raises(ValueError, match="completion"):
        adapter.collapsed_verdict_patch_gradient(
            pair,
            correct_completion_ids=(),
            incorrect_completion_ids=(20, 21, 22),
        )
    with pytest.raises(ValueError, match="branch"):
        adapter.collapsed_verdict_patch_gradient(
            pair,
            correct_completion_ids=(10, 11, 12),
            incorrect_completion_ids=(10, 11, 12),
        )


def test_qwen_replay_patch_gradient_rejects_missing_merger(tmp_path: Path) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    pair = adapter.prepare_pair(authority.fixture)
    rollouts = _MODULE.run_rollout_phase(adapter, authority.fixture)
    adapter._model.model.visual = SimpleNamespace()

    with pytest.raises(ValueError, match="vision merger authority"):
        adapter.replay_patch_gradient(
            pair,
            rollouts.completion_ids,
            _MODULE.group_normalized_advantages(authority.fixture.synthetic_rewards),
            correct_rollouts=tuple(bool(value) for value in authority.fixture.synthetic_rewards),
            attribute_spans=tuple(
                pair.attribute_token_span if value else None
                for value in authority.fixture.synthetic_rewards
            ),
            attention_layer=authority.fixture.attention_layer,
        )


def test_qwen_replay_patch_gradient_rejects_incomplete_deepstack_cut(tmp_path: Path) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    pair = adapter.prepare_pair(authority.fixture)
    rollouts = _MODULE.run_rollout_phase(adapter, authority.fixture)
    adapter._model.model.visual.deepstack_merger_list = torch.nn.ModuleList(
        tuple(adapter._model.model.visual.deepstack_merger_list)[:2]
    )

    with pytest.raises(ValueError, match="DeepStack merger authority"):
        adapter.replay_patch_gradient(
            pair,
            rollouts.completion_ids,
            _MODULE.group_normalized_advantages(authority.fixture.synthetic_rewards),
            correct_rollouts=tuple(bool(value) for value in authority.fixture.synthetic_rewards),
            attribute_spans=tuple(
                pair.attribute_token_span if value else None
                for value in authority.fixture.synthetic_rewards
            ),
            attention_layer=authority.fixture.attention_layer,
        )


def test_qwen_replay_patch_gradient_requires_a_correct_attention_teacher(tmp_path: Path) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    pair = adapter.prepare_pair(authority.fixture)
    rollouts = _MODULE.run_rollout_phase(adapter, authority.fixture)

    with pytest.raises(ValueError, match="correct rollout authority"):
        adapter.replay_patch_gradient(
            pair,
            rollouts.completion_ids,
            _MODULE.group_normalized_advantages(authority.fixture.synthetic_rewards),
            correct_rollouts=(False,) * authority.fixture.group_size,
            attribute_spans=(None,) * authority.fixture.group_size,
            attention_layer=authority.fixture.attention_layer,
        )

    with pytest.raises(ValueError, match="reward variance authority"):
        adapter.replay_patch_gradient(
            pair,
            rollouts.completion_ids,
            (0.0,) * authority.fixture.group_size,
            correct_rollouts=(True,) * authority.fixture.group_size,
            attribute_spans=(pair.attribute_token_span,) * authority.fixture.group_size,
            attention_layer=authority.fixture.attention_layer,
        )


class _HfLikeFactory:
    def __init__(self) -> None:
        self.model = _HfLikeModel()

    def load_model(self, _root: Path, **_kwargs: object) -> _HfLikeModel:
        return self.model

    def load_processor(self, _root: Path, **_kwargs: object) -> _HfLikeProcessor:
        return _HfLikeProcessor()


def _hf_authority(tmp_path: Path) -> object:
    authority = _complete_authority(tmp_path)
    return replace(
        authority,
        fixture=replace(authority.fixture, patch_tokens_per_image=2),
    )


def test_hf_shaped_qwen_adapter_executes_every_scientific_protocol(
    tmp_path: Path,
) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    adapter.validate_structure(authority)
    rollouts = _MODULE.run_rollout_phase(adapter, authority.fixture)
    replay = _MODULE.run_replay_phase(adapter, authority.fixture, rollouts)
    attention = _MODULE.run_attention_phase(adapter, adapter.pooler, authority.fixture, rollouts)
    dml = _MODULE.run_dml_floor_phase(adapter, authority.fixture)
    assert replay.vision_nonzero_gradient_parameters > 0
    assert replay.language_gradient_parameters == 0
    assert attention.layer == 26
    assert dml.embedding_shape == (64, 4096)
    assert adapter._model.forward_calls == 16


def test_complete_run_reuses_one_pooler_for_attention_and_dml(
    tmp_path: Path,
) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    calls = 0

    def count_calls(_module: object, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = adapter.pooler.register_forward_hook(count_calls)
    try:
        raw = _MODULE.run_feasibility(authority, adapter)
    finally:
        handle.remove()
    assert json.loads(raw)["outcome"] == "FITS"
    assert calls == 4


def test_complete_run_records_load_and_phase_resource_measurements(
    tmp_path: Path,
) -> None:
    authority = _hf_authority(tmp_path)
    adapter = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())

    result = json.loads(_MODULE.run_feasibility(authority, adapter))

    assert result["outcome"] == "FITS"
    assert [phase["name"] for phase in result["phases"]] == [
        "load",
        "rollout",
        "replay",
        "attention",
        "dml",
    ]
    assert all(phase["elapsed_ns"] > 0 for phase in result["phases"])
    assert all(phase["peak_rss_bytes"] > 0 for phase in result["phases"])
    assert all(phase["peak_cuda_reserved_bytes"] >= 0 for phase in result["phases"])
    scientific = result["scientific_evidence"]
    assert scientific["pooler_sha256"] == result["pooler_sha256"]
    assert scientific["rollout"]["token_counts"] == [2] * 8
    assert scientific["replay"]["generated_tokens"] == 16
    assert scientific["attention"]["teacher_shape"] == [2, 2]
    assert scientific["attention"]["patch_token_shape"] == [2, 2, 16]
    assert scientific["dml"]["embedding_shape"] == [64, 4096]


def test_qwen_pooler_initialization_is_source_bound_not_rng_bound(
    tmp_path: Path,
) -> None:
    authority = _hf_authority(tmp_path)
    torch.manual_seed(11)
    first = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())
    torch.manual_seed(29)
    second = _MODULE.load_qwen_adapter(authority, factory=_HfLikeFactory())

    assert first.pooler_sha256 == second.pooler_sha256
    assert first.pooler_sha256 == _MODULE.pooler_state_sha256(first.pooler)

    changed = replace(
        authority,
        fixture=replace(authority.fixture, source_commit="b" * 40),
    )
    third = _MODULE.load_qwen_adapter(changed, factory=_HfLikeFactory())
    assert third.pooler_sha256 != first.pooler_sha256
