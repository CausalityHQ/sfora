from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from sfora.saga_feasibility import FixtureAuthority, SnapshotAuthority

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_saga_gb10_feasibility.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_saga_gb10_feasibility", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _valid_cli_args(tmp_path: Path) -> list[str]:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    snapshot = tmp_path / "snapshot.json"
    fixture = tmp_path / "fixture.json"
    snapshot.touch()
    fixture.touch()
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
        "a" * 40,
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
def test_cli_rejects_missing_required_flags(
    tmp_path: Path, missing_index: int
) -> None:
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
    output_keys = ("attention_mask", "image_grid_thw", "input_ids", "pixel_values")


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
        model_revision="1" * 40,
        binary_sha256="6" * 64,
        environment_sha256="7" * 64,
        host="spark-fixture",
        group_size=8,
        image_count=64,
        generation_seeds=tuple(range(8)),
        synthetic_rewards=(0, 1, 0, 1, 0, 1, 0, 1),
        attention_layer=26,
        prompt_sha256="8" * 64,
        message_serialization_sha256="9" * 64,
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
    assert all(
        parameter.requires_grad
        for parameter in factory.model.parameters_by_role["vision"]
    )
    assert all(
        not parameter.requires_grad
        for parameter in factory.model.parameters_by_role["language"]
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
def test_model_load_rejects_structural_drift(
    tmp_path: Path, attribute: str, value: object
) -> None:
    factory = _FakeFactory()
    setattr(factory.model, attribute, value)
    with pytest.raises(ValueError, match="model authority"):
        _MODULE.load_qwen_adapter(_loaded_authority(tmp_path), factory=factory)


def _fixture_authority() -> FixtureAuthority:
    return FixtureAuthority(
        source_commit="5" * 40,
        model_revision="1" * 40,
        binary_sha256="6" * 64,
        environment_sha256="7" * 64,
        host="spark-fixture",
        group_size=8,
        image_count=64,
        generation_seeds=tuple(range(8)),
        synthetic_rewards=(0, 1, 0, 1, 0, 1, 0, 1),
        attention_layer=26,
        prompt_sha256="8" * 64,
        message_serialization_sha256="9" * 64,
    )


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generation_calls: list[tuple[int, float, float, int]] = []
        self.replay_calls: list[tuple[tuple[float, ...], bool]] = []
        self.cleared = 0

    def prepare_pair(self, fixture: FixtureAuthority) -> object:
        assert fixture is _FIXTURE
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
    assert adapter.replay_calls == [
        ((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0), False)
    ]
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
def test_replay_rejects_gradient_role_drift(
    gradient: dict[str, object], message: str
) -> None:
    adapter = _RecordingAdapter()
    adapter.assert_gradient_roles = lambda: _MODULE.GradientEvidence(  # type: ignore[method-assign]
        gradient_sha256="a" * 64, **gradient
    )
    with pytest.raises(ValueError, match=message):
        _MODULE.run_replay_phase(
            adapter, _FIXTURE, _MODULE.run_rollout_phase(adapter, _FIXTURE)
        )
    assert adapter.cleared == 1


class _AttentionDmlAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.teacher = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self.patch_tokens = torch.arange(
            2 * 4 * 16, dtype=torch.float32
        ).reshape(2, 4, 16)
        self.patch_tokens.requires_grad_(True)
        self.microbatch_embeddings: torch.Tensor | None = None

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
        assert fixture is _FIXTURE
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
