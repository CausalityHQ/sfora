from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sfora.data import ImageExample
from sfora.pass209_m4 import (
    REGISTERED_M4_CELLS,
    ScorerEnvironment,
    decode_descriptor_file,
)
from sfora.substrate_screen import score_frozen_substrate_evidence

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_pass209_m4_cell.py"
_SPEC = importlib.util.spec_from_file_location("run_pass209_m4_cell", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _argv(tmp_path: Path, *, cell: str = "dinov2-large") -> list[str]:
    return [
        "--cell",
        cell,
        "--prerequisite",
        str(tmp_path / "prerequisite.json"),
        "--error-manifest",
        str(tmp_path / "errors.json"),
        "--receipt-output",
        str(tmp_path / "receipt.json"),
        "--descriptor-output",
        str(tmp_path / "descriptors.bin"),
        "--query-output",
        str(tmp_path / "queries.json"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
        "--source-revision",
        "1" * 40,
        "--source-tree-digest",
        "2" * 64,
        "--uv-lock",
        str(tmp_path / "uv.lock"),
        "--execute",
    ]


def test_encoding_checkpoint_is_authenticated_and_resumable(tmp_path: Path) -> None:
    spec = replace(
        REGISTERED_M4_CELLS["dinov2-large"],
        expected_rows=4,
        descriptor_dimensions=2,
        batch_size=2,
    )
    prefix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    checkpoint_dir = tmp_path / "checkpoint"
    _MODULE._publish_encoding_checkpoint(
        checkpoint_dir,
        spec=spec,
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_examples_ordered_sha256="3" * 64,
        descriptors=prefix,
    )

    loaded = _MODULE._load_encoding_checkpoint(
        checkpoint_dir,
        spec=spec,
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_examples_ordered_sha256="3" * 64,
    )
    torch.testing.assert_close(loaded, prefix, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="source"):
        _MODULE._load_encoding_checkpoint(
            checkpoint_dir,
            spec=spec,
            source_revision="4" * 40,
            source_tree_digest="2" * 64,
            dataset_examples_ordered_sha256="3" * 64,
        )


def test_registered_m4_cells_are_exact_original_fp32_ladder() -> None:
    assert tuple(REGISTERED_M4_CELLS) == (
        "dinov2-large",
        "siglip2-so400m",
        "siglip-so400m",
    )
    assert tuple(cell.expected_correct for cell in REGISTERED_M4_CELLS.values()) == (
        1196,
        1227,
        1242,
    )
    assert tuple(cell.batch_size for cell in REGISTERED_M4_CELLS.values()) == (32, 8, 8)
    assert {cell.expected_rows for cell in REGISTERED_M4_CELLS.values()} == {1345}
    assert REGISTERED_M4_CELLS["siglip-so400m"].legacy_descriptor_sha256 == (
        "4031dc2da90588dcc39005eab92c6c519f3058c581222421ca917501dd3df071"
    )
    assert REGISTERED_M4_CELLS["dinov2-large"].legacy_descriptor_sha256 is None
    assert REGISTERED_M4_CELLS["siglip2-so400m"].legacy_descriptor_sha256 is None


def test_cell_cli_requires_exact_local_authority_and_execute(tmp_path: Path) -> None:
    args = _MODULE.parse_args(_argv(tmp_path))
    assert args.cell == "dinov2-large"
    assert args.prerequisite == tmp_path / "prerequisite.json"
    assert args.execute is True

    for flag in (
        "--execute",
        "--cell",
        "--prerequisite",
        "--error-manifest",
        "--receipt-output",
        "--descriptor-output",
        "--query-output",
        "--source-revision",
        "--source-tree-digest",
        "--uv-lock",
    ):
        argv = _argv(tmp_path)
        index = argv.index(flag)
        del argv[index : index + (1 if flag == "--execute" else 2)]
        with pytest.raises(SystemExit):
            _MODULE.parse_args(argv)


@pytest.mark.parametrize(
    "forbidden",
    (
        "--test-split",
        "--clean-classes",
        "--model-name",
        "--model-revision",
        "--readout",
        "--batch-size",
        "--query-block",
        "--resume-from-other-cell",
    ),
)
def test_cell_cli_refuses_scientific_overrides(tmp_path: Path, forbidden: str) -> None:
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*_argv(tmp_path), forbidden, "override"])


def test_runner_requires_all_offline_environment_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    _MODULE._require_offline_environment()

    monkeypatch.delenv("HF_DATASETS_OFFLINE")
    with pytest.raises(RuntimeError, match="offline"):
        _MODULE._require_offline_environment()


def test_gpu_environment_binds_nvidia_smi_to_torch_visible_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    properties = SimpleNamespace(
        name="fixture-gpu",
        major=9,
        minor=0,
        uuid="12345678-1234-1234-1234-123456789abc",
    )
    monkeypatch.setattr(_MODULE.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(_MODULE.torch.cuda, "get_device_properties", lambda _: properties)

    def run_identity(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="GPU-12345678-1234-1234-1234-123456789abc, 999.1\n")

    monkeypatch.setattr(_MODULE.subprocess, "run", run_identity)
    environment = _MODULE._gpu_environment(torch.device("cuda"))
    assert calls == [
        [
            "nvidia-smi",
            "--id=GPU-12345678-1234-1234-1234-123456789abc",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert environment["uuid"] == "GPU-12345678-1234-1234-1234-123456789abc"


def test_gpu_environment_accepts_torch_cuuid_string_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureCUuid:
        def __str__(self) -> str:
            return "12345678-1234-1234-1234-123456789abc"

    properties = SimpleNamespace(
        name="fixture-gpu",
        major=9,
        minor=0,
        uuid=FixtureCUuid(),
    )
    monkeypatch.setattr(_MODULE.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(_MODULE.torch.cuda, "get_device_properties", lambda _: properties)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="GPU-12345678-1234-1234-1234-123456789abc, 999.1\n"
        ),
    )

    environment = _MODULE._gpu_environment(torch.device("cuda"))

    assert environment["uuid"] == "GPU-12345678-1234-1234-1234-123456789abc"


def _v1_receipt() -> dict[str, object]:
    return {
        "schema": "sfora-frozen-substrate-screen-v1",
        "claim_eligible": False,
        "source_revision": "a" * 40,
        "source_tree_digest": "b" * 64,
        "dataset": "cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "dataset_examples_sha256": (
            "83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f"
        ),
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "model_name": "facebook/dinov2-large",
        "model_revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        "readout": "last_hidden_state_cls",
        "compute_dtype": "float32",
        "processor_image_shape": [224, 224],
        "descriptors_validated": True,
        "norm_tolerance": 1e-6,
        "metrics": {"correct": 1196, "queries": 1345, "recall_at_1": 1196 / 1345},
        "gates": {"expected_queries": 1345, "recall_at_1_minimum": 0.94},
        "passed": False,
    }


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_prerequisite_loader_requires_exact_bytes_schema_and_bindings(tmp_path: Path) -> None:
    value = _v1_receipt()
    payload = _canonical(value)
    path = tmp_path / "prerequisite.json"
    path.write_bytes(payload)
    spec = replace(
        REGISTERED_M4_CELLS["dinov2-large"],
        prerequisite_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert _MODULE.load_prerequisite(path, spec) == value

    for mutation in (
        {**value, "extra": 1},
        {**value, "claim_eligible": 0},
        {**value, "source_revision": 1},
        {**value, "model_revision": "0" * 40},
        {**value, "metrics": {"correct": 1195, "queries": 1345, "recall_at_1": 1195 / 1345}},
    ):
        mutated_payload = _canonical(mutation)
        path.write_bytes(mutated_payload)
        mutated_spec = replace(
            spec,
            prerequisite_sha256=hashlib.sha256(mutated_payload).hexdigest(),
        )
        with pytest.raises(ValueError, match="prerequisite"):
            _MODULE.load_prerequisite(path, mutated_spec)


def test_prerequisite_loader_rejects_digest_before_json(tmp_path: Path) -> None:
    path = tmp_path / "prerequisite.json"
    path.write_bytes(b"not-json\n")
    with pytest.raises(ValueError, match="digest"):
        _MODULE.load_prerequisite(path, REGISTERED_M4_CELLS["dinov2-large"])


def test_fake_cell_run_publishes_three_cross_bound_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [
        ImageExample(example_id="q0", image=None, label=82),
        ImageExample(example_id="q1", image=None, label=82),
        ImageExample(example_id="q2", image=None, label=83),
        ImageExample(example_id="q3", image=None, label=83),
    ]
    examples_digest = _MODULE._dataset_examples_sha256(examples)
    spec = replace(
        REGISTERED_M4_CELLS["dinov2-large"],
        descriptor_dimensions=2,
        expected_rows=4,
        expected_correct=1,
        batch_size=2,
    )
    prerequisite = _v1_receipt()
    prerequisite["dataset_examples_sha256"] = examples_digest
    prerequisite["metrics"] = {"correct": 1, "queries": 1345, "recall_at_1": 1 / 1345}
    prerequisite["passed"] = False
    prerequisite_payload = _canonical(prerequisite)
    prerequisite_path = tmp_path / "prerequisite.json"
    prerequisite_path.write_bytes(prerequisite_payload)
    spec = replace(
        spec,
        prerequisite_sha256=hashlib.sha256(prerequisite_payload).hexdigest(),
    )

    manifest = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "error_count": 103,
        "errors": [{"query_position": index} for index in range(103)],
    }
    manifest_payload = _canonical(manifest)
    manifest_path = tmp_path / "errors.json"
    manifest_path.write_bytes(manifest_payload)
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_bytes(b"fixture-lock\n")
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    environment = ScorerEnvironment(
        schema="sfora-pass209-m4-scorer-environment-v1",
        torch_version="fixture",
        torch_build_config="fixture",
        cpu_architecture="fixture",
        cpu_capability="fixture",
        cpu_isa_flags=(),
        intraop_threads=1,
        interop_threads=1,
        deterministic_algorithms=True,
        uv_lock_sha256=hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
        pillow_version="fixture",
        libjpeg_version="fixture",
        transformers_version="fixture",
    )
    dataset_calls: list[dict[str, object]] = []

    def load_fixture_examples(**kwargs: object) -> list[ImageExample]:
        dataset_calls.append(kwargs)
        return examples

    monkeypatch.setitem(_MODULE.REGISTERED_M4_CELLS, "dinov2-large", spec)
    monkeypatch.setattr(_MODULE, "_EXAMPLES_SHA256", examples_digest)
    monkeypatch.setattr(
        _MODULE,
        "_ERROR_MANIFEST_SHA256",
        hashlib.sha256(manifest_payload).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "SUBSTRATE_F0_CLASSES", (82, 83))
    monkeypatch.setattr(_MODULE, "load_image_retrieval_examples", load_fixture_examples)
    monkeypatch.setattr(_MODULE, "validate_substrate_holdout", lambda **_: None)
    monkeypatch.setattr(_MODULE, "configure_reference_scorer", lambda _: environment)
    monkeypatch.setattr(_MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_MODULE, "_gpu_environment", lambda _: {"schema": "fixture-gpu"})
    checkpoint_dir = tmp_path / "checkpoint"
    ordered_digest = _MODULE._ordered_examples_sha256(examples)
    _MODULE._publish_encoding_checkpoint(
        checkpoint_dir,
        spec=spec,
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_examples_ordered_sha256=ordered_digest,
        descriptors=descriptors[:2],
    )
    encode_starts: list[int] = []

    def encode_fixture(
        *_args: object,
        existing_descriptors: torch.Tensor,
        checkpoint_callback: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        encode_starts.append(int(existing_descriptors.shape[0]))
        assert callable(checkpoint_callback)
        checkpoint_sha256 = checkpoint_callback(descriptors[:2])
        assert isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
        return descriptors, (224, 224)

    monkeypatch.setattr(_MODULE, "_encode", encode_fixture)
    historical = score_frozen_substrate_evidence(
        descriptors,
        torch.tensor([82, 82, 83, 83], dtype=torch.int64),
        query_block=32,
    )
    cuda_queries = _MODULE.score_descriptor_plane(
        descriptors,
        tuple(
            _MODULE.M4Example(index, str(row.example_id), int(row.label))
            for index, row in enumerate(examples)
        ),
        block_size=32,
    )
    monkeypatch.setattr(_MODULE, "_historical_cuda_evidence", lambda *_: historical)
    monkeypatch.setattr(_MODULE, "_historical_cuda_queries", lambda *_: cuda_queries)

    args = _MODULE.parse_args(_argv(tmp_path))
    result = _MODULE._run(args)
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    query_payload = (tmp_path / "queries.json").read_bytes()
    descriptor_payload = (tmp_path / "descriptors.bin").read_bytes()
    header, decoded = decode_descriptor_file(descriptor_payload)

    assert result["correct"] == 1
    assert receipt["correct"] == 1
    assert receipt["reproduction_passed"] is True
    assert receipt["historical_cuda_correct"] == 1
    assert receipt["cpu_reference_correct"] == 1
    assert receipt["query_evidence_sha256"] == hashlib.sha256(query_payload).hexdigest()
    assert receipt["descriptor_file_sha256"] == hashlib.sha256(descriptor_payload).hexdigest()
    assert header.rows == 4
    assert encode_starts == [2]
    assert not checkpoint_dir.exists()
    assert dataset_calls == [{"dataset_name": "cars", "split": "train"}]
    torch.testing.assert_close(decoded, descriptors, rtol=0.0, atol=0.0)
    assert not tuple(tmp_path.glob(".*.partial"))

    with pytest.raises(FileExistsError, match="overwrite"):
        _MODULE._run(args)

    inconsistent_cuda = (
        replace(cuda_queries[0], correct=not cuda_queries[0].correct),
        *cuda_queries[1:],
    )
    monkeypatch.setattr(_MODULE, "_historical_cuda_queries", lambda *_: inconsistent_cuda)
    inconsistent_args = _MODULE.parse_args(_argv(tmp_path))
    inconsistent_args.receipt_output = tmp_path / "inconsistent-receipt.json"
    inconsistent_args.descriptor_output = tmp_path / "inconsistent-descriptors.bin"
    inconsistent_args.query_output = tmp_path / "inconsistent-queries.json"
    with pytest.raises(RuntimeError, match="CUDA query evidence"):
        _MODULE._run(inconsistent_args)
    assert not inconsistent_args.receipt_output.exists()
    monkeypatch.setattr(_MODULE, "_historical_cuda_queries", lambda *_: cuda_queries)

    correct_index = next(index for index, row in enumerate(cuda_queries) if row.correct)
    different_index = next(
        index
        for index, row in enumerate(cuda_queries)
        if row.query_label != cuda_queries[correct_index].query_label
    )
    mismatch_row = replace(
        cuda_queries[correct_index],
        nearest_position=different_index,
        nearest_example_id=cuda_queries[different_index].query_example_id,
        nearest_label=cuda_queries[different_index].query_label,
        correct=False,
    )
    mismatch_cuda = list(cuda_queries)
    mismatch_cuda[correct_index] = mismatch_row
    mismatch_cuda_tuple = tuple(mismatch_cuda)
    error_type = type(historical.errors[0])
    mismatch_errors = tuple(
        error_type(
            query_position=row.query_position,
            nearest_position=row.nearest_position,
            query_label=row.query_label,
            nearest_label=row.nearest_label,
        )
        for row in mismatch_cuda_tuple
        if not row.correct
    )
    mismatch = replace(
        historical,
        metrics=replace(historical.metrics, correct=0, recall_at_1=0.0),
        errors=mismatch_errors,
    )
    monkeypatch.setattr(_MODULE, "_historical_cuda_evidence", lambda *_: mismatch)
    monkeypatch.setattr(_MODULE, "_historical_cuda_queries", lambda *_: mismatch_cuda_tuple)
    mismatch_args = _MODULE.parse_args(_argv(tmp_path))
    mismatch_args.receipt_output = tmp_path / "mismatch-receipt.json"
    mismatch_args.descriptor_output = tmp_path / "mismatch-descriptors.bin"
    mismatch_args.query_output = tmp_path / "mismatch-queries.json"
    mismatch_result = _MODULE._run(mismatch_args)
    mismatch_receipt = json.loads(mismatch_args.receipt_output.read_bytes())
    assert mismatch_result["reproduction_passed"] is False
    assert mismatch_receipt["reproduction_passed"] is False
    assert mismatch_receipt["historical_count_passed"] is False
    assert mismatch_args.descriptor_output.is_file()
    assert mismatch_args.query_output.is_file()

    for prefix, encoded, message in (
        ("shape", (descriptors, (225, 224)), "image-shape"),
        ("count", (descriptors[:3], (224, 224)), "descriptor shape"),
    ):
        monkeypatch.setattr(_MODULE, "_encode", lambda *_args, _value=encoded, **_kwargs: _value)
        failed_args = _MODULE.parse_args(_argv(tmp_path))
        failed_args.receipt_output = tmp_path / f"{prefix}-receipt.json"
        failed_args.descriptor_output = tmp_path / f"{prefix}-descriptors.bin"
        failed_args.query_output = tmp_path / f"{prefix}-queries.json"
        with pytest.raises(RuntimeError, match=message):
            _MODULE._run(failed_args)
        assert not failed_args.receipt_output.exists()
        assert not failed_args.descriptor_output.exists()
        assert not failed_args.query_output.exists()
