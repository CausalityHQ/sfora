from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rate_matched_pq_transfer.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rate_matched_pq_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


def test_normalize_embedding_rows_validates_and_returns_unit_float32_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, -2.0]], dtype=torch.float32)

    normalized = SUBJECT.normalize_embedding_rows(values)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[0.6, 0.8], [0.0, -1.0]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert normalized.is_contiguous()
    with pytest.raises(ValueError, match="embedding row authority"):
        SUBJECT.normalize_embedding_rows(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="embedding row authority"):
        SUBJECT.normalize_embedding_rows(torch.ones((2, 2), dtype=torch.float64))


def test_encode_in_batches_never_exceeds_the_registered_batch() -> None:
    values = torch.arange(14, dtype=torch.float32).reshape(7, 2)

    def bounded_encoder(batch: torch.Tensor) -> torch.Tensor:
        if len(batch) > 3:
            raise AssertionError("encoder batch exceeded")
        return batch[:, :1].to(torch.uint8)

    codes = SUBJECT.encode_in_batches(values, encoder=bounded_encoder, batch_size=3)

    assert codes.tolist() == [[0], [2], [4], [6], [8], [10], [12]]
    assert codes.dtype == torch.uint8
    assert codes.is_contiguous()


def test_score_adc_retrieval_excludes_self_and_uses_lowest_ordinal_ties() -> None:
    queries = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    codes = torch.arange(4, dtype=torch.uint8).unsqueeze(1)
    labels = (1, 2, 1, 2)
    distances = torch.tensor(
        [
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 0.0, 4.0, 1.0],
            [0.5, 3.0, 0.0, 2.0],
            [3.0, 0.5, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def bounded_distances(batch: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        if len(batch) > 2 or gallery.shape != (4, 1):
            raise AssertionError("score batch exceeded")
        return distances[batch[:, 0].long()]

    score = SUBJECT.score_adc_retrieval(
        queries,
        codes,
        labels,
        distance=bounded_distances,
        batch_size=2,
    )

    assert score == {
        "map_at_r": 0.5,
        "per_query_ap": (0.0, 0.0, 1.0, 1.0),
        "per_query_r1": (0.0, 0.0, 1.0, 1.0),
        "r1": 0.5,
    }


def test_score_adc_retrieval_computes_ap_over_more_than_one_positive() -> None:
    queries = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    codes = torch.arange(6, dtype=torch.uint8).unsqueeze(1)
    labels = (1, 1, 1, 2, 2, 2)
    distances = torch.tensor(
        [
            [0.0, 0.2, 0.3, 0.1, 1.0, 1.0],
            [0.1, 0.0, 0.2, 1.0, 1.0, 1.0],
            [0.1, 0.2, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.1, 0.2],
            [1.0, 1.0, 1.0, 0.1, 0.0, 0.2],
            [1.0, 1.0, 1.0, 0.1, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )

    score = SUBJECT.score_adc_retrieval(
        queries,
        codes,
        labels,
        distance=lambda batch, gallery: distances[batch[:, 0].long()],
        batch_size=3,
    )

    assert score["per_query_ap"] == (0.25, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert score["map_at_r"] == 7.0 / 8.0


def test_score_float_retrieval_is_bounded_excludes_self_and_retains_neighbors() -> None:
    values = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    labels = (1, 2, 1, 2)

    score = SUBJECT.score_float_retrieval(
        values,
        values,
        labels,
        batch_size=2,
        neighbor_width=2,
    )

    assert score == {
        "map_at_r": 0.25,
        "neighbor_ordinals": ((1, 2), (0, 2), (0, 1), (0, 1)),
        "per_query_ap": (0.0, 0.0, 1.0, 0.0),
        "per_query_r1": (0.0, 0.0, 1.0, 0.0),
        "r1": 0.25,
    }
    with pytest.raises(ValueError, match="float retrieval authority"):
        SUBJECT.score_float_retrieval(
            values,
            values.clone().index_fill(0, torch.tensor([3]), torch.nan),
            labels,
            batch_size=2,
            neighbor_width=2,
        )


def test_paired_class_bootstrap_interval_is_deterministic_and_clustered() -> None:
    candidate = (1.0, 1.0, 0.0, 0.0)
    baseline = (0.0, 0.0, 1.0, 1.0)
    labels = (1, 1, 2, 2)

    first = SUBJECT.paired_class_bootstrap_interval(
        candidate, baseline, labels, seed=7, samples=1000
    )
    second = SUBJECT.paired_class_bootstrap_interval(
        candidate, baseline, labels, seed=7, samples=1000
    )

    assert first == second
    assert first == (-1.0, 1.0)

    unequal = SUBJECT.paired_class_bootstrap_interval(
        (0.0, 0.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1, 1, 2, 2, 2),
        seed=7,
        samples=1000,
    )
    assert unequal == (0.0, 1.0)


def _write_transfer_archive(path: Path, *, fit_ready: bool = False) -> str:
    train_embeddings = np.pad(
        np.asarray(
            [[3.0, 4.0], [0.0, 2.0], [-3.0, 4.0], [0.0, -2.0]],
            dtype=np.float32,
        ),
        ((0, 0), (0, 78)),
    )
    train_labels = np.asarray([10, 10, 20, 20], dtype=np.int64)
    if fit_ready:
        train_embeddings = np.tile(train_embeddings, (64, 1))
        train_labels = np.tile(train_labels, 64)
    np.savez(
        path,
        metadata_json=np.asarray('{"encoder":"fixture"}'),
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=np.pad(
            np.asarray(
                [[4.0, 3.0], [2.0, 0.0], [-4.0, 3.0], [-2.0, 0.0]],
                dtype=np.float32,
            ),
            ((0, 0), (0, 78)),
        ),
        test_labels=np.asarray([30, 30, 40, 40], dtype=np.int64),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_transfer_archive_authenticates_required_arrays_and_normalizes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "embedding-pair.npz"
    digest = _write_transfer_archive(path)

    loaded = SUBJECT.load_transfer_archive(path, sha256=digest)

    assert loaded.sha256 == digest
    assert loaded.archive_keys == (
        "metadata_json",
        "test_embeddings",
        "test_labels",
        "train_embeddings",
        "train_labels",
    )
    assert loaded.train_embeddings.dtype == torch.float32
    assert loaded.train_embeddings.shape == (4, 80)
    torch.testing.assert_close(
        torch.linalg.vector_norm(loaded.test_embeddings.double(), dim=1),
        torch.ones(4, dtype=torch.float64),
        rtol=0.0,
        atol=2e-6,
    )
    assert loaded.train_labels == (10, 10, 20, 20)
    assert loaded.test_labels == (30, 30, 40, 40)

    with pytest.raises(ValueError, match="transfer archive authority"):
        SUBJECT.load_transfer_archive(path, sha256="0" * 64)


def test_load_transfer_archive_rejects_malformed_required_arrays(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        train_embeddings=np.ones((4, 2), dtype=np.float64),
        train_labels=np.asarray([1, 1, 2, 2], dtype=np.int64),
        test_embeddings=np.ones((4, 2), dtype=np.float32),
        test_labels=np.asarray([3, 3, 4, 4], dtype=np.int64),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="transfer archive authority"):
        SUBJECT.load_transfer_archive(path, sha256=digest)


def _literal_transfer_result() -> dict[str, Any]:
    return {
        "arms": [
            {
                "code_bytes_per_vector": 24,
                "block_dimensions": [3] * 16 + [4] * 8,
                "evaluation_seconds": 3.0,
                "fit_seconds": 2.0,
                "map_at_r": 1.0,
                "name": "pca80-opq24",
                "parameter_bytes": 133440,
                "per_query_ap": [1.0, 1.0, 1.0, 1.0],
                "per_query_r1": [1.0, 1.0, 1.0, 1.0],
                "r1": 1.0,
            },
            {
                "code_bytes_per_vector": 24,
                "block_dimensions": [3] * 16 + [4] * 8,
                "evaluation_seconds": 4.0,
                "fit_seconds": 5.0,
                "map_at_r": 0.0,
                "name": "opq24",
                "parameter_bytes": 107520,
                "per_query_ap": [0.0, 0.0, 0.0, 0.0],
                "per_query_r1": [0.0, 0.0, 0.0, 0.0],
                "r1": 0.0,
            },
            {
                "code_bytes_per_vector": 32,
                "block_dimensions": [2] * 16 + [3] * 16,
                "evaluation_seconds": 6.0,
                "fit_seconds": 7.0,
                "map_at_r": 1.0,
                "name": "opq32",
                "parameter_bytes": 107520,
                "per_query_ap": [1.0, 1.0, 1.0, 1.0],
                "per_query_r1": [1.0, 1.0, 1.0, 1.0],
                "r1": 1.0,
            },
        ],
        "bootstrap": {"samples": 100, "seed": 7},
        "claim_eligible": False,
        "config": {
            "bytes_per_vector": 24,
            "codebook_size": 256,
            "dimensions": 80,
            "maximum_iterations": 20,
            "rotation_iterations": 4,
            "seed": 50,
        },
        "contrasts": [
            {
                "baseline": "opq24",
                "map_at_r_delta": 1.0,
                "map_at_r_interval_95": [1.0, 1.0],
                "r1_delta": 1.0,
                "r1_interval_95": [1.0, 1.0],
            },
            {
                "baseline": "opq32",
                "map_at_r_delta": 0.0,
                "map_at_r_interval_95": [0.0, 0.0],
                "r1_delta": 0.0,
                "r1_interval_95": [0.0, 0.0],
            },
        ],
        "execution": {"encode_batch_size": 2048, "query_batch_size": 64},
        "input": {
            "archive_keys": [
                "metadata_json",
                "test_embeddings",
                "test_labels",
                "train_embeddings",
                "train_labels",
            ],
            "dimensions": 80,
            "sha256": "1" * 64,
            "test_rows": 4,
            "train_rows": 256,
        },
        "provenance": {
            "cuda_version": "none",
            "device": "cpu",
            "device_name": "cpu",
            "evaluator_sha256": "3" * 64,
            "input_uri": "file:///registered/input.npz",
            "numpy_version": "fixture",
            "scikit_learn_version": "fixture",
            "source_commit": "2" * 40,
            "torch_version": "fixture",
        },
        "schema": "sfora-rate-matched-pq-transfer-v1",
        "timing_scope": "full-corpus diagnostic evaluation; not serving latency",
    }


def test_transfer_result_recomputes_paired_evidence_and_canonical_bytes() -> None:
    result = _literal_transfer_result()
    labels = (30, 30, 40, 40)

    SUBJECT.validate_transfer_result(result, labels=labels)
    encoded = SUBJECT.canonical_transfer_result_bytes(result, labels=labels)

    assert (
        encoded
        == (
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )

    for mutation in (
        "aggregate",
        "contrast",
        "code-width",
        "claim",
        "batch",
        "parameter-bytes",
        "impossible-metrics",
    ):
        changed = json.loads(json.dumps(result))
        if mutation == "aggregate":
            changed["arms"][0]["map_at_r"] = 0.7
        elif mutation == "contrast":
            changed["contrasts"][0]["map_at_r_delta"] = 0.4
        elif mutation == "code-width":
            changed["arms"][2]["code_bytes_per_vector"] = 24
        elif mutation == "claim":
            changed["claim_eligible"] = True
        elif mutation == "batch":
            changed["execution"]["query_batch_size"] = 0
        elif mutation == "parameter-bytes":
            changed["arms"][0]["parameter_bytes"] += 4
        else:
            changed["arms"][0]["per_query_ap"] = [0.5, 0.5, 0.5, 0.5]
            changed["arms"][0]["map_at_r"] = 0.5
        with pytest.raises(ValueError, match="transfer result authority"):
            SUBJECT.validate_transfer_result(changed, labels=labels)


def _literal_arguments(tmp_path: Path) -> list[str]:
    return [
        "--archive",
        str(tmp_path / "input.npz"),
        "--archive-sha256",
        "1" * 64,
        "--input-uri",
        "file:///registered/input.npz",
        "--output",
        str(tmp_path / "result.json"),
        "--source-commit",
        "2" * 40,
        "--device",
        "cuda",
        "--encode-batch-size",
        "2048",
        "--query-batch-size",
        "64",
        "--bootstrap-samples",
        "10000",
        "--execute-transfer-evaluation",
    ]


def test_transfer_arguments_require_explicit_execution_and_reject_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _literal_arguments(tmp_path)

    parsed = SUBJECT.parse_arguments(arguments)

    assert parsed.archive == tmp_path / "input.npz"
    assert parsed.output == tmp_path / "result.json"
    assert parsed.device == "cuda"
    assert parsed.execute_transfer_evaluation is True
    for changed in (
        arguments[:-1],
        [*arguments, "--device", "cpu"],
        [*arguments, "--unknown"],
        [*arguments[:-2], "--query-batch-size", "0", arguments[-1]],
    ):
        with pytest.raises(SystemExit):
            SUBJECT.parse_arguments(changed)
    for suffix in (("--device", "cpu"), ("--device=cpu",)):
        monkeypatch.setattr(
            sys,
            "argv",
            ["evaluate_rate_matched_pq_transfer.py", *arguments, *suffix],
        )
        with pytest.raises(SystemExit):
            SUBJECT.parse_arguments()


def test_evaluate_transfer_runs_frozen_arm_order_and_recomputes_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = torch.zeros(256, 80, dtype=torch.float32)
    train[:, 4] = 1.0
    test = torch.eye(4, 80, dtype=torch.float32).contiguous()
    archive = SUBJECT.TransferArchive(
        archive_keys=(
            "test_embeddings",
            "test_labels",
            "train_embeddings",
            "train_labels",
        ),
        sha256="1" * 64,
        test_embeddings=test,
        test_labels=(30, 30, 40, 40),
        train_embeddings=train,
        train_labels=(10,) * 128 + (20,) * 128,
    )
    perfect = torch.tensor(
        [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    poor = torch.tensor(
        [
            [0.0, 1.0, 0.1, 1.0],
            [1.0, 0.0, 1.0, 0.1],
            [0.1, 1.0, 0.0, 1.0],
            [1.0, 0.1, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    class FakeCodec:
        def __init__(self, distances: torch.Tensor, width: int, parameter_bytes: int) -> None:
            self.distances = distances
            self.parameter_bytes = parameter_bytes
            self.width = width
            self.spec = SUBJECT.balanced_product_quantization_spec(
                dimensions=80, bytes_per_vector=width
            )

        def to(self, device: torch.device) -> FakeCodec:
            self.distances = self.distances.to(device)
            return self

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"fixture": torch.ones(self.parameter_bytes, dtype=torch.uint8)}

        def hard_encode(self, values: torch.Tensor) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            rows = torch.argmax(values[:, :4], dim=1).to(torch.uint8)
            return rows[:, None].repeat(1, self.width).contiguous()

        def encode(self, values: torch.Tensor) -> torch.Tensor:
            return self.hard_encode(values)

        def asymmetric_squared_distances(
            self, queries: torch.Tensor, codes: torch.Tensor
        ) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            return self.distances[torch.argmax(queries[:, :4], dim=1)][:, codes[:, 0].long()]

        def score_codes(self, queries: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
            return self.asymmetric_squared_distances(queries, codes)

    fit_order: list[str] = []

    def fit_rate(*args: object, **kwargs: object) -> FakeCodec:
        assert args[0] is archive.train_embeddings
        rate_spec: Any = args[1]
        assert rate_spec.block_dimensions == (3,) * 16 + (4,) * 8
        assert kwargs == {"seed": 50, "maximum_iterations": 20, "rotation_iterations": 4}
        fit_order.append("pca80-opq24")
        return FakeCodec(perfect.clone(), 24, 133440)

    def fit_opq(values: torch.Tensor, spec: Any, **kwargs: object) -> FakeCodec:
        assert values is archive.train_embeddings
        assert kwargs == {"seed": 50, "maximum_iterations": 20, "rotation_iterations": 4}
        width = spec.bytes_per_vector
        fit_order.append(f"opq{width}")
        return FakeCodec((poor if width == 24 else perfect).clone(), width, 107520)

    monkeypatch.setattr(SUBJECT, "fit_rate_matched_product_quantizer", fit_rate)
    monkeypatch.setattr(SUBJECT, "fit_optimized_product_quantizer", fit_opq)

    result = SUBJECT.evaluate_transfer(
        archive,
        source_commit="2" * 40,
        input_uri="file:///registered/input.npz",
        device=torch.device("cpu"),
        encode_batch_size=2,
        query_batch_size=2,
        bootstrap_samples=100,
    )

    assert fit_order == ["pca80-opq24", "opq24", "opq32"]
    assert [arm["name"] for arm in result["arms"]] == [
        "pca80-opq24",
        "opq24",
        "opq32",
    ]
    assert result["arms"][0]["map_at_r"] == 1.0
    assert result["arms"][1]["map_at_r"] == 0.0
    assert result["contrasts"][0]["map_at_r_delta"] == 1.0
    assert result["execution"] == {"encode_batch_size": 2, "query_batch_size": 2}
    SUBJECT.validate_transfer_result(result, labels=archive.test_labels)


def test_main_exclusively_publishes_validated_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "input.npz"
    digest = _write_transfer_archive(archive_path, fit_ready=True)
    output = tmp_path / "result.json"
    result = _literal_transfer_result()
    result["input"]["sha256"] = digest

    monkeypatch.setattr(SUBJECT, "_verify_source_authority", lambda commit: "3" * 64)

    def fixed_evaluation(*args: object, **kwargs: object) -> dict[str, Any]:
        assert output.with_name(f".{output.name}.partial").exists()
        return result

    monkeypatch.setattr(SUBJECT, "evaluate_transfer", fixed_evaluation)
    arguments = _literal_arguments(tmp_path)
    arguments[1] = str(archive_path)
    arguments[3] = digest

    assert SUBJECT.main(arguments) == 0
    assert (
        output.read_bytes()
        == (
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert not output.with_name(f".{output.name}.partial").exists()

    with pytest.raises(FileExistsError):
        SUBJECT.main(arguments)
    assert output.read_bytes() == SUBJECT.canonical_transfer_result_bytes(
        result, labels=(30, 30, 40, 40)
    )


def test_output_reservation_never_deletes_another_writers_partial(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    partial = output.with_name(f".{output.name}.partial")
    partial.write_bytes(b"other-writer")

    with pytest.raises(FileExistsError):
        SUBJECT._reserve_output(output)

    assert partial.read_bytes() == b"other-writer"
    assert not output.exists()


def test_main_rejects_source_commit_before_reserving_output(tmp_path: Path) -> None:
    arguments = _literal_arguments(tmp_path)

    with pytest.raises(ValueError, match="source authority"):
        SUBJECT.main(arguments)

    output = tmp_path / "result.json"
    assert not output.exists()
    assert not output.with_name(f".{output.name}.partial").exists()
