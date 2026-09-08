from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest
import torch

from sfora.joint_relational_compaction import (
    PackedInt8Embeddings,
    RelationalLinearEncoder,
    pack_int8_unit_embeddings,
)

_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_inshop_relational_linear.py"
_SPEC = importlib.util.spec_from_file_location("probe_inshop_relational_linear", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_committed_evidence_replays_query_aggregates_and_digest_bindings() -> None:
    evidence = Path(__file__).resolve().parents[1] / "docs/evidence/relational_linear_compaction"
    quality_path = evidence / "relational-linear-inshop-v22.json"
    latency_path = evidence / "relational-linear-latency-v22.json"
    model_path = evidence / "relational-linear-v22.sfora-rl1"
    quality = json.loads(quality_path.read_bytes())
    latency = json.loads(latency_path.read_bytes())

    assert hashlib.sha256(quality_path.read_bytes()).hexdigest() == (
        "88c82a82dfac2e685e01505fd2c4e7b72961b9af29292b40930b3cf10b7f68e0"
    )
    assert hashlib.sha256(latency_path.read_bytes()).hexdigest() == (
        "d16d8833cd39d77ab47ba204167029ec055765f6e908595c873c5a817eb2a9fd"
    )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_sha256 == "c46d5c7eff99b4962b9491688ac9a1ad5d345ea1e2bc9c4bd7ae3bfc0b186521"
    assert quality["model_artifact_sha256"] == model_sha256
    assert (
        quality["latency_receipt_sha256"] == hashlib.sha256(latency_path.read_bytes()).hexdigest()
    )
    assert latency["model_artifact_sha256"] == model_sha256
    assert quality["script_sha256"] == latency["script_sha256"]
    library_sha256 = hashlib.sha256(
        (
            Path(__file__).resolve().parents[1] / "src/sfora/joint_relational_compaction.py"
        ).read_bytes()
    ).hexdigest()
    assert quality["library_sha256"] == latency["library_sha256"] == library_sha256

    baseline = quality["per_query_evidence"]["pca64_int8"]
    identities = torch.tensor(quality["per_query_evidence"]["identity_cluster"], dtype=torch.int64)
    assert len(baseline["ap"]) == len(baseline["r1"]) == 14_218
    assert identities.shape == (14_218,)
    assert statistics.fmean(baseline["ap"]) == pytest.approx(
        quality["controls"]["pca64_int8"]["map_at_r"]
    )
    assert statistics.fmean(baseline["r1"]) == pytest.approx(
        quality["controls"]["pca64_int8"]["r1"]
    )
    for seed in ("17", "1729", "65537"):
        per_query = quality["per_query_evidence"]["arms"][seed]
        assert len(per_query["ap"]) == len(per_query["r1"]) == 14_218
        assert statistics.fmean(per_query["ap"]) == pytest.approx(
            quality["arms"][seed]["int8"]["map_at_r"]
        )
        assert statistics.fmean(per_query["r1"]) == pytest.approx(
            quality["arms"][seed]["int8"]["r1"]
        )
        numeric_seed = int(seed)
        assert _MODULE._familywise_lower_bound(
            tuple(per_query["ap"]),
            tuple(baseline["ap"]),
            identities,
            seed=numeric_seed,
            samples=quality["bootstrap_samples"],
            comparisons=quality["bootstrap_comparisons"],
        ) == pytest.approx(quality["arms"][seed]["bootstrap_map_lower_bound"])
        assert _MODULE._familywise_lower_bound(
            tuple(per_query["r1"]),
            tuple(baseline["r1"]),
            identities,
            seed=numeric_seed,
            samples=quality["bootstrap_samples"],
            comparisons=quality["bootstrap_comparisons"],
        ) == pytest.approx(quality["arms"][seed]["bootstrap_r1_lower_bound"])

    assert quality["status"] == "complete"
    assert quality["passes_quality"] is quality["passes_latency"] is True
    assert latency["passes_latency"] is True
    assert len(latency["baseline"]["samples_ns"]) == 10_000
    assert len(latency["treatment"]["samples_ns"]) == 10_000


def test_recipe_is_frozen_and_matches_the_validated_linear_method() -> None:
    assert _MODULE.DIMENSIONS == 64
    assert _MODULE.PCA_CONTROL_DIMENSIONS == 128
    assert _MODULE.BATCH == 1024
    assert _MODULE.EPOCHS == 20
    assert _MODULE.TEMPERATURE == 0.05
    assert _MODULE.LEARNING_RATE == 1e-4
    assert _MODULE.WEIGHT_DECAY == 1e-4
    assert _MODULE.SEEDS == (17, 1729, 65537)
    assert _MODULE.DEPLOYMENT_SEED == 17
    assert _MODULE.CANDIDATE_WIDTH == 256
    assert _MODULE.BOOTSTRAP_COMPARISONS == 6
    assert _MODULE.R1_LOSS_GATE == 0.002


def test_probe_requires_local_artifacts_and_explicit_execution() -> None:
    with pytest.raises(SystemExit):
        _MODULE.parse_args([])

    digest = "a" * 64
    parsed = _MODULE.parse_args(
        [
            "--student-embeddings",
            "/tmp/student.npz",
            "--student-embeddings-sha256",
            digest,
            "--teacher-embeddings",
            "/tmp/teacher.npz",
            "--teacher-embeddings-sha256",
            digest,
            "--output",
            "/tmp/result.json",
            "--model-output",
            "/tmp/model.sfora-rl1",
            "--latency-output",
            "/tmp/latency.json",
            "--execute-relational-linear",
        ]
    )
    assert parsed.execute_relational_linear is True
    assert parsed.model_output == Path("/tmp/model.sfora-rl1")
    assert parsed.latency_output == Path("/tmp/latency.json")
    with pytest.raises(SystemExit):
        _MODULE.parse_args(
            [
                "--student-embeddings",
                "relative.npz",
                "--student-embeddings-sha256",
                digest,
                "--teacher-embeddings",
                "/tmp/teacher.npz",
                "--teacher-embeddings-sha256",
                digest,
                "--output",
                "/tmp/result.json",
                "--model-output",
                "/tmp/model.sfora-rl1",
                "--latency-output",
                "/tmp/latency.json",
                "--execute-relational-linear",
            ]
        )


def test_familywise_identity_bootstrap_is_deterministic_and_conservative() -> None:
    treatment = (0.8, 0.7, 0.9, 0.6, 0.5, 0.9)
    baseline = (0.5, 0.4, 0.5, 0.4, 0.3, 0.6)
    identities = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    first = _MODULE._familywise_lower_bound(
        treatment, baseline, identities, seed=17, samples=1_000, comparisons=3
    )
    second = _MODULE._familywise_lower_bound(
        treatment, baseline, identities, seed=17, samples=1_000, comparisons=3
    )
    assert first == second
    assert 0.0 < first < 0.4
    with pytest.raises(ValueError, match="bootstrap authority"):
        _MODULE._familywise_lower_bound(
            treatment, baseline, object(), seed=17, samples=1_000, comparisons=3
        )


def test_identity_clusters_are_anonymized_stable_and_library_is_bound() -> None:
    assert _MODULE._identity_cluster_ids(("zebra", "ant", "zebra")) == (1, 0, 1)
    assert (
        _MODULE._library_sha256()
        == hashlib.sha256(
            (
                Path(__file__).resolve().parents[1] / "src/sfora/joint_relational_compaction.py"
            ).read_bytes()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="identity authority"):
        _MODULE._identity_cluster_ids(("",))


def test_familywise_bootstrap_preserves_query_weighted_estimand() -> None:
    treatment = (0.1,) * 99 + (-1.0,) * 1_000
    baseline = (0.0,) * len(treatment)
    identities = torch.tensor(tuple(range(99)) + (99,) * 1_000, dtype=torch.int64)

    lower = _MODULE._familywise_lower_bound(
        treatment, baseline, identities, seed=17, samples=10_000, comparisons=3
    )

    assert lower < 0.0


def test_out_of_budget_pca_control_can_be_encoded_as_float_only() -> None:
    encoder = RelationalLinearEncoder(torch.eye(129, dtype=torch.float32)[:128])
    values = torch.eye(2, 129, dtype=torch.float32)
    encoded = _MODULE._encode_floating(encoder, values, device=torch.device("cpu"))
    assert encoded.shape == (2, 128)


def test_deployed_encoder_scores_the_exact_packed_66_byte_representation() -> None:
    encoder = RelationalLinearEncoder(torch.eye(65, dtype=torch.float32)[:64])
    values = torch.arange(1, 131, dtype=torch.float32).reshape(2, 65)

    floating, deployed = _MODULE._encode_deployed(encoder, values, device=torch.device("cpu"))
    packed = pack_int8_unit_embeddings(floating)

    assert packed.bytes_per_vector == _MODULE.PERSISTENT_BYTES_PER_ITEM
    assert isinstance(deployed, PackedInt8Embeddings)
    assert deployed.to_bytes() == packed.to_bytes()


def test_retrieval_scorer_does_not_renormalize_packed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "CANDIDATE_WIDTH", 2)
    calls = []

    def exact_scores(
        self: PackedInt8Embeddings,
        other: PackedInt8Embeddings,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        calls.append((self.to_bytes(), other.to_bytes(), device))
        return torch.tensor([[0.9, 0.8]], dtype=torch.float32)

    monkeypatch.setattr(PackedInt8Embeddings, "cosine_similarity", exact_scores)
    queries = pack_int8_unit_embeddings(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    gallery = pack_int8_unit_embeddings(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    result = _MODULE._score(
        queries,
        gallery,
        ("match",),
        ("match", "other"),
        device=torch.device("cpu"),
    )

    assert result["r1"] == 1.0
    assert calls == [(queries.to_bytes(), gallery.to_bytes(), torch.device("cpu"))]


def test_random_control_basis_is_seeded_and_orthonormal() -> None:
    first = _MODULE._random_basis(8, 3, seed=17)
    second = _MODULE._random_basis(8, 3, seed=17)
    assert torch.equal(first, second)
    torch.testing.assert_close(first @ first.T, torch.eye(3), atol=1e-6, rtol=0)


def test_paired_latency_profiler_alternates_and_recomputes_p95() -> None:
    order = []
    ticks = iter(range(1, 10_000))

    def baseline(index: int) -> None:
        order.append(("baseline", index))

    def treatment(index: int) -> None:
        order.append(("treatment", index))

    result = _MODULE._profile_paired_latency(
        baseline,
        treatment,
        query_count=2,
        warmup_pairs=1,
        measured_pairs=3,
        clock=lambda: next(ticks),
    )

    assert order == [
        ("baseline", 0),
        ("treatment", 0),
        ("treatment", 1),
        ("baseline", 1),
        ("baseline", 0),
        ("treatment", 0),
        ("treatment", 1),
        ("baseline", 1),
    ]
    assert result["baseline"]["samples_ns"] == [1, 1, 1]
    assert result["treatment"]["samples_ns"] == [1, 1, 1]
    assert result["baseline"]["p95_ns"] == 1


def test_paired_archives_require_row_authority_and_disjoint_training_identities() -> None:
    def archive(image_list_sha256: object = "a" * 64) -> dict[str, object]:
        return {
            "metadata": {"image_list_sha256": image_list_sha256},
            "train": torch.zeros((2, 3)),
            "train_labels": ("train-a", "train-b"),
            "query": torch.zeros((1, 3)),
            "query_labels": ("test",),
            "gallery": torch.zeros((2, 3)),
            "gallery_labels": ("test", "other"),
        }

    _MODULE._validate_paired_archive_rows(archive(), archive())
    with pytest.raises(ValueError, match="paired row authority"):
        _MODULE._validate_paired_archive_rows(archive(None), archive(None))
    drifted = archive()
    drifted["train_labels"] = ("train-a", "test")
    with pytest.raises(ValueError, match="paired row authority"):
        _MODULE._validate_paired_archive_rows(drifted, archive())
