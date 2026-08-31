from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import (
    FeasibilityEvidence,
    FeasibilityOutcome,
    ObjectAuthority,
    PhaseMeasurement,
    ResourceEnvelope,
    ScientificEvidence,
    canonical_feasibility_result_bytes,
    fixture_message_serialization_sha256,
    load_fixture_authority,
    load_snapshot_authority,
    parse_canonical_object,
    project_best_case_step_ns,
    validate_feasibility_result_bytes,
)


def _phase(name: str, *, elapsed_ns: int) -> PhaseMeasurement:
    return PhaseMeasurement(
        name=name,
        completed=True,
        elapsed_ns=elapsed_ns,
        peak_cuda_reserved_bytes=1_000,
        peak_rss_bytes=2_000,
    )


def coherent_feasibility_evidence() -> FeasibilityEvidence:
    return FeasibilityEvidence(
        source_commit="a" * 40,
        controller_commit="b" * 40,
        binary_sha256="c" * 64,
        environment_sha256="d" * 64,
        host="spark-fixture",
        model=ObjectAuthority(
            role="model-snapshot-manifest",
            relative_path="model/manifest.json",
            byte_length=101,
            sha256="e" * 64,
        ),
        fixture=ObjectAuthority(
            role="synthetic-fixture",
            relative_path="fixture.json",
            byte_length=202,
            sha256="f" * 64,
        ),
        envelope=ResourceEnvelope(
            cuda_reserved_limit_bytes=103_079_215_104,
            rss_limit_bytes=118_111_600_640,
            wall_limit_ns=7_200_000_000_000,
            progress_limit_ns=300_000_000_000,
        ),
        load=_phase("load", elapsed_ns=5),
        rollout=_phase("rollout", elapsed_ns=20),
        replay=_phase("replay", elapsed_ns=30),
        attention=_phase("attention", elapsed_ns=40),
        dml=_phase("dml", elapsed_ns=10),
        deterministic=True,
        attention_available=True,
        backend_valid=True,
        authority_valid=True,
        memory_within_envelope=True,
        time_within_envelope=True,
        dataset_reads=0,
        label_reads=0,
        evaluation_reads=0,
        optimizer_steps=0,
        pooler_sha256="1" * 64,
        scientific=ScientificEvidence(
            pooler_sha256="1" * 64,
            rollout_group_size=8,
            rollout_token_counts=(2,) * 8,
            rollout_completion_sha256=tuple("2" * 64 for _ in range(8)),
            replay_loss_f64_bits="3ff0000000000000",
            replay_generated_tokens=16,
            replay_vision_nonzero_gradient_parameters=2,
            replay_language_gradient_parameters=0,
            replay_gradient_sha256="3" * 64,
            attention_layer=26,
            attention_head_count=16,
            attention_teacher_shape=(2, 4),
            attention_patch_token_shape=(2, 4, 16),
            attention_kl_f64_bits="3fe0000000000000",
            attention_teacher_unit_mass=True,
            attention_teacher_gradient_parameters=0,
            attention_pooler_nonzero_gradient_parameters=3,
            dml_batch_size=64,
            dml_embedding_shape=(64, 4096),
            dml_loss_f64_bits="3ff0000000000000",
            dml_maximum_norm_delta_ppm=0,
            dml_vision_nonzero_gradient_parameters=3,
            dml_language_gradient_parameters=0,
        ),
    )


def test_projection_uses_one_dml_microbatch_and_eight_pair_groups() -> None:
    assert (
        project_best_case_step_ns(
            dml_microbatch_ns=10,
            rollout_group_ns=20,
            replay_pair_ns=30,
            attention_pair_ns=40,
        )
        == 730
    )


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_projection_rejects_nonpositive_or_nonconcrete_timings(bad: object) -> None:
    with pytest.raises(ValueError, match="timing authority"):
        project_best_case_step_ns(
            dml_microbatch_ns=bad,  # type: ignore[arg-type]
            rollout_group_ns=20,
            replay_pair_ns=30,
            attention_pair_ns=40,
        )


def test_result_recomputes_outcome_and_rejects_incomplete_phase() -> None:
    evidence = coherent_feasibility_evidence()
    raw = canonical_feasibility_result_bytes(evidence)
    value = parse_canonical_object(raw, role="SAGA feasibility result")

    assert raw.endswith(b"\n")
    assert value["claim_eligible"] is False
    assert value["quality_metrics"] == []
    assert value["outcome"] == FeasibilityOutcome.FITS.value
    assert value["best_case_step_ns"] == 730
    assert value["pooler_sha256"] == "1" * 64
    assert value["scientific_evidence"]["attention"]["layer"] == 26
    assert len(value["result_sha256"]) == 64

    with pytest.raises(ValueError, match="phase evidence"):
        canonical_feasibility_result_bytes(
            replace(evidence, replay=replace(evidence.replay, completed=False))
        )


def test_result_validator_rejects_self_digest_and_schema_drift() -> None:
    raw = canonical_feasibility_result_bytes(coherent_feasibility_evidence())
    assert validate_feasibility_result_bytes(raw)["outcome"] == "FITS"
    value = parse_canonical_object(raw, role="result")
    scientific = dict(value["scientific_evidence"])
    replay = dict(scientific["replay"])
    replay["language_gradient_parameters"] = 1
    scientific["replay"] = replay
    for mutation in (
        {**value, "result_sha256": "0" * 64},
        {**value, "extra": 1},
        {**value, "claim_eligible": 0},
        {**value, "quality_metrics": [1]},
        {**value, "scientific_evidence": scientific},
    ):
        with pytest.raises(ValueError, match="result|scientific"):
            validate_feasibility_result_bytes(canonical_json_bytes(mutation))


def test_result_recomputes_outcome_precedence() -> None:
    evidence = coherent_feasibility_evidence()
    cases = (
        (replace(evidence, time_within_envelope=False), "TIME_BUDGET_FAIL"),
        (replace(evidence, attention_available=False), "ATTENTION_UNAVAILABLE"),
        (replace(evidence, memory_within_envelope=False), "MEMORY_FAIL"),
        (replace(evidence, deterministic=False), "DETERMINISM_FAIL"),
        (replace(evidence, backend_valid=False), "BACKEND_INVALID"),
        (replace(evidence, authority_valid=False), "AUTHORITY_INVALID"),
    )
    for mutated, expected in cases:
        value = parse_canonical_object(canonical_feasibility_result_bytes(mutated), role="result")
        assert value["outcome"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_reads", 1),
        ("label_reads", 1),
        ("evaluation_reads", 1),
        ("optimizer_steps", 1),
        ("dataset_reads", False),
    ],
)
def test_result_rejects_nonzero_or_nonconcrete_capability_counters(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="capability counters"):
        canonical_feasibility_result_bytes(
            replace(coherent_feasibility_evidence(), **{field: value})
        )


def test_object_authority_rejects_schema_type_and_digest_drift() -> None:
    valid = {
        "role": "model",
        "relative_path": "manifest.json",
        "byte_length": 1,
        "sha256": "0" * 64,
    }
    assert ObjectAuthority.from_mapping(valid).byte_length == 1

    mutations = (
        {**valid, "extra": 1},
        {**valid, "byte_length": True},
        {**valid, "sha256": "0" * 63},
        {**valid, "relative_path": "../manifest.json"},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="object authority"):
            ObjectAuthority.from_mapping(mutation)


def test_parse_canonical_object_rejects_noncanonical_bytes() -> None:
    assert parse_canonical_object(b'{"a":1}\n', role="fixture") == {"a": 1}
    for raw in (b'{"a": 1}\n', b'{"a":1}', b"[]\n"):
        with pytest.raises(ValueError, match="canonical JSON"):
            parse_canonical_object(raw, role="fixture")


def _object_row(role: str, relative_path: str, payload: bytes) -> dict[str, object]:
    return {
        "role": role,
        "relative_path": relative_path,
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_snapshot_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "model"
    root.mkdir()
    payloads = {
        "config.json": b'{"architectures":["Qwen3VLForConditionalGeneration"]}\n',
        "model.safetensors": b"fixture-weights",
    }
    for relative_path, payload in payloads.items():
        (root / relative_path).write_bytes(payload)
    rows = [
        _object_row("model-file", relative_path, payloads[relative_path])
        for relative_path in sorted(payloads)
    ]
    tree_sha256 = hashlib.sha256(canonical_json_bytes({"files": rows})).hexdigest()
    manifest = tmp_path / "snapshot.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema": "sfora-saga-snapshot-v1",
                "repository_id": "Qwen/Qwen3-VL-8B-Instruct",
                "model_revision": "1" * 40,
                "processor_revision": "2" * 40,
                "tokenizer_revision": "3" * 40,
                "snapshot_tree_sha256": tree_sha256,
                "architecture": "Qwen3VLForConditionalGeneration",
                "dtype": "bfloat16",
                "attention_backend": "eager",
                "trust_remote_code": False,
                "files": rows,
            }
        )
    )
    return root, manifest


def test_snapshot_loader_authenticates_every_registered_regular_file(
    tmp_path: Path,
) -> None:
    root, manifest = _write_snapshot_fixture(tmp_path)
    loaded = load_snapshot_authority(root=root, manifest_path=manifest)
    assert loaded.repository_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert loaded.model_revision == "1" * 40
    assert loaded.architecture == "Qwen3VLForConditionalGeneration"
    assert tuple(row.relative_path for row in loaded.files) == (
        "config.json",
        "model.safetensors",
    )


@pytest.mark.parametrize(
    "mutation",
    ["mutable-revision", "symlink", "extra-file", "wrong-length", "wrong-digest"],
)
def test_snapshot_loader_rejects_authority_drift(tmp_path: Path, mutation: str) -> None:
    root, manifest = _write_snapshot_fixture(tmp_path)
    value = parse_canonical_object(manifest.read_bytes(), role="snapshot")
    if mutation == "mutable-revision":
        value["model_revision"] = "main"
        manifest.write_bytes(canonical_json_bytes(value))
    elif mutation == "symlink":
        (root / "link").symlink_to(root / "config.json")
    elif mutation == "extra-file":
        (root / "extra.bin").write_bytes(b"extra")
    else:
        rows = value["files"]
        assert type(rows) is list and type(rows[0]) is dict
        field = "byte_length" if mutation == "wrong-length" else "sha256"
        rows[0][field] = 99 if field == "byte_length" else "0" * 64
        manifest.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        load_snapshot_authority(root=root, manifest_path=manifest)


def _image_bytes(source_commit: str, ordinal: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < 224 * 224 * 3:
        output.extend(
            hashlib.sha256(
                source_commit.encode()
                + ordinal.to_bytes(4, "little")
                + counter.to_bytes(4, "little")
            ).digest()
        )
        counter += 1
    return bytes(output[: 224 * 224 * 3])


def _write_fixture(tmp_path: Path) -> Path:
    source_commit = "4" * 40
    image_sha256 = [
        hashlib.sha256(_image_bytes(source_commit, ordinal)).hexdigest() for ordinal in range(64)
    ]
    prompt = b"List the visible car attributes and relations."
    path = tmp_path / "fixture.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "sfora-saga-synthetic-fixture-v1",
                "source_commit": source_commit,
                "controller_commit": "3" * 40,
                "model_revision": "1" * 40,
                "binary_sha256": "5" * 64,
                "environment_sha256": "6" * 64,
                "host": "spark-fixture",
                "image_width": 224,
                "image_height": 224,
                "image_sha256": image_sha256,
                "pair_ordinals": [0, 1],
                "microbatch_ordinals": list(range(64)),
                "prompt_utf8": prompt.decode(),
                "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                "message_serialization_sha256": (
                    fixture_message_serialization_sha256(prompt.decode(), (0, 1))
                ),
                "group_size": 8,
                "temperature_ppm": 700_000,
                "top_p_ppm": 950_000,
                "max_new_tokens": 1024,
                "generation_seeds": list(range(8)),
                "synthetic_rewards": [0, 1, 0, 1, 0, 1, 0, 1],
                "attention_layer": 26,
                "attribute_token_span": [0, 1],
                "patch_tokens_per_image": 2,
                "pseudo_labels": [ordinal % 2 for ordinal in range(64)],
            }
        )
    )
    return path


def test_fixture_loader_reconstructs_source_bound_images(tmp_path: Path) -> None:
    loaded = load_fixture_authority(_write_fixture(tmp_path))
    assert loaded.group_size == 8
    assert loaded.controller_commit == "3" * 40
    assert loaded.image_count == 64
    assert loaded.attention_layer == 26
    assert loaded.attribute_token_span == (0, 1)
    assert loaded.patch_tokens_per_image == 2
    assert loaded.generation_seeds == tuple(range(8))
    assert loaded.prompt_utf8 == "List the visible car attributes and relations."
    assert loaded.pair_ordinals == (0, 1)
    assert loaded.microbatch_ordinals == tuple(range(64))
    assert loaded.pseudo_labels == tuple(ordinal % 2 for ordinal in range(64))
    assert len(loaded.image_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_width", 223),
        ("group_size", True),
        ("attribute_token_span", [1, 1]),
        ("patch_tokens_per_image", 0),
        ("generation_seeds", list(range(7))),
        ("synthetic_rewards", [0] * 8),
        ("attention_layer", 25),
        ("message_serialization_sha256", "0" * 64),
        ("pseudo_labels", [0] * 64),
    ],
)
def test_fixture_loader_rejects_schema_and_science_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _write_fixture(tmp_path)
    fixture = parse_canonical_object(path.read_bytes(), role="fixture")
    fixture[field] = value
    path.write_bytes(canonical_json_bytes(fixture))
    with pytest.raises(ValueError, match="fixture"):
        load_fixture_authority(path)
