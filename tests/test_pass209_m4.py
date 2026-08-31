from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from sfora.pass209_m4 import (
    DESCRIPTOR_MAGIC,
    M4DescriptorHeader,
    M4Example,
    QueryEvidence,
    RescueRow,
    bootstrap_reachable,
    configure_reference_scorer,
    decode_descriptor_file,
    dominant_pair_rescuable,
    encode_descriptor_file,
    publish_new_outputs,
    rgb_record_sha256,
    score_descriptor_plane,
    validate_query_evidence,
)


def _descriptors() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)


def _header(descriptors: torch.Tensor | None = None) -> M4DescriptorHeader:
    values = _descriptors() if descriptors is None else descriptors
    payload = values.numpy().astype("<f4", copy=False).tobytes(order="C")
    return M4DescriptorHeader(
        schema="sfora-pass209-m4-descriptor-v1",
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset="cars",
        dataset_revision="3" * 40,
        dataset_examples_sha256="4" * 64,
        cell="fixture-cell",
        model_name="fixture/model",
        model_revision="5" * 40,
        readout="fixture_readout",
        rows=2,
        dimensions=2,
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_descriptor_codec_has_exact_framing_and_round_trips() -> None:
    descriptors = _descriptors()
    header = _header(descriptors)

    encoded = encode_descriptor_file(header, descriptors)
    assert DESCRIPTOR_MAGIC == b"SFORA-M4-F32-V1\n"
    assert len(DESCRIPTOR_MAGIC) == 16
    (header_length,) = struct.unpack_from("<Q", encoded, len(DESCRIPTOR_MAGIC))
    header_start = len(DESCRIPTOR_MAGIC) + 8
    header_bytes = encoded[header_start : header_start + header_length]
    assert header_bytes.endswith(b"\n")
    assert header_bytes == (
        json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert encoded[header_start + header_length :] == struct.pack("<4f", 1.0, 0.0, 0.0, -1.0)

    decoded_header, decoded = decode_descriptor_file(encoded)
    assert decoded_header == header
    torch.testing.assert_close(decoded, descriptors, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("magic", lambda blob: b"X" + blob[1:]),
        ("trailing-byte", lambda blob: blob + b"\x00"),
        ("short", lambda blob: blob[:-1]),
    ],
)
def test_descriptor_codec_rejects_framing_drift(
    name: str, mutate: Callable[[bytes], bytes]
) -> None:
    del name
    encoded = encode_descriptor_file(_header(), _descriptors())
    with pytest.raises(ValueError):
        decode_descriptor_file(mutate(encoded))


def test_descriptor_codec_rejects_header_and_tensor_authority_drift() -> None:
    descriptors = _descriptors()
    header = _header(descriptors)
    with pytest.raises(ValueError, match="payload byte count"):
        encode_descriptor_file(replace(header, payload_bytes=15), descriptors)
    with pytest.raises(ValueError, match="payload digest"):
        encode_descriptor_file(replace(header, payload_sha256="0" * 64), descriptors)
    with pytest.raises(ValueError, match="finite"):
        encode_descriptor_file(header, torch.tensor([[float("nan"), 0.0], [0.0, -1.0]]))
    with pytest.raises(ValueError, match="unit norm"):
        encode_descriptor_file(header, torch.tensor([[0.5, 0.0], [0.0, -1.0]]))


def test_descriptor_codec_rejects_noncanonical_and_duplicate_header_keys() -> None:
    encoded = encode_descriptor_file(_header(), _descriptors())
    header_offset = len(DESCRIPTOR_MAGIC) + 8
    (header_length,) = struct.unpack_from("<Q", encoded, len(DESCRIPTOR_MAGIC))
    payload = encoded[header_offset + header_length :]

    noncanonical = (json.dumps(_header().to_dict(), separators=(",", ":")) + "\n").encode()
    assert noncanonical != encoded[header_offset : header_offset + header_length]
    with pytest.raises(ValueError, match="canonical"):
        decode_descriptor_file(
            DESCRIPTOR_MAGIC + struct.pack("<Q", len(noncanonical)) + noncanonical + payload
        )

    canonical = encoded[header_offset : header_offset + header_length]
    duplicate = canonical[:-2] + b',"rows":2}\n'
    with pytest.raises(ValueError, match="duplicate"):
        decode_descriptor_file(
            DESCRIPTOR_MAGIC + struct.pack("<Q", len(duplicate)) + duplicate + payload
        )


def test_descriptor_decoder_revalidates_nonfinite_and_unit_norm_payloads() -> None:
    for values, message in [
        (torch.tensor([[float("inf"), 0.0], [0.0, -1.0]]), "finite"),
        (torch.tensor([[0.25, 0.0], [0.0, -1.0]]), "unit norm"),
    ]:
        raw = values.numpy().astype("<f4", copy=False).tobytes(order="C")
        header = replace(_header(), payload_sha256=hashlib.sha256(raw).hexdigest())
        header_bytes = (
            json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        blob = DESCRIPTOR_MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + raw
        with pytest.raises(ValueError, match=message):
            decode_descriptor_file(blob)


def test_publication_refuses_existing_aliases_and_rolls_back(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_new_outputs(((first, b"new"), (second, b"second")))
    assert first.read_bytes() == b"existing"
    assert not second.exists()

    alias = tmp_path / "nested" / ".." / "same.bin"
    same = tmp_path / "same.bin"
    with pytest.raises(ValueError, match="distinct"):
        publish_new_outputs(((alias, b"a"), (same, b"b")))


def test_publication_rolls_back_if_second_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    original_link = __import__("os").link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original_link(source, target)

    monkeypatch.setattr("sfora.pass209_m4.os.link", fail_second)
    with pytest.raises(OSError, match="injected"):
        publish_new_outputs(((first, b"first"), (second, b"second")))
    assert not first.exists()
    assert not second.exists()
    assert not first.with_name(".first.bin.partial").exists()
    assert not second.with_name(".second.bin.partial").exists()


def _score_fixture() -> tuple[torch.Tensor, tuple[M4Example, ...]]:
    descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    examples = (
        M4Example(position=0, example_id="q0", label=82),
        M4Example(position=1, example_id="q1", label=82),
        M4Example(position=2, example_id="q2", label=83),
        M4Example(position=3, example_id="q3", label=83),
    )
    return descriptors, examples


def test_reference_scorer_uses_exact_scores_lowest_ties_and_ragged_block() -> None:
    descriptors, examples = _score_fixture()
    rows = score_descriptor_plane(descriptors, examples, block_size=3)

    assert tuple(row.query_position for row in rows) == (0, 1, 2, 3)
    assert tuple(row.nearest_position for row in rows) == (1, 2, 1, 1)
    assert tuple(row.correct for row in rows) == (True, False, False, False)
    assert rows[0].nearest_score_bits == 0x00000000
    assert rows[0].best_same_position == 1
    assert rows[0].best_different_position == 2
    assert rows[0].margin_bits == 0x00000000
    assert rows[1].nearest_score_bits == 0x3F800000
    assert rows[1].best_same_position == 0
    assert rows[1].best_different_position == 2
    assert rows[1].margin_bits == 0xBF800000
    validate_query_evidence(rows, descriptors, examples, block_size=3)


def test_query_evidence_validator_rejects_score_identity_and_order_drift() -> None:
    descriptors, examples = _score_fixture()
    rows = score_descriptor_plane(descriptors, examples, block_size=3)

    mutations: tuple[tuple[QueryEvidence, ...], ...] = (
        (replace(rows[0], nearest_score_bits=rows[0].nearest_score_bits ^ 1), *rows[1:]),
        (replace(rows[0], nearest_position=2), *rows[1:]),
        (replace(rows[0], query_example_id="wrong"), *rows[1:]),
        rows[::-1],
        rows[:-1],
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="query evidence"):
            validate_query_evidence(mutation, descriptors, examples, block_size=3)


def test_reference_scorer_environment_binds_runtime_and_lock(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"fixture-lock\n")
    environment = configure_reference_scorer(lock)
    assert environment.schema == "sfora-pass209-m4-scorer-environment-v1"
    assert environment.intraop_threads == 1
    assert environment.interop_threads == 1
    assert environment.uv_lock_sha256 == hashlib.sha256(b"fixture-lock\n").hexdigest()
    assert environment.torch_version == torch.__version__
    assert environment.torch_build_config
    assert environment.cpu_architecture
    assert environment.cpu_capability
    assert environment.pillow_version
    assert environment.libjpeg_version
    assert environment.transformers_version


def test_rgb_record_digest_binds_magic_dimensions_and_exact_rgb_bytes() -> None:
    image = Image.new("RGB", (2, 1))
    image.putdata([(1, 2, 3), (4, 5, 6)])
    expected = hashlib.sha256(
        b"SFORA-M4-RGB-V1\n" + struct.pack("<II", 2, 1) + bytes((1, 2, 3, 4, 5, 6))
    ).hexdigest()
    assert rgb_record_sha256(image) == expected

    with pytest.raises(ValueError, match="RGB"):
        rgb_record_sha256(image.convert("L"))


def test_dominant_pair_uses_exact_census_materiality_floor() -> None:
    assert not dominant_pair_rescuable(rescued=15, count=63)
    assert dominant_pair_rescuable(rescued=16, count=63)
    assert dominant_pair_rescuable(rescued=63, count=63)
    with pytest.raises(ValueError, match="63"):
        dominant_pair_rescuable(rescued=16, count=62)
    with pytest.raises(ValueError, match="concrete"):
        dominant_pair_rescuable(rescued=True, count=63)


def test_pair_cluster_bootstrap_is_repeatable_and_keeps_directions_together() -> None:
    rows = tuple(
        [
            RescueRow(error_ordinal=i, query_label=82, nearest_label=83, reachable=True)
            for i in range(40)
        ]
        + [
            RescueRow(
                error_ordinal=40 + i,
                query_label=83,
                nearest_label=82,
                reachable=False,
            )
            for i in range(23)
        ]
        + [
            RescueRow(
                error_ordinal=63 + i,
                query_label=85,
                nearest_label=86,
                reachable=(i % 2 == 0),
            )
            for i in range(40)
        ]
    )
    first = bootstrap_reachable(rows)
    second = bootstrap_reachable(rows)
    assert first == second
    assert first.observed_share == pytest.approx(60 / 103)
    assert first.sample_count == 10_000
    assert first.bootstrap_mean == pytest.approx(0.5744815611034056)
    assert (first.p2_5, first.p10, first.p97_5) == pytest.approx(
        (0.5, 0.5, 0.6349206349206349)
    )
    assert first.samples_sha256 == (
        "5a969668a5c559a5d97c22ebc93520438ebaf9b035db70ea60245abf5c59161e"
    )

    # Reversing query/nearest direction leaves the unordered-pair partition unchanged.
    reversed_rows = tuple(
        replace(row, query_label=row.nearest_label, nearest_label=row.query_label)
        for row in rows
    )
    assert bootstrap_reachable(reversed_rows) == first


def test_pair_cluster_bootstrap_rejects_noncanonical_rows() -> None:
    rows = (
        RescueRow(error_ordinal=0, query_label=82, nearest_label=83, reachable=True),
        RescueRow(error_ordinal=2, query_label=83, nearest_label=82, reachable=False),
    )
    with pytest.raises(ValueError, match="ordinal"):
        bootstrap_reachable(rows)
    with pytest.raises(ValueError, match="distinct"):
        bootstrap_reachable(
            (RescueRow(error_ordinal=0, query_label=82, nearest_label=82, reachable=True),)
        )
