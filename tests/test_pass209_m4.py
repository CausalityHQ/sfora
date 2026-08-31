from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sfora.pass209_m4 import (
    DESCRIPTOR_MAGIC,
    M4DescriptorHeader,
    decode_descriptor_file,
    encode_descriptor_file,
    publish_new_outputs,
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
