from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from sfora.cross_seed_denoising import read_tensor_artifact, write_tensor_artifact

BINDINGS = {
    "checkpoint_sha256": "11" * 32,
    "source_commit": "22" * 20,
}


def _state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("tower.a", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("tower.count", torch.tensor([3], dtype=torch.int64)),
        )
    )


def _assert_state_equal(
    actual: OrderedDict[str, torch.Tensor],
    expected: OrderedDict[str, torch.Tensor],
) -> None:
    assert tuple(actual) == tuple(expected)
    for name in expected:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].shape == expected[name].shape
        assert torch.equal(actual[name], expected[name])


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def test_tensor_artifact_round_trip_is_byte_deterministic(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"

    left = write_tensor_artifact(left_root, _state(), role="tower", bindings=BINDINGS)
    right = write_tensor_artifact(right_root, _state(), role="tower", bindings=BINDINGS)

    assert left == right
    assert left_root.joinpath("manifest.json").read_bytes() == left
    _assert_state_equal(read_tensor_artifact(left_root, left, role="tower"), _state())


def test_tensor_artifact_rejects_nonfinite_and_concrete_type_drift(tmp_path: Path) -> None:
    nonfinite = _state()
    nonfinite["tower.a"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_tensor_artifact(tmp_path / "nan", nonfinite, role="tower", bindings=BINDINGS)

    with pytest.raises((TypeError, ValueError), match="bindings"):
        write_tensor_artifact(
            tmp_path / "bool-binding",
            _state(),
            role="tower",
            bindings={"checkpoint_sha256": False},  # type: ignore[dict-item]
        )


def test_tensor_artifact_rejects_payload_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    tensor_path = root / manifest["tensors"][0]["file"]
    payload = bytearray(tensor_path.read_bytes())
    payload[0] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(ValueError, match="digest"):
        read_tensor_artifact(root, manifest_bytes, role="tower")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update({"role": "head"}), "role"),
        (lambda value: value["bindings"].update({"source_commit": "33" * 20}), "bindings"),
        (lambda value: value["tensors"][0].update({"dtype": "torch.complex64"}), "dtype"),
        (lambda value: value["tensors"][0].update({"shape": [3]}), "shape"),
        (lambda value: value["tensors"][0].update({"bytes": 7}), "length"),
        (lambda value: value["tensors"][0].update({"file": "../escape.bin"}), "path"),
    ),
)
def test_tensor_artifact_rejects_manifest_semantic_drift(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    value = json.loads(manifest_bytes)
    mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        read_tensor_artifact(root, _canonical(value), role="tower")


def test_tensor_artifact_rejects_noncanonical_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    noncanonical = json.dumps(json.loads(manifest_bytes), indent=2).encode() + b"\n"

    with pytest.raises(ValueError, match="canonical"):
        read_tensor_artifact(root, noncanonical, role="tower")


def test_tensor_artifact_rejects_symlinked_payload(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    tensor_path = root / manifest["tensors"][0]["file"]
    original = tensor_path.with_suffix(".original")
    tensor_path.rename(original)
    tensor_path.symlink_to(original.name)

    with pytest.raises(ValueError, match="symlink"):
        read_tensor_artifact(root, manifest_bytes, role="tower")


def test_tensor_artifact_manifest_binds_complete_payload_digest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    payload_digests = tuple(record["sha256"] for record in manifest["tensors"])

    assert payload_digests == tuple(
        hashlib.sha256((root / record["file"]).read_bytes()).hexdigest()
        for record in manifest["tensors"]
    )
    assert len(manifest["state_sha256"]) == 64
