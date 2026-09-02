from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from sfora.cross_seed_denoising import (
    CandidateStates,
    build_cross_seed_candidates,
    read_tensor_artifact,
    wiener_gain,
    write_tensor_artifact,
)

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


def _candidate_fixture(
    updates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    name: str = "tower.weight",
) -> tuple[OrderedDict[str, torch.Tensor], dict[int, OrderedDict[str, torch.Tensor]]]:
    initial = OrderedDict(((name, torch.zeros_like(updates[0])),))
    endpoints = {
        seed: OrderedDict(((name, update.clone()),))
        for seed, update in zip((17, 29, 43), updates, strict=True)
    }
    return initial, endpoints


def test_wiener_gain_has_registered_closed_form_and_domain() -> None:
    assert wiener_gain(0.0) == 0.0
    assert wiener_gain(0.5) == 0.75
    assert wiener_gain(1.0) == 1.0
    with pytest.raises(ValueError, match="rho"):
        wiener_gain(-0.01)
    with pytest.raises(ValueError, match="rho"):
        wiener_gain(float("nan"))


def test_wiener_candidate_uses_one_group_per_named_tensor_and_reports_gjs() -> None:
    shared = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
    )
    orthogonal = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
    )
    initial = OrderedDict(
        (
            ("tower.a", torch.zeros(2)),
            ("tower.b", torch.zeros(2)),
            ("tower.counter", torch.tensor([7], dtype=torch.int64)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.a", shared[index]),
                ("tower.b", orthogonal[index]),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
            )
        )
        for index, seed in enumerate((17, 29, 43))
    }

    result = build_cross_seed_candidates(initial, endpoints)

    assert isinstance(result, CandidateStates)
    assert tuple(row.name for row in result.groups) == ("tower.a", "tower.b")
    assert result.groups[0].rho == 1.0
    assert result.groups[0].beta == 1.0
    assert result.groups[0].g_js == 1.0
    assert result.groups[1].rho == 0.0
    assert result.groups[1].beta == 0.0
    assert 0.0 <= result.groups[1].g_js <= 1.0
    assert torch.equal(result.wiener_denoise["tower.a"], shared[0])
    assert torch.equal(result.wiener_denoise["tower.b"], torch.zeros(2))
    assert torch.equal(result.tower_soup["tower.counter"], initial["tower.counter"])
    assert result.tower_soup["tower.a"].dtype == torch.float32


def test_wiener_zero_norm_forces_zero_rho_and_candidate_update() -> None:
    initial, endpoints = _candidate_fixture(
        (
            torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
    )
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.groups[0].cosines == (0.0, 0.0, 1.0)
    assert result.groups[0].rho == 0.0
    assert torch.equal(result.wiener_denoise["tower.weight"], torch.zeros(3))


def test_candidate_construction_is_invariant_to_all_seed_mapping_permutations() -> None:
    initial, endpoints = _candidate_fixture(
        (
            torch.tensor([[3.0, 0.4], [0.0, 0.2]]),
            torch.tensor([[3.0, -0.4], [0.0, 0.2]]),
            torch.tensor([[3.0, 0.0], [0.0, 0.2]]),
        )
    )
    authority = build_cross_seed_candidates(initial, endpoints)
    for order in itertools.permutations((17, 29, 43)):
        permuted = {seed: endpoints[seed] for seed in order}
        candidate = build_cross_seed_candidates(initial, permuted)
        assert candidate.groups == authority.groups
        assert candidate.spectral == authority.spectral
        for role in ("tower_soup", "wiener_denoise", "spectral_denoise"):
            assert torch.equal(
                getattr(candidate, role)["tower.weight"],
                getattr(authority, role)["tower.weight"],
            )


def test_spectral_candidate_uses_symmetric_contrast_edge_and_hard_rank_cut() -> None:
    mean = torch.diag(torch.tensor([3.0, 0.1]))
    noise = torch.diag(torch.tensor([0.0, 0.5]))
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))

    result = build_cross_seed_candidates(initial, endpoints)

    evidence = result.spectral[0]
    assert evidence.name == "tower.weight"
    assert evidence.kept_rank == 1
    assert evidence.total_rank == 2
    assert evidence.edge == pytest.approx(math.sqrt(2.0 / 3.0) * 0.5)
    assert evidence.retained_energy == pytest.approx(9.0)
    assert evidence.total_energy == pytest.approx(9.01)
    assert torch.equal(
        result.spectral_denoise["tower.weight"],
        torch.tensor([[3.0, 0.0], [0.0, 0.0]]),
    )


def test_spectral_vector_and_scalar_tensors_use_wiener_fallback() -> None:
    initial = OrderedDict(
        (
            ("tower.scalar", torch.tensor(0.0)),
            ("tower.vector", torch.zeros(2)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.scalar", torch.tensor(value)),
                ("tower.vector", torch.tensor([value, value * 2])),
            )
        )
        for seed, value in zip((17, 29, 43), (1.0, 1.1, 0.9), strict=True)
    }
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.spectral == ()
    assert torch.equal(
        result.spectral_denoise["tower.scalar"], result.wiener_denoise["tower.scalar"]
    )
    assert torch.equal(
        result.spectral_denoise["tower.vector"], result.wiener_denoise["tower.vector"]
    )


def test_spectral_convolution_and_rectangular_updates_preserve_shape() -> None:
    mean = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 10
    noise = torch.zeros_like(mean)
    noise[0, 0, 0, 0] = 0.05
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.spectral_denoise["tower.weight"].shape == (2, 3, 2, 2)
    assert result.spectral[0].total_rank == 2


def test_spectral_rejects_singular_value_at_registered_edge() -> None:
    mean = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))
    noise = torch.diag(torch.tensor([0.0, math.sqrt(1.5)], dtype=torch.float64))
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))
    with pytest.raises(ValueError, match="spectral edge"):
        build_cross_seed_candidates(initial, endpoints)


def test_candidate_construction_rejects_state_and_nonfloating_drift() -> None:
    initial = OrderedDict(
        (
            ("tower.counter", torch.tensor([7], dtype=torch.int64)),
            ("tower.weight", torch.zeros(2)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.ones(2)),
            )
        )
        for seed in (17, 29, 43)
    }
    endpoints[29]["tower.counter"][0] = 8
    with pytest.raises(ValueError, match="non-floating"):
        build_cross_seed_candidates(initial, endpoints)

    with pytest.raises(ValueError, match="seeds"):
        build_cross_seed_candidates(initial, {17: endpoints[17], 29: endpoints[29]})
