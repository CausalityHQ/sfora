from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sfora
import sfora.factorized_residual_ann as factorized_residual_ann
from sfora.factorized_residual_ann import (
    FactorizedResidualArtifact,
    FactorizedResidualComponents,
    FactorizedResidualPostings,
    FactorizedResidualSpec,
    VectorStoreIdentity,
    write_factorized_residual_artifact,
)


def _spec() -> FactorizedResidualSpec:
    return FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=4,
        list_count=2,
        subquantizers=2,
        bits_per_subquantizer=3,
        probe_count=1,
        shortlist_width=3,
        return_width=2,
    )


def test_factorized_residual_spec_derives_packed_wire_geometry() -> None:
    spec = _spec()

    assert spec.codebook_size == 8
    assert spec.subvector_dimensions == 2
    assert spec.code_bytes == 1


def test_factorized_residual_spec_is_available_from_public_package() -> None:
    assert sfora.FactorizedResidualSpec is FactorizedResidualSpec


def test_factorized_residual_artifact_api_is_available_from_public_package() -> None:
    assert sfora.FactorizedResidualArtifact is FactorizedResidualArtifact
    assert sfora.FactorizedResidualComponents is FactorizedResidualComponents
    assert sfora.FactorizedResidualPostings is FactorizedResidualPostings
    assert sfora.VectorStoreIdentity is VectorStoreIdentity
    assert sfora.write_factorized_residual_artifact is write_factorized_residual_artifact


@pytest.mark.parametrize(
    "update",
    (
        {"metric": "cosine"},
        {"vector_dtype": "float16"},
        {"dimensions": True},
        {"dimensions": 0},
        {"list_count": 0},
        {"subquantizers": 0},
        {"subquantizers": 3},
        {"bits_per_subquantizer": 0},
        {"bits_per_subquantizer": 9},
        {"probe_count": 0},
        {"probe_count": 3},
        {"shortlist_width": 0},
        {"return_width": 0},
        {"return_width": 4},
    ),
)
def test_factorized_residual_spec_rejects_type_and_geometry_drift(
    update: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="factorized residual spec differs"):
        replace(_spec(), **update)


def test_factorized_residual_components_own_exact_finite_read_only_arrays() -> None:
    spec = _spec()
    coarse = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype="<f4")
    pq = np.arange(2 * 8 * 2, dtype="<f4").reshape(2, 8, 2)

    components = FactorizedResidualComponents(spec, coarse, pq)
    coarse[0, 0] = 99.0
    pq[0, 0, 0] = 99.0

    assert components.coarse_centroids[0, 0] == 0.0
    assert components.pq_codebooks[0, 0, 0] == 0.0
    assert not components.coarse_centroids.flags.writeable
    assert not components.pq_codebooks.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        components.coarse_centroids.flags.writeable = True


@pytest.mark.parametrize(
    ("coarse", "pq"),
    (
        (np.zeros((2, 3), dtype="<f4"), np.zeros((2, 8, 2), dtype="<f4")),
        (np.zeros((2, 4), dtype="<f8"), np.zeros((2, 8, 2), dtype="<f4")),
        (np.zeros((2, 4), dtype="<f4"), np.zeros((2, 7, 2), dtype="<f4")),
        (
            np.array([[0.0, 1.0, 2.0, np.nan], [4.0, 5.0, 6.0, 7.0]], dtype="<f4"),
            np.zeros((2, 8, 2), dtype="<f4"),
        ),
    ),
)
def test_factorized_residual_components_reject_shape_dtype_and_nonfinite_drift(
    coarse: np.ndarray,
    pq: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="factorized residual components differ"):
        FactorizedResidualComponents(_spec(), coarse, pq)


def test_factorized_residual_postings_own_dense_permutation_and_codes() -> None:
    offsets = np.array([0, 2, 4], dtype="<u8")
    ids = np.array([2, 0, 3, 1], dtype="<u4")
    codes = np.array([[8], [26], [44], [62]], dtype=np.uint8)

    postings = FactorizedResidualPostings(_spec(), offsets, ids, codes)
    ids[0] = 0

    assert postings.rows == 4
    assert postings.ids.tolist() == [2, 0, 3, 1]
    assert not postings.offsets.flags.writeable
    assert not postings.ids.flags.writeable
    assert not postings.codes.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        postings.ids.flags.writeable = True


@pytest.mark.parametrize(
    ("offsets", "ids", "codes"),
    (
        ([0, 3, 2], [2, 0, 3, 1], [[8], [26], [44], [62]]),
        ([0, 2, 3], [2, 0, 3, 1], [[8], [26], [44], [62]]),
        ([0, 2, 4], [2, 0, 2, 1], [[8], [26], [44], [62]]),
        ([0, 2, 4], [2, 0, 3, 1], [[8], [26], [44], [255]]),
    ),
)
def test_factorized_residual_postings_reject_offset_id_and_code_drift(
    offsets: list[int],
    ids: list[int],
    codes: list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="factorized residual postings differ"):
        FactorizedResidualPostings(
            _spec(),
            np.asarray(offsets, dtype="<u8"),
            np.asarray(ids, dtype="<u4"),
            np.asarray(codes, dtype=np.uint8),
        )


@pytest.mark.parametrize("bits_per_subquantizer", range(1, 9))
def test_packed_code_decoder_covers_every_width_and_cross_byte_boundary(
    bits_per_subquantizer: int,
) -> None:
    subquantizers = 9
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=18,
        list_count=1,
        subquantizers=subquantizers,
        bits_per_subquantizer=bits_per_subquantizer,
        probe_count=1,
        shortlist_width=1,
        return_width=1,
    )
    mask = (1 << bits_per_subquantizer) - 1
    expected = [(index * 3 + 1) & mask for index in range(subquantizers)]
    packed = bytearray(spec.code_bytes)
    for subquantizer, value in enumerate(expected):
        bit = subquantizer * bits_per_subquantizer
        for value_bit in range(bits_per_subquantizer):
            if value & (1 << value_bit):
                packed[(bit + value_bit) >> 3] |= 1 << ((bit + value_bit) & 7)
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 1], dtype="<u8"),
        np.array([0], dtype="<u4"),
        np.frombuffer(bytes(packed), dtype=np.uint8).reshape(1, spec.code_bytes),
    )

    observed = [
        int(factorized_residual_ann._code_indexes(postings.codes, spec, 0, 1, index)[0])
        for index in range(subquantizers)
    ]

    assert observed == expected


def _store_identity() -> VectorStoreIdentity:
    return VectorStoreIdentity(
        sha256="12" * 32,
        logical_bytes=24,
        physical_bytes=32,
        rows=4,
        dimensions=4,
        dtype="uint8",
        header_bytes=8,
        row_stride=4,
        zero_padding_bytes=8,
        generation="fixture-generation",
    )


def test_vector_store_identity_binds_logical_and_physical_geometry() -> None:
    identity = _store_identity()

    assert identity.payload_bytes == 16


@pytest.mark.parametrize(
    "update",
    (
        {"sha256": "12" * 31},
        {"sha256": "GG" * 32},
        {"logical_bytes": True},
        {"logical_bytes": 23},
        {"physical_bytes": 31},
        {"rows": 0},
        {"dimensions": 0},
        {"dtype": "float16"},
        {"header_bytes": 7},
        {"row_stride": 5},
        {"zero_padding_bytes": 7},
        {"generation": ""},
    ),
)
def test_vector_store_identity_rejects_schema_and_arithmetic_drift(
    update: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="vector store identity differs"):
        replace(_store_identity(), **update)


def _components_and_postings() -> tuple[
    FactorizedResidualSpec,
    FactorizedResidualComponents,
    FactorizedResidualPostings,
]:
    spec = _spec()
    coarse = np.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]], dtype="<f4"
    )
    pq = np.zeros((2, 8, 2), dtype="<f4")
    pq[:, 1, 0] = 1.0
    components = FactorizedResidualComponents(spec, coarse, pq)
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 2, 4], dtype="<u8"),
        np.array([2, 0, 3, 1], dtype="<u4"),
        np.array([[0], [9], [0], [9]], dtype=np.uint8),
    )
    return spec, components, postings


def test_writer_emits_exact_packed_norm_and_canonical_manifest_bytes(tmp_path: Path) -> None:
    spec, components, postings = _components_and_postings()
    destination = tmp_path / "artifact"

    manifest_sha256 = write_factorized_residual_artifact(
        destination,
        spec,
        components,
        postings,
        _store_identity(),
    )

    assert (destination / "codes.u8").read_bytes() == bytes((0, 9, 0, 9))
    assert (destination / "norms.u8").read_bytes() == bytes((0, 255, 0, 255))
    np.testing.assert_array_equal(
        np.fromfile(destination / "norm-low.f32", dtype="<f4"),
        np.array([0.0, 400.0], dtype="<f4"),
    )
    np.testing.assert_allclose(
        np.fromfile(destination / "norm-scale.f32", dtype="<f4"),
        np.array([2.0 / 255.0, 42.0 / 255.0], dtype="<f4"),
        rtol=0.0,
        atol=0.0,
    )
    manifest_bytes = (destination / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest["resident_bytes"] == sum(role["bytes"] for role in manifest["roles"].values())
    assert manifest["vector_store_identity"]["sha256"] == "12" * 32


def test_writer_uses_ties_to_even_for_halfway_norm_code(tmp_path: Path) -> None:
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=510,
        list_count=1,
        subquantizers=1,
        bits_per_subquantizer=2,
        probe_count=1,
        shortlist_width=3,
        return_width=1,
    )
    components = FactorizedResidualComponents(
        spec,
        np.zeros((1, 510), dtype="<f4"),
        np.stack(
            (
                np.zeros(510, dtype="<f4"),
                np.concatenate(
                    (np.ones(255, dtype="<f4"), np.zeros(255, dtype="<f4"))
                ),
                np.ones(510, dtype="<f4"),
                np.zeros(510, dtype="<f4"),
            )
        ).reshape(1, 4, 510),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 3], dtype="<u8"),
        np.array([0, 1, 2], dtype="<u4"),
        np.array([[0], [1], [2]], dtype=np.uint8),
    )
    identity = VectorStoreIdentity(
        sha256="56" * 32,
        logical_bytes=1538,
        physical_bytes=1538,
        rows=3,
        dimensions=510,
        dtype="uint8",
        header_bytes=8,
        row_stride=510,
        zero_padding_bytes=0,
        generation="rounding-tie",
    )
    destination = tmp_path / "rounding-tie"

    write_factorized_residual_artifact(
        destination, spec, components, postings, identity
    )

    assert (destination / "norms.u8").read_bytes() == bytes((0, 128, 255))


def test_writer_removes_published_directory_after_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, components, postings = _components_and_postings()
    destination = tmp_path / "artifact"

    real_fsync = factorized_residual_ann.os.fsync
    parent_stat = tmp_path.stat()

    def fail_parent_fsync(fd: int) -> None:
        descriptor_stat = factorized_residual_ann.os.fstat(fd)
        if (
            factorized_residual_ann.stat.S_ISDIR(descriptor_stat.st_mode)
            and descriptor_stat.st_dev == parent_stat.st_dev
            and descriptor_stat.st_ino == parent_stat.st_ino
        ):
            raise OSError("injected parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(factorized_residual_ann.os, "fsync", fail_parent_fsync)

    with pytest.raises(OSError, match="injected parent fsync failure"):
        write_factorized_residual_artifact(
            destination,
            spec,
            components,
            postings,
            _store_identity(),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_writer_fsyncs_completed_directory_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, components, postings = _components_and_postings()
    observed_modes: list[int] = []
    real_fsync = factorized_residual_ann.os.fsync

    def observe_fsync(fd: int) -> None:
        observed_modes.append(factorized_residual_ann.os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(factorized_residual_ann.os, "fsync", observe_fsync)

    write_factorized_residual_artifact(
        tmp_path / "artifact",
        spec,
        components,
        postings,
        _store_identity(),
    )

    assert [factorized_residual_ann.stat.S_ISDIR(mode) for mode in observed_modes[-2:]] == [
        True,
        True,
    ]


def test_atomic_directory_publication_never_replaces_concurrent_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        factorized_residual_ann._rename_directory_no_replace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_writer_rejects_nonfinite_derived_norm_parameters_without_output(tmp_path: Path) -> None:
    spec, _, postings = _components_and_postings()
    maximum = np.finfo(np.float32).max
    components = FactorizedResidualComponents(
        spec,
        np.full((2, 4), maximum, dtype="<f4"),
        np.zeros((2, 8, 2), dtype="<f4"),
    )
    destination = tmp_path / "overflow"

    with pytest.raises(ValueError, match="factorized residual norms differ"):
        write_factorized_residual_artifact(
            destination,
            spec,
            components,
            postings,
            _store_identity(),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def _single_dimension_norm_fixture(
    codeword: np.float32,
) -> tuple[
    FactorizedResidualSpec,
    FactorizedResidualComponents,
    FactorizedResidualPostings,
    VectorStoreIdentity,
]:
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="float32",
        dimensions=1,
        list_count=1,
        subquantizers=1,
        bits_per_subquantizer=1,
        probe_count=1,
        shortlist_width=2,
        return_width=1,
    )
    components = FactorizedResidualComponents(
        spec,
        np.zeros((1, 1), dtype="<f4"),
        np.array([[[0.0], [codeword]]], dtype="<f4"),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 2], dtype="<u8"),
        np.array([0, 1], dtype="<u4"),
        np.array([[0], [1]], dtype=np.uint8),
    )
    identity = VectorStoreIdentity(
        sha256="34" * 32,
        logical_bytes=16,
        physical_bytes=16,
        rows=2,
        dimensions=1,
        dtype="float32",
        header_bytes=8,
        row_stride=4,
        zero_padding_bytes=0,
        generation="norm-boundary",
    )
    return spec, components, postings, identity


def test_writer_rejects_finite_parameters_whose_decoded_norm_overflows(
    tmp_path: Path,
) -> None:
    spec, components, postings, identity = _single_dimension_norm_fixture(
        np.float32(2.0e19)
    )
    destination = tmp_path / "decoded-overflow"

    with pytest.raises(ValueError, match="factorized residual norms differ"):
        write_factorized_residual_artifact(
            destination, spec, components, postings, identity
        )

    assert not destination.exists()


def test_writer_rejects_positive_norm_range_that_underflows_stored_scale(
    tmp_path: Path,
) -> None:
    smallest_positive = np.nextafter(np.float32(0.0), np.float32(1.0))
    spec, components, postings, identity = _single_dimension_norm_fixture(
        smallest_positive
    )
    destination = tmp_path / "scale-underflow"

    with pytest.raises(ValueError, match="factorized residual norms differ"):
        write_factorized_residual_artifact(
            destination, spec, components, postings, identity
        )

    assert not destination.exists()


def test_writer_rejects_manifest_exceeding_reader_limit_without_output(
    tmp_path: Path,
) -> None:
    spec, components, postings = _components_and_postings()
    identity = replace(_store_identity(), generation="x" * (64 << 10))
    destination = tmp_path / "oversized-writer-manifest"

    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        write_factorized_residual_artifact(
            destination, spec, components, postings, identity
        )

    assert not destination.exists()


def test_artifact_open_authenticates_and_maps_exact_read_only_roles(tmp_path: Path) -> None:
    spec, components, postings = _components_and_postings()
    destination = tmp_path / "artifact"
    manifest_sha256 = write_factorized_residual_artifact(
        destination,
        spec,
        components,
        postings,
        _store_identity(),
    )

    artifact = FactorizedResidualArtifact.open(
        destination,
        manifest_sha256=manifest_sha256,
    )

    assert artifact.spec == spec
    assert artifact.rows == 4
    assert artifact.vector_store_identity == _store_identity()
    assert artifact.resident_bytes == 224
    with pytest.raises(AttributeError):
        artifact.rows = 5
    artifact.close()
    artifact.close()


def test_artifact_close_releases_every_resource_and_retries_failed_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Mapping:
        def __init__(self, *, fail_once: bool) -> None:
            self.fail_once = fail_once
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.fail_once:
                self.fail_once = False
                raise BufferError("injected mapping close failure")

    retry_mapping = Mapping(fail_once=True)
    successful_mapping = Mapping(fail_once=False)
    closed_descriptors: list[int] = []
    monkeypatch.setattr(
        factorized_residual_ann.os,
        "close",
        lambda descriptor: closed_descriptors.append(descriptor),
    )
    artifact = FactorizedResidualArtifact(
        spec=_spec(),
        rows=4,
        vector_store_identity=_store_identity(),
        resident_bytes=0,
        arrays={},
        mappings=[retry_mapping, successful_mapping],  # type: ignore[list-item]
        descriptors=[10, 11],
    )

    with pytest.raises(BufferError, match="injected mapping close failure"):
        artifact.close()

    assert closed_descriptors == [11, 10]
    assert not artifact._closed
    artifact.close()
    assert artifact._closed
    assert retry_mapping.calls == 2
    assert successful_mapping.calls == 1


def test_artifact_close_never_retries_a_descriptor_after_close_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def consumed_close(descriptor: int) -> None:
        calls.append(descriptor)
        if descriptor == 11:
            raise OSError("descriptor was consumed before error")

    monkeypatch.setattr(factorized_residual_ann.os, "close", consumed_close)
    artifact = FactorizedResidualArtifact(
        spec=_spec(),
        rows=4,
        vector_store_identity=_store_identity(),
        resident_bytes=0,
        arrays={},
        mappings=[],
        descriptors=[10, 11],
    )

    with pytest.raises(OSError, match="descriptor was consumed before error"):
        artifact.close()

    assert artifact._closed
    artifact.close()
    assert calls == [11, 10]


def test_artifact_open_preserves_original_error_and_attempts_all_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "partial-open-cleanup"
    _write_fixture_artifact(destination)
    real_close = factorized_residual_ann.os.close
    close_calls = 0

    def fail_authority(
        _spec: object,
        _arrays: object,
    ) -> None:
        raise ValueError("injected authority failure")

    def close_all_then_report_first(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        if close_calls == 1:
            raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        factorized_residual_ann,
        "_validate_derived_norm_roles",
        fail_authority,
    )
    monkeypatch.setattr(factorized_residual_ann.os, "close", close_all_then_report_first)

    with pytest.raises(ValueError, match="injected authority failure"):
        FactorizedResidualArtifact.open(destination)

    assert close_calls == 9


def _write_fixture_artifact(path: Path) -> str:
    spec, components, postings = _components_and_postings()
    return write_factorized_residual_artifact(
        path,
        spec,
        components,
        postings,
        _store_identity(),
    )


def _reroot_role_in_manifest(path: Path, role: str) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    role_bytes = (path / role).read_bytes()
    manifest["roles"][role]["bytes"] = len(role_bytes)
    manifest["roles"][role]["sha256"] = hashlib.sha256(role_bytes).hexdigest()
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )


def test_artifact_open_rejects_manifest_and_role_digest_drift(tmp_path: Path) -> None:
    wrong_manifest = tmp_path / "wrong-manifest"
    expected_sha256 = _write_fixture_artifact(wrong_manifest)
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(wrong_manifest, manifest_sha256="34" * 32)

    role_drift = tmp_path / "role-drift"
    _write_fixture_artifact(role_drift)
    role_path = role_drift / "codes.u8"
    mutated = bytearray(role_path.read_bytes())
    mutated[0] ^= 1
    role_path.write_bytes(mutated)
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(role_drift)

    assert len(expected_sha256) == 64


def test_artifact_open_rejects_reauthenticated_nonderived_norm_roles(
    tmp_path: Path,
) -> None:
    code_drift = tmp_path / "norm-code-drift"
    _write_fixture_artifact(code_drift)
    norm_path = code_drift / "norms.u8"
    norm_bytes = bytearray(norm_path.read_bytes())
    norm_bytes[0] = 1
    norm_path.write_bytes(norm_bytes)
    _reroot_role_in_manifest(code_drift, "norms.u8")

    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(code_drift)

    decode_overflow = tmp_path / "norm-decode-overflow"
    _write_fixture_artifact(decode_overflow)
    np.full(2, np.finfo(np.float32).max, dtype="<f4").tofile(
        decode_overflow / "norm-scale.f32"
    )
    _reroot_role_in_manifest(decode_overflow, "norm-scale.f32")

    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(decode_overflow)


def test_artifact_open_error_detaches_frames_that_reference_closed_mappings(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "detached-error"
    _write_fixture_artifact(destination)
    norm_path = destination / "norms.u8"
    norm_bytes = bytearray(norm_path.read_bytes())
    norm_bytes[0] = 1
    norm_path.write_bytes(norm_bytes)
    _reroot_role_in_manifest(destination, "norms.u8")

    with pytest.raises(ValueError, match="factorized residual artifact differs") as raised:
        FactorizedResidualArtifact.open(destination)

    frame_names: list[str] = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        frame_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "_validate_derived_norm_roles" not in frame_names


def test_artifact_open_rejects_noncanonical_unknown_and_concrete_type_drift(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical"
    _write_fixture_artifact(noncanonical)
    manifest_path = noncanonical / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(noncanonical)

    unknown = tmp_path / "unknown"
    _write_fixture_artifact(unknown)
    manifest_path = unknown / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["unknown"] = 1
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(unknown)

    concrete_type = tmp_path / "concrete-type"
    _write_fixture_artifact(concrete_type)
    manifest_path = concrete_type / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["rows"] = True
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(concrete_type)

    concrete_derived = tmp_path / "concrete-derived"
    _write_fixture_artifact(concrete_derived)
    manifest_path = concrete_derived / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["code_bytes"] = True
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(concrete_derived)


def test_artifact_open_rejects_oversized_manifest_before_json_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "oversized-manifest"
    _write_fixture_artifact(destination)
    (destination / "manifest.json").write_bytes(b"{" + b" " * (64 << 10))

    def parsing_must_not_run(_value: object) -> object:
        raise AssertionError("oversized manifest reached JSON parser")

    monkeypatch.setattr(factorized_residual_ann.json, "loads", parsing_must_not_run)

    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(destination)


def test_artifact_open_rejects_undeclared_directory_role(tmp_path: Path) -> None:
    destination = tmp_path / "extra-role"
    _write_fixture_artifact(destination)
    (destination / "undeclared.bin").write_bytes(b"not authenticated")

    with pytest.raises(ValueError, match="factorized residual artifact differs"):
        FactorizedResidualArtifact.open(destination)
