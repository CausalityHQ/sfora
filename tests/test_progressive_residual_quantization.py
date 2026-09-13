from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from sfora.product_quantization import (
    ProductQuantizationSpec,
    ProductQuantizer,
    balanced_product_quantization_spec,
)
from sfora.progressive_residual_quantization import (
    ProgressiveCandidateResult,
    ProgressiveResidualCodes,
    ProgressiveResidualQuantizer,
    ProgressiveResidualSpec,
    fit_progressive_residual_quantizer,
    pack_residual_bitplanes,
    unpack_residual_prefix,
)


def test_progressive_spec_counts_physical_prefix_bytes() -> None:
    spec = ProgressiveResidualSpec(
        metric="angular",
        base_spec=balanced_product_quantization_spec(
            dimensions=100,
            bytes_per_vector=24,
        ),
    )

    assert spec.dimensions == 100
    assert spec.residual_plane_bytes == 13
    assert spec.bytes_per_vector(residual_bits=3) == 65
    assert spec.bytes_per_vector(residual_bits=5) == 91
    assert spec.bytes_per_vector(residual_bits=8) == 130


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"metric": "cosine"}, "progressive residual spec differs"),
        ({"metric": 1}, "progressive residual spec differs"),
        ({"maximum_residual_bits": True}, "progressive residual spec differs"),
        ({"maximum_residual_bits": 0}, "progressive residual spec differs"),
        ({"maximum_residual_bits": 9}, "progressive residual spec differs"),
    ],
)
def test_progressive_spec_rejects_schema_and_concrete_type_drift(
    updates: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "metric": "angular",
        "base_spec": balanced_product_quantization_spec(
            dimensions=9,
            bytes_per_vector=3,
        ),
        "maximum_residual_bits": 8,
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=message):
        ProgressiveResidualSpec(**arguments)  # type: ignore[arg-type]


def test_progressive_spec_rejects_invalid_prefix_width() -> None:
    spec = ProgressiveResidualSpec(
        metric="squared_l2",
        base_spec=balanced_product_quantization_spec(
            dimensions=9,
            bytes_per_vector=3,
        ),
    )

    for residual_bits in (True, 0, 9):
        with pytest.raises(ValueError, match="progressive residual prefix differs"):
            spec.bytes_per_vector(residual_bits=residual_bits)  # type: ignore[arg-type]


def test_bitplanes_have_exact_msb_first_wire_bytes() -> None:
    indexes = torch.tensor(
        [[0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x55, 0xAA]],
        dtype=torch.uint8,
    )

    planes = pack_residual_bitplanes(indexes, dimensions=8)

    assert planes.shape == (1, 8, 1)
    assert planes.flatten().tolist() == [0x1D, 0x2E, 0x2D, 0x2E, 0x2D, 0x2E, 0x2D, 0x66]


def test_bitplanes_zero_padding_and_decode_every_exact_prefix() -> None:
    indexes = torch.tensor(
        [
            [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x55, 0xAA, 0xFF],
            [0xFF, 0xFE, 0x80, 0x7F, 0x01, 0x00, 0xAA, 0x55, 0x00],
        ],
        dtype=torch.uint8,
    )

    planes = pack_residual_bitplanes(indexes, dimensions=9)

    assert planes.shape == (2, 8, 2)
    assert torch.equal(planes[0, :, 1], torch.full((8,), 0x80, dtype=torch.uint8))
    assert torch.equal(planes[1, :, 1], torch.zeros(8, dtype=torch.uint8))
    for residual_bits in range(1, 9):
        assert torch.equal(
            unpack_residual_prefix(
                planes,
                dimensions=9,
                residual_bits=residual_bits,
            ),
            torch.bitwise_right_shift(indexes, 8 - residual_bits),
        )


def test_bitplane_boundaries_reject_shape_type_and_padding_drift() -> None:
    indexes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.uint8)
    planes = pack_residual_bitplanes(indexes, dimensions=9)
    bad_padding = planes.clone()
    bad_padding[0, 0, 1] |= 0x01

    invalid_indexes = (
        indexes.float(),
        indexes[:, :8],
        indexes.reshape(1, 3, 3),
    )
    for value in invalid_indexes:
        with pytest.raises(ValueError, match="progressive residual indexes differ"):
            pack_residual_bitplanes(value, dimensions=9)

    invalid_planes = (
        planes.float(),
        planes[:, :7],
        planes[:, :, :1],
        bad_padding,
    )
    for value in invalid_planes:
        with pytest.raises(ValueError, match="progressive residual planes differ"):
            unpack_residual_prefix(value, dimensions=9, residual_bits=3)


def test_progressive_codes_own_one_concrete_contiguous_device_matched_batch() -> None:
    codes = ProgressiveResidualCodes(
        base_codes=torch.zeros((2, 3), dtype=torch.uint8),
        scales=torch.ones(2, dtype=torch.float16),
        residual_planes=torch.zeros((2, 8, 2), dtype=torch.uint8),
    )

    assert codes.rows == 2
    assert codes.device == torch.device("cpu")
    with pytest.raises(FrozenInstanceError):
        codes.scales = torch.zeros(2, dtype=torch.float16)  # type: ignore[misc]

    invalid = (
        {"base_codes": torch.zeros((2, 3), dtype=torch.int16)},
        {"scales": torch.ones(2)},
        {"residual_planes": torch.zeros((2, 7, 2), dtype=torch.uint8)},
        {"residual_planes": torch.zeros((3, 8, 2), dtype=torch.uint8)},
        {"base_codes": torch.zeros((0, 3), dtype=torch.uint8)},
    )
    baseline = {
        "base_codes": torch.zeros((2, 3), dtype=torch.uint8),
        "scales": torch.ones(2, dtype=torch.float16),
        "residual_planes": torch.zeros((2, 8, 2), dtype=torch.uint8),
    }
    for update in invalid:
        arguments = {**baseline, **update}
        with pytest.raises(ValueError, match="progressive residual codes differ"):
            ProgressiveResidualCodes(**arguments)


def _fixed_codec(metric: str = "squared_l2") -> ProgressiveResidualQuantizer:
    base_spec = ProductQuantizationSpec(block_dimensions=(2,), codebook_size=2)
    base = ProductQuantizer.from_codebooks(
        base_spec,
        (torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32),),
    )
    return ProgressiveResidualQuantizer(
        ProgressiveResidualSpec(metric=metric, base_spec=base_spec),  # type: ignore[arg-type]
        base,
    )


def test_metric_preparation_normalizes_only_angular_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [-5.0, 12.0]], dtype=torch.float32)

    angular = _fixed_codec("angular").prepare_values(values)
    squared_l2 = _fixed_codec().prepare_values(values)

    torch.testing.assert_close(
        angular,
        torch.tensor([[0.6, 0.8], [-5.0 / 13.0, 12.0 / 13.0]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.equal(squared_l2, values)
    assert squared_l2.data_ptr() != values.data_ptr()


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, float("inf")]], dtype=torch.float32),
        torch.ones((1, 3), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.float64),
    ],
)
def test_metric_preparation_rejects_invalid_angular_rows(values: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="progressive residual input differs"):
        _fixed_codec("angular").prepare_values(values)


def test_encode_uses_float16_scale_and_one_maximum_rate_code() -> None:
    codec = _fixed_codec()

    codes = codec.encode(torch.tensor([[1.0, -1.0]], dtype=torch.float32))

    expected_scale = torch.tensor([1.0 / 127.5], dtype=torch.float16)
    assert torch.equal(codes.base_codes, torch.tensor([[0]], dtype=torch.uint8))
    assert torch.equal(codes.scales, expected_scale)
    assert torch.equal(
        unpack_residual_prefix(codes.residual_planes, dimensions=2, residual_bits=8),
        torch.tensor([[255, 0]], dtype=torch.uint8),
    )
    assert codes.residual_planes.shape == (1, 8, 1)


def test_decode_uses_exact_prefix_centers_and_metric_semantics() -> None:
    values = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    l2_codec = _fixed_codec()
    angular_codec = _fixed_codec("angular")
    l2_codes = l2_codec.encode(values)
    angular_codes = angular_codec.encode(values)
    scale = l2_codes.scales.float()[0]

    decoded_l2 = l2_codec.decode_prefix(l2_codes, residual_bits=4)
    decoded_angular = angular_codec.decode_prefix(angular_codes, residual_bits=4)

    torch.testing.assert_close(
        decoded_l2,
        torch.tensor([[120.0 * scale, -120.0 * scale]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        decoded_angular,
        torch.tensor([[2.0**-0.5, -(2.0**-0.5)]], dtype=torch.float32),
        rtol=2e-6,
        atol=2e-6,
    )


def test_encode_handles_zero_and_tiny_residuals_but_rejects_scale_overflow() -> None:
    codec = _fixed_codec()

    zero = codec.encode(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    tiny = codec.encode(torch.tensor([[1e-12, -1e-12]], dtype=torch.float32))

    assert zero.scales.item() == torch.finfo(torch.float16).tiny
    assert tiny.scales.item() == torch.finfo(torch.float16).tiny
    assert bool(torch.isfinite(codec.decode_prefix(zero, residual_bits=8)).all())
    with pytest.raises(ValueError, match="progressive residual scale differs"):
        codec.encode(torch.tensor([[1e10, -1e10]], dtype=torch.float32))


def test_fit_prepares_only_caller_rows_and_is_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[torch.Tensor, int, int]] = []
    base_spec = ProductQuantizationSpec(block_dimensions=(2,), codebook_size=2)

    def fake_fit(
        values: torch.Tensor,
        spec: ProductQuantizationSpec,
        *,
        seed: int,
        maximum_iterations: int,
    ) -> ProductQuantizer:
        assert spec == base_spec
        received.append((values.clone(), seed, maximum_iterations))
        return ProductQuantizer.from_codebooks(
            spec,
            (torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),),
        )

    monkeypatch.setattr(
        "sfora.progressive_residual_quantization.fit_product_quantizer",
        fake_fit,
    )
    values = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)

    fitted = fit_progressive_residual_quantizer(
        values,
        ProgressiveResidualSpec(metric="angular", base_spec=base_spec),
        seed=17,
        maximum_iterations=9,
    )

    assert fitted.spec.metric == "angular"
    assert len(received) == 1
    torch.testing.assert_close(
        received[0][0],
        torch.tensor([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert received[0][1:] == (17, 9)


def test_codec_artifact_roundtrip_and_mutations() -> None:
    codec = _fixed_codec("angular")
    artifact = codec.export_artifact()
    restored = ProgressiveResidualQuantizer.from_artifact(artifact)
    values = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)

    original_codes = codec.encode(values)
    restored_codes = restored.encode(values)

    assert torch.equal(restored_codes.base_codes, original_codes.base_codes)
    assert torch.equal(restored_codes.scales, original_codes.scales)
    assert torch.equal(restored_codes.residual_planes, original_codes.residual_planes)

    mutations = []
    for key, value in (
        ("schema", "other"),
        ("metric", "cosine"),
        ("maximum_residual_bits", True),
        ("block_dimensions", (1, 1)),
        ("codebook_size", 3),
    ):
        mutations.append({**artifact, key: value})
    mutations.append({key: value for key, value in artifact.items() if key != "metric"})
    mutations.append({**artifact, "extra": 1})
    bad_codebooks = list(artifact["codebooks"])  # type: ignore[arg-type]
    bad_codebooks[0] = bad_codebooks[0].double()
    mutations.append({**artifact, "codebooks": tuple(bad_codebooks)})
    for mutation in mutations:
        with pytest.raises(ValueError, match="progressive residual artifact differs"):
            ProgressiveResidualQuantizer.from_artifact(mutation)


def test_candidate_scoring_uses_metric_order_and_lowest_ordinal_ties() -> None:
    l2_codec = _fixed_codec()
    angular_codec = _fixed_codec("angular")
    l2_gallery = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        dtype=torch.float32,
    )
    angular_gallery = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    candidates = torch.tensor([[3, 2, 1, 0]], dtype=torch.int64)

    l2 = l2_codec.score_candidates(
        torch.tensor([[0.9, 0.0]], dtype=torch.float32),
        l2_codec.encode(l2_gallery),
        candidates,
        residual_bits=8,
        return_width=4,
    )
    angular = angular_codec.score_candidates(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        angular_codec.encode(angular_gallery),
        candidates,
        residual_bits=8,
        return_width=4,
    )
    tied = l2_codec.score_candidates(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        l2_codec.encode(l2_gallery[:2]),
        torch.tensor([[1, 0]], dtype=torch.int64),
        residual_bits=8,
        return_width=2,
    )

    assert l2.ordinals.tolist() == [[1, 0, 2, 3]]
    assert angular.ordinals.tolist() == [[1, 2, 3, 0]]
    assert tied.ordinals.tolist() == [[0, 1]]
    assert bool(torch.all(l2.scores[:, 1:] >= l2.scores[:, :-1]))
    assert bool(torch.all(angular.scores[:, 1:] <= angular.scores[:, :-1]))


def test_candidate_scoring_is_bounded_and_counts_physical_stage_bytes() -> None:
    codec = _fixed_codec()
    gallery = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        dtype=torch.float32,
    )
    candidates = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=torch.int64)

    result = codec.score_candidates(
        torch.tensor([[0.9, 0.0], [-0.9, 0.0]], dtype=torch.float32),
        codec.encode(gallery),
        candidates,
        residual_bits=4,
        return_width=2,
    )

    assert type(result) is ProgressiveCandidateResult
    assert result.ordinals.shape == (2, 2)
    assert result.scores.shape == (2, 2)
    assert result.base_bytes_read == 2 * 4 * 1
    assert result.residual_bytes_read == 2 * 4 * (2 + 4 * 1)


@pytest.mark.parametrize(
    "candidates",
    [
        torch.tensor([[0, 0]], dtype=torch.int64),
        torch.tensor([[0, 4]], dtype=torch.int64),
        torch.tensor([[0, -1]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int64),
    ],
)
def test_candidate_scoring_rejects_ordinal_authority_drift(candidates: torch.Tensor) -> None:
    codec = _fixed_codec()
    gallery = codec.encode(torch.tensor([[float(i), 0.0] for i in range(4)]))

    with pytest.raises(ValueError, match="progressive candidate authority differs"):
        codec.score_candidates(
            torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            gallery,
            candidates,
            residual_bits=4,
            return_width=1,
        )


def test_candidate_scoring_rejects_width_query_and_prefix_drift() -> None:
    codec = _fixed_codec()
    gallery = codec.encode(torch.tensor([[float(i), 0.0] for i in range(4)]))
    candidates = torch.tensor([[0, 1]], dtype=torch.int64)

    invalid = (
        {"queries": object()},
        {"queries": torch.ones((1, 3), dtype=torch.float32)},
        {"queries": torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)},
        {"return_width": 3},
        {"return_width": True},
        {"residual_bits": 0},
    )
    baseline: dict[str, object] = {
        "queries": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "residual_bits": 4,
        "return_width": 1,
    }
    for update in invalid:
        arguments = {**baseline, **update}
        with pytest.raises(ValueError):
            codec.score_candidates(
                arguments["queries"],  # type: ignore[arg-type]
                gallery,
                candidates,
                residual_bits=arguments["residual_bits"],  # type: ignore[arg-type]
                return_width=arguments["return_width"],  # type: ignore[arg-type]
            )
