from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from sfora.product_quantization import balanced_product_quantization_spec
from sfora.progressive_residual_quantization import (
    ProgressiveResidualCodes,
    ProgressiveResidualSpec,
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
