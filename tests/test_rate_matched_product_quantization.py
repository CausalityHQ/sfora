from __future__ import annotations

import io
from typing import cast

import pytest
import torch

from sfora.product_quantization import (
    ProductQuantizationSpec,
    balanced_product_quantization_spec,
)
from sfora.rate_matched_product_quantization import (
    RateMatchedProductQuantizer,
    fit_rate_matched_product_quantizer,
)


def _codec() -> RateMatchedProductQuantizer:
    return RateMatchedProductQuantizer.from_components(
        mean=torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32),
        components=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        spec=ProductQuantizationSpec(block_dimensions=(1, 1), codebook_size=2),
        rotation=torch.eye(2, dtype=torch.float32),
        codebooks=(
            torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
            torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
        ),
    )


def test_rate_matched_codec_is_available_from_the_public_library_surface() -> None:
    import sfora

    assert sfora.RateMatchedProductQuantizer is RateMatchedProductQuantizer
    assert sfora.fit_rate_matched_product_quantizer is fit_rate_matched_product_quantizer
    assert sfora.ProductQuantizationSpec is ProductQuantizationSpec
    assert sfora.balanced_product_quantization_spec is balanced_product_quantization_spec


def test_balanced_product_quantization_spec_uses_the_exact_requested_code_width() -> None:
    spec = balanced_product_quantization_spec(dimensions=80, bytes_per_vector=24)

    assert spec.block_dimensions == (3,) * 16 + (4,) * 8
    assert spec.dimensions == 80
    assert spec.bytes_per_vector == 24

    with pytest.raises(ValueError, match="product quantization rate geometry"):
        balanced_product_quantization_spec(dimensions=23, bytes_per_vector=24)


@pytest.mark.parametrize("codebook_size", (1, 257, True))
def test_balanced_product_quantization_spec_rejects_invalid_codebook_geometry(
    codebook_size: object,
) -> None:
    with pytest.raises(ValueError, match="product quantization rate geometry"):
        balanced_product_quantization_spec(
            dimensions=80,
            bytes_per_vector=24,
            codebook_size=cast(int, codebook_size),
        )


def test_rate_matched_codec_composes_normalized_projection_encoding_and_adc() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[2.0, -1.0, 7.0], [1.0, -3.0, -4.0], [0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    queries = torch.tensor([[1.0, 2.0, 99.0], [0.0, -2.0, 0.5]], dtype=torch.float32)

    prepared = codec.prepare_queries(queries)
    codes = codec.encode(gallery)
    observed = codec.score_codes(queries, codes)
    expected = codec.quantizer.asymmetric_squared_distances(prepared, codes)

    torch.testing.assert_close(
        torch.linalg.vector_norm(prepared.double(), dim=1),
        torch.ones(2, dtype=torch.float64),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    assert codes.shape == (3, 2)
    assert codec.input_dimensions == 3
    assert codec.reduced_dimensions == 2
    assert codec.bytes_per_vector == 2
    assert codec.shared_parameter_bytes == (3 + 2 * 3 + 2 * 2 + 4) * 4


def test_rate_matched_adc_matches_independent_nonidentity_codec_distance() -> None:
    codec = RateMatchedProductQuantizer.from_components(
        mean=torch.tensor([0.5, -0.5, 0.25, 1.0], dtype=torch.float32),
        components=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        spec=ProductQuantizationSpec(block_dimensions=(2, 1), codebook_size=2),
        rotation=torch.tensor(
            [[0.8, 0.0, 0.6], [0.0, 1.0, 0.0], [-0.6, 0.0, 0.8]],
            dtype=torch.float32,
        ),
        codebooks=(
            torch.tensor([[-0.8, -0.4], [0.6, 0.9]], dtype=torch.float32),
            torch.tensor([[-0.7], [0.5]], dtype=torch.float32),
        ),
    )
    gallery = torch.tensor([[1.5, -0.5, 0.25, 8.0], [0.5, 0.5, -0.75, -2.0]], dtype=torch.float32)
    queries = torch.tensor([[1.0, 0.0, 0.75, 4.0]], dtype=torch.float32)

    codes = codec.encode(gallery)
    prepared = codec.prepare_queries(queries)
    restored = codec.decode_reduced(codes)
    expected = (prepared[:, None, :] - restored[None, :, :]).square().sum(dim=-1)

    assert codes.dtype == torch.uint8
    assert codes.numel() * codes.element_size() == 4
    torch.testing.assert_close(codec.score_codes(queries, codes), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        codec.score_prepared_codes(prepared, codes), expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("bad_key", ("_components", "quantizer._rotation", "codebook"))
def test_rate_matched_checkpoint_rejects_invalid_state_atomically(bad_key: str) -> None:
    codec = _codec()
    before = {name: value.clone() for name, value in codec.state_dict().items()}
    mutation = {name: value.clone() for name, value in before.items()}
    if bad_key == "_components":
        mutation[bad_key][1] = mutation[bad_key][0]
    elif bad_key == "quantizer._rotation":
        mutation[bad_key].mul_(2.0)
    else:
        mutation["quantizer.quantizer.codebooks.0"][0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="rate-matched checkpoint authority"):
        codec.load_state_dict(mutation)

    for name, value in codec.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_rate_matched_checkpoint_round_trip_preserves_scores() -> None:
    source = _codec()
    restored = _codec()
    restored.load_state_dict(source.state_dict())
    query = torch.tensor([[2.0, 0.0, 0.5]], dtype=torch.float32)
    gallery = torch.tensor([[0.0, -1.0, 0.5], [1.0, 2.0, 0.5]], dtype=torch.float32)

    torch.testing.assert_close(
        restored.score_codes(query, restored.encode(gallery)),
        source.score_codes(query, source.encode(gallery)),
        rtol=0.0,
        atol=0.0,
    )


def test_rate_matched_exported_artifact_is_self_describing_and_round_trips() -> None:
    source = _codec()
    stream = io.BytesIO()
    torch.save(source.export_artifact(), stream)
    stream.seek(0)

    restored = RateMatchedProductQuantizer.from_artifact(
        torch.load(stream, map_location="cpu", weights_only=True)
    )
    values = torch.tensor([[2.0, -1.0, 0.5], [1.0, 2.0, 0.5]], dtype=torch.float32)

    assert restored.spec == source.spec
    assert restored.input_dimensions == source.input_dimensions
    torch.testing.assert_close(restored.encode(values), source.encode(values), rtol=0.0, atol=0.0)


def test_rate_matched_deployed_operations_never_build_autograd_graphs() -> None:
    codec = _codec()
    assert all(not parameter.requires_grad for parameter in codec.parameters())
    gallery = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    query = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
    codes = codec.encode(gallery)

    assert not codec.prepare_queries(query).requires_grad
    assert not codec.score_codes(query, codes).requires_grad
    assert not codec.decode_reduced(codes).requires_grad


def test_rate_matched_checkpoint_rejects_extra_key_before_copying_valid_tensors() -> None:
    codec = _codec()
    before = {name: value.clone() for name, value in codec.state_dict().items()}
    mutation = {name: value.clone() for name, value in before.items()}
    mutation["_mean"].add_(4.0)
    mutation["unexpected"] = torch.ones(1)

    with pytest.raises(RuntimeError):
        codec.load_state_dict(mutation)

    for name, value in codec.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_rate_matched_nested_checkpoint_rejection_is_atomic() -> None:
    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codec = _codec()

    parent = Parent()
    before = {name: value.clone() for name, value in parent.state_dict().items()}
    mutation = {name: value.clone() for name, value in before.items()}
    mutation["codec._mean"].add_(3.0)
    mutation["codec.unexpected"] = torch.ones(1)

    with pytest.raises(RuntimeError):
        parent.load_state_dict(mutation)

    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_rate_matched_nested_checkpoint_allows_absent_subtree_in_nonstrict_load() -> None:
    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(3, 3)
            self.codec = _codec()

    parent = Parent()
    codec_before = {name: value.clone() for name, value in parent.codec.state_dict().items()}
    backbone_state = {
        "backbone.weight": torch.full_like(parent.backbone.weight, 2.0),
        "backbone.bias": torch.full_like(parent.backbone.bias, -3.0),
    }

    incompatible = parent.load_state_dict(backbone_state, strict=False)

    assert set(incompatible.missing_keys) == {
        "codec._mean",
        "codec._components",
        "codec.quantizer._rotation",
        "codec.quantizer.quantizer.codebooks.0",
        "codec.quantizer.quantizer.codebooks.1",
    }
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(parent.backbone.weight, backbone_state["backbone.weight"])
    torch.testing.assert_close(parent.backbone.bias, backbone_state["backbone.bias"])
    for name, value in parent.codec.state_dict().items():
        torch.testing.assert_close(value, codec_before[name], rtol=0.0, atol=0.0)


def test_rate_matched_checkpoint_rejects_noncontiguous_assign_state() -> None:
    codec = _codec()
    mutation = {name: value.clone() for name, value in codec.state_dict().items()}
    wide = torch.zeros((2, 6), dtype=torch.float32)
    wide[:, ::2] = mutation["_components"]
    mutation["_components"] = wide[:, ::2]
    assert not mutation["_components"].is_contiguous()

    with pytest.raises(RuntimeError, match="rate-matched checkpoint authority"):
        codec.load_state_dict(mutation, assign=True)


def test_rate_matched_checkpoint_rejects_mixed_devices_before_assignment() -> None:
    codec = _codec()
    before = {name: value.clone() for name, value in codec.state_dict().items()}
    mutation = {name: value.clone() for name, value in before.items()}
    mutation["_components"] = mutation["_components"].to(device="meta")

    with pytest.raises(RuntimeError, match="rate-matched checkpoint authority"):
        codec.load_state_dict(mutation, assign=True)

    for name, value in codec.state_dict().items():
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_rate_matched_artifact_rejects_scalar_mean_with_validation_error() -> None:
    artifact = _codec().export_artifact()
    state = artifact["state_dict"]
    assert type(state) is dict
    state["_mean"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match="rate-matched artifact authority"):
        RateMatchedProductQuantizer.from_artifact(artifact)


@pytest.mark.parametrize(
    "mutation",
    ("schema", "extra-key", "block-shape", "codebook-size", "tensor-dtype"),
)
def test_rate_matched_artifact_rejects_version_shape_and_tensor_drift(mutation: str) -> None:
    artifact = _codec().export_artifact()
    state = cast(dict[str, torch.Tensor], artifact["state_dict"])
    if mutation == "schema":
        artifact["schema"] = "sfora-rate-matched-product-quantizer-v2"
    elif mutation == "extra-key":
        artifact["unexpected"] = 1
    elif mutation == "block-shape":
        artifact["block_dimensions"] = [1, 1]
    elif mutation == "codebook-size":
        artifact["codebook_size"] = 3
    else:
        state["_components"] = state["_components"].double()

    with pytest.raises(ValueError, match="rate-matched artifact authority"):
        RateMatchedProductQuantizer.from_artifact(artifact)


def test_rate_matched_codec_rejects_dtype_conversion_without_mutation() -> None:
    codec = _codec()
    with pytest.raises(ValueError, match="rate-matched module dtype"):
        codec.double()
    assert all(value.dtype == torch.float32 for value in codec.state_dict().values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rate_matched_codec_supports_device_transfer_for_serving() -> None:
    codec = _codec().cuda()
    gallery = torch.tensor([[2.0, -1.0, 0.5], [0.0, -1.0, 0.5]], device="cuda")
    query = torch.tensor([[1.0, 2.0, 0.5]], device="cuda")

    codes = codec.encode(gallery)
    assert codes.device.type == "cuda"
    assert codec.score_codes(query, codes).device.type == "cuda"

    direct = RateMatchedProductQuantizer.from_components(
        mean=codec.detached_mean().cuda(),
        components=codec.detached_components().cuda(),
        spec=codec.spec,
        rotation=codec.quantizer.detached_rotation().cuda(),
        codebooks=tuple(value.cuda() for value in codec.quantizer.detached_codebooks()),
    )
    assert direct.encode(gallery).device.type == "cuda"


def test_rate_matched_fit_is_deterministic_and_dataset_agnostic() -> None:
    generator = torch.Generator().manual_seed(23)
    latent = torch.randn((256, 3), generator=generator)
    values = torch.cat(
        (
            latent @ torch.tensor([[1.0, 0.2], [0.3, -0.7], [0.5, 0.8]]),
            0.02 * torch.randn((256, 4), generator=generator),
        ),
        dim=1,
    ).float()
    spec = ProductQuantizationSpec(block_dimensions=(2, 2), codebook_size=8)

    first = fit_rate_matched_product_quantizer(
        values,
        spec,
        seed=7,
        maximum_iterations=8,
        rotation_iterations=2,
    )
    second = fit_rate_matched_product_quantizer(
        values,
        spec,
        seed=7,
        maximum_iterations=8,
        rotation_iterations=2,
    )

    assert first.input_dimensions == 6
    assert first.reduced_dimensions == 4
    assert first.encode(values[:9]).shape == (9, 2)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("mean", "components"),
    (
        (torch.zeros(3), torch.eye(3)),
        (torch.zeros(3), torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        (torch.zeros(3), torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        (torch.tensor([0.0, float("nan"), 0.0]), torch.eye(2, 3)),
    ),
)
def test_rate_matched_codec_rejects_incompatible_projection_authority(
    mean: torch.Tensor, components: torch.Tensor
) -> None:
    base = _codec()
    with pytest.raises(ValueError, match="rate-matched projection authority"):
        RateMatchedProductQuantizer.from_components(
            mean=mean.float(),
            components=components.float(),
            spec=base.spec,
            rotation=base.quantizer.detached_rotation(),
            codebooks=base.quantizer.detached_codebooks(),
        )


def test_rate_matched_codec_rejects_nonfinite_and_zero_projected_rows() -> None:
    codec = _codec()
    with pytest.raises(ValueError, match="rate-matched input authority"):
        codec.prepare_queries(torch.tensor([[float("nan"), 0.0, 0.0]]))
    with pytest.raises(ValueError, match="rate-matched input authority"):
        codec.prepare_queries(torch.tensor([[1.0, -1.0, 9.0]]))


@pytest.mark.parametrize(
    ("dimensions", "bytes_per_vector"),
    ((0, 1), (3, 0), (True, 1), (3, True), (3, 4)),
)
def test_balanced_rate_geometry_rejects_nonconcrete_or_impossible_values(
    dimensions: object, bytes_per_vector: object
) -> None:
    with pytest.raises(ValueError, match="product quantization rate geometry"):
        balanced_product_quantization_spec(
            dimensions=cast(int, dimensions),
            bytes_per_vector=cast(int, bytes_per_vector),
        )


@pytest.mark.parametrize(
    "values",
    (
        torch.empty((0, 3), dtype=torch.float32),
        torch.ones((1, 3), dtype=torch.float32),
        torch.ones((8, 2), dtype=torch.float32),
        torch.ones((8, 3), dtype=torch.float64),
        torch.tensor([[float("nan"), 1.0, 2.0]] * 8),
    ),
)
def test_rate_matched_fit_rejects_invalid_fit_matrices(values: torch.Tensor) -> None:
    spec = ProductQuantizationSpec(block_dimensions=(1, 1), codebook_size=2)
    with pytest.raises(ValueError, match="rate-matched fit authority"):
        fit_rate_matched_product_quantizer(
            values, spec, seed=1, maximum_iterations=2, rotation_iterations=1
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seed": True, "maximum_iterations": 2, "rotation_iterations": 1},
        {"seed": 1, "maximum_iterations": 0, "rotation_iterations": 1},
        {"seed": 1, "maximum_iterations": 2, "rotation_iterations": 0},
    ),
)
def test_rate_matched_fit_rejects_invalid_optimization_contract(kwargs: dict[str, object]) -> None:
    values = torch.randn((8, 3), generator=torch.Generator().manual_seed(3))
    spec = ProductQuantizationSpec(block_dimensions=(1, 1), codebook_size=2)
    with pytest.raises(ValueError, match="rate-matched fit authority"):
        fit_rate_matched_product_quantizer(values, spec, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    (
        torch.empty((0, 3), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.float32),
        torch.ones((1, 3), dtype=torch.float64),
        torch.tensor([[float("inf"), 0.0, 0.0]], dtype=torch.float32),
    ),
)
def test_rate_matched_prepare_rejects_invalid_input_matrices(values: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="rate-matched input authority"):
        _codec().prepare_queries(values)
