"""Rate-matched dimensionality reduction composed with product quantization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn

from sfora.product_quantization import (
    OptimizedProductQuantizer,
    ProductQuantizationSpec,
    fit_optimized_product_quantizer,
)
from sfora.representation_ceiling import fit_centered_pca


class RateMatchedProductQuantizer(nn.Module):
    """A fitted normalized linear representation followed by a fixed-byte OPQ codec."""

    _mean: torch.Tensor
    _components: torch.Tensor

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        components: torch.Tensor,
        spec: ProductQuantizationSpec,
        rotation: torch.Tensor,
        codebooks: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        if (
            type(spec) is not ProductQuantizationSpec
            or type(mean) is not torch.Tensor
            or mean.dtype != torch.float32
            or mean.ndim != 1
            or len(mean) < 2
            or not mean.is_contiguous()
            or not bool(torch.isfinite(mean).all())
            or type(components) is not torch.Tensor
            or components.dtype != torch.float32
            or components.ndim != 2
            or components.shape != (spec.dimensions, len(mean))
            or components.device != mean.device
            or not components.is_contiguous()
            or not bool(torch.isfinite(components).all())
        ):
            raise ValueError("rate-matched projection authority differs")
        gram = components.double() @ components.double().T
        identity = torch.eye(spec.dimensions, dtype=torch.float64, device=components.device)
        if float(torch.linalg.matrix_norm(gram - identity, ord="fro")) > 2e-5:
            raise ValueError("rate-matched projection authority differs")
        quantizer = OptimizedProductQuantizer.from_components(spec, rotation, codebooks)
        if quantizer._rotation.device != mean.device:
            raise ValueError("rate-matched projection authority differs")
        self.spec = spec
        self.register_buffer("_mean", mean.detach().clone().contiguous())
        self.register_buffer("_components", components.detach().clone().contiguous())
        self.quantizer = quantizer
        self.quantizer.requires_grad_(False)
        self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
            self._validate_checkpoint
        )

    def _apply(
        self, function: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> RateMatchedProductQuantizer:
        probe = function(self._mean.detach()[:0])
        if probe.dtype != torch.float32:
            raise ValueError("rate-matched module dtype differs")
        return cast(
            RateMatchedProductQuantizer,
            super()._apply(function, recurse),  # type: ignore[no-untyped-call]
        )

    def _validate_checkpoint(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_messages: list[str],
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_messages
        mean = state_dict.get(f"{prefix}_mean")
        components = state_dict.get(f"{prefix}_components")
        rotation = state_dict.get(f"{prefix}quantizer._rotation")
        codebooks = [
            state_dict.get(f"{prefix}quantizer.quantizer.codebooks.{index}")
            for index in range(len(self.spec.block_dimensions))
        ]
        expected_keys = {
            f"{prefix}_mean",
            f"{prefix}_components",
            f"{prefix}quantizer._rotation",
            *{
                f"{prefix}quantizer.quantizer.codebooks.{index}"
                for index in range(len(self.spec.block_dimensions))
            },
        }
        observed_keys = {key for key in state_dict if key.startswith(prefix)}
        if not observed_keys:
            return
        if (
            observed_keys != expected_keys
            or type(mean) is not torch.Tensor
            or mean.dtype != torch.float32
            or mean.shape != self._mean.shape
            or not mean.is_contiguous()
            or not bool(torch.isfinite(mean).all())
            or type(components) is not torch.Tensor
            or components.dtype != torch.float32
            or components.shape != self._components.shape
            or components.device != mean.device
            or not components.is_contiguous()
            or not bool(torch.isfinite(components).all())
            or type(rotation) is not torch.Tensor
            or rotation.dtype != torch.float32
            or rotation.shape != self.quantizer._rotation.shape
            or rotation.device != mean.device
            or not rotation.is_contiguous()
            or not bool(torch.isfinite(rotation).all())
            or any(
                type(codebook) is not torch.Tensor
                or codebook.dtype != torch.float32
                or codebook.shape != (self.spec.codebook_size, width)
                or codebook.device != mean.device
                or not codebook.is_contiguous()
                or not bool(torch.isfinite(codebook).all())
                for codebook, width in zip(codebooks, self.spec.block_dimensions, strict=True)
            )
        ):
            raise RuntimeError("rate-matched checkpoint authority differs")
        component_gram = components.double() @ components.double().T
        component_identity = torch.eye(
            self.reduced_dimensions, dtype=torch.float64, device=components.device
        )
        rotation_gram = rotation.double().T @ rotation.double()
        rotation_identity = torch.eye(
            self.reduced_dimensions, dtype=torch.float64, device=rotation.device
        )
        if (
            float(torch.linalg.matrix_norm(component_gram - component_identity, ord="fro")) > 2e-5
            or float(torch.linalg.matrix_norm(rotation_gram - rotation_identity, ord="fro")) > 2e-5
        ):
            raise RuntimeError("rate-matched checkpoint authority differs")

    @classmethod
    def from_components(
        cls,
        *,
        mean: torch.Tensor,
        components: torch.Tensor,
        spec: ProductQuantizationSpec,
        rotation: torch.Tensor,
        codebooks: Sequence[torch.Tensor],
    ) -> RateMatchedProductQuantizer:
        """Construct a codec from an exact projection, rotation, and codebooks."""

        return cls(
            mean=mean,
            components=components,
            spec=spec,
            rotation=rotation,
            codebooks=codebooks,
        )

    @property
    def input_dimensions(self) -> int:
        """Return the original embedding width."""

        return len(self._mean)

    @property
    def reduced_dimensions(self) -> int:
        """Return the normalized representation width consumed by OPQ."""

        return self.spec.dimensions

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact database code width."""

        return self.spec.bytes_per_vector

    @property
    def shared_parameter_bytes(self) -> int:
        """Return the exact tensor bytes shared by every encoded vector."""

        return sum(value.numel() * value.element_size() for value in self.state_dict().values())

    def detached_mean(self) -> torch.Tensor:
        """Return a canonical CPU copy of the fitted centering vector."""

        return self._mean.detach().cpu().float().contiguous().clone()

    def detached_components(self) -> torch.Tensor:
        """Return a canonical CPU copy of the fitted projection rows."""

        return self._components.detach().cpu().float().contiguous().clone()

    def export_artifact(self) -> dict[str, object]:
        """Return one versioned, self-describing, CPU-portable fitted codec artifact."""

        return {
            "schema": "sfora-rate-matched-product-quantizer-v1",
            "input_dimensions": self.input_dimensions,
            "block_dimensions": self.spec.block_dimensions,
            "codebook_size": self.spec.codebook_size,
            "state_dict": {
                name: value.detach().cpu().clone() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_artifact(cls, artifact: object) -> RateMatchedProductQuantizer:
        """Restore and fully validate one versioned fitted codec artifact."""

        if type(artifact) is not dict or set(artifact) != {
            "schema",
            "input_dimensions",
            "block_dimensions",
            "codebook_size",
            "state_dict",
        }:
            raise ValueError("rate-matched artifact authority differs")
        schema = artifact.get("schema")
        input_dimensions = artifact.get("input_dimensions")
        block_dimensions = artifact.get("block_dimensions")
        codebook_size = artifact.get("codebook_size")
        state = artifact.get("state_dict")
        if (
            schema != "sfora-rate-matched-product-quantizer-v1"
            or type(input_dimensions) is not int
            or input_dimensions < 2
            or type(block_dimensions) is not tuple
            or type(codebook_size) is not int
            or type(state) is not dict
        ):
            raise ValueError("rate-matched artifact authority differs")
        try:
            spec = ProductQuantizationSpec(
                block_dimensions=block_dimensions, codebook_size=codebook_size
            )
        except ValueError as error:
            raise ValueError("rate-matched artifact authority differs") from error
        expected_keys = {
            "_mean",
            "_components",
            "quantizer._rotation",
            *{
                f"quantizer.quantizer.codebooks.{index}"
                for index in range(len(spec.block_dimensions))
            },
        }
        if set(state) != expected_keys:
            raise ValueError("rate-matched artifact authority differs")
        mean = state["_mean"]
        components = state["_components"]
        if type(mean) is not torch.Tensor or mean.ndim != 1 or len(mean) != input_dimensions:
            raise ValueError("rate-matched artifact authority differs")
        try:
            return cls.from_components(
                mean=mean,
                components=cast(torch.Tensor, components),
                spec=spec,
                rotation=cast(torch.Tensor, state["quantizer._rotation"]),
                codebooks=tuple(
                    cast(torch.Tensor, state[f"quantizer.quantizer.codebooks.{index}"])
                    for index in range(len(spec.block_dimensions))
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("rate-matched artifact authority differs") from error

    @torch.no_grad()
    def prepare_queries(self, values: torch.Tensor) -> torch.Tensor:
        """Center, project, and normalize original-coordinate float rows."""

        if (
            type(values) is not torch.Tensor
            or values.dtype != torch.float32
            or values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != self.input_dimensions
            or values.device != self._mean.device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("rate-matched input authority differs")
        projected = (values.double() - self._mean.double()) @ self._components.double().T
        norms = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
            raise ValueError("rate-matched input authority differs")
        prepared = (projected / norms).float().contiguous()
        output_norms = torch.linalg.vector_norm(prepared.double(), dim=1)
        if (
            not bool(torch.isfinite(prepared).all())
            or bool((output_norms < 1.0 - 2e-6).any())
            or bool((output_norms > 1.0 + 2e-6).any())
        ):
            raise ValueError("rate-matched projection authority differs")
        return cast(torch.Tensor, prepared)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Encode original-coordinate rows into exact fixed-width hard codes.

        Callers should pass bounded batches when encoding large galleries because
        product assignment materializes distances to every codeword.
        """

        with torch.no_grad():
            return self.quantizer.hard_encode(self.prepare_queries(values))

    def score_prepared_codes(
        self, prepared_queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor:
        """Score already-prepared queries against hard gallery codes.

        The result has one value per query/code pair; retrieval indexes should
        supply bounded candidate batches rather than an unbounded full corpus.
        """

        with torch.no_grad():
            return self.quantizer.asymmetric_squared_distances(prepared_queries, gallery_codes)

    def score_codes(self, queries: torch.Tensor, gallery_codes: torch.Tensor) -> torch.Tensor:
        """Prepare queries and score a caller-bounded batch of hard gallery codes."""

        with torch.no_grad():
            return self.score_prepared_codes(self.prepare_queries(queries), gallery_codes)

    def decode_reduced(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode hard codes in normalized reduced coordinates."""

        with torch.no_grad():
            return self.quantizer.hard_decode(codes)


def fit_rate_matched_product_quantizer(
    values: torch.Tensor,
    spec: ProductQuantizationSpec,
    *,
    seed: int,
    maximum_iterations: int,
    rotation_iterations: int,
) -> RateMatchedProductQuantizer:
    """Fit normalized PCA followed by OPQ using only caller-supplied fitting rows.

    Select the reduced dimension without consulting untouched evaluation data. For
    class-disjoint studies, :func:`sfora.deterministic_class_partition` can define
    the fitting population before this function is called.
    """

    if (
        type(spec) is not ProductQuantizationSpec
        or type(values) is not torch.Tensor
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < max(2, spec.codebook_size)
        or values.shape[1] < spec.dimensions
        or spec.dimensions > values.shape[0] - 1
        or not values.is_contiguous()
        or not bool(torch.isfinite(values).all())
        or type(seed) is not int
        or seed < 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
        or type(rotation_iterations) is not int
        or rotation_iterations < 1
    ):
        raise ValueError("rate-matched fit authority differs")
    try:
        projection = fit_centered_pca(values, dimensions=spec.dimensions)
        prepared = projection.apply(values)
        quantizer = fit_optimized_product_quantizer(
            prepared,
            spec,
            seed=seed,
            maximum_iterations=maximum_iterations,
            rotation_iterations=rotation_iterations,
        )
    except ValueError as error:
        raise ValueError("rate-matched fit authority differs") from error
    return RateMatchedProductQuantizer.from_components(
        mean=projection.mean,
        components=projection.components,
        spec=spec,
        rotation=quantizer.detached_rotation(),
        codebooks=quantizer.detached_codebooks(),
    )
