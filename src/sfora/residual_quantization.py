"""Full-dimensional residual codes with additive asymmetric dot scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from torch import nn


@dataclass(frozen=True, slots=True)
class ResidualQuantizationSpec:
    """Wire and codebook geometry for a fixed-width residual quantizer."""

    dimensions: int
    stages: int
    codebook_size: int = 256

    def __post_init__(self) -> None:
        if (
            type(self.dimensions) is not int
            or self.dimensions < 2
            or type(self.stages) is not int
            or self.stages < 1
            or type(self.codebook_size) is not int
            or not 2 <= self.codebook_size <= 256
        ):
            raise ValueError("residual quantization spec differs")

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact one-byte-per-stage code width."""

        return self.stages

    @property
    def codebook_shape(self) -> tuple[int, int, int]:
        """Return stages, codewords, and full vector dimensions."""

        return (self.stages, self.codebook_size, self.dimensions)


class ResidualQuantizer(nn.Module):
    """Greedy full-dimensional residual quantizer with additive dot tables."""

    def __init__(self, spec: ResidualQuantizationSpec, codebooks: torch.Tensor) -> None:
        super().__init__()
        if (
            type(spec) is not ResidualQuantizationSpec
            or type(codebooks) is not torch.Tensor
            or codebooks.dtype != torch.float32
            or codebooks.shape != spec.codebook_shape
            or not bool(torch.isfinite(codebooks).all())
        ):
            raise ValueError("residual quantization codebook authority differs")
        self.spec = spec
        self.codebooks = nn.Parameter(codebooks.detach().clone().contiguous())

    @classmethod
    def from_codebooks(
        cls, spec: ResidualQuantizationSpec, codebooks: torch.Tensor
    ) -> ResidualQuantizer:
        """Construct a trainable residual quantizer from exact codebooks."""

        return cls(spec, codebooks)

    def _validate_values(self, values: torch.Tensor) -> None:
        if (
            type(values) is not torch.Tensor
            or values.dtype != torch.float32
            or values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != self.spec.dimensions
            or values.device != self.codebooks.device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("residual quantization input authority differs")

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if (
            type(codes) is not torch.Tensor
            or codes.dtype != torch.uint8
            or codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] != self.spec.stages
            or codes.device != self.codebooks.device
            or bool((codes.to(torch.int16) >= self.spec.codebook_size).any())
        ):
            raise ValueError("residual quantization code authority differs")

    def hard_encode(self, values: torch.Tensor) -> torch.Tensor:
        """Greedily quantize each successive full-dimensional residual."""

        self._validate_values(values)
        residual = values.detach().clone()
        codes = []
        with torch.no_grad():
            for codebook in self.codebooks:
                distances = torch.cdist(
                    residual,
                    codebook,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                indexes = torch.argmin(distances, dim=1)
                codes.append(indexes.to(torch.uint8))
                residual -= codebook[indexes]
        return torch.stack(codes, dim=1).contiguous()

    def hard_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum selected full-dimensional codewords without renormalization."""

        self._validate_codes(codes)
        restored = torch.zeros(
            (codes.shape[0], self.spec.dimensions),
            dtype=torch.float32,
            device=codes.device,
        )
        for stage, codebook in enumerate(self.codebooks):
            restored = restored + codebook[codes[:, stage].long()]
        return restored.contiguous()

    def asymmetric_dot_scores(
        self, queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor:
        """Score float queries by summing one dot-product table lookup per stage."""

        self._validate_values(queries)
        self._validate_codes(gallery_codes)
        scores = torch.zeros(
            (queries.shape[0], gallery_codes.shape[0]),
            dtype=torch.float32,
            device=queries.device,
        )
        for stage, codebook in enumerate(self.codebooks):
            table = queries @ codebook.T
            scores = scores + table[:, gallery_codes[:, stage].long()]
        return scores.contiguous()


def fit_residual_quantizer(
    values: torch.Tensor,
    spec: ResidualQuantizationSpec,
    *,
    seed: int,
    maximum_iterations: int,
) -> ResidualQuantizer:
    """Fit successive deterministic Lloyd codebooks on CPU residuals."""

    if (
        type(values) is not torch.Tensor
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < spec.codebook_size
        or values.shape[1] != spec.dimensions
        or not bool(torch.isfinite(values).all())
        or type(seed) is not int
        or seed < 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
    ):
        raise ValueError("residual quantization fit authority differs")
    residual = values.detach().numpy().astype(np.float32, copy=True)
    codebooks = []
    for stage in range(spec.stages):
        estimator = KMeans(
            n_clusters=spec.codebook_size,
            init="k-means++",
            n_init=1,
            max_iter=maximum_iterations,
            random_state=seed + stage,
            algorithm="lloyd",
        )
        with threadpool_limits(limits=1):
            assignments = estimator.fit_predict(residual)
        centers = estimator.cluster_centers_.astype(np.float32, copy=True)
        if centers.shape != (spec.codebook_size, spec.dimensions) or not np.isfinite(centers).all():
            raise RuntimeError("residual quantization fit produced invalid codebooks")
        residual -= centers[assignments]
        codebooks.append(torch.from_numpy(centers))
    stacked = torch.stack(codebooks).contiguous()
    if not math.isfinite(float(stacked.square().sum())):
        raise RuntimeError("residual quantization fit produced invalid codebooks")
    return ResidualQuantizer.from_codebooks(spec, stacked)
