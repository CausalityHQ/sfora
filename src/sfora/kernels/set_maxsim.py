"""Symmetric weighted MaxSim with a bounded fused Triton forward kernel."""

from __future__ import annotations

import torch


def _validate_inputs(
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    query_weights: torch.Tensor,
    gallery_weights: torch.Tensor,
) -> None:
    if query.ndim != 3 or gallery.ndim != 3:
        raise ValueError("query and gallery tokens must be rank-three tensors")
    if query.shape[-1] != gallery.shape[-1]:
        raise ValueError("query and gallery token dimensions differ")
    if query.shape[1] < 1 or gallery.shape[1] < 1 or query.shape[2] < 1:
        raise ValueError("token sets and token dimensions must be nonempty")
    if query.device != gallery.device:
        raise ValueError("query and gallery tokens must share a device")
    if query.dtype != gallery.dtype or not query.is_floating_point():
        raise ValueError("query and gallery tokens must share a floating dtype")
    if query_weights.shape != query.shape[:2]:
        raise ValueError("query weights do not match query token sets")
    if gallery_weights.shape != gallery.shape[:2]:
        raise ValueError("gallery weights do not match gallery token sets")
    if query_weights.device != query.device or gallery_weights.device != gallery.device:
        raise ValueError("token weights must share the token device")
    if not query_weights.is_floating_point() or not gallery_weights.is_floating_point():
        raise ValueError("token weights must use floating dtypes")
    if not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("tokens must be finite")
    if not torch.isfinite(query_weights).all() or not torch.isfinite(gallery_weights).all():
        raise ValueError("token weights must be finite")
    if (query_weights < 0).any() or not torch.allclose(
        query_weights.sum(dim=1),
        torch.ones(query.shape[0], dtype=query_weights.dtype, device=query.device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("query weights must be probability vectors")
    if (gallery_weights < 0).any() or not torch.allclose(
        gallery_weights.sum(dim=1),
        torch.ones(gallery.shape[0], dtype=gallery_weights.dtype, device=gallery.device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("gallery weights must be probability vectors")


def symmetric_set_maxsim_reference(
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    query_weights: torch.Tensor,
    gallery_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the exact eager symmetric weighted MaxSim score matrix.

    This deliberately materializes the four-dimensional interaction tensor and is
    therefore only an oracle for tests and small batches. Production scoring uses
    :func:`fused_set_maxsim`, which streams one image-pair tile per program.
    """

    _validate_inputs(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )
    interactions = torch.einsum("bkd,nmd->bknm", query.float(), gallery.float())
    query_to_gallery = (interactions.max(dim=3).values * query_weights.float()[:, :, None]).sum(
        dim=1
    )
    gallery_to_query = (interactions.max(dim=1).values * gallery_weights.float()[None, :, :]).sum(
        dim=2
    )
    return 0.5 * (query_to_gallery + gallery_to_query)


try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - the research extra provides Triton on Linux.
    triton = None
    tl = None


if triton is not None:

    @triton.jit  # type: ignore[untyped-decorator]
    def _symmetric_set_maxsim_kernel(  # type: ignore[no-untyped-def]  # pragma: no cover
        query_ptr,
        gallery_ptr,
        query_weight_ptr,
        gallery_weight_ptr,
        output_ptr,
        query_count: tl.constexpr,
        gallery_count: tl.constexpr,
        query_tokens: tl.constexpr,
        gallery_tokens: tl.constexpr,
        dimensions: tl.constexpr,
        query_stride_0: tl.constexpr,
        query_stride_1: tl.constexpr,
        gallery_stride_0: tl.constexpr,
        gallery_stride_1: tl.constexpr,
        block_query_tokens: tl.constexpr,
        block_gallery_tokens: tl.constexpr,
        block_dimensions: tl.constexpr,
    ) -> None:
        query_index = tl.program_id(0)
        gallery_index = tl.program_id(1)
        query_offsets = tl.arange(0, block_query_tokens)
        gallery_offsets = tl.arange(0, block_gallery_tokens)
        dimension_offsets = tl.arange(0, block_dimensions)

        query_mask = query_offsets < query_tokens
        gallery_mask = gallery_offsets < gallery_tokens
        dimension_mask = dimension_offsets < dimensions
        query_values = tl.load(
            query_ptr
            + query_index * query_stride_0
            + query_offsets[:, None] * query_stride_1
            + dimension_offsets[None, :],
            mask=query_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        gallery_values = tl.load(
            gallery_ptr
            + gallery_index * gallery_stride_0
            + gallery_offsets[:, None] * gallery_stride_1
            + dimension_offsets[None, :],
            mask=gallery_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        interactions = tl.dot(query_values, tl.trans(gallery_values), out_dtype=tl.float32)
        interactions = tl.where(
            query_mask[:, None] & gallery_mask[None, :],
            interactions,
            -float("inf"),
        )
        query_maxima = tl.max(interactions, axis=1)
        gallery_maxima = tl.max(interactions, axis=0)
        query_maxima = tl.where(query_mask, query_maxima, 0.0)
        gallery_maxima = tl.where(gallery_mask, gallery_maxima, 0.0)
        query_weights = tl.load(
            query_weight_ptr + query_index * query_tokens + query_offsets,
            mask=query_mask,
            other=0.0,
        )
        gallery_weights = tl.load(
            gallery_weight_ptr + gallery_index * gallery_tokens + gallery_offsets,
            mask=gallery_mask,
            other=0.0,
        )
        score = 0.5 * (
            tl.sum(query_maxima * query_weights, axis=0)
            + tl.sum(gallery_maxima * gallery_weights, axis=0)
        )
        tl.store(output_ptr + query_index * gallery_count + gallery_index, score)


def fused_set_maxsim(
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    query_weights: torch.Tensor,
    gallery_weights: torch.Tensor,
) -> torch.Tensor:
    """Score two token-set batches without materializing all token interactions.

    CPU tensors and autograd-enabled calls use the exact eager oracle. CUDA inference
    dispatches to the fused Triton forward kernel. The training backward is a separate
    TSPA milestone and is intentionally not implied by this inference-only surface.
    """

    _validate_inputs(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )
    requires_grad = any(
        tensor.requires_grad for tensor in (query, gallery, query_weights, gallery_weights)
    )
    if query.device.type != "cuda" or requires_grad:
        return symmetric_set_maxsim_reference(
            query,
            gallery,
            query_weights=query_weights,
            gallery_weights=gallery_weights,
        )
    if triton is None:
        raise RuntimeError("Triton is required for fused CUDA MaxSim scoring")

    query = query.contiguous()
    gallery = gallery.contiguous()
    query_weights = query_weights.contiguous()
    gallery_weights = gallery_weights.contiguous()
    output = torch.empty(
        (query.shape[0], gallery.shape[0]),
        device=query.device,
        dtype=torch.float32,
    )
    block_query_tokens = max(16, triton.next_power_of_2(query.shape[1]))
    block_gallery_tokens = max(16, triton.next_power_of_2(gallery.shape[1]))
    block_dimensions = max(16, triton.next_power_of_2(query.shape[2]))
    _symmetric_set_maxsim_kernel[(query.shape[0], gallery.shape[0])](
        query,
        gallery,
        query_weights,
        gallery_weights,
        output,
        query.shape[0],
        gallery.shape[0],
        query.shape[1],
        gallery.shape[1],
        query.shape[2],
        query.stride(0),
        query.stride(1),
        gallery.stride(0),
        gallery.stride(1),
        block_query_tokens,
        block_gallery_tokens,
        block_dimensions,
    )
    return output
