"""Dataset-agnostic deterministic arithmetic authority for similarity experiments."""

from __future__ import annotations

import os
import random
from typing import NamedTuple, cast

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]


class DeterministicSimilarityRuntimeReceipt(NamedTuple):
    """Exact process arithmetic state after deterministic configuration."""

    seed: int
    cpu_threads: int
    blas_threads: int
    cublas_workspace_config: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_tf32: bool
    cudnn_tf32: bool
    float32_matmul_precision: str
    math_sdp_enabled: bool
    flash_sdp_enabled: bool
    memory_efficient_sdp_enabled: bool
    cudnn_sdp_enabled: bool
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    cuda_device_name: str | None
    cuda_device_capability: str | None


def _current_runtime_receipt(seed: int) -> DeterministicSimilarityRuntimeReceipt:
    blas_threads = {
        int(pool["num_threads"]) for pool in threadpool_info() if pool["user_api"] == "blas"
    }
    if len(blas_threads) != 1:
        raise ValueError("deterministic runtime authority differs")
    cuda_available = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda_available else None
    capability = (
        torch.cuda.get_device_capability(device_index) if device_index is not None else None
    )
    return DeterministicSimilarityRuntimeReceipt(
        seed=seed,
        cpu_threads=torch.get_num_threads(),
        blas_threads=blas_threads.pop(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        math_sdp_enabled=cast(bool, torch.backends.cuda.math_sdp_enabled()),  # type: ignore[no-untyped-call]
        flash_sdp_enabled=cast(bool, torch.backends.cuda.flash_sdp_enabled()),  # type: ignore[no-untyped-call]
        memory_efficient_sdp_enabled=cast(
            bool,
            torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
        ),
        cudnn_sdp_enabled=cast(
            bool,
            torch.backends.cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
        ),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        cuda_device_name=(
            torch.cuda.get_device_name(device_index) if device_index is not None else None
        ),
        cuda_device_capability=(
            f"{capability[0]}.{capability[1]}" if capability is not None else None
        ),
    )


def validate_deterministic_similarity_runtime(
    receipt: DeterministicSimilarityRuntimeReceipt,
) -> None:
    """Reject a receipt that does not describe the active arithmetic process."""

    if (
        type(receipt) is not DeterministicSimilarityRuntimeReceipt
        or type(receipt.seed) is not int
        or not 0 <= receipt.seed < 2**63
        or receipt != _current_runtime_receipt(receipt.seed)
    ):
        raise ValueError("deterministic runtime authority differs")


def configure_deterministic_similarity_runtime(
    seed: int,
    *,
    cpu_threads: int,
) -> DeterministicSimilarityRuntimeReceipt:
    """Pin RNG, CPU parallelism, and CUDA math backends before computation."""

    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**63
        or type(cpu_threads) is not int
        or cpu_threads < 1
        or workspace != ":4096:8"
    ):
        raise ValueError("deterministic runtime authority differs")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    threadpool_limits(limits=cpu_threads, user_api="blas")
    blas_threads = {
        int(pool["num_threads"]) for pool in threadpool_info() if pool["user_api"] == "blas"
    }
    if blas_threads != {cpu_threads}:
        raise ValueError("deterministic runtime authority differs")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    receipt = _current_runtime_receipt(seed)
    if receipt.blas_threads != cpu_threads:
        raise ValueError("deterministic runtime authority differs")
    return receipt
