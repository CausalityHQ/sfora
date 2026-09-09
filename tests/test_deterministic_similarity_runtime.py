from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
    validate_deterministic_similarity_runtime,
)


def test_runtime_replays_rng_and_pins_cuda_arithmetic() -> None:
    threadpool_limits(limits=4, user_api="blas")
    receipt = configure_deterministic_similarity_runtime(17, cpu_threads=2)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    replay = configure_deterministic_similarity_runtime(17, cpu_threads=2)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert receipt == replay
    assert first == second
    assert receipt.seed == 17
    assert receipt.cpu_threads == 2
    assert receipt.blas_threads == 2
    assert receipt.torch_version == torch.__version__
    assert receipt.cuda_version == torch.version.cuda
    assert receipt.cudnn_version == torch.backends.cudnn.version()
    if torch.cuda.is_available():
        assert receipt.cuda_device_name == torch.cuda.get_device_name(torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        assert receipt.cuda_device_capability == f"{capability[0]}.{capability[1]}"
    else:
        assert receipt.cuda_device_name is None
        assert receipt.cuda_device_capability is None
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"} == {2}


def test_runtime_rejects_invalid_seed_threads_and_cublas_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for seed, threads in ((True, 2), (-1, 2), (17, True), (17, 0)):
        with pytest.raises(ValueError, match="deterministic runtime authority"):
            configure_deterministic_similarity_runtime(seed, cpu_threads=threads)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="deterministic runtime authority"):
        configure_deterministic_similarity_runtime(17, cpu_threads=2)


def test_runtime_receipt_must_match_active_process_state() -> None:
    receipt = configure_deterministic_similarity_runtime(17, cpu_threads=2)
    validate_deterministic_similarity_runtime(receipt)

    torch.use_deterministic_algorithms(False)
    try:
        with pytest.raises(ValueError, match="deterministic runtime authority"):
            validate_deterministic_similarity_runtime(receipt)
    finally:
        torch.use_deterministic_algorithms(True)
    with pytest.raises(ValueError, match="deterministic runtime authority"):
        validate_deterministic_similarity_runtime(receipt._replace(seed=True))
