from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "_scratch_siglip2_so400m_cars_screen.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "scratch_siglip2_so400m_cars_screen", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_candidate_requires_effect_ci_and_recall_nonregression() -> None:
    subject = _load_subject()
    baseline = {"map_at_r": 0.86, "recall_at_1": 0.97}

    passing = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.971},
        {"lower": 0.001, "median": 0.006, "upper": 0.011},
    )
    too_small = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.8649, "recall_at_1": 0.971},
        {"lower": 0.001, "median": 0.0049, "upper": 0.009},
    )
    uncertain = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.971},
        {"lower": 0.0, "median": 0.006, "upper": 0.012},
    )
    recall_loss = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.9699},
        {"lower": 0.001, "median": 0.006, "upper": 0.011},
    )

    assert passing["passed"] is True
    assert passing["map_at_r_delta"] == 0.006000000000000005
    assert passing["recall_at_1_delta"] == 0.0010000000000000009
    assert too_small["passed"] is False
    assert uncertain["passed"] is False
    assert recall_loss["passed"] is False


def test_processor_authority_accepts_only_frozen_siglip2_shape() -> None:
    subject = _load_subject()
    valid = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "resample": 2,
        "rescale_factor": 1 / 255,
        "size": {"height": 384, "width": 384},
    }

    subject.validate_processor_authority(valid)
    invalid = dict(valid)
    invalid["size"] = {"height": 224, "width": 224}
    try:
        subject.validate_processor_authority(invalid)
    except ValueError as error:
        assert str(error) == "SigLIP2 processor authority differs"
    else:
        raise AssertionError("processor drift was accepted")


def test_processor_collate_batches_images_in_one_authenticated_call() -> None:
    subject = _load_subject()

    class _Processor:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *, images, return_tensors):
            self.calls.append((images, return_tensors))
            return {"pixel_values": ("batch", tuple(images))}

    processor = _Processor()
    collate = subject._ProcessorCollate(processor)

    assert collate(["a", "b", "c"]) == ("batch", ("a", "b", "c"))
    assert processor.calls == [(["a", "b", "c"], "pt")]


def test_fused_inference_backend_is_explicit_and_repeatability_bound() -> None:
    subject = _load_subject()
    original = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }
    try:
        receipt = subject.enable_repeatable_fused_inference()
        assert receipt == {
            "backend": "upstream-default-fused-sdp-v1",
            "deterministic_algorithms": False,
            "flash_sdp": True,
            "memory_efficient_sdp": True,
            "cudnn_sdp": True,
            "repeatability_probe_equal": True,
            "repeatability_probe_sha256": (
                "71cc58509212167962fbedd843abb9efe7d654181981759b4ff48e764e383253"
            ),
            "strict_fast_min_cosine": 0.9999939203262329,
        }
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cuda.flash_sdp_enabled() is True
    finally:
        torch.use_deterministic_algorithms(original["deterministic"])
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cuda.enable_flash_sdp(original["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(original["memory_efficient"])
        torch.backends.cuda.enable_cudnn_sdp(original["cudnn_sdp"])
