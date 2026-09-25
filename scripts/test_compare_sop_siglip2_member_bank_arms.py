"""Authority and product-cluster uncertainty for the matched bank decision."""

import numpy as np
import pytest
from compare_sop_siglip2_member_bank_arms import (
    COST_SHA256,
    PREFLIGHT_SHA256,
    TRAINER_SHA256,
    product_bootstrap,
    validate_pair,
)


def test_product_bootstrap_keeps_query_weighted_point_estimate() -> None:
    delta = np.asarray([1.0, 1.0, 0.0, -1.0, -1.0])
    labels = np.asarray([10, 10, 20, 30, 30])
    result = product_bootstrap(delta, labels)
    assert result["point"] == 0.0
    assert result["lower_95"] < 0 < result["upper_95"]


def test_validate_pair_rejects_mismatched_augmented_input_tickets() -> None:
    control = {
        "arm": "arcface",
        "source_sha256": TRAINER_SHA256,
        "updates": 1000,
        "seed": 179019,
        "fit_images": 53700,
        "holdout_queries": 5851,
        "gallery_images": 59551,
        "member_bank_preflight_sha256": None,
        "member_bank_cost_sha256": None,
        "first_input_batch_sha256": ["same"] * 10,
        "rank_to_arcface_head_gradient_ratio": [],
        "quality": {"native_top10_exact": True, "gallery_wire_bytes_per_row": 130},
    }
    treatment = {
        **control,
        "arm": "float_rank_member_bank",
        "member_bank_preflight_sha256": PREFLIGHT_SHA256,
        "member_bank_cost_sha256": COST_SHA256,
        "first_input_batch_sha256": ["different"] * 10,
    }
    with pytest.raises(ValueError, match="authority"):
        validate_pair(control, treatment)
