"""The secondary attribution comparison must use a same-source float arm."""

import pytest
from compare_sop_siglip2_member_bank_arms import TRAINER_SHA256
from compare_sop_siglip2_member_bank_vs_float import validate_pair


def test_validate_pair_rejects_historical_float_source() -> None:
    common = {
        "updates": 1000,
        "seed": 179019,
        "holdout_queries": 5851,
        "first_input_batch_sha256": ["same"] * 10,
        "quality": {"native_top10_exact": True, "gallery_wire_bytes_per_row": 130},
    }
    float_arm = {**common, "arm": "float_rank", "source_sha256": "historical"}
    bank_arm = {**common, "arm": "float_rank_member_bank", "source_sha256": TRAINER_SHA256}
    with pytest.raises(ValueError, match="authority"):
        validate_pair(float_arm, bank_arm)
