"""Self exclusion for native full-gallery SOP top-10 replay."""

import numpy as np
import pytest
from verify_sop_substrate_native_full_gallery import first_nonself


def test_first_nonself_handles_self_at_different_ranks_or_absent() -> None:
    ordinals = np.array([[5, 2, 3], [8, 4, 1], [7, 8, 9]], dtype=np.int64)
    queries = np.array([5, 4, 6], dtype=np.int64)
    assert np.array_equal(first_nonself(ordinals, queries), [2, 8, 7])


def test_first_nonself_rejects_a_missing_nonself_result() -> None:
    with pytest.raises(ValueError, match="nonself"):
        first_nonself(np.array([[5, 5]], dtype=np.int64), np.array([5], dtype=np.int64))
