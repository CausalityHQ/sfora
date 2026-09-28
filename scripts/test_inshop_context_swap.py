import numpy as np
import pytest
from probe_inshop_context_swap_geometry import context_mask


def test_context_mask_preserves_all_three_garments_and_has_same_sham_geometry():
    boxes = ((1, 1, 2, 2), (3, 3, 4, 4), (2, 2, 3, 3))
    mask = context_mask(boxes, (4, 4))
    protected = {
        (x, y) for x1, y1, x2, y2 in boxes for x in range(x1 - 1, x2) for y in range(y1 - 1, y2)
    }
    assert np.array_equal(mask, [[(x, y) not in protected for x in range(4)] for y in range(4)])
    assert np.array_equal(mask, context_mask(tuple(reversed(boxes)), (4, 4)))
    with pytest.raises(ValueError):
        context_mask(((0, 1, 2, 2),), (4, 4))
