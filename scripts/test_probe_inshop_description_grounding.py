"""Product descriptions provide semantic text without identity shortcuts."""

import numpy as np
import pytest
from probe_inshop_description_alignment import shuffled_caption_names
from probe_inshop_description_grounding import fit_captions


def test_captions_use_only_fit_items_and_reject_identity_tokens():
    rows = [
        {"item": "id_a", "color": "Cream", "description": ["A sheer woven top."]},
        {"item": "id_b", "color": "Black", "description": ["A long-sleeve shirt."]},
    ]
    assert fit_captions(rows, {"id_a"}) == {"id_a": "Cream. A sheer woven top."}
    assert fit_captions([rows[0], rows[0]], {"id_a"}) == fit_captions(rows, {"id_a"})
    with pytest.raises(ValueError):
        fit_captions([rows[0], {**rows[0], "color": "Black"}], {"id_a"})
    with pytest.raises(ValueError):
        fit_captions([{**rows[0], "description": ["A top named id_00000001."]}], {"id_a"})


def test_shuffled_caption_control_preserves_category_without_fixed_points():
    categories = {"a": "tops", "b": "tops", "c": "dress", "d": "dress", "e": "dress"}
    result = shuffled_caption_names(categories, np.random.default_rng(179019))
    assert set(result) == set(result.values()) == set(categories)
    assert all(
        name != other and categories[name] == categories[other] for name, other in result.items()
    )
