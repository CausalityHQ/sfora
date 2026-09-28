"""Product descriptions provide semantic text without identity shortcuts."""

import pytest
from probe_inshop_description_grounding import fit_captions


def test_captions_use_only_fit_items_and_reject_identity_tokens():
    rows = [
        {"item": "id_a", "color": "Cream", "description": ["A sheer woven top."]},
        {"item": "id_b", "color": "Black", "description": ["A long-sleeve shirt."]},
    ]
    assert fit_captions(rows, {"id_a"}) == {"id_a": "Cream. A sheer woven top."}
    with pytest.raises(ValueError):
        fit_captions([rows[0], rows[0]], {"id_a"})
    with pytest.raises(ValueError):
        fit_captions([{**rows[0], "description": ["A top named id_00000001."]}], {"id_a"})
