import torch
from probe_inshop_siglip2_base_pilot import packed_hits


def test_packed_hits_use_lower_gallery_ordinal_on_a_tie():
    query = torch.zeros(2, 128)
    query[:, 0] = 1
    gallery = query.clone()
    assert packed_hits(query, gallery).tolist() == [1, 0]
