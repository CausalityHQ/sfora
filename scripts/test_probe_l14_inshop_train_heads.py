import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_l14_inshop_train_heads import score_query_gallery, split_query_gallery_rows


def test_train_roles_are_disjoint_and_each_identity_has_both() -> None:
    labels = (4, 4, 4, 8, 8, 11, 11, 11, 11)
    query, gallery = split_query_gallery_rows(labels)
    assert set(query).isdisjoint(gallery)
    assert set(query) | set(gallery) == set(range(len(labels)))
    assert {labels[index] for index in query} == {4, 8, 11}
    assert {labels[index] for index in gallery} == {4, 8, 11}


def test_query_gallery_scorer_excludes_no_row_and_resolves_ties_by_ordinal() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels_q = torch.tensor([1, 2], dtype=torch.int64)
    labels_g = torch.tensor([1, 9, 2], dtype=torch.int64)
    result = score_query_gallery(query, gallery, labels_q, labels_g)
    assert result["recall_at_1"] == 1.0
    assert result["map_at_r"] == 1.0
    assert result["top1_gallery_ordinals"] == [0, 2]


def test_query_gallery_packed_scorer_uses_wire_inverse_norms() -> None:
    query = torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    gallery = torch.tensor([[4.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    packed_query = pack_int8_unit_embeddings(F.normalize(query, dim=1))
    packed_gallery = pack_int8_unit_embeddings(F.normalize(gallery, dim=1))
    result = score_query_gallery(
        packed_query.codes.float(),
        packed_gallery.codes.float(),
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        query_inverse_norms=packed_query.inverse_norms,
        gallery_inverse_norms=packed_gallery.inverse_norms,
    )
    assert result["recall_at_1"] == 1.0
    assert result["map_at_r"] == 1.0
