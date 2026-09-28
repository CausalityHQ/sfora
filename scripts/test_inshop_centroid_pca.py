import torch
from probe_inshop_centroid_pca import centroid_pca


def test_centroid_basis_removes_cancelling_within_product_direction():
    values = torch.tensor([[-1.0, -10.0], [-1.0, 10.0], [1.0, -10.0], [1.0, 10.0]])
    pca = centroid_pca(values, torch.tensor([0, 0, 1, 1]), 1)
    assert torch.allclose(pca.mean, torch.zeros(2))
    assert torch.allclose(pca.components, torch.tensor([[1.0, 0.0]]))
