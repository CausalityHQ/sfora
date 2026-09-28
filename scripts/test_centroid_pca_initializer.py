import pytest
import torch
from probe_inshop_centroid_pca import centroid_pca
from train_sop_siglip2_compact import initialize_head_and_classifier


def test_centroid_initializer_preserves_rng_native_default_and_proxy_contract():
    generator = torch.Generator().manual_seed(93)
    source = torch.randn(512, 768, generator=generator)
    labels = tuple(i for i in range(128) for _ in range(2)) + tuple(
        i for i in range(128, 192) for _ in range(4)
    )
    torch.manual_seed(179024)
    native, proxy, digest = initialize_head_and_classifier(source, labels)
    after = torch.get_rng_state()
    torch.manual_seed(179024)
    explicit, explicit_proxy, explicit_digest = initialize_head_and_classifier(
        source, labels, pca_basis="images"
    )
    assert torch.equal(native.weight, explicit.weight)
    assert torch.equal(native.bias, explicit.bias)
    assert torch.equal(proxy, explicit_proxy) and digest == explicit_digest
    assert torch.equal(after, torch.get_rng_state())
    torch.manual_seed(179024)
    product, product_proxy, product_digest = initialize_head_and_classifier(
        source, labels, pca_basis="products"
    )
    assert torch.equal(after, torch.get_rng_state())
    assert product_digest != digest
    assert torch.allclose(product.weight @ product.weight.T, torch.eye(128), atol=1e-6)
    unit = torch.nn.functional.normalize(source, dim=1)
    pca = centroid_pca(unit, torch.tensor(labels), 128)
    assert torch.equal(product.weight, pca.components)
    assert torch.equal(product.bias, -(pca.components @ pca.mean))
    raw = torch.nn.functional.normalize(product(unit), dim=1)
    sums = torch.zeros(192, 128)
    sums.index_add_(0, torch.tensor(labels), raw)
    expected = torch.nn.functional.normalize(sums, dim=1)
    assert torch.allclose(product_proxy, expected, atol=1e-6)
    with pytest.raises(ValueError):
        initialize_head_and_classifier(source, labels, pca_basis="unknown")
