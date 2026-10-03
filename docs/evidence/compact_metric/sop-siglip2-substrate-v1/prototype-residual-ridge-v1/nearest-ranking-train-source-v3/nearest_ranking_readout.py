"""Differentiable fixed concat readout over the authenticated control head.

The caller authenticates the source primitive/head and coefficients. Layout
admission here cannot establish provenance. Pass the same features as the
accepted inference helper; no outer normalization, fitting or statistics are
added. Native CPU/CUDA parity and autograd qualification belong to the root.
"""


def raw_features(features, base, A, means, primitive):
    """FP32 H0raw128 + A * ([Z32,H0raw128] - mean), input graph intact."""
    import torch
    from torch.nn import functional as F

    device = features.device
    primitive._require(device.type in ('cpu', 'cuda'), 'CPU or CUDA device required')
    primitive._check_features(features, device)
    source = primitive._check_base(base, device)
    primitive._check_tensor(A, (128, 160), device, frozen=True)
    primitive._require(isinstance(means, dict) and means.keys() == {'linear', 'concat'},
                       'both fixed signed means required')
    selected = str(means['linear'].device)
    primitive._require(selected in ('cpu', str(device)), 'means must be CPU or selected device')
    for arm, width in (('linear', 32), ('concat', 160)):
        primitive._check_tensor(means[arm], (width,), selected, frozen=True)

    with torch.autocast(device.type, enabled=False):
        primitive._finite(torch, [features, A, *source, *means.values()])
        x = features.float()
        norms = torch.linalg.vector_norm(x, dim=1)
        primitive._finite(torch, [x, norms])
        primitive._require((norms > 0).all().item(), 'nonzero feature row norms required')
        h0 = base(x)
        z = base.down(F.normalize(x, dim=1) - base.center)
        phi = torch.cat((z, h0), dim=1) - means['concat'].to(device=device)
        primitive._check_tensor(h0, (features.shape[0], 128), device)
        primitive._check_tensor(z, (features.shape[0], 32), device)
        primitive._check_tensor(phi, (features.shape[0], 160), device)
        primitive._finite(torch, [h0, z, phi])
        raw = h0 + F.linear(phi, A)
        primitive._check_tensor(raw, (features.shape[0], 128), device)
        norms = torch.linalg.vector_norm(raw, dim=1)
        primitive._finite(torch, [raw, norms])
        primitive._require((norms > 0).all().item(), 'nonzero raw row norms required')
    return raw
