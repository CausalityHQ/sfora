"""Prospective connected FP32 readout; adopted by no trainer or evaluator.

Source correspondence (pinned by test_connected_residual_readout.py): the concat arm of
prototype_residual_readout.raw_features plus the C term of fullfeature_raw_features.
Both detach the features, so d raw / d features is zero. This helper keeps the exact
op order and arithmetic but leaves x/h0/z/phi on the autograd graph:

    raw = (h0 + linear(phi, A)) + linear(x - mu_train, C)

head, means and mu_train stay frozen. Caller authenticates head/primitive/readout as for
the source. Training grad mode only: under no_grad the output would silently disconnect.
"""


def raw_features(features, head, A, means, C, mu_train, primitive, readout):
    import torch
    from torch.nn import functional as F

    device = features.device
    primitive._require(device.type in ('cpu', 'cuda'), 'CPU or CUDA device required')
    primitive._check_features(features, device)
    source = primitive._check_base(head, device)
    readout.check_weight(A, device, 'concat', primitive)
    readout.check_means(means, device, primitive)
    primitive._check_tensor(C, (128, 1152), device)
    primitive._check_tensor(mu_train, (1152,), device, frozen=True)
    primitive._require(isinstance(C, torch.nn.Parameter) and C.requires_grad and
                       C.is_leaf and C.grad_fn is None, 'trainable leaf Parameter C required')
    with torch.autocast(device.type, enabled=False):
        with torch.no_grad():
            primitive._finite(torch, [features, A, C, mu_train, *source, *means.values()])
        x = features.float()
        h0 = head(x)
        z = head.down(F.normalize(x, dim=1) - head.center)
        phi = readout.basis(z, h0, 'concat') - means['concat'].detach().to(device=device)
        primitive._check_tensor(h0, (features.shape[0], 128), device)
        with torch.no_grad():
            primitive._finite(torch, [h0, z, phi])
        raw = h0 + F.linear(phi, A)
        raw = raw + F.linear(x - mu_train, C)
        with torch.no_grad():
            primitive._finite(torch, [raw])
    primitive._require(raw.requires_grad, 'connected readout requires grad mode')
    return raw
