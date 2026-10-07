"""Procedure-owned v2 signed basis; caller authenticates us and the source primitive.

TRAIN uses original outer normalization, then unchanged source head arithmetic.
Direct FIT callers pass direct rows. target_mean is never an inference input.
"""


def feature_width(arm):
    if type(arm) is not str or arm not in ('linear', 'concat'):
        raise ValueError('fixed signed readout arm required')
    return 32 if arm == 'linear' else 160


def basis(Z, H0, arm):
    import torch

    feature_width(arm)
    if tuple(Z.shape) != (H0.shape[0], 32) or tuple(H0.shape) != (Z.shape[0], 128):
        raise ValueError('signed basis widths/order differ')
    return Z if arm == 'linear' else torch.cat((Z, H0), dim=1)


def check_weight(A, device, arm, primitive):
    import torch

    primitive._check_tensor(A, (128, feature_width(arm)), device)
    primitive._require(isinstance(A, torch.nn.Parameter) and A.requires_grad and
                       A.is_leaf and A.grad_fn is None, 'trainable leaf Parameter A required')


def check_means(means, device, primitive):
    primitive._require(isinstance(means, dict) and means.keys() == {'linear', 'concat'},
                       'both fixed signed means required')
    selected = str(means['linear'].device)
    primitive._require(selected in ('cpu', str(device)), 'means must be CPU or selected device')
    for arm, value in means.items():
        primitive._check_tensor(value, (feature_width(arm),), selected, frozen=True)


def fit_means(canonical_train, base, primitive):
    import torch
    from torch.nn import functional as F

    primitive._check_features(canonical_train, 'cpu', train=True)
    source = primitive._check_base(base, 'cpu')
    with torch.autocast('cpu', enabled=False), torch.no_grad():
        primitive._finite(torch, [canonical_train, *source])
        x = F.normalize(canonical_train.detach().float(), dim=1)
        H0 = base(x).detach()
        Z = base.down(F.normalize(x, dim=1) - base.center)
        means = {arm: basis(Z, H0, arm).mean(dim=0).detach() for arm in ('linear', 'concat')}
        check_means(means, 'cpu', primitive)
        primitive._finite(torch, [H0, Z, *means.values()])
    return means


def raw_features(features, base, A, means, arm, primitive):
    import torch
    from torch.nn import functional as F

    device = features.device
    primitive._require(device.type in ('cpu', 'cuda'), 'CPU or CUDA device required')
    primitive._check_features(features, device)
    source = primitive._check_base(base, device)
    check_weight(A, device, arm, primitive)
    check_means(means, device, primitive)
    with torch.autocast(device.type, enabled=False):
        with torch.no_grad():
            primitive._finite(torch, [features, A, *source, *means.values()])
            x = features.detach().float()
            h0 = base(x).detach()
            z = base.down(F.normalize(x, dim=1) - base.center)
            phi = basis(z, h0, arm) - means[arm].detach().to(device=device)
            primitive._check_tensor(h0, (features.shape[0], 128), device, frozen=True)
            primitive._finite(torch, [h0, z, phi])
        raw = h0 + F.linear(phi.detach(), A)
        primitive._finite(torch, [raw])
    return raw
