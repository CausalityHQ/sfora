"""Fixed-basis residual over the authenticated, frozen So400 control head.

The caller authenticates the original control factory, learned source061 state,
canonical TRAIN cache and its complete ordered rows before calling. Layout
checks here cannot establish that provenance. Native gradients, row/batch and
raw/unit/packed numerical qualification belong to the parent runner.

fit_means takes the raw canonical FP32 CPU[6355,1152] cache in existing row
order: original TRAIN normalization, internal head normalization, center,
down, then FP32 mean(z) and mean(z.square()). TRAIN raw_features callers pass
the original training_features-normalized rows; FIT callers pass direct rows.
No flag changes, CUDA work, standardization, folding or batching. Both arms
start with new_weight()==0; only A remains connected to the returned graph.
"""

from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_tensor(value, shape, device, frozen=False, dtype="torch.float32"):
    _require(tuple(value.shape) == shape and str(value.dtype) == dtype and
             str(value.device) == str(device) and str(value.layout) == "torch.strided",
             "tensor shape/dtype/device/layout differs")
    if frozen:
        _require(not value.requires_grad and value.grad_fn is None, "frozen detached tensor required")


def _check_arm(arm):
    _require(type(arm) is str and arm in ("control", "candidate"), "unknown readout arm")


def _check_base(base, device):
    _require(type(base).__qualname__ == "head_from.<locals>.Residual" and
             not {"forward", "residual"}.intersection(base.__dict__), "original control factory required")
    for name in ("forward", "residual"):
        method = getattr(type(base), name)
        _require(method.__code__.co_qualname == "head_from.<locals>.Residual." + name and
                 Path(method.__code__.co_filename).name == "train_siglip2_cached_readout.py",
                 "original source method required")
    residual = type(base).residual
    captured = dict(zip(residual.__code__.co_freevars, residual.__closure__ or (), strict=True))
    _require("arm" in captured and captured["arm"].cell_contents == "control", "control source factory required")
    shapes = {"primary.weight": (128, 1152), "primary.bias": (128,),
              "down.weight": (32, 1152), "up.weight": (128, 32)}
    params, buffers = dict(base.named_parameters()), dict(base.named_buffers())
    _require(params.keys() == shapes.keys() and buffers.keys() == {"center", "preactivation_std"},
             "complete source parameters/buffers required")
    for name, shape in shapes.items():
        _check_tensor(params[name], shape, device, frozen=True)
    _check_tensor(buffers["center"], (1152,), device, frozen=True)
    _check_tensor(buffers["preactivation_std"], (), device, frozen=True)
    return [*params.values(), *buffers.values()]


def _check_features(features, device, train=False):
    shape = tuple(features.shape)
    _require((shape == (6355, 1152) if train else
              len(shape) == 2 and shape[0] > 0 and shape[1] == 1152), "feature matrix shape differs")
    dtype = str(features.dtype)
    _require(dtype == "torch.float32" if train else dtype in
             ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"), "floating feature dtype required")
    _check_tensor(features, shape, device, dtype=dtype)


def _check_weight(A, device):
    _check_tensor(A, (128, 32), device)
    _require(A.requires_grad and A.is_leaf and A.grad_fn is None, "trainable leaf A required")


def _check_means(means, device):
    _require(isinstance(means, dict) and means.keys() == {"linear", "quadratic"}, "both fixed means required")
    selected = str(means["linear"].device)
    _require(selected in ("cpu", str(device)), "means must be CPU or selected device")
    for value in means.values():
        _check_tensor(value, (32,), selected, frozen=True)


def _finite(torch, values):
    _require(all(torch.isfinite(value).all().item() for value in values), "nonfinite readout tensor")


def new_weight(device):
    """New zero FP32[128,32] trainable leaf; one per arm."""
    import torch

    device = torch.device(device)
    _require(device.type in ("cpu", "cuda"), "CPU or CUDA device required")
    return torch.nn.Parameter(torch.zeros((128, 32), dtype=torch.float32, device=device))


def fit_means(canonical_train, base):
    """Raw full TRAIN -> original CPU normalize -> head normalize -> means."""
    import torch
    from torch.nn import functional as F

    _check_features(canonical_train, "cpu", train=True)
    source = _check_base(base, "cpu")
    with torch.autocast("cpu", enabled=False), torch.no_grad():
        _finite(torch, [canonical_train, *source])
        x = F.normalize(canonical_train.detach().float(), dim=1)
        z = base.down(F.normalize(x, dim=1) - base.center)
        means = {"linear": z.mean(dim=0).detach(), "quadratic": z.square().mean(dim=0).detach()}
        _check_means(means, "cpu")
        _finite(torch, [z, *means.values()])
    return means


def raw_features(features, base, A, means, arm):
    """FP32 h0+A*phi; TRAIN x is training_features output, FIT x is direct."""
    import torch
    from torch.nn import functional as F

    _check_arm(arm)
    device = features.device
    _require(device.type in ("cpu", "cuda"), "CPU or CUDA device required")
    _check_features(features, device)
    source = _check_base(base, device)
    _check_weight(A, device)
    _require(isinstance(A, torch.nn.Parameter), "A must be a Parameter")
    _check_means(means, device)
    with torch.autocast(device.type, enabled=False):
        with torch.no_grad():
            _finite(torch, [features, A, *source, *means.values()])
            x = features.detach().float()
            h0 = base(x).detach()
            z = base.down(F.normalize(x, dim=1) - base.center)
            phi = (z if arm == "control" else z.square()) - means[
                "linear" if arm == "control" else "quadratic"].detach().to(device=device)
            _check_tensor(h0, (features.shape[0], 128), device, frozen=True)
            _finite(torch, [h0, z, phi])
        raw = h0 + F.linear(phi.detach(), A)
        _finite(torch, [raw])
    return raw
