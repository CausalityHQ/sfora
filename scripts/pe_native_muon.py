#!/usr/bin/env python3
"""Fixed native hidden-matrix Muon/remaining-parameter AdamW split."""

import torch

from pe_core_training import named_training_parameters


def build(vision, head, classifier):
    native = dict(named_training_parameters(vision, "pe"))
    shapes = {
        "attn.in_proj_weight": (2304, 768),
        "attn.out_proj.weight": (768, 768),
        "mlp.c_fc.weight": (3072, 768),
        "mlp.c_proj.weight": (768, 3072),
    }
    names = sorted(
        f"transformer.resblocks.{i}.{suffix}" for i in range(6, 12) for suffix in shapes
    )
    matrices = [native[n] for n in names]
    assert all(
        p.requires_grad
        and p.dtype == torch.float32
        and p.is_contiguous()
        and tuple(p.shape) == shapes[n.split(".", 3)[3]]
        for n, p in zip(names, matrices, strict=True)
    )
    matrix_ids = {id(p) for p in matrices}
    assert len(matrix_ids) == 24 and sum(p.numel() for p in matrices) == 42467328
    other = [p for p in native.values() if p.requires_grad and id(p) not in matrix_ids]
    params = matrices + other + list(head.parameters()) + [classifier]
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {
        id(p) for p in native.values() if p.requires_grad
    } | {id(p) for p in head.parameters()} | {id(classifier)}
    muon = torch.optim.Muon(
        matrices,
        lr=1e-5,
        weight_decay=0.05,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adam = torch.optim.AdamW(
        [
            {"params": other, "lr": 1e-5},
            {"params": list(head.parameters()), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    assert {
        id(p) for opt in (muon, adam) for g in opt.param_groups for p in g["params"]
    } == {id(p) for p in params}
    return (muon, adam), params, names


def step(optimizers, scaler, params):
    # Reject a nonfinite global gradient before either optimizer mutates weights.
    for optimizer in optimizers:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
    scale = scaler.get_scale()
    for optimizer in optimizers:
        scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() >= scale, "an optimizer update was skipped"
    assert all(torch.isfinite(p).all() for p in params)
    assert all(
        torch.isfinite(v).all()
        for opt in optimizers
        for state in opt.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    return float(norm)
