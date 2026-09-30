"""Fixed normalized final-token residual; runtime qualification belongs to the caller."""

METHOD = "token-quadrant-residual-v1"


def new_weight(device, trainable):
    """Allocate exact zeros without drawing from either CPU or CUDA RNG."""
    import torch

    assert type(trainable) is bool, "residual role must be explicit"
    return torch.nn.Parameter(torch.zeros((128, 4096), dtype=torch.float32, device=device),
                              requires_grad=trainable)


def validate_weight(weight, arm):
    """Validate live or serialized W; control always retains exact zeros."""
    import torch

    assert arm in ("control", "candidate"), "foreign residual arm"
    assert isinstance(weight, torch.Tensor), "residual tensor missing"
    assert weight.shape == (128, 4096) and weight.dtype == torch.float32, "residual geometry differs"
    assert bool(torch.isfinite(weight).all()), "nonfinite residual weight"
    if arm == "control":
        assert not bool(torch.count_nonzero(weight)), "control residual must remain zero"


def quadrant_features(tokens):
    """FP32 row-major NW/NE/SW/SE means, concatenated and unit-normalized."""
    import torch
    from torch.nn import functional as F

    assert isinstance(tokens, torch.Tensor) and tokens.ndim == 3, "final patch tokens required"
    assert tokens.shape[1:] == (256, 1024), "final patch layout differs"
    assert tokens.is_floating_point() and bool(torch.isfinite(tokens).all()), "invalid final patch values"
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        grid = tokens.float().reshape(tokens.shape[0], 16, 16, 1024)
        values = torch.cat([grid[:, :8, :8].mean(dim=(1, 2)),
                            grid[:, :8, 8:].mean(dim=(1, 2)),
                            grid[:, 8:, :8].mean(dim=(1, 2)),
                            grid[:, 8:, 8:].mean(dim=(1, 2))], dim=1)
        norm = torch.linalg.vector_norm(values, dim=1)
        assert bool(torch.isfinite(norm).all()) and bool((norm > 0).all()), "invalid quadrant norm"
        return F.normalize(values, dim=1)


def raw_features(pooled, tokens, head, weight):
    """Keep the actual biased compact head and add the live FP32 token readout."""
    import torch
    from torch.nn import functional as F
    from sfora.sop_compact_training import compact_head_features

    validate_weight(weight, "candidate")
    assert isinstance(pooled, torch.Tensor) and pooled.ndim == 2, "pooled batch required"
    assert isinstance(tokens, torch.Tensor) and tokens.ndim == 3, "token batch required"
    assert pooled.shape == (tokens.shape[0], 1024), "pooled/token batch differs"
    assert pooled.device == tokens.device == weight.device, "residual device differs"
    assert isinstance(head, torch.nn.Linear) and head.weight.dtype == torch.float32, "FP32 head required"
    assert head.bias is None or (head.bias.dtype == torch.float32 and head.bias.device == pooled.device)
    with torch.autocast(device_type=pooled.device.type, enabled=False):
        raw = compact_head_features(pooled.float(), head) + F.linear(quadrant_features(tokens), weight)
        assert bool(torch.isfinite(raw).all()), "nonfinite combined descriptor"
        return raw


def _validate_members(state, arm=None):
    import pe_large_coverage as coverage

    expected = coverage.parameters(state["model"], state["head"], state["classifier"])
    assert len(expected) == 208, "fresh boundary12 members required"
    if arm == "candidate":
        expected.append(("residual", state["residual"]))
    assert [(n, id(p)) for n, p in state["params"]] == [(n, id(p)) for n, p in expected], "parameter order differs"
    groups = state["optimizer"].param_groups
    assert len(groups) == (4 if arm == "candidate" else 3), "optimizer groups differ"
    expected_groups = [[p for p in state["model"].parameters() if p.requires_grad],
                       list(state["head"].parameters()), [state["classifier"]]]
    if arm == "candidate":
        expected_groups.append([state["residual"]])
    assert [[id(p) for p in group["params"]] for group in groups] == [
        [id(p) for p in group] for group in expected_groups], "optimizer group membership differs"
    ids = [id(p) for group in groups for p in group["params"]]
    assert ids == [id(p) for _, p in expected] and len(ids) == len(set(ids)), "optimizer membership differs"
    assert [group["lr"] for group in groups] == [1e-5] + [1e-4] * (len(groups) - 1), "optimizer rates differ"


def attach(state, arm):
    """Attach W once to the fresh source state; preserve the original three groups."""
    import torch

    assert arm in ("control", "candidate"), "foreign residual arm"
    assert "residual" not in state, "residual already attached"
    optimizer = state["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW), "fresh AdamW required"
    assert type(state["counter"]) is int and state["counter"] == 0 and not optimizer.state, "fresh optimizer required"
    _validate_members(state)
    weight = new_weight(state["classifier"].device, arm == "candidate")
    validate_weight(weight, arm)
    if arm == "candidate":
        optimizer.add_param_group({"params": [weight], "lr": 1e-4})
        state["params"].append(("residual", weight))
    state["residual"] = weight
    _validate_members(state, arm)
    assert weight.requires_grad == (arm == "candidate")
    assert weight.device == state["classifier"].device and not optimizer.state and state["counter"] == 0
    if arm == "candidate":
        assert {k: v for k, v in optimizer.param_groups[-1].items() if k != "params"} == {
            **optimizer.defaults, "lr": 1e-4}, "residual optimizer options differ"
    return weight
