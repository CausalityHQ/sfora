"""Explicit fresh adaptation from the authenticated native128 TRAIN1000 state."""
RESUME_KEYS = frozenset(("identity", "vision", "buffers", "head", "classifier",
                         "bank", "optimizer", "scaler", "cpu_rng", "cuda_rng"))
SOURCE_SHA = "cc58377f0e9aa90be529bf9a3eca1746a2dc467f765dd9681a3b9b690e324566"
SOURCE_CODE = "a85dd55c516f054c4c63341b31e0c7e4e77fb6925fd9ea29adabf097b1156bcb"
SOURCE_SCHEDULE = "a38d1be83d261856153dd75f59039250137f033fe3880245c13c658b0582d1aa"


def source_runtime(runtime):
    """Authenticate the recorded CUDA source separately from the current device."""
    devices = runtime["buffer_devices"]
    assert devices in ({"embeddings.position_ids": "cpu"},
                       {"embeddings.position_ids": "cuda:0"})
    return {**runtime, "buffer_devices": {"embeddings.position_ids": "cuda:0"}}


def validate_source(saved, expected):
    assert set(saved) == RESUME_KEYS, "complete TRAIN resume state required"
    assert saved["identity"] == {**expected, "global_step": 1000}, "foreign source identity"
    assert len(saved["vision"]) == 400
    moments = saved["optimizer"]["state"]
    assert len(moments) == 208, "source must contain all original optimizer members"
    assert all(int(v["step"]) == 1000 for v in moments.values()), "source counter differs"


def frozen_state(model, boundary):
    assert boundary in (10, 12)
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(boundary))
    values = {n: v for n, v in model.state_dict().items() if n.startswith(roots)}
    values.update((n, v) for n, v in model.named_buffers() if n.startswith(roots))
    return values


def initialize(state, proof, resume, boundary):
    """Caller authenticates immutable source closure and constructs original F5 state."""
    import torch
    import large_dense_boundary as dense
    import pe_large_optimization as old

    assert boundary in (10, 12)
    assert state["counter"] == 0 and not state["optimizer"].state
    assert old.pair.SEED == 179032 and old.pair.sha(resume) == SOURCE_SHA
    rng = torch.random.get_rng_state().clone()
    expected = {**old.identity(state, proof, "half", SOURCE_CODE, SOURCE_SCHEDULE),
                "total_updates": 1000, "precision": "cuda_fp16",
                "intervention": "teacher-preserving-corrected-width-v1",
                "width": 128, "tail_sha256": None}
    expected["runtime"] = source_runtime(expected["runtime"])
    saved = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
    validate_source(saved, expected)
    buffers = dict(state["model"].named_buffers())
    assert buffers.keys() == saved["buffers"].keys()
    assert old.fingerprint(saved["buffers"]) == expected["buffers_sha256"]
    assert saved["classifier"].shape == state["classifier"].shape == (2004, 128)
    assert saved["bank"].shape == state["bank"].shape == (13283, 128)
    assert len(proof["arms"]["half"]["rows"]) == len(state["target"]) == 13283
    state["model"].load_state_dict(saved["vision"], strict=True)
    state["head"].load_state_dict(saved["head"], strict=True)
    with torch.no_grad():
        state["classifier"].copy_(saved["classifier"])
        state["bank"].copy_(saved["bank"])
        for n, value in buffers.items():
            value.copy_(saved["buffers"][n])
    # Authenticate the original frozen12 prefix before enabling blocks10–11.
    assert old.coverage.frozen_digest(state["model"], state["inventory"]) == proof["frozen_prefix_sha256"]
    source_values = {"vision": state["model"].state_dict(), "head": state["head"].state_dict(),
                     "classifier": state["classifier"].detach(), "bank": state["bank"],
                     "buffers": dict(state["model"].named_buffers())}
    assert all(torch.isfinite(v).all() for group in source_values.values()
               for v in (group.values() if isinstance(group, dict) else (group,)))
    source_fingerprint = old.fingerprint(source_values)
    assert source_fingerprint == old.fingerprint({k: saved[k] for k in source_values})
    if boundary == 10:
        state["inventory"] = dense.configure(state["model"])
    state["params"] = old.coverage.parameters(state["model"], state["head"], state["classifier"])
    state["optimizer"] = torch.optim.AdamW([
        {"params": [p for p in state["model"].parameters() if p.requires_grad], "lr": 1e-5},
        {"params": state["head"].parameters(), "lr": 1e-4},
        {"params": [state["classifier"]], "lr": 1e-4}], weight_decay=.05)
    assert len(state["params"]) == (240 if boundary == 10 else 208)
    ids = [id(p) for g in state["optimizer"].param_groups for p in g["params"]]
    assert ids == [id(p) for _, p in state["params"]] and len(ids) == len(set(ids))
    if boundary == 10:
        dense.verify_optimizer(state)
    assert not state["optimizer"].state and state["counter"] == 0
    assert state["scaler"] is None or state["scaler"].get_scale() == 128
    assert torch.equal(rng, torch.random.get_rng_state())
    return source_fingerprint
