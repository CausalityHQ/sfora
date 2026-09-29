"""Fixed dense10 boundary; historical freeze12 helpers remain unchanged."""
ROOTS = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(10))


def verify_boundary(vision):
    assert len(vision.encoder.layers) == 24
    parameters = dict(vision.named_parameters())
    assert all(p.requires_grad == (not name.startswith(ROOTS)) for name, p in parameters.items())
    assert all(any(name.startswith(f"encoder.layers.{i}.") for name in parameters) for i in range(24))
    return {"frozen": tuple(n for n in parameters if n.startswith(ROOTS)),
            "trainable": tuple(n for n in parameters if not n.startswith(ROOTS))}


def configure(vision):
    assert len(vision.encoder.layers) == 24
    vision.requires_grad_(True)
    for module in (vision.embeddings, *vision.encoder.layers[:10]):
        module.requires_grad_(False)
    return verify_boundary(vision)


def frozen_state(vision):
    verify_boundary(vision)
    # state_dict includes buffers as well as parameters under the immutable roots.
    return {n: v for n, v in vision.state_dict().items() if n.startswith(ROOTS)}


def verify_frozen(vision, before):
    import torch
    current = frozen_state(vision)
    assert current.keys() == before.keys()
    assert all(torch.equal(current[n].detach().cpu(), before[n].detach().cpu()) for n in current)


def verify_optimizer(state):
    inventory = verify_boundary(state["model"])
    assert len(inventory["frozen"]) == 163 and len(inventory["trainable"]) == 237
    expected = [(n, p) for n, p in state["model"].named_parameters() if not n.startswith(ROOTS)]
    expected += [("compact_head." + n, p) for n, p in state["head"].named_parameters()]
    expected += [("classifier", state["classifier"])]
    assert len(expected) == 240 and all(p.requires_grad for _, p in expected)
    assert [(n, id(p)) for n, p in state["params"]] == [(n, id(p)) for n, p in expected]
    ids = [id(p) for group in state["optimizer"].param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == 240
    assert ids == [id(p) for _, p in expected]
    assert [g["lr"] for g in state["optimizer"].param_groups] == [1e-5, 1e-4, 1e-4]
    assert all(g["weight_decay"] == .05 for g in state["optimizer"].param_groups)


def fresh(control, native, proof, device):
    import torch
    import pe_large_optimization as old
    model, head, processor, original_inventory = old.coverage.load_native(control, native)
    assert old.coverage.frozen_digest(model, original_inventory) == proof["frozen_prefix_sha256"]
    # Set the boundary before collecting parameters or constructing AdamW.
    inventory = configure(model)
    model.to(device).train()
    head.to(device).train()
    values = old.initializers(proof, "half")
    classifier = torch.nn.Parameter(values["classifier"].to(device))
    bank, target = values["bank"].to(device), values["target"].to(device)
    params = old.coverage.parameters(model, head, classifier)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-4},
        {"params": [classifier], "lr": 1e-4}], weight_decay=.05)
    state = {"model": model, "head": head, "processor": processor, "inventory": inventory,
        "classifier": classifier, "bank": bank, "target": target,
        "positive": old.pair.smoke.member_bank_positive_ordinals(target.cpu().numpy(), allow_singletons=True).to(device),
        "params": params, "optimizer": optimizer,
        "scaler": old.pair.smoke.training_precision("fp16", device="cuda")[1] if str(device).startswith("cuda") else None,
        "counter": 0}
    verify_optimizer(state)
    return state
