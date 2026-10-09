def inference_outputs(endpoint, images):
    if "_serving" in endpoint:
        _check_runtime()
        context = endpoint["_serving"]
        _serving_helper_binding((context["helpers"], context["helper_guards"], context["checker"]))
    import torch
    from torch.nn import functional as F

    modules, device = endpoint["modules"], endpoint["device"]
    _check_runtime()
    if "_serving" in endpoint:
        _serving_live_identity(endpoint)
    else:
        for module in modules.values():
            bound_file({}, module.__file__, endpoint["guards"][module.__file__])
        for filename in SERVING_FILES | {"joint_relational_compaction.py"}:
            path = endpoint["directory"] / filename
            bound_file({}, path, endpoint["guards"][str(path)])
    require(
        0 < len(images) <= 32 and numerical_flags() == endpoint["flags"],
        "serving batch/numerics differ",
    )
    require(
        endpoint["encoder_identity"] == endpoint["manifest"]["encoder_identity"]
        and endpoint["vision_sha256"] == endpoint["manifest"]["vision_sha256"]
        and all(
            p.grad is None
            and not p.requires_grad
            and p.dtype == torch.float32
            and p.device.type == device
            for p in endpoint["head_object"].parameters()
        )
        and all(
            m.training
            and not m._forward_hooks
            and not m._forward_pre_hooks
            and not m._backward_hooks
            for m in endpoint["head_object"].modules()
        ),
        "serving authenticated encoder/frozen head roles/hooks differ",
    )
    require(
        encoder_facts(endpoint, endpoint["manifest"]["environment"]["packages"], serving=True)[
            "vision_sha256"
        ]
        == endpoint["vision_sha256"]
        and fingerprint(inference_readout_tree(endpoint)) == endpoint["readout_sha256"],
        "current .data updated encoder/readout/role substitution rejected",
    )
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
    pixels = endpoint["processor_object"](images=images, return_tensors="pt")["pixel_values"]
    require(
        pixels.shape == (len(images), 3, 256, 256)
        and pixels.dtype == torch.float32
        and torch.isfinite(pixels).all().item(),
        "owned processor pixels differ",
    )
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            pooled = endpoint["model"](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device, enabled=False):
            features = F.normalize(pooled.float(), dim=1)
            raw = fullfeature_raw_features(
                features,
                endpoint["head_object"],
                endpoint["A"],
                endpoint["means"],
                endpoint["C"],
                endpoint["mu_train"],
                endpoint["arm"],
            )
            require((raw.norm(dim=1) > 0).all().item(), "nonzero portable raw required")
            unit = F.normalize(raw, dim=1)
            packed = _pack(unit.cpu())
    require(
        torch.equal(cpu_rng, torch.random.get_rng_state())
        and all(
            torch.equal(a, b)
            for a, b in zip(
                cuda_rng, torch.cuda.get_rng_state_all() if device == "cuda" else [], strict=True
            )
        ),
        "serving complete RNG changed",
    )
    return {
        "raw": raw.cpu(),
        "unit": unit.cpu(),
        "codes": packed.codes.cpu(),
        "inverse_norms": packed.inverse_norms.cpu(),
        "wire": packed.to_bytes(),
    }
