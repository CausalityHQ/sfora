"""Genuine loader body candidate; failed-load lifecycle candidate."""

def _serving_load_payload(prepared):
    directory = Path(prepared['directory'])
    manifest = copy.deepcopy(prepared['origin'])
    manifest['environment'] = copy.deepcopy(prepared['installed']['expected_environment'])
    guards, device = prepared['guards'], 'cuda'
    modules = {"runtime": sys.modules[__name__]}
    import torch

    path = directory / "endpoint.pt"
    disk = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    primary = None
    try:
        expected = _serving_expected_identities(prepared, disk)
        manifest['encoder_identity'] = expected['model']
        env = manifest["environment"]
        construct_context = {
            "_serving": prepared,
            "packages": env["packages"],
            "guards": guards,
            "sources": {
                "native_environment": {"vision_constructor": {"path": env["vision_constructor"]}}
            },
        }
        owned_base = {
            "checkpoint": {
                "path": str(directory / "vision.pt"),
                "sha256": manifest["files"]["vision.pt"],
            },
            "sha256": disk["base_vision"]["sha256"],
        }
        model, processor, cache, _, structure = construct_encoder(
            construct_context,
            disk["config"],
            disk["buffers"],
            directory / "processor.json",
            owned_base,
            disk["encoder"],
        )
        # The overlay is copied by apply_overlay within the shared strict CPU
        # constructor before device transfer; the full updated identity is checked.
        require(
            fingerprint(model.state_dict()) == disk["vision_sha256"]
            and _serving_exact_json(structure, expected["model"]["runtime"]),
            "updated full448 portable reload differs",
        )
        model.requires_grad_(False).eval().to(device)
        with path.open("rb") as stream:
            pages = CheckpointPages(stream)
            copied = {
                k: owned_copy(disk[k], pages, device)
                for k in ("head", "A", "C", "means", "mu_train", "common_statistics")
            }
            head = (
                head_from("control", tensors=copied.pop("head"))
                .to(device)
                .requires_grad_(False)
                .train()
            )
            endpoint = {
                **copied,
                "A": torch.nn.Parameter(copied["A"], requires_grad=True),
                "C": torch.nn.Parameter(copied["C"], requires_grad=True),
                "head_object": head,
                "model": model,
                "processor_object": processor,
                "processor_cache": cache,
                "processor": expected["processor"],
                "arm": disk["arm"],
                "scope": copy.deepcopy(disk["scope"]),
                "mu_train_provenance": copy.deepcopy(disk["mu_train_provenance"]),
                "encoder_identity": expected["model"],
                "vision_sha256": disk["vision_sha256"],
                "flags": copy.deepcopy(disk["numerical_flags"]),
                "device": device,
                "modules": modules,
                "guards": guards,
                "manifest": manifest,
                "directory": directory,
                "_serving": prepared,
            }
            endpoint["readout_sha256"] = fingerprint(inference_readout_tree(endpoint))
            del copied, pages, model, processor, head
    except BaseException as error:
        primary = error
        raise
    finally:
        del disk
        try:
            gc.collect()
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                primary.add_note('serving endpoint collection failed: '+repr(cleanup))
        try:
            mapping_absent(path)
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                primary.add_note('serving endpoint mapping check failed: '+repr(cleanup))
        if primary is not None:
            raise primary
    _serving_live_identity(endpoint)
    require(
        encoder_facts(endpoint, manifest["environment"]["packages"], serving=True)["vision_sha256"]
        == endpoint["vision_sha256"],
        "public updated encoder identity differs",
    )
    return endpoint
