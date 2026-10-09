def construct_encoder(construct_context, config, buffers, processor_config, base, overlay=None):
    """One genuine CPU factory/strict base load, then exact overlay and transfer."""
    import torch
    from transformers import AutoImageProcessor

    guards = construct_context["guards"]
    fact = base["checkpoint"]
    path = bound_file(guards, fact["path"], fact["sha256"])
    model = construct(config, construct_context).eval()
    disk = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    primary = None
    try:
        with path.open("rb") as stream:
            pages = CheckpointPages(stream)
            require(
                disk.keys() == {"vision", "buffers", "config", "runtime", "cpu_rng"}
                and disk["config"] == config
                and fingerprint(disk["buffers"]) == fingerprint(buffers)
                and len(disk["vision"]) == 448,
                "complete authenticated original base differs",
            )
            digest = fingerprint(disk["vision"], consumed=pages.consume)
            require(
                base.get("sha256", digest) == digest, "original typed base vision identity differs"
            )
            load_vision(model, disk["vision"], pages)
            require(
                dict(model.named_buffers()).keys() == buffers.keys() == {"embeddings.position_ids"},
                "complete nonpersistent buffer inventory differs",
            )
            with torch.no_grad():
                for name, value in model.named_buffers():
                    require(
                        value.shape == disk["buffers"][name].shape
                        and value.dtype == disk["buffers"][name].dtype,
                        "buffer shape/dtype differs",
                    )
                    value.copy_(disk["buffers"][name])
                    pages.consume(disk["buffers"][name])
            del value
            require(
                fingerprint(model.state_dict()) == digest,
                "genuine strict copied original448 differs",
            )
        processor = AutoImageProcessor.from_pretrained(
            processor_config, local_files_only=True, backend="torchvision"
        )
        cache = _processor_cache(processor, guards, empty=True)
        require(
            "position_ids" in model.embeddings._non_persistent_buffers_set
            and torch.equal(
                dict(model.named_buffers())["embeddings.position_ids"],
                torch.arange(256).expand(1, -1),
            ),
            "original nonpersistent position buffer differs",
        )
        if overlay is not None:
            apply_overlay(model, overlay)
        structure = model_structure(model, construct_context["packages"])
        base = {"checkpoint": copy.deepcopy(fact), "sha256": digest}
        del pages
    except BaseException as error:
        primary = error
        raise
    finally:
        del disk
        if "_serving" not in construct_context:
            gc.collect()
            mapping_absent(path)
        else:
            try:
                gc.collect()
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    primary.add_note("serving constructor collection failed: " + repr(cleanup))
            try:
                mapping_absent(path)
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    primary.add_note("serving constructor mapping check failed: " + repr(cleanup))
            if primary is not None:
                raise primary
    return model, processor, cache, base, structure
