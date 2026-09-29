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
