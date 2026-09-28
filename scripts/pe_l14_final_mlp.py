"""Final native token MLP and attention pool learn; attention/norms stay frozen."""

import pe_l14_native_pool as pool
from pe_l14_final_block import (
    assert_frozen_tokens as assert_frozen_tokens,
    runtime_identity as runtime_identity,
    verified_features as verified_features,
)
from pe_core_training import named_training_parameters


def freeze(model):
    pool.freeze(model)
    assert len(model.transformer.resblocks) == 24
    model.transformer.resblocks[-1].mlp.requires_grad_(True)
    return {
        key: tuple(
            n
            for n, p in named_training_parameters(model, "pe")
            if p.requires_grad == active
        )
        for key, active in (("frozen", False), ("trainable", True))
    }


def frozen_state(model):
    return {
        n: value
        for n, value in pool.frozen_state(model).items()
        if not n.startswith("transformer.resblocks.23.mlp.")
    }
