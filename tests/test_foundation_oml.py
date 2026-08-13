from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from sfora.foundation_oml import oml_vit_state_dict


def test_oml_vit_state_dict_removes_only_the_lightning_wrapper() -> None:
    checkpoint = {
        "state_dict": OrderedDict(
            (
                ("model.model.cls_token", torch.ones(1, 1, 3)),
                ("model.model.blocks.0.norm1.weight", torch.ones(3)),
            )
        )
    }

    state = oml_vit_state_dict(checkpoint)

    assert tuple(state) == ("cls_token", "blocks.0.norm1.weight")


def test_oml_vit_state_dict_rejects_foreign_checkpoint_keys() -> None:
    with pytest.raises(ValueError, match="wrapper"):
        oml_vit_state_dict({"state_dict": {"encoder.weight": torch.ones(2)}})
