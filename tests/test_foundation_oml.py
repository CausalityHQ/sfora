from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from sfora import foundation_oml
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


def test_configure_oml_input_size_uses_timm_positional_resize() -> None:
    class Resizable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observed: tuple[int, int] | None = None

        def set_input_size(self, size: tuple[int, int]) -> None:
            self.observed = size

    model = Resizable()

    foundation_oml.configure_oml_input_size(model, input_size=224)
    assert model.observed is None
    foundation_oml.configure_oml_input_size(model, input_size=288)
    assert model.observed == (288, 288)


def test_load_oml_inshop_examples_uses_released_highres_protocol(tmp_path: Path) -> None:
    partition = tmp_path / "list_eval_partition.txt"
    partition.write_text(
        "7\n"
        "image_name item_id evaluation_status\n"
        "img/WOMEN/Tops/id_00000001/a.jpg id_00000001 train\n"
        "img/WOMEN/Tops/id_00000001/b.jpg id_00000001 train\n"
        "img/WOMEN/Tops/id_00000003/only.jpg id_00000003 train\n"
        "img/WOMEN/Tops/id_00000002/q.jpg id_00000002 query\n"
        "img/WOMEN/Tops/id_00000002/g.jpg id_00000002 gallery\n"
        "img/MEN/Tees/id_00000007/q.jpg id_00000007 query\n"
        "img/MEN/Tees/id_00000007/g.jpg id_00000007 gallery\n"
    )
    image_root = tmp_path / "img_highres"
    for relative in (
        "WOMEN/Tops/id_00000001/a.jpg",
        "WOMEN/Tops/id_00000001/b.jpg",
        "WOMEN/Tops/id_00000003/only.jpg",
        "WOMEN/Tops/id_00000002/q.jpg",
        "WOMEN/Tops/id_00000002/g.jpg",
        "MEN/Tees/id_00000007/q.jpg",
        "MEN/Tees/id_00000007/g.jpg",
    ):
        path = image_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    train = foundation_oml.load_oml_inshop_examples(partition, image_root=image_root, split="train")
    query = foundation_oml.load_oml_inshop_examples(partition, image_root=image_root, split="query")
    gallery = foundation_oml.load_oml_inshop_examples(
        partition, image_root=image_root, split="gallery"
    )

    assert [row.image for row in query] == [
        image_root / "WOMEN/Tops/id_00000002/q.jpg",
        image_root / "MEN/Tees/id_00000007/q.jpg",
    ]
    assert [row.label for row in train] == [0, 0]
    assert [row.label for row in query] == [0, 1]
    assert [row.label for row in gallery] == [0, 1]
