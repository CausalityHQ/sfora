from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_saga_gb10_feasibility.py"
_SPEC = importlib.util.spec_from_file_location("asgcv_qwen_pair_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
PreparedPair = _MODULE.PreparedPair
QwenSagaAdapter = _MODULE.QwenSagaAdapter


class _Processor:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        assert isinstance(messages, list)
        content = messages[0]["content"]
        self.images = [content[0]["image"], content[1]["image"]]
        self.images[0][0, 0, 0] = 255
        return {
            "input_ids": torch.arange(12).reshape(1, 12),
            "attention_mask": torch.ones((1, 12), dtype=torch.int64),
            "pixel_values": torch.ones((8, 3), dtype=torch.float32),
            "image_grid_thw": torch.ones((2, 3), dtype=torch.int64),
            "mm_token_type_ids": torch.tensor(
                [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]],
                dtype=torch.int64,
            ),
        }


def _adapter(processor: _Processor) -> object:
    adapter = object.__new__(QwenSagaAdapter)
    adapter._processor = processor
    adapter._vision_parameters = ()
    return adapter


def test_prepare_image_pair_accepts_only_two_copied_rgb_arrays_and_matches_fixture_path() -> None:
    processor = _Processor()
    adapter = _adapter(processor)
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.ones((8, 8, 3), dtype=np.uint8)

    direct = adapter.prepare_image_pair(
        (first, second),
        "Describe the relation.",
        (2, 5),
        4,
    )
    assert type(direct) is PreparedPair
    assert direct.image_token_ranges == ((1, 5), (6, 10))
    assert direct.attribute_token_span == (2, 5)
    assert direct.patch_tokens_per_image == 4
    assert first[0, 0, 0] == 0
    assert not np.shares_memory(processor.images[0], first)
    assert "label" not in inspect.signature(adapter.prepare_image_pair).parameters

    adapter._images = lambda fixture, ordinals: [first, second]  # type: ignore[method-assign]
    fixture = SimpleNamespace(
        pair_ordinals=(0, 1),
        prompt_utf8="Describe the relation.",
        attribute_token_span=(2, 5),
        patch_tokens_per_image=4,
    )
    delegated = adapter.prepare_pair(fixture)
    assert delegated.inputs.keys() == direct.inputs.keys()
    assert delegated.image_token_ranges == direct.image_token_ranges
    assert delegated.attribute_token_span == direct.attribute_token_span


@pytest.mark.parametrize(
    "images",
    [
        (np.zeros((8, 8, 3), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.uint8)),
        (np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)),
        (np.zeros((8, 8, 3), dtype=np.uint8),),
        [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
    ],
)
def test_prepare_image_pair_rejects_non_authoritative_image_inputs(images: object) -> None:
    with pytest.raises(ValueError):
        _adapter(_Processor()).prepare_image_pair(images, "prompt", (0, 1), 4)
