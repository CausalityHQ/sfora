from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.run_native_twin_probe import (
    matched_control_crop,
    native_crop_boxes,
    parse_args,
)


def test_native_crop_boxes_are_exact_label_independent_three_by_three_grid() -> None:
    assert native_crop_boxes(600, 300) == (
        (0, 0, 400, 200),
        (100, 0, 500, 200),
        (200, 0, 600, 200),
        (0, 50, 400, 250),
        (100, 50, 500, 250),
        (200, 50, 600, 250),
        (0, 100, 400, 300),
        (100, 100, 500, 300),
        (200, 100, 600, 300),
    )
    assert len(set(native_crop_boxes(599, 301))) == 9
    with pytest.raises(ValueError, match="dimensions"):
        native_crop_boxes(0, 300)
    with pytest.raises(TypeError, match="dimensions"):
        native_crop_boxes(True, 300)


def test_matched_control_crop_preserves_geometry_but_caps_pixel_information() -> None:
    """Removing the 256px matched control must fail this causal contract."""

    image = Image.fromarray(
        np.arange(600 * 300 * 3, dtype=np.uint8).reshape(300, 600, 3),
        mode="RGB",
    )
    crop = image.crop(native_crop_boxes(600, 300)[0])

    control = matched_control_crop(crop)

    assert crop.size == (400, 200)
    assert control.size == crop.size
    assert control.mode == "RGB"
    assert control.tobytes() != crop.tobytes()
    small = Image.new("RGB", (200, 100), color=(1, 2, 3))
    assert matched_control_crop(small).tobytes() == small.tobytes()
    with pytest.raises(ValueError, match="dimensions"):
        matched_control_crop(Image.new("RGB", (1, 1)))


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--seed-receipt",
        str(tmp_path / "seed.json"),
        "--checkpoint-receipt",
        str(tmp_path / "checkpoint.json"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--output",
        str(tmp_path / "result.json"),
        "--source-revision",
        "1" * 40,
        "--source-tree-digest",
        "2" * 64,
        "--probe-revision",
        "3" * 40,
        "--probe-tree-digest",
        "4" * 64,
        "--seed",
        "17",
        "--execute-native-twin-probe",
    ]


def test_native_probe_cli_is_explicit_and_refuses_policy_or_data_selection_flags(
    tmp_path: Path,
) -> None:
    parsed = parse_args(_arguments(tmp_path))
    assert parsed.seed == 17
    assert parsed.batch_size == 16
    for forbidden in (
        "--crop-policy",
        "--candidate-block",
        "--labels",
        "--clean-manifest",
        "--test-manifest",
        "--prism-result",
        "--checkpoint-selection",
    ):
        with pytest.raises(SystemExit):
            parse_args([*_arguments(tmp_path), forbidden, "anything"])
    without_execute = _arguments(tmp_path)[:-1]
    with pytest.raises(SystemExit):
        parse_args(without_execute)
