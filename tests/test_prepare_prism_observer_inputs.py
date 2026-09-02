from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_prism_observer_inputs import prepare_prism_observer_inputs
from sfora.data import ImageExample
from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import PRISM_CHANNELS
from sfora.prism_observer import PrismChannelPrompt, PrismPromptBundle


def _prompt_bundle() -> PrismPromptBundle:
    rows = []
    for channel in PRISM_CHANNELS:
        prompt = (
            f"channel={channel}; compare the two anonymous vehicle images using only "
            "the named visual cue. Return the registered compact completion grammar."
        )
        rows.append(
            PrismChannelPrompt(
                channel=channel,
                prompt_utf8=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                max_new_tokens=192,
                temperature_ppm=1_000_000,
                top_p_ppm=1_000_000,
            )
        )
    return PrismPromptBundle(
        schema="sfora-prism-prompt-bundle-v1", rows=tuple(rows)
    )


def _examples() -> tuple[ImageExample, ...]:
    rows: list[ImageExample] = []
    ordinal = 0
    for label in range(49):
        for within in range(8):
            color = (ordinal & 255, (ordinal >> 8) & 255, 17)
            rows.append(
                ImageExample(
                    example_id=f"private/optimization-{label}-{within}.png",
                    image=Image.new("RGB", (2, 2), color),
                    label=label,
                )
            )
            ordinal += 1
    for label in (82, 83):
        for within in range(32):
            color = (ordinal & 255, (ordinal >> 8) & 255, 29)
            rows.append(
                ImageExample(
                    example_id=f"private/diagnostic-{label}-{within}.png",
                    image=Image.new("RGB", (2, 2), color),
                    label=label,
                )
            )
            ordinal += 1
    return tuple(rows)


def _load(path: Path) -> object:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json_bytes(value)
    return value


def test_preparation_separates_anonymous_payloads_from_private_truth(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"

    prepare_prism_observer_inputs(
        output,
        _examples(),
        source_identity="prism-preparation-fixture-v1",
        prompt_bundle=_prompt_bundle(),
    )

    observations = _load(output / "private" / "observations.json")
    scoring = _load(output / "private" / "scoring.json")
    capability = _load(output / "calibration" / "capability.json")
    calibration_manifest = _load(output / "calibration" / "payload-manifest.json")
    diagnostic_manifest = _load(
        output / "diagnostic-sealed" / "payload-manifest.json"
    )
    assert len(observations["rows"]) == 1_280
    assert len(scoring["rows"]) == 160
    assert len(capability["rows"]) == 1_024
    assert len(calibration_manifest["payloads"]) == 256
    assert len(diagnostic_manifest["payloads"]) == 64
    assert len(tuple((output / "calibration" / "payloads").glob("*.png"))) == 256
    assert len(tuple((output / "diagnostic-sealed" / "payloads").glob("*.png"))) == 64
    public_bytes = b"".join(
        path.read_bytes()
        for path in (
            output / "calibration" / "capability.json",
            output / "calibration" / "payload-manifest.json",
            output / "diagnostic-sealed" / "payload-manifest.json",
        )
    ).lower()
    for forbidden in (
        b"label",
        b"relation",
        b"example_id",
        b"pair_ordinal",
        b"fold",
        b"private/",
    ):
        assert forbidden not in public_bytes
    assert not (output / "calibration" / "scoring.json").exists()
    for manifest in (calibration_manifest, diagnostic_manifest):
        for row in manifest["payloads"]:
            assert frozenset(row) == frozenset(
                ("payload_sha256", "byte_length", "width", "height", "mode")
            )
            assert row["mode"] == "RGB"
            assert row["width"] == 2 and row["height"] == 2


def test_preparation_rejects_disallowed_labels_without_partial_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    rows = list(_examples())
    rows[0] = ImageExample(
        example_id=rows[0].example_id,
        image=rows[0].image,
        label=49,
    )

    with pytest.raises(ValueError, match="label"):
        prepare_prism_observer_inputs(
            output,
            tuple(rows),
            source_identity="prism-preparation-fixture-v1",
            prompt_bundle=_prompt_bundle(),
        )

    assert not output.exists()


def test_preparation_rejects_duplicate_pixels_and_existing_output(
    tmp_path: Path,
) -> None:
    rows = list(_examples())
    rows[1] = ImageExample(
        example_id=rows[1].example_id,
        image=rows[0].image.copy(),
        label=rows[1].label,
    )
    with pytest.raises(ValueError, match="duplicate image bytes"):
        prepare_prism_observer_inputs(
            tmp_path / "duplicate",
            tuple(rows),
            source_identity="prism-preparation-fixture-v1",
            prompt_bundle=_prompt_bundle(),
        )
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_prism_observer_inputs(
            output,
            _examples(),
            source_identity="prism-preparation-fixture-v1",
            prompt_bundle=_prompt_bundle(),
        )


def test_preparation_rejects_non_rgb_and_symlink_payloads(tmp_path: Path) -> None:
    rows = list(_examples())
    rows[0] = ImageExample(
        example_id=rows[0].example_id,
        image=Image.new("L", (2, 2), 1),
        label=rows[0].label,
    )
    with pytest.raises(ValueError, match="RGB"):
        prepare_prism_observer_inputs(
            tmp_path / "non-rgb",
            tuple(rows),
            source_identity="prism-preparation-fixture-v1",
            prompt_bundle=_prompt_bundle(),
        )
    target = tmp_path / "target.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(target)
    link = tmp_path / "link.png"
    link.symlink_to(target)
    rows = list(_examples())
    rows[0] = ImageExample(
        example_id=rows[0].example_id,
        image=link,
        label=rows[0].label,
    )
    with pytest.raises(ValueError, match="symlink"):
        prepare_prism_observer_inputs(
            tmp_path / "symlink",
            tuple(rows),
            source_identity="prism-preparation-fixture-v1",
            prompt_bundle=_prompt_bundle(),
        )
