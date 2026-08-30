"""Render authenticated frozen-substrate retrieval errors for human taxonomy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

_SHEET_SIZE = (768, 432)
_IMAGE_BOX = (352, 264)


def _read_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _validated_rows(
    *, manifest: dict[str, object], examples: Sequence[Any], class_names: Sequence[str]
) -> list[dict[str, Any]]:
    if manifest.get("schema") != "sfora-frozen-substrate-errors-v1":
        raise ValueError("error manifest schema differs")
    if (
        manifest.get("claim_eligible") is not False
        or manifest.get("dataset") != "cars"
        or manifest.get("split") != "train"
        or manifest.get("cell") != "siglip-so400m"
        or manifest.get("holdout_classes") != list(range(82, 98))
    ):
        raise ValueError("error manifest authority differs")
    if len(class_names) <= 97:
        raise ValueError("class name authority is incomplete")
    manifest_class_names = manifest.get("class_names")
    expected_class_names = [
        {"id": label, "name": class_names[label]} for label in range(82, 98)
    ]
    if manifest_class_names != expected_class_names:
        raise ValueError("class name authority differs from error manifest")
    raw_rows = manifest.get("errors")
    if not isinstance(raw_rows, list):
        raise ValueError("errors must be a list")
    error_count = _read_int(manifest.get("error_count"), field="error count")
    if error_count < 1 or error_count != len(raw_rows):
        raise ValueError("error count differs from rows")

    rows: list[dict[str, Any]] = []
    previous_query = -1
    expected_keys = {
        "query_position",
        "query_example_id",
        "query_label",
        "nearest_position",
        "nearest_example_id",
        "nearest_label",
    }
    for raw in raw_rows:
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("error row schema differs")
        query_position = _read_int(raw["query_position"], field="query position")
        nearest_position = _read_int(raw["nearest_position"], field="nearest position")
        query_label = _read_int(raw["query_label"], field="query label")
        nearest_label = _read_int(raw["nearest_label"], field="nearest label")
        if query_position <= previous_query:
            raise ValueError("errors must retain increasing query order")
        if not 0 <= query_position < len(examples) or not 0 <= nearest_position < len(
            examples
        ):
            raise ValueError("error position is out of range")
        query = examples[query_position]
        nearest = examples[nearest_position]
        if (
            str(query.example_id) != raw["query_example_id"]
            or str(nearest.example_id) != raw["nearest_example_id"]
            or int(query.label) != query_label
            or int(nearest.label) != nearest_label
        ):
            raise ValueError("example identity differs from error manifest")
        if not 0 <= query_label < len(class_names) or not 0 <= nearest_label < len(
            class_names
        ):
            raise ValueError("class name authority is incomplete")
        rows.append(
            {
                **raw,
                "query": query,
                "nearest": nearest,
                "query_class_name": class_names[query_label],
                "nearest_class_name": class_names[nearest_label],
            }
        )
        previous_query = query_position
    return rows


def _paste_image(canvas: Image.Image, image: object, *, x: int, y: int) -> None:
    if not isinstance(image, Image.Image):
        raise ValueError("contact-sheet examples must contain PIL images")
    materialized = ImageOps.contain(image.convert("RGB"), _IMAGE_BOX)
    left = x + (_IMAGE_BOX[0] - materialized.width) // 2
    top = y + (_IMAGE_BOX[1] - materialized.height) // 2
    canvas.paste(materialized, (left, top))


def _render_row(row: dict[str, Any], *, ordinal: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", _SHEET_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"sealed error {ordinal + 1}/{total}", fill="black")
    draw.text(
        (16, 34),
        f"query {row['query_position']}: {row['query_class_name']}",
        fill="black",
    )
    draw.text(
        (400, 34),
        f"nearest {row['nearest_position']}: {row['nearest_class_name']}",
        fill="black",
    )
    _paste_image(canvas, row["query"].image, x=16, y=80)
    _paste_image(canvas, row["nearest"].image, x=400, y=80)
    draw.text((16, 360), str(row["query_example_id"]), fill="black")
    draw.text((400, 360), str(row["nearest_example_id"]), fill="black")
    return canvas


def render_error_contact_sheets(
    *,
    manifest: dict[str, object],
    examples: Sequence[Any],
    class_names: Sequence[str],
    output_dir: Path,
    pairs_per_sheet: int,
) -> tuple[Path, ...]:
    """Render ordered error pairs without overwriting an existing evidence image."""

    if pairs_per_sheet < 1:
        raise ValueError("pairs_per_sheet must be positive")
    rows = _validated_rows(
        manifest=manifest, examples=examples, class_names=class_names
    )
    page_count = math.ceil(len(rows) / pairs_per_sheet)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_dir / f"errors-{index:03d}.png" for index in range(page_count))
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    for page_index, output in enumerate(outputs):
        start = page_index * pairs_per_sheet
        page_rows = rows[start : start + pairs_per_sheet]
        page = Image.new("RGB", (_SHEET_SIZE[0], _SHEET_SIZE[1] * len(page_rows)), "white")
        for offset, row in enumerate(page_rows):
            rendered = _render_row(row, ordinal=start + offset, total=len(rows))
            page.paste(rendered, (0, offset * _SHEET_SIZE[1]))
        page.save(output, format="PNG")
    return outputs
