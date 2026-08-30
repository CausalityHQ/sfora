from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sfora.error_contact_sheet import render_error_contact_sheets


def _manifest() -> dict[str, object]:
    return {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "dataset": "cars",
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "class_names": [
            {"id": index, "name": f"class-{index}"} for index in range(82, 98)
        ],
        "cell": "siglip-so400m",
        "error_count": 2,
        "errors": [
            {
                "query_position": 0,
                "query_example_id": "cars-train-000",
                "query_label": 82,
                "nearest_position": 1,
                "nearest_example_id": "cars-train-001",
                "nearest_label": 83,
            },
            {
                "query_position": 2,
                "query_example_id": "cars-train-002",
                "query_label": 84,
                "nearest_position": 3,
                "nearest_example_id": "cars-train-003",
                "nearest_label": 85,
            },
        ],
    }


def _examples() -> list[SimpleNamespace]:
    colors = ("red", "green", "blue", "yellow")
    return [
        SimpleNamespace(
            example_id=f"cars-train-{index:03d}",
            label=82 + index,
            image=Image.new("RGB", (12, 8), color=color),
        )
        for index, color in enumerate(colors)
    ]


def test_renderer_preserves_manifest_order_and_writes_bounded_pages(
    tmp_path: Path,
) -> None:
    outputs = render_error_contact_sheets(
        manifest=_manifest(),
        examples=_examples(),
        class_names=[f"class-{index}" for index in range(196)],
        output_dir=tmp_path,
        pairs_per_sheet=1,
    )

    assert [path.name for path in outputs] == ["errors-000.png", "errors-001.png"]
    first = Image.open(outputs[0])
    second = Image.open(outputs[1])
    assert first.size == second.size == (768, 432)
    assert first.getpixel((100, 150))[0] > first.getpixel((100, 150))[2]
    assert second.getpixel((100, 150))[2] > second.getpixel((100, 150))[0]


def test_renderer_groups_only_the_registered_number_of_pairs_per_sheet(
    tmp_path: Path,
) -> None:
    outputs = render_error_contact_sheets(
        manifest=_manifest(),
        examples=_examples(),
        class_names=[f"class-{index}" for index in range(196)],
        output_dir=tmp_path,
        pairs_per_sheet=2,
    )

    assert [path.name for path in outputs] == ["errors-000.png"]
    assert Image.open(outputs[0]).size == (768, 864)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"error_count": 3}, "error count"),
        ({"schema": "drift"}, "schema"),
        ({"errors": []}, "error count"),
    ],
)
def test_renderer_rejects_manifest_authority_drift(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    manifest = _manifest()
    manifest.update(mutation)
    with pytest.raises(ValueError, match=message):
        render_error_contact_sheets(
            manifest=manifest,
            examples=_examples(),
            class_names=[f"class-{index}" for index in range(196)],
            output_dir=tmp_path,
            pairs_per_sheet=2,
        )


def test_renderer_rejects_example_identity_drift_before_writing(tmp_path: Path) -> None:
    examples = _examples()
    examples[1] = SimpleNamespace(
        example_id="wrong-id", label=83, image=examples[1].image
    )
    with pytest.raises(ValueError, match="example identity"):
        render_error_contact_sheets(
            manifest=_manifest(),
            examples=examples,
            class_names=[f"class-{index}" for index in range(196)],
            output_dir=tmp_path,
            pairs_per_sheet=2,
        )
    assert list(tmp_path.iterdir()) == []


def test_renderer_rejects_class_names_that_differ_from_manifest_authority(
    tmp_path: Path,
) -> None:
    class_names = [f"class-{index}" for index in range(196)]
    class_names[82] = "wrong"
    with pytest.raises(ValueError, match="class name authority"):
        render_error_contact_sheets(
            manifest=_manifest(),
            examples=_examples(),
            class_names=class_names,
            output_dir=tmp_path,
            pairs_per_sheet=2,
        )
    assert list(tmp_path.iterdir()) == []


def test_renderer_reports_incomplete_class_name_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="class name authority is incomplete"):
        render_error_contact_sheets(
            manifest=_manifest(),
            examples=_examples(),
            class_names=["too-short"],
            output_dir=tmp_path,
            pairs_per_sheet=2,
        )


def test_renderer_refuses_to_overwrite_any_sheet(tmp_path: Path) -> None:
    existing = tmp_path / "errors-001.png"
    existing.write_bytes(b"sealed")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_error_contact_sheets(
            manifest=_manifest(),
            examples=_examples(),
            class_names=[f"class-{index}" for index in range(196)],
            output_dir=tmp_path,
            pairs_per_sheet=1,
        )
    assert (tmp_path / "errors-000.png").exists() is False
    assert existing.read_bytes() == b"sealed"
