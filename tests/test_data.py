from typing import cast

import pytest

from sfora.data import (
    ImageExample,
    TextExample,
    load_image_retrieval_examples,
    load_imdb_examples,
    mine_group_triplets,
    mine_triplets,
    select_balanced_examples,
    select_labeled_image_examples,
)


def _records() -> list[dict[str, object]]:
    return [
        {"text": "bad acting", "label": 0},
        {"text": "weak plot", "label": 0},
        {"text": "poor pacing", "label": 0},
        {"text": "dull scenes", "label": 0},
        {"text": "great acting", "label": 1},
        {"text": "sharp plot", "label": 1},
        {"text": "strong pacing", "label": 1},
        {"text": "vivid scenes", "label": 1},
    ]


def test_select_balanced_examples_returns_deterministic_per_class_sample() -> None:
    examples = select_balanced_examples(_records(), limit_per_class=2, seed=13)

    assert len(examples) == 4
    assert [example.example_id for example in examples] == [
        "imdb-0-0",
        "imdb-0-3",
        "imdb-1-4",
        "imdb-1-7",
    ]
    assert [example.label for example in examples] == [0, 0, 1, 1]


def test_select_balanced_examples_rejects_missing_classes() -> None:
    with pytest.raises(ValueError, match="at least two labels"):
        select_balanced_examples(_records()[:3], limit_per_class=2)


def test_mine_triplets_pairs_anchor_positive_by_label_and_negative_by_other_label() -> None:
    examples = select_balanced_examples(_records(), limit_per_class=3, seed=0)

    triplets = mine_triplets(examples, max_triplets_per_label=2)

    assert len(triplets) == 4
    for triplet in triplets:
        assert triplet.anchor.label == triplet.positive.label
        assert triplet.anchor.label != triplet.negative.label


def test_mine_group_triplets_uses_groups_instead_of_single_points() -> None:
    examples = select_balanced_examples(_records(), limit_per_class=4, seed=0)

    groups = mine_group_triplets(examples, group_size=2)

    assert len(groups) == 4
    for triplet in groups:
        assert len(triplet.anchor) == 2
        assert len(triplet.positive) == 2
        assert len(triplet.negative) == 2
        assert {example.label for example in triplet.anchor} == {triplet.anchor[0].label}
        assert {example.label for example in triplet.positive} == {triplet.anchor[0].label}
        assert {example.label for example in triplet.negative} != {triplet.anchor[0].label}


def test_mine_group_triplets_drops_label_remainder_deterministically() -> None:
    examples = [
        TextExample(example_id=str(index), text=str(index), label=index % 2) for index in range(6)
    ]

    groups = mine_group_triplets(examples, group_size=2)

    assert len(groups) == 2
    used_ids = {
        example.example_id
        for triplet in groups
        for group in [triplet.anchor, triplet.positive, triplet.negative]
        for example in group
    }
    assert used_ids == {"0", "1", "2", "3"}


def test_load_imdb_examples_accepts_injected_dataset_loader() -> None:
    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        assert name == "stanfordnlp/imdb"
        assert split == "train"
        return _records()

    examples = load_imdb_examples(limit_per_class=2, seed=13, dataset_loader=fake_loader)

    assert [example.text for example in examples] == [
        "bad acting",
        "dull scenes",
        "great acting",
        "vivid scenes",
    ]


def test_select_labeled_image_examples_filters_classes_and_balances_sample() -> None:
    records: list[dict[str, object]] = [
        {"image": f"image-{label}-{index}", "label": label}
        for label in range(4)
        for index in range(3)
    ]

    examples = select_labeled_image_examples(
        records,
        image_key="image",
        label_key="label",
        class_ids=(1, 3),
        limit_per_class=2,
        seed=7,
        id_prefix="cub-train",
    )

    assert examples == [
        ImageExample(example_id="cub-train-1-3", image="image-1-0", label=1),
        ImageExample(example_id="cub-train-1-5", image="image-1-2", label=1),
        ImageExample(example_id="cub-train-3-9", image="image-3-0", label=3),
        ImageExample(example_id="cub-train-3-11", image="image-3-2", label=3),
    ]


def test_load_image_retrieval_examples_uses_metric_learning_class_split() -> None:
    calls: list[tuple[str, str]] = []

    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        calls.append((name, split))
        return [
            {"image": f"{split}-{label}-{index}", "label": label}
            for label in range(200)
            for index in range(2)
        ]

    examples = load_image_retrieval_examples(
        dataset_name="cub",
        split="test",
        limit_per_class=1,
        seed=3,
        dataset_loader=fake_loader,
    )

    assert calls == [
        ("bentrevett/caltech-ucsd-birds-200-2011", "train"),
        ("bentrevett/caltech-ucsd-birds-200-2011", "test"),
    ]
    assert len(examples) == 100
    assert min(example.label for example in examples) == 100
    assert max(example.label for example in examples) == 199


def test_select_labeled_image_examples_can_crop_images_to_bbox() -> None:
    class CropImage:
        def __init__(self, name: str) -> None:
            self.name = name
            self.boxes: list[tuple[float, float, float, float]] = []

        def crop(self, box: tuple[float, float, float, float]) -> "CropImage":
            cropped = CropImage(f"{self.name}-cropped")
            cropped.boxes = [*self.boxes, box]
            return cropped

    records: list[dict[str, object]] = [
        {
            "image": CropImage(f"image-{label}-{index}"),
            "label": label,
            "bbox": [10.0, 20.0, 30.0, 40.0],
        }
        for label in range(2)
        for index in range(2)
    ]

    examples = select_labeled_image_examples(
        records,
        image_key="image",
        label_key="label",
        limit_per_class=1,
        seed=0,
        crop_bbox=True,
    )

    assert examples
    cropped_image = cast(CropImage, examples[0].image)
    assert cropped_image.name.endswith("-cropped")
    assert cropped_image.boxes == [(10.0, 20.0, 30.0, 40.0)]


def test_load_image_retrieval_examples_caps_explicit_test_classes_without_extra_skip() -> None:
    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        return [
            {"image": f"{split}-{label}-{index}", "label": label}
            for label in range(200)
            for index in range(2)
        ]

    examples = load_image_retrieval_examples(
        dataset_name="cub",
        split="test",
        limit_per_class=1,
        max_classes=100,
        seed=3,
        dataset_loader=fake_loader,
    )

    assert len(examples) == 100
    assert min(example.label for example in examples) == 100
    assert max(example.label for example in examples) == 199


def test_load_image_retrieval_examples_can_cap_classes_for_debug_runs() -> None:
    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        return [
            {"image": f"{split}-{label}-{index}", "id": f"{label}_{index}"}
            for label in range(20)
            for index in range(3)
        ]

    examples = load_image_retrieval_examples(
        dataset_name="sop",
        split="train",
        limit_per_class=2,
        max_classes=5,
        seed=0,
        dataset_loader=fake_loader,
    )

    assert len(examples) == 10
    assert sorted({example.label for example in examples}) == [0, 1, 2, 3, 4]


def test_load_image_retrieval_examples_can_keep_all_eligible_classes_without_cap() -> None:
    records: list[dict[str, object]] = [
        {"image": f"image-{label}-{index}", "id": f"{label}_{index}"}
        for label, count in [(0, 2), (1, 4), (2, 5), (3, 1), (4, 4)]
        for index in range(count)
    ]

    examples = select_labeled_image_examples(
        records,
        image_key="image",
        label_key="id",
        min_per_class=4,
        seed=0,
    )

    assert len(examples) == 13
    assert sorted({example.label for example in examples}) == [1, 2, 4]


def test_load_image_retrieval_examples_offsets_sop_test_classes_with_debug_cap() -> None:
    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        return [
            {"image": f"{split}-{label}-{index}", "id": f"{label}_{index}"}
            for label in range(20)
            for index in range(3)
        ]

    examples = load_image_retrieval_examples(
        dataset_name="sop",
        split="test",
        limit_per_class=2,
        max_classes=5,
        seed=0,
        dataset_loader=fake_loader,
    )

    assert len(examples) == 10
    assert sorted({example.label for example in examples}) == [5, 6, 7, 8, 9]


def test_load_image_retrieval_examples_partitions_sop_classes_without_debug_cap() -> None:
    def fake_loader(name: str, split: str) -> list[dict[str, object]]:
        return [
            {"image": f"{split}-{label}-{index}", "id": f"{label}_{index}"}
            for label in range(6)
            for index in range(4)
        ]

    train = load_image_retrieval_examples(
        dataset_name="sop",
        split="train",
        min_per_class=4,
        seed=0,
        dataset_loader=fake_loader,
    )
    test = load_image_retrieval_examples(
        dataset_name="sop",
        split="test",
        min_per_class=4,
        seed=0,
        dataset_loader=fake_loader,
    )

    train_labels = {example.label for example in train}
    test_labels = {example.label for example in test}
    assert train_labels == {0, 1, 2}
    assert test_labels == {3, 4, 5}
    assert train_labels.isdisjoint(test_labels)


def test_select_labeled_image_examples_reads_sop_product_ids() -> None:
    records: list[dict[str, object]] = [
        {"image": f"image-{product}-{index}", "id": f"{product}_{index}"}
        for product in [101, 202, 303]
        for index in range(3)
    ]

    examples = select_labeled_image_examples(
        records,
        image_key="image",
        label_key="id",
        limit_per_class=2,
        max_classes=2,
        skip_classes=1,
        seed=0,
        id_prefix="sop-test",
    )

    assert len(examples) == 4
    assert sorted({example.label for example in examples}) == [202, 303]


def test_select_labeled_image_examples_skips_underfilled_classes() -> None:
    records: list[dict[str, object]] = [{"image": "single", "id": "101_0"}]
    for product in [202, 303]:
        for index in range(3):
            records.append({"image": f"enough-{product}-{index}", "id": f"{product}_{index}"})

    examples = select_labeled_image_examples(
        records,
        image_key="image",
        label_key="id",
        limit_per_class=2,
        max_classes=2,
        seed=0,
    )

    assert len(examples) == 4
    assert sorted({example.label for example in examples}) == [202, 303]
