from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Literal


@dataclass(frozen=True)
class TextExample:
    """A labeled text example used for mining similarity-learning batches."""

    example_id: str
    text: str
    label: int


@dataclass(frozen=True)
class TextTriplet:
    """A standard anchor/positive/negative text triplet."""

    anchor: TextExample
    positive: TextExample
    negative: TextExample


@dataclass(frozen=True)
class TextGroupTriplet:
    """A sfora triplet where each role is a group of examples."""

    anchor: tuple[TextExample, ...]
    positive: tuple[TextExample, ...]
    negative: tuple[TextExample, ...]


@dataclass(frozen=True)
class ImageExample:
    """A labeled image example used for image retrieval benchmarks."""

    example_id: str
    image: object
    label: int


DatasetLoader = Callable[[str, str], Iterable[dict[str, object]]]
ImageDatasetName = Literal["cub", "cars", "sop"]
ClassPartition = Literal["first_half", "second_half"]
IMDB_DATASET_ID = "stanfordnlp/imdb"


@dataclass(frozen=True)
class _ImageDatasetSpec:
    dataset_id: str
    image_key: str
    label_key: str
    train_class_ids: tuple[int, ...] | None
    test_class_ids: tuple[int, ...] | None
    class_split_uses_all_records: bool
    config_name: str | None = None
    source_split_name: str | None = None
    crop_bbox: bool = False


_IMAGE_DATASET_SPECS: dict[ImageDatasetName, _ImageDatasetSpec] = {
    "cub": _ImageDatasetSpec(
        dataset_id="bentrevett/caltech-ucsd-birds-200-2011",
        image_key="image",
        label_key="label",
        train_class_ids=tuple(range(100)),
        test_class_ids=tuple(range(100, 200)),
        class_split_uses_all_records=True,
    ),
    "cars": _ImageDatasetSpec(
        dataset_id="tanganke/stanford_cars",
        image_key="image",
        label_key="label",
        train_class_ids=tuple(range(98)),
        test_class_ids=tuple(range(98, 196)),
        class_split_uses_all_records=True,
    ),
    "sop": _ImageDatasetSpec(
        dataset_id="JamieSJS/stanford-online-products",
        image_key="image",
        label_key="id",
        train_class_ids=None,
        test_class_ids=None,
        class_split_uses_all_records=False,
        config_name="corpus",
        source_split_name="corpus",
    ),
}


def load_imdb_examples(
    *,
    split: str = "train",
    limit_per_class: int = 128,
    seed: int = 0,
    dataset_loader: DatasetLoader | None = None,
) -> list[TextExample]:
    """Load a balanced IMDb sample through Hugging Face datasets."""
    loader = dataset_loader or _load_huggingface_dataset
    records = loader(IMDB_DATASET_ID, split)
    return select_balanced_examples(records, limit_per_class=limit_per_class, seed=seed)


def load_image_retrieval_examples(
    *,
    dataset_name: ImageDatasetName,
    split: Literal["train", "test"],
    limit_per_class: int | None = None,
    min_per_class: int | None = None,
    max_classes: int | None = None,
    seed: int = 0,
    dataset_loader: DatasetLoader | None = None,
) -> list[ImageExample]:
    """Load a standard image metric-learning split through Hugging Face datasets."""
    loader = dataset_loader or _load_huggingface_dataset
    spec = _IMAGE_DATASET_SPECS[dataset_name]
    class_ids = spec.train_class_ids if split == "train" else spec.test_class_ids
    source_splits = (
        ("train", "test")
        if spec.class_split_uses_all_records
        else (spec.source_split_name or split,)
    )
    records: list[dict[str, object]] = []
    for source_split in source_splits:
        if dataset_loader is None and spec.config_name is not None:
            records.extend(
                _load_huggingface_dataset_config(
                    spec.dataset_id,
                    spec.config_name,
                    source_split,
                )
            )
        else:
            records.extend(loader(spec.dataset_id, source_split))
    return select_labeled_image_examples(
        records,
        image_key=spec.image_key,
        label_key=spec.label_key,
        class_ids=class_ids,
        limit_per_class=limit_per_class,
        min_per_class=min_per_class,
        max_classes=max_classes,
        class_partition=(
            None
            if class_ids is not None or max_classes is not None
            else ("first_half" if split == "train" else "second_half")
        ),
        skip_classes=0 if split == "train" or class_ids is not None else (max_classes or 0),
        seed=seed,
        id_prefix=f"{dataset_name}-{split}",
        crop_bbox=spec.crop_bbox,
    )


def select_balanced_examples(
    records: Iterable[dict[str, object]],
    *,
    limit_per_class: int,
    seed: int = 0,
    id_prefix: str = "imdb",
) -> list[TextExample]:
    """Select a deterministic balanced sample from IMDb-shaped records."""
    if limit_per_class < 1:
        raise ValueError("limit_per_class must be at least 1")

    grouped: dict[int, list[TextExample]] = defaultdict(list)
    for index, record in enumerate(records):
        label = _read_int(record, "label")
        text = _read_str(record, "text")
        grouped[label].append(
            TextExample(
                example_id=f"{id_prefix}-{label}-{index}",
                text=text,
                label=label,
            )
        )

    if len(grouped) < 2:
        raise ValueError("balanced selection requires at least two labels")

    rng = Random(seed)
    selected: list[TextExample] = []
    for label in sorted(grouped):
        candidates = grouped[label].copy()
        if len(candidates) < limit_per_class:
            raise ValueError(f"label {label} has fewer than {limit_per_class} examples")
        rng.shuffle(candidates)
        selected.extend(
            sorted(candidates[:limit_per_class], key=lambda example: example.example_id)
        )

    return selected


def select_labeled_image_examples(
    records: Iterable[dict[str, object]],
    *,
    image_key: str,
    label_key: str,
    class_ids: Sequence[int] | None = None,
    limit_per_class: int | None = None,
    min_per_class: int | None = None,
    max_classes: int | None = None,
    class_partition: ClassPartition | None = None,
    skip_classes: int = 0,
    seed: int = 0,
    id_prefix: str = "image",
    crop_bbox: bool = False,
) -> list[ImageExample]:
    """Select a deterministic class-balanced image sample from dataset records."""
    if limit_per_class is not None and limit_per_class < 1:
        raise ValueError("limit_per_class must be at least 1")
    if min_per_class is not None and min_per_class < 1:
        raise ValueError("min_per_class must be at least 1")
    if max_classes is not None and max_classes < 2:
        raise ValueError("max_classes must be at least 2")
    if skip_classes < 0:
        raise ValueError("skip_classes must be non-negative")
    if class_partition is not None and (skip_classes or max_classes is not None):
        raise ValueError("class_partition cannot be combined with skip_classes or max_classes")

    allowed_classes = None if class_ids is None else set(class_ids)
    grouped: dict[int, list[ImageExample]] = defaultdict(list)
    for index, record in enumerate(records):
        label = _read_image_label(record, label_key)
        if allowed_classes is not None and label not in allowed_classes:
            continue
        if image_key not in record:
            raise ValueError(f"record is missing {image_key!r}")
        image = record[image_key]
        if crop_bbox:
            image = _crop_image_to_bbox(image, record.get("bbox"))
        grouped[label].append(
            ImageExample(
                example_id=f"{id_prefix}-{label}-{index}",
                image=image,
                label=label,
            )
        )

    if len(grouped) < 2:
        raise ValueError("image retrieval selection requires at least two labels")

    selected: list[ImageExample] = []
    labels = sorted(grouped)
    minimum_count = min_per_class if min_per_class is not None else limit_per_class
    if minimum_count is not None:
        labels = [label for label in labels if len(grouped[label]) >= minimum_count]
    if class_partition is not None:
        midpoint = len(labels) // 2
        labels = labels[:midpoint] if class_partition == "first_half" else labels[midpoint:]
    if skip_classes:
        labels = labels[skip_classes:]
    if max_classes is not None:
        labels = labels[:max_classes]
    if len(labels) < 2:
        raise ValueError("image retrieval selection requires at least two labels after filtering")
    for label in labels:
        rng = Random(seed)
        candidates = grouped[label].copy()
        rng.shuffle(candidates)
        if limit_per_class is not None:
            candidates = candidates[:limit_per_class]
        selected.extend(sorted(candidates, key=lambda example: _image_example_sort_key(example)))

    return selected


def mine_triplets(
    examples: Sequence[TextExample],
    *,
    max_triplets_per_label: int | None = None,
) -> list[TextTriplet]:
    """Mine deterministic text triplets from a balanced labeled sample."""
    grouped = _group_by_label(examples)
    triplets: list[TextTriplet] = []

    for label in sorted(grouped):
        same_label = grouped[label]
        other_label_examples = _examples_from_other_labels(grouped, label)
        count = len(same_label)
        if max_triplets_per_label is not None:
            count = min(count, max_triplets_per_label)

        for position, anchor in enumerate(same_label[:count]):
            triplets.append(
                TextTriplet(
                    anchor=anchor,
                    positive=same_label[(position + 1) % len(same_label)],
                    negative=other_label_examples[position % len(other_label_examples)],
                )
            )

    return triplets


def mine_group_triplets(
    examples: Sequence[TextExample],
    *,
    group_size: int,
    max_triplets_per_label: int | None = None,
) -> list[TextGroupTriplet]:
    """Mine deterministic group triplets from a balanced labeled sample."""
    if group_size < 1:
        raise ValueError("group_size must be at least 1")

    grouped = _group_by_label(examples)
    grouped_chunks = {
        label: _chunk_label_examples(label_examples, group_size=group_size)
        for label, label_examples in grouped.items()
    }
    triplets: list[TextGroupTriplet] = []

    for label in sorted(grouped_chunks):
        same_label_groups = grouped_chunks[label]
        other_label_groups = _groups_from_other_labels(grouped_chunks, label)
        count = len(same_label_groups)
        if max_triplets_per_label is not None:
            count = min(count, max_triplets_per_label)

        for position, anchor_group in enumerate(same_label_groups[:count]):
            triplets.append(
                TextGroupTriplet(
                    anchor=anchor_group,
                    positive=same_label_groups[(position + 1) % len(same_label_groups)],
                    negative=other_label_groups[position % len(other_label_groups)],
                )
            )

    return triplets


def _load_huggingface_dataset(name: str, split: str) -> Iterable[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to load IMDb: uv sync --group dev --extra research"
        ) from error

    dataset = load_dataset(name, split=split)
    return dataset  # type: ignore[no-any-return]


def _load_huggingface_dataset_config(
    name: str,
    config_name: str,
    split: str,
) -> Iterable[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to load datasets: uv sync --group dev --extra research"
        ) from error

    dataset = load_dataset(name, config_name, split=split)
    return dataset  # type: ignore[no-any-return]


def _group_by_label(examples: Sequence[TextExample]) -> dict[int, list[TextExample]]:
    grouped: dict[int, list[TextExample]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)

    if len(grouped) < 2:
        raise ValueError("triplet mining requires at least two labels")
    if any(len(label_examples) < 2 for label_examples in grouped.values()):
        raise ValueError("triplet mining requires at least two examples per label")

    return {
        label: sorted(values, key=lambda example: example.example_id)
        for label, values in grouped.items()
    }


def _examples_from_other_labels(
    grouped: dict[int, list[TextExample]],
    label: int,
) -> list[TextExample]:
    examples: list[TextExample] = []
    for other_label in sorted(grouped):
        if other_label != label:
            examples.extend(grouped[other_label])
    return examples


def _groups_from_other_labels(
    grouped: dict[int, list[tuple[TextExample, ...]]],
    label: int,
) -> list[tuple[TextExample, ...]]:
    groups: list[tuple[TextExample, ...]] = []
    for other_label in sorted(grouped):
        if other_label != label:
            groups.extend(grouped[other_label])
    return groups


def _chunk_label_examples(
    examples: Sequence[TextExample],
    *,
    group_size: int,
) -> list[tuple[TextExample, ...]]:
    usable_examples = len(examples) - (len(examples) % group_size)
    return [
        tuple(examples[start : start + group_size])
        for start in range(0, usable_examples, group_size)
    ]


def _read_int(record: dict[str, object], key: str) -> int:
    value = _read_value(record, key)
    if not isinstance(value, int):
        raise ValueError(f"record {key!r} must be an integer")
    return value


def _read_image_label(record: dict[str, object], key: str) -> int:
    value = _read_value(record, key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw_label = value.split("_", 1)[0]
        if raw_label.isdigit():
            return int(raw_label)
    raise ValueError(f"record {key!r} must be an integer or product id string")


def _image_example_sort_key(example: ImageExample) -> tuple[int, str]:
    try:
        return (int(example.example_id.rsplit("-", 1)[1]), example.example_id)
    except ValueError:
        return (0, example.example_id)


def _crop_image_to_bbox(image: object, bbox: object) -> object:
    if bbox is None or not hasattr(image, "crop"):
        return image
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return image
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return image.crop((x0, y0, x1, y1))


def _read_str(record: dict[str, object], key: str) -> str:
    value = _read_value(record, key)
    if not isinstance(value, str):
        raise ValueError(f"record {key!r} must be a string")
    return value


def _read_value(record: dict[str, object], key: str) -> Any:
    if key not in record:
        raise ValueError(f"record is missing {key!r}")
    return record[key]
