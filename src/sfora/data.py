import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
ImageDatasetName = Literal["cub", "cars", "sop", "inshop", "inat2018"]
ImageDatasetSplit = Literal["train", "test", "query", "gallery"]
ImageRetrievalProtocol = Literal["self", "query_gallery"]
ClassPartition = Literal["first_half", "second_half"]
IMDB_DATASET_ID = "stanfordnlp/imdb"


@dataclass(frozen=True)
class ImageRetrievalBundle:
    """Training and evaluation splits for one image-retrieval protocol."""

    train: list[ImageExample]
    query: list[ImageExample]
    gallery: list[ImageExample] | None
    protocol: ImageRetrievalProtocol
    protocol_name: str


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
    split: ImageDatasetSplit,
    limit_per_class: int | None = None,
    min_per_class: int | None = None,
    max_classes: int | None = None,
    seed: int = 0,
    dataset_loader: DatasetLoader | None = None,
    dataset_root: Path | None = None,
) -> list[ImageExample]:
    """Load a standard image metric-learning split through Hugging Face datasets."""
    if dataset_name in {"inshop", "inat2018"}:
        bundle = load_image_retrieval_bundle(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            limit_per_class=limit_per_class,
            train_min_per_class=min_per_class if split == "train" else None,
            evaluation_min_per_class=min_per_class if split != "train" else None,
            max_classes=max_classes,
            seed=seed,
        )
        if split == "train":
            return bundle.train
        if split in {"test", "query"}:
            return bundle.query
        if bundle.gallery is None:
            raise ValueError(f"dataset {dataset_name!r} has no gallery split")
        return bundle.gallery
    if split not in {"train", "test"}:
        raise ValueError(f"dataset {dataset_name!r} supports only train/test splits")
    loader = dataset_loader or _load_huggingface_dataset
    spec = _IMAGE_DATASET_SPECS[dataset_name]
    class_ids = spec.train_class_ids if split == "train" else spec.test_class_ids
    source_splits = (
        ("train", "test")
        if spec.class_split_uses_all_records
        else (spec.source_split_name or split,)
    )
    records: list[dict[str, object]] = []
    if dataset_name == "sop" and dataset_loader is None:
        records.extend(_load_original_sop_records(split))
    else:
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
    if dataset_name == "sop" and dataset_loader is None:
        official_products = _sop_official_product_ids(split)
        records = [
            record
            for record in records
            if str(record[spec.label_key]).rsplit("_", 1)[0] in official_products
        ]
        observed_products = {
            str(record[spec.label_key]).rsplit("_", 1)[0] for record in records
        }
        missing = official_products - observed_products
        if missing:
            raise ValueError(
                f"SOP corpus is missing {len(missing)} products from the official {split} split"
            )
        expected_images = 59_551 if split == "train" else 60_502
        if len(records) != expected_images:
            raise ValueError(
                f"official SOP {split} has {len(records)} images; expected {expected_images}"
            )
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
            if class_ids is not None
            or max_classes is not None
            or (dataset_name == "sop" and dataset_loader is None)
            else ("first_half" if split == "train" else "second_half")
        ),
        skip_classes=0 if split == "train" or class_ids is not None else (max_classes or 0),
        seed=seed,
        id_prefix=f"{dataset_name}-{split}",
        crop_bbox=spec.crop_bbox,
    )


def _sop_official_product_ids(split: str) -> set[str]:
    """Resolve the official SOP split from its digest-pinned metadata file."""
    import hashlib

    from huggingface_hub import hf_hub_download

    if split not in {"train", "test"}:
        raise ValueError(f"unknown SOP split: {split!r}")
    filename = f"Stanford_Online_Products/Ebay_{split}.txt"
    path = Path(
        hf_hub_download(
            repo_id="pawlo2013/StanfordOnlineProducts",
            repo_type="dataset",
            filename=filename,
            revision="11cf4cdcd89ac209f05277fcabebf10f47d3467d",
        )
    )
    expected = {
        "train": "77abb1e82af49f2f1f272dc6bd0b8480904f58742091764f9511b152cad1824e",
        "test": "63d559dc1ca1e03671224c58a26b18985bbff22ae20344d41dabcfe3873cc276",
    }[split]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"official SOP {split} metadata digest mismatch: {observed}")
    expected_count = 11_318 if split == "train" else 11_316
    return _parse_sop_metadata(path, expected_count=expected_count)


def _load_original_sop_records(split: str) -> Iterable[dict[str, object]]:
    """Load and materialize digest-pinned original-resolution official SOP images."""
    try:
        from datasets import Image, load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to load original SOP images"
        ) from error

    revision = "24a1b9b8ec6c0b1fc4dd324f24b2d829413a6c69"
    dataset = load_dataset(
        "nyris/stanford-online-products-v1",
        split=split,
        revision=revision,
    ).cast_column("image", Image(decode=False))
    expected = 59_551 if split == "train" else 60_502
    if len(dataset) != expected:
        raise ValueError(
            f"pinned original SOP {split} has {len(dataset)} rows; expected {expected}"
        )

    cache_root = Path(
        os.environ.get(
            "SFORA_SOP_ORIGINAL_CACHE",
            Path.home() / ".cache" / "sfora" / "sop-original" / revision,
        )
    )
    for index, record in enumerate(dataset):
        item_id = str(record["item_id"])
        relative = Path(str(record["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SOP image path at row {index}: {relative}")
        if relative.stem.rsplit("_", 1)[0] != item_id:
            raise ValueError(
                f"SOP item/path mismatch at row {index}: {item_id} versus {relative}"
            )
        target = cache_root / relative
        if not target.is_file():
            raw_image = record["image"]
            if not isinstance(raw_image, dict) or not isinstance(raw_image.get("bytes"), bytes):
                raise ValueError(f"SOP row {index} does not contain embedded image bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.partial")
            temporary.write_bytes(raw_image["bytes"])
            temporary.replace(target)
        yield {"image": target, "id": item_id}


def _parse_sop_metadata(path: Path, *, expected_count: int) -> set[str]:
    products: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            raise ValueError(f"invalid SOP metadata row {line_number}: {line!r}")
        products.add(Path(fields[3]).stem.rsplit("_", 1)[0])
    if len(products) != expected_count:
        raise ValueError(
            f"SOP metadata has {len(products)} products; expected {expected_count}"
        )
    return products


def load_image_retrieval_bundle(
    *,
    dataset_name: ImageDatasetName,
    dataset_root: Path | None = None,
    limit_per_class: int | None = None,
    train_min_per_class: int | None = None,
    evaluation_min_per_class: int | None = None,
    max_classes: int | None = None,
    seed: int = 0,
    dataset_loader: DatasetLoader | None = None,
) -> ImageRetrievalBundle:
    """Load all splits needed to train and evaluate an image retrieval dataset."""
    if dataset_name == "inshop":
        root = _required_dataset_root(dataset_name, dataset_root)
        return _load_inshop_bundle(
            root,
            limit_per_class=limit_per_class,
            train_min_per_class=train_min_per_class,
            evaluation_min_per_class=evaluation_min_per_class,
            max_classes=max_classes,
            seed=seed,
        )
    if dataset_name == "inat2018":
        root = _required_dataset_root(dataset_name, dataset_root)
        return _load_inat2018_bundle(
            root,
            limit_per_class=limit_per_class,
            train_min_per_class=train_min_per_class,
            evaluation_min_per_class=evaluation_min_per_class,
            max_classes=max_classes,
            seed=seed,
        )
    train = load_image_retrieval_examples(
        dataset_name=dataset_name,
        split="train",
        limit_per_class=limit_per_class,
        min_per_class=train_min_per_class,
        max_classes=max_classes,
        seed=seed,
        dataset_loader=dataset_loader,
    )
    test = load_image_retrieval_examples(
        dataset_name=dataset_name,
        split="test",
        limit_per_class=limit_per_class,
        min_per_class=evaluation_min_per_class,
        max_classes=max_classes,
        seed=seed,
        dataset_loader=dataset_loader,
    )
    return ImageRetrievalBundle(
        train=train,
        query=test,
        gallery=None,
        protocol="self",
        protocol_name=f"{dataset_name}-standard-zero-shot",
    )


def materialize_image(image: object) -> object:
    """Open a filesystem-backed image lazily; pass already-decoded images through."""
    if not isinstance(image, Path):
        return image
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Install Pillow to load filesystem-backed image datasets") from error
    with Image.open(image) as opened:
        return opened.convert("RGB").copy()


def _required_dataset_root(dataset_name: str, dataset_root: Path | None) -> Path:
    if dataset_root is None:
        raise ValueError(f"dataset {dataset_name!r} requires dataset_root")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist or is not a directory: {root}")
    return root


def _load_inshop_bundle(
    root: Path,
    *,
    limit_per_class: int | None,
    train_min_per_class: int | None,
    evaluation_min_per_class: int | None,
    max_classes: int | None,
    seed: int,
) -> ImageRetrievalBundle:
    partition_path = root / "Eval" / "list_eval_partition.txt"
    if not partition_path.is_file():
        raise ValueError(f"missing In-Shop partition file: {partition_path}")
    lines = [line.strip() for line in partition_path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 3:
        raise ValueError(f"invalid In-Shop partition file: {partition_path}")
    try:
        declared_count = int(lines[0])
    except ValueError as error:
        raise ValueError("In-Shop partition first line must be an image count") from error
    rows: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(lines[2:], start=3):
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid In-Shop partition row {line_number}: {line!r}")
        image_name, item_id, status = fields
        if status not in {"train", "query", "gallery"}:
            raise ValueError(f"unknown In-Shop evaluation status {status!r} on row {line_number}")
        rows.append((image_name, item_id, status))
    if len(rows) != declared_count:
        raise ValueError(
            f"In-Shop partition declares {declared_count} images but contains {len(rows)} rows"
        )
    label_by_item = {item_id: index for index, item_id in enumerate(sorted({r[1] for r in rows}))}
    by_status: dict[str, list[ImageExample]] = {"train": [], "query": [], "gallery": []}
    items_by_status: dict[str, set[str]] = {"train": set(), "query": set(), "gallery": set()}
    for image_name, item_id, status in rows:
        image_path = root / "Img" / image_name
        if not image_path.is_file():
            raise ValueError(f"In-Shop partition references missing image: {image_path}")
        items_by_status[status].add(item_id)
        by_status[status].append(
            ImageExample(
                example_id=f"inshop-{status}-{image_name}",
                image=image_path,
                label=label_by_item[item_id],
            )
        )
    evaluation_items = items_by_status["query"] | items_by_status["gallery"]
    overlap = items_by_status["train"] & evaluation_items
    if overlap:
        raise ValueError(f"In-Shop train/evaluation identities overlap: {sorted(overlap)[:3]}")
    train = _select_prebuilt_image_examples(
        by_status["train"],
        limit_per_class=limit_per_class,
        min_per_class=train_min_per_class,
        max_classes=max_classes,
        seed=seed,
    )
    query, gallery = _select_paired_query_gallery_examples(
        by_status["query"],
        by_status["gallery"],
        limit_per_class=limit_per_class,
        min_per_class=evaluation_min_per_class,
        max_classes=max_classes,
        seed=seed,
    )
    return ImageRetrievalBundle(
        train=train,
        query=query,
        gallery=gallery,
        protocol="query_gallery",
        protocol_name="deepfashion-inshop-official",
    )


def _load_inat2018_bundle(
    root: Path,
    *,
    limit_per_class: int | None,
    train_min_per_class: int | None,
    evaluation_min_per_class: int | None,
    max_classes: int | None,
    seed: int,
) -> ImageRetrievalBundle:
    train_all = _load_coco_image_examples(root, root / "train2018.json", split="train")
    validation_all = _load_coco_image_examples(root, root / "val2018.json", split="validation")
    train_labels = {example.label for example in train_all}
    validation_labels = {example.label for example in validation_all}
    eligible = sorted(train_labels & validation_labels)
    if len(eligible) < 4:
        raise ValueError(
            "inat2018-zero-shot-species-v1 requires at least four species present "
            "in both train and validation annotations"
        )
    midpoint = len(eligible) // 2
    optimization_labels = eligible[:midpoint]
    evaluation_labels = eligible[midpoint:]
    if max_classes is not None:
        if max_classes < 2:
            raise ValueError("max_classes must be at least 2")
        optimization_labels = optimization_labels[:max_classes]
        evaluation_labels = evaluation_labels[:max_classes]
    train = _select_prebuilt_image_examples(
        [example for example in train_all if example.label in set(optimization_labels)],
        limit_per_class=limit_per_class,
        min_per_class=train_min_per_class,
        max_classes=None,
        seed=seed,
    )
    query, gallery = _select_paired_query_gallery_examples(
        [example for example in validation_all if example.label in set(evaluation_labels)],
        [example for example in train_all if example.label in set(evaluation_labels)],
        limit_per_class=limit_per_class,
        min_per_class=evaluation_min_per_class,
        max_classes=None,
        seed=seed,
    )
    if {example.label for example in train} & {example.label for example in query}:
        raise ValueError("iNaturalist optimization and evaluation species overlap")
    return ImageRetrievalBundle(
        train=train,
        query=query,
        gallery=gallery,
        protocol="query_gallery",
        protocol_name="inat2018-zero-shot-species-v1",
    )


def _load_coco_image_examples(
    root: Path,
    annotation_path: Path,
    *,
    split: str,
) -> list[ImageExample]:
    if not annotation_path.is_file():
        raise ValueError(f"missing iNaturalist annotation file: {annotation_path}")
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid iNaturalist annotation file: {annotation_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid iNaturalist annotation object: {annotation_path}")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(
            f"iNaturalist annotations need images and annotations lists: {annotation_path}"
        )
    paths_by_id: dict[int, Path] = {}
    for raw_image in images:
        if not isinstance(raw_image, dict):
            raise ValueError(f"invalid image record in {annotation_path}")
        image_id = int(raw_image["id"])
        image_path = root / str(raw_image["file_name"])
        if not image_path.is_file():
            raise ValueError(f"iNaturalist annotation references missing image: {image_path}")
        paths_by_id[image_id] = image_path
    examples: list[ImageExample] = []
    seen_images: set[int] = set()
    for raw_annotation in annotations:
        if not isinstance(raw_annotation, dict):
            raise ValueError(f"invalid annotation record in {annotation_path}")
        image_id = int(raw_annotation["image_id"])
        if image_id in seen_images:
            raise ValueError(f"iNaturalist image {image_id} has multiple category annotations")
        if image_id not in paths_by_id:
            raise ValueError(f"iNaturalist annotation references unknown image id {image_id}")
        seen_images.add(image_id)
        examples.append(
            ImageExample(
                example_id=f"inat2018-{split}-{image_id}",
                image=paths_by_id[image_id],
                label=int(raw_annotation["category_id"]),
            )
        )
    return sorted(examples, key=_image_example_sort_key)


def _select_prebuilt_image_examples(
    examples: Sequence[ImageExample],
    *,
    limit_per_class: int | None,
    min_per_class: int | None,
    max_classes: int | None,
    seed: int,
) -> list[ImageExample]:
    grouped: dict[int, list[ImageExample]] = defaultdict(list)
    for example in examples:
        grouped[int(example.label)].append(example)
    labels = sorted(grouped)
    if min_per_class is not None:
        if min_per_class < 1:
            raise ValueError("min_per_class must be at least 1")
        labels = [label for label in labels if len(grouped[label]) >= min_per_class]
    if max_classes is not None:
        if max_classes < 2:
            raise ValueError("max_classes must be at least 2")
        labels = labels[:max_classes]
    if len(labels) < 2:
        raise ValueError("image retrieval selection requires at least two labels after filtering")
    selected: list[ImageExample] = []
    for label in labels:
        candidates = grouped[label].copy()
        Random(seed).shuffle(candidates)
        if limit_per_class is not None:
            if limit_per_class < 1:
                raise ValueError("limit_per_class must be at least 1")
            candidates = candidates[:limit_per_class]
        selected.extend(sorted(candidates, key=_image_example_sort_key))
    return selected


def _select_paired_query_gallery_examples(
    query_examples: Sequence[ImageExample],
    gallery_examples: Sequence[ImageExample],
    *,
    limit_per_class: int | None,
    min_per_class: int | None,
    max_classes: int | None,
    seed: int,
) -> tuple[list[ImageExample], list[ImageExample]]:
    query_by_label: dict[int, list[ImageExample]] = defaultdict(list)
    gallery_by_label: dict[int, list[ImageExample]] = defaultdict(list)
    for example in query_examples:
        query_by_label[int(example.label)].append(example)
    for example in gallery_examples:
        gallery_by_label[int(example.label)].append(example)
    labels = sorted(set(query_by_label) & set(gallery_by_label))
    if min_per_class is not None:
        if min_per_class < 1:
            raise ValueError("evaluation_min_per_class must be at least 1")
        labels = [
            label
            for label in labels
            if len(query_by_label[label]) >= min_per_class
            and len(gallery_by_label[label]) >= min_per_class
        ]
    if max_classes is not None:
        if max_classes < 2:
            raise ValueError("max_classes must be at least 2")
        labels = labels[:max_classes]
    if len(labels) < 2:
        raise ValueError("query/gallery selection requires at least two shared labels")
    selected_query: list[ImageExample] = []
    selected_gallery: list[ImageExample] = []
    for label in labels:
        query_candidates = query_by_label[label].copy()
        gallery_candidates = gallery_by_label[label].copy()
        Random(seed).shuffle(query_candidates)
        Random(seed).shuffle(gallery_candidates)
        if limit_per_class is not None:
            if limit_per_class < 1:
                raise ValueError("limit_per_class must be at least 1")
            query_candidates = query_candidates[:limit_per_class]
            gallery_candidates = gallery_candidates[:limit_per_class]
        selected_query.extend(sorted(query_candidates, key=_image_example_sort_key))
        selected_gallery.extend(sorted(gallery_candidates, key=_image_example_sort_key))
    return selected_query, selected_gallery


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
        # Need at least two same-label groups (otherwise the positive wraps back to
        # the anchor group — a degenerate self-positive) and at least one other-label
        # group to draw a negative from.
        if len(same_label_groups) < 2 or not other_label_groups:
            continue
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
