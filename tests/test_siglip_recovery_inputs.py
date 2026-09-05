"""Image decoding must follow metadata authentication and optimization selection."""

import hashlib
import io
import json

import pytest
from PIL import Image

from sfora.siglip_recovery_inputs import load_optimization_images


class _ColumnOnlyDataset:
    """Model HF column access separately from record access (which decodes images)."""

    def __init__(self, labels, offset, allowed):
        self.labels, self.offset, self.allowed = labels, offset, allowed
        self.read_rows = []

    def __getitem__(self, key):
        if key == "label":
            return self.labels
        if type(key) is not int:
            raise AssertionError("only label-column or selected row access is allowed")
        global_index = self.offset + key
        if global_index not in self.allowed:
            raise AssertionError(f"unselected pixel access: {global_index}")
        self.read_rows.append(global_index)
        return {
            "label": self.labels[key],
            "image": Image.new("RGB", (2, 3), (global_index % 256, 5, 6)),
        }


def _fixtures():
    labels = (
        [i % 49 for i in range(3963)]
        + [49 + i % 33 for i in range(2746)]
        + [82 + i % 16 for i in range(1345)]
        + [98 + i % 98 for i in range(8131)]
    )
    rows = sorted(
        ({"example_id": f"cars-train-{y}-{i}", "label": y} for i, y in enumerate(labels) if y < 98),
        key=lambda x: x["example_id"],
    )
    raw = (json.dumps({"examples": rows}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    selected_ids = [row["example_id"] for row in rows if row["label"] < 49][:128]
    selected_indices = {int(name.rsplit("-", 1)[1]) for name in selected_ids}
    # Deliberately split inside the optimization population: global IDs must not reset.
    datasets = {
        "train": _ColumnOnlyDataset(labels[:1000], 0, selected_indices),
        "test": _ColumnOnlyDataset(labels[1000:], 1000, selected_indices),
    }

    def loader(name, *, split, revision):
        assert name == "tanganke/stanford_cars"
        assert revision == "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
        return datasets[split]

    return datasets, loader, digest, selected_ids


def test_metadata_precedes_decode_and_only_sorted_optimization_prefix_is_read():
    datasets, loader, digest, selected_ids = _fixtures()
    examples = load_optimization_images(limit=128, dataset_loader=loader, expected_manifest=digest)
    assert [e.example_id for e in examples] == selected_ids
    assert len(examples) == 128 and all(0 <= e.label < 49 for e in examples)
    observed = datasets["train"].read_rows + datasets["test"].read_rows
    assert sorted(observed) == sorted(int(name.rsplit("-", 1)[1]) for name in selected_ids)
    assert all(e.image.getpixel((0, 0))[1:] == (5, 6) for e in examples)


def test_real_hf_image_feature_never_decodes_corrupt_unselected_pixels():
    from datasets import Dataset
    from datasets import Image as ImageFeature

    datasets, _, digest, selected_ids = _fixtures()
    selected = {int(name.rsplit("-", 1)[1]) for name in selected_ids}
    png = io.BytesIO()
    Image.new("RGB", (2, 3), (7, 8, 9)).save(png, format="PNG")
    real = {}
    for split, source in datasets.items():
        images = [
            {
                "bytes": png.getvalue() if source.offset + i in selected else b"not-an-image",
                "path": None,
            }
            for i in range(len(source.labels))
        ]
        real[split] = Dataset.from_dict({"label": source.labels, "image": images}).cast_column(
            "image", ImageFeature()
        )
    examples = load_optimization_images(
        limit=128,
        dataset_loader=lambda name, split, revision: real[split],
        expected_manifest=digest,
    )
    assert [e.example_id for e in examples] == selected_ids
    assert all(e.image.getpixel((0, 0)) == (7, 8, 9) for e in examples)


@pytest.mark.parametrize("mutation", ["digest", "label", "missing-row", "bool-label"])
def test_bad_metadata_is_rejected_before_any_pixel(mutation):
    datasets, loader, digest, _ = _fixtures()
    if mutation == "digest":
        digest = "0" * 64
    elif mutation == "label":
        datasets["test"].labels[-1] = 999
    elif mutation == "missing-row":
        datasets["test"].labels.pop()
    else:
        datasets["train"].labels[0] = False
    with pytest.raises(ValueError):
        load_optimization_images(limit=128, dataset_loader=loader, expected_manifest=digest)
    assert datasets["train"].read_rows == datasets["test"].read_rows == []
