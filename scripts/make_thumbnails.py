"""Extract per-point CUB thumbnails + species names for the website's viz.

The separation visualisation plots 360 sampled test images (8 classes × 45) in
several embedding spaces. This script turns each plotted point into a real
clickable bird photo and gives the legend true species names.

It consumes the `indices` and `classIds` a projection JSON already recorded
(`scripts/make_projection.py`), so image k lines up with point k in *every*
panel — no re-sampling, no ordering drift between this script's environment and
wherever the projections were generated.

Run it where the CUB dataset is available (the `research` extra installed):

    uv run python scripts/make_thumbnails.py \
        --proj site/src/data/proj_herd.json \
        --thumb-dir site/public/thumbs/cub \
        --meta-out site/src/data/proj_meta.json \
        --size 140
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _species_names(dataset_id: str) -> list[str] | None:
    """Best-effort ClassLabel names from the raw HF dataset; None if unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="test")
        feat = ds.features.get("label")
        names = getattr(feat, "names", None)
        return list(names) if names else None
    except Exception:
        return None


def _pretty(name: str) -> str:
    """`013.Bobolink` / `Black_footed_Albatross` -> `Bobolink` / `Black Footed Albatross`."""
    if "." in name and name.split(".", 1)[0].isdigit():
        name = name.split(".", 1)[1]
    return name.replace("_", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proj", required=True, help="projection JSON with indices/classIds")
    ap.add_argument("--thumb-dir", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--dataset", default="cub")
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--check-npz", help="embedding npz whose labels must match example order")
    args = ap.parse_args()

    try:
        from sfora.data import _IMAGE_DATASET_SPECS, load_image_retrieval_examples
    except ModuleNotFoundError:  # pre-rename checkouts still ship `group_learning`
        from group_learning.data import (  # type: ignore[import-not-found,no-redef]
            _IMAGE_DATASET_SPECS,
            load_image_retrieval_examples,
        )

    proj = json.loads(Path(args.proj).read_text())
    indices: list[int] = proj["indices"]
    class_ids: list[int] = proj["classIds"]

    # Same deterministic order the embeddings were exported in (shuffle=False loader).
    examples = load_image_retrieval_examples(dataset_name=args.dataset, split="test")

    # Safety: the export ran in a possibly different environment. Prove the example
    # order matches the embedding rows before we trust `indices` to map to images.
    if args.check_npz:
        import numpy as np

        ref = np.load(args.check_npz)["labels"]
        got = np.array([e.label for e in examples])
        if got.shape != ref.shape or not np.array_equal(got, ref):
            raise SystemExit(
                f"ALIGNMENT MISMATCH: examples ({got.shape}) vs npz labels ({ref.shape}). "
                "Thumbnails would not correspond to points — aborting."
            )
        print(f"alignment OK: {len(got)} examples match {args.check_npz} labels")

    spec = _IMAGE_DATASET_SPECS[args.dataset]
    raw_names = _species_names(spec.dataset_id)
    class_names = [
        _pretty(raw_names[cid]) if raw_names and cid < len(raw_names) else f"class {cid}"
        for cid in class_ids
    ]

    thumb_dir = Path(args.thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[str] = []
    for k, orig in enumerate(indices):
        img = examples[orig].image  # PIL image from the HF dataset
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((args.size, args.size))
        name = f"{k:03d}.webp"
        thumb.save(thumb_dir / name, "WEBP", quality=80, method=6)
        thumbs.append(name)

    meta = {
        "dataset": args.dataset,
        "size": args.size,
        "classIds": class_ids,
        "classNames": class_names,
        "thumbBase": f"thumbs/{args.dataset}",
        "thumbs": thumbs,  # thumbs[k] is the file for point k in every panel
    }
    Path(args.meta_out).write_text(json.dumps(meta))
    print(
        f"{len(thumbs)} thumbnails -> {thumb_dir}\n"
        f"classes: {list(zip(class_ids, class_names, strict=True))}\n"
        f"meta -> {args.meta_out}"
    )


if __name__ == "__main__":
    main()
