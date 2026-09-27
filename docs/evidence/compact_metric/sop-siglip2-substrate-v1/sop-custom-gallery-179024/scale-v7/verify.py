#!/usr/bin/env python3
"""Source-bound production custom-gallery parity on 64 SOP TRAIN images."""

import hashlib
import inspect
import json
import resource
import time
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sfora import (
    cutile_int8,
    joint_relational_compaction,
    siglip2_compact_serving,
    sop_compact_training,
)
from sfora.siglip2_compact_serving import Siglip2CompactIndex

RUN = Path("/home/riomus/runs/sfora-sop-custom-gallery-scale-179024-v7")
TRAINING = Path("/home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1")
ARCHIVE = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
DATA = Path("/home/riomus/datasets/Stanford_Online_Products")
MODEL = Path(
    "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
)
LIBRARY = Path("/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so")
SOURCE_HASHES = {
    "siglip2_compact_serving": "73ad606c78ad22e6b377627e4e7c8fe07d7dfaafd07a029c40df2982221a7fd1",
    "joint_relational_compaction": (
        "4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67"
    ),
    "cutile_int8": "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409",
    "sop_compact_training": "12ccd15c3b943fa519ba37f8d09870b105939872d9279541fc0ab42f20a7ead9",
}
ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
TRAINING_SHA = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
LIBRARY_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_paths() -> list[Path]:
    with np.load(ARCHIVE, allow_pickle=False) as source:
        relatives = np.asarray(source["train_relative_paths"]).astype(str)[:64]
    paths = []
    for relative in relatives:
        part = PurePosixPath(str(relative))
        if part.is_absolute() or ".." in part.parts or not part.parts:
            raise ValueError("SOP custom gallery path differs")
        path = DATA.joinpath(*part.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP custom gallery image missing")
        paths.append(path)
    return paths


def main() -> None:
    output = RUN / "receipt.json"
    if output.exists() or not torch.cuda.is_available():
        raise ValueError("SOP custom gallery invocation differs")
    imported = {
        name: sha(Path(inspect.getfile(module)))
        for name, module in (
            ("siglip2_compact_serving", siglip2_compact_serving),
            ("joint_relational_compaction", joint_relational_compaction),
            ("cutile_int8", cutile_int8),
            ("sop_compact_training", sop_compact_training),
        )
    }
    if (
        imported != SOURCE_HASHES
        or sha(ARCHIVE) != ARCHIVE_SHA
        or sha(LIBRARY) != LIBRARY_SHA
        or sha(TRAINING / "receipt.json") != TRAINING_SHA
    ):
        raise ValueError("SOP custom gallery source differs")
    paths = image_paths()
    hashes = [sha(path) for path in paths]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    opened_gallery = []
    original_open = cutile_int8.CutilePackedInt8Gallery.open_packed

    def capture_gallery(library: Path, packed: object):
        opened_gallery.append(packed)
        return original_open(library, packed)

    started = time.perf_counter()
    with patch.object(
        cutile_int8.CutilePackedInt8Gallery, "open_packed", staticmethod(capture_gallery)
    ):
        index = Siglip2CompactIndex.from_artifacts(
            model_snapshot=MODEL,
            training_receipt=TRAINING / "receipt.json",
            training_checkpoint=TRAINING / "checkpoint.pt",
            train_embeddings=TRAINING / "train_embeddings.npy",
            native_library=LIBRARY,
            expected_receipt_sha256=TRAINING_SHA,
            precision="fp16_native",
            custom_gallery_image_paths=paths[:32],
        )
    with index:
        torch.cuda.synchronize()
        build_wall = time.perf_counter() - started
        if index.encoder is None or index.gallery is None:
            raise ValueError("SOP custom gallery did not load")
        with ExitStack() as stack:
            gallery_images = [stack.enter_context(Image.open(path)) for path in paths[:32]]
            query_images = [stack.enter_context(Image.open(path)) for path in paths[32:]]
            gallery = index.encoder.encode_images(gallery_images)
            queries = index.encoder.encode_images(query_images)
            if (
                len(opened_gallery) != 1
                or not torch.equal(opened_gallery[0].codes, gallery.codes)
                or not torch.equal(opened_gallery[0].inverse_norms, gallery.inverse_norms)
            ):
                raise ValueError("SOP custom gallery packed build differs")
            native_ordinals, native_scores = index.gallery.search_packed(queries)
            search_started = time.perf_counter()
            public_ordinals, public_scores = index.search_images(query_images)
            torch.cuda.synchronize()
            search_wall = time.perf_counter() - search_started
        if not (
            np.array_equal(native_ordinals, public_ordinals)
            and np.array_equal(native_scores, public_scores)
        ):
            raise ValueError("SOP custom gallery public/native result differs")
    score = (
        (queries.codes.float().cuda() @ gallery.codes.float().cuda().T)
        * queries.inverse_norms.float().cuda()[:, None]
        * gallery.inverse_norms.float().cuda()[None, :]
    )
    expected = torch.argsort(score, dim=1, descending=True, stable=True)[:, :10]
    expected_scores = score.gather(1, expected).cpu().numpy()
    max_error = float(np.max(np.abs(public_scores - expected_scores)))
    if (
        not np.array_equal(public_ordinals, expected.cpu().numpy())
        or not np.isfinite(public_scores).all()
        or max_error > 1e-5
    ):
        raise ValueError("SOP custom gallery packed oracle differs")
    result = {
        "schema": "sfora-sop-custom-image-gallery-public-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "imported_sha256": imported,
        "source_archive_sha256": ARCHIVE_SHA,
        "training_receipt_sha256": TRAINING_SHA,
        "checkpoint_sha256": sha(TRAINING / "checkpoint.pt"),
        "native_library_sha256": LIBRARY_SHA,
        "image_sha256": hashes,
        "gallery_rows": 32,
        "query_rows": 32,
        "gallery_wire_bytes_per_row": gallery.bytes_per_vector,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gallery_packed_sha256": hashlib.sha256(
            gallery.codes.numpy().tobytes() + gallery.inverse_norms.numpy().tobytes()
        ).hexdigest(),
        "query_packed_sha256": hashlib.sha256(
            queries.codes.numpy().tobytes() + queries.inverse_norms.numpy().tobytes()
        ).hexdigest(),
        "public_top10_ordinals": public_ordinals.tolist(),
        "public_top10_scores": public_scores.tolist(),
        "public_top10_exact": True,
        "max_score_abs_delta": max_error,
        "gallery_build_wall_seconds": build_wall,
        "one_batch32_search_wall_seconds": search_wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "public_top10_exact",
                    "gallery_build_wall_seconds",
                    "one_batch32_search_wall_seconds",
                    "max_score_abs_delta",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
