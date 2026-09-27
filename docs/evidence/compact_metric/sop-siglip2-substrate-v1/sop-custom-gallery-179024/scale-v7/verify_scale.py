#!/usr/bin/env python3
"""Source-bound 59,551-row custom image-gallery build and exact search check."""

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
from verify import (
    ARCHIVE,
    ARCHIVE_SHA,
    DATA,
    LIBRARY,
    LIBRARY_SHA,
    MODEL,
    SOURCE_HASHES,
    TRAINING,
    TRAINING_SHA,
    sha,
)

from sfora import (
    cutile_int8,
    joint_relational_compaction,
    siglip2_compact_serving,
    sop_compact_training,
)
from sfora.siglip2_compact_serving import Siglip2CompactIndex

RUN = Path("/home/riomus/runs/sfora-sop-custom-gallery-scale-179024-v7")
MANIFEST = Path("/home/riomus/runs/sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin")
MANIFEST_SHA = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
VERIFY_SHA = "922c0c28ab81cec8ffb2f26056396c3d9001e49da2708af156259ed734c83b7c"


def paths_from_relatives(relatives: np.ndarray) -> list[Path]:
    paths = []
    for relative in relatives.astype(str):
        part = PurePosixPath(str(relative))
        if part.is_absolute() or ".." in part.parts or not part.parts:
            raise ValueError("SOP scale gallery path differs")
        path = DATA.joinpath(*part.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP scale gallery image missing")
        paths.append(path)
    return paths


def main() -> None:
    output = RUN / "receipt.json"
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
        output.exists()
        or not torch.cuda.is_available()
        or imported != SOURCE_HASHES
        or sha(Path(__file__).with_name("verify.py")) != VERIFY_SHA
        or sha(ARCHIVE) != ARCHIVE_SHA
        or sha(MANIFEST) != MANIFEST_SHA
        or sha(LIBRARY) != LIBRARY_SHA
        or sha(TRAINING / "receipt.json") != TRAINING_SHA
    ):
        raise ValueError("SOP scale gallery authority differs")
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        gallery_paths = paths_from_relatives(np.asarray(archive["train_relative_paths"]))
        query_paths = paths_from_relatives(np.asarray(archive["test_relative_paths"])[:32])
    if len(gallery_paths) != 59_551 or len(query_paths) != 32:
        raise ValueError("SOP scale gallery rows differ")
    gallery_digest = hashlib.sha256()
    for path in gallery_paths:
        gallery_digest.update(bytes.fromhex(sha(path)))
    manifest = MANIFEST.read_bytes()
    query_hashes = [sha(path) for path in query_paths]
    if any(
        bytes.fromhex(digest) != manifest[i * 32 : (i + 1) * 32]
        for i, digest in enumerate(query_hashes)
    ):
        raise ValueError("SOP scale query image content differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    opened = []
    original_open = cutile_int8.CutilePackedInt8Gallery.open_packed

    def capture(library: Path, packed: object):
        opened.append(packed)
        return original_open(library, packed)

    started = time.perf_counter()
    with patch.object(cutile_int8.CutilePackedInt8Gallery, "open_packed", staticmethod(capture)):
        index = Siglip2CompactIndex.from_artifacts(
            model_snapshot=MODEL,
            training_receipt=TRAINING / "receipt.json",
            training_checkpoint=TRAINING / "checkpoint.pt",
            train_embeddings=TRAINING / "train_embeddings.npy",
            native_library=LIBRARY,
            expected_receipt_sha256=TRAINING_SHA,
            precision="fp16_native",
            custom_gallery_image_paths=gallery_paths,
        )
    with index:
        torch.cuda.synchronize()
        build_wall = time.perf_counter() - started
        if index.encoder is None or index.gallery is None or len(opened) != 1:
            raise ValueError("SOP scale gallery did not load")
        gallery = opened[0]
        sample_starts = (
            0,
            (len(gallery_paths) // 2 // 32) * 32,
            ((len(gallery_paths) - 1) // 32) * 32,
        )
        for start in sample_starts:
            stop = min(start + 32, len(gallery_paths))
            with ExitStack() as stack:
                images = [
                    stack.enter_context(Image.open(path)) for path in gallery_paths[start:stop]
                ]
                direct = index.encoder.encode_images(images)
            if not (
                torch.equal(gallery.codes[start:stop], direct.codes)
                and torch.equal(gallery.inverse_norms[start:stop], direct.inverse_norms)
            ):
                raise ValueError("SOP scale gallery packed sample differs")
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)) for path in query_paths]
            queries = index.encoder.encode_images(images)
            native_rows, native_scores = index.gallery.search_packed(queries)
            search_started = time.perf_counter()
            public_rows, public_scores = index.search_images(images)
            torch.cuda.synchronize()
            search_wall = time.perf_counter() - search_started
        if not (
            np.array_equal(native_rows, public_rows)
            and np.array_equal(native_scores, public_scores)
        ):
            raise ValueError("SOP scale public/native top-10 differs")
    scores = (
        (queries.codes.float().cuda() @ gallery.codes.float().cuda().T)
        * queries.inverse_norms.float().cuda()[:, None]
        * gallery.inverse_norms.float().cuda()[None, :]
    )
    expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
    expected_scores = scores.gather(1, expected).cpu().numpy()
    max_error = float(np.max(np.abs(public_scores - expected_scores)))
    peak_cuda = torch.cuda.max_memory_allocated()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    expected_rows = expected.cpu().numpy()
    checks = {
        "top10_exact": bool(np.array_equal(public_rows, expected_rows)),
        "scores_finite": bool(np.isfinite(public_scores).all()),
        "score_error_within_1e-5": bool(max_error <= 1e-5),
        "build_within_600s": bool(build_wall <= 600),
        "peak_cuda_within_2gb": bool(peak_cuda <= 2_000_000_000),
        "peak_rss_within_6gb": bool(peak_rss <= 6_000_000_000),
    }
    mismatch_rows = np.flatnonzero(np.any(public_rows != expected_rows, axis=1))
    result = {
        "schema": "sfora-sop-custom-image-gallery-scale-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "imported_sha256": imported,
        "training_receipt_sha256": TRAINING_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "native_library_sha256": LIBRARY_SHA,
        "test_manifest_sha256": MANIFEST_SHA,
        "gallery_image_digests_sha256": gallery_digest.hexdigest(),
        "query_image_sha256": query_hashes,
        "gallery_rows": len(gallery_paths),
        "query_rows": len(query_paths),
        "sample_starts": sample_starts,
        "sample_rows_exact": sum(min(32, len(gallery_paths) - start) for start in sample_starts),
        "gallery_wire_bytes_per_row": gallery.bytes_per_vector,
        "gallery_packed_sha256": hashlib.sha256(
            gallery.codes.numpy().tobytes() + gallery.inverse_norms.numpy().tobytes()
        ).hexdigest(),
        "public_top10_ordinals": public_rows.tolist(),
        "public_top10_scores": public_scores.tolist(),
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "mismatch_query_rows": mismatch_rows.tolist(),
        "expected_top10_ordinals": expected_rows.tolist(),
        "expected_top10_scores": expected_scores.tolist(),
        "max_score_abs_delta": max_error,
        "gallery_build_wall_seconds": build_wall,
        "one_batch32_search_wall_seconds": search_wall,
        "peak_cuda_allocated_bytes": peak_cuda,
        "peak_parent_host_rss_bytes": peak_rss,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    if not result["gate_pass"]:
        print(json.dumps({"gate_checks": checks, "mismatch_query_rows": mismatch_rows.tolist()}, sort_keys=True), flush=True)
        raise ValueError("SOP scale custom-gallery gate fails")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "gallery_rows",
                    "sample_rows_exact",
                    "max_score_abs_delta",
                    "gallery_build_wall_seconds",
                    "one_batch32_search_wall_seconds",
                    "peak_cuda_allocated_bytes",
                    "peak_parent_host_rss_bytes",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
