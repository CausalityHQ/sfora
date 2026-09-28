"""Bounded TRAIN-only native compiler smoke; no public compiler option."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import signal
import time

import numpy as np
import torch

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex
from benchmark_sop_siglip2_cuda_graph_public import ARCHIVE_SHA, NATIVE_SHA, RECEIPT_SHA, decode, sha256, stats


def timeout(signum: int, frame: object) -> None:
    raise TimeoutError("120-second compile/parity cap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    archive = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
    native = Path("/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so")
    run = Path("/home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1")
    assert sha256(archive) == ARCHIVE_SHA and sha256(native) == NATIVE_SHA
    assert sha256(run / "receipt.json") == RECEIPT_SHA
    assert sha256(Path(serving.__file__)) == "ef454200ca17b80917fe7aa52cf232703779d21af7e8ffe0d25f8e811724bc62"
    with np.load(archive, allow_pickle=False) as source:
        relative = [PurePosixPath(str(p)) for p in source["train_relative_paths"][:32]]
    assert all(not p.is_absolute() and ".." not in p.parts for p in relative)
    paths = [Path("/home/riomus/datasets/Stanford_Online_Products").joinpath(*p.parts) for p in relative]
    assert all(p.is_file() and not p.is_symlink() for p in paths)
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"schema": "sfora-sop-siglip2-compiler-smoke-v1", "claim_eligible": False,
              "source_sha256": sha256(Path(__file__)), "serving_sha256": sha256(Path(serving.__file__)),
              "receipt_sha256": RECEIPT_SHA, "archive_sha256": ARCHIVE_SHA,
              "image_sha256": [sha256(p) for p in paths], "batch": 32,
              "gallery_rows": 59551, "hardware": torch.cuda.get_device_name(),
              "torch": torch.__version__, "compiler_mode": "default/fullgraph/static"}
    started = time.monotonic()
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=Path("/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"),
        training_receipt=run / "receipt.json", training_checkpoint=run / "checkpoint.pt",
        train_embeddings=run / "train_embeddings.npy", native_library=native,
        expected_receipt_sha256=RECEIPT_SHA, precision="fp16_native",
    ) as index:
        encoder, gallery = index.encoder, index.gallery
        assert encoder is not None and gallery is not None
        eager = encoder.vision
        images = decode(paths)
        reference = encoder.encode_images(images)
        expected = gallery.search_packed(reference)
        torch.cuda.reset_peak_memory_stats()
        signal.signal(signal.SIGALRM, timeout)
        smoke_started = time.monotonic()
        signal.alarm(120)
        try:
            compiled = torch.compile(eager, fullgraph=True, dynamic=False, mode="default")
            encoder.vision = compiled
            candidate = encoder.encode_images(images)
            actual = gallery.search_packed(candidate)
            report["packed_exact"] = bool(torch.equal(reference.codes, candidate.codes) and torch.equal(reference.inverse_norms, candidate.inverse_norms))
            report["top10_exact"] = bool(np.array_equal(expected[0], actual[0]) and np.array_equal(expected[1], actual[1]))
            report["code_mismatches"] = int((reference.codes != candidate.codes).sum())
            report["norm_mismatches"] = int((reference.inverse_norms != candidate.inverse_norms).sum())
            if not report["packed_exact"] or not report["top10_exact"]:
                raise ValueError("compiled packed output differs")
            if torch.cuda.max_memory_allocated() >= 16_000_000_000:
                raise ValueError("compiled allocation exceeds 16GB cap")
        except Exception as error:
            report.update(decision="KILL_COMPILE_OR_PARITY", error=f"{type(error).__name__}: {error}")
        finally:
            signal.alarm(0)
            report["compile_parity_seconds"] = time.monotonic() - smoke_started
        if "decision" not in report:
            arms = {"eager": eager, "compiled": compiled}
            for arm in arms.values():
                encoder.vision = arm
                for _ in range(2):
                    index.search_images(decode(paths))
            raw = {name: [] for name in arms}
            for block in range(10):
                for name in (("eager", "compiled", "compiled", "eager") if block % 2 == 0 else ("compiled", "eager", "eager", "compiled")):
                    encoder.vision = arms[name]
                    for _ in range(2):
                        torch.cuda.synchronize()
                        tick = time.perf_counter_ns()
                        result = index.search_images(decode(paths))
                        torch.cuda.synchronize()
                        raw[name].append(time.perf_counter_ns() - tick)
                        assert np.array_equal(expected[0], result[0]) and np.array_equal(expected[1], result[1])
            report["raw_ns"] = raw
            report["results"] = {name: stats(values) for name, values in raw.items()}
            r = report["results"]
            useful = all(r["compiled"][k] <= .95 * r["eager"][k] for k in ("p50_ms", "p95_ms"))
            report["decision"] = "PROMISING_PILOT_ONLY" if useful else "KILL_LATENCY_FLOOR"
        encoder.vision = eager
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["whole_seconds"] = time.monotonic() - started
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
