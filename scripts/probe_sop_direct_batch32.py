"""Bounded CPU-only screen of the existing direct processor at batch32."""
import json
from pathlib import Path, PurePosixPath
import resource
import signal
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor

from probe_sop_siglip2_processor_lut import ARCHIVE_SHA, PROCESSOR_SHA, sha256, summary
import sfora.siglip2_compact_serving as serving


def direct_batch(images):
    return torch.cat([serving._direct_preprocess(image) for image in images])


def main():
    started = time.monotonic()
    def timeout(_signal, _frame):
        raise TimeoutError("CPU batch32 screen120-second cap")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    root = Path(sys.argv[1])
    root.mkdir(exist_ok=False)
    report = {"schema": "sfora-sop-direct-batch32-cpu-v1", "claim_eligible": False,
              "quality_measured": False, "full_latency_measured": False, "cuda_used": False,
              "source_sha256": sha256(Path(__file__)), "serving_sha256": sha256(Path(serving.__file__)),
              "raw_ns": {"control": [], "direct": []}, "image_sha256": [], "batches_checked": 0}
    try:
        torch.set_num_threads(20)
        archive = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
        model = Path("/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c")
        assert sha256(archive) == ARCHIVE_SHA
        assert sha256(model / "preprocessor_config.json") == PROCESSOR_SHA
        assert serving._direct_processor_supported(PROCESSOR_SHA)
        report.update(archive_sha256=ARCHIVE_SHA, processor_sha256=PROCESSOR_SHA,
                      torch=torch.__version__, threads=torch.get_num_threads())
        processor = AutoImageProcessor.from_pretrained(model / "preprocessor_config.json", local_files_only=True, backend="torchvision")
        rng = np.random.default_rng(179035)
        synthetic = [Image.fromarray(rng.integers(0, 256, (77 + i, 105 + i, 3), dtype=np.uint8)) for i in range(32)]
        assert torch.equal(processor(images=synthetic, return_tensors="pt")["pixel_values"], direct_batch(synthetic))
        report["synthetic_exact"] = True
        seen, paths = set(), []
        with np.load(archive, allow_pickle=False) as source:
            for value in source["train_relative_paths"]:
                relative = PurePosixPath(str(value))
                assert relative.parts and not relative.is_absolute() and ".." not in relative.parts
                path = Path("/home/riomus/datasets/Stanford_Online_Products").joinpath(*relative.parts)
                assert path.is_file() and not path.is_symlink()
                digest = sha256(path)
                if digest not in seen:
                    seen.add(digest)
                    paths.append(path)
                    report["image_sha256"].append(digest)
                if len(paths) == 1024:
                    break
        assert len(paths) == 1024
        for block in range(32):
            images = []
            for path in paths[block * 32:(block + 1) * 32]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            expected = processor(images=[image.convert("RGB") for image in images], return_tensors="pt")["pixel_values"]
            actual = direct_batch(images)
            assert expected.shape == actual.shape == (32, 3, 256, 256)
            if not torch.equal(expected, actual):
                report["mismatched_entries"] = int((expected != actual).sum())
                raise ValueError("direct batch32 pixel parity differs")
            report["batches_checked"] += 1
            for arm in (("control", "direct", "direct", "control") if block % 2 == 0 else ("direct", "control", "control", "direct")):
                tick = time.perf_counter_ns()
                if arm == "control":
                    value = processor(images=[image.convert("RGB") for image in images], return_tensors="pt")["pixel_values"]
                else:
                    value = direct_batch(images)
                report["raw_ns"][arm].append(time.perf_counter_ns() - tick)
                assert torch.equal(expected, value)
            print(json.dumps({"batches_checked": block + 1}), flush=True)
        stats = {arm: summary(values) for arm, values in report["raw_ns"].items()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gates = {"exact": report["batches_checked"] == 32,
                 "p50": stats["direct"]["p50_ms"] <= .70 * stats["control"]["p50_ms"],
                 "p95": stats["direct"]["p95_ms"] <= .80 * stats["control"]["p95_ms"],
                 "rss": rss < 2_000_000_000, "budget": time.monotonic() - started <= 120}
        report.update(timing=stats, gates=gates, peak_rss_bytes=rss,
                      decision="GO_FREEZE_PUBLIC_PILOT" if all(gates.values()) else "KILL_DIRECT_BATCH32_CPU")
    except Exception as error:
        report.update(decision="KILL_DIRECT_BATCH32_EXECUTION", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report["whole_seconds"] = time.monotonic() - started
        (root / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("raw_ns", "image_sha256")}), flush=True)


if __name__ == "__main__":
    main()
