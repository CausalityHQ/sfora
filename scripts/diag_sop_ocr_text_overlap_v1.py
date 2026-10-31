#!/usr/bin/env python3
"""OCR text-overlap diagnostic on SOP TRAIN-held rank-1 errors (Qwen3-VL-8B-Instruct as a transcriber).

Measurement, not a retrieval system. Question: how often does an error query carry legible text that also appears in
an image of the same product, and how often in the wrongly retrieved image?

Pre-registered (committed before any number exists):
  base    : cache global_512.npy (59,551 unit rows, archive order); packed int8 cosine (codes x inverse norms, stable
            descending, self excluded). Held per-query R@1 must equal receipt quality.per_query_r1, else exit 1.
  split   : deterministic_class_partition(train_labels, fit 0.9, seed 179019): 5,851 held queries vs all 59,551.
  cases   : every held ERROR (wrong top-1) + CONTROL = 300 held correct queries (seed 0). Images per case: query, top-1
            (wrong for errors, correct for controls), up to 3 positives = same label, not the query, not the top-1,
            in ascending global rank. Images deduplicated across cases; each transcribed once.
  ocr     : greedy generate, max_new_tokens 48, bf16, one image (longer side <= 768 px), PROMPT below.
  text    : lowercase; tokens = alphanumeric runs of length >= 3 plus digit-containing runs of length >= 2; drop
            STOPWORDS; a NONE transcript is the empty set.
  per case: has_text = |Q| >= 1; overlap_pos = max over positives |Q & P| >= 1; overlap_wrong = |Q & W| >= 1 (W =
            top-1; for controls it is the correct top-1); discriminative = overlap_pos and not overlap_wrong.
  report  : fractions with Wilson 95% intervals for errors and controls.
  decision: BUILD an OCR-text index screen iff the discriminative fraction among held errors has Wilson lower95 >= 0.10
            (i.e. >= ~30 of ~300 errors are plausibly fixable by exact text match); else CLOSE the text lane.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_sop_token_cache_v1 as s  # noqa: E402
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402

N_POS, N_CONTROL, MAX_SIDE, MAX_NEW, CHECKPOINT_EVERY, EXAMPLES, Z = 3, 300, 768, 48, 50, 30, 1.959963984540054
PROMPT = ("Transcribe all legible printed text (brand names, model numbers, labels) in this image. "
          "If there is none, answer NONE.")
STOPWORDS = frozenset({"the", "and", "made", "for", "with", "usa", "china", "none"})
FLAGS = ("has_text", "overlap_pos", "overlap_wrong", "discriminative")


def tokens(transcript: str) -> set:
    """Normalised token set of one transcript; NONE -> empty."""
    t = transcript.strip().lower()
    if t.rstrip(".") == "none":
        return set()
    return {w for w in re.findall(r"[a-z0-9]+", t)
            if (len(w) >= 3 or (len(w) >= 2 and any(c.isdigit() for c in w))) and w not in STOPWORDS}


def case_flags(q: set, wrong: set, positives: list) -> dict:
    pos = max((len(q & p) for p in positives), default=0) >= 1
    wr = len(q & wrong) >= 1
    return {"has_text": bool(q), "overlap_pos": pos, "overlap_wrong": wr, "discriminative": pos and not wr}


def wilson(k: int, n: int, z: float = Z) -> list:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def summarise(flags: list) -> dict:
    n = len(flags)
    out: dict = {"n": n}
    for f in FLAGS:
        k = sum(x[f] for x in flags)
        out[f] = {"count": k, "fraction": k / n if n else math.nan, "wilson95": wilson(k, n)}
    return out


def image_rows(cases: list) -> list:
    """Sorted unique image rows over all cases."""
    return sorted({r for c in cases for r in (c["query"], c["top1"], *c["positives"])})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("cache-dir", "archive", "dataset-root", "receipt", "vlm-snapshot", "output-dir"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--resume-cache", type=Path, help="transcripts.json from a crashed run (same model, same prompt)")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict = {"schema": "sop-ocr-text-overlap-diagnostic-v1", "official_read": False, "seconds": {},
                    "prompt": PROMPT, "stopwords": sorted(STOPWORDS), "max_side": MAX_SIDE, "max_new_tokens": MAX_NEW}

    def save() -> None:
        result["peak_host_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["seconds"]["total"] = time.perf_counter() - started
        (a.output_dir / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    def lap(label: str, t0: float) -> float:
        torch.cuda.synchronize()
        result["seconds"][label] = time.perf_counter() - t0
        save()
        return time.perf_counter()

    t = time.perf_counter()
    with np.load(a.archive, allow_pickle=False) as archive:
        labels_np = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    part = deterministic_class_partition(tuple(map(int, labels_np)), fit_fraction=0.9, seed=179019)
    held_np = np.asarray(part.validation_row_indexes, dtype=np.int64)
    rows = len(labels_np)
    unit_np = np.load(a.cache_dir / "global_512.npy")
    if rows != 59_551 or len(held_np) != 5_851 or unit_np.shape != (rows, 512):
        raise ValueError("SOP TRAIN partition or cache differs")
    labels = torch.from_numpy(labels_np).cuda()
    held = torch.from_numpy(held_np).cuda()
    every = torch.arange(rows, device="cuda")
    score = s.packed_scorer(torch.from_numpy(unit_np), every)
    ranked = s.rank(score, held, every, rows)
    base_r1, base_ap, rec = s.metrics(ranked, held, every, labels)
    result["baseline"] = {"recall_at_1": 100 * float(base_r1.mean()), "map_at_r": 100 * float(base_ap.mean()),
                          "held_recall_at_k": rec}
    if not np.array_equal(base_r1, np.asarray(json.loads(a.receipt.read_text())["quality"]["per_query_r1"],
                                              dtype=np.float64)):
        result["decision"] = "INVALID: baseline does not reproduce receipt per-query R@1"
        save()
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(1)

    errors = np.flatnonzero(base_r1 == 0)
    controls = np.sort(np.random.default_rng(0).choice(np.flatnonzero(base_r1 == 1), size=N_CONTROL, replace=False))
    top1 = ranked[:, 0].cpu().numpy()
    cases = []
    for kind, idx in (("error", errors), ("control", controls)):
        for i in idx:
            q, w = int(held_np[i]), int(top1[i])
            pos = np.flatnonzero(labels_np == labels_np[q])
            pos = pos[(pos != q) & (pos != w)]
            sc = score(torch.tensor([q], device="cuda"))[0, torch.from_numpy(pos).cuda()]
            order = torch.argsort(sc, descending=True, stable=True)[:N_POS].cpu().numpy()  # pos ascending -> global tie order
            cases.append({"kind": kind, "query": q, "top1": w, "positives": [int(x) for x in pos[order]]})
    images = image_rows(cases)
    result["counts"] = {"errors": len(errors), "controls": len(controls), "images": len(images),
                        "image_slots": sum(2 + len(c["positives"]) for c in cases)}
    t = lap("cases", t)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(a.vlm_snapshot, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm_snapshot, dtype=torch.bfloat16,
                                                            local_files_only=True).cuda().eval()
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (8, 8))},
                                      {"type": "text", "text": PROMPT}]}], tokenize=False, add_generation_prompt=True)
    result["chat_text"] = text
    t = lap("vlm_load", t)

    cache: dict = {}
    if a.resume_cache is not None:
        cache = {int(k): v for k, v in json.loads(a.resume_cache.read_text()).items()}
    stats = {"generated": 0, "seconds": 0.0, "resumed": len(cache)}
    result["ocr"] = stats

    def checkpoint() -> None:
        (a.output_dir / "transcripts.json").write_text(json.dumps({str(k): v for k, v in sorted(cache.items())}))
        stats["seconds_per_image"] = stats["seconds"] / max(stats["generated"], 1)
        save()

    with torch.inference_mode():
        for row in images:
            if row in cache:
                continue
            t0 = time.perf_counter()
            with Image.open(a.dataset_root / relatives[row]) as im:
                rgb = im.convert("RGB")
            rgb.thumbnail((MAX_SIDE, MAX_SIDE))
            x = processor(text=[text], images=[rgb], return_tensors="pt").to("cuda")
            out = model.generate(**x, max_new_tokens=MAX_NEW, do_sample=False)
            cache[row] = processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            stats["seconds"] += time.perf_counter() - t0
            stats["generated"] += 1
            if stats["generated"] % CHECKPOINT_EVERY == 0:
                checkpoint()
                print(json.dumps({"generated": stats["generated"], "of": len(images),
                                  "s_per_image": stats["seconds_per_image"]}), flush=True)
    checkpoint()
    t = lap("ocr", t)

    tok = {row: tokens(cache[row]) for row in images}
    for c in cases:
        c.update(case_flags(tok[c["query"]], tok[c["top1"]], [tok[r] for r in c["positives"]]))
    result["errors"] = summarise([c for c in cases if c["kind"] == "error"])
    result["controls"] = summarise([c for c in cases if c["kind"] == "control"])
    lower = result["errors"]["discriminative"]["wilson95"][0]
    result["decision"] = "BUILD" if lower >= 0.10 else "CLOSE"

    def show(c: dict) -> dict:
        def img(r):
            return {"path": relatives[r], "transcript": cache[r], "tokens": sorted(tok[r])}
        return {**{f: c[f] for f in FLAGS}, "query": img(c["query"]), "top1": img(c["top1"]),
                "positives": [img(r) for r in c["positives"]]}

    err_cases = [c for c in cases if c["kind"] == "error"]
    pick = np.sort(np.random.default_rng(0).choice(len(err_cases), size=min(EXAMPLES, len(err_cases)), replace=False))
    (a.output_dir / "error_examples.json").write_text(json.dumps([show(err_cases[i]) for i in pick], indent=1) + "\n")
    (a.output_dir / "cases.json").write_text(json.dumps(
        [{**c, "query_path": relatives[c["query"]]} for c in cases]) + "\n")
    lap("report", t)
    print(json.dumps({k: result[k] for k in ("baseline", "counts", "errors", "controls", "ocr", "decision", "seconds",
                                             "peak_cuda_allocated_gib", "peak_host_rss_gib")},
                     indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
