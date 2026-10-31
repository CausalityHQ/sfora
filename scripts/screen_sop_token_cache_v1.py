#!/usr/bin/env python3
"""TRAIN-held screen: per-image token cache from the fine-tuned SOP So400m@384/512 checkpoint plus three
training-free re-scorers (docs/fundamental_redesign_proposals_2026-10-31.md: candidate 1 stage 0, 2 and 3).

Pre-registered (committed before any screen number exists):
  model : run dir via So400mCompactEncoder.from_checkpoint (384, 512-D). Encode = training export: fp16 autocast
          vision, pooler_output (1152) -> L2 -> Linear(1152->512) -> L2, batches of 64 in archive row order.
  split : SOP TRAIN, class partition seed 179019 fit 0.9; 5,851 held queries vs all 59,551 TRAIN, self excluded.
  base  : packed int8 512-D cosine (codes x inverse norms, stable descending). Held per-query R@1 must equal
          receipt quality.per_query_r1 exactly, else INVALID and stop.
  cache : 512-D global; final-block tokens (post_layernorm = pooling-head input) 27x27 avg-pooled to 9x9, fp16
          memmap; the pooling head's own MHA weights (forward hook on vision.head.attention, head-averaged, 729);
          context = sum (1 - a_i) t_i / sum (1 - a_i) over the 729 tokens.
  S0    : PCA256 of tokens (fit images) + per-token L2; rerank global top-32 by z(global) + w z(MaxSim),
          z within the 32, w in {0.25, 0.5, 1.0}.
  S2    : PCA128 of context (fit images) + L2; score = global + w context cosine, full gallery, w in {0.1, 0.2, 0.3}.
          S0/S2 w = best fit R@1 (then mAP@R, then smaller w), <=5,000 fit queries (seed 0) vs the fit gallery.
  S1    : crop the original image to the smallest 27x27-grid box holding >=80% attention mass, +10% of the box
          size per side (clipped), re-encode at 384. Variants: zoom, sum of the two cosines, renormalised mean of
          the two unit vectors (all packed int8). Skipped if the median box area fraction > 0.8.
  stats : held dR@1, dmAP@R in pp vs base; paired product bootstrap (5000 draws, seed 179019, resample held
          products). PASS iff dR@1 lower95 > 0 and dmAP@R point >= -0.2, else KILL. One held read per variant.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

GRID, TOP, BLOCK, WIDTH = 27, 32, 64, 1152
S0_W, S2_W = (0.25, 0.5, 1.0), (0.1, 0.2, 0.3)


def attach_attention_hook(vision: torch.nn.Module) -> dict:
    """Capture the head-averaged weights the pooling head's MHA actually returns (it is called with the
    nn.MultiheadAttention defaults need_weights=True, average_attn_weights=True)."""
    captured: dict = {}

    def hook(_module, _args, output):
        captured["weights"] = output[1]

    vision.head.attention.register_forward_hook(hook)
    return captured


def maxsim(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """q (b,T,D), c (b,K,S,D) unit tokens -> (b,K): mean over query tokens of the max cosine over candidate tokens."""
    return torch.einsum("btd,bksd->bkts", q.float(), c.float()).amax(-1).mean(-1)


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, unbiased=False, keepdim=True).clamp_min(1e-12)


def fuse_order(g: torch.Tensor, m: torch.Tensor, w: float) -> torch.Tensor:
    """Permutation (b,K) by z(global) + w z(maxsim), descending; ties keep the global order."""
    return torch.argsort(zscore(g.float()) + w * zscore(m.float()), dim=1, descending=True, stable=True)


def context_pool(tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    w = 1.0 - attn.float()
    return (w[..., None] * tokens.float()).sum(1) / w.sum(1, keepdim=True)


def attention_box(attn: torch.Tensor, grid: int = GRID, mass: float = 0.8) -> torch.Tensor:
    """(B, grid*grid) -> (B,4) int [r0, r1, c0, c1) of the smallest-area box holding >= mass (first on ties)."""
    a = attn.float().reshape(-1, grid, grid)
    p = F.pad((a / a.sum((1, 2), keepdim=True)).cumsum(1).cumsum(2), (1, 0, 1, 0))
    lo, hi = torch.triu_indices(grid + 1, grid + 1, 1, device=a.device)
    total = p[:, hi][:, :, hi] - p[:, lo][:, :, hi] - p[:, hi][:, :, lo] + p[:, lo][:, :, lo]
    side = (hi - lo).float()
    cost = torch.where(total >= mass - 1e-6, side[:, None] * side[None, :], torch.inf).flatten(1)
    best = cost.argmin(1)
    i, j = best // len(lo), best % len(lo)
    return torch.stack([lo[i], hi[i], lo[j], hi[j]], 1)


@torch.inference_mode()
def rank(score, queries: torch.Tensor, gallery: torch.Tensor, rows: int, rerank=None) -> torch.Tensor:
    """Top-TOP gallery rows per query by stable descending score(q) (b, len(gallery)); self excluded.
    rerank(q, candidate_rows, candidate_scores) -> permutation of the top TOP."""
    col = torch.full((rows,), -1, dtype=torch.long, device=gallery.device)
    col[gallery] = torch.arange(len(gallery), device=gallery.device)
    if int(col[queries].min()) < 0:
        raise ValueError("every query must be in the gallery")
    out = []
    for q in queries.split(BLOCK):
        s = score(q)
        s[torch.arange(len(q), device=s.device), col[q]] = -torch.inf
        r = torch.argsort(s, dim=1, descending=True, stable=True)[:, :TOP]
        if rerank is not None:
            r = r.gather(1, rerank(q, gallery[r], s.gather(1, r)))
        out.append(gallery[r])
    return torch.cat(out)


def metrics(ranked: torch.Tensor, queries: torch.Tensor, gallery: torch.Tensor, labels: torch.Tensor):
    """Per-query R@1 and AP@R (R = gallery positives minus self, as score_packed_full_gallery) + R@{1,5,10,32}."""
    rel = torch.bincount(labels[gallery], minlength=int(labels.max()) + 1)[labels[queries]] - 1
    if int(rel.min()) < 1 or int(rel.max()) > ranked.shape[1]:
        raise ValueError("positive inventory differs")
    m = labels[ranked] == labels[queries][:, None]
    ranks = torch.arange(1, m.shape[1] + 1, device=m.device)
    ap = ((m.cumsum(1) / ranks[None, :]) * m * (ranks[None, :] <= rel[:, None])).sum(1) / rel
    recall = {f"R@{k}": float(m[:, :k].any(1).double().mean()) for k in (1, 5, 10, 32)}
    return m[:, 0].double().cpu().numpy(), ap.double().cpu().numpy(), recall


def product_bootstrap(d: np.ndarray, products: np.ndarray, draws: int = 5000, seed: int = 179019) -> np.ndarray:
    """d (Q,m) paired deltas -> (draws,m) means with whole products resampled with replacement."""
    _, inv = np.unique(products, return_inverse=True)
    k = int(inv.max()) + 1
    s = np.stack([np.bincount(inv, col, minlength=k) for col in d.T], 1)
    n = np.bincount(inv, minlength=k)
    rng = np.random.default_rng(seed)
    c = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(draws)]).astype(np.float64)
    return (c @ s) / (c @ n)[:, None]


def compare(base: tuple, new: tuple, products: np.ndarray) -> dict:
    d = 100.0 * np.stack([new[0] - base[0], new[1] - base[1]], 1)
    lo, hi = np.percentile(product_bootstrap(d, products), [2.5, 97.5], axis=0)
    point = d.mean(0)
    return {
        "recall_at_1": 100 * float(new[0].mean()), "map_at_r": 100 * float(new[1].mean()),
        "d_r1": float(point[0]), "d_r1_95": [float(lo[0]), float(hi[0])],
        "d_map": float(point[1]), "d_map_95": [float(lo[1]), float(hi[1])],
        "decision": "PASS" if lo[0] > 0 and point[1] >= -0.2 else "KILL",
    }


def pca(chunks, k: int):
    """Mean and top-k principal axes (D,k) from streamed (n,D) chunks."""
    # ponytail: one-pass E[xx^T] - mu mu^T in float64 accumulators; two-pass centering if conditioning bites.
    n, s, ss = 0, 0.0, 0.0
    for x in chunks:
        x = x.float()
        n, s, ss = n + len(x), s + x.sum(0).double(), ss + (x.T @ x).double()
    mean = s / n
    _, vecs = torch.linalg.eigh((ss / n - torch.outer(mean, mean)).cpu())
    return mean.float(), vecs[:, -k:].flip(1).float().to(mean.device)


class Rows(Dataset):
    def __init__(self, paths, crops=None):
        self.paths, self.crops = paths, crops

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            rgb = image.convert("RGB")
        if self.crops is None:
            return rgb
        top, bottom, left, right = self.crops[i]
        w, h = rgb.size
        x0, y0 = math.floor(left * w), math.floor(top * h)
        return rgb.crop((x0, y0, max(math.ceil(right * w), x0 + 1), max(math.ceil(bottom * h), y0 + 1)))


class Collate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]


@torch.inference_mode()
def encode(encoder, paths, crops=None, on_batch=None, workers: int = 8) -> torch.Tensor:
    """Unit 512-D rows in export arithmetic; on_batch(start, last_hidden_state) sees each batch's tokens."""
    loader = DataLoader(Rows(paths, crops), batch_size=BLOCK, num_workers=workers, pin_memory=True,
                        collate_fn=Collate(encoder.processor))
    out, start = [], 0
    for pixels in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            o = encoder.vision(pixel_values=pixels.cuda(non_blocking=True))
        with torch.autocast("cuda", enabled=False):
            out.append(F.normalize(encoder.head(F.normalize(o.pooler_output.float(), dim=1)), dim=1).cpu())
        if on_batch is not None:
            on_batch(start, o.last_hidden_state)
        start += len(pixels)
        if start % (BLOCK * 100) == 0:
            print(json.dumps({"encoded": start}), flush=True)
    return torch.cat(out)


def packed_scorer(unit: torch.Tensor, gallery: torch.Tensor):
    p = pack_int8_unit_embeddings(unit)
    c, inv = p.codes.float().cuda(), p.inverse_norms.float().cuda()
    cg, ig = c[gallery], inv[gallery]
    return lambda q: (c[q] @ cg.T) * inv[q, None] * ig[None, :]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("run-dir", "model-snapshot", "archive", "dataset-root", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_sop_siglip2_substrate_pair as pair
    from sfora.so400m_compact_serving import So400mCompactEncoder

    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict = {"schema": "sop-so400m-token-cache-screen-v1", "official_read": False, "seconds": {}}

    def save() -> None:
        result["peak_host_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["seconds"]["total"] = time.perf_counter() - started
        (a.output_dir / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    def lap(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        result["seconds"][name] = time.perf_counter() - t0
        save()
        return time.perf_counter()

    receipt = json.loads((a.run_dir / "receipt.json").read_text())
    if pair.sha256(a.run_dir / "checkpoint.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from receipt")
    with np.load(a.archive, allow_pickle=False) as archive:
        labels_np = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    paths = pair.paths_from_archive(a.dataset_root, relatives)
    part = pair.deterministic_class_partition(tuple(map(int, labels_np)), fit_fraction=0.9, seed=179019)
    held_np = np.asarray(part.validation_row_indexes, dtype=np.int64)
    fit_np = np.asarray(part.fit_row_indexes, dtype=np.int64)
    rows = len(labels_np)
    if rows != 59_551 or len(held_np) != 5_851:
        raise ValueError("SOP TRAIN partition differs")
    fit_q_np = np.sort(np.random.default_rng(0).choice(fit_np, size=min(5000, len(fit_np)), replace=False))
    labels = torch.from_numpy(labels_np).cuda()
    held, fit, fit_q = (torch.from_numpy(x).cuda() for x in (held_np, fit_np, fit_q_np))
    every = torch.arange(rows, device="cuda")
    products = labels_np[held_np]

    encoder = So400mCompactEncoder.from_checkpoint(a.model_snapshot, a.run_dir / "checkpoint.pt", resolution=384,
                                                   output_dim=512, device=torch.device("cuda"))
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 must be disabled")
    captured = attach_attention_hook(encoder.vision)

    # Pass 1: global + 9x9 tokens + attention + context + box.
    tokens = np.lib.format.open_memmap(a.output_dir / "tokens_9x9_fp16.npy", "w+", np.float16, (rows, 81, WIDTH))
    attention = torch.empty(rows, GRID * GRID)
    context = torch.empty(rows, WIDTH)
    boxes = torch.empty(rows, 4, dtype=torch.long)

    def on_batch(start: int, hidden: torch.Tensor) -> None:
        weights = captured.pop("weights")
        b = len(hidden)
        if hidden.shape != (b, GRID * GRID, WIDTH) or weights.shape != (b, 1, GRID * GRID):
            raise ValueError("token or attention geometry differs")
        w = weights[:, 0].float()
        grid = hidden.float().transpose(1, 2).reshape(b, WIDTH, GRID, GRID)
        tokens[start:start + b] = F.avg_pool2d(grid, 3).flatten(2).transpose(1, 2).half().cpu().numpy()
        attention[start:start + b] = w.cpu()
        context[start:start + b] = context_pool(hidden, w).cpu()
        boxes[start:start + b] = attention_box(w).cpu()

    t = time.perf_counter()
    unit = encode(encoder, paths, on_batch=on_batch, workers=a.workers)
    tokens.flush()
    result["encode_images_per_second"] = rows / (time.perf_counter() - t)
    t = lap("encode_pass1", t)
    np.save(a.output_dir / "global_512.npy", unit.numpy())
    np.save(a.output_dir / "attention_729.npy", attention.numpy())
    np.save(a.output_dir / "context_1152.npy", context.numpy())
    np.save(a.output_dir / "boxes_27.npy", boxes.numpy())
    result["attention_row_sum_max_abs_dev"] = float((attention.sum(1) - 1).abs().max())

    base_score = packed_scorer(unit, every)
    base_ranked = rank(base_score, held, every, rows)
    base_r1, base_ap, recall = metrics(base_ranked, held, every, labels)
    receipt_r1 = np.asarray(receipt["quality"]["per_query_r1"], dtype=np.float64)
    result["baseline"] = {"recall_at_1": 100 * float(base_r1.mean()), "map_at_r": 100 * float(base_ap.mean()),
                          "held_recall_at_k": recall,
                          "receipt_ap_max_abs_diff": float(np.abs(base_ap - np.asarray(
                              receipt["quality"]["per_query_ap"], dtype=np.float64)).max())}
    if not np.array_equal(base_r1, receipt_r1):
        result["decision"] = "INVALID: baseline does not reproduce receipt per-query R@1"
        save()
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(1)
    side = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    area = side.double().numpy() / GRID**2
    result["box_area_fraction"] = {"mean": float(area.mean()), **{f"p{q}": float(np.percentile(area, q))
                                                                   for q in (5, 25, 50, 75, 95)}}
    t = lap("baseline", t)

    fit_base = packed_scorer(unit, fit)
    base_fit = metrics(rank(fit_base, fit_q, fit, rows), fit_q, fit, labels)
    result["fit_baseline"] = {"recall_at_1": 100 * float(base_fit[0].mean()), "map_at_r": 100 * float(base_fit[1].mean())}

    def choose(name: str, grid_w, run_fit, run_held) -> None:
        fits = {}
        for w in grid_w:
            r1, ap, _ = run_fit(w)
            fits[str(w)] = [100 * float(r1.mean()), 100 * float(ap.mean())]
        best = max(grid_w, key=lambda w: (fits[str(w)][0], fits[str(w)][1], -w))
        r1, ap, rec = run_held(best)
        result[name] = {"fit_grid": fits, "w": best, "held_recall_at_k": rec,
                        **compare((base_r1, base_ap), (r1, ap), products)}

    # S0: token MaxSim fused with the global score on the global top-32.
    def chunks(index):
        for i in range(0, len(index), 512):
            yield torch.from_numpy(np.asarray(tokens[index[i:i + 512]])).cuda().reshape(-1, WIDTH)

    mean, axes = pca(chunks(fit_np), 256)
    tok = torch.empty(rows, 81, 256, dtype=torch.float16, device="cuda")
    for i in range(0, rows, 512):
        x = torch.from_numpy(np.asarray(tokens[i:i + 512])).cuda().float()
        tok[i:i + 512] = F.normalize((x - mean) @ axes, dim=-1).half()

    def s0(w):
        return lambda q, cand, g: fuse_order(g, maxsim(tok[q], tok[cand]), w)

    choose("S0_token_maxsim", S0_W,
           lambda w: metrics(rank(fit_base, fit_q, fit, rows, rerank=s0(w)), fit_q, fit, labels),
           lambda w: metrics(rank(base_score, held, every, rows, rerank=s0(w)), held, every, labels))
    del tok
    t = lap("S0", t)

    # S2: complement-of-attention context vector.
    ctx_full = context.cuda()
    mean, axes = pca([ctx_full[fit]], 128)
    ctx = F.normalize((ctx_full - mean) @ axes, dim=1)
    del ctx_full

    def s2(gallery, scorer, w):
        cg = ctx[gallery]
        return lambda q: scorer(q) + w * (ctx[q] @ cg.T)

    choose("S2_context", S2_W,
           lambda w: metrics(rank(s2(fit, fit_base, w), fit_q, fit, rows), fit_q, fit, labels),
           lambda w: metrics(rank(s2(every, base_score, w), held, every, rows), held, every, labels))
    t = lap("S2", t)

    # S1: attention zoom (second encode pass).
    if result["box_area_fraction"]["p50"] > 0.8:
        result["S1_zoom"] = {"decision": "SKIPPED: median box area > 0.8"}
    else:
        frac = boxes.double() / GRID
        hm = 0.1 * (frac[:, 1] - frac[:, 0])
        wm = 0.1 * (frac[:, 3] - frac[:, 2])
        crops = torch.stack([frac[:, 0] - hm, frac[:, 1] + hm, frac[:, 2] - wm, frac[:, 3] + wm], 1).clamp(0, 1)
        zoom = encode(encoder, paths, crops=crops.numpy(), workers=a.workers)
        t = lap("encode_pass2", t)
        np.save(a.output_dir / "zoom_global_512.npy", zoom.numpy())
        zoom_score = packed_scorer(zoom, every)
        variants = {"zoom": zoom_score,
                    "sum": lambda q: base_score(q) + zoom_score(q),
                    "mean": packed_scorer(F.normalize(unit + zoom, dim=1), every)}
        result["S1_zoom"] = {}
        for name, scorer in variants.items():
            r1, ap, rec = metrics(rank(scorer, held, every, rows), held, every, labels)
            result["S1_zoom"][name] = {"held_recall_at_k": rec, **compare((base_r1, base_ap), (r1, ap), products)}
        result["S1_zoom"]["decision"] = (
            "PASS" if any(v["decision"] == "PASS" for v in result["S1_zoom"].values()) else "KILL")
        lap("S1", t)
    save()
    summary = {k: result[k] for k in ("baseline", "box_area_fraction", "seconds", "peak_host_rss_gib")}
    for name in ("S0_token_maxsim", "S2_context"):
        summary[name] = {k: result[name][k] for k in ("w", "d_r1", "d_r1_95", "d_map", "decision")}
    summary["S1_zoom"] = {k: (v if isinstance(v, str) else {x: v[x] for x in ("d_r1", "d_r1_95", "d_map", "decision")})
                          for k, v in result["S1_zoom"].items()}
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
