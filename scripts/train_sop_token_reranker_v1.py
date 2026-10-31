#!/usr/bin/env python3
"""Learned pairwise re-ranker over cached 9x9 patch tokens (SOP TRAIN, fine-tuned SigLIP2 So400m@384/512).

Pre-registered (committed before any number exists; nothing below is tuned on held):
  data   : cache from screen_sop_token_cache_v1 (global_512.npy, tokens_9x9_fp16.npy); labels = archive
           train_labels; split = deterministic_class_partition(labels, fit 0.9, seed 179019): held 5,851, fit 53,700.
  base   : packed int8 512-D cosine (codes x inverse norms, stable descending, self excluded). Held per-query R@1
           must equal receipt quality.per_query_r1 exactly, else INVALID and exit 1 (checked before training).
  lists  : every FIT query -> top-32 of the FIT gallery (self excluded) by the base score. Inner split of FIT
           products 90/10 (deterministic_class_partition over fit labels, seed 0). Inner-train lists with >=1
           positive are the training set; inner-val queries (all of them, gallery = full fit gallery) are used only
           for early stopping / checkpoint choice.
  model  : 81 query + 81 candidate tokens, shared Linear(1152->256), + segment embedding + 9x9 position embedding,
           + CLS (+ Linear(1->256)(pair base cosine)); 2-layer pre-norm transformer (256, 4 heads, MLP 1024,
           dropout 0.1); score = Linear(256->1)(CLS) + base cosine x learned scale (init 10).
  train  : listwise softmax CE over the 32 (target uniform over positives) + 0.5 x per-pair BCE; AdamW 3e-4,
           wd 0.01, cosine on progress = max(step / 6 passes, train seconds / 1500); bf16 autocast; 8 lists per
           step, sampled with replacement, weight 4 if the base top-1 is wrong else 1. Inner-val R@1 every 2,000
           steps and at the end; best by inner-val R@1 then mAP@R.
  held   : once, chosen checkpoint: held query -> top-32 of the full 59,551 gallery (self excluded), re-ordered by
           the learned score (stable, ties keep base order), ranks below 32 unchanged; R@1 and AP@R per query.
           Primary = fp16 pre-projected tokens (81 x 256 x 2 B per image); also int8 per-token absmax gallery
           tokens (81 x 256 + 81 x 2 B per image).
  stats  : paired product bootstrap vs base (5000 draws, seed 179019, resample held products), dR@1 and dmAP@R pp.
  rule   : "PASS iff ΔR@1 lower95 > 0 AND ΔmAP point >= -0.2; official read only if also ΔR@1 point >= +0.5. Else KILL."
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_sop_token_cache_v1 as s  # noqa: E402
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402

TOP, GRID2, WIDTH, D = s.TOP, 81, s.WIDTH, 256
LISTS, LR, WD, PASSES, BUDGET_S, EVAL_EVERY, HARD_W = 8, 3e-4, 0.01, 6, 25 * 60, 2000, 4.0


class Reranker(nn.Module):
    def __init__(self, width: int = WIDTH, d: int = D, heads: int = 4, mlp: int = 1024, layers: int = 2,
                 dropout: float = 0.1, tokens: int = GRID2):
        super().__init__()
        self.proj = nn.Linear(width, d)
        self.segment = nn.Parameter(0.02 * torch.randn(2, d))
        self.position = nn.Parameter(0.02 * torch.randn(tokens, d))
        self.cls = nn.Parameter(0.02 * torch.randn(d))
        self.cosine = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d, heads, mlp, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def pair(self, q: torch.Tensor, c: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """Projected q (B,81,d), c (B,K,81,d), base cosine (B,K) -> scores (B,K); candidates are independent."""
        b, k = cos.shape
        qe = (q + self.segment[0] + self.position)[:, None].expand(b, k, -1, -1)
        ce = c + self.segment[1] + self.position
        cls = (self.cls + self.cosine(cos[..., None].to(self.cls.dtype)))[:, :, None]
        x = torch.cat([cls.to(ce.dtype), qe.to(ce.dtype), ce], 2).flatten(0, 1)
        return self.head(self.encoder(x)[:, 0]).view(b, k).float() + self.scale * cos.float()

    def forward(self, q_tokens, c_tokens, cos):
        return self.pair(self.proj(q_tokens), self.proj(c_tokens), cos)


def listwise_loss(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    pos = positive.float()
    ce = -((pos / pos.sum(1, keepdim=True)) * logits.float().log_softmax(1)).sum(1).mean()
    return ce + 0.5 * F.binary_cross_entropy_with_logits(logits.float(), pos)


def list_weights(top1_correct: torch.Tensor) -> torch.Tensor:
    return torch.where(top1_correct, 1.0, HARD_W)


def apply_rerank(ranked: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute the first perm.shape[1] columns of ranked by perm; the tail is unchanged."""
    k = perm.shape[1]
    return torch.cat([ranked[:, :k].gather(1, perm), ranked[:, k:]], 1)


def candidate_lists(score, queries, gallery, rows):
    """s.rank top-TOP row ids plus their base scores (identity re-rank captures the scores)."""
    saved = []

    def keep(_q, cand, sc):
        saved.append(sc.float())
        return torch.arange(cand.shape[1], device=cand.device).expand_as(cand)

    return s.rank(score, queries, gallery, rows, rerank=keep), torch.cat(saved)


def autocast(x: torch.Tensor):
    return torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda)


@torch.inference_mode()
def project(model: Reranker, tokens: torch.Tensor, index: torch.Tensor, out: torch.Tensor, chunk: int = 1024):
    model.eval()
    for i in range(0, len(index), chunk):
        r = index[i:i + chunk]
        with autocast(out):
            out[r] = model.proj(tokens[r]).to(out.dtype)
    return out


def quantize_int8(x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Per-token symmetric absmax int8 (fp16 scale), returned dequantized as fp16."""
    out = torch.empty_like(x)
    for i in range(0, len(x), chunk):
        v = x[i:i + chunk].float()
        scale = (v.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8).half().float()
        out[i:i + chunk] = ((v / scale).round().clamp(-127, 127) * scale).half()
    return out


@torch.inference_mode()
def rerank_perm(model: Reranker, qtok, ctok, queries, cand, cos, chunk: int = 64) -> torch.Tensor:
    """Stable descending order of learned scores per list (ties keep base order)."""
    model.eval()
    out = []
    for i in range(0, len(queries), chunk):
        with autocast(cos):
            sc = model.pair(qtok[queries[i:i + chunk]], ctok[cand[i:i + chunk]], cos[i:i + chunk])
        out.append(torch.argsort(sc, dim=1, descending=True, stable=True))
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cache-dir", "archive", "receipt", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    started = time.perf_counter()
    result: dict = {"schema": "sop-so400m-token-reranker-v1", "official_read": False, "seconds": {},
                    "inner_val_curve": []}

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

    t = time.perf_counter()
    with np.load(a.archive, allow_pickle=False) as archive:
        labels_np = np.asarray(archive["train_labels"], dtype=np.int64)
    receipt = json.loads(a.receipt.read_text())
    part = deterministic_class_partition(tuple(map(int, labels_np)), fit_fraction=0.9, seed=179019)
    held_np = np.asarray(part.validation_row_indexes, dtype=np.int64)
    fit_np = np.asarray(part.fit_row_indexes, dtype=np.int64)
    rows = len(labels_np)
    if rows != 59_551 or len(held_np) != 5_851 or len(fit_np) != 53_700:
        raise ValueError("SOP TRAIN partition differs")
    inner = deterministic_class_partition(tuple(map(int, labels_np[fit_np])), fit_fraction=0.9, seed=0)
    itrain_np = np.sort(fit_np[np.asarray(inner.fit_row_indexes, dtype=np.int64)])
    ival_np = np.sort(fit_np[np.asarray(inner.validation_row_indexes, dtype=np.int64)])

    unit_np = np.load(a.cache_dir / "global_512.npy")
    mm = np.load(a.cache_dir / "tokens_9x9_fp16.npy", mmap_mode="r")
    if unit_np.shape != (rows, 512) or mm.shape != (rows, GRID2, WIDTH) or mm.dtype != np.float16:
        raise ValueError("cache geometry differs")
    tokens = torch.empty(mm.shape, dtype=torch.float16, device="cuda")
    for i in range(0, rows, 4096):
        tokens[i:i + 4096] = torch.from_numpy(np.ascontiguousarray(mm[i:i + 4096])).cuda()
    del mm
    unit = torch.from_numpy(unit_np)
    labels = torch.from_numpy(labels_np).cuda()
    held, fit, itrain, ival = (torch.from_numpy(x).cuda() for x in (held_np, fit_np, itrain_np, ival_np))
    every = torch.arange(rows, device="cuda")
    products = labels_np[held_np]
    t = lap("load", t)

    # Baseline validity (global-only, no learned component): must reproduce the receipt before any training.
    base_score = s.packed_scorer(unit, every)
    held_ranked, held_cos = candidate_lists(base_score, held, every, rows)
    base_r1, base_ap, base_rec = s.metrics(held_ranked, held, every, labels)
    result["baseline"] = {"recall_at_1": 100 * float(base_r1.mean()), "map_at_r": 100 * float(base_ap.mean()),
                          "held_recall_at_k": base_rec}
    if not np.array_equal(base_r1, np.asarray(receipt["quality"]["per_query_r1"], dtype=np.float64)):
        result["decision"] = "INVALID: baseline does not reproduce receipt per-query R@1"
        save()
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(1)

    # FIT lists (fit gallery only); held labels/tokens are not touched again until the single held read.
    fit_ranked, fit_cos = candidate_lists(s.packed_scorer(unit, fit), fit, fit, rows)
    pos_of = torch.full((rows,), -1, dtype=torch.long, device="cuda")
    pos_of[fit] = torch.arange(len(fit), device="cuda")
    tr = pos_of[itrain]
    tr_pos = labels[fit_ranked[tr]] == labels[itrain][:, None]
    keep = tr_pos.any(1)
    tr_q, tr_cand, tr_cos, tr_pos = itrain[keep], fit_ranked[tr][keep], fit_cos[tr][keep], tr_pos[keep]
    weights = list_weights(tr_pos[:, 0])
    va = pos_of[ival]
    va_cand, va_cos = fit_ranked[va], fit_cos[va]
    vb = s.metrics(va_cand, ival, fit, labels)
    result["lists"] = {"inner_train_queries": len(itrain_np), "inner_train_lists": int(keep.sum()),
                       "inner_train_hard_lists": int((~tr_pos[:, 0]).sum()), "inner_val_queries": len(ival_np),
                       "inner_val_base": [100 * float(vb[0].mean()), 100 * float(vb[1].mean())]}
    t = lap("lists", t)

    model = Reranker().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = PASSES * math.ceil(len(tr_q) / LISTS)
    ptok = torch.empty(rows, GRID2, D, dtype=torch.float16, device="cuda")

    def inner_val(step: int, train_s: float) -> tuple:
        project(model, tokens, fit, ptok)
        r1, ap, _ = s.metrics(apply_rerank(va_cand, rerank_perm(model, ptok, ptok, ival, va_cand, va_cos)),
                              ival, fit, labels)
        point = (100 * float(r1.mean()), 100 * float(ap.mean()))
        result["inner_val_curve"].append({"step": step, "train_seconds": train_s, "r1": point[0], "map": point[1],
                                          "scale": float(model.scale)})
        save()
        print(json.dumps(result["inner_val_curve"][-1]), flush=True)
        return point

    best, best_key, step, train_s, losses = None, (-1.0, -1.0), 0, 0.0, []
    while max(step / total_steps, train_s / BUDGET_S) < 1:
        t0 = time.perf_counter()
        model.train()
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * max(step / total_steps, train_s / BUDGET_S)))
        i = torch.multinomial(weights, LISTS, replacement=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens[tr_q[i]], tokens[tr_cand[i]], tr_cos[i])
        loss = listwise_loss(logits, tr_pos[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        losses.append(loss.detach())
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        done = max(step / total_steps, train_s / BUDGET_S) >= 1
        if step % EVAL_EVERY == 0 or done:
            result["train_loss_mean_last"] = float(torch.stack(losses).mean())
            losses = []
            key = inner_val(step, train_s)
            if key > best_key:
                best_key, best = key, {"step": step, "state_dict": copy.deepcopy(model.state_dict())}
    result["training"] = {"steps": step, "total_steps_budget": total_steps, "train_seconds": train_s,
                          "passes": step * LISTS / len(tr_q), "best_step": best["step"],
                          "best_inner_val": list(best_key)}
    torch.save({"state_dict": best["state_dict"], "step": best["step"], "inner_val": best_key},
               a.output_dir / "best.pt")
    model.load_state_dict(best["state_dict"])
    t = lap("train", t)

    # Single held read with the chosen checkpoint: fp16 and int8 pre-projected gallery tokens.
    project(model, tokens, every, ptok)
    storage = {"fp16": GRID2 * D * 2, "int8": GRID2 * D + GRID2 * 2}
    variants = {}
    for name, ctok in (("fp16", ptok), ("int8", None)):
        ctok = quantize_int8(ptok) if ctok is None else ctok
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm = rerank_perm(model, ptok, ctok, held, held_ranked, held_cos)
        torch.cuda.synchronize()
        amortised = (time.perf_counter() - t0) / len(held_np)
        r1, ap, rec = s.metrics(apply_rerank(held_ranked, perm), held, every, labels)
        out = s.compare((base_r1, base_ap), (r1, ap), products)
        out["official_read_eligible"] = out["decision"] == "PASS" and out["d_r1"] >= 0.5
        variants[name] = {"bytes_per_image": storage[name], "held_recall_at_k": rec,
                          "amortised_ms_per_query": 1e3 * amortised, **out}
        del ctok
    result["held"] = variants
    result["decision"] = variants["fp16"]["decision"]
    result["official_read_eligible"] = variants["fp16"]["official_read_eligible"]
    t = lap("held", t)

    # Batch-1 latency: project one query's raw tokens, score 32 pre-projected fp16 candidates, sort.
    q, cand, cos = held[:1], held_ranked[:1], held_cos[:1]
    times = []
    with torch.inference_mode():
        for rep in range(220):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc = model.pair(model.proj(tokens[q]), ptok[cand], cos)
            torch.argsort(sc, dim=1, descending=True, stable=True)
            torch.cuda.synchronize()
            if rep >= 20:
                times.append(time.perf_counter() - t0)
    result["latency_batch1_ms"] = {"median": 1e3 * float(np.median(times)), "p90": 1e3 * float(np.percentile(times, 90))}
    lap("latency", t)
    print(json.dumps({k: result[k] for k in ("baseline", "lists", "training", "held", "decision",
                                             "latency_batch1_ms", "seconds", "peak_cuda_allocated_gib",
                                             "peak_host_rss_gib")}, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
