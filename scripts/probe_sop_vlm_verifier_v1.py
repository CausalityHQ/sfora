#!/usr/bin/env python3
"""VLM pairwise verifier ceiling probe on SOP TRAIN-held (Qwen3-VL-8B-Instruct as a "same product?" judge).

Ceiling probe, not a serving candidate: SIGNAL iff ΔR@1 lower95 > 0; NO-SIGNAL otherwise. SIGNAL means rank-1 errors are partly fixable by fine-grained pairwise evidence; NO-SIGNAL raises the near-duplicate-ceiling hypothesis.

Pre-registered (committed before any number exists):
  base   : cache global_512.npy (59,551 unit rows, archive order); packed int8 cosine (codes x inverse norms, stable
           descending, self excluded). Held per-query R@1 must equal receipt quality.per_query_r1, else exit 1.
  split  : deterministic_class_partition(train_labels, fit 0.9, seed 179019): held 5,851 vs all 59,551; fit queries
           vs the fit gallery.
  gate   : query is ambiguous iff base top-1 score - top-2 score < tau. Only ambiguous queries are touched.
  verify : per ambiguous query, each of its top-K=3 candidates gets ONE forward pass of Qwen3-VL-8B-Instruct (bf16,
           local snapshot) on [query image, candidate image (longer side <= 448 px), PROMPT]; v = max yes-variant
           logit - max no-variant logit at the assistant answer position.
  fuse   : the K candidates re-ordered by z(cosine) + alpha z(v) (z within the K; alpha = inf: v only), stable;
           ranks beyond K unchanged.
  choose : tau in {0.02, 0.05, 0.10}, alpha in {0.5, 1, 2, inf} by R@1 on <= 600 fit queries (seed 0) sampled among
           fit queries ambiguous at the largest tau kept, gallery = fit gallery; ties -> smaller alpha, then smaller
           tau. VLM scores cached, so the grid costs no extra passes.
  budget : if K x (fit sample + held ambiguous at the largest tau) > 6,000 passes, drop the largest tau (recorded).
  held   : ONE read at the chosen (tau, alpha) over all 5,851 held queries (ungated unchanged); per-query R@1 and
           AP@R vs base, paired product bootstrap (5000 draws, seed 179019, resample held products) in pp.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_sop_token_cache_v1 as s  # noqa: E402
import train_sop_token_reranker_v1 as r  # noqa: E402
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402

K, MAX_SIDE, FIT_N, MAX_PASSES, CHECKPOINT_EVERY, ERROR_CASES = 3, 448, 600, 6000, 500, 40
TAUS, ALPHAS = (0.02, 0.05, 0.10), (0.5, 1.0, 2.0, math.inf)
PROMPT = ("Do these two photos show the exact same product (same model and variant, possibly from a different "
          "angle)? Answer yes or no.")


def yes_no_ids(tokenizer) -> dict:
    """{"yes": {variant: id}, "no": {...}} over case / leading-space variants that encode to exactly one token."""
    out = {}
    for word in ("yes", "no"):
        ids = {}
        for variant in (word, word.capitalize(), " " + word, " " + word.capitalize()):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if len(t) == 1:
                ids[variant] = int(t[0])
        if not ids:
            raise ValueError(f"no single-token id for {word!r}")
        out[word] = ids
    if set(out["yes"].values()) & set(out["no"].values()):
        raise ValueError("yes and no ids overlap")
    return out


def verdict(logits: torch.Tensor, ids: dict) -> float:
    """Next-token logits (V,) -> max over yes variants minus max over no variants."""
    x = logits.float()
    return float(x[list(ids["yes"].values())].max() - x[list(ids["no"].values())].max())


def fuse_perm(cos: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """(n,K) cosines and verifier scores -> stable descending permutation of z(cos) + alpha z(v); inf = v only."""
    key = v.float() if math.isinf(alpha) else s.zscore(cos.float()) + alpha * s.zscore(v.float())
    return torch.argsort(key, dim=1, descending=True, stable=True)


def gated_rerank(ranked: torch.Tensor, cos: torch.Tensor, v: torch.Tensor, gate: torch.Tensor,
                 alpha: float) -> torch.Tensor:
    """Re-order the first K of gated rows; ungated rows and ranks beyond K keep the base order."""
    out = ranked.clone()
    if bool(gate.any()):
        if bool(v[gate].isnan().any()):
            raise ValueError("gated row without verifier scores")
        out[gate] = r.apply_rerank(ranked[gate], fuse_perm(cos[gate, :K], v[gate], alpha))
    return out


def choose(grid: dict) -> tuple:
    """grid {(tau, alpha): fit R@1} -> best key; ties to the smaller alpha, then the smaller tau."""
    return max(grid, key=lambda ta: (grid[ta], -ta[1], -ta[0]))


def name(tau: float, alpha: float) -> str:
    return f"tau={tau},alpha={'inf' if math.isinf(alpha) else alpha}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("cache-dir", "archive", "dataset-root", "receipt", "vlm-snapshot", "output-dir"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--resume-cache", type=Path, help="vlm_cache.json from a crashed run (same pairs, same model)")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict = {"schema": "sop-vlm-verifier-ceiling-probe-v1", "official_read": False, "serving_candidate": False,
                    "seconds": {}, "K": K, "prompt": PROMPT}

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
    fit_np = np.asarray(part.fit_row_indexes, dtype=np.int64)
    rows = len(labels_np)
    unit_np = np.load(a.cache_dir / "global_512.npy")
    if rows != 59_551 or len(held_np) != 5_851 or unit_np.shape != (rows, 512):
        raise ValueError("SOP TRAIN partition or cache differs")
    unit = torch.from_numpy(unit_np)
    labels = torch.from_numpy(labels_np).cuda()
    held, fit = torch.from_numpy(held_np).cuda(), torch.from_numpy(fit_np).cuda()
    every = torch.arange(rows, device="cuda")
    products = labels_np[held_np]

    held_ranked, held_cos = r.candidate_lists(s.packed_scorer(unit, every), held, every, rows)
    base_r1, base_ap, base_rec = s.metrics(held_ranked, held, every, labels)
    result["baseline"] = {"recall_at_1": 100 * float(base_r1.mean()), "map_at_r": 100 * float(base_ap.mean()),
                          "held_recall_at_k": base_rec}
    if not np.array_equal(base_r1, np.asarray(json.loads(a.receipt.read_text())["quality"]["per_query_r1"],
                                              dtype=np.float64)):
        result["decision"] = "INVALID: baseline does not reproduce receipt per-query R@1"
        save()
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(1)
    fit_ranked, fit_cos = r.candidate_lists(s.packed_scorer(unit, fit), fit, fit, rows)
    torch.cuda.empty_cache()
    held_gap = (held_cos[:, 0] - held_cos[:, 1]).cpu()
    fit_gap = (fit_cos[:, 0] - fit_cos[:, 1]).cpu()
    result["ambiguous_counts"] = {str(tau): {"held": int((held_gap < tau).sum()), "fit": int((fit_gap < tau).sum())}
                                  for tau in TAUS}

    # Budget guard (label-free: uses only base score gaps).
    taus, dropped = list(TAUS), []
    while True:
        n_fit = min(FIT_N, int((fit_gap < taus[-1]).sum()))
        estimate = K * (n_fit + int((held_gap < taus[-1]).sum()))
        if estimate <= MAX_PASSES or len(taus) == 1:
            break
        dropped.append(taus.pop())
    result["budget"] = {"taus": taus, "dropped_taus": dropped, "estimated_passes_upper": estimate,
                        "max_passes": MAX_PASSES, "over_budget_after_drops": estimate > MAX_PASSES}
    fit_sample_np = np.sort(np.random.default_rng(0).choice(
        np.flatnonzero(fit_gap.numpy() < taus[-1]), size=n_fit, replace=False))  # positions into fit
    fs = torch.from_numpy(fit_sample_np).cuda()
    fq, f_ranked, f_cos, f_gap = fit[fs], fit_ranked[fs], fit_cos[fs], fit_gap[fit_sample_np].cuda()
    result["fit_sample"] = {"queries": n_fit, "sampled_among_ambiguous_at_tau": taus[-1]}
    t = lap("lists", t)

    # VLM (scores cached by (query row, candidate row); checkpointed every CHECKPOINT_EVERY passes).
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(a.vlm_snapshot, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm_snapshot, dtype=torch.bfloat16,
                                                            local_files_only=True).cuda().eval()
    ids = yes_no_ids(processor.tokenizer)
    result["token_ids"] = ids
    blank = Image.new("RGB", (8, 8))
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": blank}, {"type": "image", "image": blank},
                                      {"type": "text", "text": PROMPT}]}], tokenize=False, add_generation_prompt=True)
    result["chat_text"] = text
    t = lap("vlm_load", t)

    @functools.lru_cache(maxsize=64)
    def image(row: int) -> Image.Image:
        with Image.open(a.dataset_root / relatives[row]) as im:
            rgb = im.convert("RGB")
        rgb.thumbnail((MAX_SIDE, MAX_SIDE))
        return rgb

    cache: dict = {}
    if a.resume_cache is not None:
        cache = {(int(q), int(c)): float(v) for q, c, v in json.loads(a.resume_cache.read_text())}
    stats = {"passes": 0, "seconds": 0.0, "resumed": len(cache)}
    result["vlm"] = stats

    def checkpoint() -> None:
        (a.output_dir / "vlm_cache.json").write_text(json.dumps([[q, c, v] for (q, c), v in cache.items()]))
        stats["seconds_per_pass"] = stats["seconds"] / max(stats["passes"], 1)
        save()

    @torch.inference_mode()
    def verify(queries: torch.Tensor, ranked: torch.Tensor) -> torch.Tensor:
        """(n,) query rows, (n,>=K) candidate rows -> (n,K) v, running the pairs not yet cached."""
        q_list, c_list = queries.tolist(), ranked[:, :K].tolist()
        for q, cands in zip(q_list, c_list):
            for c in cands:
                if (q, c) in cache:
                    continue
                t0 = time.perf_counter()
                x = processor(text=[text], images=[image(q), image(c)], return_tensors="pt").to("cuda")
                cache[(q, c)] = verdict(model(**x, logits_to_keep=1).logits[0, -1], ids)
                stats["seconds"] += time.perf_counter() - t0
                stats["passes"] += 1
                if stats["passes"] % CHECKPOINT_EVERY == 0:
                    checkpoint()
                    print(json.dumps({"passes": stats["passes"], "s_per_pass": stats["seconds_per_pass"]}), flush=True)
        return torch.tensor([[cache[(q, c)] for c in cands] for q, cands in zip(q_list, c_list)], device="cuda")

    def full_v(queries, ranked, gate) -> torch.Tensor:
        v = torch.full((len(queries), K), math.nan, device="cuda")
        if bool(gate.any()):
            v[gate] = verify(queries[gate], ranked[gate])
        return v

    # FIT grid.
    f_v = verify(fq, f_ranked)
    checkpoint()
    fb = s.metrics(f_ranked, fq, fit, labels)
    grid, table = {}, {"baseline": [100 * float(fb[0].mean()), 100 * float(fb[1].mean())]}
    for tau in taus:
        for alpha in ALPHAS:
            r1, ap, _ = s.metrics(gated_rerank(f_ranked, f_cos, f_v, f_gap < tau, alpha), fq, fit, labels)
            grid[(tau, alpha)] = 100 * float(r1.mean())
            table[name(tau, alpha)] = [grid[(tau, alpha)], 100 * float(ap.mean())]
    tau, alpha = choose(grid)
    result["fit_grid_r1_map"] = table
    result["chosen"] = {"tau": tau, "alpha": "inf" if math.isinf(alpha) else alpha}
    t = lap("fit", t)

    # Single held read.
    gate = (held_gap < tau).cuda()
    h_v = full_v(held, held_ranked, gate)
    checkpoint()
    new_ranked = gated_rerank(held_ranked, held_cos, h_v, gate, alpha)
    r1, ap, rec = s.metrics(new_ranked, held, every, labels)
    out = s.compare((base_r1, base_ap), (r1, ap), products)
    out["decision"] = "SIGNAL" if out["d_r1_95"][0] > 0 else "NO-SIGNAL"
    gate_np = gate.cpu().numpy()
    result["held"] = {"gated_queries": int(gate_np.sum()), "held_recall_at_k": rec,
                      "fixed": int(((base_r1 == 0) & (r1 == 1)).sum()), "broken": int(((base_r1 == 1) & (r1 == 0)).sum()),
                      **out}
    result["decision"] = out["decision"]

    # Gate diagnostics on held errors.
    match = (labels[held_ranked[:, :K]] == labels[held][:, None]).cpu().numpy()
    vh = h_v.cpu().numpy()
    err_in = (base_r1 == 0) & gate_np

    def yes_rate(mask: np.ndarray) -> dict:
        return {"pairs": int(mask.sum()), "yes_rate": float((vh[mask] > 0).mean()) if mask.any() else None,
                "mean_v": float(vh[mask].mean()) if mask.any() else None}

    pairs_err = np.repeat(err_in[:, None], K, 1)
    pairs_gate = np.repeat(gate_np[:, None], K, 1)
    result["held_gate_diagnostics"] = {
        "held_errors": int((base_r1 == 0).sum()), "held_errors_in_gate": int(err_in.sum()),
        "errors_in_gate_positive_in_top_k": float(match[err_in].any(1).mean()) if err_in.any() else None,
        "errors_in_gate_true_pairs": yes_rate(pairs_err & match),
        "errors_in_gate_false_pairs": yes_rate(pairs_err & ~match),
        "all_gated_true_pairs": yes_rate(pairs_gate & match), "all_gated_false_pairs": yes_rate(pairs_gate & ~match),
    }
    err_idx = np.flatnonzero(err_in)
    pick = np.sort(np.random.default_rng(0).choice(err_idx, size=min(ERROR_CASES, len(err_idx)), replace=False))
    hr, hc = held_ranked[:, :K].cpu().numpy(), held_cos[:, :K].cpu().numpy()
    cases = []
    for i in pick:
        true_k = np.flatnonzero(match[i])
        cases.append({"query": relatives[held_np[i]], "wrong_top1": relatives[hr[i, 0]],
                      "true_positive_in_top_k": relatives[hr[i, true_k[0]]] if len(true_k) else None,
                      "true_positive_rank": int(true_k[0]) + 1 if len(true_k) else None,
                      "candidates": [relatives[c] for c in hr[i]], "cosines": hc[i].tolist(), "v": vh[i].tolist(),
                      "match": match[i].tolist(), "fixed": bool(r1[i] == 1)})
    (a.output_dir / "error_cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    lap("held", t)
    print(json.dumps({k: result[k] for k in ("baseline", "budget", "chosen", "held", "held_gate_diagnostics", "vlm",
                                             "decision", "seconds", "peak_cuda_allocated_gib", "peak_host_rss_gib")},
                     indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
