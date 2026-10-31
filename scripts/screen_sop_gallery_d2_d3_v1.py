#!/usr/bin/env python3
"""Two cached-data gallery screens on SOP TRAIN-held (docs/gallery_linking_proposals_2026-10-31.md, D2 and D3).

Pre-registered (committed before any number exists):
  data    : cache global_512.npy (59,551 unit rows, archive order) and tokens_9x9_fp16.npy (59,551 x 81 x 1152).
            Labels = archive train_labels; deterministic_class_partition(fit 0.9, seed 179019); 5,851 held queries.
  base    : packed int8 cosine (codes x inverse norms, stable descending, self excluded), top-32 per query.
            G_full = all 59,551 images; G_held = the 5,851 held images. Both reported. G_full held per-query R@1 must
            equal receipt quality.per_query_r1 exactly, else exit 1.
  CV      : held products split into 5 folds (permutation seed 0, fold = position mod 5). Per fold the knobs with the
            best mean R@1 on the other 4 folds (then mAP@R, then the earlier grid entry) score that fold; the held-out
            per-query results are concatenated. Paired product bootstrap vs base (5000 draws, seed 179019), pp.

  D2 competition normalisation (label-free at index time):
    rho(c)    = mean packed cosine of gallery image c to its neighbours at ranks K1..K2 (1-based, c excluded) in the
                same gallery, LEAVE-QUERY-OUT: q is removed from c's neighbour list before ranks are counted (exact).
                (K1, K2) in {(6, 50), (11, 100)}.
    detail(c) = ridge (lambda 1.0, features z-scored on FIT) from [512-D vector, mean and std of the 81 raw token
                norms] to the mean (1 - float cosine) to same-product mates, fitted on FIT images with >= 1 mate;
                detail = max(pred, 0) / mean over FIT of max(pred, 0).
    s'(q, c)  = cos(q, c) - beta * rho(c) * (1 + gamma * detail(c)), beta in {0.1, 0.25, 0.5}, gamma in {0, 1, 2};
                top-32 re-ranked by s' (stable), ranks beyond 32 unchanged.
    AUC       = over held base errors with a positive in the top-32: P(rho(wrong top-1) > rho(best-ranked positive)),
                ties 1/2, per K; the KILL uses the larger of the two.
    KILL if CV ΔR@1 < +0.2, or its 95 % interval includes 0, or that AUC ≤ 0.58.

  D3 oracle ceiling (uses TRUE labels; a ceiling, never a method):
    tokens    : PCA256 fitted on FIT image tokens, L2 per token.
    groups    : the query's top-32 grouped by true product; a group's best member = its highest-ranked candidate.
    union     : mean over the query's 81 tokens of the max cosine over all tokens of ALL gallery images of that
                product (whole gallery, query image excluded).
    order     : groups by z(best member cosine) + w * z(feature), z over the query's groups, stable (ties keep base
                order); members keep base order inside a group; w in {0.25, 0.5, 1, 2} by the same 5-fold CV.
                Rank-1 = best member of the top group.
    control   : best-member MaxSim = max single-image MaxSim over the group's top-32 members. Diagnostic only:
                the same max over all gallery images of the product (best_member_all).
    KILL the D3 lane if oracle ΔR@1 < +1.0, or union minus best-member < +0.3.
  Decisions are reported per gallery.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_sop_token_cache_v1 as s  # noqa: E402
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402

TOP, FOLDS, SEED = 32, 5, 179019
KS = ((6, 50), (11, 100))
BETAS, GAMMAS = (0.1, 0.25, 0.5), (0.0, 1.0, 2.0)
D2_GRID = [(k, b, g) for k in KS for b in BETAS for g in GAMMAS]
D3_W = (0.25, 0.5, 1.0, 2.0)
RIDGE = 1.0


@torch.inference_mode()
def topn(score, queries, gallery, rows: int, n: int, block: int = 64):
    """(Q,n) gallery row ids and scores by stable descending score(q) (b, len(gallery)); self excluded (as s.rank)."""
    col = torch.full((rows,), -1, dtype=torch.long, device=gallery.device)
    col[gallery] = torch.arange(len(gallery), device=gallery.device)
    if int(col[queries].min()) < 0:
        raise ValueError("every query must be in the gallery")
    ids, val = [], []
    for q in queries.split(block):
        sc = score(q)
        sc[torch.arange(len(q), device=sc.device), col[q]] = -torch.inf
        r = torch.argsort(sc, dim=1, descending=True, stable=True)[:, :n]
        ids.append(gallery[r])
        val.append(sc.gather(1, r).float())
    return torch.cat(ids), torch.cat(val)


def loo_rho(nbr_ids, nbr_cos, cand, queries, k1: int, k2: int):
    """rho (Q,C) of candidate rows cand: mean cosine at neighbour ranks k1..k2 after removing the query from each
    candidate's neighbour list. nbr_* (rows, >= k2+1) hold each gallery row's neighbours, self excluded."""
    ids, cs = nbr_ids[cand][..., :k2 + 1], nbr_cos[cand][..., :k2 + 1]
    keep = ids != queries[:, None, None]
    kr = keep.cumsum(-1)
    return (cs * (keep & (kr >= k1) & (kr <= k2))).sum(-1) / (k2 - k1 + 1)


def rerank_head(ranked, key):
    """Stable descending re-order of the first key.shape[1] columns of ranked by key; the tail is unchanged."""
    perm = torch.argsort(key, dim=1, descending=True, stable=True)
    return torch.cat([ranked[:, :key.shape[1]].gather(1, perm), ranked[:, key.shape[1]:]], 1)


def product_folds(products: np.ndarray, k: int = FOLDS, seed: int = 0) -> np.ndarray:
    """Fold id per query; whole products are assigned (seeded permutation of unique products, position mod k)."""
    u, inv = np.unique(products, return_inverse=True)
    fold_of = np.empty(len(u), np.int64)
    fold_of[np.random.default_rng(seed).permutation(len(u))] = np.arange(len(u)) % k
    return fold_of[inv]


def cv_select(r1: np.ndarray, ap: np.ndarray, folds: np.ndarray):
    """r1, ap (configs, Q) -> held-out per-query (r1, ap) and the config index chosen per fold."""
    out_r1, out_ap, chosen = np.empty(r1.shape[1]), np.empty(r1.shape[1]), []
    for f in range(int(folds.max()) + 1):
        tr, te = folds != f, folds == f
        best = max(range(len(r1)), key=lambda i: (r1[i, tr].mean(), ap[i, tr].mean(), -i))
        out_r1[te], out_ap[te] = r1[best, te], ap[best, te]
        chosen.append(best)
    return out_r1, out_ap, chosen


def mate_distance(unit: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean (1 - cosine) of each unit row to the other rows with its label; NaN for singletons."""
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    u = torch.from_numpy(unit).double()
    sums = torch.zeros(len(cnt), u.shape[1], dtype=torch.float64).index_add_(0, torch.from_numpy(inv), u)
    mates = ((sums[inv] * u).sum(1) - (u * u).sum(1)).numpy()
    n = cnt[inv] - 1
    return np.where(n > 0, 1.0 - mates / np.maximum(n, 1), np.nan)


def ridge_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, lam: float = RIDGE) -> np.ndarray:
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-12
    z = (x - mu) / sd
    zt, ym = z[train], y[train].mean()
    w = np.linalg.solve(zt.T @ zt + lam * np.eye(z.shape[1]), zt.T @ (y[train] - ym))
    return z @ w + ym


def error_auc(rho, match, base_r1: np.ndarray) -> dict:
    """Over base errors with a positive in the list: P(rho(wrong top-1) > rho(best-ranked positive)), ties 1/2."""
    err = torch.from_numpy(base_r1 == 0).to(match.device) & match.any(1)
    first = match.int().argmax(1)
    rw, rp = rho[:, 0][err], rho.gather(1, first[:, None])[:, 0][err]
    return {"errors": int(err.sum()), "auc": float(((rw > rp).double() + 0.5 * (rw == rp).double()).mean())}


def group_slots(lab):
    """(Q,C) candidate labels -> (Q,C) position of the first candidate with the same label (the group's slot)."""
    return (lab[:, :, None] == lab[:, None, :]).int().argmax(2)


def group_perm(slot, gcos, feat, w: float):
    """Permutation (Q,C): groups ordered by z(gcos) + w z(feat) over the query's groups (values read at each group's
    slot), ties by slot; candidates keep base order inside a group."""
    c = slot.shape[1]
    ar = torch.arange(c, device=slot.device)
    valid = slot == ar
    n = valid.sum(1, keepdim=True)

    def z(x):
        x = torch.where(valid, x.double(), 0.0)
        m = x.sum(1, keepdim=True) / n
        sd = (torch.where(valid, x - m, 0.0) ** 2).sum(1, keepdim=True).div(n).sqrt().clamp_min(1e-12)
        return (x - m) / sd

    key = torch.where(valid, z(gcos) + w * z(feat), -torch.inf)
    order = torch.argsort(key, dim=1, descending=True, stable=True)
    grank = torch.empty_like(order).scatter_(1, order, ar.expand_as(order))
    return torch.argsort(grank.gather(1, slot) * c + ar, dim=1)


def union_and_best(qtok, img_tok, seg, n_seg: int):
    """qtok (T,D), img_tok (n,S,D) unit tokens, seg (n,) group index -> per group (union MaxSim, best single-image
    MaxSim), -inf for empty groups."""
    m = torch.einsum("td,nsd->nts", qtok.float(), img_tok.float()).amax(-1)
    u = torch.full((n_seg, m.shape[1]), -torch.inf, device=m.device)
    u = u.scatter_reduce(0, seg[:, None].expand_as(m), m, "amax")
    b = torch.full((n_seg,), -torch.inf, device=m.device).scatter_reduce(0, seg, m.mean(1), "amax")
    return u.mean(1), b


@torch.inference_mode()
def oracle_features(tok, ranked, queries, labels, labels_np: np.ndarray, gallery_np: np.ndarray):
    """(Q,C) per group slot: union MaxSim and best single MaxSim over all gallery images of the product (query
    excluded), and best single MaxSim over the group's top-C members."""
    slot = group_slots(labels[ranked])
    q_count, c = ranked.shape
    union = torch.zeros(q_count, c, device=ranked.device)
    best_all = torch.zeros(q_count, c, device=ranked.device)
    order = gallery_np[np.argsort(labels_np[gallery_np], kind="stable")]
    sorted_labels = labels_np[order]
    ranked_np, slot_np, q_np = ranked.cpu().numpy(), slot.cpu().numpy(), queries.cpu().numpy()
    for i, q in enumerate(q_np):
        groups = np.flatnonzero(slot_np[i] == np.arange(c))
        lab = labels_np[ranked_np[i, groups]]
        lo, hi = np.searchsorted(sorted_labels, lab, "left"), np.searchsorted(sorted_labels, lab, "right")
        parts = [order[a:b] for a, b in zip(lo, hi)]
        seg = np.concatenate([np.full(len(p), g) for p, g in zip(parts, groups)])
        rows = np.concatenate(parts)
        keep = rows != q
        u, b = union_and_best(tok[int(q)], tok[torch.from_numpy(rows[keep]).to(tok.device)],
                              torch.from_numpy(seg[keep]).to(tok.device), c)
        union[i], best_all[i] = u.nan_to_num(neginf=0.0), b.nan_to_num(neginf=0.0)
    single = torch.cat([s.maxsim(tok[q], tok[r]) for q, r in zip(queries.split(64), ranked.split(64))])
    best_group = torch.full_like(single, -torch.inf).scatter_reduce(1, slot, single, "amax")
    return slot, union, best_all, best_group.nan_to_num(neginf=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ("cache-dir", "archive", "receipt", "output-dir"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict = {"schema": "sop-gallery-d2-d3-screen-v1", "official_read": False, "seconds": {},
                    "d2_grid": [[list(k), b, g] for k, b, g in D2_GRID], "d3_w": list(D3_W)}

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
    part = deterministic_class_partition(tuple(map(int, labels_np)), fit_fraction=0.9, seed=179019)
    held_np = np.asarray(part.validation_row_indexes, dtype=np.int64)
    fit_np = np.asarray(part.fit_row_indexes, dtype=np.int64)
    rows = len(labels_np)
    unit_np = np.load(a.cache_dir / "global_512.npy")
    tokens = np.load(a.cache_dir / "tokens_9x9_fp16.npy", mmap_mode="r")
    if rows != 59_551 or len(held_np) != 5_851 or unit_np.shape != (rows, 512) or tokens.shape != (rows, 81, 1152):
        raise ValueError("SOP TRAIN partition or cache differs")
    dev = torch.device("cuda")
    labels = torch.from_numpy(labels_np).to(dev)
    held = torch.from_numpy(held_np).to(dev)
    every = torch.arange(rows, device=dev)
    unit = torch.from_numpy(unit_np)
    products = labels_np[held_np]
    folds = product_folds(products)
    result["folds"] = {"seed": 0, "queries": np.bincount(folds).tolist(),
                       "products": [int(len(np.unique(products[folds == f]))) for f in range(FOLDS)]}

    # Tokens: PCA256 on FIT, unit tokens for D3; raw token-norm mean/std for D2 detail.
    def chunks(index):
        for i in range(0, len(index), 512):
            yield torch.from_numpy(np.asarray(tokens[index[i:i + 512]])).to(dev).reshape(-1, 1152)

    mean, axes = s.pca(chunks(fit_np), 256)
    tok = torch.empty(rows, 81, 256, dtype=torch.float16, device=dev)
    norm_stats = torch.empty(rows, 2, dtype=torch.float64)
    for i in range(0, rows, 512):
        x = torch.from_numpy(np.asarray(tokens[i:i + 512])).to(dev).float()
        nrm = x.norm(dim=-1)
        norm_stats[i:i + 512] = torch.stack([nrm.mean(1), nrm.std(1, unbiased=False)], 1).double().cpu()
        tok[i:i + 512] = F.normalize((x - mean) @ axes, dim=-1).half()
    del x
    t = lap("tokens", t)

    # detail(c): ridge on FIT images only.
    target = np.full(rows, np.nan)
    target[fit_np] = mate_distance(unit_np[fit_np], labels_np[fit_np])
    train = np.zeros(rows, bool)
    train[fit_np] = True
    train &= np.isfinite(target)
    pred = ridge_predict(np.concatenate([unit_np.astype(np.float64), norm_stats.numpy()], 1), target, train)
    pos = np.maximum(pred, 0.0)
    detail_np = pos / pos[fit_np].mean()
    held_target = mate_distance(unit_np[held_np], labels_np[held_np])  # diagnostic only

    def r2(y, f):
        ok = np.isfinite(y)
        return float(1 - ((y[ok] - f[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())

    result["detail"] = {"fit_rows": int(train.sum()), "fit_r2": r2(target[train], pred[train]),
                        "held_r2": r2(held_target, pred[held_np]),
                        "held_pearson": float(np.corrcoef(held_target[np.isfinite(held_target)],
                                                                     pred[held_np][np.isfinite(held_target)])[0, 1]),
                        "clamped_fraction": float((pred < 0).mean())}
    detail = torch.from_numpy(detail_np).float().to(dev)
    t = lap("detail", t)

    receipt_r1 = np.asarray(json.loads(a.receipt.read_text())["quality"]["per_query_r1"], dtype=np.float64)
    summary: dict = {}
    for gname, gallery in (("G_full", every), ("G_held", held)):
        out: dict = {}
        result[gname] = out
        scorer = s.packed_scorer(unit, gallery)
        ranked, cos = topn(scorer, held, gallery, rows, TOP)
        base_r1, base_ap, base_rec = s.metrics(ranked, held, gallery, labels)
        base = (base_r1, base_ap)
        out["baseline"] = {"recall_at_1": 100 * float(base_r1.mean()), "map_at_r": 100 * float(base_ap.mean()),
                           "held_recall_at_k": base_rec}
        if gname == "G_full" and not np.array_equal(base_r1, receipt_r1):
            result["decision"] = "INVALID: baseline does not reproduce receipt per-query R@1"
            save()
            print(json.dumps(result, sort_keys=True), flush=True)
            raise SystemExit(1)
        match = labels[ranked] == labels[held][:, None]
        t = lap(f"{gname}_baseline", t)

        def cv(r1s, aps, names, ref=base) -> dict:
            r1, ap, chosen = cv_select(np.stack(r1s), np.stack(aps), folds)
            res = s.compare(ref, (r1, ap), products)
            res.pop("decision")
            res["chosen_per_fold"] = [names[i] for i in chosen]
            return res, (r1, ap)

        # D2.
        nbr_n = KS[-1][1] + 1
        g_ids, g_cos = topn(scorer, gallery, gallery, rows, nbr_n, block=512)
        nbr_ids = torch.full((rows, nbr_n), -1, dtype=torch.long, device=dev)
        nbr_cos = torch.zeros(rows, nbr_n, device=dev)
        nbr_ids[gallery], nbr_cos[gallery] = g_ids, g_cos
        del g_ids, g_cos
        rho = {k: loo_rho(nbr_ids, nbr_cos, ranked, held, *k) for k in KS}
        del nbr_ids, nbr_cos
        det = detail[ranked]
        r1s, aps, grid = [], [], {}
        for k, b, g in D2_GRID:
            r1, ap, _ = s.metrics(rerank_head(ranked, cos - b * rho[k] * (1 + g * det)), held, gallery, labels)
            r1s.append(r1)
            aps.append(ap)
            grid[f"K={k},beta={b},gamma={g}"] = [100 * float(r1.mean()), 100 * float(ap.mean())]
        d2, _ = cv(r1s, aps, list(grid))
        auc = {str(k): error_auc(rho[k], match, base_r1) for k in KS}
        auc_kill = max(v["auc"] for v in auc.values())
        d2["decision"] = ("KILL" if d2["d_r1"] < 0.2 or d2["d_r1_95"][0] <= 0 or auc_kill <= 0.58 else "PASS")
        out["D2"] = {"in_sample_grid_r1_map": grid, "error_auc": auc, "auc_for_kill": auc_kill, "cv": d2}
        del rho
        t = lap(f"{gname}_D2", t)

        # D3 oracle.
        slot, union, best_all, best_group = oracle_features(tok, ranked, held, labels, labels_np,
                                                            gallery.cpu().numpy())
        d3, cv_r1 = {}, {}
        for method, feat in (("union", union), ("best_member", best_group), ("best_member_all", best_all)):
            r1s, aps = [], []
            for w in D3_W:
                r1, ap, _ = s.metrics(ranked.gather(1, group_perm(slot, cos, feat, w)), held, gallery, labels)
                r1s.append(r1)
                aps.append(ap)
            d3[method], cv_r1[method] = cv(r1s, aps, list(D3_W))
        diff, _ = cv([cv_r1["union"][0]], [cv_r1["union"][1]], ["union"], ref=cv_r1["best_member"])
        d3["union_minus_best_member"] = diff
        d3["decision"] = ("KILL" if d3["union"]["d_r1"] < 1.0 or diff["d_r1"] < 0.3 else "PASS")
        out["D3_oracle"] = d3
        t = lap(f"{gname}_D3", t)
        summary[gname] = {"baseline_r1": out["baseline"]["recall_at_1"],
                          "D2": {x: d2[x] for x in ("d_r1", "d_r1_95", "d_map", "chosen_per_fold", "decision")},
                          "D2_auc_for_kill": auc_kill,
                          "D3": {m: {x: d3[m][x] for x in ("d_r1", "d_r1_95", "d_map")}
                                 for m in ("union", "best_member", "best_member_all", "union_minus_best_member")},
                          "D3_decision": d3["decision"]}
        del scorer
        torch.cuda.empty_cache()
    result["summary"] = summary
    save()
    print(json.dumps({"summary": summary, "detail": result["detail"], "seconds": result["seconds"],
                      "peak_cuda_allocated_gib": result["peak_cuda_allocated_gib"]}, indent=1, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
