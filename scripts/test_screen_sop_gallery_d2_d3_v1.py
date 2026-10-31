#!/usr/bin/env python3
"""Tiny CPU checks for screen_sop_gallery_d2_d3_v1; no GPU, cache or images.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts python scripts/test_screen_sop_gallery_d2_d3_v1.py
"""
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before running this CPU-only test")
import numpy as np
import torch
from torch.nn import functional as F

import screen_sop_gallery_d2_d3_v1 as d


class GalleryScreenTests(unittest.TestCase):
    def test_leave_query_out_rho_equals_brute_force(self):
        g = torch.Generator().manual_seed(0)
        rows, k1, k2 = 40, 2, 6
        x = F.normalize(torch.randn(rows, 3, generator=g), dim=1)
        gallery = torch.arange(5, rows)
        queries = torch.arange(5, 15)

        def score(q):
            return x[q] @ x[gallery].T

        cand, _ = d.topn(score, queries, gallery, rows, 8)
        g_ids, g_cos = d.topn(score, gallery, gallery, rows, k2 + 1)
        nbr_ids = torch.full((rows, k2 + 1), -1)
        nbr_cos = torch.zeros(rows, k2 + 1)
        nbr_ids[gallery], nbr_cos[gallery] = g_ids, g_cos
        got = d.loo_rho(nbr_ids, nbr_cos, cand, queries, k1, k2)
        hits = 0
        for i, q in enumerate(queries.tolist()):
            for j, c in enumerate(cand[i].tolist()):
                others = [r for r in gallery.tolist() if r not in (c, q)]
                sims = sorted((float(x[c] @ x[r]) for r in others), reverse=True)
                self.assertAlmostEqual(float(got[i, j]), float(np.mean(sims[k1 - 1:k2])), places=5)
                hits += q in g_ids[gallery.tolist().index(c), :k2].tolist()
        self.assertGreater(hits, 0)  # the query really sat inside some candidate's window

    def test_rerank_keeps_ranks_beyond_32(self):
        g = torch.Generator().manual_seed(1)
        ranked = torch.stack([torch.randperm(100, generator=g)[:40] for _ in range(3)])
        key = torch.randn(3, d.TOP, generator=g)
        out = d.rerank_head(ranked, key)
        self.assertTrue(torch.equal(out[:, d.TOP:], ranked[:, d.TOP:]))
        for r, o, k in zip(ranked, out, key):
            self.assertEqual(sorted(o[:d.TOP].tolist()), sorted(r[:d.TOP].tolist()))
            self.assertEqual(o[0].item(), r[int(k.argmax())].item())

    def test_product_folds_never_split_a_product(self):
        products = np.random.default_rng(3).integers(0, 37, 300)
        folds = d.product_folds(products)
        for p in np.unique(products):
            self.assertEqual(len(np.unique(folds[products == p])), 1)
        self.assertEqual(set(folds.tolist()), set(range(d.FOLDS)))
        np.testing.assert_array_equal(folds, d.product_folds(products))
        # CV scores each fold with the config chosen on the other folds.
        r1 = np.zeros((2, len(products)))
        r1[1, folds == 0] = 1.0  # config 1 wins only on fold 0, so fold 0 itself must not pick it
        out, _, chosen = d.cv_select(r1, np.zeros_like(r1), folds)
        self.assertEqual(chosen, [0] + [1] * (d.FOLDS - 1))
        self.assertEqual(out.sum(), 0.0)

    def test_union_maxsim_beats_best_member_on_toy(self):
        e = torch.eye(3)
        q = e[:2]                      # query tokens e1, e2
        imgs = torch.stack([e[[0, 2]], e[[1, 2]], e[[2, 2]]])  # A covers e1, B covers e2, C neither
        seg = torch.tensor([0, 0, 1])  # A and B are one product, C another
        union, best = d.union_and_best(q, imgs, seg, 3)
        torch.testing.assert_close(union[:2], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(best[:2], torch.tensor([0.5, 0.0]))
        self.assertTrue(torch.isinf(union[2]) and torch.isinf(best[2]))
        # group ordering: labels a b a c; a large weight on the feature puts product b first, a's members stay together
        lab = torch.tensor([[7, 8, 7, 9]])
        slot = d.group_slots(lab)
        self.assertEqual(slot.tolist(), [[0, 1, 0, 3]])
        cos = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        feat = torch.tensor([[0.1, 0.9, 0.0, 0.2]])
        self.assertEqual(d.group_perm(slot, cos, feat, 100.0).tolist(), [[1, 3, 0, 2]])
        self.assertEqual(d.group_perm(slot, cos, torch.zeros_like(feat), 1.0).tolist(), [[0, 2, 1, 3]])

    def test_mate_distance(self):
        u = F.normalize(torch.randn(5, 4, generator=torch.Generator().manual_seed(4)), dim=1).numpy()
        lab = np.array([0, 0, 0, 1, 2])
        got = d.mate_distance(u, lab)
        self.assertAlmostEqual(got[0], 1 - (u[0] @ u[1] + u[0] @ u[2]) / 2, places=6)
        self.assertTrue(np.isnan(got[3]) and np.isnan(got[4]))


if __name__ == "__main__":
    unittest.main()
