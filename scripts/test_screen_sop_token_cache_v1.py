#!/usr/bin/env python3
"""Tiny random CPU checks for screen_sop_token_cache_v1; no GPU, checkpoint or images.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts python scripts/test_screen_sop_token_cache_v1.py
"""
import itertools
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before running this CPU-only test")
import numpy as np
import torch
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

import screen_sop_token_cache_v1 as s


class ScreenTests(unittest.TestCase):
    def test_maxsim_and_fusion(self):
        g = torch.Generator().manual_seed(1)
        q = F.normalize(torch.randn(2, 5, 8, generator=g), dim=-1)
        c = F.normalize(torch.randn(2, 4, 6, 8, generator=g), dim=-1)
        ref = torch.tensor([[np.mean([max(float(q[b, t] @ c[b, k, u]) for u in range(6)) for t in range(5)])
                             for k in range(4)] for b in range(2)])
        m = s.maxsim(q, c)
        torch.testing.assert_close(m, ref.float())
        gs = torch.randn(2, 4, generator=g)
        for w in (0.0, 0.5, 2.0):
            fused = (gs - gs.mean(1, keepdim=True)) / gs.std(1, unbiased=False, keepdim=True) + w * (
                (m - m.mean(1, keepdim=True)) / m.std(1, unbiased=False, keepdim=True))
            want = np.stack([np.argsort(-row, kind="stable") for row in fused.numpy()])
            np.testing.assert_array_equal(s.fuse_order(gs, m, w).numpy(), want)
        self.assertTrue(torch.equal(s.fuse_order(gs, torch.zeros_like(m), 1.0),
                                    torch.argsort(gs, dim=1, descending=True, stable=True)))

    def test_context_pool(self):
        g = torch.Generator().manual_seed(2)
        tok, att = torch.randn(3, 9, 4, generator=g), torch.rand(3, 9, generator=g).softmax(1)
        for b in range(3):
            w = (1 - att[b]).numpy()
            np.testing.assert_allclose(s.context_pool(tok, att)[b].numpy(),
                                       (w[:, None] * tok[b].numpy()).sum(0) / w.sum(), rtol=1e-5, atol=1e-6)

    def test_attention_box(self):
        n = 5
        a = torch.rand(6, n * n, generator=torch.Generator().manual_seed(3)) ** 4
        boxes = s.attention_box(a, grid=n)
        for b in range(6):
            m = (a[b] / a[b].sum()).reshape(n, n).numpy()
            best = min((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in itertools.product(range(n + 1), repeat=4)
                       if r0 < r1 and c0 < c1 and m[r0:r1, c0:c1].sum() >= 0.8 - 1e-6)
            r0, r1, c0, c1 = boxes[b].tolist()
            self.assertEqual((r1 - r0) * (c1 - c0), best)
            self.assertGreaterEqual(m[r0:r1, c0:c1].sum(), 0.8 - 1e-6)
        delta = torch.zeros(1, 27 * 27)
        delta[0, 3 * 27 + 7] = 1
        self.assertEqual(s.attention_box(delta).tolist(), [[3, 4, 7, 8]])
        r0, r1, c0, c1 = s.attention_box(torch.ones(1, 27 * 27))[0].tolist()
        self.assertEqual((r1 - r0) * (c1 - c0), 594)  # smallest rectangle >= 0.8 * 729 cells

    def test_rank_and_metrics_vs_numpy(self):
        rng = np.random.default_rng(4)
        labels = np.repeat(np.arange(30), rng.integers(2, 7, 30))
        rows = len(labels)
        unit = F.normalize(torch.from_numpy(rng.normal(size=(rows, 16))).float() +
                           torch.from_numpy(rng.normal(size=(30, 16))[labels]).float(), dim=1)
        unit[5] = unit[6]  # exact tie
        lab = torch.from_numpy(labels)
        for gallery in (np.arange(rows), np.sort(rng.choice(rows, 90, replace=False))):
            queries = np.sort(rng.choice(gallery, 25, replace=False))
            gal = torch.from_numpy(gallery)
            if (np.bincount(labels[gallery], minlength=30)[labels[queries]] < 2).any():
                queries = queries[np.bincount(labels[gallery], minlength=30)[labels[queries]] >= 2]
            ranked = s.rank(lambda q: unit[q] @ unit[gal].T, torch.from_numpy(queries), gal, rows)
            r1, ap, rec = s.metrics(ranked, torch.from_numpy(queries), gal, lab)
            for i, q in enumerate(queries):
                others = gallery[gallery != q]
                score = (unit[q] @ unit[torch.from_numpy(others)].T).numpy()
                order = others[np.argsort(-score, kind="stable")]
                match = labels[order] == labels[q]
                rel = int(match.sum())
                prec = np.cumsum(match) / np.arange(1, len(match) + 1)
                self.assertEqual(r1[i], float(match[0]))
                self.assertAlmostEqual(ap[i], float((prec * match)[:rel].sum() / rel), places=5)
                np.testing.assert_array_equal(ranked[i].numpy(), order[:s.TOP])
            self.assertAlmostEqual(rec["R@1"], r1.mean())

    def test_product_bootstrap(self):
        rng = np.random.default_rng(5)
        d, products = rng.normal(size=(40, 2)), rng.integers(0, 12, 40)
        a, b = s.product_bootstrap(d, products), s.product_bootstrap(d, products)
        self.assertEqual(a.shape, (5000, 2))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, s.product_bootstrap(d, products, seed=1)))
        np.testing.assert_allclose(a.mean(0), d.mean(0), atol=0.1)
        out = s.compare((np.zeros(40), np.zeros(40)), (np.ones(40), np.zeros(40)), products)
        self.assertEqual((out["d_r1"], out["d_r1_95"], out["decision"]), (100.0, [100.0, 100.0], "PASS"))

    def test_attention_hook_matches_manual_softmax(self):
        cfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                 image_size=42, patch_size=14)
        vision = SiglipVisionModel(cfg).eval()  # transformers 5: SiglipModel.vision_model is this class
        captured = s.attach_attention_hook(vision)
        with torch.no_grad():
            out = vision(pixel_values=torch.randn(2, 3, 42, 42))
        weights = captured["weights"]
        mha, h = vision.head.attention, out.last_hidden_state
        wq, wk, _ = mha.in_proj_weight.chunk(3)
        bq, bk, _ = mha.in_proj_bias.chunk(3)
        qp = (vision.head.probe @ wq.T + bq).reshape(1, 1, 4, 8).transpose(1, 2)
        kp = (h @ wk.T + bk).reshape(2, 9, 4, 8).transpose(1, 2)
        manual = ((qp @ kp.transpose(-1, -2)) / 8**0.5).softmax(-1).mean(1)
        self.assertEqual(tuple(weights.shape), (2, 1, 9))
        torch.testing.assert_close(weights, manual)


if __name__ == "__main__":
    unittest.main()
