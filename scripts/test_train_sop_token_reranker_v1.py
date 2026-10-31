#!/usr/bin/env python3
"""Tiny CPU checks for train_sop_token_reranker_v1; no GPU, cache or checkpoint.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts python scripts/test_train_sop_token_reranker_v1.py
"""
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before running this CPU-only test")
import numpy as np
import torch

import screen_sop_token_cache_v1 as s
import train_sop_token_reranker_v1 as r


class RerankerTests(unittest.TestCase):
    def test_forward_shape_and_permutation(self):
        torch.manual_seed(0)
        m = r.Reranker(width=12, d=16, heads=2, mlp=32, tokens=4).eval()
        q, c, cos = torch.randn(3, 4, 12), torch.randn(3, 5, 4, 12), torch.rand(3, 5)
        with torch.no_grad():
            out = m(q, c, cos)
            self.assertEqual(tuple(out.shape), (3, 5))
            perm = torch.tensor([3, 0, 4, 1, 2])
            torch.testing.assert_close(m(q, c[:, perm], cos[:, perm]), out[:, perm])
            self.assertAlmostEqual(float(m.scale), 10.0)
            # rerank_perm on projected tokens agrees with a direct argsort of the forward scores
            pt = m.proj(torch.cat([q, c.flatten(0, 1)]))
            qi = torch.arange(3)
            ci = 3 + torch.arange(15).view(3, 5)
            want = torch.argsort(out, dim=1, descending=True, stable=True)
            self.assertTrue(torch.equal(r.rerank_perm(m, pt, pt, qi, ci, cos, chunk=2), want))

    def test_listwise_loss_manual(self):
        logits = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
        pos = torch.tensor([[True, False, True, False], [False, False, False, True]])
        manual = 0.0
        for z, y in zip(logits.tolist(), pos.tolist()):
            lse = np.log(np.sum(np.exp(z)))
            manual += -np.mean([z[k] - lse for k in range(4) if y[k]]) / 2
        bce = np.mean([-(yk * np.log(1 / (1 + np.exp(-zk))) + (1 - yk) * np.log(1 - 1 / (1 + np.exp(-zk))))
                       for z, y in zip(logits.tolist(), pos.tolist()) for zk, yk in zip(z, y)])
        self.assertAlmostEqual(float(r.listwise_loss(logits, pos)), manual + 0.5 * bce, places=5)

    def test_rerank_keeps_tail_and_ap(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(20), rng.integers(2, 6, 20))
        rows = len(labels)
        lab = torch.from_numpy(labels)
        queries = torch.from_numpy(np.sort(rng.choice(rows, 15, replace=False)))
        gallery = torch.arange(rows)
        full = torch.from_numpy(np.stack([rng.permutation(np.delete(np.arange(rows), q)) for q in queries.numpy()]))
        k = 8
        perm = torch.from_numpy(np.stack([rng.permutation(k) for _ in range(len(queries))]))
        out = r.apply_rerank(full, perm)
        self.assertTrue(torch.equal(out[:, k:], full[:, k:]))
        self.assertTrue(torch.equal(out[:, :k], full[:, :k].gather(1, perm)))
        r1, ap, _ = s.metrics(out, queries, gallery, lab)
        for i, q in enumerate(queries.numpy()):
            match = labels[out[i].numpy()] == labels[q]
            rel = int(match.sum())
            prec = np.cumsum(match) / np.arange(1, len(match) + 1)
            self.assertEqual(r1[i], float(match[0]))
            self.assertAlmostEqual(ap[i], float((prec * match)[:rel].sum() / rel), places=6)

    def test_hard_list_weights(self):
        w = r.list_weights(torch.tensor([True, False, True, False, False]))
        self.assertEqual(w.tolist(), [1.0, 4.0, 1.0, 4.0, 4.0])
        torch.manual_seed(0)
        draws = torch.multinomial(r.list_weights(torch.tensor([True, False])), 20000, replacement=True)
        self.assertAlmostEqual(float((draws == 1).float().mean()), 0.8, delta=0.02)

    def test_int8_quantize(self):
        x = torch.randn(3, 4, 16).half()
        y = r.quantize_int8(x)
        self.assertEqual(y.dtype, torch.float16)
        bound = x.float().abs().amax(-1, keepdim=True) / 127 * 0.51 + 1e-3
        self.assertTrue(bool(((y.float() - x.float()).abs() <= bound).all()))


if __name__ == "__main__":
    unittest.main()
