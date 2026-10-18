#!/usr/bin/env python3
"""Tiny random two-layer CPU checks; no GPU, network, checkpoint or model images.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/test_bench_sop_so400m_encode_variants_v1.py
"""

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before running this CPU-only test")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

import bench_sop_so400m_encode_variants_v1 as bench


class BenchmarkTests(unittest.TestCase):
    def test_configuration_grid(self):
        configs = bench.configurations()
        self.assertEqual(len(configs), 54)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in configs}), 54)
        self.assertEqual({c["attention"] for c in configs}, {"eager", "sdpa"})
        self.assertEqual({c["precision"] for c in configs}, {"fp32_autocast", "fp16", "bf16"})
        self.assertEqual({c["batch_size"] for c in configs}, {64, 128, 256})
        self.assertEqual({c["compile_mode"] for c in configs}, {None, "default", "max-autotune-no-cudagraphs"})
        self.assertGreaterEqual(bench.TIMED_ITERS, 20)
        self.assertGreaterEqual(bench.DRIFT_ROWS, 2048)
        self.assertIn(bench.reference_config("sdpa"), configs)

    def test_two_layer_encoder_and_schema(self):
        torch.manual_seed(179024)
        config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, image_size=28, patch_size=14,
                                   attention_dropout=0.0)
        source = SiglipVisionModel(copy.deepcopy(config)).eval()
        head = nn.Linear(32, 128).eval()
        state = {"vision": source.state_dict(), "head": head.state_dict()}
        device = torch.device("cpu")
        variant = {**bench.reference_config("eager"), "batch_size": 16}
        model = bench.build_encoder(config, state, variant, device)
        # 42px exercises positional interpolation from the native 28px config.
        samples = np.random.default_rng(1).integers(0, 256, (32, 42, 42, 3), dtype=np.uint8)
        processor = SiglipImageProcessor(size={"height": 42, "width": 42})
        values = bench.encode_samples(model, samples, processor, variant, device)
        pixels = bench.pixels_for(samples[:16], processor, variant, device)
        with torch.inference_mode():
            pooled = model.vision(pixel_values=pixels, interpolate_pos_encoding=True).pooler_output
            expected = F.normalize(head(F.normalize(pooled.float(), dim=1)), dim=1)
        self.assertTrue(torch.equal(values[:16], expected))
        self.assertEqual(values.shape, (32, 128))
        self.assertEqual(model.head.weight.dtype, torch.float32)
        exact = bench.drift_metrics(values, values.clone())
        self.assertEqual(exact, {"rows": 32, "max_abs_cosine_delta": 0.0,
                                "byte_identical_code_fraction": 1.0,
                                "top10_neighbour_set_agreement": 1.0,
                                "packed_top10_neighbour_set_agreement": 1.0})
        sdpa = bench.build_encoder(config, state, {**variant, "attention": "sdpa"}, device)
        sdpa_values = bench.encode_samples(sdpa, samples, processor, variant, device)
        drift = bench.drift_metrics(values, sdpa_values)
        self.assertLess(drift["max_abs_cosine_delta"], 1e-4)
        for precision in ("fp16", "bf16"):
            native_variant = {**variant, "precision": precision}
            native = bench.build_encoder(config, state, native_variant, device)
            self.assertEqual(native.vision.embeddings.patch_embedding.weight.dtype, bench.PRECISIONS[precision])
            self.assertEqual(native.head.weight.dtype, torch.float32)
            native_values = bench.encode_samples(native, samples, processor, native_variant, device)
            self.assertTrue(torch.isfinite(native_values).all())
            self.assertLess(bench.drift_metrics(values, native_values)["max_abs_cosine_delta"], .05)
        default = bench.build_encoder(config, state, bench.reference_config(), device)
        print("Installed HF default attention:", default.vision.config._attn_implementation)
        good = {"config": variant, "status": "ok", "effective_attention": "eager",
                "gpu_img_per_s": 1.0, "peak_cuda_bytes": 0, "warmup_iters": 5,
                "warmup_seconds": 0.0, "timed_iters": 20, "timed_seconds": 1.0, "drift": drift}
        bad = {"config": bench.configurations()[-1], "status": "unsupported", "error": "test failure"}
        report = json.loads(json.dumps(bench.make_report(
            {"device": "CPU fixture"}, {"config": bench.reference_config("sdpa"), "self_drift": exact},
            [good, bad]), allow_nan=False))
        self.assertEqual(set(report), {"schema", "official_read", "environment", "reference", "results", "metrics"})
        self.assertEqual(report["schema"], bench.SCHEMA)
        self.assertFalse(report["official_read"])
        self.assertEqual([r["status"] for r in report["results"]], ["ok", "unsupported"])
        self.assertEqual(report["results"][0]["drift"], drift)

    def test_drift_matches_dense_oracle_and_detects_changes(self):
        torch.manual_seed(7)
        reference = F.normalize(torch.randn(300, 128), dim=1)
        candidate = reference.roll(1, dims=0).contiguous()
        result = bench.drift_metrics(reference, candidate)
        ref_scores, cand_scores = reference @ reference.T, candidate @ candidate.T
        self.assertAlmostEqual(result["max_abs_cosine_delta"], float((ref_scores - cand_scores).abs().max()), places=5)
        ref_scores.fill_diagonal_(-torch.inf)
        cand_scores.fill_diagonal_(-torch.inf)
        ref_top = torch.argsort(ref_scores, dim=1, descending=True, stable=True)[:, :10]
        cand_top = torch.argsort(cand_scores, dim=1, descending=True, stable=True)[:, :10]
        agreement = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref_top, cand_top)) / 3000
        self.assertAlmostEqual(result["top10_neighbour_set_agreement"], agreement, places=6)
        self.assertEqual(result["byte_identical_code_fraction"], 0.0)
        self.assertLess(result["packed_top10_neighbour_set_agreement"], .5)
        tied = F.normalize(torch.ones(12, 128), dim=1)
        self.assertEqual(bench.neighbours(tied @ tied.T, 0)[0].tolist(), list(range(1, 11)))
        self.assertEqual(bench.drift_metrics(tied, tied)["top10_neighbour_set_agreement"], 1.0)
        for invalid in (reference[:10], reference[:, :127], reference * float("nan"), reference * 0):
            with self.assertRaises(ValueError):
                bench.drift_metrics(invalid, invalid)

    def test_sample_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npy"
            good = np.zeros((2048, 2, 3, 3), dtype=np.uint8)
            np.save(path, good, allow_pickle=False)
            self.assertEqual(bench.load_samples(path, 280).shape, good.shape)
            for invalid in (good[:2047], good.astype(np.float32), good[:, :, :, :2], good[:, 0]):
                np.save(path, invalid, allow_pickle=False)
                with self.assertRaises(ValueError):
                    bench.load_samples(path, 384)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
