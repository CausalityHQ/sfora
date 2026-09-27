# In-Shop encoder resolution feasibility, 27 September 2026

After the valid-anchor rank confirmation failed, test whether reducing
SigLIP2 Large/256 input size is a plausible serving-speed route before
training a smaller-resolution student. This is an exploratory zero-training
screen on the already-selected official In-Shop **TRAIN** held-only symmetric
12,599-query/gallery panel. It cannot support an official or SOTA claim.

The [source-bound probe](../scripts/probe_inshop_siglip2_resolution.py)
(SHA-256 `0d2ee04d5da800b2e580a7279f8232d5d0ed590d73beae9d17347747a3149597`)
loaded the seed-179026 `freeze_emb` checkpoint, re-encoded the same held images
at 256 and 192 pixels, and used the same 128-D int8 packed exact scorer.
At 192 it used the model's positional-embedding interpolation. The 256 replay
matched the frozen checkpoint receipt's per-query packed R@1 exactly. The
sole DGX Spark GB10 service `sfora-inshop-resolution-probe-179026-v1.service`
invocation `8d1a2d4e2a974d9da831ff05f0942112` exited successfully.

| Input | Packed R@1 | mAP@R | Full 12,599-image export wall |
| --- | ---: | ---: | ---: |
| 256 pixels | 98.5237% | 0.841910 | 67.433 s |
| 192 pixels | 96.9125% | 0.768205 | 60.775 s |

The 192-pixel result loses **203 net correct queries** (45 gained, 248 lost),
or **1.6112 percentage points** R@1, while the one-pass export is **9.87%**
shorter. Export wall includes preprocessing and batched encoding, not public
image-to-top-k latency, and this sequential timing is not a latency gate.
This is too large a quality regression to justify immediate 192-pixel
distillation or production serving changes. A 224-pixel screen is the last
cheap resize check: before reading it, require the packed R@1 loss versus
the same 256 checkpoint to be **at most 0.5 percentage points** to consider
training; otherwise stop the resize lane. Any promising result still needs
paired new-seed training and public image-to-top-k timing. Both screens use
selected TRAIN held products and remain exploratory.

The full [per-query report](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-resolution-probe-179026-v1.json)
SHA-256 is `fdf12abff9bc170d89536b2caf607f10e5aa7a95300365c25f658095f123c873`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-resolution-probe-179026-v1.journal.log)
SHA-256 is `be09d7c41c51520111cf7a8af2f623c00b5560a74e05ce4be3224703012a7c23`.
