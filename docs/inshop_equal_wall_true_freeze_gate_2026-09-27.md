# In-Shop equal-wall true-freeze TRAIN gate, 27 September 2026

The 1,000-update `freeze_emb` checkpoint saved 34.8% training wall versus
full control on seed 179024, with a one-query R@1 shortfall and higher mAP@R.
Its frozen promotion gate failed; it is not the production default. The
trained raw 1,024-D pooler scored below its 128-D head on the same held panel,
so a wider head has weaker direct support than spending the observed compute
saving on more training exposure. This is a new exploratory recipe comparison,
not a claim that the official In-Shop gap or SOTA has been solved.

## Authority and frozen decision

Use only official In-Shop **TRAIN** standard `img.zip` retrieval pixels,
partition SHA-256 `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`.
The full 52,712-row standard path/size/content digest is
`608373be84bc4e5b95e3c6f87e712e4d1ca53299e6f59d1627d033a70128f8fe`;
the separate high-resolution corpus is invalid. Pin the existing
13,283-image/2,004-product fit and 12,599-image/1,993-product held split,
SigLIP2 Large/256 checkpoint/cache, seed 179024, batch 64, BF16, optimizer,
ArcFace plus member-bank SmoothAP, PCA-initialized 128-D head, signed-int8
plus f16 inverse-norm packed scorer, and the official TRAIN held-only
self-excluded symmetric gallery. Do not read official query/gallery.

Run a **new same-source control** (`control`, 1,000 updates) and then the
treatment (`freeze_emb`, **1,533 updates**) serially on the DGX Spark GB10
under one GPU lock. The 1,533 budget is fixed from the archived seed-179024
wall ratio, 1,000 × 1,147.928/748.621 ≈1,533. The treatment consumes the
first 1,533 batches of the already frozen 3,000-step schedule; its first
1,000 batches must equal the original schedule. No learning-rate decay depends
on the requested update total: the trainer uses fixed per-group rates. The
only treatment changes are the lower-stack and embedding freeze plus update
count; no causal claim isolates either component. Code source SHA-256:
`de7a3fde54decd78c0c42aa29973c8c714d01c4a85a6a6ce1d19358642470801`.
The [serial DGX launcher](../scripts/run_inshop_equal_wall_true_freeze_dgx.sh)
SHA-256 is `33ecbecec132deb2925a15460311faf0e1113d67b205e1bc44a445d474723eb2`.
The [frozen paired analyzer](../scripts/analyze_inshop_equal_wall.py) SHA-256
is `909237b68ff474afedb3be9787273e3fe5aa259b16b28372de07dde51c6cf9ac`;
its reused product-bootstrap helper SHA-256 is
`a4ee6ce18ca67cfab1838c505fff977f818e1865dcf85600f39f5dc900280bbe`.

Before scoring, verify source/model/cache/preflight hashes, the standard
pixel corpus, matched split, PCA initialization, first ten input batches,
finite updates, checkpoint/receipt hashes, and exact packed scorer. The
training scripts must record all executed steps and the executed-schedule
digest. Stop on any authority, finiteness, parity or resource failure.

Advance only if all four seed-179024 gates pass on the same 12,599 held
queries: (1) treatment minus control R@1 product-cluster paired-bootstrap
95% **lower bound >0**; (2) treatment mAP@R **≥control−0.002**; (3)
treatment training wall including bank setup **≤1.02× control**; and (4)
treatment peak allocated CUDA **≤control**. Report exact R@1, mAP@R,
wall, images/s, VRAM and receipt hashes for both arms. The fixed
6,354-query/6,245-gallery asymmetric roles may be reported as an auxiliary
readout only; they cannot rescue a failed gate.

If seed 179024 fails, stop before another seed or update count. If it passes,
freeze a same-source seed-179026 control/treatment pair with the same gates;
both seeds must pass individually before a fresh split/seed confirmation and
public checkpoint parity/latency gate. The current held panel and seed have
been inspected repeatedly; even a pass remains exploratory and cannot
authorize a SOTA claim or a production default. Serving architecture and
130-byte gallery format stay fixed, but public latency for the resulting
checkpoint must be measured before promotion.

## Terminal seed-179024 result

The sole locked DGX Spark GB10 unit `sfora-inshop-equal-wall-179024-v1`
(invocation `155cc1935c82421dbda6259f9e2de916`) exited 0 after both arms.
The launcher verified the full standard-pixel manifest before training. The
[control](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-equal-wall-179024-v1/control.json)
receipt SHA-256 is `420099def5528239e50cd65fe4c3451168c4352a486e7a5e357d4a08177ee9bc`;
[treatment](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-equal-wall-179024-v1/treatment.json)
is `91c6847879d683e7793caf04ccdf537124cf691be1153d7861b2276c9cd6dcc5`;
[paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-equal-wall-179024-v1/decision.json)
is `c3f7279718690385b71ecc3d0d2ba59940ea160af63a0caf81f63a7350155d4e`;
[terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-equal-wall-179024-v1/journal.log)
is `846504c3aee2f2729cbeb2436ed247cc826099fbc0010f25e8cee604cb3db388`.
The analyzer verified the source files, seed, split, first ten input batches,
PCA initialization, full and executed schedule digests, 1,000/1,533 finite
step receipts, checkpoints, per-query vectors and scorer output. The fresh
control's packed quality exactly matches the archived same-seed control.

| Official TRAIN, 12,599 held-only self-excluded queries | New control, 1,000 updates | True freeze, 1,533 updates |
| --- | ---: | ---: |
| Packed R@1 | **98.5554%** | **98.4761%** |
| Packed mAP@R | **0.826739** | **0.842585** |
| Training wall including bank setup | **1,151.440 s** | **1,142.074 s** |
| Training throughput including setup | **55.58 images/s**, 64,000 images | **85.91 images/s**, 98,112 images |
| Peak allocated CUDA | **22.554 GB** | **12.941 GB** |

Treatment minus control R@1 was **−0.07937 percentage points**, paired
product-bootstrap 95% **[−0.27599, +0.11431] pp**. mAP@R improved
**+0.015846**, paired interval **[+0.011813, +0.020213]**. Wall ratio was
**0.9919** and peak CUDA ratio **0.5738**. The mAP, wall and memory gates
pass, but R@1 lower confidence bound is below zero; the frozen joint gate
fails (`advance_seed179026=false`). Stop before seed 179026, another update
budget, official query/gallery or production promotion. This establishes a
same-cost mAP gain on this repeatedly inspected TRAIN panel, not a verified
R@1 improvement or a public serving result.
