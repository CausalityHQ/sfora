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
