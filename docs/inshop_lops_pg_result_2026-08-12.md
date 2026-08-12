# LOPS-PG Train-Only Confirmation — 2026-08-12

## Provisional decision

**PASS.** Leave-One-Out Positive-Safe
Proxy Gradient (LOPS-PG) passed all seven frozen confirmation predicates on
previously untouched folds 1-3. This is train-embedding directional evidence,
not a retrieval result or SOTA claim. It authorizes a small real-training
comparison; it does not establish that the trained encoder will improve
held-out Recall@1.

## Execution

- Source HEAD: `caef56b` (`fix LOPS-PG shuffled centroid control`)
- Input SHA-256: `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`
- Output: `reports/generated/inshop_lops_pg_confirmation.json`
- Output SHA-256: `fac85fcb286124bc3d3da58bf3888a5f3f92fa37cc5829815e235d165e655a02`
- Output size/mode: 4,469,516 bytes, mode 0600
- Environment: CPU only, NumPy 2.5.0, CUDA hidden, one OpenMP/MKL/OpenBLAS thread
- Process exit: 0 (`PASS`)

The evaluator used only the registered In-Shop train embedding archive. Fold 0
was disclosed discovery data and was excluded in code and persisted protocol.
Folds 1, 2, and 3 supplied 10,620 evaluated rows and 2,655 frozen bottom-margin
primary rows spanning 1,389 identities.

## Confirmation evidence

The positive-safety constraint was active for 10,619 of 10,620 eligible rows
(`0.9999058380414313`). LOPS-PG minus unchanged PA hard-row margin was positive
in every untouched fold:

```text
fold 1  +0.0010552690367211458
fold 2  +0.0011481495626101536
fold 3  +0.0010418630129327020
pooled  +0.0010806352997062690
99% LB  +0.0010484854445500706
```

The gain was 32.5% of the median absolute PA margin change
(`0.0033244575172974855`), above the frozen 10% materiality threshold.

The identity-specific controls did not explain the result:

```text
LOPS minus shuffled-centroid mean       +0.0013237415249388635
one-sided 99% lower bound               +0.0012777954946163184

LOPS minus pure-positive mean           +0.0015478013685748212
one-sided 99% lower bound               +0.0014686477403046201

LOPS minus PA positive-similarity mean  +0.0015098979754477044
one-sided 99% lower bound               +0.0014772144550084239
```

The registered batch-hard-triplet direction remained substantially stronger on
its own one-step objective (`+0.0056138290524235645` versus PA). This arm was
prospectively non-gating because it directly optimizes the diagnostic margin.
It is a required baseline for any real training comparison and limits the
claim: LOPS-PG is a minimal PA-preserving correction, not the best possible
one-step margin direction.

All seven predicates passed: coverage, raw advantage, fold consistency,
control superiority, material effect, positive similarity, and exact
half-space constraint integrity. The four 10,000-replicate identity-bootstrap
hashes were:

```text
raw            7a52dcdcb473c18614d289a7c6f003a3fe327661d0205db3e62f87c3596134bf
shuffled       3eac2649f543381e7cdc6aa58a225ab381ece113800a9b7d4f637f624f25d2a5
positive-only  602b202bc9a8dfe13c30b0d72d176f96af99ec7a63e1c08907031bd64a9bd9f8
positive       55a8e5e41da69d3f5cc94ed8d819de5de51ba31f4878bab5c24ce3fae7026410
```

## Consequence

An external review attempt timed out without returning a verdict. Independent
self-review found that the first provisional process shuffled already-projected
tangents instead of raw sibling centroids. That process was preserved under an
explicit invalid filename, the defect was caught by an independent-oracle RED
test and fixed in `caef56b`, and the confirmation above is the fresh post-fix
process. The corrected control separation increased; all other registered
values were unchanged.

The next experiment is a small multi-seed real-training comparison using the
live Proxy Anchor embedding cotangent:

1. unchanged Proxy Anchor;
2. Proxy Anchor with LOPS-PG embedding-gradient projection;
3. Proxy Anchor plus an ordinary positive-compactness auxiliary loss;
4. batch-hard triplet under the same backbone, data, augmentation, optimizer,
   schedule, and cosine inference.

The primary question is whether LOPS-PG preserves PA's open-set proxy benefit
while preventing the pervasive within-class cohesion damage seen in the
virtual steps. GPU nondeterminism must be handled with multiple seeds and
distributional reporting, not claimed away. Until that training comparison
passes, the established reproducible operating point remains Proxy Anchor plus
the already validated fixed local-scaling retrieval correction.
