# In-Shop expected-gallery-recall TRAIN falsifier, 27 September 2026

The current detached-bank SmoothAP objective improves the saturated symmetric
In-Shop TRAIN gallery but the selected three-seed official query/gallery mean
remains 95.4823% packed R@1. A proposed replacement weights positives by the
chance that a gallery subset contains no positive above the top impostor.
This is a *candidate*, not an established novel method: [random-gallery
rank-1 averaging has prior art](https://people.cs.umass.edu/~elm/papers/Erdos.pdf),
and the official gallery removes negatives as well as
positives. Holding negatives fixed makes the proposed positive-subset miss
probability an upper bound under uniform random positive and negative
subsampling, not an exact official-protocol probability.

Before changing the trainer, re-export the existing seed-179026 `freeze_emb`
checkpoint on the fixed 12,599-image, 1,993-product official **TRAIN** held
panel. Require exact packed per-query symmetric R@1 parity with its source
receipt. Use the existing path-hash asymmetric split of 6,354 queries and
6,245 gallery rows. For each query, count `n` eligible positives and `m`
positives below the best full-gallery impostor using exact packed score and
ordinal ties. With the observed per-product gallery-positive count `k`, the
uniform positive-subset fixed-negative miss probability is `C(m,k)/C(n,k)`.
Compare its mean predicted R@1 with the actual asymmetric packed R@1 from
the same export. This is a diagnostic of the loss target, not an official
evaluation or a new model result.

Freeze these kill rules before export: the predicted and actual asymmetric
R@1 must differ by at most **0.004**; at least **3%** of the 6,354 query rows
must have only one or two positives above the best full-gallery impostor; and
among below-impostor positives on those fragile rows, at least **30%** must be
within **0.05 packed-cosine** of that boundary. If any fails, stop this
candidate before loss code or training. A pass only authorizes checking the
same counts and stability on the two other archived `freeze_emb` seeds, then
a separately frozen paired TRAIN training gate. No current SOTA or novelty
claim follows. The sole DGX Spark remains occupied by the serial serving
p99 job; this probe must wait for that job to close.

## Terminal TRAIN-only result

After the serving job closed, the sole DGX Spark GB10 probe exited 0. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-expected-gallery-179026/receipt.json)
has SHA-256 `d48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4`;
the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-expected-gallery-179026/service-journal.txt)
has SHA-256 `c10f5c81c837a698a42bc49a1e9d8b53b76177aa67ff980d0a5825eb03f3d36c`.
Hashes of all 12,599 held image bytes were recorded; the checkpoint, parser,
packing source and prior pose receipt were pinned. The re-export replayed the
prior packed full and asymmetric R@1 exactly. Independent local recomputation
checked every query's
combination formula and all three frozen thresholds.

| In-Shop official TRAIN held-only, seed 179026 | Result | Frozen gate |
| --- | ---: | ---: |
| Fixed-negative predicted asymmetric packed R@1 | 96.5051% | Prediction error ≤0.4 pp |
| Actual 6,354-query/6,245-gallery asymmetric packed R@1 | 97.6235% | **Error 1.1184 pp; fail** |
| Queries with 1–2 positives above best full-gallery impostor | 729/6,354 = 11.4731% | ≥3%; pass |
| Below-impostor positives near boundary on those queries | 391/2,579 = 15.1609% | ≥30%; **fail** |

The fixed-negative surrogate is too pessimistic for this gallery because the
actual subset also removes negatives; the predeclared near-boundary lever is
too small. The proposed expected-gallery rank loss stops **before any trainer
change, second-seed export or training**. This is a falsified method candidate,
not a model-quality regression. Export took 67.442 s and scoring 0.208 s;
peak allocated CUDA memory was 1.886 GB. No official TEST data was used.
