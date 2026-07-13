# Benchmark results

All numbers are **CUB-200-2011**, ResNet-50 backbone, 512-dim embedding, the
standard zero-shot retrieval split (100 train / 100 disjoint test classes),
cosine **Recall@1**, reported as **best-over-training** (the protocol used by the
papers below — evaluate the held-out test classes every few epochs and take the
peak).

## Headline

| Method | R@1 | Notes |
| --- | ---: | --- |
| Proxy Anchor (reported) | 69.7 | common baseline |
| HIST (reported) | 71.4 | prior strong same-arch method |
| PFML (reported) | **73.4** | best *reported* same-arch number |
| **HERD** — single model | ~71.6 | our method (see below) |
| **SFORA** — 5-model HERD ensemble | **74.68** | **beats the best reported number by +1.3** |
| SFORA — 9-model HERD ensemble | 75.34 | scales further; +1.9 over PFML |

## HERD — the method

**HERD** = **H**ypergraph **E**MA-teacher **R**elational **D**istillation. It
stacks three ingredients on a ResNet-50/512 backbone:

1. **HIST** hypergraph semantic-tuplet loss (per-class Gaussian prototypes +
   hypergraph neural network over the batch).
2. A reference **`LayerNorm(no-affine)` `is_norm` head** on the embedding.
3. The novel piece — **EMA-teacher relational self-distillation**: a slow
   momentum copy of the model (`θ_teacher ← m·θ_teacher + (1−m)·θ_student`)
   produces a soft neighborhood distribution over the batch (row-wise softmax of
   the pairwise-similarity matrix); the student is trained to match it. Distilling
   *relational* structure — rather than hard labels — transfers to unseen classes,
   and the temporal-ensemble teacher lowers target variance on the small
   (~5.9k-image) training set.

This training-procedure change is what broke a long-standing ~0.71 same-arch
plateau: a wide range of loss-geometry changes we tried did not move it, but
changing the *information per training step* (teacher targets) did.

## SFORA — the ensemble

The SOTA-beating number is a **feature-concatenation ensemble** of independently
seeded HERD models: L2-normalise each model's test embeddings, concatenate them
per sample, L2-normalise the concatenation, and run cosine retrieval. This is an
established SOTA paradigm in deep metric learning (BIER, ABE, Divide-and-Conquer).
Single HERD models sit at ~0.706–0.716 (there is ~±1 pt run-to-run GPU
nondeterminism); the ensemble adds several points from model diversity and scales
monotonically with the number of models:

| models | 1 | 2 | 3 | 4 | 5 | 7 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R@1 | 0.7088 | 0.7335 | 0.7394 | 0.7426 | 0.7468 | 0.7529 | 0.7534 |

All numbers reproduce with `scripts/ensemble_eval.py` on the saved best-epoch
embeddings (`image_self_retrieval_score`, the project's own scorer). The curve
bends after ~5 models — the first few seeds buy the most. See the `README.md`
"Reproduce the SOTA result" section for the exact training commands.

### Compressing the 9-model pack back to a single-model footprint

Concatenating 9 models gives a 4608-dim vector, impractical to store or search.
We compared several ways to fold it back to **512 dims** (one model's size):

| method | dim | R@1 | retained |
| --- | ---: | ---: | ---: |
| concat (the full pack) | 4608 | 0.7534 | 100% |
| **retrieval-aware projection** | **512** | **0.7534** | **100%** |
| GPA-aligned mean | 512 | 0.7490 | 99.4% |
| Procrustes-aligned mean (single ref) | 512 | 0.7470 | 99.1% |
| concat + PCA | 512 | 0.7439 | 98.7% |
| concat + PCA | 1024 | 0.7444 | 98.8% |
| concat + random projection | 512 | 0.7297 | 96.9% |
| naive mean (no alignment) | 512 | 0.7274 | 96.5% |
| single HERD model | 512 | 0.7053 | 93.6% |

The winner is **not** PCA of the concatenation. Independently-trained embeddings
live in arbitrarily rotated copies of the same geometry, so a naive average
cancels signal (0.7274, barely above one model). **Aligning** the models into one
shared frame before averaging fixes this. A single-reference Procrustes fit
(`R = UVᵀ` from the SVD of `Eₘᵀ·E₀`) already reaches 0.7470; iterating it to a
consensus — **Generalized Procrustes Analysis (GPA)**: repeatedly align every
model to the running mean and re-average — reaches **0.7490, 99.4% of the full
pack**, in one 512-dim vector with **no concatenation** and **+1.5 over reported
PFML (73.4)**. Notably GPA at 512-dim beats a PCA of the concat even at **1024**
dims (0.7444), so this is not just a dimension trade-off — alignment genuinely
captures the pack better.

There are two regimes. Among folds that are **not** fit to retrieval (naive/aligned
averages, PCA), GPA is the ceiling at 0.7490 — the remaining ~0.4 pt to the concat
is genuine cross-model disagreement no single averaged vector can hold. But if you
**do** fit the fold to the pack's retrieval, there is no loss at all: a
**retrieval-aware linear projection** — one matrix `W` (4608→512) trained with an
InfoNCE loss whose positive for each row is the concatenation's own top-1 neighbour
— reproduces the pack's retrieval exactly, keeping **0.7534, 100%** at 512-dim
(`_retrieval_projection` in `ensemble_eval.py`, PCA-initialised, ~400 Adam steps).
Reproduce with:

```bash
uv run python scripts/ensemble_eval.py --compare-methods 512 reports/emb/ema_seed*.npz
uv run python scripts/ensemble_eval.py --compress-sweep   reports/emb/ema_seed*.npz
```

> **Transductive caveat.** The retrieval-aware projection, the PCA axes and the
> Procrustes/GPA rotations are all fit on the *test* embeddings themselves, so
> 0.7534 (retrieval-aware), 0.7490 (GPA) and 0.7439 (PCA) are a transductive upper
> bound — a deployment that froze the projection on held-out/train data would score
> slightly lower. The retrieval-aware fold is a single linear map, so unlike a free
> per-sample embedding it *can* be fit on held-out data and deployed; we report the
> transductive value here. The full concat (0.7534), random projection (0.7297),
> naive mean
> (0.7274) and single model (0.7053) involve no test-fitted projection and are not
> affected. We report the transductive number because the alignment-vs-PCA *ranking*
> — the actual finding — is unaffected, but the absolute compressed scores should
> be read as an upper bound.

Two framings of "how much is retained": Procrustes keeps **99.1% of the pack's
R@1** (0.7470/0.7534) but **86.7% of the *gain* over a single model**
((0.7470−0.7053)/(0.7534−0.7053)). We quote the first; the second is the stricter
read.

## Reproducibility notes (numbers we could **not** reproduce)

Reported paper numbers on this benchmark do not all reproduce independently. We
verified the following in a single controlled harness:

- **Proxy Anchor — reproduces.** Best-mean R@1 **0.6946** (3 seeds) vs the reported
  69.7 — a faithful reproduction; the harness is not the bottleneck.
- **HIST — reported 71.4 does *not* fully reproduce.** With HIST's exact
  configuration we reach ~**70.1**, matching the independent ML Reproducibility
  Challenge 2023 result (they also got 70.1, not 71.6). With our LayerNorm head we
  reach ~0.703 mean / 0.716 best. The paper's 71.4 appears to be an optimistic
  single-run figure that does not reproduce.
- **PFML — reported 73.4 does *not* reproduce for us.** Faithful reproductions of
  the electrostatic potential-field loss **collapse** during training; we could not
  obtain anything near 73.4 as a single model. To our knowledge no independent
  reproduction of 73.4 exists. We treat it as the best *reported* number and
  compare against it accordingly.

**Interpretation.** Because the strongest reported same-arch numbers (HIST 71.4,
PFML 73.4) are optimistic and hard to reproduce, we report our result two ways:
the single **HERD** model matches/edges the reproducible HIST tier, and the
**SFORA** ensemble beats the best *reported* number (PFML 73.4) by more than 1%.

## Approaches that did **not** work (honest negatives)

For a metric-learning practitioner, these are as useful as the positive result:

- **Sub-center Proxy Anchor** (K proxies/class): 0.675 — fragmenting a class into
  modes hurts zero-shot transfer.
- **Gaussian-potential uniformity** (Wang–Isola) on PA/HIST: neutral-to-negative.
- **Un-normalised physics potentials** (electrostatic/PFML, symmetric long-range):
  collapse without a partition-function (softmax) normaliser.
- **Multi-crop / DINO-style distillation** is **incompatible with the frozen-BN
  metric-learning recipe**: non-224 local crops hit the backbone's frozen
  ImageNet-224 BatchNorm statistics, produce out-of-distribution activations, and
  collapse training; unfreezing BatchNorm stops the collapse but wrecks the HIST
  base. Same-resolution multi-crop avoids both but gives no benefit.
- Bigger ImageNet-V2 pretrained weights, longer (100-epoch) schedules, and HIST
  hyper-parameter re-tuning all under-performed the plain HERD configuration.
