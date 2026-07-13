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
established SOTA paradigm in deep metric learning (feature-concatenation ensembling,
e.g. BIER). Single HERD models sit at **0.705 mean / 0.716 best across 9 seeds**
(measured standard deviation σ ≈ 0.006 across seeds); the ensemble adds several
points from model diversity and scales monotonically with the number of models:

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
| **GPA-aligned mean** | **512** | **0.7490** | **99.4%** |
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

GPA is the ceiling among folds that use only the embeddings' geometry: the
remaining ~0.4 pt to the concat is genuine cross-model disagreement no single
averaged vector can hold. **We do not close it by fitting a projection to the test
set** — that would be test-set overfitting, and reporting the resulting number is
not honest.

#### The honest, inductive answer: fit the fold on the disjoint train split

The legitimate way to compress is to fit the projection on the disjoint **train**
classes, freeze it, and only then apply it to test — nothing about the test split
informs the fold. We ran this on a 3-seed HERD pack (each seed exports its
best-epoch train and test embeddings via `--save-train-embeddings` /
`--save-test-embeddings`); `scripts/train_fit_fold.py` fits each 512-dim fold on
the train concat and evaluates it, frozen, on the test concat:

| 512-dim fold (this 3-seed pack) | R@1 | vs concat | vs single |
| --- | ---: | ---: | ---: |
| full concat (1536 dims) | 0.7259 | 100% | — |
| **PCA fit on train** | **0.7078** | **97.5%** | **+1.4 pt** |
| Proxy-Anchor head fit on train | 0.7076 | 97.4% | +1.4 pt |
| single HERD model | 0.6940 | 95.6% | — |

So a genuinely train/test-clean projection recovers **~97.5%** of the pack at one
model's footprint and beats a single model by **+1.4 pt** — but it does **not**
reach 100%. The concat *is* the 100% point (no compression); closing the last
~2.5 pt at 512 dims would require fitting the projection to the test set, which we
refuse. That is the honest ceiling for an inductive fold. (The transductive GPA
number below, 0.7490, is higher because it uses the test embeddings' own geometry
— see the caveat.) Reproduce with:

```bash
uv run python scripts/train_fit_fold.py --dim 512 \
    --train 'reports/emb/herd_tt_seed*.train.npz' \
    --test  'reports/emb/herd_tt_seed*.test.npz'
```

Reproduce the transductive folds above with:

```bash
uv run python scripts/ensemble_eval.py --compare-methods 512 reports/emb/ema_seed*.npz
uv run python scripts/ensemble_eval.py --compress-sweep   reports/emb/ema_seed*.npz
```

> **Transductive caveat.** The PCA axes and the Procrustes/GPA rotations use only
> the embeddings' geometry (no labels, no retrieval targets), but they are *computed
> on the test embeddings themselves*, so 0.7490 (GPA) and 0.7439 (PCA) are a
> transductive upper bound — a deployment that froze the projection on held-out/train
> data would likely score slightly lower. The full concat (0.7534), random projection
> (0.7297), naive mean (0.7274) and single model (0.7053) involve no fitted projection
> at all. We do **not** fit any projection to the test set's *retrieval* (labels or
> nearest-neighbour targets) to inflate the compressed number — that would be
> test-set overfitting.

Two framings of "how much is retained": Procrustes keeps **99.1% of the pack's
R@1** (0.7470/0.7534) but **86.7% of the *gain* over a single model**
((0.7470−0.7053)/(0.7534−0.7053)). We quote the first; the second is the stricter
read.

## Cars196 — a second dataset

The same protocol on Cars196 (ResNet-50/512, zero-shot split, best-over-training).
Here the HERD recipe (tuned on CUB) does **not** transfer at the single-model level:

| method | R@1 | provenance |
| --- | ---: | --- |
| Proxy Anchor (reported) | 87.7 | paper |
| HIST (reported) | 89.6 | paper |
| Proxy Anchor (our run) | 88.5 | reproduces above the reported 87.7 |
| HERD — single (our run) | 87.1 | *below* our own Proxy Anchor run |
| **SFORA — 3-model ensemble** | **90.3** | above reported PA (87.7) and HIST (89.6) |

A single HERD model (0.871) lands between the reported PA and HIST and below our
own PA reproduction (0.885) — the `is_norm`+EMA recipe was tuned on birds and does
not transfer as-is. But the **ensemble still wins**: a 3-model SFORA-HIST-style
pack of HERD models reaches **0.903**, the entire gain coming from the pack.

## SFORA on raw HIST — what does the ensemble alone buy? (ablation)

To separate the ensemble from the HERD recipe we ensembled **plain HIST** models
(no `is_norm` head, no EMA teacher) the same way. Cumulative first-N CUB seeds:

| models | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFORA-HIST | 0.6972 | 0.7242 | 0.7330 | 0.7402 | 0.7443 |
| SFORA-HERD | 0.7088 | 0.7335 | 0.7394 | 0.7426 | 0.7468 |

The ensemble is the **main driver**: a pack of raw HIST models clears reported PFML
(0.734) at 4 models (HERD clears it at 3) and reaches 0.7443 at 5. The full HERD
recipe then adds a steady margin — **~0.7 pt single-model (0.705 vs 0.698 mean) and
~0.25 pt at 5 models (0.7468 vs 0.7443)**. This isolates HERD vs plain HIST, not the
EMA term alone. Reproduce: `ensemble_eval.py reports/emb/hist_only_seed*.npz`.

## Reproducibility notes (numbers we could **not** reproduce)

Reported paper numbers on this benchmark do not all reproduce independently. We
verified the following in a single controlled harness:

- **Proxy Anchor — reproduces.** Best-mean R@1 **0.6946** (3 seeds) vs the reported
  69.7 — a faithful reproduction; the harness is not the bottleneck.
- **HIST — reported 71.4 does *not* fully reproduce.** Plain HIST in our harness
  reaches ~**70.1** best (and ~0.698 mean over 5 seeds), consistent with independent
  reproductions that also land near 70 rather than 71.4. The **full HERD recipe**
  (HIST + `is_norm` head + EMA-teacher distillation) reaches **0.705 mean / 0.716
  best** over 9 seeds — we did not measure the `is_norm` head in isolation, so 0.716
  is a full-HERD number, not a LayerNorm-only one. The paper's 71.4 appears to be an
  optimistic single-run figure that does not reproduce.
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
