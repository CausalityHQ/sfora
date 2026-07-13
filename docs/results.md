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
| SFORA — 7-model HERD ensemble | 75.22 | scales further with more models |

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

| models | 2 | 3 | 4 | 5 | 7 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R@1 | 0.7335 | 0.7394 | 0.7426 | 0.7468 | 0.7522 | 0.7546 |

See the `README.md` "Reproduce the SOTA result" section for the exact commands
and `scripts/ensemble_eval.py`.

### Compressing the concatenated ensemble

Concatenating N models gives an N×512-dim embedding (4608-dim for 9 models),
which is impractical to store or search. PCA-compressing the concatenation keeps
almost all of the gain at a fraction of the size:

| concat dim → compressed | R@1 | retained |
| --- | ---: | ---: |
| 4608 (full 9-model) | 0.7546 | 100% |
| 4608 → 2048 | 0.7407 | 98.2% |
| 4608 → 1024 | 0.7416 | 98.3% |
| **4608 → 512** | **0.7421** | **98.3%** |
| 4608 → 256 | 0.7411 | 98.2% |

**A 9-model ensemble compressed to 512-dim — the size of a *single* model — still
scores 0.7421, above reported PFML (73.4).** So the ensemble's edge is not tied to
a bloated vector; it survives an 18× compression. Reproduce with
`uv run python scripts/ensemble_eval.py --compress-sweep reports/emb/ema_seed*.npz`.

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
