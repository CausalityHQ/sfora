# Benchmark results

All numbers are **CUB-200-2011**, ResNet-50 backbone, 512-dim embedding, the
standard zero-shot retrieval split (100 train / 100 disjoint test classes),
cosine **Recall@1**, reported as **best-over-training** (the protocol used by the
papers below — evaluate the held-out test classes every few epochs and take the
peak).

> **Status correction (2026-07-20).** Results below predate strict
> publication-backed method×dataset recipes and are retained as historical
> `modified_legacy` evidence. They must not be called official Proxy Anchor or HIST
> reproductions. Corrected reference/selected-extension experiments have been queued
> on the DGX; this document will add a separate table only after recipe IDs and
> digests validate. No legacy number is used to choose an unpublished recipe.

## Corrected reference-recipe evidence

The first corrected results have landed, and they are **negative for the method**.

### CUB-200 under the official recipes (seed 0 — single seed, screening only)

The comparison the headline claim never had. Both HIST arms carry official
LayerNorm; the arms differ only in the declared distillation delta.

| arm | R@1 | vs published | vs its base |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.6825 | reported 0.697 (**−1.45**) | — |
| PA + distillation | **0.6916** | — | **+0.91** |
| HIST | **0.7183** | reported 0.714 (**+0.43**) | — |
| HERD (HIST + distillation) | 0.7156 | — | **−0.27** |

Three things follow, and the first is the important one.

**1. The legacy "HERD beats HIST" result was the LayerNorm, not the
distillation.** The legacy pair was HIST 0.700 (control run *without* embedding
LayerNorm) → HERD 0.716, read as +1.6 for distillation. Corrected, with LayerNorm
held constant as official HIST specifies: HIST 0.7183 → HERD 0.7156, i.e.
**−0.27**. The entire historical gain is accounted for by the confound. This is
the single clearest outcome of the audit so far.

**2. HIST reproduces above its published number** (0.7183 vs 0.714). Proxy Anchor
lands 1.45 points below its own (0.6825 vs 0.697).

*Caveat on that second half, added after checking.* Calling this a "reproduction
gap" was premature at **one seed**: CUB seed noise in this harness has historically
been σ ≈ 0.6 pt, so 1.45 points is ≈ 1.4σ — unremarkable. A line-by-line fidelity
audit against the official repository also found nothing to blame it on:

| checked | official Proxy Anchor | ours |
| --- | --- | --- |
| CUB ResNet-50 command | `--batch-size 120 --lr 1e-4 --warm 5 --bn-freeze 1 --lr-decay-step 5` | identical |
| warmup semantics | freeze backbone, train embedding + proxies only | identical (`backbone_warmup_parameters`) |
| train transform | `RandomResizedCrop(224)`, torchvision default scale | identical |
| eval transform | `Resize(256)` + `CenterCrop(224)` | identical |
| proxy LR multiplier | `lr * 100` | identical |

(The audit was prompted by a specific suspicion — that we used torchvision's
default crop scale where the authors set `scale=(0.16, 1)`. The authors do **not**;
they call `RandomResizedCrop(224)` bare. Hypothesis wrong, fidelity confirmed.)

So the honest statement is that PA's single-seed number sits low but within noise,
with no identified fidelity defect. Seeds 1 and 2 settle it.

### The first arm to beat HIST — and it is the one predicted not to

| arm (CUB seed 0) | R@1 | vs HIST | peak epoch | decay |
| --- | ---: | ---: | ---: | ---: |
| HIST | 0.7183 | — | 27 / 41 | −0.16 |
| HERD — pairwise EMA distillation | 0.7156 | −0.27 | 19 | −1.05 |
| herd_hg — propagated HGNN logits | 0.7174 | −0.09 | 26 | −0.13 |
| **herd_hg_incidence — Mahalanobis prototype affinity** | **0.7235** | **+0.52** | 22 | −0.61 |

**This inverts the argument that motivated the work, and the inversion is the
finding.** `herd_hg_incidence` was built as an *ablation control*, not a candidate:
`test_only_the_propagated_target_is_a_genuine_n_ary_quantity` proves its target
`H_i` depends solely on sample `i`'s own distances to the class Gaussians, is
invariant to the rest of the batch, and is therefore ordinary dark-knowledge KD over
Mahalanobis-proxy logits — **expressible without any hypergraph and carrying no
novelty claim**. The arm that *does* carry the novelty claim, the propagated HGNN
logits whose normalisation depends on the whole batch's hyperedge population, scored
−0.09.

So on this evidence the n-ary property is not what helps. The plausible reading,
consistent with §2b: the propagated target mixes the entire batch and is
correspondingly noisier, while the per-sample prototype affinity is a cleaner,
lower-variance target. Its curve supports that — it is far ahead early (0.6931 at
25% of the run against HIST's 0.6487) and peaks at 54%.

Caveats that matter more than the number:

* **One seed, and +0.52 is below CUB's σ ≈ 0.6.** This is a screening signal, not a
  result. Seeds 1 and 2 are running; a claim needs ≥ 6 seeds (see the p-value floor
  in `research_reset_plan.md` Phase 4).
* If it survives, the honest framing is **not** "hypergraph-native distillation
  works". It is "distilling an EMA teacher's class-prototype affinities helps a
  strong HIST base" — closer to Hinton-style KD than to anything novel, and the
  novelty question would have to be re-asked from scratch.

**2b. What the training curves say: the distillation is a regulariser.** The
best-over-training numbers hide the mechanism; the curves show it plainly.

| arm | best R@1 | peak epoch | position in run | decay after peak |
| --- | ---: | ---: | ---: | ---: |
| Proxy Anchor | 0.6825 | 12 / 60 | **20%** | **−1.35 pt** |
| PA + distillation | 0.6916 | 20 / 60 | 33% | −0.95 pt |
| HIST | 0.7183 | 27 / 41 | **66%** | **−0.16 pt** |
| HERD (pairwise distill) | 0.7156 | 19 / 41 | 46% | −1.05 pt |
| herd_hg (hypergraph distill) | 0.7174 | 26 / 41 | 63% | −0.13 pt |

Read down the "position in run" column:

* **Proxy Anchor overfits hard** — it peaks a fifth of the way through and then
  sheds 1.35 points. Distillation pushes the peak out to a third of the run and
  lifts it by +0.91. That is textbook regularisation.
* **HIST barely overfits** — it peaks two thirds through and sheds 0.16. It has no
  use for another regulariser, and the pairwise target actively *over*-regularises
  it: the peak is pulled forward to 46% and the decay triples to −1.05.
* **The hypergraph target is gentler** — it leaves HIST's curve shape essentially
  intact (63%, −0.13) and lands −0.09 instead of −0.27. Directionally better than
  pairwise, still not an improvement.

This single mechanism accounts for every distillation result we have, including
In-Shop, and it resolves the apparent contradiction in §2: the effect tracks how
much headroom the base has, not the dataset or the loss family. It also says
plainly what beating HIST requires — **added structure, not added regularisation.**

**3. Distillation helps the weaker base and not the stronger one.** On CUB it adds
+0.91 to Proxy Anchor (0.6825, underfitting) and subtracts 0.27 from HIST (0.7183).
Combined with In-Shop, where it costs both bases, the picture is:

| base | CUB (BatchNorm frozen) | In-Shop (BatchNorm trainable) |
| --- | ---: | ---: |
| Proxy Anchor | **+0.91** | −0.41 |
| HIST | −0.27 | −1.39 |

Two effects appear to be superimposed: a roughly −1.2 pt shift moving from CUB to
In-Shop for *both* bases (consistent with the BatchNorm teacher/student mismatch,
or simply with dataset size), and within each dataset a benefit that accrues only
to the weaker, underfitting base (consistent with a variance-reducing regularizer).
The queued `_bnfix` runs separate the first effect from the second.

Single seed — CUB seed noise has historically been σ ≈ 0.6 pt, so −0.27 is well
inside noise while +0.91 is roughly 1.5σ. Neither is a result yet.

**DeepFashion In-Shop**, best-over-training R@1, 3 seeds. Proxy Anchor rows use
the official `reference` recipe; HIST rows use a frozen `selected_extension`
(HIST published no In-Shop recipe, so it was selected from SOP using
**training-split-only** scoring — no test leakage). Within each base, the plain
and distilled arms differ in `ema_distill_weight`/`ema_momentum`/`ema_distill_tau`
and nothing else (`derive_recipe`); `embedding_layer_norm` is held constant, so
the legacy confound cannot recur.

| arm | seed 0 | seed 1 | seed 2 | mean | Δ vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy Anchor | 0.9024 | 0.9048 | 0.9032 | **0.9035** | — |
| PA + distillation | 0.8999 | 0.8994 | 0.8990 | **0.8994** | **−0.41 pt** |
| HIST | 0.9046 | 0.9037 | 0.9031 | **0.9038** | — |
| HERD (HIST + distillation) | 0.8906 | 0.8892 | 0.8900 | **0.8899** | **−1.39 pt** |

All six paired per-seed deltas are negative (PA: −0.25 / −0.55 / −0.41; HIST:
−1.40 / −1.45 / −1.31). With the correct paired test (df=2): HERD vs HIST
t = −33.9, p ≈ 0.0009 — robust; PA+distill vs PA t = −4.75, p ≈ 0.042 — marginal,
and only under a normality assumption that three points cannot evidence. The
assumption-free exact sign test cannot go below 0.25 at n=3 for either leg, so
both are screening evidence. HIST − PA is +0.03 pt — the two bases are
indistinguishable on this dataset.

**iNaturalist 2018** — recorded for completeness, but the *recipe* is broken, not
just the method. Both arms peak at **epoch 5 of 60** and decay thereafter, because
the `selected-from-cars` recipe transfers Cars196 hyperparameters (8k images, 98
classes) to a ~450k-image / ~8k-class dataset:

| arm | best R@1 | best epoch | final R@1 |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.2099 | 5 / 60 | 0.1734 |
| PA + distillation | 0.2094 | 5 / 60 | (run stopped at epoch 46; best already fixed at epoch 5) |

These arms are 0.05 pt apart — statistically empty. iNat needs its own recipe work
before it can support any conclusion.

**CUB and Cars under official reference recipes have never been run.** That matrix
is now running; it is the experiment that decides whether the headline claim
survives. See [research_reset_plan.md](research_reset_plan.md).

## Headline (legacy recipes — see status correction above)

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
2. The official HIST **`LayerNorm(no-affine)` `is_norm` head** on the embedding
   (baseline behavior, not a HERD addition).
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

## The distillation on small datasets — broad gains (legacy recipes)

> **Claim withdrawn (2026-07-27).** This section was previously titled "The
> distillation is universal — and beats Proxy Anchor on every dataset". That
> universality claim is **falsified** by the corrected In-Shop matrix, where the
> same distillation *regresses* both bases across 3 seeds (PA −0.41 pt, HIST
> −1.39 pt; see "Corrected reference-recipe evidence" below). Every number in
> this section is CUB or Cars under **legacy** recipes.
>
> The surviving, weaker reading: the gains below are real measurements on the two
> *smallest* benchmarks (CUB 5.9k train images, Cars 8.1k) and the regressions are
> on a much larger one (In-Shop ~25k, 3997 classes). That pattern is what a
> **variance-reducing regularizer** looks like — helpful when data is scarce,
> pure capacity cost when it is not. Testing that hypothesis is the point of the
> sweeps in [research_reset_plan.md](research_reset_plan.md) §5.

In our harness the distill term is applied ungated on top of whatever objective is
training, so it can augment Proxy Anchor just as it augments HIST. We measured it
in-harness (same code, same protocol, reseeded where noted):

| dataset | base loss | plain | **+ our distillation** | Δ |
| --- | --- | ---: | ---: | ---: |
| CUB-200 | HIST | 0.700 | **0.716** (= HERD)† | +1.6† |
| CUB-200 | Proxy Anchor | 0.666 | 0.678 | +1.2 |
| Cars196 | HIST | 0.871 | 0.884 (= HERD)† | +1.3† |
| Cars196 | Proxy Anchor | 0.888 | **0.8961** | +0.8 |

† These historical HIST → HERD rows used a legacy HIST control without the
LayerNorm that official HIST enables. Their Δ is therefore a combined
head-plus-distillation change and is **not** the corrected paired estimate. In the new
recipe system both HIST and HERD retain official LayerNorm, and HERD changes only the
declared EMA distillation fields. Proxy Anchor rows remain historical until rerun from
their exact dataset recipe as well.

**It is not specific to HIST or Proxy Anchor.** To check that the distillation is a
*general* training-procedure improvement rather than a two-loss coincidence, we ran
it on three more bases (CUB, seed 0, same protocol). It lifts every one:

| additional base (CUB, seed 0) | plain | + our distillation | Δ |
| --- | ---: | ---: | ---: |
| SupCon | 0.580 | 0.611 | +3.2 |
| triplet (semi-hard) | 0.492 | 0.511 | +1.9 |
| batch-hard triplet | 0.244 | 0.617 | +37.3 |

So across **five** bases — HIST, Proxy Anchor, SupCon, triplet, batch-hard triplet —
the EMA-teacher relational distillation improves retrieval every time **on CUB**.
(It does not on In-Shop; see the corrected table below.) The batch-hard
row is a special case worth naming honestly: our plain batch-hard baseline collapses
(0.244, a known hard-mining failure mode), and the distillation *stabilises* it back to
a competitive 0.617 — so that +37 is "rescued a collapse", not a uniform-quality gain.
The SupCon and triplet rows are ordinary healthy baselines lifted by a few points. (These
three are single-seed spot checks, not reseeded means like the HIST/PA rows above.)

**On CUB and Cars, the distillation improves every base tested.** And on those two
datasets the *best base + our distillation* beats every plain baseline:

- **CUB** — HIST is the stronger base, so **HERD (HIST + distillation) = 0.716** is
  best (> Proxy Anchor 0.666/0.695, > HIST 0.700).
- **Cars** — Proxy Anchor is the stronger base, so **PA + distillation = 0.8961**
  is best (mean over 3 seeds `[0.8944, 0.8974, 0.8963]`, every seed above PA's best
  single run 0.8892; > PA 0.8879, > HERD 0.8835, > HIST 0.8709).

So **our method beats Proxy Anchor on both datasets** — via the *same procedure*
applied to whichever base is stronger. Honest caveats we verified:

- **No single fixed loss is best everywhere.** A fused `HIST + Proxy Anchor`
  objective in one model is a *compromise worse than the best base on both*
  datasets (CUB 0.69 < 0.716; Cars 0.880 < 0.896). HIST genuinely wins CUB, Proxy
  Anchor genuinely wins Cars; mixing them helps neither. The unifying method is the
  distillation *procedure*, not one loss.
- **Single HERD does not beat Proxy Anchor on Cars** (reseeded mean 0.8835 < 0.8879)
  — the HIST base is simply weaker than PA there. We reached "beats PA on Cars" only
  by putting the distillation on the *PA* base, and we say so rather than pretending
  the HIST-based HERD wins there. This is not for lack of tuning: an exhaustive sweep
  of every HIST-internal lever (samples-per-class, LR schedule, distillation
  temperature/weight/momentum, variance floor, hypergraph cross-entropy weight
  `λ_s ∈ {0.5, 2.0}`, incidence sharpness `α`, and HGNN width) leaves the HIST base
  plateaued at ~0.884 seed-0 — every variant landed *at or below* the baseline, none
  reached PA's 0.8857. HIST is genuinely the weaker Cars base; the distillation
  procedure is what transfers, so we apply it to PA there. The dead-end holds at **three
  independent levels**: (1) HIST-internal tuning above; (2) the ensemble — the HIST-based
  HERD ensemble (0.9026) trails the PA+distill ensemble (0.9172); and (3) **cross-teacher
  distillation** — distilling a frozen *trained PA* teacher's relational geometry into a
  HERD/HIST student (`teacher_checkpoint` + `teacher_similarity_weight`) scored 0.8737/0.8743
  (teacher weight 0.5/1.0), *below* HERD's 0.884, because the PA teacher's geometry conflicts
  with HIST's hypergraph rather than complementing it.
- Relaxing HIST's variance floor (an ablation) is a null/negative result; the
  faithful `relu6` default stands.

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

These historical numbers recompute with `scripts/ensemble_eval.py` on the saved best-epoch
embeddings (`image_self_retrieval_score`, the project's own scorer). The curve
bends after ~5 models — the first few seeds buy the most. See the `README.md`
for the corrected publication-backed training command; it intentionally does not
promise the legacy score before rerunning.

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
| group-SupCon-XBM head fit on train | 0.7039 | 96.9% | +1.0 pt |
| single HERD model | 0.6940 | 95.6% | — |

Notably the **unsupervised PCA edges both supervised metric-learning heads** — for
folding an already-trained pack the discriminative geometry is already present, so
re-optimising it on train labels only risks overfitting the train classes.

**At the aggressive 512-dim footprint, the best inductive fold is an *uncentered*
train-fit projection.** Retrieval is cosine similarity, so a projection onto an
orthonormal train-fit basis (the top-k right singular vectors of the raw, **un-mean-
centered** train concat) is cosine-preserving — whereas subtracting the train mean
shifts test cosines (the train mean is not the test mean) and caps every *centered*
fold (PCA, whitening) at ~98%. On the 5-seed pack this uncentered fold recovers
**98.9%** at 512-dim (0.7272 of the 0.7350 concat), beating both train-fit GPA
alignment (**98.0%**, 0.7205 — align each model to a train consensus, then average)
and centered PCA (97.4%).

**And "decrease dims to 100%" is achievable honestly — just not at 512-dim.** The five
models are highly correlated (they encode the same classes), so the concat's effective
retrieval rank is well below 2560. Sweeping the uncentered train-fit projection's
target dimension (nothing fit on test) shows it stays **lossless** as it discards the
lowest-energy directions:

| train-clean uncentered fold | R@1 | retained |
| --- | ---: | ---: |
| 512 dims | 0.7272 | 98.94% |
| 1024 dims | 0.7318 | 99.56% |
| 1536 dims | 0.7331 | 99.75% |
| 1792 dims | 0.7343 | 99.91% |
| **2048 dims** | **0.7350** | **100.00%** |
| full concat (2560) | 0.7350 | 100% |

So a **train/test-clean projection reduces the pack 2560 → 2048 dims (a 20% cut) with
zero retrieval loss** — the honest reading of "decrease vectors dims to get 100%,
trained on train." What is *not* achievable without fitting on test is 100% at the
single-model **512-dim** footprint (a 5× cut): the bottom ~500 directions still carry
~1% of retrieval signal, so an honest 512-dim fold tops out at 98.9%, and only a fold
that peeks at the test geometry (transductive GPA, 99.4% below) closes more. Reproduce
both with:

```bash
# 512-dim folds (GPA, PCA, uncentered) + the dimension-vs-retention sweep
uv run python scripts/explore_trainclean_projection.py \
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
On Cars the HIST base is weaker than Proxy Anchor, so the **HIST-based HERD does not
beat PA** — but our distillation *procedure* on the **PA** base does (see the
universal-distillation section above). In-harness, reseeded where noted:

| method | R@1 | provenance |
| --- | ---: | --- |
| HIST (our run) | 87.1 | in-harness baseline |
| Proxy Anchor (our run) | 88.8 | reseeded mean; above reported 87.7 |
| HERD = HIST + distillation | 88.4 | reseeded mean; *below* PA — HIST base weaker |
| **PA + our distillation** | **89.6** | reseeded mean `[0.8944, 0.8974, 0.8963]`; **beats PA** |
| SFORA — HERD (HIST) ensemble, 3 models | 90.3 | HIST-based ensemble; still below the PA-based one |
| **SFORA — PA+distill ensemble, 3 models** | **91.7** | best Cars ensemble (PA is the stronger Cars base) |

The HIST-based single HERD (mean 0.8835) lands below our own PA reproduction (0.8879)
— the HIST base simply loses to PA on Cars. The honest win is the **distillation
procedure**: applied to the stronger PA base it gives **PA + distillation = 0.8961**
(every seed above PA's best single run 0.8892), so **our method beats PA on Cars**. A
single fused HIST+PA loss is a compromise worse than each base (0.880), so no single
fixed loss is best everywhere. The **ensemble** confirms the base-adaptive story at a
higher level: the PA+distill ensemble (**0.9172**) beats the HIST-based HERD ensemble
(**0.9026**) — HIST is the weaker Cars base at *both* single-model and ensemble scale,
so on Cars we ensemble the PA base.

## SOP — a third dataset (a genuine reproduction gap, thoroughly investigated)

Stanford Online Products (11,318 train classes, ResNet-50/512, best-over-training). At
this scale the **base-adaptive story holds and extends**: Proxy Anchor is again the
stronger base, and our distillation on it wins.

| method (SOP, seed 0) | R@1 | note |
| --- | ---: | --- |
| **PA + our distillation** | **~0.72** | stronger base at scale; distill neutral here |
| HIST-HERD | 0.678 | HIST is weaker at 11k-class scale, as on Cars |

So across all three datasets the pattern is consistent — **HIST wins CUB, Proxy Anchor
wins Cars and SOP**, and "best base + our distillation" is the method each time.

**Honest caveat: our SOP Proxy Anchor reproduces at ~0.72, ~7.5 pt below the reported
0.796, and we could not close it.** This is not for lack of trying — we ran a full
investigation:

- **Hyperparameters (8 configs):** batch 120/180/256, lr 1e-4/2e-4, LR decay
  γ 0.25/0.5, samples-per-class 2/3/4 (spc=2 sees all 11,317 classes; spc=4 silently
  excluded 36% of them), the `is_norm` LayerNorm head on/off, 60 vs 90 epochs, and
  distillation on/off. **Every config plateaus at 0.712–0.721.** is_norm was neutral
  (0.719); 90 epochs peaked at 0.721 then overfit down.
- **Implementation audit (code trace):** the Proxy-Anchor loss normalization (positive
  term over |P⁺|, negative over all 11,318 proxies), proxies excluded from weight decay,
  the disjoint first-half/second-half class split, the standard `Resize(256)→CenterCrop`
  eval transform, and the eval protocol (60,698 test images / 11,317 classes ≈ the
  standard 60,502 / 11,316) are all faithful.

So SOP behaves like HIST (our 0.701 vs reported 0.714) and PFML (collapses): **the
reported number is hard to reproduce, not a knob we failed to turn.** We report our
honest ~0.72 as supporting evidence for the base-adaptive finding, **not** a SOTA claim.

## DeepFashion In-Shop and iNaturalist 2018 — results pending

The harness now supports the official DeepFashion In-Shop train/query/gallery
partition and the project-defined `inat2018-zero-shot-species-v1` protocol. No result
number is claimed here yet: the three-seed Proxy Anchor, PA+distillation, HIST, and
HERD artifacts must exist and pass dataset preflight before any table is published.

In-Shop results will use query-to-gallery retrieval with R@1/10/20/30 as the canonical
cutoffs (plus the harness's R@2/4/8 and MAP@R). iNaturalist results will be labeled as
SFORA's project protocol, not as a canonical iNaturalist metric-learning benchmark or
a SOTA comparison. The exact setup and sequential runner are documented in
[`library_usage.md`](library_usage.md#deepfashion-in-shop-and-inaturalist-2018).

### Recipe matrix used by the corrected queue

| Dataset | Base method | Recipe | Backbone | Source / expected R@1 | Status |
| --- | --- | --- | --- | --- | --- |
| CUB | Proxy Anchor | `proxy_anchor.cub.official-51db570` | ResNet-50 | official repo 69.9 | registered |
| CUB | HIST | `hist.cub.official-e7d650c` | ResNet-50 | paper 71.4±0.2 | registered |
| Cars | Proxy Anchor | `proxy_anchor.cars.official-51db570` | ResNet-50 | official repo 87.7 | registered |
| Cars | HIST | `hist.cars.official-e7d650c` | ResNet-50 | paper 89.6±0.2 | registered |
| SOP | Proxy Anchor | `proxy_anchor.sop.official-51db570` | BN-Inception | official repo 79.2 | registered |
| SOP | HIST | `hist.sop.official-e7d650c` | ResNet-50 | paper 81.4±0.2 | registered |
| In-Shop | Proxy Anchor | `proxy_anchor.inshop.official-51db570` | BN-Inception | official repo 91.9 | queued reference |
| In-Shop | HIST | `hist.inshop.selected-from-<winner>-e7d650c` | winner preserved | no published pair | train-only selection queued |
| iNat2018 v1 | Proxy Anchor | `proxy_anchor.inat2018.selected-from-<winner>-51db570` | winner preserved | no published pair | train-only selection queued |
| iNat2018 v1 | HIST | `hist.inat2018.selected-from-<winner>-e7d650c` | winner preserved | no published pair | train-only selection queued |

Primary sources are the pinned [Proxy Anchor official repository](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/tree/51db57031e38f75c03f69bbdfad1a3233afd9787)
and [HIST official repository](https://github.com/ljin0429/HIST/tree/e7d650c80460f464c55bcdc2262d785923c50dc4),
with HIST expected scores from its CVPR 2022 Table 3. `pa_distill` and `herd`
inherit the resolved base recipe unchanged and add only their recorded EMA delta.

### DGX handoff audit (2026-07-20)

At `2026-07-20T10:53:15+02:00`, before changing remote state, the legacy controller
was captured on `spark-2751`:

- controller PID `1595927`, process group/session `1595925`, command
  `/home/riomus/experiment-logs/run-inshop-matrix.sh`;
- active trainer PID `1612789` plus eight DataLoader workers, running legacy
  In-Shop PA-distill seed 1 at step 3700/12360;
- GPU 0 (`NVIDIA GB10`) was 96% utilized;
- completed legacy artifacts were In-Shop seed-0 PA, PA-distill, HIST, HERD and
  seed-1 PA. Their JSON files and per-run logs were preserved in place;
- the controller script checksum was
  `1f7f5f000f85dfd832708b126a90a6d4d6a7c79b0261acece9aa2c43b01379e5`.

A second legacy controller was then found waiting independently for iNaturalist:
PID `1596894`, process group `1596892`, script
`run-inat2018-matrix.sh` (SHA-256
`77b2bff7475fc98f40ec4723233371a540ac9d28790f0c21a2d7c1e68b29d3a6`). It had
started legacy PA/iNat seed 0 and reached step 1400/130380. Both captured legacy
process groups were terminated with SIGTERM; their logs and partial state remain
recoverable, and no unrelated process was signaled.

The captured legacy R@1 values were PA seed 0 `0.8636`, PA-distill seed 0 `0.8633`,
HIST seed 0 `0.7731`, HERD seed 0 `0.6831`, and PA seed 1 `0.8590`. These are the
poor common-preset results that triggered the recipe audit; they are retained as
`modified_legacy`, not reference evidence.

Deployment then passed both dataset preflights and installed the official
`bn_inception-52deb4733.pth` checkpoint after verifying its full SHA-256
`52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353`. A one-step,
two-class In-Shop smoke run loaded that checkpoint and emitted the official recipe
identity plus BN-Inception, batch 180, LR `6e-4`, updating BatchNorm, step-20,
γ=`0.25`, and shuffled batches. Its cap-related changes are correctly recorded as a
`modified` smoke artifact.

The corrected detached controller is PID `1619657` (log
`logs/reference_recipes.controller.log`). Its first full command is Proxy Anchor /
In-Shop seed 0 from recipe `proxy_anchor.inshop.official-51db570`, digest
`50137fe7f9d84cb567ee092f68b3bee6be58716dfab4d9bfbe964de8bc78fe57`; trainer PID
`1619742` was observed at 95% GPU utilization. The controller will run the paired
PA-distill seeds, perform frozen training-only selection for unsupported pairs, and
then execute their base/derived seed matrices. A duplicate controller created while
repairing PID-file detachment was detected immediately and its isolated process group
was stopped; PID `1619657` is the sole surviving corrected controller.

## SFORA on raw HIST — what does the ensemble alone buy? (ablation)

The historical ablation ensembled a legacy control labeled **plain HIST** that omitted
both `is_norm` and the EMA teacher. Because official HIST includes `is_norm`, this is
not an official HIST baseline. Cumulative first-N CUB seeds:

| models | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFORA-HIST | 0.6972 | 0.7242 | 0.7330 | 0.7402 | 0.7443 |
| SFORA-HERD | 0.7088 | 0.7335 | 0.7394 | 0.7426 | 0.7468 |

The ensemble is the **main driver**: a pack of raw HIST models clears reported PFML
(0.734) at 4 models (HERD clears it at 3) and reaches 0.7443 at 5. The full HERD
recipe then adds a steady margin — **~0.7 pt single-model (0.705 vs 0.698 mean) and
~0.25 pt at 5 models (0.7468 vs 0.7443)**. This isolates the old combined
LayerNorm-plus-EMA change, not the corrected EMA-only HERD delta. Recompute:
`ensemble_eval.py reports/emb/hist_only_seed*.npz`.

## Legacy reproducibility observations (official-recipe reruns pending)

The old common-preset harness produced the observations below. The audit found enough
recipe drift that none of them is evidence for or against the official recipes; the
new digest-tracked runs must finish before drawing that conclusion.

- **Proxy Anchor:** the legacy run reached best-mean R@1 **0.6946** over three seeds,
  close to the reported CUB value, but it did not carry an official recipe digest.
- **HIST/HERD:** the legacy control reached about **0.698 mean**, while the combined
  LayerNorm-plus-EMA variant reached **0.705 mean / 0.716 best** over nine seeds. Since
  official HIST already includes LayerNorm and uses a different optimizer, sampler,
  schedule, and dataset-specific parameters, this is not a valid official HIST
  reproduction or a clean EMA ablation.
- **PFML:** legacy attempts collapsed and did not approach the reported 73.4. This
  recipe audit covers Proxy Anchor and HIST, so it does not upgrade those attempts to
  an official PFML reproduction claim.

**Interpretation.** The strongest reported same-architecture numbers remain reference
targets. Only corrected artifacts with a `reference` or frozen `selected_extension`
track can update the comparison.

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
