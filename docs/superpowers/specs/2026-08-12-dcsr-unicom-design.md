# DCSR-UNICOM: Deployment-Consistent Subspace Risk

**Status:** rejected at design review; do not implement

**Closure (2026-08-12):** adversarial review found that the objective is a
hard two-group DRO variant overlapping slimmable/sandwich training, nested
dropout, and Group-DRO, while its authorization statistic tests coordinate
non-exchangeability that UNICOM's uniform mask objective should remove.  The
review also discovered that the official four-rank implementation samples a
different feature mask on each rank before combining the class-shard logits in
one distributed softmax.  That source-level distributed objective, not DCSR,
is the next mechanism-audit target.  This document is retained as negative
research evidence; none of its experiment stages is authorized.

**Target:** improve UNICOM's supervised In-Shop retrieval frontier while
preserving a single 512-dimensional, gallery-independent descriptor.

## 1. Motivation

The official UNICOM implementation trains and deploys different geometries.
For ViT-B/16 on In-Shop, the model emits 768 coordinates and the recipe uses
`num_feat=512`.

During training, `PartialFC_V2.forward` draws a fresh random set of 512
coordinates and then normalizes the selected embedding and selected prototype.
During evaluation, `get_metric` first normalizes all 768 coordinates, then keeps
the first 512 coordinates, and ranks with Euclidean distance without
renormalizing the truncated vectors.

Consequently:

1. training sees a random subspace while deployment always sees one prefix;
2. training uses unit vectors within the selected subspace while deployment
   permits prefix-energy variation to alter Euclidean ranking; and
3. test-split R@1 is exposed every epoch and embedded in checkpoint filenames,
   creating a selection affordance even though the code does not automatically
   select a best checkpoint.

This is source evidence, not yet evidence that the mismatch harms R@1. The first
stage is therefore a no-training falsifier.

Primary evidence:

- UNICOM paper: <https://arxiv.org/abs/2304.05884>
- official source: <https://github.com/deepglint/unicom>
- inspected source revision:
  `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
- Matryoshka Representation Learning:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html>
- DADA: <https://ojs.aaai.org/index.php/AAAI/article/view/29400>

## 2. Alternatives

### 2.1 DCSR-UNICOM (selected)

Train the exact deployed prefix and a random subspace together. Optimize the
worse of their two classification risks. This directly repairs the deployment
view without discarding UNICOM's random-feature compactness.

### 2.2 Mean two-view loss (matched Matryoshka control)

Average the fixed-prefix and random-subspace losses. This is a necessary control
but not the candidate: it is close to Matryoshka-style multi-capacity training,
and a strong random view can numerically conceal a weak deployment view.

### 2.3 Evaluation-only normalization repair

Slice the 512-coordinate prefix and then normalize it before Euclidean ranking.
This exactly matches the metric used by UNICOM's selected-subspace training.
It is a zero-training control and possible operational fix, not a new learning
mechanism.

## 3. Method

Let the encoder emit `z_i in R^768` and let the PartialFC prototype for class
`c` be `w_c in R^768`. Let `P` select coordinates `[0, 512)`. At each training
step, let `R_t` be a uniformly sampled set of 512 distinct coordinates.  The
official implementation samples `R_t` independently in each rank's PartialFC
process; it is shared by the gathered minibatch only within that rank.  The
resulting rank-local class-shard logits are then combined by one distributed
softmax.

For a selector `S`, define

```text
u_i^S = S z_i / clamp(||S z_i||_2, min=1e-12)
q_c^S = S w_c / clamp(||S w_c||_2, min=1e-12)
s_ic^S = <u_i^S, q_c^S>
ell_S  = ArcFaceCrossEntropy(s^S, y)
```

using the official ArcFace margin, scale, class set, and reduction.

The selected objective is

```text
L_DCSR = max(ell_P, ell_Rt)
```

The identity

```text
max(a, b) = (a + b + |a - b|) / 2
```

shows that DCSR is the mean two-view risk plus a fixed, parameter-free risk-gap
penalty. No new continuous loss weight is introduced. At non-tied points, only
the worse subspace determines the gradient. Exact ties use PyTorch's defined
subgradient and are recorded diagnostically.

The matched mean control is

```text
L_MEAN = (ell_P + ell_Rt) / 2.
```

The official control remains `L_RANDOM = ell_Rt`.

At deployment, DCSR uses

```text
d_P(x, x') = ||normalize(P z(x)) - normalize(P z(x'))||_2.
```

This is equivalent to cosine ranking in the 512-dimensional prefix and exactly
matches the normalization order used to construct `ell_P`.

## 4. Why this is not a renamed existing method

- **UNICOM** samples one feature subspace per step and never makes the fixed
  deployed prefix an explicit competing risk. DCSR adds a paired deployment
  view and a worst-view objective.
- **Matryoshka Representation Learning** trains an ordered family of nested
  prefixes, conventionally by summing their losses. DCSR has exactly one
  deployed prefix plus a stochastic exchangeability view and optimizes the
  worst paired risk. The mean control measures how much of any gain is merely
  Matryoshka-style prefix supervision.
- **DADA** aligns sample and proxy distributions through augmented intermediate
  domains and adversarial discriminators. DCSR has no discriminator, feature
  mixup, or sample/proxy domain classifier. DADA is a possible later composition
  only after DCSR independently passes.
- **SoftTriple, CCP-DML, and HIER** change the class representative or class
  geometry. DCSR changes which coordinate projection must carry the same class
  decision.
- **MCPS-PG** constrains a sparse subset of encoder gradients using memory
  centroids in the BN-Inception/Proxy-Anchor lane. DCSR acts on every UNICOM
  minibatch and targets a measured train/deploy contract, not centroid safety.

The claim is a combination/repair contribution, not that random subspaces,
nested embeddings, or distributionally robust optimization are individually
new.

## 5. Prospective no-training falsifier

### 5.1 Inputs

- official UNICOM ViT-B/16 released checkpoint;
- official DeepFashion In-Shop query/gallery split;
- one deterministic FP32 export of every full 768-dimensional query and gallery
  embedding;
- the official image transform and one-image inference path;
- no test-time augmentation and no gallery-dependent learned transform.

The exporter records the model identifier, source revision, dataset split
counts, and SHA-256 hashes of the embedding arrays. Ordinary Git and a JSON
experiment report are sufficient; no handoff or authorization framework is
part of this design.

### 5.2 Scores

From the same frozen full embeddings, compute:

1. `official`: normalize 768, take prefix 512, Euclidean rank;
2. `prefix_unit`: take prefix 512, normalize, Euclidean rank;
3. `random_unit[j]`: select and sort 512 coordinates drawn without replacement
   using NumPy `Generator(PCG64(j))`, for `j=0..31`; normalize the selected
   vectors and rank by Euclidean distance.

All ties use gallery order as the stable secondary key. R@1 is evaluated for
all 14,218 queries against all 12,612 gallery images.

Also record per-query top-1 correctness, selected-subspace norm, top-1 identity,
and top-10 identity list. Bootstrap intervals use 10,000 paired query resamples
from `PCG64(205)` and the ordinary percentile interval.

### 5.3 Frozen decisions

Define

```text
delta_norm = R1(prefix_unit) - R1(official)
delta_mask = median_j R1(random_unit[j]) - R1(prefix_unit)
mask_wins  = count_j[R1(random_unit[j]) > R1(prefix_unit)]
disagree   = median_j mean_q[top1_random_j(q) != top1_prefix(q)]
```

Decisions are mutually exclusive and evaluated in this order:

1. **DCSR premise PASS** if `delta_mask >= 0.002`, `mask_wins >= 24`, and
   `disagree >= 0.10`. Continue to the training smoke.
2. **Evaluation-only repair** if DCSR does not pass, but `delta_norm >= 0.002`
   and the paired 95% bootstrap lower bound for `delta_norm` is positive.
   Adopt post-slice normalization and close DCSR training.
3. **CLOSE** otherwise. Do not change these thresholds after observing results.

If both training and evaluation conditions hold, correct the evaluator first,
then compare all training methods with the corrected evaluator.

The zero-shot checkpoint is a premise test. It cannot establish a supervised
SOTA improvement.

## 6. Training evidence ladder

Training starts only after a DCSR premise PASS.

### 6.1 Algebraic and CPU tests

- selectors contain exactly 512 unique in-range coordinates;
- `P` is exactly `[0, 512)`;
- normalization occurs after selection for embeddings and prototypes;
- DCSR equals `torch.maximum(ell_P, ell_R)`;
- swapping which view is worse swaps the active gradient;
- exact ties remain finite and symmetric;
- the official random-only path is unchanged when DCSR is disabled;
- evaluation prefix scoring is identical to cosine ranking;
- fixed seeds reproduce masks and reports byte-for-byte on CPU.

### 6.2 Eight-epoch seed-0 smoke

Use the official ViT-B/16 In-Shop supervised recipe, shortened only to eight
epochs, and compare from the same initial checkpoint:

- official random-only UNICOM;
- mean two-view control;
- DCSR.

The smoke passes only if all are true:

- DCSR corrected-prefix R@1 exceeds both controls by at least `0.002`;
- DCSR median random-mask R@1 is no more than `0.001` below random-only;
- both view losses are finite at every step;
- each view is the active maximum on at least 10% of steps after epoch one;
- measured epoch time is no more than 1.35 times random-only.

Otherwise DCSR closes without a full run.

### 6.3 Full matched experiment

Only after the smoke passes, run the official 128-epoch ViT-B/16 recipe for
seeds 0, 1, and 2, paired across random-only, mean-control, and DCSR. GPU runs
are treated as statistically reproducible, not bitwise deterministic.

DCSR counts as a matched mechanism improvement only if:

- its paired final R@1 delta versus both controls is positive in at least two of
  three seeds;
- its paired mean delta versus random-only is at least `0.002` and exceeds its
  sample standard error;
- its mean R@1 is at least the official reproduced random-only mean;
- it retains one 512-dimensional prefix descriptor and exact cosine-equivalent
  inference; and
- no best-epoch selection is substituted for the registered final checkpoint.

## 7. SOTA boundary

The published UNICOM In-Shop points are 95.5 R@1 for ViT-B/16, 96.0 for
ViT-L/14, and 96.7 for ViT-L/14@336. A positive DCSR delta on ViT-B/16 is only a
matched mechanism result. A SOTA claim requires:

1. reproducing the relevant official UNICOM baseline locally;
2. exceeding the strongest comparable published point under the same backbone,
   input size, descriptor dimension, and inference lane;
3. paired multi-seed uncertainty;
4. replication on at least two of CUB, Cars196, and SOP; and
5. comparison against contemporary strong baselines without mixing rerankers,
   ensembles, transductive scoring, or different pretraining regimes.

LOCORE remains a separate local-reranking lane and is not a descriptor baseline.

## 8. Components and boundaries

Implementation, if authorized by the falsifier, should have four isolated
units:

1. a pure selector-plus-normalizer used by training and evaluation;
2. a paired-view PartialFC loss wrapper that exposes both scalar losses and the
   active view;
3. a frozen-embedding diagnostic evaluator with no training imports; and
4. an experiment report writer that records masks, metrics, gates, and timing.

The existing UNICOM source is imported or adapted minimally. The first
falsifier must not modify model weights or launch training.

## 9. Failure handling

- Nonfinite embeddings, zero selected norms, duplicate mask indices, incomplete
  query/gallery counts, or source/model mismatch invalidate the attempt.
- Existing reports are never overwritten.
- A failed premise closes this exact mechanism; it does not authorize threshold
  changes, mask cherry-picking, or a different prefix selected from test data.
- Any new mechanism after closure receives a new prospective design and gates.
