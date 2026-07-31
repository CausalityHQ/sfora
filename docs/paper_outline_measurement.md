# Paper outline: what this project can actually claim

Drafted 2026-07-31, after sixteen method candidates failed and two independent
literature reviews recommended stopping the search. This is the deliverable that the
evidence supports. It is not a method paper, and the central discipline of writing it is
refusing to dress it up as one.

**Working title.** *Best-over-training hides stable methods: a measurement audit of deep
metric learning.*

---

## 1. The claim

The standard DML protocol — report `max` over ~60 test evaluations of one run — is a
**maximum over noisy observations**, so it is upward-biased, and **the bias grows with a
method's noise**. Two methods with identical true performance therefore report different
numbers, and the less stable one wins. This is not a small correction:

| CUB arm | evaluates | selection bonus |
| --- | --- | ---: |
| `pa_distill` | student | 0.836 pt |
| `proxy_anchor` | student | 0.769 pt |
| `pa_ema_avg_fast` | averaged weights | 0.306 pt |
| `pa_ema_avg` | averaged weights | 0.074 pt |

On a *flat simulated plateau* with 0.5 pt evaluation noise, the estimator recovers
**+1.16 pt of pure selection** — larger than most published DML gains, from a ground truth
containing no improvement at all.

**The consequence that matters.** Removing each arm's own bonus **reverses a ranking**:

| paired comparison | reported | corrected |
| --- | ---: | ---: |
| `pa_ema_avg_fast` − `proxy_anchor` | +0.414 | **+0.732** |
| `pa_distill` − `proxy_anchor` | +0.658 | +0.592 |

Weight averaging — which the protocol under-credits because it *smooths the evaluated
curve* — is the stronger intervention, and the benchmark says the opposite.

## 2. Why this is not just Musgrave et al.

*A Metric Learning Reality Check* (ECCV 2020) established unfair comparisons, criticised
tuning on test classes, and ran ten seeds with confidence intervals. It did **not**:
decompose seed variance from GPU-nondeterministic variance; quantify the selection bias of
best-over-training; show that the bias *differs between arms* and therefore contaminates
comparisons rather than cancelling; or pre-register causal interventions and refute them.

## 3. Evidence, in the order it should be presented

1. **Fixed-seed nondeterminism.** Three effectively identical CUB runs at the *same seed*:
   0.7183 / 0.7154 / 0.7075 — a **1.08 pt spread with no seed difference at all**. More
   than half the apparent "seed variance" in this field is the GPU.
2. **Power is a property of the comparison, not the dataset.** Detecting +0.5 pt needs
   **5 seeds** on the Proxy Anchor leg and **17** on the HIST leg, because pairing removes
   most seed variance on one (0.60 → 0.37) and none on the other (0.68 → 0.73). A
   dataset-level σ cannot power a paired test.
3. **σ from three seeds is worthless.** Ours was 55% too high for arms and 2–4× too *low*
   for paired differences. `pa_distill`'s 3-seed paired sd was 0.153; at six it was 0.367.
4. **The winner's curse, measured.** In-sample seeds gave +0.890; out-of-sample **+0.427**.
   Screening estimates were inflated more than twofold.
5. **Selection bias** (§1), including the `local_nca` sanity check: a collapsed run that
   peaked in its first epochs still reported **0.5733 against a 0.3394 trend**.
6. **Two confounds the provenance system caught**: a LayerNorm mismatch between a method
   and its own control, and an EMAN-class BatchNorm mismatch between teacher and student
   costing 0.3–1.4 pt, recovering proportionally on two bases over six paired seeds.

## 4. The negative corpus

Sixteen candidates, each with a *mechanism* rather than a number: relational and
hypergraph distillation, balanced sampling, persistent memory, Sinkhorn coupling, local
NCA, region-based proxies, hubness correction, asymmetric query/gallery embeddings,
Procrustes fusion, Shepard's generalisation kernel, Tversky contrast similarity, and four
capacity/timescale variants.

**Six pre-registered predictions, all refuted**: headroom-proportional distillation;
capacity-starved regularisation; averaging-explains-distillation; additivity of the two
EMA roles; monotone gain in embedding width; and the momentum-contamination story once
corrected for selection bias. Each was written down *with a numeric falsification
condition before the deciding run*. That is the section reviewers will not have seen
before.

## 5. Threats to validity, stated by us first

- The correction is one estimator (leave-one-out neighbour mean). Corrected values are
  **not** the benchmark metric; a reader who wants the leaderboard number should use the
  reported one. The argument is about the protocol, not a proposal to restate scores.
- Averaging arms are at n=2–3 on some cells.
- Single architecture family (ResNet-50/512, BN-Inception/512) and three datasets.
- The deeper problem is that best-test checkpointing is **test-set feedback**, not merely
  a noisy estimator. The right fix is class-disjoint validation checkpointing, which
  Musgrave et al. already advocated; debiasing is a diagnostic, not a remedy.

## 6. What we do not claim

No novel method. Weight averaging is Polyak/Izmailov; the BatchNorm fix is EMAN (Cai et
al. CVPR 2021), independently rediscovered here. The contribution is that this subfield
has not adopted the fix, that its benchmark protocol structurally hides the technique, and
that both effects are large enough to change conclusions.

---

## Open items before submission

- `pa_dual_ema` on CUB (queued): the last untested cell. Expected to fail; recorded either
  way.
- Cars196 under corrected recipes has **never been run** — the matrix is CUB + In-Shop
  only, and Cars is one of the three benchmarks claimed.
- Selection-bias numbers should be recomputed at n≥3 for every averaging arm.
- Verify the BLENDER citation (arXiv 2026) directly rather than second-hand.
