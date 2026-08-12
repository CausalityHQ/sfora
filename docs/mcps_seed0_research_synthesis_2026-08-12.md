# MCPS seed-0 research synthesis

Date: 2026-08-12

## Scope and status

This is a prospective research triage, not the three-seed MCPS decision.  It
uses the already disclosed seed-0 reports only to identify the next cheapest
falsifiers.  It does not change any running command, threshold, seed, or model.
The original PA -> MCPS sequence remains authoritative, followed by the frozen
proxy-compactness controls and the frozen UNICOM audit.

## What seed 0 establishes

| Quantity | Proxy Anchor | MCPS-PG | MCPS minus PA |
|---|---:|---:|---:|
| final R@1 | 0.9136306090870727 | 0.9164439442959629 | +0.0028133352088902 |
| best R@1 | 0.9170769447179632 | 0.9173582782388522 | +0.0002813335208890 |
| best minus final | 0.0034463356308905 | 0.0009143339428893 | -0.0025320016880012 |
| final mAP@R | 0.6490443891671460 | 0.6503176256156041 | +0.0012732364484581 |

The final-R@1 gain is positive, but 90.0% of it is explained by reduced
best-to-final decay: `0.0025320016880012 / 0.0028133352088902`.  The seed-0
evidence therefore supports an optimization-stability hypothesis much more
strongly than a representation-ceiling hypothesis.  PA's best checkpoint also
exceeds MCPS's final checkpoint by 0.0006330004220003 R@1.  Three paired seeds and the
compactness control remain necessary because one seed cannot distinguish a
repeatable mechanism from GPU noise or checkpoint luck.

The full-run conflict rate is 0.0015853827965840908, far below the frozen 0.05
diagnostic even though the one-epoch smoke passed at 0.05415012464223064.  The
implementation aggregates conflict counts over all 8,580 steps, so the two
numbers measure different training horizons.  The original gate is retained
without reinterpretation; the paired analyzer reports the diagnostic failure
separately from effect estimates.

## Candidate triage

### 1. Solenoidal memory drift — close without execution

An antisymmetric update in the embedding-memory plane,
`g' = g + gamma (m z^T - z m^T) g`, is formally distinct from MCPS's
dissipative projection.  Non-reversible perturbations can accelerate mixing,
but their defining theory preserves the stationary distribution rather than
improving the attainable optimum.  That targets convergence speed, while the
observed problem is a 1.5--4 point representation gap to modern anchors.

### 2. Learned Tversky retrieval — close as the next mechanism

A differentiable Tversky feature bank could score common and distinctive
features with separate query and gallery weights.  This is not a sufficiently
novel or well-matched next mechanism:

- Rahnama and Huellermeier already learn Tversky similarity on image data
  ([arXiv:2006.11372](https://arxiv.org/abs/2006.11372)).
- Doumbouya, Jurafsky, and Manning already provide differentiable Tversky
  feature/projection layers and a frozen-image-encoder adapter
  ([arXiv:2506.11035](https://arxiv.org/abs/2506.11035)).
- In-Shop relevance is same-item identity and is semantically symmetric.  Query
  and gallery are protocol partitions, not different semantic roles.  An
  `alpha != beta` gain can therefore encode partition artifacts instead of a
  transferable notion of identity.
- The symmetric version remains a possible off-the-shelf control on future
  frozen frontier embeddings, but it is an application of occupied prior art,
  not the requested new similarity mechanism.

### 3. Retrieval-induced forgetting — close before training

An inverted-U penalty on persistent class competitors is a falsifiable transfer
from retrieval-induced forgetting, but confusion-graph supervision and
non-monotone negative weighting are already close to occupied hard-negative and
confusion-flow families in this repository.  Its decisive test requires another
full training arm, while its plausible effect scale is below the current
backbone/pretraining gap.  It fails the cost-versus-capability test.

## Decision and next evidence

Do not add a new BN-Inception training arm.  The next evidence is:

1. finish all paired PA/MCPS seeds unchanged;
2. run the already frozen proxy-compactness controls even if the diagnostic
   controller stops after reporting the low conflict rate;
3. report both the frozen MCPS gate and a stricter mechanism-specific verdict;
4. run the released-weight UNICOM ViT-B/16 export and frozen E1/E2 audit;
5. reopen mechanism invention only on the strongest reproducible modern
   representation, where a head or similarity operator can plausibly affect
   the 93--95.5 R@1 frontier rather than polishing a 91.x baseline.

Checkpoint averaging or EMA/SWA should be included as a conventional stability
control in any future baseline training design.  Existing seed-0 directories
contain only the final checkpoint, so a post-hoc last-k average cannot be
constructed honestly from the current run.

This closure is deliberately reversible: a three-seed MCPS ceiling gain, a
compactness-control separation, or the UNICOM audit can supply new evidence.
Absent one of those, another train-time mechanism is not justified.
