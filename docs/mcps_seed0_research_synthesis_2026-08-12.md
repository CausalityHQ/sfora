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
control in any future baseline training design.  This is not merely a proposed
implementation: `ImageEndToEndConfig.ema_weight_averaging` and the
`ema_weight_average` evaluation-model path already exist in
`src/sfora/image_end_to_end.py`, but the frozen PA and MCPS-PG recipes leave the
switch off.  Because roughly 90% of the seed-0 final-score delta is reduced
late-training decay, any later MCPS attribution must include an EMA-on PA arm
before crediting the memory-centroid mechanism.  Existing seed-0 directories
contain only the final checkpoint, so a post-hoc last-k average cannot be
constructed honestly from the current run.

The official UNICOM source also exposes a sharper conditional candidate than a
new BN-Inception loss.  In its four-rank PartialFC path, each rank samples a
different 512-of-768 coordinate subset but the class-shard logits are combined
inside one global softmax.  This means the denominator compares logits computed
in different similarity spaces.  Separately, the evaluator normalizes all 768
coordinates, truncates to 512, and does not renormalize, introducing a
gallery-dependent prefix-energy term into Euclidean ranking.  The existing E1
and E2 audit is the prospective falsifier: evaluator renormalization or full
768 dimensions are simpler controls, while synchronized feature masks are a
training candidate only if E2 shows material independent-mask sensitivity that
the coherent-mask control removes.  Matryoshka Representation Learning covers
[prefix-consistent normalized embeddings](https://arxiv.org/abs/2205.13147),
and [PartialFC](https://arxiv.org/abs/2203.15565) covers class-center sampling;
neither makes cross-rank feature sketches coherent.  This is a defect-repair
hypothesis, not yet an improved method or a SOTA result.

This closure is deliberately reversible: a three-seed MCPS ceiling gain, a
compactness-control separation, or the UNICOM audit can supply new evidence.
Absent one of those, another train-time mechanism is not justified.

## Seed-1 checkpoint

The unchanged seed-1 pair completed after this note's prospective decisions
were frozen:

| Quantity | Proxy Anchor | MCPS-PG | MCPS minus PA |
|---|---:|---:|---:|
| final R@1 | 0.9160219440146293 | 0.9180616120410747 | +0.0020396680264454 |
| best R@1 | 0.9189759459839640 | 0.9199606133070756 | +0.0009846673231115 |
| best minus final | 0.0029540019693347 | 0.0018990012660008 | -0.0010550007033339 |
| final mAP@R | 0.6517248216313138 | 0.6523812637090788 | +0.0006564420777649 |

Seeds 0 and 1 are both positive in final R@1; their provisional paired mean is
`+0.002426501617667798`.  This clears the frozen `+0.0015` effect-size floor at
the two-seed checkpoint but cannot be evaluated against the three-seed standard
error or compactness control yet.  Seed 1 again contradicts the proposed
conflict mechanism: memory-target rate is `0.9973510748510749` and skip rate is
zero, but conflict rate is only `0.0013315510069732762`, versus the frozen
`0.05` requirement.  The final effect remains potentially useful, but any
surviving explanation must be stability/regularization rather than the claimed
conflict-projection activation.

## Late-trajectory noise diagnostic

The apparent best-to-final decay was checked without another GPU run.  For each
completed 60-epoch trajectory, fit an OLS line to epochs 45--59, detrend those
15 values, and circular-block-bootstrap the residuals with length-three blocks,
100,000 `PCG64(212)` draws.  The null statistic is the bootstrapped window
maximum minus its final value.  This is an exploratory attribution diagnostic,
not a replacement for the frozen three-seed decision.

| Arm | late slope / epoch | residual SD | observed late max-final | fitted drift contribution | null p(gap >= observed) |
|---|---:|---:|---:|---:|---:|
| PA seed 0 | +0.0000497357 | 0.001390012 | 0.003446336 | 0 | 0.41040 |
| MCPS seed 0 | -0.0000464703 | 0.001173043 | 0.000914334 | 0.000650584 | 0.66747 |
| PA seed 1 | -0.0001477001 | 0.001116667 | 0.002954002 | 0.002067801 | 0.24980 |
| MCPS seed 1 | -0.0000195929 | 0.001053594 | 0.001899001 | 0.000274300 | 0.46544 |

None of the observed gaps is unusual under its stationary residual null.
Seed-0 PA even has a slightly positive fitted late slope; seed-1 PA has a
negative slope, but its max-final gap remains within ordinary evaluation
fluctuation.  Consequently, reduced best-to-final decay cannot currently be
claimed as MCPS's causal benefit.  The positive paired final deltas remain real
observations, but their mechanism is unresolved and must be compared against
EMA-on PA and proxy compactness.  No additional BN-Inception stability
mechanism is authorized by these curves.
