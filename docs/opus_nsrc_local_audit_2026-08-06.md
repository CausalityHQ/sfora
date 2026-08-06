# Pass 41 local evidence-aware audit: NSRC

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_nsrc_proposal_pass41_2026-08-06.md`  
Blind-proposal job: `8d9377e3a3d64e2b` (direct Opus after the first
Fable-fallback job recursively consulted and was cancelled)  
Proposal source SHA-256: `53ba3319caa37c73cb62eac6b87f4cb391d9fe785f71796fda72ba7cfe3c2437`  
Committed copy SHA-256: `ea16c9a60bacc5794fe4d850b468592db79f6f17a4fe040cb9f2afde37f66aa2`
(the committed copy differs only by one terminal newline)

This audit was written before launching or reading the mandatory independent
review. It binds the blind proposal to the repository evidence and audits the
method exactly as frozen rather than silently replacing the 95%-energy
projector with a full proxy-span projector.

## Verdict

**DEAD at Gate 1 and Gate 2; independently dead because the frozen projector
invalidates the claimed orthogonality theorem and the only forecast frontier
arm can have no null space. No preregistration, implementation, or candidate
GPU.**

The log-determinant is a legitimate finite coding-rate objective, and the
fixed-projector feature gradients of the two terms occupy complementary
coordinate subspaces under stronger assumptions than the proposal uses. Those
pieces do not establish the measured premise, novelty, executable theorem, or
frontier result.

## Gate 1: no repository measurement establishes proxy-complement collapse

NSRC requires the following measurements before an intervention:

1. a corrected baseline's descriptor energy in the exact proxy complement
   decays monotonically below `0.15`;
2. this decay causes official-query retrieval errors rather than merely
   accompanying successful class separation;
3. augmentation-invariant information in that complement separates unseen
   identities; and
4. the effect follows the proposed class-count/proxy-rank dose across datasets.

None exists. F6 is a proposed future premise test, not Gate-1 provenance. The
claims that one proxy “annihilates ~80%” of the descriptor, that PFML works by
raising proxy rank, and that In-Shop/SOP should be null are hypotheses inferred
from hyperparameters, not corrected measurements in this repository.

The closest locked evidence is adverse or already exhausted. Candidate 365's
Blind-Subspace Allocation had the same unmeasured proxy-complement-headroom
premise. Pass 30's Null-Space Provisioning found that a centroid complement was
not label-null and that a class-exchangeable deployed block diluted retrieval.
Pass 21 RIM and Pass 36 DARC found no corrected evidence that augmentation-
repeatable residual information is useful rather than stable nuisance.
Candidate 225's disjoint-identity subspace-transfer ratios were
`0.9312/0.9287/0.9345`, all below its locked `1.15` threshold. Candidate 371,
Pass 29 DSA, and Pass 39 FCS further record that a proxy/classifier rank ceiling
is dimensional algebra, not an observed encoder-capacity deficit.

## The frozen 95%-energy projector makes the central theorem false

Let the normalized proxy-loss feature gradient be

```
g_PA = sum_c w_c p_c - alpha hhat.
```

The first term lies in the **full** proxy span. NSRC, however, defines `Pi` from
only the leading eigenvectors containing 95 percent of proxy energy. In
general, the omitted proxy tail has nonzero projection into `range(Pi_perp)`:

```
Pi_perp g_PA = Pi_perp sum_c w_c p_c - alpha Pi_perp hhat.
```

The first term is not generally parallel to `b=Pi_perp h`. Therefore the task
gradient can rotate `b`, and the claimed pointwise orthogonality to the null
loss is false. F5's required cosine below `1e-6` is incompatible with the
specified `tau=0.95` except by empirical accident. Replacing `Pi` with the
exact full proxy span would repair this algebraic statement but would be a
different frozen method and would destroy the proposed combined arm below.

Even under a full-span repair, feature-gradient orthogonality does not imply
optimization noninterference. With shared backbone/head Jacobian `J`, two
orthogonal descriptor gradients have parameter gradients `J^T g_1` and
`J^T g_2`, whose inner product is `g_1^T J J^T g_2`, generally nonzero. AdamW,
BatchNorm state, and proxy/projector drift add further coupling. This is the
same shared-backbone error recorded for candidate 365 and Pass 30.

The projector is recomputed every 50 iterations and its rank is the smallest
integer crossing a hard cumulative-energy threshold. Eigenvalue crossings can
rotate `Q_perp`, and threshold crossings change the dimension of `c`. The loss
and its optimizer state therefore change discontinuously; the proposal gives
no transport rule for the code when the basis or dimension changes.

## The escape and coding-rate claims are false as written

`L_rate` uses `logdet(I + a Chat Chat^T)`. Its eigenvalues are at least one, so
rank deficiency does **not** send the loss to infinity. A rank-one or constant
normalized code has a finite objective; collapsed directions contribute
`log(1)=0`. Minimizing negative log-determinant can prefer greater rank, but
the advertised infinite barrier is absent.

At exact complement collapse, both escape terms fail operationally:

- `chat=c/||c||` is undefined because no stabilizing epsilon is specified;
- the energy ratio `q=||b||^2/||h||^2` has zero first derivative with respect
  to `b` at `b=0`, so the squared hinge has zero first-order restoring force.

Thus the claimed strict exclusion of `||b|| -> 0` is neither a well-defined
loss nor a gradient barrier. The rate objective is also bounded by batch rank,
`rank(Chat Chat^T) <= min(B,d-r)`, so “high rate” cannot certify useful
identity information in a 512-D deployed complement.

The strongest semantic degeneracy remains open. Two views of one image plus a
batch volume reward can encode instance identity, stable background, crop
artefacts, or other augmentation-invariant nuisance. D1 treats near-chance
training-class accuracy as reassuring, but a class-exchangeable null code is
exactly what dilutes cosine retrieval. With energy share `gamma=0.15`,

```
cos_total = 0.85 cos_task + 0.15 cos_null.
```

If positive and negative `cos_null` distributions match, the enforced channel
adds noise rather than unseen-class supervision. C4 can diagnose this only
after spending GPU; it does not identify the method ex ante.

## The only forecast frontier arm is internally undefined

NSRC alone forecasts `0.706` CUB and `0.893` Cars, below its own PFML references
`0.734/0.927`. The claimed result exists only for “NSRC + multi-proxy.” Yet the
proposal's causal story says 15 proxies per class raise the proxy span toward
the full 512 dimensions. With 1,500 CUB or 1,470 Cars proxies, a generic proxy
matrix is full rank. Then `r=512`, `d-r=0`, `Q_perp` is empty, and the frozen
loss divides by zero in both its dimension normalization and its empty code.

If the 95%-energy truncation leaves a complement, it does so only by
reclassifying the omitted five percent of **supervised proxy directions** as
null directions, which is precisely what invalidates the orthogonality proof.
The proposal therefore cannot simultaneously retain its theorem, its null
space, and its only forecast frontier crossing.

The energy floor creates a second zero-set problem. If the null code is
retrieval-irrelevant but augmentation-stable, it forces at least 15 percent of
descriptor energy into it. The full-space cosine can then be worse even when
`L_inv` and `L_rate` are optimized perfectly. This is not a mere tuning risk;
it contradicts the claim that rate in an unsupervised complement is
necessarily useful deployed content.

## Gate 2: the supervision object and action recur internally and publicly

NSRC's exact wrapper is new, but the claimed training object is not: preserve
augmentation-invariant, sample-specific information outside a class-
discriminative component and deploy it jointly with that component.

- Candidate 365 Blind-Subspace Allocation already split Proxy Anchor from an
  auxiliary complementary block and filled the block with residualized
  relations. Its different content estimator does not make NSRC's occupied
  shared/private decomposition new.
- Pass 30 Null-Space Provisioning used an online class-statistic complement,
  same-image augmentation supervision, sample-specific contrast, and an energy
  floor. Switching centroid span to proxy span and contrast to coding rate does
  not create a new supervision relation.
- Pass 21 RIM and Pass 36 DARC already attempted to preserve augmentation-
  repeatable class-residual information and expand its rank/capacity. NSRC is a
  less class-conditioned rate estimator for the same action.
- DiVA (Milbich et al., ECCV 2020) jointly learns class-discriminative and
  complementary self-supervised relations from the same DML training data and
  aggregates them into one deployed representation. MIC (Roth et al., ICCV
  2019) explicitly learns cross-class intra-class characteristics such as
  viewpoint and illumination in an auxiliary encoder. Fixed blocks versus a
  live proxy-derived block is an allocation wrapper, not a new supervision
  source.
- Anti-Collapse Loss for DML already applies a coding-rate expansion objective
  to sample features or proxies alongside pair/proxy losses. MCR-squared is the
  primary coding-rate source. Limiting the same rate action to a derived
  subspace would need a valid, measured causal separation to support novelty;
  NSRC has neither.
- S2SD supplies auxiliary high-dimensional information during DML training and
  compresses it into the deployed descriptor. It is not an exact collision,
  but it further occupies the claimed “unused descriptor capacity” objective.

Primary sources:

- DiVA: <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530579.pdf>
- MIC: <https://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf>
- Anti-Collapse Loss: <https://arxiv.org/abs/2407.03106>
- MCR-squared: <https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html>
- S2SD: <https://proceedings.mlr.press/v139/roth21a.html>

The proposal admits its search was one batch and did not inspect the primary
sources exhaustively. The internal recurrences alone are sufficient for Gate
2; the primary papers independently occupy the decomposition and rate actions.

## Protocol and experimental defects

1. The protocol requires corrected paired In-Shop screening after Gates 1--3.
   NSRC instead predicts In-Shop should be near-null and makes five-seed CUB
   and Cars its first decision. A method whose scoping prediction excludes the
   project's efficient screen needs positive provenance before changing the
   protocol; none exists.
2. `B/2` unique images with two views is not exposure-matched to a baseline
   with `B` unique images. It changes the number of identities/examples and
   all pair/proxy combinatorics. The decisive control needs the same paired-
   view batch structure with `lambda=0`, as RIM correctly specified earlier.
3. The proposal fixes an unresolved PA recipe (`B=128`, no decay) rather than
   the corrected repository recipe and does not supply raw-best versus final-
   checkpoint metrics or a selection-correction analysis.
4. `gamma=1` makes the advertised folded deployment matrix exactly `W`; this
   is clean deployment but not a new calibration mechanism.
5. The `+/-` values are unsupported forecast SDs, not paired seed data or
   paired standard errors. The proposed frontier margins of 0.3--0.8 points
   cannot be claimed from them.
6. A `B x B` Cholesky and eigendecomposition are plausibly small relative to a
   backbone, but two-view data loading/augmentation, basis changes, and
   backward computation make an exact `1.00x` wall-clock claim unmeasured.

## What survives

- For a **fixed exact full proxy-span projector**, the projection of the
  proxy-loss feature gradient into its complement is radial.
- A loss depending only on a normalized complement code has a feature gradient
  tangent to that code and inside the complement.
- The Gram-form log-determinant is a compact finite coding-rate calculation.
- C2, C3, C4, C5 and a paired-view `lambda=0` baseline would be useful controls
  for some future measured proposal.
- Measuring exact full-proxy-complement energy and its retrieval relevance is
  a legitimate CPU/checkpoint diagnostic. It is not evidence already present.

Process lesson: a truncated energy subspace is not a span for orthogonality
proofs. Check whether the proposed combination erases the very complement on
which the method depends, and distinguish descriptor-gradient orthogonality
from shared-parameter-gradient orthogonality before claiming noninterference.

## Reconciliation with the independent cold review

The cold reviewer returned **DEAD at Gate 1** without seeing this audit and
independently found the same earliest failure: the 95%-energy projector leaves
proxy-tail directions in the alleged null block, so the omitted tangential
Proxy Anchor gradient invalidates both frozen theorems. In a disclosed
synthetic construction its complement proxy energy was `0.0491/0.0493`, the
tangential/radial norm ratio was `2.23`, and the task/null feature-gradient
cosine was `9.2e-3`, far above frozen F5's `1e-6`. These magnitudes are
construction results, not checkpoint measurements; only the algebraic
possibility of a nonzero tail term is required for the kill.

It also independently retained the bounded-rate, zero-gradient floor,
undefined normalization, batch-rank, deployed-noise, full-rank multi-proxy,
shared-parameter-interference, moving-basis, exposure-control, missing-
provenance, weak-recipe, and protocol failures. Its paired retrieval table is
a stylized simulation and is not promoted to repository evidence. Likewise,
the statement that `tau=0.95` *guarantees* approximately five percent omitted
energy is too strong in general: the hard cumulative threshold bounds the tail
at at most five percent and a spectrum can cross with a smaller or zero tail.
The exact objection is conditional and sufficient: whenever the truncation
omits nonzero proxy energy, the theorem is false; when it omits none, the
claimed large null block is absent.

The review added a useful control omission: `PA + L_f` alone is needed because
the floor has the largest coefficient and could account for any gain without
rate coding. It also identified Domain Separation Networks as another public
shared/private orthogonal-allocation neighbour. Neither changes the Gates 1
and 2 verdict.
