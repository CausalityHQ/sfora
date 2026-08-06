# Pass 38 local evidence-aware audit: CHIMERA

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_chimera_proposal_pass38_2026-08-06.md`  
Frozen proposal SHA-256 (file with the repository-standard terminal newline):
`bacb1dfe1f55287c3bf548092391b5fb0cbc4dad961669b16657a34aa3486a19`  
Durable blind consultation: `ee129cf981db4845` (Fable credit failure followed by
the configured Claude Opus fallback in the same job)

This audit was written without reading a second review. It is a local Gate 0--2
analysis, not an implementation plan. No preregistration, code, or GPU is
authorized by it.

## Local disposition

**DEAD at Gate 1 and independently DEAD at Gate 2; no GPU.** CHIMERA has no
verified repository measurement that identifies a finite-training-class rank
bottleneck or shows that shared atom recombination repairs retrieval errors. Its
only claimed empirical support is PFML's published choice of 15 proxies per
class on CUB/Cars and two on SOP. That is a recipe difference, not a measured
causal quantity, and it changes with class count, memory, per-class sample count,
and multimodality simultaneously.

At the mechanism level, CHIMERA composes two families already rejected in this
repository: (i) factor every class proxy through a shared nonnegative dictionary
(candidate 264 and CoVeR/375), and (ii) synthesize virtual identities by
recombining training-class representations (Composite Class Expansion/367 and
RECOMB/370). The inherited-mass calibration is a new wrapper around the latter,
but it does not supply an observed image with the synthetic identity and does
not repair the false rank or sharing claims.

The written objective also contains executable contradictions: balanced
marginals do not forbid class-private atoms; the rank of the actual class
prototype matrix still cannot exceed `C-1`; the Sinkhorn operator is
underdefined; `U` is assigned both SGD and AdamW; and its random-orthonormal
control is impossible when `K=1024>d=512`.

## Gate 0 and Gate 1: no eligible motivating measurement

The proposal does not depend on a new repository artifact, so there is no
candidate-specific Gate-0 number to validate. Its causal premise is inferred
from three facts:

1. a class-mean span from `C` training classes has centered rank at most `C-1`;
2. neural collapse can contract training examples toward class means; and
3. PFML publishes 15 proxies per class for CUB/Cars and two for SOP.

The first is a dimensional identity, not evidence that the trained 512-D image
cloud or its errors are confined to that span. The second is a limiting training
phenomenon; the proposal supplies no saved spectrum showing it in the relevant
fine-tuned checkpoints and no intervention linking it to corrected unseen-class
R@1. The third is a selected hyperparameter. `15*C > 512` does not show that
PFML restored rank, and `2*C > 512` on SOP does not show that rank was the reason
for choosing two. The project's earlier rank audit already recorded this exact
distinction: a dimensional ceiling on a proxy or class-mean matrix is not a
causal measurement of representation starvation.

The nearest reliable repository evidence is therefore absence, while the
mechanism has two exact internal predecessors:

- candidate 264 rejected sparse nonnegative shared-atom class proxies as
  dictionary/low-rank proxy regularization whose bank disappears at retrieval;
- CoVeR/375 attempted to make a shared vocabulary operational in the deployed
  descriptor, yet still failed provenance and admitted class-template
  absorption. CHIMERA returns the atoms to train-only proxy construction, so it
  does not retain even CoVeR's proposed test-time escape.

Gate 1 alone ends the candidate under `docs/search_protocol.md`.

## Gate 2 algebra: the named shortcuts remain

### Uniform marginals do not imply semantic sharing

CHIMERA constrains every row of `Gamma` to sum to `1/K` and every column to
`1/C`, then asserts that a class-private atom is mathematically inadmissible.
This is false. A transportation-polytope row can place its entire `1/K` mass in
one class while that class receives the remaining `1/C-1/K` from other rows.
For `K=256,C=100`, a sparse extreme point can allocate roughly 2.56 atoms per
class with only boundary rows shared. Uniform aggregate usage says every atom is
used; it does not say an atom is used by multiple classes, much less that it is a
reusable visual factor.

The image-code entropy floor does not establish sharing either. It requires
per-image perplexity of at least 32 only when its small weighted hinge is worth
paying. A solution can carry identity in a private or arbitrary group code and
fill the remaining entropy with common atoms. On In-Shop/SOP, where `K<C`, a
balanced atom can merely partition unrelated identities into arbitrary groups.
Neither constraint connects an atom to a stable cross-class semantic.

The claimed infinite KL cost is not present in the written loss. `Gamma` is
projected, not charged an OT/KL penalty in the total objective. Moreover,
`Sinkhorn_epsilon(Gamma)` is not fully defined: Sinkhorn scaling of a positive
matrix to fixed marginals needs no separate epsilon, while entropic OT needs a
cost or logits from which its kernel is constructed. These alternatives produce
different updates and gradients. A substantive choice would be a repaired
proposal.

### The class-indexed table and prototype rank were misdescribed

`Gamma` is a `K x C` identity-indexed state table updated from labelled samples.
Calling it a non-gradient EMA buffer does not make it cease to be class-private
training memory. The training prototype bank is exactly

```
P = normalize_columns(U Gamma).
```

Ignoring column normalization, its centered rank is at most
`min(d,K,C-1)`, not `min(K,512)` independent of `C`. The cone of every
hypothetical `U gamma` may have larger span, but training provides only `C`
empirical code columns. The mask-crossed `q` vectors can add algebraic
constraints, but no training image is labelled with a chimeric identity and no
test identity is represented by averaging a labelled class code. The asserted
train/test operator match is therefore false: training uses a labelled
class-mean lookup `Gamma[:,c]`; deployment returns each single-image `f(x)` and
never computes that class statistic.

The neural-collapse argument also overreaches. Rank at most `C-1` for centered
class means does not imply that the whole feature cloud has that rank unless
within-class variability has collapsed exactly. Even then it describes the
seen training cloud, not a theorem that a shared dictionary recovers useful
unseen-class directions.

### Crossover is synthetic-class supervision, not evidence of a real identity

The core constructs `q` by coordinate-wise crossover of two class codes. It
then asks `q` to interpolate the two parent class means according to inherited
mass and to beat a third class. This is synthetic proxy/virtual-class geometry.
There is no corresponding observation whose identity is `q`; the phrase
"chimera is a real identity" is contradicted by the construction.

Proxy Synthesis already generates synthetic embeddings and proxies as virtual
classes to mimic unseen identities. Metrix and related work occupy mixed
embedding supervision. Repository candidates 367 and 370 independently added
hard composite identities from different parents and were rejected because
the mask, generator, and consistency details changed the estimator rather than
the supervision primitive. Binary atom masks and the fixed calibration slope
do not create a new source of supervision.

The duplicate-atom proof is also unsupported. If two columns of `U` coincide,
the calibration target can change under a mask while `q` does not distinguish
their individual contributions, but the trainable parent means can change their
projections onto `q`. The proposal gives no expectation calculation proving
that all duplicated-atom solutions have positive irreducible loss. A fixed
`kappa` blocks the scalar `kappa -> 0` shortcut, not all joint adaptations of
`U`, the encoder, and the class table.

## Gate 2 prior-art and recurrence map

The literal conjunction may be uncommon; the protocol judges the supervision
mechanism rather than conjunction novelty.

- Gu, Ko, and Kim, *Proxy Synthesis: Learning with Synthetic Classes for Deep
  Metric Learning*, AAAI 2021, generates synthetic embeddings and synthetic
  proxies that act as virtual classes for unseen-class generalization:
  <https://arxiv.org/abs/2103.15454>.
- Zheng et al., *Deep Compositional Metric Learning*, CVPR 2021, already trains
  multiple sub-embeddings through learned composites to preserve generalizable
  diversity; it is adjacent to the claimed compositional-rank action:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Deep_Compositional_Metric_Learning_CVPR_2021_paper.html>.
- Yang et al., *Hierarchical Proxy-based Loss for Deep Metric Learning*, WACV
  2022, explicitly learns class-shared characteristics through structured
  proxies:
  <https://arxiv.org/abs/2103.13538>.
- Nauta et al., *PIP-Net*, CVPR 2023, learns class-shareable visual prototypes
  with sparse nonnegative class connections:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.html>.
- Kundu et al., *Subsidiary Prototype-space Alignment*, NeurIPS 2022, learns a
  shared word-prototype vocabulary, soft-quantizes images, and uses the shared
  primitives to support unknown classes:
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/bf121b033db3bac31c3193e8a0dcbf66-Paper-Conference.pdf>.
- SwAV supplies balanced Sinkhorn prototype assignments, but balanced usage is
  not a proof of semantic factor reuse:
  <https://arxiv.org/abs/2006.09882>.

CHIMERA's inherited-mass calibration is narrower than any single neighbour in
this list. It does not survive as a novel method because its load-bearing
actions are the conjunction of the occupied dictionary-proxy and
synthetic-class families, and its claimed distinctions rely on false algebra.

## Executability, protocol, and forecast audit

1. The object table says `U` is learned by SGD, while the schedule assigns `U`
   an AdamW learning rate. These are different optimizers, not notation.
2. `U` initialization is omitted even though the frozen-encoder initialization
   of `Gamma` depends on it.
3. Control C8 requests `K` orthonormal columns in 512 dimensions. This is
   impossible for the frozen `K=1024` SOP/In-Shop setting.
4. A `45 x 4` sampler is arithmetically compatible with batch 180, but SOP and
   In-Shop contain classes with fewer than four images. Sampling with
   replacement introduces duplicates; filtering changes the identity
   population. Neither behavior is specified or matched in the controls.
5. A full `1024 x 11,318` FP32 table is about 46 MB before temporary buffers.
   Rebalancing all 11.6 million entries three times every optimization step is
   not justified by the stated "sub-millisecond" assertion and is not included
   in the claimed memory multiplier.
6. The CUB/Cars forecasts clear PFML's matched 512-D lane in point estimate but
   do not clear the standing capacity-unrestricted observations (0.766 CUB and
   0.949 Cars). In-Shop 0.932 clears neither VAPNet 0.939 nor CRT 0.9448. The
   forecast standard deviations are priors without a repository estimator.
7. The protocol requires a paired corrected In-Shop screen first. The frozen
   method instead makes its decisive mechanism thresholds CUB-based and treats
   an unpaired local PFML reconstruction as a condition for its headline.
8. F4 computes test-class-mean covariance using test labels. It can be a locked
   post-hoc diagnostic, but using it for tuning or model selection would violate
   the test-data prohibition. Its effective-rank increase would still not prove
   that useful unseen identity information, rather than arbitrary variance,
   was added.

## What is worth retaining

The proposal usefully makes three usually hidden issues explicit: class-indexed
proxy lookup differs from deployed image encoding; balanced use is weaker than
semantic sharing; and a proposed factor vocabulary must be attacked by a
class-private matched-anchor control and a convex-synthesis control. Those are
good audit questions. They do not validate CHIMERA's answer.

Process lesson: a hyperparameter pattern in an external method is not repository
provenance, a balanced transport marginal is not a factor-sharing theorem, and
combining two occupied mechanisms does not become novel merely because their
exact conjunction is absent from a title search.

## Independent-review reconciliation (written after the frozen review)

The cold review is `docs/opus_chimera_review_2026-08-06.md`, produced by durable
consultation `de2e47fdb43c48a3`. Fable exhausted its allocation, so the same
durable job used its configured Claude Opus fallback. The frozen review returns
**DEAD** and independently agrees on the missing provenance, false rank story,
underdefined Sinkhorn operator, optimizer/bootstrap contradictions, invalid C8,
sampler problem, and synthetic-class/dictionary adjacency.

The review adds one genuinely stronger mechanism failure. If two parent classes
assign equal positive mass `g` to an atom, flipping only that atom's crossover
bit leaves the unnormalized chimera, its normalization, and therefore `q`
unchanged, while changing `beta` by `g/Z`. The two masks demand different
calibration targets for the same similarity difference. Their minimum paired
residual is `2*kappa^2*g^2/Z^2 > 0`. Thus the calibration term applies an
anti-sharing pressure exactly where equal cross-class atom reuse occurs. The
proposal's D3 duplicate-coordinate argument and its claimed semantic-sharing
mechanism use the same algebra with opposite interpretations. This is accepted
and strengthens the Gate-2 rejection.

The review also finds a severe EMA scaling defect that the local audit missed.
After balancing, a historical column has mass `1/C`, while the incoming batch
mean has mass one. Blending them before rebalancing gives effective historical
weight

```
(m/C) / (m/C + 1-m),
```

not `m`. At SOP's first-update `m=0.5`, history is outweighed 11,318 to one.
Even after `m` grows, the intended count-debiased EMA is radically shortened.
The proposed mitigation for rare-class staleness therefore destroys itself.
This is accepted as an independent executable failure.

### Corrections to the cold review

The cold result is evidence only where its calculations survive checking. Four
material statements need correction:

1. Its displayed CUB common-plus-private matrix is **not** a Sinkhorn fixed
   point. For classes with two versus three private atoms, the row contribution
   `0.5/|S_c| + 100*0.5/256` differs; `0.5/|S_c|` is not the stated `0.195` in
   either case. The entropy line also writes `ln(510)` although only 255 tail
   coordinates exist (its numerical value happens to match the 255-coordinate
   calculation). Therefore the review does not prove that *all* 256 rows are
   class-private for the actual noninteger ratio `K/C=2.56`, nor that the
   displayed object equals C3.
2. The correct counterexample is weaker but sufficient. A greedy exact
   transportation vertex with CUB marginals `row=1/256`, `column=1/100` has 352
   nonzeros: **160 of 256 atom rows have support in exactly one class**, and the
   other 96 touch only two classes. Hence a class-private atom is plainly not
   “mathematically inadmissible.” Mixing that coupling `T` with the uniform
   coupling as `Gamma=lambda*T+(1-lambda)/(KC)` preserves both marginals. At
   `lambda=0.6`, every normalized class code has entropy between `3.504` and
   `3.589`, above `ln(32)=3.466`, while 60 percent of its discriminative mass
   remains in the sparse near-private transport component. Balance plus entropy
   therefore fails to identify semantic reuse without relying on the review's
   incorrect fixed point.
3. The paired-mask calculation proves anti-sharing pressure for equal shared
   mass. It does **not** prove that disjoint supports are the only global
   minimizers for unequal masses, finite sampling of 64 masks, normalized `q`,
   and the other three loss terms. The justified conclusion is conflict with
   the claimed sharing mechanism, not a proved C3 equilibrium or a prediction
   that F2/F3/F6 must fire.
4. The review's cost check is more persuasive than the local audit's skepticism:
   11.6 million entries are about 46 decimal MB, and three fused scaling passes
   can plausibly remain below a millisecond on a DGX GPU. The exact wall time is
   implementation-dependent, but cost is **not** a reason to reject CHIMERA.
   The memory still has to be reported, but the earlier implication that the
   sub-millisecond claim was likely false is withdrawn.

Two additional reviewer endorsements are not retained. `tau` and `k*` remain
two operational choices: temperature shapes the whole assignment distribution,
while `k*` independently sets the hinge threshold; the fact that entropy is a
function of temperature does not collapse them into one hyperparameter. Also,
the chimeric term can reasonably be called a new derived constraint even though
it contains no new label bits. What it cannot be called is an observed “real
identity,” and its supervision primitive remains mechanism-equivalent to
synthetic/virtual-class regularization for this protocol.

Finally, the review labels executable mathematics “Gate 1” according to its own
numbered rubric. Under `docs/search_protocol.md`, the authoritative earliest
failure is **Gate 1: no repository provenance**; the algebra and prior-art
failures are Gate 2. With those corrections, the two independent analyses still
support the same disposition: CHIMERA is dead before preregistration or GPU.
