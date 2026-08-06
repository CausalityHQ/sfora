# Pass 53 local evidence-aware audit: CoDiF

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_codif_proposal_pass53_2026-08-06.md`  
Blind-proposal job: `e368f4caf5454264` (Fable credit failure, Opus fallback;
process exit 1 after emitting a complete terminal partial)  
Frozen recovered artifact SHA-256:
`be2183383bc3d208d3a265552e88a457ebf5eac8352a7719b5ef2d4320038951`

This audit was written before launching or reading the mandatory independent
review. The durable terminal partial was frozen rather than rerunning the blind
proposal. It audits Correlation-Dimension Flattening (CoDiF) exactly as frozen.

## Verdict

**DEAD at Gate 1 and independently non-executable as frozen; Gate 2 is occupied
at the supervision-object/action level. No preregistration, implementation, or
candidate GPU.**

CoDiF's motivating “scale gap” has not been measured in a corrected repository
checkpoint or associated with official-query errors. Its executable loss is
singular at the exact collapse it claims to attack and is driven toward that
singularity by its own positive-pair term. Its pooled distance-CDF constraints
can be satisfied by class-mean geometry, private training-class/image codes,
or view-shared nuisance without preserving unseen-identity semantics. Local
intrinsic-dimensionality regularization and benchmark-matched multilevel
distance regularization already occupy the target and action; curvature of a
particular smoothed CDF is an estimator wrapper, not new supervision.

The proposal also predicts **no standalone frontier crossing on any dataset**.
Its only crossing is an explicitly unclaimed, unimplemented additivity guess on
PFML. Even a fully successful frozen standalone arm would not meet the standing
objective.

## Gate 1: no measured scale-gap cause

CoDiF requires all of the following before intervention:

1. a corrected baseline has a plateau between within- and between-training-
   identity distance scales;
2. that plateau grows during training rather than merely reflecting successful
   supervised separation;
3. unseen-identity distinctions occupy the same scale as training within-class
   variation;
4. reducing log-correlation-integral curvature improves official-query errors
   at fixed ordinary compactness, exposure, and compute; and
5. the preserved small-scale variation is identity-semantic rather than pose,
   crop, background, acquisition, or image-instance information.

None has been measured in this repository. The proposed `a/b`, NC1 scatter,
correlation-dimension, curvature, and held-out-class probe are future
diagnostics. Neural-collapse literature and the benchmark's class-disjoint
split do not establish that unseen classes are separated at the training
within-class median scale.

The closest prospective transfer evidence is adverse. Candidate 225 learned
within-class directions on source identities and obtained corrected disjoint-
identity `rho_32` values **0.9312, 0.9287, and 0.9345**, all below its locked
1.15 threshold and below one. ARCG's corrected augmentation-response graph was
also image-specific on In-Shop, retaining **0.3631--0.3640** of same-class
pairs. Those measurements do not prove every nonparametric distance penalty
fails, but they deny positive provenance for CoDiF's claim that preserved
small-scale training variation transfers as unseen identity structure.

F3 is a sensible train-identity-only premise test, but it is proposed after the
method and requires a new GPU training programme. Under Gate 1 it cannot be used
to authorize the candidate whose hypothesis generated it.

## The frozen objective is singular at its named failure state

The grid begins at the batch median two-view distance `epsilon` and defines

```
rho = (theta_max / epsilon) ** (1/M)
theta_m = epsilon * rho**m
```

with sigmoid bandwidth `beta*theta_m` and curvature divided by
`(log rho)^2`. At exact view collapse, `epsilon=0`: the ratio, `rho`, every
quantity involving `theta_0`, and the sigmoid denominator are undefined. When
`theta_max=epsilon`, `log rho=0` and the curvature denominator is undefined.
No clamp, skip rule, or limiting expression is specified.

This is not a remote boundary. `L_epsilon` explicitly minimizes the mean
squared two-view distance and therefore drives the lower grid anchor toward
zero. Near zero, the live distance derivatives include factors
`1/(beta*theta_m)`, while the stop-gradient grid relocates between steps. The
claim that CoDiF is “maximally violated” at exact neural collapse is therefore
about a non-executable limit, not its frozen loss.

The statement “zero iff `C(theta)` is proportional to `theta^D` over the
interval” is also too strong. Zero discrete second differences say only that
the nine sampled `g_m` values lie on a line; the smoothed CDF can curve between
grid points. The claimed equivalence to memorylessness of log distance requires
additional support/truncation assumptions and is not derived.

## Cheap semantic escapes remain

The global curvature term pools all `B(B-1)` ordered pairs. With the proposed
30-by-6 sampler, only about 2.8% are same-class. A network can shape the
**training class means** so their inter-class distances fill the log grid while
every class remains internally collapsed. That changes a label-code geometry,
not unseen-class supervision.

The alleged intra-class closure does not prevent heterogeneity: it pools all
900 same-class pairs into one CDF rather than constraining each class. Classes
can receive private radii or private low-dimensional curves whose mixture is a
power law. Within a class, training-image lookup, stable background, pose, and
other view-shared information can fill the distance scales at little or no
two-view alignment cost. A finite proxy loss does not identify an interior
solution as semantic; it only trades class separation against the junk code.

Consequently the decisive failure is stronger than D6's acknowledged “partial
junk” risk. The loss supplies no positive relation saying which small-scale
pairs or directions should agree across unseen identities. It shapes a
marginal distance histogram while leaving the supervision referent unchanged.

`L_dim` does not repair identification. Its endpoint slope is called a
correlation dimension even when the curve is not flat, and its target is
estimated from unnormalized 2048-D frozen GAP features while the trained object
is a normalized 512-D descriptor. The proposal does not specify identical
sampling, normalization, grid support, or a finite-sample calibration that
makes those values comparable. Preserving a scalar dimension from ImageNet does
not preserve identity-relevant directions.

## The second-moment lemma overclaims

For an exact simplex of `C` collapsed class atoms, the centered second moment
has rank at most `C-1` and equal nonzero eigenvalues under ideal balancing. That
narrow algebra is correct. It does **not** follow that every second-moment or
spectral regularizer is optimized there:

- VICReg's variance floor is violated in the `512-(C-1)` zero-variance
  directions;
- covariance decorrelation depends on coordinate orientation and target
  variances, not only the nonzero eigenvalue multiplicities;
- an intrinsic-dimension or singular-value floor may explicitly demand rank
  above `C-1`; and
- coding-rate objectives depend on their noise scale and normalization.

The proposal therefore does not establish its claimed categorical separation
from the occupied anti-collapse family. More fundamentally, the geometry of
training class atoms says nothing by itself about how the same encoder maps an
unseen class.

## The invented carrier is not a matched baseline

The smooth class score is

```
s_c = logsumexp(gamma * cosine_k) / gamma.
```

It omits the usual `-log(K)/gamma` normalization. Thus duplicating equal
proxies raises every class score by `log(K)/gamma`; with `K=15` the offset is
about **0.271**, and `s_c` can exceed one even though every cosine is at most
one. The absolute Proxy-Anchor margins therefore acquire a large proxy-count-
dependent bias, with opposite effects in the positive and negative terms.
Using `K=15` on CUB/Cars and `K=2` on SOP/In-Shop changes the base loss itself,
not merely its capacity.

This MPA carrier is neither the corrected repository Proxy Anchor recipe nor a
verified PFML reproduction. The proposed GAP-only, 200-epoch, 30-by-6,
two-view recipe also changes pooling, sampler, exposure, schedule, and proxy
mechanics. A paired C1 controls the CoDiF delta within that invented carrier,
but cannot transfer its absolute numbers or claimed additivity to PFML.

## Gate 2: occupied target and action

Two primary neighbours are load-bearing:

- Huang et al., **LDReg: Local Dimensionality Regularized Self-Supervised
  Learning** (ICLR 2024), explicitly derives and optimizes local distance
  distributions using a local-intrinsic-dimensionality target to prevent local
  dimensional collapse. This directly occupies differentiable distance-law
  and intrinsic-dimension supervision, including the claim that global spectra
  can miss local collapse.
- Kim and Park, **Multi-level Distance Regularization for Deep Metric Learning**
  (AAAI 2021), is benchmark-matched DML and directly regularizes pairwise
  embedding distances into multiple levels alongside an existing DML loss to
  improve unseen-class retrieval.

IDRR, VICReg, HORDE, uniformity, `rho` spectral regularization, NIR, and coding-
rate methods occupy surrounding global/local dimension and distribution-shape
actions. CoDiF replaces LDReg's local distance-law target and MDR's explicit
levels with the squared curvature of one smoothed pooled empirical CDF. It does
not add a new label relation, correspondence, transformation, or supervision
source. Its literal curvature formula may be uncommon, but after its causal
distinction and anti-degeneracy claims fail, that is an estimator-level wrapper
around occupied multiscale distance/dimension regularization.

Primary sources:

- Huang et al., *LDReg*, ICLR 2024:
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/496d8e7c79c39e284d3b461d3fed13d7-Abstract-Conference.html>
- Kim and Park, *Multi-level Distance Regularization for Deep Metric Learning*,
  AAAI 2021: <https://ojs.aaai.org/index.php/AAAI/article/view/16277>
- Jacob et al., *HORDE*, ICCV 2019:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.html>
- Roth et al., *Non-Isotropy Regularization for Proxy-Based Deep Metric
  Learning*, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html>

## Controls, cost, and standing objective

- C8's grid permutation is not magnitude- or gradient-matched. Permuting the
  `g_m` sequence changes every second difference, the zero set, and generally
  the loss scale; it is not a pure gradient-noise placebo.
- C3 is selected after observing CoDiF's final `a/b`, so it needs a nested
  train-identity-only selection protocol. Matching one output statistic does
  not match gradients or training trajectory.
- Five seeds for C0--C8 plus a six-point lambda sweep is a large programme, not
  an In-Shop-first one-seed screen. Raw-best versus independently selected/final
  reporting and a fresh out-of-sample stage are absent.
- The stated regularizer cost omits construction/backpropagation of a new
  `B x B x 512`-equivalent cross-view distance operation; it need not be large
  relative to ResNet-50, but it is not merely twelve sigmoid passes over a
  matrix the MPA carrier already computes.
- Every standalone forecast is below the proposal's own frontier: CUB 0.727
  versus 0.734, Cars 0.913 versus 0.927, SOP 0.817 versus 0.829, and In-Shop
  0.924 versus 0.930. The PFML composition is explicitly “conditional” and
  “not asserted,” with an invented 0.4 attenuation. It cannot satisfy the
  standing objective or serve as a preregistered candidate result.

## Gate decision and reusable lesson

CoDiF stops at Gate 1. The repository has no measured scale-gap cause, and the
closest transfer measurements are adverse. It is independently non-executable
at its named collapse boundary, has cheap class/private-code solutions, and
reduces to occupied local-dimension and multilevel-distance regularization.

Reusable lesson: **a smooth marginal distance spectrum is not supervision.**
Before optimizing a global or pooled geometric statistic, enumerate which
training-label lookup, class-mean arrangement, image-instance code, and stable
nuisance distributions share its zero set. A scale grid must also remain
defined at the very collapse boundary it claims to repel.

## Cold-review reconciliation (Pass 53)

The independent frozen-proposal review agrees with the Gate 1/2 death and
strengthens it; it does not rescue any part of CoDiF. In particular, the
review's exact scale argument shows that the proposed same-class flatness term
is invariant under a -> lambda a, while the pooled term can shrink as the
within-class scale shrinks. This is a direct counterexample to the claimed
causal mechanism, not merely an empirical concern. It also confirms the
undefined epsilon=0/rho=1 boundaries, the unnormalised logsumexp proxy
carrier's log(K)/gamma shift, and the class-mean/private-code zero set.

The review independently identifies LDReg (ICLR 2024) and Multi-level
Distance Regularization (AAAI 2021) as mechanism-level prior art, so the
curvature parameterisation cannot support a Gate 2 distinction after the
causal claim fails. It further catches that the proposed variance/coding-rate
lemma is not a valid consequence of the ETF identity, the grid permutation is
not a matched placebo, and the stated standalone forecasts do not cross the
audited frontier. These findings are accepted. One reviewer illustration
uses a plausible operating point rather than a persisted checkpoint; that
does not affect the exact invariance, boundary, prior-art, or forecast
decisions. CoDiF remains DEAD at Gates 1 and 2 and receives no GPU run.

Independent review artifact:
docs/opus_codif_review_2026-08-06.md.
