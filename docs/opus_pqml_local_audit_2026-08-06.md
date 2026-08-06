# Local Gate 1/2 audit: Phase-Quotient Metric Learning (blind pass 47)

**Verdict: DEAD at Gates 1 and 2. No diagnostic, preregistration,
implementation, or candidate GPU run is authorized.** The fixed complex-modulus
map is mathematically interesting, but the repository measures against its
shared-nuisance premise, the proposed losses do not identify a nuisance or a
shared group action, and the executable definitions contradict several of the
claimed exact identities. Public work already occupies the supervision object
and action: learning class-exogenous intra-class characteristics and removing
them from the deployed task representation.

## Frozen provenance

The proposal is frozen verbatim in
`docs/opus_pqml_proposal_pass47_2026-08-06.md` at commit `4fc697c`. Its file
SHA-256 is
`e2f5a28167b0c2472d7dc1e6dbbad5d2b5b852a4902deba810458d7e3210372d`;
the provider text before the single terminal newline hashes to
`c2b48d65f1f28132579e2ce38e08ebaea881b24b83dc7c935e9fc7ecbe76dd4c`.
The durable proposal job was `5c4b15d4504a4b25`. The worker read only the
frozen blind prompt and verified its SHA-256 before using public Web search and
fetch; it reported no other repository state, delegation, or consultation.
That bootstrap verifies the assignment and does not supply the evidence below.

The frozen object is **Phase-Quotient Metric Learning (PQML)**. An internal
544-D unit vector contains 32 designated complex coordinate planes. Deployment
replaces each complex pair by its modulus and retains 480 ordinary coordinates,
yielding 512 values. Proxy Anchor or PFML acts on that deployed vector. A
same-class hinge requires the quotient to destroy at least `0.15` squared
distance, while first- and second-order circular-moment hinges try to make phase
class-independent. The claim is that phase therefore becomes a transferable
pose/viewpoint nuisance and modulus preserves identity.

## Gate 1: the measured premise is absent and the nearest prospective test is adverse

No artifact in the verified packet shows that official-query failures are caused
by a circular nuisance, that viewpoint or pose is represented as a common
two-plane rotation, or that class-independent phase can be identified from
identity labels. The proposal's `S1` causal story, `L=32`, `0.15` target,
dataset ordering, and R@1 conversion are forecasts rather than repository
measurements. Horizontal flip and scale in its future P2 are not azimuth labels
and cannot retrospectively supply provenance.

The closest prospective repository measurement points the other way. Candidate
225 learned leading within-class subspaces on one set of identities and tested
whether they transferred as nuisance-heavy, identity-light directions to
disjoint identities. Its fold-averaged `rho_32` values were **0.9312, 0.9287,
and 0.9345** for seeds 0--2, all below the locked `1.15` falsifier and below
one: source within-class directions captured proportionally at least as much
target between-identity signal as target within-identity signal. That does not
prove every nonlinear invariant harmful, but it directly denies positive
evidence for PQML's shared class-exogenous nuisance frame.

This is also a recurrence, not an isolated missing measurement. Candidate 176
already considered a cross-instance nuisance-tangent quotient; candidate 369
Exchangeable-Nuisance Embedding and blind passes 24 (EFML), 25 (CFR), 27
(FRAME), and 40 (CNW) all required class-independent intra-class structure to
transfer across identities. They failed for adverse provenance, missing
correspondence, non-identification, or occupied supervision. A torus coordinate
system does not reverse those measurements.

## Gate 2 algebra: the objective does not identify pose, nuisance, or a shared action

### Exact cheap solution: manufacture disposable random phase

The decisive counterexample is class-independent within-class phase variation.
Let the encoder place any deterministic per-image code in the 32 phases—for
example background, crop detail, instance hash, or arbitrary image texture—so
its class-conditional first two moments match the pooled moments. Give different
same-class images enough angular separation to reach `Delta >= 0.15`, and put
all identity information used by `L_base` in the deployed moduli and remainder.
Then:

1. `L_inv = 0` because same-class phase differs;
2. `L_prot = 0` whenever the first two phase moments are class-independent up to
   its permissive hinge; and
3. `L_base` cannot distinguish this code because phase is discarded before the
   task loss.

This solution contains no viewpoint, no transformation correspondence between
images, and no common physical group action. It simply manufactures noise that
the quotient later removes while preserving baseline retrieval behavior. The
proposal's D5 assertion that the common solution is *therefore* a nuisance
factor is false: class independence plus within-class variation is not
identification. Identity labels provide no observation linking the phase angle
of one image to the phase angle of another identity.

The impossibility is not merely semantic. ZIN proves in a broader invariant-
learning setting that invariant features cannot in general be identified
without environment partitions, additional information, or explicit inductive
bias. PQML supplies a coordinate topology but no observation identifying which
image factor should occupy it. Its premise that a circular channel will become
azimuth rather than a cheaper arbitrary code is unsupported.

### The advertised anti-degeneracy arguments fail

- **Collapse is first-order stationary.** The proposal itself admits that
  `Delta` is quadratic in angular separation. At equal phases, including exact
  collapse, the angular derivative is zero. A positive hinge value does not
  imply an escape gradient. Random initialization is not a proof against later
  convergence to the stationary solution. The epoch-40 `L_disp` contingency is
  an unfrozen replacement, not part of PQML.
- **Two circular moments do not establish independence.** Class information can
  occupy orders `k >= 3` or arbitrary multimodal phase laws while the first two
  moments agree. The proposed adversarial classifier is another post hoc method,
  not a control of the frozen object.
- **The protection null is miscalibrated.** For `B` independent unit phasors of
  population moment `mu`, with a class subset of size `n_s`, the exact finite-
  sample expectation is
  `E|m_c-m_B|^2 = (1-|mu|^2)(1/n_s - 1/B)`, not
  `(1-|m_B|^2)/n_s`. The global mean includes the class samples. With
  `n_s=4`, the written threshold is too large and itself random, making the
  hinge easier to satisfy. Even a correctly calibrated two-moment test would
  not identify a nuisance.
- **Energy mis-setting is selective reporting.** D4/F8 says a converged run
  above `0.6` phase energy is discarded rather than counted as method failure.
  That is a post-result exclusion keyed to a mechanism outcome. A frozen method
  must report such a run as a failure.
- **Test-triggered repair is illegal.** F4 and F5 compute contraction transfer
  and phase/class mutual information on unseen test identities, then say a
  failed probe triggers a repair. Post hoc diagnosis is legal; choosing and
  rerunning a method from test identities violates the no-test-selection rule.

## Gate 2 algebra: the written quotient is not the claimed exact quotient

The unsmoothed modulus has the clean torus identities, but the executable uses
`|zeta_l|_eps = sqrt(a_l^2+b_l^2+eps^2)`.

- Because `z` is unit norm, the written deployed vector satisfies
  `||y||^2 = 1 + L eps^2`, not exactly one. The error is tiny, but the exact
  norm-preservation claim and the decision not to renormalize are false.
- With epsilon smoothing,
  `sum_l |zeta_l|_eps |zeta'_l|_eps + <r,r'>` is not
  `max_theta <R_theta z,z'>`. F2 and the equality between quotient distance and
  orbit-minimized distance hold for raw moduli, not the forward pass as written.
- The displayed derivative of `Delta` mixes raw and smoothed moduli. For the
  executable smoothed expression it must use the smoothed partner modulus and
  `zeta_i/|zeta_i|_eps`; for the exact group expression it must use raw moduli
  and is nondifferentiable at zero. The proposal cannot simultaneously claim
  the exact group identity and the displayed globally smoothed gradients.
- F1 also does not hold literally for the smoothed `pi` on the unit sphere in
  the stated codomain with exact norm one. The orbit-equivalence content can be
  repaired by removing epsilon or renormalizing and re-deriving the scorer, but
  either is a substantive change to the frozen executable object.

The circle-versus-line argument in section 2.1 is conditionally correct for a
known plane containing a radial identity variable and angular nuisance. The
loss never observes that precondition. An unconstrained head can instead put
identity in the discarded phase, nuisance in modulus, or arbitrary noise in
phase. A theorem about quotienting a supplied group action is not a theorem that
the proposed supervision discovers that action.

## Gate 2 prior art: the supervision object and action are occupied

The narrow coordinate wrapper may be unusual, but novelty under this protocol
is judged at mechanism level.

- Roth, Brattoli, and Ommer, **MIC: Mining Interclass Characteristics for
  Improved Metric Learning** (ICCV 2019), explicitly learns latent visual
  characteristics such as viewpoint and illumination that are shared across
  classes, separates them from a class encoder through mutual-information
  reduction, and evaluates standard zero-shot DML benchmarks. That is PQML's
  claimed supervision referent and removal action.
- Jaiswal et al., **Unsupervised Adversarial Invariance** (NeurIPS 2018), learns
  a split task/nuisance representation without nuisance labels and deploys the
  nuisance-invariant task component.
- Feige, **Invariant-Equivariant Representation Learning for Multi-Class Data**
  (ICML 2019), splits a class-invariant representation from an equivariant
  within-class transformation representation using class labels.
- Group-invariant instance-retrieval descriptors, scattering/complex modulus,
  Phase Collapse, Augerino, learned canonicalization, and latent-symmetry
  discovery occupy the quotient, modulus, and learned-invariance operators.
  Internally, candidate 176 occupied tangent quotienting and candidate 23 IDNR
  occupied an augmentation-estimated nuisance quotient.

PQML's possible distinction is the fixed learned complex-coordinate torus plus
the particular moment and contraction losses. Those losses do not identify the
claimed characteristic, however, so wrapper novelty cannot rescue the same
occupied target and action. MIC is especially close and benchmark-matched: its
primary abstract says it learns characteristics “shared by and go across object
classes” and explains away structured visual variability rather than treating
it as noise.

Primary sources:

- Roth, Brattoli, and Ommer, *MIC*, ICCV 2019:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html>
- Jaiswal et al., *Unsupervised Adversarial Invariance*, NeurIPS 2018:
  <https://proceedings.neurips.cc/paper_files/paper/2018/hash/03e7ef47cee6fa4ae7567394b99912b7-Abstract.html>
- Feige, *Invariant-Equivariant Representation Learning for Multi-Class Data*,
  ICML 2019: <https://proceedings.mlr.press/v97/feige19a.html>
- Lin et al., *ZIN: When and How to Learn Invariance Without Environment
  Partition?*, NeurIPS 2022:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/9b77f07301b1ef1fe810aae96c12cb7b-Abstract-Conference.html>

## Protocol, frontier, and cost failures

The proposal begins with five-seed CUB and Cars arms instead of the mandatory
same-seed corrected In-Shop screen. Its headline is conditional on an
unimplemented local PFML reproduction under an invented sampler and learning
schedule, while its own source audit says PFML does not disclose enough detail
to establish recipe parity. A frontier cannot be inherited if R0 misses the
published interval, and a later interval check does not make the prior recipe
matched.

It omits the required raw-best versus independently selected/final split and
uses independent-paper SD arithmetic rather than a paired current-digest
comparison. Its PA recipe is not portable across datasets: primary-source
checking elsewhere in this repository found `6e-4` learning rate for SOP and
In-Shop and batch 150 for Cars, whereas the proposal fixes `1e-4` and 120.
The claim that pairwise auxiliary work is `1.6e-5` of forward compute also
converts to 0.0016%, not evidence of total measured training overhead; backward,
pair construction, and memory traffic remain unmeasured.

Even the frozen forecast gives only `0.40` probability of crossing both chosen
PFML bars, and those are not the capacity-unrestricted SOTA horizons. The
project's standing matched-lane rule permits a Lane-A claim, but only after a
faithful matched reproduction, corrected In-Shop screening, out-of-sample
confirmation, raw/final reporting, and second-dataset replication. PQML fails
before all of those at Gates 1 and 2.

## What survives

For raw moduli and nonzero complex coordinates, the torus maximal-invariant and
orbit-maximized-cosine identities are useful mathematical components. The
proposal also names strong controls: quotient-only, linear WCCN, random-pair
routing, generic anti-collapse, and removal of phase protection. None authorizes
this run. A future method would need a new verified measurement and an observed
cross-identity correspondence that identifies a transformation before selecting
a quotient; identity labels and class-independent variation alone are
insufficient.

**Process lesson:** proving that a quotient removes a specified group action is
not proving that weak label-only losses discover that action. Always construct
the cheapest class-independent disposable code, check finite-sample null levels,
and separate diagnostic use of test identities from method-selection triggers
before treating invariance as learned supervision.
