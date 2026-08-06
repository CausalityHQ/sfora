# Pass 34 local evidence-aware audit: GVF

Date: 2026-08-06 UTC  
Frozen proposal: docs/opus_gvf_proposal_pass34_2026-08-06.md  
Independent review: d1422459f0614e0d (running when frozen)

This audit was written without reading reviewer partials or results.

## Verdict

**DEAD at Gate 1 and Gate 2; no diagnostic, preregistration, implementation, or
GPU.** The shared-frame target is prospectively adverse in this repository and
occupied by benchmark-matched prior art. Several proof and recipe defects
reinforce but are not needed for rejection.

## Gate 1 is measured against GVF

GVF assumes a low-rank within-class tangent frame shared across identities and
claims tying proxy displacements to it forces transferable nuisance handling.
Candidate 225 prospectively tested the underlying premise on corrected,
disjoint In-Shop identity folds. Its locked source-to-target top-32
within-class-subspace ratios were 0.9312, 0.9287, and 0.9345, all below the
preregistered 1.15 threshold and below the random-subspace reference. Source
within-class directions captured about 35--37% of target within-class energy
but 38--40% of target between-class energy.

That experiment does not exactly train GVF proxies, but it directly denies the
claim that this repository measures a transferable linear nuisance frame.
Pass 25 CFR, Pass 27 FRAME, Pass 24 EFML, and Pass 18 CITTR already died on
variants of the same premise. GVF supplies no new prospective measurement.
Its F0 is not a free diagnostic: it first requires a full PFML GPU training,
uses CUB despite the protocol's In-Shop-first screen, and measures in-sample
explained variance rather than disjoint-identity transfer. High in-sample E7
would not reverse candidate 225.

The parameter-count argument is not provenance. Fourteen free proxies per
class provide many parameters, but shared class-level vectors cannot thereby
memorize arbitrary 512-D embeddings image by image. Counting 122 scalar
parameters per image neither proves an interpolation construction nor shows
that optimization uses proxies instead of the backbone. Restricting proxy
capacity may simply underfit PFML.

## Gate 2 is occupied at the supervision-object level

Spherical Feature Transform (Zhu, Bai, and Wei, ECCV 2020) already transfers
within-class feature variation between identities by spherical rotation on
CUB, Cars, and SOP. DVML explicitly assumes intra-class variation is
class-independent and represents it with a shared latent distribution.
ESRC/intraclass variant dictionaries import variation learned from other
identities in undersampled face recognition. SoftTriple and structured
multi-proxy methods occupy learned class manifolds; NIR shapes proxy-local
residual distributions.

GVF's learned low-rank parameter, tangent projection, geodesic proxy
generation, and deletion at deployment are a new scaffold around the same
cross-class variation-transfer object. Replacing SFT's nonparametric observed
variation with a learned shared basis, or replacing DVML's latent law with a
deterministic frame, does not change what supervision exists.

Primary sources:

- SFT: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf
- DVML: https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html
- SoftTriple: https://arxiv.org/abs/1909.05235
- NIR: https://arxiv.org/abs/2203.08547

## Mathematical and operational defects

The strict-saddle derivation includes only a signed attractive pair around one
mean. PFML also contains negative sample forces, proxy-proxy attraction and
repulsion, all other generated proxies, tangent gates that depend on class
means, and Locus 2. Therefore the 1/(alpha+2) E7 threshold does not follow for
the full frozen objective, and its proposed diagnostic is not calibrated.

The asserted stationary frame as top eigenvectors of within-minus-between
scatter is likewise a second-order local analogy, not a derivation through the
complete nonlinear PFML energy, geodesic projection, QR Jacobian, learned
means, and amplitude gates. The same trainable means can rotate relative to
the global frame, letting auxiliary geometry change without identified
backbone nuisance suppression.

Locus 2 is adjacent to proxy jitter/label smoothing. Its stopped anchor does
not force the other samples to learn a transferable factor; Omega and tau can
adapt virtual proxies to the current training classes while the anchor has no
corrective gradient. The decisive per-class-frame and random-frame controls
would diagnose this only after spending GPU on a Gate-1/2 failure.

The SOP construction is not executable as written. The universal formula gives
p0 plus a plus/minus pair for K=1, hence three proxies, while PFML SOP uses two.
Calling it unsigned K=1 silently changes the formula, symmetry, cancellation
proof, and controls.

Annealing beta only to 0.25 does not close the train/deploy metric gap; every
generated proxy remains displaced at the end and all are deleted at test.
The claim that tied displacements keep nearby proxy sets parallel is also
false after class-specific tangent projection: the same ambient vector
projects differently at different poles.

## Protocol and frontier

GVF explicitly adopts best-epoch-on-test selection despite this repository's
measured 0.35--0.84 point winners-curse bonuses and arm-dependent differences
large enough to reverse rankings. Equal evaluation cadence does not remove
selection bias. The protocol requires raw and selection-corrected values and
an In-Shop-first screen; GVF instead proposes a CUB PFML baseline diagnostic
and forecasts no In-Shop crossing.

Its CUB/Cars crossings rely on an in-house PFML reproduction forecast 0.8/0.7
points below the published base while comparing GVF against the published
reference. Until the base reproduces, this is not a matched frontier arm. The
claimed probability of reaching either crossing is only 0.40 by the proposal's
own forecast.

## Mechanism lesson

Parameter sharing is not evidence that the shared parameter represents
nuisance, and an in-sample shared-subspace fit is not evidence of transfer.
The project has now repeatedly measured against and rejected a global
cross-identity nuisance frame. Future proposals in this family require a new
prospective disjoint-identity intervention, not another manifold, frame, or
proxy parameterization.
