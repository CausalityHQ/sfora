# Pass 56 local evidence-aware audit: IFR

Date: 2026-08-06 UTC  
Recovered proposal: `docs/opus_ifr_proposal_pass56_2026-08-06.md`  
Recovered-file SHA-256: `6431dc6b276748e651544c3be8dcc3c64c46491d892fe95b0b3c7f5740dba091`

## Provisional verdict

**UNRESOLVED/NO GPU AUTHORIZATION pending exact-proposal recovery and cold
review.** The mechanism is materially different from the exhausted tail and
EMA families, but Gate 1 is not yet passed and the shell transcript omitted a
middle segment of the provider answer. A recovered summary cannot silently be
treated as the exact frozen proposal.

## Gate 1 — provenance

The proposer’s causal premise is class-gated nuisance cancellation: seen-class
proxy training may learn identity-specific invariance circuits that do not
transfer to unseen identities. The only cited positive evidence is Zhou et al.
(arXiv:2203.09739/ICLR 2022), an external study, not a repository measurement.
The verified repository packet contains no corrected artifact measuring
class-conditional scatter-operator heterogeneity, held-out-class invariance,
or a relation between such heterogeneity and official-query R@1. Thus Gate 1
is not passed. The proposed held-out-training-class probe is future evidence,
not provenance for the frozen method.

## Gate 2 — prior-art attack

The leave-one-class-out prediction of a per-class displacement covariance is a
plausibly distinct supervision object, but the recovered answer’s novelty
search must be checked against primary sources before GPU work. Closest
mechanism families are:

- class-conditional invariance and episodic/meta-learning, including Zhou et
  al.’s class-transfer invariance study;
- covariance-preserving augmentation, IAA/ISDA and feature-distribution
  augmentation, which estimate per-class covariance and use it for training;
- WCCN and within-class covariance analysis, which use a global covariance;
- Anti-Collapse/coding-rate, VICReg and related variance/covariance penalties;
- augmentation-response/self-supervised consistency methods; and
- domain-generalization methods that match leave-one-domain-out statistics.

The mechanism-level distinction to verify is strict: IFR predicts the *shape*
of a held-out class’s within-class displacement second moment from other class
operators, with a detached per-class scale, and uses the residual as a
train-time penalty. A method that merely matches global covariance, samples
synthetic features, or adds two-view consistency occupies a different object;
if a primary source already performs this leave-one-class-out covariance
prediction, IFR is dead at Gate 2.

## Algebra and protocol risks to audit

1. The denominator and per-class scale are stop-gradient, so the displayed
   scalar is a custom update rather than the gradient of an ordinary normalized
   objective. Verify the claimed absence of common-mode amplification under
   per-step recomputation.
2. `SR_c` has rank at most 3 and `SF_c` rank at most 5 in a 512-D space. The
   Frobenius residual may be dominated by sampling noise and augmentation
   artifacts, especially on SOP; effective sample size must be measured.
3. The consistency term is an occupied mechanism and must be separated from
   IFR. C2, C6, C7, C8, C10, and C11 are necessary; a single headline gain is
   not sufficient.
4. The recovered recipe completes many undisclosed PFML settings and forecasts
   below or only marginally above the audited frontier. It cannot inherit PFML
   numbers without a matched reproduction.
5. Corrected In-Shop must be the first GPU screen after Gates 1–3, with paired
   same-digest control, raw-best and independently selected/final metrics,
   out-of-sample confirmation, and second-dataset replication.

No implementation or GPU queue is authorized until the exact-answer limitation,
Gate 1 measurement, and cold prior-art review are resolved.

## Cold-review reconciliation

The independent reviewer returns **DEAD**. In addition to the recovered-stream
specification failure, it derives the central algebraic defect: fitting the
per-class scalar makes the normalized residual a weighted `1−CKA²`, which is
zero for every `S_c=lambda_c S*`, including a rank-one common operator. With
rank at most 3/5 and descriptor dimension 512, IFR has no rank or direction
floor and can satisfy itself through low-dimensional collapse. This confirms
the local concern that the term does not identify class-generic nuisance
structure. The external Zhou evidence is not repository provenance, and the
missing In-Shop-first screen is moot. Review details are in
`docs/opus_ifr_review_2026-08-06.md`.
