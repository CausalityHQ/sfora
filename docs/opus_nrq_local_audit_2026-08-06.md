# Pass 52 local evidence-aware audit: NRQ

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_nrq_proposal_pass52_2026-08-06.md`  
Blind-proposal job: `2d34287adc3b4fa9` (Fable credit failure, durable Opus fallback)  
Proposal source SHA-256: `e6fd873a9f4e1177511c43b37e31c4914087083b6768e465320d5918a9035bae`  
Committed copy SHA-256: `d12787effc0353a399c688404d1009974b962ff691b78cec9abb660f43c87b83`
(the committed copy adds one terminal newline)

This audit was written before launching or reading the mandatory independent
review. It audits Nuisance-Register Quotient Metric Learning (NRQ) exactly as
frozen.

## Verdict

**DEAD at Gate 1 and Gate 2. No preregistration, implementation, or candidate
GPU run is authorized.**

NRQ's linear-regression tail bound is valid in its narrow form. It does not
establish the claimed useful anti-collapse mechanism: unregularized decoder
gain lets an arbitrarily small, high-rank side channel in the deployed
descriptor satisfy the certificate for arbitrarily small invariance cost.
Rank is discontinuous and carries no amplitude or semantic guarantee. The
split/reconstruct/discard action is also occupied by prior shared/private and
invariant/equivariant representation methods, while the frozen-initial-feature
anchor is occupied by transfer/plasticity regularization. The conjunction does
not repair the certificate or create an identified new supervision object.

## Gate 1: the proposed error mode has not been measured here

NRQ requires evidence that a corrected baseline:

1. achieves train augmentation invariance through seen-class lookup rather
   than an input-general computation;
2. has a harmful train/test-identity variance ratio `R > 3`;
3. collapses the deployed descriptor below the teacher-spectrum rank needed
   for unseen-identity retrieval; and
4. loses useful ImageNet-initialized feature variation during fine-tuning.

No such measurement exists in this repository. `R`, the teacher spectrum,
`k*(epsilon)`, the achieved sufficiency error, and the association of any rank
deficit with corrected retrieval errors are all proposed future diagnostics.
The proposal itself says the spectrum has not been measured.

The closest locked evidence is adverse to a shared nuisance route. Candidate
225 prospectively learned leading within-class subspaces on source identities
and tested their transfer to disjoint identities. Its corrected `rho_32`
values were **0.9312, 0.9287, and 0.9345**, all below the locked `1.15`
threshold and below one: the learned directions captured at least as much
target between-identity signal as target within-identity nuisance. Corrected
ARCG also found augmentation response heterogeneous and image-specific on
In-Shop: its graph retained only **0.3631--0.3640** of same-class pairs. That
does not support one compact, class-exogenous register containing the nuisance
that should be deleted for every unseen identity.

The repository has repeatedly exhausted the same premise or action through
CINA, EFML, CFR, FRAME, CNW, Pass 47 PQML, and Pass 48 NRC. A new forecast is
not provenance. Xue et al.'s external class-collapse result for supervised
contrastive learning does not show that the corrected PFML or Proxy Anchor
baseline in this repository has NRQ's specific failure.

## The rank theorem admits an arbitrarily cheap near-collapse escape

Let `H` be the centered frozen-teacher feature and `u=[zhat;n]`. Proposition 1
correctly says that affine prediction from a random vector whose covariance
has rank `rho` cannot beat the PCA tail after component `rho`. Its corollary
can therefore exclude the exact equality `zhat = mu_y` when the proposed
spectrum inequality and achieved reconstruction error both hold.

That exact, discontinuous statement is too weak for NRQ's causal claim. Choose
any high-rank bounded code `q(x,theta)` in tangent directions at the normalized
class code `mu_y`, and define

```
zhat_delta = normalize(mu_y + delta q(x,theta)).
```

For every nonzero `delta`, `Cov(zhat_delta)` may have the required high rank.
An unregularized decoder can recover the code with weights proportional to
`1/delta`. Meanwhile the cosine invariance cost produced by this side channel
is `O(delta^2)` and tends to zero as `delta -> 0`. Thus a representation can be
arbitrarily close in amplitude and behavior to class lookup while satisfying
the rank certificate and reconstructing the teacher target.

NRQ places no spectral-norm constraint on `V`, no minimum singular-value or
variance floor on the deployed code, and no signal-to-noise lower bound. Any
unspecified optimizer weight decay is absent from the theorem and is neither
bounded nor calibrated against `delta`. The theorem excludes a measure-zero
equality; it does not make the harmful near-collapse solution expensive.

This also shows why the random-teacher control is not merely a negative
control: the certificate can force recoverability of high-rank random or
background variation. Covariance rank alone says nothing about whether the
variation separates unseen identities.

## The nuisance is not identified or confined to the deleted register

The decoder jointly reconstructs the full teacher feature from `[zhat;n]`.
Nothing in the frozen objective assigns the view-dependent component uniquely
to `n`:

- finite `L_inv` only encourages two descriptors to agree; the infinitesimal
  side channel lets `zhat` retain view detail cheaply;
- the two linear heads share the same trainable trunk, so deleting `W_n` does
  not delete nuisance already encoded in trunk features or `zhat`;
- no orthogonality, conditional independence, adversarial exclusion, or
  decoder-block restriction separates content from nuisance; and
- `L_reg` penalizes only variance across **batch class means** of each register
  coordinate. Identity can remain in zero-mean within-class patterns,
  per-instance residuals, higher moments, or nonlinear codes. Four samples per
  class make those means noisy.

The teacher target is `h0` of the augmented crop, which mixes augmentation
effects with identity-useful fine detail and background. The recorded
augmentation parameters do not train the routing; they are used only in a
post-hoc probe. Consequently a successful reconstruction and low class probe
would still not prove that the deleted coordinates contain only nuisance.

Calling coordinate deletion a “fixed quotient” adds no guarantee. It is a
linear projection for all inputs, but the objective does not establish that a
nuisance action occupies its kernel.

## Gate 2: the supervision object and action are occupied

The closest primary-source mechanisms are:

- Bousmalis et al., **Domain Separation Networks** (NeurIPS 2016), explicitly
  partition representations into shared and private subspaces and reconstruct
  inputs from their combination while training the shared representation for
  the task. Treating augmentations as domains maps directly to NRQ's
  content/private-register/reconstruction/discard action.
- Jaiswal et al., **Unsupervised Adversarial Invariance** (NeurIPS 2018), learns
  a split representation through a prediction task and reconstruction coupled
  with disentanglement, specifically to preserve task information while making
  the task representation invariant to nuisance.
- Feige, **Invariant-Equivariant Representation Learning for Multi-Class Data**
  (ICML 2019), routes class information and within-class transformation into
  separate invariant and equivariant representations.
- Roth, Brattoli, and Ommer, **MIC** (ICCV 2019), is benchmark-matched DML: it
  learns a separate encoder for cross-class latent characteristics such as
  viewpoint and illumination so that structured variability can be explained
  away from the class representation.
- L2-SP and DELTA already preserve information from the pretrained initial
  model during fine-tuning through parameter- or feature-level constraints;
  InFeR uses regression to initial features to resist effective-rank collapse.

Primary sources:

- Bousmalis et al., *Domain Separation Networks*, NeurIPS 2016:
  <https://papers.nips.cc/paper_files/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html>
- Jaiswal et al., *Unsupervised Adversarial Invariance*, NeurIPS 2018:
  <https://papers.nips.cc/paper_files/paper/2018/hash/03e7ef47cee6fa4ae7567394b99912b7-Abstract.html>
- Feige, *Invariant-Equivariant Representation Learning for Multi-Class Data*,
  ICML 2019: <https://proceedings.mlr.press/v97/feige19a.html>
- Roth et al., *MIC*, ICCV 2019:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html>
- Li et al., *Explicit Inductive Bias for Transfer Learning with Convolutional
  Networks* (L2-SP), ICML 2018:
  <https://proceedings.mlr.press/v80/li18a.html>
- Li et al., *DELTA*, ICLR 2019:
  <https://openreview.net/forum?id=rkgbwsAcYm>
- Lyle et al., *Understanding and Preventing Capacity Loss in Reinforcement
  Learning* (InFeR), ICLR 2022 submission:
  <https://openreview.net/forum?id=5G7fT_tJTt>

NRQ changes the reconstruction target from pixels to frozen ImageNet features,
uses augmentation views as the nuisance source, and adds a rank-floor wrapper.
Those are implementation-level conjunctions around the already occupied
split/reconstruct/discard target and action. The only potentially distinctive
piece is the rank certificate, and the infinitesimal-gain counterexample makes
that piece causally ineffective. This is not a defensible novel mechanism.

## Protocol and benchmark failures

- The certificate is mathematically vacuous on In-Shop (`C=3997`) and SOP,
  which the proposal admits. The project protocol requires In-Shop screening
  first, but NRQ gives no numeric In-Shop prediction. A positive In-Shop result
  would contradict the claimed rank-floor mechanism; a null cannot establish
  it.
- The proposed training costs about `2.6--2.8x` per epoch. Equal-epoch
  comparison is not matched compute. A 540-epoch base changes its optimization
  schedule and sample exposure, so it is not an adequate wall-clock control
  without a frozen matched-update schedule.
- CUB/Cars/SOP forecasts are judgmental combinations of external numbers, not
  paired current-digest estimates. The proposal does not preregister raw and
  independently selected/final metrics, out-of-sample confirmation seeds, or
  the required In-Shop decision threshold.
- `r=64` is not derived uniquely. The inequality defines an admissible family
  conditional on an unmeasured spectrum and error; index 400 and the chosen
  register size remain design choices, and C10 later sweeps `r`.
- The selection section says “all four lambdas” although the objective has
  three lambdas. More importantly, selecting on CUB pseudo-identities and then
  screening a theorem-vacuous In-Shop arm cannot test the claimed mechanism.

## Gate decision

The earliest failure is Gate 1: NRQ has no positive repository measurement for
class-lookup invariance or harmful rank starvation, while the closest nuisance-
transfer measurement is adverse. Gate 2 independently kills it: its central
action is an occupied shared/private reconstruction-and-discard design, and
its advertised new rank floor permits arbitrarily cheap near-collapse through
unbounded decoder gain. NRQ does not proceed to Gate 3 or GPU.

The reusable lesson is sharper than “rank is not semantics”: **a covariance-
rank certificate without a lower singular-value/amplitude bound and a bounded
decoder is topological, not operational.** It can exclude exact collapse while
leaving arbitrarily close collapse essentially free.

## Independent cold-review reconciliation

The frozen review job `49727c9e13374d85` returned **DEAD at Gate 1** after the
local audit was committed. It independently constructed the same tangent-code
escape and made it explicit on CUB: choose the code in a complement of the at
most 100 class means and scale the corresponding decoder block as `1/delta`.
This agreement is decisive because the construction does not depend on the
unmeasured teacher spectrum or on candidate 225.

The review adds four valid defects that strengthen, but are not needed for, the
verdict:

1. With an EMA teacher mean rather than the exact population mean, the
   denominator of `L_suf` is `T + ||E[h0]-bar_h0||^2`. Therefore
   `L_suf <= epsilon` does not imply the numerator is at most `epsilon*T`, and
   the advertised `k*(epsilon)` certificate need not follow until mean error is
   separately bounded.
2. `r=64` is not uniquely derived. Setting the calibration tail after component
   400 leaves the CUB inequality true for a wide range of `r`; the window width
   is independent of `r`, and C10 later sweeps the supposedly derived value.
3. The controls omit register-plus-reconstruction with the alleged rank floor
   disabled at fixed compute. C4 (`r=0`) actually tightens the claimed rank
   burden, so the forecast ordering does not isolate the headline mechanism.
4. A 40% register class-probe falsifier is far too permissive for a 100-class
   dataset and contradicts the proposal's own predicted `<=15%`. The theta
   probe also omits pose, articulation, background, and other view-shared
   variation that the method calls nuisance.

The review also confirms that the mandatory In-Shop screen cannot adjudicate
NRQ's stated rank mechanism: the proposal declares the rank inequality vacuous
there and gives no In-Shop forecast. A positive arm would support only the
occupied auxiliary/quarantine family; a null would be non-diagnostic.

Several reviewer formulations are deliberately **not** propagated:

- PFML attraction being flat inside its radius does not prove exact class
  collapse is not a minimizer of the complete potential-field objective;
  repulsion, proxy-proxy terms, and the placement of all distances still
  matter. The valid point is narrower: the proposal did not prove its claim
  that collapse is globally optimal for its chosen base.
- Unit-sphere trace at most one permits arbitrarily small nonzero eigenvalues;
  it does not “force” every retained direction to be low-amplitude. The fatal
  absence is a lower bound, not the trace upper bound itself.
- Projecting `[zhat;n]` to `zhat` does delete the auxiliary coordinates from a
  conceptual joint representation. Saying it removes “nothing” is too strong.
  What remains true is that the objective never identifies those coordinates
  as a nuisance-action kernel, and the shared trunk can retain the same
  information in `zhat`.
- A 540-epoch matched-cost control changes schedule and exposure, but the claim
  that it necessarily biases toward NRQ through DML overfitting is unmeasured.

The independent review is preserved verbatim in
`docs/opus_nrq_review_2026-08-06.md`. These corrections leave both failed gates
unchanged.
