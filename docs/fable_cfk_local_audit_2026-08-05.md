# Pass 19 local audit: Common-Frame Kent Supervision

Date: 2026-08-05.

## Process record

Blind invention consultation `436e657a8ba74784` was correctly attributed to
`sfora/emafactorial`. Fable failed during its literature-tool setup and the
configured same-job Claude Opus fallback streamed a coherent CFK proposal but
exited nonzero. The durable answer window begins during section 1.6, so the
exact surviving provider text is frozen without reconstruction at
`docs/fable_cfk_proposal_pass19_2026-08-05.md`.

The incomplete prefix is already non-authorizing under
`docs/search_protocol.md`: an API failure or incomplete executable method is
not a candidate. The local scientific audit below nevertheless evaluates the
surviving mechanism rather than using the process failure as an easy rejection.
It was written while the mandatory fresh review was still running and therefore
does not depend on model agreement.

## Surviving proposed mechanism

CFK replaces Proxy Anchor's isotropic cosine score with a training-only
class-conditional diagonal quadratic score in one learned embedding basis. Each
seen class has a learned proxy and a learned per-coordinate precision vector;
the proposal claims a multiplicative gauge, clipping, count-dependent shrinkage,
and a small norm penalty prevent global precision collapse and one-coordinate
shortcuts. The per-class parameters are discarded and deployment remains a
single normalized 512-D descriptor with ordinary cosine retrieval.

The claimed causal error is that isotropic positive proxy attraction uniformly
suppresses factors which vary within seen classes but separate unseen classes.
Class-specific low precision is supposed to attenuate that attraction along the
affected coordinates while retaining negative discrimination.

## Gate 1: no eligible causal provenance

The verified packet does not measure the proposed quantity. It establishes:

- two corrected official-pixel In-Shop Proxy Anchor final checkpoints;
- near-saturated leave-one-out training retrieval;
- a descriptive foreign-image/foreign-proxy agreement error stratum;
- persistent query difficulty across two seeds; and
- a controlled augmentation-response relation which is not reducible to
  embedding distance.

None measures class-conditional coordinate precision, within-class factor
suppression by proxy attraction, transfer of a common diagonal frame to unseen
identities, or a causal relationship between retained within-class covariance
and official-query error. The packet explicitly withholds support for a shared
cross-class nuisance basis. Historical covariance, CUB, and Cars observations
are quarantined unless their artifacts and current path are independently
recomputed.

The augmentation-response relation is not provenance for CFK: it distinguishes
pairs by response to controlled image transformations, whereas CFK observes no
paired transformation or factor identity and merely learns a class-specific
axis weighting from the retrieval objective. The frozen CUB > Cars > In-Shop >
SOP ordering and all forecast deltas are therefore priors, not deductions from
repository measurements.

**Gate 1 fails.** No new diagnostic is justified because Gate 2 independently
occupies the executable mechanism.

## Gate 2: the mechanism is already benchmark-matched prior art

The proposal itself named the decisive paper but did not read it. Primary-source
inspection resolves the ambiguity against CFK.

Kirchhof et al., *A Non-isotropic Probabilistic Take on Proxy-based Deep Metric
Learning* (ECCV 2022), define for each class proxy a learnable diagonal
concentration matrix

```text
K_p = diag(kappa_p,1, ..., kappa_p,M)
```

in the shared embedding coordinates. Their construction stretches the unit
sphere into an ellipsoid: high-concentration coordinates emphasize distance
and low-concentration coordinates downweight it. They derive a
distribution-to-point non-isotropic proxy score, a distribution-to-distribution
variant, and explicitly combine the latter with Proxy Anchor. They report
ResNet-50/512-D results on CUB, Cars, and SOP, and show that training-time
non-isotropic scoring improves ordinary cosine retrieval even when the norm is
not used at test time.

This is the substantive CFK object: a learned per-seen-class, per-axis
non-isotropic proxy score in one shared basis, used only to shape the encoder for
unseen-class cosine retrieval. A bounded log-precision parameterization,
count-dependent shrinkage, or a hard scale gauge changes regularization and
optimization of that occupied object; it does not create a new supervision
relation or a new comparison mechanism.

Two further primary neighbors close the causal story independently:

- Roth, Vinyals, and Akata, *Non-isotropy Regularization for Proxy-based Deep
  Metric Learning* (CVPR 2022), identify proxy-induced local isotropy as a
  generalization problem and use a normalizing-flow regularizer to preserve
  class-local structure on CUB, Cars, and SOP.
- Lim et al., *Hypergraph-Induced Semantic Tuplet Loss for Deep Metric
  Learning* (CVPR 2022), learn a mean and diagonal covariance for every class
  prototype specifically to represent pose, viewpoint, and background
  variation, then train the embedding with the resulting distribution loss.
  This repository already implements that distributional HIST component.

L-GM (Wan et al., CVPR 2018), common-principal-component models, semi-tied
covariance models, and heteroscedastic discriminant analysis occupy still older
forms of class-conditional diagonal covariance in a learned shared frame. The
zero-shot proxy-DML papers above are closer and decisive.

Primary sources:

- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf
- https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2022/papers/Lim_Hypergraph-Induced_Semantic_Tuplet_Loss_for_Deep_Metric_Learning_CVPR_2022_paper.pdf
- https://openaccess.thecvf.com/content_cvpr_2018/papers/Wan_Rethinking_Feature_Distribution_CVPR_2018_paper.pdf

**Gate 2 fails.** No GPU, implementation, or preregistration follows.

## Frozen formal and control defects

The missing prefix removes the definitions of the raw precision transform,
count shrinkage, score insertion into Proxy Anchor, schedules, optimizer, and
regularizer. Consequently the surviving claims that the final precision has
product exactly one, the method reduces exactly to PA, and its gradients have a
particular form cannot be checked from the frozen artifact. They may not be
silently reconstructed.

Several surviving claims are independently false or underdefined:

1. Discarding a train-only class table does not prove memorization cannot affect
   zero-shot features. Gradients through an auxiliary seen-class table can steer
   the shared backbone toward seen-class shortcuts even when the table is later
   removed.
2. The class-independent diagonal-metric control C5 is not guaranteed to equal
   the baseline under normalized embeddings and ordinary cosine deployment. A
   global anisotropic training metric can change the learned head and backbone;
   absorbing it into a linear head does not preserve the subsequent unit
   normalization and deployed cosine metric without a compensating test-time
   transform.
3. C2 is not matched compute as stated: drawing and applying a fresh 512-by-512
   orthogonal frame each iteration is materially different from two diagonal
   score GEMMs unless a structured transform is specified.
4. C3 does not uniquely define which random precision row scores each
   sample/class pair. C4 matches nominal parameter count but changes the
   objective to multi-proxy pooling, so it cannot by itself isolate generic
   parameter capacity.
5. The implementation section has two score GEMMs total versus Proxy Anchor's
   one, but the cost section calls them two *extra* GEMMs. The table cannot be
   batch-sparse when all class scores require all `C*d` precisions each step.
6. The causal optimum argument considers positive compactness in isolation.
   Proxy Anchor's negative terms and all competing class proxies also reward
   retaining within-class factors when those factors reject impostors; uniform
   positive pressure does not prove global suppression is the only
   loss-reducing representation.

## Forecast and outcome

CFK on its fully specified PA base forecasts `0.723` CUB, `0.903` Cars,
`0.808` SOP, and `0.925` In-Shop: below every supplied Lane-A frontier. The only
forecasted crossing is conditional on an undefined CFK+PFML composition and a
future source-matched PFML reproduction. Even there, the stated clean-cross
probabilities are only `0.45`, `0.33`, and `0.30`; Cars and SOP do not exceed
their own two-standard-deviation thresholds in the frozen arithmetic.

## Local verdict

**DEAD at Gates 1 and 2, and non-authorizing as an incomplete frozen artifact.**
The verified repository does not motivate class-conditional precision
reallocation, and ECCV/CVPR 2022 already implement and benchmark the same
non-isotropic-proxy diagnosis and mechanism. The forecast does not establish a
frontier cross without an undefined composition. No diagnostic, implementation,
preregistration, or GPU work follows.

## Independent review and reconciliation

The mandatory cold review completed as consultation `e35e30324c1c4ade` after
Fable failed at tool startup and the configured same-job Claude Opus fallback
finished the unchanged prompt. Its exact result is preserved at
`docs/fable_cfk_review_2026-08-05.md`. It returned **DEAD** and independently
confirmed the decisive collision: Kirchhof et al.'s learned `diag(kappa_p)` is
per class, per embedding dimension, diagonal in the shared global basis,
composed with Proxy Anchor, and absent from cosine deployment.

The review also independently confirms or strengthens the formal defects:

- C5 is not an absorbable null after unit normalization and Proxy Anchor's
  log-sum-exp-with-one objective; therefore F8 would halt a correct
  implementation.
- C1 uses an across-class scalar gauge rather than CFK's within-class
  per-coordinate gauge and is not a nested isotropic control.
- Discarding the class precision table does not prevent it from relocating
  class memorization out of the deployed descriptor.
- The positive-only causal argument ignores equal attenuation of the negative
  term.
- The common-frame control changes learnability and noise simultaneously.
- The stated clean-cross probabilities are mutually incoherent.
- The missing prefix makes the exact transform and joint constraint
  unverifiable without repair.

One reviewer statement is **not adopted**: that Kirchhof et al. measured the
mechanism negatively by comparing their literature-table R50/512 values
`69.3/86.2/79.4` to CFK's forecast PA priors. Those rows are not a paired recipe.
Kirchhof et al.'s standardized Table 1 instead reports its own matched Proxy
Anchor comparison improving `64.4 -> 66.5` CUB, `82.4 -> 83.6` Cars, and
`78.0 -> 78.2` SOP. The paper therefore establishes occupied positive evidence
at a much lower historical horizon, not a negative causal estimate under CFK's
forecast recipe. This correction does not weaken the Gate-2 death: exact prior
art, not effect direction, is decisive, and the published R50/512 headline still
falls below the current PFML frontier.
