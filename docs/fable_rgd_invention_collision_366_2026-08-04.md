# Candidate 366: Reliability-Gated Descriptors invention and collision audit

Date: 2026-08-04.

## Search design and frozen proposal

This was a second independent maximum-effort `claude-fable-5` invention pass
using the same outcome-only brief as candidate 365. It was not shown candidate
365 or the repository catalogue.

Fable proposed **Reliability-Gated Descriptors (RGD)**. A network would produce
a unit semantic direction split into groups and a positive per-image reliability
`gamma_g`. The exported 512-float vector would concatenate
`sqrt(a) gamma_g u_g` and `sqrt(b/G) gamma_g`. Ordinary squared-Euclidean nearest
neighbour search would then yield reliability-scaled cross terms plus the usual
query and gallery norm penalties. A Proxy Anchor term would train semantic
direction, an image-to-image listwise NCA term would train the deployed
Euclidean score, and a class-explained-variance penalty would discourage the
reliability head from encoding training identity.

Fable motivated the form by a first-term separable expansion of a
heteroscedastic Gaussian same-class likelihood ratio. It predicted In-Shop
final-state R@1 0.927--0.935 against about 0.915, CUB 0.698--0.708, and Cars196
0.874--0.882. It stated plainly that the forecast would not beat the audited
0.939 In-Shop CNN horizon.

## Factual correction

The invention report claimed that VAPNet could not be verified. This is false.
The primary NeurIPS 2023 paper is *Learning to Parameterize Visual Attributes
for Open-set Fine-grained Retrieval* and names the Visual Attribute
Parameterization Network in its abstract and method:

https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf

The brief supplied that exact title and venue. This is a Fable search failure,
not a defect in the recorded horizon.

## Gate 1: contradicted by corrected measurements

RGD's proposed causal target is generic gallery hubness/reliability. Corrected
repository measurements already argue against that target:

- the In-Shop training-neighbour audit found a maximum of only **six** queries
  sharing any top-1 neighbour and a 99th percentile occurrence count of three;
  it concluded that the signal is relational ambiguity, not generic hubness;
- **90.1%** of top-1 errors were still reciprocal within top 10, so a small
  unreliable/hub subset cannot explain most errors;
- direct hubness interventions were causally negative: CSLS changed R@1 by
  **-0.65 points** and Sinkhorn by **-3.16 points**; and
- the prospectively fixed raw-score diagnostic found normalized cosine
  **0.84365**, raw Euclidean **0.82163** (-2.201 points), and raw dot product
  **0.60051** (-24.314 points).

RGD learns a bounded reliability rather than reusing raw norm, so the raw-score
result does not by itself refute every calibrated Euclidean embedding. It does
refute the invention report's claim that the supplied evidence positively
identifies a decision-layer failure. Its lane comparison is confounded by
architecture, pretraining data, training recipe, and tuning. It cannot establish
that a stronger representation becoming worse on one row leaves only scoring as
the cause.

The proposed D1 is not a valid strict upper bound on an inductive per-image
penalty. Fitting one free scalar per validation-gallery item by coordinate ascent
uses item identity and labels and can memorize that particular gallery. An
image-to-scalar function available for unseen images is a much smaller function
class. The oracle can quantify a transductive item-bias ceiling, but it cannot
license the claimed inductive mechanism.

## Algebraic and mechanism reduction

The deployed score is exactly ordinary squared Euclidean distance in a learned
512-dimensional vector. Its gallery self-term is not a new comparison primitive;
every unnormalized Euclidean descriptor has `-||phi_y||^2 / 2` in a fixed-query
ranking. The reliability construction is a constrained parameterization of that
descriptor.

The Gaussian expansion does not rescue novelty. The exact likelihood requires
an infinite series; RGD deploys a learned first-order approximation and replaces
coupled self-terms with ordinary endpoint norms. The learnable scalars are fitted
by the retrieval objective, so the probabilistic derivation supplies an
initialization/interpretation rather than an exact new metric.

Gate 2 is occupied from every executable side:

1. **Per-image uncertainty/quality.** Probabilistic Face Embeddings, FastMLS,
   IDML, NIR/nivMF, MagFace, and AdaFace learn image-specific uncertainty,
   concentration, or quality and use it in comparison or training. Candidate
   116 already records norm-as-concentration as exact NIR prior art.
2. **Unnormalized or norm-aware scoring.** Candidates 105--111 audited this
   family after a positive within-identity norm diagnostic. Candidate 110's
   canonical retrieval test was decisively negative. A June-2026 primary paper
   identified by Fable additionally performs inference-time norm-aware retrieval.
3. **Listwise identification.** RGD trains its score with ordinary
   image-to-image listwise NCA. This is an established retrieval objective, not
   new supervision.
4. **Class-agnostic reliability.** Penalizing class-explained variance is an
   invariance/disentanglement regularizer. It does not create a new observed
   relation.

Combining an occupied uncertainty head, ordinary Euclidean distance, listwise
NCA, and an invariance regularizer is not a novel similarity-learning method
under the protocol. The exact closest-paper boundary may distinguish this
specific finite-dimensional parameterization, but that is an implementation
combination rather than a new training referent or comparison operator.

Primary sources checked after freezing the proposal:

- Wang et al., *Introspective Deep Metric Learning*:
  https://arxiv.org/abs/2309.09982
- Chen et al., *Fast and Reliable Probabilistic Face Embeddings in the Wild*:
  https://arxiv.org/abs/2102.04075
- Kirchhof et al., *A Non-isotropic Probabilistic Take on Proxy-based Deep
  Metric Learning*: https://arxiv.org/abs/2207.03784

## Verdict

**DEAD at Gates 1 and 2. No oracle fitting, implementation, preregistration, or
GPU.** The corrected repository evidence contradicts generic hubness as the
load-bearing In-Shop failure, and the executable method is an occupied
uncertainty-aware Euclidean embedding trained with an occupied listwise loss.
Its own forecast also falls below existing methods.

