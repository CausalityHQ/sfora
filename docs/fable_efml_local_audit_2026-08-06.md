# Pass 24 local evidence-aware audit: EFML

## Verdict before the independent review

**DEAD at Gates 1 and 2. No diagnostic, preregistration, implementation, or
GPU is authorized.**

This audit was frozen after the exact blind proposal and independent-review
prompt were committed, but before the independent reviewer returned. It checks
Exchangeable-Fiber Metric Learning (EFML) against the repository's verified
measurements, prior candidates, and primary literature.

## Gate 1: the causal premise is not measured

EFML assumes that zero-shot retrieval error is caused by different training
identities having different transported within-class displacement laws, and
that making those laws identical will transfer to unseen identities. The
verified packet establishes none of those links. In particular:

1. the repository does not measure an identity-independent fiber law, an
   association between classwise law discrepancy and corrected official-query
   errors, or an intervention showing that homogenization repairs those errors;
2. CINA's immediately preceding audit records the opposite limitation: the
   available measurements do **not** support a shared cross-class nuisance
   basis;
3. EFML's proposed P0 is prospective and only asks whether the 90th/10th ratio
   of scalar scatter exceeds 1.5. Heteroskedasticity alone does not show that
   the variation is nuisance, that its shape should be shared, or that
   equalizing it improves nearest-neighbor ranking; and
4. the claimed (+2.4/+1.9/+2.6/+1.5) point gains and all crossing
   probabilities are forecasts with no quantitative derivation from a
   repository measurement or reproduced matched base.

The exchangeability lemma is conditional on the very fact the method needs to
establish. It says what follows *if* all unseen and seen identities share one
fiber law and comparable center spacing; a loss on training identities cannot
prove either condition for unseen identities. The support-radius bound is a
generic triangle-inequality sufficient condition and does not show that EFML
reduces the relevant radius.

## Gate 2: the supervision target is occupied

EFML is not a new supervision object. It changes the estimator and geometry of
an established class-independent intra-class-variation target.

- Lin et al., *Deep Variational Metric Learning* (ECCV 2018), state that “the
  distribution of variance within classes is actually independent on classes.”
  Their KL term makes the intra-class-variation latent share one isotropic
  distribution, while a separate class-center representation carries identity.
  That is EFML's center/fiber decomposition and cross-class law-equality premise
  in a generative Euclidean parameterization.
- Zhu, Bai, and Wei, *Spherical Feature Transform for Deep Metric Learning*
  (ECCV 2020), transport feature variation between classes by spherical
  rotation under similar class covariances and evaluate on CUB, Cars, and SOP.
  EFML's proxy-to-pole transport plus distribution matching turns SFT's
  spherical shared-variation assumption into a penalty; “enforce rather than
  assume” does not create a different supervision target.
- Cheng et al., *Learning Deep Classifiers Consistent with Fine-Grained Novelty
  Detection* (CVPR 2021), train different class means with one covariance
  shared across classes. This already occupies the homoscedastic
  class-conditional representation target on CUB.
- Deep CORAL and class-conditional MMD/domain-alignment work make covariance or
  kernel-distribution equality differentiable training losses. A Gaussian MMD
  against a pooled reference is standard distribution alignment machinery.

EFML's remaining distinctions—sphere log maps, transport to a random pole,
64-D Johnson–Lindenstrauss projection, FIFO buffers, Gaussian-kernel MMD,
log-sum-exp aggregation, and a warm-up dispersion pin—are an estimator and
anti-collapse wrapper around that occupied target. They do not change what
supervision exists.

Primary sources:

- DVML: <https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html>
- SFT: <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf>
- Cheng et al.: <https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.html>
- Deep CORAL: <https://arxiv.org/abs/1607.01719>

## Frozen mathematical and experimental failures

1. **One random projection does not enforce the advertised law.** A
   characteristic Gaussian kernel identifies a distribution only in the space
   on which it is applied. Equality after one fixed map from 511 tangent
   dimensions to 64 leaves a roughly 447-dimensional nullspace in which
   class-specific structure can differ arbitrarily. The proposal nevertheless
   reasons as if full transported fiber laws coincide.
2. **The gauge is extrinsic and pole-dependent.** Minimal rotation to a pole is
   defined away from the antipode, but the comparison changes when the arbitrary
   pole changes because spherical parallel transport has holonomy. No
   pole-invariance control is included. The antipodal branch is also
   underdefined; clamping the cosine does not define the rotation axis.
3. **The global dispersion pin does not pin each class.** It constrains one
   batch mean. Classes can trade radius, and classwise collapse/expansion can be
   hidden by finite projected MMD error, detached buffers, bandwidth adaptation,
   or the projection nullspace. It therefore does not establish the claimed
   non-collapsed common law.
4. **The MMD estimator is underdefined and mostly stale.** “MMD squared” does
   not specify biased versus unbiased estimation. On SOP/In-Shop each class
   contributes only two live samples mixed with up to eight detached buffer
   entries, while the target is another detached, encoder-stale pooled queue.
   The value compares mixtures from different encoder times; most of each
   class statistic supplies no sample-side gradient.
5. **The pooled target is endogenous.** Every class is matched to a distribution
   partly made from the same classes, with class frequency, buffer age, and
   sampler history determining the target. This is not an independently fixed
   class-exchangeable law.
6. **Median bandwidth admits scale feedback.** The kernel scale is refreshed
   from the learned queue. Jointly rescaling or reshaping projected
   displacements changes both the samples and their measuring instrument; the
   frozen full-dimensional scalar pin does not make the nonparametric shape
   test scale- or history-independent.
7. **Cosmetic scatter is not excluded.** A deterministic network can encode
   background, crop, illumination, or other identity-irrelevant variation with
   the pinned norm and shared projected distribution. Those are “real”
   displacements but can worsen retrieval. Computing the loss from images does
   not prove that the retained fiber information is useful nuisance structure.
8. **Legitimate class-specific structure can be erased.** Pose, sex, life
   stage, product view, and clothing mode weights genuinely vary by identity.
   Forcing their marginal laws to match the pooled training law is bias, not an
   exchangeability theorem. RSPG already measured strong dataset dependence in
   derived intra-class structure.
9. **The controls cannot rescue novelty or provenance.** Scalar
   homoscedasticity, Euclidean residuals, a self-buffer placebo, and SFT may
   diagnose pieces of a result, but even a positive full-law arm remains an
   MMD implementation of DVML/SFT's occupied class-independent-variation
   premise. The placebo also changes the target from cross-class to
   self-history matching and is not machinery-equivalent.
10. **The benchmark base is invented and contradicts the active protocol.**
    The proposal changes backbone recipe, optimizer, sampler, batch
    composition, schedule, BN behavior, and evaluation convention; admits
    unresolved official defaults; assumes SOP/In-Shop baselines; and forecasts
    a 0.917 In-Shop PA base rather than using the repository's corrected
    three-seed 0.9035 baseline. Gate 4 requires the established In-Shop screen,
    not a new unmeasured lane.
11. **The tuning budget is not “one method.”** A 3-by-3 validation grid,
    sampler replacement, projection seed, buffer sizes, pole, queue length,
    bandwidth refresh, beta, warm-up statistic, and eight controls create a
    substantial design/selection surface. The cost forecast omits the separate
    class-disjoint validation trainings and five-seed control matrix.
12. **The crossing arithmetic is not evidence.** Probabilities computed from
    forecast means and forecast standard deviations against single-run or
    unmatched references are numerology. Best-test-epoch reporting preserves
    the selection bias this repository explicitly measured.

## Mechanism lesson

EFML is the fifth recent return to a shared cross-class nuisance law: CINA,
CRS, CITTR's transferable displacement, DVML/SFT-style variation transfer, and
now full-law MMD. The recurrence does not supply the missing measurement.
Future proposals should not revive class-independent nuisance or
homoscedasticity unless a prospective intervention first shows that a specific
measured classwise discrepancy causes corrected unseen-identity retrieval
errors. A different kernel, manifold, transport, or anti-collapse floor is not
that intervention.
