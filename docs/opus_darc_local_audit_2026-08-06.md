# Pass 36 local audit: Discriminant-Axis Residual Capacity (DARC)

Date: 2026-08-06 UTC. This audit was written after the proposal and review
prompt were frozen, while the independent consultation
`65d94bfa5f8c4532` was running, and before reading any reviewer partial.
No implementation, preregistration, diagnostic requiring a learned artifact,
or GPU work is authorized.

## Verdict before reviewer reconciliation

**DEAD at Gates 1 and 2.** DARC is the same supervision premise already tested
as Pass 21 RIM: preserve augmentation-repeatable within-class residual
information because class-discriminative training may erase factors useful for
unseen identities. The verified repository packet does not establish that
premise, and Candidate 225 supplies adverse disjoint-identity evidence for a
shared transferable within-class subspace. Independently, DARC's load-bearing
dimension-free dither floor is false after normalization, its default dither
dominates the unit descriptor, and its finite-batch ANOVA is biased. The
written objective is therefore not the absolute, encoder-independent capacity
measurement it claims to be.

## Frozen object

DARC draws two augmentations of each training image, adds independent
`N(0, sigma^2 I_512)` dither after descriptor normalization, normalizes again,
and class-centres those dithered descriptors. It maintains a detached EMA of
between-class scatter, keeps its top 16 eigenvectors, and maximizes a weighted
sum of `0.5 log(1 + tau_hat/(nu_hat+epsilon))`, where a two-view decomposition
is intended to separate repeatable between-image within-class variation from
view noise. Nothing survives at deployment.

## Gate 1: the repository evidence is absent and partly adverse

Pass 21 RIM already proposed the same causal object with a different estimator:
class-centre two augmented-view batches and maximize repeat-reliable
within-class residual capacity. Its frozen audit found that the corrected
In-Shop packet measures an augmentation-*response relation between different
same-class images*, not the rank, magnitude, semantic content, or zero-shot
utility of same-image augmentation-repeatable residuals. No subsequently
verified artifact establishes DARC's new claims that Proxy Anchor quantizes a
continuous attribute into seen-class levels, that residual variance along the
top between-class eigenvectors is useful to unseen identities, or that
preserving it repairs R@1.

The closest prospective repository measurement points the other way. Candidate
225 learned a pooled within-class subspace on one half of corrected In-Shop
training identities and evaluated transfer on disjoint identities. Its locked
captured-within/captured-between ratios were **0.9312, 0.9287, and 0.9345** over
seeds 0--2, below the preregistered `1.15` falsifier and below the random-
subspace reference near one. That does not prove that every nonlinear residual
factor is useless, but it is adverse evidence for exactly the class-exogenous,
transferable within-class structure DARC needs.

Selecting the *top between-class* axes does not repair provenance. A direction
that separates training-class means is not thereby a continuous semantic
coordinate shared by unseen identities. Expanding each seen class along that
direction can instead increase class overlap. The proposal itself names the
concrete adverse cases—sex on CUB, colour on Cars, pose and background—and
provides no measurement that its desired “quantization resolution” dominates
them.

Relevant frozen artifacts:

- `docs/fable_rim_local_audit_2026-08-05.md`
- `docs/fable_rim_review_2026-08-05.md`
- `docs/location_dependent_nuisance_audit_226_2026-08-02.md`
- `docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md`

## The dither theorem is false

The proposal asserts, for every encoder and measurement axis,

```
E[nu_hat_k] >= sigma^2 / (1 + sigma^2).
```

That ignores the 512-dimensional norm in

```
normalize(z + sigma epsilon).
```

For a generic axis orthogonal to `z`, the projected dither variance is roughly
`sigma^2/(1+d sigma^2)`, not `sigma^2/(1+sigma^2)`. If `z` aligns with the
measurement axis, radial noise is cancelled to first order by normalization,
and the coordinate variance is smaller still. The encoder and slowly updated
frame can therefore change the supposed absolute noise ruler.

A deterministic standard-library Monte Carlo check used `d=512`, 30,000
samples per cell, seed `20260806`, fixed unit `z`, and either `u=z` or `u`
orthogonal to `z`:

| sigma | sigma sqrt(d) | claimed floor | aligned variance | orthogonal variance | mean aligned coordinate |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.6788 | 0.000899 | 0.000126 | 0.000613 | 0.8275 |
| 0.05 | 1.1314 | 0.002494 | 0.000487 | 0.001108 | 0.6624 |
| 0.10 | 2.2627 | 0.009901 | 0.001258 | 0.001633 | 0.4043 |
| 0.20 | 4.5255 | 0.038462 | 0.001746 | 0.001869 | 0.2153 |

At the frozen default, the claimed lower bound is too large by about **7.9x**
in the aligned case and **6.1x** in the orthogonal case. More fundamentally,
`sigma=0.10` is not a small calibration perturbation in 512 dimensions:
its expected noise norm is `2.263` against a unit signal. Every point in the
proposed sweep is material (`sigma sqrt(d)=0.679` to `4.525`), and the default
leaves only about `0.404` mean projection onto the original descriptor in the
aligned construction.

The proposal labels the dither bound load-bearing: without an absolute floor,
the ratio can be gamed and the capacity interpretation becomes a scale-free
reliability regularizer. The bound's failure therefore kills the frozen
mechanism rather than suggesting a hyperparameter adjustment.

## The stated ANOVA estimators are not both unbiased

Even granting the additive population model

```
y_iv = t_i + xi_iv,
Var(t_i)=tau, Var(xi_iv)=nu,
```

the implementation first subtracts the same empirical class mean from all
`2m` observations. The paired difference is unaffected, so `nu_hat` is
unbiased under the ideal additive assumptions. But with denominator `m`, the
variance of the centred pair means has expectation

```
E[(1/m) sum_i (b_i - b_bar)^2]
  = ((m-1)/m) (tau + nu/2).
```

Consequently the frozen estimator obeys

```
E[tau_hat] = ((m-1)/m) tau - nu/(2m),
```

not `tau`. The bias is material for the frozen samplers: `m=6` on CUB/Cars and
`m=3` on SOP/In-Shop. Normalized dither adds a second failure because its
denominator depends on all coordinates and on `z`; the additive IID replicate
model does not hold exactly. Applying `softplus(tau_hat/s)*s` then adds a
positive floor (`s log 2 = 0.000693` at zero for `s=0.001`) and changes both
the estimand and gradients.

The displayed “exact” derivatives omit these executed operations. For

```
p = s softplus(tau_hat/s),
C = 0.5 [log(nu_hat+epsilon+p) - log(nu_hat+epsilon)],
```

the derivatives include `epsilon`, and

```
dC/d tau_hat
  = 0.5 sigmoid(tau_hat/s)/(nu_hat+epsilon+p).
```

The proposal instead differentiates a hard positive `p=tau_hat` with
`epsilon=0`. Its formula is not exact for the specified loss.

## “Reverse water-filling” is not derived

The scalar stationarity display freezes `g_k`, `nu_k`, `w_k`, the frame, class
means, and all other coordinates while treating each `tau_k` as an independent
control variable. In the executable model all of them are coupled through one
unit descriptor, empirical centring, normalized dither, Proxy Anchor, and the
EMA frame. No Lagrange multiplier for the unit-sphere power budget is present.
The equation is at most a local scalar balance under assumptions, not the
Gaussian rate-distortion reverse-water-filling solution, and it proves neither
an interior nor a unique optimum. The assertion that Proxy Anchor's marginal
cost necessarily grows with within-class spread also does not follow from the
convexity of one exponential after the complete log-sum-exp and moving proxies
are included.

## The semantic signal is not identifiable

Two views of the same source image certify repeatability, not usefulness for
unseen identity. Stable background, acquisition condition, native resolution,
watermark, pose, and an image-specific code all contribute to `tau_hat` just as
a desired latent attribute does. Independent augmentation parameters remove
only factors that the chosen augmentation family resamples. Dither is
content-independent noise and cannot distinguish the remaining signals.

The proposed C5 image-index probe does not decide this shortcut. If it trains
and tests on the same one frozen descriptor per image, the classifier has one
sample for each of 5,864 labels and millions of class-weight parameters; setting
each class weight to its sample descriptor already makes self-score one and
usually gives trivial training accuracy. If it is intended to generalize
across augmentations or held-out observations of an image, that data split is
not specified. A nonlinear or distributed hash can also evade a linear probe.
Saturation and a sphere budget cap how much code is stored but provide no
preference for semantic over nuisance code.

C7 is likewise not causal. Increasing dispersion moves readings away from the
nearest seen-class scalar levels almost by definition. Both the capacity and
the “level snapping” mediator can improve under generic variance expansion,
stable nuisance, or instance coding without increasing unseen-class R@1.

## Gate 2: an internal recurrence inside occupied public mechanisms

DARC's exact dither-plus-ANOVA-plus-eigenframe wrapper may be bibliographically
distinct. Its supervision object and action are not. Pass 21 RIM already froze
and rejected the same operation: maximize augmentation-repeatable
class-residual information in a deployed DML descriptor. DARC replaces a
whitened cross-covariance trace with a per-axis signal-to-noise log and chooses
a between-class frame; those are estimator choices inside the same mechanism.

The public neighbourhood is also explicit:

- Wang et al., *Ranked List Loss for Deep Metric Learning* (CVPR 2019), state
  that pulling each class to one point destroys intra-class similarity and
  deliberately preserve structure inside a class hypersphere:
  https://arxiv.org/abs/1903.03238
- Lin et al., *Deep Variational Metric Learning* (ECCV 2018), diagnose the
  suppression of intra-class variance as harmful to unseen-class
  generalization and explicitly model that variance on CUB, Cars and SOP:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html
- Ermolov et al., *Whitening for Self-Supervised Representation Learning*
  (ICML 2021), and Barlow Twins (ICML 2021) train augmented views using
  variance/whitening and cross-view agreement:
  https://proceedings.mlr.press/v139/ermolov21a.html and
  https://proceedings.mlr.press/v139/zbontar21a.html
- Zhang, Jayasuriya, and Berisha (NeurIPS 2023) add an intra-class-correlation
  repeatability regularizer to contrastive embedding training:
  https://arxiv.org/abs/2310.17049
- Roth, Brattoli, and Ommer, *MIC* (ICCV 2019), explicitly target cross-class
  latent characteristics such as viewpoint and illumination for DML
  generalization:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html

The exact rate wrapper is not needed to decide Gate 2 because its only claimed
novel discriminator—the absolute dither ruler—is mathematically false. After
that failure, the executable residue is class-conditional variance preservation
plus cross-view reliability, already occupied and already rejected internally.

## Protocol, forecasts, and cost

The proposal is candid that standalone DARC forecasts **zero of three** Lane-A
frontier crossings: `0.708<0.734` CUB, `0.894<0.927` Cars, and `0.806<0.829`
SOP. Every crossing depends on an unreproduced PFML base and an assumed
`0.5--0.7` additive fraction. That fraction has no measurement behind it and
cannot authorize an objective whose standing goal is to outperform the
matched frontier.

The frozen experiment order also violates the governing protocol. It supplies
no corrected In-Shop forecast, explicitly excludes In-Shop, and makes the
primary falsifier a five-seed CUB/Cars conjunction. Gate 4 requires a corrected
In-Shop-first screen after Gates 1--3. The proposal also does not commit to raw
plus independently selected/final metrics. C1 matches forward-pass count but
not information exposure: two augmentations of 90 images are not the same
sampling object as 180 distinct images. The 60-cell hyperparameter grid before
controls and multiple mechanism tests requires a selection plan absent from
the proposal.

Per-step overhead can be small and deployment cost is correctly zero, but
“1.01x training” does not describe the proposed programme of five-seed bases,
eight controls with subarms, sweeps, two datasets, and an unreproduced PFML
composition. Low per-step arithmetic cannot rescue failed provenance,
mechanism, or expected-value gates.

## Authorizing condition

There is none for frozen DARC. Correcting the dither scaling, changing the
estimator, introducing an observable that distinguishes useful residuals from
stable nuisance, or replacing the supervision target is a substantive new
proposal and must restart blind generation and freezing. No GPU work should be
spent on DARC.

## Independent-review reconciliation

The sole cold review, job `65d94bfa5f8c4532`, is frozen byte-for-byte in
`docs/opus_darc_review_2026-08-06.md`. It independently reproduced the two
decisive arithmetic failures: the normalized-dither variance is far below the
proposal's dimension-free bound, and finite-sample class centring biases
`tau_hat` downward. It also found a still sharper executable failure. At the
proposal's observed noise scales (`nu` about `0.0006--0.0019`), the supposedly
innocuous clamp contributes `softplus(0)*s = 0.000693` even when the signal is
zero. That is 0.37--1.13 times the entire noise denominator and manufactures
about 0.16--0.38 nats of capacity from no reliable signal. The optimized
residue is therefore a strong class-conditioned, rotated cross-view invariance
penalty, not the claimed capacity allocator.

The review correctly narrows one statement in the local audit. Although the
proposal's lower bound and numerical constants are false, in a generic
high-dimensional asymptotic the common attenuation cancels and
`tau_hat/nu_hat` can converge to `tau_raw/sigma^2`. Thus exogenous dither can
serve as an idealized absolute ruler in that restricted sense. This does not
rescue frozen DARC: the floor depends on alignment `u_k^T z`, so the encoder can
change it; the executed softplus term dominates the measured scale; and
`tau_hat` remains biased. The audit's broader claim that the false bound alone
destroys every scale-free ratio is withdrawn.

Similarly, the two displayed gradients are exact for the ideal hard expression
`0.5 log(1+tau/nu)`, but they are not the derivatives of the specified
softplus-and-epsilon loss. D2 and D3 are sound for nuisance factors actually
resampled independently by the augmentation family; they do not identify
stable background, acquisition, pose, watermark, or image-code signal. The
review also confirms that `nu_hat` itself is unbiased under the ideal additive
model.

Further review findings strengthen rather than change the disposition:

- the default `sigma=0.10` dither supplies roughly 84 percent of the normalized
  vector, and the false floor is encoder-controllable through alignment;
- the claimed reverse water-filling is only an equal-marginal stationarity
  display with no sphere-budget multiplier, coupled-noise dynamics, or
  uniqueness proof;
- C5's one-example-per-image probe is trivially self-classifiable and C6 omits
  the plain cross-view-consistency control;
- 180 forwards over 90 unique images do not match the baseline's 180 unique
  images, so F1 compares DARC to the wrong exposure control;
- the assumed PFML additivity is mechanism-inconsistent: PFML's 15 proxies per
  class already attack the one-proxy quantization DARC claims to repair, making
  a shrink factor near zero at least as plausible as the frozen 0.55; and
- the standalone forecasts remain zero frontier crossings.

The public literature may not contain this literal wrapper, but the repository
already does contain the same supervision object and action in Pass 21 RIM:
preserve augmentation-repeatable class-centred residual capacity. The public
neighbourhood separately occupies intra-class variance preservation,
cross-view repeatability, whitening/variance regulation, and class-discriminant
frames. Changing the ruler, estimator, and frame does not clear Gate 2 here.

## Authoritative disposition

**DEAD at Gates 1 and 2; no preregistration, implementation, or GPU.** The
restricted dither-ratio observation, unbiased `nu_hat`, ideal hard-loss
derivatives, and augmentation-noise rejection are preserved as correct
subcomponents. They do not supply repository provenance for useful residual
signal, distinguish semantics from stable nuisance, make the executable loss a
capacity allocator, create a new supervision object, or forecast a standalone
frontier crossing. Any repair must return as a new blind proposal rather than
an amendment to DARC.
