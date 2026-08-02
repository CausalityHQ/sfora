# Candidate 224: randomized relation ablation for causal supervision labelling

Date: 2026-08-02. Status: **DEAD at Gate 2**, with an independent identification
failure that would kill it even if the prior art did not exist. No diagnostic,
implementation, or GPU run.

## The proposal as audited

During ordinary DML training, randomly suppress selected same-class relation
contributions on a balanced factorial schedule; measure each factor's causal
effect on a fixed held-out **training-class** retrieval panel; fit a cheap
pixel-pair predictor of the estimated treatment effect; in a fresh run use the
predicted sign to assign a three-way relation label (positive / unknown /
repulsive). Constraints: pixels + labels only, roughly 1x cost via analytic loss
masking plus held-out panels, class-disjoint test, single embedding.

Claim under audit: randomized intervention identifies *causal supervision
usefulness*, which embedding similarity does not.

## Verdict

**DEAD.** One mechanism-level reason:

> The supervision object is unchanged from candidate 135 — a same-class edge
> labelled by its causal training helpfulness — and only the **estimator** of
> that helpfulness moves, from gradient influence to randomized ablation. This
> repository has already ruled twice that swapping the edge estimator is not a
> new supervision mechanism (133, 209), and the specific estimator being
> imported is itself occupied end-to-end: randomized subset ablation plus a
> cheap fitted surrogate of the held-out effect **is** Datamodels; doing it at
> ~1x inside one training run **is** In-Run Data Shapley; learning a
> feature-conditioned value predictor from stochastically-dropped training
> contributions scored on a held-out set, then reusing it in a fresh run, **is**
> DVRL.

## Closest primary sources

Operator (same-class edge labelled by causal helpfulness) — already closed here:

- Liu, R., Lin, Y., et al. *Debugging and Explaining Metric Learning
  Approaches: An Influence Function Based Perspective.* NeurIPS 2022. Empirical
  influence function for DML; identifies training samples responsible for
  retrieval generalization error; evaluates CUB, Cars196, In-Shop. This is the
  source that killed candidate 135.
- Pruthi, G., Liu, F., Sundararajan, M., Kale, S. *Estimating Training Data
  Influence by Tracing Gradient Descent* (TracIn). NeurIPS 2020. In-run,
  trajectory-based helpfulness of a training example for a held-out example.

Estimator + amortization (the actual delta) — occupied:

- Ilyas, A., Park, S.M., Engstrom, L., Leclerc, G., Madry, A. *Datamodels:
  Predicting Predictions from Training Data.* ICML 2022 (arXiv:2112.01008).
  Random training-subset masks → outcome on a fixed target example → fit a
  cheap (linear) surrogate that predicts the effect from the mask. This is
  candidate 224's measurement-and-surrogate template verbatim, at example
  granularity.
- Wang, J.T., Mittal, P., Song, D., Jia, R. *Data Shapley in One Training Run.*
  ICLR 2025 (arXiv:2406.11011). In-Run Data Shapley: per-iteration contribution
  to a validation objective, accumulated along one trajectory at negligible
  runtime overhead. Occupies the "roughly 1x, single run, held-out panel"
  position the candidate claims as its cost advantage.
- Yoon, J., Arik, S.O., Pfister, T. *Data Valuation using Reinforcement
  Learning* (DVRL). ICML 2020 (arXiv:1909.11671). A data value estimator network
  predicts value **from the datum's own features**; training contributions are
  stochastically dropped and the estimator is trained against held-out
  performance; the learned estimator then governs a subsequent training run.
  This is the "fit a cheap predictor of the treatment effect, then use it in a
  fresh run" half of the candidate.
- Ghorbani, A., Zou, J. *Data Shapley: Equitable Valuation of Data for Machine
  Learning.* ICML 2019. The retraining-based valuation the above accelerate.

Downstream policy (three-way relation label) — occupied:

- Xu, X., Yang, Y., Deng, C., Zheng, F. *Deep Asymmetric Metric Learning via
  Rich Relationship Mining* (DAMLRRM). CVPR 2019. Rejects all-positive
  adjacency; per-class visual MST leaves distant same-class pairs connected only
  indirectly. Directly re-verified in this repo. "Harmful → unknown" is this.
- Xuan, H., Stylianou, A., Pless, R. *Improved Embeddings with Easy Positive
  Triplet Mining.* WACV 2020. Loosened same-class relation to avoid class
  collapse.
- Ren, M., Zeng, W., Yang, B., Urtasun, R. *Learning to Reweight Examples for
  Robust Deep Learning.* ICML 2018; Shu, J., et al. *Meta-Weight-Net: Learning
  an Explicit Mapping For Sample Weighting.* NeurIPS 2019. Bilevel/meta
  weighting from a clean held-out set — the continuous form of the same map.
- Zheng, W., Chen, Z., Lu, J., Zhou, J. *Deep Metric Learning via Adaptive
  Learnable Assessment.* CVPR 2020. Meta-learned, episode-trained assessment of
  training tuples in DML — the closest "bilevel DML" primary.
- Han, B., et al. *Co-teaching: Robust Training of Deep Neural Networks with
  Extremely Noisy Labels.* NeurIPS 2018; Ibrahimi, S., et al. *Learning with
  Label Noise for Image Retrieval by Selecting Interactions* (T-SINT). WACV
  2022. Interaction selection over the retrieval distance matrix.
- Zhu, X. *Machine Teaching: an Inverse Problem to Machine Learning.* AAAI 2015;
  Liu, W., et al. *Iterative Machine Teaching.* ICML 2017. The algorithmic
  teaching framing.
- Hudgens, M.G., Halloran, M.E. *Toward Causal Inference With Interference.*
  JASA 103(482), 2008. The design-side primary for the interference objection
  below.

**Not resolved:** the "ACDC" referent in the brief could not be matched to a
primary source in DML, bilevel data curation, or data selection. Two searches
returned only the semantic-segmentation dataset of that name. If a specific
paper was intended it should be supplied; nothing in this verdict rests on it,
and the bilevel-DML slot is filled by Zheng et al. CVPR 2020 above.

## Why the "randomized, not correlational" defence fails at the operator level

The claim's force is epistemic (randomization > similarity), but the deployed
object is a per-pair scalar `w_ij` thresholded into `{+1, 0, -1}` and multiplied
into that pair's loss term. That is signed pair weighting. Every prior candidate
in this repository that produced a per-pair scalar died on the same reduction —
133 (estimator swap), 135 (causal helpfulness edges), 204 (design-based edge
statistics), 214 (trajectory-derived relabelling), 222 (endogenous interaction
selection). Randomization improves the *estimate* of `w_ij`; it does not add a
supervision relation that is not already `w_ij`.

Note also that this is the second time a fractional-factorial import has been
audited here: candidate 209 died as "an estimator for the already tried
augmentation-response displacement." The design methodology is not the novel
part in either case.

## Independent kill: the design cannot be run at 1x

This section stands on its own; it does not depend on any prior art.

Reference In-Shop recipe: **25,882** training images, batch **180**, **60**
epochs → **8,631** steps. The repo's measured same-class pair count is
**153,115**. Under 4-per-class sampling a batch carries ~45 × C(4,2) = **270**
same-class pairs, so the run contains ~2.33M pair presentations — about **15.2
presentations per pair**.

Two regimes, and the candidate needs both at once:

1. **Run-long assignment** (a factor is suppressed for the whole run). This
   produces a persistent, measurable end-of-run effect — but each factor then
   has exactly **one** contrast, and one training run yields one outcome. At 1x
   you buy on the order of **one** identifiable effect, not 153,115. Getting
   10⁵ effects at SNR ≈ 1 this way requires ~10⁵ runs, which is exactly
   Datamodels' cost and exactly why In-Run Data Shapley exists.

2. **Window-scoped re-randomization** (needed to get many contrasts from one
   run). Now the number of estimable factors is bounded by the number of panel
   readouts `T`. Even a generous `T = 1,000` (a panel every ~9 steps) leaves
   ~153 pairs bundled per factor, i.e. <0.1% of supervision per factor. Worse,
   a pair suppressed in one window is still presented in ~14 other windows, so
   the optimizer **re-supplies the withheld gradient** and the persistent
   component of the window-scoped effect decays toward zero. The design measures
   a transient the training process then erases.

The two regimes trade off along the same axis — treatment-window length — and
their product is bounded by one run's total learning. Against the repository's
own noise floor (1.08-point spread between nominally identical fixed-seed runs;
sd from 2–3 seeds already shown near-worthless), the per-factor SNR is far below
1 in regime 2 and the factor count is 1 in regime 1. Cost is *not* the binding
constraint here — a 2,000-image panel at T = 200 is only ~8% of run FLOPs.
**Identification is.**

## The treatment unit does not exist in the pinned baseline

`_proxy_anchor_loss` (`src/sfora/image_end_to_end.py:4939`) computes
image-to-**proxy** similarities only. The pinned `proxy_anchor` baseline has no
image–image same-class relation terms to suppress. Running candidate 224
therefore requires first replacing the positive term with a pair-based
(SupCon-type) term — a second declared delta from official Proxy Anchor, and
precisely the contested change RSPG was built around. The comparison would no
longer be one delta from the baseline it is scored against.

## Identification objections, individually

**Interference is the mechanism, not a nuisance.** Positive terms in both PA and
SupCon are log-sum-exp over the anchor's positives, so relation contributions
are *substitutive*: suppressing one positive mechanically raises the gradient
weight on the remaining positives for that anchor. Suppressed pairs also share
images, and every suppression moves the embedding that every other relation is
scored in. Balanced factorial main effects are therefore direct effects under
one particular Bernoulli(p) allocation (Hudgens & Halloran), effect-modified by
co-assignment. The deployment policy flips *all* predicted-harmful pairs
simultaneously — a saturated allocation with no support in the randomization
distribution. The extrapolation from sparse-marginal to saturated is not
licensed, and under substitution the sign can flip: a pair whose removal is
harmless when its partners remain is not harmless when they are removed too.

**The repulsive arm was never randomized.** The design's support is
`{present, suppressed}`. The policy deploys `{present, absent, repulsive}`. No
unit is ever assigned the repulsive arm, so its effect is not estimable from
this experiment under any sample size. Inferring "adding repulsion helps" from
"removing attraction helped" is a sign extrapolation outside the design, and it
is the single arm carrying the candidate's novelty over DAMLRRM's
already-occupied "leave it unconnected."

**Trajectory dependence breaks transport.** The estimand is a derivative of one
trajectory's state — which pairs are currently hard depends on where the proxies
and embedding happen to be. The fresh run's trajectory diverges from step 1
because the relabelling changes the loss. This is candidate 214's ruling
(training-indexed statistics are endogenous to the trajectory) plus the measured
1.08-point spread, and it is why Data Shapley variants have needed explicit
treatment of training stochasticity.

**The panel is the wrong outcome, and the repo already established that.** A
training-class retrieval panel is the outcome that the 195–198 audit ruled a
poor causal proxy: 95.604% of In-Shop queries have a same-series gallery
positive and 42.720% have *only* same-series positives, so training-class
retrieval rewards acquisition structure rather than identity generalization.
Test classes are disjoint and have no proxies, so the proxy-geometry component
of every estimated effect (the repository measured a 7.91-point LOO R@1 gap by
proxy ownership) cannot transport at all.

**The amortization step collapses the claim into the baseline it attacks.** The
pixel-pair predictor is fit to group-level, noise-dominated targets with a
capacity-limited function of two images. What it can recover is the smooth
appearance/acquisition component — and that component is exactly what this
repository has measured on In-Shop same-class pairs: 28.58% edge rate for
same-acquisition pairs versus 1.29% cross-acquisition, and accepted-pair mean
cosine 0.85 versus 0.63 for rejected pairs. So the predictor's learnable signal
*is* appearance similarity, i.e. Easy Positive / OSM-CAA / DAMLRRM weighting.
The claim "unlike embedding similarity" fails at the surrogate, not at the
estimator: even a perfect causal measurement is laundered back into a similarity
rule by the step that makes it deployable.

**Harmful → repulsive is label corruption under these constraints.** In-Shop and
CUB labels are clean by construction. "Suppressing this pair helped the panel"
is a statement about the current optimizer state — a hard pair with a
high-variance gradient — not evidence that the two images are different
identities. Repulsion changes the objective's minimizer to one that splits a
labelled class, while test-time retrieval requires matching across views of one
identity. The construction is only defensible under label noise, which the
constraint set excludes; with clean labels it injects the noise that T-SINT and
co-teaching exist to remove. Nothing in the repository authorizes it either: the
fragmentation/LOO association that might have been read as support is recorded
as **unidentified** after the partner-exclusion reversal.

## What survives from this audit

Nothing reopens. The process lesson to carry: a randomized design changes the
*epistemic status* of an edge weight but not the *operator* it feeds, and this
project's Gate 2 tests the operator. Candidate 224 is the fourth import in a row
(204, 209, 214, 224) whose novelty lived entirely in the measurement apparatus.
