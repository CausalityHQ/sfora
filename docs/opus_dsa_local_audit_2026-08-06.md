# Local protocol audit: DSA (Pass 29)

Date: 2026-08-06. Frozen proposal:
`docs/opus_dsa_proposal_pass29_2026-08-06.md` at commit `9499280`.
Independent cold review consultation: `97f3aa216ed54605`, launched only after
the proposal and review prompt were separately committed. This audit authorizes
no implementation or GPU.

## Local verdict

**DEAD at Gate 1; independently occupied at Gate 2; no GPU.** The repository
contains no verified measurement that low between-training-identity scatter
rank causes official-query error, that test-identity contrasts are exchangeable
against the training eigensystem, or that increasing participation ratio
improves corrected zero-shot retrieval. Candidate 371 specifically found that
the available proxy-rank observation was dimensional algebra rather than
evidence of channel starvation and that its proposed demanded-rank statistic
would not identify an intrinsic benchmark requirement.

The frozen proposal additionally predicts no statistically defensible crossing
of PFML on CUB or Cars and no In-Shop crossing. Even its forecast therefore does
not satisfy the standing objective.

The proposed matrix is also not the stated between-identity scatter: means of
four stochastic images carry within-class and augmentation covariance whose
bias survives EMA averaging. Its exact rotation-equivariance proof fails when
the cutoff eigenvalue is repeated, precisely the regime approached by the
desired flattened spectrum. Finally, the supervision action is Representation
Self-Challenging in a spectral wrapper: remove the currently dominant
label-predictive representation and reapply the same label loss so the network
must use remaining features. A global between-class eigenspace is a different
mask selector, but not a new causal supervision object.

## Gate 1: the rank-starvation premise is unmeasured

The proposal supplies three numbers as if they were evidence: a CUB `27%`
failure rate, a base participation ratio of `8--15`, and an idealized `+4.1`
point ceiling. None is a measured repository quantity tied to this mechanism.
The first simply recasts a forecast retrieval error as the probability that an
unobserved quadratic form falls below an unmeasured threshold. The second is a
forecast. The third follows only after assuming that novel-identity contrast
directions are exchangeable with respect to the learned training eigensystem,
that scatter eigenvalues are encoder gains, that retrieval error is a scalar
threshold event, and that variance falls by 20% at fixed mean.

Those assumptions are not identified by R@1. Between-class scatter combines
the distribution of training identity means with the representation map; its
eigenvalues are not separately identifiable encoder gains. Transfer is possible
precisely because novel identities share visual factors with training and
ImageNet identities, so uniform exchangeability against the learned basis is
not an innocuous approximation. Conversely, if novel identity evidence were
truly independent and uniformly oriented, flattening finite training-identity
means would still not show that new transferable image evidence was learned.

The verified evidence packet explicitly does not support low-rank shortcut
monopoly. Candidate 371's proposed CUB rank audit was killed before execution:
centered `C` proxies having rank at most `C-1` is guaranteed, and the proposed
held-out metric-fit curve would depend on estimator, shrinkage, and split
rather than identify a minimum demanded rank. A 99-dimensional space can encode
arbitrarily many novel identities; `C-1 < 512` is not itself a capacity error.

## The actual EMA matrix is noise-contaminated and can be full rank

Let a batch contain `P` identities and `m` stochastic images per identity. Write
the batch mean as

```
mhat_c = mu_c + eps_c,
E[eps_c] = 0,
Cov(eps_c) = Sigma_c / m.
```

Conditional on the chosen identities, the proposal's denominator-`P` centered
scatter has expectation

```
E[S_batch] = S_between_subset
           + (P-1)/P * mean_c(Sigma_c) / m.
```

For the frozen `P=30, m=4` recipe, the retained within-class term is
`29/120` times the mean within-class covariance. Random crops and flips are
part of that covariance. EMA reduces sampling variance but does not remove this
bias. Because the noisy class means vary from step to step, a sum of rank-29
batch matrices can be full rank even with only 100 fixed training identities.

An independent NumPy Monte Carlo using the frozen `P,m` reproduced the formula
to maximum matrix-entry error `0.00227` after 20,000 draws. This is a formula
check, not dataset evidence. It shows that the claimed `rank(S_B) <= C-1`
mechanism does not describe the matrix the method actually diagonalizes.
Leading eigenvectors may mix identity separation with within-class nuisance and
augmentation response. Removing them could conceivably regularize training,
but that is a different, unidentified mechanism.

An unbiased repair would require independent half-means/cross-products or an
explicit within-class covariance subtraction. That changes the frozen method
and must not be silently substituted.

## The exact equivariance proof fails at spectral ties

For a symmetric matrix with a strict gap `lambda_r > lambda_(r+1)`, the
top-`r` spectral projector is unique and transforms as `Q P_r Q^T`. When the
cutoff eigenvalue is repeated, no unique rank-`r` subspace exists inside the
degenerate eigenspace. A deterministic eigensolver must choose an arbitrary
basis, so the selected projector need not transform equivariantly.

A direct 8-dimensional check used
`S=diag(2,2,2,2,1,.5,.2,.1)`, `r=2`, and a rotation mixing one selected and
one unselected vector inside the four-dimensional top eigenspace. The matrix
was invariant to numerical precision (`6.30e-17`), while
`||P(QSQ^T)-Q P(S) Q^T||_F = sqrt(2)`. This is structural, not a library bug:
an equivariant rank-2 choice inside an isotropic 4-space cannot be defined from
`S` alone.

DSA is designed to flatten leading eigenvalues, so the missing-gap case is not
remote. Near ties also make the refreshed projector discontinuous and invite
eigenbasis cycling. The frozen method specifies no cluster-aware cutoff,
random-Haar selection within tied eigenspaces, or stability condition. D1's
correlated-copy observation remains valid when its direction is selected, but
the stronger claim of exact basis immunity does not.

## Shortcut and objective audit

1. **Training-identity codes satisfy the loss.** A network can spread arbitrary
   training-identity artifacts over more than `R` directions. The residual
   Proxy-Anchor term has no information that distinguishes transferable pose or
   part evidence from background, collection, compression, or augmentation-
   stable identity codes. Held-out training identities reduce tuning bias but
   do not prove the causal quantity.

2. **EMA staleness is only operationally bounded.** `gamma=0.99` and a 50-step
   refresh state how old the estimate is; they do not prove a bound on how
   quickly `W`, the backbone, or proxy geometry can rotate. The cold reviewer
   judged this secondary rather than fatal, because the window is finite; no
   causal conclusion should depend on the stronger frozen claim.

3. **Both sides adapt.** The method projects and renormalizes samples and
   proxies together. Proxy adaptation can make the residual task easy without
   showing that the deployed embedding contains independently transferable
   evidence. The proposed participation-ratio signature does not separate this
   path from image-feature repair.

4. **Collapse resistance survives; singular projection does not.** The cold
   review correctly preserves D4: the same discriminative task on the residual
   does not reward collapse or volume inflation, and centering keeps the shared
   mean outside the between-class eigenspace. This is stronger than the local
   high-loss caution. The method still omits an epsilon for `||P_r f||` and
   `||P_r p||`; exact or near containment in the removed subspace makes the
   forward operation and displayed gradient undefined or unbounded.

5. **The multi-proxy base is a new recipe.** Hard nearest-proxy positives,
   ignored unassigned same-class proxies, zero proxy weight decay, a 100x proxy
   learning rate, cosine schedule, and `K=15` are not an exact reduction of
   published one-proxy Proxy Anchor. They may be executable, but every claimed
   delta must be against a reproduced matched base and cannot inherit PFML's
   frontier or causal story.

## Gate 2: the action is self-challenging feature suppression

The exact selector—an EMA top eigenspace of noisy batch class means—may be a new
wrapper. The training action and causal claim are established:

- Fei et al., *Beyond ID Bias: PCA-Guided Dropout for Robust Fine-tuning*,
  SCSL at ICLR 2025, is the closest spectral collision. It identifies dominant
  principal components of in-distribution representations, suppresses those
  components with structured dropout during fine-tuning, and claims that this
  forces reliance on alternative features that generalize OOD. DSA replaces
  total ID variance by labelled between-identity variance and reuses a metric
  loss, but the dominant-PCA-component suppression action and transfer story
  are already explicit: <https://openreview.net/forum?id=UvrSxutZK9>.
- Huang et al., *Self-Challenging Improves Cross-Domain Generalization*, ECCV
  2020, iteratively discards dominant label-predictive representation features
  and forces the network to activate remaining label-correlated features. DSA
  substitutes a global spectral selector for RSC's gradient-selected feature
  mask, but applies the same supervision to claim the same generalization
  mechanism: <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3018_ECCV_2020_paper.php>.
- Dai et al., *Batch DropBlock Network for Person Re-identification and Beyond*,
  ICCV 2019, uses shared erasure across a batch to force complementary
  discriminative evidence. Moving erasure from spatial feature maps to a
  descriptor eigenspace changes invariance and engineering, not the
  information-to-action map:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html>.
- Xiao et al., *Learning Deep Feature Representations with Domain Guided
  Dropout for Person Re-Identification*, CVPR 2016, already uses learned,
  nonuniform feature dropout in an identity-retrieval setting:
  <https://openaccess.thecvf.com/content_cvpr_2016/html/Xiao_Learning_Deep_Feature_CVPR_2016_paper.html>.

RSC is the decisive collision because its object is not merely ordinary
dropout: it finds the currently dominant task-predictive representation,
removes it during training, and reapplies label supervision to elicit alternate
evidence. PCA Dropout separately occupies dominant-component suppression in a
learned PCA basis. DSA's labelled scatter selector is a potentially useful
ablation within this known self-challenging family, not a new supervision
mechanism.

## Controls, cost, and objective mismatch

- C1 and C6 do not isolate novelty; they compare the proposed learned selector
  only with random/coordinate masks while omitting the closest gradient-based
  RSC control at matched erase fraction and schedule.
- C3 changes the deployed metric through train-scatter whitening. It is legal
  if frozen without test data, but it is not byte-identical deployment and does
  not isolate whether DSA extracted new evidence.
- C8's participation ratio inherits the within-class-noise bias and can rise
  without transferable between-identity signal. Held-out labels are legal for
  validation, but the statistic is not mediation.
- F5 compares CUB with SOP while changing class count, images per identity,
  proxy count, visual domain, and baseline difficulty; it does not isolate a
  rank ceiling.
- The `+2 MB` claim counts roughly the 512-square buffers but omits the
  second logit graph and projected sample/proxy activations. On In-Shop,
  `K=15` means 59,955 proxies and a `120 x 59,955` logit matrix: about 29 MB in
  fp32 before backward storage. The cold review independently found about
  11.4 MB extra on CUB and 128.5 MB on SOP. It verified that the backbone runs
  backward once and judged `1.02--1.03x` time defensible; the memory claim, not
  the time claim, is false.

Most decisively for the standing objective, the proposal itself forecasts DSA
below PFML on both CUB (`0.731` versus `0.734`) and Cars (`0.923` versus
`0.927`), below PA+DADA on In-Shop, and only at parity on SOP. It declines to
forecast the only proposed crossing configuration, DSA+PFML. A successful run
matching the frozen prediction would therefore not find a method that
outperforms the audited existing methods.

## Independent-review reconciliation

The frozen cold review (`docs/opus_dsa_review_2026-08-06.md`) returns **DEAD**
without qualification and independently reproduces the two main local results:
the batch-mean EMA contains `Sigma_W/m` bias, and RSC already publishes the
same remove-dominant-features/reapply-task-loss action. It adds five decisive
findings.

First, DSA confuses realized scatter with encoder gain. If latent training
identity contrasts have directional spectrum `s_i` and encoder squared gain is
`g_i`, then the observed between-identity eigenvalue is `lambda_i=g_i s_i`.
The proposal's sphere-uniform transfer calculation is minimized by constant
`g_i`, whereas flattening observed `lambda_i` forces `g_i` proportional to
`1/s_i` whenever the data are anisotropic. A perfectly flat-gain encoder can
therefore exhibit exactly the spiky scatter DSA calls diseased. The mechanism
observable does not identify the mechanism target.

Second, the registered probe is impossible under its natural full-split class-
mean definition. Participation ratio is at most rank, and centered means from
20 CUB validation identities have rank at most 19; the ten Cars identities have
rank at most nine. The forecast `30--45` cannot occur and a twofold Cars rise
cannot pass. If C8 instead means the noisy batch EMA, it can exceed those ranks
only through the within-class bias already shown above. Neither interpretation
is a valid mediation test.

Third, proxy refitting is an explicit short circuit. The ablated term projects
and trains 1,500 CUB proxies, giving roughly 720k residual proxy degrees of
freedom updated at 100 times the backbone learning rate with no weight decay.
The review's synthetic CUB-like residual remained perfectly separable through
`r=32`, so proxies can reduce the auxiliary loss while the encoder is fixed.
Nonlinear re-encodings such as `(s, ReLU(s), ReLU(-s))` separately turn one
scalar cue into several second-moment directions without adding image evidence;
D1 only defeats linear copies.

Fourth, every registered erasure control is easier than DSA. In the review's
CUB-like calculation, top-32 removal retained 13.5% of between-class energy,
while a random-32 removal retained 93.6%, with different renormalization and
gradient scales. C1 therefore does not isolate the selector. As the target
spectrum flattens, the top eigenspace becomes arbitrary and DSA itself converges
toward the random-subspace control, so F2 can fire on the claimed success state.

Fifth, the proposed base is internally confounded and partly unspecified.
Raising hard-assigned Proxy Anchor from `K=1` to `K=15` changes the positive-to-
negative normalization balance by about 3.75 times; at most four proxies per
present class can be attracted in a four-image batch. The frozen proposal also
does not say whether positive proxy assignment is recomputed after projection.
The primary Proxy Anchor source discloses AdamW, `alpha=32`, `delta=0.1`, the
100x proxy learning rate, and 40/60 epochs; the proposal incorrectly calls much
of this unresolved and substitutes a five-times-longer 200-epoch recipe.

The review preserves the projection-gradient derivation, the single-backbone-
backward construction and low time overhead, the exact sphere-uniform
quadratic-form variance lemma, collapse resistance, and the discipline of not
forecasting an unsupported frontier crossing. None rescues novelty, provenance,
executability, or the objective mismatch.

## Final local disposition

Preserve two observations: using a task loss after a data-dependent spectral
ablation is an executable way to avoid a direct log-determinant objective, and
the top-projector is rotation-equivariant when a strict cutoff gap exists. Do
not implement DSA. Its causal premise is unmeasured, its actual estimator mixes
within- and between-identity structure, its exact invariance fails at its
intended flat-spectrum state, its action is occupied by representation
self-challenging, and its own forecasts do not meet the research objective.
