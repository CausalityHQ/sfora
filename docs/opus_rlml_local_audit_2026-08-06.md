# Local Gate 1/2 audit: Return-Level Metric Learning (blind pass 49)

**Verdict: DEAD at Gates 1 and 2. No diagnostic, preregistration,
implementation, or candidate GPU run is authorized.** RLML is the eleventh
internal recurrence of the gallery-tail/return-level family, only five blind
passes after TERL proposed the same pooled PWM-to-GPD estimator, EMA-stabilized
return level, positive-side hinge, and differentiated shape path. Prospective
repository evidence already says fitted shape adds essentially no signal and
is worse than a constant estimator. Public work occupies train-time EVT tail
correction and differentiable tail-index regularization.

## Frozen provenance

The proposal is frozen verbatim in
`docs/opus_rlml_proposal_pass49_2026-08-06.md` at commit `6bbc7ab`. Its file
SHA-256 is
`b4e8e7083d4326022c186988b1010e89027845b5b0f0cf27fe91dca80eee0533`.
The durable proposal job was `e5d7570e027846da`; Fable exhausted its credit and
the same job continued under Claude Opus. The worker verified the byte-identical
blind-prompt hash and used only public Web search/fetch after that bootstrap.

RLML takes a PFML embedding batch, selects the top 15% negative similarities,
fits pooled GPD shape and scale by probability-weighted moments, extrapolates a
nominal hardest-negative similarity to training-set size, and hinges it against
a smooth best-positive score. It deliberately differentiates through the PWM
shape. Its claimed fingerprint is an affine tail gradient that **attracts** the
lower part of the selected negative tail while repelling the upper part.

## Gate 1: the exact prospective measurement is adverse

Pass 31 XTail already measured the premise before training a return-level arm.
Across corrected embeddings, fitted tail shape added only **0.0087 incremental
R-squared** beyond threshold and mean excess. Shape noise consumed **67--86%**
of the full gallery-depth signal. A tuned constant deep-quantile estimator had
RMSE **0.0918**, versus **0.1575** for the fitted estimator: the constant was
about 1.7 times better. Those are not generic objections to EVT; they are the
prospective tests of the precise quantity RLML makes load-bearing.

RLML offers pooling and an EMA, which can reduce sampling variance but cannot
manufacture incremental predictive information or make a biased identity
mixture transport to disjoint identities. It supplies no new measurement that
shape predicts official-query errors after threshold/mean excess, no
train-to-unseen-identity transport result, and no intervention varying shape
while holding location and scale fixed. Its SOP `0.17` cosine example and R@1
forecasts are constructed arithmetic.

The closest completed blind object is Pass 44 TERL. TERL pooled normalized
exceedances across anchors, fit a Hosking--Wallis PWM GPD shape, used a
straight-through EMA, extrapolated to a larger nominal retrieval depth, and
hinged the result against a smooth best-positive score. Its cold review proved
that the differentiated shape path gamed the estimator and that parking at the
shape clamp reduced it to the constant estimator already favored by XTail.
RLML does not reverse either finding; it removes per-anchor normalization and
changes the cap/gradient wrapper.

## Gate 2: exact internal recurrence and occupied public mechanism

RLML is the **eleventh** gallery-tail recurrence after EVPC, RLM, EGR-PFML,
PORTAL, PORT, XTail, POTER, CFEV, GEVS, and TERL. Its defining sequence—select a
negative-score upper tail, fit or approximate its shape/scale, extrapolate an
extreme at a nominal larger gallery, and backpropagate that extreme—is identical
at supervision-object and action level. Pooling, plotting positions, a tanh
cosine cap, gradient-norm matching, and a different extremal-index heuristic are
estimator wrappers, not a new method mechanism.

The proposal itself finds the decisive public neighbours:

- **WEINCE** fits a POT tail of in-batch negative cosine similarities and uses
  the tail correction inside the train-time contrastive loss. Its deliberate
  stop-gradient through the fit is a gradient choice, not a new supervision
  object for RLML.
- **LDReg** differentiates a tail-index-family statistic of learned distances
  as a representation-learning regularizer.
- Recall@k surrogates, Smooth-AP/ROADMAP, XBM, rank/top-k/CVaR losses, and
  hardness-weighted mining occupy training pressure aimed at the operating
  retrieval depth.
- Post-hoc biometric EVT is less close, but confirms that return-level/FAR
  extrapolation is itself established.

RLML's residual distinction is the exact raw-PWM return-level gradient. That
gradient is not only a wrapper; it is openly adverse. The proposal derives that
for SOP, exceedances below about the 71st percentile of the already-hard top
15% receive an update that pulls a **negative closer**. It makes dependence on
that attraction its F5 novelty test. This is estimator gaming: changing the
sample shape to lower a fitted functional is not evidence of improved retrieval
geometry. TERL's independent algebra already demonstrated the same class of
shortcut in the normalized-PWM variant.

## Executable, selection, and arithmetic failures

- F4 fits the same tail statistic on **test-split negatives** and kills or
  advances the method from their discrepancy. That is explicitly prohibited
  test-identity method selection, not a legal post-hoc diagnostic.
- The mandatory corrected paired In-Shop screen is absent. The proposal starts
  five-seed controls across datasets, reports no raw-best versus frozen-final
  path, provides no out-of-sample confirmation stage, and does not specify a
  second-dataset escalation after In-Shop.
- The In-Shop forecast is internally inconsistent: `0.930 + 1.3` percentage
  points equals **0.943**, not the table's **0.934**. Its text then calls the
  `0.930` crossing a coin flip. The headline is not a frozen numerical
  prediction one can adjudicate.
- `N=|D_train|` is justified because it happens to resemble test-gallery size,
  while the no-test rule forbids selecting it from that resemblance. The
  four-dataset `N` dose is confounded with class count, images per class, and
  dataset identity; it does not identify gallery depth.
- PFML's batch size, sampler, schedule, weight decay, augmentation, proxy
  normalization, and alpha/delta choices remain unresolved, yet RLML fixes a
  4-by-45 batch and calls the carrier PFML. This is not a uniquely executable
  matched reproduction.
- Seven new knobs plus sweeps over kappa, Delta, and N have no complete
  class-disjoint train-only selection rule. Gradient-norm matching itself uses
  a noisy 50-step ratio and changes the effective intervention across arms.
- Collapse having a larger loss than a centroid construction does not prove it
  is nonstationary under normalized cosine; exact all-equal embeddings are
  first-order stationary for Gram-only losses.

The stated compute is plausibly small and deployment is clean, but free GPU
does not authorize an exact Gate-1/2 recurrence.

## What survives

The Hosking--Wallis PWM algebra, exponential/uniform sanity checks, stable
near-zero-shape series, cosine-cap projection, train-only deployment, and
explicit constant/quantile/XBM controls are useful. C3—the exponential-tail
constant-shape arm—is especially important because the repository measurement
predicts it should beat the fitted-shape arm. If it did, it would confirm the
existing negative mechanism, not create RLML novelty.

**Process lesson:** gallery-tail/return-level methods are now a hard family stop
unless a new prospective measurement first reverses XTail's incremental-signal
result. A different estimator, cap, or backward rule cannot be treated as a new
candidate when the supervision object has recurred eleven times. Never use
unseen test identities in a proposed cheap gate, and verify every headline
forecast by elementary percentage-point arithmetic before freezing it.

Primary neighbours:

- Erol et al., *WEINCE*, 2026: <https://arxiv.org/abs/2606.00262>
- Huang et al., *LDReg*, ICLR 2024:
  <https://openreview.net/forum?id=oZyAqjAjJW>
- Patel et al., *Recall@k Surrogate Loss*, CVPR 2022:
  <https://arxiv.org/abs/2108.11179>
- Wang et al., *Cross-Batch Memory*, CVPR 2020:
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html>

## Post-freeze reconciliation with the independent review

The frozen cold review in `docs/opus_rlml_review_2026-08-06.md` independently
returns **DEAD** and identifies the prohibited F4 test-identity gate as the
earliest failure. It confirms the internal recurrence and adds several stronger
exact results:

- The attractive-gradient fraction is **38.7% Cars, 39.7% CUB, 69.3%
  In-Shop, and 70.4% SOP** of the selected hardest-negative band. It rises with
  the claimed extrapolation depth and tends to 75%. Near the exponential
  reference, raising the bottom 70% is about **5.6 times** more loss-efficient
  than honestly lowering the top 30%.
- RLML's three changes from TERL cannot alter that sign: raw versus normalized
  exceedances leave PWM shape invariant to scale; the tanh cap is strictly
  monotone; and theta only shifts `log m`. They are inert estimator wrappers on
  the rejected gradient.
- Return level is monotone increasing in fitted shape, so descent has no
  interior resistance to driving shape toward the `-4` clamp. There it is
  essentially independent of `m` and reduces to a fixed scale offset. Worse,
  separately clamping/EMA-averaging shape and scale breaks the point-mass limit
  invoked in P2: scale can diverge while shape remains pinned.
- With the frozen `theta=1/mean_images_per_class` and
  `mean_images_per_class=N/C`, the proposed depth simplifies exactly to
  `m=theta*rho*N=rho*C`. Training cardinality cancels. The method therefore
  carries **training class count**, not gallery size. The In-Shop value is
  `0.15*3997=599.55`, not 598; the heuristic is not a valid extremal-index
  estimate because exceedance cluster size is not class size.
- The claim that training cardinality is within about 2% of test-gallery size
  is false on In-Shop: 25,882 is 105% larger than its 12,612-image gallery. It
  is close only to query plus gallery, a quantity unavailable to legal method
  design.
- The review confirms the In-Shop forecast is irreconcilable and counts about
  eleven, not seven, new constants. Tuning on CUB is especially uninformative
  because the proposal itself says extrapolation is inactive there, while the
  adverse attractive fraction nearly doubles on SOP.

The review preserves the PWM algebra, series, split cardinalities, gradient-
norm matching, several controls, legal zero-cost deployment, and the unpaired
SE as conservative rather than optimistic. None changes the Gate-1/2 death.
