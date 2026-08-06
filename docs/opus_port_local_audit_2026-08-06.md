# Pass 26 local evidence-aware audit: PORT

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_port_proposal_pass26_2026-08-06.md`  
Independent-review prompt: `docs/opus_port_review_prompt_2026-08-06.txt`  
Review prompt SHA-256: `2b6094e8e3f1020c141087554ec50fde2f9189312ac19397912e49787648314c`  
Durable independent review: `ec5bbd01382d4b00` (running when this audit was frozen)

This audit was written without reading the independent review result.

## Verdict

**DEAD at Gates 1 and 2.** PORT is an equation-level return of Pass 23 PORTAL
and the earlier EVPC/RLM/EGR-PFML line. No diagnostic, preregistration,
implementation, or GPU is warranted.

## Gate 1: the same unsupported transfer premise

PORT assumes that a peaks-over-threshold law fitted to negative similarities of
seen training identities transfers to the hardest confuser among disjoint test
identities. The verified repository evidence contains no prospective
goodness-of-fit, class-holdout tail-index stability, queue-to-gallery transport,
or causal response showing that corrected zero-shot errors are caused by this
gap. Pass 23 already records this premise as explicitly outside verified
evidence.

The proposal's own cheapest test is illegal as written: it proposes fitting
tail parameters on **test identities of an already-trained model** before the
200-epoch sweep. The protocol forbids test-data access for method selection.
A legal class-holdout measurement on training identities would be prospective
evidence, but designing it now would not make frozen PORT novel and is not
authorized after the Gate-2 collision.

The proposed population sizes and `sigma log(N/M)` calculation show only that
an assumed tail model can produce a nonzero extrapolation. They do not measure
that the extrapolation predicts official-query errors or survives identity
shift. The SOP/In-Shop gain forecasts are invented from that story.

## Gate 2: PORT is PORTAL with renamed and slightly rearranged estimators

Pass 23 PORTAL already:

- fitted a differentiable GPD to the upper tail of negative similarities;
- used probability-weighted moments and an XBM/batch negative source;
- extrapolated a return level beyond the observed maximum to a nominal identity
  population;
- inserted that extrapolated hard negative into a retrieval loss;
- ramped/capped the extrapolation, treated WEINCE as the closest work, and
  claimed novelty over observed top-k/CVaR/Recall@k pressure; and
- proposed tail-transfer diagnostics, hard-negative/frozen-tail controls, and
  gallery-size sweeps.

PORT changes the source from the prior pooled/queue formulation to per-anchor
class-softmax block maxima, replaces the surrounding instance/PFML loss with
Multi-Similarity, floors the estimate at the observed maximum, and changes the
exact ramp, clamps, and controls. Those are estimator and robustness choices
inside the identical supervision object: **train against a GPD-extrapolated
unobserved population maximum**. It cannot be candidate-new after PORTAL was
frozen, independently reviewed, and recorded DEAD one pass earlier.

It also collides with Pass 9 EVPC, Pass 14 RLM, and Pass 15 EGR-PFML. Primary
literature occupies the components through WEINCE (online training-time EVT
correction), TriSim (GPD similarity tails used in retrieval training), XBM,
Recall@k surrogates, top-k/CVaR, and classical POT. The proposal omits TriSim
and all four repository-identical predecessors because the blind protocol hid
them; local Gate 2 exists precisely to catch this recurrence.

## Frozen mathematical failures

1. **Collapse is stationary, not a strict first-order escape.** At identical
   normalized descriptors all cosine similarities equal one. Although the
   scalar negative loss is high and `dL/d s_hat` is nonzero, the tangent
   derivative of cosine between identical normalized descriptors is zero. The
   asserted “any separating perturbation strictly decreases at first order”
   confuses derivative with respect to a score with derivative through the
   normalized embedding. The earliest change is second order.
2. **The anti-gaming floor can erase the advertised estimator.** Whenever the
   fitted return level is below the observed maximum, the `max` branch sends no
   gradient through the GPD fit. The loss then is ordinary hard-negative
   mining. Domination proves only that PORT is at least as large as that loss;
   it does not prove that extrapolation is identified or useful.
3. **The endpoint target can be gamed without representing an unseen law.** A
   network can make the fitted per-batch tail terminate at the observed maximum
   through batch-composition- and estimator-specific score shaping. Resampling
   classes does not establish POT stationarity while the encoder changes, nor
   transport that endpoint to unseen identities.
4. **The fit is a tiny, dependent, moving-tail estimate.** CUB/Cars use only 15
   order statistics per anchor. Class soft-maxima still share the anchor and
   encoder, batches are sampler-conditioned, and all scores move during
   optimization. Calling the blocks approximately independent does not supply
   the stationary iid sample assumed by POT/PWM theory.
5. **The clamp contradicts the claimed endpoint domain.** Positive fitted
   shape is called a sample artifact and clipped, while the same noisy fit is
   treated as a differentiable causal target. The loss can optimize into clamp
   flats or the exponential guard; neither behavior estimates the asserted
   finite endpoint.
6. **Population size is an adaptive-margin control, not new supervision.** For
   near-zero shape the return level is `u + sigma L`; changing nominal `N`
   changes an anchor-dependent margin computed from observed scores. Outside
   saturation this changes gradients, but it does not make an unobserved
   confuser observed or identified. In the softplus-saturated regime the
   additive shift becomes parameter-gradient-inert except through the fitted
   scale. The proposed `N` sweep therefore cannot by itself distinguish
   extrapolation from margin strength.
7. **Sign-mixed PWM gradients are not a unique causal fingerprint.** They are
   an algebraic property of this estimator and can attract some negative
   ranks. A rank-matched signed weighting can reproduce them without any
   population interpretation. The proposal recognizes the attraction risk but
   does not establish that it improves retrieval rather than destabilizing the
   tail fit.
8. **The test-data premise check violates the contamination rule.** Reporting
   train-versus-test fitted tail parameters before deciding whether to run is
   direct test-set method selection, even if no gradient is taken on test data.

## Forecast and control failures

The matched Multi-Similarity baselines and PORT deltas are forecasts, not
measurements, under a newly specified AdamW/200-epoch/batch recipe. Their
uncertainty and crossing probabilities cannot be combined with literature
reference error bars as if both were observed independent means. PORT itself
states that it is the wrong tool for CUB/Cars; its claimed frontier case is
therefore concentrated on the large-class datasets where recipe-matched
official controls are weakest.

The controls improve on PORTAL by including hard mining and rank-weight
comparisons, but still do not identify population extrapolation. A fixed or
learned anchor-dependent margin, a rank-matched signed-weight loss, and the same
tail fit with shuffled nominal population sizes can reproduce the operational
channels without claiming unseen-order-statistic supervision. Strictly beating
them would be necessary but would not repair Gate 1 or the exact Pass-23
collision.

## Mechanism lesson

Five neutral proposals have now converged on gallery-tail extrapolation. The
recurrence is evidence that the story is easy to invent, not that the
repository motivates it. Blind generation correctly withholds the failure
catalogue; local Gate 2 must therefore kill equation-level rediscoveries even
when the new proposal is more carefully written. Future proposals should not
return to GPD/EVT deployment-size corrections without a prospectively verified
new causal measurement and a genuinely different supervision object.

## Independent-review reconciliation

The separately frozen cold review (`docs/opus_port_review_2026-08-06.md`) also
returns **DEAD**. It accidentally opened the unrelated root file
`RSPG_TASK.md`, immediately declared it irrelevant, and inspected no repository
failure evidence; this is a minor protocol blemish but gives it no information
about PORTAL or the earlier return-level line.

The review independently verifies the PWM formulas, all five advertised
partials, `E'(xi)>0`, the domination inequality, the declustering upper bound,
and the broad FLOP estimate. It then supplies several stronger fatal results
that are adopted:

- every gradient weight on the retained order statistics is **exactly affine
  in rank**, a two-parameter family containing uniform top-k/CVaR; the claimed
  non-rank-determined fingerprint is false;
- at the capped SOP/In-Shop operating point the return level is already 85.5%
  of the way to the fitted endpoint, and changing nominal population size by
  six orders of magnitude moves it only another 14% of that span; the practical
  object is endpoint shaping, adjacent to WEINCE, not an identified deployment
  population;
- the threshold class receives the dominant repulsion while much of the hard
  retained tail receives net attraction, with large cancellation and severe
  PWM variance at `k=15/31`;
- on CUB/Cars the hard floor binds about 45.6% of anchors and the mean raw EVT
  increment is approximately zero, contradicting the forecasted mechanism
  gains;
- the motivating `0.07--0.13` cosine gap uses the `xi=0` limit even though the
  proposal insists on a bounded-tail `xi≈-0.3` regime, where its own formula
  gives only `0.013--0.026`;
- the exact-collapse guard evaluates `0/0`, while class log-sum-exp maxima can
  exceed one, so the frozen degeneracy branch is non-executable as written;
- a cliff-shaped tail makes `a1=0`, `sigma_hat=0`, and PORT exactly hard mining,
  directly falsifying the stated anti-threshold-gaming argument; and
- the controls do not isolate an arbitrary affine rank-weight rule, scaled
  CVaR, adaptive weighting, or the dataset-family batch-configuration change.

The review did not find a public method named PORTAL because the cold search
was barred from repository history. That absence is not evidence of novelty:
the frozen Pass-23 PORTAL artifact is an internal prior proposal, and Gate 2
rejects rediscovery regardless of publication status. Its stronger statement
that the operational endpoint is “precisely” WEINCE's object is treated as
adjacency rather than equation-level identity: WEINCE fixes the bounded
endpoint and stop-graduates the fit, while PORT estimates and differentiates
through it. The exact internal PORTAL collision is already sufficient.
