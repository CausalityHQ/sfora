# EGR-PFML authoritative local audit

Date: 2026-08-05.

Verdict: **DEAD at Gates 1 and 2.** No diagnostic, preregistration,
implementation, or GPU run follows.

The exact blind Sol proposal is
`docs/sol_egr_pfml_proposal_pass15_2026-08-05.md`; the exact independent cold
Sol review is `docs/sol_egr_pfml_review_2026-08-05.md`. Both native
`devbox-cross` Codex launches failed before a provider receipt. The documented
shell fallback completed in two separate GPT-5.6-Sol sessions. Both reported
medium internal reasoning effort despite the requested maximum effort. The
proposal and review were frozen from their respective Codex JSONL transcripts,
with matching SHA-256 checks, before this local verdict.

## Gate 1: no measured causal premise

EGR-PFML assumes that corrected zero-shot R@1 headroom is caused by a
seen-to-unseen transferable GPD law for rare negative-image similarities and by
the mismatch between a finite training negative pool and a full deployment
gallery. The verified repository packet establishes neither claim. It explicitly
states that it does not support an unseen-class EVT tail shift. Corrected
In-Shop artifacts show persistent query difficulty but materially less stable
wrong-impostor identity, not a calibrated stationary tail law. Training
leave-one-out R@1 near 0.995 also supplies no evidence that stronger tail
pressure on seen identities repairs official-query errors.

The proposed `+0.007` CUB, `+0.004` Cars, and `+0.005` SOP forecasts are
assertions. No repository artifact estimates GPD goodness of fit, held-out
identity maximum coverage, error prevalence attributable to gallery
multiplicity, or the response of those errors to tail pressure. Gate 1 therefore
fails independently of implementation and novelty.

## Gate 2: repeated occupied mechanism

This is the return-level repair already exposed by blind pass 9 EVPC and then
returned almost exactly by pass 14 RLM: fit a per-anchor POT/GPD model to hard
negative similarities, extrapolate an extreme maximum, and train a margin loss
against that return level. Replacing RLM's proxies with an XBM-style image queue
and attaching the term to PFML does not change the mechanism class.

External literature is already close enough to defeat the broad novelty case:

- [WEINCE](https://arxiv.org/abs/2606.00262) applies anchor-wise online EVT
  correction to the top of a contrastive score distribution during training,
  explicitly handling the bounded cosine endpoint and altering hard-negative
  gradient allocation without learned tail parameters.
- [TriSim](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf)
  fits a generalized Pareto tail to high-similarity retrieval observations and
  feeds its probabilities into a triplet training loss. Its cross-modal task
  and false-negative purpose differ, but train-time GPD shaping of similarity
  extremes is prior art.
- [Cross-Batch Memory](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)
  supplies the detached FIFO embedding queue and current-query gradient path.
- [Pickands (1975)](https://doi.org/10.1214/aos/1176343003) and
  [Balkema and de Haan (1974)](https://doi.org/10.1214/aop/1176996548)
  justify asymptotic POT limits under suitable distributions and thresholds;
  they do not validate an adaptive, stale, correlated queue or transfer from
  seen to unseen identities.

The exact conjunction of PFML, an image queue, and a nominal larger-gallery
return level may be unpublished. It is not a substantively new training
mechanism relative to WEINCE plus XBM, and the proposal provides no measured
reason to expect its narrow wrapper to cross a frontier.

## Frozen mathematical failures

The independent review's specification objection is decisive and locally
accepted:

1. The proposal defines `(xi_hat, beta_hat)` both as an exact constrained MLE
   and as the result of eight damped Newton iterations. Eight iterations need
   not equal the argmin. No damping, line search, feasibility, projection,
   barrier, active-set, or boundary rule is specified, so the loss and gradient
   are not unique.
2. The displayed likelihood is undefined at `xi = 0`, although zero lies in
   the allowed interval. Only the CDF and return-level limits are supplied.
3. The initialization can violate GPD support; constrained optima and rank
   transitions are nonsmooth; all-zero exceedances leave `xi` unidentified.
4. The main return-level equation uses undefined `nu_i` where its continuous
   branch uses `u_i`.
5. Upper-clipping a raw return level above `0.9999` lowers the nominal quantile.
   The claimed 90% maximum-coverage implication then no longer follows. If the
   clamp saturates, the negative-tail gradient can disappear and leave only an
   easiest-positive compactness term.
6. The queue is class-balanced, stale, correlated, and generated by changing
   encoders. FIFO turnover does not establish iid sampling, stationarity, or
   seen-to-unseen tail transport.
7. On SOP, 11,318 training identities can exceed the queue's 64-step lifetime
   before an identity repeats. Under a normal without-replacement identity
   sampler, the valid-positive set can be empty and EGR can contribute no
   gradient. The replacement policy is absent.
8. The proposal changes PFML's optimizer, learning-rate ratios, schedule,
   augmentations, and regularization while comparing its forecasts directly
   with published PFML numbers. Those recipe changes prevent causal attribution
   even if a later completed implementation improved.

The monotonicity claim is also unavailable. A lowered top-tail observation
refits the threshold, shape, and scale jointly; no implicit-derivative sign is
proved. Outside the top 64 or under clipping it changes nothing, while within
the tail a heavier fitted shape can raise the extrapolated return level.

## Process conclusion

The Sol substitution worked as an invention/review process but not as a source
of a new mechanism: it independently regenerated the already-dead EVPC/RLM
family. This is useful evidence that a neutral proposer repeatedly gravitates
to gallery-tail correction, not evidence that the method works. The next blind
proposal must receive the same neutral task statement rather than this failure
catalogue. No GPU is authorized.
