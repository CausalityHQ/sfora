# Pass 23 local evidence-aware audit: PORTAL

Date: 2026-08-05 UTC  
Frozen proposal: `docs/fable_portal_proposal_pass23_2026-08-05.md`  
Independent-review prompt: `docs/fable_portal_review_prompt_2026-08-05.txt`  
Review prompt SHA-256: `4d33619268e18e652f393bc70ac384f9a0125d0efebc33787ed9179a930f5b35`  
Durable independent review: `fa1510780f0840f5` (still running when this audit was frozen)

This audit was written without reading the independent review result.

## Verdict

**DEAD at Gates 1 and 2, and the forecasted PFML composition is not an
executable frozen method.** No diagnostic, preregistration, implementation, or
GPU run is warranted.

## Gate 1: the causal premise is explicitly outside verified evidence

PORTAL assumes that corrected zero-shot errors are caused by a transferable
seen-to-unseen negative-similarity tail law and a training-class-to-gallery
extreme-value mismatch. The verified packet explicitly lists an unseen-class
EVT tail shift among the premises it does not support. It contains no GPD/PWM
goodness-of-fit result, tail-index stability across held-out identities,
gallery-multiplicity error decomposition, or causal response to tail pressure.

The proposal derives `N_train/C_train` ratios from dataset sizes, then simply
assigns gains that follow their ordering. A ratio is not a measured error
mechanism. The `+0.014/+0.012/+0.005` CUB/Cars/SOP forecasts have no empirical
or identified causal bridge.

## Gate 2: fourth return of the same occupied line

PORTAL is the same mechanism as three earlier neutral proposals:

- pass 9 EVPC fitted a GPD negative-score tail and attempted a deployment-size
  partition correction; its own proposed repair was a return level;
- pass 14 RLM fitted a differentiable POT/GPD tail and trained a margin against
  the extrapolated negative maximum; and
- pass 15 EGR-PFML used an XBM-style detached image queue, fitted an online GPD
  hard-negative tail, extrapolated a full-gallery maximum, and attached that
  return-level pressure to PFML.

Changing EGR-PFML's MLE/Newton fit to a shared closed-form PWM fit, replacing a
return-level hinge by `log(1 + N * survival)`, and adding a dimension-chosen
tail-index penalty do not change the supervision object: extrapolated
gallery-scale rare-negative pressure from a stale training-image queue.

Primary prior art independently occupies the pieces. WEINCE (arXiv
2606.00262) applies online anchor-wise EVT correction to top contrastive scores
during training. TriSim (CVPR 2026) fits a generalized Pareto similarity tail
and feeds its probabilities into a retrieval training loss. XBM (CVPR 2020)
supplies the detached FIFO embedding queue. Recall@k surrogates, top-k/CVaR
losses, and classical POT supply the remaining observed-tail and extrapolation
machinery. An explicit nominal `N` and a uniform-sphere anchor are distinctions
inside an occupied mechanism, not a new kind of supervision.

Primary sources:

- WEINCE: <https://arxiv.org/abs/2606.00262>
- TriSim: <https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf>
- XBM: <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html>

## Frozen mathematical and specification failures

1. **The advertised PFML arm is undefined.** The executable objective is a
   standalone instance loss and explicitly has no proxies. The forecast table
   then introduces “PORTAL replacing PFML's repulsive term” without defining
   which PFML attractive/sample-proxy terms, proxies, normalization, or recipe
   remain. That frontier-crossing arm is not the frozen objective.
2. **The neural-collapse proof uses nonexistent excesses.** If every negative
   similarity is the same atomic value, the empirical 95th-percentile threshold
   equals that value and every excess `s-u` is zero; the strict exceedance set
   is empty. It is not a nonzero common `y0`. The PWM estimator is therefore
   undefined, or the denominator guard produces a constant branch—not the
   claimed `xi -> -infinity` with a nonvanishing gradient.
3. **Constant collapse is stationary despite its high scalar loss.** The bulk
   fallback at equal scores is `sigmoid(0)=0.5`, not 1. More importantly, the
   tangent derivative of cosine between identical normalized descriptors is
   zero. Calling collapse a global maximum does not prove gradient escape.
4. **One pooled tail is mislabeled per-anchor risk.** Excesses are centered by
   anchor-specific thresholds but pooled into one `(xi,sigma)`. Heterogeneous
   anchor tails become a mixture that is generally not GPD; substituting the
   pooled parameters into `F_i` does not estimate each anchor's survival law.
5. **The uniform-sphere tail anchor is not identified for a learned semantic
   embedding.** The reverse-Weibull index `-2/(d-1)` is an asymptotic fact for
   independent uniform points, not for class-structured, mined, queue-stale
   descriptors. Enforcing it imports an unmeasured geometry prior and may erase
   useful anisotropy.
6. **Its own endpoint arithmetic contradicts bounded cosine support.** For a
   negative GPD shape the excess endpoint is `-sigma/xi` and must not exceed
   the physical `1-u <= 2`. The proposal celebrates an implied endpoint about
   12.8, which is outside cosine support and therefore evidence of a
   misspecified scale/index pair, not harmless near-exponential behavior.
7. **Nominal gallery size is often gradient-negligible.** In the proposal's
   own `xi -> 0`, `Lambda >> 1` reduction, `log(Np)` is additive and the
   gradient is independent of `N`. Thus the regime advertised as most relevant
   collapses toward an adaptive-temperature margin; the `N`-sweep control is
   likely to test saturation rather than deployment-scale extrapolation.
8. **The EVT assumptions fail on the training stream.** Queue entries are
   stale, correlated, class-sampler-dependent, and produced by changing
   encoders. Negatives are neither stationary nor exchangeable draws from an
   unseen-gallery law. Pooling 24,000 dependent excesses does not restore the
   missing effective sample size or tail transport.
9. **The optimizer claim is false.** AdamW's decoupled weight decay does not
   make arbitrary loss rescaling operationally invariant in the presence of
   epsilon, moment transients, multiple terms, clipping/guards, and different
   parameter groups. The proposal correctly notes normalized-head scale can be
   operational, then asserts it away.
10. **Controls do not isolate the occupied alternatives.** C3 uses
    `bar_sigma` measured from the deciding PORTAL run and is therefore not a
    prospective independent constant control. No control replaces the GPD
    extrapolation with an empirical top-k/CVaR loss at matched queue and
    gradient mass. The 5x5 tuned baseline budget is not matched to PORTAL's
    multiple fixed design choices and ablations.
11. **Forecasts are fabricated and cross-codebase arithmetic is not a claim.**
    The “my matched repro” and method rows are predictions, not measurements.
    A z-score against forecast standard deviations and a hypothesized
    reproduction gap cannot establish a frontier crossing; selecting among
    compositions and controls adds unaccounted multiplicity.

## Mechanism lesson

Four neutral passes have now converged on gallery-tail correction. That
recurrence says the story is cognitively attractive, not that it is supported.
The repository's verified packet deliberately excludes the needed EVT-transfer
premise, and the primary literature now occupies training-time EVT similarity
shaping. A future blind return-level variant should be killed immediately at
Gates 1 and 2 unless it first brings a prospectively verified causal tail
measurement and a genuinely different supervision object.
