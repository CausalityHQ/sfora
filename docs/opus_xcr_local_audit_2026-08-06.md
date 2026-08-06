# Pass 55 local evidence-aware audit: XCR

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_xcr_proposal_pass55_2026-08-06.md`  
Frozen proposal SHA-256: `1092316716077b0c6984dafd99a692d0ca531534c0d927e0108506e113450fa9`

This audit is independent of the proposer and was written before the mandatory
cold review result.

## Provisional verdict

**DEAD at Gates 1 and 2; no GPU work is authorized.** XCR is a mathematically
interesting gallery-risk surrogate, but its load-bearing premise has already
been measured adversely in this repository and its supervision object is a
recurrence of the fitted negative-tail family.

## Gate 1 — provenance is adverse, not supportive

XCR requires that a negative-only local intrinsic-dimension estimate from about
16 of roughly 176 class-balanced in-batch negatives predicts the unseen-gallery
collision probability after extrapolation by the training image count `M`.
The verified evidence packet contains no positive measurement of that transport.

The closest prospective repository measurements point the other way. Pass 31
XTail found that fitted GPD shape added only `0.0087` incremental R-squared
beyond threshold and mean excess; a tuned constant predicted the deep quantile
better (`0.0918` versus `0.1575` RMSE), and fitted-shape noise consumed about
67–86% of the full batch-to-SOP depth signal. The independent TERL/GEVS audits
record the same result. XCR changes the estimator from GPD shape to Hill LID,
but supplies no new artifact demonstrating that this different estimate
predicts disjoint-identity R@1. Its forecasts are explicitly hypothetical.

The proposer’s claim that gallery-scale error is an exact function of
`M_neg F_q(rho_+)` is only exact after conditioning on an unknown deployment
distribution. The displayed `1-exp(-N)` is an approximation for iid negatives,
not a measured identity-mixture law. Training-set `M` is not the query/gallery
exposure (especially for In-Shop), and the balanced sampler is neither iid nor
representative of the unseen gallery. Thus the required causal measurement is
absent and the best available evidence is adverse.

## Gate 2 — occupied mechanism family

XCR’s training action is: estimate a local tail/LID statistic of observed
negative similarities, extrapolate it to a nominal gallery depth, and optimize
the resulting risk. This is the same supervision referent as the repository’s
EVPC, RLM, EGR-PFML, PORTAL, PORT, XTail, POTER, CFEV, and GEVS proposals.
Changing the tail estimator, shrinkage, and `M` coupling does not change what
supervision exists.

Primary neighbours occupy the same or an immediately adjacent mechanism:

- LDReg (ICLR 2024) differentiates local intrinsic-dimensionality statistics
  through the representation to prevent dimensional collapse.
- WEINCE (ICML 2026) applies online extreme-value corrections to bounded
  contrastive scores.
- TriSim (CVPR 2026) fits generalized-Pareto similarity tails in retrieval
  training.
- Recall@k surrogate (CVPR 2022), Smooth-AP/FastAP, ranked-list/top-k/CVaR
  objectives, and XBM occupy train-time operating-depth pressure.
- Coding-rate and uniformity/anti-collapse objectives occupy global dimension
  expansion controls.

XCR’s distinction—LID appears as the exponent of an extrapolated collision
  bound rather than as a free `-beta log LID` penalty—is a wrapper-level
  distinction until a matched control shows that the coupling, not ordinary
  LID maximization or hard-negative weighting, does the work. That evidence
  cannot be obtained before Gate 2; the proposal therefore does not clear the
  prior-art gate.

## Mathematical and protocol defects

1. The estimator uses order statistics from only `k=16` dependent negatives;
   the Hill power-law model is not identified for a class-mixture gallery and
   its variance is large. The `M` multiplier creates thousands-fold
   extrapolation without a transport measurement.
2. `rho_+` is the nearest positive, so the method inherits best-positive
   selection and can neglect other within-class modes. The out-of-range branch
   is a smoothed hard-negative count, making a large part of the method an
   adaptive mining term.
3. At identical normalized descriptors, cosine has zero first-order tangent
   derivative. A positive `log(1+N)` value therefore does not prove the claimed
   first-order escape from collapse.
4. The proposal’s claim that all DML losses cannot represent probabilities
   below `1/B_neg` confuses a batch estimator with the function class of the
   loss; ranking and proxy losses can still impose continuous penalties whose
   effects are not probability estimates.
5. The advertised 1.00–1.02x cost omits sorting/top-k synchronization and
   autograd through selected order statistics; it needs measurement, not FLOP
   division by a convolution estimate.
6. The frozen screen is specified as five seeds and forecasts, but no corrected
   In-Shop paired artifact exists for XCR. The protocol requires In-Shop first,
   raw plus independently selected/final metrics, out-of-sample confirmation,
   and second-dataset replication only after Gates 1–3.
7. Even the frozen forecasts do not establish the standing objective: B2+XCR
   is forecast at 0.740/0.933/0.834, only 0.9–1.1 standard deviations above
   PFML’s published CUB/Cars/SOP values, while B1+XCR is below the frontier.
   No decisive frontier crossing is predicted.

## What is worth preserving

The gallery-risk formulation is a useful diagnostic question, and the proposed
controls (frozen exponent, LDReg transplant, tuned mining, `M`/`k` ablations,
and gallery-size sensitivity) would be appropriate if a future corrected
measurement first established tail transport. The deployment constraint is
legal and unchanged. None of these merits authorize GPU work for this frozen
candidate.

## Required next step

Reconcile this audit with the mandatory cold review. If the reviewer agrees,
append XCR’s mechanism and the adverse tail-evidence lesson to
`docs/method_search_verdict.md`, commit/push, and start the next neutral blind
proposer pass. If it finds a substantive disagreement, resolve it without
silently repairing XCR; a repair would be a new proposal.
