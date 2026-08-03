# Selection-bias estimator retraction (candidate-independent audit 254)

Date: 2026-08-03

## Claim under audit

`scripts/measure_selection_bias.py` treated the difference between the maximum test
R@1 and a leave-one-out mean of its neighbouring epochs as a best-over-training
“selection bonus.” Subtracting that gap was repeatedly called “selection-corrected”
R@1 and used to rank arms.

## Deterministic falsifiers

An estimator of noise-induced winner's-curse bias must return zero when observations
are noiseless. The existing estimator does not:

| noiseless history | reported maximum | neighbour estimate | invented gap |
| --- | ---: | ---: | ---: |
| flat | 0.8000 | 0.8000 | 0.00 pt |
| quadratic peak | 0.8000 | 0.7975 | +0.25 pt |
| rising endpoint | 0.8000 | 0.7850 | +1.50 pt |
| transition into plateau | 0.8000 | 0.7925 | +0.75 pt |

The gap therefore contains ordinary curve curvature and endpoint slope even with no
evaluation noise. Real training histories additionally contain autocorrelation and
nonstationarity. Excluding the selected observation prevents direct reuse of its noise,
but it does not make neighbouring epochs counterfactual observations of the selected
epoch's latent score.

## Consequences

1. Historical raw best-over-training scores are unaffected by this software issue, but
   remain test-selected and optimistic in the usual qualitative sense.
2. Historical “selection-corrected” scores are renamed local-neighbour diagnostics.
   They must not support effect sizes, rank reversals, or claims about method stability.
3. In particular, the claimed `pa_ema_avg_fast` corrected gain of +0.732 point and the
   conclusion that averaging was the strongest single intervention are retracted. Its
   raw CUB and In-Shop observations, including failure to replicate, remain valid only
   subject to their separate recipe-fidelity audits.
4. A defensible benchmark comparison must use a checkpoint epoch fixed in advance or
   selected without the evaluation set (nested class-disjoint validation), then report
   the untouched evaluation score. A final-epoch artifact is also descriptive and
   independent of test-epoch maximisation, though not automatically optimal.

## Process lesson

A simulation showing that an estimator reacts to noise is not a calibration test. A
zero-noise negative control is mandatory before interpreting an estimated bias. The
existing unit test even documented a nonzero gap on a noiseless monotone curve but
mistook “does not scale with noise” for “does not bias the correction.” Historical
claims can fail through analysis code even when the training and retrieval code are
correct.
