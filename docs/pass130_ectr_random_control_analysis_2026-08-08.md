# Pass 130 — corrected ECTR random control closeout (2026-08-08)

The corrected random-area control for the In-Shop ECTR screen completed all 60
epochs (8,580 optimizer steps) on the original DGX process.  The process and
its artifact were preserved; no overlapping run was started.

The report artifact omitted `test_recall_history`, so the generic selection
diagnostic correctly refused to invent a curve.  The original immutable log
contains the complete 60-epoch history and was used for the descriptive
local-neighbour calculation below.  This is the same statistic used by
`scripts/measure_selection_bias.py`: it is a selection diagnostic, not an
identified unbiased correction.

* best-over-training raw R@1: **0.8738** (epoch 10)
* local-neighbour trend at the selected epoch: **0.85915**
* local peak gap: **+1.465 pt**
* final epoch R@1: **0.6564**

The corrected random control therefore does not supply evidence for ECTR.  Its
artifact-format defect is itself recorded: reports must persist the recall
history whenever a selection diagnostic is required.  The next GPU gate is
CEA, but only after this closeout is recorded and the matched-control queue is
explicitly authorized.
