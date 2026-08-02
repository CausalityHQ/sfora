# Post-RS@k measurement-space audit

Date: 2026-08-02. This audit was performed while the faithful Cars196 RS@k
reference was still running. Its result was unknown, and no incomplete metric
was used to generate a candidate.

## Question

Claude Opus was given the complete search protocol, the 200-candidate verdict
catalogue, the current search boundary, and the source-level RS@k audit. It was
asked for at most one post-run measurement computable from the eventual
checkpoint and labelled training pixels that could expose a supervision
relation absent from the catalogue. The prompt explicitly excluded renamed
measurements of hardness, rank, density, gradients, augmentation response,
fragmentation, modes, topology, proxy ownership, uncertainty, and acquisition
groups. It requested a measurement rather than an implementation.

## External answer and adjudication

The answer was `NONE`. Its defensible argument partitions any proposed
statistic `S(theta, D)` into three cases:

1. If it depends on the learned checkpoint, it is endogenous. Converting it to
   supervision must route through a refinement, coarsening, weighting,
   subsetting, mining, gating, curriculum, or regularisation operator. Those
   routes and all presently non-vacuous endogenous observables are already in
   the verdict catalogue.
2. If it depends only on labelled pixels, it estimates an edge, subgroup, or
   graded label inside the known class partition. Changing the estimator does
   not make the downstream clustering, pseudo-label, hierarchy, mining, or
   weighting operator novel.
3. A relation not in either case needs information outside one checkpoint and
   the labelled training pixels: another trajectory/model, metadata, or an
   external source. Those violate the locked information or deployment budget.

This is accepted as an **evidence-bounded negative**, not a mathematical
completeness theorem. It agrees with the prior independent operator audits and
does not adjudicate RS@k performance. In particular, a surprising final Cars
number cannot by itself turn an endogenous post-hoc statistic into new
supervision.

## Consequence

No candidate 201 is named and no candidate GPU work is authorized from the
pending checkpoint. The search can reopen only if a new experiment observes an
information-bearing relation outside the current catalogue, or if a
primary-source audit demonstrates that a supposedly occupied mechanism is
substantively different or unevaluated. The faithful RS@k run remains an
occupied reference reproduction and must be judged under its locked thresholds.
