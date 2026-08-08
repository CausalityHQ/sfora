# Pass 145 — class-disjoint signal review (2026-08-08)

## Verdict

**NONE.** No cheap Gate-1 diagnostic currently has both (1) a class-disjoint
residual information channel and (2) a gradient route outside the occupied
mechanism families. No new GPU arm is justified.

## The one missing measurement

For a prospectively declared pixel-derived signal `S`, measure its incremental
held-out-identity predictive value:

`Delta AUC = AUC(P(Y=1 | C,S)) - AUC(P(Y=1 | C))`.

`Y=1` means a query's nearest support image has the same identity; query and
support must have different acquisition tokens. `C` contains cosine
positive/rival distances, proxy margin, acquisition token, augmentation
dispersion, target-excluded rival signature, and Proxy-Anchor gradient
coefficients. `S` must be fixed before observing `Y`, derived only from training
pixels, and fitted without evaluated identities.

Protocol: assign identities to five fixed-hash outer folds; fit `S`, nuisance
models, and hyperparameters only on identity-disjoint inner folds; apply frozen
models to the fifth fold; pool outer predictions; repeat on two independently
trained packs; and bootstrap whole identities stratified by acquisition
availability. Controls are identity-label permutation, `S` permutation within
acquisition/class-size strata, a cosine-budget-matched control, and
cross-acquisition-only evaluation.

Prospective pass thresholds are all required: pooled `Delta AUC >= 0.05`, an
identity-bootstrap 95% lower bound above zero, positive delta in at least four
of five folds, cross-acquisition-only delta `>= 0.05`, cross-seed rank
reliability Spearman `>= 0.50`, and placebo deltas with absolute value `< 0.01`.
Anything weaker is descriptive, not provenance.

## Why no method follows yet

Every presently expressible route for a passing `S` collapses to occupied
machinery: weighting/gating/sampling is pair mining or weighting (DML-ALA);
estimating which examples improve held-out performance is Datamodels, DVRL, or
In-Run Data Shapley; matching an `S`-derived representation is distillation or
augmentation invariance; support-set targets are contextual/graph or episodic
learning; and parameter-only routing is optimizer/preconditioning or
architecture regularisation. Treating conditional information as the objective
is information-theoretic feature selection rather than a new referent (PIDF).

Thus a passing measurement would reopen referent search but would not authorize
training until its exact Jacobian is shown not to reduce to one of these routes.
The cold Fable consultation and automatic Claude fallback were unavailable at
their weekly limit; Sol was used, and no cross-model agreement is claimed.
