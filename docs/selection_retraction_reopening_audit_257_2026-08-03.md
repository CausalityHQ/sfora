# Candidate reopening audit after selection-estimator retraction (257)

Date: 2026-08-03

Retracting the leave-neighbour pseudo-correction could invalidate a candidate death if
that death depended on the corrected sign or magnitude. Every occurrence in the method
verdict and candidate/preregistration files was inspected before generating another
arm.

## Result

No completed candidate reopens:

- Dual EMA failed its registered **raw** +0.24-point incremental threshold and absolute
  0.9048 threshold; observed raw gain over averaging was only +0.014 point.
- Weight averaging failed cross-dataset replication on its registered raw paired
  effects (+0.19, -0.12, +0.14 point). The invalid +0.203 pseudo-corrected estimate was
  explicitly non-deciding.
- RSPG missed its raw threshold by -5.72 points and self-erased its positive objective.
- ARCG collapsed before a completed benchmark; its pre-activation raw score was already
  below threshold and its objective fell nearly to zero.
- TIRD failed by -7.237 raw points. IPSR also failed its registered raw screen.
- The Cars RS@k and PFML documents use the old diagnostic for reference reporting, not
  for a novel-method survival decision.

Candidate 58's provenance invoked the old correction, so that motivation is withdrawn.
Its Gate-2 death remains independently valid because KD-as-gradient-variance-reduction
and control-variate/SVRG machinery occupy the proposed operator.

## Consequence

The analysis bug retracts several effect-size and ranking interpretations—most notably
that averaging became the strongest intervention—but it does not conceal a surviving
candidate. Reopening the search still requires a corrected artifact-derived measurement,
now expected from the running official SOP reference. This audit prevents both errors:
keeping a false negative merely because it is old, and reviving a method whose raw
falsifier had already fired.
