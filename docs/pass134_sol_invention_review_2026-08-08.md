# Pass 134 — Codex Sol invention review (2026-08-08)

This was a read-only, adversarial fallback review while the sequential CIS
In-Shop controller was running.  The reviewer read the search protocol,
method ledger, CIS/SRC preregistrations, and the CEA screen.  It was asked for
three train-time methods spanning architecture, activation, optimizer,
geometry, or supervision, with exact objects, repository provenance, primary
prior-art collisions, CPU falsifiers, and preregisterable screens.

## Verdict: no candidate survives Gate 2

Sol returned `NONE`.  The repository measurements motivate further questions,
but the strongest proposals collapse into existing mechanisms before GPU work:

1. **Counterfactual batch-composition invariance.**  Penalising the change in
   an image descriptor when same-class batchmates are replaced would measure
   the observed batch/context sensitivity.  Its operative mechanism is still
   making outputs invariant to batch composition, which is the purpose of
   Batch Renormalization (Ioffe, 2017), not a defensible new DML mechanism.

2. **Order-symmetrized class-gradient updates.**  Averaging the two update
   compositions for disjoint class shards cancels a gradient commutator.  It
   is gradient-conflict manipulation plus parameter/update averaging, covered
   in mechanism by CAGrad/gradient-surgery methods and by the ledger's closed
   averaging and game-dynamics families.

3. **Class-jackknife predictive encoding.**  Predicting an image descriptor
   from the remaining images of its class and training a self-only deployment
   branch is cross-image representation transfer.  That mechanism is already
   present in Wang et al., *Joint Learning of Single-Image and Cross-Image
   Representations* (CVPR 2016), alongside the ledger's closed peer-feature
   exchange and masked set-completion families.

No numeric In-Shop prediction is registered for these proposals: a passing
CPU diagnostic would not repair their Gate-2 collisions.  CIS/SRC coalition
variants and CEA are not repackaged as new candidates.  The next candidate
must come from a genuinely different supervision object or representation
mechanism, and must be audited against primary literature before GPU use.

## Process note

Fable and its Claude fallback were unavailable at the time of review because
the devbox weekly limit was exhausted.  Codex Sol was used as the requested
fallback; this memo reports its result as an independent advisory review, not
as a benchmark claim.

