# Pass 86 — label-only search boundary (2026-08-07)

## Verdict: no executable candidate survives the stated constraints

The only live corrected Gate-1 signal is the four-seed cardinality-matched
positive-transfer deficit: unseen-minus-seen nearest-positive similarity is
`-0.04968`. It measures how much identity-induced contraction fails to transfer,
but contains no observable telling an optimizer which within-identity
differences are transferable evidence and which are nuisance.

The recent searches exhausted architecture, optimizer, weight-space, scalar
loss, graph, higher-order, coding, exchange, and topological routes. Any
label-only action therefore reduces to one of three cases:

1. refine identity equivalence from current geometry (mining, weighting,
   pair/group/order/graph supervision);
2. change geometry without a new referent (regularization, projection, coding,
   or an unmotivated architectural prior); or
3. introduce a new referent (cross-image exchange, attributes, or human
   judgments).

The first two are occupied or already falsified; the third violates the
label-only benchmark scope. PEBH is the strongest recent attempt and failed
its CPU gate: +0.004747 positive gain, +0.008855 nearest-foreign increase,
and a leave-one-out violation.

The precise assumption that must be relaxed to reopen an executable search is:

> Identity equality is the only permitted supervision referent.

Allowing one fixed exogenous within-identity substitutability relation would
make the multi-threshold proposal in Pass 85 testable. Under the current
assumption, no honest preregistration or GPU run remains.
