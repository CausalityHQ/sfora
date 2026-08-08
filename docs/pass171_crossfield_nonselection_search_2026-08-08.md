# Pass 171 — cross-field non-selection mechanisms (NONE before GPU)

An independent subagent searched for a train-time mechanism that changes
supervision without selecting or reweighting pairs, using the In-Shop
between:local error ratio `3.05–3.22`, the `-0.04968` unseen-minus-seen gap,
and CMR's CPU failure as constraints.

Candidate families included symplectic/Hamiltonian volume-preserving heads,
replica/energy dynamics, variance-expansion heads, relational supervision,
virtual classes, and principal-direction augmentation. The symplectic head was
the only apparently unoccupied architecture, but Pass 66 already identifies
symplectic/coupling flows as a prior-art/degeneracy family; no repaired
mechanism-equivalence distinction was found. ESA, SEE, DRML, Deep Metric
Learning Beyond Binary Supervision, MemVir, and Potential-Field DML occupy the
other families.

Decision: `NONE` before GPU. This is scoped to the searched non-selection
families and is not evidence that all future architectures are exhausted. No
implementation or GPU run occurred.
