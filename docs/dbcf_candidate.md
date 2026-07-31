# Candidate 36: detailed-balance confusion flow (DBCF)

**Gate-1 death recorded 2026-07-31; no prior-art claim, implementation, or GPU
run.**

DBCF was a thermodynamics-inspired proposal to treat ordered class-to-class
nearest-neighbour errors as probability flux and penalize violations of detailed
balance. Its apparent provenance was 82.16% one-way connected error pairs and
67.25% mass-weighted directional imbalance at the exact In-Shop epoch-10
operating point.

That provenance fails under the required sparse-graph null. Permuting destinations
while preserving all source and receiver marginals gives only 0.43% reciprocal
directed cells versus 30.28% observed, and 99.43% imbalance versus 67.25% observed.
The learned geometry is dramatically *more* reciprocal than chance conditional on
its degree sequence. DBCF would regularize an absent defect and risks suppressing
legitimate class-density differences.

**Verdict: DEAD at Gate 1.** This is a statistical-specification failure, not a
negative model result. The null was cheaper and more decisive than a literature
search or GPU screen.
