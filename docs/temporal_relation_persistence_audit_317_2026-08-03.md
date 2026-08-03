# Candidate 317: temporal relation-persistence supervision

The motivation was that an image's nearest same-class neighbor may be a more
reliable positive when the relation persists from the epoch-10 operating point
to the final state. On corrected In-Shop seed-0 artifacts, the nearest
same-class identity persisted for **64.99%** of samples. Final leave-one-out
error was **0.4518%** for persistent relations and **0.5849%** for changing
relations, only a **0.133-point** single-trajectory difference.

This is too weak and selection-dependent to establish a causal Gate-1 signal or
justify GPU work. Temporal consistency is also established adjacent work.
**Dead before implementation; no GPU.**
