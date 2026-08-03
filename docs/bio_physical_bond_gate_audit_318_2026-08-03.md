# Candidate 318: bio-physical bond objective

The repository contains an unqueued `bio_physical_bond` objective combining a
Lennard-Jones shell around proxies, Proxy-Anchor-style softplus terms, and a
coding-rate proxy niche term. Its stated motivation is preventing class
collapse. Corrected In-Shop geometry does not show that premise: within-class
pair cosine ranges from **0.4031** to **0.99998** (q10 **0.6030**, median
**0.7701**), and proxy-margin failures are only **0.108%**. There is no measured
collapse or proxy-niche failure to motivate a new intervention.

**Gate 1: dead.** No GPU was used.
