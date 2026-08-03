# SOP structural-operator audit 235

Date: 2026-08-03. Conducted while the preregistered corrected SOP baseline was
running; no result from that run was available.

An independent Claude audit tested whether official-training embeddings could
authorize a new supervision operator. Its strongest three proposals were:

1. ecology-inspired rarefaction of per-class dispersion, which reduces to a
   per-class scalar and shrinkage of proxy targets;
2. statistical-physics packing diagnostics for 11,318 classes in 512 dimensions,
   which feed occupied uniformity/spread and spectral regularizers; and
3. detailed-balance/hubness diagnostics on the kNN graph, whose corrections are
   local scaling, CSLS, or neighbor mining.

All die at operator identification before a diagnostic: weighting/shrinkage,
proxy geometry, and similarity correction are occupied mechanisms. SOP adds one
observed channel, `super_class_id`, but its uses are likewise occupied by HPL,
HIER, Divide-and-Conquer embeddings, category-conditioned mining, or adversarial
invariance. Saved embeddings can reveal behavior in a new class-semantic regime,
but they cannot alone create an observed relation.

This closes embedding-derived *supervision* for SOP. It does not close training
dynamics. SOP has an unusually separated two-timescale system: 11,318 learned
proxies use a 100x learning-rate multiplier while sharing a slowly updated
backbone. A separate Gate-2 audit will therefore test control-theoretic
fast-variable elimination, proxy lookahead, and block-coordinate response as a
possible faster optimizer rather than claiming another supervision signal.
