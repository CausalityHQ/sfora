# Pass 148 — In-Shop Gate-1 signal diagnostic: label semantics (2026-08-08)

## Result

The registered class-disjoint residual-signal diagnostic cannot be evaluated on
the available In-Shop training pack as written.  This is a Gate-0 data-semantic
finding, not a candidate result and not a GPU screen.

The exported epoch-10 operating pack contains 25,882 images, 3,997 unique
labels, and 3,997 unique product directories (`id_*`).  The mapping is exactly
one product directory per label: every label maps to one product and no label
maps to more than one product.  Therefore, in this pack the metric-learning
class is the product identity.  There is no independent within-class identity
or acquisition-independent class variable from which to construct the
registered outcome `Y` while controlling class.

An attempted low-level pixel diagnostic (48x48 RGB histograms plus first-order
edge statistics, 12,000-image sample) consequently produced zero valid
same-label/different-identity negative pairs.  Treating the product directory
as a second identity would make all same-label pairs the same identity and
would test a tautology; treating category path components as class would be a
new, unregistered target and would not match the corrected protocol.

## Consequence

No pixel-derived training signal, AUC, or method forecast is authorized from
this pack.  The missing measurement must first specify a genuine hierarchy
(for example category versus product identity) and obtain that metadata for
the exact train split, then rerun the cross-fitted held-out-identity diagnostic
with its preregistered thresholds.  This prevents inventing a positive or a
negative from a label/identity mismatch.  The active Pass133 GPU run is
unrelated and remains unchanged.
