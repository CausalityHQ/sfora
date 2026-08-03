# Claude APGC proposal: Gate-2 audit (candidate 309)

Claude proposed “Adaptive Positive Gradient Curvature” (APGC): retain every
Proxy Anchor positive but multiply each positive contribution by a sigmoid of its
current embedding cosine, increasing weight on distant positives.

## Gate 1

The proposal points to three repository facts: corrected In-Shop Proxy Anchor
0.916303 raw best / 0.913701 final, corrected SOP 0.791098 / 0.790007, and
RSPG's graph collapse from 0.0895 to 0.0144 density.  None measures that a
continuous positive-weight function would improve retrieval.  The SOP
fragmentation association (+2.195 class-balanced points) is observational and
does not identify a causal positive weighting rule.  Thus APGC has no exact
effect-size provenance and would already fail the registered provenance gate.

## Gate 2: DEAD

The mechanism-level sentence is “multiply positive loss by a function of current
positive hardness while retaining all positives.”  That is ordinary hard-example
weighting / focal-style loss scaling, not a new supervision object.  Proxy Anchor
already weights sample gradients by relative proxy hardness; Multi-Similarity and
Generalized Pair Weighting explicitly reweight pair contributions by similarity;
focal loss is the general difficulty-modulated loss precedent.  A sigmoid and the
name “curvature” do not change that mechanism.  It is also explicitly excluded
by the project search protocol as generic mining or weighting.

No implementation, CPU diagnostic, preregistration, or GPU run follows.  APGC is
recorded because it was the strongest remaining ownership-preserving suggestion,
and its failure leaves no live candidate in the current training-pixels/class-label
search space.

