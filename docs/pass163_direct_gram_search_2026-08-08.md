# Pass 163 — direct Gram-geometry search (NONE before GPU)

The repaired measurement-conditioned lane used the verified unseen-minus-seen
nearest-positive gap (`-0.04968`) and required any candidate to act directly on
the normalized 512-D Gram matrix.  The strongest concrete construction was
class-conditional instance discrimination added to Proxy Anchor: align two
views of each instance while keeping distinct same-identity instances apart.

Primary-art review found the training object occupied by Chen et al. (ICML
2022) and Lee et al. (AISTATS 2025), with compatible-positive selection in Easy
Positive mining and within-class variation transfer in DVML.  The measurement
does not identify which variation should be retained; assigning a gain forecast
would be invented precision.  No candidate-specific CPU falsifier or
preregistration is defensible, so no implementation or GPU run was authorized.
