# Pass 61 blind proposal — NONE

The blind proposer returned NONE. It corrected Pass 60’s claim that
equivariance is a remaining escape: pretrained equivariant conversion is
possible (equi-tuning), but an invariant readout is rank-reducing and cannot
provide the needed off-support selectivity; a bump-function argument rules out
the desired behavior from equivariance alone.

The reusable closure is a three-part rank/identifiability argument. Gradient
row-space truncation says off-support direction selectivity cannot be learned
from the supervised row space; the C−1 label-rank cap is binding on CUB/Cars;
and the three-source bound (labels, pixels/augmentations, initialization/
parameterization) explains why the prior families share one ceiling. Conjunction
features are saturated prior art. The proposer recommends only cheap D1–D3 rank
probes (~1 GPU-hour) as a possible falsifier before another blind pass. No
method forecast or GPU candidate run followed.
