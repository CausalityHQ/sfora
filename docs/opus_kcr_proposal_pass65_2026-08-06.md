# Pass 65 blind proposal: Krylov Continuation Readout (KCR)

KCR changes the extrapolation operator itself rather than adding a detached
loss. It appends one zero-initialized, GroupNorm residual bottleneck block to a
pretrained ResNet-50 stage-4 map and unrolls it eight times with tied weights:
`x_{k+1}=x_k+g_theta(x_k)`. Instead of reading the final iterate, form residual
Gram matrix `G_ij=<u_i,u_j>`, `u_i=x_{i+1}-x_i`, and solve the ridged affine
minimum-residual coefficients

`gamma=(G+lambda_bar I)^(-1)1 / (1^T(G+lambda_bar I)^(-1)1)`,
`x*=sum_j gamma_j x_j`, `e=normalize(W GAP(x*))`.

This is differentiable through both unrolled states and the coefficient solve;
the coefficient gradient lets training shape the transient spectrum. It is a
parameter-free forward continuation readout, not post-hoc averaging: negative
gamma values analytically continue divergent/oscillatory residual dynamics.
Initialize the last bottleneck convolution to zero (exact baseline at step 0),
use GroupNorm, 8 steps at evaluation, and randomize training horizon 3–7.
Everything else follows the Lane-A recipe.

## Causal claim and controls

Seen identities are hypothesized to form fast residual attractors while unseen
identities have slow/divergent transients. KCR removes phase dependence by
continuing the tied recurrence, potentially correcting unseen retrieval without
quotienting dimensions. Required controls: baseline C0; same block/final
iterate C1; uniform depth average C2; fixed-point DEQ C3; untied depth C4;
stop-gradient coefficients C5; and no horizon randomization C6. Report KCR−C1
as the mechanism result. Falsifiers include no seen/unseen contraction-ratio
gap after the cheap gate, no divergent unseen regime, KCR−C1 <=.2 CUB, large
lambda reducing to averaging, coefficients in the simplex (no extrapolation),
horizon dependence, equivalence to DEQ, or effective-rank collapse.

## Frozen forecasts and cost

Lane A three-seed forecasts versus C1: CUB .739 (+.5), Cars .930 (+.3), SOP
.830 (+.1), with only ~20% chance of a lane-relevant +1pt CUB crossing. A cheap
5-epoch D0 gate (~2 GPU-hours) measures seen/unseen residual contraction; abort
if the gap is <.02, both ratios <.9, or unseen divergent probability <5%.
The block adds ~1.64M parameters, ~16% compute, and zero deployment data
dependencies; GroupNorm avoids BatchNorm coupling.

## Prior-art claim to audit

Nearest works are Anderson/RRE/MPE/epsilon acceleration, DEQ/Anderson solvers,
Richardson extrapolation, recurrent logical-extrapolation networks, and
rational/Padé activations. The distinction is a finite, unrolled,
sample-specific, differentiable continuation operator used as the deployed
metric descriptor, intentionally useful in divergent/limit-cycle regimes;
classical methods solve an assumed fixed point or accelerate an optimizer and
do not train a metric readout through the coefficient path.

No GPU run is authorized before the cold review and D0 gate.
