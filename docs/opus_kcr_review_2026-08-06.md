# Cold Opus review — KCR (Pass 65)

**Decision: DEAD at Gates 1–2; no GPU.**

The proposed coefficients are exactly the published Regularized Nonlinear
Acceleration operator (Scieur, d’Aspremont & Bach, NeurIPS 2016): same residual
Gram matrix, ridge, affine normalization, and extrapolant. The paper explicitly
identifies the object with Anderson/Eddy–Mešina/MPE/RRE. Differentiability is
also occupied by neural DEQ solvers and unrolled Anderson; deployment inside a
CNN is a relocation, not a mechanism change; divergent/limit-cycle use is the
classical Shanks/Wynn rationale.

Algebraically, KCR is just seven scalar gains on the tied block’s own increments:
`x*=x0+sum_i c_i u_i`. Its final-iterate and uniform-average controls are exact
points of that parameterization, not independent causal controls. Zero
initialization makes `G=0`, so the coefficient gradient is zero to first order
and KCR equals the uniform average at initialization. The ridge is not
scale-invariant, Krylov Gram conditioning collapses near the linearized block,
and the metric loss supplies no force to maintain a divergent transient; the
block naturally contracts into a DEQ/fixed-point or ordinary deeper backbone.
The written state/difference index ranges are also under-specified.

Gate 1 independently fails because seen/unseen contraction mismatch was never a
repository measurement. The forecast (+.5 CUB over C1) is below a decisive
frontier and not separable at three seeds. No D0 or GPU run followed.

Primary source: Scieur et al., “Regularized Nonlinear Acceleration,” NeurIPS
2016 / arXiv:1606.04133; corroborating Neural DEQ Solvers (ICLR 2022),
Nonlinear Acceleration of CNNs (arXiv:1806.00370), and classical Shanks/Wynn
extrapolation.
