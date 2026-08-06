# Pass 56 independent cold review: IFR

Reviewer: explicit Claude Opus fallback (shell consultation), completed after
reading only the frozen recovered proposal and review prompt. Verdict: **DEAD**.

## Earliest failures and decisive mechanisms

The reviewer treats the recovered stream limitation as an operator/specification
failure: the exact provider answer was not fully frozen, so the object is not
fully adjudicable. Independent scientific failures make the verdict dead even
if that retrieval defect is repaired.

### 1. The isothermality residual is only weighted `1 - CKA^2`

With the per-class scale `a_c` fitted and detached, each residual
`||S_c-a_c Sbar_-c||_F^2 / ||S_c||_F^2` is a normalized alignment penalty. It is
zero whenever `S_c = lambda_c S*` for a common operator `S*`, including a
rank-one operator. The term therefore does not enforce the claimed shared
nuisance subspace in a meaningful rank/direction sense and can be satisfied by
low-rank collapse. Leave-one-out does not change this degeneracy.

### 2. Low-rank estimates cannot support the claimed signal

`SR_c` has rank at most 3 and `SF_c` rank at most 5, while the descriptor has
512 dimensions. The proposed second-moment statistics are consequently noisy,
and the gradient can reduce the residual by collapsing onto a common
low-dimensional direction. The same objective has no rank floor, variance
floor, or direction-identifiability term; the claimed class-genericity effect
is not isolated.

### 3. Independent gate failures

The external Zhou et al. result is not a repository measurement, so Gate 1 is
degraded/failed under this project’s evidence rule. Gate 2/novelty is unresolved
by the recovered answer’s incomplete stream and requires a complete primary
source comparison against covariance-preserving augmentation, WCCN,
class-conditional invariance, coding-rate/VICReg, and leave-one-class-out
domain-generalization methods. The reviewer nevertheless marks the candidate
dead on the algebraic mechanism failure; the missing corrected In-Shop-first
screen is moot.

## Preserved components

The reviewer preserves the Gram-domain implementation idea, the separation of
same-image response from cross-image fluctuation, the held-out-training-class
probe, the useful C2/C6/C7/C11 controls, and the general warning that
class-specific invariance may not transfer. These are diagnostic or control
ideas, not a live method. No GPU work was recommended.
