# Pass 55 independent cold review: XCR

Reviewer job: `422b6f1c2e984728` (Fable exhausted credits; same job completed
under Claude Opus). The reviewer returned **DEAD**. The full provider stream is
not retained by the MCP after completion; this file preserves its mechanism-
level verdict and calculations from the completed partial answer.

## Verdict

Earliest failed gate: Gate 1 (no provenance measurement in the frozen object).
The decisive fatal defect is a Gate-4 degeneracy, so this is not repairable by
running the frozen method.

For CUB (`B=180`, `K=6`, `B_neg=174`, `M=5864`, `c=Mk/B_neg=539.1`), hold the
positive radius at `rho+=0.70`. Start with the 24 nearest negative radii spread
roughly over `[0.90,1.10]` (`u=1.10`, `d≈8.3`, `A=log(rho+/u)=-0.452`, `T=-3.75`,
`N_tail≈12.68`, loss `log(1+N)≈2.62`). Compress every one of those negatives
to `0.90`—each negative moves closer or stays no farther, while the positive
does not move. The Hill spacing tends to zero, the clamp returns `d=512`, and
`T≈-128.7`; the tail estimate collapses to about `5×10^-54` and the loss falls
to near zero (roughly 0.24–0.48 with the gate residual). Thus the objective
strictly rewards worsening retrieval geometry. The proposal’s own derivative
`∂T/∂log u=-d+d²|A|` has the wrong-sign inward direction whenever `d|A|>1`.

The reviewer rejects the proposal’s defenses: D3 mistakes vanishing gradient
at a minimum for proof of no runaway, D4 confuses mutual equidistance with
many points on a sphere around one anchor, and D1 only handles loss of contrast,
not collapse of radial spread.

## Additional independent findings

- The frozen proposal contains forecasts only; it contains no persisted
  measurement linking negative-only LID/extrapolated risk to disjoint-identity
  R@1. Gate 1 is therefore at best unresolved before the fatal degeneracy.
- The Hill/Weissman algebra and displayed gradient rows are mostly correct,
  but the tail fraction `k/B_neg` is anchored at `r_(k)` while `u` is the mean
  of `r_(k+1:k+m)`, so the two branches disagree at the seam (about 28% in the
  reviewer’s CUB calculation). The iid equality/Poisson approximation is not
  valid for clustered galleries.
- Effective independent negatives are only a few identities, not `k=16` iid
  samples; estimator variance and exponential bias are consequently much
  larger than the proposal’s `d/sqrt(k)` estimate. The balanced sampler also
  measures within-identity clumping rather than deployment tails.
- XCR’s exponent gradient is the same LID carrier as LDReg (ICLR 2024) with a
  data-dependent hardness weight. The proposed constant-beta C3 cannot isolate
  risk calibration from generic per-anchor hard-negative reweighting.
- C1 stop-gradient removes both the LID exponent path and the nearest-negative
  path, not only the claimed exponent path. The `gamma→1` limit does not equal
  stop-gradient. F2 is therefore not a clean mechanism test.
- The frozen forecasts are not a decisive matched-frontier crossing (0.9–1.1σ
  against PFML) and do not provide the required corrected In-Shop raw/final
  results or second-dataset evidence.

## Preserved subcomponents

The reviewer found the union bound `M_neg F_q(rho+)`, the D1 contrast argument,
the five local derivative calculations, and differentiable Hill/Weissman fitting
as a potentially unoccupied technique. Those are reusable research ingredients,
not evidence that XCR itself is viable.
