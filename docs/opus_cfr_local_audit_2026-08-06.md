# Pass 25 local audit: Clifford Fiber Rectification (CFR)

Date: 2026-08-06. The frozen proposal is
`docs/opus_cfr_proposal_pass25_2026-08-06.md`. This audit was written after the
proposal was frozen and while the separate cold review was still running. No
implementation, diagnostic, preregistration, or GPU work is authorized.

## Verdict

**DEAD at Gate 1.** The proposal's load-bearing empirical premise is that a
class-exogenous nuisance coordinate transfers across identities: within-class
variation should be routable through one fixed family of identity-independent
directions, and applying the same nuisance state to different identities should
produce the same latent action. The closest prospective repository measurement
is adverse. Candidate 225 estimated a pooled within-class subspace on one half
of corrected In-Shop training identities and tested whether it was
nuisance-heavy but identity-light on the disjoint half. At the preregistered
`k=32` operating point, `rho = captured-within / captured-between` was
**0.9312, 0.9287, and 0.9345** over seeds 0--2, below the locked `1.15`
falsifier in all three seeds. The source-fold subspace captured about 35--37%
of target within-class variance but 38--40% of target between-class variance.

That result does not prove that no nonlinear nuisance action exists. It does
mean this repository supplies no positive provenance for CFR's fixed
class-exogenous frame, while supplying a prospective contradiction to its
linear prerequisite. Candidate 226 already rejected the repair “make the
subspace location-dependent”: a failed global premise does not positively
motivate an unmeasured tangent field. CFR gives no new repository measurement
linking a shared action, same-augmentation cross-identity cosine inflation, a
16-dimensional Clifford orbit, or fiber confinement to corrected retrieval
errors. Its `+1.8/+2.2` deltas are therefore forecasts without an empirical
bridge.

Primary artifact:
`docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md`.

## Independent Gate-2 algebra

Gate 1 is already terminal under `docs/search_protocol.md`. The independent
review nevertheless requires the frozen training object to be checked. It has
several mechanism-level failures that cannot be repaired in place.

### 1. The advertised `S^16` fiber is not a fiber or group orbit

The Hurwitz--Radon count `rho(512)-1 = 17` and the basic Clifford identities are
plausible. They provide 17 pointwise orthonormal tangent vector fields. They do
**not** make

```
F(u) = S^511 intersect span{u, A_1u, ..., A_16u}
```

a fibration or an orbit of the rotations in the proposal. Let
`v = (cos(theta) I + sin(theta) A_a)u`, where
`A_a = sum a_k A_k`. Then `A_j v` contains terms `A_j A_k u`. For `j != k`
these bivector terms are generically outside
`span{u,A_1u,...,A_16u}`. Consequently `F(v) != F(u)` in general: moving inside
the claimed fiber changes the claimed fiber. Likewise the set
`{cos(theta)I + sin(theta)A_a}` is not closed under multiplication; products
contain `A_jA_k`. Its group closure has the much larger Lie algebra spanned by
the grade-1 and grade-2 elements, which is why the proposal itself introduces
136 matrices in `Omega`.

The primary mathematical source cited by the proposer describes the familiar
fiberwise-homogeneous Hopf families with great fibers of dimensions 1, 3 and 7,
not a great-`S^16` fibration of `S^511`:
<https://arxiv.org/abs/1407.4549>. A maximal family of independent vector
fields is not a maximal great-sphere fibration.

### 2. The cancellation theorem is true but causally vacuous

`R(a,theta)` is orthogonal, so
`(R mu_i)^T(R mu_j) = mu_i^T mu_j`. This holds for **every** shared orthogonal
matrix and does not depend on a Clifford family. The loss never observes a
nuisance state, never pairs two identities known to share one, and never
identifies common `(a,theta)` coordinates across identities. `Phi` separately
places each residual near an anchor-dependent tangent span. It neither implies
`z=R(a,theta)mu` nor makes two visually corresponding nuisances use the same
`a` or `theta`. `Omega` penalizes cross-identity bilinear responses; it does not
create correspondence. The theorem therefore proves a property of a latent
factorization the objective does not identify.

This is precisely why established equivariant methods require transformation
information. Class--pose decomposition learns an invariant class factor and a
group pose factor from **relative symmetry supervision**, not class labels
alone: <https://proceedings.mlr.press/v206/marchetti23b.html>. CARE enforces
input-transform/latent-transform correspondence; the repository already
rejected the same-controlled-augmentation cross-instance operator as occupied
and empirically unsupported (`docs/adversarial_candidate_audit_211_213_2026-08-02.md`).

### 3. Collapse is stationary, not excluded

At `w_i=0`, every coefficient `c_ik=0`. The active hinge has positive value,
but its gradient through `sum c_ik^2` is exactly zero because
`d(c^2)/dc = 2c`. Detaching and flooring `D_i` does not change that. Thus the
collapsed point is stationary for `Phi`, while Proxy Anchor continues to pull
samples toward their proxy. The statement `partial Phi / partial ||w|| == 0`
is also internally inconsistent with the next claim that `Phi` increases as
the residual collapses: away from zero, the numerator supplies a radial
gradient; exactly at zero, it supplies no escape gradient. This is not an
anti-collapse construction.

The seven-directions counting bound is a value bound, not a guarantee that
optimization activates seven directions. The `min` cap gives zero gradient to
an already saturated coefficient, but all inactive coefficients can remain at
zero-gradient zero.

### 4. The `beta` headroom arithmetic is wrong and the registered maximum is
miscalibrated

The proposal's own bound is

```
sin(theta) sqrt(m) beta > Delta/2.
```

At `theta=30 degrees`, `m=16`, and `Delta=0.9`, this becomes
`2 beta > 0.45`, hence **`beta > 0.225`**, not `beta > 0.45`. More importantly,
`beta` is a maximum over generators and class pairs. Its null scale is not the
single-inner-product standard deviation `1/sqrt(512)=0.044`; the maximum grows
with the number of comparisons. Forecasting a maximum of `0.025--0.035`, below
the standard deviation of one comparison, is not credible. F8's `beta <= 0.05`
is therefore not calibrated to its own max statistic.

`Omega` minimizes a **mean of squares** on training samples/proxies. It neither
bounds the maximum nor transfers such a bound to held-out classes. The claim
that having far more constraints than degrees of freedom supplies headroom runs
in the wrong direction: it makes simultaneous near-zero satisfaction harder.
The proposal also uses `Delta=0.9` without connecting it to measured nearest
negative margins, so even corrected arithmetic would not quantify a ranking
guarantee.

### 5. The direct probe does not identify the claimed nuisance

Applying the same sampled crop/flip parameters to two different source images
does not make their physical pose, lighting, viewpoint or background state the
same. It only reuses augmentation coordinates in two different pixel frames.
Any cosine shift can be ordinary augmentation equivariance, interpolation, or
crop-overlap response. CARE, AugSelf and EquiMod already occupy those measured
operators. The proposed probe cannot validate the premise used in the worked
`+0.19` similarity example.

### 6. Novel notation does not reopen the shared-variation mechanism

The exact signed-permutation frame may be an unpublished wrapper, but the
scientific object is class-independent intra-class variation constrained to a
shared spherical transport law. DVML explicitly assumes class-independent
intra-class variance and separates it from class centers:
<https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html>.
Spherical Feature Transform transfers shared within-class variation between
classes by sphere-respecting rotations:
<https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf>.
Candidate 369 and Pass 18 CITTR already died on exactly this premise and this
prior-art neighborhood. OLE additionally occupies low-rank within-class
geometric confinement, although with per-class subspaces rather than CFR's
fixed rotating frame:
<https://openaccess.thecvf.com/content_cvpr_2018/papers/Lezama_OLE_Orthogonal_Low-Rank_CVPR_2018_paper.pdf>.

The Clifford basis is a distinctive estimator/regularizer, but the proposal's
claimed benefit comes from the same class-exogenous shared-variation premise.
Because that premise is measured adverse here and the objective does not
identify shared nuisance coordinates, exact-wrapper novelty cannot authorize a
run.

## Recipe, forecast and cost corrections

- The proposal correctly labels all method rows as forecasts and admits that no
  Lane-A frontier is forecast to clear cleanly. That honesty survives.
- Its CUB crossing arithmetic (`74.0` against about `73.94`) is internally close,
  but a point only `0.06` above a constructed two-sided bar is not a 0.45
  probability claim without a stated joint sampling model. The Cars/SOP calls
  correctly say “does not clear.” None is repository evidence.
- The claimed `Omega` proxy cost is understated for the 512-proxy subsample.
  A naive exact computation is approximately
  `136 * 512^2 * 512 = 18.3` billion multiply-adds per step before the sample
  term, not about 1 GFLOP. Streaming can control memory, but materializing all
  transformed 512-proxy vectors is about 142 MB in fp32. The reported 3.9 MB
  activation figure counts only 16 sample-frame vectors at batch 120, not the
  full 136-element `Omega` basis or proxy term.
- `B0` changes the official PA recipe to 200 epochs, cosine decay, AdamW,
  balanced `K=4` sampling and different proxy learning-rate assumptions. The
  proposal discloses this and correctly refuses to inherit the published PA
  row, but the resulting baselines and `+1.8/+2.2` gains remain invented.
- The claim of a cost-axis frontier crossing is not established. It compares
  a forecasted CFR accuracy below PFML with a training-overhead estimate while
  ignoring the changed 200-epoch base and the unimplemented 136-map proxy term.

## Authoritative disposition

CFR is **DEAD at Gate 1**, independently reinforced by fatal identifiability,
geometry and gradient errors. The exact Clifford construction is intellectually
interesting, but it does not turn label-only tangent confinement into an
identified shared nuisance action. Repairing the proposal would require a new
observable for cross-identity nuisance correspondence, a different objective,
and a new blind pass; it is not an ablation of frozen CFR.
