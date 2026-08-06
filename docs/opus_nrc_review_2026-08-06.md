# Adjudication — Pass 48, NRC (Nuisance-Rotor Coding)

## Verdict: **DEAD**

**Earliest failed gate: Gate 1 — provenance.** NRC's load-bearing posit is a **universal fixed response law**: §2.1 states the rotor phase "is a function of `t` alone, identical for every image and every class." The only supplied measurement of augmentation response says the opposite — corrected ARCG found augmentation-response compatibility **image-specific and heterogeneous** on In-Shop, retaining only ~0.3631–0.3640 of same-class pairs while passing close-pair rejection and distant-pair acceptance. A universal, image-independent law predicts near-uniform compatibility; ~36% retention *with image-specific structure* is a direct contradiction, not a gap. IPSR separately established an augmentation-response relation **with no causal retrieval benefit** — which is precisely the transfer step NRC needs and concedes it has not shown (§6, "the transfer from ledgering photometry/scale to tolerating pose is asserted, not proven"). Provenance is therefore a forecast contradicted by the available measurement, in a family (Candidate 147, the augmentation-response supervision line, the tangent-metric line, cycle-equivariance 307) with a prior negative record and no new positive evidence.

**One decisive mechanism-level reason:** the *only* thing separating NRC from the occupied AugSelf/EquiMod supervision family is the claim that reading the ledger off **the same coordinate pairs** that produce the descriptor "prevents the trunk from servicing phase and radius through disjoint upstream channels." That claim is false. The shared-coordinate constraint binds only the final linear map `W_u`; a nonlinear 2048-channel trunk can realize `u_j = ρ_j(identity) · (cos ψ_j(t̂), sin ψ_j(t̂))` from disjoint upstream pathways combined in its last stage, giving radius = identity, phase = nuisance, and satisfying `L_rot`, `L_comp`, `L_inv`, `L_occ` while the descriptor gains nothing. The proposal itself pre-registers this as **F4** ("the coupling claim (the actual novelty) is false"), so with the argument gone, NRC is a fixed reparameterization of augmentation-parameter-difference prediction — AugSelf's exact supervision object with a frozen readout.

---

## Audit findings

### 1. Geometry of `û = u/(‖u‖+ε)` — **the stated proof is false**

`û` is **neither unit length nor zero-homogeneous**: `‖û‖ = r/(r+ε) < 1`, and `û(cu) ≠ û(u)`. The exact Jacobian is

```
∂û/∂u = (1/(r+ε))·[ I − s·P ],   s = r/(r+ε),   P = uuᵀ/r²
     radial block:      (ε/(r+ε)²)·P        tangential block:  (1/(r+ε))·(I−P)
```

so the radial:tangential operator ratio is `ε/(r+ε)`, **not zero**. Verified by finite difference at `r = 0.806`, `ε = 1e−6`: `∂L_rot/∂r` measured `3.33e−7` against predicted `−ε·cos/(r+ε)² = 3.26e−7`; `∂L_rot/∂log r = 2.69e−7`. §1.5's "`∂L_rot/∂(log r_j) ≡ 0`" and "`⟨∂L_PA/∂u_j, ∂L_rot/∂u_j⟩ = 0`" are **literally false for the executable forward pass**. The leak is ~1e−6 relative and I do not claim it is numerically material — but it is the proof that §1.5 calls "the novelty, stated formally."

**D2 is materially, not just ε-, wrong.** The per-plane loss is `1 − s⁽¹⁾s⁽²⁾cos Δ`. Driving `r⁽¹⁾ → 0` drives `s⁽¹⁾ → 0` and the loss to **1**:

| cos Δ | r=1 | r=1e−3 | r=1e−6 | r=1e−9 | r=0 |
|---|---|---|---|---|---|
| +1 | 0.000 | 0.001 | 0.500 | 0.999 | 1.000 |
| −1 | 2.000 | 1.999 | 1.500 | 1.001 | 1.000 |

A maximally-violating plane buys **50% of the maximum penalty** by collapsing — not "exactly zero." The escape survives only because its basin sits at `r ≲ ε` and the gradient along the path is `O(ε/r)`; the true closure is `L_occ`, which the proposal demotes to "numerical health." The stated architecture of closure is misattributed.

### 2. Output-space orthogonality ⇏ parameter-space orthogonality — **fails even at ε = 0, even at the head**

For the head itself, `∂L/∂W_u = Σᵢ (∂L/∂uᵢ) φᵢᵀ`, so

```
⟨g_PA, g_rot⟩_F = Σ_{i,i'} (aᵢ·b_{i'}) ⟨φᵢ, φ_{i'}⟩
```

Diagonal terms vanish exactly (`aᵢ ⊥ bᵢ` per plane, ε = 0). **The cross-terms `i ≠ i'` do not**, and they are weighted by the post-ReLU GAP feature Gram, which is strictly positive. Simulated (8 planes, batch 12, exact radial/tangential split, ε = 0): per-sample activation orthogonality holds to `2.3e−16`, while the two **parameter** gradients have cosine **+0.27**. Add: the trunk NTK `Jᵢ Jᵢᵀ` does not preserve the per-plane radial/tangential splitting even on the diagonal; AdamW's diagonal preconditioner gives `g_PAᵀ D⁻² g_rot ≠ 0` even when raw gradients are orthogonal; and gradient orthogonality is a first-order instantaneous statement — trajectory coupling is governed by the Hessian cross-block, which is generically nonzero. §1.5's scoping ("through the head") is honest, but §0's and §3's stronger coupling claim is not supported.

### 3. Maximal-invariant proposition — **false; concrete collision supplied**

Two failures.

**(a) `r` is not a maximal invariant of `ρ(A)`.** `ρ(A) ⊂ T⁵¹²` has dimension ≤ 6 (5 continuous + 1 circle + Z/2), not 512. `r` is invariant under the *full* torus, hence strictly **coarser** than the orbit map of `ρ(A)`: two points with equal radii and phase offset outside `im ρ` are separated by the maximal invariant but not by `r`. The correct statement is "`r` is *an* invariant, coarser than maximal"; the descriptor discards ~506 angular directions carrying no rotor-code content.

**(b) Full rank of `W_g` does not make the normalized map injective.** `W_g ψ(r) + b_g` is injective (ψ = log(1+·) strictly increasing, `W_g` bijective). `normalize` is not — it quotients by positive scaling, and under `ψ = log(1+r)` that quotient is a genuine one-parameter family of distinct radius vectors, `r_j = (1+r'_j)^c − 1`. Verified with a full-rank integer `W_g` (b_g = 0):

```
r' = (1.0, 2.0, 3.0, 0.5)      r = (3.0, 8.0, 15.0, 1.25)      c = 2
max |e(r) − e(r')| = 0.00e+00      → identical deployed descriptor, distinct radii
```

For `b_g ≠ 0` the collision persists in an open neighbourhood of `c = 1` whenever all `r'_j > 0` — which `L_occ` actively enforces.

### 4. `A ≅ R⁵ × S¹ × Z/2` vs. the actual image operators — **ρ is not a representation of the rendered action; L_comp tests no homomorphism**

Three independent breaks, verified against torchvision primary documentation:

- **Crop position is omitted from `t`.** `RandomResizedCrop.get_params` returns `(i, j, h, w)` — a random top-left corner in addition to scale and aspect. So `a_t(x)` is **not a function of `t`**; `t` is a partial label. §2.3 correctly proves that scale+translation generate `ax+b`, whose commutator subgroup is translation, so no torus-valued rotor can be equivariant to it — and then removes translation from the ledger without removing it from the renderer.
- **ColorJitter applies its four operations in a random order.** The docs state `get_params` returns the parameters "along with their **random order**." Hue is a nonlinear HSV shift that does not commute with contrast/saturation blending, and output clipping to [0,1] plus 8-bit quantization are non-invertible and do not commute with the multipliers. So the photometric factor is an **order-dependent, non-invertible monoid**, not `R³` — the claim "photometric multipliers compose multiplicatively (→ additively in log)" does not hold for the executed pipeline, and the sampled permutation is a hidden nuisance variable outside `t`.
- **`v3` is rendered once at `t^(1)+t^(3)`, never as `a_{t3}(a_{t1}(x))`.** Both the loss input and the loss target are computed from the same supplied parameter ledger; the image never undergoes composition. `L_comp` therefore tests **linear extrapolation of a parameter regressor onto a wider parameter range** — the proposal says so itself ("`t^(13)` has twice the variance… probes the code at parameter values outside the single-view support"). D4's group-theoretic premise is correct (continuous homs `R⁵×S¹×Z/2 → S¹` are exactly `exp(i⟨ω,ξ⟩ + inh + iπεf)`), but it constrains `ρ`, which is fixed and unlearned. It never constrains the network. D4's own conclusion — "passing `L_rot+L_comp` is equivalent to having estimated the true nuisance parameters" — **concedes that the supervision object is parameter estimation**.

Two clamp objections I checked and am *not* pressing: the composed area-scale clamp binds on only **1.25%** of `v3` draws (400k Monte Carlo), and the photometric clamp `[0.3, 2.0]` binds on **0.000%** (product range 0.36–1.96). A real but minor spec ambiguity remains: the frozen text does not say whether the `L_comp` target uses the clamped or unclamped `t^(13)`; if unclamped, that 1.25% carries a systematically unsatisfiable target.

Separately: `ColorJitter(hue=0.1)` gives `h ∈ ±0.1` **turn** = `±0.2π` rad. On the `n_j = 8` octave, `n_j·h` spans `3.2π > 2π` — the hue code **wraps**, making `φ_j` non-injective in `h` on those planes. The frozen text also never pins whether `h` is in turns or radians while asserting "hue angles add mod 2π."

### 5. Cheap shortcuts — **live, and D5's mitigation addresses the wrong channel**

The shortcut is concrete and cheap: estimate `t̂` from **low-order global image statistics**, then set phase from `⟨ω_j, ξ(t̂)⟩` and radius from identity content.

- σ, a: apparent magnification, blur scale, and the interpolation kernel's periodic second-derivative correlations — resampling factor is estimable from a single image (Popescu & Farid, *IEEE TSP* 53(2):758–767, 2005).
- β, γ, ς: rendered mean/std/chroma versus the dataset prior are near-sufficient statistics for the applied multipliers.
- h: mean chroma angle versus a strong dataset colour prior (birds, cars).
- f: half the planes ignore it anyway (`ε_j = q mod 2` is 0 for half of `q`).

**D5's mitigation is misdirected.** It argues "the three photometric coordinates + hue have no resampling signature" — true and irrelevant. Those four coordinates have a **direct photometric** signature; they need no artifact at all. D5 addresses only the geometric channel and only the artifact flavour, leaving 4 of 7 coordinates on the easiest possible route.

**C6 cannot adjudicate this.** On a constant-texture image the photometric coordinates are *trivially* readable (a flat colour shifts exactly as β, γ, ς, h dictate) while σ, a are unreadable. Because every `v_q` is a Hadamard row with all five entries ±1/√5, each plane's phase mixes **all five** `R⁵` coordinates equally, so C6 returns a partially-reduced `L_rot` with no per-coordinate attribution, and F6's 0.15 threshold is unmotivated. C6 also conflates "artifact-solved" with "photometric-statistic-solved," which are different mechanisms with different implications.

**`L_occ` does not prevent uniform all-radius shrinkage.** It is defined on the *normalized share* `r_j/Σ_k r_k`, which is invariant to `r → c·r` for any `c > 0`. It therefore provides **no floor on absolute radius** — the exact quantity that the ε-dependent `û` makes load-bearing (as `r → ε`, `‖û‖ → 1/2` and `L_rot → 1` with vanishing gradient). Uniform shrinkage is blocked only incidentally, by `L_PA` (as `r → 0`, `e → normalize(b_g)`, a constant). And per audit 1, "shrinking buys exactly zero" is **not** compatible with the ε-dependent `û`.

### 6. Prior art — **the supervision object is occupied; the architecture is narrowly new**

Verified against primary sources:

- **AugSelf** (Lee et al., NeurIPS 2021) optimizes an auxiliary loss "that learns **the difference of augmentation parameters** (e.g., cropping positions, color adjustment intensities) between two randomly augmented samples." That is *exactly* `L_rot`'s object: the target is `R(φ_j(t⁽¹⁾ − t⁽²⁾))`, a function of the parameter difference, read from a view pair.
- **EquiMod** (Devillers & Lefort, ICLR 2023): the predictor's "input is the concatenation of a representation of `t` and the input embedding," producing `ẑ` from `z` and `t`. That is `L_rot` with a learned rather than prescribed predictor.
- **SIE** (Garrido et al., ICML 2023): splits into **disjoint coordinate blocks** ("the first 256 dimensions for `y_inv` and the 256 last for `y_equi`"), fed through **separate projection heads** "to ensure no information is exchanged between the two after the split," with a hypernetwork predictor `p_{ψ,g}(z_equi) = reshape(H(g), d×d) z_equi`.

So NRC's structural distinctions from SIE and EquiMod are **real** — polar split of the *same* coordinates rather than disjoint blocks/heads, and a parameter-free prescribed predictor rather than a learned one. But magnitude-invariant/phase-equivariant polar coding is itself long-established (scattering transforms, Harmonic Networks, *Phase Collapse in Neural Networks*, ICLR 2022), and fixed multi-octave rotary codes for continuous parameters are RoPE/Fourier-feature standard.

**Narrowest defensible mechanism-level novelty:** *imposing, by loss rather than by filter structure, a modulus-invariant/phase-equivariant polar split on the descriptor head's own coordinates, for a photometric/scale nuisance group, with a parameter-free prescribed torus code as the predictor.*

**The rotor creates no new supervision.** Its information content is `t⁽¹⁾−t⁽²⁾` (D4 says so explicitly) — AugSelf's target under a fixed reparameterization. The one genuine consequence is the forced coupling at the head, which is exactly the claim falsified in audit 2. Search limitation: I found no paper doing NRC's precise polar split for augmentation parameters in supervised DML; absence at this search depth is weak evidence of absence.

### 7. Gate-1 provenance — **negative** (see verdict). ARCG's novelty audit already places single-image augmentation-prediction/equivariance auxiliaries with AugSelf and EquiMod, which audit 6 independently confirms is where NRC's supervision object lives. NRC offers no measurement supporting universality — only a forecast the one relevant local measurement contradicts.

### 8. D1–D8 / C1–C9 as causal controls

**Ledger gaps in the frozen object:** the table has six rows labelled D1, D2, D3, D4, D5, **D7** — D6 is skipped entirely and D8 never appears, though the assignment asks for D1–D8. Controls run C1–C9 (C9 marked diagnostic), plus an unnumbered C1′.

**C4 — the decisive control — is not clean.** As specified, C4 gives the parallel-head arm "a plain 2048→512 head" while NRC uses 2048→1024 → radii → 512×512. The arms therefore differ in **two** ways: coupling *and* head depth/parameterization (+1.31M params). If F4 fires, the result is attributable to head capacity, not coupling; if it does not fire, the same confound cuts the other way. This defeats the isolation C4 exists to provide.

**Which controls do separate identity-useful equivariance from parameter decoding or generic regularization:** C3 (`L_inv`+`L_occ` only) kills consistency-plus-floor; C5 (broken code) kills noise injection; C2 (multi-view PA) kills more-augmentation; C1′ is the only genuinely fair compute arm. C7 separates narrow-support from wider-support parameter regression, not "predicting `t`" from "representing `A` as a group" — that label is wrong per audit 4, but the arm remains informative. C8 is a hyperparameter ablation, not a causal control.

**Missing:** an arm supervising augmentation parameters on the radii head's *own* coordinates with a **learned** predictor (this is what isolates "prescribed parameter-free code" from "learned predictor," the SIE/EquiMod distinction NRC claims); an arm matched on **distinct images per epoch** rather than forward images per step; and a per-coordinate ledger enabling C6 to localize which of the seven coordinates is artifact-solved.

### 9. Recipe and compute matching — **not matched, and internally contradictory**

**The "1.00×" is forward-image-count only, and the metric-supervision claim is false.** §1.4 states L_PA is "view 1 only (so the metric supervision budget exactly matches the baseline)." It does not: the baseline computes `L_PA` over **150** samples per step, NRC over **50**. Positive sets `X_p⁺` shrink ~3×, the per-proxy log-sum-exp becomes far noisier, and on CUB a 50-image batch covers ~half the classes a 150-image batch does. Proxy Anchor's paper trains "all methods… with batch size of 150," and its own Tables 5–6 sweep 30–600 and report best performance at **150+ for the small datasets and 300+ for the large ones** — so NRC's 50 distinct images is below PA's small-dataset optimum and ~6× below its SOP/In-Shop recommendation, on every dataset.

This directly contradicts the proposal's own §6 admission that "distinct images per epoch drop 3×." Both cannot hold. At fixed steps and epochs NRC gets **1/3 the unique-sample exposure** of the baseline at matched FLOPs — meaning C1′ (3× step cost) is the only genuinely compute-fair arm, yet it is relegated to a "fallback."

**BN and batch/lr.** NRC fixes trainable BN and a universal 50×3 batch across all four datasets, against audited recipes that freeze BN and use dataset-specific batch sizes and learning rates. PA's paper does not disclose BN handling (I checked; the freeze convention is an implementation/audit fact, not a paper fact), which is itself a reason NRC's C1 is not a reproduction of the audited reference. Trainable BN also interacts with the 3-view batch: BN statistics are computed over **50 distinct images in correlated triplets** (v1 and v3 share `t⁽¹⁾`'s crop parameters), a different normalization regime from 150 independent images — a confound C2 controls but C1 does not.

**NRC+PFML is not a frozen carrier.** §5b leaves `α ∈ [0,6]` and `δ ∈ [0.1,0.3]` as ranges. I confirmed PFML states exactly this ("δ between [0.1,0.3]", "α between {0,6}") and discloses neither batch size nor weight decay. A carrier with two live hyperparameters permits post-hoc selection; that is a selection-integrity defect, not a source-reporting one.

**Deployment parameter count is off.** 27.96M/26.65M implies counting the full ResNet-50 *including its 1000-way fc* (25.56M) as the backbone. Against the correct headless base (23.51M + 1.05M head = 24.56M) the increase is +5.3%, not +4.9%. Immaterial to the verdict.

### 10. Protocol and forecasts — **direct violation of the frozen envelope**

The envelope requires: corrected paired In-Shop **first**; raw plus independently selected/final; out-of-sample confirmation; second-dataset replication; paired uncertainty. The proposal starts five seeds on CUB, Cars, and SOP, **never runs In-Shop**, reports one number per dataset with **no raw-vs-selected/final split**, specifies **no out-of-sample confirmation and no replication stage**, and computes **unpaired** SE. Four of five requirements are unmet and the mandatory first screen is bypassed. The local In-Shop reference exists precisely for this (three seeds: raw mean 0.918062, sd 0.001523; final mean 0.915201, sd 0.001549).

**The stated significance is also arithmetically wrong.** `SE_Δ = √(s₁²+s₂²)·√(2/5)` double-counts the pooled variance; the correct two-sample SE is `√(s₁²/5 + s₂²/5)`:

| | proposal SE | proposal σ | correct SE | correct σ |
|---|---|---|---|---|
| CUB | 0.00369 | 2.71 | 0.00261 | **3.83** |
| Cars | 0.00369 | 1.90 | 0.00261 | 2.68 |
| SOP | 0.00228 | 1.75 | 0.00161 | 2.48 |

The error is **conservative** — it understates NRC's own case by √2 — which I note as evidence the forecasting is not motivated. It does not rescue anything: the required estimator is *paired*, the seeds are not paired, and the entire 5b layer is conditional on an in-house PFML reproduction landing within 0.003 of 0.734/0.927/0.829 under an undisclosed batch size. The strongest headline remains +0.010 CUB over a frontier the proposal explicitly declines to inherit.

---

## Correct subcomponents — preserve independently of the verdict

1. **§2.3's derived exclusion is a genuine theorem.** Crop scale and translation generate `ax+b`; its commutator subgroup *is* the translation subgroup; every homomorphism to an abelian group contains it in the kernel. Therefore **no torus-valued rotor can be equivariant to crop translation**. Any future candidate placing translation in an abelian ledger is silently fitting a non-homomorphism. This result outlives NRC.
2. **The continuous-homomorphism classification is correct.** Continuous homs `R⁵ × S¹ × Z/2 → S¹` are exactly `exp(i⟨ω,ξ⟩ + inh + iπεf)` with `ω ∈ R⁵`, `n ∈ Z`, `ε ∈ {0,1}`. Correct group theory; it just does not constrain the trained network.
3. **C3, C5, C2, C1′ are correctly designed causal controls** and should be reused by any successor in this family.
4. **The C4 *intent*** — isolating coupling-through-shared-coordinates from auxiliary supervision per se — is the right decisive question. Its implementation needs the head held identical across arms.
5. **Source reporting is accurate where checkable.** PA ResNet-50/512-D/224² CUB 0.697, Cars 0.877, with **no** SOP or In-Shop ResNet-50 rows (confirmed: appendix Table 7); PFML CUB 0.734±0.003, Cars 0.927±0.003, SOP 0.829±0.002 over 5 runs, **no In-Shop row**, batch size and weight decay undisclosed, α and δ given only as ranges, M = 15/15/2, lr 5e-4, proxy lr ×100. All verified.
6. **The honesty discipline** — D5 left explicitly open, §5b declared conditional and non-inherited, F1–F7 pre-registered with F4 naming the novelty claim itself, and the self-flagged scope objection that "recording sampler state is a new class of free supervision" — is exemplary and should be the template. It does not repair a false central proof or negative provenance.

## Uncertainty

The ε leak (audit 1) is real but ~1e−6 relative; it falsifies the stated proof, not necessarily the trained behaviour, and is trivially repairable. The clamp objections are quantified and minor (1.25% / 0.000%). The parameter-space non-orthogonality (audit 2) is settled mathematics, but whether the resulting coupling *helps or hurts retrieval* is empirical — which is what C4/F4 was for, and that control is confounded as specified. Audits 3, 4, 5, 9, and 10 are decidable from the frozen text plus primary sources and require no experiment. My prior-art search found no exact NRC analogue in supervised DML; absence at this depth is weak evidence. The verdict rests on Gate-1 provenance and the falsity of the coupling claim, both of which stand independently of the smaller defects.

**Any substantive repair is a new proposal.** I have not modified the frozen object.

Sources: [AugSelf, NeurIPS 2021](https://arxiv.org/abs/2111.09613) · [EquiMod, ICLR 2023](https://ar5iv.labs.arxiv.org/html/2211.01244) · [SIE, ICML 2023](https://ar5iv.labs.arxiv.org/html/2302.10283) · [SIE (PMLR)](https://proceedings.mlr.press/v202/garrido23b/garrido23b.pdf) · [Proxy Anchor, CVPR 2020](https://ar5iv.labs.arxiv.org/html/2003.13911) · [PFML (arXiv HTML)](https://arxiv.org/html/2405.18560v3) · [Phase Collapse in Neural Networks, ICLR 2022](https://arxiv.org/abs/2110.05283) · [Popescu & Farid, IEEE TSP 2005](https://www.semanticscholar.org/paper/Exposing-digital-forgeries-by-detecting-traces-of-Popescu-Farid/1609781b81ded3cde0d8ff960b43f3cc81c5526a) · [torchvision ColorJitter](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) · [torchvision RandomResizedCrop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) · [torchvision adjust_hue](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html)
