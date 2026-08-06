I verified the algebra by hand and checked the primary sources. Here is the adjudication.

---

# Pass 49 — RLML: **DEAD**

**Earliest failed gate:** Legality — no test/gallery data or unseen test identities may enter method selection. **F4** fits the POT shape on *test-split negatives* and is designated "cheapest, **run first**… this can kill the method before any run" (§5). Advance/kill is therefore conditioned on a statistic computed over unseen test identities, with a numeric selection rule (|ξ̂_train − ξ̂_test| > 0.15). That is method selection on test identities, not a post-hoc diagnostic, and it is positioned first, where it contaminates everything downstream.

**Decisive mechanism-level reason:** The return-level gradient is affine in the plotting position with a sign flip at `c* = (log m − 1)/(4 log m − 8)`, so a large fraction of the *selected hardest* negatives receives an **attractive** gradient. `c*` is strictly decreasing in `log m` (`dc*/dL = −4/(4L−8)² < 0`), so the attraction fraction **rises monotonically with the very extrapolation depth the method exists to exploit**, approaching 3/4 as `L → ∞`: **38.7% (Cars), 39.7% (CUB), 69.3% (In-Shop), 70.4% (SOP)**. That range sits inside TERL's already-rejected 36–72%. The three deltas RLML offers over TERL cannot touch it, and I show below that each is provably inert.

---

## Verified-correct algebra (preserved separately from the verdict)

I re-derived the estimator from `a_s = α/[(s+1)(s+k+1)]` for the Hosking–Wallis GPD `F(x) = 1 − (1 − kx/α)^{1/k}`:

| Claim | Status |
|---|---|
| `ξ̂ = (a₀−4a₁)/(a₀−2a₁)`, `σ̂ = 2a₀a₁/(a₀−2a₁)`, `σ̂/ξ̂ = 2a₀a₁/(a₀−4a₁)` | ✓ correct (`ξ = −k`; `k̂ = a₀/(a₀−2a₁) − 2`) |
| Sanity: exponential `(s, s/4) → (0, s)`; uniform `(σ/2, σ/6) → (−1, σ)` | ✓ both correct |
| Zero-shape series `R = σ log m(1 + ξ log m/2 + ξ²log²m/6)` | ✓ correct Taylor expansion of `(σ/ξ)(e^{ξL}−1)` |
| P2 point mass: `a₁→a₀/2`, `ξ̂→−∞`, `σ̂→∞`, `R→a₀` | ✓ exact (mean of `c_r=(K−r)/(K−1)` over `r=1..K` is exactly 1/2) |
| `dR/de_(r) = A + B·c_r`, `A = L(L−1)/K`, `B = L(8−4L)/K`, `c* = (L−1)/(4L−8)` | ✓ I derived it independently via `(a₀,a₁)→(σ,ξ)→R`; matches exactly |
| SOP SEM 0.0009, diff SE 0.0013, 2-SE crossing 0.25 pp | ✓ (the "8.5 SE" is 8.70 SE — immaterial) |
| log m = 2.708 / 2.708 / 7.437 / 6.394 | ✓ |
| Splits CUB 5,864 / Cars 8,054 / SOP 59,551 / In-Shop 25,882 | ✓ verified against the standard DML protocol |
| WEINCE characterization | ✓ **fully verified at source**: *"we treat (λᵢ, β̂ᵢ) as stop_grad statistics (no backprop through the tail fits)"*; within-batch only, no gallery extrapolation; SimCLR on CIFAR-10/100, STL-10, ImageNet-32, Tiny-ImageNet + SimCSE — no DML benchmarks |

The mathematics is competent. That is what makes the verdict decidable on paper.

---

## 1. Pooling/EMA vs. missing signal — **variance only, and bought with bias**

Pooling (K≈4700, sd(ξ̂) 0.25→0.015) and EMA are both averaging operators. They reduce Var(ξ̂); they cannot raise the incremental R² that fitted shape carries beyond (threshold, mean excess), which Pass 31 measured at **0.0087**. No estimator extracts signal that is not there.

Worse, pooling imposes homogeneity across anchors and classes (A1). If per-anchor tails differ, pooled ξ̂ is consistent for a *mixture*-tail index, not for any anchor's own tail — a bias introduced to buy variance. And Pass 31's RMSE result (tuned constant **0.0918** vs fitted **0.1575**) is a direct prospective prediction that C3 (`ξ̂≡0 → u + σ̄ log m`) and C4 (`u + c`) will match or beat the full method. By RLML's own F3 ("C3 ≥ 80% ⇒ report as a log-N margin schedule, not an EVT method"), the frozen object pre-registers its own demotion. Note further that at the design point CUB/Cars have `m ≈ 15 < k_i ≈ 26`: on two of four benchmarks the "extrapolation" interpolates *inside the batch*, so it is definitionally a constant-margin arm there.

## 2. New supervision object, or estimator wrapper? — **wrapper**

Component-for-component identity with TERL: pooled per-query exceedances ✓, Hosking–Wallis PWM GPD shape ✓, straight-through EMA ✓, extrapolated return level ✓, hinged against a smooth best-positive score ✓. The three deltas are each provably orthogonal to the rejection reason:

- **Raw vs. normalized exceedances.** The PWM GPD shape estimator is scale-equivariant: under `e → ce`, both `a₀` and `a₁` scale by `c`, so `ξ̂ = (a₀−4a₁)/(a₀−2a₁)` is **invariant** and `c*` (a function of `log m` alone) is untouched. Per-anchor normalization changes only pooling homogeneity — variance again.
- **tanh cap.** `M̂ = u + (1−u)tanh(R/(1−u))` has derivative `sech²(·) ∈ (0,1]`, strictly positive. A positive monotone transform cannot flip a gradient sign; it rescales *all* exceedance gradients by one common scalar. Attraction fraction unchanged.
- **θ = 1/n̄_c.** `R` depends on `m` only through `m^ξ = e^{ξ log m}`, so θ is an additive shift of `log m`. Since `c*` decreases in `L`, *any* θ leaving enough extrapolation to matter (`L > 2.33`) leaves a large attraction region, and shrinking `L` below 2.33 (`m < 10.3`) kills the extrapolation itself. The heuristic cannot escape the trap.

No new supervision object. Three inert wrappers on a rejected one.

## 3. Gradient signs and the cheapest gaming path — **fatal, in three independent ways**

**Does the pull-closer term improve retrieval?** No. `R@1 error = P(max_{g∈G⁻}s > max_{p∈G⁺}s)` is monotone non-decreasing in *every* negative similarity. Raising 38.7–70.4% of the top-15% hardest negatives strictly increases every term of that probability. The only object improved is the fitted statistic.

**Cheapest gaming path (quantified).** `R` sees the ~4700 exceedances only through two scalars. At the exponential reference with L=7.4: `∂R/∂a₀ = 47.4`, `∂R/∂a₁ = −159.8` — R falls **3.4× faster** per unit of `a₁` than it rises per unit of `a₀`, and `a₁` weights the *smallest* exceedances by ≈1, the largest by ≈0.

- Honest path — lower the top 30% by ε: `Δa₀=−0.3ε`, `Δa₁≈−0.045ε` → **ΔR ≈ −7.0ε**
- Gaming path — raise the bottom 70% by ε: `Δa₀=+0.7ε`, `Δa₁≈+0.455ε` → **ΔR ≈ −39.5ε**

≈**5.6× more loss-efficient**, and it acts on the densest, cheapest-to-move pairs just above the 85th-percentile threshold rather than on the genuinely confusable extremes. Descent takes it.

**Unbounded free descent in ξ̂.** `d/dξ[(e^{ξL}−1)/ξ] = [ξLe^{ξL} − e^{ξL} + 1]/ξ² ≥ 0` for all ξ (numerator vanishes only at ξ=0), so **R is strictly increasing in ξ everywhere** — the loss has a monotone, unbounded reward for shrinking ξ̂, with no interior stationary point. Only the clamp stops it. At `ξ̂ = −4` and `m ≥ 15`, `m^{−4} ≤ 2×10⁻⁵`, so `R = (σ̂/4)(1 − m^{−4})` is **constant in m to within 0.002%**: N, θ, and the entire stated mechanism switch off and RLML reduces to `M̂ = u_i + σ̂/4` — precisely TERL's "parked at a clamp, reduced to a fixed constant arm."

**The clamp also breaks P2.** P2's non-divergence requires the endpoint-stable ratio `σ̂/ξ̂ = 2a₀a₁/(a₀−4a₁)` at the *unclamped* ξ̂. §1.5 clamps ξ̂ and EMAs `ξ̄, σ̄` separately, so the implemented quantity is `σ̄/ξ̄_clamped`. As the gradient drives `a₁ → a₀/2`, `σ̂ = 2a₀a₁/(a₀−2a₁) → +∞` while `ξ̄` is pinned at −4, so `R → +∞`. P2 is a correct theorem about an estimator the method does not implement, and the loss's own attractor sits on a pole its stated stabilisation creates.

## 4. Novelty — **does not survive**

Narrowest defensible claim: *backprop through a PWM-GPD shape fit of pooled negative-similarity exceedances to a return level at train-derived depth m ≫ n, hinged against a smooth-max positive.* Both halves are taken:

- **"Backprop through the fit, not stop-grad" → LDReg (ICLR 2024).** Its objective `ℒ_L1 = ℒ_SSL − β(1/N)Σ ln(LID*)` places an EVT tail-index estimate *inside the loss*. LID is defined extreme-value-theoretically (Houle's LID Representation Theorem), and the estimator `LID* = −μ_k/(μ_k − w_k)` is the Method-of-Moments estimator from Amsaleg et al. (DAMI 2018) — the paper that derives LID estimators "based on extreme value theory, using maximum likelihood estimation, the method of moments, **probability weighted moments**, and regularly varying functions," i.e. literally Hosking–Wallis's estimator family. A detached regularizer has zero gradient and cannot *increase* LID, which is LDReg's entire claim, so it is necessarily differentiated. **RLML's §3 novelty table does not contain LDReg** — the adversarial search missed the nearest prior art for its load-bearing distinction.
- **Return-level extrapolation to N ≫ n** is taken post hoc by biometric EVT FAR calibration and OpenMax/EVM (RLML distinguishes these correctly on post-hoc vs train-time), and taken at train time *with gradients* by ten prior internal candidates — TERL identically.

The WEINCE distinction as written is correct and verbatim-verified, but WEINCE was never the binding prior art for the load-bearing half. What remains is the conjunction of three inert wrappers.

## 5. F4 — **prohibited test-identity selection**

Not a legal post-hoc diagnostic: (i) it fits on unseen test identities; (ii) it is explicitly ordered *first* and gates whether the method proceeds; (iii) it carries a numeric threshold on test data. A legal version would fit on a held-out *train-identity* split; the frozen object does not. Structurally worse: A2 (train→test tail transfer) is what the proposal itself calls "the load-bearing zero-shot assumption," and **the only test it offers for that assumption is the one the envelope forbids.**

## 6. Headline arithmetic

**The In-Shop row is wrong and self-contradictory — on the dataset the envelope makes the first screen.** `0.930 + 1.3 pp = 0.943 ≠ 0.934` (0.9 pp gap). The interval `[0.926, 0.943]` matches neither `0.930 + [+0.5,+2.2] pp = [0.935, 0.952]` nor a symmetric band on 0.943. The same section's prose — "In-Shop crossing 0.930 is ≈ a coin flip" — is coherent with 0.934 (+0.4 pp) and flatly incoherent with +1.3 pp at `[+0.5,+2.2]`, whose *lower bound* already clears the reference. A forecast that cannot be read unambiguously has not been pre-registered.

SOP/Cars/CUB rows are arithmetically correct. SEM chain correct, with one caveat: it assumes PFML's "±0.002" is a seed std; if it is already a SEM, every SE is √5× too small and "well-powered" fails.

**θ is not an extremal index.** The extremal index is the reciprocal mean *cluster size of exceedances above the level*, not reciprocal images-per-class. `θ = 1/n̄_c` asserts every one of a class's `n̄_c` images exceeds `u` and forms one cluster. For SOP, `n̄_c = 5.26` and `ρ = 0.15`, so ~0.79 images per class are expected to exceed `u` at all — the correct extremal index is ≈1, so θ is misspecified by ≈5×. Its sole effect is the shift `log m ← log m − log n̄_c`.

**m collapses.** With `n̄_c = N/C`, `m = θρN = (C/N)·ρ·N ≡ **ρC**` — `|D_train|` cancels **identically**. CUB 0.15×100 = 15 ✓; Cars 0.15×98 = 14.7 ✓; SOP 0.15×11,318 = 1,697.7 ✓; In-Shop 0.15×3,997 = **599.6, not 598** (implies C = 3,987).

## 7. Is N causally identified as deployment depth? — **no; it *is* train class count**

After the θ discount the loss uses `0.15 × (number of training classes)` — a pure train-set-composition statistic containing **zero gallery-size information**. The claimed causal variable is not merely confounded with class count; it is class count.

This kills F1. Its ordering prediction (SOP ≳ In-Shop > Cars ≈ CUB) is *exactly* the train-class-count ordering (11,318 > 3,997 > 98 ≈ 100), so it cannot separate the EVT mechanism from anything monotone in class count — including PFML's own proxy count, which differs along precisely that axis (**M=2 on SOP vs M=15 on CUB/Cars**) and is held fixed by no control. Credit where due: F1 *does* discriminate against classical hard-negative mining, which historically favours CUB/Cars. That observation is correct and worth keeping.

**The "~2%" claim is false for In-Shop**, the mandated first screen: N = 25,882 against a test gallery of 12,612 is **105% larger**; against query+gallery (26,830) it is 3.5% low. It holds only for CUB (1.0%), Cars (0.95%), SOP (1.6%). More importantly than the numeric error — invoking the correspondence *at all* as design support draws on the test gallery's size, which is test knowledge entering method design even though the computed value uses only train data. The proposal anticipates this and offers F2 as mitigation; but F2 is algebraically vacuous once ξ̂ reaches the clamp (R constant in m to 0.002% across the whole sweep), so the mitigation does not exist.

## 8. Executable completeness and carrier fidelity

§7 is commendably candid that PFML's batch, sampler, schedule, weight decay, warmup, augmentation, and proxy normalization are undisclosed, and that "δ ∈ [0.1,0.3], α ∈ {0,6}" is set-vs-interval ambiguous. It then fixes batch 180 = 4×45 by fiat — a **carrier-fidelity break**, because RLML's own statistics depend on that composition (`K ≈ 4700` needs n=180 with 4 img/class; `k_i ≈ 26` needs ν=176). RLML is not a drop-in modifier of the disclosed PFML; it is PFML-plus-an-invented-sampler, and any Δ is against an in-house reconstruction. §7.1's remedy is right in spirit, but changing n changes K, k_i, estimator variance and A1, so the three batch arms are not matched-mechanism arms.

**Count:** ρ, β, γ, EMA, θ, clamp bounds, K-fallback 64, Δ, λ/κ, the 20→40 ramp, and the 50-step norm window — ~**11** new constants, not the 7 claimed. The selection rule (freeze seven; tune κ and Δ on CUB only, transfer) is **train-only and legal** — a genuine strength. But it is self-undermining here: CUB is where the proposal itself predicts the mechanism is inert (`m ≈ 15 < k_i ≈ 26`), so κ and Δ are chosen exactly where the tail term does nothing, then transferred to SOP where the attraction fraction jumps 39.7% → 70.4%. **The tuning set cannot see the failure mode.**

λ gradient-norm matching is genuinely good and correctly reasoned, including that weight decay on `W` leaves `z` invariant under L2-norm but changes the effective angular LR, making Δ operational. Preserve it.

## 9. C1–C8 / F1–F6

**Well isolated:** batch-max hinge (C1); location vs shape, both quantile and CVaR arms (C2); the shape parameter (C3, sharpest); constant margin (C4); observe-vs-extrapolate bound (C6); pooling/EMA vs A1 (C7); the attractive half (C8). As a control *set* this is the right construction, and C8 is a real pre-registered self-falsification.

**Not isolated:** (i) **ordinary hard mining** — no arm holds the *set of pairs receiving gradient* fixed; the correct control is C8 ∧ C2 (same top-15% selection, non-negative hardness-monotone weights), never specified jointly. (ii) **XBM** appears only as C6 at memory `M = |D_train|`, an upper bound with no matched-memory arm, so "beats a *comparable* memory" is untested. (iii) **Negative attraction as a standalone effect** — C8 removes it, but no arm *adds* it without the EVT machinery, so a positive C8 delta cannot distinguish EVT extrapolation from a lucky dose of negative attraction. (iv) **Proxy count M** is held fixed nowhere, which is fatal for F1.

**F-battery:** F1 confounded; F2 vacuous at the clamp; F3 pre-registers the demotion Pass 31 already predicts; F4 illegal; F5 sound but decided in advance by closed form; F6 fine. Of six falsifiers, one is illegal, one vacuous, one confounded, two resolved on paper.

Note the structure of C8/F5: if gains survive C8, RLML is hard-negative weighting and novelty collapses (the proposal says so); if they don't, the load-bearing half is the attraction term, i.e. the estimator-gaming path. **Both branches are fatal, and the algebra decides which without running it.**

## 10. Reporting protocol — **does not meet the envelope's ordering**

- **In-Shop first, paired, corrected:** not first — F4 (illegal) is designated "run first." In-Shop is second in the table, has no paired/corrected protocol, and §7.3 concedes PFML never evaluates it while PA+DADA 0.930 reports no seeds. Pairing (shared seeds/sampler/reconstruction, per-seed differences) is unspecified.
- **Raw best vs. independently selected/final:** not specified at all. With 200 epochs and a λ ramp ending at 40, best-epoch selection is an open inflation channel.
- **Out-of-sample confirmation:** absent — all four benchmarks are the selection surface.
- **Second-dataset replication:** nominally four, but CUB/Cars are self-declared null controls, so effective replication is SOP → In-Shop, and In-Shop's forecast is the self-contradictory row.
- **Paired uncertainty:** the arithmetic is correct but **unpaired** — `√(SE²+SE²)` assumes independent arms where the design shares seeds and sampler. The right quantity is the SD of per-seed paired differences, typically much smaller. This errs *against* RLML, so preserve it as conservative rather than as a defect.
- **Cost:** forward-FLOP-only. §6 omits the backward pass through the PWM fit (gather/scatter over K≈4700 with `c_r` weights plus the sort permutation) and cross-anchor synchronisation; the sort is a serialising, latency-bound kernel. "1.00–1.02×" is asserted, not measured. The **deployment**-cost claim (exactly zero delta; N is train-time only) is correct and verified from §1.

---

## Uncertainty

- LDReg's differentiability is **inferred**, not verbatim-quoted: the arXiv PDF returned unrenderable binary, and the HTML fetch surfaced the objective and the estimator but no explicit "we backpropagate" sentence. The inference is strong (a detached regularizer cannot increase LID), but the novelty verdict does not rest on it — the internal TERL recurrence is independently sufficient.
- Pass 31's measurements and the ten-candidate chronology are taken as supplied per the prompt; I did not verify them.
- The "≈5.6×" efficiency ratio uses the ξ≈0 linearization and mean `c_r` over each range; it varies with the empirical exceedance distribution. The **sign structure** and the 3.4× dominance of the `a₁` channel are exact at that point.
- I did not verify PFML's 0.734/0.927/0.829 or PA+DADA's 0.930 at source; no part of the verdict depends on them.

Any substantive repair — deleting F4, replacing θ, defining `m` from something with actual gallery content, or removing the sub-`c*` attraction — is a **new proposal**, not a revision of this one.

**Sources:** [WEINCE, arXiv:2606.00262](https://arxiv.org/abs/2606.00262) · [WEINCE HTML](https://arxiv.org/html/2606.00262) · [LDReg, ICLR 2024 / arXiv:2401.10474](https://arxiv.org/abs/2401.10474) · [LDReg HTML](https://arxiv.org/html/2401.10474v1) · [LDReg ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/496d8e7c79c39e284d3b461d3fed13d7-Paper-Conference.pdf) · [Amsaleg et al., *Extreme-value-theoretic estimation of local intrinsic dimensionality*, DAMI 2018](https://link.springer.com/article/10.1007/s10618-018-0578-6) · [Houle, *LID I: An Extreme-Value-Theoretic Foundation*](https://link.springer.com/chapter/10.1007/978-3-319-68474-1_5) · [Hosking & Wallis, *Parameter and Quantile Estimation for the GPD*, Technometrics 1987](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488243) · [DML split cardinalities](https://arxiv.org/html/2503.13045) · [In-Shop protocol](https://arxiv.org/pdf/1810.06951)
