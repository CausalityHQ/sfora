# VERDICT: **DEAD**

**Earliest failed gate:** Gate‑1 provenance (audit item 8). PQML requires class‑exogenous intra‑class structure shared across disjoint identities — the same premise candidate 225 measured prospectively and found absent (ρ₃₂ = 0.9312 / 0.9287 / 0.9345, all below 1 and far below its locked 1.15 falsifier), and the same premise 176, Exchangeable‑Nuisance Embedding, EFML, CFR, FRAME and CNW all failed on. PQML offers a forecast (its own P(both bars) = 0.40), not positive provenance.

**One decisive mechanism‑level reason:** §2.1 and §1.4(c) rest on **contradictory premises about the same variable**. §2.1 requires viewpoint azimuth to be the dominant within‑class nuisance worth quotienting. `L_prot` requires the quotiented factor to be *class‑independent*. Azimuth is class‑correlated in exactly these benchmarks (CUB: species‑typical posture — waterfowl side‑on, woodpeckers vertical, raptors perched; Cars196: class‑stereotyped dealer/press/auto‑show capture conventions). So routing azimuth into phase incurs a **strictly positive** `L_prot` cost, while routing the train‑time augmentation randomness — RandomResizedCrop scale/offset and RandomHorizontalFlip, class‑independent *by construction at every moment order* — drives `L_inv` and `L_prot` to **exactly zero**. The global optimum of the two novel terms is the augmentation‑noise channel, not the azimuth channel. The term introduced to make the channel zero‑shot‑safe is the term that selects against the method's own stated mechanism.

---

## A. Audit findings

### 1. Raw quotient vs. ε‑smoothed forward pass — **no defect of consequence; no repair needed**

I recomputed everything at ε = 1e−6, L = 32.

| Quantity | Raw | With ε | Verdict |
|---|---|---|---|
| ‖y‖² | 1 exactly | 1 + Lε² = 1 + 3.2e−11 | proposal's "up to O(Lε²)" is **accurate**; identical for every image, so cosine *ranking* is unaffected **exactly** |
| F1 maximal invariant | π(z)=π(z′) ⟺ z′∈T^L·z | `sqrt(u+ε²)` strictly monotone in u ⇒ equal \|ζ\|_ε ⟺ equal \|ζ\| | **exact, unchanged** |
| F2 orbit‑maximised cosine | Σ\|ζ\|\|ζ′\| + ⟨r,r′⟩ = max_θ⟨R_θz,z′⟩ | overstated by Σ_l[√((\|ζ\|²+ε²)(\|ζ′\|²+ε²)) − \|ζ\|\|ζ′\|] ≈ 3e−11 | **exact identity; ε error 5 orders below float32 eps** |
| F3 Δ | Σ2\|ζ\|\|ζ′\|(1−cos Δψ) ≥ 0 | Δ_ε = Σ_l 2(\|ζ\|_ε\|ζ′\|_ε − ε² − Re ζζ̄′) ≥ Δ_raw ≥ 0 | **still exact and non‑negative**; no spurious floor at ζ=0 |

All four displayed gradients are correct, including `∂e/∂ζ = (I − êê^T)/|ζ|_ε` which is **exactly right** with ê = ζ/|ζ|_ε (not an approximation). Only bookkeeping slip: `∂L_inv/∂Δ_ij` omits the 1/|P_same| factor.

Two real but non‑decisive ε caveats:
- `e = ζ/|ζ|_ε` is **sub‑unit** when |ζ| ≲ 1e−6, biasing the circular moments and ν_k downward — irrelevant at working energies, relevant only in the collapsed regime.
- `∂e/∂ζ` is bounded by 1/ε = **1e6**. A plane driven to near‑zero energy receives `L_prot` gradients ~6 orders above typical. Worse: ε² = 1e−12 is **below fp16's smallest subnormal (~6e−8)**, so under any AMP path that computes `z²+z²+ε²` in half precision the smoothing **flushes to zero and the Jacobian is genuinely singular**. The proposal specifies no precision.

**This is a correct subcomponent, not a failed gate.** The frozen object needs no repair here.

### 2. Cheapest solution to `L_base + L_inv + L_prot` — **manufacturable disposable channel confirmed**

Yes, trivially, and it is the *global* optimum of the two novel terms rather than a local one.

`L_base` is computed on `y = π(z)`, which is exactly T^L‑invariant ⇒ **∂L_base/∂ψ ≡ 0**. The only forces on phase are `L_inv` and `L_prot`. Set ψ_il to any class‑exogenous per‑image quantity:

- **`L_inv`** ✓ — independent phases across images give E[1−cos Δψ] = 1, so Δ ≈ 2·(phase energy). Δ* = 0.15 needs only E ≈ 0.075.
- **`L_prot`** ✓ — class‑independence by construction ⇒ moments at the null floor at *every* order.
- **`L_base`** ✓ — identity lives in the moduli and `r`; cost is 480 sign‑free dims instead of 512.

The canonical exploit is not even a learned hash: **the training augmentation supplies the signal for free.** Crop scale/offset and flip are i.i.d. and class‑independent by construction, and are readable from low‑level image statistics (resampling blur, border artifacts). The encoder learns *nothing* about pose, viewpoint, or any shared transformation action, and attains `L_inv = L_prot = 0` exactly — which the intended azimuth solution provably **cannot**, per the decisive reason above.

Under this solution Δ_same ≈ Δ_diff, so the quotient contracts same‑ and different‑class pairs equally and the expected R@1 delta is ≈ 0 or slightly negative (32 non‑negative coordinates compress cosine dynamic range). The forecast +1.5/+1.2 has no mechanism behind it.

**All three mechanism probes pass under the shortcut:** P1 (junk phase transfers perfectly to unseen identities ⇒ Δ̄_unseen ≈ Δ̄_train ⇒ **F4 passes**), P2 (phase responds strongly to flip/scale — the shortcut *is* flip/scale ⇒ **positively confirms** the shortcut), P3 (junk phase is class‑independent ⇒ I(ψ;class) ≈ 0 ⇒ **F5 passes**). The probe set is not diagnostic of the thing it was written to diagnose.

### 3. Identifiability — **not established; D5 is false as stated**

Two different theorems are conflated in §0:

- **T1** — given a *specified* torus T^L, π is the maximal invariant and cosine on π is orbit‑maximised cosine. **True** (verified above).
- **T2** — the objective *discovers* a torus action corresponding to a real nuisance group. **Not established, and refuted by §A.2.**

The torus is not learned at all: §1.1 fixes it to coordinate pairs. Nothing in `L` is a function of L, and nothing maximises group size (contrast Augerino, which adds an explicit group‑size regulariser). "The largest embedding‑isometry group that preserves the label partition" is **maximised nowhere in the objective**; L is a hyperparameter. What is learned is feature routing, not group identification.

Labels + within‑class spread + two circular moments cannot establish that phase means the same thing across images: `L_prot` constrains only per‑plane per‑class *marginal* distributions, and the sole relational quantity `Δ(z_i,z_j)` is a scalar target, not a correspondence. Every method that actually identifies a group supplies something PQML lacks — input‑space action (Augerino), known group (Kaba et al. ICML 2023), paired (x, gx) observations (equivariance losses), or adversarial distribution matching over a parameterised latent group (LieGAN/LaLiGAN). PQML's criterion is *strictly weaker* than distribution matching, which is itself non‑identifying (Locatello et al., ICML 2019 impossibility result — **uncited**).

**D5 explicitly:** its non‑redundancy argument is *correct* (constant phase violates `L_inv`; class‑indexed spread violates `L_prot`), and its warning against writing `L_prot` as a different‑class distance constraint is *correct and sharp*. But it then infers "class‑agnostic coordinate along which same‑class images differ" ⇒ "a nuisance factor." That inference is invalid: removing two points from a solution set does not pin down the remainder. Non‑redundancy ≠ identification.

### 4. Collapse and shortcut arguments — **the hinge value supplies no escape gradient**

At exactly equal phases (ψ_il = ψ_jl):

```
∂Δ/∂ζ_il = 2|ζ_jl|·ê_il − 2ζ_jl = 2|ζ_jl|(cos ψ, sin ψ) − 2|ζ_jl|(cos ψ, sin ψ) = 0
```

**identically, for any magnitudes.** So `∂L_inv/∂ζ = 0` there. `L_inv` sits at its maximum λ_inv·Δ* = 0.15 with **zero gradient** — a plateau, not a saddle you slide off. The same holds at zero plane energy (∂Δ/∂ζ_il ~ 2|ζ_jl|(ê_il − ê_jl) → 0). D1's defence (i) argues about *initialisation*, not *stability*; defence (ii) is excluded by instruction. D2 inherits the identical flaw and **does not notice it** — "collapse maximally violates the hinge" is true and irrelevant, because maximal violation with zero gradient is not escapable.

The sharpest form, and it is exact rather than approximate. In polar coordinates ζ = ρe^{iψ}:

- `L_prot` depends on z **only** through unit phasors ⇒ **∂L_prot/∂ψ is exactly ρ‑independent**.
- `∂Δ/∂ψ_i = 2ρ_iρ_j sin(ψ_i−ψ_j)` ⇒ **`L_inv`'s phase torque is O(ρ²)**.

So `L_prot`'s phase‑concentrating torque is plane‑energy‑independent while `L_inv`'s phase‑spreading torque vanishes quadratically with plane energy. Below a threshold energy (my second‑order estimate: **E ≈ 0.05**, same order as the energy the method needs) the net torque favours concentration; once phase concentrates, `∂Δ/∂ρ_i = 2ρ_j(1−cos Δψ) → 0` removes the restoring force on energy too. **The degeneracy is jointly closed in both coordinates**, and plane energy is under downward pressure from `L_base` (the proposal's own D4 argument) and weight decay. The 0.05 figure is order‑of‑magnitude; the torque‑scaling asymmetry that produces it is exact.

### 5. Finite‑batch null level — **ν_k is ~2.6% conservative (fine); the "exactly zero under the null" claim is false**

With u_i = e_{il}^k unit‑modulus i.i.d. under H0, σ² = 1−|μ|², N = n_c·n_s = 120, and the global mean **including** the class's own n_s = 4 samples:

```
E|m_c − m_global|² = σ²(1/n_s − 1/N) = (σ²/n_s)(1 − 1/n_c)     = 0.9667 σ²/n_s
E[ν_k] = E[(1−|m|²)/n_s], with E|m|² = |μ|² + σ²/N            = 0.9917 σ²/n_s
```

Ratio 1.0259 — **ν_k overstates the null mean by 2.6%**, the conservative (under‑penalising) direction. Exact correction: multiply by (1 − 1/n_c)/(1 − 1/N). Credit for constructing a finite‑sample floor at all; this is close to right.

The real defect is one‑sided hinging of a **fluctuating** statistic. |m_c − m|² is approximately Exp(mean), so E[(X−t)_+] = E·e^{−t/E} ≈ **0.37 ν** at t = ν, per (c,l,k). `L_prot` therefore has strictly positive expectation ≈ 0.09 σ² **under the exact null**, with a gradient that systematically shrinks σ² — a persistent phase‑concentration pressure toward precisely the D1 degeneracy, which item 4 shows is inescapable. §1.4(c)'s "the hinge makes the term exactly zero under the null hypothesis" is wrong: the *statistic's mean* is calibrated, the *hinge* is not. Calibrating to an upper null quantile (~3ν) would repair it — a new proposal.

**Two moments do not imply independence.** Concretely: ψ ~ uniform on {0, 2π/3, 4π/3} + class offset δ_c gives m^(1)_c = m^(2)_c = **0 for every class**, exactly matching global, so `L_prot` = 0 — while the third harmonic e^{3iψ} = e^{3iδ_c} carries δ_c in full. With L = 32 planes that is 32 continuous free parameters of pure class information, invisible to `L_prot`, *rewarded* by `L_inv` (three‑mode structure maximises Δ), and destroyed by the quotient. §7 risk 3 under‑rates this as a residual leak; it is an unconstrained channel. (Mitigating: `L_base` is phase‑blind, so there is no positive *incentive* to use it — the objective merely fails to forbid it.)

### 6. F8 and F4/F5 — **one selective‑reporting rule, one prohibited test‑selected method change**

- **F8 / D4 (discard runs with phase energy > 0.6, "not reported")** — **not a falsifier; selective reporting.** A falsifier rejects the claim; this removes an unfavourable outcome from the record. The filter is correlated with the metric (high phase energy → non‑negativity compression → worse R@1), so conditioning the 5‑seed mean on it biases upward. D4's phrasing makes it a *method* rule, which is worse. Legal form: report all seeds, add a re‑tuned‑Δ* arm as a separately labelled arm.
- **F5 → D3 repair** — **prohibited test‑selected method change.** I(ψ_l; class) on unseen identities requires **test class labels**; its firing *selects* the shipped `L_prot` (moment hinge vs. gradient‑reversal adversary). "Report both" does not cure it: the method conditioned on test data. This violates the stated envelope.
- **F4 (P1 transfer)** — **legal.** It only re‑attributes an observed improvement; it triggers no method change.
- **D1 contingency (ii)** is triggered on a training statistic, so legal in kind, but it mutates a "frozen" method mid‑run. Excluded by instruction — leaving D1 undefended per item 4.
- **F1, F2, F3, F6, F7 are genuine pre‑registered falsifiers** and well chosen. F2 in particular is correctly nominated as the most important.

### 7. Prior art by supervision object and action — **the claimed composition is occupied**

Cited from knowledge; I read only the frozen file and fetched nothing.

Three works with the **same supervision object (identity labels only)** and the **same action (route shared intra‑class variation into a channel excluded from deployment, enforcing independence)** are **absent from the "adversarial" §3 search**:

1. **MIC: Mining Interclass Characteristics for Improved Metric Learning** — Roth, Brattoli, Ommer, ICCV 2019. Class‑discriminative branch + shared intra‑class *characteristics* branch (pose, viewpoint, lighting) from labels alone, decorrelated, characteristics branch dropped at deployment. Evaluated on CUB/Cars/SOP/In‑Shop. This is PQML's premise, action, benchmarks, and deployment convention. **Its omission is material** in a section claiming an adversarial search.
2. **Unsupervised Adversarial Invariance** — Jaiswal, Wu, Abd‑Almageed, Natarajan, NeurIPS 2018. Split into predictive e₁ + nuisance e₂ with adversarial disentanglement; deploy e₁ only.
3. **Invariant‑Equivariant Representation Learning for Multi‑Class Data** — Feige, ICML 2019. Class‑invariant + instance code from class labels only, on the explicit assumption that intra‑class variation is shared across classes.
4. **Locatello et al., ICML 2019** — the identifiability impossibility result that directly governs D5.

The §3 distinctions that **hold**: Phase Collapse (Zarka/Guth/Mallat, ICLR 2022) and scattering use moduli over *fixed wavelet* channels; Augerino learns an *input‑space* group and deploys by averaging; Kaba et al. (ICML 2023) canonicalise a *known* group; LieGAN/LaLiGAN discover latent symmetry by adversarial distribution matching (and thereby possess an identification criterion PQML lacks — PQML is weaker, not stronger); group‑integration retrieval descriptors use hand‑specified groups; tangent distance requires known transformations.

**Narrowest defensible novelty:** *a deployed DML descriptor whose plain cosine equals the orbit‑maximised cosine over a rank‑L torus of embedding isometries, obtained in closed form by per‑plane complex modulus, at exactly the deployed dimension and with exact norm preservation.* I believe this specific deployment‑geometry statement is unoccupied.

**Does it survive identifiability? No.** The F2 identity is true for *any* torus and any routing; its value is entirely contingent on the routing being a real nuisance group, which items 2–3 show is not identified and item 6's decisive reason shows is actively selected against. Stripped of the identification claim, what survives is: *"replace 32 of your 512 coordinates by the ℓ₂ norms of 32 disjoint coordinate pairs of a 544‑D embedding."* That is arm **C3**, which the proposal itself forecasts at ≈ 0.

### 8. Gate‑1 provenance — **negative, not positive**

Candidate 225 is the prospective test of PQML's premise, run on these benchmarks with disjoint identities: ρ₃₂ = 0.9312 / 0.9287 / 0.9345 across seeds 0–2, all **below 1** and far below its locked 1.15 falsifier. Under PQML's own circular model a shared torus implies shared 2‑planes implies a shared within‑class subspace transferring across identity folds — which is what 225 measured and did not find. 176 (cross‑instance nuisance‑tangent quotient), ENE, EFML, CFR, FRAME and CNW all required class‑exogenous intra‑class structure and failed for adverse provenance, missing correspondence, non‑identification, or occupied supervision. PQML fails on the **same three** (missing correspondence, non‑identification, occupied supervision) plus adverse provenance.

The proposal has a principled partial reply — §2.1 predicts that *linear* subspace removal fails on circular nuisance, so 225's ρ₃₂ < 1 is arguably consistent with PQML. I take that seriously, but it does not rescue: ρ₃₂ < 1 says the leading within‑class *subspace itself* fails to transfer, which is stronger than "the linear projection fails to help," and PQML needs the planes to be shared even if the angles are not. *(Conditional on ρ₃₂ measuring subspace transfer as described; I could not read 225's definition.)*

PQML supplies a forecast, and a weak one by its own arithmetic (P(both) = 0.40). That is not Gate‑1 provenance.

### 9. Controls and causal probes — **well designed where they exist; the two most diagnostic arms are unfalsified, and P2 cannot see the claimed mechanism**

- **Good:** C1/C7 (C7 correctly identified as the true baseline), C2 (correctly nominated decisive), C3 (directly measures D1), C6, C8 (interior‑optimum prediction is a real signature).
- **Unfalsified arms:** F1–F8 cover C2, C3, C6, C8, P1, P3 and phase energy. **C4, C5 and C9 have no pre‑registered threshold.** C4 (random vs. same‑class pairs) and C9 (λ_prot = 0) are precisely the arms that test whether the two *novel* terms do anything specific. Under the §A.2 shortcut both predict "no difference" — the outcome that should falsify the identification story is exactly the outcome with no threshold attached to it. This is the single most constructive gap: **C9's prediction ("R@1 on unseen classes drops while training‑class metrics do not") is the cleanest mechanism signature in the document and it carries no falsifier.**
- **P2 cannot identify what it claims.** Horizontal flip is the order‑2 group ℤ/2, not S¹; ±10% scale is a non‑compact ℝ₊ action, not S¹; neither is viewpoint azimuth. Both are *inside the training augmentation distribution*, i.e. they are the shortcut. "Phase response ≫ modulus response" is satisfied by any encoder that writes augmentation nuisance into phase. **P2 can identify only that the channel is non‑degenerate.** A genuine azimuth probe needs azimuth annotation — Cars196 has none, CUB has parts/attributes but no azimuth, and extra annotation is outside the envelope. **The central mechanistic claim is therefore unfalsifiable within the legal envelope.**
- **Line between legal diagnosis and illegal repair:** P1 and P3 *as reported diagnostics* are legal. F5's triggered repair converts P3 into test‑driven selection (item 6).

### 10. Recipe, compute, protocol — **the required screen order is violated and checkpoint selection is unspecified**

**Fails the entry protocol on three of four sub‑requirements:**

- **No In‑Shop screen at all.** §4 states "I forecast CUB and Cars in Lane A only"; In‑Shop appears only as a nuisance example and a cost comparison. The required paired corrected In‑Shop‑first screen is the cheap, high‑power test that exists (n=3, sd ≈ 0.0015 ⇒ SEM ≈ 0.0009 per arm, tighter still paired). PQML instead opens with a two‑dataset frontier chase at self‑forecast P(both) = 0.40. Strictly worse information per unit compute.
- **No raw vs. independently‑selected/final distinction anywhere**, and **no checkpoint‑selection protocol at all** — §6 reports bare means ± SD. This is quantitatively load‑bearing: the corrected local In‑Shop PA reference has raw − final = 0.918062 − 0.915201 = **0.002861**, against a CUB crossing bar margin of **0.0044**. Unstated best‑epoch selection would supply ~65% of the required crossing margin from reporting convention alone.
- **Out‑of‑sample confirmation** is not specified as a step. **Replication** is partially met (CUB and Cars can replicate each other); SOP and In‑Shop are absent.

**Internal contradiction on Δ\*.** §1.7: *"Changing α, the weight decay, or the base loss requires re‑tuning Δ\* on the ablation grid; inheriting Δ\* = 0.15 across such a change is not permitted."* §6's **headline R3 is PQML on PFML with Δ\* = 0.15 inherited**, across a complete change of base loss (piecewise potentials, M = 15 proxies/class vs. α = 32, δ = 0.1). By its own rule the headline arm requires re‑tuning — and the proposal specifies **no validation protocol and no held‑out training‑class split** on which that tuning could legally occur.

**Invented schedule arithmetic.** 200 epochs with γ = 0.5 and step 17 (CUB) gives ⌊200/17⌋ = **11 decays**, lr 1e−4 → 4.9e−8; Cars, 6 decays → 1.6e−6. Training effectively stops around epoch 100. Rescaling the *step* preserves neither the decay count nor the final lr, so it is not a rescaling of the source schedule in any meaningful sense. Compounding this, PFML's published Adam lr is 5e−4 while R0 runs AdamW at 1e−4 — one fifth — so P(R0 ∈ 0.734 ± 0.006) is materially lower than assumed. The **R0 gate is correctly structured** (headline void if the reproduction misses), which is the right methodology; the schedule makes it likely to fire.

**Uncertainty is inconsistent between the bar and the falsifier.** The crossing arithmetic is correct as arithmetic (0.003/√5 = 0.001342; 0.004/√5 = 0.001789; √sum = 0.002236; ×1.96 = 0.0044; bar 0.7384 → 0.742 ✓). But it uses SEM 0.00179 while the R3 90% CI [0.734, 0.757] implies SD ≈ 0.007 — a 4× discrepancy, unreconciled. Neither uses **paired seed‑level differences**, which is the correct statistic for the in‑house R3 − R0 headline. F1 applies a bare point threshold (+0.005) with no uncertainty at all. The stated crossing probabilities (0.55 / 0.50 / 0.65 / 0.40) are mutually consistent as a correlated joint distribution but are **not derivable from the tabled dispersions** — they are more conservative than either reading implies, which is honest but undocumented.

**Compute claim is correct for the method, absent for the programme.** 120²×544 = 7.83 MFLOP vs 120×4.1 GFLOP = 492 GFLOP ⇒ 1.6e−5 ✓. +65,536 params = 0.256% of 25.6M ✓. Deployed dims: (544−64)+32 = 512 ✓; L=128 ⇒ (640−256)+128 = 512 ✓. The 1.00–1.02× overhead claim is credible. But the **total programme** — 9 controls × 5 seeds × 2 datasets × 200 epochs, plus a 6×4×4 ablation grid — is >100 ResNet‑50 200‑epoch runs, and is never costed.

---

## B. Correct subcomponents (preserved independently of the verdict)

These survive and should be carried forward:

1. **F1** — π is a maximal invariant of T^L. Correct, and **exact under ε** (monotonicity of √(u+ε²)).
2. **F2** — ⟨π(z),π(z′)⟩ = max_θ⟨R_θ z, z′⟩. Correct identity; ε error ≈ 3e−11, and cosine *ranking* is affected **exactly zero** because ‖y‖² = 1+Lε² is image‑independent. This is the strongest piece of the proposal and the narrowest defensible novelty (§A.7).
3. **F3** — Δ's closed form and non‑negativity via the reverse triangle inequality. Correct, including under ε.
4. **The absolute‑vs‑ratio insight** — that Δ/‖z_i−z_j‖² is *maximised* by collapse and the absolute form is load‑bearing. Correct, non‑obvious, and worth keeping.
5. **Exact norm preservation at exactly 512‑D with no renormalisation knob.** A genuine Lane‑A‑compliance advantage over branch‑based invariance methods.
6. **§2.1's linear lemma** — rank‑1 gives ρcos(ψ−ψ₀) (still ψ‑modulated), rank‑0 destroys ρ, invertible preserves ψ. **Correct within its stated model** and a valid a‑priori reason to prefer a modulus readout over WCCN. Scope caveat: it constrains a linear head applied to an *already circularly encoded* representation; it does not prove a nonlinear backbone must fail, so "this is the whole mechanism, and it is a theorem" overstates it.
7. **All four displayed gradients**, including the ε‑consistent (I − êê^T)/|ζ|_ε.
8. **ν_k's finite‑sample construction** — 2.6% conservative, correctable by ×(1−1/n_c)/(1−1/N).
9. **D5's non‑redundancy argument** and its warning against writing `L_prot` as a different‑class distance constraint (different‑class pairs at different azimuths *should* contract). Both correct.
10. **The R0 gate structure** and §4's unusually complete disclosure of invented recipe elements.
11. **C2/C3/C6/C7/C8/C9 design**, F2's nomination as most important falsifier, and C9's unseen‑vs‑seen prediction — the cleanest mechanism signature in the document, wasted for lack of a threshold.
12. **Cost accounting** — all figures check out.

---

## C. Uncertainty and scope

- I read exactly the one frozen file and fetched nothing. Prior‑art citations in §A.7 are from knowledge; I am confident about the content and venues of MIC, Jaiswal et al., Feige, Locatello et al., Phase Collapse, Augerino and Kaba et al., but did not re‑verify page‑level claims.
- The **class‑correlation of azimuth** in CUB/Cars is an empirical claim I assert from dataset construction rather than measurement. It is the load‑bearing premise of my decisive reason. Note, however, that the proposal is exposed either way: if azimuth *is* class‑correlated, `L_prot` penalises the target; if it is *not*, the augmentation shortcut still attains a strictly lower auxiliary loss than azimuth and there is no term selecting between them. The proposal never states, tests, or notices that `L_prot`'s validity requires an independence assumption its own §2.1 does not supply.
- The **E ≈ 0.05** degeneracy threshold in §A.4 is an order‑of‑magnitude estimate resting on an exponential‑tail approximation. The torque‑scaling asymmetry that produces it (`L_prot` O(1) vs `L_inv` O(ρ²) in phase; `L_inv`'s radial force → 0 as Δψ → 0) is **exact** and does not depend on that estimate.
- The candidate‑225 inference in §A.8 is conditional on ρ₃₂ measuring transfer of the leading within‑class subspace to disjoint identities as summarised; I could not read its definition.
- I did not verify the PFML/Proxy‑Anchor published numbers, PA+DADA 0.930, or any prompt‑supplied reference.

---

**Adjudication only.** I made no repair and altered no part of the frozen object. Every item above that would require a change — calibrating `L_prot` to an upper null quantile, adding a correspondence or input‑space action to make the torus identifiable, replacing F8 with an all‑seeds reporting rule, removing F5's triggered repair, adding C4/C5/C9 thresholds, specifying checkpoint selection, re‑tuning Δ* for PFML under a legal validation protocol, or running the In‑Shop‑first paired screen — constitutes a **new proposal**, not a revision of this one.
