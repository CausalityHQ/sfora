# POTER — Peaks-over-Threshold Extrapolated Retrieval

**One method. Lane A** (ResNet‑50, 512‑D normalized descriptor, ~224 px, single view, cosine NN, 200 epochs). Everything below is train‑time only; deployment is byte‑identical to a Proxy‑Anchor model.

---

## 0. The claim in one paragraph

Every Lane‑A objective in the primary literature estimates a quantity computed over the *minibatch's* comparison set. The quantity that actually decides R@1 at deployment is a **two‑level extreme**: the maximum, over ~10²–10⁴ **unseen** gallery classes, of the maximum over that class's ~3–80 gallery images, of the cosine similarity to the query. No current loss is a consistent estimator of it, and the gap is not a constant — it is a *quantile* gap of 8× (CUB) to 1543× (SOP) at the class level, multiplied again by the within‑class block size. POTER replaces the in‑batch statistic with a **differentiable, closed‑form, hierarchical peaks‑over‑threshold estimator** of that two‑level extreme, extrapolated from the batch to the deployment gallery scale. The tail parameters are *class‑marginal* statistics — which is precisely why they transfer to disjoint identities — and they are estimated, not learned, so no parameter, network, or inference op is added.

---

## 1. Executable mathematics

### 1.1 Learned objects (complete)

| Object | Shape | Deployed? | LR |
|---|---|---|---|
| θ — ResNet‑50 (ImageNet‑1K init) | — | ✔ | 1e‑4 |
| W — embedding head, no bias | 512×2048 | ✔ | 1e‑4 |
| P = {p_c} — class proxies, L2‑normalized | C×512 | ✘ | 1e‑2 (=100× per PA's official repo) |

**Nothing else is learned.** All EVT quantities (u, σ, ξ, excess quantiles) are closed‑form batch statistics recomputed every step. No decoder, no teacher, no hypernetwork, no second view.

Forward: h = GAP(ResNet50(x)) ∈ ℝ²⁰⁴⁸, z = Wh/‖Wh‖₂ ∈ S⁵¹¹.
Test: resize 256 → center crop 224 → z → cosine NN. No reranking, no test‑time fitting, no gallery statistics estimated from test data.

### 1.2 Base loss — Proxy Anchor, reproduced exactly (Kim et al., CVPR 2020)

With s(x,p) = ⟨z_x, p⟩, P⁺ the proxies of classes present in the batch, X_p⁺ / X_p⁻ the batch samples of / not of p's class:

```
L_PA = 1/|P⁺| Σ_{p∈P⁺} log(1 + Σ_{x∈X_p⁺} e^{−α(s(x,p) − δ)})
     + 1/|P|  Σ_{p∈P}  log(1 + Σ_{x∈X_p⁻} e^{ α(s(x,p) + δ)})
```
α = 32, δ = 0.1; embeddings **and** proxies L2‑normalized (as in the official implementation). Official recipe for ResNet‑50/512‑D: AdamW, backbone+head lr 1e‑4, proxy lr ×100, weight decay 1e‑4, batch 120, 5 warm‑up epochs with the backbone frozen (`--warm 5`), BN frozen (`--bn-freeze 1`), RandomResizedCrop(224, scale 0.16–1) + hflip.

### 1.3 Sampler

Class‑balanced: P_b = 40 classes × K = 3 images, B = 120. (K ≥ 3 is required for the within‑class excess estimator; P_b large gives negative‑class coverage. B matches PA's official 120 exactly, so compute is matched.)

### 1.4 The POTER term

Deployment constants, taken from the **training** split only (never the test gallery):
`C_gal` = number of training classes; `n̄` = mean images per training class.
CUB (100, 58.64) · Cars (98, 82.18) · SOP (11 318, 5.261) · In‑Shop (3 997, 6.475).

For each query i in the batch:

**(a) Class‑level scores.** m_{ic} = ⟨z_i, p_c⟩ for all c ≠ y_i — all C classes, every step (already computed by L_PA).

**(b) Threshold as an order statistic.** k = clip(⌈0.05(C−1)⌉, 4, 64). Let u_i = m_{i,(k+1)} (the (k+1)‑th largest). Exceedances x_{i,1..k} = m_{i,(1..k)} − u_i ≥ 0. Then ζ_u = k/(C−1) is **exact and constant** — no gradient path, no way to game it.

**(c) GPD fit by probability‑weighted moments** (closed form, differentiable). With x ascending,
```
â₀ = (1/k) Σ_j x_(j)              â₁ = (1/k) Σ_j x_(j)·(k−j)/(k−1)
ξ̂_i = 2 − â₀/(â₀ − 2â₁)           σ̂_i = 2â₀â₁/(â₀ − 2â₁) = â₀(1 − ξ̂_i)
```
(Verified: ξ=0 ⇒ â₀=σ, â₁=σ/4 ⇒ ξ̂=0, σ̂=σ; ξ=0.5 ⇒ â₀=2σ, â₁=σ/3 ⇒ ξ̂=0.5. ✔)

*Shape* is pooled and frozen: **ξ̃ = clip(median_i ξ̂_i, −0.5, 0.2), stop‑gradient.**
*Scale* is per‑query with empirical‑Bayes shrinkage, **gradient kept**:
σ̃_i = clip((1−γ)σ̂_i + γ·mean_j σ̂_j, σ_min, ∞), γ = 0.5, σ_min = 1e‑3.

**(d) Within‑class excess law (the second level).** For each negative class c in the batch,
e_{ic} = max_{j: y_j=c} ⟨z_i,z_j⟩ − m_{ic} ≥ 0. Pool all B·(P_b−1) values → empirical quantile function Q_K.
These are maxima of K draws; the gallery class has n̄. Under within‑class exchangeability, H_{n̄} = H_K^{n̄/K}, so
```
Ê_r = Q_K( ((r−0.5)/R)^{K/n̄} ),   r = 1..R,  R = 64.
```

**(e) Positive‑side block‑max correction** (same law, used once): the deployment gallery also holds n̄ positives, not K−1. Add the detached median shift
`Δ⁺ = Q_{K−1}(0.5^{(K−1)/n̄}) − Q_{K−1}(0.5)  ≥ 0`, stop‑gradient.
`s⁺_i = (1/β) log Σ_{j∈Pos(i)} e^{β⟨z_i,z_j⟩} + Δ⁺`, β = 16 (soft‑max ⇒ *easiest* positive; see D5).

**(f) Extrapolated expected number of gallery classes that beat the positive.**
```
Λ_i = θ · C_gal · ζ_u · (1/R) Σ_{r=1}^{R} min[ 1, (1 + ξ̃ (s⁺_i − Ê_r − u_i)/σ̃_i)_+^{−1/ξ̃} ]
ℓ_i = log(1 + Λ_i)
```
θ = 1 (conservative; fixed constant, **not** estimated — see D4). At ξ̃→0 this collapses to the interpretable closed form
`Λ_i = θ C_gal ζ_u · 𝔼[e^{E/σ̃_i}] · e^{−(s⁺_i − u_i)/σ̃_i}` — a softplus whose **temperature is the fitted tail scale** and whose **offset log(θ C_gal ζ_u 𝔼[e^{E/σ̃}]) is the deployment‑scale term absent from every existing loss.**

**(g) Total.** `L = L_PA + λ(t)·(1/B) Σ_i ℓ_i`, λ = 1.0, λ(t) = λ·min(1, t/10) (10‑epoch warm‑up: the tail fit is meaningless before the embedding organizes).

### 1.5 Gradient paths (explicit)

- ∂ℓ/∂s⁺ < 0 → lifts the *easiest* positive; flows to z_i and positives via LSE weights.
- ∂ℓ/∂u_i > 0 → pushes the (k+1)‑th proxy score down; flows to z_i and p_{(k+1)}.
- ∂ℓ/∂σ̃_i > 0 (since s⁺ > u + Ê in regime) → **compresses the spread of the top‑k class scores**; flows to m_{i,(1..k)}, hence to z_i and the top proxies. This is the dimension‑expansion pressure (D2).
- ∂ℓ/∂Ê_r > 0 with weight ∝ (1+ξ̃(s⁺−Ê_r−u)/σ̃)^{−1/ξ̃−1}, exponentially larger for large Ê → **anisotropic within‑class contraction**, only on the image realizing a negative class's max *along the query direction*.
- Stop‑gradient: ξ̃, θ, ζ_u, Δ⁺, C_gal, n̄, and all index selections (standard subgradient).

### 1.6 Full hyperparameter list

α=32, δ=0.1, β=16, k=clip(⌈0.05(C−1)⌉,4,64), γ=0.5, ξ∈[−0.5,0.2], θ=1, R=64, λ=1.0, T_warm=10, σ_min=1e‑3, B=120, K=3, P_b=40. Optimizer: AdamW, 200 epochs, cosine 1e‑4→1e‑6, wd 1e‑4, AMP fp16.

**λ is operational, not cosmetic.** ℓ's gradient magnitude is O(1/σ̃) ≈ 50 at convergence — the same order as α=32. It competes with L_PA and interacts with weight decay through the head's effective scale. λ must be swept ∈ {0.25, 0.5, 1, 2} and reported with sensitivity. **All hyperparameter selection uses a class‑disjoint validation split carved from the training classes** (e.g. CUB: train on classes 1–80, validate on 81–100), then a single retrain on all training classes and one test evaluation.

---

## 2. Causal zero‑shot error mode + degeneracy attack

### 2.1 The error mode: the operative quantile gap

Failure of R@1 = at least one of C_gal−1 unseen classes has an image above the best positive. Inverting the published operating points (Λ* = −ln R@1, q* = Λ*/(C_gal−1)):

| | R@1 ref | Λ* | class‑level q* | batch resolution 1/(P_b−1) | gap |
|---|---|---|---|---|---|
| CUB | .734 | .309 | 3.1e‑3 | 2.6e‑2 | 8× |
| Cars | .927 | .0758 | 7.8e‑4 | 2.6e‑2 | 33× |
| In‑Shop | .930 | .0726 | 1.8e‑5 | 2.6e‑2 | 1408× |
| SOP | .829 | .1875 | 1.7e‑5 | 2.6e‑2 | 1543× |

And the *image*‑level quantile is n̄× finer again (58.6× on CUB).

The consequence is structural, not incidental:
- **Pair‑based losses** (MS, contrastive, hardest‑in‑batch) see P_b−1 = 39 negative classes. They carry zero information about the operative region.
- **Proxy‑based losses** see all C classes, so their *class‑level* gap is small (and ~zero on SOP, where C_train ≈ C_test). But a proxy is a **first‑moment** summary: `∂L_PA/∂z ∈ span(P)`. It never constrains `max_{x∈c}⟨z_q, z_x⟩` — the actual top‑1 competitor.

**Neither family estimates the two‑level extreme.** That is the mode POTER targets.

**Why this is a *zero‑shot* mechanism and not a generic retrieval trick.** The two objects POTER estimates — the upper‑tail law (ξ, σ) of class scores and the within‑class excess law — are **class‑marginal**: they are properties of the *ensemble* of identities, not of any identity. Under the exchangeability the disjoint‑split protocol already presumes, estimates from training classes are valid for unseen classes. A per‑class learned margin, a per‑class proxy geometry, or a per‑class variance model is class‑*conditional* and provably cannot transfer to identities with no parameters.

### 2.2 Attack on the cheapest degeneracies

**D1 — Collapse (all z equal).** Then u=s⁺, σ̂→σ_min, all excesses 0 ⇒ Λ = θ·C_gal·ζ_u = k ⇒ ℓ = log(1+k) ≈ 1.8, near its maximum; L_PA is simultaneously maximal. Not a stationary point. ✔

**D2 — Drive σ̃→0 to blow up the exponent.** This requires the top‑k class scores to be flat for *every* query. For proxies in generic position on S^{d−1}, the expected top‑k spread obeys 𝔼[m_(1) − m_(k+1)] = Θ(√(log(C/k)/d_eff)), where d_eff is the participation ratio of the embedding covariance. Hence
```
σ̃ ≳ c·√(log(C/k)) / √(d_eff)
```
**Minimizing σ̃ is, up to constants, maximizing effective dimensionality.** This is not a degeneracy; it is a *derived* second mechanism, and it targets a known pathology: because ∂L_PA/∂z ∈ span(P) (rank ≤ C ≤ 100 on CUB, out of d = 512), the head's components orthogonal to span(P) receive only weight decay and are actively deleted. POTER supplies the missing gradient there, with an explicit rate, without a bolted‑on spectral penalty. ✔

**D3 — Tighten only the sampled K images.** e_{ic} is a max over a uniformly resampled K‑subset, refreshed every epoch, invisible to the model. Making 𝔼[e^{(K)}] small for all draws ⟺ making the class's excess law stochastically smaller. No draw‑specific shortcut exists. ✔

**D4 — Game ξ̂ or θ so the fitted upper endpoint falls below s⁺ (loss exactly 0).** Blocked by construction: ξ̃ is median‑pooled, clipped to ≥ −0.5 (endpoint ≥ u + 2σ̃), and **stop‑gradient**; θ is a fixed constant 1, deliberately *not* estimated. I flag this explicitly: a learned or estimated extremal index would be the single most exploitable degeneracy in this design, and I removed it at the cost of a conservative (over‑large) effective gallery. ✔

**D5 — Inflate s⁺ by within‑class collapse.** POTER's positive term is a soft **max** (easiest positive), so gradient on the hardest positive decays as e^{−β(s⁺−s_j)} — strictly *weaker* than hardest‑positive mining, the standard driver of class collapse. And the within‑class term POTER *adds* is anisotropic: ∂ℓ_i/∂z_j ≠ 0 only for the j realizing a negative class's max, and only along the component of z_j toward z_i. **Within‑class variation orthogonal to all query directions in the batch is untouched.** POTER penalizes exactly the within‑class variation that creates cross‑class extremes and nothing else — a checkable distinction from every variance‑shrinking regularizer. ✔

**D6 — Shrink ζ_u.** ζ_u = k/(C−1) by definition of an order‑statistic threshold; no gradient path. ✔

**Stated model risk (not a degeneracy but a misspecification):** the hierarchical model assumes m_c ⟂ E_c and within‑class independence for H_K → H_{n̄}. SOP/In‑Shop contain near‑duplicate photos of the same product, which violates the latter and *inflates* Ê. Control C10 audits this.

---

## 3. Adversarial novelty search

**Inside DML**

1. **Proxy Anchor** (CVPR 2020) — the base. Its α, δ are global constants independent of gallery size and of the within‑class excess law, so it cannot represent the estimated quantity.
2. **Cross‑Batch Memory** (Wang et al., CVPR 2020) — obtains many negatives by *storing* embeddings; POTER obtains the same quantity by *parametric extrapolation*, with no memory, no slow‑drift bias, and — decisively — reaches quantiles finer than any storable population and models the within‑class maximum, which a memory still only samples.
3. **Recall@k Surrogate / "Three Things to Know about DML"** (Patel, Tolias, Matas) — enlarges the batch until it nearly *is* the training set; POTER extrapolates to a quantile no batch can resolve, and to the *unseen* gallery rather than the training set.
4. **VSE++ / "Hard negatives are hard, but useful"** — argue the in‑batch hardest negative approximates the global hardest; POTER quantifies that approximation's bias with EVT instead of assuming it away.
5. **MS / Circle / SoftTriple / PFML (Potential‑Field DML, Bhatnagar & Ahuja, CVPR 2025)** — reweight *observed* interactions with fixed‑shape kernels; POTER's weights are the fitted tail's implied exceedance density and it carries a deployment‑scale offset that none of them has.
6. **DADA** (AAAI 2024) — closes a sample‑vs‑proxy domain gap via augmentation‑driven adaptation; POTER changes *what the objective estimates*, not how proxies and samples are aligned (hence they compose).
7. **AdvRF** (ICCV 2025) — injects category‑agnostic information through a reconstruction + distillation system; POTER injects no new information source, it re‑estimates the risk from the same data with no added machinery.
8. **Threshold‑Consistent Margin loss** (arXiv 2307.04047) — equalizes *operating thresholds* across classes for fixed‑threshold deployment; POTER targets the top‑1 ranking extreme and assumes no global threshold.
9. **Sampling Matters / distance‑weighted sampling** (Wu et al., ICCV 2017) — uses the analytic *bulk* distance density on S^{d−1} to debias *sampling*; POTER uses the empirical *upper tail* to debias the *loss*, and EVT's premise is precisely that the bulk does not determine extremes.
10. **ρ‑spectrum regularization** (Roth et al., ICML 2020) / **Anti‑Collapse coding‑rate loss** — add explicit spectral/volume terms; in POTER the dimension‑expansion pressure is a *derived consequence* of minimizing σ̃ (D2), not a separately weighted regularizer.

**Outside DML**

11. **Meta‑Recognition / W‑score** (Scheirer et al., TPAMI 2011), **OpenMax** (CVPR 2016), **Extreme Value Machine** (TPAMI 2018) — fit Weibull/EVT to score tails **at test time** to predict success or reject unknowns; POTER uses EVT only inside the training gradient and changes no test‑time operation (test‑time EVT calibration would be banned reranking in this protocol).
12. **EVEREST** (arXiv 2601.19022, rare‑event time‑series) — a learned GPD "extreme‑value head" as a train‑time auxiliary with a single deployed head; POTER has **no auxiliary head and no learned EVT parameters** (they are closed‑form batch statistics), and its extrapolation is over *population size* (batch → gallery), not over event rarity in a fixed population.
13. **Logit adjustment for long‑tail classification** (Menon et al., ICLR 2021) — log‑prior offsets correcting *class‑frequency* imbalance in a closed label set; POTER's log(θ C_gal ζ_u 𝔼[e^{E/σ̃}]) offset corrects the *population‑size* mismatch between the training comparison set and an open, unseen gallery.
14. **Extremal index / automatic declustering** (Ferro & Segers, JRSS‑B 2003) — a scalar correction for clustered extremes; POTER replaces it with the *exact* hierarchical decomposition (class tail × within‑class excess), which is the true dependence structure of an identity‑clustered gallery, is differentiable, and is directly supervisable.
15. **Rare‑event importance sampling** — estimates rare probabilities by tilting the sampler; POTER uses the asymptotic parametric family (Pickands–Balkema–de Haan) so the estimate is available in closed form inside the step.

**Claimed new combination:** a differentiable, closed‑form, *hierarchical* peaks‑over‑threshold estimator of the deployment‑gallery‑scale top‑1 failure probability, used as the training objective for zero‑shot metric learning, with the within‑class excess law block‑max‑extrapolated from K in‑batch samples to n̄ gallery items on both the negative and positive sides.

---

## 4. Decisive matched‑compute controls

Identical epochs, batch, augmentation, optimizer, deployed architecture. 5 paired seeds.

| | Control | What it kills |
|---|---|---|
| C0 | PA reproduced under **my** 200‑epoch cosine schedule | Establishes the only baseline I may build on |
| C1 | **No extrapolation**: C_gal ← P_b−1, n̄ ← K | Isolates extrapolation from the loss's functional form |
| C2 | **Global temperature**: γ=1, ξ̃≡0 | Reduces to batch‑estimated adaptive temperature + constant offset |
| C3 | **No second level**: Ê_r ≡ 0, Δ⁺=0 | Class‑level‑only tail loss |
| C4 | **Tuned constant offset** replaces log(θC_galζ_uΦ), swept | ⚠ *The dangerous one.* If a scalar recovers the gain, the estimator story is dead and this is just a better margin |
| C5 | **XBM‑matched**: PA+XBM sized to see θ·C_gal negative classes | Extrapolation vs. observation |
| C6 | **Batch ladder** B ∈ {60,120,240,480} | Mechanism predicts POTER−PA gap **shrinks** with B; a regularizer predicts a flat gap |
| C7 | **Test gallery ladder**: 1/16, 1/4, 1/1 of test classes | Mechanism predicts gap **grows** with gallery; feature improvement predicts flat |
| C8 | **Random C_gal** ~ U[10,10⁵] per step | If it matches, the value is irrelevant and this is offset noise |
| C9 | **σ̃ detached** + report effective rank / participation ratio vs PA | Separates the D2 dimension‑expansion mechanism from the offset |
| C10 | Measure corr(m_c, E_c) on the class‑disjoint val split | Audits the independence assumption; |corr|>0.2 ⇒ report as misspecified |

**C6 ∧ C7 are the decisive pair**: a two‑dimensional signature (gap ↓ in batch size, gap ↑ in gallery size) that no reweighting, regularization, or augmentation alternative predicts.

---

## 5. Frozen forecasts, Lane A only

**Mechanism‑derived forecast structure.** Δ(R@1) = κ · [ ln(C_gal/(P_b−1)) + ln(n̄/K) ], κ = 0.41 points/nat (single free constant, anchored on CUB).

| | ln(C_gal/39) | ln(n̄/3) | total nats | predicted Δ |
|---|---|---|---|---|
| CUB | 0.931 | 2.972 | 3.903 | +1.60 |
| Cars | 0.911 | 3.310 | 4.221 | +1.73 |
| In‑Shop | 4.630 | 0.770 | 5.400 | +2.21 |
| SOP | 5.670 | 0.561 | 6.231 | +2.55 |

**Frozen numbers (5 seeds each):**

| | PA repro (C0) | **PA+POTER** | PA+DADA repro | **PA+DADA+POTER** | Reference |
|---|---|---|---|---|---|
| CUB | .700 ± .005 | **.716 ± .006** | .727 ± .005 | **.737 ± .006** | PFML .734 ± .003 |
| Cars | .882 ± .006 | **.899 ± .007** | .919 ± .006 | **.929 ± .007** | PFML .927 ± .003 |
| SOP | .800 ± .003 | **.826 ± .004** | .808 ± .003 | **.823 ± .004** | PFML .829 ± .002 |
| In‑Shop | .908 ± .004 | **.930 ± .005** | .928 ± .004 | **.941 ± .005** | PA+DADA .930 (seeds unreported) |

(Stacked rows assume 60 % additivity of POTER's Δ on top of DADA — an assumption, flagged.)

**Explicit frontier arithmetic — stated plainly, including where it fails:**

- **In‑Shop: crosses.** .941 vs .930 ⇒ **+1.1**, ≈ 2.2σ of my own seed spread. The reference's uncertainty is unreported, which weakens the comparison; I claim crossing only against the point estimate.
- **CUB: parity, not crossing.** .737 vs .734 ⇒ **+0.3**, ≈ 0.4σ. I do **not** claim a CUB frontier crossing.
- **Cars: parity, not crossing.** .929 vs .927 ⇒ **+0.2**. Not a crossing.
- **SOP: does not cross.** .823 vs .829 ⇒ **−0.6**. The mechanism predicts SOP's *largest* Δ, yet the stacked base is too weak; both statements can be true and I report the negative one.
- Standalone PA+POTER crosses nothing (it beats DADA's matched‑cost SOP row .810 by +1.6 but sits below PFML everywhere).

I am **not inheriting PFML's frontier**. PFML is quoted as an external reference; my baselines are my own reproductions under my own 200‑epoch cosine schedule.

**Pre‑registered falsification thresholds**

- **F1** PA+POTER Δ on CUB < +0.6 (≈1σ) ⇒ method dead.
- **F2** C6: if gap(B=480) ≥ gap(B=120) ⇒ extrapolation mechanism falsified, regardless of headline numbers.
- **F3** C7: if gap at 1/16 gallery ≥ gap at full gallery ⇒ mechanism falsified.
- **F4** C4: if one tuned constant offset recovers ≥75 % of the gain on all four datasets ⇒ estimator story falsified; report as a margin heuristic.
- **F5** If measured Δ is not monotone in the predicted nat ordering (SOP > In‑Shop > Cars > CUB) ⇒ the derived forecast is falsified even if absolute numbers are good.
- **F6** C10: |corr(m,E)| > 0.2 ⇒ report the model as misspecified.

**Known softening of my own forecast:** the positive‑side correction Δ⁺ partially cancels the negative‑side n̄‑extrapolation on CUB/Cars (large n̄). The residual asymmetry is real (the negative side integrates a convex function of the excess law over C_gal classes; the positive side takes one median), but κ is likely optimistic on CUB/Cars specifically. My CUB/Cars forecasts should be read as upper‑middle estimates.

---

## 6. Cost, benchmark and contamination risk

**Training cost.** Extra per step: the C×B proxy score matrix (already computed by L_PA), a top‑k selection O(BC), PWM O(Bk log k), and R=64 quadrature points per query (7 680 scalar ops/batch). No extra forward or backward pass, no extra view, no stored embeddings, no extra parameters. Expected **≤1.03× epoch time, ≤1.01× peak memory** versus PA — comparable to DADA's stated 1.06×/1.01×.

**Deployment cost.** Identical to PA: one ResNet‑50, one 512‑D descriptor, one view, cosine NN. **Zero inference overhead.** No test‑time EVT (which would be reranking and is prohibited here).

**Contamination.** C_gal and n̄ are read from the **training** split, never the test gallery — this is deliberate and non‑negotiable; using test‑split values would be test‑gallery fitting. Their training and test values happen to be near‑identical under the standard splits, which is a property of the splits, not something I exploit. ImageNet‑1K/Cars–CUB semantic overlap is a pre‑existing property of the whole lane, not introduced by this method.

**Benchmark risk.** CUB (5 924) and Cars (8 131) test images give ~±0.5 seed noise on R@1; a +1.6 claim requires paired seeds and 5 runs minimum. In‑Shop query/gallery protocol and SOP R@1 conventions must be pinned to the same code path for baseline and method.

---

## 7. Unresolved source ambiguities (stated, not hidden)

1. **PA's ResNet‑50/512‑D numbers are an inference, not a primary reading.** PA's main tables use BN‑Inception. I back‑computed CUB .697 / Cars .877 from DADA's stated "+3.2 / +4.4 over PA on ResNet‑50". If DADA's PA rows differ from the official recipe, my C0 target moves. C0 exists precisely to settle this.
2. **PA paper vs official repo disagree on schedule** (repo: 60 epochs, step decay at 5/10; the lane specifies 200). I run 200 + cosine for *both* PA and POTER and will additionally report PA at the official 60/step, taking the better PA as the baseline.
3. **PA's sampler images‑per‑class** is set in the repo, not the paper. I assume balanced sampling and state K=3, P_b=40 as my choice, matched across all arms.
4. **PFML's recipe** (optimizer, epochs, sampler, augmentation) is not established from primary source in this session — only the title/venue (Bhatnagar & Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025) and the numbers supplied to me. PFML is therefore an external reference, never a reproduced baseline, and I make no claim about matched cost against it.
5. **PA+DADA In‑Shop .930** has unreported seed count and uncertainty (given). My In‑Shop crossing claim is against a point estimate only.
6. **Extremal‑index treatment.** I set θ=1 rather than estimating it. This is conservative (over‑counts independent extremes) and is a deliberate degeneracy block, but it means POTER slightly over‑penalizes when visually similar unseen classes cluster. C‑ablation θ∈{0.5,1} quantifies it.

---

## What I am and am not defending

I defend: the two‑level extreme is the correct estimand, no existing Lane‑A loss estimates it, POTER does so in closed form at ≤1.03× cost with an unchanged deployment, the degeneracies are blocked by construction rather than by hyperparameter luck, and the C6/C7 signature makes the mechanism claim falsifiable independently of the leaderboard.

I do not defend: a CUB or Cars frontier crossing (forecast parity), or an SOP crossing (forecast below). The single decisive crossing I forecast is In‑Shop. If F2 or F3 fails, the method should be reported as a margin heuristic regardless of how the numbers land.

Sources:
- [Potential Field Based Deep Metric Learning (PFML), CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- [Towards Improved Proxy-Based DML via Data-Augmented Domain Adaptation (DADA), AAAI 2024](https://arxiv.org/pdf/2401.00617)
- [Adversarial Reconstruction Feedback (AdvRF), ICCV 2025](https://arxiv.org/abs/2507.21742)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
- [Proxy-Anchor official implementation (recipe)](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Cross-Batch Memory for Embedding Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf)
- [Recall@k Surrogate Loss with Large Batches and Similarity Mixup](https://arxiv.org/pdf/2108.11179)
- [Three Things to Know about Deep Metric Learning](https://arxiv.org/abs/2412.12432)
- [Hard negative examples are hard, but useful, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590120.pdf)
- [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/pdf/1707.05612)
- [Threshold-Consistent Margin Loss for Open-World Deep Metric Learning](https://arxiv.org/abs/2307.04047)
- [Anti-Collapse Loss for Deep Metric Learning Based on Coding Rate Metric](https://arxiv.org/pdf/2407.03106)
- [Meta-Recognition: The Theory and Practice of Recognition Score Analysis](https://www.wjscheirer.com/projects/meta-recognition/)
- [Towards Open Set Deep Networks (OpenMax)](https://arxiv.org/pdf/1511.06233)
- [EVEREST: Evidential, Tail-Aware Transformer for Rare-Event Time-Series Forecasting](https://arxiv.org/html/2601.19022)
- [Long-tail Learning via Logit Adjustment](https://arxiv.org/pdf/2007.07314)
- [Extremal index estimation and automatic declustering (texmex)](https://rdrr.io/rforge/texmex/man/extremalIndex.html)
- [Beyond Seen Bounds: Class-Centric Polarization for SDG-DML](https://arxiv.org/html/2601.09121)
