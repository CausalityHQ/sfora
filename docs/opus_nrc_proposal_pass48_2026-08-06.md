I did not inspect the filesystem, git history, or any repository/process state. I used only `ToolSearch` (solely to load the schemas for `WebSearch`/`WebFetch`, which were deferred and otherwise uncallable) and then only public web search/fetch. No delegation, no subagents, no consultations.

---

# Proposal: **NRC — Nuisance-Rotor Coding**

*A train-time method in which the deployed retrieval descriptor is the maximal invariant of a fixed, parameter-free torus action indexed by the augmentation sampler's own parameters.*

**Lane: A** (ResNet-50, 512-D normalized descriptor, ~224², single-view cosine, 200 epochs). All forecasts and comparisons below are Lane A only.

---

## 0. One-paragraph statement of the mechanism

Split the descriptor head's output into 512 planes of R². The **radii** of those planes are the (only) thing the metric loss touches and the (only) thing deployed. The **angles** are forced to rotate by a *prescribed, unlearned, multi-octave rotary code* of the augmentation parameters the dataloader already sampled. Because the metric-loss gradient on each plane is exactly radial and the rotor-loss gradient on each plane is exactly tangential, the auxiliary signal cannot alter the descriptor's value at the head — yet it is read from *the same* 1024 coordinates the descriptor is computed from, so the trunk cannot serve it with a disjoint set of channels. The result is dense, class-independent, per-sample supervision that keeps the fine appearance evidence a label loss discards, delivered into a subspace that is exactly projected away at test time.

---

## 1. Executable mathematics

### 1.1 Network and deployed descriptor

* Backbone `φ_θ`: ResNet-50, ImageNet-1K init, 224², GAP over conv5 → `φ_θ(x) ∈ R^2048`.
* **Ledger head** (train and test): `u = W_u φ_θ(x) + b_u ∈ R^1024`, `W_u ∈ R^{1024×2048}`.
  Reshape to `m = 512` planes: `u = (u_1,…,u_512)`, `u_j ∈ R^2`. Write `r_j = ‖u_j‖_2`, `θ_j = atan2(u_{j,2}, u_{j,1})`, `û_j = u_j/(r_j+ε)`, `ε = 1e−6`.
* **Descriptor head** (train and test): `g = W_g ψ(r) + b_g ∈ R^512`, `W_g ∈ R^{512×512}`, `ψ(r)_j = log(1 + r_j)` elementwise (fixed monotone compressor; removes the positive-orthant restriction of raw radii and bounds dynamic range).
* **Deployed**: `e(x) = g(x)/‖g(x)‖_2 ∈ S^511`. Retrieval = cosine NN, one view, one model, 512-D. Phases are never computed at test.

### 1.2 The nuisance group `A` and the prescribed rotor `ρ`

The dataloader samples augmentation parameters; NRC merely **records** them.

`t = (log σ, log a, β, γ, ς, h, f) ∈ A`, where σ = RandomResizedCrop area scale, a = aspect ratio, (β,γ,ς) = log brightness/contrast/saturation multipliers, h = hue rotation angle ∈ S¹, f ∈ Z/2 = horizontal flip.

`A ≅ R^5 × S^1 × Z/2` is **abelian**: photometric multipliers compose multiplicatively (→ additively in log), nested crop scales/aspects multiply, hue angles add mod 2π, flips add in Z/2.

Standardize: `ξ(t) = (log σ/s_σ, log a/s_a, β/s_β, γ/s_γ, ς/s_ς) ∈ R^5`, the `s_•` being closed-form standard deviations of the sampler, computed once.

**Fixed rotor code.** For `j−1 = 64k + q`, `k ∈ {0..7}`, `q ∈ {0..63}`:

* `λ_k = 2^{k−4}` (eight octaves, 1/16 … 8),
* `v_q ∈ R^5` = row `q` of the 64×64 Sylvester–Hadamard matrix, first 5 columns, divided by √5 (so `‖v_q‖ = 1`),
* **`ω_j = π λ_k v_q ∈ R^5`**, **`n_j = k+1 ∈ {1..8}`**, **`ε_j = q mod 2`**.

```
φ_j(t) = ⟨ω_j, ξ(t)⟩ + n_j·h + π·ε_j·f
ρ(t)   = blockdiag( R(φ_1(t)), …, R(φ_512(t)) ),   R(φ)=[[cosφ,−sinφ],[sinφ,cosφ]]
```

`ρ : A → T^512 ⊂ SO(1024)` is an **exact continuous group homomorphism** — `ρ(t)ρ(t′)=ρ(t+t′)`, `ρ(0)=I` — because `φ_j` is linear on the `R^5` factor, integer-winding on `S^1`, and Z/2-valued on the flip. **`ρ` has zero learnable parameters**; `ω_j, n_j, ε_j` are deterministic (not random), and no plane is rotor-inert (`ω_j ≠ 0` and `n_j ≠ 0` for all `j`).

### 1.3 Views per step

For each image `x_i` draw `t^{(1)}, t^{(2)}, t^{(3)} ~ Aug`. Render three views:

* `v1 = a_{t^{(1)}}(x_i)`, `v2 = a_{t^{(2)}}(x_i)`,
* `v3 = a_{t^{(13)}}(x_i)` where `t^{(13)} = t^{(1)} + t^{(3)}` is formed **in parameter space** and rendered once (composed area scale clamped to ≥0.05; composed photometric multipliers clamped to [0.3, 2.0]; hue mod 2π; flip XOR). Because `t^{(13)}` has twice the variance of a single draw, `v3` probes the code at parameter values **outside the single-view support** — this is the homomorphism *extrapolation* test, not a second copy of the equivariance test.

### 1.4 Losses

**(L1) Metric loss — descriptor only, view 1 only** (so the metric supervision budget exactly matches the baseline). Proxy-Anchor (Kim et al., CVPR 2020), reproduced verbatim, with `s(x,p) = ⟨e(x), p/‖p‖⟩`:

```
L_PA = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{x∈X_p⁺} e^{−α(s(x,p)−δ)})
     + (1/|P|)  Σ_{p∈P}  log(1 + Σ_{x∈X_p⁻} e^{ α(s(x,p)+δ)})
α = 32,  δ = 0.1,  one proxy/class,  proxies ~ N(0,I) then normalized,  proxy LR ×100
```

**(L2) Rotor-ledger loss — the mechanism.** Scale-free, per plane:

```
L_rot = (1/m) Σ_j [ 1 − ⟨ û_j^{(1)} , R(φ_j(t^{(1)}−t^{(2)})) û_j^{(2)} ⟩ ]
```

**(L3) Homomorphism/composition loss.**

```
L_comp = (1/m) Σ_j [ 1 − ⟨ û_j^{(3)} , R(φ_j(t^{(13)}−t^{(2)})) û_j^{(2)} ⟩ ]
```

**(L4) Descriptor invariance.** `L_inv = 1 − ⟨e(v1), e(v2)⟩` (gradient through both branches; no stop-grad).

**(L5) Plane-occupancy guard.** With `r̄_j` the batch mean of `r_j / Σ_k r_k`:
`L_occ = (1/m) Σ_j [max(0, τ − m·r̄_j)]²`, `τ = 0.25` (one-sided, inactive above 25% of the uniform share).

**Total**

```
L = L_PA + λ_r (L_rot + L_comp) + λ_i L_inv + λ_o L_occ
λ_r = 1.0 (linear ramp 0→1 over epochs 0–5), λ_i = 0.5, λ_o = 0.1
```

### 1.5 The exact gradient-path claim (this is the novelty, stated formally)

* `∂L_PA/∂u_j` and `∂L_inv/∂u_j` and `∂L_occ/∂u_j` flow only through `r_j = ‖u_j‖`, and `∂r_j/∂u_j = û_j` ⇒ **purely radial**.
* `L_rot, L_comp` are functions of `û_j` only, and `û_j` is 0-homogeneous ⇒ `∂L_rot/∂u_j ⟂ u_j` ⇒ **purely tangential**. Equivalently `∂L_rot/∂(log r_j) ≡ 0`.

Therefore at every step, every plane: `⟨∂L_PA/∂u_j , ∂L_rot/∂u_j⟩ = 0`. The ledger provably cannot push nuisance content into the descriptor *through the head*; its only route to the descriptor is the shared trunk `φ_θ`, which is exactly the route we want and which `L_inv` polices.

**Proposition (scope stated honestly).** `r` is a maximal invariant of the torus action `ρ` on `(R²)^512`, and `e = normalize(W_g ψ(r) + b_g)` is a maximal invariant whenever `rank(W_g)=512` (not enforced; a measurable, not a guaranteed, property). Consequently, if the ledger is satisfied exactly then the nuisance state is *represented* in the phases and the descriptor is *unaffected by the code*. It does **not** follow that nuisance is only in the phase — the trunk could additionally encode nuisance in `r`. That residual is what `L_inv` and control C6 address; I do not claim it away.

### 1.6 Full recipe (Lane A)

ResNet-50 (BN trainable), 512-D, 224²; RandomResizedCrop(224, scale=(0.16,1.0), ratio=(3/4,4/3)), ColorJitter(0.4,0.4,0.4,0.1), RandomHorizontalFlip; eval = resize 256 / center-crop 224. AdamW, lr 1e-4 (CUB/Cars), 6e-4 (SOP/In-Shop), weight decay 1e-4, proxy lr ×100, 200 epochs, lr ×0.5 at epochs 120 and 160, 5 warm-up epochs with the trunk frozen. **Batch: 50 images × 3 views = 150 forward images/step**, matching the 150-image PA baseline exactly. 5 seeds everywhere.

---

## 2. Causal zero-shot error mode + proof-level degeneracy attack

### 2.1 The error mode: class-conditional nuisance memorization

With `C` seen identities the label loss carries ≤ log₂C bits/image (≈6.6 on CUB), and for a proxy loss the embedding gradient is `Σ_c (p_c − 1[c=y]) w_c ∈ span{W}`, `dim ≤ M·C`. Any image-varying direction statistically independent of the seen partition receives **zero loss gradient** but nonzero weight-decay pressure, so it is removed. The specific removal that breaks zero-shot: the network attains nuisance invariance by learning a *per-seen-class tolerance* ("for this species, ignore that red-channel swing; it's lighting"). On unseen identities that tolerance is mis-specified in both directions — nuisance leaks in as apparent identity, and identity is absorbed as apparent nuisance. NRC replaces *forgetting* with a *quotient*: the rotor phase is a function of `t` alone, identical for every image and every class, so the invariance it induces is defined on identities never seen.

**Mechanism-specific prediction I will be held to.** PFML uses M=15 proxies/class on CUB/Cars (15×100 = 1500 > 512 ⇒ proxy span is full-rank) but M=2 on SOP (2×11318 ≫ 512, already full-rank). If span-deficiency of label supervision is the operative cause, NRC's gain must be **larger on CUB/Cars than on SOP**, and **larger over a 1-proxy loss than over a 15-proxy loss**. See falsifier F7.

### 2.2 Degeneracy ledger — every cheap escape and its closure

| # | Cheapest degeneracy | Closure |
|---|---|---|
| D1 | **Collapse the group action** to `ρ ≡ I`, silently turning equivariance into invariance (exactly the failure SIE's hypernetwork predictor is built to prevent) | **Closed by construction**: `ρ` has *no* learnable parameters and `ω_j, n_j ≠ 0 ∀j`. There is nothing to collapse. |
| D2 | **Kill planes** (`r_j→0`) so phases are unconstrained noise | **Closed twice**: (i) `L_rot`/`L_comp` are cosines of *unit* 2-vectors, so `∂L_rot/∂(log r_j) ≡ 0` — shrinking a plane buys exactly zero penalty relief; (ii) `L_occ` is a one-sided floor for numerical health. |
| D3 | **Encode `t` in the magnitudes** to satisfy the ledger | Structurally impossible through the head (radial ⟂ tangential, §1.5); and `L_inv` directly penalizes `e(v1) ≠ e(v2)` despite `t^{(1)} ≠ t^{(2)}`. |
| D4 | **Per-view phase lookup table** (memorize, don't represent) | `L_comp` demands `φ_j(t+t′) = φ_j(t)+φ_j(t′)` at composed parameters **outside the single-draw support**. On `A ≅ R^5×S^1×Z/2` the only continuous torus homomorphisms are the linear/integer-winding forms already prescribed, so passing `L_rot+L_comp` is equivalent to having estimated the true nuisance parameters (mod winding). A lookup cannot extrapolate. |
| D5 | **Read the ledger off rendering artifacts** (resampling ringing, border statistics leak crop scale) | *Not fully closed, and I say so.* Mitigated by rendering all views through one identical pipeline at one output size, and by the three photometric coordinates + hue having no resampling signature. **Detected** by control C6. |
| D7 | **The guard alone is doing the work** (NRC = variance floor) | Control C3. |

### 2.3 A derived exclusion (the method's form is forced, not chosen)

Crop *translation* together with scaling generates the `ax+b` group, whose commutator subgroup **is** the translation subgroup. Every homomorphism from `ax+b` into an abelian group contains the commutator subgroup in its kernel. **Therefore no torus-valued rotor can be equivariant to crop translation** — translation must be handled by `L_inv`, never by the ledger. Any construction that puts translation in a torus ledger is silently fitting a non-homomorphism. This is also an honest ceiling: NRC ledgers photometry, hue, scale, aspect, flip — not translation, rotation, or occlusion.

---

## 3. Adversarial novelty search (primary sources, inside and outside DML)

| Nearest work | One-sentence mechanism distinction |
|---|---|
| **SIE, Self-supervised learning of Split Invariant-Equivariant representations** (Garrido et al., ICML 2023) — splits the representation *vector* into invariant and equivariant blocks with a hypernetwork predictor "with no possible collapse to invariance" | NRC does not split coordinates: every deployed coordinate is the **radial invariant of the same coordinate pair whose angle carries the equivariance**, so the equivariant task cannot be served by disjoint channels, and NRC's predictor is a fixed parameter-free homomorphism rather than a learned hypernetwork. |
| **Equivariant Contrastive Learning / E-SSL** (Dangovski et al., ICLR 2022); **EquiMod** (2022) — predict the applied transformation, or its embedding displacement, from a head alongside an invariance objective | These read the transformation from a **separate head off the shared representation**, which the trunk can and does serve with separate channels; NRC reads it from the descriptor head's own pre-invariant coordinates with provably orthogonal radial/tangential gradient paths. |
| **Homomorphism AutoEncoder** (Keurti et al., ICML 2023); **Commutative Lie Group VAE** (2021) — learn a group representation on a latent space with a homomorphism loss over observed transitions | Those aim to *discover* a learned (hence collapsible) representation inside a reconstruction autoencoder; NRC **prescribes** the representation as a fixed multi-octave rotary comb, never reconstructs, and exists to define a deployment-time quotient descriptor for retrieval. |
| **DiVA** (ECCV 2020); **S2SD** (ICML 2021) — auxiliary self-supervised feature spaces / high-dimensional teacher spaces aggregated or distilled into the embedding | Both add **extra embedding spaces** whose content must be transferred by concatenation or distillation; NRC adds no extra space — the auxiliary content lives in the angular fibre of the same coordinates and is discarded exactly, by projection, at test. |
| **Proxy Anchor** (CVPR 2020), **PFML** (CVPR 2025), SoftTriple, HIST — reshape attraction/repulsion and proxy geometry (PFML: decaying potential fields, M proxies/class) | NRC changes neither the descriptor loss nor the proxy geometry; it changes what the descriptor **is** (a maximal invariant of a prescribed nuisance action) and composes with any of them. |
| **Harmonic Networks** (CVPR 2017), capsules, scattering transforms — magnitude-invariant / phase-equivariant responses to *image-plane rotation*, from filter structure or routing | Those act on the spatial rotation group inside the conv stack with equivariance derived from filters; NRC acts on a **photometric/scale/hue group at the global descriptor**, imposed by loss against a prescribed torus code whose group elements come free from the augmentation sampler. |
| **RoPE / random Fourier features** — fixed multi-frequency rotary codes indexing sequence position or kernel frequency | NRC uses the identical algebra to index **the nuisance state of an image**, then deploys the code's *invariants* rather than the code. |
| **Metrix / Embedding Expansion, XBM, uniformity/spread-out, CSLS hubness reduction** | All concern the *arrangement* of descriptors; NRC is a statement about the descriptor's **domain** — it is a function on the quotient by the nuisance action. |

**Search limitation, stated plainly.** My targeted queries for magnitude-invariant/phase-equivariant descriptors in supervised DML returned only unrelated phase-retrieval mathematics. Absence at this search depth is weak evidence of absence. I also could not read PFML's PDF directly (CVF returned 403, the arXiv PDF exceeded the fetch limit); PFML details below come from the arXiv HTML rendering.

---

## 4. Decisive matched-compute controls

All arms: 150 forward images/step, 200 epochs, 5 seeds, identical recipe.

* **C1 — PA baseline**, 1 view × 150 images. The literature-matched reference.
* **C2 — Multi-view PA**, 3 views × 50 images, no ledger. Kills "NRC is just more augmentation".
* **C3 — `L_inv` + `L_occ` only** (`λ_r = 0`). Kills "NRC is a consistency loss plus a variance floor". *Most decisive single control.*
* **C4 — Parallel-head arm.** Augmentation parameters regressed by a separate MLP off `φ_θ(x)` (E-SSL/EquiMod-shaped), descriptor a plain 2048→512 head, matched loss weight. **This is the control that isolates the actual novelty** — coupling through shared radial/tangential coordinates versus auxiliary supervision per se.
* **C5 — Broken code.** Ledger targets replaced by angles drawn independently of `t`. If the gain survives, it is noise injection.
* **C6 — Blank-content probe.** Measure `L_rot` on a constant-texture image with only augmentation applied. Low loss ⇒ D5 is live and the mechanism claim is void.
* **C7 — Homomorphism ablation** (`L_comp` weight 0). Separates "predicting `t`" from "representing `A` as a group".
* **C8 — Comb ablation.** Single octave (`λ_k ≡ 1`) vs. eight octaves.
* **C9 — Diagnostic.** Effective rank of the descriptor Gram on *unseen* classes, C1 vs NRC (mechanism predicts higher for NRC). Diagnostic only, not a claim.

---

## 5. Frozen forecasts, matched baselines, falsification, frontier arithmetic (Lane A)

### 5a. Own-baseline layer — what I actually predict

My PA reproduction at Lane A / 200 epochs (PA's published ResNet-50 512-D 224² is CUB 0.697, Cars 0.877 at 40 epochs): forecast repro **CUB 0.700 ± 0.006**, **Cars 0.882 ± 0.006**, **SOP 0.806 ± 0.003**.

| Dataset | NRC (frozen forecast) | Δ over my PA repro |
|---|---|---|
| CUB-200-2011 | **0.719 ± 0.006** | +1.9 |
| Cars196 | **0.899 ± 0.006** | +1.7 |
| SOP | **0.812 ± 0.003** | +0.6 (deliberately small — the mechanism predicts little where proxy span is already full-rank) |

### 5b. Frontier layer — conditional, and not inherited

Composing NRC with PFML (`L_PA` replaced verbatim by PFML's potential energy `𝒰`, M = 15/15/2 for CUB/Cars/SOP, α ∈ [0,6], δ ∈ [0.1,0.3], Adam lr 5e-4, proxy lr 5e-2, 200 epochs, 224²):

| Dataset | NRC+PFML | PFML reference | Δ | SE of Δ (n=5/arm) | σ |
|---|---|---|---|---|---|
| CUB | **0.744 ± 0.005** | 0.734 ± 0.003 | +1.0 | 0.0037 | **2.7σ** |
| Cars196 | **0.934 ± 0.005** | 0.927 ± 0.003 | +0.7 | 0.0037 | 1.9σ |
| SOP | **0.833 ± 0.003** | 0.829 ± 0.002 | +0.4 | 0.0023 | 1.7σ |

Arithmetic shown: pooled SD `√(0.005²+0.003²) = 0.0058`; `SE_Δ = 0.0058·√(2/5) = 0.0037`.

**I claim a crossing only on CUB, and only marginally (2.7σ). Cars and SOP are below 2σ and I do not claim them.** I explicitly do **not** inherit PFML's frontier: 5b is conditional on an in-house PFML reproduction landing within 0.003 of 0.734 / 0.927 / 0.829 under my batch size. PFML does not disclose batch size or weight decay; if my reproduction lands lower, the comparison reported is against my reproduction, not the published row.

**Pre-registered falsifiers** (5-seed means, CUB unless stated):

* **F1** NRC − C1 < +0.8 → method dead as proposed.
* **F2** NRC − C2 < +0.5 → gain is multi-view augmentation.
* **F3** NRC − C3 < +0.5 → gain is consistency + variance floor, i.e. a renamed regularizer.
* **F4** NRC − C4 < +0.4 → **the coupling claim (the actual novelty) is false**, even if R@1 is good; honest report becomes "E-SSL-style auxiliary supervision helps DML".
* **F5** C5 reproduces ≥60% of the gain → mechanism is noise injection.
* **F6** C6 yields `L_rot < 0.15` on blank content → ledger is artifact-solved (D5).
* **F7** SOP gain > CUB gain, or NRC's gain over 15-proxy PFML > its gain over 1-proxy PA → §2.1 causal story falsified regardless of R@1.

Any of F1–F4 firing means NRC must not be published as a new mechanism.

---

## 6. Cost, benchmark risk, contamination risk

**Training.** Three views/image at 50 images/step = 150 forward images/step, so step FLOPs and activation memory match the 150-image PA baseline (~1.0×). The real cost is that **distinct images per epoch drop 3×** at fixed epochs; that is a genuine risk and could fire F1 on its own. Upper-bound arm C1′ (150 images × 3 views = 3× step cost) is specified as the fallback. `ρ` costs 512 sin/cos per view — negligible. Extra train-time parameters beyond the head: none.

**Deployment.** One extra 2048×1024 matmul plus a 512×512 matmul ≈ 4.7 MFLOP against ResNet-50's ~4.1 GFLOP ⇒ **+0.11% FLOPs**; parameters 27.96M vs 26.65M ⇒ **+4.9%**. Same latency class, one view, 512-D, cosine. No test-time machinery; phases discarded.

**Benchmark risk.** (i) Sub-1-point CUB/Cars differences sit inside the historical reproducibility spread of the DML literature; a 2.7σ crossing over 5 seeds is not a robust frontier claim and I do not present it as one. (ii) NRC conditions on crop scale during training while evaluation uses a fixed center crop — an out-of-distribution nuisance state at test. I keep the standard resize-256/center-crop-224 protocol; any deviation would be a protocol change and must be reported as such.

**Contamination.** No test data, no generated data, no text/VLM encoder, no extra annotation, no transduction, no reranking, no gallery fitting. The only external asset is the permitted ImageNet-1K initialization; NRC neither distils from nor regularizes toward it. Augmentation parameters come from our own sampler, so recording them adds no annotation and no external information — but the fair objection stands that *recording sampler state is a new class of free supervision*, and if a reviewer rules that outside "ordinary stochastic image augmentation", the method is out of scope. I flag it rather than argue it away.

**Unresolved source ambiguities.** PFML: batch size and weight decay not disclosed; α is stated only as a range [0,6] and δ as [0.1,0.3] "cross-validated"; no In-Shop row found. Proxy Anchor: weight decay, warm-up, and LR-decay schedule not disclosed; ResNet-50 512-D rows given for CUB/Cars only, not SOP/In-Shop — so my Lane A SOP baseline forecast (0.806) is an extrapolation, not a reproduction of a published row. PA+DADA In-Shop 0.930: seed count and uncertainty unreported (per the brief; I did not independently verify it).

**Honest failure modes.** (a) The weakest link is that NRC ledgers *synthetic* nuisance while CUB's dominant real nuisance is pose and background — the transfer from ledgering photometry/scale to tolerating pose is asserted, not proven; F4/F5 are the tests. (b) §2.3 forces translation out of the ledger, and translation is plausibly the factor that matters most on fine-grained birds. (c) An 8-octave comb up to λ=8 may demand phase precision the trunk cannot supply, turning `L_rot` into a noise source (C8). (d) The 3×-fewer-distinct-images cost may simply exceed what the ledger earns.

---

Sources: [PFML CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/33305) · [PFML arXiv](https://arxiv.org/abs/2405.18560) · [PFML HTML](https://arxiv.org/html/2405.18560v3) · [Proxy Anchor (ar5iv)](https://ar5iv.labs.arxiv.org/html/2003.13911) · [SIE, ICML 2023](https://proceedings.mlr.press/v202/garrido23b/garrido23b.pdf) · [Equivariant Contrastive Learning, ICLR 2022](https://arxiv.org/pdf/2111.00899) · [EquiMod](https://arxiv.org/pdf/2211.01244) · [Homomorphism AutoEncoder](https://proceedings.mlr.press/v202/keurti23a.html) · [Commutative Lie Group VAE](https://arxiv.org/pdf/2106.03375) · [DiVA, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530579.pdf) · [S2SD, ICML 2021](https://arxiv.org/abs/2009.08348)
