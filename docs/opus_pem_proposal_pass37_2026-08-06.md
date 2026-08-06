## PROPOSAL — one method, Lane A (matched 512‑D ResNet‑50 CNN lane)

**Name:** **PEM — Partition‑Evidence Metric learning**

**One‑line mechanism:** replace point‑to‑proxy scoring with a *marginal‑likelihood ratio between the true identity partition of a batch and its local neighbourhood of corrupted partitions*, under a von Mises–Fisher random‑effects model in which the identity mean direction is **integrated out** against a uniform (maximum‑entropy) identity prior. Integrating out the identity latent removes every per‑class learned parameter and converts the vMF normaliser into a Bessel **Occam factor** that supplies an *absolute*, dimension‑calibrated same/different threshold — exactly the decision the deployed cosine nearest‑neighbour rule must make on identities that were never fit.

Lane declaration: **Lane A only.** All forecasts, baselines and controls below are ResNet‑50 / 512‑D / ~224 px / single‑view cosine / 200 epochs. I make no Lane‑B claim and do not compare to AdvRF, VAPNet, or CRT.

---

## 1. Executable mathematics

### 1.1 Learned objects and gradient paths

| object | shape | init | LR | weight decay |
|---|---|---|---|---|
| `θ` ResNet‑50 backbone | 25.6 M | ImageNet‑1K supervised | 1e‑5 | 1e‑4 |
| `W, b` linear head 2048→512 | 1.05 M | Kaiming / 0 | 1e‑4 | 1e‑4 |
| `a` (κ = softplus(a)) | 1 scalar | κ₀ = 16 | 1e‑2 | **none** |

There are **no per‑class parameters at all** — no proxies, no classifier, no memory bank, no second network, no auxiliary head. On SOP this *removes* 11318 × 512 = 5.8 M parameters plus Adam state relative to any proxy method.

Forward: `h = GAP(ResNet50(x)) ∈ ℝ²⁰⁴⁸`, `z̃ = Wh + b`, `z = z̃/‖z̃‖₂ ∈ S⁵¹¹`. BatchNorm frozen (running stats and affine), standard in this lane.

### 1.2 The evidence functional

Generative model over a batch, `d = 512`, `ν = d/2 − 1 = 255`:

```
μ_c ~ Uniform(S^{d-1})            i.i.d. per identity      ← the zero-shot prior
z_i | y_i = c ~ vMF(μ_c, κ)
```

The uniform prior is the mathematical statement of the task: *test identities are arbitrary and unseen*, so no direction of the sphere may be privileged.

For a cluster `A` write `S_A = Σ_{i∈A} z_i`, `R_A = ‖S_A‖₂ ∈ [0, |A|]`. Marginalising `μ`:

```
p(Z_A) = ∫ Π_i vMF(z_i; μ, κ) · C_d(0) dμ = C_d(κ)^{|A|} · C_d(0) / C_d(κ R_A)
```

using `∫_{S^{d-1}} exp(η·μ) dμ = 1/C_d(‖η‖)` and `C_d(0) = 1/A_{d-1}`. Dropping the partition‑independent factor `C_d(κ)^N`, the **log‑evidence of a partition π is**

```
E(π) = Σ_{A ∈ π} φ(R_A),        φ(r) := log C_d(0) − log C_d(κ r),   φ(0) = 0
```

and this collapses to a single special function:

```
φ(r) = log ₀F₁( ; d/2 ; κ²r²/4 )  =  log[ Γ(ν+1) (2/(κr))^ν I_ν(κ r) ]
```

**Properties used later (all provable, not assumed):**

- `φ(0) = 0`.
- `φ′(r) = κ · A_d(κ r)` where `A_d(x) = I_{d/2}(x)/I_{d/2−1}(x) ∈ (0,1)` is the vMF mean‑resultant function. This gives the **exact analytic gradient**, so no Bessel autodiff is needed.
- `A_d` is strictly increasing ⇒ `φ` is **strictly convex** on `r>0` with `φ(0)=0` ⇒ `φ` is **superadditive**: `φ(x+y) ≥ φ(x)+φ(y)`. This single fact kills collapse (§2).
- Small argument: `φ(r) ≈ κ²r²/(2d)`. Large argument (`κr ≫ ν`): `φ(r) ≈ κr − ν log(κr) − ½log(2πκr) + const`. The `−ν log(κr)` term **is** the Occam factor and is the entire mechanism.

**Numerics.** Precompute `φ` on a 4096‑point grid over `r ∈ [0, K]` by quadrature of `φ′(r) = κ A_d(κr)`, with `A_d` from the Perron continued fraction (stable for `ν=255`); cubic‑spline interpolate forward, use the closed form `∂φ(R_A)/∂z_i = κ A_d(κR_A) · S_A/R_A` backward. Rebuild the grid whenever κ changes by >1 %. Then propagate through the L2‑normalisation Jacobian `(I − zzᵀ)/‖z̃‖` into `W, θ`.

### 1.3 The partition neighbourhood 𝒩(π\*)

Batch: `P` classes × `K` images, class‑balanced sampler without replacement over classes per epoch. **CUB / Cars / In‑Shop: P=32, K=8 (B=256). SOP: P=64, K=4 (B=256)** (SOP classes hold ≈5 images).

Three move families, all evaluated in closed form from `{S_a}` and `{z_i}`:

1. **Move** — sample `i` reassigned from `a = y_i` to `b ≠ a`:
   `Δ_{i→b} = φ(‖S_a − z_i‖) + φ(‖S_b + z_i‖) − φ(R_a) − φ(R_b)`
2. **Merge** — identities `a, b` are the same identity:
   `Δ_{a⊕b} = φ(‖S_a + S_b‖) − φ(R_a) − φ(R_b)`
3. **Split** — identity `a` is really two identities. Candidate splits: the top eigen‑direction of the centred within‑class scatter of `A_a` (3 power iterations on the K×K Gram, no gradient through the eigenvector — split assignment treated as a constant) **plus 3 random hyperplanes**; take the max‑evidence candidate.
   `Δ_{a⊘} = φ(‖S_{A₁}‖) + φ(‖S_{A₂}‖) − φ(R_a)`

### 1.4 Loss

```
L_PEM = softplus( LSE_{π ∈ 𝒩(π*)} [ E(π) − E(π*) ] )
      = log( 1 + Σ_{π∈𝒩} exp(Δ_π) )
```

i.e. the negative log posterior probability of the true partition within its neighbourhood. `L_PEM` is the **only** loss term. No margin, no temperature beyond κ, no auxiliary objective. (A partition prior `log p(π)` — CRP — is set uniform; declared assumption, so the *only* complexity pressure is the Occam factor.)

### 1.5 Schedules

- **κ warm‑up:** cosine 16 → 96 over epochs 1–20, then learned freely (clamped to [32, 256]).
- **LR:** 5‑epoch linear warm‑up, then cosine to 0 over 200 epochs. AdamW, grad‑clip global norm 5.
- **Optional cold‑start:** 5 epochs of Proxy‑Anchor warm‑up, proxies then discarded. **This contaminates the "no per‑class parameters" claim during warm‑up and is therefore itself ablated (C9); if it is required, that must be reported prominently.**
- **Augmentation:** RandomResizedCrop(224, scale 0.16–1.0, ratio 3/4–4/3) + horizontal flip. Test: Resize 256 → CenterCrop 224.
- 5 seeds (0–4). Report mean ± **sample SD and SEM both** (see §5).

### 1.6 The operative‑scale condition — *not* a harmless normalisation

`φ` has two regimes. If `κ R_a ≪ ν`, then `φ(r) ≈ κ²r²/(2d)` and

```
E(π) ≈ (κ²/2d) Σ_A ‖S_A‖²
```

which is a **k‑means‑like structured clustering score with no Occam term** — PEM degenerates into the Song‑2017 family and the claimed mechanism is *absent*. The method is only itself when

```
κ · R̄_a  ≳  ν = d/2 − 1 = 255
```

With K=8 and post‑warm‑up within‑class mean resultant ρ ≈ 0.85, `R̄_a ≈ 6.8`, so κ ≥ 38 suffices; κ = 96 gives `κR̄/ν ≈ 2.5`. **κ is therefore an operational scale, not a normalisation.** Consequences that must be honoured: κ is excluded from weight decay (decaying it silently converts PEM into control C2); `κR̄/ν` is logged every epoch and is a pre‑registered kill switch (F7). Because `z` is L2‑normalised there is no gradient path by which the network can inflate scale to fake concentration — only angular concentration moves `R`.

### 1.7 Train/test asymmetry

Train‑only: κ, the neighbourhood construction, the split search, the class‑balanced sampler. **Test: one 224 centre crop, one forward pass, `z = normalize(Wh+b)`, cosine NN over the gallery, standard R@1 protocol.** Deployment is one ResNet‑50 + one linear layer, 512‑D — bit‑identical in form to the Proxy‑Anchor baseline.

### 1.8 Baseline reduction reproduced exactly (used for B0 and for the optional warm‑up)

Proxy‑Anchor (Kim et al., CVPR 2020), as disclosed:

```
L_PA = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{x∈X_p⁺} e^{−α(s(x,p) − δ)})
     + (1/|P|)  Σ_{p∈P}   log(1 + Σ_{x∈X_p⁻} e^{ α(s(x,p) + δ)})
```

α = 32, δ = 0.1, `s` = cosine, proxies L2‑normalised, one proxy per class, Adam lr 1e‑4 with proxy lr ×100, weight decay 1e‑4. **PEM is not an extension of PFML or PA — it replaces the loss — so I inherit no published frontier and run my own B0/B1 reproductions (§4).**

---

## 2. Causal zero‑shot error mode + proof‑level degeneracy attack

### 2.1 The error mode: *proxy‑absorbed calibration*

Every strong Lane‑A method carries free per‑class parameters `{w_c}` — Proxy‑Anchor (1/class), SoftTriple, ProxyNCA++, PFML (15/class on CUB & Cars, 2 on SOP). Two causal consequences:

**(a) The train objective is strictly easier than the test objective.** At test time no per‑identity parameter exists; the same/different decision is made by raw geometry alone. A free `w_c` can *absorb* systematic mis‑calibration of the backbone geometry: a class whose samples are angularly spread and anisotropic can still be won by placing `w_c` at the cloud's centroid with a large *local* margin. The backbone is never forced to make that spread small, or isotropic, **in absolute terms**.

**(b) The backbone only ever receives ordinal supervision.** Every gradient is a relative comparison among the C seen identities. No term's value depends on whether two identities would be confusable *for a rule that must also work on identities absent from the batch*. The learned geometry is ordinally correct and metrically arbitrary.

This bites in zero‑shot because an unseen identity's cluster is *placed by generalisation, not by fitting*. The dominant unseen‑class R@1 error modes are (i) anisotropic within‑class scatter — a viewpoint/pose axis elongates the cluster along a direction that crosses a neighbour — and (ii) an uncalibrated absolute confusion threshold. Neither is supervised by a relative, proxy‑absorbed objective.

### 2.2 How PEM attacks it

- **Path (a) is closed by construction.** With μ integrated out there is nothing to absorb miscalibration; winning the merge hypothesis is a statement about `R_a, R_b, ‖S_a+S_b‖` — raw geometry only.
- **The Occam factor makes the comparison absolute.** Because `φ` is strictly convex with `φ(0)=0`, `Δ_{a⊕b} = φ(R_{ab}) − φ(R_a) − φ(R_b)` has a fixed zero‑crossing determined by `(d, κ)` alone. Define the **merge margin**: identities a, b are evidence‑separated iff `‖S_a+S_b‖ < r*` where `φ(r*) = φ(R_a)+φ(R_b)`. In the asymptotic regime this requires, for `R_a=R_b=R` and mean‑separation angle θ,

  ```
  2κR(1 − cos(θ/2))  >  ν·log( κR / (2 cos(θ/2)) ) − const(ν)
  ```

  Worked at κ=96, R=6.8, ν=255: the requirement is satisfied at `cos(θ/2) ≲ 0.83`, i.e. **θ ≳ 68° of class‑mean separation** — demanding but attainable (random directions in 512‑D give θ ≈ 90°). *Caveat: `κR ≈ 650` vs `ν = 255` is only ratio 2.5, so this asymptotic is rough; the exact threshold must be computed numerically from φ. The point that survives is structural: the required separation scales as `ν/κ = (d/2−1)/κ`, so it is set by the descriptor dimension and the learned concentration, not by a hand‑tuned margin.*
- **The split move is a targeted attack on anisotropy.** The max‑evidence 2‑way split has high evidence exactly when within‑class scatter has a dominant eigen‑direction. PEM's loss stays high until within‑class scatter is near‑isotropic. Isotropic within‑class covariance is precisely the condition under which plain cosine is the likelihood‑ratio‑optimal same/different test — i.e. **the training objective and the mandated deployment rule are made to coincide, rather than merely correlate.**

### 2.3 Degeneracies, attacked

**D1 — total collapse (all `z` equal).** Then `R_A = |A|` for every A. `φ` convex with `φ(0)=0` ⇒ superadditive ⇒ `Δ_{a⊕b} = φ(2K) − 2φ(K) ≥ 0` for every pair, so **every merge hypothesis beats the truth** and `L ≥ log(1 + P(P−1)/2)` ≈ 6.2 nats. Collapse is a near‑maximiser of PEM, not a minimiser. This is a proof, not an empirical hope.

**D2 — partial collapse / insufficient separation.** More generally `R_{ab} ≤ R_a + R_b` with equality iff the class means coincide, and `Δ_{a⊕b} < 0` requires `R_{ab}` strictly below the break‑even radius `r*`. So the loss demands a **quantitative, absolute** angular separation, with the amount fixed by φ's curvature. There is no configuration in which merge terms are trivially satisfied.

**D3 — instance memorisation.** Suppose `f` indexes each training image and arranges class clouds arbitrarily but ordinally perfectly. Move terms are satisfied; **merge terms are not** (they require the absolute D2 separation); **split terms are not** (they require isotropic second‑order class shape, which is invariant to any relabelling and to which memorisation is indifferent). This is not a proof that memorisation is impossible — nothing here rules it out — but it *is* a proof that memorisation does not **suffice** to minimise `L_PEM`, in contrast to proxy‑softmax losses where memorised features plus fitted proxies attain ≈0 loss.

**D4 — dimension stuffing / rank inflation.** Injecting isotropic nuisance of relative energy ε into unused dimensions multiplies every `R_A` by ≈ √(1−ε). In the asymptotic form the κ‑linear parts of `Δ_{a⊕b}` scale by √(1−ε) while the `−ν log` terms do not, so the merge margin **shrinks** and `L` strictly increases. **PEM contains no volume or rank bonus, so it cannot be gamed by filling dimensions with nuisance** — the sharpest separation from coding‑rate / ρ‑spectral anti‑collapse regularisers, which *reward* exactly that.

**D5 — scale cheating.** Blocked by L2 normalisation of `z`; only angular structure enters `R_A`.

**D6 — κ escaping to the degenerate regime.** κ→small makes PEM ≈ k‑means (§1.6). Blocked by clamping κ ∈ [32,256], excluding κ from weight decay, and kill switch F7.

**Honest residual:** the split move's gradient is straight‑through past a non‑differentiable eigenvector and threshold. This is a genuine approximation and the most likely source of optimisation instability; the 3‑random‑hyperplane augmentation of the candidate set is the mitigation, and C4 measures whether splits earn their keep at all.

---

## 3. Adversarial novelty search (primary sources, inside and outside DML)

The honest headline: **PEM occupies the same *slot* as Song et al. 2017 — a structured objective over clusterings — with a different scoring functional and a different neighbourhood.** That is the paper to beat, and I say so first.

| nearest work | mechanism distinction (one sentence) |
|---|---|
| **Song, Jegelka, Rathod & Murphy, "Deep Metric Learning via Facility Location" / "Learnable Structured Clustering Framework", CVPR 2017** | They score clusterings by a **facility‑location (medoid) objective** with a hand‑set NMI‑rescaled structured margin; PEM scores them by a **marginal likelihood with the identity latent integrated out**, so the same/different threshold is a Bessel Occam factor determined by (d, κ), and its neighbourhood includes evidence‑optimal **splits**, which facility location structurally cannot express because adding a facility never decreases the score. |
| **Group Loss (Elezi et al., ECCV 2020)** | Replicator‑dynamics label propagation over a batch similarity graph feeding a per‑class softmax classifier; PEM has no classifier and no propagation — it scores whole partitions. |
| **ProxyNCA / ProxyNCA++ / Proxy‑Anchor / SoftTriple / PFML (Potential Field, CVPR 2025)** | All retain free per‑class proxies (PFML: 15/class on CUB & Cars, 2 on SOP) and score point‑to‑proxy potentials; PEM has **zero** per‑class parameters, so the train‑time decision rule *is* the test‑time decision rule. |
| **PA + DADA (Data‑Augmented Domain Adaptation, AAAI 2024)** | Aligns the proxy and sample "domains" via domain adaptation on top of a proxy loss; PEM changes the hypothesis space being scored, not proxy/sample alignment, and adds no augmentation or adaptation machinery. |
| **Directional‑statistics DML / vMF loss (Zhe et al.); vMF‑mixture and vMF‑hashing losses** | Maximise a **conditional** vMF likelihood with a **learned per‑class mean direction**; PEM **marginalises that direction out**, which is precisely what turns the normaliser into an Occam factor and deletes the per‑class parameter. |
| **Non‑Isotropy Regularization (Roth et al., ECCV 2022)** | Fits per‑class normalizing flows to make class‑conditional densities **non‑isotropic** around proxies; PEM's split move drives class‑conditional scatter **toward isotropy**, on the argument that isotropy is the condition making cosine NN the LR‑optimal same/different test — a directly *opposing* prediction and therefore a clean mutual falsification target. |
| **Anti‑Collapse Loss (coding‑rate/MCR², 2024); ρ‑spectral regularisation (Roth et al., ICML 2020)** | Add an explicit volume/rank bonus to a base loss; PEM adds **no** volume term and its dimension‑dependence lives *inside* the likelihood, where it **penalises** uninformative dimensions (D4). |
| **Contrastive Bayesian Analysis for DML (TPAMI)** | Models a posterior over **labels given similarities** with a learned similarity model; PEM's Bayesian object is the **partition evidence with the identity latent integrated out**. |
| **Prototypical Networks / episodic few‑shot** | Uses support means as a within‑episode classifier; PEM never forms a classifier, and merge/split hypotheses have no analogue in an episode. |
| **Bayesian model‑based clustering, product partition models, DP‑vMF mixtures (Banerjee et al. 2005; Gopal & Yang 2014)** — *outside DML* | These **infer** a partition by maximising evidence over **fixed** features; PEM inverts the direction of use — the partition is *known* and the **representation** is optimised so that the true partition is the evidence‑maximiser within its neighbourhood. |
| **MacKay (1992) Occam factor / Bayesian model selection** — *outside ML‑vision* | Classically used to select among models; PEM uses the Occam factor as a **training signal on a representation**, which I did not find in the primary literature. |
| **PLDA (Prince & Elder 2007), speaker/face verification** — *outside DML* | A Gaussian random‑effects **scoring backend fitted post hoc** on frozen embeddings and used at test time; PEM is a **train‑time** objective on the sphere whose backend is integrated out and then discarded, leaving plain cosine. |
| **Neural‑collapse / fixed simplex‑ETF classifiers** | Prescribe the terminal geometry; PEM prescribes no geometry — the Occam term supplies only a scalar separation requirement. |

**Open novelty risk, stated plainly.** Two recent adjacent arXiv entries surfaced in search but I did not read them: *"Beyond Seen Bounds: Class‑Centric Polarization for Single‑Domain Generalized Deep Metric Learning"* and *"CouCE: A Unified Causal Framework for Debiased Deep Metric Learning"*. Neither is characterised here and either could reduce PEM's novelty margin. I could not verify PFML's internal formulation beyond "continuous potential field + proxies augmenting batch‑sample interactions".

---

## 4. Decisive matched‑compute controls

All controls: identical backbone, optimizer, epochs, batch size/composition, augmentation, 5 seeds. Each is designed to *kill* a specific claim, not to flatter it.

| id | control | claim it isolates | kill condition |
|---|---|---|---|
| **B0** | Proxy‑Anchor, my recipe, my reproduction | absolute baseline | — |
| **B1** | PFML reproduction (15 proxies/class CUB & Cars, 2 SOP) | frontier reference | if B1 < 0.734 / 0.927 in my hands, **all comparisons are made against my B1** and the published number is reported separately as unmatched |
| **C1** | **Occam‑off**: replace φ by `φ̃(r) = κr` (drop the Bessel `−ν log` term) | *the entire central mechanism* | C1 within 0.002 of PEM ⇒ **mechanism claim dead**, PEM is "another structured clustering loss" |
| **C2** | **Regime‑off**: κ pinned at 8 (`κR ≪ ν`), so `E ≈ (κ²/2d)Σ‖S_A‖²` | the operative‑scale claim of §1.6 | C2 ≈ PEM ⇒ κ is cosmetic, and the k‑means reading is the correct one |
| **C3** | **Moves‑only** neighbourhood (reduces to Bessel‑warped soft nearest‑class‑mean) | the partition‑neighbourhood claim | within 0.002 ⇒ merges/splits are inert |
| **C4** | **No‑split** (moves + merges) | the isotropy mechanism | within 0.002 **and** no reduction in unseen within‑class anisotropy ⇒ isotropy story dead |
| **C5** | **Proxies‑back**: keep φ scoring but substitute free learned per‑class proxies for the empirical `S_a` | the "no per‑class parameters" claim | within 0.002 ⇒ parameter‑freeness is not the source of the gain |
| **C6** | **K‑sweep** K ∈ {2,4,8} at fixed B=256, PEM and B0 | the regime story | PEM flat in K, or B0 rising as fast ⇒ mechanism story wrong |
| **C7** | **Isotropy + margin probe** (measurement, on the *unseen* split): top‑1 eigenvalue share of within‑class covariance; mean merge margin `φ(R_a)+φ(R_b)−φ(R_ab)` | causal chain, independent of R@1 | PEM wins R@1 without moving either ⇒ result is unexplained, report as such |
| **C8** | **Wall‑clock‑matched B0** (extra epochs to equal PEM's wall clock) | rules out "PEM just trains more" | — |
| **C9** | **PEM with no PA warm‑up** | purity of the parameter‑free claim | if PEM only works with PA warm‑up, that must headline the method, not a footnote |

C1 is the single most important experiment in the proposal.

---

## 5. Frozen forecasts, Lane A — 2026‑08‑06

ResNet‑50, 512‑D, 224 px, 200 epochs, single‑view cosine, 5 seeds (0–4), mean ± sample SD.

| dataset | **PEM forecast R@1** | PFML (CVPR 2025) | PA+DADA matched‑cost control |
|---|---|---|---|
| CUB‑200‑2011 | **0.741 ± 0.005** | 0.734 ± 0.003 | 0.729 |
| Cars196 | **0.932 ± 0.004** | 0.927 ± 0.003 | 0.921 |
| SOP | **0.831 ± 0.003** | 0.829 ± 0.002 | 0.810 |
| In‑Shop *(non‑target)* | 0.929 ± 0.004 | — | PA+DADA 0.930 (no seeds/uncertainty reported) |

The near‑parity SOP forecast is deliberate and **risky**: SOP forces K=4, so `κR̄ ≈ 96·3.5 = 336` vs `ν = 255` — barely operative — and the mechanism predicts the gain must therefore be small there. A large SOP gain would falsify the story even though it would look like success.

### Frontier‑crossing arithmetic — and the ambiguity that governs it

**Unresolved source ambiguity, material:** whether PFML's "± 0.003" is sample SD or SEM over its five runs. This changes the verdict.

*Reading 1 — ±0.003 is SD.* SEM_PFML = 0.003/√5 = 0.00134; SEM_PEM = 0.005/√5 = 0.00224.
CUB: Δ = 0.741 − 0.734 = **0.007**; Welch t = 0.007/√(0.00134² + 0.00224²) = 0.007/0.00261 = **2.68**, df ≈ 6.5, two‑sided **p ≈ 0.033** → **crosses at α = 0.05, not at α = 0.01.**
Cars: Δ = 0.005; SEM_PEM = 0.00179; t = 0.005/0.00224 = **2.24**, **p ≈ 0.062** → **does not cross at α = 0.05.**

*Reading 2 — ±0.003 is SEM.* SD_PFML = 0.0067.
CUB: t = 0.007/√(0.003² + 0.00224²) = **1.87**, **p ≈ 0.10** → **does not cross.** Crossing at α = 0.05 with 5 seeds would need Δ ≥ 0.0105, or Δ = 0.007 with ≥ 12 seeds per arm.

**Therefore the only frontier crossing I forecast is CUB, and only under Reading 1, at α = 0.05.** Cars and SOP are forecast as parity‑or‑better, not crossings; the honest plan is 10 seeds on Cars. I decline any In‑Shop frontier claim because PA+DADA's 0.930 has no reported seed count or uncertainty and the comparison would be uncontrolled.

### Pre‑registered falsification thresholds

- **F1** mean CUB < 0.734 **or** mean Cars < 0.927 over 5 seeds ⇒ no frontier claim.
- **F2** mean CUB < B0 + 0.010 ⇒ the method does not justify its complexity; retract.
- **F3** C1 (Occam‑off) within 0.002 of PEM on CUB ⇒ **central mechanism false.** Primary kill switch.
- **F4** C3 (moves‑only) within 0.002 ⇒ partition‑neighbourhood claim false.
- **F5** C5 (proxies‑back) within 0.002 ⇒ parameter‑freeness claim false.
- **F6** C7 shows no reduction in unseen within‑class anisotropy vs B0/B1 ⇒ isotropy mechanism false even if R@1 improves.
- **F7** measured `κR̄/ν < 0.8` at convergence ⇒ the run lived in the degenerate quadratic regime; the result is a k‑means‑loss result and must be reported as such, not as PEM.
- **F8** SOP gain > CUB gain ⇒ the K‑regime story is inverted; mechanism story fails.
- **F9** PEM requires the PA warm‑up (C9 fails to train) ⇒ the "no per‑class parameters" framing must be demoted.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** ResNet‑50 forward/backward unchanged. Added per step: cluster sums `O(Bd)`; `P²/2` merge terms `O(P²d)` = 0.26 M FLOPs; `B(P−1)` move terms via rank‑1 updates `O(BPd)` = 4.2 M FLOPs; `P` split searches `O(PK²d)` = 1.0 M FLOPs; φ via spline lookup. Total ≪ 0.1 % of one ResNet‑50 step at 224². **Estimated ≤ 1.02× epoch time and 1.00× memory vs Proxy‑Anchor**, and strictly *cheaper* in parameters and optimizer state than any proxy method (−5.8 M params on SOP). No second network, no reconstruction branch, no distillation — this is the whole point of comparing in Lane A rather than Lane B.

**Deployment cost.** Identical to baseline: one ResNet‑50, one linear layer, one 512‑D L2‑normalised descriptor, one view, cosine NN. κ, φ, and the partition machinery are discarded at the end of training.

**Risks, ranked.**

1. **Optimisation / cold start (largest).** `φ′ = κ A_d(κR)` and `A_d(x) ≈ x/d` for small x, so poorly formed clusters get *weak* gradients — a rich‑get‑richer dynamic and a plausible cold‑start failure. Mitigated by κ warm‑up and LR warm‑up; C9 measures whether that is enough. If PA warm‑up turns out to be required, the method's headline claim weakens.
2. **Scale sensitivity is operational.** κ decides whether the method is PEM or k‑means (§1.6, F7). Excluded from weight decay by design; a silent decay would degrade PEM into C2 without any visible symptom other than the logged ratio.
3. **Split‑move straight‑through gradient** past a non‑differentiable eigenvector/threshold — a genuine approximation with oscillation risk.
4. **Reproduction risk to the frontier claim.** I do not have PFML's exact recipe (pooling — GAP vs avg+max concat; optimizer; weight decay; batch composition; whether BN is frozen), nor the SD/SEM meaning of its ±0.003. **Every comparison must be against my own matched B1**, with the published number reported separately as unmatched. This is the single largest threat to any frontier statement. My use of AdamW rather than Adam is itself a recipe deviation and is applied identically to B0, B1 and PEM.
5. **Benchmark noise.** CUB and Cars test sets are small (5924 / 8131 images); ±0.005 seed spread is normal, and SOP differences below 0.003 are routinely within noise. Codebase‑level protocol drift (crop size at test, resize interpolation, whether the query is excluded from its own gallery) can move R@1 by more than the effect being claimed; all arms run in one codebase.
6. **Contamination.** No test images, no external or generated data, no text/VLM encoder, no extra annotations, no transduction, no re‑ranking, no test‑gallery fitting at any point; κ and the split search use training data only. The one inherited contamination is semantic overlap between ImageNet‑1K pretraining and CUB (~59 bird classes) and Cars — shared by every method in the lane, so it does not bias the comparison, but it does mean "zero‑shot" holds with respect to the benchmark's identity labels, not with respect to the visual world.

**Where I am most likely wrong:** that PEM will train stably from ImageNet init without proxy scaffolding, and that the Occam margin is the *binding* constraint at the optimum rather than a term the optimizer satisfies early and then ignores. C1 and F3 exist to find that out quickly and cheaply.

---

Sources:
- [Potential Field Based Deep Metric Learning — CVPR 2025 (PFML)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- [Towards Improved Proxy-based Deep Metric Learning via Data-Augmented Domain Adaptation — AAAI 2024 (DADA)](https://ojs.aaai.org/index.php/AAAI/article/view/29400)
- [Directional Statistics-based Deep Metric Learning](https://arxiv.org/pdf/1802.09662)
- [A Non-isotropic Probabilistic Take on Proxy-based Deep Metric Learning](https://arxiv.org/pdf/2207.03784)
- [Anti-Collapse Loss for Deep Metric Learning Based on Coding Rate Metric](https://arxiv.org/pdf/2407.03106)
- [Contrastive Bayesian Analysis for Deep Metric Learning](https://arxiv.org/pdf/2210.04402)
- [The Group Loss for Deep Metric Learning — ECCV 2020](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_17)
- [Proxy Anchor Loss for Deep Metric Learning](https://suhakwak.github.io/papers/deep_metric_learning_kccv2020.pdf)
