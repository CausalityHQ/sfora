# NSP — Null‑Space Provisioning

**Lane: A** (ResNet‑50, 512‑D L2‑normalized descriptor, ~224 px, single view, cosine NN, 200 epochs). All numbers, baselines and comparisons below are Lane A only.

---

## 0. One‑paragraph statement

Identity supervision on `C` training classes has **rank at most `C−1`**: every loss of the form `L(f(x), y)` shapes the descriptor only inside the span of the class centroids. On CUB (`C=100`) and Cars (`C=98`) that leaves **~413 of 512 descriptor directions receiving no identity‑relevant gradient**. NSP measures that label‑null subspace online from an EMA of class centroids, and *provisions* it with the only transferable signal available for free — augmentation‑invariant instance identity with same‑class negatives — under an explicit energy floor so the provisioned directions actually carry weight in the deployed cosine metric. Deployment is unchanged: one ResNet‑50, one view, one 512‑D vector, cosine NN.

---

## 1. Executable mathematics

### 1.1 Base (reproduced exactly, not inherited)

Backbone `g_θ` = ResNet‑50, ImageNet‑1K init. Head `h_φ`: linear `4096 → 512` on `concat(GAP, GMP)`. `z̃ = h_φ(g_θ(x)) ∈ R^512`, `z = z̃/‖z̃‖`.

Base loss = **Proxy‑Anchor** (Kim et al., CVPR 2020), one proxy `p_c` per class, `s(·,·)` cosine:

```
L_PA = (1/|P⁺|) Σ_{c∈P⁺} log(1 + Σ_{x∈X_c⁺} e^{−α(s(z,p_c) − δ)})
     + (1/|P |) Σ_{c∈P }  log(1 + Σ_{x∈X_c⁻} e^{+α(s(z,p_c) + δ)})
```

`α=32`, `δ=0.1`, proxies unit‑normalized, proxy LR = 100× base LR, AdamW `lr=1e‑4`, `wd=1e‑4`, batch 180, 1 epoch head‑only warm‑up (backbone frozen), train aug = `RandomResizedCrop(224, scale=(0.16,1))` + hflip, test = resize 256 → center crop 224.

**Deviation stated up front:** Proxy‑Anchor's published schedule is far shorter than 200 epochs. I adopt the lane's 200 epochs with cosine LR decay and **re‑measure the base myself**; no published PA number is inherited.

**Multi‑proxy variant** (used only for the frontier arithmetic in §5), `K` proxies/class, SoftTriple‑style intra‑class soft‑max with `λ=20`:
`s(z,c) = (1/λ) log Σ_{k=1..K} exp(λ · s(z, p_{c,k}))`, substituted into `L_PA`. `K=15` on CUB/Cars, `K=2` on SOP (matching PFML's disclosed proxy counts). **This is a stand‑in, not PFML**; PFML's actual loss is unknown to me (§5.4).

### 1.2 The label‑null projector (train‑only, stop‑grad)

EMA centroid table `μ ∈ R^{C×512}`, momentum `m=0.99`, updated per step for classes present in the batch:
`μ_c ← m·μ_c + (1−m)·mean_{i: y_i=c} z_i`, then row‑normalized.

Every `K_P = 50` steps: `M = μ − 1μ̄ᵀ` (centered), thin SVD `M = UΣVᵀ`,
```
r = min{ k : Σ_{i≤k} σ_i² ≥ 0.99 Σ_i σ_i² }         (r ≤ min(C−1, d))
P = V_{:r} V_{:r}ᵀ ,   Q = I_d − P
```
`Q` is symmetric idempotent and **detached** (no gradient to `μ`, `V`, or `θ` through it). `rank(Q) = d − r`. On CUB/Cars `r ≈ 97–99`; on SOP/In‑Shop `r → d` and `Q → 0`, so the method self‑disables (this is a prediction, not a bug — §5.2).

### 1.3 Provisioning loss

Second view `x' = A_strong(x)` at **128×128** (SimCLR aug: RRC scale (0.2,1), hflip, color‑jitter 0.8·(0.4,0.4,0.4,0.1), grayscale p=0.2, blur p=0.5). Shared weights, **separate auxiliary BatchNorm affine + running stats** for the aux branch (AdvProp‑style); test uses the main BN buffers only.

Blind component and its energy (denominator **detached** — see §2.3):
```
v  = Q z̃ ,   u = v / (‖v‖ + ε) ,  ε = 1e‑6
e  = ‖v‖² / sg(‖z̃‖²)  ∈ [0,1]
```

Symmetric NT‑Xent over the `N=180` source images of the batch, temperature `τ=0.2`, with **all other source images as negatives regardless of class label**:
```
ℓ(a,b) = − log [ exp(a·b/τ) / ( exp(a·b/τ) + Σ_{j≠i} exp(a·u_j/τ) + Σ_{j≠i} exp(a·u_j'/τ) ) ]
L_aux  = (1/2N) Σ_i [ ℓ(u_i, u_i') + ℓ(u_i', u_i) ]
```

Energy floor, ramped:
```
L_e = (1/N) Σ_i [ max(0, γ_t − e_i) ]² ,   γ_t = γ · min(1, t / T_ramp)
γ = 0.25 ,  T_ramp = 20 epochs
```

**Total:** `L = L_PA + β·L_aux + η·L_e`, `β = 0.3`, `η = 5.0`. Grid for sensitivity: `β ∈ {0.1,0.3,1.0}`, `γ ∈ {0.10,0.25,0.40}`, `τ ∈ {0.1,0.2}`, `K_P ∈ {20,50,200}`.

### 1.4 Gradient paths

- `L_PA → z → z̃ → φ, θ`. Unchanged from base.
- `L_aux → u → v = Qz̃`. Since `∂u/∂v = (I − uuᵀ)/‖v‖` and `u ∈ range(Q)`, and `∂v/∂z̃ = Q`, we get `∇_{z̃} L_aux ∈ range(Q)`. **Exact orthogonality to the class‑centroid span at every step** — the self‑supervised signal is structurally incapable of rotating the discriminative subspace. This is the mechanism, in one line.
- `L_e → v` only (denominator detached), so `∇_{z̃} L_e = −2·1[e<γ_t]·(γ_t−e)·Qz̃/sg(‖z̃‖²) ∈ range(Q)`. It can only *grow* the null component; it cannot be satisfied by shrinking the discriminative component.

### 1.5 Test time

`descriptor = z̃/‖z̃‖ ∈ S^511`. No projector, no aux head, no aux BN, no second view, no reranking. Cosine NN over the gallery.

---

## 2. Causal error mode and proof‑level degeneracy attack

**Error mode — label‑rank starvation.** Let `Φ = span{μ_c}`, `dim Φ ≤ C−1`. For any loss `L(z, y)` that depends on `z` only through similarities to class‑indexed objects, the gradient `∇_z L` lies in `Φ` (proxy losses: literally the proxy span; pair/tuple losses: the span of same/different‑class differences, which is contained in `Φ` in expectation). Therefore the `d−C+1` orthogonal directions are driven only by weight decay, initialization residue, and simplicity bias — they decay. At test time, an unseen identity that differs from its nearest seen‑class‑span neighbour **only** along those directions is metrically invisible. This is why CUB (100 train classes, 512 dims) is far harder than SOP (11 318 train classes) relative to its intra‑class difficulty.

**Degeneracy 1 — rank collapse / neural‑collapse ETF.** Could the net satisfy everything with `u` confined to few dimensions? No. NT‑Xent with `N` mutually‑negative anchors has the Welch‑bound obstruction: for unit vectors, `max_{i≠j} u_i·u_j ≥ −1/(N−1)`, with equality only for a regular simplex, which requires `N−1` dimensions. With `N=180` the aux loss cannot approach its optimum unless the effective rank inside `range(Q)` reaches `min(N−1, d−r) = 179` on CUB. **Provable lower bound on provisioned rank: 179 ≫ 0.**

**Degeneracy 2 — energy shrink (the cheapest cheat).** `u` is scale‑invariant, so the net could satisfy `L_aux` with `‖Qz̃‖ = 10⁻⁶`, leaving the deployed cosine metric untouched. `L_e` closes this, and — per the prompt's warning — the normalization here is **not** harmless: `z` is L2‑normalized and the head carries weight decay, so relative energy between `P` and `Q` components is fully operational. Two consequences I commit to: (i) the detached denominator is load‑bearing (without it, `L_e` is minimized by shrinking `Pz̃`, which destroys discriminability); (ii) **`γ=0` must produce ≤0.3 pt of the gain** — this is my sharpest falsifier (§4, C5).

**Degeneracy 3 — class copying.** Could the net just replicate class information into `Q`? At the NT‑Xent optimum, same‑class images are negatives, so `E[u | y=c] ≈ 0` for every `c`: any class‑aligned mean in the blind channel makes same‑class negatives mutually attractive and is penalized. **Class‑exchangeability of the blind channel is a corollary of the negative set, not an extra term.** Measurable: linear probe of *seen* class from `u` should stay near chance‑plus‑ε while probe of *unseen* class from `u` rises.

**Degeneracy 4 — low‑level instance shortcuts.** Instance discrimination can be solved by JPEG/crop/color residue. Strong second‑view augmentation removes the standard ones. **Residual, unpatched risk: background.** Backgrounds are instance‑level and augmentation‑stable, so they can legitimately colonize `Q`. Mitigation is only partial (RRC scale 0.2, blur); I state this as the most likely reason for a null result on CUB, and I pre‑register the diagnostic: probe background/pose from `u`.

---

## 3. Adversarial novelty search — nearest works and one‑line distinctions

**Inside DML**
- **DiVA (ECCV 2020)** — multi‑task chunks (discriminative / shared / intra / self‑supervised) on a *statically partitioned* coordinate block; NSP's subspace is the *dynamically measured null space of the class‑centroid span*, giving exact per‑step gradient orthogonality that a fixed coordinate block cannot provide.
- **S2SD (ICML 2020)** — distills from higher‑dimensional auxiliary embeddings into the deployed one; NSP adds no teacher and no extra embedding, it re‑routes an existing signal into a *provably unsupervised* subspace.
- **BIER / A‑BIER (ICCV'17/TPAMI)** — boosting over embedding groups with activation decorrelation; NSP's auxiliary learners are not residual boosters on the same label task, they optimize a *different* (instance) task in a label‑orthogonal subspace.
- **PFML / SoftTriple multi‑proxy** — fills sub‑class structure using the label loss's own discovered modes (self‑confirming); NSP fills it with an *independent* signal, and the two are predicted to overlap partially (§5.3).
- **Proxy Synthesis / EE / HDML** — synthesize fake classes/embeddings by interpolation; NSP synthesizes nothing and adds no generator.
- **Roth et al., "Revisiting Training Strategies" (ICML 2020)** — ρ‑spectrum regularization *flattens the singular spectrum* (a scale prescription); NSP prescribes *content* for specific directions and predicts that the pure‑energy control (C7) yields ~0 gain, which is precisely the ρ‑style intervention.
- **SLADE, HIER, ProxyNCA++** — extra unlabeled data / hyperbolic hierarchy / recipe engineering respectively; none defines a label‑null subspace.

**Outside DML**
- **INLP (Ravfogel et al., ACL 2020)** — iterative null‑space projection *removes* an attribute at inference; NSP *populates* the null space at train time and deploys the full vector.
- **PCGrad / gradient surgery** — projects gradients only when they conflict, in parameter space; NSP projects the *readout* unconditionally in feature space, so orthogonality is structural rather than conflict‑triggered.
- **VICReg / Barlow Twins** — variance/decorrelation applied to all coordinates symmetrically; NSP applies no decorrelation and treats the two subspaces asymmetrically by construction.
- **Neural Eigenmaps / NeuralEF (Deng et al. 2022)** — learns *ordered* eigenfunctions of a single augmentation or label kernel; NSP uses two kernels with a hard rank partition between them rather than one ordered spectrum.
- **AdvProp (Xie et al. 2020)** — auxiliary BN for a second distribution; used here as a component, claimed as engineering, not novelty.
- **Matryoshka (NeurIPS 2022)** — nested prefixes for efficiency, not for unseen‑class capacity.

I could not construct a nearest work that both (a) defines the supervised subspace from the empirical class‑centroid spectrum online and (b) imposes an operational energy floor on its complement.

---

## 4. Decisive matched‑compute controls

All at 5 seeds, identical data order, identical augmentation seeds where possible.

| ID | Control | Purpose | Prediction (CUB Δ vs base) |
|---|---|---|---|
| C1 | Base, 200 ep | reference | 0 |
| C2 | Base, 266 ep (compute‑matched to NSP's 1.33×) | "it's just more compute" | ≤ +0.2 |
| C3 | Aux NT‑Xent on **full** `z` (no projection), same views/β | "it's just SSL" | +0.3 to −0.5 (interference) |
| C4 | Aux on **fixed coordinate block** dims 100–512 (DiVA‑style) | "any fixed subspace works" | +1.0 (≈50% of gain) |
| C5 | **γ = 0** (aux, no energy floor) | energy floor is load‑bearing | **≤ +0.3** |
| C6 | Aux with same‑class negatives removed | exchangeability driver | +1.0, and seen‑class probe from `u` rises |
| C7 | `β=0, η>0` (energy floor only) | "renamed spectral regularizer" | ≈ 0 |
| C8 | NSP with `r` forced to `d` (`Q=0`) | sanity: method inert | 0 by construction |
| **NSP** | full | — | **+2.0** |

Non‑R@1 mechanism evidence, pre‑registered: unseen‑class linear probe from `u` alone (expect +3 pts or more vs C3); seen‑class probe from `u` (expect flat); measured `r_99/d` per dataset; fraction of test NN decisions changed by zeroing `Q` (expect 8–15% on CUB, <2% on SOP).

---

## 5. Frozen forecasts (Lane A, 5 seeds, ±1 s.d.)

### 5.1 Own matched reproductions and NSP

| | CUB | Cars196 | SOP |
|---|---|---|---|
| PA base, 200 ep (my repro) | 0.699 ± 0.005 | 0.882 ± 0.005 | 0.803 ± 0.003 |
| **PA + NSP** | **0.719 ± 0.006** | **0.897 ± 0.006** | **0.806 ± 0.003** |
| Multi‑proxy stand‑in (K=15/2) | 0.716 ± 0.006 | 0.898 ± 0.006 | 0.812 ± 0.003 |
| **Multi‑proxy + NSP** | **0.729 ± 0.007** | **0.908 ± 0.006** | **0.815 ± 0.003** |

### 5.2 Pre‑registered dose–response law
`ΔR@1 ≈ κ·(1 − r_99/d)`, `κ = 2.4` pts, measured *before* the R@1 runs. Expect `1 − r_99/d ≈ 0.80` (CUB/Cars) and `≈ 0.10–0.25` (SOP, In‑Shop). Hence SOP Δ ≈ +0.24 to +0.60, and **In‑Shop is forecast at parity — I do not target the 0.930 PA+DADA reference and claim no result there.**

### 5.3 Frontier arithmetic (stated plainly)
PFML: CUB 0.734 ± 0.003, Cars 0.927 ± 0.003, SOP 0.829 ± 0.002.
My best forecast is **0.729 ± 0.007 CUB and 0.908 Cars — this does not cross the frontier.** Crossing is *conditional*: if a faithful PFML reproduction lands at 0.734, NSP on top is forecast at **0.744 ± 0.007** (+1.0, discounted ~50% for overlap with multi‑proxy's own use of the blind subspace), i.e. ≈ 1.3σ_combined above 0.734 — **P(cross) ≈ 0.25–0.30**. Cars conditional: 0.934 ± 0.006 vs 0.927. I am forecasting a *mechanism with an additive effect*, not a frontier win.

### 5.4 Falsification thresholds
Any one of these refutes NSP as specified: CUB Δ over its own base < **+0.8**; C5 recovers ≥70% of the gain; C3 comes within 0.3 pt of NSP; C7 exceeds +0.5; SOP Δ exceeds the law by >0.5 pt; unseen‑class probe from `u` improves <3 pts.

### 5.5 Unresolved source ambiguities
PFML's expansion, loss form, pooling, LR/epoch schedule, batch size, augmentation and whether 0.734 is GAP or concat‑pooled 512‑D are **unknown to me**; I therefore treat it only as an external reference and never as a base I have inherited. PA+DADA's seed count and variance are unreported. Proxy‑Anchor's ResNet‑50/512 published numbers are recalled, not verified — hence every base row above is a re‑measurement, not a citation. AdvRF and VAPNet are Lane B and are not used.

---

## 6. Cost, benchmark and contamination risk

**Training cost:** one extra 128×128 forward+backward ≈ 0.33× FLOPs → **≈1.33× epoch time, ≈1.20× peak memory** (materially more than DADA's 1.06×/1.01× — the C2 compute‑matched control is mandatory, not optional). EMA table `C×512` (23 MB on SOP); thin SVD every 50 steps (~0.2 s on SOP, negligible on CUB; randomized SVD if needed). Aux BN adds ~53 K training‑only parameters.

**Deployment cost:** identical to the base — one ResNet‑50, one 224 px view, one 512‑D vector, cosine NN. Zero inference overhead.

**Risks.** (i) *Background colonization of the blind channel* — the most probable cause of a null on CUB. (ii) *Two‑resolution BN* — aux BN mitigates but does not eliminate distribution mixing in the shared convolutional stats. (iii) *γ is a genuine discriminability/capacity trade‑off*; too large will degrade CUB, and the sensitivity sweep may show a narrow optimum. (iv) *200 epochs* on CUB/Cars is far beyond the source recipes and invites overfitting that could swamp the effect. (v) **Contamination:** CUB (birds) and Cars overlap ImageNet‑1K categories, so part of any gain may be *preservation of permitted pretrained rank* rather than new learning — I pre‑register a random‑init or self‑supervised‑init probe run to bound this, and note that a large gain there would strengthen, not weaken, the claim. (vi) CUB's 100‑class / 5 924‑image test set gives ~±0.5 pt seed noise, comparable to the effect size — hence 5 seeds and paired‑seed deltas rather than best‑run reporting.

---

**Position.** I return this as the strongest complete method I can defend, not as a frontier claim. Its distinguishing bet is falsifiable and unusual: the gain must vanish on SOP and In‑Shop, must vanish without the energy floor, and must vanish when the same self‑supervision is applied unprojected. If any of those three survives, the mechanism I am claiming is not the one doing the work.
