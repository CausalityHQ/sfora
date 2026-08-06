## Answer: ONE method — **Factorial Proxy Codes (FPC)**

**Chosen lane: Lane A** (ResNet‑50, 512‑D normalized descriptor, ~224 px, single‑view cosine, 200 epochs). All forecasts and comparisons below are Lane A only.

**One‑line statement.** Keep the Proxy‑Anchor loss exactly, but delete the free per‑class proxy: a class's proxy is a *balanced, injective codeword over m shared sub‑codebooks*, and the code is annealed from soft to hard so that in the terminal phase of training the head carries exactly `m·log₂C` bits per class and nothing more.

---

## 1. Executable mathematics

### 1.1 Architecture and deployment

* Backbone `g_θ`: ImageNet‑1K‑pretrained ResNet‑50 → GAP → `h(x) ∈ ℝ²⁰⁴⁸`.
* Embedding: `z̃(x) = W_e h(x) + b_e`, `W_e ∈ ℝ^{512×2048}`; **deployed descriptor** `z(x) = z̃(x)/‖z̃(x)‖₂ ∈ S⁵¹¹`.
* Train aug: `RandomResizedCrop(224, scale=(0.16,1))` + horizontal flip. Test: `Resize(256)` + `CenterCrop(224)`, one view, cosine 1‑NN. Head is discarded at test. Deployment cost is bit‑identical to a Proxy‑Anchor model.

Block split: `z = (z^{(1)},…,z^{(m)})`, each `z^{(b)} ∈ ℝ^{d_b}`, `d_b = 512/m`.

### 1.2 Learned objects

| object | shape | role | lr |
|---|---|---|---|
| `θ, W_e, b_e` | backbone + embedding | encoder | `η` |
| `U^{(b)} = [u^{(b)}_1…u^{(b)}_C]` | `m × C × d_b`, `‖u‖₂ = 1` | **shared codewords** (class‑agnostic) | `100η` |
| `A^{(b)}_c ∈ ℝ^C` | `K × m × C` | class→code assignment logits | `100η` |

Unit norm on `U` is enforced by **projection after each optimizer step**, and weight decay is **set to 0 on `U` and `A`** (otherwise decay fights the projection and the codeword scale becomes an operational, uncontrolled hyperparameter).

### 1.3 Balanced assignment (entropic OT, per block)

Per block, solve over `P^{(b)} ∈ ℝ^{K×C}_{≥0}`:

```
P^(b) = argmax_P  ⟨P, A^(b)⟩ + τ_t H(P) − ρ·KL(Pᵀ1 ‖ k),   P1 = r = (1/K)1_K,  k = (1/C)1_C
```

3 warm‑started Sinkhorn iterations, unrolled (gradient flows to `A`). `π^{(b)}_c := K·P^{(b)}_{c,:} ∈ Δ^{C−1}`. `ρ = 10` (near‑hard balance); `ρ = 0` recovers a free softmax and is an ablation. Cost: `K×C = 100×8` per block — negligible.

### 1.4 Composed proxy (the factorial code manifold)

```
μ^(b)_c = ( Σ_j π^(b)_{cj} u^(b)_j ) / ‖ Σ_j π^(b)_{cj} u^(b)_j ‖₂  ∈ S^{d_b−1}
w_c     = ( m^{-1/2} μ^(1)_c , … , m^{-1/2} μ^(m)_c )  ∈ S^{511}
```

`M_{C,m} := {w_c}` is a product of `m` spherical hulls of `C` points: **`C^m` vertices, but full 512‑dimensional span.** (This is the distinction from a low‑rank head — see §3.19.)

### 1.5 Main loss — Proxy‑Anchor, reproduced exactly

With `s(x,p) = ⟨z(x), w_p⟩`, `α = 32`, `δ = 0.1`, `P` all proxies, `P⁺` proxies present in the batch, `X⁺_p` batch samples of class `p`, `X⁻_p` the rest:

```
L_PA = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{x∈X⁺_p} e^{−α(s(x,p) − δ)})
     + (1/|P|)  Σ_{p∈P}  log(1 + Σ_{x∈X⁻_p} e^{+α(s(x,p) + δ)})
```

**FPC = `L_PA` with `W` constrained to `M_{C,m}`.** No other change to the loss.

### 1.6 Structural terms (all image‑free, head‑only)

```
overlap      o(c,c') = (1/m) Σ_b ⟨π^(b)_c, π^(b)_{c'}⟩
T1 injectivity  L_inj = mean_{c<c'} [ o(c,c') − (1 − h/m) ]²_+          h = 2
T2 decorrelation L_dec = mean_{b<b'} KL( J^(bb') ‖ q^(b) ⊗ q^(b') ),
                 J^(bb') = (1/K)Q^(b)ᵀQ^(b'),  q^(b) = (1/K)Q^(b)ᵀ1
T3 codeword sep  L_sep = mean_b mean_{j≠j'} [ ⟨u^(b)_j, u^(b)_{j'}⟩ − 0 ]²_+
T4 block energy  L_nrm = E_x Σ_b ( ‖z^(b)(x)‖²/‖z(x)‖² − 1/m )²
L = L_PA + 1.0·L_inj + 0.5·L_dec + 1.0·L_sep + 0.1·L_nrm
```

For `K > 2000` (SOP, In‑Shop), `L_inj` is estimated on 1024 classes sampled per step (the `K×K` overlap matrix is 512 MB at `K = 11318` — this is a real memory constraint, not a detail).

### 1.7 Schedules and the soft→hard capacity anneal

* `τ_t = 1.0 → 0.05`, exponential over epochs 0–120.
* **Hard phase from epoch `T_h = 120` (0.6·T):** `π̂^{(b)}_c = onehot(argmax_j P^{(b)}_{cj})`, straight‑through (`π̂ ← π + sg(π̂ − π)`).
* **Code repair (hard phase, deterministic, `O(K²m)`):** if two classes collide, the one with lower training margin has its lowest‑confidence block overridden by that block's runner‑up code, until all codewords are distinct with Hamming distance ≥ `h`. This makes Proposition 3 hold by construction rather than by hope.
* Defaults: CUB/Cars `m = 8, C = 8` (`C^m = 1.7×10⁷ ≫ K`, 24 bits/class); SOP/In‑Shop `m = 8, C = 16` (32 bits/class).
* Optimizer AdamW, `η = 1e‑4`, wd `1e‑4` (backbone/embedding only), batch 180, cosine LR over 200 epochs with 5‑epoch warm‑up, plus Proxy‑Anchor's 1‑epoch backbone‑frozen warm‑up extended to 2 epochs; `bn_freeze = 1` for CUB/Cars.

### 1.8 The gradient path that *is* the mechanism

`u^{(b)}_j` receives gradient from **every class assigned to it**. In the hard phase each codeword is a group prototype updated by `K/C` classes — on CUB, ~733 training images per codeword per epoch instead of ~30 per class. That is the entire causal claim in one sentence: *supervision is moved from a 30‑image‑per‑target regime to a 733‑image‑per‑target regime, and the targets are chosen by the optimizer to be the groupings that are actually learnable.*

---

## 2. Causal zero‑shot error mode and proof‑level attack on degeneracies

**Error mode: head‑side class‑index memorization.**

> **Proposition 1 (the free‑proxy objective cannot exclude maximal collapse).** Let `φ` map every image of class `c` to a fixed unit vector `v_c`, with `max_{c≠c'} ⟨v_c,v_{c'}⟩ = ρ₀`. Setting `w_c = v_c` gives `L_PA ≤ 2 log(1 + n·e^{−α(1−δ−ρ₀−δ)}) → 0` as `α(1−ρ₀−2δ) → ∞`. Such a `φ` has rank ≤ `K−1` on the training set and is constant within each class. Hence the global‑minimizer set of `L_PA` *contains* the maximal‑collapse encoder, which retains only `K−1 ≤ 99` of 512 directions and is at chance on unseen classes beyond whatever the ImageNet initialization survives. ∎

This is the causal error mode: nothing in a free‑proxy objective distinguishes an attribute‑structured solution from a class‑index solution, and the terminal phase of training (where neural collapse occurs) drives toward the latter. Kornblith et al. (NeurIPS 2021) give the matching empirical regularity — objectives that increase class separation transfer *worse*.

**Attacks on the cheapest degeneracies of FPC itself.**

> **Proposition 2 (no class‑private directions exist).** In the hard phase, if `κ_b(c) = κ_b(c′)` then `μ^{(b)}_c = μ^{(b)}_{c′}`, so
> `⟨z(x),w_c⟩ − ⟨z(x),w_{c′}⟩ = m^{-1/2} Σ_{b: κ_b(c)≠κ_b(c′)} ⟨z^{(b)}(x), μ^{(b)}_c − μ^{(b)}_{c′}⟩`.
> Every direction used to separate two classes is a codeword direction shared with `K/C − 1` other classes. The head offers no class‑private direction to descend into. ∎

> **Proposition 3 (quantified capacity).** In the hard phase the head's class‑attributable information is exactly the codeword: `I(c ; head) ≤ m log₂C` = **24 bits** (CUB/Cars), versus 512 continuous dimensions for a free proxy. All remaining class‑specific information must be computed by the encoder from the image. ∎

> **Proposition 4 (balance kills the three cheap shortcuts).**
> (a) *One block does everything* — balanced column marginals partition the classes into `C` equal groups per block, so one block separates at most `C < K` classes. Under `ρ`‑relaxed balance the achievable imbalance obeys `KL(Pᵀ1‖k) ≤ (⟨P*,A⟩ − ⟨P_bal,A⟩)/ρ`; buying "one class alone in its own code" costs ≈ `ρ·log C = 20.8` nats of assignment‑logit advantage at `ρ = 10, C = 8`.
> (b) *All blocks identical* — then `H(code) = log C`, at most `C` classes are distinguishable, `L_inj` is violated for `K − C` classes and code repair forces distinct codes; `L_dec` penalizes it directly. Different thresholds of a single nuisance factor across blocks produce high mutual information and are penalized by the same term.
> (c) *Within‑block codebook collapse* — the block contributes an identical component to all proxies and therefore zero to every pairwise proxy difference, shrinking every achievable margin by `√((m−1)/m)`; `L_sep` penalizes it explicitly. ∎

> **Lemma 1 (train/test scale is not assumed harmless — it is derived).**
> `⟨z(x),w_c⟩ = Σ_b (‖z^{(b)}(x)‖/‖z(x)‖)·m^{-1/2}·⟨ζ^{(b)}(x),μ^{(b)}_c⟩ ≤ 1` by Cauchy–Schwarz, with equality **iff** `‖z^{(b)}(x)‖²/‖z(x)‖² = 1/m` for all `b` **and** `ζ^{(b)}(x) = μ^{(b)}_c` for all `b`. Since `α = 32, δ = 0.1` drives `s → 1`, block‑energy equalization and per‑block alignment are consequences of the loss, so the factorial structure is transported into the *deployed* global cosine rather than living only in training logits. `L_nrm` only accelerates this (ablation C7). ∎

**The honest boundary (stated, not hidden).** FPC cannot make the class‑index solution *infeasible*: the encoder is a universal function of the image and can compute all `m` group indicators on training data. FPC changes (i) which solutions are reachable in the terminal phase — the head supplies only group‑level target directions — and (ii) the statistical regime of each supervision signal (733 vs. 30 images per target on CUB). This is an implicit‑bias and sample‑complexity claim, **not** an impossibility theorem, and it is the first thing a reviewer should attack.

---

## 3. Adversarial novelty search, with a one‑sentence distinction each

**Inside DML.**
1. **Proxy‑Anchor (CVPR 2020)** — one free proxy per class; FPC keeps the loss verbatim and removes the free proxy.
2. **ProxyNCA / ProxyNCA++** — same free per‑class proxy.
3. **SoftTriple (ICCV 2019) / Sub‑center ArcFace (ECCV 2020) / PFML (CVPR 2025, 15 proxies/class)** — *more* centers per class for intra‑class multimodality (PFML additionally a distance‑decaying potential field); FPC has *fewer than one* free center per class and shares all centers *across* classes.
4. **ProxyGML (NeurIPS 2020)** — fewer proxies, but each remains class‑owned and free; FPC's codewords are class‑agnostic and identity is a combination *index*.
5. **Hierarchical proxy losses (HIST and relatives)** — a single clustering‑derived *tree* over proxies; FPC imposes `m` simultaneous, decorrelated, balanced groupings solved by OT, giving `C^m` rather than one hierarchy.
6. **Deep Factorized Metric Learning (CVPR 2023)** — factorizes the *backbone* into sub‑blocks with a learned router allocating *samples*; FPC factorizes the *class‑parameter layer* and touches neither backbone nor routing.
7. **BIER / A‑BIER / ABE / DREML** — group diversity via boosting sample weights or attention over independent learners; FPC's blocks are not learners, get no reweighting, and are tied by one shared codebook and one loss on the composed proxy.
8. **Proxy Synthesis (AAAI 2021), Memory‑based Virtual Classes (ICCV 2021), Learning to Generate Novel Classes (2022), Embedding Expansion, HDML/DAML, DVML** — *add* synthetic classes/samples; FPC adds nothing and *subtracts* head capacity.
9. **Non‑isotropy Regularization (CVPR 2022)** — normalizing‑flow penalty on class‑conditional shape; FPC constrains proxy *locations* and adds no generative model.
10. **Anti‑Collapse / coding‑rate (MCR²) DML** — a volume term on features; FPC has no feature‑space term at all.
11. **DADA, PA+DADA (AAAI 2024)** — augmentation/original domain adaptation; orthogonal and combinable, not competing on mechanism.
12. **AdvRF (ICCV 2025), VAPNet (NeurIPS 2023)** — auxiliary reconstruction/attribute networks; FPC's entire extra machinery is `C·512 + K·m·C` scalars.
13. **Group Loss / intra‑batch connections** — proxy‑free batch label propagation; FPC is proxy‑based with structured proxies.

**Outside DML.**
14. **ECOC (Dietterich & Bakiri 1995; Deep ECOC 2023)** — fixed/random binary codes decoded by independent binary classifiers for error correction on *seen* classes; FPC learns the code jointly with the representation under a balance constraint, its "bits" are vector‑valued sub‑descriptors over a learned shared codebook, and its purpose is unseen‑class transfer.
15. **Product quantization (PQN, SPQ, DPQ, VQ‑AE product codebooks)** — quantizes the *deployed descriptor* for memory/ANN speed; FPC quantizes the *class target* at train time and deploys an unquantized float descriptor.
16. **SwAV / Sinkhorn‑Knopp online clustering; Sinkhorn Label Allocation (ICML 2021)** — balanced OT over *samples* to produce pseudo‑labels; FPC's OT is over *classes* with labels fully known, and bounds per‑class head capacity.
17. **VQ‑VAE / Gumbel annealing** — discretizes the *representation* as a bottleneck; FPC discretizes the *supervision target* and leaves the representation continuous.
18. **Fixed / simplex‑ETF classifiers ("Fix your classifier", neural‑collapse heads)** — remove head learnability but keep one arbitrary equidistant direction per class, so class geometry cannot reflect visual similarity; FPC keeps the head learnable but shared and quantized, so similar classes are *forced* to share codewords.
19. **Low‑rank / label‑embedding heads (DeViSE, ESZSL)** — proxies restricted to an `r`‑dimensional subspace, which *reduces* usable descriptor directions; FPC's product‑of‑hulls spans all 512 dimensions while having only `C^m` vertices — a combinatorial, not a rank, restriction. **This is the distinction I would defend hardest, and control C1 is designed to test exactly it.**
20. **Kornblith et al., "Why do better loss functions lead to less transferable features?" (NeurIPS 2021)** — a diagnosis of the separation/transfer tradeoff; FPC is a train‑time mechanism acting on head capacity, not a measurement.

---

## 4. Decisive matched‑compute controls

All at ResNet‑50 / 512‑D / 224 px / 200 epochs / identical sampler, augmentation, optimizer, 5 seeds.

* **C0 — PA reproduced** (free proxies), same recipe and epoch count. Every delta is measured against this; no number is inherited.
* **C1 — low‑rank head** `W = ΘV`, `V ∈ ℝ^{56×512}`, matching FPC's soft‑phase continuous dof per class (`m(C−1) = 56`). *If C1 ≈ FPC, the mechanism is rank restriction, not factorial structure.* **Decisive.**
* **C2 — random frozen codes** (`κ(c)` uniform and fixed, codebook still learned). *If C2 ≈ FPC, the learned assignment is inert.*
* **C3 — no balance** (`ρ = 0`). *If C3 ≈ FPC, balance is inert; if C3 collapses to one code per class, Prop. 4a is confirmed necessary.*
* **C4 — `m = 1, C = K`**, which reduces algebraically to PA. Sanity anchor.
* **C5 — head learning‑rate control**: PA with proxy lr ∈ {×1, ×100} and FPC with `U,A` lr ∈ {×1, ×100}, since the ×100 multiplier was moved onto new objects.
* **C6 — soft only** (no hard phase). Isolates capacity quantization from shared‑codebook structure.
* **C7 — drop each of `λ_inj, λ_dec, λ_sep, λ_nrm` to zero.**
* **C8 — SoftTriple with centers chosen so head parameters equal FPC's.** Separates "more centers per class" from "centers shared across classes".
* **C9 — sampler control**: PA‑official random 180 vs. class‑balanced 45×4, both arms.
* **C10 — capacity sweep** `(m,C) ∈ {(4,8),(8,8),(8,16),(16,4)}` → bit budgets {12, 24, 32, 32}, plus free proxies (∞). **Predicted mechanism signature: non‑monotone R@1 with an interior optimum a few bits above `log₂K`.** *If R@1 rises monotonically with capacity up to free proxies, the capacity claim is falsified.*

Corroborating diagnostics (not the claim): spectral decay `ρ` (Roth et al., ICML 2020) should *decrease* vs. PA; code‑group coherence measured against CUB attribute annotations **from the training split only, post hoc, never in training**.

---

## 5. Frozen forecasts, thresholds, and frontier arithmetic (Lane A)

**Baseline I must reproduce (not inherit), 5 seeds:** PA @ R50/512/224/200 ep → CUB `0.690 ± 0.005`, Cars `0.885 ± 0.005`, SOP `0.797 ± 0.003`, In‑Shop `0.888 ± 0.004`.

**FPC forecast, frozen, 5 seeds:**

| dataset | FPC R@1 | Δ vs. my PA repro | reference |
|---|---|---|---|
| **CUB‑200‑2011** | **0.712 ± 0.006** | +2.2 | PFML 0.734 ± 0.003; DADA row 0.729 |
| **Cars196** | **0.905 ± 0.006** | +2.0 | PFML 0.927 ± 0.003; DADA row 0.921 |
| SOP (secondary) | 0.806 ± 0.004 | +0.9 | PFML 0.829 ± 0.002; DADA row 0.810 |
| In‑Shop (secondary) | 0.900 ± 0.005 | +1.2 | PA+DADA 0.930 (seeds unreported) |

**Frontier arithmetic.** A non‑overlapping Lane‑A crossing of PFML needs CUB ≥ `0.734 + 2(0.003) = 0.740` and Cars ≥ `0.927 + 0.006 = 0.933`. My point forecasts fall short by **0.028** on both. Sensitivity: if a 200‑epoch cosine‑schedule PA reproduction lands at 0.700 rather than 0.690, FPC becomes 0.722 — still 0.012 short. **I therefore forecast that FPC as specified does not cross the Lane‑A frontier, and I state that plainly rather than inflating it.** Subjective probabilities: crossing PFML on CUB ≈ **8%**; on Cars ≈ **7%**; beating my reproduced PA by ≥ 1.0 point on *both* CUB and Cars ≈ **55%**.

**Pre‑registered falsification thresholds.**
* **F1** — FPC − PA(repro) < +1.0 on CUB **and** < +1.0 on Cars (paired same‑seed, Welch p > 0.05) ⇒ falsified as a gain mechanism.
* **F2** — C1 (low‑rank, r = 56) within 0.5 of FPC on both ⇒ **factorial** claim falsified.
* **F3** — C2 (random frozen codes) within 0.5 ⇒ **learned assignment** claim falsified.
* **F4** — C10 monotone in bit budget up to free proxies ⇒ **capacity** claim falsified.
* **F5** — C3 (`ρ = 0`) matches FPC ⇒ **balance** claim falsified.
F2–F5 falsify the *mechanism* even if F1 passes; a real gain from an unclaimed cause is not support.

---

## 6. Cost, benchmark and contamination risk

**Cost.** Extra head parameters: `C·512 + K·m·C` = 4,096 + 6,400 = **10,496** on CUB, versus PA's 51,200 free proxies — FPC has *fewer* head parameters. Sinkhorn: 3 iterations on `100×8` per block per step. `L_inj/L_dec`: `O(K²m + m²C²)` ≈ 8×10⁵ flops/step. **Forecast 1.00–1.02× epoch time, 1.00× memory** (compare PA+DADA at ~1.06× / 1.01×). Deployment: identical to the baseline — one forward pass, one 512‑D descriptor, cosine NN, zero overhead. The one real scaling constraint is the `K×K` overlap matrix at SOP scale, handled by the 1024‑class subsample stated in §1.6.

**Risks.**
* ImageNet‑1K pretraining overlaps CUB/Cars semantics; this is inherited from the standard protocol and affects all Lane‑A methods equally, but absolute numbers are not evidence of zero‑shot ability per se.
* CUB's ~5,924 test queries with 0.003–0.006 seed std means differences below ~0.01 are not decisive; I use 5 seeds and paired tests and claim no crossing.
* SOP R@1 is dominated by near‑duplicate product photos, so SOP is weak evidence for the mechanism; In‑Shop and SOP are the datasets where forced code sharing is most likely to *hurt* fine instance‑level distinctions, and I would not be surprised by a small In‑Shop regression.
* Protocol drift is the main comparison hazard: PA's official budget is 60 epochs for CUB/Cars while this lane specifies 200. I run **both** arms and never compare a 200‑epoch FPC against a 60‑epoch PA.
* Attribute‑coherence analysis uses training‑split CUB annotations only, post hoc.

---

## 7. Unresolved source ambiguities (declared)

1. **PFML** — the prompt cites CVPR 2025; the primary preprint is arXiv:2405.18560 (2024) with a CVPR 2025 CVF entry (Bhatnagar et al.), mechanism = superposed attraction/repulsion potential fields with proxies augmenting the batch field. I did **not** verify its epoch budget, optimizer, or the PA/SoftTriple baseline rows it reports. I treat 0.734 / 0.927 / 0.829 as given.
2. **Proxy‑Anchor recipe** — I am confident about AdamW, lr 1e‑4, wd 1e‑4, `α = 32`, `δ = 0.1`, batch 180, `warm = 1`, `bn_freeze = 1`, 512‑D, RandomResizedCrop+flip. I am **not** certain of the proxy‑lr multiplier (×100 is recollection from the official repo), per‑dataset LR‑decay step/γ, or SOP/In‑Shop LR (6e‑4?) and epoch counts. These must be read from the official repository before the run; the baseline must be reproduced, not inherited.
3. **DADA** = "Towards Improved Proxy‑Based DML via Data‑Augmented Domain Adaptation" (AAAI 2024, arXiv:2401.00617). I did not verify its In‑Shop 0.930 row, backbone, or dimension from primary text.
4. **AdvRF** = "Adversarial Reconstruction Feedback for Robust Fine‑grained Generalization" (arXiv:2507.21742). One secondary snippet reports 66.3% CUB R@1 for a ResNet‑50 + classification‑loss configuration, inconsistent with the prompt's 0.766 — probably a baseline row, unverified. Lane B is not used, so this does not affect my forecasts.
5. **Highest‑priority novelty risk.** A secondary aggregation of the DML literature described an approach that "decomposes class representation into a linear combination of learned vectors shared across classes, using multiple codebooks with individual codes." I could **not** trace this to a primary paper within this consultation. If such a primary work exists, it is the nearest neighbour to FPC and the novelty claim must be re‑adjudicated against it — the surviving distinctions would then be the balanced‑OT assignment, the soft→hard capacity anneal with an explicit bit budget, and the injectivity/decorrelation constraints, not the shared‑codebook idea itself. **The reviewer should resolve this before accepting §3.**

---

**Sources:**
- [Potential Field Based Deep Metric Learning (PFML), arXiv:2405.18560](https://arxiv.org/abs/2405.18560) · [CVPR 2025 CVF entry](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [official code](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Towards Improved Proxy-Based DML via Data-Augmented Domain Adaptation (DADA), AAAI 2024](https://arxiv.org/html/2401.00617v1)
- [Adversarial Reconstruction Feedback (AdvRF), arXiv:2507.21742](https://arxiv.org/pdf/2507.21742)
- [Deep Factorized Metric Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf)
- [Hierarchical Proxy-based Loss for Deep Metric Learning](https://arxiv.org/abs/2103.13538)
- [Sub-center ArcFace, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
- [Proxy Synthesis: Learning with Synthetic Classes for DML](https://ar5iv.labs.arxiv.org/html/2103.15454) · [Memory-based Virtual Classes, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ko_Learning_With_Memory-Based_Virtual_Classes_for_Deep_Metric_Learning_ICCV_2021_paper.pdf) · [Learning to Generate Novel Classes for DML](https://arxiv.org/abs/2201.01008)
- [Non-isotropy Regularization for Proxy-based DML, CVPR 2022](https://arxiv.org/pdf/2203.08547) · [Anti-Collapse Loss (coding rate) for DML](https://arxiv.org/pdf/2407.03106)
- [Deep Error-Correcting Output Codes](https://www.mdpi.com/1999-4893/16/12/555) · [Sinkhorn Label Allocation, ICML 2021](https://arxiv.org/abs/2102.08622)
- [Self-supervised Product Quantization for Image Retrieval](https://arxiv.org/pdf/2109.02244) · [Learning Product Codebooks using VQ-AEs](https://arxiv.org/abs/1807.04629)
- [Why Do Better Loss Functions Lead to Less Transferable Features?](https://arxiv.org/pdf/2010.16402)
