Prompt read; SHA-256 matches (`03cbd57b…f7e53f4`). I acted as blind proposer under its tool policy: public web search/fetch only, no other repository file, no local state, no delegation, no consultation.

---

# PROPOSAL — one method

## Phase‑Quotient Metric Learning (PQML): deploying the maximal torus‑invariant of the embedding sphere

**Lane: A** (matched 512‑D CNN lane — ResNet‑50, 512‑D normalized global descriptor, ~224 px, single view, cosine, 200 epochs). All forecasts and comparisons below are Lane A only.

---

## 0. One‑paragraph statement

Train an embedding of internal width `m = 512 + L`, split its first `2L` coordinates into `L` complex planes, and **deploy the modulus of each plane** together with the untouched remainder. The deployed descriptor is exactly 512‑D, unit‑norm by construction, and read out by ordinary cosine NN. Mathematically this deployed map is the canonical projection onto the quotient of the sphere by an `L`‑torus of isometries, so its cosine similarity equals the *orbit‑maximized* similarity `max_θ ⟨R_θ z, z'⟩` at the cost of a plain inner product. Two auxiliary train‑time terms select *which* image factors are routed into that destroyed channel: one demands the channel absorb a target amount of **within‑class** squared distance, the other demands the channel's phase be **statistically class‑independent**. Together they identify the torus as (an estimate of) the largest group of embedding isometries that preserves the training label partition — a *maximal invariant* in the decision‑theoretic sense — and it is exactly that group that continues to act on unseen identities.

---

## 1. Executable mathematics

### 1.1 Learned objects

| object | shape | init | optimizer group |
|---|---|---|---|
| ResNet‑50 backbone `φ` (ImageNet‑1K init, GAP output) | → ℝ²⁰⁴⁸ | torchvision IN‑1K | base lr |
| head `W ∈ ℝ^{m×2048}`, `b ∈ ℝ^m`, `m = 512 + L`, `L = 32` | 544×2048 | Kaiming‑uniform | base lr |
| proxies `P = {p_c}` (Proxy‑Anchor: 1/class; PFML drop‑in: `M`/class) `p ∈ ℝ⁵¹²` | C×512 (or CM×512) | 𝒩(0, 1/512), ℓ₂‑normalized | 100× base lr |

There are **no other learned parameters**. The torus is a *fixed* coordinate structure, not a learned matrix: because `W` is an unconstrained linear map, any learned orthogonal basis `Q` for the phase planes is absorbed into `W` (`QᵀW` is again a free linear map), so WLOG the planes are the fixed coordinate pairs `(1,2), (3,4), …, (2L−1, 2L)`. This removes an entire parameter block and an entire failure mode.

### 1.2 Forward pass (identical at train and test)

```
h = W·φ(x) + b                    ∈ ℝ^m,  m = 544
z = h / ‖h‖₂                      ∈ S^{m−1}
ζ_l = z_{2l−1} + i·z_{2l}         l = 1..L        (L complex planes)
r   = (z_{2L+1}, …, z_m)          ∈ ℝ^{m−2L} = ℝ^{480}
|ζ_l|_ε = sqrt(z_{2l−1}² + z_{2l}² + ε²),  ε = 1e−6
y = π(z) := ( |ζ_1|_ε, …, |ζ_L|_ε, r )              ∈ ℝ^{512}
```

`‖y‖₂ = ‖z‖₂ = 1` **exactly** (up to `O(Lε²)`), so no renormalization is required and no scale knob is introduced here. **`y` is the deployed descriptor.** Test‑time operation = this same map; retrieval = cosine on `y`. Single model, single view, single fixed 512‑D descriptor. ✔ Lane‑A compliant.

### 1.3 The three group‑theoretic facts the method rests on

Let `T^L = { R_θ = diag(e^{iθ_1},…,e^{iθ_L}) ⊕ I_{m−2L} : θ ∈ [0,2π)^L }`, a compact abelian group acting on `S^{m−1}` by isometries.

**(F1) `π` is a maximal invariant.** `π(z) = π(z') ⟺ z' ∈ T^L·z`. (⇐ trivial; ⇒ equal moduli and equal `r` give `θ_l = arg ζ'_l − arg ζ_l`, well defined wherever `ζ_l ≠ 0`, and where `ζ_l = 0` the orbit is a point.) So `π` is the canonical projection `S^{m−1} → S^{m−1}/T^L`, and nothing coarser than the orbit is lost, nothing finer is kept.

**(F2) Cosine on `y` = orbit‑maximized cosine on `z`.**
```
⟨π(z), π(z')⟩ = Σ_l |ζ_l||ζ'_l| + ⟨r,r'⟩ = max_θ ⟨R_θ z, z'⟩
‖π(z) − π(z')‖² = min_θ ‖R_θ z − z'‖²
```
This is an *identity*, not an approximation. A plain inner product on `y` performs an optimal alignment over an `L`‑dimensional continuous nuisance group. No re‑ranking, no test‑gallery fitting, no second view — the alignment is closed‑form and folded into the descriptor.

**(F3) `π` is a contraction, and the contraction has a closed form.**
```
Δ(z,z') := ‖z−z'‖² − ‖π(z)−π(z')‖² = Σ_l 2( |ζ_l||ζ'_l| − Re(ζ_l · conj(ζ'_l)) )
         = Σ_l 2|ζ_l||ζ'_l| (1 − cos(ψ_l − ψ'_l))  ≥ 0,      ψ_l := arg ζ_l
```
`Δ ≥ 0` by the reverse triangle inequality per plane. `Δ` is *exactly* the squared distance destroyed by the quotient, decomposed per plane, and it is the quantity the two auxiliary losses control.

### 1.4 Losses

Batch: class‑balanced sampler, `n_c = 30` classes × `n_s = 4` images = **120** (matched to Proxy‑Anchor's published batch size). `P_same` = same‑class pairs in batch (30·6 = 180), `P_diff` = a uniformly subsampled set of 1800 different‑class pairs.

**(a) Discriminability — base DML loss, computed on the deployed descriptor `y`.** Proxy‑Anchor (Kim et al., CVPR 2020), `α = 32`, `δ = 0.1`, cosine `s(·,·)`:
```
L_base = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{y∈Y_p⁺} e^{−α(s(y,p)−δ)})
       + (1/|P|)  Σ_{p∈P}  log(1 + Σ_{y∈Y_p⁻} e^{ α(s(y,p)+δ)})
```
For the PFML instantiation, `L_base` is PFML's potential energy `𝒰 = Σ_i Ψ_{y_i}(y_i) + Σ_j Σ_k Ψ_j(p_{j,k})` with `Ψ_j = Ψ_{j,att} + Ψ_{j,rep}` and the disclosed piecewise potentials `ψ_att(r,z) = −1/δ^α` for `‖r−z‖ < δ` else `−1/‖r−z‖^α`, `ψ_rep(r,z) = 1/‖r−z‖^α` for `‖r−z‖ < δ` else `1/δ^α`, with `M = 15` proxies/class on CUB and Cars. **The only change is that its argument is `y = π(z)` rather than `z`.** PQML is loss‑agnostic by construction.

**(b) Nuisance routing (novel term).** Hinge to an absolute target `Δ* = 0.15`:
```
L_inv = (1/|P_same|) Σ_{(i,j)∈P_same} [ Δ* − Δ(z_i, z_j) ]₊
```
This *demands* the phase channel absorb at least `Δ*` of squared within‑class distance. It is a hinge, not a maximization, so it exerts zero pressure once satisfied and cannot drive unbounded energy into the planes.

**(c) Class‑agnosticism of phase (the zero‑shot‑specific term).** Unit phasors `e_{il} = ζ_{il}/|ζ_{il}|_ε ∈ ℂ`. Per‑class and global circular moments of order `k ∈ {1,2}`:
```
m^{(k)}_{c,l} = (1/n_s) Σ_{i∈c} e_{il}^k        m^{(k)}_{l} = (1/|B|) Σ_{i∈B} e_{il}^k
ν_k = (1 − |m^{(k)}_l|²)/n_s                    (null level for n_s samples)
L_prot = (1/(2L·n_c)) Σ_{c,l,k} [ |m^{(k)}_{c,l} − m^{(k)}_{l}|² − ν_k ]₊
```
Penalize only class‑dependent phase structure **above** the sampling‑noise floor `ν_k`; the hinge makes the term exactly zero under the null hypothesis "phase ⟂ class".

**Total:** `L = L_base + λ_inv·L_inv + λ_prot·L_prot`, `λ_inv = 1.0`, `λ_prot = 0.5`.

### 1.5 Gradient paths (all analytic; no straight‑through, no stop‑grad, no EMA)

```
∂|ζ_l|_ε/∂z_{2l−1} = z_{2l−1}/|ζ_l|_ε      ∂|ζ_l|_ε/∂z_{2l} = z_{2l}/|ζ_l|_ε
∂Δ(z_i,z_j)/∂ζ_{il} = 2|ζ_{jl}|·ê_{il} − 2ζ_{jl}       (as real 2‑vectors, ê = ζ/|ζ|_ε)
∂L_inv/∂Δ_ij = −1{Δ_ij < Δ*}
∂L_prot/∂e_{il}: standard complex-moment chain rule; ∂e/∂ζ = (I − êê^T)/|ζ|_ε
```
Gradients from `L_base` reach `z` only through `π`; gradients from `L_inv`, `L_prot` reach `z` directly. All three reach `W`, `b`, `φ` through the ℓ₂‑normalization Jacobian `(I − zzᵀ)/‖h‖`. Proxies receive gradient from `L_base` only.

### 1.6 Schedule and recipe (identical for every arm of every comparison)

- AdamW, base lr `1e−4`, proxy lr `1e−2` (×100), weight decay `1e−4`, batch 120.
- 5 warm‑up epochs with backbone frozen; BN frozen throughout (`bn-freeze=1`), per the Proxy‑Anchor repository.
- **200 epochs** (lane budget). Proxy‑Anchor's published step schedule is `γ = 0.5` with `lr-decay-step = 5` (CUB) / `10` (Cars) at its own shorter budget; I rescale the step by the epoch‑budget ratio to `17` (CUB) / `33` (Cars) and apply the identical schedule to baseline and method. This is my choice, not the source's — see §7.
- Images: RandomResizedCrop 224 + RandomHorizontalFlip (train); Resize 256 + CenterCrop 224 (test).
- `λ_inv` and `λ_prot` ramp linearly from 0 over epochs 10→30 (post‑warm‑up), then constant. Rationale: before the embedding has any class structure, "within‑class variation" is undifferentiated noise and routing it is meaningless.
- `L = 32` ⇒ `m = 544`, deployed dim = `(m − 2L) + L = 512`. Exactly matched.

### 1.7 Hyperparameter table (frozen)

`L = 32`, `Δ* = 0.15`, `λ_inv = 1.0`, `λ_prot = 0.5`, `ε = 1e−6`, `n_c×n_s = 30×4`, ramp `[10,30]`, moment orders `{1,2}`. Ablation grid (§5): `L ∈ {0,8,16,32,64,128}`, `Δ* ∈ {0.05,0.10,0.15,0.30}`, `λ_prot ∈ {0,0.25,0.5,1.0}`.

**Scale is operational, not cosmetic.** `Δ*` and `ν_k` are absolute quantities on a unit sphere (`Δ ∈ [0,4]`); they are *not* rendered harmless by the ℓ₂ normalization, because the energy split `Σ_l|ζ_l|²` vs `‖r‖²` is a free direction of the optimization that AdamW's decoupled weight decay and Proxy‑Anchor's temperature `α` both bias. Changing `α`, the weight decay, or the base loss requires re‑tuning `Δ*` on the ablation grid; inheriting `Δ* = 0.15` across such a change is not permitted.

---

## 2. The causal zero‑shot error mode, and a proof‑level attack on the cheap degeneracies

### 2.1 Error mode: **circular nuisance cannot be linearly quotiented without destroying the identity signal that shares its plane**

For an unseen identity, the images trace a continuous nuisance manifold whose dominant component in these benchmarks is *viewpoint azimuth* (Cars196: make‑model‑year photographed around the car; CUB: head/body orientation; In‑Shop: model pose). Azimuth is topologically **S¹**, not ℝ.

Rank‑1 failure is precisely the event
```
‖y_a − y_b‖ > ‖y_a − y_c‖ ,   class(a)=class(b)≠class(c),  pose(a)≈pose(c)
```
i.e. within‑class spread along the nuisance circle exceeds the between‑class margin.

**Why any linear head must fail on this.** Let a 2‑plane of the embedding carry `(a,b) = ρ·(cos ψ, sin ψ)` with `ψ` = nuisance angle and `ρ` = an identity‑carrying magnitude (e.g. "how elongated is the silhouette", whose *apparent* value rotates with viewpoint but whose amplitude does not). Consider any linear map `Λ` applied to that plane:
- rank‑1 `Λ` (kill one direction): output `ρ cos(ψ − ψ₀)` — still fully modulated by `ψ`. Nuisance **not** removed.
- rank‑0 `Λ` (kill both): output `0` — nuisance removed, but `ρ` destroyed too. **2 dimensions and the identity signal lost.**
- any invertible `Λ`: `ψ`‑dependence preserved.

So no linear projection — and therefore **no PCA/LDA/WCCN‑style within‑class‑covariance normalization, and no ρ‑spectral or coding‑rate variance regularizer** — can remove a nuisance circle while retaining its radial identity content. The phase collapse `(a,b) ↦ ρ` does exactly that, at a cost of **one** dimension, and it is (by F1) the *coarsest* map that does so. This is the whole mechanism, and it is a theorem, not an intuition. It also yields the sharp prediction that PQML's gain should be largest where nuisance is most nearly circular (Cars196 ≳ CUB) and should *not* be reproducible by the linear control C2 (§5).

### 2.2 Degeneracies and why each is blocked

**D1 — unused channel.** The encoder sets the phase planes to zero or constant phase. Then `Δ ≡ 0`, `π` is effectively an isometry, and PQML reduces to the baseline (harmless but useless). *Status: a genuine critical point.* Because `Δ = Σ_l 2ρρ'(1 − cos Δψ) ≈ Σ_l ρρ'·Δψ²`, the contraction is **second order** in phase difference, so `∂L_inv/∂Δψ = 0` exactly at `Δψ = 0`. **Attack:** (i) at random init `W` is dense, so same‑class pairs have `Δψ` uniformly distributed and `Δ > 0` — training starts strictly off the saddle, and `∂L_inv/∂Δψ < 0` there, so the dynamics amplify `Δψ` until the hinge is met; (ii) an explicit trigger: monitor `Δ̄_same`; if `Δ̄_same < 0.02` at epoch 40, add the dispersion floor `L_disp = Σ_l [v* − Var_c(ψ_{·l})]₊`, `v* = 0.05`, and re‑run. I state (ii) as a contingency repair, not a component, so that the shipped method has exactly three terms.

**D2 — global or within‑class collapse.** Neural collapse (`z_i → μ_c`) trivially makes post‑quotient within‑class distance zero. **Attack:** `L_inv` uses the *absolute* contraction `Δ`, not a ratio. Collapse gives `Δ = 0`, which *maximally violates* the hinge `[Δ* − Δ]₊ = Δ*`. Collapse is therefore the single worst point of `L_inv`. (This is why the ratio form `Δ/‖z_i−z_j‖²` — the obvious first draft — is wrong: it is maximized by collapse. The absolute form is load‑bearing.) Structurally, the quotient's fibers are at most `L`‑dimensional tori, so `π` can destroy exactly `L` real dimensions of information and no more; with `L = 32 ≪ 511`, no choice of parameters lets the deployed map collapse the sphere.

**D3 — identity routed into the destroyed channel** (the shortcut that would silently damage unseen classes while leaving seen‑class loss intact). Two sub‑cases:
- *Phase = pure identity* (constant within class, varying across classes): then `Δψ_same = 0` ⇒ `Δ_same = 0` ⇒ `L_inv` maximally violated. **Self‑defeating; blocked by `L_inv` alone.**
- *Phase = nuisance + class‑dependent offset*: `L_inv` is satisfied and `L_base` is satisfied (seen classes are still separated by the surviving channels), so **neither term detects it** — yet a class‑discriminative feature is being destroyed, which costs nothing on seen classes and everything on unseen ones. This is exactly the failure that a seen‑class objective is structurally blind to. `L_prot` is the term that catches it: a class‑dependent phase offset displaces the per‑class circular mean `m^{(1)}_{c,l}` from the global mean by more than the sampling floor `ν_1`, and a class‑dependent phase *concentration* displaces `m^{(2)}`. Residual exposure: moments `k ≥ 3` are unconstrained. Contingency repair if probe F5 (§5) fires: replace the moment hinge with a 2‑layer phase‑only adversarial class probe and a gradient‑reversal coefficient of 0.1 (adds ≈0.3 % epoch time).

**D4 — cheating by norm imbalance.** Pushing all energy into the planes (`‖r‖→0`) makes `Δ` large. **Attack:** it also makes the deployed descriptor `y` non‑negative in every coordinate, collapsing all pairwise cosines toward 1 and driving `L_base` up sharply. The hinge form of `L_inv` removes all pressure past `Δ*`, so the equilibrium is interior. Predicted stationary phase‑energy fraction `Σ_l|ζ_l|² ≈ 0.15–0.35`; this is a monitored quantity, and a run that exceeds 0.6 is to be discarded as a `Δ*` mis‑set, not reported.

**D5 — the two auxiliary terms are jointly vacuous.** `L_inv` demands within‑class phase *spread*; `L_prot` demands the phase distribution be *class‑independent*. These are compatible and jointly identifying: their common solution is "phase is a class‑agnostic coordinate along which same‑class images genuinely differ" — i.e. a nuisance factor. They are not the same constraint (a class‑independent constant phase satisfies `L_prot` and violates `L_inv`; a class‑indexed spread satisfies `L_inv` and violates `L_prot`), so neither is redundant. Note also that `L_prot` must **not** be written as "different‑class pairs must not contract": under the correct solution, two images of different classes at different azimuths *do* contract, so that formulation penalizes the intended optimum. The distinction between a distance constraint and an independence constraint is essential here.

---

## 3. Adversarial primary‑source novelty search

I searched inside DML and, adversarially, outside it (harmonic analysis, equivariant deep learning, symmetry discovery, speaker verification, statistical decision theory). Nearest works and the one‑sentence mechanism distinction for each:

**Inside DML**

1. **DVML, ECCV 2018** — states the same premise (intra‑class variance distribution is approximately class‑independent) but models it as an **additive isotropic Gaussian** in a variational latent and uses it to **synthesize hard samples**; PQML models it as a **compact group action**, *enforces* rather than assumes class‑independence, and **quotients it out of the deployed descriptor** while synthesizing nothing.
2. **Non‑isotropy Regularization (NIR), CVPR 2022** — uses normalizing flows to make residuals around proxies **isotropic**, homogenizing the nuisance distribution; PQML deliberately makes nuisance **anisotropic and concentrated** into `L` designated circular coordinates precisely so that it can be exactly destroyed.
3. **ρ‑spectral regularization ("Revisiting Training Strategies", ICML 2020) and Anti‑Collapse Loss via coding rate (2024)** — preserve embedding variance *globally* with no statement about which directions are nuisance; PQML preserves within‑class variance *only inside a certified class‑agnostic channel* and removes it at test time, which is the opposite bookkeeping.
4. **Proxy Synthesis (AAAI 2021), Embedding Expansion (CVPR 2020)** — synthesize classes or interpolants to occupy empty embedding regions; PQML adds no synthetic points and changes no class set.
5. **DREML (ECCV 2018), Divide‑and‑Conquer (CVPR 2019), BIER (ICCV 2017)** — partition *classes* or *data* across ensemble members and concatenate their embeddings; PQML has one head and one descriptor and partitions **nuisance from identity**, not data from data.
6. **Deep Within‑Class Covariance Analysis / WCCN (speaker & audio verification)** — linear within‑class covariance normalization; §2.1 proves a linear map cannot quotient a circle without destroying its radial identity content — this is the decisive control C2, not a relabelling.
7. **Proxy‑Anchor (CVPR 2020) / PFML (CVPR 2025)** — reshape the *loss geometry* over a fixed embedding map; PQML changes the **deployed map** (adds a maximal‑invariant quotient) and is loss‑agnostic, dropping into either unchanged.
8. **S2SD (ICML 2021)** — distils *into* the deployed embedding *from* higher‑dimensional auxiliary embeddings; PQML adds `L` dimensions and then **destroys their phase** — routing information into a sink rather than importing it from a richer space.
9. **Introspective DML (2022)** — augments the descriptor with a semantic‑uncertainty vector; a different object (uncertainty, not invariance) and it changes the retrieval metric.
10. **Episodic / few‑shot metric adaptation (e.g. ACCV 2022 online adaptation)** — adapts the metric at test time using support data; PQML is train‑time only, with a fixed descriptor and no adaptation, no support set, no transduction.

**Outside DML**

11. **Scattering transforms (Mallat) and Phase Collapse in Neural Networks (ICLR 2022)** — nearest mechanism relative anywhere: complex modulus over **fixed wavelet channels** to gain translation invariance and class separation in classification; PQML's planes are **not fixed filters** — the encoder *learns which factors to route into them*, under an explicit class‑agnosticism constraint that exists only because the deployment task is zero‑shot retrieval rather than closed‑set classification.
12. **Augerino, NeurIPS 2020** — learns an **input‑space** affine Lie group by maximizing group size subject to training loss, and deploys by *averaging predictions over the group*; PQML learns nuisance that has **no input‑space parameterization** (3‑D viewpoint of a rigid object, articulated pose) and deploys an **exact invariant descriptor**, not an average.
13. **LieGAN / LaLiGAN — Latent Space Symmetry Discovery, ICML 2024** — discovers latent Lie‑group symmetry by **adversarial distribution matching**, unsupervised, for dynamics and equation discovery; PQML defines the group as the largest one preserving a **given label partition**, uses a fixed closed‑form torus, and has no adversary and no generator.
14. **Equivariance with Learned Canonicalization Functions, ICML 2023** — trains a canonicalization *network* for a **known** group; PQML has no canonicalization network (for a torus the canonicalization is closed‑form: take the modulus) and the group is **not known a priori** — it is the object being identified from labels.
15. **Group Invariant Deep Representations for Image Instance Retrieval (2016)** — group‑integration descriptors for **hand‑specified** transformation groups; PQML learns which factors constitute the group.
16. **Maximal invariants in statistical decision theory (Lehmann)** — the classical principle of reducing a problem by the largest group leaving it invariant. PQML is, to my search, the first use of "*the largest embedding‑isometry group that preserves the label partition*" as a **trainable objective**, with the resulting maximal invariant shipped as the retrieval descriptor.

**Summary of the claimed novelty**, stated so it can be attacked: the novel object is *a deployed descriptor that is the maximal invariant of a nuisance group identified from identity labels alone*, together with the two‑term identification criterion (absolute within‑class contraction hinge + circular‑moment class‑independence hinge) that makes that group estimable without attributes, transformations, generators, or extra views. Each ingredient in isolation has ancestors (modulus nonlinearity: §11; group‑size maximization: §12; label‑free symmetry discovery: §13; within‑class nuisance removal: §6); their composition and the zero‑shot argument in §2.1 are, to my search, unoccupied.

---

## 4. What I could not verify — unresolved source ambiguities

- **PFML (CVPR 2025):** the source I read discloses the potential functions, `M = 15` (CUB, Cars) and `M = 2` (SOP), ResNet‑50/512‑D/224 px, Adam, lr `5e−4`, proxy lr ×100, 200 epochs, and R@1 `0.734/0.927/0.829` over 5 runs. It does **not** disclose batch size, weight decay, sampler, augmentation, warm‑up, LR schedule, the per‑dataset values of the potential exponent `α` and cutoff `δ`, the In‑Shop configuration or its proxy count, or whether `±0.3/±0.3/±0.2` are standard deviations or standard errors. I treated them as SDs (the conservative reading for a crossing claim); if they are SEMs, the crossing bar in §6 rises and my forecast's crossing probability falls.
- **Proxy‑Anchor:** the repository discloses batch 120, lr `1e−4`, `warm 5`, `bn-freeze 1`, `lr-decay-step 5/10`, ResNet‑50/512‑D R@1 `69.9` (CUB) and `87.7` (Cars). It did **not** disclose, in what I could read, the optimizer, weight decay, decay `γ`, epoch count, `α`/`δ`, sampler, or image size; `α = 32`, `δ = 0.1` and 512‑D come from the paper. My `γ = 0.5`, wd `1e−4`, AdamW, 200‑epoch rescaled schedule, and the 30×4 balanced sampler are **my choices**, applied identically to every arm.
- **PA+DADA In‑Shop 0.930, AdvRF 0.766/0.949/0.842, VAPNet In‑Shop 0.939, CRT 0.9448** are taken from the prompt; I did not independently verify their numbers (I saw only an arXiv listing consistent with AdvRF's title). None of them enters my forecast arithmetic, since I forecast CUB and Cars in Lane A only.
- Because PFML's sampler and batch size are undisclosed and my method requires same‑class pairs in the batch, **the headline claim must be measured against my own PFML reproduction under my sampler and schedule**, never against the published 0.734/0.927 directly. I state separately (§6, R0) whether that reproduction lands inside `0.734 ± 0.003` / `0.927 ± 0.003`; if it does not, the frontier claim is void regardless of the delta.

---

## 5. Decisive matched‑compute controls

Every control uses the identical backbone, sampler, batch size, epoch budget, schedule, augmentation, and 512‑D deployed descriptor; 5 seeds each.

| # | Control | What it kills if it wins |
|---|---|---|
| **C1** | Baseline: `m=512`, no quotient, `L_base` on `z` | reference Δ |
| **C2** | **Linear nuisance removal**: `m=544`, deploy 512‑D after projecting out the 32 leading within‑class‑covariance directions (deep‑WCCN), estimated on training classes | **the decisive control.** If C2 ≥ ⅔ of PQML's Δ, the §2.1 circle‑vs‑line theorem is empirically irrelevant and PQML reduces to known linear compensation |
| **C3** | Quotient architecture only: `m=544`, `π` deployed, `λ_inv = λ_prot = 0` | the gain is a free nonlinearity, not the invariance objective (predicted: ≈0, channel unused — this control also directly measures D1) |
| **C4** | `L_inv` applied to **random** pairs instead of same‑class pairs | the same‑class conditioning is inert; the term is a generic variance regularizer |
| **C5** | Baseline + 2‑layer MLP head with parameter count matched to `m=544` | extra head capacity explains it |
| **C6** | Baseline + a matched‑strength generic anti‑collapse regularizer (ρ‑spectral **or** coding‑rate anti‑collapse), tuned on the same grid size | "any within‑class variance preservation" explains it, i.e. the specific circular quotient is unnecessary |
| **C7** | Baseline with the identical 30×4 balanced sampler and 200‑epoch schedule | the sampler/schedule change explains it (this row, not the published number, is the true baseline) |
| **C8** | Dose–response: `L ∈ {0,8,16,32,64,128}`, deployed dim fixed at 512 | a monotone curve out to `L=128` means it is a dimensionality effect, not a certified‑channel effect; an interior optimum is the predicted signature |
| **C9** | `λ_prot = 0` at fixed `λ_inv` | the zero‑shot‑specific term is inert (predicted: R@1 on *unseen* classes drops while training‑class metrics do not — the cleanest possible mechanism signature) |

**Mechanism probes (annotation‑free, no test‑gallery fitting — computed on the test split only for diagnosis, never used for any training or retrieval decision):**
- **P1 — transfer of routing:** mean same‑class contraction `Δ̄` on unseen classes vs. on training classes.
- **P2 — phase carries nuisance:** apply a controlled input perturbation (horizontal flip; ±10 % scale) and measure `‖Δphase‖` vs `‖Δmodulus‖`. Prediction: phase response ≫ modulus response.
- **P3 — phase carries no identity, out of sample:** empirical mutual information `I(ψ_l ; class)` on **unseen** classes, per plane.

---

## 6. Frozen forecasts, falsification thresholds, frontier arithmetic

**Lane A.** ResNet‑50, 512‑D deployed, 224 px, single view, cosine, 200 epochs, 5 seeds, mean ± SD. Frozen now, before any run.

| Row | CUB R@1 | Cars196 R@1 |
|---|---|---|
| **R0** in‑house PFML reproduction (my sampler/schedule) | 0.731 ± 0.004 | 0.925 ± 0.004 |
| **R1** in‑house Proxy‑Anchor baseline = C7 | 0.697 ± 0.004 | 0.878 ± 0.004 |
| **R2** PQML on Proxy‑Anchor | **0.716** ± 0.004 (90 % CI [0.703, 0.728]) | **0.897** ± 0.004 (90 % CI [0.884, 0.909]) |
| **R3** PQML on PFML *(headline)* | **0.746** ± 0.004 (90 % CI [0.734, 0.757]) | **0.937** ± 0.004 (90 % CI [0.926, 0.946]) |
| **C2** linear WCCN control on PFML | 0.735 ± 0.004 | 0.929 ± 0.004 |
| Published references | PFML 0.734 ± 0.003 (n=5) | PFML 0.927 ± 0.003 (n=5) |

Point forecast deltas: **+1.9 / +1.9** over Proxy‑Anchor, **+1.5 / +1.2** over in‑house PFML.

**Frontier‑crossing arithmetic (explicit).** Reference PFML CUB `0.734 ± 0.003` SD over `n = 5` ⇒ SEM `0.003/√5 = 0.00134`. PQML at `n = 5` with forecast SD `0.004` ⇒ SEM `0.00179`. SEM of the difference `= √(0.00134² + 0.00179²) = 0.00224`. A two‑sided 95 % decisive crossing needs `Δ ≥ 1.96 × 0.00224 = 0.0044`, i.e. **CUB ≥ 0.7384**; with a margin for reproduction drift I set the declaration bar at **CUB ≥ 0.742**. Same computation for Cars (`0.927 ± 0.003`) gives a bar of **Cars ≥ 0.933**.

- Point forecast R3 clears both bars (0.746 > 0.742; 0.937 > 0.933).
- The **lower 90 % bound** of R3 clears **neither** (0.734; 0.926).
- Honest crossing probabilities under my own forecast distribution: **CUB ≈ 0.55, Cars ≈ 0.50, at least one of the two ≈ 0.65, both ≈ 0.40.**
- The claim is only admissible if R0 lands inside `0.734 ± 0.006`; otherwise the recipe change invalidates inheriting PFML's published frontier and only the R3−R0 delta may be reported.

**Falsification thresholds (pre‑registered; any one firing rejects the stated claim):**

- **F1** `R3 − R0 < +0.005` on CUB over 5 seeds ⇒ mechanism rejected as a frontier method.
- **F2** C2 (linear WCCN) achieves `≥ ⅔` of PQML's Δ ⇒ the circle‑vs‑line theorem is empirically void; PQML collapses to known linear compensation. *This is the single most important falsifier.*
- **F3** C3 (no auxiliary terms) achieves `≥ ½` of Δ ⇒ the gain is architectural, and the invariance objective is decoration.
- **F4** P1: `Δ̄_unseen < 0.3 × Δ̄_train` ⇒ routing does not transfer; the mechanism story is falsified **even if R@1 improves**, and any improvement must be re‑attributed.
- **F5** P3: `I(ψ_l ; class) > 0.1` bits/plane on unseen classes ⇒ `L_prot` failed out of sample (trigger the adversarial‑probe repair, and report both).
- **F6** C8: R@1 monotone increasing through `L = 128` ⇒ a dimensionality effect, not a certified nuisance channel.
- **F7** C6 (generic anti‑collapse regularizer) matches within 0.003 ⇒ any variance preservation suffices; the specific quotient is unnecessary.
- **F8** phase energy `Σ_l|ζ_l|²` exceeds 0.6 at convergence ⇒ run discarded as mis‑set `Δ*`, not reported as a result.

---

## 7. Cost, and benchmark / contamination risk

**Training cost.** No extra forward or backward pass, no second view, no teacher, no generator, no auxiliary network. Extra parameters: `W` is `2048×544` instead of `2048×512` ⇒ **+65 536 params, +0.26 %** of ResNet‑50's 25.6 M. Extra per‑batch compute: `L_inv`/`L_prot` need only the already‑computed `z` and `y` plus pairwise terms, `O(B²·m) ≈ 120²·544 ≈ 7.8 MFLOP` per step, against ResNet‑50's `≈ 120 × 4.1 GFLOP = 492 GFLOP` forward alone — a `~1.6×10⁻⁵` fraction. Measured overhead expectation: **≤1.02× epoch time, ≤1.01× memory**. For comparison the prompt's In‑Shop reference PA+DADA costs ≈1.06× epoch time and 1.01× memory, and Lane‑B AdvRF adds a full ResNet‑34/U‑Net reconstruction system.

**Deployment cost.** `+32` head output dims and `L = 32` square roots: `≈0.13 MFLOP` against 4.1 GFLOP, i.e. **+0.003 %**. One model, one view, 512‑D, cosine — byte‑for‑byte the same index and the same query latency as the baseline.

**Risks I am not hiding.**
1. *Non‑negative coordinates.* `L` of the 512 deployed coordinates are `≥ 0`, which adds a pair‑dependent positive floor to every cosine and slightly compresses dynamic range. At `L = 32` and phase energy ≈0.25 this should cost little, but it is a real term. Repair if F8‑adjacent symptoms appear: center the moduli, `y_l = |ζ_l| − c_l` with learned `c_l`, and renormalize — this forfeits the exact norm‑preservation of §1.2 and must be re‑derived, so it is an ablation, not the shipped method.
2. *Circularity assumption.* If the dominant unseen‑class nuisance is not well approximated by an `S¹` action in any 2‑plane, `L_inv` will still be satisfiable (any within‑class spread routed into the planes contracts) but the gain will be small and F4 may still pass — a scenario where the method is mildly useful for a reason weaker than the one I argue. I would report that as a partial confirmation, not a confirmation.
3. *`L_prot` is a moment constraint.* Only circular moments 1 and 2 are controlled; a third‑or‑higher‑moment identity leak into phase would be invisible to it and would silently destroy unseen‑class information. F5 is the detector; the gradient‑reversal probe is the repair.
4. *Statistical power.* CUB and Cars R@1 have run‑to‑run SD ≈0.003–0.005; the forecast delta is ~2–4 SDs. Five seeds is the minimum for the crossing bar in §6, and I would not report a crossing at `n < 5` per arm.
5. *Contamination.* No test data, no external or generated data, no text/VLM encoder, no extra annotation, no transduction, no re‑ranking, no test‑gallery statistics. The only external knowledge is the permitted ImageNet‑1K initialization. The known, pre‑existing overlap between ImageNet‑1K categories and CUB/Cars content is a benchmark‑level contamination shared identically by every reference row in this lane (PFML, PA, DADA, AdvRF), so it does not differentially favour PQML — but it does mean absolute Lane‑A numbers on CUB should not be read as clean zero‑shot generalization in the strict sense.
6. *Benchmark risk.* CUB/Cars are near their measurement ceiling for a 512‑D ResNet‑50; a +1.2–1.5 point delta is within the range where undisclosed recipe differences between papers (sampler, epochs, augmentation) can dominate. That is exactly why the headline claim in §6 is anchored to R3−R0, an in‑house matched pair, and why R0 must itself land inside PFML's published interval before any frontier language is used.

---

**Sources:**
- [PFML — Potential Field Based Deep Metric Learning, CVPR 2025 (CVF)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [arXiv HTML](https://arxiv.org/html/2405.18560)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [official repository](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Deep Variational Metric Learning, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html)
- [Non-isotropy Regularization for Proxy-based Deep Metric Learning, CVPR 2022](https://arxiv.org/pdf/2203.08547)
- [Anti-Collapse Loss for Deep Metric Learning Based on Coding Rate Metric](https://arxiv.org/pdf/2407.03106)
- [Proxy Synthesis: Learning with Synthetic Classes for Deep Metric Learning, AAAI 2021](https://arxiv.org/pdf/2103.15454)
- [Deep Randomized Ensembles for Metric Learning, ECCV 2018](https://www2.seas.gwu.edu/~pless/papers/DREML_ECCV2018.pdf)
- [Deep Within-Class Covariance Analysis for Robust Audio Representation Learning](https://arxiv.org/pdf/1711.04022)
- [Phase Collapse in Neural Networks, ICLR 2022](https://openreview.net/pdf?id=iPHLcmtietq) · [arXiv](https://arxiv.org/abs/2110.05283)
- [Learning Invariances in Neural Networks from Training Data (Augerino), NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/cc8090c4d2791cdd9cd2cb3c24296190-Paper.pdf)
- [Latent Space Symmetry Discovery (LaLiGAN), ICML 2024](https://arxiv.org/pdf/2310.00105)
- [Equivariance with Learned Canonicalization Functions, ICML 2023](https://proceedings.mlr.press/v202/kaba23a/kaba23a.pdf)
- [Group Invariant Deep Representations for Image Instance Retrieval](https://arxiv.org/pdf/1601.02093)
- [Characterizing Generalization under OOD Shifts in Deep Metric Learning](https://arxiv.org/pdf/2107.09562)
- [Adversarial Reconstruction Feedback for Robust Fine-grained Generalization (AdvRF)](https://arxiv.org/pdf/2507.21742)
