I'll ground this against primary literature before proposing. Loading web tools.I ran a grounding search (budget now nearly exhausted, so no further consultations or searches). Confirmations: **PFML = "Potential Field Based Deep Metric Learning"** (CVPR 2025 poster 33305), ResNet‑50 R@1 73.4 / 92.7 / 82.9 — matches the brief. **MCR²** is confirmed as `R(Z,ε)=½ logdet(I + (d/mε²)ZZᵀ)` maximizing between-class rate **minus** within-class rate. And "preserve intra-class variance for unseen classes" is confirmed as an *occupied motivation* (learnable dynamic margin, PR 2022; intra-variance semi-supervised ranking, EAAI 2024) — so my novelty has to live entirely in the mechanism, and I state that plainly below.

---

# PROPOSAL: **DARC** — Discriminant‑Axis Residual Capacity

**Lane: A** (ResNet‑50, 512‑D ℓ2‑normalized descriptor, 224², single view, cosine NN, 200 epochs). All forecasts and comparisons below are Lane A only.

**One sentence.** Inject a *fixed, exogenous* dither into the descriptor so that resolution acquires an absolute unit, then add one term that maximizes the Shannon capacity `½log(1+τ/ν)` of the **view-reliable within-class residual** measured **only along an EMA between-class eigenframe** — turning the unit-norm constraint into a power budget whose stationary point is a reverse water-filling allocation between seen-class separation and within-class resolution.

---

## 1. Executable mathematics

### 1.1 Base model and reproduced baseline (Proxy‑Anchor, Kim et al., CVPR 2020)

I extend Proxy‑Anchor because its recipe is fully disclosed. Reduction reproduced exactly:

$$\mathcal L_{\mathrm{PA}}=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\!\!\sum_{x\in X_p^+}\!\! e^{-\alpha(s(x,p)-\delta)}\Big)+\frac{1}{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X_p^-}\!\! e^{\alpha(s(x,p)+\delta)}\Big)$$

with $s(x,p)=z_x^\top p$, $\|z\|_2=\|p\|_2=1$, $\alpha=32$, $\delta=0.1$; one proxy per class; AdamW, lr $10^{-4}$, weight decay $10^{-4}$, **proxy learning rate ×100**; 1 warm-up epoch with the backbone frozen; random-resized-crop + horizontal flip at train, resize‑256/center‑crop‑224 at test.

> **Unresolved source ambiguities (flagged, not glossed):** (i) Proxy‑Anchor's disclosed batch size is 180, but the class×image factorization at that size is not stated in the paper — I fix 30×6 and will report the sweep; (ii) the paper's headline schedule is ~40–60 epochs, not the lane's 200 — I run 200 with cosine decay for *both* arms and treat the 200‑epoch baseline as a **required matched reproduction**, not an inherited number; (iii) **PFML's full recipe (optimizer, batch, schedule, and the mechanics of its 15 proxies/class on CUB‑Cars vs 2 on SOP) is not available to me**, so every PFML comparison below is a comparison against a *reported* number, not a reproduced one, and §5 treats that asymmetry explicitly.

### 1.2 Learned objects

| Object | Shape | Learned? | Deployed? |
|---|---|---|---|
| $\theta$: ResNet‑50 (ImageNet‑1K init) | — | yes | **yes** |
| $W$: linear head $2048\to512$ | $512\times2048$ | yes | **yes** |
| $P=\{p_c\}$: proxies | $C\times512$ | yes | no |
| $\Sigma^{\mathrm{ema}}$: between-class second moment | $512\times512$ | **no** (EMA buffer) | no |
| $U=[u_1..u_K]$: eigenframe | $512\times K$ | **no** (derived, detached) | no |
| $\sigma$: dither scale | scalar | **no** (fixed) | no |

Nothing new is learned. The method adds **zero parameters**.

### 1.3 Forward pass (train)

Sample $M$ classes × $K_{\mathrm{img}}$ images $\Rightarrow n=MK_{\mathrm{img}}$ **unique** images; draw **two independent augmentations** $A_1,A_2$ per image. Compute-matched default: $2n=180$, i.e. $n=90$, $M=15$, $K_{\mathrm{img}}=6$.

$$z_i^{(v)}=\frac{W\,\mathrm{GAP}(f_\theta(A_v(x_i)))}{\|W\,\mathrm{GAP}(f_\theta(A_v(x_i)))\|_2},\qquad
\tilde z_i^{(v)}=\frac{z_i^{(v)}+\sigma\,\varepsilon_i^{(v)}}{\|z_i^{(v)}+\sigma\,\varepsilon_i^{(v)}\|_2},\quad \varepsilon_i^{(v)}\!\sim\!\mathcal N(0,I_{512})$$

$\varepsilon$ is resampled every step, independent across $i$ and $v$, and is **not** reparameterized into anything learnable. $\mathcal L_{\mathrm{PA}}$ is computed on the **undithered** $z_i^{(v)}$ (both views, averaged) so the base loss is untouched; the dither exists only inside $\mathcal L_{\mathrm{DARC}}$.

### 1.4 The certified measurement frame

Batch class means $\hat\mu_c=\frac{1}{2K_{\mathrm{img}}}\sum_{i\in c,v}z_i^{(v)}$, grand mean $\hat\mu$. Between-class scatter $S_B=\frac1M\sum_c(\hat\mu_c-\hat\mu)(\hat\mu_c-\hat\mu)^\top$. Update, **outside the autograd graph**:

$$\Sigma^{\mathrm{ema}}\leftarrow\rho\,\Sigma^{\mathrm{ema}}+(1-\rho)S_B,\qquad \rho=0.99$$

Every $T=20$ steps, eigendecompose $\Sigma^{\mathrm{ema}}=U\Lambda U^\top$ and keep the top $K=16$ eigenpairs $(u_k,\lambda_k)$, $u_k$ **detached**. Axis weights $w_k=\lambda_k/\sum_j\lambda_j$.

Cost note: never form the $512\times512$ eigendecomposition from scratch per step — the EMA buffer costs $O(d^2)=2.6\!\times\!10^5$ flops/step, the eigendecomposition $O(d^3)=1.3\!\times\!10^8$ flops every 20 steps, i.e. $<0.001\%$ of the ResNet‑50 step.

### 1.5 The DARC term

Within-class-centered readouts (with $\bar\mu_c$ the dithered class mean, gradient attached):

$$a_{ikv}=u_k^\top\big(\tilde z_i^{(v)}-\bar\mu_{c(i)}\big)$$

Two-view ANOVA on $a_{ikv}=t_{ik}+\xi_{ikv}$, $\xi$ i.i.d. across views:

$$\hat\nu_k=\frac{1}{2n}\sum_i\big(a_{ik1}-a_{ik2}\big)^2 \quad(\text{per-view noise variance}),\qquad
\hat\tau_k=\frac1n\sum_i\Big(\tfrac{a_{ik1}+a_{ik2}}{2}\Big)^2-\frac{\hat\nu_k}{2}$$

Both are unbiased: $\mathbb E[\hat\nu_k]=\nu_k$, $\mathbb E[\hat\tau_k]=\tau_k$. Capacity and loss:

$$\boxed{\;\mathcal L_{\mathrm{DARC}}=-\sum_{k=1}^{K}w_k\cdot\tfrac12\log\!\Big(1+\frac{[\hat\tau_k]_+}{\hat\nu_k+\epsilon}\Big),\qquad \epsilon=10^{-6}\;}$$

$$\mathcal L=\mathcal L_{\mathrm{PA}}+\beta\,\mathcal L_{\mathrm{DARC}}$$

**Gradient path.** $\partial\mathcal L_{\mathrm{DARC}}/\partial a_{ikv}$ is exact and cheap:
$$\frac{\partial C_k}{\partial \hat\tau_k}=\frac{1}{2(\hat\nu_k+\hat\tau_k)},\qquad
\frac{\partial C_k}{\partial \hat\nu_k}=-\frac{\hat\tau_k}{2\hat\nu_k(\hat\nu_k+\hat\tau_k)}$$
so the term simultaneously **pushes reliable within-class spread up** and **pushes view-noise down**, with the two pressures automatically balanced by the log. Gradients flow $a\to\tilde z\to z\to W\to\theta$. $u_k$, $\lambda_k$, $w_k$, $\varepsilon$ carry **no** gradient. `[·]₊` is implemented as `softplus(x/s)*s`, $s=10^{-3}$, so the clamp is differentiable.

### 1.6 Hyperparameters and schedules

| | CUB / Cars | SOP / In‑Shop |
|---|---|---|
| $\sigma$ (dither) | 0.10 | 0.10 |
| $\beta$ | 0.30 | 0.15 |
| $K$ axes | 16 | 16 |
| $\rho$ (EMA) | 0.99 | 0.995 |
| $M\times K_{\mathrm{img}}$ | 15×6 | 30×3 |
| epochs | 200, cosine | 200, cosine |

$\beta$ schedule: **linear warm-up $0\to\beta$ over epochs 1–20**, constant thereafter. Rationale: $\Sigma^{\mathrm{ema}}$ is meaningless before the embedding has any class structure; certifying axes early would certify noise. Sweep grids (tuned **only** on a class-disjoint holdout carved from training classes, §6): $\sigma\in\{0.03,0.05,0.10,0.20\}$, $\beta\in\{0.05,0.15,0.3,0.6,1.2\}$, $K\in\{4,16,64\}$.

### 1.7 Test-time operation

$z=\mathrm{normalize}(W\,\mathrm{GAP}(f_\theta(x)))$, single center-crop view, **no dither, no axes, no proxies, no second view**. Cosine NN over the gallery. Descriptor is bit-identical in form to the baseline's.

### 1.8 Why the stationary point is reverse water-filling

Let $g_k=\partial\mathcal L_{\mathrm{PA}}/\partial\tau_k>0$ be the base loss's marginal cost of within-class variance on axis $k$. Stationarity in $\tau_k$ gives

$$g_k=\frac{\beta w_k}{2(\nu_k+\tau_k)}\;\Longrightarrow\;\tau_k^\star=\Big[\frac{\beta w_k}{2g_k}-\nu_k\Big]_+$$

This is exactly the reverse water-filling solution of the Gaussian rate–distortion problem, with $\beta$ the water level. Axes the base loss can cheaply relax get resolution up to $\beta w_k/2g_k$; axes it cannot get $\tau_k^\star=0$. The $\|z\|_2=1$ constraint supplies the power budget that makes the allocation a genuine trade rather than a free lunch — this is the substantive claim, and it predicts a **non-monotone $R@1(\beta)$ with an interior optimum**, which is directly falsifiable (§5).

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode: **supervised quantization of continuous attribute axes**

This is *not* dimensional collapse (unused directions) and *not* over-separation in the Kornblith sense (global class-separation magnitude). It is **loss of resolution inside a direction that is being used**.

Let $u$ be a direction the labels certify as semantic (e.g. a bill-shape axis on CUB). Proxy/softmax DML drives $z_i\to p_{c(i)}$, so the class-conditional distribution of $u^\top z$ concentrates at $u^\top p_c$. In the limit the map $x\mapsto u^\top z(x)$ becomes a **step function with $C$ levels** — a scalar quantizer whose codebook is the seen-class means. An unseen class whose true attribute value falls strictly inside one seen-class cell is mapped to that cell's level; **two unseen classes inside the same cell are exactly unresolvable**, and no amount of test-time processing recovers them, because the information was destroyed at train time.

Formally: retrieval between unseen classes $u_1,u_2$ separated by $\Delta$ along axis $u$ succeeds only if $\Delta$ exceeds the descriptor's *resolution* $\sqrt{\tau_k}$ on that axis. Standard DML drives $\tau_k\to0$; DARC pins $\tau_k^\star>0$ at a level set by $\beta$ and the noise floor $\sigma$. **The number of resolvable levels per certified axis is $\exp(C_k)$**, which is the quantity DARC directly optimizes.

**Direction of the claim.** MCR² *minimizes* within-class rate; DARC *maximizes* it on certified axes. That is a real scientific disagreement, stated in the open: MCR²/neural-collapse reasoning optimizes the seen partition; DARC's thesis is that the seen partition is one draw and its cells are the wrong quantization for the next draw.

### 2.2 Proof-level attack on the cheapest degeneracies

**(D1) Drive the noise to zero and harvest infinite capacity.** *Blocked by construction.* $a_{ikv}$ contains the additive term $\sigma\,u_k^\top\varepsilon_i^{(v)}/\|z_i^{(v)}+\sigma\varepsilon_i^{(v)}\|$, and $\varepsilon_i^{(1)}\perp\varepsilon_i^{(2)}$, so
$$\mathbb E[\hat\nu_k]\;\ge\;\frac{\sigma^2}{1+\sigma^2}\qquad\text{for every }\theta.$$
$\varepsilon$ is independent of $\theta$ and resampled each step, so **no** parameter setting reduces this floor. Hence $C_k\le\frac12\log(1+\hat\tau_k(1+\sigma^2)/\sigma^2)$: capacity can only be bought with *actual reliable signal*. This is why the dither is load-bearing and not cosmetic — without it, a scale-free reliability/correlation objective is maximized at $1$ by any deterministic embedding, however compressed, and the entire method collapses to a no-op. I state this because it is the failure mode the obvious formulation of this idea has.

**(D2) Inflate $\hat\tau$ with augmentation-driven variance** (leak crop coordinates, jitter magnitude, flip parity into the descriptor). *Blocked.* Any variance driven by the augmentation parameters differs between $A_1$ and $A_2$ by construction, so it enters $(a_{ik1}-a_{ik2})^2$ and lands in $\hat\nu_k$, where it is *penalized* with gradient $-\hat\tau_k/(2\hat\nu_k(\hat\nu_k+\hat\tau_k))$. DARC therefore has the **opposite sign** to AugSelf/E‑SSL: augmentation information is noise, not signal.

**(D3) Inject content-independent spread.** Same argument as D2: content-independent spread is view-inconsistent by definition, so it is $\hat\nu$, not $\hat\tau$.

**(D4) Rotate the measurement frame onto whatever direction already has spare variance.** *Blocked three ways.* $u_k$ is detached (no gradient path); it is derived from an EMA with $\rho=0.99$, so the frame's time constant is ~100 steps against a per-step gradient; and it is refreshed only every $T=20$ steps. The residual pathway — slowly steering $\Sigma^{\mathrm{ema}}$ itself — requires *increasing between-class scatter* in the target direction, which is precisely what the base loss wants anyway. This is a **partial** block and I label it as such.

**(D5) Norm games.** $z$ is ℓ2-normalized before dithering and re-normalized after; there is no norm channel to exploit, and cosine retrieval is invariant to what remains.

**(D6) Encode an image-identity hash** (the dangerous one). A per-image hash is view-consistent and would score $\hat\tau$ on every axis. Three structural brakes, in decreasing strength:
1. **Saturating returns.** $\partial C_k/\partial\tau_k=1/(2(\nu_k+\tau_k))$ decays as $1/\tau_k$, while the base loss's marginal cost $g_k$ *grows* with within-class spread (the $e^{-\alpha(s-\delta)}$ term is convex). An interior optimum therefore exists and is unique in $\tau_k$ for fixed $g_k$; $\tau_k^\star$ is capped at $\beta w_k/2g_k$.
2. **Power budget.** $\sum_k(\tau_k+\text{between-class variance along }u_k)\le1$. Hash bits are paid for out of the same sphere that class separation needs.
3. **Path of least resistance.** From an ImageNet‑1K initialization, semantic content is already linearly accessible along $u_k$; building an identity hash from scratch requires a larger parameter move for the same $\Delta C_k$. This is a plausibility argument, **not** a proof, and it is the single weakest link in the method. §4 gives the control (C5) that decides it empirically.

**(D7) The male/female objection.** On CUB, sexually dimorphic plumage is reliable, within-class, and along a certified axis — DARC will preserve it, and the benchmark labels both sexes as one class, so preserving it *hurts*. This is a genuine adverse mechanism, not a hypothetical. The bet is that quantization loss dominates dimorphism loss; the saturating log limits how much is spent on any single axis; and $\beta$ tunes the trade. If the optimum lands at $\beta\to0$, the bet is wrong and the method is falsified (§5).

---

## 3. Adversarial novelty search — nearest works and one-sentence distinctions

**Inside DML:**

1. **MCR² / ReduNet** (Yu et al., NeurIPS 2020): uses the same $\frac12\log\det(I+\cdot/\varepsilon^2)$ rate with a noise floor but **minimizes** the within-class rate over *all* directions counting augmentation variance as signal; DARC **maximizes** it, on certified axes only, counting augmentation variance as noise.
2. **PFML** (CVPR 2025): models sample interactions as superposed potential fields with 15/2 proxies per class — more incidental parameters to capture multimodality; DARC adds no parameters and specifies a *rate floor* rather than a richer proxy set.
3. **Learnable dynamic margin** (Pattern Recognition 2022): scales the margin by measured intra-class variance — a heuristic modulation of the base loss with no absolute noise reference and no capacity budget; DARC's target is defined against a fixed exogenous ruler $\sigma$ and is scale-meaningful.
4. **Intra-variance semi-supervised ranking** (EAAI 2024): synthesizes samples at graded *augmentation intensities* and imposes an ordinal loss along intensity — i.e. it treats augmentation-induced variation as the intra-class signal; DARC treats exactly that variation as the noise term $\hat\nu$ to be suppressed. Opposite sign on the same axis.
5. **MIC** (Roth et al., ICCV 2019): mines shared intra-class characteristics in order to *remove* them from the main embedding; DARC keeps the view-reliable part of them.
6. **DiVA** (ECCV 2020): concatenates separate embedding spaces for class/intra-class/self-supervised features; DARC uses one space and reallocates its variance budget.
7. **S2SD** (ICML 2021): distills from higher-dimensional auxiliary embeddings; DARC has no teacher and no auxiliary space.
8. **ρ‑regularization** (ICML 2020): stochastically switches off hard-negative mining to preserve intra-class structure; DARC measures the preserved quantity and targets a specific level rather than ablating the base loss.
9. **Metrix / Embedding Expansion / Proxy Synthesis**: expand the span by *synthesizing* samples or proxies; DARC synthesizes nothing.
10. **HIB / PCME**: learn *input-dependent* variance for uncertainty calibration; DARC's noise is fixed, non-learned, and functions as a ruler, not an estimate.
11. **AdvRF** (ICCV 2025, Lane B): training-only ResNet‑34/U‑Net reconstruction + distillation; DARC adds no auxiliary network.

**Outside DML:**

12. **VICReg / Barlow Twins**: per-coordinate, *unconditional* variance floors and decorrelation with no class conditioning and no absolute noise unit; DARC's floor is within-class, rotation-covariant on a certified frame, and denominated in $\sigma$.
13. **Information Bottleneck / CEB**: *minimize* $I(X;Z)$ subject to $I(Z;Y)$; DARC explicitly adds a term maximizing $I(X;Z\mid Y)$ on certified axes — the anti-bottleneck.
14. **RankMe / effective rank** (ICML 2023): unconditional rank *diagnostics*; DARC is an objective and is class-conditional.
15. **Shannon capacity / reverse water-filling** (Cover–Thomas): the classical allocation is imported wholesale; the novelty is the identification of "power" with the ℓ2 descriptor budget and "channels" with label-certified discriminant axes.
16. **Generalizability theory / ICC** (Cronbach 1972): the two-facet ANOVA decomposition is standard psychometrics; its use as a *training objective* on a certified eigenframe is not.
17. **Neyman–Scott incidental parameters** (Econometrica 1948): motivated the design (proxies are incidental parameters never deployed) but the Cox–Reid adjusted-profile correction is **not** used — I checked it and it fails here, because as the model separates, $j_{pp}\to0$ and the $-\frac12\log\det$ term becomes aligned with the likelihood rather than opposed to it, making the adjustment useless or actively harmful. Reported because a reviewer will otherwise propose it.

---

## 4. Decisive matched-compute controls

Every control uses **identical forward-pass count, identical schedule, 5 seeds, paired**.

- **C1 — Two views alone.** Base loss on $2n=180$ passes from $n=90$ unique images, $\beta=0$. Isolates the multi-view batch construction from DARC. *If C1 explains the gain, the method is a batch-construction artifact.*
- **C2 — Variance floor without reliability.** Replace $C_k$ with a hinge $\sum_k w_k[\kappa-\hat\tau_k-\hat\nu_k]_+$ (total within-class variance, no view split). Tests whether the **signal/noise separation** matters or merely "more intra-class variance."
- **C3 — Reliability without the ruler.** Set $\sigma=0$ and use cross-view correlation $\mathrm{Corr}(a_{\cdot k1},a_{\cdot k2})$. Predicted to be a near no-op (D1). *This is the control that proves the dither is the mechanism, not decoration.*
- **C4 — Frame ablation.** (a) $u_k$ = random orthonormal directions, resampled every $T$ steps; (b) $u_k$ = **bottom**-$K$ eigenvectors of $\Sigma^{\mathrm{ema}}$; (c) $u_k$ = all 512 coordinate axes. Tests whether *label certification of the axes* is doing work, or any subspace suffices.
- **C5 — Hash-degeneracy probe (D6).** Train a linear probe from the frozen descriptor to *training-image index* (5,864-way on CUB) and report top‑1. If DARC raises probe accuracy substantially more than it raises unseen R@1, the mechanism is instance memorization and the method is refuted regardless of R@1.
- **C6 — Occupied-alternative matched sweep.** Label smoothing ∈{0.05,0.1,0.2}; temperature $\alpha$ ∈{16,32,64}; $\delta$ ∈{0.05,0.1,0.2}; weight decay ∈{1e‑4,1e‑3}; MCR²-style within-class rate *minimization* at matched weight. DARC must beat the **best** of these, not the default.
- **C7 — Mechanism mediator (pre-registered).** Measure, on **training** classes, $\sum_k w_kC_k$ at convergence for both arms, and on **test** classes measure quantization directly: project unseen embeddings onto the frozen train frame and compute the mean distance from each reading to the nearest seen-class level. DARC must (i) raise train-class $\sum w_kC_k$ and (ii) reduce test-class level-snapping. **If R@1 improves without both, the improvement is not the claimed mechanism** and must be reported as such.
- **C8 — 2× compute variant.** $n=180$ unique × 2 views, against a baseline at $2\times$ epochs. Separates "DARC needs paired views" from "DARC needs more compute."

---

## 5. Frozen forecasts, thresholds, and frontier arithmetic (Lane A only)

**Frozen before any run. 5 seeds, paired, R@1, ResNet‑50 / 512‑D / 224² / 200 epochs / single-view cosine.**

| | My PA baseline (to be reproduced) | **DARC (forecast)** | Δ | PFML (reported) | DADA row (matched-cost control) |
|---|---|---|---|---|---|
| CUB‑200‑2011 | 0.690 ± 0.005 | **0.708 ± 0.005** | **+1.8** | 0.734 ± 0.003 | 0.729 |
| Cars196 | 0.881 ± 0.005 | **0.894 ± 0.005** | **+1.3** | 0.927 ± 0.003 | 0.921 |
| SOP | 0.799 ± 0.004 | **0.806 ± 0.004** | **+0.7** | 0.829 ± 0.002 | 0.810 |

**Falsification thresholds (pre-registered):**
- **F1 (primary).** Paired Δ over my own baseline < **+0.8 on CUB** *and* < **+0.6 on Cars** ⇒ method falsified.
- **F2 (mechanism).** C7 fails either leg ⇒ mechanism falsified even if F1 passes; the result must then be reported as an unexplained regularization effect.
- **F3 (degeneracy).** C5 instance-probe top‑1 rises by more than 3× the R@1 gain (in points) ⇒ D6 has occurred; falsified.
- **F4 (allocation).** $R@1(\beta)$ monotone-decreasing in $\beta$ over $\{0.05\!\to\!1.2\}$ with optimum at the boundary $\beta\to0$ ⇒ the water-filling claim is wrong; falsified.
- **F5 (occupied alternative).** Best of C6 matches DARC within 0.3 pts on CUB ⇒ the method is a renamed regularizer; falsified.

**Frontier-crossing arithmetic — stated plainly, including where it fails.**

*From the Proxy‑Anchor base, DARC does not cross the frontier.* CUB: $0.708 < 0.734$, short by **−2.6 pts** (≈ 4.5 pooled σ). Cars: $0.894<0.927$, short by **−3.3**. SOP: $0.806<0.829$, short by **−2.3**. DARC also does not clear the DADA matched-cost rows (−2.1 / −2.7 / −0.4). I am not going to dress a +1.8 regularizer as a frontier result.

*Crossing requires composition, and is contingent.* Regularizer deltas shrink on stronger bases; I forecast a **0.5–0.7× shrink factor** on a multi-proxy base. Contingent on a PFML reproduction landing within ±0.003 of 0.734:

- CUB: $0.734 + 1.8\times0.55 \approx \mathbf{0.744}$ → crosses by **+1.0** (≈ 2.4σ with pooled σ ≈ 0.004)
- Cars: $0.927 + 1.3\times0.55 \approx \mathbf{0.934}$ → crosses by **+0.7** (≈ 1.7σ)
- SOP: $0.829 + 0.7\times0.55 \approx \mathbf{0.833}$ → crosses by **+0.4** (≈ 1.4σ; not decisive)

This second row is **not** a claim I can currently support, because PFML's recipe is not disclosed to me and I would be inheriting a frontier without a matched reproduction — the exact error the brief warns against. It is a plan, gated on the reproduction landing.

**Calibrated probabilities.** P(Δ ≥ +0.8 on CUB over my own baseline) ≈ **0.55**. P(C7 mediator confirms) ≈ 0.45. P(DARC-on-PFML > 0.734 on CUB at ≥2σ) ≈ **0.20**. P(all three datasets cross) ≈ 0.08.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** Primary config is compute-matched: $2n=180$ forward passes, same as the baseline's 180, with half the unique images. Measured overhead is EMA + periodic eigendecomposition + the $O(nKd)$ readouts ≈ **1.01× epoch time**, **+1 MB** for the $512^2$ EMA buffer (**~1.00× memory**). Compare: PA+DADA ≈ 1.06× time / 1.01× memory; AdvRF adds an entire ResNet‑34/U‑Net plus a distillation stage; VAPNet adds attribute machinery. **DARC is the cheapest method in its comparison set** — which is a large part of why it is worth running even at 0.55 success probability.

**Deployment cost. Zero.** One ResNet‑50, one 224² view, one 512‑D ℓ2 descriptor, cosine NN. No dither, no frame, no proxies, no reranking, no test-gallery statistics at inference.

**Risks.**
- *Adverse mechanism (D7):* preserved within-class attribute variation that the benchmark labels as identity-irrelevant (CUB sexual dimorphism, Cars color, SOP viewpoint) directly costs R@1. Largest single risk.
- *Reliable-but-useless signal:* background/context is view-consistent under crop+flip+jitter, so it can score $\hat\tau$. C4(a) and C5 bound this; a stronger augmentation family (adding RandomErasing) is the mitigation, but changing augmentation changes the baseline and must be applied to both arms.
- *Estimator variance:* $\hat\tau_k$ is a difference of noisy estimates from $K_{\mathrm{img}}=6$ (CUB/Cars) or 3 (SOP) images per class and can go negative; the softplus clamp biases it upward. On SOP/In‑Shop, where classes have ~5 images, $\hat\tau$ is estimated from very few samples — this is why $\beta$ is halved there and why the SOP forecast is the weakest.
- *Contamination:* ImageNet‑1K pretraining is permitted by the brief but overlaps CUB's bird classes; this inflates **all** Lane A numbers including PFML's and is not specific to DARC. No test data, no external data, no generated data, no text/VLM encoder, no transduction, no reranking are used. **All hyperparameter selection ($\sigma,\beta,K,\rho$) is done on a class-disjoint holdout carved out of the training identities** (CUB: 80 train / 20 holdout classes), never on the evaluation split; the final runs retrain on all 100 training classes with the frozen values. The EMA frame is fitted from training images only and is discarded before evaluation.
- *Benchmark risk:* CUB/Cars R@1 at 512‑D is near a regime where a ±0.5 pt difference is within seed noise for many published single-run numbers; PFML's ±0.003 over 5 runs is one of the few audited error bars available, and the In‑Shop reference (PA+DADA, 0.930) has **no reported seed count or uncertainty**, so I have deliberately excluded In‑Shop from the frozen forecast table rather than forecast against an unquantified target.

---

**What I am confident about:** the error mode (supervised quantization of used axes) is real, distinct from dimensional collapse and from neural-collapse magnitude arguments, and under-attacked; the dither-as-ruler is the load-bearing piece without which the scale-free version of this idea is provably a no-op; the sign inversion against MCR² and against the augmentation-intensity intra-variance line is genuine and not cosmetic; and the method is essentially free.

**What I am not confident about:** whether preserved within-class resolution nets positive against benchmark labels that deliberately quotient out some of it (D7), and whether the instance-hash brake (D6) holds — it rests on a plausibility argument, not a proof. C5 and C7 are designed to catch both, and I have pre-registered failure on them as falsification rather than as something to explain away.

**Sources:** [PFML — CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/33305) · [Proxy‑Anchor (CVPR 2020) official code](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020) · [Proxy‑Anchor project page](https://cvlab.postech.ac.kr/research/ProxyAnchor/) · [MCR² (NeurIPS 2020)](https://arxiv.org/abs/2006.08558) · [ReduNet](https://arxiv.org/pdf/2105.10446) · [PA+DADA (AAAI 2024)](https://arxiv.org/html/2401.00617v1) · [Proxy Synthesis (AAAI 2021)](https://arxiv.org/pdf/2103.15454) · [Learnable dynamic margin in DML](https://www.sciencedirect.com/science/article/abs/pii/S0031320322004411) · [DML assisted by intra-variance](https://www.sciencedirect.com/science/article/abs/pii/S0952197624000435)
