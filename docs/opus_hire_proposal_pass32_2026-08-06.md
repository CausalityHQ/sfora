I have what I need. Chosen lane: **Lane A** (ResNet-50, 512-D, ~224px, single-view cosine, 200 epochs). One method follows.

---

# HIRE — Homoscedastic Isotropic Residual Embedding

**One-line mechanism.** Deployed retrieval is 1-NN under an *isotropic* metric, but the identity-conditional embedding law is anisotropic and heteroscedastic; test-time whitening would fix this and is banned, so HIRE trains the encoder so that every identity-conditional second moment equals a *common isotropic target*, certified by an unbiased cross-class U-statistic that needs no per-class covariance estimate. No new parameters, no extra forward pass, no test-time change.

---

## 1. Executable mathematics

### 1.1 Network, deployment, learned objects

- Backbone $\phi_\theta$: ResNet-50, ImageNet-1K init. Pooling $= \mathrm{GAP}(\cdot)+\mathrm{GMP}(\cdot)\in\mathbb R^{2048}$ (Proxy-Anchor official-code convention; see §6 ambiguity A2).
- Head: $u = Wh+b$, $W\in\mathbb R^{512\times2048}$, $b\in\mathbb R^{512}$. Descriptor $z = u/\lVert u\rVert_2\in S^{511}$, $d=512$.
- Proxies $\{p_c\}_{c=1}^{C}\subset S^{511}$, one per class, Kaiming-normal init.
- **Learned objects: $\theta, W, b, \{p_c\}$ — exactly the base method's. HIRE adds none.**
- **Deployment:** resize 256 → center crop 224, one view, one 512-D $L_2$-normalised descriptor, cosine NN. Byte-identical inference to the base.

### 1.2 Base loss (Proxy Anchor, Kim et al. CVPR 2020 — reproduced exactly)

$s(z,p)=z^\top p$; $\mathcal P$ = all proxies, $\mathcal P^+$ = proxies of classes in the batch, $Z_p^\pm$ = batch embeddings of/not-of class $p$:

$$
\mathcal L_{\text{PA}}=\frac{1}{|\mathcal P^+|}\sum_{p\in\mathcal P^+}\log\!\Big(1+\!\!\sum_{z\in Z_p^+}\!e^{-\alpha(s(z,p)-\delta)}\Big)+\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}\log\!\Big(1+\!\!\sum_{z\in Z_p^-}\!e^{+\alpha(s(z,p)+\delta)}\Big)
$$

with $\alpha=32,\ \delta=0.1$. Recipe (primary source, with A2 flagged): AdamW; lr $10^{-4}$ (CUB/Cars), $6\!\times\!10^{-4}$ (SOP/In-Shop); **proxy lr $=100\times$ backbone lr**; weight decay $10^{-4}$; warm-up 1 epoch (CUB/Cars) / 5 (SOP/In-Shop) with backbone frozen; RandomResizedCrop(224)+RandomHorizontalFlip. Published PA decays lr $\times0.5$/10 epochs over **60** epochs; **the lane budget is 200 epochs, so I do not inherit PA's published numbers — I re-run PA at 200 epochs with cosine decay as the in-house matched baseline** (§4, C6).

Sampler: balanced $P\times K$, $P=30$, $K=5$, batch $N=150$ — **identical for baseline and HIRE**.

### 1.3 The single constrained object

For identity $c$ let $\Sigma_c=\operatorname{Cov}(z\mid y=c)$. HIRE minimises the deviation of every $\Sigma_c$ from one common isotropic target $\Sigma^\star=\tau^2 I/d$. Write $s_c=\operatorname{tr}\Sigma_c$, $\hat C_c=\Sigma_c/s_c$. Then **exactly**:

$$
\big\lVert \Sigma_c-\tfrac{\tau^2}{d}I\big\rVert_F^2 \;=\; \underbrace{s_c^2\big\lVert \hat C_c-\tfrac{I}{d}\big\rVert_F^2}_{\text{shape}} \;+\; \underbrace{\tfrac{1}{d}\,(s_c-\tau^2)^2}_{\text{scale}},\qquad
\mathbb E_c(s_c-\tau^2)^2=\underbrace{(\bar s-\tau^2)^2}_{\text{level}}+\underbrace{\operatorname{Var}_c(s_c)}_{\text{spread}} .
$$

Three orthogonal coordinates of **one** quantity — shape / scale-spread / scale-level. They get three weights only because their minibatch estimators have wildly different variance at $K=5$.

### 1.4 Estimators (the technical core)

Per class $c$ in the batch form all $\binom{K}{2}=10$ same-class pairs $\mathcal A_c$. For pair $a=(i,j)$:

$$
d_a=z_i-z_j,\qquad \rho_a=\lVert d_a\rVert_2,\qquad \delta_a=\frac{d_a}{\mathrm{sg}\!\left(\max(\rho_a,\rho_{\min})\right)},\quad \rho_{\min}=\tfrac12\sqrt{2\tau^2}.
$$

$\mathbb E[d_ad_a^\top]=2\Sigma_c$ — pairwise differences give a **mean-free, unbiased** handle on $\Sigma_c$ with no class-mean estimate. `sg` = stop-gradient on the norm only, so shape and scale gradients are orthogonal by construction and the $1/\rho$ factor in $\partial\delta/\partial d=(I-\delta\delta^\top)/\rho$ is bounded.

**(T1) Shape — cross-class U-statistic.** Let $C=\mathbb E[\delta\delta^\top]$, $\operatorname{tr}C=1$. Key identity:
$$\big\lVert C-\tfrac Id\big\rVert_F^2=\lVert C\rVert_F^2-\tfrac1d,\qquad \mathbb E_{\delta_a\perp\!\!\!\perp\delta_b}\big[\langle\delta_a,\delta_b\rangle^2\big]=\lVert C\rVert_F^2 .$$
Pairs sharing an index are dependent, so the statistic uses **only cross-class pairs**, which are conditionally independent given $\theta$:
$$
U=\frac{2}{P(P-1)}\sum_{c<c'}\frac{1}{|\mathcal A_c||\mathcal A_{c'}|}\sum_{a\in\mathcal A_c}\sum_{b\in\mathcal A_{c'}}\langle\delta_a,\delta_b\rangle^2,
\qquad \mathbb E[U]=\lVert \bar C\rVert_F^2,\ \ \bar C=\mathbb E_c[\hat C_c].
$$
So $U-1/d$ is an **unbiased estimator of $\lVert\bar C-I/d\rVert_F^2\ge0$**, exactly zero iff the pooled conditional shape is isotropic. Use the log form (well-scaled gradients, direct interpretation):
$$
\boxed{\ \mathcal L_{\text{shape}}=\log\big(d\cdot U+10^{-8}\big)=\log d-\log r_{\text{eff}},\qquad r_{\text{eff}}:=1/\lVert\bar C\rVert_F^2\ }
$$
$r_{\text{eff}}$ is the participation-ratio effective rank of the conditional residual covariance; $\mathcal L_{\text{shape}}=0\iff r_{\text{eff}}=d$.

**(T2) Scale spread.** $R_c=\frac{1}{|\mathcal A_c|}\sum_{a\in\mathcal A_c}\rho_a^2$ (unbiased for $2s_c$).
$$\boxed{\ \mathcal L_{\text{homo}}=\operatorname{Var}_c\big(\log R_c\big)\ }$$

**(T3) Scale level — one-sided per-class floor.**
$$\boxed{\ \mathcal L_{\text{scale}}=\frac1P\sum_c\big[\max\big(0,\ \log R^\star-\log R_c\big)\big]^2,\qquad R^\star=2\tau^2\ }$$
One-sided: HIRE never *inflates* scatter, it only forbids over-collapse. The log parameterisation is a deliberate, stated deviation from the exact quadratic decomposition in §1.3 — it stabilises variance at $K=5$ **and** turns each term into a per-class barrier at $R_c\to0$ (§2).

### 1.5 Total objective and schedule

$$
\mathcal L=\mathcal L_{\text{PA}}+\lambda(t)\Big[\lambda_{\text{sh}}\mathcal L_{\text{shape}}+\lambda_{\text{ho}}\mathcal L_{\text{homo}}+\lambda_{\text{sc}}\mathcal L_{\text{scale}}\Big]
$$
$$
\lambda(t)=\min\!\Big(1,\ \max\big(0,\tfrac{t-T_0}{T_1-T_0}\big)\Big),\quad T_0=5,\ T_1=25\ \text{epochs}.
$$

**Frozen hyperparameters:** $\lambda_{\text{sh}}=0.3$, $\lambda_{\text{ho}}=1.0$, $\lambda_{\text{sc}}=1.0$; $\tau^2=0.10$ (CUB/Cars), $0.06$ (SOP/In-Shop). Selected on a **held-out-identity meta-val split** (80/20 identity split *of the training classes*), 60-epoch runs, grid $\lambda_{\text{sh}}\!\in\!\{0.1,0.3,1.0\}\times\lambda_{\text{ho}}\!\in\!\{0.3,1\}\times\lambda_{\text{sc}}\!\in\!\{0.3,1\}\times\tau^2\!\in\!\{0.06,0.10,0.16\}$ (36 configs), then retrain on all training identities at 200 epochs. **No test data touches tuning.** The baseline receives an equal 36-config budget (§4, C4).

Gradients flow through $\delta_a,\rho_a\to z_i\to W,b,\theta$; no stop-gradient except the norm in $\delta_a$; proxies receive gradient only from $\mathcal L_{\text{PA}}$.

### 1.6 Two structural properties

- **Orthogonal invariance.** All three terms are invariant under $z\mapsto Qz$, $Q\in O(d)$ — the exact invariance group of cosine retrieval. VICReg/Barlow-Twins-style coordinate-wise decorrelation is *not*; it imposes an arbitrary basis the deployed metric cannot see.
- **Cost.** $O(M^2 d)$ with $M=P\binom K2=300$: $\approx4.6\times10^{7}$ FLOPs vs. $\approx1.9\times10^{12}$ for the ResNet-50 fwd+bwd on 150 images — **$2.5\times10^{-5}$ overhead**. Extra memory: a $300\times512$ matrix and a $300\times300$ Gram ($\approx1$ MB).

```python
# per step, after z = F.normalize(head(backbone(x)))
Z  = z.view(P, K, d)
D  = (Z.unsqueeze(2) - Z.unsqueeze(1))[:, iu, ju]          # (P, K(K-1)/2, d)
rho = D.norm(dim=-1)                                        # (P, M_c)
R_c = (rho**2).mean(1)                                      # (P,)
dlt = D / rho.clamp_min(rho_min).unsqueeze(-1).detach()
G   = torch.einsum('pad,qbd->pqab', dlt, dlt)**2            # cross-class only
mask = ~torch.eye(P, dtype=torch.bool, device=z.device)
U   = G.mean((2,3))[mask].mean()
L   = L_PA + lam(t)*(0.3*torch.log(d*U + 1e-8)
                   + 1.0*torch.log(R_c).var()
                   + 1.0*(R_star.log() - R_c.log()).clamp_min(0).pow(2).mean())
```

---

## 2. Causal zero-shot error mode + proof-level degeneracy attack

### 2.1 The error mode: conditional anisotropy the deployed metric cannot correct

Take two **unseen** identities $a,b$ with conditional laws $(\mu_a,\Sigma)$, $(\mu_b,\Sigma)$, $w=\mu_a-\mu_b$. For a query $z\sim(\mu_a,\Sigma)$:

| rule | statistic | $\mathbb E$ | $\operatorname{Var}$ | $d'$ |
|---|---|---|---|---|
| deployed (Euclidean/cosine) | $2z^\top w+\lVert\mu_b\rVert^2-\lVert\mu_a\rVert^2$ | $\lVert w\rVert^2$ | $4w^\top\Sigma w$ | $\dfrac{\lVert w\rVert^2}{2\sqrt{w^\top\Sigma w}}$ |
| Bayes/Fisher (Mahalanobis) | $2z^\top\Sigma^{-1}w+\dots$ | $w^\top\Sigma^{-1}w$ | $4w^\top\Sigma^{-1}w$ | $\dfrac{\sqrt{w^\top\Sigma^{-1}w}}{2}$ |

$$
\frac{d'_{\text{euc}}}{d'_{\text{opt}}}=\frac{\lVert w\rVert^2}{\sqrt{w^\top\Sigma w}\sqrt{w^\top\Sigma^{-1}w}}\;\overset{\text{Kantorovich}}{\ge}\;\frac{2\sqrt\kappa}{\kappa+1},\qquad \kappa=\frac{\lambda_{\max}(\Sigma)}{\lambda_{\min}(\Sigma)},
$$
with equality iff $\kappa=1$. **The efficiency of the deployed rule is governed entirely by the condition number of the identity-conditional covariance, and is lossless only under isotropy.** The bound is tight; for a $\lambda_k\propto k^{-1}$ spectrum restricted to the top-100 directions the *average*-case efficiency is $1/\sqrt{(\bar\lambda)(\overline{1/\lambda})}\approx0.62$ — a ~38% loss of discriminability index.

Heteroscedasticity ($\Sigma_a\neq\Sigma_b$) is worse: it breaks the linear-rule family entirely and makes the per-query false-match rate identity-dependent — tight identities become hubs, diffuse identities become anti-hubs (Radovanović's $N_k$ skew). Under item-level 1-NN (the actual protocol) the impostor score $z^\top g'$ has variance $\mu_a^\top\Sigma_b\mu_a+\dots$, so R@1 depends on the **tail** of the impostor score — anisotropy and scale spread hurt *more*, not less, than the class-mean analysis above.

**Why it is a train-time problem.** The two legal fixes at test time — whitening by the gallery's within-identity scatter, or CSLS/local-scaling — are exactly test-gallery fitting and reranking, both banned. So the conditioning must be built into $f$ at train time, and it must be built in **identity-independently**, or it will not transfer to identities never seen. Standard DML does the opposite: proxy/margin losses *minimise* $\operatorname{tr}\Sigma_c$ (neural collapse NC1), driving $\kappa\to\infty$ on the surviving directions and leaving the scale spread completely unconstrained.

### 2.2 Degeneracies, attacked

**D1 — global collapse.** $R_c\to0\Rightarrow \mathcal L_{\text{scale}}=(\log R^\star-\log R_c)^2\to\infty$ with $\partial\mathcal L_{\text{scale}}/\partial\log R_c=-2(\log R^\star-\log R_c)\to-\infty$. Collapse is not merely non-stationary, it is repelled with divergent force. ∎

**D2 — per-identity collapse with global compensation.** $\mathcal L_{\text{homo}}=\operatorname{Var}_c(\log R_c)\to\infty$ as any single $R_c\to0$. The log form gives a *per-class* barrier for free. ∎

**D3 — "isotropic garbage": satisfy T1/T3 by routing nuisance (crop offset, JPEG noise) into a structureless residual.** This is the real danger. Partial argument: the residual budget is finite and pinned ($R_c=R^\star$ at equilibrium, since $\mathcal L_{\text{PA}}$ pushes down and the floor pushes up), so garbage energy is energy denied to identity-orthogonal structure; and any residual component with a nonzero projection onto an impostor direction is penalised at first order by $\mathcal L_{\text{PA}}$'s negative term. This is an argument, **not** a proof — so I make it *decidable*: control **C1** (§4) injects matched-energy isotropic Gaussian noise, reproducing $r_{\text{eff}}$ and $R^\star$ with *guaranteed* garbage. If C1 matches HIRE, D3 has occurred and the method is falsified (threshold F2).

**D4 — trivial invariances.** All terms are $O(d)$-invariant and invariant to global rescaling of $u$ (killed by $L_2$ normalisation), so no head bias/gain can fake satisfaction.

**D5 — estimator degeneracy at $K=5$ (stated limitation).** Within one class, $\{\delta_a\}$ spans $\le K-1=4$ dimensions: **per-class shape is unidentifiable**. This is precisely why T1 is a *cross-class* U-statistic estimating the **pooled** $\bar C$. HIRE therefore certifies (i) pooled conditional isotropy and (ii) per-identity *scale* homogeneity; it does **not** certify per-identity *shape* homogeneity. That is exactly the testable part of the LDA homoscedasticity hypothesis, and I state it as a limitation rather than claim it.

---

## 3. Adversarial novelty search — one-sentence mechanism distinctions

**Inside DML**

| Nearest work | Distinction |
|---|---|
| **PFML**, Bhatnagar & Ahuja, CVPR 2025 — superposed decaying attractive/repulsive potentials $\psi\propto\delta^\alpha/\lVert r-z\rVert^\alpha$, $\alpha\!\in\![3,6]$, over samples **and** proxies | PFML redesigns the *first-order force law* between points; HIRE leaves the force law untouched and constrains the *second moment of identity-conditional residuals* — the two compose. |
| **NIR** (Non-isotropy Regularization), Roth, Vinyals & Akata, CVPR 2022 — normalizing flows enforce unique translatability from proxy | NIR fights *distributional* isotropy (residual direction carrying no identity of the sample) with a learned density; HIRE enforces *second-moment* isotropy, which at fixed energy is the maximum-entropy configuration and therefore **increases** the residual informativeness NIR wants — the names collide, the mechanisms do not conflict. |
| **SoftTriple / multi-proxy (incl. PFML's 15 proxies/class)** | Multi-proxy adds *attractors* and thereby edits the conditional **mean** structure; HIRE adds no attractor and edits the **covariance** about existing ones. |
| **PA+DADA**, AAAI 2024 — data-augmented domain adaptation between sample and proxy distributions | DADA closes a sample-vs-proxy *discrepancy* gap; HIRE conditions the identity-conditional second moment for Bayes-optimality of the deployed isotropic rule. |
| **IDML (TPAMI 2023) / PFE / DUL / HIB** | All learn and **deploy** a per-image uncertainty with a non-cosine similarity; HIRE has no uncertainty head, deploys plain cosine, and uses second-moment structure purely as a train-time constraint. |
| **DiVA / S2SD / MIC / "Sharing Matters"** | Those add branches, targets, or auxiliary tasks; HIRE adds no branch, no target, and no forward pass — a closed-form certificate on residuals already in the batch. |
| **DAS (ECCV 2022)** | DAS *synthesises* embeddings around anchors; HIRE constrains the covariance of the real ones. |
| **Center loss / DeepLDA / Fisher losses** | Those *minimise* or ratio-optimise within-class scatter; HIRE **pins its trace to a floor and whitens its shape** — the opposite of scatter minimisation, and it is the pinning that makes the deployed metric optimal. |
| **"Hubs and Hyperspheres", Trosten et al., CVPR 2023** | A coordinate-wise **marginal** moment condition used **transductively** at test time in few-shot; HIRE is a basis-free **conditional** condition applied only at train time. |

**Outside DML**

| Nearest work | Distinction |
|---|---|
| **LeJEPA / SIGReg**, 2025 — sketched Epps–Pulley test forcing the **marginal** embedding law to $\mathcal N(0,I)$, unsupervised | SIGReg constrains the marginal, which is satisfiable with *every* identity-conditional a point mass and with wildly heteroscedastic identities (half collapsed, half diffuse) — precisely the hubness configuration; HIRE constrains what the marginal condition cannot see. |
| **VICReg / Barlow Twins / W-MSE / Decorrelated BN** | Basis-dependent, label-free, marginal (or view-pair) statistics; HIRE is orthogonally invariant, identity-conditional, and its target is a *common* covariance across identities. |
| **Uniformity/alignment (Wang & Isola, 2020)** | Marginal uniformity on the sphere — same blind spot as above. |
| **Neural collapse (Papyan–Han–Donoho)** | Descriptive of NC1 within-class variability collapse; HIRE is prescriptive — it pins the collapsing quantity to a non-degenerate isotropic target and argues that doing so is *required* for deployed-Euclidean Bayes-optimality on unseen identities. |
| **Box's M test / Flury's Common Principal Components** | Classical *tests* for covariance homogeneity requiring per-group covariance estimates (infeasible at $K=5$); HIRE imports the homogeneity hypothesis as a differentiable constraint with a U-statistic that never forms a per-group covariance. |
| **Learned/PCA whitening in retrieval (Jégou–Chum 2012; Radenović et al. TPAMI 2018)** | A *fixed linear map* estimated post-hoc from training statistics; no single linear map can equalise per-identity scales when $\Sigma_c$ varies with $c$ — HIRE makes the nonlinear encoder itself produce conditionally-white, scale-homogeneous residuals. (Also control C2.) |
| **CFAR radar detection** | CFAR normalises the *statistic* by locally estimated clutter power at detection time; HIRE equalises the *clutter power itself* at train time so no detection-time normalisation is needed. |
| **Kantorovich inequality / Fisher's LDA** | Classical; the new move is using Kantorovich to *quantify* the deployed metric's efficiency loss and then **training the data to fit the fixed metric** rather than fitting the metric to the data — the only legal move when test-time whitening is forbidden. |

---

## 4. Decisive matched-compute controls

All at 200 epochs, 5 seeds, identical sampler/augmentation/schedule, identical tuning budget.

- **C1 — isotropic-garbage control (kills D3).** PA + $z\leftarrow\mathrm{normalize}(u+\tau g/\sqrt d)$, $g\sim\mathcal N(0,I)$, energy matched to $R^\star$. Reproduces $r_{\text{eff}}$ and residual energy with *guaranteed* structureless content.
- **C2 — linear-whitening control.** PA, then a linear whitening/LDA projection estimated from the **training** set's within-identity scatter, folded into $W$. Legal, free. *Prediction:* recovers part of the shape benefit, **none** of the homoscedasticity benefit.
- **C3 — term ablation.** shape-only / spread-only / floor-only / all. If floor-only (≈ "just don't over-collapse", i.e. a softer margin) matches the full method, the isotropy claim is dead.
- **C4 — cheap-regulariser control.** PA with label smoothing, $\delta$, $\alpha$, weight decay, and mixup each tuned over an equal 36-config meta-val budget.
- **C5 — marginal-vs-conditional control (decisive vs. the nearest outside work).** PA + SIGReg-style marginal isotropy, and PA + uniformity, at matched compute.
- **C6 — matched-cost baseline.** PA at 200 epochs, identical everything. HIRE's overhead is $2.5\times10^{-5}$ FLOPs, so epochs are already matched; report wall-clock anyway.
- **C7 — mechanism diagnostic on *unseen* identities (report-only, never used for tuning).** Measure $\kappa$, $r_{\text{eff}}$, and $\operatorname{Var}_c(\log s_c)$ on the **test** identities for PA vs PA+HIRE. The causal story *requires* these to improve on unseen identities, not merely on training identities.
- **C8 — the sharpest prediction.** Sweep proxies-per-class $\in\{1,4,15\}$. Since multi-proxy already partially preserves conditional structure, **HIRE's gain must decrease monotonically with proxy count.** If HIRE's gain is independent of proxy count, my mechanistic account is wrong even if R@1 improves.
- **C9 — random-init control.** HIRE from scratch. If the relative gain vanishes, the mechanism is really "pretrained-feature preservation", a different claim.

---

## 5. Frozen forecasts — Lane A only

**Given references (from the brief, treated as audited):** PFML CUB $0.734\pm0.003$, Cars $0.927\pm0.003$, SOP $0.829\pm0.002$ (5 runs). DADA matched-cost controls: CUB 0.729, Cars 0.921, SOP 0.810.

**My in-house matched baselines (forecast, to be measured — not inherited):** PA @200 ep, R50/512-D: CUB $0.705\pm0.004$, Cars $0.885\pm0.004$, SOP $0.808\pm0.003$.

| Dataset | PA base (mine) | **PA + HIRE** (80% int.) | matched PFML repro (required) | **PFML + HIRE** (80% int.) | PFML reference |
|---|---|---|---|---|---|
| CUB-200-2011 | 0.705 | **0.731** [0.716, 0.744] | 0.734 ± 0.005 | **0.748** [0.736, 0.759] | 0.734 ± 0.003 |
| Cars196 | 0.885 | **0.906** [0.893, 0.917] | 0.927 ± 0.005 | **0.934** [0.925, 0.942] | 0.927 ± 0.003 |
| SOP | 0.808 | **0.822** [0.814, 0.829] | 0.829 ± 0.004 | **0.836** [0.830, 0.842] | 0.829 ± 0.002 |

**Frontier-crossing arithmetic (CUB).** $\Delta=0.748-0.734=+0.014$. My SE over 5 seeds at sd 0.005 is 0.00224. If PFML's $\pm0.003$ is a standard deviation, its SEM is 0.00134 and the combined SE is $\sqrt{0.00224^2+0.00134^2}=0.00261\Rightarrow 5.4\sigma$. If $\pm0.003$ is already a SEM, combined SE $=0.00374\Rightarrow 3.7\sigma$. Crossing survives either reading. Cars: $+0.007$ → $2.0$–$2.7\sigma$ (**weak — I do not claim Cars**). SOP: $+0.007$ → $2.4$–$3.1\sigma$.

**Mechanism-level arithmetic.** Predicted PA baseline on unseen CUB identities: $r_{\text{eff}}\!\approx\!20$–$60$ of 512, $\kappa\!\gtrsim\!50$ ⇒ Kantorovich worst-case efficiency $2\sqrt{50}/51=0.28$, average-case $\approx0.6$. HIRE target: $r_{\text{eff}}\ge300$, $\kappa\le8$ ⇒ worst-case $0.63$, average-case $\approx0.9$. The forecast $+0.014$ assumes only a modest fraction of that efficiency recovery converts to R@1, because between-identity separation $\lVert w\rVert$ is partly traded away to fund the residual floor. **This trade is the central empirical bet of the method and I state it as unresolved.**

**Calibration anchor for the magnitude.** BLenDeR (arXiv Jan 2026) reports CUB 0.770 on a PF base by *synthesising* intra-class variation with diffusion + text embeddings — **illegal here** (external generated data, text encoder), cited only as evidence that intra-class variation modelling carries roughly $+3.7$ of headroom on CUB under a far more aggressive intervention. HIRE conditions rather than synthesises, so a point forecast of $+1.4$ is deliberately a fraction of that.

**Falsification thresholds (pre-registered).**
- **F1** — on the PA base, if mean $\Delta$R@1 $<+0.010$ (CUB) **and** $<+0.008$ (Cars) over 5 seeds: falsified as a mechanism.
- **F2** — if C1 (matched isotropic noise) lands within 0.003 of HIRE on both CUB and Cars: the conditional-whitening claim is dead; D3 occurred.
- **F3** — if C5 (marginal isotropy) lands within 0.003 of HIRE: the conditional-vs-marginal distinction is dead and the method is a rename of SIGReg-family work.
- **F4** — if C7 shows $<20\%$ relative reduction in $\kappa$ or $\operatorname{Var}_c(\log s_c)$ on **unseen** identities: the transfer premise is falsified *regardless of R@1*, and any gain must be re-attributed.
- **F5** — if C8 shows HIRE's gain not decreasing with proxies-per-class: mechanistic account falsified.
- **F6 (frontier gate)** — no frontier claim unless a matched in-house PFML reproduction lands within $\pm0.005$ of 0.734 **and** PFML+HIRE exceeds it by $\ge+0.008$ on CUB over 5 seeds.

---

## 6. Cost, risks, ambiguities

**Training cost.** $\approx1.00\times$ epoch time ($2.5\times10^{-5}$ extra FLOPs), $\approx1.00\times$ memory ($\approx1$ MB), **zero** extra parameters, zero extra forward passes, no teacher, no generator. Cheaper than PA+DADA ($1.06\times$ time, $1.01\times$ memory per the brief) and far cheaper than AdvRF's ResNet-34/U-Net reconstruction + distillation or VAPNet's attribute machinery (both Lane B, not used for any comparison here).

**Deployment cost.** Identical to the base: one ResNet-50, one view, one 512-D descriptor, cosine NN. No uncertainty head, no reranking, no gallery statistics.

**Benchmark & contamination risks.**
- ImageNet-1K init contains birds and cars; "zero-shot" here is w.r.t. identity labels, not visual novelty. C9 separates HIRE from mere pretrained-feature preservation.
- SOP/In-Shop contain near-duplicate images within listings, which artificially shrink and heterogenise $\Sigma_c$. HIRE's homoscedasticity term may be *most* effective there and *most* confounded there — report per-class $R_c$ histograms alongside R@1.
- CUB's test gallery is 5,924 images with known label noise; differences $<0.005$ are not meaningful at 5 seeds.
- Canonical class splits only; all tuning on held-out **training** identities; test touched once, after freezing.

**Unresolved source ambiguities (stated, not hidden).**
- **A1 — PFML.** I retrieved the potential form ($\psi_{\text{att}}/\psi_{\text{rep}}$ with cutoff $\delta$, decay $\alpha\!\in\![3,6]$, $U=\sum_i\Psi_{y_i}(z_i)+\sum_{j,k}\Psi_j(p_{j,k})$) but **not** the full recipe: per-dataset proxies/class, $\alpha$, $\delta$, optimizer, lr, epochs, batch, pooling. Whether the reported $\pm0.003$ is sd or SEM is unstated in what I retrieved. **All PFML-base forecasts are conditional on obtaining the official code and passing F6.**
- **A2 — Proxy Anchor.** The official README documents batch size, lr, embedding dim, warm-up and decay steps but **not** optimizer, weight decay, proxy-lr multiplier, total epochs, input resolution, augmentation, $\alpha$, $\delta$, or pooling. I stated those from the paper/community-standard code (AdamW, wd $10^{-4}$, proxy lr $\times100$, $\alpha{=}32$, $\delta{=}0.1$, 224 RRC+flip, GAP+GMP sum) and they require code-level verification before any matched claim. PA's published recipe is **60** epochs, not the lane's 200 — hence the in-house reproduction (C6) rather than inheritance.
- **A3 — PA+DADA In-Shop 0.930**: seeds and uncertainty unreported (per the brief). I make no In-Shop forecast.
- **A4 — NIR naming collision.** NIR's "non-isotropy" concerns the residual *distribution*'s identifiability; mine concerns its *second moment*. I believe these are compatible and even mutually reinforcing, but if NIR's primary source turns out to penalise second-moment whiteness specifically, HIRE's shape term becomes a sign-flip of prior art and the novelty claim collapses to the homoscedasticity term alone. **This is the single largest novelty risk and it is checkable in one reading of the NIR paper.**

**Honest weaknesses.** (i) The Kantorovich bound is worst-case over $w$; realised gains will be far below it. (ii) Forcing a residual floor spends between-identity separation — the trade is unproven and is the method's central bet. (iii) HIRE certifies *pooled* shape and *per-identity scale*, not per-identity shape (D5) — unidentifiable at $K=5$. (iv) D3 ("isotropic garbage") is argued, not proved; C1/F2 decide it. (v) The Cars forecast does not cross the frontier at usable significance and I do not claim it.

---

Sources: [PFML (CVPR 2025)](https://arxiv.org/abs/2405.18560) · [PFML CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf) · [Proxy Anchor (CVPR 2020) code](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020) · [DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29400) · [NIR (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf) · [IDML](https://arxiv.org/abs/2205.04449) · [AdvRF (ICCV 2025)](https://arxiv.org/abs/2507.21742) · [SIGReg / LeJEPA](https://www.emergentmind.com/topics/sketched-isotropic-gaussian-regularization-sigreg) · [Hubs and Hyperspheres (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_CVPR_2023_paper.pdf) · [VICReg](https://arxiv.org/pdf/2105.04906) · [BLenDeR (illegal-lane reference)](https://arxiv.org/html/2601.20246)
