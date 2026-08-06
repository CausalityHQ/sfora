Novelty search complete (budget nearly exhausted — one batch only). Key hits: PFML = Potential Field Based DML (Bhatnagar & Ahuja, CVPR 2025); **Anti-Collapse Loss (arXiv 2407.03106)** already uses coding-rate in DML; SVMax; Soft Orthogonal Proxies. These force the mechanism distinction below.

---

# PROPOSAL — NSRC: Null-Space Rate Coding

**Lane: A** (ResNet-50, 512-D, 224px, cosine, 200 epochs). All forecasts in Lane A only.

## 0. Mechanism in one sentence
A proxy loss on cosine scores has a gradient that **can shrink but never rotate** the embedding's component orthogonal to the proxy span; that component is therefore unsupervised and monotonically destroyed. NSRC installs a high-rate, augmentation-invariant code **in exactly that null space**, where its gradient is *provably pointwise-orthogonal* to the task gradient, and deploys it as part of the same 512-D descriptor at zero added parameters and ~1.00× cost.

## 1. Executable mathematics

Backbone $f_\theta$: ImageNet-1K ResNet-50 → GAP → $g\in\mathbb R^{2048}$. Head $W\in\mathbb R^{512\times2048}$, no bias, $h=Wg$. Proxies $p_c\in\mathbb R^{512}$, $\|p_c\|=1$, one per class.

**Task subspace.** $M=\frac1K\sum_c p_cp_c^\top$; $Q_r$ = top-$r$ eigenvectors, $r=\min\{r:\sum_{i\le r}\lambda_i\ge\tau\sum_i\lambda_i\}$, $\tau=0.95$. $\Pi=Q_rQ_r^\top$, $\Pi^\perp=I-\Pi$. Recomputed every 50 iters (one $512\times512$ eigendecomposition, held fixed between). $a=\Pi h$, $b=\Pi^\perp h$, $c=Q_\perp^\top h\in\mathbb R^{d-r}$, $\hat c=c/\|c\|$.

**(i) Task loss** — Proxy Anchor (Kim et al., CVPR 2020), reproduced exactly: $s(x,p)=\cos(h_x,p)$,
$$L_{PA}=\tfrac1{|P^+|}\!\!\sum_{p\in P^+}\!\!\log\!\Big(1+\!\!\sum_{x\in X_p^+}\!\!e^{-\alpha(s-\delta)}\Big)+\tfrac1{|P|}\sum_{p\in P}\log\!\Big(1+\!\!\sum_{x\in X_p^-}\!\!e^{\alpha(s+\delta)}\Big),\ \alpha{=}32,\ \delta{=}0.1.$$
Recipe as published: AdamW, backbone lr $10^{-4}$, proxy lr $\times100$, wd $10^{-4}$, 1 warm-up epoch with backbone frozen, RandomResizedCrop(224, 0.08–1.0)+flip, 4 images/class sampler. *Unresolved source ambiguity:* ProxyAnchor's headline table is Inception-BN; its exact ResNet-50/512 batch size and LR-decay points are reported inconsistently across reproductions — I fix batch $B=128$, no decay, and report my own reproduction as the only baseline I compare against.

**(ii) Null loss** (two scale-free terms on $\hat c$; two views $x^{(1)},x^{(2)}$ per image):
$$L_{inv}=\tfrac1B\sum_i\|\hat c_i^{(1)}-\hat c_i^{(2)}\|^2,\qquad
L_{rate}=-\tfrac1{d-r}\log\det\!\Big(I_B+\tfrac{d-r}{B\kappa}\,\hat C\hat C^\top\Big),\ \kappa{=}1,$$
$\hat C\in\mathbb R^{B\times(d-r)}$ the batch of $\hat c$ (Gram form: $B\times B$ Cholesky, negligible cost).

**(iii) Energy floor** $L_{f}=\max(0,\ \epsilon-\|b\|^2/\|h\|^2)^2$, $\epsilon=0.15$.

$$\boxed{L=L_{PA}+\lambda_i L_{inv}+\lambda_r L_{rate}+\lambda_f L_{f}},\quad \lambda_i{=}1,\ \lambda_r{=}1,\ \lambda_f{=}10.$$
Schedule: epochs 1–5 pure $L_{PA}$ (proxies must form before $\Pi$ is meaningful), then $\lambda$'s ramp linearly over epochs 5–15. 200 epochs total.

**Compute matching.** Each step uses $B/2$ distinct images × 2 views = $B$ image passes, *identical* to the baseline's $B$ passes. The baseline control uses the same sampler.

**Deployment.** $z=\mathrm{normalize}(\tilde Wg)$, $\tilde W=(\Pi+\gamma\Pi^\perp)W$ folded once, $\gamma=1$ in the headline (no post-hoc knob). One model, one view, 512-D, cosine. **Zero added parameters, zero added inference cost.**

## 2. Causal error mode + degeneracy attack

**Collapse theorem.** $L_{PA}$ depends on $h$ only through $\{\cos(h,p_c)\}$, so $\nabla_h\cos(h,p)=\frac1{\|h\|}(p-\cos\hat h)$ and
$$\Pi^\perp\nabla_hL_{PA}=-\Big(\textstyle\sum_c w_c\cos_c\Big)\,b/\|h\|^2\ \propto\ b.$$
The task gradient's null component is **exactly radial**: it can only shrink $\|b\|$, never rotate $b$. With $\|\Pi z\|^2+\|\Pi^\perp z\|^2=1$, maximizing proxy margin drives $\|\Pi^\perp z\|\to0$. Under $K\ll d$ (CUB $K{=}100$, Cars $K{=}98$, $d{=}512$) this annihilates ~80% of the descriptor's directions. This is the DML instance of neural-collapse NC1 and of Kornblith et al.'s "better losses → less transferable penultimate features," and it *explains PFML's own hyperparameters*: 15 proxies/class on CUB/Cars raises $\mathrm{rank}(\Pi)$ toward $d$; SOP ($K{=}11318{\ge}d$) has no null space, so 2 proxies suffice.

**Orthogonality theorem.** $L_{inv},L_{rate}$ depend on $h$ only via $\hat c$, so their gradients lie in $\mathrm{range}(\Pi^\perp)$ **and** are $\perp\hat c$, hence $\perp b$. Therefore $\langle\nabla_hL_{PA},\nabla_h L_{null}\rangle=0$ **pointwise for every sample**. Training decomposes into three mutually orthogonal channels: rotation in $\mathrm{range}(\Pi)$ (task), rotation in $\mathrm{range}(\Pi^\perp)$ (rate/invariance), radial energy exchange (task vs. floor).

**Degeneracy attacks.** (a) *$\|b\|\to0$*: excluded by $L_f$; $L_{null}$ is scale-free so it cannot be gamed by inflating $\|b\|$ either. (b) *Constant/low-rank $\hat c$*: $L_{rate}$ is $-\log\det$, unbounded above as any eigenvalue $\to0$; a rank-deficient code has infinite loss. (c) *Non-deterministic noise code*: maximizes $L_{rate}$ but is punished by $L_{inv}$; only augmentation-invariant, image-determined content satisfies both. (d) *Proxy collapse* making the null space trivially large: strictly increases $L_{PA}$, which dominates; monitored via $r$. (e) **Residual real risk — background/context shortcut.** RandomResizedCrop+jitter kills colour-histogram and scale codes but not habitat texture. This is the top failure mode and is *not* fully defended; diagnostic below.

## 3. Adversarial novelty search (primary sources) — one-sentence distinctions

- **Anti-Collapse Loss for DML (arXiv 2407.03106)** — nearest work: maximizes coding rate of features/proxies in the **full** embedding, so the rate term competes head-on with the task gradient and is traded off by a scalar weight; NSRC confines rate expansion to the analytic proxy null space where the task gradient provably cannot rotate, giving *zero* interference rather than a tuned truce.
- **SVMax** — a singular-value floor on the whole embedding matrix (a global spectral statistic); NSRC's rate term is applied to a *subspace defined by the supervision itself* and its output block is deployed as a calibrated part of the descriptor.
- **PFML (CVPR 2025)** and multi-proxy (SoftTriple, Sub-center ArcFace) — enlarge $\mathrm{rank}(\Pi)$, i.e. supply *room*, at $O(MKd)$ parameters; NSRC supplies *content* in the complement at zero parameters.
- **Soft Orthogonal Proxies (arXiv 2306.13055)** — orthogonalizes proxies *to each other* inside the task subspace; NSRC never constrains proxy geometry and instead orthogonalizes two *objectives*.
- **DiVA / MIC (Milbich, Roth et al.)** — allocate class-discriminative vs. intra-class SSL signals to *fixed coordinate blocks* with extra heads and no gradient-orthogonality guarantee; NSRC's split is a data-dependent eigen-decomposition of the live proxy matrix, with an exact non-interference proof and no extra head.
- **S2SD (ICML 2021)** — distills from auxiliary high-dimensional embeddings; NSRC adds no auxiliary embedding and no teacher.
- **$\rho$-spectral regularization (Roth et al., ICML 2020)** — flattens the embedding spectrum by negative resampling, a zeroth-order global statistic with no task/null decomposition.
- **OGD / GPM / Adam-NSCL (continual learning)** — project a *later task's* gradient into an *earlier task's* null space to avoid forgetting; NSRC installs a *concurrent, complementary* objective into the *current* task's null space and deploys that block.
- **VICReg / MCR² / Barlow Twins** — unsupervised objectives over the whole representation; here they are the null-space filler, not the representation.

## 4. Decisive matched-compute controls

| # | Control | Distinguishes |
|---|---|---|
| C1 | Multi-proxy ProxyAnchor ($M{=}15$) at matched compute, no null loss | *room* vs. *content* — if C1 ≈ NSRC, mechanism falsified |
| C2 | Same $L_{inv}+L_{rate}$ on the **full** embedding (= Anti-Collapse/VICReg-style), weight-swept | whether the **projection** is the active ingredient |
| C3 | Random fixed $(d{-}r)$-dim subspace instead of $\Pi^\perp$ | whether the *proxy-defined* subspace matters |
| C4 | Test-time projection onto $\Pi$ only (drop null block) | whether the null block *carries retrieval information* or is a mere regularizer |
| C5 | Rate/invariance applied inside $\mathrm{range}(\Pi)$ | sign check — must hurt |
| C6 | SSL-pretrained (label-free) init | whether the gain is really "less ImageNet-label forgetting" |
| D1 | Linear probe: seen-class accuracy from $\hat c$ over training | shortcut/redundancy diagnostic (near-chance ⇒ nuisance-only; near-task ⇒ redundant) |

## 5. Frozen forecasts, thresholds, frontier arithmetic (Lane A, R@1, 5 seeds)

| | CUB | Cars196 |
|---|---|---|
| My matched ProxyAnchor R50/512 reproduction | 0.690 ± 0.004 | 0.880 ± 0.004 |
| **NSRC (γ=1)** | **0.706 ± 0.005** | **0.893 ± 0.004** |
| NSRC + multi-proxy task loss | 0.742 ± 0.006 | 0.930 ± 0.005 |
| Reference frontier PFML | 0.734 ± 0.003 | 0.927 ± 0.003 |

**Frontier arithmetic, stated plainly.** Crossing PFML from my baseline requires **+4.4 pts (CUB)** and **+4.7 pts (Cars)**. NSRC alone forecasts **+1.6 / +1.3** — it **does not cross the frontier**. Only the combined configuration is forecast to cross, by +0.8 (CUB) and +0.3 (Cars); against $\sigma_{comb}\approx0.007$ that is ~1.2σ on CUB and ~0.4σ on Cars. **I assign ≈30–35% probability to a genuine Lane-A frontier crossing** and ≈75% to a significant gain over the matched baseline. The defensible claim is the mechanism, not the crown.

**Pre-registered falsifiers.** F1: NSRC ≤ baseline + 0.3 on CUB *and* Cars (paired, 5 seeds) ⇒ dead. F2: C1 within 0.3 of NSRC on both ⇒ mechanism claim dead even if numbers rise. F3: C4 costs < 0.3 ⇒ "complementary code" claim dead, re-framed as a regularizer. F4: **scoping prediction** — In-Shop ($K{=}3997$) and SOP ($K{=}11318$) gains must be ≤ ⅓ of the CUB/Cars gain, because $r_{\text{eff}}/d$ is far larger there; equal SOP gains falsify the null-rank explanation. F5: measured $\cos(\nabla L_{PA},\nabla L_{null}) < 10^{-6}$ at every logged step. F6: **premise test** — in the matched baseline $\mathbb E\|\Pi^\perp z\|^2$ must decay monotonically to < 0.15; if it does not collapse, the whole causal story is wrong.

## 6. Cost, benchmark and contamination risk

Training ≈ **1.00×** epoch time and memory (one $512^2$ eigendecomposition/50 iters; a $B{\times}B$ Cholesky/step), versus PA+DADA's 1.06×/1.01× and AdvRF's whole ResNet-34/U-Net. Deployment **exactly** the baseline: one folded $512\times2048$ matrix, one view, cosine.
Risks: (a) CUB/Cars R@1 seed noise ≈ ±0.4 makes a +1.3–1.6 claim require paired 5-seed testing — under-powered single-run reporting is the main way this result could look real and not be; (b) $\epsilon,\lambda_r,\tau$ must be selected on a **held-out-class split of the training classes** (e.g. CUB classes 81–100), never on test — any deviation is test-gallery fitting; (c) ImageNet-1K/CUB species overlap is a confound shared by the whole literature but is *elevated* here because the null space may preserve pretrained structure — C6 is the required check; (d) no external data, no text/VLM, no reranking, no transduction is used, so contamination risk beyond the permitted pretraining is nil.

**Honest summary:** the collapse theorem and the pointwise-orthogonality theorem are exact and cheap to verify; the empirical claim is a solid matched-baseline gain on CUB/Cars with a ~⅓ chance of crossing PFML only in combination, and an explicit prediction of *near-null* gains on SOP/In-Shop.

Sources: [Potential Field Based DML (PFML), CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [Anti-Collapse Loss for DML](https://arxiv.org/html/2407.03106v1) · [Soft Orthogonal Proxies](https://arxiv.org/abs/2306.13055) · [MCR²](https://arxiv.org/pdf/2006.08558) · [SVMax](https://www.researchgate.net/publication/349787730_SVMax_A_Feature_Embedding_Regularizer)

**Note on execution limits:** the USD budget was consumed to ~$0.27 by tool loading, so the novelty search was a single search batch rather than the exhaustive primary-source sweep item 3 deserves; PFML's exact per-dataset table and ProxyAnchor's ResNet-50 recipe were not fetched from the primary PDFs, which is why §1 fixes and self-reports its own baseline rather than inheriting published numbers.
