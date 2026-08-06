# Blind Opus proposal: DSA (Pass 29)

Consultation ID: `86c0b7f59dbf4509`

# Method: **Discriminant-Subspace Ablation (DSA)**

**Lane: A** (matched 512-D CNN lane). All forecasts, baselines and comparisons below are Lane A only.

---

## 0. One-paragraph statement

Train a standard proxy-based metric model, but at every step also re-run the *same* loss on the descriptor after projecting out the top-$r$ eigenvectors of an EMA estimate of the **between-identity scatter** $S_B$, with $r$ resampled each step. The projector is detached and data-dependent, so the constraint is exactly invariant under global rotations of descriptor space, which makes it provably immune to the two cheap ways a network fakes redundancy (coordinate duplication and basis choice). The mechanism target is the *shape of the between-identity gain spectrum*, not the loss value, not the proxy count, and not the marginal embedding distribution. Cost: **+2–3% training wall-clock, +2 MB, zero extra parameters, deployment byte-identical to the base model.**

---

## 1. Executable mathematics

### 1.1 Objects

| Object | Definition | Learned? |
|---|---|---|
| $\phi_\theta$ | ResNet-50 to global average pool, $\mathbb R^{2048}$, ImageNet-1K init | yes |
| $W$ | $512\times2048$, no bias; $z=W\phi_\theta(x)$ | yes |
| $f(x)$ | $z/\lVert z\rVert_2 \in S^{511}$ — **the deployed descriptor** | — |
| $\Pi=\{p_{c,k}\}$ | $c=1..C$, $k=1..K$ proxies per class, $\hat p = p/\lVert p\rVert$ | yes |
| $S$ | $512\times512$ EMA between-identity scatter | **no — detached buffer** |
| $P_r$ | $I - U_rU_r^\top$, $U_r$ = top-$r$ eigenvectors of $S$ | **no — detached** |

### 1.2 Base loss (exact Proxy-Anchor reduction, Kim et al., CVPR 2020)

$$
L_{\mathrm{PA}}(F,P)=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\!\!\sum_{x\in X_p^+}\!\!e^{-\alpha(s(x,p)-\delta)}\Big)+\frac{1}{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X_p^-}\!\!e^{\alpha(s(x,p)+\delta)}\Big)
$$

$s(x,p)=\langle f(x),\hat p\rangle$, $\alpha=32$, $\delta=0.1$, $P^+$ = proxies with ≥1 positive in the batch.

Multi-proxy ($K>1$) extension used as my base — stated explicitly because it is **mine, not PFML's**: hard nearest-proxy assignment for positives, $X^+_{c,k}=\{x: y_x=c,\ k=\arg\max_j s(x,p_{c,j})\}$; all proxies of other classes remain negatives.

Per-similarity gradients (needed below):

$$
\frac{\partial L_{\mathrm{PA}}}{\partial s(x,p)}=\begin{cases}-\dfrac{\alpha}{|P^+|}\cdot\dfrac{e^{-\alpha(s-\delta)}}{1+\sum_{x'\in X_p^+}e^{-\alpha(s'-\delta)}} & x\in X_p^+\\[2mm] +\dfrac{\alpha}{|P|}\cdot\dfrac{e^{\alpha(s+\delta)}}{1+\sum_{x'\in X_p^-}e^{\alpha(s'+\delta)}} & x\in X_p^-\end{cases}
$$

### 1.3 The ablation operator

**(a) Scatter estimate.** For batch $B$ with class set $\mathcal C_B$ (balanced sampler, $m=4$ images/class ⇒ $|\mathcal C_B|=30$ at batch 120):

$$
m_c=\tfrac1{n_c}\!\!\sum_{i:y_i=c}\!\!f(x_i),\quad \bar m=\tfrac1{|\mathcal C_B|}\sum_c m_c,\quad
S_B^{\text{batch}}=\tfrac1{|\mathcal C_B|}\sum_c (m_c-\bar m)(m_c-\bar m)^\top
$$

$$
S \leftarrow \gamma S + (1-\gamma)\,\mathrm{sg}[S_B^{\text{batch}}],\qquad \gamma=0.99
$$

The EMA is **not optional**: $\operatorname{rank}(S_B^{\text{batch}})\le 29$, so a single batch cannot see more than 29 of the 512 directions. $\gamma=0.99$ gives a ~100-step memory ≈ 3000 class-mean samples.

**(b) Eigenbasis**, refreshed every $T=50$ steps: $\mathrm{eigh}(S)\to u_1..u_{512}$, $\lambda_1\ge\dots\ge\lambda_{512}$.

**(c) Rank draw**, every step: $r\sim\mathrm{Unif}\{1,\dots,R\}$, $R=32$ (CUB/Cars/In-Shop), $R=64$ (SOP).

**(d) Ablated views:** $P_r=I-U_rU_r^\top$ (detached),

$$
\tilde f(x)=\frac{P_rf(x)}{\lVert P_rf(x)\rVert},\qquad \tilde p=\frac{P_r\hat p}{\lVert P_r\hat p\rVert}
$$

### 1.4 Objective

$$
\boxed{\;L=L_{\mathrm{PA}}(f,\Pi)+\beta_t\,L_{\mathrm{PA}}(\tilde f,\tilde\Pi)\;}
$$

$\beta_t=0$ for epoch $<E_0=20$; linear ramp to $\beta_{\max}=0.5$ over $[E_0,E_1]=[20,40]$; constant after.

### 1.5 Gradient path (complete)

Both terms are functions of the same $f$, so autograd sums at $f$ and the backbone backward pass runs **once**:

$$
\frac{\partial L}{\partial f}=\frac{\partial L_{\mathrm{PA}}}{\partial f}+\frac{\beta_t}{\lVert P_rf\rVert}\big(P_r-\tilde f\tilde f^\top\big)\frac{\partial L_{\mathrm{PA}}}{\partial \tilde f}
$$

(using $P_r$ symmetric idempotent and $\tilde f\in\operatorname{range}(P_r)$, so $(I-\tilde f\tilde f^\top)P_r=P_r-\tilde f\tilde f^\top$). Then $\partial L/\partial z=\lVert z\rVert^{-1}(I-ff^\top)\,\partial L/\partial f$, then into $W$ and $\theta$. Proxy path: identical form composed twice through $\hat p$ and $\tilde p$. **No second-order terms, no second backbone pass.**

### 1.6 Frozen recipe (Lane A)

200 epochs; AdamW; lr $10^{-4}$ (backbone + $W$) for CUB/Cars, $6\times10^{-4}$ for SOP/In-Shop; **proxy lr $\times100$**; weight decay $10^{-4}$ on $\theta,W$ and **0 on proxies**; cosine decay to 0 with 5-epoch linear warm-up; backbone frozen epoch 1; batch 120, balanced $m=4$; proxies $\sim\mathcal N(0,I/512)$ then normalized; train aug = RandomResizedCrop(224, scale (0.16,1)) + hflip; test = resize 256 → center crop 224. $K=15$ (CUB/Cars/In-Shop), $K=2$ (SOP).

**On the renormalization — not harmless.** $\lVert P_rf\rVert<1$, so renormalizing the residual *amplifies* the residual geometry and changes the operational margin, and AdamW weight decay on $W$ makes the pre-normalization scale of $z$ operational. Defaults are pre-registered as $\alpha_{\text{abl}}=\alpha$, $\delta_{\text{abl}}=\delta$; I additionally pre-register a scale-sensitivity ablation at $\alpha_{\text{abl}}=\alpha\cdot\mathbb E[\lVert P_rf\rVert]$ and will report $\mathbb E[\lVert P_rf\rVert]$ per epoch.

### 1.7 Test-time operation

**Unchanged.** One ResNet-50 forward, one 224px center crop, one 512-D L2-normalized descriptor, cosine 1-NN. $S$, $U_r$, $\Pi$ are discarded.

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode: leading-direction monopoly of the between-identity scatter

Write the identity contrast $\Delta_{ab}=\mu_a-\mu_b$ in the eigenbasis of the training $S_B$: $\Delta=\sum_i\delta_iu_i$. Let $g_i$ be the encoder's gain along $u_i$. The discriminative objective is minimized most cheaply by loading a few directions; and structurally, $\operatorname{rank}(S_B)\le C-1$, so on CUB ($C{=}100$) **at most 99 of 512 directions can carry any class-mean signal at all**, and empirically the realized spectrum is far spikier than that bound.

For a *novel* identity pair the contrast direction $v$ was not selected by training. Under the zero-shot assumption that $v$ is exchangeable w.r.t. the training eigenbasis, transmitted contrast is the quadratic form $T=\sum_i g_iv_i^2$ with

$$
\mathbb E[T]=\bar g,\qquad \operatorname{Var}[T]\approx \frac{2}{d}\sigma_g^2 .
$$

Failures are the mass $T<t$ (within-class dispersion threshold). At CUB's operating point (27% failures ⇒ $z=(\bar g-t)/\sigma_T=0.613$), $\ \Delta\Phi=-\varphi(z)\,z\,(\Delta\sigma_T/\sigma_T)$. A 20% reduction of $\sigma_T$ at fixed $\bar g$ gives $0.331\times0.613\times0.20=+4.1$ points.

**Flattening the gain spectrum at fixed trace strictly reduces the failure mass.** That +4.1 is an idealized ceiling (exchangeability is only partly true — novel bird species *do* load on training attribute directions, which is why ImageNet init works at all); §5 discounts it by ~4×.

A second, additive effect: after ablating $u_1..u_r$, the residual loss can only fall by redistributing existing evidence **or extracting new image evidence**, raising $\operatorname{tr}(g)$. The erasure literature in WSOL and re-ID is direct evidence that erasure induces the latter.

### 2.2 Proof-level attack on the cheapest degeneracies

**D1 — Coordinate replication (defeated).** Suppose the encoder writes one scalar $s(x)$ into $k$ orthogonal coordinates: $z\supset (s/\sqrt k)(e_1+\dots+e_k)$. Then every class-mean deviation from that block is collinear with $w=\mathbf 1_k/\sqrt k$, so that block contributes a **rank-1** term to $S_B$ with eigenvector $w$. Ablating $r\ge1$ annihilates all $k$ copies simultaneously. ∎ Linearly-correlated duplication is provably worthless against DSA — this is precisely what defeats coordinate dropout and Matryoshka-style redundancy.

**D2 — Decoy directions (defeated).** $U_r$ is by construction the top eigenvectors of the *between-identity* scatter of the model's own current descriptors. A decoy with large between-identity energy *is* discriminative, so ablating it costs real capacity. Inflating *within*-class variance instead does not enter $S_B$ and is punished by the primary loss. The only cheap satisfaction is a flat spectrum — the intended solution.

**D3 — Basis / estimator gaming (defeated).** $L_{\mathrm{abl}}$ is **exactly equivariant under any global orthogonal transform $Q$ of descriptor space**: $f\mapsto Qf \Rightarrow S\mapsto QSQ^\top \Rightarrow U_r\mapsto QU_r \Rightarrow \tilde f\mapsto Q\tilde f$, and cosines are preserved. ∎ There is no basis in which the constraint is cheaper. Staleness gaming is bounded by $\gamma=0.99$ (100-step memory > $T=50$).

**D4 — Collapse (structurally excluded).** DSA is *not* a spectral-dispersion penalty. It is the **same discriminative loss on the residual**. $S\to0$ raises both terms. This is exactly why I did not use $-\log\det$, spectral entropy, or an explicit flatness penalty: those admit collapse and volume-inflation optima; the ablated-task-loss form has neither.

**D5 — Noise inflation (NOT defeated; the honest failure mode).** The encoder can flatten $S_B$ by loading residual directions with cues that separate *training* identities but do not generalize (background, per-collection color statistics, JPEG artifacts). $L_{\mathrm{abl}}$ cannot tell these from real evidence. Two partial guards: (i) $L_{\mathrm{abl}}$ is evaluated on augmented views with margin $\delta$, so crop/flip-fragile cues are penalized; (ii) **all hyperparameters are selected on a held-out *training-identity* split** (below), which is itself a zero-shot-to-identity test, so shortcut-driven flattening is not rewarded there. This is the mode I expect to cap the gain, and F3/F4 in §5 are designed to catch it.

---

## 3. Adversarial novelty search — nearest works, one-sentence distinctions

**Inside DML / retrieval**

1. **Batch DropBlock / Batch Feature Erasing** (Dai et al., ICCV 2019) — erases a contiguous **spatial** block of feature maps in a second network branch; DSA erases a data-dependent **discriminant subspace of the global descriptor**, with no extra branch, no extra parameters, and exact rotation-invariance.
2. **BIER (ICCV 2017) / Divide-and-Conquer (CVPR 2019) / DREML / ABE-8** — partition the embedding into **fixed axis-aligned chunks** diversified by boosting, data clustering, or attention; DSA keeps one undivided descriptor and imposes a basis-free constraint with no test-time ensemble.
3. **SoftTriple / multi-proxy Proxy-Anchor / PFML (CVPR 2025)** — raise the **rank of the target structure** by giving each class multiple centers (which is why PFML needs 15 proxies where $C{<}d$ and only 2 on SOP where $C\gg d$); DSA leaves targets untouched and applies spectral pressure to the encoder's **realized** scatter, so the two act on different objects.
4. **DADA (AAAI 2024)** — aligns the sample and proxy **distributions** via data-augmented domain adaptation; DSA never touches the sample/proxy gap.
5. **AdvRF (ICCV 2025) / VAPNet (NeurIPS 2023)** — add a **separate auxiliary network** (reconstruction / attribute) plus distillation; DSA adds zero parameters and ~2% wall-clock.
6. **Energy-Confused Adversarial Metric Learning (AAAI 2019)** — adversarially perturbs the **distance/energy distribution** toward confusion; DSA has no adversary and its projector is closed-form from $\mathrm{eigh}(S)$.
7. **Fisher/LDA-style deep metric learning** — maximizes $\operatorname{tr}(S_W^{-1}S_B)$, whose optimum is *also* attained by a spiky $S_B$; DSA requires the $(r{+}1)$-th direction onward to **individually satisfy the margin**, which the trace criterion does not.
8. **All-but-the-Top (ICLR 2018) / PCA-whitening post-processing** — a fixed post-hoc linear map that **rescales existing evidence**; control C3 isolates exactly this difference.

**Outside DML**

9. **INLP / "Null It Out"** (Ravfogel et al., ACL 2020) — post-hoc iterative nullspace projection to **delete a nuisance attribute** from a frozen representation; DSA applies nullspace projection to the **task's own leading discriminants during training**, to *add* redundancy, and never modifies the deployed descriptor.
10. **Hide-and-Seek / ACoL / Erasing-Integrated Learning** (WSOL) — erase the most-activated **image regions**; DSA erases in **feature-covariance space**, forcing discovery of non-localizable evidence (global color/texture statistics) that spatial erasure cannot reach.
11. **Matryoshka Representation Learning** (NeurIPS 2022) — trains **nested axis-aligned prefixes** so dimensions are *deliberately ordered by importance*; DSA targets the opposite geometry (rotation-invariant flatness) for a different purpose.
12. **MCR² / Maximal Coding Rate Reduction** (NeurIPS 2020) — an explicit $\log\det$ volume objective; DSA has no volume term (D4).
13. **VICReg / Barlow Twins / Uniformity (Wang & Isola)** — regularize the **marginal** embedding distribution; DSA regularizes the **between-identity** scatter, a different second-order object (a marginal can be perfectly isotropic while $S_B$ is rank-3).
14. **Coordinate dropout on embeddings** — axis-aligned and data-independent, satisfiable by choosing a redundant basis; provably insufficient against DSA by D1.

I found no primary source applying **data-dependent leading-discriminant subspace ablation as a training-time stressor** in deep metric learning.

---

## 4. Decisive matched-compute controls

| ID | Control | Isolates | Prediction |
|---|---|---|---|
| **C0** | Base $K{=}1$ and $K{=}15$, identical recipe | — | reference |
| **C1** | **Random $r$-frame ablation**, same $\beta$, same $r$ schedule, identical cost | top-discriminant vs. *any* ablation | **≤ +0.2** — by Johnson–Lindenstrauss, at $r\le32,d=512$ a random projection preserves the batch cosine geometry to $O(\sqrt{\log B/(d-r)})$, so the term is near-vacuous |
| **C2** | Bottom-$r$ ablation | direction of the spectral pressure | ≤ +0.1 |
| **C3** | **Post-hoc $S_B$-whitening of C0** (train statistics only — legal) | *rescale existing evidence* vs. *extract new evidence* | +0.2 to +0.6 |
| **C4** | C0 at 206 epochs (+3%) | compute match | ≤ +0.1 |
| **C5** | C0 + second 512-D head with its own $L_{\mathrm{PA}}$ | "extra loss term / extra gradient" | ≤ +0.3 |
| **C6** | C0 + $L_{\mathrm{PA}}$ on Bernoulli$(1-r/512)$-masked descriptor | basis-free vs. axis-aligned erasure (D1) | ≤ +0.3 |
| **C7** | C0 with $K$ raised to match parameter count / target rank | DSA vs. "more proxy capacity" | — |
| **C8** | **Mechanism probe** (not performance): $\mathrm{PR}(S_B)=(\sum\lambda_i)^2/\sum\lambda_i^2$ on the held-out-identity split | did the mechanism engage at all | base 8–15 → DSA 30–45 on CUB |

**Protocol hygiene.** A fixed identity-level validation split is carved from the *official training identities* (CUB: train identities 81–100; Cars: 89–98; SOP/In-Shop: 5% of training identities) and used for **all** hyperparameter selection, $\beta/R$ tuning, and early stopping. Test identities are touched once per frozen configuration. This is *stricter* than standard practice in this literature, which routinely selects on the test split — so my numbers are conservative relative to the references, and I will additionally report the test-selected number, clearly labelled, for apples-to-apples.

---

## 5. Frozen forecasts — Lane A (5 seeds, R@1)

| Configuration | CUB | Cars196 |
|---|---|---|
| C0 base, $K{=}1$ | 0.705 ± 0.005 | 0.885 ± 0.005 |
| C0 base, $K{=}15$ | 0.720 ± 0.004 | 0.916 ± 0.004 |
| **DSA on $K{=}1$** | **0.723 ± 0.005** (Δ +1.8, 90% CI [+0.9, +2.7]) | **0.896 ± 0.005** (Δ +1.1, [+0.3, +1.9]) |
| **DSA on $K{=}15$** | **0.731 ± 0.004** (Δ +1.1, [+0.4, +1.8]) | **0.923 ± 0.004** (Δ +0.7, [+0.1, +1.3]) |
| Reference **PFML** (CVPR 2025) | 0.734 ± 0.003 | 0.927 ± 0.003 |
| Reference **DADA** matched-cost rows | 0.729 | 0.921 |

**SOP (secondary, mechanism-differential):** C0 $K{=}2$ 0.822 ± 0.003 → **DSA 0.827 ± 0.003** (Δ +0.5, [+0.0, +1.0]); PFML 0.829 ± 0.002. **The smaller SOP gain is a prediction, not a hedge**: with $C{=}11{,}318\gg512$ the rank-starvation half of the mechanism is inoperative and only the spectral-shape half survives.

**In-Shop:** the reference (PA+DADA 0.930) has **unreported seed count and uncertainty**, so no meaningful crossing test exists. I forecast DSA on $K{=}15$ at 0.928 ± 0.003 and decline to claim a crossing either way.

### Frontier-crossing arithmetic (explicit, including the negative)

PFML CUB $0.734\pm0.003$ over 5 runs ⇒ SE $=0.003/\sqrt5=0.0013$. DSA-$K{=}15$ forecast $0.731$, SE $0.0018$. Difference $-0.003\pm0.0022$ ⇒ **P(cross) ≈ 0.15** at the point forecast, **≈ 0.30** once my ±0.006 uncertainty in reproducing a competitive $K{=}15$ base is folded in. Cars: $-0.004\pm0.0022$ ⇒ P(cross) ≈ 0.10–0.25.

**I forecast that DSA reaches PFML's error bar and does not cross it.** Against the matched-cost DADA rows the forecast crossings are +0.2 (CUB) and +0.2 (Cars) — inside noise. The defensible claim is therefore a **mechanism claim with a ≈+1-point effect at 1.02× training cost and 1.00× deployment cost**, on an axis orthogonal to the proxy-count axis — not a forecast frontier break.

The natural crossing configuration is **DSA stacked on PFML** (predicted sub-additive, ~60% of standalone Δ, since both raise effective rank by different routes). Per the rule against inheriting a published frontier without a matched reproduction, **I decline to forecast a number for that** absent a reproduction of PFML's potential-field loss, which I could not obtain from primary sources this session.

### Pre-registered falsification thresholds

- **F1** — mean Δ on CUB over 5 seeds $< +0.8$ pts on the $K{=}1$ base ⇒ falsified at the claimed effect size.
- **F2** — C1 (random subspace) Δ ≥ 60% of DSA's Δ ⇒ the effect is generic stochastic regularization ⇒ falsified.
- **F3** — C3 (post-hoc whitening) Δ ≥ 80% of DSA's Δ ⇒ it is a linear reparameterization obtainable free at test time ⇒ falsified, and C3 should be reported instead.
- **F4** — $\mathrm{PR}(S_B)$ on the held-out-identity split does not rise ≥ 2× ⇒ the stated mechanism never engaged ⇒ claim withdrawn *even if R@1 improved*.
- **F5** — Δ(SOP) ≥ Δ(CUB) ⇒ the rank-starvation differential prediction is contradicted ⇒ mechanism story falsified even if R@1 improves.
- **F6** — C6 (coordinate dropout) matches DSA ⇒ the D1 basis-invariance argument is falsified.

---

## 6. Cost, and benchmark / contamination risks

**Training cost.** One extra $L_{\mathrm{PA}}$ on 512-D vectors ($\sim$8 MFLOP), one $30\times512^2$ scatter update, one $\mathrm{eigh}(512)$ per 50 steps ($\sim$0.2 ms/step amortized). Because both loss terms are functions of the same $f$, **the backbone backward pass runs once** — the dominant cost is untouched. Measured overhead expectation **1.02–1.03× time, +2 MB, +0 parameters.** Compare: PA+DADA 1.06×, AdvRF an entire ResNet-34/U-Net plus distillation.

**Deployment cost.** Identical to the base model. Single model, one view, 512-D, cosine NN. Legal under every stated constraint: official training images + identity labels + ordinary stochastic augmentation + ImageNet-1K init only.

**Hyperparameter debt (a real cost).** DSA introduces six knobs ($R,\beta_{\max},E_0,E_1,\gamma,T$) on top of $K$. Over-tuning risk is the second-largest threat after D5. Mitigated only by the held-out-identity protocol, which I regard as necessary, not optional.

**Benchmark risks.**
- CUB/Cars test splits are ~5.9k images; seed noise is ±0.4–0.5 pts. A +1.1 claim needs ≥5 seeds and a paired test; single-run differences below ~0.6 pts are not resolvable in this lane.
- ImageNet-1K pretraining overlaps CUB (≈59 bird classes) and Cars196 (car/truck classes). This inflates *all* Lane-A numbers uniformly and is not introduced here, but it means "zero-shot" holds only w.r.t. the DML label set.
- SOP and In-Shop contain near-duplicate images within identity; R@1 there is partly a near-duplicate detection score, which is one reason I expect the mechanism to bite less on SOP.
- The comparison is asymmetric in my disfavour: most published Lane-A numbers are selected on the test split.

**Unresolved source ambiguities (stated, not papered over).**
1. PFML's exact potential-field loss, per-dataset proxy counts (the project page does not disclose the 15/15/2 figures — I take the prompt's audited values as given), epoch budget, optimizer, and input resolution. I do **not** reproduce PFML and do not inherit its frontier.
2. The Proxy-Anchor README discloses lr (1e-4 CUB/Cars, 6e-4 SOP/In-Shop), batch 120 for ResNet-50, dim 512, `--lr-decay-gamma 0.25`, warm-up 1 or 5 epochs. It does **not** disclose optimizer, weight decay, the proxy-lr multiplier, or total epochs. $\alpha{=}32$, $\delta{=}0.1$ and the ×100 proxy lr are from the paper/code as I recall them but I **could not verify them from a primary source in this session** — flagged as unverified.
3. DADA's In-Shop 0.930 has unreported seeds/uncertainty (per the prompt).
4. The lane's 200-epoch LR schedule is unspecified; I fix cosine decay and record it as **my choice**, not an inherited recipe.

---

**Largest residual uncertainties, stated plainly:** (i) D5 noise-inflation may consume most of the predicted gain, in which case F3/F4 will fire; (ii) the +4.1-point variance-reduction ceiling rests on an exchangeability assumption that is only partly true, and I have discounted it ~4× to +1.1 — that discount is judgement, not derivation; (iii) if the base $K{=}15$ reproduction lands materially below 0.720 on CUB, the whole comparison shifts and no crossing claim survives.

Sources: [PFML (CVPR 2025) project page](https://shubhangb97.github.io/potential_field_DML/) · [PFML CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/33305) · [DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29400) · [DADA arXiv](https://arxiv.org/pdf/2401.00617) · [Proxy Anchor (CVPR 2020)](https://arxiv.org/pdf/2003.13911) · [Proxy Anchor official code](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) · [Batch DropBlock (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.pdf) · [Null It Out / INLP (ACL 2020)](https://aclanthology.org/2020.acl-main.647/) · [Energy Confused Adversarial Metric Learning (AAAI 2019)](https://arxiv.org/abs/1901.07169)

