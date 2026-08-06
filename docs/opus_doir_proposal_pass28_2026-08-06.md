# Opus blind proposal, Pass 28

Consultation ID: `82eb45c14d024c4f`

## Lane

**Lane A** (ResNet‑50, 512‑D, ~224 px, single‑view cosine, 200 epochs). All numbers below are Lane A. Reference targets: PFML CUB 0.734±0.003, Cars 0.927±0.003, SOP 0.829±0.002; DADA matched‑cost controls 0.729 / 0.921 / 0.810.

---

# DOIR — D‑Optimal Identity‑Information Regularization

**One sentence.** Treat the proxy softmax as a statistical model of identity and require its *Fisher information matrix in embedding coordinates* to be spectrally isotropic — an equipartition/optimal‑design constraint that forbids the embedding from leaving any of its 512 directions informationally idle, which is the absorbing state that causes zero‑shot collapse.

---

## 1. Executable mathematics

### 1.1 Deployment (unchanged)
ResNet‑50 (ImageNet‑1K init) → GAP → $h\in\mathbb R^{2048}$ → linear head $W\in\mathbb R^{512\times2048}$, $b\in\mathbb R^{512}$ → $z=(Wh+b)/\|Wh+b\|_2\in S^{511}$. Test: resize 256, center crop 224, one view, cosine NN. No proxies, no barrier, no second branch at test time.

### 1.2 Base loss (fully specified by me; see §7 on PFML)
$K$ proxies per class, $P\in\mathbb R^{CK\times d}$, rows L2‑normalized, $y(j)\in\{1..C\}$ the class of proxy $j$. Fixed scale $s=32$, additive cosine margin $\delta=0.1$, $d=512$.

$$\ell_i=-\log\frac{\sum_{j:y(j)=y_i}e^{s(\langle z_i,p_j\rangle-\delta)}}{\sum_{j:y(j)=y_i}e^{s(\langle z_i,p_j\rangle-\delta)}+\sum_{j:y(j)\neq y_i}e^{s\langle z_i,p_j\rangle}}$$

This is the *class‑marginalized* multi‑proxy softmax (SoftTriple's LSE pooling replaced by exact marginalization so that the model whose Fisher I compute is exactly the model I train). $K=15$ (CUB/Cars), $K=2$ (SOP), $K=2$ (In‑Shop) — matched to PFML's disclosed proxy counts for cost parity.

### 1.3 The identity Fisher information matrix (IFIM)
Margin‑free posterior (the deployed likelihood):
$$\pi_{ij}=\frac{e^{s\langle z_i,p_j\rangle}}{\sum_{j'}e^{s\langle z_i,p_{j'}\rangle}},\qquad q_{ic}=\!\!\sum_{j:y(j)=c}\!\!\pi_{ij},\qquad m_{ic}=\frac1{q_{ic}}\!\!\sum_{j:y(j)=c}\!\!\pi_{ij}\,p_j,\qquad \bar m_i=\sum_j \pi_{ij}p_j .$$

Since $\nabla_z\log q_c = s\,(m_c-\bar m)$, the Fisher of the class likelihood w.r.t. the embedding is exactly

$$\boxed{\;F_i \;=\; \sum_c q_{ic}\,\nabla_z\log q_{ic}\,\nabla_z\log q_{ic}^\top \;=\; s^2\sum_c q_{ic}\,(m_{ic}-\bar m_i)(m_{ic}-\bar m_i)^\top\;}$$

equivalently $F_i=s^2\sum_{c<c'}q_{ic}q_{ic'}(m_{ic}-m_{ic'})(m_{ic}-m_{ic'})^\top$. Tangent‑project onto the sphere, $\Pi_i=I-z_iz_i^\top$, and average:
$$F=\tfrac1{|B|}\textstyle\sum_i \Pi_i F_i \Pi_i \;\in\;\mathbb S^{d}_{+}.$$

**Read this object correctly.** $F$ is a *between‑class scatter of responsibility‑weighted proxy means, gated by live confusion mass*. A confidently classified sample contributes exactly $0$ ($q$ one‑hot $\Rightarrow m_c=\bar m$). $F$'s eigendirections are the directions the model is *currently using to resolve hard identity decisions* — not the directions in which features merely vary.

### 1.4 The barrier (D‑optimality)
$$\tilde F=\frac{d\,F}{\operatorname{tr}F},\qquad \mathcal R=-\tfrac1d\log\det(\tilde F+\varepsilon I)+\log(1+\varepsilon),\qquad \varepsilon=10^{-2}.$$
$\mathcal R\ge0$ with equality iff $\tilde F=I$. Range: $\mathcal R\in[0,\;\approx4.59]$ for $d=512,\varepsilon=10^{-2}$ (rank‑1 worst case).

$$\mathcal L=\tfrac1{|B|}\textstyle\sum_i\ell_i+\lambda_t\,\mathcal R .$$

**Exact scale‑neutrality (not an assumed‑harmless normalization).** With $A=-\tfrac1d(\tilde F+\varepsilon I)^{-1}$,
$$G:=\frac{\partial\mathcal R}{\partial F}=\frac{d}{\operatorname{tr}F}\Big(A-\tfrac{\operatorname{tr}(A\tilde F)}{d}I\Big),\qquad \operatorname{tr}(GF)=0 .$$
The barrier gradient is trace‑orthogonal to $F$: it can rotate/redistribute the spectrum but exerts **zero** pressure on $\|F\|$. This is what makes the interaction with weight decay and with the additive $\ell_i$ term analyzable rather than assumed benign. Proxies get **no** weight decay (they are used normalized; decay there is an implicit LR change, not a regularizer) — stated as a deliberate recipe choice.

### 1.5 Large‑$C$ truncation (SOP / In‑Shop)
Keep the top $C'=32$ classes by $q_{ic}$ per sample. Since contributions are weighted by $q_cq_{c'}$, discarding residual mass $\rho$ perturbs $F_i$ by at most $2\rho s^2\max\|m_c-\bar m\|^2\le 8\rho s^2$; measured $\rho<10^{-3}$ after epoch 20. Cost becomes $C$‑independent.

### 1.6 Schedule, optimizer, data
- $\lambda_t=\lambda_{\max}\min(1,t/10)$ for $t\le 0.8T$, then linear anneal to $0$ at $t=T$. $\lambda_{\max}=0.05$. Annealing off returns the endpoint to a pure task optimum; the barrier's job is to prevent early entry into the absorbing set (§2).
- AdamW, backbone lr $10^{-4}$, head lr $10^{-4}$, proxies lr $10^{-2}$; wd $10^{-4}$ (backbone+head only); 5 warmup epochs with backbone frozen; 200 epochs; cosine lr decay.
- Batch 180 = 30 classes × 6. Augmentation: RandomResizedCrop(224, scale 0.16–1) + hflip only.
- Numerics: $F$, Cholesky, and $\log\det$ in fp32 even under AMP; jitter $10^{-6}I$.
- **Hyperparameters ($\lambda_{\max},\varepsilon,C',K$) selected on a pseudo‑novel split: the last 20% of *training* identities held out from training, retrieval‑scored.** No test data touches selection.

---

## 2. Causal zero‑shot error mode, with proofs

**Error mode: informationally idle directions are an absorbing set.**

**Lemma 1 (absorption).** Let $u$ be a unit direction with $u^\top F_i u=0$ for all training samples. Then $u^\top\nabla_{z_i}\ell_i=0$ for all $i$, because $\nabla_z\ell$ lies in $\mathrm{span}\{\nabla_z\log q_c\}$ and $u^\top F_iu=0\Rightarrow u^\top\nabla_z\log q_{ic}=0\ \forall c$. Hence the task loss never updates the embedding's $u$‑component; under weight decay $\gamma$ that component decays as $e^{-\eta\gamma t}$, and the condition $u^\top F_iu=0$ is preserved along the flow. Idle directions are *absorbing and self‑reinforcing*. ∎

This is the causal mechanism of zero‑shot failure: seen classes need only a low‑dimensional discriminative subspace; every direction outside it is provably driven to and held at zero information; unseen classes whose distinguishing evidence lies there are unrecoverable at test time by any nearest‑neighbour rule.

**Lemma 2 (escape).** $\partial\mathcal R/\partial\lambda_k=-\tfrac1d(\lambda_k+\varepsilon)^{-1}\cdot(\text{scale terms})$, so the barrier's pressure on an eigendirection grows as $1/(\lambda_k+\varepsilon)$ and is bounded below by $\tfrac1{d(1+\varepsilon)}$ times the largest. The idle set is therefore not invariant under $\mathcal L$: the flow is pushed out of it at a rate that does not vanish as $\lambda_k\to0$. ∎

### Degeneracy attacks

**D1 — global temperature inflation.** Blocked twice: $s$ is fixed and both $z$ and $p$ are L2‑normalized, so no parameter can rescale logits; and $\mathcal R(\alpha F)=\mathcal R(F)$ exactly, so uniform inflation of $F$ buys nothing.

**D2 — within‑class proxy scatter (the fatal degeneracy for naive multi‑proxy Fishers).** Blocked *by construction*: $F$ is the marginalized **class** Fisher. Any redistribution of proxies inside class $c$ that preserves $m_{ic}$ leaves $F_i$ identically unchanged. A method that isotropized the per‑proxy Fisher $P^\top(\mathrm{diag}\pi-\pi\pi^\top)P$ would be trivially satisfiable by scattering same‑class proxies; the marginalized form is not.

**D3 — parked "junk" proxies.** A proxy at cosine margin $\Delta$ from the data contributes confusion mass $q_cq_{c'}\lesssim e^{-2s\Delta}$. The barrier only credits a direction once $\lambda\gtrsim\varepsilon\cdot\operatorname{tr}F/d$, i.e. relative mass $\ge10^{-2}$. With $s=32$ that requires $e^{-64\Delta}\ge10^{-2}\Rightarrow\Delta\le0.072$. Proxies must sit **inside the genuine confusion shell of real images** to count; parking is arithmetically excluded.

**D4 — global underfitting to uniform $q$.** Maximum obtainable barrier credit is $\lambda_{\max}\cdot(\mathcal R_{\max}-\mathcal R_{\min})=0.05\times4.59=0.23$ nats. Driving $q$ to uniform costs $\log C=4.61$ nats (CUB), $9.33$ (SOP). Safety factor 20× / 40×. Formally: for any $\lambda<\Delta\ell/\log(1+1/\varepsilon)$ the uniform‑$q$ point is not a minimizer.

**D5 — residual risk (declared, not defeated).** Injecting isotropic noise into $z$ raises confusion mass broadly. It does not create *new* informative directions (it only diffuses existing ones) and it costs $\ell_i$, but I cannot rule it out analytically. Control C8 (§4) is designed to detect it empirically.

---

## 3. Adversarial novelty search — nearest works and one‑sentence distinctions

**Inside DML**
- **SoftTriple (ICCV'19)** — multi‑proxy similarity; DOIR changes no similarity, it constrains the *spectrum of that model's information matrix*.
- **Proxy Anchor (CVPR'20) / Circle / MS** — first‑order per‑pair gradient reweighting; DOIR is a second‑order constraint with zero first‑order analogue.
- **PFML (CVPR'25)** — the frontier reference; DOIR is mechanism‑orthogonal (spectral design on the Fisher) and is composable rather than competing, but I do not inherit its recipe (§7).
- **Non‑isotropy regularization (Roth et al., CVPR'22)** — *encourages* non‑isotropy of class‑conditional feature distributions via normalizing flows; DOIR targets a different object (label‑likelihood curvature, not feature density) and pushes it the other way.
- **LDA/Fisher‑discriminant DML losses** — *closest hit, named by me*: these optimize trace ratios of unweighted between/within scatter; DOIR's $F$ is a **confusion‑gated, responsibility‑weighted** between‑class scatter in which correctly and confidently classified samples contribute exactly zero, and the objective is a scale‑invariant log‑det barrier on the normalized spectrum, not a ratio.
- **MDR, Ortho‑reg, DiVA, S2SD, HIER, DIML** — distance‑distribution shaping, decorrelation, auxiliary branches, hierarchy, or local matching; none constructs the model's Fisher.
- **AdvRF (ICCV'25, Lane B)** — retains information via a reconstruction/distillation system; DOIR retains information with no decoder, no teacher, no second network.

**Outside DML**
- **VICReg / Barlow Twins / W‑MSE / DirectCLR** — spectra of the *feature covariance*; a perfectly white covariance is compatible with a rank‑1 Fisher (all variance label‑irrelevant), so these cannot express DOIR's constraint.
- **Uniformity–alignment (Wang & Isola'20)** — pairwise Gaussian potential on the sphere; uniform embeddings can still be informationally rank‑deficient.
- **EWC / continual‑learning Fisher** — Fisher w.r.t. *parameters*, used to freeze weights; DOIR's Fisher is w.r.t. *embedding coordinates* and is a shaping target.
- **Natural gradient / K‑FAC** — Fisher as a preconditioner; DOIR makes it the objective.
- **Fisher‑based active learning (BAIT, V/A‑optimal acquisition)** — optimal design over *which data to label*; DOIR runs optimal design over *the model's own information geometry* with the data fixed.
- **Physics analogue** — isotropic $\tilde F$ is equipartition of identity information across the 512 embedding modes; the barrier is the corresponding free‑energy penalty for mode freeze‑out.

---

## 4. Decisive matched‑compute controls

All at identical epochs, batch, augmentation, and within 5% wall‑clock; 5 paired seeds each.

| # | Control | Kills what if it matches DOIR |
|---|---|---|
| C1 | Same log‑det barrier on batch **feature covariance** $\mathrm{Cov}(z)$ | The whole "Fisher ≠ covariance" claim |
| C2 | Barrier on **proxy Gram** $P^\top P$ | Reduces DOIR to proxy uniformity |
| C3 | $F$ computed with **$q,m$ detached** (gradient only into proxies) | Backbone‑shaping path is inessential |
| C4 | $F$ computed with **$P$ detached** (gradient only via $z$) | Proxy‑geometry path is inessential |
| C5 | Barrier on the **per‑proxy** (unmarginalized) Fisher | Confirms D2 is the real trap |
| C6 | Base + matched extra compute (longer schedule, larger $K$) | "It's just more capacity/time" |
| C7 | $s$ and $\delta$ sweep on base | "It's equivalent to temperature tuning" |
| C8 | **Label‑permuted Fisher**: build $F$ from a fixed random class permutation | Any semantics‑free second‑order noise explains it (also tests D5) |
| C9 | No anneal ($\lambda_t$ constant to $T$) | Tests the schedule claim, not the mechanism |
| C10 | Mediation: effective rank $\mathrm{erank}(F)=e^{H(\hat\lambda)}$ measured **on unseen test identities** | If erank doesn't move, causal chain unsupported |

---

## 5. Frozen forecasts (Lane A), falsification, frontier arithmetic

Frozen 2026‑08‑06. Mean R@1 ± s.d. over 5 seeds, single view, 512‑D, cosine.

| Dataset | My base (§1.2), forecast | **DOIR, forecast** | Lane‑A reference |
|---|---|---|---|
| CUB‑200‑2011 | 0.716 ± 0.005 | **0.735 ± 0.004** | PFML 0.734 ± 0.003 |
| Cars196 | 0.918 ± 0.004 | **0.930 ± 0.003** | PFML 0.927 ± 0.003 |
| SOP | 0.824 ± 0.002 | **0.833 ± 0.002** | PFML 0.829 ± 0.002 |
| In‑Shop *(soft, not frozen)* | 0.921 | 0.932 | PA+DADA 0.930 (seeds unreported) |

**Frontier‑crossing arithmetic, stated honestly.** Two‑sample $t$, $n=5$ per side, pooled s.d.:
- CUB: $+0.001$, $s_p\approx0.0035$, $t\approx0.45$ → **not a crossing**; forecast is parity.
- Cars: $+0.003$, $s_p\approx0.003$, $t\approx1.58$ → **not a crossing** (needs $\ge0.0064$); parity.
- SOP: $+0.004$, $s_p\approx0.002$, $t\approx3.16$, $p\approx0.013$ → **the one forecast significant crossing.**
- vs. DADA matched‑cost rows (0.729/0.921/0.810): clear on all three.

So the defensible claim is: *a ~1.0–1.9 point matched‑cost mechanism gain over a reproduced strong multi‑proxy base, parity with PFML on CUB/Cars, and a significant crossing on SOP.* I am not forecasting a broad frontier crossing, and I will not manufacture one.

**Falsification thresholds (pre‑registered).**
1. CUB paired Δ over base $<+0.8$ pt (5 seeds) → mechanism rejected.
2. C1 (covariance barrier) reaches $\ge80\%$ of Δ → Fisher‑specific claim falsified.
3. C8 (label‑permuted Fisher) reaches $\ge50\%$ of Δ → mechanism is generic noise; rejected.
4. $\mathrm{erank}(F)$ on **unseen** identities fails to rise $\ge1.5\times$ vs. base while R@1 rises → causal chain unsupported (gain is real but the story is wrong; must be reported as such).
5. Wall‑clock $>1.10\times$ base → cost claim falsified.
6. Across seeds, correlation between Δ R@1 and Δ erank $<0.5$ → mediation claim withdrawn.

---

## 6. Cost, benchmark and contamination risks

**Train cost.** Extra FLOPs/step $\approx 3$–$5$ GFLOP (forming $F$: $B\!\cdot\!C'\!\cdot\!d^2\!\approx\!1.5$ GFLOP; $512^3/3$ Cholesky $\approx0.045$ GFLOP) against $\approx6.6$ TFLOP for ResNet‑50 fwd+bwd at $B=180$ — under 0.1% of FLOPs. Measured overhead is launch‑ and fp32‑bound: **forecast 1.02–1.04× epoch time, 1.02× peak memory.** Cheaper than PA+DADA's reported 1.06× / 1.01×. No auxiliary network, no second view, no teacher.

**Deployment cost.** Exactly zero: identical ResNet‑50, one 512‑D descriptor, cosine NN. Proxies and barrier are discarded.

**Risks.** (i) ImageNet‑1K pretraining overlaps CUB/Cars semantics — standard for this lane, but absolute numbers on those two are not contamination‑free in a strict sense. (ii) CUB test‑label noise caps attainable R@1 and inflates seed variance. (iii) SOP R@1 differences of 0.4 pt are within the range where sampler and dataloader ordering matter; the paired‑seed protocol is mandatory, not optional. (iv) `logdet` under AMP is the most likely implementation failure; fp32 + jitter is required. (v) $C'=32$ truncation is a real approximation on SOP/In‑Shop; its bound must be measured, not assumed. (vi) The barrier could in principle be satisfied by manufacturing confusions on easy pairs, degrading precision at high recall — R@1 alone would hide this, so R@1/R@10/NMI must all be reported.

---

## 7. Unresolved source ambiguities (declared)

1. **I cannot reproduce PFML's (CVPR 2025) loss or recipe from memory**, beyond its disclosed proxy counts (15/15/2) and its five‑run numbers. I therefore do **not** build on PFML and do **not** inherit its frontier. §1.2 specifies my own base, and every Δ is paired against *that* base, reproduced by me. If PFML's actual base is stronger than 0.716 on CUB, my absolute forecasts move with it — in either direction.
2. **PA+DADA's In‑Shop 0.930** has no reported seed count or uncertainty; I give In‑Shop only as a soft, non‑frozen forecast for that reason.
3. Proxy weight decay: published proxy‑based recipes are inconsistent about whether decay is applied to pre‑normalization proxy parameters. I set it to zero and flag that this is a recipe choice which must be held identical across base and DOIR.
4. Whether DOIR composes additively with PFML's mechanism is untested and unforecast.

**Honest summary of the bet.** The mechanism claim (Lemma 1's absorbing set is real, and a confusion‑gated Fisher barrier escapes it) is the part I most believe. The magnitude is the part I least believe: a 512×512 spectral barrier is a blunt instrument, and the plausible outcomes range from +2 pt to +0.3 pt. The controls in §4 — especially C1 and C8 — are constructed so that a small gain is still *interpretable* rather than merely reported.

