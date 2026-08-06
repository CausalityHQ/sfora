2})=3.24$.

Hence $\mathcal{L}_{\mathrm{PA}}(S^U)\ \ge\ L_{\text{floor}}:=\log(1+e^{\alpha\delta})=3.24$ for $\alpha=32,\delta=0.1$. $\square$

**Contrapositive (the certificate):** $\max_{U\in\mathrm{St}(d,k)}\mathcal{L}_{\mathrm{PA}}(S^U)<3.24\ \Longrightarrow\ \dim\mathrm{span}\{p_c-p_{c'}\}\ \ge k+1$.

**Honesty about its status.** The certificate is *exact* only for the exact inner max. With $R$-step ascent it is a lower-bound-based heuristic, and I pre-register the fresh-Haar audit (F6) as the check that separates certification from adversary-evasion. $L_{\text{floor}}$ is computable exactly at run time from the batch class histogram and I pre-register logging $\max_m \mathcal{L}_{\mathrm{PA}}(S^{U_m})$ against it every epoch. **Prediction: the certificate holds for $k\le 32$ on CUB and fails for $k\ge 96$** — which is the mechanistic source of the inverted-U in $k$ (F8).

### 2.4 Proposition 3 (scale invariance) — kills the norm/temperature escape

$S^U_{ic}$ is invariant under $z_i\mapsto\beta z_i$ and $p_c\mapsto\gamma p_c$ for $\beta,\gamma>0$ (numerator scales $\beta\gamma$, $n_i$ scales $\beta$, $m_c$ scales $\gamma$). So neither term can be reduced by inflating feature or proxy norms, and no learned temperature is introduced. $\square$

**I do not claim this normalization is harmless.** AdamW's decoupled weight decay on $W$ and $P$ *is* operational: it changes the effective *angular* step size (update norm ÷ parameter norm), and PA additionally multiplies the proxy lr by 100. AXE does not alter this coupling structurally (same optimizer, same decayed parameter set, zero new parameters), but a residual confound remains if AXE shifts the $\|W\|_F$ / $\overline{\|p_c\|}$ trajectories. **Pre-registered:** log both per epoch on both arms, and require the AXE gain to survive at the *baseline's own best* weight decay from the $\{0,10^{-4},5\times10^{-4}\}$ sweep run identically on both arms.

### 2.5 Proposition 4 (why both $z$ and $p$ are projected) — kills the mean-direction attack

Under a **query-only** ablation ($\tilde e=\Pi_U z/\|\Pi_U z\|$ against *full-dimensional* proxies), $\langle\tilde e,\hat p_c\rangle=\langle z,\Pi_U\hat p_c\rangle/\|\Pi_U z\|$, so an adversary aligned with the mean proxy direction $\mu$ shrinks **all** logits roughly uniformly. That inflates the positive term ($e^{-\alpha(s-\delta)}$ blows up) without deleting any *discriminative* information, and the model's cheapest response is merely to decorrelate proxies from $\mu$ — not to build redundancy.

With the subspace-restricted cosine of §1.2, deleting $\mathrm{span}(\mu)$ removes a component contributing **equally to every logit**, i.e. a strictly non-discriminative direction; the surviving cosines are the renormalized purely discriminative parts. The adversary gains nothing generic and is forced onto genuinely class-carrying directions. The ablated quantity is also *deployment-faithful*: it answers "would this exact retrieval system still work if deployed in the surviving subspace?" I pre-register the query-only variant as control **C10** to demonstrate the difference empirically rather than assert it. $\square$

### 2.6 Remaining degeneracies, honestly ranked

* **G5 — redundant memorization (the serious one).** AXE's constraint could in principle be met by a high-rank per-class *code* that the encoder memorises rather than by computing more attributes. Two defences, neither a proof: (i) the implicit bias — deleting a subspace creates no information, so the encoder must *compute* more independent functions of the image, and among robust solutions SGD + weight decay + ImageNet init + crop/flip augmentation favour low-complexity functions, i.e. genuine attributes; (ii) the pre-registered diagnostic — the 4-fold class-disjoint validation R@1 and the train-class-vs-val-class R@1 gap. If AXE raises the gap, it is memorising. **Probability this is what happens: ≈0.20.** Mitigation if triggered: cap $k_{\max}$ and lengthen the $\lambda$ ramp; do not add a second mechanism.
* **G4 — adversary evasion.** The model pushes signal out of the *current* $U_m$ rather than becoming robust. Countered by warm-started persistence (the adversary tracks the model), $M=3$ frames, and periodic Haar restarts. **Detectable, not assumed:** F6 audits 64 fresh Haar draws at end of training.
* **G6 — "spread the noise."** Adding many within-class-varying directions does not reduce the adversarial term (the adversary deletes *signal*, and deleting noise lowers its objective) while it raises the clean term. Disfavoured on both terms simultaneously.

---

## 3. Adversarial novelty search, with one-sentence mechanism distinctions

**Inside DML**

| Work | Distinction |
|---|---|
| **BIER / A-BIER** (ICCV'17, TPAMI'20) | Splits the embedding into a *fixed axis-aligned* boosted ensemble with a decorrelation penalty; AXE has no partition and no boosting and is invariant to the full orthogonal group, so it cannot be satisfied by copying one feature into several coordinate blocks — exactly BIER's escape hatch. |
| **ABE** (ECCV'18) | Multiple attention heads with a divergence loss, deployed as a concatenation; AXE adds no head, no parameter, and constrains the single deployed descriptor. |
| **DREML** | Ensemble of networks over label-space partitions; AXE is one network constraining one descriptor's geometry. |
| **DiVA / MIC / S2SD** (Roth et al.) | Add self-supervised, intra-class or higher-dimensional-teacher branches and distil; AXE adds no branch, teacher, or auxiliary supervision — the pressure is a worst-case restriction of the *same* loss. |
| **PFML** (potential-field DML) | Replaces the pairwise/proxy interaction law with a continuous potential field; AXE leaves the interaction law untouched and is composable with it. |
| **DADA** (AAAI'24) | Frames the sample/proxy mismatch as a domain gap closed by data-augmented domain adaptation; AXE makes no distributional-alignment claim and constrains the *rank* at which class information is carried. |
| **Non-Isotropy Regularization** (CVPR'22) | Normalizing-flow model of proxy neighbourhoods pushed to be non-isotropic; AXE fits no density and targets worst-case subspace survival, not local isotropy. |
| **SEC** (spherical embedding constraint) | Regularizes pre-normalization embedding *norm*; AXE's terms are exactly norm-invariant (Prop. 3) and cannot be reduced to a norm constraint. |
| **OLÉ** (orthogonal low-rank embedding) | Explicitly drives each class's feature subspace to be **low**-rank and mutually orthogonal — the opposite sign of AXE's high-effective-rank between-class requirement. |
| **Metrix / Embedding Expansion / Proxy Synthesis / HDML** | Synthesize new *points*; AXE synthesizes nothing and instead deletes *directions*. |
| **Adversarially robust DML** | Input-space $\ell_p$ robustness; AXE never perturbs a pixel and its adversary lives in the descriptor's coordinate-free geometry. |

**Outside DML**

| Work | Distinction |
|---|---|
| **Dropout / DropBlock / Batch DropBlock (ReID) / Random Erasing** | Random, axis-aligned or spatial masks; AXE's mask is a worst-case *rotation-free* subspace, which is exactly what makes duplication a non-defence. |
| **Adversarial Dropout (AAAI'18), ADR (ICLR'18), DTA (ICCV'19)** | Adversarially chosen *binary unit masks* to smooth decision boundaries or enforce the cluster assumption under domain shift; AXE optimizes a continuous orthonormal frame on $\mathrm{St}(d,k)$ under a rank budget with a retrieval-margin certificate, targeting class *extrapolation* rather than boundary geometry or domain alignment. |
| **Adversarial Erasing / ACoL / Hide-and-Seek / ADL (WSOL)** | Erase the most discriminative *spatial region* so attention spreads over the object; AXE erases a *representational subspace* — the object actually deployed — which a feature duplicated across locations cannot escape. |
| **Matryoshka Representation Learning** | Requires every nested *axis-aligned prefix* to be predictive, certifying one fixed flag of subspaces; AXE requires every co-dimension-$k$ subspace to be predictive, an $O(d)$-invariant condition that MRL does not imply (MRL is fully compatible with all class information living in the first 8 coordinates, which AXE forbids at $k\ge 8$). |
| **Sub-AT** (CVPR'22, subspace adversarial training) | Restricts the *input perturbation* to a low-dimensional subspace to stabilise fast AT; AXE has no input perturbation and the subspace is the adversarially chosen object, not a constraint on an existing attack. |
| **INLP / LEACE (concept erasure)** | Find a subspace whose removal *destroys* a concept, in order to erase it; AXE uses the same primitive with the outer optimization reversed — train until no such subspace exists. |
| **Morcos et al., ICLR'18** ("importance of single directions") | Uses unit/direction ablation as a *diagnostic* correlate of generalization; AXE converts a rank-$k$, adversarially chosen, rotation-invariant version of that diagnostic into a training objective with a margin certificate. |
| **Gradient Starvation / Spectral Decoupling** (NeurIPS'21) | Same diagnosed disease, treated by an L2 penalty on *logit magnitude* that decouples the learning dynamics; AXE makes few-feature solutions **infeasible** rather than re-weighting their dynamics, and is exactly scale-invariant where SD's cure is intrinsically a magnitude penalty. |
| **Rich Feature Construction / DivDis / Diversify-and-Disambiguate** | Train *multiple* heads/models to be diverse, then select; AXE produces one head whose single descriptor is internally redundant, with no selection step. |
| **MCR²** (maximal coding rate reduction) | Volumetric log-det rate objective on feature covariance; AXE has no log-det and targets the worst-case *retrieval margin* under deletion — the two disagree exactly where it matters, since a flat covariance spectrum can be manufactured by class-irrelevant variance that AXE's adversary ignores (deleting noise directions lowers its objective). |
| **Group testing / RIP / distributed population codes (lesion robustness)** | Conceptual ancestry only; the incoherence/redundancy requirement is imposed analytically or measured post hoc, never as a differentiable train-time min-max over the deployed descriptor. |

**Residual novelty risk, stated plainly.** My searches surfaced no work combining (a) an adversarially optimized orthonormal subspace deletion, (b) applied to a *deployed retrieval descriptor* with a rank budget, (c) formulated as a subspace-restricted cosine with *both* query and proxy projected, (d) with a margin-based rank certificate. Any one of (a)–(c) exists somewhere in isolation. I searched English-language venues via web search only, did not read every DML paper from 2017–2026, and could not access the PA PDF directly (403). The nearest-miss I most expect a reviewer to find is an embedding-space adversarial-dropout variant in re-ID or WSOL; **Prop. 1 is the distinction I would defend it on**, and control C2 is the experiment that decides it.

---

## 4. Matched-compute controls

All controls: identical backbone, init, data pipeline, epochs (200), optimizer, seeds. AXE adds **zero parameters** and ≈1–3% wall-clock, so parameter- and compute-matching is essentially automatic; C8 makes it exact.

| # | Control | What it isolates | Predicted CUB R@1 |
|---|---|---|---|
| **C0** | PA baseline ($\lambda=0$) | — | 0.701 |
| **C1** | Random excision: $U\sim$ Haar resampled each step, same $k,\lambda$ | **adversariality** | 0.706 |
| **C2** | Axis-aligned adversarial excision: top-$k$ coordinates by loss gradient | **rotation-invariance** (= embedding adversarial dropout) | 0.712 |
| **C3** | Plain dropout $p=k/d$ on $z$ pre-normalization | cheapest occupied alternative | 0.703 |
| **C4** | BIER-style fixed axis-aligned block partition, PA loss per block | redundancy-by-construction | 0.708 |
| **C5** | Direct spectral flattening: $-\frac{\beta}{d}\sum_i\log(\lambda_i(\Sigma_b)+\epsilon)$ on proxy between-class scatter, $\beta$ tuned | **is the min-max necessary, or does the certificate's signature suffice?** | 0.713 |
| **C5b** | Hyperspherical proxy uniformity $\sum_{c\ne c'}e^{-t\|\hat p_c-\hat p_{c'}\|^2}$ | occupied spread regularizer | 0.705 |
| **C6** | Single Haar $U$ frozen for the whole run | adaptivity | 0.702 |
| **C7** | Fresh Haar init each batch, $R$ steps (no persistence) | warm-starting | 0.708 |
| **C8** | PA at 200 epochs +3% steps, and PA at equal wall-clock | compute-matching | 0.701 |
| **C9** | Excision term with proxies replaced by fixed random unit vectors + shuffled labels | **is AXE just noise regularization?** | 0.700 |
| **C10** | Query-only excision (full-D proxies) | validates Prop. 4 | 0.705 |
| **C11** | $k$-sweep $\{0,16,32,48,64,96,128\}$ | inverted-U prediction | peak at 32–48 |
| **C12** | **AXE on CUB with training classes subsampled to $C\in\{25,50,100\}$ at fixed image budget** | **the cleanest mechanism test**: does the gain shrink as $C$ grows, *within one dataset*, removing the CUB↔SOP confound | gain 25 > 50 > 100 |
| — | AXE | — | **0.720** |

Predicted ordering: **AXE > C2 ≳ C5 > C4 ≈ C7 ≈ C1 > C3 ≈ C10 ≈ C5b > C0 ≈ C6 ≈ C8 ≈ C9.**

**C5 and C12 are the decisive ones.** If C5 matches AXE, the whole min-max apparatus is unnecessary and should be replaced by the two-line spectral penalty. If C12's gain does not fall with $C$, the causal account in §2.1 is wrong and AXE is a generic regularizer that happens to work.

---

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic

**Frozen 2026-08-06, Lane A, ResNet-50 / 512-D / 224² / cosine / 200 epochs, 5 seeds, mean ± sd.**

### 5.1 Primary configuration (PA base, $K=1$ proxy per class)

| Dataset | My matched PA baseline (C0) | AXE | Δ | 90% CI on Δ |
|---|---|---|---|---|
| **CUB-200-2011** | 0.701 ± 0.004 | **0.720 ± 0.005** | **+1.9** | [+0.9, +3.0] |
| **Cars196** | 0.881 ± 0.005 | **0.897 ± 0.005** | **+1.6** | [+0.5, +2.8] |
| **SOP** | 0.800 ± 0.002 | **0.806 ± 0.003** | **+0.6** | [+0.1, +1.2] |
| **In-Shop** | 0.919 ± 0.003 | **0.927 ± 0.003** | **+0.8** | [+0.2, +1.5] |

The **asymmetry** (CUB/Cars ≫ SOP/In-Shop) is not a hedge — it is the mechanism's central prediction. With $C\gg d$ (SOP: 11,318; In-Shop: 3,997) the loss already forces the class signal to near-full rank and the minimal-rank degeneracy is largely absent; with $C\approx 100$ it is severe.

### 5.2 Frontier attempt (multi-proxy base: SoftTriple smooth-max, $K=15$ CUB/Cars, $K=2$ SOP/In-Shop, $\tau=0.2$)

| Dataset | Multi-proxy PA base | + AXE | Reference | Decisive-crossing threshold | Verdict |
|---|---|---|---|---|---|
| CUB | 0.716 ± 0.005 | **0.736 ± 0.006** | PFML 0.734 ± 0.003 | **0.740** | nominal cross, **not decisive** |
| Cars | 0.898 ± 0.005 | **0.913 ± 0.006** | PFML 0.927 ± 0.003 | 0.933 | **no cross**, shortfall 1.4 |
| SOP | 0.812 ± 0.003 | **0.818 ± 0.003** | PFML 0.829 ± 0.002 | 0.832 | **no cross**, shortfall 1.1 |
| In-Shop | 0.923 ± 0.003 | **0.930 ± 0.004** | PA+DADA 0.930 (no σ) | **0.934** | tie, **not a crossing** |

**Crossing arithmetic.** 5 seeds, $t_{4,0.95}=2.132$. CUB: SEM $=0.006/\sqrt5=0.00268$; to reject $H_0:\mu\le 0.734$ one-sided at $\alpha=0.05$ requires mean $\ge 0.734+2.132(0.00268)=\mathbf{0.7397}$. In-Shop: SEM $=0.00179$; requires mean $\ge 0.930+0.0038=\mathbf{0.9338}$.

$P(\text{decisive crossing})$: CUB **0.25**, In-Shop **0.15**, SOP **0.05**, Cars **0.05**. $P(\text{nominal crossing on CUB})\approx 0.55$.

**Honest headline: AXE as specified is forecast to be a large, tightly-controlled gain over a matched baseline, and is NOT forecast to decisively cross the Lane-A frontier on any of the four datasets.** The prompt asks for the strongest defensible method with uncertainties stated plainly, not for a method whose forecast crosses. I decline to inflate the numbers to manufacture a crossing.

**Conditional (unverified) composition arithmetic.** AXE constrains the *rank* at which class information is carried; PFML replaces the *interaction law*. These are orthogonal, and PFML's potential field has no obvious rank-raising effect. If AXE transfers to a PFML base the same Δ it transfers to a PA base, CUB would reach $0.734+0.014=0.748$ and Cars $0.927+0.015=0.942$. **I flag this as arithmetic, not a forecast**: I have not reproduced PFML, its recipe is not disclosed to me, and Δ's transfer across bases is exactly the kind of assumption that fails.

**Multi-proxy caveat that weakens the frontier attempt.** With $K=15$ on CUB the proxy set has $1500$ vectors in $\mathbb{R}^{512}$, so $\mathrm{span}\{p_c-p_{c'}\}$ is trivially full-rank and Prop. 2's certificate becomes vacuous *for the proxies*. Under $K>1$ the certificate must be restated on the class-level **feature** mean configuration, which is a genuine loosening. **$K=1$ is the scientifically clean configuration; $K>1$ is the frontier attempt and should be reported as such.**

### 5.3 Falsification thresholds (pre-registered)

*Method-killing:*
* **F1.** AXE − C0 on CUB $\ge$ **+0.8** pts, one-sided Welch, 5+5 seeds, $\alpha=0.05$. Below ⇒ falsified as an improvement.
* **F2.** AXE − C1 (random excision) $\ge$ **+0.6** on CUB. Below ⇒ adversariality falsified; AXE is structured dropout.
* **F4.** AXE − C5 (spectral flattening) $\ge$ **+0.4** on CUB. Below ⇒ the min-max is unnecessary; publish the two-line penalty instead.

*Mechanism-killing (an empirical gain may survive, but the claimed causal story must then be retracted, not softened):*
* **F3.** AXE − C2 (axis-aligned adversary) $\ge$ **+0.4** on CUB. Below ⇒ rotation-invariance claim falsified.
* **F5.** $(\text{AXE}-\text{C0})_{\text{CUB}} - (\text{AXE}-\text{C0})_{\text{SOP}} \ge$ **+0.7**. Below or reversed ⇒ the class-count-limited-rank account is falsified.
* **F5b.** In C12, gain at $C{=}25$ exceeds gain at $C{=}100$ by $\ge$ **0.5** pts.
* **F6 (certificate audit).** At end of training, $\max$ over **64 fresh Haar** $U$ of $\mathcal{L}_{\mathrm{PA}}(S^U)\le 1.25\times\max_m\mathcal{L}_{\mathrm{PA}}(S^{U_m})$, **and** both $<L_{\text{floor}}=3.24$ at $k=32$. Violation ⇒ the model is evading a moving adversary, not becoming robust; the certificate is void.
* **F7.** Participation ratio $\mathrm{PR}(\Sigma_b)=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ of the class-mean feature scatter, measured on held-out training-class folds, must rise $\ge$ **30%** vs C0. No rise despite an R@1 gain ⇒ report as an unexplained confound, not a success.
* **F8.** R@1 vs $k$ on CUB must be non-monotone with an interior maximum in $\{16,\dots,128\}$. Monotone increasing ⇒ the rank-budget account is wrong.
* **F9 (protocol hygiene).** Spearman $\rho\ge 0.7$ between held-out-training-class val R@1 and test R@1 across the $k$-sweep. If not, the selection protocol is unsound and every selected hyperparameter is suspect.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** Zero added parameters. Adversary state $M\!\cdot\!d\!\cdot\!k = 3\cdot512\cdot48 \approx 74$K floats (0.3 MB). Per optimizer step the adversary costs $M R\big(2dk(B{+}C)\big)\approx 65$ MFLOP on CUB against ResNet-50 fwd+bwd at $120\times 4.1\,\text{GFLOP}\times 3\approx 1.48$ TFLOP — under 0.01% in FLOPs; wall-clock overhead is dominated by kernel-launch latency on small ops. **Forecast 1.01–1.03× epoch time, ≈1.00× peak memory.** On SOP ($C=11{,}318$) the proxy projection dominates at ≈3.4 GFLOP/step, still ≈0.2%; **forecast ≤1.05×**, with the 4096-class sampling fallback pre-registered.

**Deployment cost.** Identical to the PA baseline: one ResNet-50, one 224² view, one 512-D L2-normalized descriptor, cosine NN. 1.00× on every axis.

**Against the references.** AdvRF adds a training-only ResNet-34 + U-Net reconstruction system plus distillation; VAPNet adds attribute-discovery and online-refinement machinery; PA+DADA reports ≈1.06× epoch time and 1.01× memory. **AXE at ≈1.02× with zero parameters is the cheapest of the set**, which is a defensible contribution independent of frontier crossing — and I would rather report "matched In-Shop at 1.02× vs DADA's 1.06×" honestly than claim a crossing I do not forecast.

**Benchmark risk.**
* CUB/Cars test sets are small (5,924 / 8,131 queries): 1 R@1 point ≈ 59 / 81 images. With seed sd ≈ 0.005, **single-seed comparisons are meaningless**; 5 seeds and Welch tests are mandatory, not decorative. My own forecast sd of 0.006 on the CUB frontier run is larger than PFML's reported 0.003, which is itself a reason the crossing arithmetic is unfavourable.
* Standard CUB/Cars splits are class-ordered (first half train), so "unseen" classes are not a random draw from the label space. C12 partially disentangles class *count* from class *identity*, but not fully.
* SOP/In-Shop with random batch sampling give most batches ≤1 image per class, weakening PA's positive term; AXE inherits this. Pre-register reporting mean $|P^+|$ per batch and the images-per-class histogram.

**Contamination risk.**
* ImageNet-1K pretraining semantically overlaps CUB (≈59 bird classes) and Cars. This affects **every** Lane-A method identically, including PFML and PA+DADA, so relative comparisons are sound; absolute numbers carry the standard caveat.
* No test data, external data, generated data, text/VLM encoder, extra annotation, transduction, reranking, or gallery fitting is used anywhere. The adversary is computed solely from training batches and training proxies and is destroyed before evaluation. Hyperparameter selection uses only class-disjoint folds of the official training classes.
* **Reproduction risk.** I have reproduced neither PFML nor PA+DADA. Every claim in §5.1 is against my own matched PA reproduction; §5.2's references are quoted external numbers, one of which (PA+DADA 0.930) has no disclosed uncertainty or seed count, so even a nominal crossing there would be weak evidence.

---

## Where I expect this to fail

Ranked, with my honest priors: **(1)** AXE collapses to structured dropout, C1 ≈ AXE — 0.30. **(2)** $k_{\max}=48$ is too aggressive at $C=100$ and the inverted-U peaks at $k=16$ with a ≈+1.0 gain instead of +1.9 — 0.25. **(3)** Redundant memorization (G5) shows up as a widened train-class/val-class gap — 0.20. **(4)** Adversary evasion trips F6 and the certificate claim must be withdrawn even if R@1 improves — 0.20. **(5)** The 200-epoch schedule change swamps the effect relative to PA's 40/60-epoch regime — 0.15.

The single experiment I would run first is **C12** (class-count sweep within CUB), because it tests the causal story at ~1/4 the cost of the full grid and, if the gain does not fall with $C$, everything in §2.1 is wrong regardless of what the R@1 numbers do.

**Sources:** [Proxy Anchor (CVPR 2020)](https://arxiv.org/pdf/2003.13911) · [official PA repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) · [PFML review](https://liner.com/review/potential-field-based-deep-metric-learning) · [DADA (AAAI 2024)](https://arxiv.org/pdf/2401.00617) · [BIER (TPAMI 2020)](https://arxiv.org/abs/1801.04815) · [Gradient Starvation (NeurIPS 2021)](https://arxiv.org/abs/2011.09468) · [Morcos et al. (ICLR 2018)](https://arxiv.org/abs/1803.06959) · [Adversarial Dropout Regularization (ICLR 2018)](https://arxiv.org/abs/1711.01575) · [AdvRF (ICCV 2025)](https://arxiv.org/abs/2507.21742) · [VAPNet (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf) · [Non-Isotropy Regularization (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf) · [OLÉ](https://arxiv.org/pdf/1712.01727) · [Matryoshka Representation Learning](https://aniketrege.github.io/blog/2024/mrl/)
