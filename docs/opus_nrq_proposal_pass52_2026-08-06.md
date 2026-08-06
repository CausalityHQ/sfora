## Disclosure of process

I worked blind as instructed: no filesystem, git, repo, shell, consultation, or subagent tools were used. Only `WebSearch`/`WebFetch`. Everything below is derived from the prompt plus public sources retrieved in this session. Where I could not verify a number from a primary source in this session, I say so rather than assert it.

**Lane choice: Lane A** (ResNet-50, 512-D, ~224px, single-view cosine, 200 epochs). All forecasts and comparisons below are Lane A only. Lane B (AdvRF, VAPNet) and the MiT-B2 CRT row appear only in the novelty-distinction section.

---

# NRQ — Nuisance-Register Quotient Metric Learning

**One-line statement.** Supervised DML attains augmentation invariance by *class lookup* rather than by any input-general nuisance-removal computation; NRQ makes that solution *infeasible* with a rank floor tied to the number of training identities, forces the augmentation-driven component of the representation into a small linear register, and deletes that register at deployment — so the invariance that survives is realized by a fixed linear quotient that applies identically to unseen identities.

---

## 0. Baseline reduction and recipe, reproduced where disclosed

The Lane-A frontier reference is **PFML = Potential Field Based Deep Metric Learning** (Bhatnagar et al., CVPR 2025; arXiv:2405.18560). Retrieved primary-source content:

Attractive and repulsive potentials, for source $r$ and embedding $z_i$:

$$
\psi_{\text{att}}(r,z_i)=\begin{cases}-\delta^{-\alpha} & \|r-z_i\|_2<\delta\\[2pt] -\|r-z_i\|_2^{-\alpha} & \text{else}\end{cases}
\qquad
\psi_{\text{rep}}(r,z_i)=\begin{cases}\|r-z_i\|_2^{-\alpha} & \|r-z_i\|_2<\delta\\[2pt] \delta^{-\alpha} & \text{else}\end{cases}
$$

Total potential energy $\;\mathcal U=\sum_i \Psi_{y_i}(z_i)+\sum_j\sum_k \Psi_j(p_{j,k})$, superposing same-class attraction and different-class repulsion over embeddings and proxies, minimized jointly over embeddings and proxies.

Disclosed recipe: $\alpha$ cross-validated in $\{0,\dots,6\}$; $\delta$ cross-validated in $[0.1,0.3]$; $M=15$ proxies/class on CUB-200-2011 and Cars196, $M=2$ on SOP; Adam, base lr $5\times10^{-4}$, **proxy lr $\times 100$**; 200 epochs; 512-D $\ell_2$-normalized output; $224\times224$ (center crop from 256 at test); ResNet-50 ImageNet-pretrained. Reported R@1 over 5 runs: CUB $73.4\pm0.3$, Cars $92.7\pm0.3$, SOP $82.9\pm0.2$ — consistent with the prompt's $0.734\pm0.003$ / $0.927\pm0.003$ / $0.829\pm0.002$.

**Unresolved source ambiguities (must be recorded, not papered over):**

1. The exact superposition/normalization of $\Psi$ per anchor (whether the sum over sources is count-normalized, and whether any log-sum-exp/softmax wrapper is applied) is not in the material I could retrieve.
2. The **augmentation pipeline is not disclosed** in the retrieved pages.
3. **Batch size, sampler (random vs. $m$-per-class), weight decay, LR schedule, warm-up, BN freezing** are not disclosed in what I retrieved.
4. Whether $\pm0.3$ is a standard deviation or a standard error is not stated.
5. Whether the cross-validation for $(\alpha,\delta)$ is carved from **training identities** or touches test identities is not stated. If the latter, the reference itself is compromised and no frontier arithmetic against it is meaningful.

**Consequence I bind myself to:** I do **not** inherit 0.734 / 0.927 / 0.829. Every NRQ forecast is stated against *my own reproduction* of PFML under a fully specified recipe. A frontier claim is admissible only if that reproduction lands within $\pm0.005$ of each published mean; otherwise only the paired delta over my reproduction is claimable. I additionally specify a second base with a fully public recipe (Proxy-Anchor, Kim et al., CVPR 2020) so the mechanism can be tested free of PFML ambiguity.

Because $\delta$ is an **absolute radius** on the normalized sphere and Adam + weight decay act on the head, the scale of the pre-normalization head output is operational. I therefore never rescale or renormalize $z$, I define every NRQ term on scale-free quantities, and I include an explicit scale control (C9).

---

## 1. Executable mathematics

### 1.1 Objects

* Backbone $g_\phi$: ResNet-50, ImageNet-1K pretrained; GAP output $h(x)=g_\phi(x)\in\mathbb R^{2048}$.
* **Frozen teacher** $g_{\phi_0}$: a byte-identical copy of the initialization, `eval()` mode BN, `no_grad`. $h_0(x)\in\mathbb R^{2048}$. This is the permitted ImageNet-1K initialization, not an extra model or extra data.
* **Content head** $W_z\in\mathbb R^{512\times2048},b_z$: $z=W_zh+b_z$; deployed descriptor $\hat z=z/\|z\|_2\in S^{511}$. *Identical in form to the base head — unchanged.*
* **Nuisance register** $W_n\in\mathbb R^{r\times2048},b_n$, $r=64$: $n=\mathrm{BN}_{\text{no-affine}}(W_nh+b_n)\in\mathbb R^{r}$. The non-affine BatchNorm makes every NRQ term scale-free without touching $z$'s path.
* **Sufficiency decoder** $V\in\mathbb R^{2048\times576},\,c\in\mathbb R^{2048}$, applied to $\tilde u=[\hat z;n]$.
* Proxies $\{p_{j,k}\}$ exactly as in the base.

**Discarded at test:** $W_n,b_n$, the register BN, $V,c$, $g_{\phi_0}$, all proxies. Deployment is one ResNet-50, one 224 center crop, one 512-D $\ell_2$-normalized descriptor, cosine NN. Bit-identical inference cost to the base.

### 1.2 Views and recorded parameters

Per image, draw two augmentation parameter vectors $\theta^{(1)},\theta^{(2)}\sim\mathcal A$ and record

$$\theta=(\log s,\ \log\rho,\ c_x,\ c_y,\ \mathbb 1_{\text{flip}},\ \Delta_b,\ \Delta_c,\ \Delta_{\text{sat}},\ \Delta_{\text{hue}},\ \mathbb 1_{\text{gray}})\in\mathbb R^{10},$$

each standardized to zero mean / unit variance under $\mathcal A$. $\theta^{(1)}\sim$ **the base pipeline exactly** (RandomResizedCrop + hflip); $\theta^{(2)}\sim$ base pipeline $+$ mild color jitter $+$ $p=0.2$ grayscale. Both are ordinary stochastic augmentation of official training images.

$\mathcal L_{\text{base}}$ (PFML) is computed on **view 1 only**, with the base sampler, batch size, and proxy update unchanged — so the base optimization problem is not perturbed except by the added gradients.

### 1.3 The three terms

**(a) Content invariance.**

$$\mathcal L_{\text{inv}}=\mathbb E_x\big[1-\hat z(x,\theta^{(1)})^\top\hat z(x,\theta^{(2)})\big]$$

**(b) Anchored sufficiency — the rank floor.**

$$\mathcal L_{\text{suf}}=\frac1{2}\sum_{v\in\{1,2\}}\frac{\mathbb E\big\|h_0(t_{\theta^{(v)}}x)-V\tilde u^{(v)}-c\big\|_2^2}{\mathbb E\big\|h_0(t_{\theta^{(v)}}x)-\bar h_0\big\|_2^2},\qquad \tilde u^{(v)}=[\hat z^{(v)};n^{(v)}]$$

$\bar h_0$ is an EMA (momentum 0.99) of the teacher mean. $V,c$ use the same optimizer at $10\times$ base lr. Crucially the teacher target is the feature of the **augmented** view, so the view-dependent component of $h_0$ must be explained by something in $\tilde u$ — and $\mathcal L_{\text{inv}}$ forbids that something from being $\hat z$. The register is the only remaining route. *That is the entire routing mechanism.*

**(c) Register parsimony (identity must not leak into the register).** With $\bar n_{k,c}$ the batch mean of coordinate $k$ over samples of class $c$ (register coordinates have unit batch variance by the BN):

$$\mathcal L_{\text{reg}}=\frac1r\sum_{k=1}^{r}\operatorname{Var}_{c\in B}\big(\bar n_{k,c}\big)$$

Requires $\ge 2$ samples per class in the batch; use the standard $m{=}4$-per-class DML sampler. On SOP with 11318 classes this term is near-vacuous under random sampling — stated, not hidden.

### 1.4 Objective, hyperparameters, schedules

$$\boxed{\ \mathcal L=\mathcal L_{\text{base}}^{(v=1)}+\lambda_{\text{inv}}\mathcal L_{\text{inv}}+\lambda_{\text{suf}}(e)\,\mathcal L_{\text{suf}}+\lambda_{\text{reg}}\mathcal L_{\text{reg}}\ }$$

| Hyperparameter | Value | Chosen how |
|---|---|---|
| $r$ | 64 | **derived** in §2.2, not tuned |
| $\lambda_{\text{inv}}$ | 1.0 | pseudo-zero-shot val on CUB |
| $\lambda_{\text{suf}}$ | 0.5, warm-up $\min(1,e/5)$ | calibrated to hit $\epsilon^\star$ (§2.2) |
| $\lambda_{\text{reg}}$ | 0.05 | pseudo-zero-shot val on CUB |
| decoder lr | $10\times$ base | fixed a priori |
| teacher | frozen init, fp16, `no_grad` | — |

**Selection protocol:** all four $\lambda$'s and $r$ are fixed **once on CUB** using the last 20 *training* classes as a held-out pseudo-zero-shot validation split, then **transferred unchanged** to Cars196, SOP, In-Shop. No test identity is ever used for selection. This is a hard commitment; per-dataset retuning would void the claim.

**Gradient paths.** $\mathcal L_{\text{base}}\!:\hat z^{(1)}\!\to\! W_z\!\to\! g_\phi$ and $\to$ proxies. $\mathcal L_{\text{inv}}\!:$ both views $\to W_z\to g_\phi$. $\mathcal L_{\text{suf}}\!:\to V,c$ and $\to\{W_z,W_n\}\to g_\phi$; **no gradient into $g_{\phi_0}$**. $\mathcal L_{\text{reg}}\!:\to W_n\to g_\phi$. Nothing flows through the teacher, and the register BN's running stats are frozen at test (irrelevant — the register is deleted).

### 1.5 Train / test operations

**Train step:** sample batch ($m{=}4$/class) → draw $\theta^{(1)},\theta^{(2)}$ → student forward both views → teacher forward both views under `no_grad` → assemble $\mathcal L$ → backward → Adam.
**Test:** resize 256, center crop 224, one forward, $\hat z=z/\|z\|$, cosine NN over the gallery. No reranking, no test-set fitting, no transduction, no second view.

---

## 2. Causal zero-shot error mode, and proof-level attack on the degeneracies

### 2.1 The error mode: **class-conditional invariance**

A discriminative DML objective on $C$ training identities is minimized by any $f$ with $f(x)\approx\mu_{y(x)}$. Such an $f$ is *perfectly augmentation-invariant on the training support* while containing **no input-general nuisance-removal computation whatsoever**: invariance is realized as "recognize the identity, emit its mean." On an unseen identity there is no mean to emit, and the network falls back on whatever generic computation it happens to retain; the unseen-class manifold then inflates along exactly the nuisance directions (viewpoint, crop, illumination) that dominate fine-grained retrieval errors.

This is not speculation about optimization: Xue et al. (ICML 2023) prove that supervised contrastive learning collapses sub-class structure and that SGD's simplicity bias is the driver, and prescribe adding an unsupervised contrastive term. Their remedy is precisely the *occupied simpler alternative* I must beat (control C7).

**Fingerprint (D1), measurable before any R@1 claim:** $\mathcal R=\dfrac{\text{mean within-identity embedding variance on test identities}}{\text{mean within-identity embedding variance on train identities}}$. Prediction: $\mathcal R>3$ for the base on CUB/Cars, $\mathcal R<2$ under NRQ. Reported as analysis only; never used for model selection.

### 2.2 Proposition 1 (rank floor) — makes collapse *infeasible*, not merely penalized

Let $\Sigma_0=\operatorname{Cov}\big(h_0(t_\theta x)\big)$ over $(x,\theta)\sim p_{\text{train}}\times\mathcal A$, eigenvalues $\lambda_1\ge\cdots\ge\lambda_{2048}$, $T=\sum_k\lambda_k$.

> For any random vector $\tilde u$ with $\rho:=\operatorname{rank}\operatorname{Cov}(\tilde u)$ and any affine $(V,c)$:
> $$\mathbb E\|h_0-V\tilde u-c\|_2^2\ \ge\ \sum_{k>\rho}\lambda_k .$$

*Proof.* $V\tilde u+c$ ranges over affine images whose covariance has rank $\le\rho$. The optimal $c$ matches means, so the MSE equals $\operatorname{tr}\Sigma_0$ minus the variance explained by a rank-$\rho$ subspace; by Eckart–Young / Ky Fan that explained variance is at most $\sum_{k\le\rho}\lambda_k$. ∎

**Corollary.** $\mathcal L_{\text{suf}}\le\epsilon$ forces $\rho\ge k^\star(\epsilon):=\min\{k:\sum_{j>k}\lambda_j\le\epsilon T\}$. Since $\rho\le\operatorname{rank}\operatorname{Cov}(\hat z)+r$,

$$\operatorname{rank}\operatorname{Cov}(\hat z)\ \ge\ k^\star(\epsilon)-r .$$

Full class collapse ($\hat z=\mu_y$ a.s.) gives $\operatorname{rank}\operatorname{Cov}(\hat z)\le C-1$. Therefore collapse is **infeasible** at sufficiency level $\epsilon$ whenever

$$\boxed{\,C-1+r\ <\ k^\star(\epsilon)\ \le\ 512+r\,}$$

**This inequality derives $r$.** With $r=64$ and CUB $C=100$ we need $163<k^\star(\epsilon)\le576$ — a real but narrow window. With $r=256$ we would need $k^\star>355$, much harder. Hence $r=64$; $r$ is not a tuned knob.

**Calibration pass (one pass over training images, before training):** compute $\Sigma_0$'s spectrum; set $\epsilon^\star=\tfrac1T\sum_{j>400}\lambda_j$; require achieved $\mathcal L_{\text{suf}}\le1.25\,\epsilon^\star$ by epoch 20, and *verify numerically* that $k^\star(1.25\epsilon^\star)>C-1+r$. Report the achieved value every run.

**I have not measured this spectrum** and will not fabricate it. ReLU GAP features are non-negative with heavy spectral tails, so I *expect* $k^\star(0.05)$ in the several hundreds — but this is an expectation, not evidence. **If $k^\star$ cannot be pushed above $C-1+r$ at any achievable $\epsilon$, the anti-collapse leg of NRQ is void** and must be replaced (e.g. concatenating res4 + res5 pooled teacher features to enrich the spectrum). That is a pre-registered failure condition, not a footnote.

**Where the bound is vacuous, and why that is a feature.** On SOP ($C=11318$) and In-Shop ($C=3997$), $C-1+r\gg 2048\ge k^\star$: the rank floor cannot bind. NRQ there reduces to quarantine + invariance only. This predicts a **strict asymmetry**: CUB/Cars gains must exceed SOP/In-Shop gains. If they do not, the stated mechanism is wrong (falsifier F4).

### 2.3 Proposition 2 (what the quotient does and does not buy) — stated with its weak link exposed

The deletion of the register is a **fixed linear map** $P:\mathbb R^{576}\to\mathbb R^{512}$ applied identically to every input. For the component of variation the trunk routes into the register's coordinates, removal is *exactly uniform over all inputs, including unseen identities, by construction* — no support assumption, no Lipschitz constant.

For the remaining component, I claim only a **solution-set selection argument**, not a certificate: $\mathcal L_{\text{inv}}$ is imposed on the training support like any other term, but Proposition 1 removes the class-lookup members from the set of minimizers, leaving mechanisms that are more plausibly input-general. Converting this into a genuine uniform bound would need a controlled Lipschitz constant for ResNet-50, which nobody has. **I am not claiming that bound.** This is the single largest theoretical gap in the proposal.

### 2.4 The three cheapest degeneracies, each priced and each with a falsifier

| Degeneracy | Why it is cheap | What blocks it | Falsifier |
|---|---|---|---|
| **Class collapse** — $\hat z\to\mu_y$; invariance free | globally optimal for $\mathcal L_{\text{base}}+\mathcal L_{\text{inv}}$ | Proposition 1: infeasible when $C-1+r<k^\star(\epsilon)$ | achieved $\mathcal L_{\text{suf}}>1.25\epsilon^\star$, or $k^\star\le C-1+r$ |
| **Content dumping** — register absorbs class-relevant detail, $\hat z$ goes coarse | reduces $\mathcal L_{\text{suf}}$ cheaply | $\mathcal L_{\text{reg}}$ + $r\ll d$ | **F5:** linear class probe on frozen $n$ > 40% top-1 (predict $\le$ 15%, vs $\ge$ 90% for $\hat z$) |
| **Teacher copying** — trunk stays at init, sufficiency trivial | zero-effort $\mathcal L_{\text{suf}}$ | $\mathcal L_{\text{base}}$ opposes it | CKA$(h,h_0)$ must land strictly between base run (low) and frozen-backbone run (=1); if $\approx1$, NRQ is L2-SP in disguise (control C8) |

**Routing verification (D3):** linear probe of $\theta$ from $\hat z$ must be at chance ($R^2<0.05$); from $n$ must be high ($R^2>0.5$). If both are low, nothing was routed and $\mathcal L_{\text{suf}}$ is being satisfied some other way — the method's description is then false regardless of R@1.

---

## 3. Adversarial novelty search — nearest works, inside and outside DML

I searched deliberately *for* prior art that would kill this, not for support. Nearest neighbours found, each with a one-sentence mechanism distinction:

**Outside DML (the dangerous ones):**

* **AugSelf** (Lee et al., NeurIPS 2021) — predicts augmentation-parameter *differences* from a shared representation so nuisance is **retained in the deployed features**; NRQ confines nuisance to a disjoint register that is **deleted** at deployment, and its anti-collapse pressure is a rank floor, not parameter prediction. *(This paper also explicitly notes the supervised case discards augmentation info — so my §2.1 observation is anticipated in spirit; my contribution there is the specific infeasibility construction, not the observation.)*
* **SIE** (Garrido, Najman, LeCun, ICML 2023) — splits SSL representations into invariant/equivariant halves with a hypernetwork predictor, evaluated on synthetic 3DIEBench equivariance tasks; NRQ is supervised, has no predictor or hypernetwork, deploys only the invariant half, and derives the split size from $C$.
* **LOOC** (Xiao et al., ICLR 2021) — learns several embedding spaces each invariant to all-but-one augmentation and uses them all downstream; NRQ deploys exactly one space and treats the other block as a discard.
* **E-SSL / Equivariant Contrastive Learning** (Dangovski et al.) — adds equivariance to a chosen transform in an SSL objective; NRQ targets a shortcut (class lookup) that does not exist in SSL at all.
* **Xue et al. (ICML 2023)** — proves class collapse in supervised contrastive learning and prescribes adding an unsupervised contrastive term; NRQ replaces that remedy with a rank floor requiring no negatives, no queue, no instance-discrimination head, and additionally removes the nuisance at test.
* **InFeR** (RL plasticity literature) — auxiliary regression to random projections of the *initial network's outputs* to hold up effective rank; NRQ regresses the frozen pretrained feature **of the augmented view** from the **deployed descriptor plus register**, so one term simultaneously sets the rank floor and defines what must be routed away.
* **L2-SP / DELTA / LP-FT** (Kumar et al., ICLR 2022 and predecessors) — constrain weights or features toward the pretrained model; NRQ constrains only *linear recoverability from a 576-D head*, permitting arbitrary rotation and rescaling of the trunk — strictly weaker, and it is the weakness that makes it usable.
* **NAP / WCCN** (Solomonoff et al. 2005; Hatch et al. 2006, speaker verification) — post-hoc linear removal of a within-class nuisance subspace estimated from training identities, known to help *unseen-speaker* verification. This is the closest thing to an existence proof outside vision, and it sharpens NRQ's actual claim: **post-hoc NAP is ineffective on a collapsed deep embedding precisely because the collapse already destroyed the subspace; the subspace must be manufactured during training to be removable.** Control C8 tests exactly this and can falsify the whole method.

**Inside DML:**

* **PFML** (CVPR 2025) — a potential-field re-derivation of proxy interactions; NRQ is orthogonal to the proxy interaction and applies on top.
* **S2SD** (Roth et al., 2020) — distills *from* auxiliary high-dimensional embedding spaces *into* the deployed one; NRQ distills nothing into $z$ and instead removes a subspace from deployment.
* **DiVA / Sharing Matters** (Milbich, Roth et al.) — aggregate complementary branches (self-supervised, shared-characteristic) into the deployed embedding; NRQ has one deployed branch and one deleted register, and argues infeasibility rather than diversity.
* **DVML** (Lin et al., ECCV 2018) — *assumes* intra-class variance is class-independent and uses a VAE to synthesize; NRQ makes class-independence an enforced structural property of a linear register rather than a generative assumption.
* **$\rho$-spectrum regularization** (Roth et al., ICML 2020) — flattens the embedding singular-value spectrum as a correlate of generalization; NRQ's rank floor is not a spectral-shape penalty but a *feasibility constraint against an external code*, and it specifies where the retained variance must go.
* **AdvRF** (ICCV 2025, Lane B) — learns "category-agnostic" discrepancy features via an adversarial U-Net reconstruction loop plus distillation; conceptually adjacent in goal, but NRQ has no generator, no pixel reconstruction, no distillation, and is 512-D Lane A.
* **PA+DADA** (AAAI 2024, In-Shop 0.930) — domain-adaptation-flavoured proxy augmentation; different mechanism, and I do not forecast against it (§5).

**Honest residual:** the *combination* "split invariant/nuisance representation + augmentation awareness" has substantial SSL prior art (AugSelf, SIE, LOOC). What I claim is new is (i) the supervised class-collapse infeasibility construction with $r$ derived from $C$, (ii) the anchored-sufficiency rank floor as the anti-collapse device, and (iii) the deployed quotient. A reviewer who reads the method as "AugSelf with a discard" is not being unreasonable; controls C4 and C6 exist specifically to force that reading to be right or wrong empirically.

---

## 4. Matched-compute controls

All at identical epochs, batch size, optimizer, sampler, and **5 shared seeds**; paired differences reported.

| | Control | What it kills |
|---|---|---|
| C0 | Base (PFML repro), 1 view | reference |
| C1 | Base + the enriched augmentation as plain extra data | "it's just more augmentation" |
| C2 | Base + $\mathcal L_{\text{inv}}$ only | "it's just augmentation invariance" — predicted **neutral-to-negative**, since this *is* the collapse shortcut |
| C3 | Base + $\mathcal L_{\text{suf}}$ only, $r=0$ | "it's just pretrained-feature retention" |
| **C4** | Base + $\mathcal L_{\text{suf}}$ + $\mathcal L_{\text{inv}}$, **$r=0$** | **decisive:** same pressures, nowhere to put the nuisance |
| C5 | NRQ, register kept at test (576-D) | isolates the quotient (off-lane; diagnostic only) |
| C6 | NRQ with $n$ shuffled across the batch inside $\mathcal L_{\text{suf}}$ | kills routing, keeps compute |
| C7 | Base + SupCon⊕SimCLR aux (Xue et al. remedy); Base + DiVA-style branches | "any anti-collapse aux loss does this" |
| C8 | Base + L2-SP; Base + $\|h-h_0\|^2$; **post-hoc NAP/WCCN on C0** | the classical occupied alternatives |
| C9 | $\mathcal L_{\text{suf}}$ on unnormalized $[z;n]$ vs normalized $[\hat z;n]$ | disguised gradient-scale / weight-decay effect |
| C10 | $r\in\{0,16,64,256\}$; $\lambda_{\text{suf}}\in\{0,0.1,0.5,2\}$ | shape of the mechanism |
| **C11** | Base at **matched wall-clock**: 540 epochs; 2× batch; 2-view-as-data | **decisive:** NRQ costs ~2.7×; a 2.7× method beating a 1× baseline is not a frontier result |
| **C12** | NRQ with a **randomly initialized frozen teacher** | **decisive:** separates "rank floor" from "ImageNet knowledge retention" |

**Predicted ordering (the mechanism claim, falsifiable as an ordering independent of any absolute number):**

$$\text{NRQ} \;>\; \text{C4}\approx\text{C3} \;>\; \text{C7} \;>\; \text{C0}\approx\text{C1}\approx\text{C2},\qquad \text{C8(post-hoc NAP)}\approx\text{C0}$$

C12 is the most important experiment in the proposal. Random features have rich spectra, so the rank floor should still bind. If the gain survives C12, the mechanism is the rank floor. **If it vanishes, the honest description of NRQ changes to "structured retention of ImageNet-1K features"** — a weaker, more contamination-exposed claim that I would have to state as such rather than defend the current framing.

---

## 5. Frozen forecasts, Lane A, R@1, 5 seeds

**Gate:** my PFML reproduction must land within $\pm0.005$ of 0.734 / 0.927 / 0.829. If not, everything below collapses to paired deltas only.

| Dataset | PFML reference | NRQ central | 80% interval | Δ | Δ / pooled SE |
|---|---|---|---|---|---|
| CUB-200-2011 | 0.734 ± 0.003 | **0.746** | 0.735 – 0.756 | +0.012 | ≈ 6.3 |
| Cars196 | 0.927 ± 0.003 | **0.933** | 0.926 – 0.940 | +0.006 | ≈ 3.2 |
| SOP | 0.829 ± 0.002 | **0.833** | 0.828 – 0.838 | +0.004 | ≈ 3.0 |

(Pooled SE of a 5-vs-5 mean difference at $\sigma=0.003$ is $\approx 0.0019$.)

**Frontier-crossing arithmetic, stated against myself.** The central forecasts cross. The **lower 80% bounds do not**: CUB 0.735 is $+0.001$ ($<1$ SE), Cars 0.926 is *below* the reference, SOP 0.828 is below. My own subjective crossing probabilities: **CUB ~70–75%, Cars ~70%, SOP ~65%** — and these are not independent, since a common failure (C12, C11) sinks all three. Joint probability that all three cross: I would put it near **45%**.

**Magnitude rationale.** Gains track where Proposition 1 binds: CUB ($C{=}100$) and Cars ($C{=}98$) sit inside the feasible window, SOP ($C{=}11318$) does not, so SOP retains only the quarantine/invariance route. Cars gets a smaller gain than CUB despite viewpoint nuisance being exactly what the register should absorb, because 0.927 leaves little headroom.

**Secondary, ambiguity-free test (Proxy-Anchor base, fully public recipe):** I forecast **+0.014 CUB / +0.009 Cars / +0.004 SOP** over my own PA reproduction. I deliberately state these as *deltas*: I did not verify PA's published ResNet-50/512-D absolute numbers from a primary source in this session, and I will not quote them from memory.

**In-Shop: no forecast.** The reference (PA+DADA 0.930) has unreported seed count and uncertainty; there is no denominator for frontier arithmetic. I will run In-Shop and report it, but I claim nothing there. Additionally $C{=}3997$ makes the rank floor vacuous, so I expect little.

### Pre-registered falsifiers

* **F1** — PFML+NRQ − PFML(repro) $< +0.005$ on CUB, paired, 5 seeds → falsified as a frontier method.
* **F2** — NRQ $-$ C4 $\le +0.002$ on both CUB and Cars → the **quotient** claim is falsified even if the total gain survives; the method must be renamed "sufficiency regularization", an occupied alternative.
* **F3** — post-hoc NAP/WCCN on C0 recovers $\ge$ 60% of the gain → the train-time claim is falsified.
* **F4** — SOP gain $\ge$ CUB gain → the rank-floor causal story is falsified; the method may still work but the explanation must be withdrawn.
* **F5** — linear class probe on $n$ $>$ 40% top-1 → quarantine leaks; described mechanism is false.
* **F6** — any C11 variant closes the gap → NRQ is not a cost-matched improvement.
* **F7** — achieved $\mathcal L_{\text{suf}}>1.25\epsilon^\star$, or the calibration pass shows $k^\star\le C-1+r$ → the anti-collapse leg never engaged; any observed gain has some other cause.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.**

* *NRQ-full*: 2 student forward+backward passes + 2 frozen teacher forwards ≈ **2.6–2.8× base step time**; peak memory ≈ **1.3–1.4×** (teacher activations are transient, `no_grad`, fp16); added parameters $W_n$ (0.13M) + $V$ (1.18M) + a frozen backbone copy (25.6M, ~50MB fp16) — all discarded at test.
* *NRQ-lite*: NRQ terms on a random 50% sub-batch, teacher on view 1 only ≈ **1.6× step time**, ≈1.15× memory. Forecast ≈ 70% of NRQ-full's gain.

This cost is high relative to the matched-cost references quoted in the prompt (DADA ≈ 1.06× epoch time, 1.01× memory). I therefore treat **C11 as a gating experiment, not an ablation**: NRQ has no frontier claim unless it beats the base run at equal wall-clock.

**Deployment cost: exactly zero delta.** Same ResNet-50, same 512-D descriptor, same single 224 forward, same cosine NN, same index.

**Benchmark and contamination risks.**

1. **ImageNet-1K ↔ CUB/Cars semantic overlap.** ImageNet contains bird and car categories; $\mathcal L_{\text{suf}}$ imports ImageNet-shaped structure. This is *permitted* (ImageNet init is legal, and every reference uses it) but it means part of any gain may be better retention of ImageNet knowledge rather than the claimed quotient. **C12 decides this and I commit to reporting it first.**
2. **CUB test-set size.** 5924 images / 100 classes: R@1 differences below ~0.006 are inside seed noise. My Cars and SOP claims are near that floor, which is why paired 5-seed differences — not mean-vs-published — are the reported statistic.
3. **Hyperparameter leakage.** All NRQ hyperparameters come from held-out *training* identities on CUB and transfer unchanged. PFML's own $(\alpha,\delta)$ cross-validation protocol is not disclosed in what I retrieved; if it touched test identities, the reference is compromised and my arithmetic against it is void through no fault of NRQ. Flagged, not assumed away.
4. **Protocol compliance.** Single model, one view, one 512-D global descriptor, ordinary cosine NN. No test data, no generated data, no text/VLM encoder, no extra annotation, no transduction, no reranking, no test-gallery fitting. The two views are ordinary stochastic augmentation of official training images. The teacher is the permitted ImageNet-1K initialization, frozen.

---

## Where I think this most likely fails

Ranked by my own probability:

1. **C12 kills the story** (~35%) — the gain traces to ImageNet retention, not the rank floor. The method might still improve R@1 while the paper's claim is wrong.
2. **C11 erases the advantage** (~30%) — 2.7× compute spent on longer base training does as well.
3. **The window closes** (~20%) — $k^\star(\epsilon)$ measured on ResNet-50 GAP features never exceeds $C-1+r$ at achievable $\epsilon$, and Proposition 1 never binds.
4. **Quarantine ceiling** (~ongoing) — crop/color are a strict subset of real nuisance (pose, articulation, background). The register can only absorb what the augmentations expose, which caps the quotient's reach and is the most likely reason for a real-but-small effect.
5. **F2** — the register turns out to contribute nothing beyond C4, and NRQ reduces to sufficiency regularization.

I am returning this rather than NONE because the construction is executable end to end, the anti-collapse step is a feasibility argument with an explicit inequality rather than a penalty, the deployed quotient is a genuine structural (not support-limited) operation, and every claim above has a pre-registered way to be shown false. The adjudicating reviewer should weight C12, C11, and C8 above the R@1 table.

---

**Sources:**
- [Potential Field Based Deep Metric Learning (PFML), CVPR 2025 — arXiv HTML](https://arxiv.org/html/2405.18560v4)
- [PFML project page](https://shubhangb97.github.io/potential_field_DML/)
- [PFML CVPR 2025 poster page](https://cvpr.thecvf.com/virtual/2025/poster/33305)
- [Improving Transferability of Representations via Augmentation-Aware Self-Supervision (AugSelf), NeurIPS 2021](https://arxiv.org/abs/2111.09613)
- [Self-supervised learning of Split Invariant Equivariant representations (SIE), ICML 2023](https://arxiv.org/abs/2302.10283)
- [Which Features are Learnt by Contrastive Learning? Class Collapse and Feature Suppression, ICML 2023](https://proceedings.mlr.press/v202/xue23d.html)
- [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT), ICLR 2022](https://arxiv.org/abs/2202.10054)
- [S2SD: Simultaneous Similarity-based Self-Distillation for Deep Metric Learning](https://arxiv.org/pdf/2009.08348)
- [Sharing Matters for Generalization in Deep Metric Learning](https://arxiv.org/abs/2004.05582)
- [Revisiting Training Strategies and Generalization Performance in Deep Metric Learning (ρ-spectrum), ICML 2020](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch)
- [Proxy Anchor Loss for Deep Metric Learning, CVPR 2020](https://arxiv.org/pdf/2003.13911)
- [Proxy Synthesis: Learning with Synthetic Classes for Deep Metric Learning](https://arxiv.org/pdf/2103.15454)
- [Deep Variational Metric Learning (DVML), ECCV 2018](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_42)
- [Adversarial Reconstruction Feedback for Robust Fine-grained Generalization (AdvRF), ICCV 2025](https://arxiv.org/abs/2507.21742)
- [Nonlinear kernel nuisance attribute projection for speaker verification](https://ieeexplore.ieee.org/document/4518562/)
- [Front-End Factor Analysis for Speaker Verification (LDA/WCCN for unseen speakers)](https://sls.csail.mit.edu/publications/2010/Dehak_IEEE_Transactions.pdf)
- [Leave-One-EquiVariant / LOOC discussion of augmentation-specific subspaces](https://arxiv.org/html/2412.18955)
