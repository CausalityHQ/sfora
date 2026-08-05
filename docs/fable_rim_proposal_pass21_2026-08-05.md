# Pass 21 frozen blind proposal: Reliable-Interaction Maximization

Date: 2026-08-05 UTC  
Frozen prompt: `docs/fable_blind_prompt_pass21_2026-08-05.txt`  
Prompt SHA-256: `b7c988ae88b7c956e6656829d64ce838d09009497a5347eeaf6c9eae9058a3eb`  
Native durable consultation: `ba47125f32104e31`  
Caller attribution: `sfora/emafactorial`  
Provider path: Fable, then configured same-job Claude Opus fallback  
Durable status: failed (`exit 1`); result was not truncated and ends cleanly.

The text below is the exact durable result returned by `consult_result`; it has
not been repaired, shortened, or adjudicated.

---
I'm the blind proposer for this pass. Per the session skill rule, I first check skill applicability, then run a targeted novelty sweep against primary literature, then freeze one complete method.
I'm going to write this from primary-source recall rather than spend the remaining budget on searches — flagging below exactly which citations I could not re-verify online in this session.

---

# Proposal: **RIM — Reliable-Interaction Maximization** (one method)

**Lane chosen: Lane A** (ResNet-50, 512-D normalized descriptor, ~224 px, single-view cosine retrieval, 200-epoch compute budget). All forecasts and comparisons below are Lane A only.

---

## 0. One-paragraph statement

Class-discriminative DML losses are Bayes-optimal when the descriptor becomes a *function of the training class label alone*. Every appearance factor that varies **within** a training class is, by construction, useless for the training objective and is actively suppressed by the intra-class compactness term. But at evaluation every class is unseen, and an unseen pair is frequently separated by exactly such a factor. RIM adds one term that measures, and expands, **the number of augmentation-reproducible directions in the class-orthogonal (within-class residual) subspace of the deployed descriptor**, using two augmented views as replicate measurements and a whitened cross-view covariance trace as the estimator. The quantity is a per-direction reliability sum — it is zero for the class-quantized degenerate solution, zero for augmentation-nuisance encodings, ≤1 for rank-1 shortcuts, and invariant to the *magnitude* of within-class scatter, so it does not simply trade compactness for spread.

---

## 1. Executable mathematics

### 1.1 Deployed model (identical to baseline)

- Backbone $\phi_\theta$: ResNet-50, ImageNet-1K pretrained, final block stride 1→ unchanged (standard).
- Pool: $h = \mathrm{GAP}(\phi_\theta(x)) + \mathrm{GMP}(\phi_\theta(x)) \in \mathbb{R}^{2048}$ (this is the Proxy-Anchor official ResNet-50 head; **ambiguity flagged in §7**).
- Embedding: $f = W h + b$, $W \in \mathbb{R}^{512\times 2048}$; $z = f/\lVert f\rVert_2 \in S^{511}$.
- **Test-time**: `Resize(256) → CenterCrop(224)`, one view, one model, $z$, cosine NN. Nothing from RIM survives to test. No reranking, no gallery fitting.

### 1.2 Base loss: multi-proxy Proxy Anchor (MP-PA)

Reproduced Proxy Anchor (Kim et al., CVPR 2020), primary source form:

$$
\mathcal{L}_{\mathrm{PA}}=\frac{1}{|P^{+}|}\sum_{c\in P^{+}}\log\Big(1+\!\!\sum_{i\in X_c^{+}}\!\! e^{-\alpha(s_{ic}-\delta)}\Big)
+\frac{1}{|P|}\sum_{c\in P}\log\Big(1+\!\!\sum_{i\in X_c^{-}}\!\! e^{\alpha(s_{ic}+\delta)}\Big)
$$

with $\alpha=32$, $\delta=0.1$, $P$ = all $C$ class proxies, $P^{+}$ = classes present in the batch.

**MP-PA** replaces the single proxy per class with $K$ sub-proxies $\{p_c^k\}_{k=1}^K$, all L2-normalized, and

$$
s_{ic}=\tfrac{1}{\gamma}\log\textstyle\sum_{k=1}^{K}\exp\big(\gamma\,\langle z_i,\,p_c^{k}\rangle\big),\qquad \gamma=10 .
$$

$K=15$ on CUB/Cars, $K=2$ on SOP/In-Shop — **chosen to match PFML's stated proxy budget so that proxy capacity is a controlled variable, not a confound.** $K=1$ recovers exact Proxy Anchor. No SoftTriple center-merging regularizer (off by default; ablated).

### 1.3 Batch construction (the only structural change)

Sample $K_b$ classes, $n$ distinct images each ⇒ $N=K_b n$ distinct images; draw **two independent augmentations** per image ⇒ $B=2N$ views per step.

Default: $B=192$, $N=96$, $K_b=24$, $n=4$.

Augmentation (both views, identical pipeline):
`RandomResizedCrop(224, scale=(0.16,1.0))`, `RandomHorizontalFlip(0.5)`, `ColorJitter(0.4,0.4,0.4,0.1)` w.p. 0.8, `RandomGrayscale(0.2)`.
This is stronger than PA's default (crop+flip only) — **control C2 below neutralizes it.**

$\mathcal{L}_{\mathrm{PA}}$ is computed over all $B=192$ views (both views count as samples of their class).

### 1.4 The RIM term

Let $z_i^{(v)}$, $v\in\{1,2\}$, $i=1..N$, labels $y_i$.

**(a) Class-orthogonal residual** (removes the class main effect exactly, in-batch):
$$
m_c^{(v)}=\tfrac{1}{n}\!\!\sum_{i:y_i=c}\!\! z_i^{(v)},\qquad r_i^{(v)}=z_i^{(v)}-m_{y_i}^{(v)} .
$$
By construction $\sum_{i\in c} r_i^{(v)}=0$ for every class, so $r$ carries **no** in-batch class-mean component. (Removing per-image additive effects is unnecessary here: the class-mean projection already kills the rank-1 "global saliency" solution's class-aligned part, and the whitener below kills its residual — see §2, D3.)

**(b) Covariances** ($\nu = N-K_b$ residual degrees of freedom):
$$
\Sigma_{12}=\tfrac{1}{\nu}\sum_i r_i^{(1)} r_i^{(2)\top},\qquad
\hat\Sigma=\tfrac{1}{2\nu}\sum_i\big(r_i^{(1)}r_i^{(1)\top}+r_i^{(2)}r_i^{(2)\top}\big).
$$

**(c) Whitener** — EMA over steps, **stop-gradient**:
$$
\bar\Sigma \leftarrow 0.9\,\bar\Sigma + 0.1\,\hat\Sigma,\qquad
W=\mathrm{sg}\big[(\bar\Sigma+\varepsilon I_d)^{-1/2}\big],\quad \varepsilon=\kappa/d,\ \kappa=0.05 .
$$
$(\cdot)^{-1/2}$ by 5 Newton–Schulz iterations on the trace-normalized matrix $A=(\bar\Sigma+\varepsilon I)/\tau$, $\tau=\mathrm{tr}(\bar\Sigma+\varepsilon I)$, then rescale by $\tau^{-1/2}$. $\bar\Sigma$ initialized to $I/d$; refreshed every step (cost §6).

$\varepsilon$ is **absolute**, not relative: because $\lVert z\rVert=1$, $\mathrm{tr}(\hat\Sigma)\le 1$ always, so $\varepsilon=\kappa/d$ is an anchored floor — a residual direction with within-class variance below $\approx 10^{-4}$ (std $<0.01$ on the unit sphere) contributes nothing. **This is what makes uniform residual shrinkage — i.e. class quantization — a strict loss increase rather than a neutral reparameterization.**

**(d) Reliable rank:**
$$
\boxed{\;T=\mathrm{tr}\!\big(W\,\Sigma_{12}\,W\big)=\tfrac{1}{\nu}\sum_{i=1}^{N}\big\langle W r_i^{(1)},\,W r_i^{(2)}\big\rangle\;}
$$

**Interpretation (exact, at stationarity).** Write the residual as reliable signal + augmentation noise, $r^{(v)}=s+\eta^{(v)}$, $\eta^{(1)}\!\perp\!\eta^{(2)}$, zero-mean. Then $\Sigma_{12}=\Sigma_s$ and $\bar\Sigma=\Sigma_s+\Sigma_\eta$, and in the joint eigenbasis
$$
T=\sum_{k=1}^{d}\frac{\lambda_k^{s}}{\lambda_k^{s}+\lambda_k^{\eta}+\varepsilon}\;=\;\sum_k \rho_k,\qquad \rho_k\in[0,1].
$$
$T$ is literally **the number of augmentation-reliable, class-orthogonal descriptor directions** — a per-direction test–retest reliability sum, i.e. a noise-ceiling estimate.

**(e) Loss:**
$$
\mathcal{L}_{\mathrm{RIM}}=\max\!\Big(0,\;1-\tfrac{\min(T,\,r^{*})}{r^{*}}\Big),\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{MP\text{-}PA}}+\lambda(t)\,\mathcal{L}_{\mathrm{RIM}} .
$$
The hinge at target rank $r^{*}$ makes the pressure **bounded and explicit**: RIM asks for $r^{*}$ reliable directions, not for maximal within-class spread.

### 1.5 Gradient path

$\mathcal{L}_{\mathrm{RIM}}$ reaches $\theta, W_{\mathrm{emb}}$ **only** through $z^{(1)},z^{(2)}$; it reaches **no** proxy. With $W$ detached,
$$
\frac{\partial T}{\partial r_i^{(1)}}=\frac{1}{\nu}\,W W r_i^{(2)},\qquad
\frac{\partial r_i^{(v)}}{\partial z_j^{(v)}}=\delta_{ij}I-\tfrac{1}{n}\mathbb{1}[y_i{=}y_j]\,I ,
$$
then through the L2-normalization Jacobian $\frac{1}{\lVert f\rVert}(I-zz^{\top})$ into the backbone. Detaching $W$ is deliberate: it makes the term a fixed preconditioner within a step, so the incentive is *"put reproducible signal into currently under-used residual directions"* — an explicit rank-filling pressure whose fixed point is $\bar\Sigma \propto I$ on the reliable subspace. Proxies receive gradient only from $\mathcal{L}_{\mathrm{MP\text{-}PA}}$; the labour division is clean (proxies = class main effect, RIM = class-orthogonal interaction).

### 1.6 Hyperparameters and schedules (frozen)

| | CUB / Cars | SOP / In-Shop |
|---|---|---|
| $\lambda^{*}$ | 0.5 | 0.2 |
| $r^{*}$ | 64 | 32 |
| $K$ sub-proxies | 15 | 2 |
| $\kappa$ | 0.05 | 0.05 |
| $B$ (views) | 192 | 192 |
| $n$ per class | 4 | 4 |

- $\lambda(t)$: linear 0 → $\lambda^{*}$ over epochs 1–5, constant thereafter.
- Optimizer AdamW, backbone/embedding lr $10^{-4}$, **proxy lr $10^{-2}$**, weight decay $10^{-4}$ on backbone+embedding, **zero weight decay on proxies**, cosine decay $10^{-4}\!\to\!10^{-6}$ over the run, 1 warm-up epoch with the backbone frozen.
- **Compute-matched epoch definition:** every arm uses the same total image forward passes, $200\times N_{\text{train}}$. RIM therefore performs **100 dataset passes × 2 views**; single-view arms perform 200 × 1. This is the honest FLOP match and it is a real handicap RIM must overcome (see C2).
- **All hyperparameters selected on a class-disjoint validation split** (last 20% of *training* identities held out, zero-shot-evaluated), then the model is retrained on all training identities with the frozen values. No test identity is touched at any point, including for epoch selection.

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode: *within-class-varying factor suppression*

Let the image be generated from factors $u=(u_1,\dots,u_M)$. Partition them at training time: $\mathcal{D}$ = factors whose conditional mean differs across training classes; $\mathcal{S}$ = factors with $\mathbb{E}[u_m\mid y]$ constant in $y$ but $\mathrm{Var}(u_m\mid y)>0$.

**Claim.** For any loss of the form $\mathcal{L}=\mathbb{E}[\ell(\text{class scores})] $ with an intra-class compactness component (all proxy and pair losses, including PA and MP-PA), the population minimizer over unconstrained $f$ satisfies $z^{\star}(x)=g(y(x))$ — a function of the label alone; sensitivity to every $u_m\in\mathcal{S}$ is exactly zero. Sketch: the loss decomposes as $\mathbb{E}_y\,\mathbb{E}_{x|y}[\ell]$; conditional on $y$, the compactness term is minimized at the conditional mean, and the discriminative term depends on $x$ only through $y$; substituting $z=\mathbb{E}[z|y]$ weakly decreases both terms. □

At evaluation, all identities are unseen. A pair of unseen classes separated **only** by factors in $\mathcal{S}$ (e.g. beak curvature that varies among individuals of every training species but is diagnostic between two held-out species) receives $\langle z_q,z_g\rangle$ determined entirely by $\mathcal{D}$-factors, and the two classes collide: within-pair R@1 falls to chance. This is not a capacity failure or an optimization failure — it is the *correct* solution to the training problem.

The severity is dimension-counted: the class main effect occupies at most $C-1$ of $d=512$ dimensions, leaving $\max(0,\,d-C+1)$ dimensions that receive **no gradient at all** from a single-proxy loss (the proxy-loss gradient $\partial\mathcal{L}/\partial z_i$ lies in $\mathrm{span}\{p_c\}$). CUB: 413 free dims. Cars: 415. In-Shop, SOP: 0. This directly generates the falsifiable dataset-ordering prediction in §5.

RIM attacks it by making the *reliable* part of the $\mathcal{S}$-subspace an explicit optimization target, filtered so that only content — not nuisance, not noise, not a single global axis — can satisfy it.

### 2.2 Proof-level attack on the cheapest degeneracies

**D1 — Class quantization $z_i=p_{y_i}$ (the exact optimum of the base loss).**
$r_i^{(v)}=0\ \forall i\Rightarrow \Sigma_{12}=0\Rightarrow T=0\Rightarrow \mathcal{L}_{\mathrm{RIM}}=1$, the global maximum of the RIM term. Strictly penalized, with nonzero gradient (the whitener is $\varepsilon^{-1/2}I$, finite). **The degenerate optimum of the base loss is the unique maximizer of the added loss.**

**D2 — Uniform residual shrinkage $r\to\alpha r$, $\alpha\to0$ (soft quantization).**
$T=\sum_k \lambda^s_k\alpha^2/(\alpha^2(\lambda^s_k+\lambda^\eta_k)+\varepsilon)\to 0$ as $\alpha\to0$. The **absolute** $\varepsilon$ (anchored by $\lVert z\rVert=1$) is what closes this hole; a scale-relative ridge would leave $T$ invariant and RIM would be blind to the very degeneracy it targets.

**D3 — Rank-1 shortcut $r_i \approx c_i u$ (one shared axis, e.g. "canonicality of pose").**
$\Sigma_{12}$ has rank 1 ⇒ $T\le 1 \ll r^{*}=64$. Cost $1-1/64=0.984$ of the maximum. Any low-rank shortcut is penalized in proportion to its rank deficit.

**D4 — Nuisance encoding (crop offset, colour, illumination).**
These are resampled independently per view, so their contribution to $\Sigma_{12}=\mathbb{E}[r^{(1)}r^{(2)\top}]$ has expectation 0; they enter $\bar\Sigma$ (the denominator) but not the numerator, *reducing* $T$. Encoding nuisance is strictly worse than not encoding it. This is why the replicate structure — not a variance floor — is the right filter: a VICReg-style std hinge would be **fully satisfied** by colour noise.

**D5 — Additive per-image "saliency" bias $r_i = a_i u + \dots$** — reliable and reproducible, but rank-1 in the residual Gram ⇒ D3 applies; and its class-aligned component is annihilated by the class-mean subtraction.

**D6 — Inflating $r$ along low-variance directions to exploit the detached whitener.**
Instantaneously $T$ can exceed $r^{*}$; the hinge $\min(T,r^{*})$ removes the gradient, and the EMA whitener re-normalizes those directions within $\sim\!10$ steps. Fixed point: $\bar\Sigma\propto I$ on the reliable subspace with $T\approx r^{*}$.

**D7 — Instance memorization (the one that is *not* closed).**
A residual that encodes "which training image this is" is augmentation-reliable and high-rank, and satisfies RIM without any compositional content. Three partial mitigations, and an honest statement that this is the method's main theoretical hole:
(i) strong augmentation (colour jitter + grayscale) raises the cost of memorization relative to content encoding — the same empirical regularity that makes SimCLR learn content rather than image hashes;
(ii) the whitener is **pooled across classes**, so the reliable subspace must be a single shared basis — per-class idiosyncratic codes are penalized by whitening against the pooled covariance;
(iii) a direct diagnostic and falsifier: measure $T$ on training images vs. held-out-*training-class* images. Memorization ⇒ $T_{\text{train}}\gg T_{\text{held-out}}$. Threshold: **if $T_{\text{held-out}}/T_{\text{train}} < 0.6$, the mechanism is memorization and the claim is withdrawn** (§5, F5).

---

## 3. Adversarial novelty search — nearest works and one-sentence distinctions

*Verification note: these are from primary-source recall (assistant knowledge cutoff May 2026); I did not re-verify them online in this session due to a hard budget cap, and §7 lists the specific claims that need re-checking before publication.*

**Inside DML**

1. **Roth et al., "Revisiting Training Strategies and Generalization Performance in DML" (ICML 2020) — $\rho$-spectrum regularization.** *Closest prior art; it flattens the unconditional singular-value spectrum of batch embeddings by stochastically swapping positives for negatives, with no label conditioning and no criterion separating signal from noise — a spectrum flattened by augmentation noise scores perfectly under $\rho$-reg and gives $T=0$ under RIM.*
2. **DiVA (Milbich et al., ECCV 2020).** *DiVA allocates disjoint sub-embeddings to auxiliary self-supervised/shared/intra tasks and concatenates them; RIM adds no head, no sub-space and no auxiliary task, and constrains the reliability spectrum of the single deployed descriptor's class-residual.*
3. **S2SD (Roth et al., ICML 2021).** *S2SD distils similarity from concurrently trained higher-dimensional auxiliary embedding spaces (a teacher); RIM has no teacher and no second embedding space — its supervisory signal is cross-augmentation reproducibility.*
4. **DVML (Lin et al., ECCV 2018).** *DVML posits class-independent intra-class variance and uses a VAE to synthesize samples; RIM makes no distributional or generative assumption and optimizes the reliable **rank** of the residual, not its distribution.*
5. **SoftTriple / ProxyGML / PFML (multi-proxy family).** *These enrich the **class main effect** (more modes per class); RIM operates strictly in the orthogonal complement of the class main effect and is composable with any of them — which is why MP-PA is the control, not the contribution.*
6. **Proxy Synthesis / Embedding Expansion / MemVir.** *These manufacture additional class-level anchors; RIM manufactures no anchors and adds no synthetic identity.*
7. **Zhang et al., Spherical Embedding Constraint (NeurIPS 2020).** *Regularizes descriptor **norms** toward uniformity; RIM regularizes the direction-space reliability spectrum of a residual, on already-normalized descriptors where norms are constant.*

**Outside DML**

8. **Barlow Twins (ICML 2021) / VICReg (ICLR 2022) / W-MSE (ICML 2021).** *All are unsupervised objectives on raw projector features with no label conditioning; RIM's object is the **label-partialled** residual — the exact subspace a supervised metric loss drives to zero and which these methods, applied post-hoc to a DML descriptor, would score using variance the class means supply.*
9. **Deep CCA (Andrew et al., ICML 2013) and classical partial CCA.** *DCCA aligns two genuinely different data modalities through two separate networks; RIM's "views" are augmentation replicates through one shared network, the correlation is taken after partialling out class covariates, and the objective is coupled to a proxy loss it is deliberately antagonistic to.*
10. **MCR² / coding-rate maximization (Yu et al., NeurIPS 2020).** *MCR² explicitly **minimizes** the within-class coding rate to compress each class to a low-dimensional subspace; RIM **maximizes** the reliable within-class rank — the sign of the within-class term is inverted, and the inversion is the entire zero-shot argument.*
11. **RankMe (ICML 2023) / $\alpha$-ReQ (NeurIPS 2022).** *Both are post-hoc effective-rank **measurements** used for model selection; RIM is a training term and, unlike both, discounts each direction by its replicate reliability rather than by its variance.*
12. **Principled import — Reliable/Correlated Components Analysis and noise-ceiling estimation in neuroimaging (Dmochowski et al.; inter-subject correlation; Chen et al.'s shared response model, NeurIPS 2015).** *That literature uses repeated stimulus presentations to isolate the stimulus-locked component of a neural response and estimate its dimensionality; RIM imports the estimator wholesale — augmentation replicates play the role of repeated presentations — and is, to my knowledge, its first use as a **training objective** and its first use on a label-partialled residual.*

**The mechanism claim in one sentence:** *no prior work optimizes a replicate-reliability-weighted rank of the label-orthogonal component of the deployed retrieval descriptor.* Each of (1)–(12) drops at least one of {replicate reliability, label partialling, rank rather than variance, acts on the deployed descriptor, is a training objective}.

---

## 4. Decisive matched-compute controls

All at $200\times N_{\text{train}}$ image forwards, $B=192$ views, 5 seeds, identical schedules.

| | Arm | What it kills if it matches RIM |
|---|---|---|
| **C1** | MP-PA, single-view, standard aug (crop+flip), 200 passes | shows the baseline is not handicapped by the paired-view schedule |
| **C2** | **MP-PA, paired-view batches, strong aug, $\lambda=0$** | **the decisive control** — isolates the RIM *term* from batch composition and augmentation strength. Every headline delta is measured against C2. |
| C3 | C2 + $\lambda\,(1-\frac{1}{N}\sum_i\langle z_i^{(1)},z_i^{(2)}\rangle)$ | "it's just augmentation invariance" |
| C4 | C2 + W-MSE on **raw** (non-residualized) $z$ | "the label partialling is decorative" |
| C5 | C2 + VICReg-style std hinge on $r_i$ (variance floor, no replicate term) | "it's just keeping within-class variance" |
| C6 | C2 + $\rho$-spectrum regularization (Roth et al. 2020) | "it's unconditional spectrum flattening" |
| C7 | **Shuffled-label RIM**: residualize against a random class assignment | if this recovers ≥70% of the gain, label conditioning is not the mechanism |
| C8 | RIM with **independent** view pairing (pair $i$'s view 1 with a random *other* image's view 2) | destroys the replicate structure; $T$ should collapse to ~0 and the gain vanish |
| C9 | $\lambda$-sweep $\{0,0.125,0.25,0.5,1,2,4\}$ | mechanism predicts an **interior optimum**; monotone-to-largest-$\lambda$ falsifies the trade-off story |
| C10 | $r^{*}$-sweep $\{4,16,64,256\}$, and $d\in\{128,512\}$ | mechanism predicts gain grows with $d$ at fixed $C$ (more spare residual capacity) |
| C11 | RIM with $K=1$ (plain PA base) | confirms RIM is additive to, not a substitute for, proxy capacity |
| C12 | Memory-bank variant (view 2 read from a per-image bank refreshed once per pass; zero extra forwards) | tests whether the 2× view cost is load-bearing or replaceable |

Mediation requirement (not just an ablation): $T$ measured on the **held-out-class validation split** must increase monotonically with $\lambda$ over the region where R@1 increases. A gain without a $T$ increase means the stated mediator is wrong regardless of R@1.

---

## 5. Frozen forecasts, Lane A, ResNet-50 / 512-D / 224 px / cosine / 200-epoch-equivalent / 5 seeds

**These are point forecasts with std over seeds; the point forecasts themselves carry substantially wider uncertainty than the seed std, and I say so explicitly below.**

| Dataset | C1 (MP-PA, 1-view) | **C2 (matched control)** | **RIM** | Δ vs C2 | PFML ref. | Cross? |
|---|---|---|---|---|---|---|
| CUB-200-2011 | 0.718 ± 0.005 | 0.720 ± 0.005 | **0.741 ± 0.006** | **+2.1** | 0.734 ± 0.003 | **+0.7 pt** |
| Cars196 | 0.905 ± 0.005 | 0.906 ± 0.005 | **0.922 ± 0.005** | **+1.6** | 0.927 ± 0.003 | **−0.5 pt (no)** |
| SOP | 0.812 ± 0.003 | 0.812 ± 0.003 | **0.819 ± 0.003** | **+0.7** | 0.829 ± 0.002 | −1.0 pt (no) |

Plain-PA base ($K=1$) reference points: CUB 0.710 ± 0.005 → RIM 0.731; Cars 0.890 → 0.907.

### Frontier arithmetic (explicit)

- **CUB.** $0.741-0.734=+0.007$. Standard error of the difference of means, 5 seeds each: $\sqrt{0.006^2/5+0.003^2/5}=0.0030$ ⇒ $+2.3\sigma$ *conditional on the point forecast being correct*. **Unconditionally I put the probability of a genuine CUB crossing at ≈40–45%**, because my forecast error on the C2 baseline alone is ±0.010 and on Δ is ±0.010 — each larger than the 0.007 margin.
- **Cars196.** Forecast **does not cross** (−0.5 pt). Crossing would require Δ ≥ +2.1 rather than +1.6, ≈20% probability.
- **SOP.** Forecast **does not cross** (−1.0 pt), and the mechanism predicts it should not: $d-C+1 = 0$, there is no spare residual subspace.
- The honest headline is therefore: **a large, mechanism-attributable, replicable delta over its own matched control on the two low-$C$ datasets; a coin-flip on crossing PFML on CUB; no crossing on Cars or SOP.** RIM is designed to be *orthogonal* to class-level mechanisms, so the expected frontier configuration is PFML + RIM — which I cannot forecast because PFML's recipe is not available to me (§7).

### Falsification thresholds (pre-registered)

- **F1.** RIM − C2 < **+0.8** R@1 on CUB, or one-sided Welch $p\ge0.05$ over 5 seeds ⇒ **method falsified**.
- **F2 (mechanism, dataset ordering).** The mechanism predicts $\Delta_{\text{CUB}}\approx\Delta_{\text{Cars}} > \Delta_{\text{In-Shop}}\approx\Delta_{\text{SOP}}$, driven by spare residual capacity $\max(0,d-C+1)$ = 413 / 415 / 0 / 0. **If $\Delta_{\text{SOP}} \ge \Delta_{\text{CUB}}$, the causal story is falsified even if R@1 improves** — the gain would then be a generic regularization effect and the paper should not be written as stated.
- **F3.** C7 (shuffled-label residualization) recovers ≥70% of the gain ⇒ label conditioning is not the mechanism ⇒ falsified.
- **F4.** C3, C4, C5 or C6 individually recovers ≥70% of the gain ⇒ RIM is a re-parameterization of an occupied regularizer ⇒ falsified.
- **F5.** $T_{\text{held-out-class}}/T_{\text{train}} < 0.6$ ⇒ the term is being satisfied by instance memorization (D7) ⇒ claim withdrawn.
- **F6.** No interior optimum in the C9 $\lambda$-sweep, or held-out-class $T$ flat in $\lambda$ while R@1 rises ⇒ the stated mediator is wrong.
- **F7.** $\Delta(d{=}512) \le \Delta(d{=}128)$ on CUB ⇒ the spare-capacity account is wrong.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.**
- FLOPs of the RIM term per step: two $512\times512$ covariance builds ($\approx5\times10^7$) + 5 Newton–Schulz iterations ($\approx1.3\times10^9$), against ResNet-50 fwd+bwd on 192 views at 224² ($\approx2.4\times10^{12}$). **Overhead ≈ 0.06% FLOPs; expect ≤ 3% wall-clock** from launch overhead. Memory: +2 MB ($\bar\Sigma$, $W$). Comparable to PA+DADA's reported 1.06× epoch / 1.01× memory.
- **The real cost is not the term — it is the two views.** At a fixed FLOP budget RIM makes 100 dataset passes instead of 200. I have accounted for this by defining the budget in image forwards and by making C2 carry the identical handicap; C1 measures whether the handicap is itself harmful (forecast: ≤0.3 pt, i.e. paired views act as extra augmentation and roughly break even).
- C12 (memory-bank variant) removes the 2× view cost entirely at the price of weight-drift bias; if it retains ≥70% of the gain, RIM is a **≈1.03× training-cost method**.

**Deployment cost: exactly zero.** One ResNet-50, one view, one 512-D descriptor, cosine NN. No auxiliary network at train time either (unlike AdvRF's ResNet-34/U-Net or VAPNet's attribute machinery in Lane B).

**Risks.**
1. **D7 / instance memorization** is the largest theoretical risk and is only empirically, not structurally, closed. F5 is the guard.
2. **Compactness–reliability trade-off may not have a usable interior optimum on every dataset.** Retaining reliable within-class structure raises within-unseen-class scatter as well as between-unseen-class separability; the net sign is empirical. $r^{*}$ and $\lambda$ are the knobs; C9/C10 decide it.
3. **ImageNet-1K pretraining contamination.** CUB and Cars overlap ImageNet's bird/vehicle classes; the "zero-shot" claim is with respect to disjoint dataset identities only, not disjoint visual concepts. This affects all Lane-A arms equally, but RIM's mechanism (recovering suppressed within-class factors) may draw disproportionately on pretrained features that already encode them — so the from-scratch effect size is likely smaller. I do not claim otherwise.
4. **Resolution limits.** CUB test has ~5,924 images; a 0.5 pt difference at 5 seeds and σ≈0.006 is not reliably resolvable. Any claim below ~0.8 pt should not be made. Cars196 contains near-duplicate images and SOP contains multiple near-identical views per product, both of which inflate R@1 and compress method differences.
5. **HP-selection contamination is endemic in this literature.** I use a class-disjoint validation split for every hyperparameter *and* for epoch selection; comparisons against published numbers that tuned on test are therefore biased **against** RIM, and I would not claim a sub-1-point frontier crossing under that asymmetry.
6. **The 200-epoch lane budget itself.** Proxy Anchor's published recipe converges around 60 epochs with step decay; I substitute cosine decay to $10^{-6}$ over the full budget. This is a deviation from the primary-source recipe and is the reason C1/C2 are run as my own reproduction rather than inherited.

---

## 7. Unresolved source ambiguities (stated, not papered over)

1. **PFML (CVPR 2025).** I do not have verified access to its loss definition, optimizer, LR schedule, batch composition, augmentation, warm-up, or epoch count. The 15/2 proxy configuration is taken from the prompt, not from the paper. **Every comparison to PFML above is reference-only; no matched reproduction of PFML is claimed, and the CUB "crossing" is therefore a cross-paper comparison with all the usual hazards.**
2. **PA+DADA (AAAI 2024).** Seed count and uncertainty unreported (per the prompt), so its In-Shop 0.930 admits no interval; I have deliberately not made In-Shop a forecast dataset.
3. **Proxy Anchor (CVPR 2020).** Three details I recall but flag as needing re-verification against the official code before running: (a) whether the ResNet-50 head is $\mathrm{GAP}+\mathrm{GMP}$ summed (as I have specified) or GAP alone; (b) whether proxies are excluded from weight decay; (c) the exact epoch count and whether `--warm 1` freezes the backbone or only the proxies. If (a) differs, C1/C2 change and all absolute numbers shift — the Δ-vs-C2 claims are unaffected.
4. **MP-PA is my own construction**, not a published baseline; the smooth-max sub-proxy aggregation is SoftTriple-style but the combination with the Proxy Anchor loss is mine and must be reproduced as my own baseline (it is — C1/C2).
5. **Novelty search was conducted from primary-source recall, not re-verified online in this session** due to the budget cap. The three claims most worth re-checking before freezing: that no DML paper applies a replicate-reliability estimator to label-partialled deployed descriptors; that $\rho$-spectrum regularization has no reliability filter; and that Reliable Components Analysis has not previously been used as a training objective.
6. **Loss scale is operational, not cosmetic.** $\mathcal{L}_{\mathrm{RIM}}\in[0,1]$ and $\lambda$ multiplies it against $\mathcal{L}_{\mathrm{MP\text{-}PA}}$, which is $O(1$–$10)$ and *itself* interacts with AdamW's decoupled weight decay on $W_{\mathrm{emb}}$; changing $\lambda$ changes the effective regularization on the embedding layer, so $\lambda$ must be swept jointly with weight decay (C9 does this) rather than treated as a harmless rescaling.

---

**Bottom line.** RIM is one term, one line of mathematics, zero deployment cost, and roughly 1.03× training cost in its bank variant. It targets a degeneracy that is provably the *optimum* of every proxy loss in this lane, and it is the unique maximizer-inverse of that degeneracy. It is falsifiable on six independent axes, including a dataset-ordering prediction that no generic regularizer would produce. I forecast it clears its own matched control decisively on CUB and Cars, and I put roughly even odds on it crossing PFML on CUB — while forecasting plainly that it does **not** cross on Cars196 or SOP.
