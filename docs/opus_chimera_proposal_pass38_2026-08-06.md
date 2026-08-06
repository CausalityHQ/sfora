## Lane declaration

**Lane A** (matched 512‑D CNN lane) for every forecast and comparison below. Rationale: PFML's references carry audited 5‑run uncertainty (±0.003), which makes falsification thresholds arithmetically decisive; Lane B's AdvRF row has no reported uncertainty and bundles a ResNet‑34/U‑Net reconstruction system, so a crossing claim there is not auditable at the precision I need.

I return **ONE** method.

---

# CHIMERA — Compositional Hypothesis of Identity via Masked Exchange of Reusable Atoms

## 0. The one‑line claim

Replace the class‑private proxy table with a **class‑shared, usage‑balanced atom vocabulary** whose combinations generate class prototypes, and train the encoder against **combinatorially recombined (chimeric) identities** under a calibration law that forces atoms to be individually readable. Zero learned per‑class parameters exist anywhere in the model.

---

## 1. Executable mathematics

### 1.1 Deployed model (unchanged, Lane A)

ResNet‑50, ImageNet‑1K init → GAP $h(x)\in\mathbb R^{2048}_{\ge0}$ → linear $W\in\mathbb R^{512\times2048}$ → $f(x)=Wh(x)/\lVert Wh(x)\rVert_2\in S^{511}$.
Test: resize 256, center crop 224, single view, cosine NN. Nothing below is deployed.

### 1.2 Train‑time objects

| object | shape | learned? | notes |
|---|---|---|---|
| atom vocabulary $U=[u_1..u_K]$ | $512\times K$ | yes, SGD | columns projected to unit norm after every step (projected gradient on $(S^{511})^K$); **weight decay disabled on $U$** — AdamW decay would fight the unit‑norm projection and silently rescale the sphere learning rate |
| class code table $\Gamma$ | $K\times C$ | **no** — EMA buffer, `stop_grad` | this is the anti‑lookup‑table design |
| $\tau$ (code temperature) | scalar | fixed 0.05 | |
| $\kappa$ (calibration slope) | scalar | fixed 0.6 | **not learned** (see D4) |

$K=256$ for CUB/Cars; $K=1024$ for SOP/In‑Shop.

### 1.3 Image atom code

$$z(x)=U^\top f(x)\in\mathbb R^{K},\qquad a(x)=\mathrm{softmax}\!\big(z(x)/\tau\big)\in\Delta^{K-1}$$

### 1.4 Class code as a *balanced statistic*, not a parameter

Per batch $\mathcal B$: $\tilde\Gamma^{\mathcal B}_{jc}=\frac{1}{|\mathcal B_c|}\sum_{x\in\mathcal B_c}a_j(x)$; EMA into the global table with count‑debiased momentum $m_c=\max(0.5,\,1-1/n_c)$ ($n_c$ = images of $c$ seen so far — essential on SOP where classes recur rarely):

$$\Gamma_{:,c}\leftarrow m_c\Gamma_{:,c}+(1-m_c)\tilde\Gamma^{\mathcal B}_{:,c}$$

Then, once per iteration, **entropic OT balancing on the full $K\times C$ table** (3 Sinkhorn iterations, $\epsilon=\tau$):

$$\Gamma\leftarrow\mathrm{Sinkhorn}_\epsilon(\Gamma)\ \ \text{s.t.}\ \ \Gamma^\top\mathbf 1=\tfrac1C\mathbf 1_C\ \ (\text{column: each class code sums to }1/C),\quad \Gamma\mathbf 1=\tfrac1K\mathbf 1_K\ (\textbf{row: every atom carries equal mass across classes})$$

The **row marginal is the sharing constraint** and is the load‑bearing piece: it makes a class‑private atom mathematically inadmissible. Cost: $256\times100$ (CUB) to $1024\times11{,}318$ (SOP), 3 iterations — sub‑millisecond.

Class prototype: $\;p_c=U\Gamma_{:,c}\big/\lVert U\Gamma_{:,c}\rVert$. Gradient flows to $U$ and (through $f$) to $\theta$; **never** to $\Gamma$.

### 1.5 Loss terms

**(L1) Composed‑prototype metric loss** — Proxy Anchor form with $p_c$ substituted for the proxy table:

$$\mathcal L_{\rm PA}=\frac1{|C^+|}\!\sum_{c\in C^+}\!\log\Big(1+\!\!\sum_{x\in X^+_c}\!e^{-\alpha(\langle f(x),p_c\rangle-\delta)}\Big)+\frac1{C}\sum_{c}\log\Big(1+\!\!\sum_{x\in X^-_c}\!e^{\alpha(\langle f(x),p_c\rangle+\delta)}\Big)$$

$\alpha=32,\ \delta=0.1$.

**(L2) Code grounding + augmentation stability.** For each image, $\mathcal L_{\rm code}=\mathrm{CE}\big(C\!\cdot\!\Gamma_{:,c(x)},\,a(x)\big)$. On a random **25 % subset** of the batch a second augmented view $x'$ is drawn and the symmetric term $\tfrac12[\mathrm{CE}(C\Gamma_{:,c},a(x))+\mathrm{CE}(C\Gamma_{:,c},a(x'))]+\mathrm{KL}(\mathrm{sg}[a(x)]\Vert a(x'))$ is used. (25 %, not 100 %, to hold epoch cost at ≈1.25×.)

**(L3) Chimeric identity synthesis — the core.** Per step draw $V=64$ chimeras. For each: classes $c\neq c'$ from the batch, mask $M\sim\mathrm{Bern}(\tfrac12)^K$,

$$\gamma^\times=\frac{M\odot\Gamma_{:,c}+(\mathbf1-M)\odot\Gamma_{:,c'}}{\lVert\cdot\rVert_1},\quad q=\frac{U\gamma^\times}{\lVert U\gamma^\times\rVert},\quad \beta=\frac{\langle M,\Gamma_{:,c}\rangle}{\langle M,\Gamma_{:,c}\rangle+\langle\mathbf1-M,\Gamma_{:,c'}\rangle}$$

With batch class‑means $\bar f_c$ (embeddings, gradient‑carrying):

$$\underbrace{\mathcal L_{\rm cal}=\big(\langle\bar f_c,q\rangle-\langle\bar f_{c'},q\rangle-\kappa(2\beta-1)\big)^2}_{\textbf{linear‑readout law}}\;+\;\underbrace{\mathcal L_{\rm sep}=\big[\delta_v+\langle\bar f_{c''},q\rangle-\min(\langle\bar f_c,q\rangle,\langle\bar f_{c'},q\rangle)\big]_+}_{\text{chimera is a real identity}}\;+\;\underbrace{\mathcal L_{\rm rep}=\tfrac1{V(V-1)}\!\sum_{v\neq w}\![\langle q_v,q_w\rangle-\eta]_+^2}_{\text{chimeras don't collapse}}$$

$\delta_v=0.1$, $\eta=0.5$, $c''$ a third batch class.

**(L4) Code‑entropy floor** (differentiable path, on image codes): $\mathcal L_{\rm ent}=\mathbb E_x\big[\log k^\ast-H(a(x))\big]_+$, $k^\ast=32$.

**Total:** $\;\mathcal L=\mathcal L_{\rm PA}+\lambda_c\mathcal L_{\rm code}+\lambda_r(\mathcal L_{\rm cal}+\mathcal L_{\rm sep}+\mathcal L_{\rm rep})+\lambda_e\mathcal L_{\rm ent}$, with $\lambda_c=1.0$, $\lambda_e=0.1$, and $\lambda_r:0\to0.5$ linearly over epochs 10–30 (chimeras are meaningless before atoms exist).

### 1.6 Schedule

200 epochs, AdamW, backbone lr $10^{-4}$, $U$ lr $10^{-2}$ (100×), wd $10^{-4}$ on $\theta,W$ only, cosine decay, 5‑epoch linear warmup, batch 180 with $m=4$ images/class (SOP/In‑Shop $m=4$, batch 180), RandomResizedCrop(224)+hflip only. $\Gamma$ initialised by one forward pass with the frozen ImageNet encoder before epoch 1.

### 1.7 Operational scales (explicitly *not* harmless normalizations)

- $\lVert U\gamma\rVert$ normalization is **not** free: it interacts with the unit‑norm projection on $u_j$ and with AdamW; hence wd is off for $U$.
- $(\tau,k^\ast)$ is **one** knob, not two — $H(a(x))$ is a function of $\tau$ at fixed $\lVert f\rVert=1$. Ablate the pair jointly.
- $\kappa$ lives on the $[-1,1]$ cosine scale and is therefore operational; fixing it (rather than learning it) is a degeneracy fix, not a convenience.

---

## 2. The causal zero‑shot error mode, and the degeneracy attack

### 2.1 Error mode: **finite‑class rank bottleneck under variability collapse**

For any per‑class‑prototype loss with $C$ training classes, $\operatorname{rank}\operatorname{span}\{\mu_c-\bar\mu\}\le C-1$. Neural‑collapse terminal dynamics additionally drive within‑class variability toward zero, contracting the *entire* feature cloud toward that affine subspace. On CUB ($C{=}100$) and Cars196 ($C{=}98$) that is **≤ 99 of 512 dimensions**. Disjoint test‑class means have no reason to lie in it; the directions that separate two unseen species are exactly those the loss could afford to discard because training‑class attribute correlations made them redundant.

**Direct evidence inside the brief itself:** PFML needs **15 proxies/class on CUB/Cars** ($15C-1\approx1499\gg512$, rank fully restored) but only **2 on SOP** ($C{=}11{,}318\gg512$, rank never binding). That 15‑vs‑2 asymmetry is a rank‑bottleneck signature, and it is the single strongest empirical support for this diagnosis available without new experiments.

**Why CHIMERA is a different fix than more proxies.** Multi‑proxy adds *class‑private* anchors: each is estimated from ~1/15 of one class's images and carries no cross‑class meaning, so nothing about it is computable for an unseen class. CHIMERA adds no anchors; it forces the encoder to compute $K$ **shared** detectors. The realizable prototype set becomes the $K$‑dimensional cone $\{U\gamma:\gamma\in\Delta\}$, **rank $\min(K,512)$ independent of $C$**, and an unseen class is a new $\gamma$ computed by existing detectors from pixels — no new parameters.

**Train/test operator mismatch, resolved.** Every proxy method realises training identity as an index lookup $c\mapsto P_{:,c}$ but test identity as an encoder output. CHIMERA's $\Gamma_{:,c}=\mathbb E_{x\sim c}[a(x)]$ is the *same operator* available at test time on a gallery item. This is an architectural difference from PFML, ProxyAnchor, ProxyNCA++, and SoftTriple, not a loss reweighting.

### 2.2 Cheapest degeneracies, and why each is blocked

**D1 — collapse to multi‑proxy (one private atom per class).** Blocked twice. (i) A one‑hot class code has $H(a)=0<\log 32$, incurring the full $\lambda_e\log 32$ penalty. (ii) A class↔atom perfect matching leaves $K-C$ atoms with zero row mass; the Sinkhorn row constraint $\Gamma\mathbf1=\mathbf1/K$ assigns each such atom KL cost $\to\infty$ in the limit, and finite $\epsilon$ leaves a strictly positive residual that grows as $\log(K/C)$ per unused atom. *This is the degeneracy that would silently reduce CHIMERA to PFML, so it gets two independent blockers.*

**D2 — uniform/constant codes.** Then $p_c=p_{c'}\ \forall c,c'$ and the PA repulsion term is bounded below by $\log(1+|X^-|e^{\alpha\delta})$, which for $\alpha=32,\delta=0.1$ is $\gtrsim\log(1+24.5|X^-|)$ — catastrophic. Blocked.

**D3 — duplicated atoms $u_i=u_j$.** A Bernoulli mask splits the duplicate pair with probability $1/2$; then $\beta$ counts the split mass while $\langle\bar f,q\rangle$ responds to the *combined* direction, so $\mathcal L_{\rm cal}$'s residual is strictly positive in expectation and cannot be driven to zero by any choice of $\bar f$. No explicit coherence penalty is required — the crossover operator *is* the decorrelation test.

**D4 — trivial satisfaction of the calibration law.** If $\kappa$ were learned it would drive to 0 with all similarities equalized. Two fixes: $\kappa$ is **fixed at 0.6**, and $\mathcal L_{\rm sep}$ independently forbids equal‑similarity collapse by requiring both parents to beat a third class by $\delta_v$.

**D5 — atoms encode nuisance (pose, lighting, crop) rather than content.** Blocked by the two‑view code term: both views of one image must map to the same class code and to each other.

**D6 — satisfy calibration by globally shrinking similarities.** $\mathcal L_{\rm PA}$'s attraction pins $\langle f(x),p_{c(x)}\rangle$ near 1 while $\mathcal L_{\rm cal}$ constrains only a *difference*.

---

## 3. Adversarial novelty search — nearest works and mechanism distinctions

**Inside DML**

- **PFML, *Potential Field based Deep Metric Learning*, CVPR 2025** — places continuous attractive/repulsive potential fields around class‑private proxies; CHIMERA has no class‑private parameter at all, its prototypes being Sinkhorn‑balanced compositions of a class‑shared vocabulary.
- **Proxy Anchor / ProxyNCA++ / SoftTriple / Fewer‑is‑More graph proxies** — all learn an index‑addressed per‑class (or per‑class‑multi‑center) free table; CHIMERA's class code is an EMA statistic of the encoder's own output, so train‑ and test‑time identity operators coincide.
- **Proxy Synthesis (AAAI 2021), Embedding Expansion, Metrix (ICLR 2022), Learning to Generate Novel Classes (2022)** — synthesise virtual classes by convex interpolation with a Beta coefficient, or by a learned generator, i.e. a one‑parameter family along a segment; CHIMERA synthesises by **binary crossover over a discrete vocabulary**, giving $2^K$ virtual identities, and imposes a calibration law tying similarity to *inherited atom mass* that a scalar interpolant cannot even express.
- **HIER (CVPR 2023)** — discovers a *tree* of hyperbolic ancestor proxies, so a class is a leaf on one path; CHIMERA discovers a *factorial* code, so a class is a conjunction of many co‑equal atoms and unseen classes occupy new conjunctions, not new leaves.
- **DiVA, BIER/A‑BIER, Divide‑and‑Conquer, S2SD** — partition or ensemble the *embedding* with decorrelation, boosting, or self‑distillation; CHIMERA leaves the embedding monolithic and constrains the *prototype‑generating map* to factor through a balanced shared vocabulary.
- **PA+DADA (AAAI 2024)** — data‑augmented domain adaptation between proxy and sample distributions; CHIMERA changes what a proxy *is*, not how proxy and sample distributions are aligned.

**Outside DML**

- **SwAV / DINO / MSN** — Sinkhorn balances a *sample×prototype* assignment as a self‑supervised pseudo‑label with near‑one‑hot targets; CHIMERA balances a *class×atom* matrix under ground‑truth labels, deliberately holds the code high‑entropy via a floor, and composes prototypes from it.
- **Compositional ZSL (survey 2025; clustering‑based prototypes; Visual Proxy 2025)** — requires state/object attribute annotations and usually a text encoder; CHIMERA discovers its vocabulary from identity labels alone, with no attribute names and no language, which is what makes it legal here.
- **NMF (Lee & Seung 1999) / sparse coding (Olshausen & Field 1996)** — unsupervised parts‑based factorization of the *data* matrix; CHIMERA factorizes the *supervised prototype* set and adds a recombination consistency law with no analogue in dictionary learning.
- **RSC, adversarial dropout, Spectral Decoupling (Pezeshki 2021)** — fight feature suppression by perturbing gradients or muting coordinates; CHIMERA makes the label geometry itself combinatorial so that no small feature subset is ever sufficient.
- **Neural‑collapse transfer analyses (Galanti et al. ICLR 2022; variability‑collapse work 2022‑23)** — diagnostic; CHIMERA is a training method whose predicted signature is *derived from* that diagnosis and is separately falsifiable (F4, F5).

I found no primary source combining: shared atom vocabulary + class×atom marginal balancing + combinatorial crossover synthesis + inherited‑mass calibration. The nearest single hits are Proxy Synthesis (synthesis, but convex) and SwAV (Sinkhorn, but self‑supervised and one‑hot).

---

## 4. Decisive matched‑compute controls

All at identical wall‑clock. CHIMERA costs ≈1.25× epoch time (the 25 % second view), so **baselines get 250 epochs where CHIMERA gets 200**; the 200/200 equal‑epoch comparison is reported alongside.

| # | Control | Isolates | Decisive against |
|---|---|---|---|
| **C3** | Class‑private multi‑proxy with total anchors matched to $K$ (CUB: 3 proxies × 100 classes ≈ 300 ≈ $K{=}256$) | shared vs private anchors | PFML / SoftTriple |
| **C6** | Replace binary mask $M$ with scalar $\lambda\sim\mathrm{Beta}(a,a)$ on the *same* composed prototypes | combinatorial vs convex synthesis | **Proxy Synthesis / Metrix — the nearest work** |
| C1 | Proxy Anchor, same recipe, 200 ep and 250 ep | headline baseline | — |
| C2 | **Local PFML reproduction** (15/15/2 proxies) | frontier inheritance | see §7 ambiguity 1 |
| C4 | Sinkhorn off ($\Gamma$ = plain EMA) | the sharing constraint | — |
| C5 | Chimeras off ($\lambda_r=0$) | the recombination core | — |
| C7 | PA baseline given the same 25 % extra views | view budget | — |
| C8 | $U$ frozen at a random orthonormal frame | learned vocabulary vs random projection | — |
| C9 | $U$ initialised from ImageNet fc rows projected to 512‑D | pretrained‑feature preservation confound | §6 risk |

**C3 and C6 are the two that decide whether the mechanism claim survives.**

---

## 5. Frozen forecasts, thresholds, frontier arithmetic

Lane A, ResNet‑50, 512‑D, 224 px, single view, cosine, 200 epochs, **5 seeds**, mean ± sd. Frozen now.

| dataset | reference (audited) | CHIMERA forecast | 80 % interval | Δ | σ of Δ | z |
|---|---|---|---|---|---|---|
| CUB‑200‑2011 | PFML 0.734 ± 0.003 | **0.748 ± 0.005** | [0.735, 0.760] | +0.014 | 0.0026 | **5.4** |
| Cars196 | PFML 0.927 ± 0.003 | **0.933 ± 0.004** | [0.924, 0.941] | +0.006 | 0.0022 | **2.7** |
| SOP | PFML 0.829 ± 0.002 | **0.833 ± 0.003** | [0.826, 0.840] | +0.004 | 0.0016 | **2.5** |
| In‑Shop | PA+DADA 0.930 (seeds unreported) | 0.932 ± 0.004 | [0.927, 0.937] | +0.002 | — | **no crossing claimed** |

σ of Δ = $\sqrt{(s_1^2+s_2^2)/5}$. In‑Shop is reported but **not** claimed as a crossing: the reference's uncertainty is unreported, so the arithmetic cannot be closed.

**Mechanism‑specific prediction (the part that makes this falsifiable beyond the score):** gain must be ordered **CUB ≈ Cars ≫ SOP**, because $C{=}11{,}318$ on SOP means the rank bottleneck was never binding there.

**Pre‑registered falsification thresholds**

- **F1** CUB 5‑seed mean < 0.740 → no crossing → reject.
- **F2** C3 (anchor‑matched class‑private proxies) within 0.003 of CHIMERA on CUB → the *sharing* mechanism is not operative → reject the mechanism claim even if the number is good.
- **F3** C5 (chimeras off) costs < 0.004 R@1 on CUB → the recombination core is inert; the method reduces to "Sinkhorn‑balanced shared proxies" and must be rescoped and renamed.
- **F4** Effective rank (participation ratio of the *test*‑class‑mean covariance) does not rise ≥1.5× over C1 → the causal story is wrong regardless of R@1.
- **F5** SOP gain > CUB gain → rank‑bottleneck mechanism falsified (prediction is the opposite ordering).
- **F6** C6 (convex mixing) within 0.003 of CHIMERA → the method is a reparameterisation of Proxy Synthesis; withdraw the novelty claim.

F4–F6 can kill the *mechanism* independently of the *score*. That asymmetry is deliberate.

---

## 6. Cost, and honest risks

**Parameters:** $U$ = 512×256 = 131 k (0.51 % of ResNet‑50's 25.6 M); $\Gamma$ = 256×$C$ non‑gradient buffer. **All train‑only; deployment is one ResNet‑50 and one 512‑D descriptor, byte‑identical in form to the baseline.**
**Compute:** $U^\top f$ ≈ 1.3×10⁵ FLOPs/image vs 4.1 GFLOPs backbone (<0.01 %); Sinkhorn 3 iters on ≤1024×11,318; 64 chimeras/step = 64 matvecs. **The only real cost is the 25 % second view: ≈1.25× epoch time, ≈1.15× peak memory.** For reference, PA+DADA reports 1.06×/1.01× — CHIMERA is more expensive, which is why every control runs at matched wall‑clock.

**Risks I will not paper over**

1. **Five auxiliary scales** ($\lambda_c,\lambda_r,\lambda_e,\tau/k^\ast,\kappa$). Each blocks a named degeneracy, but the joint optimisation is fragile and the $\lambda_r$ warmup is load‑bearing. This is the most likely cause of a null result.
2. **$\Gamma$ staleness on SOP/In‑Shop.** With ~5 images per SOP class, each class's code is refreshed rarely; the count‑debiased momentum and the frozen‑encoder initialisation are mitigations, not guarantees. **CHIMERA may fail worst exactly where I forecast the smallest gain, for a second and unrelated reason.**
3. **My own variance.** I forecast sd 0.005 on CUB vs PFML's 0.003; on Cars and SOP my forecast sd exceeds the forecast gain's margin comfort. The Cars/SOP crossings are 2.5–2.7σ, i.e. suggestive, not decisive, at 5 seeds — 10 seeds would be needed there.
4. **Pretrained‑feature‑preservation confound.** A learned atom vocabulary may preferentially recover ImageNet‑1K class directions, in which case part of the gain is "better preservation of pretrained features," not novel composition. C8 and C9 quantify this; I cannot rule it out a priori.
5. **Benchmark/contamination.** ImageNet‑1K pretraining overlaps CUB and Cars196 semantics (bird and automobile synsets). This is inherited equally by every Lane A baseline and is the standard‑split status quo; CHIMERA adds no data, no generated images, no text, no transduction, no reranking, no test‑gallery fitting. The only *new* contamination surface is (4), which is a mechanism‑attribution risk rather than a leakage risk.

---

## 7. Unresolved source ambiguities (stated, not hidden)

1. **PFML's full recipe** (optimizer, lr, batch, augmentation, proxy‑lr multiplier, BN freezing, head normalisation) is not available to me in this blind pass. I therefore **reproduce its mathematical reduction nowhere and inherit its frontier nowhere** — control **C2**, a matched local PFML reproduction, is a hard precondition on any crossing claim. Its published 0.734/0.927/0.829 are treated as targets, not as a baseline I may build on.
2. **Proxy Anchor's exact hyperparameters.** I state $\alpha=32$, $\delta=0.1$, proxy lr = 100× backbone, batch 180, AdamW. I am confident on $\alpha,\delta$; **less confident** on the lr multiplier and on the per‑dataset batch size. These must be checked against the primary source before the recipe is frozen, because L1 inherits them verbatim.
3. **PA+DADA In‑Shop 0.930** — seeds and uncertainty unreported; only 1.06× epoch / 1.01× memory disclosed. Hence no In‑Shop crossing claim.
4. **Whether PFML's 15 proxies are freely learned or constrained** (tied norms, initialisation from class clusters). This changes how C3 must match anchor counts, and therefore changes what F2 tests.
5. **In‑Shop split conventions** — I assume 3,997 train / 3,985 test classes with the standard query and gallery lists.
6. $K$ and $k^\ast$ are set by argument ($K\gtrsim2C$ on CUB/Cars, perplexity 32), not by any measurement I can make blind; they are the most likely hyperparameters to need retuning, and a retune invalidates the frozen forecast unless done on a train‑split holdout of *training* identities only.

---

**Summary of what is new:** identity is represented as a conjunction over a class‑shared, marginal‑balanced atom vocabulary rather than as a lookup into a class‑private table; virtual identities are produced by binary crossover over that vocabulary rather than by convex interpolation; and a fixed‑slope calibration law ties chimeric similarity to inherited atom mass, which is what forces atoms to be individually readable and is what makes the collapse‑to‑multi‑proxy degeneracy provably costly.

Sources:
- [Potential Field Based Deep Metric Learning (CVPR 2025 poster)](https://cvpr.thecvf.com/virtual/2025/poster/33305)
- [Proxy Synthesis: Learning with Synthetic Classes for DML (AAAI 2021)](https://arxiv.org/pdf/2103.15454)
- [Learning to Generate Novel Classes for Deep Metric Learning](https://arxiv.org/pdf/2201.01008)
- [HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.pdf)
- [Towards Improved Proxy-based DML via Data-Augmented Domain Adaptation (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/download/29400/30645)
- [On the Role of Neural Collapse in Transfer Learning](https://arxiv.org/abs/2112.15121)
- [Neural Collapse: A Review on Modelling Principles and Generalization](https://arxiv.org/pdf/2206.04041)
- [Compositional Zero-Shot Learning: A Survey](https://arxiv.org/html/2510.11106)
- [Learning Clustering-based Prototypes for Compositional Zero-shot Learning](https://arxiv.org/pdf/2502.06501)
- [Fewer is More: Deep Graph Metric Learning with Fewer Proxies](https://arxiv.org/pdf/2010.13636)

Note: several claude.ai connectors (Gmail, Calendar, Drive, IBKR) reported needing authorization this session — irrelevant to this task, but they'd need authorizing via claude.ai connector settings if wanted later.
