# Recovered proposer terminal output

<!-- The fallback process exited after emitting this complete proposal. The source disclosed its limited two-query web verification; this is preserved for audit, not treated as sufficient Gate 2 evidence. -->

# LORD — Label-Orthogonal Rank Distillation from a diffusion teacher

**Lane: A** (ResNet-50, 512-D normalized global descriptor, ~224 px, single-view cosine, 200 epochs). All numbers, baselines and comparisons below are Lane A. Lane B references (AdvRF, VAPNet) are never used as targets.

## 0. The one-paragraph mechanism

Test-time diffusion/reranking demonstrably repairs nearest-neighbour retrieval on *unseen* classes, and is exactly what deployment forbids. LORD makes reranking unnecessary by driving the parametric metric toward a **diffusion fixed point**: the embedding is trained so that its *one-hop* cosine geometry already equals its *sharpened multi-hop* (personalized-PageRank) geometry — but **only in the directions the labels leave unspecified**. Ground-truth labels supply the positive/negative partition; the diffusion teacher supplies only the *within-positive* and *within-negative* orderings, injected through group-restricted softmax KLs that are provably invariant to the partition. The teacher is a nonparametric graph functional of a stop-gradient memory bank of training embeddings: no extra network, no extra forward pass, no extra proxies, nothing at deployment.

---

## 1. Executable mathematics

### 1.1 Network and deployment operation

Backbone $F_\theta$: ResNet-50, ImageNet-1K init, stride-32 conv5 map $F\in\mathbb R^{2048\times7\times7}$.

$$h(x)=\mathrm{GAP}(F_\theta(x))+\mathrm{GMP}(F_\theta(x))\in\mathbb R^{2048},\qquad
z(x)=\frac{W_e\,h(x)}{\lVert W_e\,h(x)\rVert_2}\in S^{511},\quad W_e\in\mathbb R^{512\times2048}.$$

**Train op:** RandomResizedCrop(224, scale (0.16,1.0)) + horizontal flip; ImageNet mean/std. Nothing else. **Test op:** Resize 256 → CenterCrop 224, one forward, $z(x)$, cosine 1-NN over the gallery. No memory bank, no graph, no diffusion, no reranking, no gallery statistics. Deployment is byte-identical to a Proxy-Anchor model.

### 1.2 Base loss (Proxy Anchor, reproduced)

One L2-normalized proxy $p_c\in S^{511}$ per training class, $P=\{p_c\}_{c=1}^{C}$, $P^+$ the proxies of classes present in batch $\mathcal B$, $X_p^+$ its samples of class $p$, $X_p^-=\mathcal B\setminus X_p^+$, $s(x,p)=z(x)^\top p$:

$$\mathcal L_{\mathrm{PA}}=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\!\!\sum_{x\in X_p^+}\!\! e^{-\alpha(s(x,p)-\delta)}\Big)
\;+\;\frac{1}{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X_p^-}\!\! e^{\alpha(s(x,p)+\delta)}\Big),$$

with $\alpha=32,\ \delta=0.1$ (paper values, **from memory — flagged in §7**).

### 1.3 Memory bank (zero extra forward passes)

$M\in\mathbb R^{N\times512}$, one row per official training image. Initialized after a 1-epoch warm-up (backbone frozen, $W_e$ + proxies only) by one no-grad pass. Thereafter updated online with the *student's own* stop-gradient embeddings: after each step, $M[i]\leftarrow \mathrm{sg}(z_i)$ for $i\in\mathcal B$. So $M$ is a one-epoch-stale snapshot — an implicit temporal EMA, relying on the XBM slow-drift property. Cost: $N{\times}512{\times}4$ B (CUB 12 MB, Cars 17 MB, SOP 122 MB) and **zero** additional forward/backward passes.

### 1.4 Diffusion teacher (per anchor, per step)

For anchor $i$, all quantities from $M$ (stop-grad throughout):

1. **Local subgraph.** Frontier $\mathcal F_0=\{i\}$; for $t=0,1,2$: $\mathcal F_{t+1}=\bigcup_{u\in\mathcal F_t}\mathrm{kNN}_k(u;M)$, $k=32$; node set $V_i=\bigcup_t\mathcal F_t$ capped at $M_{\text{cap}}=512$ by descending cosine to $M[i]$.
2. **Affinity.** $A_{uv}=\max(0,\,M[u]^\top M[v])^{\beta}$ for $v\in\mathrm{kNN}_k(u)$, $\beta=3$; symmetrize $A\leftarrow\max(A,A^\top)$; $A_{uu}=0$; $\mathcal A=D^{-1/2}AD^{-1/2}$, $D=\mathrm{diag}(A\mathbf 1)$.
3. **Truncated personalized PageRank.** $\displaystyle \tilde f_i=(1-\alpha_d)\sum_{t=0}^{T}\alpha_d^{\,t}\,\mathcal A^{t}e_i$, $T=3$, $\alpha_d=0.85$ (three sparse mat-vecs on a $\le512$-node subgraph).
4. **Teacher score.** $\tilde s_{ij}=\log\big(\tilde f_i[j]+\varepsilon\big)$, $\varepsilon=10^{-8}$. With this choice, $\mathrm{softmax}_j(\tilde s_{ij}/\tau_t)$ is the PPR mass renormalized after raising to the power $1/\tau_t$ — sharpening is literally a power on diffusion mass.

**Comparison sets.** $\mathcal P_i$: $m_p=8$ same-class memory rows ($\le4$ from the top of $\tilde f_i$, remainder uniform from the class); require $|\mathcal P_i|\ge2$ else drop the positive term for $i$ (binding on SOP/In-Shop, ~5.6 images/class → effective $m_p\approx4$). $\mathcal N_i$: $m_n=64$ different-class rows, 32 highest-$\tilde f_i$ (manifold-hard) + 32 uniform.

### 1.5 The two group-restricted distillation terms

$$q^{\mathcal G}_{ij}=\frac{e^{\tilde s_{ij}/\tau_t^{\mathcal G}}}{\sum_{j'\in\mathcal G_i}e^{\tilde s_{ij'}/\tau_t^{\mathcal G}}},\qquad
p^{\mathcal G}_{ij}=\frac{e^{\,z_i^\top M[j]/\tau_s}}{\sum_{j'\in\mathcal G_i}e^{\,z_i^\top M[j']/\tau_s}},\qquad \mathcal G\in\{\mathcal P,\mathcal N\}$$

$$\boxed{\;\mathcal L=\mathcal L_{\mathrm{PA}}+\lambda_p(e)\cdot\frac1B\sum_i \mathrm{KL}\!\big(q^{\mathcal P}_{i\cdot}\,\|\,p^{\mathcal P}_{i\cdot}\big)+\lambda_n(e)\cdot\frac1B\sum_i \mathrm{KL}\!\big(q^{\mathcal N}_{i\cdot}\,\|\,p^{\mathcal N}_{i\cdot}\big)\;}$$

**Gradient path.** $q$ is a constant (teacher, stop-grad). $M[j]$ is stop-grad. Gradient enters **only** through $z_i$, the current strongly-augmented view of the anchor:
$$\frac{\partial \mathrm{KL}}{\partial z_i}=\frac{1}{\tau_s}\sum_{j\in\mathcal G_i}\big(p^{\mathcal G}_{ij}-q^{\mathcal G}_{ij}\big)\,M[j],$$
then through the $S^{511}$ projection $\partial z/\partial \hat z=(I-zz^\top)/\lVert W_eh\rVert$, then $W_e$, then $\theta$. $\mathcal L_{\mathrm{PA}}$ additionally reaches the proxies $P$. No gradient reaches $M$, $A$, or $\tilde f$.

### 1.6 Hyperparameters and schedules (frozen)

| object | value |
|---|---|
| $\alpha,\delta$ (PA) | 32, 0.1 |
| optimizer | AdamW, backbone/$W_e$ lr $10^{-4}$, proxy lr $10^{-2}$, wd $10^{-4}$, $\beta=(0.9,0.999)$ |
| schedule | 200 epochs, 1 warm-up epoch (backbone frozen), cosine lr decay to $10^{-6}$ |
| batch | 180, balanced sampler 30 classes × 6 images (SOP/In-Shop: 45 × 4) |
| $\tau_s$ | 0.05 (fixed) |
| $\tau_t^{\mathcal P},\tau_t^{\mathcal N}$ | init 0.5, **controlled online** (§2.2) |
| $\lambda_p,\lambda_n$ | linear ramp $0\to(1.0,\,0.5)$ over epochs 2–20, constant thereafter |
| graph | $k=32$, $T=3$, $\alpha_d=0.85$, $\beta=3$, $M_{\text{cap}}=512$ |
| sets | $m_p=8$, $m_n=64$ (32 hard + 32 uniform) |

The ramp exists because a randomly-initialized embedding graph is meaningless; $\lambda$ is zero while the memory is still garbage.

**Note on normalization, per the protocol's warning:** $\tau_s$ is *not* a harmless rescale. It multiplies the distillation gradient by $1/\tau_s$ while AdamW's decoupled weight decay on $W_e$ is scale-fixed, so $\tau_s$ shifts the equilibrium pre-normalization radius $\lVert W_eh\rVert$ and hence the effective learning rate on the sphere. It is a load-bearing hyperparameter and appears explicitly in the stability condition of §2.2. It must be held identical across all arms of every control.

---

## 2. One causal zero-shot error mode, and proof-level attack on the cheapest degeneracies

### 2.1 The error mode: manifold-inconsistent impostor dominance

Fix an unseen class $y$, query $q$, gallery $\mathcal Gal$. R@1 fails iff
$$\exists\,x'\in\mathcal Gal,\ y(x')\ne y:\quad z(q)^\top z(x') > \max_{x\in\mathcal Gal,\,y(x)=y} z(q)^\top z(x).$$
Empirically on CUB/Cars, a large share of these top-1 impostors are *manifold-far*: they are not reachable from $q$ by a short path of high-affinity steps through the data, which is exactly why test-time diffusion/reranking repairs them. The causal reason ERM permits this: the label loss on $C$ seen classes constrains only the **partition** $\{$same, different$\}$. Within the positive set it says "as close as possible" (a single point is optimal → neural collapse), and within the negative set it says "as far as possible" (all impostors interchangeable). So the *graded local geometry* of $f_\theta$ — the very thing that determines which impostor wins the argmax on classes never labelled — receives **no gradient signal at all** under any purely label-derived loss. It is fixed by architecture, initialization and weight decay. LORD supplies gradient exactly there, and only there.

Transfer assumption, stated plainly: diffusion-consistency enforced on the seen-class manifold is a property of the *function* $f_\theta$, not of the seen labels, so it is expected to hold on unseen regions of the same image domain. This is an assumption, not a theorem, and it is the main scientific risk of the method (§6).

### 2.2 Degeneracy 1 — within-class collapse (the cheapest solution). Attacked by an instability condition, not by an assertion.

Under the class-code solution $z_i=\mu_{y_i}$ for all $i$, the student's within-group softmax $p^{\mathcal P}_{i\cdot}$ is uniform. Naively one says $\mathrm{KL}(q\|\mathcal U)=\log m_p-H(q)>0$ and declares collapse penalized. **That argument is wrong**, because if the memory has collapsed too, $\tilde f_i$ is uniform on $\mathcal P_i$ and the loss is $0$: collapse is a fixed point. The correct question is whether it is *stable*.

Linearize. Write $z_j=\mu_y+u_j$, $u_j\perp\mu_y$, $\sum_j u_j=0$, $\lVert u\rVert$ small. Memory cosines within the class: $M[i]^\top M[j]\simeq 1-\tfrac12\lVert u_i-u_j\rVert^2$. Let $g_{ij}=-\tfrac{\beta}{2}\lVert u_i-u_j\rVert^2$ be the first-order log-affinity, and $\hat g_{ij}=g_{ij}-\bar g_i$ its within-group centering. Two facts:

- **Teacher side.** $\tilde f_i[j]\propto 1+\kappa\,g_{ij}+O(\lVert u\rVert^4)$ with $\kappa=\kappa(T,\alpha_d)>0$ from the PPR series, so after sharpening $q^{\mathcal P}_{ij}\simeq \tfrac1{m_p}\big(1+\tfrac{\kappa}{\tau_t}\hat g_{ij}\big)$.
- **Student side.** $z_i^\top M[j]-\overline{(\cdot)}\simeq \tfrac1\beta \hat g_{ij}$, so $p^{\mathcal P}_{ij}\simeq \tfrac1{m_p}\big(1+\tfrac{1}{\beta\tau_s}\hat g^{\text{cur}}_{ij}\big)$.

Because $q$ is **stop-gradient**, the KL is a quadratic pull of the *current* dispersion onto a *frozen* target, not a symmetric penalty on a shared variable. Its minimizer is
$$\hat g^{\text{cur}} = \rho\;\hat g^{\text{mem}},\qquad \rho \;=\; \frac{\kappa(T,\alpha_d)\,\beta\,\tau_s}{\tau_t}.$$
Across memory refreshes this is the linear map $\hat g^{(n+1)}=\rho\,\hat g^{(n)}$.

> **Proposition (collapse instability).** The within-class collapsed configuration $\hat g=0$ is a repelling fixed point of LORD's positive term iff $\rho>1$. For $\rho>1$ any infinitesimal within-class structure is amplified geometrically until balanced by $\mathcal L_{\mathrm{PA}}$'s contraction; the equilibrium dispersion is where PA's intra-class pull equals LORD's expansion.

Had I fixed the target with a *symmetric* penalty on both sides (no stop-grad), the loss would be $\propto(\tfrac{\kappa}{\tau_t}-\tfrac{1}{\beta\tau_s})^2\hat g^2$, minimized at $\hat g=0$ — collapse would be *attracting*. The stop-gradient is therefore not an implementation convenience; it is what flips the sign of the mechanism.

**Enforcement, not hope.** $\kappa$ is unknown analytically, so $\rho$ is measured and controlled. Per epoch, for anchors with $|\mathcal P_i|\ge3$, let $\hat a_i$ = within-group centered memory cosines and $\hat b_i = (\tau_s/\tau_t)\,\tilde s_{i\cdot}$ centered (the target-implied cosines; matching softmaxes means matching logits up to a group constant). Define and control:
$$\hat\rho=\frac{\mathbb E_i\lVert \hat b_i\rVert_2}{\mathbb E_i\lVert \hat a_i\rVert_2},\qquad \tau_t\leftarrow \mathrm{clip}\Big(\tau_t\big(\hat\rho/\rho^\ast\big)^{\eta},\;0.05,\;2\Big),\quad \rho^\ast=1.2,\ \eta=0.1.$$
Same controller, independent $\tau_t^{\mathcal N}$, for negatives. $\hat\rho$ is logged every epoch; if it leaves $[1.05,1.5]$ for 10 consecutive epochs the run is declared a mechanism failure (§5).

### 2.3 Degeneracy 2 — target duplicates the labels (the term does nothing)

> **Proposition (exact label-orthogonality).** For any $a,b\in\mathbb R$, the substitution $\tilde s\mapsto \tilde s+a\mathbf 1_{\mathcal P_i}+b\mathbf 1_{\mathcal N_i}$ leaves $q^{\mathcal P}$ and $q^{\mathcal N}$ — and therefore both KL terms and all their gradients — exactly invariant.

Hence the distillation carries **zero** information about the positive/negative partition. It cannot act as a pseudo-label, cannot re-derive the class structure, cannot be traded off against $\mathcal L_{\mathrm{PA}}$, and cannot inject label noise into the partition. Every bit it contributes is in the two orderings that no label-derived loss constrains. This is also what separates it from every similarity-KD method I know of (§3).

### 2.4 Degeneracy 3 — instance memorization

The student could satisfy a frozen ordering by memorizing idiosyncratic per-image evidence. Three structural blocks: (i) $z_i$ is a **fresh strong crop** while $M[j]$ is a stale, differently-cropped embedding, so any satisfying feature must survive RandomResizedCrop; (ii) targets are regenerated from a memory that turns over every epoch, so the target is non-stationary; (iii) the loss is on *relative* order within a group, which is invariant to any per-anchor bias the network could memorize. Empirically falsified by control C7 (frozen targets after epoch 50) and C6 (within-group permuted targets), §4.

### 2.5 Degeneracy 4 — smoothing collapse of the target itself

Diffusion is a low-pass operator; naive repeated self-distillation of smoothed targets converges to the leading eigenvector (uniform similarity). Three blocks: PPR is *personalized* — the $(1-\alpha_d)e_i$ seed term keeps $\Theta(1-\alpha_d)=0.15$ of the mass on the anchor and prevents convergence to the stationary distribution; sharpening $\tau_t<1$ expands non-uniform modes, and the §2.2 controller keeps that expansion above the smoothing contraction ($\rho^\ast=1.2>1$); and $\mathcal L_{\mathrm{PA}}$ independently maintains the partition.

### 2.6 The fixed point (what the method actually converges to)

$$\mathrm{softmax}_{\mathcal P_i}\!\big(z_i^\top M/\tau_s\big)=\mathrm{softmax}_{\mathcal P_i}\!\big(\log\mathrm{PPR}_i/\tau_t\big),\quad\text{likewise on }\mathcal N_i.$$
In words: **the one-hop cosine geometry equals the sharpened multi-hop manifold geometry, within groups.** A metric with this property gains nothing from diffusion reranking — which is exactly the deployment constraint we must satisfy.

---

## 3. Adversarial novelty search — nearest works and one-sentence mechanism distinctions

*(Two works re-verified this session; the rest from memory and marked accordingly.)*

**Inside DML / retrieval**

1. **STML, CVPR 2022** *(verified)* — replaces labels entirely with a smoothed contextualized similarity as the whole unsupervised supervision; **LORD** keeps ground-truth labels as the *sole* source of the partition and adds a target that is provably invariant to that partition (§2.3), so the two objectives cannot substitute for each other.
2. **Embedding Transfer with Label Relaxation, CVPR 2021** *(memory)* — distills a *different, already-trained source model's* full similarity distribution into a target embedding; LORD's teacher is a parameter-free graph functional of the student's own memory and its target is group-restricted, so no partition information is transferred.
3. **XBM, CVPR 2020** *(memory)* — a memory bank supplying more negatives to the *same* label-based loss; LORD uses the memory as a *graph substrate* for a multi-hop operator, and its loss is orthogonal to the label information the memory is normally used to amplify.
4. **Diffusion / manifold reranking (Iscen et al. CVPR 2017), k-reciprocal encoding (Zhong et al. CVPR 2017), α-QE** *(memory)* — transductive operators applied to the *test gallery* at inference; LORD applies the operator only to official training images and distills its effect into the parametric metric, leaving inference a bare cosine 1-NN.
5. **Mining on Manifolds, CVPR 2018** *(memory)* — uses manifold-vs-Euclidean discrepancy to *select* hard training pairs for a binary loss; LORD does no selection and regresses the full graded within-group ordering.
6. **S2SD, ICML 2021** *(memory)* — distills from an auxiliary *higher-dimensional* embedding branch (extra parameters, extra heads); LORD adds no parameters, no branch and no forward pass — the "high-capacity teacher" is the training manifold itself.
7. **Relational KD / Similarity-Preserving KD** *(memory)* — a separate teacher network's pairwise structure is matched globally; LORD matches only within-group orderings of a self-generated diffused structure.
8. **SoftTriple (ICCV 2019), ProxyGML, PFML (CVPR 2025)** *(memory / PFML per prompt)* — preserve intra-class structure with $K$ **free per-class centers** (PFML: 15 on CUB/Cars, 2 on SOP); LORD preserves intra-class structure with **zero per-class parameters**, from a data-derived ordering, and additionally shapes the inter-class ordering — which is why the two are hypothesized complementary rather than competing.
9. **HIER, CVPR 2023** *(memory)* — shapes inter-class geometry with learned hierarchical proxies in hyperbolic space; LORD shapes it from a diffusion operator on the training manifold, with no hierarchy parameters and no curvature change.
10. **Proxy Synthesis (AAAI 2021), Embedding Expansion (CVPR 2020)** *(memory)* — synthesize virtual classes/points to simulate unseen identities; LORD synthesizes nothing and instead constrains the metric's local geometry.

**Outside DML**

11. **APPNP / PPR-GNN, ICLR 2019** *(memory)* — personalized PageRank as an *inference-time* propagation layer inside a graph network; LORD uses PPR purely as a *supervisory target generator* whose operator never appears at inference.
12. **DINO / BYOL sharpening-and-centering** *(memory)* — sharpening prevents collapse of an unsupervised invariance objective, tuned by hand; LORD ties sharpening to a *measured* expansion ratio $\hat\rho$ with an explicit instability condition $\rho>1$ and a closed-loop controller (§2.2).
13. **Diffusion maps (Coifman & Lafon, 2006)** *(memory)* — builds an embedding *from* the diffusion operator's spectrum; LORD leaves the embedding parametric and instead imposes agreement between one-hop and multi-hop geometry as a training constraint.
14. **RankDistil / ranking distillation** *(memory)* — distills a teacher ranker's top-$k$ list; LORD's distillation is *group-conditional*, i.e. deliberately restricted to the exact complement of the information the relevance labels already provide.

**Honest assessment of the closest threat.** STML is the work a reviewer should press hardest. The mechanism boundary is sharp and testable, not rhetorical: STML's target *is* the supervision (remove labels and it still trains); LORD's target is by construction incapable of supplying supervision (Prop. 2.3 — the partition is unrecoverable from it), and removing $\mathcal L_{\mathrm{PA}}$ leaves an objective with no notion of class at all. Control C3 (§4) operationalizes this boundary.

---

## 4. Decisive matched-compute controls

All arms: identical backbone, recipe, epochs, augmentation, batch size, sampler, seeds (5), and identical $\tau_s$. Wall-clock differences below 5% are equalized by padding the cheaper arm's step.

| # | Arm | Isolates | Prediction if LORD's claimed mechanism is real |
|---|---|---|---|
| **C0** | PA only | reference | baseline |
| **C1** | PA + memory bank as extra negatives (XBM-style), no distillation | memory itself | recovers ≤ 25% of LORD's gain |
| **C2** | PA + LORD with **$T=0$** (raw memory cosine target, no diffusion), all else identical | **the multi-hop operator** | **recovers ≤ 40% of the gain — this is the decisive control**; if it recovers ≥70%, the mechanism is plain self-distillation, not manifold internalization |
| **C3** | PA + diffusion KD with a **single softmax over $\mathcal P_i\cup\mathcal N_i$** (no group restriction) | label-orthogonality | strictly *worse* than LORD and possibly worse than C0, because the target then partially re-encodes (and corrupts) the partition |
| **C4a/b** | positive term only / negative term only | term decomposition | each ≈ 40–70% of full; positive term dominates on CUB/Cars, negative term dominates on SOP |
| **C5** | PA with $K{=}8$ proxies/class, matched cost | "intra-class structure preservation" as a simpler occupied explanation | C5 alone < PA+LORD; **and C5+LORD > both**, i.e. non-redundant |
| **C6** | LORD with within-group target orderings **randomly permuted** (magnitudes preserved) | is the gain just target noise / implicit smoothing? | ≤ 15% of the gain; if ≥40%, the mechanism claim is dead |
| **C7** | targets frozen after epoch 50 | bootstrapping vs. one-shot prior | loses ≥ 50% of the gain |
| **C8** | controller off, $\tau_t$ fixed at the controller's time-average | is §2.2 load-bearing? | loses 20–50%, and $\hat\rho$ drifts below 1 with measurable within-class dispersion decay |
| **C9** | LORD on a **random** kNN graph (shuffled memory rows) | is the graph carrying semantics? | ≤ 5% of the gain |
| **D1** | *diagnostic only:* diffusion-consistency gap and test-time-diffusion headroom measured on a **held-out split of 20% of the training identities** | how much reranking headroom was internalized | headroom shrinks ≥ 40% relative to C0 |

D1 uses **training identities only** — never test images, never the evaluation gallery. All hyperparameter selection is done on that same held-out training-identity split.

---

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic — Lane A

All: ResNet-50, 512-D, 224 px, 200 epochs, single-view cosine R@1, **mean ± std over 5 seeds**. Frozen before any experiment.

| Dataset | C0: PA (my repro) | **PA + LORD** | C5: PA-K8 (my repro) | **PA-K8 + LORD** |
|---|---|---|---|---|
| CUB-200-2011 | 0.693 ± 0.004 | **0.716 ± 0.005** | 0.707 ± 0.005 | **0.727 ± 0.006** |
| Cars196 | 0.879 ± 0.004 | **0.898 ± 0.004** | 0.891 ± 0.005 | **0.909 ± 0.005** |
| SOP | 0.798 ± 0.002 | **0.811 ± 0.002** | 0.804 ± 0.002 | **0.816 ± 0.003** |

Claimed matched deltas: CUB **+2.3**, Cars **+1.9**, SOP **+1.3** over PA; **+2.0 / +1.8 / +1.2** over the matched multi-proxy control C5.

### Frontier-crossing arithmetic (stated plainly, including where I fall short)

Lane-A references: **PFML** CUB 0.734 ± 0.003, Cars 0.927 ± 0.003, SOP 0.829 ± 0.002 (5 runs). Matched-cost control rows: DADA CUB 0.729, Cars 0.921, SOP 0.810.

- **CUB:** best forecast 0.727 ± 0.006 vs 0.734 ± 0.003. Deficit **−0.007**; pooled $\sigma=\sqrt{0.006^2+0.003^2}=0.0067$, so **−1.0σ — I forecast a miss, within noise of a tie.** Also below DADA's 0.729 by 0.002.
- **Cars:** 0.909 ± 0.005 vs 0.927 ± 0.003. Deficit **−0.018 ≈ −3.1σ. Clear miss.**
- **SOP:** 0.816 ± 0.003 vs 0.829 ± 0.002. Deficit **−0.013 ≈ −3.6σ. Clear miss**, though above DADA's 0.810.

**I do not forecast that LORD crosses the Lane-A frontier standalone.** The honest claim is a mechanism claim: LORD delivers a +1.2 to +2.3 point matched-compute, matched-recipe gain at ~1.05× training cost and 1.00× deployment cost, from a supervision channel (within-group ordering) that is provably disjoint from what every label-derived DML loss uses. The conditional frontier claim — PFML + LORD ≥ 0.745 CUB / 0.936 Cars / 0.836 SOP — follows from the non-redundancy prediction in C5, but **I cannot make it**, because PFML's loss reduction and training recipe were not available to me in this session and I therefore have no matched reproduction of it. Per the protocol, I refuse to inherit that frontier. Closing the Cars/SOP gaps requires PFML's multi-proxy machinery, not more of LORD.

### Falsification thresholds (any one kills the method)

1. **Primary:** mean Δ over 5 seeds < **+1.0** on CUB **or** < **+0.8** on Cars, under matched recipe/compute vs C0.
2. **Mechanism:** control **C2** ($T=0$, no diffusion) recovers **≥70%** of the gain → this is self-distillation, not manifold internalization.
3. **Mechanism:** control **C6** (permuted targets) recovers **≥40%** of the gain → the gain is target noise/smoothing.
4. **Mechanism:** control **C9** (random graph) recovers **≥25%** → the graph is not carrying semantics.
5. **Stability:** $\hat\rho<1.0$ for 10 consecutive epochs while within-class dispersion monotonically decays → §2.2's instability condition does not bind and the collapse analysis is wrong.
6. **Orthogonality:** control **C3** (ungrouped softmax) ≥ LORD → the group restriction is decorative and Prop. 2.3 is not the operative mechanism.
7. **Non-redundancy:** C5+LORD ≤ max(C5, PA+LORD) + 0.003 → LORD and multi-proxy are the same mechanism, and the conditional PFML+LORD claim is void.
8. **Cost:** epoch time > 1.15× C0 → the "free teacher" claim fails and the method must be re-benchmarked against equal-cost longer training.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** Extra forward/backward passes: **zero**. Graph work per step: 180 anchors × 3 sparse mat-vecs on a ≤512-node, ≤32-degree subgraph ≈ $9\times10^6$ sparse MACs — under 1% of a ResNet-50 step. Memory: $N{\times}512{\times}4$ B (CUB 12 MB, Cars 17 MB, In-Shop 53 MB, SOP 122 MB) plus a kNN index. One extra no-grad pass at warm-up (≈0.3 epoch, amortized to 0.15% over 200 epochs). **Forecast: 1.04–1.07× epoch time, 1.02–1.05× peak memory** — the same envelope as the PA+DADA reference (1.06× / 1.01×). If measured cost exceeds 1.15×, threshold 8 fires.

**Deployment cost.** Exactly 1.00×: one model, one view, one 512-D vector, cosine 1-NN. No memory bank, no graph, no reranking, no gallery statistics — the transduction ban is satisfied structurally, not by policy.

**Risks.**
- *Transduction boundary.* The graph is built **only** over official training images and only at train time. This is the same class of operation as XBM/cluster-based DML. No test or gallery embedding ever enters $M$. The one place this could be violated by sloppy implementation — hyperparameter selection — is handled by selecting only on a held-out split of *training identities*.
- *ImageNet pretraining confound.* ImageNet-1K contains many bird and some car categories overlapping CUB/Cars semantics. This is permitted by the protocol and affects every arm equally, but it means absolute CUB/Cars numbers are not evidence of pure zero-shot transfer. Deltas are the meaningful quantity.
- *Benchmark noise.* CUB's test split (100 unseen classes, ~5.9k images) gives per-seed R@1 σ ≈ 0.4–0.6 points. A single-seed +1.0 report is uninformative; hence the 5-seed requirement on every arm, and hence the deficits in §5 that are ~1σ are reported as ties-within-noise rather than as wins.
- *SOP structural risk.* SOP's ~5.6 images per class caps $m_p$ at ~4, so the positive term is weak there; and many SOP positives are near-duplicate photographs, which makes the diffusion graph locally degenerate (near-clique). This is my honest reason for the smaller SOP forecast, and control C4a/b is designed to expose it.
- *Slow-drift assumption.* The teacher's validity depends on memory embeddings being nearly current. If the lr schedule causes rapid drift (e.g. after a warm restart, which I therefore do not use), the targets become stale and the method degrades toward C6. Monitored via mean $\lVert z_i - M[i]\rVert$ per epoch; abort threshold 0.3.
- *The main scientific risk*, restated: diffusion-consistency on the seen-class manifold may simply not transfer to unseen classes, in which case LORD becomes an elaborate intra-class-structure regularizer and C5 (multi-proxy, matched cost) will match it. Threshold 7 is the test.

---

## 7. Unresolved source ambiguities (all frozen identically across every arm)

1. **Proxy Anchor learning rate.** The paper text I recall specifies AdamW with backbone lr $10^{-4}$ and proxy lr ×100; the official repo's example command uses lr **6e-4** with "reduce lr by 2 every 5 steps." These are not reconcilable from the snippets I retrieved. **I freeze backbone $10^{-4}$ / proxy $10^{-2}$ with cosine decay over 200 epochs and use it for C0 and every LORD arm.** My C0 numbers are therefore *my reproduction*, not the published PA row, and must not be compared to published PA figures.
2. **PA's $\alpha=32$, $\delta=0.1$** — from memory; **not re-verified** in this session's search results. If the true values differ, both arms change together.
3. **PA's embedding head** — whether pooling is GAP+GMP *sum* or *concatenation*, and whether a BatchNorm1d precedes the linear layer, differs across public reimplementations. I freeze sum + no BN.
4. **PA's original epoch count** (~40 in the paper, per memory) differs from Lane A's 200-epoch budget. Running PA for 200 epochs is *not* PA's published recipe; that is precisely why C0 is a reproduction under the lane's budget rather than an inherited published row.
5. **PFML (CVPR 2025)** — loss reduction, proxy update rule, sampler and recipe are unknown to me. No matched reproduction is claimed; PFML is used only as an external reference target, and the PFML+LORD stack in §5 is explicitly a *conditional* prediction, not a forecast.
6. **DADA (AAAI 2024)** — mechanism unverified; per the prompt its seed count and uncertainty are unreported, so its 0.729/0.921/0.810 rows are treated as single-point references of unknown variance.
7. **AdvRF, VAPNet, CRT** — Lane B / transformer lane; deliberately unused.
8. **$\kappa(T,\alpha_d)$** in §2.2 has no closed form I derived; the method does not depend on knowing it, since $\hat\rho$ is measured directly and controlled. This is a deliberate substitution of an online measurement for an unavailable constant, and it is itself falsifiable (threshold 5).

---

**Summary of what I am claiming.** One new training constraint — *within-group diffusion consistency*, i.e. the embedding's one-hop geometry must equal its sharpened multi-hop geometry in exactly the directions the labels leave free — implemented as two group-restricted KLs against a parameter-free personalized-PageRank teacher over a stop-gradient memory of training embeddings, with a measured expansion controller that makes the collapsed solution provably repelling. It internalizes at train time the reranking operation that deployment forbids, costs ~5% extra training and nothing at inference, and I forecast +1.3 to +2.3 matched R@1 points across CUB/Cars/SOP in Lane A **without** crossing the PFML frontier on its own.

Sources: [Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf), [official Proxy-Anchor implementation](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020), [Self-Taught Metric Learning without Labels (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.pdf), [STML project page](https://cvlab.postech.ac.kr/research/STML/).
