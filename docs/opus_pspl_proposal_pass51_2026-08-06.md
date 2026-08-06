## PSPL — Product‑Simplex Prototype Lattice supervision

**Lane: A** (ResNet‑50, 512‑D normalized descriptor, ~224 px, single‑view cosine retrieval, 200 epochs). All forecasts and comparisons below are Lane A only.

**One paragraph.** Replace the learned prototype matrix of a normalized‑softmax/CosFace head with a *frozen* geometric object: a product of $G$ mutually‑orthogonal $K$‑point simplex ETFs whose $K^G$ sums form a lattice of unit vectors in $S^{511}$. Each training class is not given a private prototype but a *discrete code* $\mathbf k^{(c)}\in[K]^G$ selecting one codeword per group; the code is the only "prototype" object and it is optimized by a balanced discrete assignment (E‑step) interleaved with SGD (M‑step). Because every codeword is shared by $\approx C/K$ classes and no single group can identify a class ($K\ll C$), the objective is minimized only by features that compute $G$ *reusable, cell‑level* functionals rather than $C$ class‑private directions. The lattice's margin structure is class‑index‑free (Lemma 1), so the geometry allocated to an identity never seen at train time is identical to that allocated to a seen one. Deployment is an unmodified ResNet‑50 → 512‑D → $\ell_2$ → cosine NN; the codebook and codes are discarded. Added train cost is $<0.01\%$ FLOPs and *negative* parameter count versus proxy losses.

---

## 1. Executable mathematics

### 1.1 Network and deployment

$x \to$ ResNet‑50 (ImageNet‑1K init, all layers trained, BN affine+stats frozen after warmup) $\to$ GAP $p\in\mathbb R^{2048}$ $\to$ BN1d $\to$ linear $W\in\mathbb R^{512\times2048}, c\in\mathbb R^{512}$ $\to z=Wp+c$ $\to f=z/\lVert z\rVert_2\in S^{511}$.
**Test:** resize 256 → center crop 224 → one forward → $f$ → cosine NN over the official gallery. Nothing else. No reranking, no multi‑crop, no test‑set statistics.

### 1.2 The frozen lattice (no learned parameters)

Fix $K=9$, $G=64$, so $G(K-1)=512=d$ exactly.

* $S=[s_1\ldots s_K]\in\mathbb R^{(K-1)\times K}$: the regular simplex frame, $\lVert s_k\rVert=1$, $\langle s_k,s_{k'}\rangle=-\tfrac1{K-1}$ $(k\ne k')$, $\sum_k s_k=0$. Closed form: $S=\sqrt{\tfrac{K}{K-1}}\,V^\top\!\big(I_K-\tfrac1K\mathbf 1\mathbf 1^\top\big)$ with $V\in\mathbb R^{K\times(K-1)}$ any orthonormal basis of $\mathbf 1^\perp$.
* $U=[U_1\ldots U_G]\in\mathbb R^{512\times512}$: a **fixed** orthogonal matrix (QR of a Gaussian at seed 0, or a DCT‑II basis — pre‑registered as seed‑0 QR), block‑partitioned into $U_g\in\mathbb R^{512\times 8}$. $U_g^\top U_{g'}=\delta_{gg'}I$.
* Codewords $b_{g,k}=\rho\,U_g s_k$ with $\rho=G^{-1/2}=1/8$.
* Lattice point for code $\mathbf k$: $\mu(\mathbf k)=\sum_{g}b_{g,k_g}$.

$U$ and $S$ are registered buffers with `requires_grad=False`. They receive no gradient, ever.

**Lemma 1 (exact cosine–Hamming law).** $\lVert\mu(\mathbf k)\rVert=1$ for every $\mathbf k$, and for any two codes at Hamming distance $h$,
$$\big\langle\mu(\mathbf k),\mu(\mathbf k')\big\rangle \;=\; 1-\frac{h}{G}\cdot\frac{K}{K-1}.$$
*Proof.* Cross‑group terms vanish ($\mathrm{col}\,U_g\perp\mathrm{col}\,U_{g'}$). Matching groups contribute $\rho^2$, mismatched $-\rho^2/(K-1)$. Sum $=\rho^2\big[(G-h)-\tfrac{h}{K-1}\big]=1-\tfrac hG\tfrac{K}{K-1}$; setting $h=0$ gives $\lVert\mu\rVert^2=G\rho^2=1$. $\square$

With $G=64,K=9$: $\cos = 1-0.017578\,h$; range $[-1/8, 1]$.

**Corollary (reserved geometry).** All $K^G\approx 10^{61}$ lattice points are unit vectors and their mutual similarity depends *only* on code distance — not on which of them happen to be occupied by the $C$ training identities. An identity that appears for the first time at test time and lands near a lattice point inherits exactly the same margin structure as a training identity. This is the formal sense in which the supervision "reserves room" for unseen classes; free‑prototype losses have no analogue (see §2).

### 1.3 Loss

For image $x$ of class $c$ with code $\mathbf k^{(c)}$: let $u_g=U_g^\top f\in\mathbb R^{8}$, $\hat u_g=u_g/(\lVert u_g\rVert+\epsilon)$, $\epsilon=10^{-6}$, and $\pi_g=\lVert u_g\rVert^2$ (note $\sum_g\pi_g=1$ since $U$ is orthogonal and $\lVert f\rVert=1$).

$$\boxed{\;\mathcal L_{\mathrm{grp}}=\frac1G\sum_{g=1}^{G}\;-\log\frac{\exp\!\big(\gamma(\langle\hat u_g,s_{k_g}\rangle-m)\big)}{\exp\!\big(\gamma(\langle\hat u_g,s_{k_g}\rangle-m)\big)+\sum_{k\ne k_g}\exp\!\big(\gamma\langle\hat u_g,s_k\rangle\big)}\;}$$

$$\mathcal L_{\mathrm{eng}}=\log G+\sum_{g}\pi_g\log\pi_g \;\;(\ge 0,\;=0 \iff \pi\equiv 1/G),\qquad \mathcal L=\mathcal L_{\mathrm{grp}}+\lambda_e\,\mathcal L_{\mathrm{eng}}.$$

**Reduction to the named baseline.** $\mathcal L_{\mathrm{grp}}$ with $G=1$, $K=C$, identity code, and $S$ replaced by a learned unit‑norm $W_{\mathrm{cls}}$ is *exactly* CosFace/normalized‑softmax, $-\log\frac{e^{\gamma(\cos\theta_y-m)}}{e^{\gamma(\cos\theta_y-m)}+\sum_{j\ne y}e^{\gamma\cos\theta_j}}$. With $G=1,K=C$ and $S$ frozen it is exactly the fixed simplex‑ETF classifier of Yang et al. (NeurIPS 2022). PSPL is the $G>1$ product generalization plus a learned discrete class→lattice assignment. This is my baseline of record and I reproduce it in house rather than inheriting anyone's frontier.

**The per‑group renormalization is operational, not cosmetic.** Because $\hat u_g$ is unit norm, $\mathcal L_{\mathrm{grp}}$ is invariant to $\lVert u_g\rVert$; combined with weight decay this changes the stationary set relative to the un‑normalized variant (which is a plain fixed‑classifier on $\mu(\mathbf k)$). Control C6 in §4 isolates it. It also alters the gradient magnitude per group — see §1.4.

**Proposition 2 (consistency: the loss recovers the lattice point).** For $\lambda_e>0,\gamma>0$, $\inf_{f\in S^{511}}\mathcal L$ is attained exactly at $f=\mu(\mathbf k^{(c)})$.
*Proof.* $\mathcal L_{\mathrm{grp}}$ depends only on $\{\hat u_g\}$ and decomposes per group. Per group, write $\mathrm{CE}(\hat u)=-\gamma(\langle\hat u,s_{k^*}\rangle-m)+\log\sum_k e^{\gamma\langle\hat u,s_k\rangle}$. Its Euclidean gradient is $\gamma\big(\sum_k p_k s_k-s_{k^*}\big)$. At $\hat u=s_{k^*}$, using $\sum_{k\ne k^*}s_k=-s_{k^*}$ and $p_k$ equal for $k\ne k^*$, this equals $\gamma s_{k^*}\big(p_{k^*}-1-\tfrac{1-p_{k^*}}{K-1}\big)$, i.e. purely radial, so the spherical (tangential) gradient vanishes; by the symmetry of $S$ and log‑sum‑exp convexity along geodesics it is the unique minimizer on $S^{K-2}$. $\mathcal L_{\mathrm{eng}}$ is minimized uniquely at $\pi_g\equiv1/G$. The point $f$ with $\hat u_g=s_{k_g}$ and $\pi_g=1/G$ is $f=\sum_g G^{-1/2}U_gs_{k_g}=\mu(\mathbf k)$, which is feasible ($\lVert\mu\rVert=1$). $\square$

So $\mathcal L_{\mathrm{eng}}$ is load‑bearing: without it the group‑normalized loss does not pin $f$ to the lattice.

### 1.4 Gradient path

Only $\theta=\{$backbone, BN1d, $W,c\}$ receives gradient. With $p_{g,\cdot}=\mathrm{softmax}(\ell_{g,\cdot})$, $y_{g,\cdot}$ the one‑hot target, $r_g=\sum_k(p_{g,k}-y_{g,k})s_k$:
$$\frac{\partial\mathcal L_{\mathrm{grp}}}{\partial f}=\frac{\gamma}{G}\sum_g U_g\frac{I-\hat u_g\hat u_g^\top}{\lVert u_g\rVert}\,r_g,\qquad
\frac{\partial\mathcal L_{\mathrm{eng}}}{\partial f}=2\lambda_e\sum_g\big(\log\pi_g+1\big)U_gu_g,$$
then $\partial/\partial z=(I-ff^\top)\,\partial/\partial f\,/\lVert z\rVert$. The $1/\lVert u_g\rVert$ factor *amplifies* gradients into starved groups — this is the mechanism that defeats degeneracy D1 dynamically, and the reason for $\epsilon$ and global‑norm gradient clipping at 5.0. $A$ (the code table) is updated outside autograd; targets enter under `stop_gradient`.

### 1.5 E‑step: the class→lattice assignment

Maintain per‑class EMA centroids $\bar f_c \leftarrow 0.9\,\bar f_c+0.1\,\mathrm{mean}_{\text{batch}}f$, renormalized. Every $T_A=5$ epochs after a 10‑epoch warmup:

1. For each group $g$: cost $J^{(g)}_{c,k}=-\big\langle \tfrac{U_g^\top\bar f_c}{\lVert U_g^\top\bar f_c\rVert},\,s_k\big\rangle$.
2. **Balanced assignment.** $\min_{A_g\in\{0,1\}^{C\times K}}\sum_{c,k}A_{g,c,k}J^{(g)}_{c,k}$ s.t. row sums $=1$ and column sums $\in[\lfloor C/K\rfloor-\beta,\lceil C/K\rceil+\beta]$, $\beta=\max(1,\lceil 0.25C/K\rceil)$. Solved exactly by min‑cost flow ($C\le 11{,}318$, $K=9$ — milliseconds), or entropic Sinkhorn ($\eta=0.05$, 50 iters) followed by rounding for large $C$.
3. **Pairwise‑independence penalty (anti‑duplication).** Process groups in a fixed order; when assigning group $g$, add $\lambda_{\mathrm{ind}}\!\sum_{g'<g}\!\big[\log n_{gk,g'k'_{c}}-\log(n_{gk}n_{g'k'_c}/C)\big]$ to $J^{(g)}_{c,k}$, where $n$ are code‑table co‑occurrence counts from the previous assignment and $k'_c$ is class $c$'s current digit in group $g'$. $\lambda_{\mathrm{ind}}=0.1$. (Iterated‑conditional‑modes; one sweep.)
4. **Separation repair.** Compute all pairwise Hamming distances by one one‑hot matmul ($C\times576$ by $576\times C$). Any pair with $h<h_{\min}=24$: move the class with the worse mean group‑CE to its next‑best feasible codeword in the group with the smallest margin loss; repeat $\le R=20$ passes; if still infeasible, retain the previous code for that class.
5. **Hysteresis.** Accept class $c$'s new code only if it lowers $c$'s mean group‑CE by $\ge\kappa=0.02$ nats; else keep the old code.

**Initialization ($\text{epoch }0$).** Run steps 1–4 on $\bar f_c$ computed from the ImageNet‑initialized backbone with the randomly‑initialized head. Pre‑registered control C11 replaces this with a random balanced code, to show the result does not hinge on ImageNet‑derived code structure.

### 1.6 Frozen hyperparameters and schedule

| | value |
|---|---|
| $d,K,G$ | $512,\;9,\;64$ (all four datasets) |
| $\gamma,m,\lambda_e,\lambda_{\mathrm{ind}}$ | $12,\;0.15,\;0.5,\;0.1$ |
| $h_{\min}$ | 24 (sweep $\{12,24,40\}$ pre‑registered) |
| $T_A,\kappa$, EMA | $5$ epochs, $0.02$, $0.9$ |
| optimizer | AdamW; lr $1{\times}10^{-4}$ backbone / $1{\times}10^{-3}$ head; wd $4{\times}10^{-4}$; cosine decay; 5‑epoch linear warmup |
| epochs / batch | 200 / 128 (random shuffle; $P{\times}M$ sampler as control C12) |
| augmentation | RandomResizedCrop(224, scale (0.16,1)) + horizontal flip. Nothing else. |
| BN | backbone BN frozen (affine + running stats) after epoch 5 |
| seeds | 5 for all ablations; **10** for headline comparisons |

Legality: official train images + identity labels + ordinary stochastic augmentation + ImageNet‑1K init only. No test data, no generated data, no text/VLM, no extra annotations, no transduction, no reranking, no gallery fitting.

---

## 2. Causal zero‑shot error mode, and the attack on degeneracies

### 2.1 The error mode: prototype privacy / unranked solution set

Let $L(\theta,M)$ be any free‑prototype loss depending on $\theta$ only through $\{\langle f_\theta(x_i),\mu_c\rangle\}$ with $M=[\mu_1\ldots\mu_C]$ learned. Two facts jointly cause the zero‑shot failure:

1. **Rotation/relabel invariance.** $L(Qf_\theta,QM)=L(f_\theta,M)$ for all $Q\in O(d)$, and $L$ is invariant under any simultaneous permutation of prototypes and labels. Nothing in $L$ ties the direction $\mu_c$ to a *reusable* visual quantity; $\mu_c$ is a private coordinate of class $c$.
2. **Unseen inputs are unconstrained.** Let $(\theta^\ast,M^\ast)$ minimize $L$ on the training support $\mathcal S$. For any disjoint compact $\mathcal U$ with $\mathcal U\cap\mathcal S=\emptyset$, and any $\eta>0$, a network of sufficient capacity admits $\theta'$ with $\sup_{x\in\mathcal S}\lVert f_{\theta'}(x)-f_{\theta^\ast}(x)\rVert<\eta$ while $f_{\theta'}$ maps all of $\mathcal U$ to a single point. Hence $L(\theta',M^\ast)\to L(\theta^\ast,M^\ast)$ although unseen‑class R@1 collapses to chance.

Together: **the objective imposes no ordering on solutions by unseen‑class separability.** The entire zero‑shot result is delivered by implicit bias plus the ImageNet prior. Terminal‑phase neural collapse makes this concrete — the class‑mean subspace is $(C{-}1)$‑dimensional (99 on CUB) and within‑class variability, which is exactly where unseen‑class distinctions live, is driven to zero.

PSPL changes the ranking, not merely the regularizer. Every codeword $b_{g,k}$ is the target of $\approx C/K$ *distinct* classes ($\approx11$ on CUB, $\approx1258$ on SOP). Achieving $\langle\hat u_g,s_{k}\rangle\approx1$ for all of them requires a functional that is *shared across a class cell*, and the E‑step chooses the cells to be the max‑margin $K$‑way partitions under current features. At any joint stationary point of $(\theta,A)$, each of the $G\cdot K$ codeword directions is the max‑margin center of a visually coherent super‑class, estimated from $\approx N/K$ images rather than $N/C$. Those are precisely the quantities that can be evaluated on an identity never labeled.

**Honest limit.** This is an argument about the *joint* objective's landscape and about optimization bias. It is not a theorem excluding memorization: a network that identifies the class can emit any class‑level code. The mechanism claim is that the loss makes the shared‑functional solution the cheaper one, and the controls in §4 (especially C3 and C4) are designed to be able to *refute* it, not to confirm it.

### 2.2 Degeneracies and their attacks

**D1 — group starvation** (dump all embedding energy into a few groups, ignore the rest).
*Attack, proof‑level:* $\mathcal L_{\mathrm{grp}}$ is a function of $\{\hat u_g\}$ only, hence invariant to $\lVert u_g\rVert$. A group whose direction is uninformative contributes $\ge\log K-\gamma m$ ... more precisely, a group with $\hat u_g$ independent of the label contributes exactly $\mathbb E[\mathrm{CE}]\ge \log K$ in expectation over a balanced cell distribution. Starving $j$ groups therefore incurs an *irreducible* $\tfrac jG\log 9$ nats, which no energy reallocation can offset. Additionally $\partial\mathcal L_{\mathrm{grp}}/\partial f$ carries a $1/\lVert u_g\rVert$ factor, so starved groups receive amplified gradient. $\square$

**D2 — code duplication** (all groups converge to the same partition; effective code length collapses to $\log K$).
*Attack:* duplication is loss‑*neutral*, not loss‑favored (a duplicated group adds no constraint but is exactly as easy), so it can only arise by drift. Blocked by (i) the pairwise‑independence penalty in E‑step step 3, (ii) hysteresis $\kappa$ (neutral moves are rejected), (iii) a **pre‑registered falsifier**: mean pairwise NMI between group partitions must remain $<0.35$ at epoch 200.

**D3 — a group becomes a class identifier** (the cheapest shortcut: one codeword per class).
*Attack, structural and exact:* the balance constraint forces every column sum to $\lfloor C/K\rfloor\pm\beta$. With $K=9$, each codeword is shared by $\ge 8$ classes on CUB/Cars and $\ge 1000$ on SOP/In‑Shop. Therefore **no single group can separate any pair of classes on its own**, and identifying a class requires $\ge\lceil\log_K C\rceil$ groups to fire jointly and correctly ($3$ on CUB/Cars, $5$ on SOP). This is the central structural guarantee and it is enforced by a hard constraint, not a penalty. $\square$

**D4 — nuisance‑aligned cells** (a group's partition tracks background/pose rather than object structure).
*Attack:* partial, and stated as a residual risk. The balance constraint forces a nuisance to partition *classes*, not images, which is much harder for pose (pose varies within class). Backgrounds correlated with species/model are a genuine confound. Detection without extra annotation: cell‑prediction accuracy on the **held‑out training‑identity validation split** (§2.3); nuisance‑driven cells do not transfer to unseen identities. Pre‑registered: mean group‑cell top‑1 on held‑out classes must exceed $2\times$ chance ($2/9$).

**D5 — the head alone solves it, backbone unshaped.** $W$ is linear on pooled features; realizing 64 independent 9‑way class partitions linearly is exactly the property we want, and control C10 ("PSPL, backbone frozen") pre‑registers the lower bound.

**D6 — E/M self‑confirmation** (the E‑step picks the code the features already encode; nothing improves).
*Attack:* the E‑step is *constrained* (balance + $h_{\min}$), so it cannot select a trivially‑satisfied code; hysteresis requires strict improvement; $T_A=5$ gives the M‑step time to move; the epoch‑0 code comes from un‑adapted features. Residual risk acknowledged; monitored by code‑churn rate (fraction of digits changed per E‑step), which should decay but not hit 0 before epoch ~120.

### 2.3 Held‑out‑identity validation (legal, train‑images‑only)

Split *training* identities once per dataset at a frozen seed into $90\%$ fit / $10\%$ val. All model selection, all falsification thresholds, and all diagnostics use val‑identity R@1. Final reported models retrain on all training identities with epoch count fixed by this protocol. No test image or test label is ever touched.

---

## 3. Adversarial novelty search, and one‑sentence distinctions

*Inside DML.*
- **PFML / Potential Field DML (CVPR'25 per the prompt; listed as CVPR'24 in some indexes — see §7)** — models all sample interactions as a continuous potential field with 15 class‑private proxies; PSPL has *zero* learned proxies and instead constrains prototypes to a frozen shared lattice with a discrete class code.
- **ProxyAnchor / ProxyNCA++ / SoftTriple / Normalized‑Softmax‑CosFace** — free (or multi‑center) class‑private prototypes; PSPL's prototypes are non‑learned lattice points whose coordinates are *shared across classes*.
- **Proxy Synthesis (AAAI'21)** — interpolates proxies/embeddings to *sample* a few synthetic unseen classes; PSPL synthesizes nothing and instead makes all $K^G$ unoccupied lattice slots margin‑equivalent to occupied ones by construction (Lemma 1).
- **Deep Factorized Metric Learning (CVPR'23) / Deep Compositional Metric Learning (CVPR'21)** — factorize the *sample representation* (class‑related vs. class‑independent components, with auxiliary train‑time modules and/or sample synthesis); PSPL factorizes the *supervision target* and adds no auxiliary module. *(PDF retrieval returned HTTP 403; this distinction is stated at mechanism level and flagged in §7.)*
- **BIER / A‑BIER / DREML / Divide‑and‑Conquer** — split the *embedding* into independently‑trained learners with decorrelation, each with its own class prototypes; PSPL keeps one embedding and one loss, and the split is in the shared *codeword dictionary*, not in the learner set.
- **MIC (ICCV'19), DiVA, S2SD, Grouplet** — add auxiliary tasks / self‑distillation / mined characteristics; PSPL has a single head family, no teacher, no mined targets.
- **Anti‑Collapse Loss (coding‑rate DML)** — maximizes a rate objective to resist embedding collapse; PSPL does not optimize rate and instead removes the free‑prototype degrees of freedom that allow collapse to be optimal.
- **PA+DADA (AAAI'24)** — data‑augmented domain adaptation on proxy‑based losses; PSPL leaves the proxy‑based loss family entirely and changes the prototype hypothesis class.

*Outside DML.*
- **Fixed simplex‑ETF classifier (Yang et al., NeurIPS'22)** — one $C$‑vertex ETF spanning $C{-}1$ dims for imbalanced closed‑set classification; PSPL is a $G$‑fold *product* of $K$‑vertex ETFs spanning $G(K{-}1)=d$ dims with $K^G\gg C$ available prototypes and a learned class→vertex assignment. (PSPL with $G=1,K=C$ reduces *exactly* to it.)
- **ECOC (Dietterich & Bakiri '95; Deep‑ECOC; ECOC‑for‑robustness)** — fixed, hand‑ or randomly‑designed binary code matrices for closed‑set accuracy/robustness; PSPL uses $K$‑ary groups realized as orthogonal simplex frames in the *embedding* metric and optimizes the code by balanced assignment during training.
- **Evron et al., "The Role of Codeword‑to‑Class Assignments in ECOC" (PMLR v206, 2023)** — shows *pre‑chosen* similarity‑aware assignments help closed‑set accuracy; PSPL makes the assignment an online constrained optimization coupled to the features and targets zero‑shot retrieval geometry, with a hard $h_{\min}$ and balance constraint they do not have.
- **"Deep Representation Learning with Target Coding" (AAAI'15)** — fixed Hadamard binary targets with squared loss; PSPL has group structure, simplex geometry, per‑group renormalization, balance, and a learned assignment, none of which are present there.
- **Product / additive / composite quantization (PQ, OPQ, AQ, CQ)** — factorized codebooks applied to *features* for ANN compression at test time; PSPL applies the product codebook to the *label targets* at train time only and deploys a fully continuous, unquantized descriptor.
- **HDC‑ZSC (DATE'24) and VSA factorizers** — fixed product codebooks binding *given attribute groups* for zero‑shot classification; PSPL is given no attributes — the group semantics are discovered by the balanced assignment from identity labels alone.
- **SwAV / DINO Sinkhorn prototypes** — unsupervised, per‑*image* balanced assignment to one flat prototype bank; PSPL is supervised, per‑*class*, factorized into $G$ groups, with a persistent discrete code table.
- **Barlow's factorial codes / redundancy reduction** — a criterion on unsupervised representations; PSPL imposes near‑factoriality on the *label code* as a constraint in a discrete assignment problem.
- **Group Softmax (long‑tail detection)** — disjoint groups of *categories* to protect tail classes' logits; PSPL's groups partition the *embedding subspace*, every class appears in every group, and the target is a code digit not a category.

I did not find a primary source combining: frozen product‑of‑simplex‑ETF prototypes + online balanced class→code assignment + per‑group renormalized cosine CE + zero‑shot DML. I would revise this claim if a reviewer surfaces one.

---

## 4. Matched‑compute controls (all Lane A, identical recipe, 5 seeds, paired by seed)

| # | Control | Isolates |
|---|---|---|
| C1 | CosFace/Normalized‑Softmax, learned 512‑D prototypes | the occupied simple alternative; PSPL's reduction at $G=1$ |
| C2 | Fixed single simplex‑ETF classifier ($G=1,K=C$) | "fixed vs. learned classifier" — the whole prior claim |
| C3 | **Random frozen balanced code, no E‑step** | the *learned assignment* |
| C4 | **Class‑private group codewords** (each class gets its own $G$ vectors; same arch, same FLOPs) | codeword **sharing** — the core claim |
| C5 | Learned unconstrained group codebook $B_g$ (drop frozen simplex/orthogonality) | the geometric constraint |
| C6 | No per‑group renormalization (global cosine to $\mu(\mathbf k)$) | the operational normalization |
| C7 | Balance constraint off | (C1)/D3 |
| C8 | $\lambda_e=0$ | Proposition 2's necessity |
| C9 | $h_{\min}\in\{12,24,40\}$; $G\in\{8,16,32,64,128\}$ at fixed $G(K{-}1)=512$ | interior optimum in $G$ (mechanism predicts one; monotone toward $G{=}1$ refutes) |
| C10 | PSPL, backbone frozen | D5 lower bound |
| C11 | Random code init instead of ImageNet‑feature init | ImageNet‑derived code contamination |
| C12 | $P{\times}M$ sampler vs. random shuffle | batch construction confound |
| C13 | In‑house PFML reproduction (15 proxies CUB/Cars, 2 SOP) | reference reproduction, not inherited |

**C3 and C4 are decisive.** If C3 matches PSPL within 0.3 R@1, the learned assignment is not the mechanism and the paper must be re‑scoped to "product‑ETF fixed classifier." If C4 matches PSPL, sharing is not the mechanism and the proposal fails.

---

## 5. Frozen forecasts, falsification, and frontier arithmetic

All Lane A, R@1, ResNet‑50, 512‑D, 224 px, 200 epochs, single‑view cosine, mean ± s.d. over **5 seeds** (headline comparisons at 10).

**Predicted in‑house baselines (my recipe):**

| | CUB | Cars196 | SOP | In‑Shop |
|---|---|---|---|---|
| C1 CosFace 512‑D | 0.702 ± 0.005 | 0.886 ± 0.005 | 0.801 ± 0.003 | 0.911 ± 0.003 |
| C2 fixed ETF | 0.706 ± 0.005 | 0.890 ± 0.005 | 0.803 ± 0.003 | — |
| C13 PFML repro | 0.728 ± 0.006 | 0.921 ± 0.005 | 0.826 ± 0.003 | — |

**PSPL forecast (frozen now):**

| | point | 80% CI |
|---|---|---|
| **CUB‑200‑2011** | **0.738** | 0.719 – 0.752 |
| **Cars196** | **0.934** | 0.920 – 0.943 |
| SOP (secondary) | 0.827 | 0.817 – 0.835 |
| In‑Shop (secondary) | 0.929 | 0.918 – 0.937 |

**Frontier‑crossing arithmetic** against the prompt's audited Lane A references (PFML: CUB $0.734\pm0.003$, Cars $0.927\pm0.003$, SOP $0.829\pm0.002$, 5 runs ⇒ SEM $\approx0.0013$, $0.0013$, $0.0009$).

- **Cars196.** $\Delta=0.934-0.927=+0.007$. My 5‑seed SEM $=0.005/\sqrt5=0.0022$; combined SEM $=\sqrt{0.0022^2+0.0013^2}=0.0026$; $z=2.7$, two‑sided $p\approx0.007$. At 10 seeds (SEM $0.0016$): combined $0.0021$, $z=3.3$, $p\approx0.001$. **Cars196 is the decisive test.**
- **CUB.** $\Delta=0.738-0.734=+0.004$. 5 seeds: combined SEM $0.0026$, $z=1.5$, $p\approx0.13$ — **not decisive**. 10 seeds: combined $0.0021$, $z=1.9$, $p\approx0.06$ — still not decisive. Stated plainly: at my own forecast, **CUB cannot deliver a statistically decisive crossing at any realistic seed budget** (≈35 seeds would be needed for $p<0.05$ at $\Delta=0.004$). I will not claim CUB SOTA on a 0.4‑point margin; 1 R@1 point on CUB is ~59 test images.
- **SOP.** I forecast $0.827 < 0.829$: **I do not predict crossing PFML on SOP.** Reason: SOP has ~5.3 images/class over 11,318 train classes, so EMA class centroids are noisy and the E‑step degrades; $K=9$ cells of ~1258 classes are also very coarse.
- **In‑Shop.** $0.929$ vs PA+DADA $0.930$ (seed count and uncertainty **unreported**, so no significance test is possible). I claim parity at ~1.00× training cost versus their 1.06× epoch time / 1.01× memory, not a crossing.

**Subjective probabilities (frozen):** P(PSPL beats C1 by ≥1.0 pt on *both* CUB and Cars) = 0.60. P(Cars mean > 0.927) = 0.62. P(decisive Cars crossing at 10 seeds, $p<0.05$) = 0.45. P(CUB mean > 0.734) = 0.55. P(both CUB and Cars means exceed PFML) = 0.40. P(SOP crossing) = 0.20.

**Falsification thresholds (pre‑registered, decided on held‑out‑identity val first, then reported on test):**

- **F1** PSPL − C1 $<+1.0$ R@1 on either CUB or Cars (paired, 5 seeds) → **method falsified**.
- **F2** mean pairwise partition NMI $>0.35$ at epoch 200 → **factorization claim falsified** (D2).
- **F3** |PSPL − C3| $<0.3$ on both CUB and Cars → learned assignment is not the mechanism; re‑scope.
- **F4** |PSPL − C4| $<0.3$ → **codeword sharing is not the mechanism; method falsified**.
- **F5** energy entropy $H(\pi)/\log G<0.8$ at convergence → D1 not defeated.
- **F6** $G$‑sweep monotone increasing toward $G=1$ → the product structure is inert; **falsified**.
- **F7** held‑out‑identity mean group‑cell top‑1 $<2\times$ chance → cells are not transferable (D4).
- **F8** |PSPL − C11| $>0.5$ → the result depends on ImageNet‑derived code initialization; must be reported as such.

---

## 6. Cost, and benchmark / contamination risk

**Train cost.** Extra forward compute per sample: one $512\times512$ matmul (the full $U^\top f$; the $G$ projections are one matmul), $G$ vector normalizations, $G$ nine‑way softmaxes $\approx 0.3$ MFLOP against ResNet‑50's $\approx4.1$ GFLOP forward / $\approx12.3$ GFLOP fwd+bwd ⇒ **$<0.01\%$**. E‑step: $O(CGK)$ costs + $G$ min‑cost‑flows + one $C\times576$ matmul for Hamming repair — $\approx1$ s (CUB/Cars) to $\approx3$ s (SOP) every 5 epochs, i.e. $<0.1\%$ of wall clock. **Predicted epoch time 1.00×** the CosFace baseline (measurable overhead $<0.5\%$), memory $+1$ MB (the $U$ buffer) $+\,C{\times}64$ int8.
**Learned parameters: fewer than the baselines** — PSPL has *no* prototype matrix. vs. ProxyAnchor/CosFace: $-0.05$M (CUB) and $-5.8$M (SOP). vs. AdvRF (extra ResNet‑34 + U‑Net + distillation, $\ge2\times$ train) and VAPNet (attribute machinery) this is a large cost advantage — though those are Lane B and I am not comparing to them numerically.
**Deployment.** Byte‑identical to a plain ResNet‑50/512‑D model: one model, one view, one descriptor, cosine NN. No codebook, no code table, no quantization at test.

**Risks.**
- *ImageNet overlap.* CUB and Cars196 overlap semantically with ImageNet‑1K classes. ImageNet init is permitted, but the epoch‑0 code is ImageNet‑derived; C11/F8 quantify the dependence.
- *Benchmark resolution.* CUB test = 5,924 images / 100 classes; 1 R@1 pt ≈ 59 images. Cars196 contains near‑duplicates; SOP has multi‑view products with high intra‑class viewpoint variance; In‑Shop has a fixed query/gallery split that must be used verbatim.
- *Method‑specific.* $\lVert u_g\rVert\to0$ instability (mitigated by $\epsilon$, $\mathcal L_{\mathrm{eng}}$, clipping); E/M oscillation (hysteresis + $T_A$); the assignment problem is solved to optimality per group but the *joint* code optimization is coordinate‑descent and can stall; SOP/In‑Shop centroid noise from ~5 images/class is the most likely cause of a null result there.
- *Hyperparameter surface.* $\gamma,m$ interact with $K$ (only $K=9$ competitors per group, min cosine $-1/8$, so the achievable logit range is $\gamma(1+1/8)$); a $\gamma$ that works for a $C$‑way head is not transferable, and I have set $\gamma=12$ without tuning. A null result caused by $\gamma$ mis‑set would be indistinguishable from a mechanism failure unless the $\gamma\in\{8,12,16,24\}$ sweep (on held‑out identities only) is run — that sweep is part of the protocol.

---

## 7. Unresolved source ambiguities (stated, not papered over)

1. **PFML.** The prompt attributes it to CVPR 2025; public indexes I reached describe "Potential Field based Deep Metric Learning" with matching numbers (CUB 73.4 / Cars 92.7 / SOP 82.9) and at least one listing gives CVPR **2024**. Venue/year unresolved.
2. **PFML recipe.** Its optimizer, LR schedule, whether it uses a BN‑neck 512‑D head, its proxy update rule, and whether the reported "±" is over seeds or splits are not established for me under this blind protocol. I therefore **do not inherit** its frontier: C13 is my own reproduction and every comparison is against both C13 and the prompt‑quoted number, labeled separately.
3. **PA+DADA In‑Shop 0.930** — seed count and uncertainty explicitly unreported; no significance test is possible, so I claim parity, not crossing.
4. **Deep Factorized Metric Learning (CVPR'23)** — PDF fetch returned HTTP 403; my distinction is stated at mechanism level (factorizes sample representations with auxiliary modules vs. PSPL factorizing the target hypothesis class) and should be re‑verified against the primary text.
5. **BN freezing** — standard in several Lane A recipes but not uniform; if C13's source recipe differs, the reproduction must follow the source and PSPL must be re‑run under the same setting.
6. **DADA's matched‑cost rows** (0.729 / 0.921 / 0.810) are treated as matched‑cost controls, not targets, per the prompt.

---

**What I am claiming, and what I am not.** I claim a new *supervision hypothesis class* — prototypes confined to a frozen product‑of‑simplex‑ETF lattice with an online balanced discrete class→code assignment — whose structural guarantee (D3: no group can identify a class, by hard constraint) and geometric guarantee (Lemma 1: margin structure is class‑index‑free over all $K^G$ slots) are new relative to the primary sources I could reach. I do **not** claim it adds information: the code is a deterministic function of the label, so the Shannon content of supervision is unchanged; the mechanism is a hypothesis‑class and optimization‑bias mechanism, and §4's C3/C4 are built to be able to kill it. I forecast a decisive crossing on Cars196 only, explicit non‑crossing on SOP, parity on In‑Shop, and an underpowered CUB comparison that I will report as such rather than claim.

**Sources:** [Yang et al., fixed ETF classifier (arXiv 2203.09081)](https://arxiv.org/abs/2203.09081) · [Evron et al., codeword‑to‑class assignments in ECOC (PMLR v206)](https://proceedings.mlr.press/v206/evron23a/evron23a.pdf) · [Deep Representation Learning with Target Coding (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/download/9796/9655) · [Scalable design of ECOCs (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/41792f041a3a0774418791993cf887fe-Paper-Conference.pdf) · [Deep Factorized Metric Learning (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf) · [Deep Compositional Metric Learning (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Deep_Compositional_Metric_Learning_CVPR_2021_paper.html) · [Potential Field Based Deep Metric Learning](https://shubhangb97.github.io/potential_field_DML/) · [PA+DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/download/29400/30645) · [Anti‑Collapse Loss for DML (arXiv 2407.03106)](https://arxiv.org/pdf/2407.03106) · [Non‑isotropy Regularization for Proxy‑based DML (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf) · [Zero‑shot Classification using Hyperdimensional Computing](https://arxiv.org/html/2401.16876) · [Factorizers for Distributed Sparse Block Codes](https://arxiv.org/pdf/2303.13957)
