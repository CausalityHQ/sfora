## Lane declaration

**Lane A** (matched 512‑D CNN lane: ResNet‑50, 512‑D normalized global descriptor, ~224 px, single‑view cosine, 200 epochs). All forecasts, baselines and references below are Lane A. No Lane B number is used anywhere in this proposal.

---

# FCS — Factorial Coordinate Supervision

**One sentence.** Replace the single $C$‑way class‑private readout of proxy/softmax DML with $K$ *independent* $d$‑way readouts over the **fixed canonical basis** of $K$ disjoint, L2‑normalized slices of the 512‑D descriptor, where each identity is a discrete codeword $y_c\in[d]^K$ over an **over‑subscribed** alphabet ($\sim C/d$ identities per symbol) that is re‑optimized every epoch by balanced assignment under a minimum‑Hamming‑distance constraint; the code table is train‑only and discarded at test time.

The method allocates **zero parameters per class** — there are no proxies, no learned atoms, no auxiliary network, no pair mining, no memory, no sampler requirement.

---

## 1. Executable mathematics

### 1.1 Objects

| Symbol | Object | Learned? | Deployed? |
|---|---|---|---|
| $\theta$ | ResNet‑50 backbone, ImageNet‑1K init, GAP → $g(x)\in\mathbb R^{2048}$ | SGD/AdamW | yes |
| $W\in\mathbb R^{512\times2048},\,b\in\mathbb R^{512}$ | linear embedding | AdamW | yes |
| $\{e_1,\dots,e_d\}$ | per‑block alphabet = **canonical orthonormal basis of $\mathbb R^d$** | **fixed, not learned** | n/a |
| $Y\in[d]^{C\times K}$ | identity code table | discrete, re‑solved each epoch by min‑cost flow | **no — discarded** |
| $s,\ \varepsilon_{\rm ls},\ \delta_{\min},\ \gamma,\ K$ | scale, label smoothing, min code distance, hysteresis, block count | fixed hyperparameters | n/a |

### 1.2 Forward map (identical at train and test)

$$\tilde z(x)=W g_\theta(x)+b\in\mathbb R^{512},\qquad \tilde z=[\tilde z^{(1)};\dots;\tilde z^{(K)}],\ \ \tilde z^{(k)}\in\mathbb R^{d},\ d=512/K$$

$$u^{(k)}(x)=\frac{\tilde z^{(k)}}{\max(\|\tilde z^{(k)}\|_2,\epsilon)},\ \epsilon=10^{-6};\qquad \boxed{\varphi(x)=K^{-1/2}\big[u^{(1)};\dots;u^{(K)}\big]\in S^{511}}$$

Retrieval: ordinary cosine $=\langle\varphi(x),\varphi(x')\rangle=\frac1K\sum_k\langle u^{(k)}(x),u^{(k)}(x')\rangle$. One model, one view, one fixed 512‑D descriptor, plain NN search. **Block normalization is part of the model**, applied identically in training and deployment.

### 1.3 Loss

With target coordinate $y_{c,k}$ and smoothed target $t^{(k)}_j=(1-\varepsilon_{\rm ls})\mathbb 1[j=y_{c,k}]+\varepsilon_{\rm ls}/d$:

$$\mathcal L=\frac1B\sum_{i=1}^{B}\frac1K\sum_{k=1}^{K}\ \Big[-\sum_{j=1}^{d}t^{(k)}_j\log \frac{\exp\!\big(s\,u^{(k)}_{i,j}\big)}{\sum_{j'=1}^{d}\exp\!\big(s\,u^{(k)}_{i,j'}\big)}\Big]$$

Note $u^{(k)}_{i,j}=\langle u^{(k)}(x_i),e_j\rangle$ — the logit **is** the cosine to a fixed atom. There is no other term in the objective.

**Gradient path (complete).** With $p^{(k)}=\mathrm{softmax}(s\,u^{(k)})$, $r_k=\|\tilde z^{(k)}\|$:

$$\frac{\partial \ell_i}{\partial u^{(k)}}=\frac{s}{K}\big(p^{(k)}-t^{(k)}\big),\qquad \frac{\partial \ell_i}{\partial \tilde z^{(k)}}=\frac{s}{K r_k}\big(I-u^{(k)}u^{(k)\top}\big)\big(p^{(k)}-t^{(k)}\big)$$

then standard backprop into $W,b,\theta$. No gradient flows to $Y$ (discrete). **Every one of the 512 coordinates receives non‑zero gradient on every sample** — this is the load‑bearing structural fact (§2).

**The normalization is not harmless.** $\partial\ell/\partial\tilde z^{(k)}\propto 1/r_k$, so decoupled weight decay $\lambda$ on $W$ (and the backbone) shrinks $r_k$ and thereby *raises* the effective angular learning rate — the classic normalized‑network scale/decay coupling. The scale is therefore operational, not cosmetic: $\lambda$ is frozen at $10^{-4}$, per‑block $r_k$ trajectories are logged, and $\lambda\in\{0,10^{-4},5\cdot10^{-4}\}$ is a mandatory ablation.

### 1.4 Code constraints

* **(B) Balance.** For every block $k$ and symbol $j$: $\lfloor C/d\rfloor \le |\{c: y_{c,k}=j\}| \le \lceil C/d\rceil$.
* **(S) Separation.** For all $c\ne c'$: $H(y_c,y_{c'})\ge\delta_{\min}$, Hamming over $K$ blocks.

**Feasibility (Gilbert–Varshamov).** A length‑$K{=}32$, alphabet‑$q{=}16$, min‑distance‑8 code exists with at least $q^K/\sum_{i<8}\binom{32}{i}15^i \approx 3.4\cdot10^{38}/5.8\cdot10^{14}\approx 6\cdot10^{23}$ words $\gg C_{\rm SOP}=11318$. Constraint (S) is never infeasible at our operating points.

### 1.5 Code update (the only non‑standard operation; once per epoch, no extra forward passes)

1. **Accumulate** during the epoch, online: $\hat S^{(k)}_{c,j}=\frac1{n_c}\sum_{i:c_i=c}u^{(k)}_{i,j}$ (window of $E_w=5$ epochs when $n_c<10$, i.e. SOP/In‑Shop).
2. **Hysteresis:** $\hat S^{(k)}_{c,\,y_{c,k}}\mathrel{+}=\gamma$ ($\gamma=0.05$) — incumbent bonus, kills oscillation.
3. **Balanced assignment per block:** $\max_{\sigma}\sum_c \hat S^{(k)}_{c,\sigma(c)}$ s.t. (B). This is a transportation problem with $C$ unit sources and $d$ sinks with lower bound $\lfloor C/d\rfloor$ and capacity $\lceil C/d\rceil$; solved **exactly** by min‑cost flow with lower bounds, $O(Cd\log C)$ ($\le 2.5\cdot10^6$ ops for SOP). Deterministic given $\hat S$.
4. **Exact separation check by pigeonhole (no $O(C^2)$ scan).** Partition the $K$ blocks into $\delta_{\min}$ groups of $K/\delta_{\min}$. Two codewords at Hamming distance $<\delta_{\min}$ must agree *entirely* on at least one group. Bucket classes by their per‑group sub‑codeword and compare only within buckets — no false negatives, $O(KC)$ expected.
5. **Repair.** For each violating pair, take a block where they agree, move the class with the smaller affinity margin to its best alternative symbol having spare capacity that increases $H$. Cap at 100 rounds; on failure, **revert the involved classes to the previous epoch's code** (feasible by induction) — repair therefore always terminates with a feasible code.
6. **Freeze** $Y$ at epoch $T_{\rm freeze}=160$ of 200 so the final 40 epochs optimize against a stationary target.

Initialization: a random balanced code satisfying (S) at epoch 0.

### 1.6 Frozen hyperparameters and recipe

$K=32$, $d=16$ (so $Kd=512$ exactly); $s=16$; $\varepsilon_{\rm ls}=0.1$; $\delta_{\min}=\lceil K/4\rceil=8$; $\gamma=0.05$; $T_{\rm freeze}=160$.
AdamW, lr $10^{-4}$, weight decay $10^{-4}$, cosine decay to 0 with 5‑epoch linear warm‑up, 200 epochs, batch 120, BN frozen (matched to the ProxyAnchor reference recipe), 224 px `RandomResizedCrop(scale=(0.16,1))` + horizontal flip; test `Resize(256)+CenterCrop(224)`. **No P‑K sampler needed** — the loss is per‑sample; plain shuffling is used (and the P‑K sampler is run as a controlled variant so the sampler is never a confound).

**All hyperparameters, including $(K,d)$ and $s$, are selected on a class‑disjoint 20 % held‑out split of the *training* identities, then the model is retrained on the full training split.** No test identity, image, or gallery statistic is ever touched.

### 1.7 Train/test operation table

| | train | test |
|---|---|---|
| forward | backbone + linear + block‑norm | identical |
| loss/readout | $K$ coordinate softmaxes | **none** |
| code table $Y$ | updated per epoch | **discarded** |
| retrieval | — | cosine on $\varphi$, single view |

---

## 2. The causal zero‑shot error mode, and a proof‑level attack on the cheap degeneracies

### 2.1 Error mode: **prototype‑span confinement**

For a class‑private readout (NormSoftmax, ProxyNCA++, ProxyAnchor, SoftTriple, ProxyGML, and the proxy component of PFML), the loss depends on $z$ only through $\{\langle z,p_c\rangle\}_{c=1}^C$, hence

$$\nabla_z\mathcal L\in\mathrm{span}\{p_1,\dots,p_C\},\qquad \dim\le\min(C,512).$$

On CUB ($C{=}100$) and Cars ($C{=}98$) that is a $\le99$‑dimensional *governed subspace* inside a 512‑D descriptor; the orthogonal complement (413 directions) is loss‑invisible and is annihilated by weight decay, while the terminal‑phase equilibrium (neural collapse NC2/NC3) actively pulls the descriptor onto the seen‑class simplex. The deployed function is therefore, up to a decaying remainder, a projection onto a 99‑dimensional subspace **chosen to separate 100 seen birds**. Two unseen species whose distinguishing evidence lies outside that subspace are mapped to the same point. No retrieval rule can recover them; the damage is in the trained function, and it is invisible on seen‑class validation. This is causal, zero‑shot‑specific, and quantitative.

FCS removes the cause rather than penalizing the symptom:

> **Proposition 1 (full‑rank governance).** Under (B) with $C\ge d$, every symbol of every block is the target of at least $\lfloor C/d\rfloor\ge1$ class. Hence $\mathrm{span}\{\varphi(\text{codewords})\}=\mathbb R^{512}$, and since $\nabla_{u^{(k)}}\ell=\tfrac{s}{K}(p^{(k)}-t^{(k)})$ is generically dense in $\mathbb R^d$, all $512$ coordinates are in the loss's active span at every step. The governed dimension rises from $\min(C,512)$ to $K\cdot d=512$.

> **Proposition 2 (no loss of discriminative power).** If $\mathcal L\to0$ with $\varepsilon_{\rm ls}=0$ and (S) with $\delta_{\min}\ge1$, then for every training image the block‑argmax tuple equals $y_{c_i}$, and $c\mapsto y_c$ is injective — the encoder implements exact $C$‑way discrimination. FCS does **not** reduce the information the readout demands; it changes the *topology* of the constraint set from one $C$‑way margin in a $\le C$‑dim span to $K$ independent $d$‑way margins in $K$ orthogonal subspaces.

I explicitly reject the tempting but wrong "more supervision bits" framing: $I(\text{code};\text{label})=\log C$ exactly as before (the code is a deterministic function of the label). What multiplies is **constraint multiplicity in orthogonal subspaces**, not information.

> **Proposition 3 (design‑time, $C$‑independent retrieval margin).** If at test time same‑identity block similarity is $\ge 1-\eta$ and different‑identity block similarity is $\le\xi$ in the $\ge\delta_{\min}$ blocks that separate the two codewords (and $\le1$ elsewhere), then $\cos(\text{same})\ge1-\eta$ and $\cos(\text{diff})\le1-\tfrac{\delta_{\min}}{K}(1-\xi)$. Correct ranking holds whenever $\eta<\tfrac{\delta_{\min}}{K}(1-\xi)$: with $\delta_{\min}/K=0.25$, **up to 25 % of the block‑similarity budget may be destroyed by nuisance variation before ranking fails**, and this tolerance is fixed by the code at design time, independent of $C$ and of optimization success. Learned proxies attain their separation only asymptotically (and in practice need a $100\times$ proxy learning rate).

### 2.2 Degeneracies, and why each is excluded

| Degeneracy | Why it is the cheapest cheat | Exclusion |
|---|---|---|
| **D1** All classes share one codeword; $u^{(k)}\equiv e_1$; $\mathcal L\to0$; $\varphi$ constant; R@1 → chance | global infimum of $\mathcal L$ | **Infeasible under (B)**: $|\sigma^{-1}(j)|\le\lceil C/d\rceil<C$ for $d\ge2$. The constraint set is load‑bearing; the loss alone is provably insufficient. |
| **D2** Two classes share a codeword: zero loss, zero discrimination | second‑cheapest | Excluded by (S) with $\delta_{\min}\ge1$; repair + revert guarantees a feasible code every epoch (§1.5.5). |
| **D3** All $K$ blocks compute the same function | collapses $K$ problems to 1 | Then at most $d$ distinct codewords exist, contradicting injectivity whenever $C>d$ (always: $98>16$). Stronger: (S) forces **every** class pair to be separated by $\ge\delta_{\min}=8$ *distinct* blocks, so the pairwise signal is replicated across 8 orthogonal subspaces. |
| **D4** A block outputs a constant symbol | silences a block | Under (B) the block's targets are near‑uniform over $d$ symbols, so the best constant predictor incurs CE $\ge\log d=2.77$ nats, versus $\to0$ for a partition‑consistent block: strictly suboptimal, with a strict descent direction away from constancy. |
| **D5** A block shrinks its norm to opt out | free‑riding | The loss reads $u^{(k)}$, which is scale‑free, so there is no gradient path that silences a block; and because deployment uses the *same* block normalization, no block can dominate the test‑time metric either. This is why block normalization is **necessary**, not decorative (control C3 tests exactly this). |
| **D6** Code‑chasing: the assignment latches onto a nuisance factor correlated with identity (background, pose, lighting) | genuinely possible | Partially, not fully, excluded. (S) forbids any *set* of partitions that fails to separate some class pair, so nuisance factors survive only if collectively identity‑determining; hysteresis $\gamma$ and the epoch‑160 freeze damp latching. **Residual risk stated plainly**, with the diagnostic: block‑argmax stability under the standard crop/flip augmentations (a nuisance‑aligned block is unstable). This is FCS's honest weak point. |

**What FCS does *not* prove.** The constraint binds the *readout*, not the backbone: a backbone that memorizes identity in 2048‑D can satisfy all $K$ partitions by linear decoding. The claim is not that memorization is impossible; it is that (i) the governed descriptor subspace is 512‑D rather than $\le C$‑D, (ii) every class pair is separated redundantly in $\ge\delta_{\min}$ orthogonal subspaces, and (iii) no identity owns a private direction. Controls C2/C5/C6 are designed to falsify exactly this.

---

## 3. Adversarial novelty search (primary sources), with a one‑sentence mechanism distinction each

**Nearest work, inside DML**

1. **DREML** (ECCV 2018) — *random* meta‑class partitions, $L{=}48$ **separate** ResNet‑18s, $D{=}12$, deploy the $D{\times}L{=}576$‑D concatenation. → FCS delivers the same partition‑supervision diversity inside **one** network and **one** descriptor by giving each partition a disjoint normalized slice, replaces random partitions with per‑epoch balanced‑assignment codes carrying a guaranteed minimum Hamming distance, and — verified in the primary source — DREML contains **no single‑network multi‑head variant** and no code constraints.
2. **Divide‑and‑Conquer the Embedding Space** (CVPR 2019) — partitions the **data** into $K$ clusters; each embedding chunk trains on its own cluster only. → FCS partitions the **label alphabet**, every block trains on every sample, and the $K$ partitions are constrained to be jointly injective with min distance $\delta_{\min}$.
3. **BIER / A‑BIER** — embedding groups made diverse by boosting reweighting and decorrelation penalties. → FCS's groups differ by their *supervision target* (a learned $d$‑ary partition of the class set), and FCS adds no decorrelation or reweighting term at all.
4. **HPL** (WACV 2022) / **HIER** (CVPR 2023) — hierarchical (tree) proxies capturing class‑shared structure. → A hierarchy is a *nested* family with $\approx C$ leaves, so representable identity count does not exceed the seen‑class count; FCS uses $K$ *non‑nested, jointly injective* partitions whose intersection lattice has $d^K$ cells, and it deletes class‑private proxies entirely, which no hierarchical method does.
5. **PFML** (CVPR 2025), ProxyAnchor, ProxyNCA++, SoftTriple, **ProxyGML** ("fewer proxies" = fewer *per class*) — all keep class‑private (multi‑)proxies. → FCS allocates **zero parameters per class**; identity is a discrete codeword over fixed orthonormal coordinates, so the readout cannot spend a private direction on any identity.
6. **CSQ** (CVPR 2020) and **minimal‑distance‑separated hash centers** (CVPR 2023) — assign classes well‑separated centers with a Hamming‑distance guarantee. → Those centers are binary (alphabet 2), Hadamard‑fixed or learned as continuous vectors, deployed as **quantized codes** on **seen‑class** hashing benchmarks; FCS uses alphabet $d=512/K$ of orthonormal atoms, re‑solves the assignment from data by balanced transport each epoch, deploys a **continuous float** descriptor under cosine, and targets identity‑disjoint retrieval.
7. **OPQN** (PR 2023), DPQ, **SPQ** (CVPR 2022) — product codebooks for compact retrieval codes, with a per‑identity classifier retained in each subspace. → PQ uses product structure to **compress the deployed descriptor** while identities keep private parameters; FCS uses it as an **over‑subscribed label alphabet** ($\sim C/d$ identities per symbol, 6.25 on CUB, 707 on SOP) with no per‑class parameters and no deployed quantization.
8. **Anti‑Collapse Loss** (2024, coding rate), **ρ‑spectrum regularization** (ICML 2020), **Non‑isotropy Regularization** (CVPR 2022), **MMCR** (NeurIPS 2023) — additive penalties that flatten/enlarge the used subspace. → FCS adds **no penalty term**; full‑rank governance follows from the readout topology plus constraint (B) (Prop. 1). Control **C6** exists precisely to kill FCS if spectral flattening alone reproduces the gain.
9. **Proxy Synthesis** (AAAI 2021), **MemVir**, **Metrix** — synthesize virtual classes/embeddings to mimic unseen classes. → FCS creates unseen‑identity capacity **structurally** ($d^K-C$ unoccupied codewords) without synthesizing or interpolating anything.
10. **VAPNet** (NeurIPS 2023, "Learning to Parameterize Visual Attributes for Open‑set Fine‑grained Retrieval") — patch‑level attribute exploration + parameterization modules. → FCS adds no module, no patch machinery, no attribute head; its "attributes" are the $d$‑ary block partitions implied by the code, at zero architectural cost.
11. **Fixed classifiers** (Hoffer et al., "Fix your classifier", 2018) — fixed Hadamard/orthonormal classifier weights. → FCS's $K{=}1$ case is essentially this; the novelty is $K>1$ over an **over‑subscribed** alphabet with a learned assignment — and $K{=}1$ is run as control **C4** to isolate it.
12. **SwAV / DINO** prototypes with Sinkhorn balancing — balanced online assignment of *samples* to prototypes, unsupervised, one prototype set. → FCS balances the assignment of **classes** to symbols using ground‑truth identity, factorized across $K$ blocks, with a min‑distance constraint that has no SwAV analogue.

**Nearest work, outside DML**

13. **ECOC** (Dietterich & Bakiri 1995) — decompose multiclass into fixed binary dichotomies for seen‑class accuracy. → FCS's codes are data‑adaptive, balanced, $d$‑ary, tied to orthogonal descriptor slices, and its object is the deployed geometry on **unseen** identities.
14. **Shu & Nakayama, "Compressing Word Embeddings via Deep Compositional Code Learning"** (ICLR 2018) — $K$‑way $D$‑dimensional codes over shared codebooks, Gumbel‑softmax. → Their codes are fit to **reconstruct a pretrained embedding table** for compression, with no encoder and no unseen‑token claim; FCS learns the code jointly with an image encoder by exact balanced assignment (no Gumbel / straight‑through), and the code is the *only* supervision channel.
15. **Barlow's factorial codes / sparse distributed coding; combinatorial coding in olfaction** — the scientific import: identity as a pattern over reusable, over‑subscribed features rather than a grandmother cell.
16. **Neural collapse in transfer** (e.g., NeurIPS 2024 geometric‑complexity work) — supplies the diagnosis (feature space confined to a $C{-}1$ simplex limits expressivity for unseen classes), not the remedy. → FCS's remedy is a factorized label alphabet, not an ETF prescription.

I found **no** primary source that (a) removes per‑class parameters entirely in DML, (b) supervises through $K$ orthogonal‑slice $d$‑way readouts with $d\ll C$, and (c) re‑optimizes the class→symbol assignment by balanced transport under a min‑Hamming constraint for identity‑disjoint retrieval. An informal precedent exists in practitioner writing ("embedding as a concatenation of softmax feature groups"), which I disclose rather than ignore; it carries no code constraints, no assignment optimization, and no zero‑shot mechanism claim.

---

## 4. Decisive matched‑compute controls

All controls share backbone, recipe, epochs, augmentation, seeds (5), and evaluation code; only the named factor changes.

| # | Control | What it kills if it wins |
|---|---|---|
| **C1** | **Random frozen code** (balanced + $\delta_{\min}$, never updated) = "DREML in one net" | Kills the *learned‑code* claim (factorial readout survives). |
| **C2** | **NormSoftmax / ProxyAnchor, 512‑D, whole‑vector L2** at identical cost | Kills the whole method if it matches. |
| **C3** | **Block‑normalized descriptor + ordinary $C$‑way proxy loss** | Separates deployment geometry from supervision structure. |
| **C4** | **$K{=}1$, $d{=}512$** (fixed orthonormal per‑class proxies; needs $C\le512$, so CUB/Cars only) | Kills the *product* claim, leaving "fixed classifier". |
| **C5** | **Over‑subscription sweep** $(K,d)\in\{(64,8),(32,16),(16,32),(8,64),(4,128)\}$, i.e. $\rho=C/d$ from 12.5 down to 0.78 on CUB | The stated mechanism *requires* monotone degradation as $\rho\to1$; non‑monotonicity falsifies it. |
| **C6** | **Rank‑matched control**: NormSoftmax + coding‑rate/ρ‑spectrum regularizer, tuned so its measured *test* effective rank matches FCS's | **The decisive one.** If it comes within 0.3 pt, FCS is a re‑parameterization of an occupied mechanism and must be rejected. |
| **C7** | **PFML reproduction** (§7 ambiguities resolved by sweep), same code base, 5 seeds | Establishes whether the frontier number survives an in‑house recipe. |
| **C8** | Code‑update cadence: every epoch / every 10 / frozen after epoch 1; $T_{\rm freeze}\in\{100,160,200\}$ | Isolates the alternating‑optimization contribution. |
| **C9** | $\delta_{\min}\in\{0,1,4,8,12\}$ | Tests Prop. 3's margin claim directly. |
| **C10** | Sampler: plain shuffle vs P‑K (4 per class) for FCS *and* baselines | Removes batch composition as a confound. |

**Diagnostics (measurements, not claims):** test‑set effective rank (participation ratio of $\varphi$ singular values on *unseen* classes); pairwise normalized MI between block‑argmax variables on test data (mechanism predicts low redundancy); symbol‑usage histograms; block‑argmax stability under augmentation (D6 probe); per‑block $\|\tilde z^{(k)}\|$ trajectories (scale/decay coupling).

---

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic

**Frozen before any run.** Lane A, ResNet‑50, 512‑D, 224 px, single‑view cosine, 200 epochs, 5 seeds, mean ± std.

| Dataset | Reference (Lane A) | **FCS forecast R@1** | Predicted delta |
|---|---|---|---|
| CUB‑200‑2011 | PFML **0.734 ± 0.003** | **0.745 ± 0.005** | **+1.1 pt (crossing claimed)** |
| Cars196 | PFML **0.927 ± 0.003** | **0.931 ± 0.004** | +0.4 pt (crossing *not* claimed) |
| SOP | PFML **0.829 ± 0.002** | **0.829 ± 0.003** | **0.0 pt — a deliberate null** |
| In‑Shop (secondary) | PA+DADA **0.930** (seeds unreported) | 0.928 ± 0.004 | no crossing predicted |

The SOP/In‑Shop null is a *risky* prediction that distinguishes FCS's mechanism from a generic improvement: for $C>512$ the baseline's governed subspace is already full rank, so the span mechanism has nothing to add and only the (weaker) optimization‑stability argument remains. **A uniform gain across all four datasets would be evidence against my stated mechanism, not for it.**

**Frontier‑crossing arithmetic (CUB).** With $\sigma\approx0.004$ and $n=5$ per arm, $\mathrm{SE}_{\rm diff}=\sqrt{2\sigma^2/5}\approx0.0025$. The forecast $+0.011$ is $\approx4.4\,\sigma$ (Welch $p\approx0.005$). The crossing is claimed **only if**
$$\overline{\rm FCS}_{\rm CUB}-\max\big(0.734,\ \overline{\rm PFML^{repro}}_{\rm CUB}\big)\ \ge\ 0.007\quad\text{and}\quad p<0.01 .$$
If the in‑house PFML reproduction lands below 0.734 (likely — see §7), the comparison against the published 0.734 is reported as **recipe‑unmatched** and labelled as such; I will not inherit a published frontier obtained under an undisclosed recipe by beating my own weaker reproduction.

**Falsification thresholds (any one triggers rejection or the stated retraction):**

* **F1** CUB gain over in‑house NormSoftmax (C2) $<+1.5$ pt → the factorial readout does not carry the mechanism → **reject**.
* **F2** FCS test effective rank on unseen CUB classes not $\ge1.5\times$ C2's → span‑expansion mechanism **falsified even if R@1 improves** (the method would then work for an unknown reason and must not be published as this mechanism).
* **F3** C5 non‑monotone in $\rho$ → over‑subscription mechanism falsified.
* **F4** C6 (rank‑matched spectral regularizer) within 0.3 pt → occupied mechanism → **reject**.
* **F5** C1 (random codes) within 0.2 pt → the learned‑code claim is withdrawn; the method is re‑reported as "DREML‑in‑one‑network", with DREML credited as the origin.
* **F6** CUB crossing $<+0.7$ pt over the better of published/reproduced PFML → **no frontier claim is made**, only a cost‑parity claim.

**Calibrated belief (mine, stated plainly):** P(F1 passes) ≈ 0.55; P(CUB frontier crossing as forecast) ≈ 0.35; P(SOP null holds within ±0.4 pt) ≈ 0.6; P(C6 kills it) ≈ 0.2; P(C1 kills the learned‑code component) ≈ 0.35. The single most likely disappointing outcome is that FCS matches C2 within noise on CUB while raising effective rank — mechanism confirmed, benefit absent.

---

## 6. Cost, and benchmark/contamination risk

**Training.** Added FLOPs: $K$ softmaxes over 512 total logits per sample ($\sim1.5$k FLOPs) against ResNet‑50's $\sim4.1$ GFLOPs → $<10^{-6}$ relative. Code update: $O(KCd)$ affinity + $K$ min‑cost flows $\approx2.5$M ops/epoch (SOP) → negligible. **Memory strictly lower than the baselines**: FCS holds $CK$ integers (CUB 3.2 k; SOP 362 k) and *no* proxy tensor, versus PFML's $M{\cdot}C{\cdot}512$ floats plus Adam moments ($15{\times}100{\times}512=0.77$ M on CUB; $2{\times}11318{\times}512=11.6$ M on SOP, $\times3$ with optimizer state). Forecast wall clock **1.00–1.01×** baseline epoch time, memory $\le$ baseline. Compare PA+DADA at 1.06× epoch time / 1.01× memory.

**Deployment.** Byte‑for‑byte the baseline: one ResNet‑50 forward, one 224 px view, one 512‑D descriptor, cosine NN. No auxiliary encoder, no reranking, no test‑time fitting. The code table is not shipped.

**Benchmark and contamination risk.** CUB and Cars categories overlap ImageNet‑1K pretraining; this is lane‑standard and applies equally to every reference, but it inflates all absolute Lane‑A numbers and should not be read as clean zero‑shot transfer. SOP/In‑Shop splits are identity‑level with less pretraining overlap. FCS uses only official training images and identity labels plus ordinary stochastic augmentation; no test data, no external or generated data, no text/VLM encoder, no extra annotation, no transduction, no reranking, no gallery fitting. Hyperparameters are chosen on a class‑disjoint split of the *training* identities. Residual risk: the per‑epoch code update reads training‑set statistics, which is a mild extra fit to the training identities (not to the test set) — measured by the train‑identity/held‑out‑identity generalization gap in C8.

---

## 7. Unresolved source ambiguities (stated, not papered over)

1. **PFML's disclosed reduction** (fetched from the arXiv HTML): $\psi_{\rm att}(r,z)=-\|r-z\|^{-\alpha}$ (clipped to $-\delta^{-\alpha}$ inside $\delta$), $\psi_{\rm rep}$ the mirror image; $\Psi_j=\Psi_{j,\rm att}+\Psi_{j,\rm rep}$ aggregating batch embeddings **and** $M$ proxies per class; objective $\mathcal U=\sum_i\Psi_{y_i}(z_i)+\sum_{j,k}\Psi_j(p_{j,k})$; $M{=}15$ (CUB/Cars), $M{=}2$ (SOP); Adam, lr $5\cdot10^{-4}$, proxy lr $\times100$, 200 epochs, ResNet‑50/512‑D/224 px.
2. **Ambiguous or undisclosed in the fetched source:** the exact $\alpha$ (rendered as "$\alpha\in\{0,6\}$", most plausibly the range $[0,6]$), the per‑dataset $\delta\in[0.1,0.3]$, batch composition (classes × samples), weight decay, LR schedule/warm‑up, BN freezing, augmentation, and In‑Shop's $M$. C7 must sweep these.
3. **A material recipe gap.** PFML's ResNet‑50/512‑D CUB 0.734 sits ~3.5 pt above the *official ProxyAnchor repository's* ResNet‑50/512‑D number (0.699 CUB, 0.877 Cars). That gap is too large to attribute to the loss alone, so the Lane‑A frontier cannot be treated as recipe‑matched to a ProxyAnchor‑style baseline without C7.
4. **PA+DADA In‑Shop 0.930** — seed count and uncertainty unreported (as given), so any In‑Shop comparison is single‑point.
5. **Sources I could not read in primary form:** the DREML PDF and the OPQN PDF returned binary/403; my DREML facts (random partitions, $L{=}48$, $D{=}12$, per‑member dim $=D$, total 576‑D, CUB 63.9 / Cars 86.0 / In‑Shop 78.4, **no single‑network multi‑head ablation**, SOP not reported) come from a rendered HTML version, and my OPQN characterization from its abstract and secondary summaries. Both mechanism distinctions in §3 must be re‑verified against the primary PDFs before publication.
6. **Not exhaustively searched:** whether any deep‑hashing follow‑up has applied CSQ‑style separated centers to identity‑disjoint DML splits under a continuous descriptor. If such a paper exists, it is FCS's nearest neighbour and the novelty claim narrows to the over‑subscribed $d$‑ary alphabet + balanced re‑assignment.

---

### Plain statement of what could sink this

The mechanism binds the readout, not the backbone (D6 and §2.2 caveat). The strongest single threat is **C6**: if flattening the descriptor spectrum with an existing regularizer reproduces the gain, FCS is a re‑parameterization of occupied work and should be rejected. The second is **C1**: if random codes match learned codes, the contribution reduces to "DREML with one network", which is still a real single‑model/single‑descriptor result at 1.00× cost but is a much smaller claim, and DREML must then be credited as the mechanism's origin. I forecast a frontier crossing on CUB only, an insignificant gain on Cars, and a deliberate null on SOP.

**Sources:**
- [Potential Field Based Deep Metric Learning (PFML)](https://arxiv.org/html/2405.18560v1)
- [Deep Randomized Ensembles for Metric Learning (DREML)](https://ar5iv.labs.arxiv.org/html/1808.04469)
- [Proxy Anchor Loss official repository](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Divide and Conquer the Embedding Space for Metric Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)
- [Hierarchical Proxy-based Loss for Deep Metric Learning](https://arxiv.org/abs/2103.13538)
- [HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.pdf)
- [Fewer is More: Deep Graph Metric Learning Using Fewer Proxies (ProxyGML)](https://arxiv.org/pdf/2010.13636)
- [Central Similarity Quantization (CSQ)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf)
- [Deep Hashing With Minimal-Distance-Separated Hash Centers](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Hashing_With_Minimal-Distance-Separated_Hash_Centers_CVPR_2023_paper.pdf)
- [Orthonormal Product Quantization Network](https://arxiv.org/pdf/2107.00327)
- [Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/pdf/1711.01068)
- [Anti-Collapse Loss for Deep Metric Learning Based on Coding Rate Metric](https://arxiv.org/pdf/2407.03106)
- [Non-isotropy Regularization for Proxy-based Deep Metric Learning](https://arxiv.org/pdf/2203.08547)
- [Proxy Synthesis: Learning with Synthetic Classes for DML](https://arxiv.org/pdf/2103.15454)
- [Learning to Parameterize Visual Attributes for Open-set Fine-grained Retrieval (VAPNet)](https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf)
- [Adversarial Reconstruction Feedback for Robust Fine-grained Generalization (AdvRF)](https://arxiv.org/abs/2507.21742)
- [Deep Factorized Metric Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf)
- [Sharing Matters for Generalization in Deep Metric Learning](https://arxiv.org/abs/2004.05582)
- [The Impact of Geometric Complexity on Neural Collapse in Transfer Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b24015f3af598e1d9179f6e06353780-Paper-Conference.pdf)
