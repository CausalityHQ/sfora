# Frozen Pass 78 proposal (blind proposer; verbatim)

 You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
 ## Proposal: **MCB‑PA — Meta‑Class Block Codes for Proxy Anchor**
 
 One method. Train‑time only. Deployed descriptor is bit‑identical in shape to the baseline (512‑d, L2‑normalized, cosine, single view, single model, no re‑ranking).
 
 ---
 
 ### 1. Causal failure mode
 
 Proxy Anchor's loss depends on an embedding $x$ **only** through the $C$ scalars $\{x\cdot p_c\}_{c=1}^C$. Therefore
 
 $$\frac{\partial L_{\mathrm{PA}}}{\partial x}=\sum_{c}\frac{\partial L}{\partial s_c}\,p_c\ \in\ \mathrm{span}(P),\qquad \frac{\partial L_{\mathrm{PA}}}{\partial f}=\tfrac{1}{\|f\|}(I-xx^\top)\frac{\partial L}{\partial x}.$$
 
 The normalization Jacobian adds only a radial‑removal term along $x$ — it carries no label information. So **the label‑bearing supervisory signal on a 512‑d descriptor lives in a subspace of rank $\le C$.** On CUB ($C{=}100$) and Cars196 ($C{=}98$) that is $\le 100$ of 512 directions. The remaining $\gtrsim 400$ directions are shaped only by backbone inductive bias and weight decay, yet they contribute variance to the deployed cosine similarity on *unseen* classes — signal‑free dilution. This is consistent with the observed fast spectral decay of DML embeddings and with the finding that spectral entropy $\rho$ correlates with zero‑shot generalization (Roth et al., ICML 2020).
 
 The fix must add *more independent supervised directions* using only training labels. Repeating the same $C$‑way task per direction adds rank but not diversity. Random **coarse partitions of the class set** add both: each block must compute a visual predicate that separates an arbitrary grouping of species/models — such a predicate cannot be a class detector, so it is attribute‑like and transfers to disjoint identities.
 
 ---
 
 ### 2. Training object
 
 Embedding $f=W\,\mathrm{pool}(\phi(I))\in\mathbb R^{512}$, $x=f/\|f\|$. Split coordinates into $M$ contiguous blocks $B_1..B_M$ of size $k=512/M$; sub‑descriptor $u^m=x_{B_m}/\|x_{B_m}\|\in S^{k-1}$, block energy $a_m=\|x_{B_m}\|$, $\sum_m a_m^2=1$.
 
 For each block $m$ draw $J$ balanced random surjections $\pi_{m,j}:\{1..C\}\to\{1..G\}$ once at start of training (frozen; seeded), with learned meta‑proxies $\Pi^{m,j}=\{q^{m,j}_1..q^{m,j}_G\}\subset S^{k-1}$.
 
 With $L_{\mathrm{PA}}(X,P)=\frac{1}{|P^+|}\sum_{p\in P^+}\log\!\big(1+\!\!\sum_{x\in X^+_p}\!e^{-\alpha(x\cdot p-\delta)}\big)+\frac{1}{|P|}\sum_{p\in P}\log\!\big(1+\!\!\sum_{x\in X^-_p}\!e^{\alpha(x\cdot p+\delta)}\big)$:
 
 $$\boxed{\;L=\underbrace{L_{\mathrm{PA}}(x,P;y)}_{\text{fine, }C\text{ proxies}}\;+\;\frac{\lambda}{MJ}\sum_{m=1}^{M}\sum_{j=1}^{J}L_{\mathrm{PA}}\big(u^m,\Pi^{m,j};\pi_{m,j}(y)\big)\;+\;\beta\sum_{m=1}^{M}\Big(a_m^2-\tfrac1M\Big)^2\;}$$
 
 **Design rule (pre‑registered, not tuned):** $J\cdot G\ge k$, so $\bigcup_{m,j}\mathrm{span}(\Pi^{m,j})$ covers all 512 directions; and $G=\mathrm{clip}(\mathrm{round}(C/6),8,64)$ (CUB/Cars $\to 16$, In‑Shop $\to 64$). Defaults $M{=}8,k{=}64,J{=}4,\lambda{=}0.5$ (linear warmup over epochs 1–5), $\beta{=}1$, $\alpha{=}32,\delta{=}0.1$.
 
 **Why the third term exists.** Deployed similarity decomposes exactly as $x\cdot y=\sum_m a_m(x)a_m(y)\,(u^m\!\cdot v^m)$ — a norm‑weighted sum of $M$ block metrics. Because $u^m$ is renormalized, there is a degenerate solution: drive $a_m\to0$ and satisfy the block loss on a direction that no longer influences retrieval. The energy‑balance penalty blocks it; block‑energy histograms are a mandatory training diagnostic.
 
 **Notes on rigor.** A fixed random rotation before blocking is a no‑op (absorbable into $W$; cosine is rotation‑invariant, $\|QW\|_F=\|W\|_F$) — so contiguous blocks are used, and any paper‑style claim about "random rotations" would be vacuous. Meta‑proxies are near‑always populated per batch ($G\ll$ batch classes), which incidentally densifies PA's positive term, normally active on only a few proxies per step.
 
 ---
 
 ### 3. Expected effect, forecast, falsifiers
 
 **Baseline arm:** PA, ResNet‑50, 512‑d, published recipe verbatim. Extra params $MJGk=32{,}768$ ($\approx 0.13\%$ of the head, $\approx 10^{-3}$ of the model); extra FLOPs $\sim10^{-6}$ of the backbone; test‑time cost exactly zero. Matched compute is therefore trivially met, but still enforced: identical epochs, batch, schedule, augmentation, seed set, plus a $+1\%$‑epoch baseline arm and wall‑clock reporting.
 
 **Forecast** (mean paired $\Delta$R@1 over 5 shared seeds, vs matched baseline):
 
 | | CUB | Cars196 | In‑Shop |
 |---|---|---|---|
 | point | **+1.0** | **+0.8** | **+0.1** |
 | 80% interval | −0.3 … +2.3 | −0.5 … +2.0 | −0.6 … +0.8 |
 
 The **differential** is the mechanism‑specific prediction: $C\ll512$ on CUB/Cars (binding) vs $C{=}3997\gg512$ on In‑Shop (not binding). No generic regularizer predicts this pattern. Secondary: under random (non‑ImageNet) init the CUB gain should *grow*, since the unsupervised complement has no useful generic prior to fall back on.
 
 **Falsifiers, pre‑registered:**
 - **Primary (kills the method):** CUB mean $\Delta$R@1 $\le +0.3$ with a 95% paired‑bootstrap CI containing 0.
 - **Mechanism (kills the explanation, not necessarily the method):** gain exists but (a) CUB/Cars advantage does not exceed In‑Shop advantage by $\ge0.5$ R@1, **and** (b) singular‑value entropy of the test embedding matrix does not increase vs baseline. Then report it as unexplained generic regularization and say so.
 - **Redundancy (kills the additivity claim):** if MCB‑PA $+$ S2SD $\approx$ S2SD alone, the mechanisms overlap; report the interaction, not the standalone delta.
 - **Degeneracy check:** if $\min_m a_m^2 < 0.3/M$ at convergence, the run is invalid regardless of R@1.
 
 **Ablation lattice** (separates the two mechanisms; each is a degenerate corner of the same code path): A0 PA; A1 meta‑class head on the *full* 512‑d (no blocking, $M{=}1$); A2 blocks with *true class* labels (no meta‑classes, $G{=}C$ — a BIER‑lite control); A3 proposed; A4 A3 with $\beta{=}0$; A5 semantic (clustered) instead of random partitions.
 
 **Selection correction:** all of $(M,k,G,J,\lambda,\beta)$ chosen on a held‑out *pseudo‑zero‑shot split of training classes* (last 20 train identities, never test), one final test run at the frozen config. Report raw best, the full sweep distribution, seed‑level scatter, and a BH‑corrected p‑value over the number of configs evaluated. Metrics: R@1/2/4/8, NMI, and mAP@R.
 
 ---
 
 ### 4. Deployment constraints
 
 Descriptor, dimensionality, normalization, similarity, and inference cost are unchanged — protocol‑compliant for single‑model/single‑view/fixed‑descriptor evaluation. Needs only class labels (already available); no external models. Requires a class‑balanced sampler (standard for PA) and a batch large enough that most meta‑proxies see positives (≥64). Block proxies use the same 100× proxy LR as PA. Assumption to flag: ImageNet‑pretrained init is the field‑standard and is not a generative/text model, but if the program forbids it, the method is unchanged and only absolute numbers move. On In‑Shop the $G{=}64$ rule leaves ~62 classes per meta‑group; the block predicate may be uninformative there — that is a predicted null, not a bug.
 
 ---
 
 ### 5. Closest primary‑literature risks
 
 1. **BIER / A‑BIER** (Opitz et al., ICCV'17; TPAMI) — *the* structural prior art: one embedding split into groups, each with its own loss, plus online boosting and adversarial decorrelation. Delta here: PA supervision, random meta‑class label codes instead of boosting reweighting, no decorrelation term, retained full‑descriptor loss, energy‑balance term, explicit rank‑coverage rule. **Rediscovery risk is real and must be stated in any writeup.**
 2. **DREML** (Xuan et al., ECCV'18) — random meta‑class partitions, but an ensemble of separate networks with concatenated descriptors (violates single‑model/fixed‑descriptor). Delta: one backbone, one 512‑d vector, zero test cost.
 3. **SoftTriple / Sub‑center ArcFace** — multiple centers per class: raises proxy rank via *finer* labels, the opposite direction from coarse codes.
 4. **ECOC** (Dietterich & Bakiri, 1995) and deep variants — the label‑code idea itself.
 5. **Matryoshka Representation Learning** (Kusupati et al., NeurIPS'22) — losses on sub‑descriptors, but nested prefixes with identical labels and a different goal (adaptive dimensionality).
 6. **Roth et al., ICML'20** — supplies the $\rho$/spectral‑entropy mechanism metric *and* the warning that many DML gains vanish under equalized protocols; **Musgrave et al., ECCV'20** ("Reality Check") is why the forecast above is deliberately ~1 point, not ~3.
 7. **S2SD** (ICML'21) and **NIR** (CVPR'22) — the strongest current PA‑compatible regularizers; MCB‑PA must be benchmarked against at least one and tested for additivity.
 8. **Divide‑and‑Conquer the Embedding Space** (CVPR'19) — also splits the embedding, but partitions *data* by clustering rather than *labels* by random code. **HIST** (CVPR'22) — hierarchical proxies from a semantic hierarchy; MCB‑PA's A5 arm is the semantic‑partition version of the same knob.
