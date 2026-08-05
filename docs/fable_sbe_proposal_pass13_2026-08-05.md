# PROPOSAL (one method): Self-Balancing Embeddings (SBE) — Balanced-Coupling Likelihood with a Deployment Balance-Gap Penalty

**Lane: A** (ResNet-50, 512-D L2-normalized descriptor, ~224 px, single-view cosine retrieval, ≤200 epochs). All forecasts and comparisons below are Lane A only.

**One-paragraph thesis.** Zero-shot retrieval fails partly through *retrieval-mass anisotropy* (hubness): the learned kernel concentrates rank-1 mass on a small set of gallery points. Supervised training erases the *symptoms* among seen classes (each seen misranking is individually corrected by label gradients) without removing the anisotropy of the kernel itself, so the errors re-manifest exactly where labels never reached — unseen identities. All published fixes for this live at inference (CSLS, inverted softmax, mutual proximity, transductive noHub) and are banned under this protocol's deployment rules. SBE internalizes the fix at train time: the batch retrieval likelihood is computed under the *symmetric Sinkhorn-balanced* (doubly-stochastic) coupling of the batch kernel, and a *balance-gap penalty* forces the raw kernel itself — the thing deployed — to be the balanced one. Deployment is untouched 512-D cosine NN.

---

## 1. Executable mathematics

**Architecture / deployment.** ResNet-50 (ImageNet-1K init, torchvision weights), FC removed, BatchNorm layers frozen in eval mode (standard Proxy-Anchor practice). GAP → Linear(2048→512), no bias → `u = z/‖z‖₂`. Test: resize 256, center-crop 224, one view, cosine NN. Nothing else at test — no centering, no whitening, no normalization fit, no gallery statistics.

**Sampler.** Class-balanced batches: CUB/Cars 30 classes × 3 images (B=90); SOP/In-Shop 45 classes × 2 (B=90). Augmentation: RandomResizedCrop(224, scale (0.16,1)), horizontal flip. Nothing else.

**Batch kernel.** For batch embeddings u₁…u_B, similarities S_ij = uᵢ·uⱼ, kernel

  K_ij = exp(S_ij/τ_b) for i≠j, K_ii = 0,  τ_b = 0.1.

**Symmetric Sinkhorn balancing (learned object: none — it is a differentiable fixed-point operator).** Find s ∈ ℝ^B_{>0} with

  sᵢ Σⱼ K_ij sⱼ = 1 ∀i  ⟺ P = diag(s) K diag(s) is symmetric doubly stochastic.

Existence/uniqueness holds for symmetric K with positive off-diagonal (Sinkhorn–Knopp; symmetric case per Knight 2008). Solve in log domain with aᵢ = log sᵢ, damped fixed-point (damping is required for symmetric Sinkhorn; cf. Feydy et al. 2019):

  aᵢ^{(t+1)} = (1−η) aᵢ^{(t)} + η · [ −LSEⱼ≠ᵢ( S_ij/τ_b + aⱼ^{(t)} ) ], η = 0.5, T = 15 iterations, a^{(0)} = 0.

Backprop by unrolling all T iterations (no detach anywhere: gradients flow into S through both the kernel entries and the potentials a). P_ij = exp(aᵢ + aⱼ + S_ij/τ_b).

**Loss 1 — Balanced-Coupling Likelihood.** With P(i) = same-class batch mates of i (≥1 by the sampler):

  L_bal = −(1/B) Σᵢ log( Σ_{j∈P(i)} P_ij ).

This is the log-mass the balanced retrieval coupling puts on the correct identity: an NCE-type likelihood, but normalized with *row and column* constraints instead of row-only softmax.

**Loss 2 — Balance-Gap penalty (the deployment-consistency term).**

  L_gap = Var(a) = (1/B) Σᵢ (aᵢ − ā)².

Exact certificate (Prop. 3 below): Var(a)=0 ⟺ K𝟙 = c𝟙 ⟺ the *raw* kernel is already balanced, so no scaling is needed at deployment.

**Base loss.** Proxy-Anchor, standard hyperparameters (α=32, δ=0.1; one 512-D proxy per class; proxy LR ×100; 3 proxy-only warmup epochs).

**Total.** L = L_PA + λ_b L_bal + λ_g L_gap, with λ_b ramped 0→1 over epochs 0–10 and λ_g ramped 0→0.1 over epochs 10–50 (let clusters form, then squeeze the gap).

**Optimization.** AdamW, LR 1e-4 (backbone and head), wd 1e-4, cosine decay, 200 epochs, evaluate final checkpoint only. Hyperparameters (τ_b ∈ {0.05, 0.1, 0.2}, λ_g ∈ {0.03, 0.1, 0.3}) tuned once on an internal class-disjoint validation split (last 15% of *training* classes, Musgrave-protocol style), then frozen and retrained on the full training classes; 5 seeds; report mean ± 95% CI. Numerical guards: clamp S/τ_b ≤ 60; monitor marginal error ‖P𝟙−𝟙‖_∞ < 1e-3 (met in ≤15 damped iterations at τ_b=0.1 in this regime).

Gradient path summary: ∂L_bal/∂θ has two channels — the direct positive-pair channel (pull genuine pairs together) and the potential channel: a hub point j has large row mass ⇒ aⱼ ≪ 0 ⇒ every positive pair containing j pays −(aᵢ+aⱼ), so the gradient specifically pushes j's *impostor* similarities down. That column-wise correction channel is what row-softmax losses do not have.

## 2. Causal zero-shot error mode and degeneracy attacks

**Error mode (one, causal).** On the unit sphere, points whose position aligns with locally dense regions / the data mean direction receive systematically higher similarity to random queries and become rank-1 "thieves" (hubness; Radovanović et al., JMLR 2010; demonstrated for hyperspherical few-shot features by noHub, CVPR 2023). Crucially this mode is *train-invisible and test-manifest*: on seen classes, CE/margin gradients correct each individual theft using labels, but nothing constrains the kernel's column-mass profile on the residual degrees of freedom, so anisotropy persists and bites precisely the identities that never had labels. Inference-side corrections (CSLS, inverted softmax, mutual proximity, transductive noHub) demonstrably recover such errors in adjacent fields but are *banned here* — which is exactly why a train-time internalization is the valuable move under this protocol.

**Lemma 1 (no conflict with the supervised optimum — the key difference from uniformity).** For a P×M batch with perfect clustering (within-class similarity 1, cross-class → −∞ scaled), setting sᵢ = ((M−1)e^{1/τ_b})^{−1/2} for all i makes P exactly block-uniform doubly stochastic; then Σ_{j∈P(i)} P_ij = 1 so L_bal = 0 (its minimum) and a is constant so L_gap = 0. Hence both new terms are *inactive at the supervised optimum* and act only as a selector among imperfect solutions. Wang–Isola uniformity, by contrast, is maximally violated by perfect clustering — it fights the supervised objective at its optimum. This is the proof-level reason SBE is not "uniformity renamed."

**Proposition 3 (deployment certificate — exact).** If Var(a)=0 then K𝟙 = c𝟙; by symmetry the row-softmax matrix P^{row} = K/c then has column sums (Kᵀ𝟙)ⱼ/c = 1: the *deployed* raw kernel has exactly uniform soft retrieval mass at temperature τ_b, with no test-time operator. So L_gap is not a proxy — it is the deployed quantity.

**Degeneracy A — scaling absorption (the cheapest cheat).** Could the network satisfy L_bal while the *raw* geometry stays hubby, letting the Sinkhorn factors absorb the anisotropy (train/test operator mismatch)? Two independent blocks: (i) L_gap explicitly minimizes the spread of the factors, and by Prop. 3 its zero is exactly "raw kernel balanced"; (ii) even without L_gap, the −(aᵢ+aⱼ) terms inside L_bal charge hub points' own positive-pair likelihood, routing gradient into their embeddings, not just into the scalings (the scalings are not free parameters — they are a deterministic function of the embeddings).

**Degeneracy B — trivial balance.** The all-uniform kernel is doubly stochastic, but then Σ_{j∈P(i)} P_ij = (M−1)/(B−1) and L_bal = log((B−1)/(M−1)) ≈ log 44.5 ≫ 0: the likelihood term forbids buying balance with discriminability. Balance is a *constraint direction*, discrimination remains the objective.

**Degeneracy C — batch-locality.** Balance is enforced in expectation over random class-balanced batches, not on the global gallery. This is a real slack; it is why L_gap matters (a pointwise function-property, "constant local mass," which transfers to any mixture of batches) and why the diagnostics in §4 measure global test-gallery hubness directly rather than trusting the train constraint.

**Degeneracy D — soft vs. hard @1.** The constraint equalizes the τ_b-MGF of each point's similarity profile, not the hard rank-1 count. At τ_b = 0.1 the row sums are dominated by the top few neighbors, so equalizing them equalizes near-neighbor mass, which is the rank-1-relevant quantity; the residual soft-hard gap is monitored (§4) and is a stated failure channel, not hidden.

## 3. Adversarial novelty search (primary sources; one-sentence mechanism distinctions)

Web-verified this session:
- **noHub / noHub-S (Hubs and Hyperspheres, CVPR 2023):** optimizes *test-time* support/query embeddings transductively to reduce hubness in few-shot — inference-side and transductive, i.e., the banned category SBE internalizes into training.
- **Inductive few-shot hubness reduction via uniform loss (Neurocomputing 2025):** trains toward *global hyperspherical uniformity* to kill hubs — a mechanism that provably conflicts with class clustering (their own limitation), whereas SBE's balance constraint is exactly compatible with it (Lemma 1).
- **SinSim (arXiv 2502.10478):** adds a Sinkhorn/entropic-Wasserstein *view-alignment regularizer* to SimCLR — OT as a dispersion/alignment distance in label-free SSL, with no doubly-stochastic constraint on the retrieval coupling and no deployment-gap term. (Abstract-level only; reviewer should read its §method — flagged.)
- **Learning Deep Optimal Embeddings with Sinkhorn Divergences (arXiv 2209.06469):** uses Sinkhorn divergence as a *distance between image/class distributions* inside a ranking loss — OT-as-metric, not OT-marginals-as-calibration of the batch retrieval kernel.
- **DIML (ICCV 2021):** computes OT structural matching between spatial feature maps *at inference* as a second stage — banned deployment shape here, and a matching mechanism, not a balancing one.
- **Doubly Stochastic Neighbor Embedding on Spheres (arXiv 1609.01977) / low-rank DS word embedding (arXiv 1812.10401):** normalize similarity to doubly stochastic to fix crowding when *directly optimizing coordinates* for visualization/co-occurrence factorization — no parametric encoder, no supervision, no unseen-class deployment, no gap penalty forcing the raw kernel itself to be balanced.
- **Sinkhorn doubly-stochastic attention (Sinkformer line; arXiv 2604.07925):** balances *internal attention maps* for rank/stability — a layer normalization choice, not a retrieval likelihood or a hubness claim.
- **DS affinity learning for graph clustering (EJOR 2022):** post-hoc kernel normalization for clustering a *fixed* dataset — no encoder training, no generalization claim.
- **DADA (arXiv 2401.00617, AAAI 2024)** — confirmed as the Lane-A In-Shop reference method; mechanism (augmentation-based domain adaptation on proxies) orthogonal to SBE.

From memory (venues given; reviewer should verify quotes/details):
- **Zass & Shashua, NIPS 2006:** DS normalization of affinities improves spectral clustering — normalization *at use time* of a fixed kernel; SBE forbids use-time normalization and forces the encoder to produce the balanced kernel.
- **CSLS (Conneau et al., ICLR 2018), inverted softmax (Smith et al., ICLR 2017), mutual proximity (Schnitzer et al., JMLR 2012):** inference-time hub rescoring — the banned family SBE bakes in.
- **RCSLS (Joulin et al., EMNLP 2018):** trains a *linear orthogonal map between frozen word spaces* by relaxing CSLS — closest philosophical ancestor; SBE differs in operator (exact symmetric Sinkhorn coupling, not a k-NN penalty), in scope (end-to-end deep encoder under identity supervision), and in the deployment-gap penalty, which has no RCSLS analogue.
- **SwAV / SeLa / MSN (2020–22):** Sinkhorn balances *cluster-assignment* matrices to prevent collapse in label-free SSL — different matrix (instances×prototypes), different purpose (collapse avoidance), no claim about deployed kernel balance.
- **Wang & Isola uniformity (ICML 2020):** pairwise repulsive potential — conflicts with clustering at the optimum (Lemma 1 separates the fixed points).
- **Group Loss (ECCV 2020) / Spectral Feature Transformation (arXiv 2019):** train-time similarity-graph *propagation/refinement* operators (replicator dynamics; row-stochastic smoothing) to improve supervision — neither constrains column marginals nor targets hubness.
- **t-SNE/symmetric SNE:** normalizes by a single global constant — no per-point marginal constraint, hence no mass equalization.
- **Threshold-Consistent Margin (TCM, ~2024):** calibrates pairwise *threshold* statistics across classes — fixes bias in the accept threshold, not rank-1 mass theft; orthogonal error mode.
- **MagFace/AdaFace:** *want* informative per-sample scale; SBE drives the per-sample scaling degenerate (Var→0) — opposite mechanism.
- **PFML (CVPR 2025) / SoftTriple:** proxy-count/probabilistic-proxy axis, orthogonal to kernel balance; SBE adds zero parameters.

Honest gap: I am blind to the 12+ earlier frozen proposals in this protocol series; if any froze a retrieval-mass-balancing mechanism, that is a collision only the reviewer can detect. I found no published occupant of "supervised DML + symmetric-Sinkhorn-balanced likelihood + raw-kernel balance-gap penalty."

## 4. Decisive matched-compute controls

All controls: same backbone, sampler, optimizer, schedule, 5 seeds, same wall-clock ±3%.

- **C1** Proxy-Anchor base (the Δ baseline).
- **C2** PA + fixed train-mean centering before L2 (first-order hub fix; SBE must beat it, else "it's just centering").
- **C3** PA + Wang–Isola uniformity, weight tuned on internal val (the occupied alternative; Lemma 1 predicts it underperforms SBE and can hurt).
- **C4** SBE without L_gap (isolates the deployment-consistency term; prediction: partial gain).
- **C5** *Killer control:* replace the balanced coupling with row-only softmax at identical τ_b, λ_b (a supervised-contrastive term of matched compute). If C5 ≈ SBE (±0.2 everywhere), the column constraint is inert and the claimed mechanism is falsified regardless of headline gains.
- **C6** Pre-registered diagnostics: skew of the gallery N₁ (top-1 occurrence) distribution; hub-attributed error rate := errors whose wrongly retrieved item lies in the top 5% of N₁. Claims: baseline hub-attributed errors ≥15% on SOP, ≥8% on CUB; SBE reduces N₁ skewness ≥40% (relative) and hub-attributed errors ≥30%.
- **C7** *Stage gate before full runs:* apply CSLS/inverted-softmax to the C1 baseline *as a diagnostic only* (never as a deployed result). Its gain upper-bounds internalizable headroom; if < +0.3 R@1 on all four datasets, the premise is dead and I would withdraw the method (cheap kill switch).
- **C8** Base-swap robustness: Multi-Similarity base instead of PA (mechanism must not be PA-specific).

## 5. Frozen forecasts (Lane A), falsification thresholds, frontier arithmetic

Assumed frozen base (our PA reproduction, 5-seed mean): CUB 71.8, Cars 91.4, SOP 81.0, In-Shop 92.3 (if reproduction deviates >0.5, interpret Δs, not absolutes).

**SBE forecasts (R@1, 5-seed mean ± 95% CI):**

| Dataset | SBE forecast | Δ vs PA base | Lane-A reference | Crossing arithmetic |
|---|---|---|---|---|
| SOP (primary) | **83.2 ± 0.6** | +2.2 | PFML 82.9 ± 0.2 | decisive cross needs mean ≥ 83.3 (CI separation); central forecast borderline; P(mean ≥ 83.1) ≈ 0.5 |
| In-Shop (primary) | **93.5 ± 0.3** | +1.2 | PA+DADA 93.0 (no seeds reported) | CI lower bound 93.2 > 93.0 ⇒ decisive if hit; P(≥93.1) ≈ 0.65 |
| CUB (secondary) | 73.5 ± 0.5 | +1.7 | PFML 73.4 ± 0.3 | cross needs ≥ 73.7; P ≈ 0.35 |
| Cars (secondary) | 92.8 ± 0.3 | +1.4 | PFML 92.7 ± 0.3 | cross needs ≥ 93.0; P ≈ 0.4 |

Pre-registered *structure*: Δ(SOP) ≥ Δ(CUB) ≥ Δ(Cars) — hub severity ordering by gallery size and class sparsity (SOP: 11,316 unseen classes, ~5 images each; Cars: 49 dense classes). A gain with violated ordering counts as attribution failure even if means improve.

**Falsification (any one kills the method):** (a) C7 gate < +0.3 everywhere; (b) 5-seed Δ < +0.5 on both SOP and In-Shop; (c) C6 skew reduction < 20% relative; (d) C5 within 0.2 of SBE on all datasets; (e) ordering violated with otherwise positive results ⇒ mechanism claim retracted even if numbers stand.

**Honest headroom uncertainty:** no published measurement of hub-attributed error exists on these benchmarks (my search confirms the gap); the effect size is theory-motivated, gated by C7 before any expensive claim. Primary claimed crossings are SOP and In-Shop; CUB/Cars are forecast as ties-to-marginal-crosses, stated as such.

## 6. Costs and risks

**Cost.** Sinkhorn adds T·O(B²) log-sum-exps per step: ≈ 15×8100 entries — < 3% of a ResNet-50 batch-90 step; zero extra parameters, zero auxiliary networks (contrast: AdvRF's ResNet-34+U-Net, PFML's 15 proxies/class). Epoch time ≤ 1.03× PA; memory +O(B²) floats (negligible). Deployment bit-identical in cost and interface to any 512-D ResNet-50.

**Risks.** (1) Small headroom on strong modern embeddings — C7 gates this cheaply. (2) CUB/Cars/SOP use leave-one-out query=gallery evaluation while In-Shop has disjoint query/gallery; In-Shop is therefore the cleanest test of transfer, and it is one of my two primary claims. (3) SOP near-duplicates make some concentration legitimate; training-time balancing does not alter the test gallery, but could marginally under-serve duplicate-heavy classes. (4) Symmetric Sinkhorn oscillation — handled by η=0.5 damping, monitored marginal error. (5) Batch-locality slack (Degeneracy C) — mitigated by L_gap, measured by C6. (6) Contamination: no external data, no text, no generated data; ImageNet-1K init is lane-standard; known benchmark label noise (Cars, SOP duplicates) affects all lane methods equally. (7) Novelty residual risk: SinSim's full method text and any unpublished 2025–26 "balanced contrastive" variants; and possible collision with an earlier frozen proposal in this series, which I cannot see — both flagged for the adversarial reviewer.

**Failure modes stated plainly:** if hub-attributed error is a minority of a minority (e.g., 3% of errors), SBE lands ≈ +0.3 and crosses nothing — the C7 gate exists precisely to catch that before full commitment; if the soft-hard gap (Degeneracy D) dominates, diagnostics improve but R@1 does not, and criterion (c)+(b) kills it.

---

Sources: [Hubs and Hyperspheres (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_CVPR_2023_paper.pdf) · [Inductive few-shot hubness reduction (Neurocomputing)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015516) · [SinSim (arXiv 2502.10478)](https://arxiv.org/html/2502.10478) · [Deep Optimal Embeddings with Sinkhorn Divergences (arXiv 2209.06469)](https://arxiv.org/pdf/2209.06469) · [DIML: Interpretable DML with Structural Matching](https://www.alphaxiv.org/overview/2108.05889v1) · [DS Neighbor Embedding on Spheres (arXiv 1609.01977)](https://arxiv.org/abs/1609.01977) · [Low-rank DS word embedding (arXiv 1812.10401)](https://arxiv.org/abs/1812.10401) · [Sinkhorn DS attention rank decay (arXiv 2604.07925)](https://arxiv.org/abs/2604.07925) · [DS + idempotent affinity for clustering (EJOR)](https://www.sciencedirect.com/science/article/abs/pii/S0377221721010900) · [DADA (arXiv 2401.00617)](https://arxiv.org/html/2401.00617v1) · [Metric Learning Reality Check protocol](https://kevinmusgrave.github.io/powerful-benchmarker/papers/mlrc/)
