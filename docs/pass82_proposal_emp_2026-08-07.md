# Frozen Pass 82 proposal (Codex blind proposer; verbatim)

 I did not inspect the repository, its files, or candidate history.
 
 ## Proposal: Excitable-Medium Pooling (EMP)
 
 Target: class-disjoint CUB-200-2011, using a standard ImageNet-initialized ResNet-50 and an otherwise ordinary uniformly sampled metric loss.
 
 The core idea is to replace global average/GeM pooling with four steps of a fixed, differentiable FitzHugh–Nagumo-like reaction–diffusion process over the final spatial activation map. The descriptor represents which semantic activations can sustain a locally connected excitation—not merely their marginal magnitudes.
 
 ### Exact model
 
 Let the backbone output
 
 \[
 F(x)\in\mathbb{R}^{C\times H\times W}.
 \]
 
 A learned \(1\times1\) projection, with the same parameter count as a conventional 512-D head, produces
 
 \[
 s_{kq}=\tanh\!\left(P_k^\top F_q+b_k\right),
 \qquad k=1,\ldots,512,
 \]
 
 where \(q\) indexes spatial positions.
 
 Initialize an activator and refractory field:
 
 \[
 u^0_{kq}=s_{kq},\qquad v^0_{kq}=0.
 \]
 
 Run \(T=4\) explicit steps:
 
 \[
 u^{t+1}_{kq}=u^t_{kq}
 +h\left[
 D\Delta_Nu^t_{kq}
 +u^t_{kq}-\frac{(u^t_{kq})^3}{3}
 -v^t_{kq}
 +\eta s_{kq}
 \right],
 \]
 
 \[
 v^{t+1}_{kq}=v^t_{kq}
 +h\epsilon\left(u^t_{kq}-\gamma v^t_{kq}\right).
 \]
 
 Here \(\Delta_N\) is the five-point spatial Laplacian with reflecting boundaries. Fix, rather than tune,
 
 \[
 h=0.2,\quad D=0.12,\quad \eta=0.35,\quad
 \epsilon=0.08,\quad\gamma=0.5,\quad T=4.
 \]
 
 Pool the transient:
 
 \[
 a_k=\frac1{HW}\sum_q
 \left(u^T_{kq}+\frac{1}{2T}\sum_{t=1}^{T}u^t_{kq}\right),
 \qquad
 z(x)=\frac{a}{\lVert a\rVert_2}.
 \]
 
 Thus deployment returns exactly one 512-D normalized descriptor from one image view, compared by cosine similarity.
 
 Use a plain uniformly sampled soft triplet objective:
 
 \[
 \mathcal L=
 \frac1B\sum_i
 \log\!\left[
 1+\exp\left(
 \frac{
 z_i^\top z_{n(i)}
 -z_i^\top z_{p(i)}
 +m}{\tau}
 \right)
 \right],
 \]
 
 with \(m=0.1,\tau=0.05\). Select \(p(i)\) uniformly from the same class and \(n(i)\) uniformly from other classes before the forward pass; never consult descriptor distances.
 
 No auxiliary loss or special augmentation is required. All baseline crops, optimization, and sampling remain identical.
 
 ### Why spatial arrangement survives pooling
 
 The Laplacian itself has zero spatial mean, but it changes the local field before the cubic reaction. Consequently,
 
 \[
 \frac1{HW}\sum_q (u^t_{kq})^3
 \]
 
 depends on whether high activations form connected patches or scattered isolated sites. Two maps with identical mean, maximum, and activation histogram can therefore produce different EMP descriptors.
 
 The intended causal story is:
 
 \[
 \text{physical part layout}
 \rightarrow
 \text{connected activation support}
 \rightarrow
 \text{excitation transient}
 \rightarrow z.
 \]
 
 Fine-grained train-class texture correlations can disappear on unseen bird species. The fact that an eye, wing edge, wheel arch, or grille occupies a locally coherent region is more class-invariant. EMP makes connected evidence easier to transmit and fragmented texture responses more likely to decay into the refractory field. The metric loss must therefore choose projected channels whose evidence possesses spatially coherent support.
 
 This is an inductive-bias claim, not a guarantee. It produces a particularly strong counterfactual test: independently permuting the \(H\times W\) positions within every channel leaves GAP and all channelwise histograms unchanged, but should eliminate most of EMP’s advantage.
 
 ### Why it is outside the excluded families
 
 - No whitening, projection of nuisance subspaces, coding constraint, proxies, or class centers.
 - No masks or gates multiply features; every spatial site participates in the dynamics.
 - No adversary, teacher, EMA, distillation, extra image view, or generated augmentation.
 - No mining or cross-image context; pairs are sampled independently of embeddings.
 - No post-hoc transform: EMP is inside the end-to-end forward pass and remains at deployment.
 - No added regularization term. It is a pooling architecture with a specific nonlinear spatial mechanism.
 - It is unrelated to generative diffusion models, OT, hypergraphs, or distributional embeddings.
 
 ### Required motivating measurement and CPU preflight
 
 Before any training, take approximately 600 CUB training images, balanced across the 100 training classes:
 
 1. Run the frozen ImageNet backbone on CPU with deterministic center crops and cache its final \(7\times7\times C\) maps.
 2. Compute frozen GAP vectors.
 3. Apply the fixed EMP dynamics channelwise to all \(C\) cached channels—no learned 512-D projection is needed for this diagnostic.
 4. Compute same-class versus different-class cosine-separation AUC for GAP and EMP.
 5. Independently spatially permute every channel and recompute EMP.
 
 Proceed only if
 
 \[
 \operatorname{AUC}_{EMP}
 -\operatorname{AUC}_{EMP\text{-shuffle}}
 \ge 0.015
 \]
 
 and its class-bootstrap 95% interval excludes zero. Also require EMP’s unshuffled AUC to exceed GAP by at least \(0.01\).
 
 This measurement establishes that class information exists in spatial adjacency beyond marginal activation statistics. If it fails, the proposed causal substrate is absent and a GPU experiment is unjustified.
 
 ### Prediction and falsifier
 
 For a baseline whose five-seed CUB Recall@1 is \(68.0\%\), my preregistered prediction is:
 
 - Raw EMP Recall@1: \(69.5\%\), a \(+1.5\) percentage-point gain.
 - Selection-corrected expectation: approximately \(68.9\%\), a \(+0.9\)-point gain.
 - Inference overhead: below \(0.1\%\) of ResNet-50 FLOPs; the \(1\times1\) head has baseline-like parameter count.
 
 Primary falsifiers:
 
 - Mean five-seed gain below \(+0.5\) Recall@1 points.
 - The paired class-bootstrap interval includes zero.
 - Spatial permutation preserves more than 30% of EMP’s gain.
 - Replacing the cubic reaction with four ordinary linear diffusion steps performs equally well. That would mean smoothing, rather than excitability, explains the result.
 
 ### Raw and selection-corrected reporting
 
 Report every official-test seed, mean Recall@1/2/4/8, and paired per-query confidence intervals as the raw result. The official test set must never choose a configuration.
 
 Maintain a ledger \(H\) containing every examined configuration or abandoned variant. On train-class validation data, class-cluster bootstrap it \(B\) times. For replicate \(b\),
 
 \[
 h_b=\arg\max_{h\in H}R_h(D_b),
 \]
 
 \[
 o_b=R_{h_b}(D_b)-R_{h_b}(D_{\mathrm{val}}).
 \]
 
 Estimate selection optimism as
 
 \[
 \widehat o=\frac1B\sum_b o_b,
 \qquad
 \Delta_{\mathrm{corrected}}
 =\Delta_{\mathrm{raw}}-\widehat o.
 \]
 
 Resample entire classes and then seeds, not individual queries. This correction must include unsuccessful variants; otherwise it understates invention-selection bias.
 
 ### Likely prior art and collision boundary
 
 The physical mechanism comes from FitzHugh’s two-state excitability model and Nagumo, Arimoto, and Yoshizawa’s spatial active transmission line: [FitzHugh 1961](https://pubmed.ncbi.nlm.nih.gov/19431309/), [Nagumo et al. 1962](https://cir.nii.ac.jp/crid/1360574095742788736).
 
 The closest ML collision I found is trainable reaction–diffusion for image restoration, not retrieval or semantic-map pooling: [Chen, Yu, and Pock, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Chen_On_Learning_Optimized_2015_CVPR_paper.html). The relevant conventional retrieval comparator is trainable GeM pooling: [Radenović, Tolias, and Chum](https://pubmed.ncbi.nlm.nih.gov/29994246/).
 
 The potentially novel unit is specifically: a short excitable/refractory spatial transient in learned semantic feature maps, globally pooled into a class-disjoint metric descriptor, with spatial permutation as its causal falsifier. This is a collision hypothesis, not a claim of exhaustive novelty.
