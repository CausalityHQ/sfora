## Proposal: Extreme-Gallery Risk PFML

I choose **Lane A only**. The method adds a differentiable extreme-value retrieval-risk term to PFML. Its purpose is to correct a specific train–test mismatch: training losses see hundreds or thousands of negatives at once, whereas evaluation R@1 is determined by the single most similar impostor in a gallery that may contain tens of thousands of images.

The proposed method is **Extreme-Gallery Risk Potential-Field Metric Learning (EGR-PFML)**.

### 1. Executable method

#### Encoder and descriptor

Use ImageNet-1K-pretrained ResNet-50 at \(224\times224\). For image \(x\),

\[
h_\theta(x)=\operatorname{GAP}(\operatorname{R50}_\theta(x))\in\mathbb R^{2048},
\]

\[
z_\theta(x)=
\frac{W h_\theta(x)}
{\|W h_\theta(x)\|_2+\epsilon}\in\mathbb S^{511},
\qquad W\in\mathbb R^{512\times2048},\quad \epsilon=10^{-12}.
\]

The learned objects are \(\theta,W\), and PFML’s normalized proxies \(p_{c,j}\in\mathbb S^{511}\): 15 per class on CUB/Cars, two on SOP, matching the audited PFML configuration.

#### Base loss

Use PFML’s published potential-field objective unchanged. In executable notation, for distance \(d=\|r-v\|_2\),

\[
\psi_{\rm att}(d)=-
  \big(\max(d,\delta)\big)^{-\alpha},
\qquad
\psi_{\rm rep}(d)=
  \big(\min(d,\delta)\big)^{-\alpha}.
\]

For each current embedding or active proxy \(r\) of class \(c\),

\[
\Psi_c(r)=
\frac{1}{|A_c|}
\sum_{v\in A_c}\psi_{\rm att}(\|r-v\|_2)
+
\frac{1}{|R_c|}
\sum_{v\in R_c}\psi_{\rm rep}(\|r-v\|_2),
\]

where \(A_c\) contains other current-batch embeddings and active proxies of class \(c\), and \(R_c\) contains those of other active classes. Self-interactions are excluded. The base objective is

\[
L_{\rm PF}=\frac1{|E|}\sum_{r\in E}\Psi_{y(r)}(r).
\]

Use PFML’s audited \(\alpha,\delta\) values; if reconstructing independently, freeze \(\alpha=4,\delta=0.5\) for every dataset. PFML’s distinguishing mechanism is that interactions decay rather than strengthen with distance. [PFML primary paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)

#### FIFO training gallery

Maintain a non-learned FIFO queue

\[
Q=\{(\bar z_j,y_j,\operatorname{id}_j)\}_{j=1}^{8192}
\]

of detached descriptors, labels, and source-image IDs from previous iterations. Queue descriptors receive no gradient. Current query descriptors do receive gradients through their dot products with \(Q\). An entry is at most 64 optimizer steps old; older entries are overwritten. Same-source-image entries are always excluded, including when two augmentations of an image have appeared at different times.

For query \(z_i\), define positive and negative similarities

\[
P_i=\{z_i^\top\bar z_j:y_j=y_i,\ \operatorname{id}_j\neq\operatorname{id}_i\},
\]

\[
N_i=\{s_{ij}=z_i^\top\bar z_j:y_j\neq y_i\}.
\]

If the queue contains no valid positive, the query contributes only to \(L_{\rm PF}\). Otherwise set

\[
p_i=\max P_i .
\]

The ordinary subgradient of `max` is used; ties are averaged.

#### Differentiable peaks-over-threshold fit

Let \(n_i=|N_i|\) and take the \(k=64\) largest negative similarities. Let \(u_i\) be the 64th-largest value and

\[
e_{ij}=s_{ij}-u_i,\qquad s_{ij}\ge u_i.
\]

Fit a generalized Pareto distribution to these exceedances:

\[
G(e;\xi_i,\beta_i)
=
1-\left(1+\xi_i e/\beta_i\right)^{-1/\xi_i}.
\]

The \(\xi_i=0\) value is its continuous exponential limit.

The fit is the constrained maximum-likelihood solution

\[
(\hat\xi_i,\hat\beta_i)
=
\arg\min_{\substack{-0.45\le\xi\le0.10\\
\beta\ge10^{-4}\\
1+\xi e_{ij}/\beta>0}}
\left[
k\log\beta+
\left(1+\frac1\xi\right)
\sum_{j=1}^{k}\log\left(1+\frac{\xi e_{ij}}{\beta}\right)
\right].
\]

Implement it with eight damped Newton iterations initialized at \(\xi=-0.1\), \(\beta=\operatorname{mean}(e)+10^{-4}\). Backpropagate through all eight iterations, including through \(u_i\) and the exceedances. There are no freely learned tail parameters.

The generalized-Pareto choice follows the Pickands–Balkema–de Haan peaks-over-threshold result: sufficiently high exceedances of a broad class of distributions converge to a GPD. [Primary EVT exposition](https://arxiv.org/abs/cond-mat/0011168)

#### Gallery extrapolation

Let

\[
M=|\mathcal D_{\rm train}|-
\frac{|\mathcal D_{\rm train}|}{C_{\rm train}},
\]

using only the official training split. Thus \(M\) approximates the number of negative images in a full training-sized gallery; no test-gallery size is consulted.

Freeze maximum-tail confidence \(1-\rho=0.90\), so the required one-negative survival probability is

\[
r_M=1-(1-\rho)^{1/M}.
\]

Since the empirical probability of exceeding \(u_i\) is \(k/n_i\), define the extrapolated 90th percentile of the maximum of \(M\) negatives as

\[
q_i =
u_i+
\frac{\hat\beta_i}{\hat\xi_i}
\left[
\left(\frac{k}{n_i r_M}\right)^{\hat\xi_i}-1
\right],
\]

with continuous limit

\[
q_i=u_i+\hat\beta_i
\log\frac{k}{n_i r_M}
\quad\text{when }|\hat\xi_i|<10^{-4}.
\]

Clip \(q_i\) to \(1-10^{-4}\). The extreme-gallery loss is

\[
L_{\rm EGR}
=
\frac1{|\mathcal I|}
\sum_{i\in\mathcal I}
\tau\log\left[
1+\exp\left(\frac{q_i-p_i+m}{\tau}\right)
\right],
\]

where \(\mathcal I\) is the set of queries with valid positives, \(m=0.02\), and \(\tau=0.02\).

The final loss is exactly

\[
L=L_{\rm PF}+\lambda(t)L_{\rm EGR},
\]

with

\[
\lambda(t)=
\begin{cases}
0,&t<20,\\
0.1(t-20)/20,&20\le t<40,\\
0.1,&t\ge40.
\end{cases}
\]

The queue is filled during epochs 1–19 but EGR gradients begin only at epoch 20.

#### Optimization

Use class-balanced batches of 32 identities and four images per identity. Apply one ordinary stochastic view per selected image: random resized crop to 224 with scale \([0.16,1]\), horizontal flip \(0.5\), color jitter \(0.4/0.4/0.4/0.1\) with probability \(0.8\), grayscale \(0.2\), and ImageNet normalization.

Train 200 epochs with AdamW:

- backbone learning rate \(3\times10^{-5}\);
- \(W\) and proxies \(3\times10^{-4}\);
- weight decay \(10^{-4}\) on \(\theta,W\), zero on proxies;
- five-epoch linear warm-up;
- cosine decay to \(10^{-6}\);
- gradient clipping at norm 10;
- mixed precision allowed.

Normalize every proxy after each optimizer step. All hyperparameters above are frozen across datasets.

At test time discard proxies, the queue, and all EVT machinery. One unaugmented resized/center-cropped image produces one 512-D normalized descriptor. Retrieval is ordinary cosine nearest neighbour.

### 2. Causal error mode and degeneracy argument

The targeted error is a **rare impostor collision**. A model can have excellent average negative separation yet fail R@1 because one unseen-identity image shares a nuisance configuration—pose, background, color, product presentation—with the query. With \(M\) approximately independent negatives,

\[
\Pr(\max_j S_j^-<s)=F_-(s)^M.
\]

Consequently, a tail event with probability only \(10^{-4}\) per negative becomes common in a gallery containing \(10^4\) negatives. PFML deliberately weakens distant interactions, but neither its mean potential nor a minibatch-hard negative estimates this gallery-multiplicity effect.

Conditional on a correct GPD tail and fixed positive similarity \(p_i\), the construction of \(q_i\) gives

\[
\Pr\!\left(\max_{1\le j\le M}S^-_{ij}\le q_i\right)=0.90.
\]

Therefore,

\[
p_i\ge q_i+m
\quad\Longrightarrow\quad
\Pr\!\left(p_i>\max_jS^-_{ij}\right)\ge0.90.
\]

This is a conditional training certificate, not a distribution-free claim about unseen identities.

The cheapest shortcuts are blocked as follows:

- **Constant collapse:** if all descriptors coincide, \(p_i=q_i=1\), giving \(L_{\rm EGR}\ge\tau\log(1+\exp(m/\tau))>0\); PFML’s repulsive field also has nonzero descent directions.
- **Norm inflation:** impossible because every descriptor and proxy is normalized before any similarity or loss.
- **Proxy lookup codes:** EGR contains no proxies. It compares each image directly with queued images from other identities.
- **Self-augmentation leakage:** source-image IDs remove every copy of the query image from its positive set.
- **A freely manipulated tail head:** \(\xi_i,\beta_i\) are not learned parameters; they are deterministically refitted from current similarities under compact constraints.
- **Suppressing only one hard negative:** after one top exceedance moves down, another enters the top-64 set. Apart from order-statistic ties, zero EGR loss requires suppression of the whole moving upper tail or an increase in genuine cross-image positive similarity.
- **Stale-gallery memorization:** FIFO turnover changes the negative population continually, while the age cap prevents a fixed bank from becoming a permanent set of training targets.

The substantive vulnerability is the exchangeability approximation: negative similarities are correlated by class and visual subtype, so the stated 90% certificate need not be calibrated on unseen identities.

### 3. Adversarial novelty search

I searched both DML and EVT/open-set literature for combinations of GPD tails, nearest-neighbour maxima, hard-negative training, rank-loss optimization, and metric-learning queues. The nearest primary works are:

- **PFML** superposes decaying sample/proxy potentials; EGR-PFML instead fits and differentiates through the upper impostor-similarity tail and extrapolates it to a full gallery. [PFML](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- **PADS** learns which hardness regions to sample through a reinforcement-learning teacher; it neither models a tail law nor turns gallery multiplicity into an extreme quantile. [PADS](https://openaccess.thecvf.com/content_CVPR_2020/html/Roth_PADS_Policy-Adapted_Sampling_for_Visual_Similarity_Learning_CVPR_2020_paper.html)
- **Smooth-AP** differentiably approximates within-batch AP; it optimizes observed ranks rather than an extrapolated maximum of negatives not present in the batch. [Smooth-AP](https://arxiv.org/abs/2007.12163)
- **SupRank/ROADMAP** constructs robust decomposable surrogates for AP and R@\(k\); it still acts on observed rankings and does not estimate an EVT tail for a larger latent gallery. [Optimization of Rank Losses](https://arxiv.org/abs/2309.08250)
- **Proxy Anchor** aggregates sample–proxy hardness but has no sample-gallery extreme model. [Proxy Anchor](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)
- **OpenMax** fits extreme tails after classifier training to reject unknown inputs; it changes open-set inference and does not train a deployable nearest-neighbour descriptor against future impostor maxima. [OpenMax](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)
- **Extreme Value Machine** uses EVT to form variable-bandwidth open-set classification regions; it is an inference classifier rather than an end-to-end global-descriptor retrieval loss. [EVM](https://pubmed.ncbi.nlm.nih.gov/28541894/)
- **DRO pairwise DML** adversarially reweights observed mini-batch pairs; it does not estimate the order statistic induced by a larger gallery. [Pairwise DRO framework](https://openreview.net/forum?id=SJl3CANKvB)

I found adjacent ingredients, but no primary work using a differentiable peaks-over-threshold fit to extrapolate unseen negative-gallery maxima during standard zero-shot DML training. That is the claimed mechanism distinction, not merely the use of “hard negatives” or generic CVaR.

### 4. Decisive matched-compute controls

All comparisons must use the same ResNet-50, 512-D head, augmentations, optimizer, batch composition, 200 epochs, queue dot products, and five seeds.

1. **PFML reproduction:** establishes the audited base.
2. **Queue-hard PFML:** replace \(q_i\) by \(\max N_i\). This tests whether the gain is merely a larger hard-negative pool.
3. **Top-64 CVaR PFML:** replace \(q_i\) by the mean of the top 64 negatives. This tests generic tail emphasis.
4. **No-extrapolation control:** retain the GPD fit but set \(M=n_i\). This isolates full-gallery extrapolation from flexible tail fitting.
5. **Log-sum-exp control:** tune its temperature so its mean gradient norm matches EGR. This tests ordinary smooth hard mining.
6. **Shuffled-tail control:** randomly assign each query another query’s fitted \((\hat\xi,\hat\beta,u)\), preserving compute and marginal penalties while destroying query-conditional tail calibration.

The claimed mechanism is supported only if EGR beats controls 2–5 and the shuffled-tail control loses at least half of EGR’s improvement over PFML. If queue-hard or CVaR matches it within 0.001 R@1, the EVT interpretation is falsified even if the full method improves.

### 5. Frozen Lane-A forecasts

These are prospective five-seed means and between-seed standard deviations, frozen before running experiments.

| Dataset | Matched PFML reference | EGR-PFML forecast | Frontier arithmetic |
|---|---:|---:|---:|
| CUB-200-2011 | \(0.734\pm0.003\) | **\(0.741\pm0.004\)** | \(0.741-0.734=+0.007\) |
| Cars196 | \(0.927\pm0.003\) | **\(0.931\pm0.003\)** | \(0.931-0.927=+0.004\) |
| SOP | \(0.829\pm0.002\) | **\(0.834\pm0.002\)** | \(0.834-0.829=+0.005\) |

I do not forecast In-Shop because the supplied strongest matched reference, PA+DADA \(0.930\), differs in base mechanism and lacks reported seed uncertainty; inventing a precise PFML-matched anchor would be misleading.

Primary falsification thresholds:

- CUB mean \(<0.737\);
- Cars mean \(<0.929\);
- SOP mean \(<0.831\);
- or failure to beat queue-hard PFML by at least 0.002 on two of the three datasets.

A strict frontier claim requires the five-seed 95% confidence interval for the paired EGR-minus-PFML difference to lie above zero. Merely obtaining one best seed above PFML is not frontier crossing. This caution is important because standardized audits have shown that many historical DML gains shrink under matched evaluation. [Metric Learning Reality Check](https://arxiv.org/abs/2003.08505)

### 6. Cost and risks

Training retains one ResNet-50. The queue occupies approximately \(8192\times512\times2\approx8\) MB in FP16, plus negligible metadata. A \(128\times8192\) similarity matrix adds roughly \(0.54\) billion multiply-adds per step. Batched top-\(k\) and eight two-parameter Newton fits are small relative to the backbone. I forecast \(1.10\)–\(1.18\times\) PFML epoch time and \(1.03\)–\(1.07\times\) peak memory, depending on similarity-kernel implementation.

Deployment cost is exactly Lane A: one ResNet-50, one view, one 512-D normalized descriptor, and ordinary cosine search. No queue, proxy, EVT fit, auxiliary network, or reranking survives training.

The major scientific risks are:

- GPD approximation may be poor for bounded, multimodal cosine tails.
- Queue staleness can bias fitted tail shape early in training.
- SOP’s very small classes may make \(p_i\) noisy even when negative-tail estimation is good.
- The method may over-separate visually legitimate cross-identity attributes and damage transferable structure.
- A top-64 estimate may have excessive variance on CUB; a larger \(k\) reduces variance but biases the asymptotic tail fit.
- Gallery negatives are correlated, making the independence-based maximum arithmetic optimistic.
- Standard benchmarks have small margins and substantial implementation sensitivity; all preprocessing, split files, and model-selection rules must be fixed before test evaluation.
- Hyperparameters must be selected only through identity-disjoint subdivisions of the official training identities, followed by one full-training rerun. Test embeddings, gallery statistics, dataset names, and test R@1 must never influence \(k,\rho,\lambda,m\), epoch selection, or queue size.
