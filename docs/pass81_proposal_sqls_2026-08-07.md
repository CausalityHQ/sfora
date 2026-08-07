# Frozen Pass 81 proposal (Codex blind proposer; verbatim)

 ## Proposal: Subquorum Label Secrecy
 
 Train the 512-D descriptor so that the complete vector separates training classes, while no small coordinate subset can reliably identify them. This approximates threshold secret sharing: useful class evidence must accumulate across many weak visual cues instead of residing in a few train-class-specific shortcut coordinates.
 
 ### Objective
 
 For one standard crop per image, produce
 
 \[
 z_i=\frac{f_\theta(x_i)}{\|f_\theta(x_i)\|_2}\in\mathbb S^{511}.
 \]
 
 Use a single-view, batch-all NCA loss:
 
 \[
 \mathcal L_{\mathrm{NCA}}
 =-\frac1B\sum_i
 \log
 \frac{\sum_{p\ne i:y_p=y_i}\exp(z_i^\top z_p/\tau)}
 {\sum_{a\ne i}\exp(z_i^\top z_a/\tau)}.
 \]
 
 No proxies or hard-example selection are required.
 
 For each batch, sample \(R=2\) masks \(m_r\in\{0,1\}^{512}\), uniformly without replacement, with
 
 \[
 k_r=\|m_r\|_0\sim\operatorname{Uniform}\{32,64,128\}.
 \]
 
 The masked observation is
 
 \[
 q_{ir}=\left[\sqrt{\frac{512}{k_r}}\,(m_r\odot z_i),\,m_r\right].
 \]
 
 A disposable two-layer classifier \(A_\phi\) tries to recover the training label from \(q_{ir}\). Giving it the mask prevents missing-coordinate ambiguity.
 
 Train the adversary on detached descriptors:
 
 \[
 \phi \leftarrow \arg\min_\phi
 \left[
 \operatorname{CE}(A_\phi([z_i,\mathbf1]),y_i)
 +\frac1R\sum_r \operatorname{CE}(A_\phi(q_{ir}),y_i)
 \right].
 \]
 
 Define empirical label leakage
 
 \[
 \widehat I_r=\left[\log C-\operatorname{CE}(A_\phi(q_{ir}),y_i)\right]_+,
 \]
 
 and full-vector leakage
 
 \[
 \widehat I_{\mathrm{full}}
 =\operatorname{stopgrad}
 \left[\log C-\operatorname{CE}(A_\phi([z_i,\mathbf1]),y_i)\right]_+.
 \]
 
 Permit only a sublinear fraction of the full leakage:
 
 \[
 \mathcal L_{\mathrm{SQLS}}
 =
 \frac1R\sum_r
 \left[
 \widehat I_r
 -\alpha\frac{k_r}{512}\widehat I_{\mathrm{full}}
 -\delta
 \right]_+,
 \]
 
 with initial fixed values \(\alpha=0.5,\delta=0.05\) nat and
 
 \[
 \min_\theta\quad
 \mathcal L_{\mathrm{NCA}}+\lambda\mathcal L_{\mathrm{SQLS}},
 \qquad \lambda=0.1.
 \]
 
 Alternate one adversary update and one encoder update. Warm \(\lambda\) linearly from zero during epochs 5–15. The adversary is a training-only head, not another encoder. Inference remains exactly one raw, normalized 512-D descriptor searched by cosine/FAISS.
 
 ### Why it could transfer causally
 
 A metric loss can reduce training loss using a small collection of coordinates carrying large class-specific mean shifts—background, pose, paint pattern, or photographer bias. The proposed intervention closes that path: whenever a small subset becomes sufficiently label-predictive, its gradient is reversed until its advantage falls below the quota. Meanwhile, the full-vector metric objective still demands class discrimination.
 
 The easiest remaining solution is to accumulate many individually weak cues—part shape, texture, proportions, and their conjunctions. Novel bird or car classes can reuse those cues in combinations not present among training labels. This is a causal hypothesis, not a theorem: the leakage intervention must precede both a shift in the quorum curve and improved unseen-class retrieval.
 
 ### Why this is not an excluded method
 
 - No inference transform, whitening, projection, coding, proxy recoding, or contextual scoring.
 - No second view, image augmentation loss, teacher, EMA, or ensemble.
 - Coordinate masks exist only inside the training adversary.
 - It is not dropout: dropout minimizes label loss under masks, making small subsets more predictive. This method does the opposite—explicitly limiting their training-label information.
 - It is not ordinary information bottleneck: the full descriptor remains label-informative; only unauthorized coordinate subsets are bottlenecked.
 
 ### Closest prior art and collision risk
 
 The most likely collision is the combination of:
 
 - [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html), which discourages reliance on individual units but has the opposite masked-label objective.
 - [Domain-adversarial training](https://www.jmlr.org/papers/v17/15-239.html), which supplies the gradient-reversal machinery but removes nuisance/domain information from the complete representation.
 - The [information bottleneck](https://arxiv.org/abs/physics/0004057), which supplies the leakage interpretation.
 - Work showing that supervised contrastive objectives can produce class-collapsed geometries, including [Graf et al.](https://proceedings.mlr.press/v139/graf21a) and transfer-focused analysis by [Chen et al.](https://proceedings.mlr.press/v162/chen22d.html).
 
 I would claim novelty only for the particular “retain full target information while adversarially denying it below a coordinate quorum” construction in class-disjoint DML. A proper literature review could still uncover a collision.
 
 ### Motivating measurement
 
 Measure the coordinate-quorum curve on a trained baseline:
 
 \[
 \Lambda(k)=
 \frac{\log C-\operatorname{CE}^{CV}_k}
 {\log C-\operatorname{CE}^{CV}_{512}},
 \]
 
 where \(\operatorname{CE}^{CV}_k\) is a fresh cross-validated probe’s loss averaged over 100 random \(k\)-coordinate subsets.
 
 The proposal is motivated if, on training classes:
 
 - \(\Lambda(64)>0.20\), or
 - fewer than 128 coordinates recover 80% of the full probe information,
 
 and this concentration correlates negatively with class-disjoint validation Recall@1 across seeds/checkpoints.
 
 ### CPU-only, protocol-clean preflight
 
 Use only official training classes:
 
 1. Split them class-disjointly into 80% inner-train and 20% inner-validation; never load official test images.
 2. Run the ImageNet-initialized backbone once on deterministic single crops using CPU and cache its activations.
 3. Freeze the backbone and train only a 512-D head plus adversary for three fixed seeds.
 4. Compare \(\lambda=0\) against the single preregistered configuration above.
 5. Reject before GPU work unless:
    - the minimum \(k\) reaching \(80\%\) full leakage at least doubles;
    - full-vector inner-validation Recall@1 falls by no more than 1 point;
    - adversary accuracy repeatedly returns toward chance rather than oscillating or diverging.
 
 This preflight does not establish retrieval improvement; it cheaply tests whether the intended control mechanism is achievable.
 
 ### Prediction and falsifier
 
 On CUB-200-2011 with matched backbone, schedule, batches, and compute:
 
 - Predicted raw effect: \(+2.0\) Recall@1 points, plausible range \(+1.0\) to \(+3.0\).
 - Predicted selection-corrected effect: approximately \(+1.2\) points.
 - Expected compute overhead: below 5%.
 
 Falsify the mechanism if any of these occurs across three preregistered seeds:
 
 - selection-corrected improvement is below \(+0.7\) Recall@1 points;
 - \(k_{80}\) fails to increase by at least \(2\times\);
 - full-vector validation Recall@1 drops by more than 1 point;
 - retrieval improves without a preceding quorum-curve shift.
 
 Report the locked three-seed result raw. If several configurations are inspected on the official test set, additionally report class-cluster bootstrap bias correction: select the configuration on each bootstrap’s in-bag classes, score it on out-of-bag classes, and average the paired method-minus-baseline effect. If only the preregistered configuration reaches the test set, selection count is one and the corrected estimate equals the raw locked estimate.
