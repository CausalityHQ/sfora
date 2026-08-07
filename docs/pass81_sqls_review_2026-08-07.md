# Pass 81 SQLS independent review (verbatim)

 # DEAD
 
 The proposal fails Gate 1 immediately and also fails Gate 2 independently. No files were edited.
 
 ## Gate 1 — killed
 
 The repository contains no audited coordinate-quorum measurement supporting the proposed causal premise. The proposal says to “Measure the coordinate-quorum curve” and is motivated only *if* future thresholds are met ([proposal, lines 113–130](/home/rb/worktrees/sfora-emafactorial/docs/pass81_proposal_sqls_2026-08-07.md:113)). Repository search found no existing `Λ(k)`, `k80`, random-coordinate leakage, or equivalent artifact outside the frozen proposal.
 
 The verified packet instead contains:
 
 - Four corrected In-Shop Proxy Anchor frozen-final seeds: **0.9153889436 ± 0.0013195712**.
 - Near-saturated training retrieval and confusion/fragmentation measurements.
 - No measurement that label evidence is concentrated in coordinates, no negative correlation with class-disjoint R@1, and no evidence that coordinate concentration causes transfer failure.
 
 See [current evidence audit](/home/rb/worktrees/sfora-emafactorial/docs/current_evidence_reliability_audit_321_2026-08-03.md:35). A proposed future probe cannot satisfy Gate 1’s requirement for an existing audited motivating number ([protocol, lines 109–112](/home/rb/worktrees/sfora-emafactorial/docs/search_protocol.md:109)).
 
 **Exact Gate-1 kill:** absent an independently recomputed corrected-recipe baseline satisfying either `Λ(64)>0.20` or `k80<128`, plus the proposed cross-seed/checkpoint negative association with class-disjoint R@1, the candidate is DEAD.
 
 ## Gate 2 — independently killed
 
 ### Direct prior-art collision
 
 Iwasawa, Nakayama, and Matsuo’s **MARS** trains adversarial classifiers on different random subsets of a representation, reverses their signal into the encoder, and preserves full-representation utility. That is the core SQLS mechanism, including the finite-adversary motivation and random-coordinate access restriction. [MARS, ICLR 2018 Workshop](https://openreview.net/pdf?id=ByuP8yZRb).
 
 SQLS changes the protected variable from a separate sensitive attribute to the same class label used by the full-vector utility objective, resamples masks, and adds a quota. Those are formulation details, not a materially new mechanism class. The occupied lineage also includes:
 
 - Adversarial censoring through an alternating minimax classifier: [Edwards & Storkey, ICLR 2016](https://arxiv.org/abs/1511.05897).
 - Gradient-reversal utility/invariance training: [Ganin et al., JMLR 2016](https://www.jmlr.org/papers/v17/15-239.html).
 - Information-theoretic adversarial leakage suppression: [Bertran et al., ICML 2019](https://proceedings.mlr.press/v97/bertran19a.html).
 - Mask-based adversarial regularization: [Park et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11634).
 
 The proposal is correct that ordinary and adversarial dropout optimize a different masked prediction direction. That distinction does not avoid the much closer MARS collision.
 
 **Novelty decision:** “retain full label information while denying it below a coordinate quorum” is a narrow same-target variant of adversarial random-subspace censoring/regularization, not materially novel.
 
 ### The “secret sharing” claim is unsupported
 
 True threshold secret sharing guarantees reconstruction from every authorized set and zero information from every unauthorized set. [Shamir 1979](https://doi.org/10.1145/359168.359176). SQLS provides neither:
 
 - It samples only two masks per batch at three sizes.
 - It controls expected finite-classifier loss, not every subset.
 - It never tests or constrains subsets between 129 and 511 coordinates.
 - It supplies no randomized sharing construction, reconstruction rule, access structure, or information-theoretic guarantee.
 
 Calling it “approximate secret sharing” therefore does not establish a distinct mechanism.
 
 ## Mathematical failures
 
 1. **The causal mediator is coordinate-gauge dependent.** For any orthogonal matrix \(O\), replacing \(z\) by \(Oz\) preserves every cosine, the NCA loss, and every retrieval ranking. Yet it can radically change all coordinate-subset probe scores. The head can therefore improve `k80` simply by rotating an unchanged class code into a dense basis. A quorum shift does not identify more weak visual cues or better transfer.
 
 2. **Distributed class coding is the easiest escape.** NCA can retain the same class-collapsed geometry while the head distributes that code across coordinates. Supervised contrastive objectives admit class-collapsed simplex solutions [Graf et al., ICML 2021](https://proceedings.mlr.press/v139/graf21a.html), while transfer work shows that spread alone is insufficient without a mechanism preserving meaningful within-class structure [Chen et al., ICML 2022](https://proceedings.mlr.press/v162/chen22d.html).
 
 3. **The “leakage” estimator is not mutual information.** For classifier \(A\),
    \[
    \mathrm{CE}(A)=H(Y\mid Q)+E_Q\,\mathrm{KL}(p(Y\mid Q)\|A(Y\mid Q)).
    \]
    Hence
    \[
    \log C-\mathrm{CE}
    =I(Y;Q)+\log C-H(Y)-E\mathrm{KL}.
    \]
    It has the claimed interpretation only with uniform labels and a Bayes-optimal adversary. Corrected In-Shop classes are unbalanced, so even a label-independent representation can have positive “leakage” because `log C > H(Y)`. The Information Bottleneck does not justify substituting this training CE statistic for mutual information. [Tishby et al.](https://arxiv.org/abs/physics/0004057).
 
 4. **Detached-gradient semantics are incomplete.** Detachment is correct for the adversary update. For the encoder update, \(A_\phi(q)\) must be recomputed with frozen \(\phi\) but non-detached \(z\). If the stated detached observations are reused, SQLS has exactly zero encoder gradient. The frozen text does not specify the required two-forward-pass boundary.
 
 5. **Mask metadata is not a direct encoder escape**, because masks are sampled after encoding and independently of labels. It does, however, demand that one unspecified two-layer network learn potentially astronomical mask-conditional decoders. Failure of that finite head demonstrates classifier weakness, not label secrecy—precisely the adversary-design problem identified by MARS.
 
 6. **No coordinate quorum is enforced.** The allowance is linear in \(k\), not a threshold:
    - \(k=32:\ 0.03125 I_{\rm full}+0.05\)
    - \(k=64:\ 0.0625 I_{\rm full}+0.05\)
    - \(k=128:\ 0.125 I_{\rm full}+0.05\)
 
 7. **NCA may be undefined for singleton labels in a batch.** The numerator is zero unless every anchor has another same-class sample. No class-balanced sampler or singleton handling is specified. The original NCA objective is a leave-one-out neighborhood objective [Goldberger et al., NeurIPS 2004](https://papers.nips.cc/paper_files/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html).
 
 ## Control, preflight, and frontier failures
 
 The proposed two-arm comparison is NCA versus NCA+SQLS. It does not isolate SQLS against the corrected Proxy Anchor recipe. A valid attribution requires at least:
 
 1. Corrected Proxy Anchor.
 2. Matched NCA without SQLS.
 3. Matched NCA with SQLS.
 
 Proxy Anchor and batch-all NCA have materially different supervision and optimization [Proxy Anchor, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html).
 
 The CPU preflight can establish only that a frozen feature head learns a coordinate rotation/code. It cannot test the claim that backbone learning shifts from shortcuts to transferable visual cues. Its inner-validation query/gallery construction is also unspecified, and “adversary accuracy near chance” is ill-defined under class imbalance.
 
 The deciding forecast is on **CUB**, directly violating the protocol’s requirement to screen first on corrected In-Shop ([protocol, lines 125–131](/home/rb/worktrees/sfora-emafactorial/docs/search_protocol.md:125)). It supplies no In-Shop prediction against **0.9153889 ± 0.0013196**:
 
 - Hypothetical +0.7 point: **0.9223889**
 - Hypothetical +1.2 points: **0.9273889**
 - Hypothetical +2.0 points: **0.9353889**
 
 Only the last would numerically exceed the published ResNet-50/512-D In-Shop 0.930 lane, but applying a CUB prediction to In-Shop is invalid, and BN-Inception/short-recipe versus ResNet-50/long-recipe is not a matched lane. On CUB, even adding the optimistic +3 points to the historical 0.6919 PA observation gives **0.7219**, still below the 0.734 PFML 512-D frontier—and that historical local base is not a fully rescored current-recipe control.
 
 Finally, the proposed bootstrap over already inspected official-test configurations is not an identified selection correction. It reuses official test classes for selection and scoring and does not correct adaptive search. The protocol requires nested validation, a frozen epoch, or independent selection.
 
 ## Exact disposition
 
 - **Gate 1:** DEAD—no existing audited motivating coordinate measurement.
 - **Gate 2 novelty:** DEAD—MARS is mechanism-equivalent random-subspace adversarial censoring.
 - **Gate 2 algebra:** DEAD—the quorum shift is non-identifying under an orthogonal rotation.
 - **Gate 3:** forecast is ungrounded and names the wrong deciding dataset/lane.
 - **Gate 4:** proposed CUB-first screen violates the corrected In-Shop rule.
 - **GPU authorization:** none.
 
 The protocol-mandated fresh Fable review and automatic Claude fallback were attempted, but both failed on account limits. That makes the pass procedurally non-authorizing as well; the local **DEAD** verdict does not depend on model agreement.
