# Pass 77 independent review (verbatim)

 I have what I need. I did not edit the proposal or run any GPU work.
 
 ---
 
 # Verdict: **DEAD**
 
 Dead independently at Gate 1, at Gate 2, and on frontier arithmetic. Any one is dispositive under `docs/search_protocol.md`.
 
 ## Single strongest reason
 
 **NAM's load-bearing premise was preregistered and measured in this repository on the audited corrected In-Shop packs, and it failed — with the sign going the wrong way.**
 
 NAM rests entirely on §1: *"within-class nuisance directions … are largely class-agnostic and therefore transfer to unseen identities … the pooled second moment is the part of the training-class distribution that is legitimately extrapolable to test identities."*
 
 `docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md` tested exactly that, correctly: pooled within-class covariance estimated on one In-Shop identity fold, leading-`k` subspace evaluated on a **disjoint** identity fold, three digest-bound seeds, both fold directions, with Haar-random subspace and label-permuted nulls, thresholds locked before the numbers were read. At `k=32`, fold-mean `rho_32` = **0.9312 / 0.9287 / 0.9345**, all below the preregistered `1.15` falsifier. The source-fold within-class subspace captured 35–37% of target **within**-class variance but 38–40% of target **between**-class variance.
 
 That is not merely "no support." It inverts NAM's §4 mechanism prediction. Suppressing energy along those directions removes ~39% of between-identity signal and ~36.5% of within-identity signal on unseen identities, so `tr(S_w)/tr(S_b)` on test identities moves to `(1−0.365)/(1−0.39) ≈ 1.04×` its baseline — **up ~4%**, against NAM's registered prediction of a **≥10% drop**. NAM's gradient term `−αδβ·Σ̄_λẑ/σ` pushes embeddings off precisely these directions, so on disjoint identities it strips identity signal at least as fast as nuisance.
 
 **And NAM's own §3 pre-flight cannot detect this.** It tests (i) eigen-spectrum Spearman ≥ 0.5 and (ii) top-50 subspace overlap *above chance*. Chance overlap for a random 50-d subspace of ℝ⁵¹² is 50/512 ≈ 9.8%; candidate 225 measured 35–37% at k=32 (chance 6.25%) and confirmed random subspaces give `rho ≈ 0.998–1.002`. So the pre-flight would have **passed** on the very data that killed the mechanism. It tests *transferability of the subspace*; the mechanism needs *nuisance-selectivity* of the subspace (`w` vs `b`), which it never measures. The proposal's self-described "single most informative experiment" is uninformative about its own load-bearing assumption.
 
 *Honest limit:* candidate 225 uses residuals about the class mean, NAM about the proxy (near-identical at a PA operating point), and it is In-Shop epoch-10, not CUB. Per the `RSPG_SPECDEFECT.md` lesson I apply this symmetrically — it is a near-match, not an exact one. It does not need to be exact: Gate 1 requires *positive* provenance, which NAM has none of. Candidate 225 only converts "unsupported" into "contradicted on the dataset the protocol makes decisive" (Gate 4: screen on corrected In-Shop, not CUB).
 
 ## 1. Objective: the claimed optimum is not the executed one
 
 The support-function algebra is correct: for `Q_p = {p̂+u : uᵀΣ̄_λ⁺u ≤ ε²}`, `max ẑᵀu = ε√(ẑᵀΣ̄_λẑ)`. Three defects follow anyway.
 
 - **"Both fold *exactly* into a per-sample margin" is false for β ∈ (0,1).** The worst-case margin is `εσ_i`. The executed margin is `δ_i = δ(1−β) + δβ·σ_i/mean(σ)`. These coincide only at `β=1`. At every recommended interior β the robust/minimax interpretation — the entire justification for the √ form — does not hold for the objective actually optimized. What executes is an affine rescaling of PA's margin by a normalized per-sample scalar.
 - **`δ_i` is constant across proxies, so it cannot address the §1 failure mode.** `σ(ẑ)` depends on the query only. For a fixed sample `i`, the margin against *every* negative proxy is identical, so the NAM term exerts **zero re-ranking pressure among negative classes**. The stated error mode — "the closest plausible instance of a *negative class* is closer than that class's proxy" — is per-negative-class and pooled Σ̄ gives every class the same ellipsoid. The mechanism is structurally incapable of the correction it is motivated by.
 - **The objective self-annihilates toward its own control.** Σ̄ is an EMA of residuals; the gradient actively suppresses residual energy in Σ̄'s top directions, flattening Σ̄. Because σ̃ is mean-normalized, only spectral *spread* drives the term — and the term's own action is to remove spread. At isotropy σ̃ ≡ 1 and NAM ≡ PA exactly. This is negative feedback toward the β=0 arm, and at m ∈ {0.9, 0.99} over ~2940 steps the EMA tracks fast enough for it to bite. This is an argument, not a measurement, but it predicts a shrinking effect and makes the §3 `std(σ̃) < 0.1` kill self-fulfilling late in training.
 
 ## 2. Gate-1 provenance: absent
 
 NAM cites **zero repository measurements**. §1 is analytic (PA's gradient lies in `span({p}) ∪ {ẑ}`) — true, but it is armchair algebra, which Gate 1 exists to reject. Every number in §4 is a forecast. Passes 22 (CINA), 24 (EFML), 27 (FRAME), 32 (HIRE), and 40 (CNW) were each recorded dead for this same missing measurement, all citing candidate 225. NAM is the sixth.
 
 ## 3. Prior art: two uncited, benchmark-matched occupations
 
 The proposal's own citations (ISDA — Wang et al., NeurIPS 2019; Roth/Vinyals/Akata, CVPR 2022; Shivaswamy/Bhattacharyya/Smola, JMLR 2006; SoftTriple, ICCV 2019; Proxy Synthesis, AAAI 2021; MagFace CVPR 2021 / CurricularFace CVPR 2020 / AdaCos CVPR 2019; WCCN, Interspeech 2006) are accurate and the concessions are fair. Two closer items are missing.
 
 - **Chen et al., *Intra-class Adaptive Augmentation with Neighbor Correction for Deep Metric Learning* (arXiv:2211.16264).** Repo-verified as prior art in verdict entry **102**. It estimates class-wise embedding covariances, **corrects sparse per-class estimates by borrowing across classes**, and samples adaptive virtual embeddings for DML losses — on CUB, Cars196, SOP **and In-Shop**, at ~2% overhead. NAM's differentiator ("pooled Σ̄ is required because In-Shop has ~5 images/class, where per-class covariance is hopeless") *is* IAA's stated motivation. NAM is the sup-over-ellipsoid closed form of the same estimate-covariance-then-synthesize operator that ISDA is the expectation closed form of. Entry 102's recorded ruling: "Changing the covariance estimator, conditioning it…, or using a stricter uncertainty threshold would be a variant of the same … operator."
 - **The repo's recorded reduction for margin reweighting.** Entry 27: a proposal that "only reweights or enlarges the margin of an existing [relation]" adds no supervision relation. Entries 105–108 and `docs/imsdo_2025_margin_variance_prior_art_2026-08-04.md` (Winston & Kang, *IMSDO*, Neurocomputing 655, 2025 — benchmark-matched on all four datasets) record margin curricula, dynamic margins, and feature-deviation targeting as occupied. NAM's executed object is a per-sample scalar adaptive margin with a warmup schedule.
 
 From my own knowledge (not repo- or web-verified here, flagging accordingly): **Yin et al., *Feature Transfer Learning for Face Recognition with Under-Represented Data*, CVPR 2019** is the closest primary source for NAM's transfer premise — it explicitly transfers intra-class variance across identities on the assumption it is class-agnostic. **Shi & Jain, *Probabilistic Face Embeddings*, ICCV 2019** and **Chang et al., *Data Uncertainty Learning in Face Recognition*, CVPR 2020** occupy per-sample variance modulating a proxy/softmax objective. **Lanckriet et al., *A Robust Minimax Approach to Classification*, JMLR 2002** likely predates Shivaswamy 2006 for the `√(wᵀΣw)` margin.
 
 ## 4. Controls: do not isolate the mechanism
 
 Controls 1–3 are well-chosen for what they target, and control 2 (spectrum-matched random rotation) is genuinely good. The gap: **no arm matches σ̃ to a simple difficulty scalar.** `σ(ẑ)` is largest for samples aligned with dominant residual directions — i.e. off-center, hard samples. The obvious confound is `ρ(σ̃, ẑ·p̂)` or `ρ(σ̃, ‖r_i‖)`, and a difficulty-keyed margin is CurricularFace / AdaptiveFace / IMSDO. Control 2 fails under *both* "the eigenbasis carries real structure" and "the difficulty correlation carries it," so it cannot separate them. The degeneracy diagnostic only checks `ρ(σ̃, ‖z‖)` — the MagFace axis — and `z` is L2-normalized in the loss, so that check is the narrowest of the family. The missing arm is a rank-matched σ̃ constructed from own-proxy similarity.
 
 ## 5. Frontier: does not cross, and is under-powered against its own reference
 
 Per `docs/current_method_search_frontiers_2026-08-05.md`, "a forecast and falsifier must name exactly one row/lane and a same-lane base." **NAM names no lane.** Its forecasts are deltas against its own internal β=0 arm.
 
 | | Forecast applied to repo base | Same-lane horizon | Crosses? |
 |---|---|---|---|
 | CUB | 0.6919 + 0.007 = 0.699 | Lane A 0.734 (PFML, CVPR 2025) | No |
 | Cars196 | no repo base under corrected recipes | Lane A 0.927 | No |
 | In-Shop | 0.9137 + 0.0015 = 0.9152 | Lane A 0.930 (PA+DADA); Lane B 0.939 (VAPNet) | No |
 
 Worse, the In-Shop arm cannot resolve its own forecast. The two corrected-pixel In-Shop controls are **0.9137** and **0.9168** — a 0.31-point spread across two seeds — against a **+0.15** prediction at 3 seeds. Gate 4 states a single seed "cannot establish a small gain." The proposal's own §5 selection arithmetic is correct (σ√(2 ln 12) = 0.5 × 2.229 = 1.11) and self-refuting: **its stated selection inflation (1.1) exceeds its largest predicted effect (+0.7)**. Also note the repo's controlled lane is **BN-Inception/512-D**, not the ResNet-50 lane the recipe implies.
 
 ## 6. Hidden violations
 
 - **Test-split selection.** §3 computes `Σ_w` from **test-class** residuals and makes a preregistered abandon/proceed decision on it. That is selecting on test — `HANDOFF_BRIEF.md` discipline #4 and protocol Gate 0. It also contradicts §5's own (correct) class-disjoint-validation rule. The clean construction is candidate 225's: fold-split *within training identities*. §4's mechanism prediction is likewise evaluated on test identities.
 - **Tuning.** "α, δ … unchanged" and "equal HPO trial count" are in tension: β, λ, m, and the warmup fraction have no baseline counterpart, so the baseline either re-tunes (α, δ) — contradicting "unchanged" — or the trial counts are not equal.
 - **Clean:** compute (~5e7 FLOPs/step vs ~1.1e12 for the backbone; the <1% claim is conservative and honest), data legality, and deployment (Σ̄ discarded, zero inference cost) all check out.
 
 ## Note on the artifact
 
 Line 3 of the frozen file contains a stray CLI error — `You're out of usage credits…` — captured in the verbatim transcript. The proposal body appears structurally complete (§1–6 plus recommendation); flagging it as a provenance artifact, not a basis for the verdict.
 
 ---
 
 I have not recorded this to `docs/` or updated `method_search_verdict.md`. Say the word and I'll write it up as `docs/pass77_nam_review_2026-08-07.md` plus a ledger entry with the mechanism, per the protocol's "when a candidate dies" rule.
