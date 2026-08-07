# Pass 80 ETSD independent review (verbatim)

 I'll start by reading the frozen proposal and the search protocol that defines the verdict semantics.Now the literature attack. Let me load web search and verify the closest primary sources.I read the frozen proposal and the protocol, audited the verified repository packet, and ran a targeted literature check. No file edits, no GPU, no implementation.
 
 # Verdict: **DEAD**
 
 Dead independently at **Gate 1** (no eligible repository measurement, and the premise has already been adjudicated twice in-repo), at **Gate 2** (the frozen objective's own global optimum *is* the shortcut ETSD claims to price out, and the descriptor is an occupied reparameterization), on **preflight protocol** (hyperparameters selected on the CUB test split; diagnostic dataset ≠ gated dataset), and on **frontier arithmetic** (its own preregistered success crosses no audited horizon). Any one is dispositive under `docs/search_protocol.md` ("stop at the first gate a candidate fails").
 
 ---
 
 ## The single strongest reason
 
 **With `z ≥ 0`, the MS loss's global optimum is the class-private one-hot code.** ETSD's mechanism 1 asserts the fixed mass budget makes concentration *expensive*:
 
 > "A class signature concentrated in one or two coordinates consumes the entire budget and drives that image's similarity to everything else toward 0."
 
 Driving similarity to everything else toward 0 is exactly what the negative term of Multi-Similarity **rewards**. Since `z = √p ≥ 0`, all similarities lie in `[0,1]`, so the negative term's infimum is `S_neg = 0`, attained at **disjoint support**; the positive term's supremum `S_pos = 1` is attained when all images of a class share one `p`. On CUB (`C=100`) and Cars196 (`C=98`), both `≤ 512`, the joint minimizer is **one atom per class** — maximally private, maximally concentrated, exactly "a private direction per class, worthless on class 101."
 
 The cosine-logit bound does not rescue it. With `ℓ ∈ [−1,1]` and a realizable `ℓ_win = 1`, `ℓ_other ≈ 0`:
 
 | τ | `p_max` (k=32) | cross-class `S_neg` for two single-atom classes |
 |---|---:|---:|
 | 0.05 | ≈ 1 − 10⁻⁸ | ≈ 0.001 |
 | 0.10 | 0.9986 | ≈ 0.015 |
 | 0.20 | 0.827 | ≈ 0.30 |
 
 So at **two of the three preregistered τ points the collapse is essentially free**. The mechanism argument predicts the winner is τ = 0.2 — a **grid edge**, which by §6's own rule ("If the winning τ sits at a grid edge, the honest read is that the grid was wrong") is a self-declared failure.
 
 Mechanism 2 ("atoms are shared, so gradients are shared") has **no term enforcing it** in the main arm; load balancing is explicitly a *fallback arm*. `C(512,32) ≈ 10⁵²` is capacity, not supervision — the exact error Pass 51 (PSPL) died of: *"sharing target coordinates does not force sharing the computation that produces them, and unused geometric capacity is not unseen-class supervision"* (`docs/method_search_verdict.md`).
 
 This repository has already ruled on this identical geometry. Candidate **375 (CoVeR)**, `docs/fable_cover_collision_375_2026-08-05.md:83`: *"Both images and proxies lie in the nonnegative orthant, so cosine is in `[0,1]` … The negative loss therefore keeps pushing different classes toward disjoint supports, directly opposing cross-class reuse."* ETSD is that ruling with MS substituted for Proxy Anchor. MS's negative term has the same sign and the same infimum, so the ruling transfers unchanged — and ETSD is *worse* off, because CoVeR at least carried an (unsatisfiable) reuse term.
 
 ---
 
 ## Gate 1 — no repository provenance
 
 **ETSD cites zero repository measurements.** §5 concedes: *"I have not run a literature search for this."* Its causal error mode is stated as an armchair claim about "what everyone does."
 
 What the verified packet (`docs/current_evidence_reliability_audit_321_2026-08-03.md`, `docs/inshop_corrected_reference_seed3_result_2026-08-06.md`) actually measures on corrected In-Shop: train leave-one-out error **0.0049841589**; nearest-foreign-image/nearest-foreign-proxy confusion agreement **0.1569044123** (seed 0) and **0.1259176261** (seed 1), with error|agreement **0.0238857424** vs error|disagreement **0.0014664772**; class fragmentation **0.3932075472**; cross-seed persistent errors **908** queries, overlap coefficient **0.7675401522**, top-1 gallery agreement **0.8084822057**, same-wrong-identity only **0.6475770925**. **None** of these identify code-space geometry, mass concentration, private class directions, or a compositional-vocabulary deficit.
 
 The premise has been rejected here twice already:
 
 - **Candidate 371** (supervised-rank vs demanded-rank) — DEAD; the "label-bearing rank / private direction" measurement is tautological and unidentifiable (`docs/method_search_verdict.md:4704`).
 - **Pass 78 MCB-PA review** (`docs/pass78_mcb_pa_review_2026-08-07.md:20`): *"No Gate-1 repository measurement supports the claimed missing rank; the existing DOIR null-space control and candidate 371 reject that premise."*
 - **Candidate 375 CoVeR**, Gate 1: *"The verified repository packet contains no measurement of the proposed class-template/compositional-vocabulary deficit."* Same premise, same absence.
 
 **The secondary prediction is contradicted.** ETSD predicts hubness skewness reduced ≥ 20%. `docs/HANDOFF.md:310` records hubness reduction as correlation −0.82 but **causally negative** (CSLS **−0.65**, Sinkhorn **−3.16**); `docs/fable_hde_collision_373_2026-08-05.md:60` records the CUB hubness audit as directionally adverse; candidate 366 rejected generic hubness as an identified error cause. Those CUB arms sit below the audit-321 reliability boundary and so cannot *promote* a positive — but ETSD supplies no positive either, and Gate 1 fails on **absence**, per the CoVeR precedent.
 
 **Mechanism 4 is backwards on its own terms.** All-nonnegative codes force every pairwise similarity `≥ 0`, giving the Gram matrix a large leading eigenvalue along the mean direction — the canonical hubness generator (Radovanović et al., JMLR 2010). Equal L1 mass removes *norm* heterogeneity, but sphere hubness is driven by alignment with the local density/mean direction, not norm. And Roth et al. ICML 2020's evidence is that *higher* spectral entropy predicts better transfer; pinning every descriptor to a 32-of-512 nonnegative support lowers it. Same adverse reading the Pass 79 review applied to GCNP (`docs/pass79_gcnp_review_2026-08-07.md:40`).
 
 **Lane mismatch.** ETSD specifies ResNet-50/512-D/224px. The only audited controlled repository lane is **BN-Inception/512-D on the corrected 256-px corpus** — and audit 321 explicitly warns that the legacy `proxy-anchor-resnet50-512` protocol-family string is *misleading*. There is no same-lane repo base for ETSD's In-Shop arm, and **Cars196 has no corrected-recipe baseline at all** (`HANDOFF_BRIEF.md`).
 
 ---
 
 ## Gate 2 — the algebra is correct, and the descriptor is occupied
 
 **The algebra checks out.** `z_i/‖z‖₂ = exp(ℓ_i/2τ)/√(Σ_{j∈S} exp(ℓ_j/τ)) = √p_i` ✓. `⟨z_a,z_b⟩ = Σ_{i∈S_a∩S_b} √(p^a_i p^b_i)` = Bhattacharyya ✓. The "no ε-floor needed" observation is right. One minor slip, not load-bearing: under the standard Fisher–Rao normalization (`p ↦ 2√p`, sphere of radius 2) `BC = cos(d_FR/2)`, not `cos(d_FR)`; §1 is using the radius-1 convention.
 
 **But that algebra is a known map, and every claimed delta has been closed:**
 
 | ETSD's claimed delta | Status |
 |---|---|
 | "RootSIFT is post-hoc on hand-crafted histograms; here atoms are *learned*" | A learned, L1-normalized, √-then-L2 code over a learned dictionary is classical sparse-coding image representation: **Yang et al., CVPR 2009 (ScSPM)**; **Mairal, Bach & Ponce, Task-Driven Dictionary Learning** — already logged against CoVeR. Signed square-rooting predates RootSIFT in **Perronnin, Sánchez & Mensink, ECCV 2010**. |
 | "SPLADE is text, vocabulary-tied dimensions" | **Superseded.** **LexLIP (Luo et al., ICCV 2023)** and **STAIR (EMNLP 2023)** deploy learned **sparse nonnegative image-side** retrieval vectors. The image/text delta no longer exists. |
 | "sparsemax/entmax/top-k MoE were never used as a retrieval descriptor" | Also gone — LexLIP/STAIR *are* sparse-activation retrieval descriptors. |
 | "compositional atoms → graceful degradation on unseen classes" | **SPA (Kundu et al., NeurIPS 2022)** globally pools a sparse learned word histogram *specifically arguing that shared visual primitives represent unknown/private classes* — ETSD's stated transfer mechanism, published, in an open-set setting, already logged against CoVeR. |
 | (unnamed) response over a bank of learned atoms for novel categories | **Classemes** (Torresani et al., ECCV 2010), **PiCoDes** (Bergamo et al., NeurIPS 2011). Repo already recorded this branch as occupied (candidate 371 audit: *"the alternate overcomplete-measurement branch is Classemes/proxy-response or attribute/concept embedding"*). |
 | (unnamed) in-repo recurrence | **375 CoVeR** (√-histogram over sparse nonnegative learned atoms + atom-reuse claim, DEAD Gates 1&2); **264** (sparse nonnegative shared-atom proxies, rejected); **51 PSPL** (fixed-simplex atoms in the descriptor, DEAD). |
 
 **Mechanism-level statement: ETSD is RootSIFT/Hellinger normalization applied to a top-k-masked softmax over a row-normalized linear head — i.e. an activation reparameterization of the existing head.** Its three novel-sounding components (√-simplex geometry, learned sparse nonneg. retrieval code, shared-atom compositionality) are each separately occupied, and their conjunction is candidate 375 minus the patch stage.
 
 **Top-k gradients: annealing is not the fix.** The mask is hard and non-differentiable; `∂z_j/∂ℓ_j = 0` exactly for `j ∉ S`, and no straight-through or noisy estimator is specified. Annealing 512→32 over the first third distributes gradient early but supplies **no restoring force afterwards**: once `k` freezes, an atom outside every sample's top-32 gets zero gradient and its row `ŵ_i` is frozen while `f̂` keeps drifting. Shazeer et al. 2017 — the paper §5 itself cites for this concern — solved it with **noisy** top-k gating **plus** a load-balancing loss, *both in the method*. ETSD keeps the failure mode and demotes both fixes to a contingency arm. Separately, ending the anneal at exactly 1/3 of epochs commits the support to whichever atoms are momentarily on top; no anneal-fraction control exists. §6 calls the annealing "load-bearing" — correct, and it does not bear the load.
 
 ---
 
 ## Preflight — not protocol-clean (four defects)
 
 1. **Hyperparameters are selected on the test split.** §3 preregisters τ and `k_final` "from preflight"; §3-preflight step 3 is *"Evaluate R@1 on the 100 **unseen** classes."* On CUB's standard 100/100 split those unseen classes **are the test set**. This violates `HANDOFF_BRIEF.md` discipline #4 ("Never train, fit, or select on the test split") and protocol Gate 6. Contrast the Pass 79 review, which certified GCNP clean *precisely because* it used leave-class-out folds **inside the training split**. ETSD does not.
 2. **Wrong dataset for the gated decision.** The go/no-go is CUB frozen features; protocol Gate 4 makes corrected In-Shop the decisive screen and warns *"CUB screening at n=1 produced a false positive here once already (+0.52, retracted)."* This is verbatim the spec defect already recorded in `RSPG_SPECDEFECT.md:15`: *"A diagnostic whose job is 'does this signal exist in the data I am about to train on' cannot answer that question from a different dataset."*
 3. **The probe cannot exercise the claimed mechanism.** With the backbone frozen at ImageNet init, "atoms" are linear functionals of fixed features; the compositional claim requires the backbone to learn sub-class part detectors. So neither a pass nor a fail identifies the causal quantity, and the support-overlap / atom-entropy diagnostics inherit that non-identifiability.
 4. **The gates are weak or ill-defined.** "ETSD mean ≥ baseline mean (3 seeds)" passes with probability ≈ 0.5 under the null, and repo discipline #3 records that a 2–3-seed sd is worthless (n=3 → 0.153, n=6 → 0.367). And "usage entropy over 512 atoms" never says **mask-based or mass-based**: the top-k mask selects exactly 32 atoms every sample, so a mask-based entropy is saturated by construction and **blind to the one-hot collapse above**. Only a mass-weighted entropy can detect it.
 
 ---
 
 ## Frontier arithmetic — crosses nothing
 
 Against the corrected four-seed In-Shop frozen-final reference — mean **0.9153889436**, sample SD **0.0013195712**, four-seed power floor **0.9185** (+0.31 pt):
 
 - ETSD supplies **no quantitative frontier-crossing argument at all**. §4 states "+1.5 points absolute" on CUB with **no named base on any dataset**. The protocol requires that argument from the proposer.
 - Its In-Shop bar is only the replication threshold ΔR@1 ≥ +1.0 → **0.9254**. That nominally clears 0.9185, but it is a pass/fail bar, not a forecast with an interval, and the base is a *different lane*.
 - Taking its own success thresholds at face value against audited horizons: CUB 0.6919 + 1.5 = **0.7069** vs Lane A **0.734**; In-Shop 0.9154 + 1.0 = **0.9254** vs Lane A **0.930** (Lane B 0.939; CRT 0.9448); Cars196 has no corrected base to add to. **ETSD's own preregistered "success" crosses no audited matched-capacity horizon on any dataset.** Its falsifier threshold (+0.5 CUB) falls further short still.
 
 ---
 
 ## What is right, and should be preserved in the ledger
 
 - The `z = √p` / Bhattacharyya derivation and the "exponential *is* the square root, so no ε-floor" observation are correct and cleanly stated.
 - Deployment legality is clean: single model, single view, 512-D unit-norm float, plain inner product, no test-time transform, added inference cost genuinely ~nil.
 - §4/§6 reporting discipline — fixed final epoch as headline, best-epoch numbers **labelled inflated**, selection correction rather than a naked max — is materially better than recent passes and matches this repo's own retraction lessons.
 - Falsifier 3 (a gain requiring per-dataset `k`/`τ` is hyperparameter search, not a method) and the "effect without mechanism" falsifier are both good gates.
 - §6 honestly names the dead-atom and grid-edge risks — and both fire.
 
 ---
 
 ## Exact kill gates a revival must clear, in order
 
 1. **Gate 2, algebra, zero compute.** Exhibit a term *in the frozen objective* whose minimizer excludes the disjoint-support / one-hot-per-class solution for `C ≤ 512`, given `z ≥ 0 ⇒ S ∈ [0,1]` and MS's negative infimum at `S = 0`. A fallback arm does not count. Until then mechanisms 1 and 2 are refuted by the objective's own optimum.
 2. **Gate 1, ≤ 1 CPU-hour, on the existing frozen seed-0–3 corrected In-Shop embeddings.** Show that the 908 cross-seed-persistent wrong queries are caused by class-private descriptor geometry rather than the already-measured foreign-proxy confusion channel (agreement 0.1569/0.1259; error|agreement 0.0239 vs 0.00147). Thresholds locked in advance, both seeds, permutation null. CoVeR failed this same gate.
 3. **Gate 2, prior art.** Distinguish the *deployed object* from RootSIFT / Fisher-√ (geometry), ScSPM and task-driven dictionary learning (learned sparse code as descriptor), SPLADE + **LexLIP (ICCV 2023)** + **STAIR (EMNLP 2023)** (learned sparse **image-side** retrieval vectors — the "text-only" delta is gone), **SPA (NeurIPS 2022)** (pooled sparse word histogram argued as the unknown-class transfer mechanism), Classemes/PiCoDes, and in-repo **375**, **264**, **51**.
 4. **Preflight, rebuild.** Move all τ/`k`/loss-scale selection to leave-class-out folds **inside the training split**; run the diagnostic on the dataset it gates (corrected In-Shop); define atom-usage entropy as **mass-weighted**; replace the 3-seed `≥` gate with a threshold of stated power. A frozen-backbone probe cannot certify the compositional mechanism at any seed count.
 5. **Top-k validity.** Put noisy gating **and** load balancing in the method (Shazeer et al. 2017), or measure that post-anneal out-of-top-32 atoms actually re-enter; and add an anneal-fraction control.
 6. **Frontier.** Name one lane and one same-lane base with recipe digest, raw **and** frozen-final. Any In-Shop forecast must exceed **0.9185** at four seeds against **0.9153889 ± 0.0013196**; a general claim must confront **0.930** / **0.939** / **0.9448**.
 
 None of this is reachable by tuning: gate 1 is an inequality on the frozen objective and gate 3 is a literature fact. Per `docs/search_protocol.md`, any substantive repair is a **new proposal** that must restart blind generation and freezing — I have not reinterpreted or repaired the training object.
 
 **Process note, stated honestly:** this is the local evidence-aware critic verdict you asked for. The protocol's mandatory *second fresh isolated session* review is a separate artifact and remains outstanding (the frozen proposal's own header shows the Fable session hit a credit limit). Since ETSD fails Gates 1 and 2 independently and the protocol says to stop at the first gate failed, that outstanding review does not change the disposition — but it is not yet on record. I made no file edits and did not update `docs/method_search_verdict.md`; say the word and I'll write this up as `docs/pass80_etsd_review_2026-08-07.md` plus a ledger entry with the mechanism.
 
 **Sources:** [LexLIP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_LexLIP_Lexicon-Bottlenecked_Language-Image_Pre-Training_for_Large-Scale_Image-Text_Sparse_Retrieval_ICCV_2023_paper.html) · [STAIR, EMNLP 2023](https://arxiv.org/pdf/2301.13081)
