Choose **same-product feature mixing on the stronger So400 affine readout**. This changes training data support while holding representation capacity and losses fixed. It is a bounded test of the fixed-view hypothesis; the latest negative does not establish that hypothesis as the cause.

The [measured result](/home/rb/worktrees/sfora-positive-causality/docs/inshop_so400_cached_readout_result_2026-10-02.md) closes the fixed GELU recipe: real nonlinear deformation worsened both metrics in both seeds. Preserve that negative and the historical full2000 official **96.307498% R@1 / 81.882593% mAP@R**, without comparing its official scores directly with TRAIN-held scores.

1. **Freeze one input intervention.** Both arms use `phi(z)=0.5*z`, the same five optimizer tensors, shared fresh initialization, and frozen So400 features. For each scheduled anchor, candidate training has a fixed 50% probability of using

   `normalize((1−λ)·u_anchor + λ·u_donor)`

   where the donor is sampled uniformly from another image of the **same FIT product**, and λ is uniform on `[0,1]`. Singletons remain unchanged. Generate donor, coefficient and treatment-mask tables prospectively from separate fixed streams; bind them to resume identity.

   Keep 1,000 B64/micro16 updates, FP32 readout, CE margin .3/scale64, corrected rank weight8, AdamW LR1e−4/decay .05, clip1 and scaler128. Preserve anchor schedules and loss denominators. Refresh the detached bank with **unmixed, pre-update descriptors** in both arms; synthetic features never become bank members.

   This differs from the failed cross-product proxy synthesis, which introduced virtual classes and mixed compact features/proxies. Repeated coverage, image queues, teacher anchors and role masks remain closed. The older local-chord proposal stopped on novelty grounds before training; this proposal makes no novelty claim.

2. **Create fresh prospective TRAIN authority.** Use only the authenticated So400 FIT cache, SHA `c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716`. Keep singleton products in training. Use the existing `deterministic_class_partition` twice on nonsingletons: seeds **179071/179072**, producing approximately half training, quarter selection and quarter sealed validation.

   Freeze exact products, rows, query/gallery roles and counts before fitting. Fit PCA, center/scaling, proxies, bank and schedules exclusively on the new training partition. Retain the existing deterministic query/gallery role rule.

   Use optimizer-input seeds **179061/179069**, with fresh initialization and identical paired anchor schedules. Neither archived trained heads nor old PCA/proxies/banks initialize these arms. These TRAIN identities have been explored previously: freshness concerns the prospective decision protocol, not previously unseen data. The observed outer TRAIN-held panel has no selection authority.

3. **Run one decisive quality test, with an early stop.** After necessary qualification and discarded replay mechanics, train control061/candidate061 and score only selection. Stop immediately if candidate R@1 delta is nonpositive or mAP@R delta is negative.

   Otherwise complete candidate069/control069. Require the existing gate: each seed R@1 positive and AP nonnegative; equal-seed mean gains **at least +0.20pp in both metrics**; both paired product95% lower bounds positive, using 5,000 shared draws/seed179019. Report query intervals and both seeds.

   Only a selection survivor unlocks the sealed validation panel, using the same four terminal checkpoints and identical gate. No fitting between panels, intermediate checkpoint choice, best seed, coefficient sweep or threshold rescue. Any failure closes this fixed mixing procedure.

4. **Keep implementation small and isolated.** Three prospective files suffice:

   - `scripts/train_siglip2_identity_mix.py`: partition/initializer authority, deterministic mixing tables and matched training/replay.
   - `scripts/evaluate_siglip2_identity_mix.py`: cache-based packed export, selection/validation scoring and terminal decision.
   - `scripts/test_siglip2_identity_mix.py`: one runnable check covering product-safe donors, singleton identity, normalization, clean-bank refresh and resume correspondence.

   Reuse existing PCA, affine-head arithmetic, valid-anchor ranking, packer, tie semantics and bootstrap implementation. Current trainer/evaluator validators hardcode row/class counts, seeds and endpoint inventories; give the new driver explicit prospective inventories rather than weakening those frozen validators. Preserve original files and checkpoints.

5. **Enforce costs before escalation.** Use the current cached-readout policy: CPU120; mechanics/TRAIN/export/score300; 8GiB host, no swap, zero disallowed memory events, CUDA allocation `<10GB`, both lifetime locks and complete terminal accounting.

   | Work | Measured reference | Prospective estimate |
   |---|---|---|
   | TRAIN1000 endpoint | Whole service143–155s | Roughly100–180s per new endpoint |
   | Discarded mechanics | Whole service53–54s per arm | Roughly50–80s per arm |
   | Updated CPU qualification | Whole service91s | Approximately60–120s |
   | Selection/validation | Original scorer83s on the larger panel | Approximately20–90s per panel |

   Estimates are not admission evidence. Smaller banks reduce ranking work; clean-bank forwards add work. The first trained-quality falsifier should take roughly **8–15 minutes after implementation**, including qualification. No new encoder export is needed for that test. Any original terminal resource failure stops it without retries at altered caps.

6. **Give a survivor a complete deployment path.** Freeze a fresh full-TRAIN comparison using the authenticated FIT and held source caches, identical mixing policy and terminal1000 budget. Full-data resource fit is unmeasured and must pass unchanged caps. Package the complete frozen So400 vision state, learned factored affine readout, processor and normalization/packing contract.

   Public serving currently enforces `nn.Linear(1024,128)`. A survivor therefore needs explicit **1152-input, factored-affine checkpoint support**, followed by native/public raw-unit-packed parity qualification; do not assume folding the affine factors preserves packed bits.

   Then obtain fresh independently executed, fixed-checkpoint official confirmation, disclose prior benchmark exposure, and satisfy **both SOP and InShop protocol-matched external quality plus matched image-to-top-k B1/B32 speed**. No official-derived retuning. Any p99 claim retains the 10,000 interleaved paired-call protocol.

The main scientific risk is that interpolated features leave the useful image manifold or make training positives artificially easy. The sealed validation test directly rejects that outcome. A success supports this training procedure on frozen So400 features; it does not isolate source capacity.

The full public pipeline remains an engineering risk: the measured 12,599-image export already took284s. There is no evidence that complete official export fits300s. A survivor must demonstrate that exact whole-unit requirement; if it fails, it is blocked under current limits.

Carry forward the existing fixed Opus/Astra review conditions without a duplicate critique. Inspection was read-only; trainer/evaluator code matches the requested commit despite workspace HEAD advancing. No files, native jobs or quality evaluations were changed or run.
