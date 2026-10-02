# Prospective So400 post-layernorm adaptation

Status: read-only plan completed; review pending. No implementation or native launch authorized by this artifact alone. Root retains protocol/resource/quality authority under the SAME active full production goal.

Planning consultation b3ca91099b144a74, GPT-6.1 Sol High, original terminal exit0; repository evidence90a5b2b5. This proposes one next bounded procedure; gains and fit are unmeasured.

**Choose one intervention: train only So400’s existing `post_layernorm.weight` and `post_layernorm.bias`, before its frozen attention pool.** Keep native256, the affine128 readout, objective, images and serving graph fixed. This adds **2,304 trainable scalars in two native tensors**.

The cached negatives establish failure of those recipes; they do not identify a frozen-backbone bottleneck. This intervention directly changes token representations entering nonlinear attention pooling, which an affine head on cached pooled vectors cannot generally reproduce. The ledger’s pool-only and final-block procedures froze post-layernorm; the So400 twelve-block procedure stopped on runtime without a quality result. Norm-only adaptation is a distinct, cheaper allocation, with uncertain quality.

| Explanation | Existing evidence | Decision |
|---|---|---|
| Frozen pooled representation limits useful learning | GELU, same-product mixing and genuine augmented-view recipes all regressed | Test a change before pooling; no universal bottleneck claim |
| Broad encoder adaptation is needed | Partial-allocation results are mixed; twelve-block So400 training remains runtime NO-GO | Start with two native vectors, preserving the pretrained pool |
| More resolution or view support is needed | Resize screens and wider crops failed; trained288’s proposed detail mechanism was rejected; genuine mild views failed | Leave pixels, resolution and support unchanged |

1. **Use one authenticated learned starting point.** Both arms reconstruct the original frozen So400 source and load the accepted genuine-view **control061** head, classifier and bank from `resume.pt`, SHA `97db53ab934edbef2656c6e9509e4518421eaba02d3864bdc336eedd7787c7d2`. This is a qualified TRAIN-trained control, not the killed candidate. Authenticate its complete original state before extracting those members. Reset AdamW/scaler in both arms under a new explicit continuation identity; preserve provenance of the original 1,000 updates.

   Control freezes all 448 native tensors. Candidate trains exactly the two `[1152]` post-layernorm tensors and freezes the other 446. Head/classifier membership remains five tensors: **five optimizer members for control, seven for candidate**. No pool, block, position, resolution or cached-input treatment.

2. **Reuse the existing implementation narrowly.** Add a small `train_siglip2_postln_adaptation.py` driver and runnable contract check. Reuse:
   - `qualify_siglip2_substrate_cpu.fresh_source()` for authenticated construction, then declare and independently verify the new roles.
   - The genuine-view affine head factory, initializer/state semantics and class-balanced schedules.
   - Native image forward and update arithmetic from `train_siglip2_substrate_adaptation.py`: FP32 parameters, FP16 encoder autocast, FP32 normalized pooled/head path, ArcFace `.3/64`, corrected valid-anchor rank coefficient8, B64/micro16, clip1 and detached pre-update bank refresh with last-duplicate semantics.
   - Its **original serial** fingerprint, frozen-tensor digest/version audit, bounded checkpoint writer and mapped-page restoration.

   Use native AdamW rate `1e-5` for the two vectors and existing `1e-4` head/classifier rates, decay `.05`. New local membership/payload checks must explicitly enforce five/seven members; do not repurpose the old 208-member admission or modify archived helper globals.

3. **Freeze one TRAIN-only paired100 rejection test.** Retain partition SHA `702f763e…95c676c`: TRAIN6355/1008, selection1734q/1715g/498, sealed validation1749q/1730g/498. Both arms receive identical canonical RGB and processed pixels, original row mappings and anchor schedules. Actual image forwards supply every update; the canonical cache serves only initialization/provenance. Neither arm uses augmented-cache rows.

   Qualify actual CPU roles, finite gradients, observable norm perturbations, frozen complement and independent complete reload first. Require graph-free post-layernorm input and candidate gradients through the frozen pool; control native gradients remain absent. Then discarded **17 versus independently restored8+9** mechanics for both arms, complete state/output replay and uncached exit.

   Run fresh control061 and candidate061 for **100 online updates each**, then one new selection comparison. The historical 96.309112/80.572295% control point is context, not the fresh comparator. **KILL if ΔR1≤0 or ΔAP<0; no second seed or validation.**

   Only a first-screen pass admits candidate069/control069. Retain each-seed R1>0/AP≥0, equal-seed mean improvements **≥+0.20pp on both metrics**, and both paired-product95% lower bounds>0 using shared5,000 draws/179019. These are continuation seeds conditional on one trained source, not independent pretraining replications. Full selection GO alone permits the frozen sealed-validation comparison.

4. **Keep whole-job resource accounting.** Every original unit retains 8GiB host/noSwap/zero disallowed memory events, allocated CUDA<10GB, both lifetime locks and lifetime peaks. CPU qualification≤120s; mechanics, TRAIN, export and scoring≤300s. Updated export must run two independent complete native reloads/forwards sequentially, verify every panel row’s raw/unit/packed outputs and per-query replay, then finish source/input/output rehash and normal exit.

   Added norm weights, gradients and two Adam moments total roughly **36KiB**; this excludes activations, model ownership, archives and page cache. The substantial cost remains image forward, frozen-pool backward, source admission and strict reload. Prior So400 sourceCPU peak6.855GB demonstrates limited headroom, not new qualification. Prior v6 mechanics took194.051s, while broad TRAIN100 remained runtime NO-GO.

   Record admission/construction, decode/preprocess, forward/backward, integrity snapshots, serialization/reload and exit rehash separately. Report preparation independently; compare fresh whole TRAIN-service and median-update ratios≤1.50. The first screen’s maximum envelope is **41 minutes**: initial CPU, two mechanics, two TRAIN, two updated CPU proofs, two exports and one scorer. Estimates admit nothing; an overrun stops the procedure.

5. **Delegate one independent slice while DGX qualifies mechanics.** Root freezes the endpoint/wire schema and owns the native driver and execution. A later bounded child can implement `score_siglip2_postln_adaptation.py` plus one stdlib contract check, reusing the genuine-view metric/bootstrap/gate functions. Its scope is complete terminal/chain admission, exact native-state and wire bindings, five/seven optimizer inventories, summed costs and rejection of missing/stale/parity-failed endpoints. Synthetic inputs only; no DGX, quality reads or trainer edits.

6. **Promote only the frozen recipe.** A selection-and-validation survivor leads to fresh full-TRAIN affine initialization/training followed by the same matched100 post-layernorm continuation. Declare preparation shards and training chunks prospectively, preserving the complete update budget and resume chain; never partition an overrun retrospectively.

   Risks are insufficient plasticity, ineffective gradients through the frozen pool, harmful channel reweighting, bank staleness and lifecycle overhead. Any failure closes this allocation without a rate, site, budget or precision rescue. Preserve full2000 and all valid Pareto points. Native256 adds no serving operations, but speed remains unmeasured: root still owes strongest protocol-matched SOP/InShop quality, public B1/B32 parity and matched end-to-end speed, including the required10,000 paired calls and p99 uncertainty.

Read-only planning completed at `90a5b2b5`; no files edited, jobs launched, consultations started or quality data reopened.
