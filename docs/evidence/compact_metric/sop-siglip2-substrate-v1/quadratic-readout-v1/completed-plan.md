Read-only specialist d8dc6602a25e40e8, Jev-routed GPT-6.1 Sol Max/XHigh, completed. No quality gain or resource fit measured.

**Choose one fixed-basis quadratic readout continuation at 128 dimensions. Reject 128→256 for this decision.** This is a new, falsifiable allocation hypothesis; no quality gain or resource fit is demonstrated.

The proposed nested widening already appears in [the teacher-retained trainer](/home/rb/worktrees/sfora-positive-causality/scripts/train_pe_teacher_retained256.py:31): copy the leading128 head rows, zero the additional outputs, append independent nonzero classifier tails, and pad the bank. Its Large result was KILL. Frozen So400 caches would change the experimental procedure, but the initializer itself is not distinct. The older Large1024 source/PCA diagnostic supplies no So400 width proof.

The gradient issue is real but solvable mathematically. At initialization, let the descriptor be `[a,0]` and a classifier row `[c,r]`. ArcFace uses

`cos = (a·c)/(||a|| sqrt(||c||²+||r||²))`.

Nonzero `r` can give the new output rows a CE gradient, but changes classifier normalization and therefore the original logits. With both classifier and bank tails zero, the extra output gradient is zero; the padded rank bank cannot unlock it initially. Nonzero tails permit learning, rather than guarantee a nonzero aggregate gradient. Shape-dependent GEMM and normalization also require actual raw/unit/code/inverse-bit witnesses before claiming numerical preservation.

Widening would add280,704 trainable scalars in this So400/1008-class head, approximately4.28MiB for weights, gradients and two moments, plus3.10MiB of bank storage. Codes grow130→258 bytes/row. These are allocation calculations, not measured costs. More decisively, the [Python wrapper](/home/rb/worktrees/sfora-positive-causality/src/sfora/cutile_int8.py:19), [native FFI](/home/rb/worktrees/sfora-positive-causality/rust/sfora-cutile-int8-score/src/ffi.rs:16) and top-k kernel require128. The generic private256 scorer does not provide the required deployed endpoint.

The selected intervention instead preserves the qualified affine representation and adds a small nonlinear correction:

```text
u = existing normalized-source path
z = D061(u - center061)
h0 = primary061(u) + up061(0.5*z)

control:   raw = h0 + A(z - meanTRAIN(z))
candidate: raw = h0 + A(z² - meanTRAIN(z²))
```

Both arms initialize `A[128,32]=0`; freeze the complete source061 head, its buffers, classifier and encoder. Train **only A:4096 scalars, one optimizer tensor**. Compute both fixed means from canonical TRAIN6355 alone. Preserve the existing normalization operations and their order.

The hypothesis is that second-order features in an already learned basis can improve retrieval without relearning the broad deformation that accompanied the failed GELU recipe. The demonstrated facts are narrower: the affine control reached96.309112% R1/80.572295% mAP@R, and the tested GELU, mixing and genuine-view treatments regressed. They do not identify this mechanism as the missing cause.

1. **Bind the accepted starting state.** Authenticate control061’s complete1000-update checkpoint, SHA `97db53ab934edbef2656c6e9509e4518421eaba02d3864bdc336eedd7787c7d2`, and canonical cache SHA `55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8`. Copy its stale terminal bank exactly; no refit, new PCA or augmentation. Start fresh, identical AdamW/scaler state for A in both arms, with a new local counter0 and recorded source counter1000.

2. **Implement one trainer and evaluator adapter.** Reuse the authenticated objective, schedule, refresh, fingerprint and packing mathematics. Give the new allocation its own schema and one-member checks; archived five-member checks remain provenance. Do not mutate imported helper globals or reuse the old zero-up/down-gradient predicates.

3. **Qualify learning before TRAIN.** CPU≤120s must verify initial source/control/candidate raw, unit and packed parity, frozen-state integrity and independent complete reload. Here
   `∂L/∂A = Σ gradient_raw × featureᵀ`;
   a zero A has no second zero factor blocking learning. Require an actual finite, nonzero A gradient on the fixed first TRAIN batch. Then discard paired mechanics17 versus independently restored8+9, verifying complete state and output replay.

4. **Run the cheapest decisive TRAIN comparison.** One seed179061 pair, **1000 cached updates per endpoint**, using the unchanged class-balanced anchors, canonical inputs, B64/micro16, ArcFace `.3/64`, corrected rank coefficient8, AdamW `1e-4/.05`, clip1 and scaler128. Refresh the bank from detached canonical **pre-update** descriptors, last duplicate wins. Score only the existing selection1734q/1715g/498 panel, exporting complete raw/unit/packed wires and independently replaying every query. No new encoder export is needed for this falsifier.

   Retain early KILL if ΔR1≤0 or ΔmAP@R<0. Only a pass admits the matched179069 pair, again conditional on source061. Retain each-seed R1>0/AP≥0, equal-seed gains≥+0.20pp on both metrics, and both paired-product95% lower bounds>0 using shared5000 draws/179019. Also require candidate nonregression against the preserved source061 selection point. These are continuation seeds, not independent source-training replications.

5. **Keep the original resource gates.** CPU≤120s; each mechanics, TRAIN and scoring/export unit≤300s;8GiB host, no swap, zero disallowed events, CUDA<10GB, both lifetime locks and complete peaks. Require whole-service and median-update ratios≤1.50. A’s weight/gradient/moment allocation is64KiB per arm; admission, inherited state and bank work still dominate unknown whole-unit cost. Failure stops this recipe without rescue.

6. **Make a survivor deployable.** The native128 backend already accepts its wire geometry. The concrete blocker is the [public encoder](/home/rb/worktrees/sfora-positive-causality/src/sfora/siglip2_compact_serving.py:110), which accepts only a1024→128 `nn.Linear`. Add an explicitly authenticated So4001152→128 readout path that loads the complete factored base, fixed means and A, preserves the FP32 arithmetic, and uses existing packing/gallery/search. Do not fold either arm and assume byte equality. Public B1/B32 packed/top-k parity must qualify the port.

A selection survivor permits the unchanged sealed-validation gate, followed by a separately frozen full-TRAIN model: canonical affine stage1000, then this same fixed-basis continuation1000. Full-scale cache preparation and resource fit remain unmeasured. Preserve full2000 and all valid Pareto models; root retains official SOP/InShop qualification, strongest-reference reconciliation, matched serving speed and the10000-pair p99 gate.

Main risks are quadratic amplification of nuisance variation, weak gradients under frozen proxies, stale-bank effects, and cached-to-public arithmetic mismatch. The bounded TRAIN result decides this recipe; it cannot establish a universal representation ceiling.

Read-only planning only: no edits, jobs, native execution, reviews, SSH, operator contact, or sealed/official quality reads.
