**NO-GO for another consumer launch based on a claimed minimal speed fix.** I found redundant work, but no single intervention with defensible savings sufficient to complete both CPU qualification and subsequent scoring within 300 seconds while preserving every required predicate. This is an evidence-limited decision, not proof that optimization is impossible.

No files were edited or native jobs started. The live HEAD advanced during inspection, but the evaluator and fitter bytes match both requested commit `540df95c` and the frozen execution hashes.

**Measured evidence**

- Admission finished at **144.255880 seconds**. The service timed out at **300.217 seconds**, with peak memory **3,206,475,776 bytes**, zero swap and zero memory events. No accepted receipt exists. See [decision.json](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-evaluation-cpu-v1/decision.json:2).
- The log contains `authority_end`, then termination; no `exit_rehash_start`. It cannot distinguish interruption during native preparation, reload qualification or synthetic bootstrap. See [original.log](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-evaluation-cpu-v1/original.log:26).
- Fresh matched fits remain cost-qualified: linear **212.169 seconds**, concat **234.821 seconds**, whole-service ratio **1.106764**, fit-core ratio **0.923030**. These measurements do not establish evaluator timing. See [cost-decision.json](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-fit-concat-v1/cost-decision.json:6).
- Fitter CPU qualification passed at native wall **266.657774 seconds**, with whole service **268.001 seconds**. Its different admission and call schedule prevent treating its remaining margin as evaluator headroom. See [fitter decision](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-cpu-v2/decision.json:9).

After the evaluator’s measured admission, only approximately **155.744 seconds** remained for native startup, four complete reload/witness paths, bootstrap, full exit authentication, receipt publication and service overhead.

**Actual call/read/ownership chain**

| Path | Source correspondence | Required predicate |
|---|---|---|
| Native startup | [evaluator:666](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:666) | Qualified origins/interpreter/cgroup precede encoder retention and native preparation. |
| Encoder retention | [evaluator:643](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:643) | Fresh SHA authentication, exact 1,711,945,083-byte file, read-only mapping, complete page population. |
| Each independent head load | [evaluator:695](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:695) | `reload → integrity → canonical TRAIN feature fingerprint`; complete endpoint identity and payload binding. |
| Reload reconstruction | [fitter:922](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:922), [fitter:735](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:735) | Fresh original preparation, independent head/input ownership, serialized complete typed digest, restored fitted statistics without fitting, reconstructed frozen state and output parity. |
| Every head forward | [evaluator:713](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:713) | Complete integrity immediately before and after arithmetic. |
| Release before reload | [evaluator:725](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:725), [original release:965](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7/train_siglip2_quadratic_readout.py:965) | State is cleared; weak model ownership must be empty before reconstruction. |
| Final exit | [evaluator:1087](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:1087) | Original complete exit guard union, fresh encoder composition and every pinned source closure. |

The CPU schedule is two independent reloads per arm, each with one TRAIN witness forward: [qualify_heads:731](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:731). Successful scoring also retains four independent reloads, but each performs both TRAIN and full-panel forwards before candidate quality: [score_panel:1015](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:1015).

**What the source implies about the two suspected bottlenecks**

The encoder sweeps are substantial, but they are explicit authentication gates. From retention through qualification, before exit, the source requires **at least 22 fresh complete encoder-file SHA passes**:

- One in retention and one in initial fitter preparation.
- Five per reload: reconstruction preparation, reload integrity, caller integrity, and the forward’s two integrity boundaries.
- Four reloads.

That is approximately **37.66 GB of logical encoder-byte hashing**, even with encoder pages retained. Scoring increases this minimum to **30 passes**, approximately **51.36 GB**, because each reload performs two forwards. These are **source-derived counts, not measured phase durations**. The fitter’s [integrity loop](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:887) and [bound_file implementation](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:130) freshly read and hash bytes on every call. Eliminating those calls would skip required authentication.

The native metadata load is different. Pinned [encoder_metadata](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7/train_siglip2_quadratic_readout.py:644) deserializes tensor metadata, checks all 448 vision layouts, hashes RNG/buffer facts, and clones config/buffers. **Its vision loop does not convert or hash every vision tensor’s contents.** Calling it an additional full 1.7 GB tensor-conversion sweep would be incorrect.

The startup assignment to `context['typed_encoder']` at evaluator line 688 appears unused by this evaluator’s actual path; reconstruction obtains its own typed config/buffers. Removing it is the smallest plausible deletion. However:

- It removes one metadata reconstruction, **not the mandatory encoder SHA sweeps**.
- It changes startup validation coverage and the native-start AST locked by [the existing falsifier](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_prototype_residual_evaluation.py:168).
- Its savings are **UNMEASURED**; there is no basis for claiming it closes either cap.

For bootstrap, the verified pinned [bootstrap_lower](/home/rb/worktrees/sfora-positive-causality/scripts/score_inshop_crop_view_pair.py:78) uses 5,000 draws in batches of 512. Across both synthetic panels, the 16 unchanged calls generate:

`4 × 5000 × (498 + 1734) + 4 × 5000 × (498 + 1749) = 89,580,000`

sampled group indices. Complexity is **O(draws × groups)**, not O(draws × queries²). The largest `picked` batch contains `512 × 1749` indices. Actual runtime is **UNMEASURED**. Skipping calls, deriving the AP intervals from R1, sharing previously computed draws or replacing the pinned implementation would violate the frozen call/source contract.

**Actionable disposition and bounded falsifier**

Keep the failed attempt closed and the accepted fit endpoints intact. Do not promote partial state or launch scoring. The [freeze](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-evaluation-source-v1/freeze.json:9) requires complete CPU qualification before scoring.

The root can falsify this audit’s structural claims with **one ≤30-second stdlib-only source check**, without native imports, dataset reads or a profile job:

1. Verify evaluator/fitter hashes against frozen execution and original helper hashes against their pins.
2. Parse the actual call paths and verify the four reloads and the **22 CPU / 30 score** minimum encoder SHA-pass counts.
3. Verify that `encoder_metadata`’s vision loop performs layout checks rather than per-vision byte conversion/hashing, and resolve every reachable `typed_encoder` consumer.
4. Verify the pinned bootstrap loop bounds, batch size, draw count and seed.

A contradiction rejects this audit before spending another capped job. Passing establishes structural correspondence only; it supplies **no runtime savings evidence**. Under the current restrictions, no honest fast source-only test can demonstrate completion within 300 seconds.

The main risk is mistaking removable metadata overhead for sufficient critical-path savings. CPU qualification would also leave scoring’s extra integrity, archived-source replay, quality replay and persisted-wire work unproven. The full SOP+InShop production quality-and-speed goal remains unmet.
