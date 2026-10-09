What this change does: computes the genuine full-B64 total gradient, releases that graph, then independently recomputes the full-B64 ranking gradient under restored RNG. The microbatch training path remains unchanged.

**Verdict: no blocking flaw found in the proposed mathematical design. It is suitable for bounded implementation review; native qualification remains blocked.** The trainer and original-log hashes match the supplied identities.

The following are essential implementation requirements, not newly demonstrated defects in the proposal:

1. **Make cleanup enforceable on failures as well as successful passes.**  
   The current helper initializes empty weakrefs and populates them only after all calculations succeed; its failure path suppresses secondary cleanup errors. See [trainer:1320](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_rank_routed_mlp.py:1320) and [trainer:1366](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_rank_routed_mlp.py:1366). Simply wrapping this structure around two passes would not establish the proposal’s stronger exceptional-lifetime guarantees.

   Register disposable owners before dropping them, clear pass-local aliases, and restore RNG in an unconditional outer cleanup path. A first-pass failure must prevent replay. Preserve the primary exception and existing [outer traceback cleanup](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_rank_routed_mlp.py:2020). Verify state equality; do not silently repair mutated parameters or optimizer state.

2. **Test gradient provenance, not just equal numbers.**  
   The existing [fake autograd implementation](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_rank_routed_mlp.py:562) recursively adds gradient components. Numerical agreement alone therefore cannot distinguish a genuine total-gradient call from the forbidden synthesized result.

   Extend the existing [full-reference falsifier](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_rank_routed_mlp.py:850) to require:
   - Original A/C and total-six calls on graph 1; ranking-six on distinct graph 2.
   - Graph 1 dead before the second forward.
   - Exactly one **encoder-reaching** backward per graph, with `retain_graph=False`; the six-input detached-regression check must not count as encoder backward.
   - Rejection of synthesized totals/ranking, retained original-MSE aliases, replay mutations, and faults in either pass. Check RNG restoration and primary-error preservation on those failures.

3. **Keep replay identity and source authority independent of the returned gradients.**  
   Fresh hashes of pixels, features and raw outputs, exact ranking-scalar agreement, restored RNG, and unchanged complete payload make the proposed replay scientifically defensible. The [pixel loader](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_actual_objective_encoder_gradients.py:163) already authenticates images and checks global CPU RNG preservation.

   Use the existing current-byte fingerprint path without a cache or retained tensor snapshots. Preserve genuine `loss_terms`, original A/C comparison, all existing tolerances, and the independent [full-versus-micro comparisons](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_rank_routed_mlp.py:1460). Add the proposed exact inverse to `2fb4f237…`; retain the existing [historical inverse chain](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_rank_routed_mlp.py:234).

**Resource fit remains unresolved.** The first violation is bracketed at canonical total backward, but augmented ranking already raises `max` from 228 to 842, and subsequent microbatch work raises it from 1367 to 1643. See [original.log:88](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/rank-routed-mlp-cpu-v2-result/original.log:88) and [original.log:113](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/rank-routed-mlp-cpu-v2-result/original.log:113). Removing graph overlap does not establish allocator release or whole-unit fit.

The proposed 121/123 record counts fit the 128-record cap. They do not independently prove the byte budget; retain both byte guards and verify the implemented records. The original 111 records total 115,664 bytes.

The smallest complete change remains the helper, its fixed observation sites, existing falsifiers, and exact inverse. No derived gradients, relaxed witnesses, optimizer changes, or cap changes are justified.

Not checked: no edits, tests, Torch/native execution, SSH, images, children, or consultations.  
Risk: the revised implementation may still fail unchanged CPU600/8GiB qualification; no performance or quality conclusion follows.
