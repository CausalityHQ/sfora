# Fitter retention runtime audit

Consultation 240d0fc29d0f4b71; actual gpt-6.1-sol/xhigh; exit 0; elapsed 588s. Jev d457ae1a-295f-4c2e-a36a-8f3e99cc5994. Read-only; native UNRUN.

**GO for one source-only implementation: fitter-local encoder retention plus bounded phase timestamps.** This is justified for a single prospective qualification; its runtime benefit remains unmeasured.

The working fitter exactly matches the frozen source. Its [terminal log](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-cpu-v1/original.log) contains one useful marker: legacy `exit_rehash` completed in **16.153s**. The only reachable caller runs after `fit_run`. Thus both arms reached the original exit audit; completion of the subsequent new guard sweep and receipt publication remains unproved. An empty CPU output directory is consistent with qualification checkpoints having been deleted.

The source establishes repeated work, but does not measure its time:

| Existing authenticated receipt | Input guards | Guards under `/datasets/` | Known bulk bytes |
|---|---:|---:|---:|
| Original CPU v7 | 15,122 | 13,333 | 3,519,801,881 |
| Historical fitter CPU v3 | 15,129 | 13,333 | 3,519,801,881 |
| Historical linear fit | 15,133 | 13,333 | 3,587,539,998 |

Historical admission adds **30,262 uncached guard attempts**, including two full encoder hashes. On the successful CPU path, six reconstructions and seventeen successful integrity boundaries perform **23 further full encoder hashes: 39,374,736,909 bytes**. These counts exclude inherited admission and exit reads. The old fitter’s four measured fit cores total only **0.432s**; that measurement does not establish the new fitter’s phase costs.

Implement only these two source files:

1. In [fit_siglip2_prototype_residual.py](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:1168), add `retain_encoder(context)` using the evaluator’s [existing implementation](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:643). Obtain the descriptor through `context['old'].owned_encoder(context['legacy']).materialize()['checkpoint']`. Require the exact descriptor keys, canonical path, pinned SHA and basename `fresh_vision.pt`; perform the existing full `bound_file` authentication. Before mapping, require:

   ```python
   size == 1711945083 and 0 < size <= 2 * 1024**3
   ```

   Map exactly that file with `mmap.ACCESS_READ`, touch one byte at every page offset, and retain only the mapping in `context['_encoder_mapping']`.

   Insert the call after complete authority, cgroup and interpreter admission, immediately before the first Torch import. Enclose retention through receipt publication and the final cap check in `try/finally`; close/pop the mapping on success or rejection, with immediate cleanup on population failure. Include the existing evaluator-style retention descriptor in the receipt. Apply this same fixed behavior to CPU and subsequent matched fit units.

2. In [test_siglip2_prototype_residual.py](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_prototype_residual.py:489), add one bounded stdlib retention falsifier. Keep the readout helper, arithmetic, solver adaptation, cost timer boundaries and all existing guards unchanged.

For telemetry, add `phase_mark(started, phase, edge, arm=None, pass_index=None)`. Emit flushed monotonic elapsed timestamps around authority, retention, each fresh pass, historical parity, save/reload, the original exit audit, the new exit sweep/closure and receipt publication. Bound a successful CPU run to **30 events**; emit none per guard. Place logging outside the existing fit-core timers.

The VMA restriction permits this particular retention. [CheckpointPages](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:1240) requires exactly one mapping for its inode, with mode `rw-p`, offset zero and page-rounded length. Its reachable fitter uses concern warm, historical and newly serialized payloads. The retained encoder’s [metadata reader](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:644) uses Torch mmap **without `CheckpointPages`**. Consequently its temporary second encoder VMA needs no predicate relaxation. Retaining any warm or fitted checkpoint would violate this reasoning.

The bounded falsifier should use temporary payloads totaling at most 64KiB and prove read-only access, exact page coverage including the partial final page, descriptor/size rejection, cleanup, and a fresh SHA failure after same-size corruption while retention remains live. Exercise the actual `CheckpointPages` constructor against a distinct private mapping, and verify that two mappings of the same inode still fail. Check lifetime through both exit audits and publication. Preserve all arithmetic and guard-call structure. Parent verification: `timeout 15s python3 -B scripts/test_siglip2_prototype_residual.py`, with no native imports.

The principal risks are unchanged admission cost before retention, repeated hashing CPU cost that retention cannot remove, and increased charged file residency. The evaluator’s **249.204s versus 299.405s** result supports the hypothesis; it predicts neither fitter time nor memory.

**Prospective stop rule:** after parent integration and a new freeze, permit exactly one separately cold CPU300 qualification under unchanged 8GiB/noSwap/CUDA-hidden/both-lock conditions. Accept only normal terminal success below 300s with every existing parity, common-lambda, refit, reload, tamper, source, exit and resource gate passing. Any timeout or failed gate closes this intervention before matched fits or quality work, without retry or hot-set expansion.

No files were changed or trials launched. The closed trial, both learned Pareto heads and the full production goal remain preserved.
