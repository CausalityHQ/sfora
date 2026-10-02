Delete the duplicate origin scan at [evaluator:887](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:887):

```diff
-    origins = source.imported_origins(context['selected']['extract'],
-                                     context['selected']['packages'])
-    check_origins(context, origins)
     origins = exit_rehash(context)
     check_origins(context, origins)
```

`exit_rehash` already performs the same scan and rejection at lines 861–862, **before** exporter rehash at 863. This deletion preserves foreign-origin rejection, whole-payload validation, independent strict reload, raw/unit/packed equality, and uncached exit reads.

The frozen evaluator/trainer hashes and relevant helper hashes match repository source. No files were edited or qualification run.

| Work | Exact caller locations and duplication accounting |
|---|---|
| CPU admission | Evaluator `run:873–875` calls authority, prerequisites, then native admission. CPU prerequisites return immediately at `612–614`; this phase does not score a panel. |
| Fresh cache admission | Trainer `authority:386–387` calls `cache_facts` twice. Each call hashes the complete file at `313`, then reads/hash-validates every payload row at `315–339`. Together, the row checks inspect **14,641,920 FP32 values**. Native consumption retains another fresh hash at `training_features:561`. |
| TRAIN guard inventories | Evaluator `authority:540–541` makes **30,230 full-file hash calls covering 15,116 unique paths**: 15,112 shared paths plus two checkpoint rereads. Shared paths include all **13,283 images**, source weights and native libraries. This duplication remains outside the proposed correction. |
| Actual reloads | `qualify_heads:750,759` invokes `load_head` twice per endpoint. First CPU therefore performs **four complete terminal loads and four initializer loads** at `669,681`, including complete fingerprints, finite checks and strict head construction. It also performs eight schedule/mask generations and 4,000 update-provenance checks; feature tensor hashing alone covers **2,359,296,000 bytes**. |
| Selected avoidable scan | Direct origin scans occur at `native_start:649`, `run:887`, and `exit_rehash:861`. Removing `887–889` reduces three direct scans to two. The adjacent pre-exit scans change from **2R to R file SHA reads**, where R is the actual distinct loaded-origin inventory. |
| Remaining exit work | `exit_rehash:863` retains genuine exporter rehash; `865` retains fresh reads of every evaluator guard. The nested original exporter also hashes all images, followed by source guard rehash. These predicates remain intact. |

The [previous cached evaluator](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_cached_readout.py:322) used `FlatAdmission.bound_file` for overlapping endpoint inventories at 323–324. Its exit used the [single fresh union pass](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:965). It also constructed and independently reloaded the frozen encoder (`583,596`); the genuine evaluator does not, but adds initializer and per-update provenance work. The immediately preceding identity-mix evaluator likewise had four terminal/initializer load pairs, without the new 1,000-step feature-provenance loop. Historical CPU services were 91.223s and 102.513s respectively; neither is a matched measurement of this correction.

The bounded falsifier should be **one stdlib-only, in-memory test**, under five seconds with a 1MiB synthetic origin. AST-extract the actual pre-exit statements and `exit_rehash`/origin validators; supply recording doubles for origin hashing and exporter rehash. Assert:

- Exactly one pre-exit origin scan; current v1 fails with two.
- Foreign or conflicting origins reject before exporter rehash.
- A same-size byte mutation during exporter rehash fails the retained fresh guard read.
- Returned origins and guard inventories match the unmodified path.

The risk is effectiveness: the [failed log](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/genuine-view-v1/evaluation-source-v1/first-cpu-v1-failed.log) has no phase timings. This is proven redundant work, **not a proven location or explanation of the timeout**.

The parent’s next gate is a separately frozen evaluator version containing this deletion, the falsifier, and flushed `perf_counter` markers around authority, native admission, each endpoint/reload, and exit rehash. Run one prospective **CPU120** unit with unchanged locks and resource predicates. Require complete normal exit and uncached-exit acceptance before admitting first selection. Preserve v1’s terminal **KILL**; quality remains **UNREAD**, with no scientific rejection inferred and no cap rescue.
