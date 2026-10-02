**Implement one evaluator-owned fresh union pass in `exit_rehash`; preserve every non-hashing exit predicate explicitly.** Change only the evaluator and its existing stdlib test. Keep the frozen trainer/exporters unchanged.

The v3 evidence supports this target. Its exit rehash consumed **42.112694 seconds**. The internal receipt was published; its resource timestamp is **119.636076 seconds**. The enclosing service still timed out at 120.119 seconds, so that receipt remains inadmissible. The log cannot distinguish interpreter teardown from subsequent wrapper finalization. [Evidence](docs/evidence/compact_metric/sop-siglip2-substrate-v1/genuine-view-v1/evaluation-source-v3/terminal-outcome.md)

Current bulk-read accounting, verified against the saved inventories:

| Caller | Full-file SHA visits |
|---|---:|
| Evaluator → `source.imported_origins` | 1,358 |
| Genuine exporter → reference → `all_fit_images` | 13,283 |
| Reference → source `rehash`, prior guards | 15,070 |
| Genuine exporter’s own guards | 14 |
| Evaluator’s outer guards | 15,132 |
| **Total / distinct union** | **44,857 / 15,132** |

This excludes the small closure, partition and AST rereads. All nested inventories and actual origins are contained in the saved evaluator inventory with matching digests. Thus **29,725 repeated bulk hash visits** are removable: images and actual loaded origins currently receive three passes.

The caller/predicate ledger is:

| Existing location | Required replacement or retained operation |
|---|---|
| [Evaluator `exit_rehash`](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:862) → [source origin discovery](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_siglip2_substrate_cpu.py:480) | Enumerate current `sys.modules` and `/proc/self/maps`; retain genuine `loaded_module_origin`, including `torch.ops`/`torch.classes` identity checks, canonical files and package-root containment. Preserve returned modules/native-files inventories. |
| [Evaluator origin qualification](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:591) | Retain package equality, authenticated qualified-origin backing, conflicting-digest rejection and rejection of every unqualified actual origin. Admission’s distribution/version resolution remains unchanged. |
| [Reference `all_fit_images`](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_substrate_fit.py:134) and [source `rehash`](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_siglip2_substrate_cpu.py:523) | Call genuine `fit_rows`, compare its first-two resolution with `prior['images']`; resolve all rows, enforce containment, 13,283 distinct paths and ordered equality with `prior['all_images']`; bind every image digest into the checked union. |
| Source/prior, genuine and evaluator guard loops | Conflict-check their complete union, then freshly stream every union file once. No admission cache, stat shortcut or last-writer-wins dictionary merge. |
| [Genuine exporter `rehash`](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:258) | Retain fresh authenticated partition read, genuine `selected_manifest`, resolved TRAIN mapping equality and genuine `image_rows_node` AST validation. |
| Nested closures plus [evaluator final closures](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:898) | Retain genuine source `bootstrap(...)[1] == prior['code']`, reference `bootstrap(...) == prior['own_code']`, genuine-export closure equality, and separate evaluator/trainer/evaluation-reference closures. Source bootstrap’s live extractor import/loaded-origin checks must survive. |

The bounded implementation is:

1. Replace the nested bulk scans inside evaluator `exit_rehash` using the existing [single-union implementation](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:965) as the template. Adapt its context references explicitly; do not fabricate an initialized context to call it.
2. Discover actual origin **paths without hashing**, using the genuine module-origin validator. Obtain their expected digests from `qualified_origins(context)` and reject unknown/conflicting paths.
3. Form a conflict-checked union of evaluator, genuine and prior guards plus image/origin authorities. Reuse evaluator `bound_file({}, path, digest)` for each distinct path: it already performs a complete fresh read with bounded buffering and cache advice. Populate returned origin hashes only after successful reads. Preserve the complete receipt inventory.
4. Execute every mapping, AST and closure check in the ledger. Leave admission, four complete reloads, resource checks, receipt validation/publication and service wrapper unchanged.

The existing historical union permits discovering new origins; **do not copy that permission**. This evaluator requires every actual origin to belong to its authenticated qualified union.

One bounded stdlib falsifier should extend the existing test file, adapting [the proven exit-chain checks](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_substrate_adaptation.py:532). Budget **30 seconds and at most 4 MiB of additional synthetic payload**, with no native imports:

- Compare returned origins and complete guard inventories against the current nested traversal on identical synthetic inputs. Count actual streamed bytes: one complete read per non-metadata union file; explicitly allow retained closure/partition/AST rereads. Current v3 must fail the read-count assertion.
- Invoke the new exit twice and require fresh reads both times.
- Independently mutate an image beyond the first two, source checkpoint, view cache, endpoint checkpoint and origin-file tail; preserve size and restore `mtime_ns`. Every mutation must reject, including mutation after discovery but before the union read.
- Reject conflicting authorities, unqualified mapped libraries, invalid dynamic module objects, escaped/aliased/reordered image paths, changed TRAIN mapping, changed ImageRows AST and each closure/import-origin mismatch.
- Inject open, mid-read and cache-advice failures; none may return successful origins or publish a passing receipt. Update the existing tests that currently require the literal exporter-rehash call.

The principal risks are omitting a nested predicate and accidentally borrowing the historical routine’s more permissive origin handling. The ledger and differential falsifier address both. Runtime improvement remains unmeasured; fewer reads do not justify predicting a specific service duration.

After that check passes, freeze **one new evaluator version** and run **one prospective CPU120 gate** with unchanged checkpoints, locks, 8 GiB/no-swap/zero-event requirements and all reload checks. Acceptance requires receipt publication, post-command checks, final footer, normal process exit and successful enclosing service completion within 120 seconds. Preserve CPU v1–v3 failures; quality remains **UNREAD** and the production goal unchanged.

No files were edited or qualification workloads run.
