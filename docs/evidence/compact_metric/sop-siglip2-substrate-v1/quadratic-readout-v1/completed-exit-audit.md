# Exit lifecycle audit

Original consultation eacb1fbc4814438b; GPT-6 Astra XHigh; exit 0; 671 seconds. CPUv1 remains engineering NO-GO. Future runtime savings are unmeasured.

**Yes: one trainer-owned, EXIT-ONLY composition is feasible using the pinned `FlatAdmission`.** The unchanged nested `rehash` functions cannot share a reader; preserve their predicates through explicit composition, without rebinding helpers or copying functions.

I verified the local closure hashes against the frozen manifests, including trainer `ea141763…5fc28` and `FlatAdmission` source `a1684917…b543`.

The original [exit callgraph](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:393) is:

- `require_no_model` → `audit_origins`: genuine `imported_origins` hashes actual origins; quadratic `bound_file` hashes them again.
- Genuine exporter `rehash` → FIT exporter `rehash` → `all_fit_images` → qualifier `rehash`.
- Those enforce FIT metadata, canonical containment, 13,283 distinct resolved paths, complete ordered-image equality, first-two equality, prior guards and source/export closures.
- Genuine exit then checks its guards, closure, freshly authenticated partition, exact TRAIN mapping and ImageRows AST.
- Quadratic exit finally hashes every own guard and checks its closure.

Metadata supports this **lower bound for a completed traversal**, excluding additional origin passes and small closure rereads:

| Repeated traversal | Redundant SHA requests |
|---|---:|
| 13,283 images: three passes instead of one | 26,566 |
| Other baseline prior guards: nested plus outer | 1,787 |
| Genuine guards: nested plus outer | 14 |
| **Minimum** | **28,367** |

The reconstructed prior inventory matches the saved 15,070-entry startup inventory; its guards and all 14 genuine guards occur in the authenticated warm receipt. Matching size/digest facts establish **≥2,029,648,993 duplicate bytes**: derived vision file 1,711,601,328; nine observed native files 259,479,985; two cache payloads ≥58,567,680. This excludes image bytes and checkpoint bytes. It establishes neither completed reads during the timeout nor timing savings.

The single intervention should touch only the quadratic trainer and its existing stdlib test:

1. Construct a **new empty** pinned `FlatAdmission` inside `exit_rehash`. Retain genuine `source.imported_origins` unchanged. Preserve `audit_origins`’ conflict checks and exact CPU/warm **module and file** membership checks. Register origins as verified only after their genuinely computed, fresh **same-exit** hashes match authority; this replaces their second read. Never seed from startup admission, expected hashes alone, stat, mtime or native caches.
2. Call pinned [`FlatAdmission.all_fit_images`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:433), compare its result with `prior['all_images']`, and retain genuine `fit_rows(...) == prior['images']`. Its metadata/path predicates are AST-identical to the original before the hashing loop.
3. Pass every prior, genuine and quadratic guard through that same reader, rejecting conflicts. Retain genuine source bootstrap—including live extractor-origin checks—FIT bootstrap, genuine closure, fresh partition/`selected_manifest`, resolved TRAIN equality, ImageRows AST and quadratic closure. Keep their small metadata rereads.
4. Leave complete source448/config/processor/nonpersistent-buffer composition, typed payload fingerprints, use-boundary byte checks, reloads, RNG, ownership, resource checks and terminal admission unchanged.

**Bounded falsifier:** one stdlib differential fixture, ≤30 seconds and ≤4 MiB synthetic payload. Compare the original traversal and proposed composition’s predicates, inventories and origin results. Count actual streamed bytes: each unique bulk file exactly once per successful exit, and freshly again on a second invocation. Independently mutate bulk-file tails while preserving size/mtime; exercise conflicting guards, unknown/conflicting module and file origins, escaped/aliased/reordered images, first-two and TRAIN mapping changes, partition/AST/closure/import-origin failures, and read failures. Every original rejection must remain a rejection. Pin unchanged typed-state/use-boundary code by AST comparison; stdlib fixtures cannot certify native behavior.

The principal risks are incorrectly promoting unverified origins and omitting a nested predicate. Failure of this falsifier blocks admission. **CPUv1 remains engineering NO-GO; resource fit and future execution remain unmeasured.** No files were edited or native workloads run.
