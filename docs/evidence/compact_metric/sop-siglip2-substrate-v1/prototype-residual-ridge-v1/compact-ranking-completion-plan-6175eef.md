Original consultation 6175eefda8d04e14; actual gpt-6.1-sol / xhigh; terminal exit 0; 782 seconds. Read-only, no native jobs or quality reads. Root retains integration and admission authority.

**GO for one compact-owned scheduling patch combining three changes. KILL the exit-reader-only approach and any claim that this establishes TRAIN300.** Expected saving is **about 50–52 seconds, ESTIMATE**; the current estimated deficit is **53.343 seconds**.

Implement only in [train_siglip2_compact_ranking.py](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_compact_ranking.py) and its existing test file. Keep every external source/reference pin immutable.

1. **Share one fresh reader across the nested exit checks.** Add `exit_admission_adapter(context, api, reader)`. Create the genuine `FlatAdmission()` locally in compact `exit_rehash`, pass it to the existing source origin audit, then compile private copies with exactly these substitutions:

   | Original function | Exact substitution |
   |---|---|
   | Quadratic `exit_rehash` | `context['original'].FlatAdmission()` → `_compact_exit_reader` |
   | Fitter `exit_rehash` | `context['old'].exit_rehash(context['legacy'])` → `_compact_quadratic_exit(context['legacy'])` |
   | Fitter `exit_rehash` | `bound_file({}, path, digest)` → `_compact_exit_reader.bound_file({}, path, digest)` |

   Bind the private quadratic namespace’s `audit_origins` to the existing authenticated `api.audit_origins`. Use the same reader for compact’s own guard loop. Each substitution occurs exactly once; require inverse AST equality. I checked that these transformations compile and reverse exactly.

   Authenticate original full SHA, live code/defaults/globals/classes and closures through the accepted `native_source_api`; authenticate the new private namespaces/functions too. Preserve genuine FIT resolution/cardinality checks, source/export bootstraps, TRAIN mapping, image-row authentication, terminal-reader checks and all closure checks. Never rebind original modules.

2. **Replace bundle pre-admission with authenticated manifest preflight.** In `qualify_bundle`, replace:

   ```python
   bundle, _ = portable.admit_bundle(directory, sha)
   ```

   with the existing authenticated `portable.read_json` call for `bundle.json`. Use that authenticated environment to establish the deny hook. Both sequential calls to the **unchanged public `load_inference`** must still run full `admit_bundle` under that hook, followed by all payload, vision448, processor, reload and oracle checks.

3. **Reuse only the immediately preceding post-run origin read for promotion.** Give `run`’s post-run audit a fresh local `post_run_reader`. Replace the subsequent promotion’s `bound_file` with `post_run_reader.bound_file(context['guards'], p, h)`, then discard that reader before entering exit. The accepted audit already hashes, checks authority, registers and verifies those paths; promotion retains canonical-path, size and conflicting-authority checks.

**Reuse scope:** one uninterrupted admission boundary, with no intervening model work or file mutation. Preserve all five compact origin audits and the nested quadratic collector’s actual `imported_origins` reads. Keep supplemental authority authentication fresh. The final post-exit audit gets another new reader and freshly hashes again. Neither public loader receives an earlier admission cache.

The [accepted timing evidence](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/compact-ranking-mechanics-control-v3/decision.json) supports this budget:

| Change | Saving, ESTIMATE |
|---|---:|
| Repeated fitter/compact union reads | 28–30s |
| Two bundle pre-admissions | ≈16.4s |
| Immediate origin promotion reread | ≈5.9s |
| Combined | **≈50–52s** |

Reader plus bundle preflight alone is insufficient. The absolute sum of targeted phases is **54.160s**, before retaining unique reads and metadata checks. Compact-only inputs include the accepted CPU bundle’s vision and initializer files; TRAIN additionally retains its checkpoint and bundle. Consequently, this patch cannot honestly promise the required margin. The central estimate gives **301.268s before TRAIN-specific additions and mechanics-only deductions**.

**Bounded falsifier:** add one stdlib test using the existing `NativeAdmissionFixture` and tiny bundle fixtures, with ≤16MiB payload and ≤55s runtime. Require:

- One physical union SHA read per shared non-origin fixture path; unchanged stage guard inventories and predicate outcomes.
- Fresh collectors/readers at independent boundaries; same-size, restored-mtime mutation after startup rejected at exit, and native mutation before the final audit rejected there.
- Wrong digest/size, instance-method shadowing, changed private globals/code/closures, unknown origins and missing supplemental membership rejected; original module bindings unchanged.
- Authenticated manifest preflight followed by byte corruption rejected by full `admit_bundle` under the deny hook; forbidden original dependency opens rejected.
- Exact inverse AST correspondence and unchanged public loader/admission ASTs.

Kill the patch if any falsifier fails. Root alone owns subsequent timing, integration and TRAIN admission; the recipe, scientific/parity/resource gates and historical FAIL receipts remain authoritative. No files were edited, and no native jobs or imports were started.

Root decision: implement the exact three-part scheduling correction as one bounded source change. Estimated savings are not native measurements or a TRAIN-fit claim. Fresh qualification remains mandatory; no cap, recipe, scientific threshold, parity, public-loader or historical-verdict changes.
