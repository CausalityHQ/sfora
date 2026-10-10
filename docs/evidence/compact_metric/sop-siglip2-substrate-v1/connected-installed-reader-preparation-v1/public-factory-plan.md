# Public installed-artifact factory plan

Read-only specialist result; implementation and native execution remain unrun. Original consultation `12b3f97b4c28403b`, exit 0, source baseline `9b4c0fca`; documentation/CI-only head `8e146973` preserved. No duplicate consultation was started.

Add one factory on the existing `ConnectedCompactIndex`; keep the integrated loader and five helpers unchanged. Restrict implementation to the bridge, its source-contract tests, a finite inverse record, and the existing CI source-check list.

The checkout is now `8e146973`, two commits beyond the supplied `9b4c0fca`. The specified implementation/evidence files are unchanged between those commits, and bridge/runtime hashes match `integration-verification.json`. Preserve the later documentation/CI commits and existing dirty files.

Proposed public API in [connected_compact_serving.py](/home/rb/worktrees/sfora-positive-causality/src/sfora/connected_compact_serving.py:128):

```python
@classmethod
def from_serving_artifact(
    cls,
    *,
    serving_dir: Path,
    trusted_serving_sha256: str,
    trusted_fragment_sha256: dict[str, str],
    installed_environment: bytes,
    trusted_installed_environment_sha256: str,
    native_library_path: Path,
    expected_native_library_sha256: str,
) -> ConnectedCompactIndex:
    ...
```

Derive gallery path/count from genuine admission. Keep CUDA, dimensions128, batches1..32, k10 and ordinal ties fixed. Do not expose `serving_helpers`, add another gallery authority, or require a package-root export.

**Ordered implementation plan**

1. **Pin the bridge increment and its inverse before changing behavior.** Record the complete current bridge SHA `5da8991b…3303e06`, exact inserted spans and exact replacements with occurrence counts. The new inverse must reconstruct all current bridge bytes, then feed the existing `bridge_factory_inverse` and historical pipeline. Keep `serving-reader-inverse.json` unchanged. Adapt its consumers to apply the new bridge inverse first; do not replace old expectations with new hashes or broadly delete AST definitions.

2. **Create the private executable closure.** Require the canonical installed bridge and exact `ConnectedCompactIndex` class for this factory; subclasses cannot satisfy the loader’s owner check. Validate/copy caller inputs, authenticate the existing literal runtime authority, and read all five helper sources against independently frozen bridge literals **before executing any helper**:

   | Helper, in execution/binding order | Frozen SHA256 |
   |---|---|
   | `connected_gallery_provenance.py` | `0e552eb81c3a200bda4568b43a7955cef8018ca43d83c4b0172fc903debaa1ec` |
   | `connected_serving_artifact.py` | `102ab0ab54eff3a0f1d4fa9526966b94f18e648a2a69ad41020148e71741bd08` |
   | `connected_serving_admission.py` | `c3f608bb98225525629badb07e84567dd198973a7ffa5be161d4f1e600875b0b` |
   | `connected_artifact_identity.py` | `75d6e7a862fcee1b1b324e87b7181e3959e539afb0c240da5200a6dc57cf2156` |
   | `connected_installed_environment.py` | `1ec916430dee8dc08c8f4a9a05f0616ffc4d0b0731444fd6314fc53533d9dd0a` |

   Under `_REGISTRY_LOCK`, create a fresh private package with an empty discovery path and five named children. Register exact ownership before each `exec(compile(authenticated_raw, canonical_path, ..., dont_inherit=True))`. Relative imports must resolve only the already-owned dependencies. Never use `exec_module`, cached public helpers, pyc, or historical source roots.

3. **Finish initialization before freezing snapshots.** Preserve `_guards[0:3]` as runtime, authority and packed source, in that order; append only small executable-source guards. Pass exactly those first three entries to `_bind_runtime`.

   Use the genuine `_snapshot` to authenticate runtime code before binding, then discard that temporary runtime snapshot and snapshot the bound runtime. Snapshot the canonical bridge once: it already records `ConnectedCompactIndex` and its methods, including exactly one `_check_current` row. Snapshot all five helpers and the finalized package namespace. Explicitly check package/spec structure because namespace identity alone does not detect mutation inside a `ModuleSpec`.

   Retain this binding:

   ```python
   (tuple(helper_modules), tuple(helper_guards), self._check_current)
   ```

   Run genuine `_check_current` before passing it to the loader. Do not clear all snapshots after recording bridge/helpers.

4. **Invoke the integrated loader before packing/native imports.** Call [load_serving_inference](/home/rb/worktrees/sfora-positive-causality/src/sfora/connected_inference.py:1865) with the copied independent pins, immutable installed authority and retained binding. Do not call legacy `admit_bundle`.

   Assign the returned endpoint immediately and retain its complete `modules` inventory. Then obtain the admitted gallery fact/count, reread `gallery.bin` against its independent pin, and perform the existing lazy packing/CuTile imports. Authenticate/snapshot shared packing, parse the unchanged wire, recheck the native-library pin, and open the native gallery. Any failure after loader success must release that endpoint.

   Per-call `_guards` must contain executable closure sources, not serialized payloads or the selected wheel inventory. Preserve the loader’s existing uncached full startup/exit checks and every request tensor, role, backend, RNG and numerical predicate.

5. **Give the artifact route independent release state.** Retain the existing legacy branch unchanged. For the artifact route, clone authenticated runtime functions into one independent globals dictionary before native allocation, copying mutable literal globals/defaults without copying module identities.

   On successful load, save a release-only serving context without retaining model/tensor aliases. The nested `exit_check` needs special treatment: merely cloning top-level functions leaves its `__globals__` pointing at the live runtime, so cloned `_serving_exit_callables` rejects it. Reconstruct it from saved code/defaults and independently saved closure cells using the release globals; preserve the retained verifier graph and its mutation checks. Do not rebind live runtime globals.

   In artifact `close()`, pass the complete endpoint inventory into authenticated release. The current outer `endpoint["modules"] = {}` must remain only on the legacy branch: `_serving_release` owns that delegation after its inventory checks. Keep runtime/helpers registered through release and full exit, then remove exact owned identities, children before package. Report replacement entries without deleting them; preserve primary exceptions and secondary notes.

6. **Separate failed-load ownership from successful-endpoint release.** [ `_capture_failure` ](/home/rb/worktrees/sfora-positive-causality/src/sfora/connected_compact_serving.py:473) currently discovers only `load_inference`/`load_authenticated` frames. Add a finite artifact-route branch that recognizes saved authenticated serving API code and clears only finished owned frames.

   The integrated loader already releases partial/returned endpoints on its own failures. Do not recover an emptied traceback-local endpoint and release it again. Before loader success, the bridge owns registry rollback; after success, it owns the returned endpoint and gallery. Never infer ownership from a registry difference.

**Snapshot and lifecycle traps to falsify**

- `_bind_runtime` changes `_binding`; a pre-bind snapshot cannot remain the final expectation.
- Child imports/parent attribute installation change the package namespace. Freeze it after registration; an empty discovery path must remain empty.
- `_snapshot(bridge, ...)` already captures the checker through the class. Saving it again causes the loader’s “one authenticated checker snapshot” check to fail.
- Bridge/helper namespace snapshots reject new keys, replaced aliases and changed literal containers. Keep imports local and avoid post-snapshot global caches.
- Preserve `_guards[:3]`, `_module.__name__`, `__file__`, spec name/origin and genuine `_apis` population. Adding helpers must not shift packed-source indexing.
- The saved release must survive live runtime function/global mutation, while full exit still rejects changed artifact/environment bytes.
- Artifact close must retain helpers outside endpoint `modules`; failed-load cleanup must not double-release.

**Bounded falsifying test**

First add one stdlib-only test using the canonical bridge, genuine snapshots/checker, all five real helpers, and the existing tiny artifact/installed-RECORD fixtures. Run the actual integrated loader with a trace sentinel raising at entry to `_serving_load_payload`, before its first native import. Forbid native imports throughout.

Require: genuine preparation succeeds; the public factory reaches that exact loader boundary once; packing/CuTile remain untouched; the sentinel remains the primary exception; full failure exit runs; exact owned registry entries disappear; a foreign replacement survives; restored fixtures can repeat successfully. This isolates factory reachability without qualifying native execution.

Extend that same test with source-compiled, explicitly authenticated stand-in native seams to cover successful return/close and failures during gallery read, packing, gallery open and release. Do not patch live functions after snapshotting. Include wrong helper bytes, checker mutation, package/spec mutation, imported-alias substitution, release-clone globals correspondence, double close, retained exceptions and finite-inverse smuggling negatives.

Run serially under the existing **120-second / 1-GiB** limits with Python3.14.4 and `-B -S`; then run the affected extraction, integrated-reader and both legacy factory checks. Add the new test to the existing CI source job without increasing caps.

No planning blocker remains. Native execution still depends on the separately authorized interpreter/import/ISA/external-anchor/live-mapping gate; metadata verification cannot supply it. Factory reachability will leave the historical full-model hash cost of approximately **1.377 seconds/call** unchanged.

I made no edits, ran no tests or native imports, and launched no children or duplicate reviews. Native parity, cleanup on real framework traces, quality and speed remain **UNQUALIFIED**.

## Root reconciliation before production writes

Accept the single-factory scope. Keep the runtime, five helpers, numerical body and resource gates unchanged. The checker-global mismatch was independently reproduced by compiling the actual `_serving_exit_functions` and `_serving_exit_callables` definitions with an empty synthetic helper inventory: a live-global checker rejects in cloned release globals; the same nested code and closure bound to those release globals passes. This is a finite source seam observation, not complete helper-graph or native qualification.

The implementation must derive an independent release-only context from authenticated startup state, retain complete endpoint module inventory until `_serving_release` delegates removal, and never adopt a traceback-local endpoint already disposed by the loader. Scope production changes to the bridge. Introduce an exact whole-byte inverse ahead of historical inverses; preserve all old hashes and assertions. First test genuine five-helper preparation and trace-stop before native import, then authenticated stand-in success/failure seams. One affected serial gate under existing 120 seconds / AS 1 GiB; no native launch until the separate installed-environment/ISA/native-mapping gate.

