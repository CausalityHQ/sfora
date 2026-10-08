Add an **explicit probe factory backed by a separately pinned, generated inference runtime**, while retaining the existing MLP factory and sharing its lifecycle implementation. Do not broaden the MLP schema or introduce a configurable model adapter.

The source comparison supports a small, finite extraction: of the 20 trainer definitions currently extracted for serving, **15 are AST-identical** between MLP and probe. Five differ—`parameter_roles`, `apply_overlay`, `construct_encoder`, `encoder_facts`, `admit_bundle`—and probe adds `_probe_decorators`, `_probe_vision_forward`, and `probe_source`. The accepted trainer/test bytes match `e2496033…` / `4accfd6c…`.

I checked the [existing extraction proof](/home/rb/worktrees/sfora-positive-causality/scripts/test_connected_inference_extraction.py:20), original packaged-source qualification, pooling-head source, and its decorator hashes against the original CPU-v5 receipt. These support source correspondence; they do not qualify an installed probe endpoint.

1. **Generate one closed probe runtime and its authority.**

   Add `scripts/extract_connected_probe_inference.py`, using stdlib AST/text processing without importing source modules. Give it a deterministic `--check` mode.

   Produce:

   - `src/sfora/connected_probe_inference.py`
   - `src/sfora/_connected_probe_inference_authority.py`

   Extract the existing `CLOSURE` inventory from `scripts/test_connected_inference_extraction.py`, substituting the accepted probe trainer for the MLP trainer and adding the three probe guards: **41 source definitions**, plus the existing four binding/packing helpers and two hash-worker helpers.

   Preserve these concrete APIs:

   ```python
   _bind_runtime(historical_code, runtime_guards)
   admit_bundle(directory, digest)
   load_inference(directory, bundle_sha256, device)
   inference_outputs(endpoint, images)
   release_inference(endpoint)
   ```

   Preserve output keys `raw`, `unit`, `codes`, `inverse_norms`, `wire`; use the existing probe schemas:

   - `siglip2-connected-probe-bundle-v1`
   - `siglip2-connected-probe-inference-v1`

   Use a separate literal extraction-authority schema, `sfora-connected-probe-inference-extraction-v1`, retaining the current ledger fields.

   The complete allowed correspondence is the existing extraction’s counted substitutions: helper calls become local calls; the unused `source_runtime` branch is removed; trainer self-authentication becomes exact historical-closure authentication; authenticated package execution replaces historical module execution; installed head-method authentication replaces the historical filename predicate; packing uses canonical `sfora.packed_int8`. Add only the probe-specific runtime/authority filenames and the preserved hash-pipeline transformation.

   Record whole-source hashes, source/output symbol byte and AST hashes, exact substitution counts and diffs. Reject unexpected definitions, statements, imports or dependencies. The independent test must reconstruct correspondence without trusting the generator’s own comparison.

   Carry the [probe source guards](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_probe.py:589) intact: genuine classes, constructors and forwards; actual wrapper closure links; globals, defaults and flags; uncached source reads. Keep the exact absolute `head.probe` overlay, shape `[1,1,1152]`, complete 448-parameter inventory and frozen-447 complement.

   Copy `_sha_cpu_bytes` and `_fingerprint_cuda_dict` unchanged. Preserve calling-thread tensor snapshots, CPU-only workers, four-task/96-MiB bounds, typed framing, fresh duplicate reads, ordered failures and joined cleanup. Only the two existing large-dictionary call sites use that helper; `fingerprint`, buffers, readout and individual probe hashing stay unchanged.

2. **Add one explicit public binding to the existing bridge.**

   In `src/sfora/connected_compact_serving.py`, add:

   ```python
   ConnectedCompactIndex.from_probe_bundle(
       *,
       bundle_dir,
       expected_bundle_sha256,
       gallery_path,
       expected_gallery_sha256,
       gallery_count,
       native_library_path,
       expected_native_library_sha256,
   )
   ```

   Return the same `ConnectedCompactIndex`. Keep `from_bundle(...)` explicitly MLP-only.

   Move the common factory body into one private implementation with exactly two fixed internal choices. Each choice fixes its schema, nine-file closure, runtime filename, authority filename and authority-schema literal. Add `_installed_probe_authority()` with a literal digest; retain `_installed_authority()` and the MLP authority/runtime bytes unchanged. Neither bundle contents nor caller arguments may supply executable paths or an authority implementation.

   Share `search_images`, snapshot checks, locks and cleanup unchanged. Preserve CUDA device 0, dimensions 128, 130-byte rows, batches 1–32, exact native top-10 and ordinal ties. Preserve saved release globals, full registry checks, foreign-entry preservation, failure-frame cleanup, processor-cache teardown and weak-reference lifetime checks.

   **Proof trap:** `pipeline_contract_check()` pins the historical whole bridge. Add an exact finite inverse for this factory refactor before its existing inverse/assertion; do not replace the old expected hash or weaken the assertion.

3. **Keep export production unchanged; bind deployment to actual artifacts.**

   The accepted probe trainer already provides `inference_members`, `export_bundle`, copied public loading and independent bundle qualification. No trainer or evaluator edit is needed for this packaging boundary.

   The probe authority must authenticate the actual nine-file vector:

   ```text
   train_siglip2_connected_probe.py
   test_siglip2_connected_probe.py
   qualify_siglip2_substrate_cpu.py
   extract_siglip2_vision_source.py
   train_siglip2_cached_readout.py
   train_siglip2_substrate_adaptation.py
   prototype_residual_readout.py
   quadratic_readout.py
   joint_relational_compaction.py
   ```

   Source work can proceed from accepted source-v2. Deployment must obtain these facts from the **future accepted actual bundle/export**, with independent provenance:

   - `bundle.json` SHA; exact nine-file code vector; `vision.pt`, `endpoint.pt`, `processor.json` SHAs.
   - Endpoint-state/fixed fingerprints, full vision identity, probe overlay, frozen complement, encoder runtime/buffer identity and numerical flags.
   - Existing scope, processor/backend/model/decorator sources, package roots, constructor and native-file inventory.
   - Matching gallery bytes, count, ordered row mapping, image/batch witnesses and exporter receipt tying that gallery to the chosen endpoint.
   - Qualified native binary/build provenance and installed wheel/runtime/authority/packing identities.

   Derive new runtime and authority pins from reviewed generated bytes. Do not invent future hashes, establish trust by hashing arbitrary current files, or relabel a mechanics artifact as a deployment candidate.

   Retain the nine historical `.py` files as authenticated evidence. Installed inference must never execute them. Existing absolute environment/source paths remain required; relocation is outside this change.

4. **Use this exact implementation inventory.**

   | Action | Files |
   |---|---|
   | Add generator/runtime/authority | `scripts/extract_connected_probe_inference.py`; `src/sfora/connected_probe_inference.py`; `src/sfora/_connected_probe_inference_authority.py` |
   | Modify shared bridge | `src/sfora/connected_compact_serving.py` |
   | Add extraction falsifiers | `scripts/test_connected_probe_inference_extraction.py` |
   | Extend bridge and historical-proof tests | `scripts/test_connected_compact_serving.py`; `scripts/test_connected_inference_extraction.py` |
   | Add bounded native gate and its source tests | `scripts/qualify_connected_probe_serving.py`; `scripts/test_connected_probe_serving.py` |
   | Document API/status | `README.md`; prospective `connected-probe-library-source-v1/` and `connected-probe-library-native-v1/` evidence directories under the existing substrate evidence root |

   No changes to `packed_int8.py`, `cutile_int8.py`, Rust/CUDA/CuTile, trainer/evaluator sources, or old authority/runtime files. Hatch already includes `src/sfora`; no dependency, build-config or top-level export change is needed.

   Existing observer/request/control drivers are MLP-bound. Leave them intact and run their affected regression tests against the shared bridge. The new native gate needs parity and admission, not their timing instrumentation.

5. **First falsify the binding on CPU without native imports.**

   Use one serial source gate, capped at **120 seconds and 1 GiB address space**, with third-party imports forbidden.

   The smallest decisive test constructs correctly hashed synthetic evidence for each binding and verifies that:

   - MLP factory rejects probe closure; probe factory rejects MLP or mixed closure, even when the supplied manifest digest is correct.
   - Removing a probe guard, accepting four MLP overlay names, or excluding an MLP leaf from frozen-447 checking fails correspondence or an executed stand-in check.
   - Wrong overlay name/shape/dtype, forged decorator closure, live callable/global/default mutation, and unchanged-version byte mutation reject.
   - Missing, changed, linked or substituted evidence/runtime/authority/packing files reject before native import; fresh-read and hash-before-use checks remain.
   - Genuine cleanup handles cancellation, pre-return failure, foreign registry replacements, cache failures and retained references; repeated close is safe.
   - B0/B33 reject; the old MLP proof still restores its original bytes and passes.
   - Existing hash-worker falsifiers still establish fresh snapshots, exact serial digests, bounded overlap and joined failure cleanup.

   A deliberate guard-removal mutant must fail. A green test that merely regenerates and accepts its own ledger is insufficient.

6. **Roll out only after a fresh installed public-parity gate.**

   The root owns this later gate, after collecting the existing worker’s terminal result and obtaining the resource slot. Preserve candidate061 and all original receipts.

   Install the wheel non-editably outside the checkout, retaining authenticated dependency roots. Permit **one live encoder owner at a time**: obtain independent reference outputs from the accepted copied loader, release it completely, then create the installed probe index. Repeat ownership sequentially for reload/lifecycle checks and the MLP regression.

   Require:

   - Identical image membership/order and preprocessing at B1, B2, B32 and an actual export tail. Do not demand cross-batch floating-point equality.
   - Exact raw/unit values, codes, inverse-norm bits and wire; genuine public `search_images` top-10 ordinals and score bits; ties, repeated calls and gallery ownership.
   - Probe, frozen-leaf, readout, source, callable, processor, flag and RNG mutation rejection.
   - Complete constructor/load/readout/packing math checks, dependency-denial checks, release/cache/registry/mmap lifetime checks, double-close and post-close rejection.
   - Both locks, complete native provenance, fresh uncached exit rehash, original normal terminal and resource evidence.

   A prospective envelope can retain the existing diagnostic ceilings: **1,500 seconds whole unit, 300-second body, 300-second exit reserve, 8 GiB host, zero swap/events, CUDA allocation below 10 GB**. Freeze the exact gate before launch; failure closes that attempt without automatic extension. Runtime, implementation effort and cost are **unmeasured**; these are ceilings, not forecasts.

   Source acceptance permits a packaging preview. Deployment requires the chosen artifact’s existing acceptance gates plus this installed parity pass. CPU/control mechanics and source-reviewed evaluator evidence do not establish candidate quality or a speed improvement.

The main risks are losing probe/decorator authority during extraction, accidentally retaining frozen-444 logic, weakening historical proof to accommodate the bridge refactor, and pairing a valid gallery with the wrong endpoint. The currently dirty Rust file must not silently enter native release provenance.

No files edited, tests/builds launched, third-party imports, SSH/GPU work, jobs or children used; live candidate work was untouched.  
Installed probe compatibility, native parity, release eligibility, runtime and cost remain unmeasured.
