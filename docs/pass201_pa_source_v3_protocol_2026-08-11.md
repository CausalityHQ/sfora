# Pass201 ordinary-PA source-v3 and CPU-replay protocol — 2026-08-11

## Decision and evidence boundary

Create one fresh ordinary Proxy Anchor seed-0 source checkpoint, then run the
already frozen Pass201 CIS operator diagnostic on CPU. This is a new protocol
and attempt, not a retry or reinterpretation of source-v2. Source-v2 remains
`BLOCKED` with no result because its interpreter did not provide `python -m
pip`.

This protocol changes no dataset, model, objective, recipe, seed, schedule,
Pass201 input selection, operator, metric, bootstrap, threshold, decision, or
authorized next action. It prospectively changes only:

1. package/runtime authority for the new source;
2. output paths and schemas needed to distinguish source-v3 from source-v2;
3. the diagnostic's obsolete source-manifest binding; and
4. the execution device of the diagnostic, from auto-selected CUDA to required
   CPU, so the frozen exact replay gate is executable in the stated
   nondeterministic GPU environment.

No CIS value or generated scientific artifact may be read while implementing
or reviewing these changes.

## Why this measurement is next

Pass204's corrected CIS observation is confounded by the exact `sqrt(m)`
coefficient difference between summed and atomic operators and by the missing
per-image full-union control. The existing Pass201 diagnostic tests those two
explanations at equal update norm on disjoint-image, same-class-set contexts.

CJFS is rejected because its statistic is contributor-gradient variance/RDGC
under a new partition and admits a rank-one Jacobian minimum. EMA relational
secant is rejected because its first-order effect is an EMA momentum change.
Neither receives code or GPU time.

## Frozen source computation

The source remains the official In-Shop ordinary Proxy Anchor recipe frozen by
source-v2: `proxy_anchor`, BN-Inception, seed `0`, batch `180`, 60 epochs, one
proxy per class, no checkpoint-selection split, label noise, class cap, or
recipe override, and the registered final state after 8580 steps.

The reviewed source commit, complete `src/sfora` tree, controller, contract,
diagnostic, dependency files, Python executable, import roots, pretrained
checkpoint, partition, image tree, exact command/environment, and every output
are authenticated by Git blob and worktree bytes.

Source-v3 reuses the historical implementation paths
`scripts/run_pass201_pa_source_v2.py` and
`scripts/pass201_pa_source_v2_contract.py` to avoid copying thousands of lines.
All new behavior is selected by the exact source-v3 schema. Historical
source-v2 fixtures retain the old two-key package schema and must remain
validatable. Source-v2 manifests, checkouts, paths, and absent outputs remain
immutable.

## Exact package/runtime authority

Replace only the unavailable pip interface. Invoke the bound interpreter under
the exact training replacement environment and checkout working directory as:

```text
<absolute-python> -B -c <literal-child-source>
```

`-I` is forbidden because it implies `-E` and would ignore the registered
`PYTHONPATH=<checkout>/src`. The child calls
`importlib.metadata.distributions()` and emits a canonical UTF-8 JSON object
plus one LF with exact top-level key set:

```text
schema_version, python, distributions
```

`schema_version` is `pass201-importlib-environment-v1`. `python` has exact key
set `executable`, `prefix`, `sys_path`; all are built-in strings except
`sys_path`, which is an ordered list of built-in strings. They must exactly
match a separate child that imports the same values immediately before loading
the training entry point under the same environment/cwd.

Each distribution record has exact key set:

```json
{"name":"<Name>","normalized_name":"<PEP-503 name>","version":"<version>"}
```

Values are nonempty built-in strings with no NUL or surrounding whitespace.
Normalize with lowercase `re.sub(r"[-_.]+", "-", name)`. Reject duplicate
normalized names. Sort by UTF-8 bytes of `(normalized_name, name, version)`.
Serialize with `sort_keys=true`, `ensure_ascii=false`, `allow_nan=false`, compact
separators, and one LF, matching the existing controller's canonical encoder.
Serialized object keys are lexicographic; array record order remains semantic.
The parent strictly reparses and reserializes the bytes, then records package
evidence with exact key set:

```text
algorithm, distribution_count, bytes, sha256
```

`algorithm` is `importlib-metadata-v1`; count/bytes are positive built-in ints;
SHA-256 is lowercase 64-hex. The same object is relationally identical in
authority, execution, and receipt.

Immediately before both captures, record one literal RFC3339 UTC
`frozen_absence_checked_utc` after verifying every frozen path absent. Pass that
same literal argument to both processes; neither process may sample its own
clock. Two fresh `freeze-authority` top-level processes at the independently reviewed
source commit must produce byte-identical complete manifest candidates before
the manifest-only authorization commit. Their package inventories are therefore
also byte-identical. Empty/malformed output, path/runtime mismatch, duplicates,
nonzero exit, or byte drift stops before authorization. No package is installed
or changed.

## Exact source-v3 authority/receipt delta

The source-v3 prelaunch manifest keeps the source-v2 fields and nested schemas
byte-semantically unchanged except for the following complete, version-selected
delta. Every object uses exact key sets and canonical `sort_keys=true` bytes:

- top-level key set is `schema_version`, `status`, `purpose`, `protocol`,
  `plan`, `source_commit`, `authorization`, `controller`, `source`, `execution`,
  `dataset`, `outputs`, `sidecars`, `postconditions`;
- `schema_version` is `pass201-pa-source-v3-prelaunch-v1`;
- `protocol` and `plan` each have exact key set `path`, `sha256`, `commit` and
  bind the A3/P3 Git/worktree bytes;
- `authorization.manifest_path` and its sole required diff path are exactly
  `docs/pass201_pa_source_v3_authorization_manifest.json`;
- `execution.python_packages` uses the four-key source-v3 evidence above;
- `source.files` is the source-v2 UTF-8-byte-sorted list with exactly
  `scripts/diagnose_pass201_cis_operator.py` inserted first, immediately before
  `scripts/pass201_pa_source_v2_contract.py`; all prior file paths/hashes retain
  their relative order and V3 bytes;
- `outputs.run_directory` is exactly
  `reports/generated/pass201_source_v3/run-v3`; the six filenames under it stay
  `report.json`, `checkpoint.pt`, `training.log`, `resolved_config.json`,
  `train_manifest.json`, `receipt.json`;
- all other source-v2 key sets, types, literals, and relations are unchanged.

The Git chain is linear `A3→P3→I3→V3→H3`: I3 is the independently reviewed
package-authority implementation, V3 adds only the source-v3 binding/CPU
diagnostic source and tests, and H3 is manifest-only. No merge or unrelated
changed path is allowed; the manifest's `source_commit` is V3.

The complete receipt likewise retains every source-v2 field-set/relation
except `schema_version=pass201-pa-source-v3-receipt-v1`, the four-key package
evidence, source-v3 paths, and two new `protocol`/`plan` objects inside
`authorization`. Historical v2 validators
select only their old schemas; no union or coercion is permitted inside one
schema version.

## GPU source and CPU diagnostic boundary

The one source-training run may use the registered GPU. GPU bitwise
reproducibility is not a premise: one frozen seed/config/input/environment is
executed once and its checkpoint/report are accepted by semantic and provenance
validation, not by repeating training.

The diagnostic is different. Its frozen implementation requires exact equality
of input tensors, gradient/update digests, and the complete context-0 record
across three fresh processes, in addition to tensor/scalar tolerances. The
predicate is not weakened. All diagnostic roles therefore run with
`CUDA_VISIBLE_DEVICES=""` and must select CPU. Process records must report
accelerator `cpu`, `visible_cuda_devices=["cpu"]`, and the exact registered CPU
placeholder for CUDA RNG evidence. `cuda_version` and `cudnn_version` remain the
observed build-time strings (for the bound runtime currently `13.0` and `92000`)
and must agree across all roles; they are not evidence of a visible device.
Any visible GPU, nonfinite value, digest mismatch, or
tolerance failure is `INVALID` with no retry.

Before importing Torch, each role's replacement environment retains
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and additionally sets `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. After Torch import and before
any tensor/model construction, call exactly once
`torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`, then require
both getters equal built-in int `1`. Extend `deterministic_settings` with exact
keys `torch_num_threads` and `torch_num_interop_threads`, both `1`, and require
equality across all three process records. This prospective CPU scheduling
binding changes no mathematical formula and prevents thread-count-dependent
FP32 reduction drift.

CPU execution changes no mathematical object: every operator within a context
uses one shared FP32 train-mode graph and identical input bytes; all registered
FP64 reductions, 32 paired contexts, 20,000 paired bootstrap samples, formulas,
thresholds, and decisions remain unchanged.

Integrity A is the prospective CPU feasibility smoke: it constructs all frozen
input contexts and scores only context 0. Record elapsed time and peak resident
memory. Any timeout, resource exhaustion, or exact replay failure stops before
integrity B/science; it does not authorize a CUDA fallback or altered batch.

## Prospective diagnostic source-binding amendment

The existing diagnostic hard-pins an obsolete path and digest. Replace that
tuple with a Git-handoff predicate, not another compile-time digest. The public
controller accepts only the literal normalized path
`docs/pass201_pa_source_v3_authorization_manifest.json`, requires detached clean
HEAD H3, requires H3 to have sole parent V3 and sole diff `A 100644` at that
path, reads the manifest from both `H3:path` and the worktree, and requires byte
identity plus strict source-v3 validation. The manifest then authenticates V3's
complete source mapping, including the executing diagnostic blob. No literal
H3 or manifest SHA is stored in V3, so there is no cycle.

For source-v3, the public CLI must unconditionally reject a non-null
`--runtime-factory`, the `PASS201_RUNTIME_FACTORY` environment variable, and
every external runtime factory before importing it. The historical digest-based
test seam remains legal only under the historical schema in unit tests; it is
never reachable from a source-v3 public controller.

Before importing Torch or constructing contexts, authenticate:

- this protocol and its plan commit;
- the independently reviewed source commit;
- the direct manifest-only handoff;
- all controller/contract/diagnostic Git blobs and worktree bytes;
- the complete source-v3 receipt/report/checkpoint relations; and
- the historical v1/v2 paths as rejected, never fallback authorities.

The amended diagnostic and tests are part of V3; H3 binds their hashes.
Mutation tests must prove that aliases, old paths,
wrong ancestry/order/type, dirty bytes, and coordinated valid-hash drift fail.

No formula, replay predicate, process count/order, metric, threshold, or
decision may change under this amendment.

## Attempt, publication, and stop rules

Before the training child starts, a structural freeze/preflight defect may be
repaired only through a reviewed source/protocol commit and fresh absent paths.
Once the sole training child starts, any nonzero exit, missing output, drift, or
validation failure consumes source-v3 permanently.

Publication is exclusive: same-directory `xb` temporary, file fsync, hard-link
no-replace, directory fsync, strict reload, and inode-owned cleanup. The source
receipt states `candidate_values_computed=false` and
`authorized_action=source_binding_only`.

Only an authenticated source permits the CPU diagnostic. `PASS` authorizes only
a separate training preregistration. `FAIL` or ordinary `UNRESOLVED` closes CIS
and the RSTA/RDGC/CJFS tangent-field line; no new reference, rescaling, or
replacement penalty may rescue it.
