# Pass 200 RSTA immutable scientific-artifact roundtrip recovery amendment — 2026-08-10

## Status, outcome blindness, and scientific boundary

This is a prospective recovery and validation amendment for one already-written,
immutable Pass 200 RSTA scientific artifact. It repairs a JSON representation
contract and authorizes exactly one offline validation attempt. It does not
authorize a candidate-free audit, a scientific rerun, a model or dataset load,
GPU execution, field construction, receiver scoring, aggregation production,
bootstrap production, decision production, artifact rewrite, or a second
validation attempt.

The immutable artifact is bound only by the following disclosed metadata:

```text
path: reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json
sha256: e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
producer_pid: 1002393
producer_exit_code: 0
producer_handoff_commit H: c04574e2bb751c3229bce673408577cfedc00a88
producer_source_commit S: 15234a529a181c39c1c8b6477ad7eb7823fd0798
producer_manifest_path: docs/pass200_rsta_receipt_stage_a_manifest.json
producer_manifest_sha256: 9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe
producer_diagnostic_path: scripts/diagnose_pass200_rsta_stage_a.py
producer_diagnostic_sha256: 85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e
```

During defect triage, only the top-level header word `UNRESOLVED` was
inadvertently seen. No candidate value, receiver row, metric, aggregate,
bootstrap value, decisive-clause value, or other scientific value was opened,
read, copied, compared, interpreted, or used to choose this recovery. The
header observation is disclosed for chronology only. It is not validation
evidence and is never copied into the validation receipt.

The artifact remains opaque to amendment and plan authors and reviewers. Only
the isolated validator process may parse its scientific content, and that
process may emit only the exact validation status and provenance registered
below. A `VALID` receipt establishes byte-exact roundtrip validity under this
prospective recovery; it does not create or change a scientific outcome. An
`INVALID` receipt, any structural failure, or any interrupted attempt stops the
workflow permanently with no rerun under this amendment.

## Exact chronology and defect disclosure

The chronology is exact:

1. The reviewed sign-control source was committed at
   `15234a529a181c39c1c8b6477ad7eb7823fd0798`.
2. The manifest-only handoff was committed at
   `c04574e2bb751c3229bce673408577cfedc00a88`; its sole changed path was
   `docs/pass200_rsta_receipt_stage_a_manifest.json`, and its parent was the
   source commit above.
3. The handoff manifest bytes had SHA-256
   `9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe`.
4. One scientific producer process, PID `1002393`, exited `0` and published the
   immutable artifact at the exact path above with SHA-256
   `e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae`.
5. No second scientific command was run. The artifact was not modified,
   replaced, normalized, migrated, or reserialized.
6. Read-only schema triage found a JSON roundtrip defect: the live producer
   validated `panel_binding.primary.support_ids_by_label` while its keys were
   Python integers. JSON object serialization necessarily converted those keys
   to strings, so reloaded bytes could not be submitted unchanged to the same
   relational validator.
7. A local test-first diagnosis produced an uncommitted canonical-key producer
   repair and tests in the existing producer/test files. Those uncommitted
   changes produced no artifact, candidate computation, scientific execution,
   or result interpretation and have no authority until incorporated into the
   reviewed source commit `V` defined below.
8. This amendment and its implementation plan are committed and independently
   reviewed before any verifier source is committed, before the manifest is
   refrozen, and before the one offline validation attempt.

The defect is representational. It does not itself establish whether the
immutable artifact is valid or invalid and does not authorize inferring either
status from the disclosed header.

## Canonical live-producer JSON contract

The live producer's registered scientific computations, inputs, actions,
fields, rows, metrics, aggregates, bootstraps, decision rules, thresholds,
predicates, ordering, and atomic writer remain unchanged. Only the in-memory
representation of one mapping at the scientific-payload boundary changes.

Immediately before `scientific_payload` validates `panel_binding`, the producer
must construct
`panel_binding.primary.support_ids_by_label` as an insertion-ordered Python
`dict` whose keys are the exact canonical decimal strings derived from
`panel_binding.primary.eligible_labels` in order:

```text
expected_keys = [str(label) for label in eligible_labels]
observed_keys = list(support_ids_by_label)
observed_keys == expected_keys
```

The exact canonical schema is:

- `eligible_labels` is a JSON/Python list;
- every label satisfies `type(label) is int` and `label >= 0`;
- labels are unique, so the derived strings are unique;
- `support_ids_by_label` is a concrete insertion-ordered `dict`;
- its key sequence is exactly `[str(label) for label in eligible_labels]`;
- it has no missing, extra, duplicated, reordered, integer, boolean, signed,
  whitespace-padded, zero-padded, exponent, or other alias key; and
- each support value and its order remain unchanged.

`str(label)` is normative. Thus `"0"` is canonical for integer `0`; `"00"`,
`"+0"`, `"-0"`, `"0.0"`, and an integer or boolean key are invalid. Exact key
order is semantic evidence and validators must compare `list(mapping)`, not a
set of keys.

The producer's `_validate_registered_rows` indexes the canonical mapping with
`str(row["label"])`. It validates the canonical key sequence before any row
lookup. The implementation test must exercise the live producer's real
`scientific_payload`, strict loader, and writer semantics. Given one complete
synthetic valid argument set, it computes `first`, then exactly:

```text
first_bytes =
  UTF-8(json.dumps(first, indent=2, sort_keys=False, allow_nan=False) + "\\n")
persisted = strict JSON load(first_bytes)
second = scientific_payload(the complete ten components from persisted)
exact_ordered_equal(second, persisted) is true
UTF-8(json.dumps(second, indent=2, sort_keys=False, allow_nan=False) + "\\n")
  == first_bytes
```

`exact_ordered_equal` here is the verifier's concrete-JSON recursive comparator:
it requires identical concrete types, exact mapping key order, exact list
order, and IEEE-754 byte equality for finite floats, including distinguishing
`0.0` from `-0.0`. The test carries a synthetic `-0.0` through a live-payload
field accepted and preserved by the producer, proves a post-hoc `+0.0` mutant
is rejected, and never uses ordinary `dict ==` as its validation predicate.
All prior live relational checks remain in force. This repair does not loosen
a scientific validator and does not authorize re-running the live producer.

## Separate offline legacy validator

Create one standalone verifier at:

```text
scripts/verify_pass200_rsta_scientific_artifact.py
```

Its tests are confined to:

```text
tests/test_verify_pass200_rsta_scientific_artifact.py
```

The verifier is not imported or called by the live producer. It loads no
dataset, checkpoint, image, embedding, or model; constructs no candidate
field, receiver row, score, aggregate, bootstrap distribution, or verdict; and
never imports a current scientific function as a substitute for the legacy
producer. Its sole scientific-content operation is isolated validation of the
already-persisted bytes through the authenticated legacy producer function.

### Raw artifact gate

Before parsing scientific content, the verifier must:

1. require the exact repository-relative artifact path registered above;
2. reject a symlink, non-regular file, path escape, missing file, or changed
   inode between open and read;
3. open once read-only with no-follow semantics;
4. read the bytes from that one file descriptor without printing them;
5. require exact SHA-256
   `e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae`;
6. strictly parse JSON with duplicate-object-key and nonfinite-constant
   rejection; and
7. retain both the original bytes and a never-mutated raw object.

No code may rewrite, rename, chmod, touch, normalize, re-indent, or replace the
artifact. A temporary byte-identical process input is unnecessary and is not
permitted; the child receives the already-open read-only descriptor.

### Exact legacy provenance

The verifier authenticates all of the following before legacy validation:

- `H` resolves to exact commit
  `c04574e2bb751c3229bce673408577cfedc00a88`;
- `S` resolves to exact commit
  `15234a529a181c39c1c8b6477ad7eb7823fd0798`;
- `H` has exactly one parent and that parent is `S`;
- the `H` commit changes only
  `docs/pass200_rsta_receipt_stage_a_manifest.json`;
- the manifest Git blob at `H` has exact SHA-256
  `9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe`;
- the manifest's `current_scientific_source.git_revision` is exactly `S`;
- its exact prior 31-path source mapping/order is the one frozen below;
- every listed digest equals both the `S:path` Git blob and the `H:path` Git
  blob, proving that the manifest-only handoff did not change source;
- the legacy diagnostic blobs at `S` and `H` are identical and have exact
  SHA-256
  `85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e`;
  and
- the artifact's persisted `manifest` object is recursively type- and
  order-identical to the exact ten-key old projection frozen below.

### Exact old persisted manifest projection

The old persisted scientific `manifest` object has exactly these ten keys in
this order:

```text
path
sha256
base_preregistration
amendment
deterministic_pool_amendment
zero_jacobian_classifier_amendment
binding_receipt
historical
artifact_schema
source
```

The verifier independently derives that projection from the authenticated
manifest Git blob at `H`, never from current source and never by selecting keys
present in the artifact. `path` is exactly
`docs/pass200_rsta_receipt_stage_a_manifest.json`; `sha256` is exactly
`9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe`;
the next seven named values are the recursively exact values of the same-named
keys in the H manifest; and `source` is the recursively exact value of H's
`current_scientific_source`.

Although the H manifest itself also contains
`adjoint_integrity_amendment`, `normwise_adjoint_calibration_protocol`,
`normwise_adjoint_calibration_result`, `normwise_adjoint_amendment`, and
`normwise_adjoint_sign_control_amendment`, producer `S` constructed its
scientific projection with the explicit ten-key literal above and therefore
omitted those five later authorities. H's top-level `schema_version` and
`seeds` are likewise not part of the persisted scientific projection. Adding
an omitted key, dropping a registered key, changing order or type, or deriving
the projection through the current producer is invalid.

### Isolated old-H execution

The authoritative parent verifier runs only from a new clean detached checkout
at the future handoff commit `HV`. It first authenticates its own `V`/`HV`
provenance as specified below. It then creates a fresh temporary directory and
a local no-hardlink clone of the authenticated repository, checks out exact
`H` detached, and proves the temporary checkout is clean and at `H`.

The parent invokes its already-authenticated verifier bytes by absolute path
with the pinned repository `.venv/bin/python`, flags `-I -B`, closed stdin,
captured bounded stdout/stderr, a new process session, an exact timeout, and an
environment that sets `CUDA_VISIBLE_DEVICES` to the empty string. The child
working directory is the temporary old-`H` checkout. No network access, shell,
user-site import, `PYTHONPATH`, GPU, or inherited open file is permitted except
the single read-only artifact descriptor explicitly passed by the parent.

The timeout is exactly `600` seconds. `close_fds=True` is required and
`pass_fds` contains exactly the one artifact descriptor. Captured stdout and
stderr are each bounded to `64` bytes; exceeding either bound is structural.

Before the artifact is opened, the parent derives the registered interpreter
as the exact absolute live-repository path `.venv/bin/python`, requires
`sys.executable` to equal that exact path, and requires both paths to resolve to
the same existing regular executable. The observed runtime must satisfy
`sys.version_info[:3] == (3, 12, 3)` and its canonical recorded value is
`"3.12.3"`. The child is launched with that authenticated
`sys.executable`, independently repeats the exact path/resolution/version
checks against the separately supplied live repository, and rejects an
interpreter argument, executable path, or runtime-version drift.

The child reauthenticates:

- its executing verifier `__file__` is the exact absolute path authenticated by
  the parent and its bytes equal the `V` Git blob and future manifest digest;
- the live repository supplied separately by the parent has `HEAD == HV` and
  `HV^ == V`;
- its current working repository has `HEAD == H` and a clean status;
- the old manifest `__file__` path is the old-H checkout path and has the old
  manifest digest;
- the imported legacy diagnostic `__file__` is exactly
  `<old-H checkout>/scripts/diagnose_pass200_rsta_stage_a.py`;
- that file's worktree bytes and the `S` and `H` Git blobs have the registered
  diagnostic digest; and
- the loaded callable is exactly the legacy module's `scientific_payload`.

After strict parsing and before adapter construction or any call to
`scientific_payload`, the child requires the raw artifact's `environment` to
be a concrete ordered `dict`, its `numpy_version` to be a concrete nonempty
`str`, the imported old diagnostic's `np` object to be exactly the child
runtime module at `sys.modules["numpy"]`, and
`str(legacy_module.np.__version__)` to equal that persisted
`environment.numpy_version`. The parent records its own observed
`str(numpy.__version__)`; the child also requires that parent value to equal
both its module value and the persisted value. Any wrong NumPy module identity,
missing or mistyped persisted version, or version drift is artifact-invalid
when only the persisted field differs and structural when the authenticated
runtime or module differs. These gates precede all legacy recomputation.

The child suppresses scientific values and exception representations. Its
complete permitted process outcomes are exactly:

```text
stdout = "RSTA_LEGACY_VALID\n", stderr = "", exit = 0
stdout = "RSTA_LEGACY_INVALID\n", stderr = "", exit = 1
stdout = "RSTA_LEGACY_STRUCTURAL\n", stderr = "", exit = 2
```

The first is valid evidence, the second is artifact-invalid evidence, and the
third is structural failure that publishes no receipt. Any other stdout,
stderr, exit, timeout, signal, import path, or file identity is structural
failure.

## Exact in-memory legacy adapter and recomputation

The raw artifact schema must already use canonical string keys. The adapter is
not a migration and never changes the raw object or bytes. It makes one deep
copy and performs exactly one permitted transformation in that copy:

```text
path:
  panel_binding.primary.support_ids_by_label

before:
  {str(label): support_value, ...}

after, in the legacy-call copy only:
  {label: support_value, ...}
```

Keys are paired positionally with `eligible_labels`. Each raw key must equal
`str(label)` before conversion; parsing an arbitrary string with `int(key)` is
forbidden. Values are reused without mutation. A recursive mutation ledger must
prove that no other key, value, type, order, or path changed.

The old-H child passes the adapted components to old
`scientific_payload` in this exact argument mapping and order:

```text
manifest_audit   = adapted["manifest"]
execution_audit  = adapted["execution_audit"]
environment      = adapted["environment"]
seed_audits      = adapted["seed_audits"]
primary_rows     = adapted["rows"]["primary"]
alternate_rows   = adapted["rows"]["alternate"]
integrity        = adapted["integrity"]
aggregation      = adapted["aggregation"]
bootstrap        = adapted["bootstrap"]
panel_binding    = adapted["panel_binding"]
```

This invokes the complete legacy relational validation, including execution
binding, integrity, registered receiver rows, aggregation recomputation,
bootstrap recomputation and hashes, and top-level scientific payload
reconstruction. No isolated subvalidator or selected-field shortcut is valid.

The returned recomputed payload must then satisfy both gates below:

1. **Recursive exact equality:** after the legacy producer's existing
   `_json_ready` inverse-canonicalizes the adapted integer keys, the recomputed
   payload and the never-mutated raw JSON object have identical concrete types,
   mapping key sequences, list sequences, scalar types, and scalar values at
   every path. Python `dict ==`, set comparison, coercion, tolerance, and
   selected-field comparison are insufficient.
2. **Byte-exact producer serialization:** serialize the recomputed payload with
   exactly
   `json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\\n"`,
   UTF-8 encode it, and require byte equality with the original immutable
   artifact bytes. A hash-only comparison is insufficient at this final gate.

Any raw-schema, adapter-ledger, legacy-validation, equality, or byte gate
failure returns only the fixed invalid token. It may not expose the differing
path or either value outside the child.

## New verifier provenance: V and HV

`V` is defined, not left as a placeholder: it is the full lowercase 40-hex
commit obtained after the exact four-file source/test implementation has passed
all tests and independent full-source review. Only these paths may differ from
the parent of `V`:

```text
scripts/diagnose_pass200_rsta_stage_a.py
scripts/verify_pass200_rsta_scientific_artifact.py
tests/test_diagnose_pass200_rsta_stage_a.py
tests/test_verify_pass200_rsta_scientific_artifact.py
```

`HV` is the full lowercase 40-hex commit obtained later from a manifest-only
refreeze whose sole changed path is:

```text
docs/pass200_rsta_receipt_stage_a_manifest.json
```

`HV` has exactly one parent, `V`. This sequencing avoids a self-cycle: source
and verifier bytes are frozen and reviewed in `V`; the manifest in `HV` binds
`V`; the validation receipt is created only after `HV` exists and therefore is
not a source or manifest input.

Before opening the artifact, both parent and child independently prove:

- live checkout `HEAD == HV` and `HV^ == V`;
- the checkout is detached, clean, and contains no pre-existing receipt path;
- the future manifest worktree bytes equal its `HV` Git blob;
- its `current_scientific_source.git_revision == V`;
- every one of its exact 32 source paths has a worktree digest equal to the
  manifest digest and the `V:path` Git blob digest;
- the verifier `__file__` equals the repository verifier path and its bytes
  equal the manifest digest and `V` Git blob; and
- the recovery-amendment authority bytes, Git blob, SHA-256, commit, order, and
  ancestry are exact.

## Future manifest authority and exact source order

Add one future manifest authority named exactly:

```text
scientific_artifact_roundtrip_recovery_amendment
```

Its value has exact nested order `path`, `sha256`, `commit` and binds this
amendment's final reviewed path, bytes, and commit. Insert it immediately after
`normwise_adjoint_sign_control_amendment` and before `binding_receipt`.

The future manifest top-level key order is exactly:

```text
schema_version
base_preregistration
amendment
deterministic_pool_amendment
zero_jacobian_classifier_amendment
adjoint_integrity_amendment
normwise_adjoint_calibration_protocol
normwise_adjoint_calibration_result
normwise_adjoint_amendment
normwise_adjoint_sign_control_amendment
scientific_artifact_roundtrip_recovery_amendment
binding_receipt
historical
current_scientific_source
artifact_schema
seeds
```

The candidate-free projected manifest schema is updated for authority
consistency even though this amendment forbids running it. Its exact order is:

```text
path
sha256
base_preregistration
amendment
deterministic_pool_amendment
zero_jacobian_classifier_amendment
adjoint_integrity_amendment
normwise_adjoint_calibration_protocol
normwise_adjoint_calibration_result
normwise_adjoint_amendment
normwise_adjoint_sign_control_amendment
scientific_artifact_roundtrip_recovery_amendment
binding_receipt
historical
artifact_schema
source
```

The previous 31 source paths retain their relative order. Insert the verifier
as the exact fourth path, immediately after `scripts/rsta_normwise_adjoint.py`
and before `src/sfora/__init__.py`, producing this exact 32-path order:

```text
scripts/diagnose_pass159_cotangent_stage_a.py
scripts/diagnose_pass200_rsta_stage_a.py
scripts/rsta_normwise_adjoint.py
scripts/verify_pass200_rsta_scientific_artifact.py
src/sfora/__init__.py
src/sfora/ablation.py
src/sfora/api.py
src/sfora/arcg.py
src/sfora/benchmark.py
src/sfora/bn_inception.py
src/sfora/catalog.py
src/sfora/cea.py
src/sfora/cem.py
src/sfora/cli.py
src/sfora/compose.py
src/sfora/data.py
src/sfora/encoder_ablation.py
src/sfora/encoder_training.py
src/sfora/evaluation.py
src/sfora/experiments.py
src/sfora/image_benchmark.py
src/sfora/image_end_to_end.py
src/sfora/image_recipes.py
src/sfora/ipsr.py
src/sfora/losses.py
src/sfora/method.py
src/sfora/oapf.py
src/sfora/publication.py
src/sfora/remote.py
src/sfora/report.py
src/sfora/text_baselines.py
src/sfora/training.py
```

At `HV`, refresh only `current_scientific_source.git_revision`, the changed
producer digest, and the new verifier digest. Every unchanged path retains its
old digest. Tests remain excluded. Apart from the one authority insertion and
the exact source revision/path/digest transition, all prior manifest domains
remain byte-semantically identical. The old manifest remains immutable as the
Git blob at `H` and is never rewritten in history.

## Exact CLI and one-attempt rule

The public verifier CLI accepts exactly:

```text
--manifest docs/pass200_rsta_receipt_stage_a_manifest.json
--artifact reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json
--output receipt_path(HV)
--validate-immutable-artifact-once
```

where `receipt_path` is the exact function:

```text
receipt_path(commit) =
  reports/generated/pass200_rsta_receipt/
  + lowercase_full_40_hex(commit)
  + -scientific-artifact-roundtrip-validation.json
```

No alternative path, stdin artifact, glob, symlink, directory scan, overwrite
flag, retry flag, scientific flag, or candidate-free flag is accepted. The
output parent must already exist. Output and its process-specific temporary
path must both be absent and not symlinks before the artifact is opened.

After authenticated preflight, opening the artifact begins attempt `1`. Any
status, exception, signal, timeout, publication failure, or operator
interruption consumes the one attempt. Neither deletion of a failed receipt nor
absence of a receipt after interruption authorizes another attempt. The CLI is
never invoked a second time under this amendment.

## Exact atomic validation receipt

The receipt contains exactly these top-level keys in order:

```text
schema_version
validation
mode
attempt
status
outcome_disclosed
artifact
legacy_provenance
verifier_provenance
process
```

Their exact fixed values and schemas are:

```text
schema_version = 1
validation = "pass200-rsta-scientific-artifact-roundtrip"
mode = "offline_immutable_artifact"
attempt = 1
status = exactly "VALID" or "INVALID"
outcome_disclosed = false

artifact keys, in order:
  path = exact registered artifact path
  sha256 = e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
  producer_pid = 1002393
  producer_exit_code = 0
  immutable = true

legacy_provenance keys, in order:
  handoff_commit = H
  source_commit = S
  manifest_path = docs/pass200_rsta_receipt_stage_a_manifest.json
  manifest_sha256 = 9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe
  diagnostic_path = scripts/diagnose_pass200_rsta_stage_a.py
  diagnostic_sha256 = 85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e

verifier_provenance keys, in order:
  source_commit = V
  handoff_commit = HV
  manifest_path = docs/pass200_rsta_receipt_stage_a_manifest.json
  manifest_sha256 = exact SHA-256 of the HV manifest Git blob/worktree bytes
  verifier_path = scripts/verify_pass200_rsta_scientific_artifact.py
  verifier_sha256 = exact SHA-256 of the V verifier Git blob/worktree bytes
  amendment = exact path/sha256/commit authority object from the HV manifest

process keys, in order:
  parent_pid = exact positive Python int of the authoritative CLI process
  child_pid = exact positive Python int of the isolated legacy child
  child_exit_code = 0 exactly when status is VALID, otherwise 1
  python_executable = ".venv/bin/python"
  python_version = "3.12.3"
  numpy_version = exact observed parent/child version that matched the persisted artifact environment.numpy_version
  isolated = true
  child_head_commit = H
  cuda_visible_devices = ""
```

The three runtime receipt fields are emitted only from authenticated observed
values: `python_executable` is normalized to the registered
repository-relative path only after the exact absolute `sys.executable` gate;
`python_version` is formatted from the observed exact version tuple; and
`numpy_version` is copied from the observed parent runtime only after the child
has proved exact parent/child/persisted equality. They are runtime provenance,
not candidate or scientific values.

The receipt has no scientific verdict, decisive clause, candidate flag or
value, field, row, score, metric, aggregate, bootstrap value or hash,
criterion, exclusion, or excerpt. Exact schema validation recursively rejects
every missing, extra, reordered, mistyped, or inconsistent field. `status` is
`VALID` if and only if every preflight and child validation gate succeeds and
the child exits `0` with its exact fixed token. It is `INVALID` if authenticated
execution reaches artifact validation and any artifact/adapter/legacy/equality/
serialization gate fails with the exact fixed invalid token and child exit `1`.
An unauthenticated parent, unexpected child behavior, or receipt-publication
failure is structural: stop without claiming either status.

The parent serializes the validated receipt with two-space indentation,
`sort_keys=False`, `allow_nan=False`, UTF-8, and one final newline. It writes a
process-specific temporary file with exclusive creation, flushes and fsyncs it,
hard-links it to the absent final path without replacement, fsyncs the parent
directory, and removes the temporary. It then strictly reloads and validates
the final bytes. No existing path is replaced. CLI exit is `0` only for a
published `VALID` receipt, `1` only for a published `INVALID` receipt, and `2`
for structural/preflight/publication failure.

## Test-first implementation and review authority

Implementation is limited to the exact four source/test paths defining `V`.
The complete RED suite precedes verifier implementation and manifest-validator
GREEN work. It must prove:

- canonical string keys and exact order after JSON roundtrip;
- a live-producer exact recursive concrete-type/key-order/list-order/signed-zero
  roundtrip through `exact_ordered_equal`, plus byte-identical first/second
  producer serialization, with ordinary `dict ==` forbidden as the predicate;
- rejection of integer, boolean, alias, missing, extra, reordered, duplicate,
  and collision keys;
- unchanged live scientific relational checks and serialized output order;
- strict raw JSON duplicate/nonfinite/path/symlink/hash gates;
- the single permitted adapter mutation and rejection of every other mutation;
- full old `scientific_payload` invocation and no selected-field shortcut;
- old-H cwd, diagnostic `__file__`, HEAD, manifest, S/H blob, and 31-path gates;
- the exact ten-key old persisted manifest projection derived from H and
  rejection of every omitted-authority/current-projection mutant;
- V/HV verifier `__file__`, manifest, source-path, Git-blob, worktree, parent,
  and clean-checkout gates;
- exact recursive type/order equality and byte-identical serialization;
- exact registered `sys.executable`, observed Python `3.12.3`, old-module NumPy
  identity, and parent/child/persisted NumPy version equality before recomputation;
- candidate/row/value-free stdout, stderr, and receipt;
- canonicalization, tolerance, ordinary-dict-equality, selected-field,
  current-producer, wrong-import, wrong-cwd, and detached-source mutants fail;
- exact VALID/INVALID receipt predicates and every nested mutation;
- atomic no-clobber behavior, interruption/failure stop, and one-attempt guard;
- no dataset/model/GPU/candidate-free/scientific producer reachability; and
- exact future manifest authority/projection order and 32-path source order.

Run focused RED nodes before each implementation unit, then GREEN nodes, then
the complete affected CPU-only test, Ruff, `py_compile`, and diff gates. No test
may open the real artifact; tests use synthetic fixtures whose values are
unrelated to the immutable artifact.

Commit the exact four-file source/test implementation, obtain a fresh
independent full-source review, repair every Critical or Important finding
test-first within those four files, and repeat until clean. The final reviewed
commit is `V`.

Only after `V` is clean may the real manifest be edited alone and committed.
Obtain a fresh independent full manifest/provenance review and repair only the
manifest until clean. The final manifest-only commit is `HV`. No result path is
created in either commit.

After a clean detached checkout at `HV` is authenticated, run the verifier CLI
exactly once. No GPU command, candidate-free audit, scientific command, or
artifact rewrite follows. On `INVALID`, structural failure, unexpected exit,
timeout, signal, interrupted publication, or receipt validation failure: STOP.
On `VALID`: preserve the receipt as outcome-blind validation evidence and STOP;
any interpretation or publication of the scientific artifact requires a new,
separate authority not granted here.
