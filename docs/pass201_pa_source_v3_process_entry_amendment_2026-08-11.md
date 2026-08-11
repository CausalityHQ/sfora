# Pass201 Source-v3 Process-entry Environment Amendment

Date: 2026-08-11 UTC

## Status and scope

This is a prospective operational amendment to the Pass201 ordinary-PA
source-v3 authority. It changes no dataset, model, optimizer, seed, training
command, package inventory, output schema, scientific formula, threshold, or
decision rule. It preserves the original source-v3 protocol and H3 as immutable
historical authority and records why H3 could not launch the registered training
child.

The authenticated chain through the failed preflight is:

- V3 `03d0ed509fe7b65aee0162941a9f6a3b6fea228f`;
- H3 `183fa5b9cf99b7f860e954c9be38c06a477b3912`;
- H3 manifest SHA-256
  `bec53f011c5a85802fdf443acf8827236efe67bfd18243286f2858188e9c11aa`.

H3 passed strict schema, Git/worktree, package, runtime, source, absence, and
independent review gates. Two top-level controller invocations then exited in
`validate_runtime_preflight` before `create_and_lock_private_run_directory`,
before the registered training child, and before any GPU process:

1. PID 1031337 used an incomplete ambient environment and exited with
   `controller environment drift`.
2. PID 1032024 used the exact frozen 16-key process-entry environment via
   `env -i` and exited with the same error.

After the second exit, the run directory and all six outputs remained absent,
and NVIDIA reported no compute process. A read-only import-boundary probe under
the exact frozen environment proved the root cause: importing the authenticated
training stack adds exactly

```text
KMP_DUPLICATE_LIB_OK=True
KMP_INIT_AT_FORK=FALSE
```

and removes or changes no process-entry key. The existing implementation
compares the post-import live environment to the pre-import frozen environment,
so the registered command is structurally unreachable. Neither exit is a
training attempt. The protocol's one-attempt rule begins only when the
registered image-end-to-end child is spawned.

## Exact process-entry contract

The replacement source must snapshot `dict(os.environ)` exactly once after
stdlib imports and before importing `pass201_pa_source_v2_contract`, `typer`, or
any `sfora` module. Call this immutable-by-convention private snapshot
`_PROCESS_ENTRY_ENVIRONMENT`.

`_require_replacement_environment` must require both:

1. the snapshot is exactly equal, including key set, built-in string types, and
   values, to `authority.payload["execution"]["environment"]`; and
2. the post-import live `dict(os.environ)` is exactly the snapshot plus the two
   ordered additions above, with no removed, changed, or additional key.

The comparison is exact dictionary equality. No filtering, coercion, default,
wildcard, prefix allowance, or `os.environ` mutation is authorized. The
controller must not delete or rewrite the two observed KMP entries.

Every subprocess involved in runtime recapture, sidecar derivation, or training
continues to receive the original exact frozen 16-key replacement environment
through `subprocess`'s explicit `env` argument. The two KMP additions are parent
import evidence only and must not appear in the manifest's execution
environment, the receipt's command environment, or the registered child.

## TDD and falsifiers

Before production changes, a fresh-process RED test must enter with the exact
16-key source-v3 environment, import the real controller, prove the two exact
KMP additions exist, and show the old preflight rejects it. GREEN must prove the
new predicate accepts that exact state.

Mutation tests must independently reject:

- missing, changed, or extra process-entry keys;
- either missing, mistyped, or changed KMP addition;
- any third post-import addition;
- a KMP key present at process entry;
- snapshot capture after contract, Typer, or `sfora` import;
- any registered child environment containing either KMP key.

Historical source-v2 schema and receipt validators remain byte-semantically
unchanged. The replacement source must retain the existing full source-v3 suite,
Ruff, `py_compile`, and diff-check gates and obtain independent read-only review.

## Replacement Git and manifest chain

The prospective chain is linear:

```text
A3 -> P3 -> I3 -> V3 -> H3 -> A4 -> P4 -> S4 -> H4
```

- A4 adds only this amendment.
- P4 adds only its bound implementation plan.
- S4 changes exactly the same six source/test paths authorized for V3.
- H4 changes only
  `docs/pass201_pa_source_v3_authorization_manifest.json` in mode `100644`.

S4 must authenticate every preceding edge, with no merge. H4 has sole parent S4,
its manifest `source_commit` is S4, and its `protocol` and `plan` objects bind A4
and P4. Since H3 already added the manifest path, H4's sole edge is exact `M`,
not `A`. All runtime, package, source, dataset, output, and frozen scientific
domains are recomputed from S4 and otherwise unchanged.

The authority freezer must again run in exactly two fresh top-level processes
with one shared prospective absence timestamp and the two registered sibling
temporary paths. Their bytes must be identical before H4. No training or GPU
process is authorized until H4 independently validates as READY.

After H4 is READY, exactly one top-level controller may pass preflight and spawn
the one registered source-training child. GPU bitwise reproducibility remains
explicitly outside the premise; the resulting single observation is accepted
only through semantic and provenance validation.
