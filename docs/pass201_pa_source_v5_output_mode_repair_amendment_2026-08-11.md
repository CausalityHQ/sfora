# Pass 201 source-v5 immutable-output mode repair amendment — 2026-08-11

## Status and scope

This is a prospective repair of one activation-time evidence validator. It does
not alter or rerun the completed ordinary Proxy Anchor training, any output
bytes, the dataset, checkpoint, configuration, candidate, diagnostic
arithmetic, contexts, thresholds, bootstrap, or decisions. It authorizes no GPU
process.

The completed ordinary Proxy Anchor result remains the reproducible baseline:
In-Shop Recall@1 `0.9174989449992966`. Higher published results remain external
comparison targets until reproduced under the same data and evaluation
protocol.

## Bound predecessor chain

```text
historical source S4: 53a9db9e9dbe54fcebb33769b915c3f33699d522
historical handoff H4: 32c4d39322fca2a5a906f785bdb612dcd7008647
reviewed source V5: 656b5f2069f76ee6d8c5079bee8ae6a371a89f69
manifest-only handoff H5: 18b225f33b61dd221d6878cf8b14eb75a0037323
H5 manifest path: docs/pass201_pa_source_v5_authorization_manifest.json
H5 manifest SHA-256: 2cf3b9a1c5cb41304f8d653e839d5372fa9570c4f442d4948ecdec4256c0de20
H5 preservation ref: pass201-source-v5-handoff-18b225f
H5 recovery bundle: /home/rb/pass201-h5-18b225f.bundle
H5 recovery bundle SHA-256: 838b3f65435b374e172220caa1612910fc3ca73fd24560a0ab7affec6e7ceb75
historical receipt SHA-256: a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9
```

H5 is a sole-manifest child of V5. H4 is the separately preserved
sole-manifest child of S4. Every source output below remains byte-identical to
the historical receipt. During independent review, H5 was found absent from
the local shared Git object store. Before source work, the complete registered
bundle above was verified, H5 was imported, and the exact local preservation
ref above was created. The ref resolves to H5, H5's sole parent is V5, its sole
edge adds the v5 manifest, and the manifest Git bytes have the SHA-256 above.

## Third structural activation failure

The first H5 activation process was launched once on
`riomus@spark-2751` from the fresh detached checkout
`/home/riomus/pass201-h5-18b225f`:

```text
PID: 1061572
mode: --activate-source
device visibility: CUDA_VISIBLE_DEVICES=
exit: 1
failure: ValueError: source-v3 output evidence differs
failing function: _validate_source_v3_output
```

The failure occurred before activation publication, source-manifest
publication, model/checkpoint loading for candidate computation, smoke,
science, or any GPU work. After exit, the activation, source-manifest, smoke,
scientific result, and owned temporary paths were all absent; no diagnostic or
GPU compute process remained. This attempt is permanently recorded as a
structural failure and is not scientific evidence.

## Root cause

The historical producer intentionally makes completed outputs immutable.
`hash_open_regular` opens each regular file without following symlinks, hashes
it, applies mode `0444`, re-hashes it, and records the resulting complete
`st_mode`. The authenticated receipt therefore records decimal `33060`, equal
to octal `0100444`, for each required output. The copied H5 files are regular,
non-symlink files with exact mode `0444`, size, and SHA-256.

The activation validator instead requires the unrelated literal
`evidence["mode"] == 0o100644`. It does not compare the named file's actual
mode with the receipt. Consequently, valid read-only evidence is rejected and
a writable copy would still fail because the receipt remains `0100444`.

## Considered repairs

1. **Change or copy the outputs as `0644`: rejected.** This mutates an
   authenticated property and cannot make the receipt's `0100444` equal the
   validator's `0100644` without also rewriting authority.
2. **Ignore mode completely: rejected.** This weakens immutable-file and
   symlink substitution protection.
3. **Bind the live file to the authenticated receipt mode: required.** This is
   the producer-consistent invariant and preserves every other strict check.

## Exact repaired invariant

For each of the four activation inputs, in the existing literal order
`report`, `checkpoint`, `resolved_config`, `train_manifest`, validation must:

1. rely on the already-required canonical receipt byte equality to bind JSON
   key order, and require the evidence object to have the exact key set
   `bytes,file_type,mode,path,sha256` and exact concrete JSON types;
2. require the receipt mode to be exactly the historical literal decimal
   `33060` (`0o100444`), not merely any regular-file mode;
3. resolve only the exact authority-declared repository-relative path under
   the detached checkout root;
4. open the named file with `O_RDONLY | O_NOFOLLOW` where available;
5. require a regular file, exact complete `st_mode == evidence["mode"]`, exact
   byte count, and exact SHA-256 while holding the descriptor;
6. require the path identity and parent identity not to change across the
   read; and
7. compare the live absolute path separately to
   `root / evidence["path"]`; the descriptor verifier reports live mode, size,
   and digest and does not pretend its absolute path equals the receipt's
   repository-relative path; and
8. reject symlinks, mode drift including `0644`, path aliases, non-regular
   files, size drift, hash drift, and replacement races before parsing or
   deserializing content.

The repaired code may reuse the authenticated producer contract's existing
descriptor-based `hash_open_regular` only if reuse does not chmod or otherwise
mutate the file. The producer helper currently enforces immutability by chmod,
so activation must use a read-only verifier rather than that mutating helper.

No caller-selected mode, chmod, normalization, fallback, warning-only path, or
permission-bit mask is allowed. Comparing only `stat.S_IMODE` is insufficient:
the complete regular-file mode is the receipt authority.

## TDD and review requirements

Before production edits, a real-file RED must use a mode-`0444` regular file
with independent expected bytes/hash and receipt evidence `33060`; the current
validator must reject it only because of the hard-coded `0644`. Mutation tests
must independently cover:

- receipt modes `0100644`, `0100400`, boolean, float, string, and negative;
- live-file chmod to `0644` after receipt construction;
- symlink, directory, FIFO where supported, size/hash/path drift;
- replacement between pre-open and post-read identity checks; and
- all four output keys, proving failure occurs before JSON/checkpoint parsing,
  model access, candidate construction, or publication.

The minimal source change and tests receive an independent Opus-to-Sol review.
The reviewer must confirm no scientific function, constant, model path,
candidate path, or result schema changed.

## Git and refreeze sequence

The chronology is prospective and linear. The initial amendment commit is
`622e145b2dfeafdf6a202c7012ad92813a2932c2`; its first sole-file fix is
`a8119bcf4b97de6a7a948d75ffd393e64c406b10`; the initial plan is
`651920e80126d2c8f31b2acce6d04438fe0c12a8`. Final A6 is the sole-file
review-fix child of that initial plan and incorporates the independent review;
final P6 is A6's sole-file child:

```text
V5
 -> 622e145b2dfeafdf6a202c7012ad92813a2932c2
 -> a8119bcf4b97de6a7a948d75ffd393e64c406b10
 -> 651920e80126d2c8f31b2acce6d04438fe0c12a8
 -> final amendment A6
 -> bound implementation plan P6
 -> reviewed source/test repair V6
 -> manifest-only handoff H6
```

A6 changes only this document. P6 changes only its plan. Every P6-to-V6 commit
is single-parent, merge-free, nonempty, and confined to:

```text
scripts/diagnose_pass201_cis_operator.py
scripts/pass201_pa_source_v2_contract.py
tests/test_diagnose_pass201_cis_operator.py
tests/test_pass201_pa_source_v2_contract.py
```

H6 is the sole-manifest child of V6. It adds a new source-v6 authorization
manifest rather than modifying H5 or any historical authority. The new
manifest incorporates this amendment and plan by exact path, SHA-256, and
commit; preserves every historical producer domain and output digest; binds
the reviewed V6 source rows; and contains no candidate or scientific values.
The contract change is restricted to the exact source-v6 authority shape and
the non-mutating descriptor-based existing-output verifier; it cannot alter the
historical producer, receipt builder, or publication helper.

## Execution authorization after H6

Only after source review and independent H6 review may one fresh CPU activation
process run in a new detached clean H6 checkout. It must reuse byte-identical
copies of the six immutable historical outputs, verify exact `0444` modes and
receipt evidence before launch, and keep CUDA unavailable.

If activation passes, the pre-existing plan gates continue in order:
binding-only, seed-0 smoke, then one scientific CPU process. Any structural or
integrity failure stops the chain. Source training and GPU execution remain
forbidden. The two structural attempts that preceded PID `1061572`, plus PID
`1061572` itself, remain the exact three disclosed structural attempts and are
never reclassified.
