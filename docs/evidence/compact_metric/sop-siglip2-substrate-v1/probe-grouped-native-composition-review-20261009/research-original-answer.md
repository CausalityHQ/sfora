**Verdict: conditional GO for source work only. Not ready for native freeze.**

The committed design is sound in outline. Supplemental authority (S) comes only from the owner, the pure map preflight runs before any reads, the complete-S rule is kept, there are exact AST inverses, and the exits are untouched. Three items have to be closed before freezing, and one of them is in the OWN2 draft. I made no edits and ran no tests, imports, native runs or SSH.

## 1. The culprit can be derived from source plus the log (not observed directly)

**S is part of M−K at the failing entry. That follows from the source and log, assuming no library was unmapped and remapped.** M is the mapped set, K the historical known set.
- `native_ties` opens `candidate.so` (`qualify_connected_serving_requests.py:401`) before `api.audit_origins` runs at `qualify_connected_probe_serving.py:1107`. `collect` then requires the complete frozen S (`connected_control_native_authority.py:267-269`).
- The exit failure message is "exact four" (`train_siglip2_nearest_ranking.py:512`). That check is only reached after `private_audit`→`collect` returned. `dispatch` had set `admitted=True` (`:301`), so both S members were mapped at exit.
- S cannot overlap K. `__init__` enforces disjointness from H (`:142`), and K holds only historical native files plus pins with cuDNN names. So `mapped <= known` at line 1620 had to be false.
- Remaining bound: the failing set is inside the exit set, which is inside H's `.so` files plus S. So M−K ⊆ S ∪ (H's `.so` files outside K). That right-hand term can be computed offline from the frozen origins and `required_guards`.

**The first entry never evaluated line 1620.** `origin-causal-audit.json` records that `charset_normalizer.md` was absent before import, so `grouped_md_origin` returned None at `:1597`. md was imported inside the first scope (the load at `qualify_connected_probe_serving.py:465`). The batch entry at `:480` was the first time the predicate ever ran in this setup, so there is no evidence it passes even without S.

**The exit exact-four failure looks like a knock-on effect.** Because `collect` passed, any extra files beyond the known set can only be cuDNN supplement members. The failure therefore means at least one of the four cuDNN libraries was unmapped, which fits "no forward ran" (v3 recorded the same thing). That it clears once a forward runs is conjecture.

## 2. Conditions before freeze

| # | Severity | Issue |
|---|---|---|
| C1 | **Draft blocker** (OWN2, `native_paths`) | The preflight adds `not alias.exists() and not alias.is_symlink()`. Neither the 95cb predicate nor the committed design has this. The charset_normalizer RECORD lists `md.cpython-313…so` at 201304 bytes as installed. If the file is on disk, every grouped entry rejects. The original only requires `alias ∉ M`. Remove the check, or confirm absence with one stat on the host. |
| C2 | **Missing evidence** | The four cuDNN libraries are supplements precisely because they are not in the historical native files. Their only route into K is through the cuDNN-named pins in `required_guards`, snapshotted at `evaluate_siglip2_connected_probe.py:1225`. The first forward maps them. The next batch entry then re-runs line 1620, and if the pins are missing it fails on cuDNN members that S does not cover. This can be checked offline against the frozen guard set. |
| C3 | **Resource and side effects** | `api.audit_origins` inside the callback runs a full genuine collection at every entry after md loads (one per batch kind). `imported_origins` hashes every module and mapped native. The old audit then re-hashes all of them via `bound_file` with `FADV_DONTNEED`, so every read is cold. `validate_runtime` re-reads S and its provenance. The call also mutates `legacy['origins']`, the legacy and prior guards, and `owner.inventory`. The original helper never hashed K members. The cost is unmeasured: v4 has no timing for the line-1107 audit, only about 60 s for the whole failure-to-exit stretch. This conflicts with "no resource increase". **Smaller alternative:** after the preflight, call `api.authenticate()`. Its `owner.check()` already re-reads S bytes, provenance, build evidence and S inode/dev. Add the pure check `mappings()==M` for S identities. That authenticates exactly what the adapter adds and leaves hashing of H to the unchanged line-1107 and exit audits, as the original intended. If you keep the audit, freeze a source-derived budget (entries × cost) first. |
| C4 | Should-fix | Pin S to exactly the two expected paths. Require it to exclude G, the md alias and anything under `site`. Today it rests only on `owner.files`. |
| C5 | Keep | The draft fixes the insertion index from the exact matching predicate and pops at that index. Keep this: an inverse that only removes the matching node from anywhere would not fix its order. The late `bind_native` is needed because the boundary is built at `:1062`, before `install` at `:1071`. The callback must keep failing closed while unbound. |

**Alternative that deletes code:** run the reference step (R) before `native_ties` and the line-1107 audit. S would then be unmapped at every reads-only entry, the original predicate passes unchanged, and no new authority code is needed. Two things are unverified: whether `from sfora import cutile_int8` or `native_start` maps S earlier, and whether the reload owner (B) re-enters the reads-only scope after S is mapped. It also changes the frozen order, so it needs its own review.

## 3. Smallest falsifier (stdlib-only, extracted real AST, read counters)

Count calls to `open`, `bound_file`, `extract.sha`, `observer.file_bytes` and `Source.check`.

**Should pass:**
- H plus the exact S.
- Ordinary H alone, with S unmapped and not admitted.
- The four cuDNN libraries mapped and pinned.
- The alias file present on disk but unmapped (this catches C1).

**Should reject with zero authentication or content reads:**
- A group member missing.
- The physical md mapped.
- An unknown `.so`, even if guard-pinned.
- A deleted, noncanonical, or inode/dev-replaced mapping.
- `candidate.so` mapped while `gpucomp` is unmapped and S is admitted.
- An owner whose `mappings` is shadowed or class-patched.

**Should reject later:**
- A forged API, `authenticate` code, closure or global.
- A wrong source pin, hash or RECORD row.
- A projected inventory missing one member, run through the retained real exact-four wrapper.

**Should also hold:**
- The four cuDNN libraries mapped but unpinned still rejects at line 1620 (this documents C2).
- If md is not loaded, the callback is never invoked.
- The callback does not mutate `mapped`, `known` or `group`.
- The byte and AST inverses hold, the original globals are unchanged, and the weak-reference owner fails closed once released.

**Offline, no native run:** compute H's `.so` files outside K, and the intersection of `required_guards` with the four cuDNN paths.

Even if the grouped boundary passes, that does not qualify the exit, parity, speed or the product. All earlier FAIL results stand.
