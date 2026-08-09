# Pass201 PA source-v2 checkout-r4 amendment — 2026-08-09

Status: prospective transport repair, written before any authority capture at
the r4 path, authorization commit, controller launch, model training, receipt,
report metric, or Pass201 candidate value.

## Trigger and preserved state

The r3 checkout was created detached and clean at source commit
`bf7ddeded7b801b7859fc3c3e15088550b40bb2a`. Its checked wrapper then ran the
new pre-capture Git capability gate. `git var GIT_AUTHOR_IDENT` and
`git var GIT_COMMITTER_IDENT` succeeded, but the non-mutating HTTPS push dry run
failed because the DGX has no GitHub push credentials.

This happened before the shared RFC3339 capture timestamp was computed and
before `freeze-authority` loaded. No temporary authority output, canonical
manifest, authorization commit, controller, training process, private run
directory, receipt, report, metric, or candidate value was created. The r3
checkout remains detached and clean at its source commit. It is evidence only
and is never reused, removed, cleaned, moved, or copied.

The failed capability check did not consume the single authorized scientific
attempt. It did consume the r3 checkout as a prospectively absent final path.

## Correct transport boundary

Requiring the DGX to push to GitHub was an unnecessary authentication coupling.
The DGX needs only read access to create the exact detached source checkout and
local Git identity to create the authorization commit. The devbox already has
authenticated GitHub push access and can read the DGX repository over SSH.

The only final execution checkout is the previously absent literal path:

```text
/home/riomus/sfora-pass201-pa-source-v2-r4
```

This amendment and the checked r4 procedure are committed and pushed from the
authenticated devbox before that path is created. Their commit is source commit
`C4`. The r4 checkout is cloned once, detached at `C4`, and never moved or
copied. The two temporary outputs are exactly:

```text
/home/riomus/sfora-pass201-pa-source-v2-r4.pass201-prelaunch-freeze-1.tmp
/home/riomus/sfora-pass201-pa-source-v2-r4.pass201-prelaunch-freeze-2.tmp
```

The remote procedure verifies author and committer identities before capture,
but performs no GitHub push. After it creates detached authorization commit
`A4`, it stops and prints the exact `C4`, `A4`, checkout path, manifest byte
count, manifest SHA-256, timestamp, and two capture PIDs.

The devbox then fetches the DGX checkout's exact detached `HEAD` over the
already-authenticated SSH host connection into a new local review ref. It
requires that fetched object to equal the reported `A4`, have exactly parent
`C4`, and contain exactly the sole `100644` manifest addition. An independent
review checks `C4..A4`, the manifest blob/current bytes, the live DGX checkout,
and runtime bindings. Only after READY does the devbox push the reviewed local
`A4` object to `origin/devbox/emafactorial`. The DGX never needs GitHub write
credentials.

Before this amendment was committed, the devbox verified that transport without
touching r4: `git ls-remote` and `git fetch --dry-run` over the same SSH URL
successfully resolved the preserved r3 detached `HEAD` as
`bf7ddeded7b801b7859fc3c3e15088550b40bb2a`. No report or candidate artifact was
read.

## Preserved scientific contract

The descriptor-bound publication, non-clobber hard link, conditional temporary
name removal, fsyncs, exact byte checks, single controller launch, receipt
checks, non-selective activation, scientific constants, runtime bindings,
dataset, recipe, and Pass201 decision rules are unchanged.

Any failure after the first `freeze-authority` process starts permanently stops
this source attempt. Any controller start consumes the single scientific
attempt and is never retried. No report metric or candidate path is read before
non-selective activation.
