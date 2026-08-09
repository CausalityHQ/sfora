# Pass201 PA source-v2 checkout-r3 amendment — 2026-08-09

Status: prospective operational repair, written before any authority capture at
the r3 path, authorization commit, controller launch, model training, receipt,
report metric, or Pass201 candidate value.

## Trigger and preserved state

The reviewed source commit
`dd61c0ebbd069ed006fb19224a4762de57c8f1a4` repaired the temporary-output
contract. A later shell wrapper created the intended r2 checkout and detached it
at that commit, then failed its own parent-count assertion. The assertion used
`git rev-list --count --parents HEAD`, whose output is a single revision count;
counting its fields therefore returned zero instead of the number of parents.

The wrapper stopped before computing the shared RFC3339 capture timestamp and
before loading `freeze-authority`. No temporary authority output, canonical
manifest, authorization commit, controller process, training process, private
run directory, receipt, report, metric, or candidate value was created. The r2
checkout remains detached, clean, and preserved at the reviewed commit. It is
evidence only and is never reused, removed, cleaned, moved, or copied.

This failure did not consume the single authorized scientific attempt. It did
consume the r2 checkout path as a prospectively absent final path.

## Final checkout and source commit

The only final execution checkout is the previously absent literal path:

```text
/home/riomus/sfora-pass201-pa-source-v2-r3
```

This amendment is committed before that path is created. Its commit is the new
reviewed source commit `C3`. The checkout is cloned once from
`origin/devbox/emafactorial`, detached at `C3`, and never moved or copied.
Authorization `A3` has exactly one parent, `C3`, and adds only
`docs/pass201_pa_source_v2_prelaunch.json` with mode `100644`.

The two permitted temporary outputs are therefore exactly:

```text
/home/riomus/sfora-pass201-pa-source-v2-r3.pass201-prelaunch-freeze-1.tmp
/home/riomus/sfora-pass201-pa-source-v2-r3.pass201-prelaunch-freeze-2.tmp
```

All scientific constants, source-v2 schemas, runtime bindings, dataset,
training recipe, receipt checks, activation rules, and Pass201 decision rules
remain unchanged.

## Checked operational procedure

The remote procedure is written once as a shell program, syntax-checked, and
read-only reviewed before execution. It uses `set -euo pipefail`, changes to the
exact checkout before every controller call, and verifies `pwd -P`.

The one-parent assertion is exactly the output shape of:

```text
git rev-list --parents -n 1 HEAD
```

split into a shell array. It requires exactly two fields: the commit
and its single parent. It does not use `--count` or parse presentation text with
`awk`.

Both top-level `freeze-authority` invocations use the same operational RFC3339
UTC value, separate fresh processes, the bound Python interpreter, the exact
clean environment, and the two permitted output paths. Neither invocation is
retried. After both exit zero, descriptor-bound byte and SHA-256 equality is
required before same-filesystem non-clobber hard-link publication, directory
fsync, temporary-name removal, and canonical-byte revalidation.

Before computing that timestamp, the procedure also verifies nonempty Git
author and committer identities and completes a non-mutating authenticated
dry-run push of `C3` to the already-equal remote branch. The later authorization
commit explicitly disables commit signing so an untested signing agent cannot
consume the captured path.

Before commit, porcelain must contain exactly the one canonical untracked file.
After commit, the checkout must be detached and clean, the commit must have the
single parent `C3`, and `git diff-tree C3 A3` must contain exactly the sole
`100644` addition. The edge is independently reviewed, then pushed, before the
single `run` command may start.

Any failure after the first `freeze-authority` process starts permanently stops
this source attempt. Any controller start consumes the single scientific
attempt and is never retried. No report metric or candidate path is read before
non-selective activation.
