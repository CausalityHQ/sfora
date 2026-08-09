# Pass201 PA source-v2 prelaunch-output amendment — 2026-08-09

Status: prospective operational repair, written before any authorization commit,
controller launch, model training, receipt, report metric, or Pass201 value.

## Trigger and preserved failed state

Reviewed code commit `dd7ccf05a18b9a565f20f3bb4f3d856fc11af9bb`
implemented the Task-4 public-interface shorthand that required
`freeze-authority --output` to name the final canonical manifest.  Task 5 instead
requires two fresh top-level controller processes to write two distinct temporary
outputs, prove byte identity, and only then exclusively move one accepted byte stream
to the canonical manifest path.  These requirements are incompatible.

The first remote wrapper also omitted the required checkout `cd`.  Bound Python
therefore exited with code 2 because it could not open the controller script.  The
controller did not load.  Both temporary paths, the canonical manifest, the private
run directory, all scientific outputs, and the receipt remained absent.  No model or
candidate value was computed.  The failed checkout
`/home/riomus/sfora-pass201-pa-source-v2` remains detached, clean, and preserved at
the reviewed commit; it is not reused, removed, cleaned, or reinterpreted.

This did not consume the single authorized training attempt because no authorization
`A` and no controller/training process existed.  It does require a new reviewed code
commit and a new, previously absent fixed checkout path.

## Repaired output contract

For a resolved checkout root `R`, define exactly two permitted temporary authority
paths as siblings of the checkout:

```text
R.parent / (R.name + ".pass201-prelaunch-freeze-1.tmp")
R.parent / (R.name + ".pass201-prelaunch-freeze-2.tmp")
```

`freeze-authority --output` accepts exactly one of these two normalized absolute
paths.  It rejects the canonical manifest path, every relative/aliased spelling,
every other sibling, and every path inside the checkout, dataset, import roots, run
directory, or another filesystem location.  The selected temporary path, canonical
manifest, and private run directory must be absent before and after each internal
capture child.  The other permitted temporary may exist during the second top-level
freeze invocation and is not part of captured authority.

The output path is only a sink for the already-canonical authority bytes.  It is not
included in the manifest, so the two top-level processes must remain byte-identical.
Publication remains exclusive and fail-closed.

After both processes exit zero, Task 5 must require literal byte identity, bind both
temporary files by descriptor, and require equal byte count/SHA-256.  It then creates
the absent canonical manifest by a same-filesystem non-clobber hard link from the
accepted first temporary, fsyncs the canonical parent, removes both temporary names,
and reopens/revalidates the canonical bytes.  The checkout must then contain exactly
the sole untracked canonical addition before commit `A`.

## Repaired execution path

- New final checkout path:
  `/home/riomus/sfora-pass201-pa-source-v2-r2`.
- Every remote controller command begins with an exact
  `cd /home/riomus/sfora-pass201-pa-source-v2-r2` and verifies `pwd -P` before
  invoking bound Python.
- The new reviewed repair commit is a new source commit `C2`; authorization `A2`
  has exactly one parent `C2` and adds only the canonical manifest.
- The original failed checkout and its nonexistent outputs are evidence only.
- There is still exactly one controller/training launch: from detached `A2`, after
  the same queue/runtime/data/absence checks.  Any failure after that controller
  starts consumes the attempt and is never retried.

No scientific threshold, recipe, dataset, seed, model, training schedule, sidecar,
receipt, activation, or Pass201 decision changes in this amendment.
