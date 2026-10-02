Consultation22be9f89b01c41b7, Jev-routed GPT-6 Astra XHigh, completed exit0. Source audit only; original CPU v1 remains FAIL.

**One correction candidate is demonstrated: reuse authenticated admission results instead of streaming the entire warm guard inventory again. This is a source-level redundancy, not a demonstrated cause of the timeout.** The original invocation remains **FAIL**, with no accepted qualification or GPU admission.

I verified the adapter matches commit `560dd4d4` and its frozen copy. The local genuine trainer and external driver files also match their pinned hashes.

The call chain is [`genuine.authority` → guard merge → warm `admit_unit`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_postln_adaptation.py:288). Genuine admission authenticates prerequisite inventories using its existing [`FlatAdmission.bound_file`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_genuine_views.py:464). The adapter then discards that verified-read ownership: [`admit_unit`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_postln_adaptation.py:220) calls its unconditional streaming hasher for every warm guard.

A bounded, stdlib-only check authenticated the copied receipts, reconstructed the inventory overlap, and executed the actual admission-loop AST with counting stubs:

| Warm admission accounting | Count |
|---|---:|
| Unconditional full-file hash calls | 15,114 |
| Already authenticated by genuine admission | 15,113 |
| Repeated image-file hashes included above | 13,283 |
| Newly introduced file | Warm `resume.pt` only |

The redundant pass covers **at least 3,481,672,192 payload bytes** from the two full vision artifacts and two feature caches alone. Images, libraries, initializer, headers and other files add more. This is a logical hashing-byte lower bound, **not measured physical I/O or elapsed time**.

The smallest prospective correction is two adapter changes, reusing the pinned [`FlatAdmission`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:214):

```python
# After genuine.authority succeeds and check_warm_record passes:
for path, digest in selected['guards'].items():
    admission.verified.add(str(admission.register(guards, path, digest)))

# In admit_unit's existing input_guards loop:
context['admission'].bound_file(context['guards'], path, digest)
```

Seed this ownership **only from the successful current-call `selected['guards']`**, never directly from receipt claims or the general merged dictionary. `register` retains canonical-file, digest-format, size and conflict checks; unknown files still receive complete SHA admission. No helper globals or pinned original drivers change.

For the CPU warm-inventory loop, expected full-file hash calls become **1 instead of 15,114**. Keep the warm checkpoint’s separate [pre-load byte hash and complete logical fingerprint](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_postln_adaptation.py:429): its existing two pre-native byte hashes are real, but removing the use-boundary check is unnecessary for this correction. Preserve every original validator, four sequential native factories, complete state checks, raw/unit/packed witnesses and full uncached exit.

**Bounded falsifying test:** before any prospective native decision, add one stdlib-only test, capped at 10 seconds, using two tiny temporary files and a counting reader. Require inherited admission bytes to be reused, the new file to be fully hashed, and final guard inventories to remain identical. Require conflicting digests, noncanonical paths and corrupt new files to fail; a same-size mutation with restored mtime must still fail the uncached exit reader. Reject the correction if any assertion fails.

**Risk and limit:** ownership transfer relies on fresh successful admission and allows an inherited file’s later mutation to be detected at its retained use/exit checks. It must never become a persistent cache. The terminal contains no phase trace, so this invocation may have timed out before reaching the redundant loop. No CPU120 fit claim follows; any prospective source decision remains root-owned.

No files changed; no native execution, jobs, reviews or operator contact.
