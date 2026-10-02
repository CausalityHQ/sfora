Recommend **one correction: seal the complete admitted `partition` into owned typed frames**, reusing the encoder-frame helper. Leave source/identity metadata and tensor hashing unchanged.

The verified hot graph is:

`update → pre-integrity(STATIC + frozen_tree) → unchanged update → post-integrity(STATIC + frozen_tree) → complete payload fingerprint`

The pinned partition has **24,268 nodes and 286,299 original framed bytes**. Five traversals therefore visit **121,340 nodes per completed update**. Frames remove those repeated Python traversals; the enclosing digest still receives all **1,431,495 bytes**. These are source counts, not measured savings or evidence of 300-second fit.

1. Extend [quadratic_encoder_frames.py](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_encoder_frames.py:49) to `seal(original, *roots)`, returning an immutable capsule tuple, the adapted fingerprint, and `check_owned(slot, value)`. Require exact capsule type and identity for each slot; reject foreign, replacement, conflicting and wrong-role capsules. Preserve the untouched original serializer and framing.

2. In [the trainer](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:663), seal encoder plus partition **after** `prepare_native` authenticates the complete warm payload, verifies warm partition equality with the authenticated selected partition, and checks original STATIC. Replace the live partition root with its capsule. Add `owned_partition` beside `owned_encoder`; use it in `fresh`, `frozen_tree`, `payload` and live payload admission. Route the existing complete STATIC fingerprint through the adapted serializer. Persist materialized ordinary dictionaries for both roots, requiring the untouched original full-payload fingerprint to equal the optimized digest. Keep ordinary disk admission and independent reload checks intact.

3. Extend [the existing differential falsifier](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_quadratic_readout.py:760): **≤30 seconds, ≤4 MiB, stdlib/fake tensors only**. Assert old/new complete fingerprints and persisted materialization agree, including tuple/list and integer/string-key distinctions. Exercise every existing source/encoder/STATIC/payload mutation witness, failed partition admission, cross-owner and wrong-role replacement, mutable aliases and subclasses. After each successful check, mutate fake tensor bytes through ordinary access and `.data` without changing pointer/version; every subsequent same-boundary check must reread bytes and reject. Any mismatch stops the correction.

The risks are admitting a partition before its full predicates pass, accepting a capsule in the wrong role, changing typed framing, or accidentally retaining tensor facts. **Both live integrity boundaries, all448/roles checks, optimizer/RNG/parity, independent reload and fresh exit remain mandatory.** This correction may still save insufficient time.

After the falsifier passes, the root freezes the changed closure and runs fresh **CPU120 → both mechanics300 → control061 TRAIN1000/300 once**. Require exact first17 replay, complete reload, fresh exit and accepted normal terminal; candidate follows only after control passes. Pass the final closure to the existing evaluator owner.

Source-v4 remains NO-GO: 666 updates, timeout300.155s, no accepted receipt or reusable partial state. No files were edited or native jobs launched.
