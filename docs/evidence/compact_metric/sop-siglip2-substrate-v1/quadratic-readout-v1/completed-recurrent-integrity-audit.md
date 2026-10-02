Recommend **one prospective intervention: an invocation-owned immutable encoder proof whose precompiled typed frames replace repeated metadata traversal. Keep tensor hashing fresh and uncached.** This is a plausible engineering correction, not evidence that TRAIN1000 will fit 300 seconds.

The existing pinned API cannot implement this safely without amendment: `fingerprint(value, frozen=None, consumed=None)` caches **tensor facts only**. It has no immutable metadata-subtree API. Caching the current mutable encoder dictionary, replacing it with a SHA string, or passing serialized bytes as an ordinary value would violate the required checks or change the typed fingerprint.

I verified trainer `3639f0d…3124`, primitive `12f0d8d…b3c6`, and original helper `a1684917…b543` against the local sources.

The expensive update graph is:

```text
update
├─ integrity: pre-update
│  ├─ check_encoder
│  ├─ fingerprint(encoder)
│  ├─ fingerprint(buffers / head buffers / means / STATIC)
│  ├─ check_payload → check_encoder
│  └─ check_complement → fingerprint(frozen_tree)
├─ unchanged four microbatch forward/backward operations
├─ unchanged optimizer/scaler/bank update
├─ integrity: post-update — same graph
└─ diagnostic → fingerprint(complete typed payload)
```

Consequently, each completed update performs:

| Work | Count |
|---|---:|
| Complete `check_encoder` calls | 4 |
| Inventory-row loop iterations | 1,792 |
| Complete encoder metadata fingerprint traversals | 5 |
| Complete STATIC fingerprint traversals | 5 |
| Full canonical-feature byte hashes | 2 |
| Complete dynamic payload fingerprint | 1 |

The actual authenticated encoder composition contains **45,678 visited nodes and 1,630,214 framed bytes per traversal**. Reconstructing it from the pinned source proof and export receipt reproduced the accepted typed encoder SHA exactly: `aac9378c…11da`. Thus encoder fingerprinting alone entails **228,390 node visits and 8,151,070 framed bytes per completed update**.

Separately, recorded shapes and dtypes support a lower bound of **102,869,396 tensor bytes hashed per update**, excluding dynamic A/moments/RNG and small diagnostic tensors. Canonical features account for 58,567,680 bytes of that total. These are source-count bounds, **not measured runtime savings**.

The correction should have this exact scope:

1. **Prospectively extend the canonical fingerprint helper**, retaining its existing traversal and framing rules, to emit validated immutable frame segments directly into the enclosing digest. A segment must contain the original dictionary/list/scalar frame stream—not a subtree digest and not a newly framed `bytes` value. Ordinary inputs retain the original behavior. Load the newly pinned helper explicitly; preserve `a1684917…b543` as original provenance without rebinding its globals.

2. **Seal only the tensor-free encoder composition after full original admission.** Run every existing `check_encoder` and composition predicate before sealing: all 448 source facts, ordered roles, original 205 trainable roles, all-frozen export, config/processor/nonpersistent buffers, source/export bindings, accepted proofs and warm-state bindings. Retain an invocation-owned immutable capsule containing the complete frames. Expose no mutable descendants. At every boundary, reject replacement, foreign ownership or conflicting contents.

   This is an explicit amendment to the **in-memory composition and fingerprint interface**: repeated traversal of immutable metadata becomes verification of its exact owned capsule. Persisted checkpoints retain the original complete dictionary representation, and the optimized fingerprint must equal the original helper’s fingerprint of that representation.

3. **Leave live tensor checks unchanged.** Keep both integrity boundaries, fresh full frozen-tree tensor bytes, `.data` rejection, versions/pointers/gradients, complete STATIC checks and dynamic payload hashing. Keep standalone `check_complement`, serialized reload authentication and fresh exit admission uncached. Do not optimize mutable STATIC metadata in this intervention.

Modify the canonical helper, trainer and existing stdlib contract test; prospectively freeze the changed closure and document the composition amendment. Leave the quadratic primitive, mathematical operations, schedules, RNG and scientific recipe unchanged. If changing the operative serializer closure is forbidden, this intervention is **NO-GO under the current API**.

The bounded falsifier should be one stdlib-only differential test, **≤30 seconds and ≤4 MiB of fixtures**, using the existing fake-tensor/AST test approach:

- Require old/new fingerprint equality for complete representative payloads, including tuples versus lists, integer versus string keys, optimizer state and repeated metadata.
- Exercise the production sealing/admission path with every existing encoder mutation witness; each must still fail.
- Reject unknown mutable objects, mutable scalar subclasses, replacement/foreign capsules and conflicting frame bytes. Failed admission must never create a usable capsule.
- Mutate fake tensor bytes in place and through `.data`, including unchanged pointer/version and mutation after a previous check within the same nominal boundary. Subsequent checks must read current bytes and reject; no tensor fact table may survive.
- Verify persisted materialization fingerprints identically with the untouched old helper. Verify final reload and exit use fresh readers.

**Any mismatch or mutation admission falsifies the correction; stop before native qualification.** The main risks are accidentally changing typed framing, sealing only a projection of the proof, allowing mutable aliases, or losing original provenance through helper rebinding. Even a correct implementation may save insufficient time because fresh tensor checks and native computation remain.

After that falsifier passes, freeze the exact future code, requalify CPU120 and both discarded mechanics17-versus8+9 units, then admit fresh control061 TRAIN1000 under the unchanged 300-second cap, with first17 replay, complete reload and normal terminal admission. Candidate follows only after control passes.

Original session89164 remains an engineering failure: 531 logged updates, timeout/TERM, no accepted receipt, reusable partial state, candidate result or quality claim. Preserve that negative evidence and all previously valid Pareto results. No files were edited, native libraries imported, or jobs launched.
