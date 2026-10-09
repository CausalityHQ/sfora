**Conditional GO for source-only implementation; BLOCKED for native freeze or release.** The private grouped-function insertion and private bundle-call substitution are appropriately narrow. The root disposition states the right requirements, but the existing helpers do not automatically provide all of them.

Reviewed committed `54b80524f5d5bdff275cb29ac70870f15f671bd7`, including both complete design documents. The identity evaluator hashes to `95cb8823…`; the qualifier and combined-authority source hashes match freeze-v4. These findings concern the design and existing source, not the separately developing OWN2 implementation.

1. **The preflight must check exact H∪S membership, independently of `known`.**  
   Original `known` includes any required-guard path below the site with one of the historical four basenames; it does not require the exact historical path. See [grouped_md_origin](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_identity_diversity.py:1617). A canonical, guard-pinned `site/other/libcudnn_graph.so.9`, absent from authenticated H/S, therefore satisfies the proposed `M ⊆ K∪S` check.

   `CombinedAuthority.collect()` would eventually reject it, but invokes the genuine hashing collector first. See [collect](/home/rb/worktrees/sfora-positive-causality/scripts/connected_control_native_authority.py:242). That violates the required zero-read rejection order. **Smallest fix:** additionally require `M ⊆ frozen_H_paths ∪ frozen_S_paths` before authentication, preserving the original `known` construction and predicate unchanged. Test the same-basename, guard-pinned impostor specifically.

2. **Current-file identity is insufficient; compare mapped S against frozen identities before authentication.**  
   [`mappings()`](/home/rb/worktrees/sfora-positive-causality/scripts/connected_control_native_authority.py:227) checks mapping device/inode against the current file. Replace an S file with identical bytes and remap it at the same path: those identities agree, yet differ from the admitted identity.

   The existing frozen-identity check occurs in [`check()`](/home/rb/worktrees/sfora-positive-causality/scripts/connected_control_native_authority.py:211), after `validate_runtime()` has read library content. **Smallest fix:** independently retain the authenticated S identities and compare every mapped S member against them before `audit_origins`. Also enforce complete S at that point when `owner.admitted` or the exact library path is mapped. Test both conditions, including `admitted=True` with the library now absent.

3. **“Malformed maps reject” needs an explicit coverage check.**  
   Both [the original witness](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_identity_diversity.py:1597) and [`mappings()`](/home/rb/worktrees/sfora-positive-causality/scripts/connected_control_native_authority.py:229) skip rows whose split does not yield six fields. Consequently, both can agree while ignoring an additional malformed native-looking row.

   This is a source-visible mismatch with the requested malformed-input contract, **not evidence of malformed kernel output in v4**. Require byte-free rejection of malformed native-looking rows before authentication, and include a truncated row alongside otherwise valid G/H/S mappings in the falsifier.

The authority-binding requirement is also a release gate. Capture the original owner/API/owned dictionary and source state immediately after successful [`install()`](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_connected_probe_serving.py:1071). A later callback must compare against those independent bindings before invoking any owner method. Comparing a replacement API with its replacement dictionary would authenticate nothing. Protect the new callback, derived function, mapping method, globals, defaults, attributes and closure cells; extend the existing [active in-memory checks](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_connected_probe_serving.py:1351) without introducing active-scope source reads or strong audit-hook ownership.

The smallest sufficient falsifier is one isolated stdlib harness extracting the real ASTs and exercising the production callback and retained wrappers. Require:

- Legitimate H+S success and ordinary H success when completeness rules allow it; exactly one boundary `audit_origins(legacy)` invocation, fresh inventory paths equal to original M, and only genuine frozen S returned.
- Missing G, mapped physical md, unknown paths—including the guard-pinned impostor—malformed/deleted maps, device/inode replacement, and required-but-missing S: **zero authentication and library-content reads**.
- Forged owner/API, code/global/closure/source/hash bindings, provenance and RECORD mutations: rejection, including transient active-scope mutations before restoration.
- Missing one historical exact-four member: rejection through the retained real wrapper.
- Both exact function AST inverses, the production byte inverse, unchanged original helpers/globals, repeated boundary entry, and weak-owner collection after success and failure.

The existing [falsifier receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-installed-control-serving-v4/grouped-known-set-falsifier.json:12) establishes the set-composition mechanism only. The existing [identity fixture](/home/rb/worktrees/sfora-positive-causality/scripts/test_connected_probe_serving.py:2733) does not exercise this combined grouped-authority path.

V4 remains **FAIL**: 357.822 seconds, 3,699,171,328-byte host peak, zero swap/events, grouped rejection before reference forward, followed by exact-four exit failure. The [original log](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-installed-control-serving-v4/original.log:151) does not identify the failing map conjunct or member. Neither `candidate.so` nor `libnvidia-gpucomp.so` is a proven culprit.

Keep original API ownership, evaluator exit, full uncached exit, both locks and body300/whole1500/exit300/8GiB/noSwap/CUDA<10GB unchanged. Fresh auditing adds unmeasured cost at repeated boundary entries. Passing this boundary establishes no later-exit, parity, speed, SOTA or product qualification.

No edits, tests, repository imports, native execution, SSH, children, consultations or operator messages performed. Root verification and native freezing remain outstanding.
