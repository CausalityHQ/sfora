# Full-TRAIN chunk500 source design (2026-10-15)

Base: exactly `origin/devbox/so400-full-scale-source-20261013` at
`289d18a351a49d680090b8f59275b694d9471294`. All historical/pinned files remain
byte-identical. The four new `_v2` entrypoints copy the reviewed campaign code;
only unit boundaries, v2 namespaces/schema, and receipt `chunk_length` change.
The sole `CHUNK = 500` declaration lives in the v2 trainer, with
`TOTAL_UPDATES % CHUNK == 0` asserted. The staging tool imports it and supplies
the driver's endpoints. Each arm admits exactly four units: 500, 1000, 1500,
2000. Code/output roots and systemd unit names end in v2. v1 receipts cannot
enter a v2 chain, even when their step numbers coincide.

Boundary inventory: trainer CLI choices; layout count/names/completion;
receipt start, step-row span and first-unit checkpoint link; chain endpoint
modulus and walk; run layout, predecessor path, initial/exit chain checks,
update span and receipt start; stage target choices, arm ordering,
predecessor threshold/terminal/chain, and driver endpoint list. Scorer final
step2000 and complete-pair admission remain unchanged, using the generalized
layout/chain checks. The two retained trainer literals `100` admit/restore the
original TRAIN100 pilot; the scorer's `* 100` converts to percentage points.
They are not unit boundaries. Scaler growth interval 2000 is unchanged.

Schedule, augmentation seeds, update math, optimizer/scaler/RNG handling,
independent restore/parity, held-v2 `service_runtime`, lifetime guard,
locks/resources/watchdog, exit rehash and official-read gating are unchanged.
The v2 stage also freezes the unchanged v1 trainer/scorer and test helpers so
the stdlib tests and offline comparison can read the historical definitions.

## Parent-owned native falsifier plan (not executed)

1. Freeze the new source in the separate v2 code/output roots using the v2
   staging tool. Use the exact Large v9r1 seed-179032 source resume, receipt SHA
   `89f0d4a726db59e9e53a29bcfe7032144665b92131310dfe981467834fdc4ebe`, and
   checkpoint SHA from that receipt. Preserve the original pilot closure.
2. After the unchanged detached-driver lifetime/environment admission, launch
   **one** Large v2 training unit 0->500 through `launch train large 500`, then
   admit its original terminal with `admit train large 500`. Retain both GPU
   locks, CUDA/CUBLAS configuration, 8 GiB host/zero swap/<10 GB CUDA caps,
   watchdog, cgroup footers and `RuntimeMaxSec=3600`. Expected updates ~1000 s
   plus source-bank/restore/save/rehash overhead, to be measured below 3600 s.
   Do not run the full driver: it includes subsequent training and official
   scoring. No official read is authorized by this plan.
3. Under the same resource/lock supervision, run the new
   `falsify_siglip2_substrate_full_v2.py` offline with `--v1-chunk500` pointing
   to `/home/riomus/runs/sfora-native256-full-run-v1/large/chunk-0500`,
   `--v2-chunk500` pointing to the new endpoint, and both
   `--v1-terminal-sha256` / `--v2-terminal-sha256` read from parent-admitted
   original terminal descriptors. It verifies receipt/checkpoint SHAs and
   original terminal chains; v1 steps 1..500 come from receipts chunk-0100,
   0200, 0300, 0400 and 0500. It checks each step's loss/CE/rank/gradient norms,
   batch/augmentation/pixels/RGB/schedule/scale, all numerical checkpoint
   components (including optimizer/scaler/counter/CPU and CUDA RNG), with exact
   equality and no tolerance. Never overwrite archived outputs or retry an orphan.

**Parent pre-registered PASS rule (instruction 1792089785229015754-1401517):**
Both endpoints must have exactly the pilot's `PAYLOAD_KEYS`. Removing exactly
`schema` and `identity` must leave exactly `STATE_FIELDS`; the complement of
`STATE_FIELDS` in `pilot.PAYLOAD_KEYS` must be exactly those two metadata keys.
Every numerical component's typed pilot fingerprint and every `STEP_FIELDS`
value for steps 1..500 must match exactly. Canonical identities must match
after deleting exactly `schema` and `code`; any other difference is FAIL.
Receipt schemas must be exactly full-v1 -> full-v2 and v2 `chunk_length` must
be 500. Missing/extra payload components, schema/chunk mismatches, any other
identity difference, numerical fingerprint difference or step-field difference
all fail admission. No additional identity key is whitelisted.

Each endpoint's raw `terminal_state_sha256` is still verified against its own
checkpoint. Cross-version raw hash equality is a reported diagnostic,
`raw_terminal_hash_equal`, with false expected: both terminal and per-step
full-state hashes include schema and identity/code metadata. Per-step
`state_sha256` and timing are excluded from equality for this reason and
because timing is not numerical training state, respectively. Neither raw
cross-version hash inequality nor timing variation overrides the PASS rule.

Expected future paired-campaign saving: 40 -> 8 units, removing 32 repetitions
of measured ~130 s checkpoint overhead: ~4160 s (~1.1 h). This is an efficiency
proposal for the retained So400m Pareto candidate, not new training science or
a measured runtime/quality result. Native execution, official reads, campaign
authorization and integration remain parent-owned.

Source checks: nice'd, 1 GiB-address-space stdlib tests only. They exercise
four-unit chains, v1/v2 rejection, alternate CHUNK boundaries, literal-100
mutations, staged launch/admission gating, archived-row collection and exact
comparison fixtures, and compare unchanged training/guard ASTs. The offline
checkpoint tool itself and all GPU/model/image work remain unrun.
