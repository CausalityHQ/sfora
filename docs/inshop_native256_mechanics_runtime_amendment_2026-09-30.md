# Prospective mechanics runtime amendment — approval pending

The corrected native Large mechanics procedure remains NO-GO under its frozen whole-unit 120-second limit. Original v4 invocation `9e61f1c141a84319aaf6a3ec7f721415` terminated at 120.738 seconds after 17 uninterrupted updates and two exactly matching replay rows. Independent step8 restoration took 21.181 seconds and completed at 114.421 seconds. The complete nine-row replay, second independent final reload/raw-packed calibration, and full uncached exit rehash did not finish.

Measured terminal peak was 6,390,661,120 bytes, all memory events zero and swap zero. The prior mapped-archive retention failure is corrected for the observed restore. These observations do not qualify whole mechanics or model quality.

The proposed change is **only** a prospective 300-second whole-unit cap for mechanics qualification, retaining:

- 8-GiB host memory, zero swap/events, CUDA allocation below 10 GB, both resource locks and complete-unit peaks.
- Every source/authority/CPU admission and serialized-state hash, independent model/optimizer/RNG restoration, exact 17 versus 8+9 replay, final whole/raw-packed reload and complete uncached exit guard.
- Discarded mechanics state, four fresh TRAIN100 endpoints only after both arms qualify, unchanged quality confidence and training-cost gates.
- Existing CPU120/FIT300/TRAIN300/held-export300 caps and the full SOP/InShop quality-and-speed objective.

Implementation would change mechanics `policy()` and its original terminal-duration admission together, update the narrow frozen-contract test pins, then freeze NEW immutable source/launch authorities. It would run one fresh Large mechanics job; So400 remains gated on a full Large pass. No failed v1–v4 invocation, receipt or state would be rescued or reclassified. Runtime beyond 120 seconds would be reported as measured preparation cost, not a training speed win.

No amendment has been applied or launched. The operator's existing instruction to keep the 120s/300s execution caps requires an explicit decision before this prospective change. If declined, the fixed procedure remains closed at its reproducible runtime blocker and the broader product goal stays open.

The exact proposed patch is preserved as `docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-runtime-amendment-proposed-v1.patch.txt` (SHA256 `4a0ecc62115a4af92f5eb599da05fa4d5e62325527bcd8aa0704360043b9c438`). It changes two production lines: mechanics policy duration and mechanics terminal-duration admission. Three test lines update the whole-contract AST pin, expected mechanics duration, and malformed resource-policy negative (301 seconds, since mechanics/train would share 300). The isolated proposal passed the full narrow stdlib check (original session6161 exit0); an independent AST diff confirms only policy/admit_mechanics production definitions change. Active production source remains unchanged, approval pending, native unrun.

## Authorization and prospective v5 boundary

Operator authorization delivered by devbox/main message immediate-1790882142067309118-1991058 permits ONE new Large engineering-only mechanics qualification up to300s. The reviewed patch is applied prospectively; all v1–v4 failures retain their verdicts. Root owns this trivial integration and DGX gate; no additional specialist is needed for the already-reviewed two-line correction. No TRAIN/quality read until complete17 plus independent8+9 and strict final reload pass. Any integrity/resource/parity/runtime failure kills this arm; no retry or cap escalation.

Frozen source root `/home/riomus/runs/sfora-native256-adaptation-source-v5`, execution SHA `d00d7d70ac2af4acb654d2054e28ffd6f0a0c286ad00d86d02869cfa5534b326`, Large authority SHA `5065f464218827ee96aca444af4e08ec8102cc725ae0fa0b5c251db917283fb1`. Original initialized CPU descriptors and all remaining inputs/limits unchanged. Output is new `/home/riomus/runs/sfora-native256-mechanics-large-v5`.

### v5 terminal outcome

Original session10687/invocatione3c02b979367415f8985c2eb69117440 exited0 successfully in189.303s whole service (driver187.290s). Full17 versus serialized8+independent9 and strict final whole/raw-packed reload, uncached exit rehash PASS; state discarded/no quality read. Driver host peak6,548,062,208B/events0/swap0, CUDA6,128,466,432B; median uninterrupted update2.085331s. Raw log and compact pinned receipt are committed beside prior failure evidence. Engineering mechanics GO only; product quality and speed remain unqualified. No second engineering job or TRAIN started under this one-job authorization.

### Prospective sequential So400 v5 gate

Authorization immediate-1790882776853633660-1991058 permits ONE sequential So400 mechanics-only300s gate after Large terminal authentication. Same exact v5 source execution; So400 authority SHA371cfbaa518b9b5df33934087f037bd9cac0d125635d171b0574ad3ba34fb3d7 binds unchanged original fullCPU proof. New output `/home/riomus/runs/sfora-native256-mechanics-so400-v5`. DGX compute/services idle and both nonblocking resource locks checked before launch. Full17+independent8+9/strict final reload/exit and all resource limits remain mandatory; any failure kills this arm. Report paired terminal decision before any freshTRAIN100 launch. No prior failure is reclassified.
