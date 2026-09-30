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
