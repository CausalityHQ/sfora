# SOP representation-ceiling evidence

This directory seals the claim-ineligible, train-only representation diagnosis
run from source commit `4602f2357730916e8863d8875e3a3297282b0662` on the
DGX Spark host. The evaluator used class-disjoint outer splits and never
materialized Stanford Online Products test arrays.

## Frozen inputs

- UNICOM ViT-B/16 source archive SHA-256:
  `6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818`.
- UNICOM ViT-L/14@336px teacher archive SHA-256:
  `1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`.
- Training population: 59,551 rows and 11,318 classes.
- Outer split seeds: 17, 1729, and 65537. Inner ridge penalties:
  `1e-6`, `1e-4`, and `1e-2`.

## Result

`sop-representation-ceiling-v1.json` is 4,580,401 bytes with SHA-256
`a89a09f73661fd64acc666b84732c411cb74b215103dbf7d818ba47c71624f3e`.
Its canonical bytes, loaded-source closure, input identities, partitions,
per-arm aggregates, clustered bounds, and decisions were independently
recomputed after execution.

- Ridge source-to-teacher full-width gain: `0.0008692580286114859` mAP@R.
- One-sided original-class-cluster lower bound: `-0.0003904282392322747`.
- Teacher PCA-128 loss against full teacher: `-0.021379992039264994`.
- One-sided width-comparison lower bound: `-0.022848996060488007`.
- Decision: backbone-quality work warranted; neighborhood sampling not
  warranted; width outcome intermediate; a wider code is not yet warranted.

The successful named systemd unit ran for approximately 32.3 seconds. One
observed process-cgroup sample was 4,633,362,432 bytes; an exact peak was not
retained. Host memory PSI full `avg10` remained `0.00`, no resource stop fired,
and no evaluator process remained afterward. Two earlier launch attempts ended
before science because of a missing source-layout `PYTHONPATH` and an
interactive SSH lifecycle; neither produced a result.

The evidence separates causes but is not a deployment or absolute-SOTA claim.
It shows that the frozen linear head cannot recover the teacher gap and that
representation quality is the next research bottleneck.
