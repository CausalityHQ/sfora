# UniCOM Rank-Finish Confirmation Design

## Purpose

Confirm the promoted seed-0 SmoothAP rank-finish result without tuning the
method on additional outcomes. The confirmation keeps the epoch-4 parent,
descriptor prefix, temperature, batch shape, optimizer state, epoch count, and
quality thresholds unchanged. Only the finish-phase random stream changes.

The seed-0 screen result is immutable evidence with SHA-256
`3a8cf818e66248fa124cbfd6231a17298cf9fb1734ce9dd1b9a47d4274d8b111`.
It contributes one of three registered finish seeds. Seeds 1 and 2 are run from
the same authenticated epoch-4 checkpoint. The finish seed controls the
identity-balanced order, augmentation-worker stream, and CPU/CUDA stochastic
model stream after checkpoint restoration.

## Frozen confirmation

For each of finish seeds 1 and 2, train epochs 5--8 exactly as in the promoted
screen and evaluate the same identity-disjoint development holdout at epoch 8.
Persist a canonical result and an inference-only epoch-8 model artifact. Seed 1
is fixed in advance as the release-candidate model; seed 2 measures robustness
and is never selected as a replacement because of its observed score.

Confirmation passes only when all three seeds satisfy:

- mAP@R delta at least +0.003 from the epoch-4 control;
- Recall@1 delta at least -0.001;
- Recall@10 delta at least -0.001.

The three-seed mean mAP@R delta must also be at least +0.010. No seed is stopped
early from the epoch-6 metric; all registered seed-8 outcomes are collected.
Failure closes this exact finish method without changing a threshold or seed.

## Final readout

Only after confirmation passes, evaluate the fixed seed-1 inference artifact on
the standard In-Shop test split once, paired against the already authenticated
epoch-4 control geometry. The final readout must report mAP@R and Recall@1/10/20/30,
per-query AP@R evidence, exact artifact identities, and deterministic descriptor
checks. It passes the release-quality gate when mAP@R improves by at least
+0.003 and Recall@1 and Recall@10 each decline by no more than 0.001.

The standard-test split has been used historically and is not represented as a
never-seen statistical holdout. The confirmation is therefore suitable for a
transparent benchmark result and release decision, not an undisclosed adaptive
claim. All receipts remain `claim_eligible=false` until independent replication.

## Resource and publication boundary

Run one seed at a time on the DGX because a finish process uses about 88 GB of
allocated CUDA memory. Each process has a two-hour wall limit. Stop if available
host memory falls below 8 GiB, swap grows by more than 256 MiB, or memory PSI
full avg10 remains at least 0.79 for twelve consecutive five-second samples.
The sustained PSI rule deliberately excludes the reproducible sub-minute
checkpoint/page-cache load transient, observed with more than 22 GiB available
and no swap growth; that transient does not predict the stable training
footprint. Nonfinite and progress stops remain fail-closed. Publish with no
replacement. Bind the source commit, screen result, parent checkpoint and
receipt, partition, finish seed, method implementation, model artifact digest,
metrics, elapsed time, and peak CUDA allocation into every result.
