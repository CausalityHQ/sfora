# SOP/In-Shop resource serialization audit 276

Date: 2026-08-03. Repaired while the corrected SOP reference was at epoch 32,
before SOP final artifacts existed and before the queued In-Shop reference
created a report, checkpoint, or GPU process.

## Defect

The original In-Shop controller waited for the SOP final train/test artifact
audits, then launched immediately. The independently preregistered SOP
fragmentation diagnostic also launched after those audits. Its global
leave-one-out retrieval computation processes 59,551 training embeddings in
512-row chunks, roughly 1.82 trillion multiply-accumulates on CPU (about 3.63
TFLOP under the usual two-operations convention). Running
it concurrently with In-Shop would compete with eight image-loading and
augmentation workers, reducing or destabilising training throughput for a
reason unrelated to either recipe.

This is a scheduling/evidence-path defect. It does not change a loss, prediction,
or method result, and no contaminated In-Shop run existed to retract.

## Repair

`scripts/run_inshop_after_sop_evidence.sh` now waits for all four artifacts:

1. the SOP final-test verification;
2. the joint train/test/checkpoint/report verification;
3. the locked SOP fragmentation result;
4. the independent fragmentation support verifier.

It requires both verifier payloads to report `status="verified"` before launching
the unchanged official In-Shop Proxy Anchor command. It deliberately does not
require the fragmentation hypothesis to pass: scientific outcome cannot decide
whether a separately preregistered reference benchmark runs.

The liveness check treats the fragmentation verifier as a valid producer only
after its input result exists, so a waiting verifier controller cannot mask a
failed fragmentation producer indefinitely. The superseded waiting controller
was terminated only after read-only checks proved that neither In-Shop output
artifact existed. The replacement controller PID was `3137282`; the only GPU
process remained corrected SOP trainer PID `3108066`.

## Boundary

Serialization removes known CPU contention; it does not prove bitwise hardware
determinism or guarantee that future unrelated host load is absent. The In-Shop
report retains its ordinary timing and artifact audits. Candidate generation
must use the verified scientific outputs, never throughput changes caused by
this operational repair.
