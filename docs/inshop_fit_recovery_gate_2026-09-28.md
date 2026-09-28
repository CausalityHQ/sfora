# Exact converged TRAIN-fit artifact recovery

**Terminal decision: defer this proposed GPU recovery; no encoder ran.**
The proposal below is retained as reviewed scope, not an executable next gate.

The previous tail probe discarded its tensor but recorded its raw float32 SHA.
The checkpoint retains the matching classifier, not the fit embeddings or bank.
This proposal recovers that missing artifact; it does not reopen the rejected
K3/sham training screen, initial conflict gate, worst-positive loss or held data.

CPU preflightbe07d9edac0441f3817e28c746699894 passed with CUDA hidden in
4.80s GNU wall/2313448KiB RSS. It verified original checkpoint/training receipt,
tail receipt, partition,13283 fit-row digest,2004 labels, classifier2004×128,
head128×1024, every fit path and original model/helper hashes. No pixel decode,
encoder or optimizer ran. The retained helper remains byte-identical:
`e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6`.

Proposed implementation: one artifact-only branch in the existing tail probe,
reusing its entire checkpoint/model/fit authority checks and exact export path.
Require the original receipt SHA
`953be9e6eab846f22e3e187f043eebfaf3161af0bbe4a752f6c68987ffb7fb49`.
Export only the same13283 TRAIN fit paths, unchanged eval mode, FP16 vision
autocast, float32 head/normalized descriptors, batch64/workers4/TF32 disabled.
Require shape13283×128, finite float32 values and raw data SHA exactly
`ea78db9abaf5200bc9952caef8fecd5cd7c38dd29f8cd33ad55de00bd8e1371b`.
Only then save a new exclusive NPY file and sidecar receipt; retain file SHA,
raw SHA, fit/checkpoint/source/helper/reference hashes, wall and CUDA peak.
Return BEFORE margin computation or any new statistical/model-selection read.
Existing tail-probe default behavior and historical files stay intact.

CPU-check the writer's exact-match acceptance, deliberate hash-mismatch rejection
before any file write, and refusal to overwrite. One DGX process only after
review reconciliation and writer checks:150s/14GiB host cap/shared GPU lock,
no downloads. The original whole export+tail70.675s and8.7G host peak provide
cost evidence, not a guarantee. Assert CUDA peak<=original1886321152+1GiB.
Any authority/digest/resource failure stops this attempt: no numerical retuning,
alternate checkpoint/precision/batch, retries, relaxed hash or partial acceptance.

Passing supplies the exact retained tensor needed by the previously suggested
static fit-gradient read. It does not itself prove classifier/rank conflict,
encoder harm, K3 relief or any quality improvement. A separately frozen CPU-only
gradient read must distinguish eval normalized descriptors/fresh gallery from
the unavailable stale training bank and BF16 augmented training activations.
No automatic training, official evaluation or new loss follows from recovery.
The full joint SOP/In-Shop quality and image-to-top-k speed goal remains unmet.

## Review reconciliation and terminal evidence

Dual62c6e60d326d4554 completed normally in111s, no fallback. Opus5.5
2e5c3dcb245347d6 in93s rejects the scope; Astra278902c83ff2473c in111s
conditionally approves exact artifact acquisition but requires persistence and
authority checks before any GPU launch. Full separately labelled reviews and
CPU evidence are retained in the
[archive](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-recovery-v1/dual-result.json).

Opus's utility objection is decisive: no separately frozen CPU read has an
action that its outcome can open, and the rejected K3/sham training screen
cannot supply that action. Acquiring a tensor merely to feed another undefined
probe does not meet the user's stop rule. Its warning against25% floor reuse is
also valid across different spaces/profiles; the earlier initialization screen
uses training activations, while these eval descriptors cannot reconstruct the
historical augmented BF16/stale-bank training gradient. Do not turn this deferral
into a negative experiment or a universal closure of classifier geometry.

Opus also asks for written lifting of the earlier export stop. That is a review
recommendation, not a new operator-permission requirement: the user authorized
autonomous bounded work. The prior research did not authorize export, and the
new scope still lacks decision-bearing utility. No permission request is needed
to defer it. Astra's conditional approval is not unanimous approval to launch.

While reviews ran, an unshipped optional writer branch passed CPU checks for
exact-match saving, deliberate hash-mismatch rejection before writes and NPY
overwrite refusal; it returned before margin computation. CPU writer invocation
bf90910611a640d5bb11dd833ee9a2cd passed with CUDA hidden. Final prototype also
passed focused Ruff format/lint and its self-check. Astra read the original
probe before this implementation and therefore described the branch as missing;
that finding was stale by reconciliation. Its additional sidecar/read-back
requirements remained unmet, so the prototype was not launch-ready regardless.
The prototype is archived only; all its uncommitted script changes were removed,
restoring the original probe SHA1f655cebb7155c1b980bb5222ac3894b582f916f4c9c85a2cc203f14d1978b69.

Both CPU jobs and reviews are terminal. No recovery GPU unit was launched,
no NPY or matching fit artifact was produced, and no held/official data was
encoded or scored. The preflight establishes source availability only; it does
not prove repeat numerical reproducibility or supply a training mechanism.
Next quality work must have a causally distinct supported mechanism and a
concrete decision rule before paying for data acquisition or training. Existing
representation/scale/loss gates stay closed; official quality and speed claims
are unchanged. Preserve the full joint goal as unmet.
