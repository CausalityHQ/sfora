# Fixed saved-state held attribution, no training

CPU fit-only counterfactual shows PE ahead at native1024 and PCA128, losing less
quality to PCA than Large. This does not identify the trained held reversal.
Resolve the missing step-0 and encoder/readout contributions on the SAME existing
TRAIN-held roles, using original and already saved100-update parameter states.
The100-update recipe remains STOP; no continuation, fit, optimizer or loss tweak.

Pre-result review ruling: primary reads are each architecture's initial00 quality
and total11−00 change, and PE-minus-Large initial and total differences, for
Recall@1 and mAP@R. Component/interactions and other mixed-state comparisons
are descriptive diagnostics; their marginal95% intervals are not simultaneous
or multiplicity-corrected evidence. Head0 was fitted to encoder0: a mismatch
between encoder100 and head0 may dominate an interaction and cannot identify
a training-causal mechanism. Declare one secondary CPU read after golden parity:
score raw1024 slices for original/final encoders on identical held roles, with
paired uncertainty, to inspect encoder quality independently of the compact
readout. No selection, additional image passes, fitting, widths or training.

Fixed2x2 per architecture: vision0/head0, vision0/head100, vision100/head0,
vision100/head100. Encoder factor includes native pool/projection; readout factor
is Linear1024→128. Four encoder image passes total produce both heads each;
eight predeclared states, no best-state/checkpoint selection or extra variants.
Original FP32 weights and final trusted checkpoint bytes, nativeFP16/TF32off,
unaugmented native224/256 inputs, batch32, all original held file hashes and
6,354query/6,245gallery roles. Both heads stay immutable and inference-only.

CPU preflight authenticates priorSTOP receipt/preflight/init/audit and both
checkpoint/model SHAs, heads/shape/finiteness, executing paths/code authority;
freeze external preflight digest beforeGPU. Reject existing/partial output and
reserve one exclusive attempt. One DGXshared-lock user service600s/8GiBhost,
allocatedCUDA<10GB, cleanup<8MiB after every encoder state. Preserve original
training/artifacts; no cap/dtype/floor rescue or overlapping job. Calibration
first4held raw FP16/F32 cosine≥.999, logbeforeassert; all vectors finite/unit,
no parameter changes. Structured matrices pack1024raw+128head0+128head100 per
row; finalvision/finalhead must reproduce saved100held vectors (cos≥.999999)
and all per-query packed hits/AP within1e-6 before attribution is accepted.

CPU packed scoring/stableties/mAP@R and paired product/query bootstrap asbefore.
For each metric and architecture report initial→final total, encoder effect with
head0, head effect with encoder0, and interaction (11−10−01+00), plus differences
between architectures; record all8states evenifearlycaseisnegative. Exact
algebra requires total=encoder+head+interaction. These are saved-component
counterfactuals with coadaptation, not training-causal interventions or evidence
a hybrid generalizes. No official/SOP/CUB/Cars/serving inference. Runtime/source/
parameter/score/role/cost integrity failures stop diagnosis without retry.

Interpretation: determine whether the reversal is present atheldstep0 or emerges
in relative encoder/readout changes/coadaptation. Retain uncertainty and any
ambiguity. Choose ONE responsible-layer next hypothesis only after terminal
parity/data audit; no automatic training of a preferred mixed state and no
reopening the closed recipe. Full production joint goal remains unmet.
