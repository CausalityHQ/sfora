# Saved direct256 checkpoint qualification

Corrected audit source `e9c3b639`, hash
`06fcccb18b75fb14d2488ef3ea442b00273ffe6f78522912a19dea52ab30019f`.
Original DGX unit `sfora-inshop-direct-width-saved-audit-v2`, invocation
`fae0975fc54b4b8e9f9a667804e8eb36`, terminal inactive/MainPID0/exit0/success.
Measured audit wall **11.897309seconds**, within90-second total cap; no
training, optimizer or new quality/evaluation read. Receipt SHA
`c86c47c64d49df087b3db5103637009ac8d5ef22ff382bbe65035c2422f06a61`.

| Saved-checkpoint guard | Native128 | Direct256 |
|---|---|---|
| Frozen pretrained tensors exactly preserved | 195 | 195 |
| Changed unfrozen tensors | Yes | Yes |
| Saved vision/head/classifier parameters finite | Pass | Pass |
| Common64fit-image pixels exact between arms | Pass | Pass |
| Strict nativeFP16 reload reproduces every code/norm | Pass | Pass |
| Normalized variance on common64fit fixture | 0.87019336 | 0.86468881 |
| Effective rank on common64fit fixture | 20.208023 | 22.683987 |

The replacement gross-collapse guard passes: both direct256 metrics≥50%native.
This panel is not the original augmented training-batch geometry comparison;
it cannot recover that lost attestation. The separate recovered100-update
TRAIN quality result remains R@1 +0.802644pp/mAP@R +0.01683096, not repeated here.
Private nativeFP16/FP32-head reload parity is not public256/native-search parity.
Audit time is neither training time nor serving latency. No new p50/p95/p99/QPS.

Version1 failed before GPU forwards in2.175480seconds because the audit used
original FP32 pretrained tensors. The actual native trainer loads FP16 then
expands FP32 before training. A read-only CPU follow-up proved all195frozen
tensors match that precise round trip in both arms. Version2 corrects this
invalid audit baseline; original version1 failure/source and original real100
failed receipt remain preserved. No method, checkpoint, thresholds, data,
precision, deadline or training was retuned. The original real100 execution
gate remains FAILED; original bank-init wall and training-batch geometry
attestation are still missing. No fabricated final candidate receipt.

**GO for optional256 production design review**, with existing128 default.
No format/kernel/default promotion or SOTA claim. A new single consequential
dual critique `c112576b70054d1a` is reviewing whether longer quality confirmation
or an optional exact native/public256 serving pilot should come first; this is
distinct from the completed cached-stage mechanics review. Both reviewers are
read-only, Opus5.5 and GPT-6 Astra, no new GPU jobs. Raw v1/v2 and journal are
archived beside the original real100 evidence. DGX is idle after the audit.
