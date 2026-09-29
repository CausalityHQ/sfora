# Full native valid-anchor2000 terminal result

The fixed corrected procedure is **KILL**. Integrity and resource checks pass;
the candidate remains a valid measured mAP@R improvement. It is preserved with
its complete optimizer state and serving weights. The full production SOP and
In-Shop quality, learning-cost and deployed-speed goal remains active and unmet.

| Dataset / split | Matched control R@1 / mAP@R | Corrected candidate R@1 / mAP@R | Training cost | Public latency | Remaining gap / next decisive test |
| --- | --- | --- | --- | --- | --- |
| DeepFashion In-Shop, previously observed official 14,218 query / 12,612 gallery; same full-TRAIN2000, F5 source, seed179032, native FP16 / packed128 | 96.216064% / 81.675230% | 96.307498% / 81.882593% | C/T 2,424.832571 / 2,441.790976 s for128,000 images; 52.787150 / 52.420539 images/s; archival cost ratio1.006994 | Not measured for this candidate | 0.392502 pp below necessary dated96.7%; current strongest-reference and joint speed gates open. Next: one frozen TRAIN gate for uniform late-trajectory weight averaging, with no additional training updates. |

All quality values above were independently replayed on CPU from the complete
saved packed wires. Candidate-minus-control product-cluster95% intervals are
R@1 **+0.091433 pp [-0.083843,+0.262319]** and mAP@R
**+0.207363 pp [+0.057734,+0.360077]**. Query95% intervals are respectively
[-0.077367,+0.260234] and [+0.073562,+0.344282] pp. The unchanged5,000-draw
bootstrap seed179019 conditions on these fixed checkpoints; it does not
estimate training-seed uncertainty. Neither the necessary dated reference
screen nor the requirement for both positive product lower bounds passes.

The TRAIN candidate used25,882 images /3,997 identities,2,000 B64 updates and
recovered all375 original omitted rank updates. All20 original units exited
normally and were collected; no intermediate quality selection or control
retraining occurred. Training peak allocated CUDA was6,560,978,432 bytes,
with no swap and unchanged source, input pixels, frozen prefix and complete
optimizer/scaler/bank/classifier/buffer/RNG continuation. The cost ratio is an
operational comparison with archived control timing, not a fresh paired public
speed result.

Weights-only export exited0 in11.401 s, actual updated nativeCPU qualification
in21.784 s, publicB32 official evaluation in254.080 s, independent saved-wire
CPU audit in13.275 s, and the paired decision in11.213 s. Every unit stayed
within its frozen120/300-second cap and8GiB/no-swap guard. Public evaluation
qualified all26,830 images, exact native top10 ordinals and float32 score bits
for all14,218 queries, and separate-model/native-packed agreement for
the first32 images of each role. A supplemental16.000-second GPU audit then
used the independently reloaded original processor for all64 sentinels and
matched the saved public codes and inverse norms exactly, closing the shared
processor finding. Inference peak allocated CUDA was1,528,136,704
bytes. These are correctness and resource results, not additional model gains
or latency/QPS measurements.

Frozen qualification generation101 extends the unchanged training closure by
one adapter: `4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375`.
The actual manifest contains102 imported-source entries. Export includes
nonpersistent runtime buffers in its whole-model hash and checks finite
weights/bank/classifier/buffers/optimizer state. The unchanged CPU/public
qualification implementation is reused after genuine historical authority
authentication. The separate full/full paired decision never relabels a
control arm or bypasses the old half/full comparison guard.

Evidence:

- [Complete training summary](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/training-summary.json)
- [Actual updated CPU proof](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/qualification/updated-cpu-proof.json)
- [Public official receipt](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/official/receipt.json)
- [Independent CPU audit](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/official/cpu-audit.json)
- [Terminal paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/decision/decision.json), SHA`0727ce389aab093a657b61fa121c7240ef53fd9e84f838e9833bf980d0a314f5`

The original controller completed during the explicitly authorized reload.
The same native conversation/goal was resumed; no training was restarted.
DGX is idle after the original decision was collected. A single result-gated
Opus5.5/Astra critique `e49878044cda4062` completed; both support the fixed
KILL and candidate preservation. Their distinct findings and reconciliation
are recorded below. No duplicate consultation or training was launched.

## Next intervention boundary

Recovered rank supervision improves mAP@R but does not close top1 quality.
That result alone does not establish a representation root cause. Both critics
confirm that this loss consumes detached full-bank positives; paired32×2
sampling would change gradient grouping, identity diversity and refresh timing,
not make current partner vectors interact. Astra accepts that as an untested
bounded experiment; Opus rejects its stated causal rationale and recommends
fixed trajectory averaging. Choose the cheaper generalization intervention,
with no additional training updates, rather than another sampling/loss probe.

Declare one averaging rule prospectively: uniform average of trainable vision
and head tensors at steps1100,1200,...,2000; copy frozen prefix and runtime
buffers exactly. First compare the archived half-TRAIN2000 terminal model with
its averaged model on the existing6,354-query /6,245-gallery TRAIN-held split.
Do not select snapshots or the rule from official scores. A frozen TRAIN gate
requires positive paired product95 R@1 lower bound and nonnegative mAP@R point
effect, actual updated-source/native/public/CPU parity and bounded resources.
Only a survivor licenses the same rule on the preserved corrected full-TRAIN
trajectory, followed promptly by serving qualification and fresh confirmation.
This is a new prospective intervention; earlier fixed KILL labels stay intact.
No averaging job has started at this checkpoint. A separate specification must
freeze authority, cost and execution details before it does.

## Review findings and narrow corrections

- Both new authority entrypoints now reject `-O`, `-OO` and
  `PYTHONOPTIMIZE` before importing model code. The original frozen entry's
  optimized `--help` returned0 (RED); the focused
  `scripts/test_pe_full_valid_anchor_entrypoints.py` rejects all three modes
  for the two entrypoints and two supplemental audits (GREEN).
- Original measured entrypoint snapshots are archived in the qualification
  and decision evidence directories with their original hashes. The shipped
  entrypoints add only the unconditional optimized-mode rejection; their
  normal-mode AST bodies match those exact measured snapshots. The frozen
  remote imported closure and all existing receipts remain unchanged.
- A54.190-second CUDA-hidden/no-swap audit rehashed every physical retained
  `resume.pt`, not only the terminal state. All20 match their original receipts:
  [preservation proof](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/preservation/receipt.json).
- The16.000-second separate original-processor audit qualifies the64 packed
  sentinels and explicitly binds common query/gallery order through the
  identical frozen evaluator, identical protocol SHA and control-source subset:
  [processor/order proof](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/processor/receipt.json).
- The control-cost receipt chain was already authenticated by the unchanged
  historical startup authority and anchored terminal receipt. The hash-bound
  bootstrap implementation matches the stated5,000 draws /179019 seed; no
  historical settings or scientific thresholds were changed.

[Both separately labelled critic results](evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1/dual-critique.json)
are preserved. The research critic's small training-rank-loss observation is
a diagnostic, not a new measured TRAIN AP or proof of a generalization cause.

## Bounded external-reference recheck

The [UNICOM primary paper, ICLR2023](https://arxiv.org/html/2304.05884v1)
Table4 reports official supervised SOP91.2% and In-Shop96.7% R@1 for
ViT-L/14-336. Its large-scale LAION pretraining and dataset-specific supervised
fine-tuning differ from this SigLIP2/F5 recipe; those are published quality
screens, not matched training or latency controls. This check reconfirms the
dated thresholds, not that they exhaust the current literature.

The newer [TIPS primary paper, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a15a2ece7f0663d1ba7db91103ac61c9-Paper-Conference.pdf)
reports UnED-domain retrieval, including SOP/InShop, which must not be
substituted for the frozen official protocols. The [ICLR2025 AE-SVC/SS2D
paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a2370db7c99791ad5d9f3ef48ad6d464-Abstract-Conference.html)
targets foundation-model distribution and descriptor-size tradeoffs; its
abstract does not establish a stronger comparable official In-Shop result.
An exhaustive strongest current protocol-matched reference remains an open
production acceptance item. No SOTA, independent untouched-TEST, SOP-transfer
or matched public speed claim follows from this procedure.
