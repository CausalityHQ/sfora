# InShop nearest-ranking encoder trial — 2026-10-03

Status: prospective bounded procedure. No native qualification, training, quality gain or production release claimed. SAME SOP+InShop quality-and-speed goal remains unmet.

## Decision and review provenance

Proceed with one matched final-MLP encoder trial after its technical gates pass. The original plan is nearest-ranking-plan-6808efe4c21a4f52.md; dual critique f64090982b394ae0 completed both legs exit0. Research leg fe7bbb3055494bd2 used the recorded GPT6.1Sol fallback after Opus5.5 OAuth failure; engineering leg38d8feb4c3184a0d completed under Astra. Their full separate reports are committed alongside the plan.

Root independently verified the detach/frozen-A mismatch, original205-trainable factory, immutable encoder descriptor, and TRAIN6355/1008 including12 singleton classes. Lack of cached-separability evidence does not prove inseparability or identify the encoder as the cause.

Accepted concat selection R1/mAP@R is96.48212226066897%/81.77754035543956%; matched linear96.42445213379469%/81.56035338692806%. Concat's R1 gate failed; preserve its improved development head and all prior verdicts. This new comparison tests adding nearest-impostor supervision to four-tensor encoder adaptation.

## Frozen method

Both arms initialize from accepted concat resume.pt SHA b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf and full typed state a118fd98cce0b8fafa51c897be70b2b6e2ec93ebb226b2382dd264721776b644. Original source vision, warm061 head/classifier, fitted coefficients/means, processor, complete source authority and TRAIN rows remain authenticated dependencies.

Only encoder.layers.26.mlp.fc1.weight[4304,1152], fc1.bias[4304], fc2.weight[1152,4304], fc2.bias[1152] are trainable:9,921,872 FP32 scalars. Freeze all parameters first, then enable exactly these four before optimizer construction. The other444 vision parameters, complete head/classifier/concat A/means, configuration and persistent/nonpersistent buffers remain unchanged. Frozen downstream operations must propagate input gradients.

Use canonical native256 TRAIN images only, pinned processor, no augmentation. Teacher descriptorsT are accepted concat outputs reconstructed from the normalized canonical TRAIN cache with its original CPU normalization/readout sequence. Do not reuse stored pre-correction prototypes. V=normalize(T), P[class]=member-inclusive meanT, e0=mean_images(sum128coordinates((T-P[target])**2)); require finite positivee0. Teachers and original-row mappings are immutable saved state, not regenerated from updated encoder.

Control objective: mean_images(sum_coordinates((raw-P[target])**2))/e0.
Candidate adds mean_valid_anchors(relu(.05 + dot(unit,Vnegative)-dot(unit,Vpositive)))/.05.
Positive is highest-cosine OTHER original image of the same identity; negative is highest-cosine different-identity TRAIN image among all6355 teacher rows. Detached indices/teachers, connected current anchor. Exact ties prefer ascending original-row ID. Singleton anchors receive regression only. Invalid positive indices must never be indexed.

For B64/micro16 accumulation, regression microbatch sums divide by64; ranking hinge sums divide by FULL B64 valid-anchor count and .05. All-invalid batches contribute exactly zero ranking. Both arms execute the same mining/diagnostics; only candidate ranking contributes backward.

Fresh four-member AdamW: lr1e-5, betas(.9,.999), eps1e-8, weight_decay.05, amsgradFalse, foreachFalse, fusedFalse, defaults/original parameter names/order pinned. Global clipping1. FP32 params/gradients/moments; FP16 vision autocast, FP32 pooled/readout/loss, initial scaler128, source CUDA flags/workspace and eval-mode vision. Reject nonfinite values and skipped scaler updates.

Exactly32 updates/B64=2048 presentations per fresh arm; regenerate and authenticate first32 batches of original genuine TRAIN seed179061. No margin/LR/precision/boundary/step-count rescue.

## Corrected readiness and endpoint contracts

Procedure-owned nearest_ranking_readout.py preserves existing concat operation order with connected inputs and frozenA. Original helper remains unchanged. Oracle comparison uses isolated same-value A copy satisfying the old trainable-A validator. Require exact forward parity within identical input/device/precision roles and complete native training/inference path parity. CPU teacher and nativeFP16 micro16 contexts are separate roles; record zero-update TRAIN witness drift rather than claiming cross-role bit identity.

Native witness must show finite nonzero gradients and actual changes confined to all four tensors, and separately a nonzero ranking-gradient contribution for an active valid hinge. MSE-only nonzero total gradients are insufficient evidence of ranking activity. Objective activity is resolved in required mechanics; an empty initial teacher-triplet panel alone is not a KILL.

New payload owns complete updated vision state and separate original provenance. Bind all448 parameters, configuration and nonpersistent position buffer; strict independent reload must load trained vision. Substituting original vision must fail authentication. Save teachers/P/e0, original partition/rows/targets/schedule, frozen head/classifier/A/means, optimizer moments and name/order, scaler/counters, CPU/CUDA RNG and numerical flags. Resume restores saved teachers. Original immutable-encoder endpoint schema cannot certify this updated model.

Use one model at a time; release old model/archive/mmap-backed optimizer scalars before independent construction. Full state/current-byte checks, typed fingerprints, same-size restored-mtime/data-bypass mutation rejection and uncached exit checks remain mandatory. Any amortized immutable-complement fingerprint requires explicit actual-byte ownership and mutation witnesses; no stat/hash shortcut or original helper-global mutation.

Inference artifact must load updated vision+fixedconcat without TRAIN cache/teacher dependencies and produce packed128 outputs. Actual public serving remains a later qualified product path.

## Units, costs and stop rules

All native/Torch/CPU qualification/GPU work on authenticated DGX Spark only. No AWS GPU. One original native unit at a time, both lifetime locks, host<=8GiB, zero swap/disallowed events, CUDA whole-unit allocated<10GB, original terminal receipts. Source/code/inputs/output exclusivity/stop rule frozen before each launch. Cold start/cache policy and complete setup/save/reload/exit work included; no retroactive relabeling of old failures.

1. One prospective CPU engineering qualification<=500s, CUDA hidden, no quality scoring. Source/layout/teacher/miner reductions, exact forward oracle, malformed-state/role mutation rejection and complete serialization contracts.
2. Sequential discarded mechanics<=300s per arm: uninterrupted17 versus independent8+9 with exact state/RNG/schedule/optimizer replay, gradient/activity witnesses, strict final reload/raw-unit-packed parity and all exit checks. No state reuse for TRAIN.
3. Only paired mechanicsPASS permits fresh control32 then candidate32, <=300s each. Measure phase costs including serialization/reload/exit. Both candidate/control whole-service and total declared training-core ratios<=1.50. Core includes target construction, mining, decoding, forward/backward and optimizer; report shared preparation separately. Control also mines, so cost comparison is instrumentation matched. Historical ridge timing is not denominator.
4. Independently reload valid/cost-qualified endpoints, then frozen TRAIN diagnostic and one fresh selection image export/score. Never evaluate updated encoder through old feature caches.

Freeze at most128 initially active teacher triples in original-row order. Re-encode all three roles through each updated encoder, report violation counts and paired margin changes. This is diagnostic only: failure means mechanism not demonstrated on this panel, not a utility veto. No remine/tuning after outcome. Integrity/nonfinite/gradient/resource failures remain hard stops.

Selection remains previously observed TRAIN1734queries/1715gallery/498products. Replay original source wires/metrics, independently freshly encode both updated models, verify raw/unit/packed/wire readback before candidate quality. Both candidate-control gains>=.002fraction, both paired-product95% lower bounds>0; shared5000draws/seed179019 and query intervals. Preserve all original immediate failures/source floors; candidateR1 must exceed96.48212226066897% and mAP>=81.77754035543956%. No threshold adjustment. Only selectionGO admits sealed validation; official SOP/InShop and matched end-to-end public speed remain subsequent mandatory gates.

A negative result closes this procedure only. Preserve valid control/candidate/Pareto artifacts. No external SOTA, serving latency or sufficient convergence claim follows from this32-update trial.

