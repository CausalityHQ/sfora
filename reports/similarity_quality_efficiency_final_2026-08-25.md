# Similarity-learning quality--efficiency result

## Decision

The strongest supported result in this research line is **class-proxy
imprinting for a pretrained UniCOM backbone**.  On the official In-Shop
query/gallery protocol it improves the strongest matched random-proxy baseline
by **+2.648 mAP@R points** and **+1.404 Recall@1 points** over five prospective
paired seeds.  Every paired delta is positive, the paired 95% intervals exclude
zero, and all five candidate runs reach the baseline's epoch-16 mAP@R by epoch
8.  Deployment inference, architecture, and storage are unchanged.

This is a statistically supported matched-baseline quality--efficiency
improvement across prospective paired seeds.  The official and holdout
readouts reuse the same trained checkpoints; no independent re-training has
been performed.  It is
**not a global state-of-the-art result**: the candidate's mean official In-Shop
Recall@1 is 95.15%, below UniCOM's published 96.7% anchor under its longer,
larger-scale recipe.  The evidence is suitable for a paper about a cheap,
training-only initialization improvement and its reproducibility boundary, but
not for a paper whose headline claims SOTA.

The Cars196 experiment below validates a separate established intervention,
Proxy Anchor EMA self-distillation (`pa_distill`).  It must not be described as
a cross-dataset replication of proxy imprinting.

## Method: class-proxy imprinting

The pretrained UniCOM encoder embeds every training image.  For each training
identity, the method averages its normalized embeddings, normalizes the class
mean, and norm-matches it to the random proxy initializer.  It consumes the
same initialization RNG stream and restores every RNG domain before training.
Both arms then use the same images, batches, loss, optimizer, schedule, epochs,
encoder, and deployment path.  The only intervention is the initial class-proxy
tensor.

This makes the gain operationally useful in two ways:

1. training starts from class geometry already present in the pretrained
   embedding instead of relearning proxy locations from random vectors;
2. the one-time feature pass is absent from deployment, so inference and model
   storage do not change.

## Official UniCOM/In-Shop evidence

The primary evidence is the strict artifact
`reports/generated/unicom_ema_imprint_official_426afa4.json`, SHA-256
`a67371bcf3f727ab39cec358c66552287503d46c4442dfc1de5e6d1d25ca5b24`.
The evaluator source is commit
`426afa464c7c32e7adbba81d29a6777cae9ed972`; the frozen run-config commit is
`367d319535a8c368885b98ccc8f80ec59070a831`.

| Metric | Matched result |
| --- | ---: |
| Prospective seeds | 5 paired seeds (2--6) |
| Official mAP@R delta | +0.026484 (+2.648 points) |
| Paired Student-t 95% interval | [+0.024766, +0.028202] |
| Positive mAP@R pairs | 5/5 |
| Official Recall@1 delta | +0.014039 (+1.404 points) |
| Paired Student-t 95% interval | [+0.012904, +0.015173] |
| Candidate mean Recall@1 | 0.951456 (95.15%) |
| Candidate best-seed Recall@1 | 0.952033 (95.20%) |
| Epoch-to-baseline-quality | by epoch 8 on 5/5 seeds |

Across the preregistered sensitivity seed plus the five gating seeds, the
six-seed mAP@R interval is `[+0.024814, +0.027681]`.  The same six training
pairs, evaluated on the earlier fixed train/holdout split, give a mean mAP@R
delta of +0.017578, a paired 95% interval `[+0.014804, +0.020353]`, and 6/6
positive seeds.  This is a second readout of the same checkpoints, not an
independent replication.

### Quality--cost frontier (train/holdout protocol)

- On the fixed train/holdout protocol, compute to matched quality is
  **30.8%--73.1% lower** on the prospective training-gate seeds.
- On that same protocol, full 16-epoch compute is **1.7%--2.2% higher** because
  imprinting adds one feature pass.
- Deployment storage is unchanged at 3,632,816,144 bytes; paired checkpoint
  storage is equal.
- Registered batch-128 inference is approximately 11.96--12.11 ms/image with
  the same architecture and code path.
- The profiled fusible non-backbone fraction is about 0.046%, far below the 10%
  kernel threshold.  A custom CUDA kernel cannot materially improve this
  candidate's time-to-quality frontier.

The exact registered command from the frozen config checkout was:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PYTHONHASHSEED=0 \
.venv/bin/python -I -B scripts/evaluate_unicom_ema_imprint_official.py \
  --config docs/unicom_ema_imprint_official_run_config.json
```

The detailed seed table, checkpoint inventory, resource accounting, and
validator evidence are in
`reports/unicom_ema_imprint_official_result_2026-08-23.md` and
`reports/unicom_ema_imprint_replication_result_2026-08-22.md`.

## Cars196 confirmation: EMA self-distillation

Cars196 tested the already-established `pa_distill` intervention against the
exact Proxy Anchor reference recipe.  The preregistration is Git object
`f8078b1:docs/cars_pa_distillation_confirmation_2026-08-24.md`; the execution
source is commit `b5e949f75d8eaf1f0198bcaea48413db92f3f99b`, with the residual
CUDA nondeterminism disclosure at commit
`0fad150`.

- Baseline recipe digest:
  `d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a`.
- Candidate recipe digest:
  `080a45b8c14460d43b6f5f1d352f10854adb0d6c8d434fc6d2f02f2dbd501b02`.
- Seeds: exactly 0--5, seed-major, baseline then candidate.
- Protocol: Cars196 official class-disjoint split, ResNet-50, 512 dimensions,
  Proxy Anchor, 60 epochs, 4,080 steps.

The confirmatory final-epoch Recall@1 result is:

| Metric | Value |
| --- | ---: |
| Baseline mean Recall@1 | 0.863321 |
| Candidate mean Recall@1 | 0.876768 |
| Mean paired delta | +0.013446 (+1.345 points) |
| Positive pairs | 6/6 |
| Paired t statistic / p value | 10.2862 / 0.0001493 |
| Paired Student-t 95% interval | [+1.009, +1.681] points |
| Registered bootstrap 95% interval | [+1.125, +1.576] points |
| Exact two-sided sign-test p value | 0.03125 |
| Registered decision | `CONFIRMED` |

The six registered final-epoch deltas, in seed order 0--5, are
`[1.217562, 1.684910, 0.910097, 1.758701, 1.279055, 1.217562]` points;
their sample standard deviation is `0.320205` points.  The registered point
prediction was `[+0.5, +1.0]` points; the observed +1.345-point mean exceeded
that interval without changing the preregistered `CONFIRMED` decision rule.
The registered best-over-training sensitivity deltas are
`[0.639528, 0.565736, 1.266757, 1.008486, 0.787111, 0.762514]` points.

As an **unregistered secondary analysis recomputed post hoc** from the twelve
registered artifacts, final-epoch mAP@R improves from 0.297294 to 0.303599: +0.006306
(+0.631 points), 6/6 positive, paired p=0.001335, paired 95% interval
approximately `[+0.379, +0.882]` points.  The best-over-training Recall@1
sensitivity improves by +0.838 points (6/6, p=0.000512).  Best-over-training
mAP@R is not significant (+0.244 points, 4/6, p=0.2915) and is not a supported
claim.

In another **unregistered secondary analysis recomputed post hoc**, the
candidate's mean wall time was 4,977 seconds versus 4,062.17 seconds for the
baseline, a descriptive ratio of 1.2252 (**22.52% slower full training**).
Thus this experiment supports quality, not training speed.  The EMA teacher is
training-only, so deployed inference and storage are unchanged.

The frozen launcher command was:

```bash
nohup bash scripts/run_priority_queue_v47.sh \
  > /home/riomus/experiment-logs/reference-matrix/cars-pa-distill-v47.nohup.log \
  2>&1 &
```

Each arm was executed by that launcher as:

```bash
.venv/bin/sfora image-end-to-end \
  --dataset-name cars --objectives proxy_anchor \
  --recipe <auto-or-pa_distill> --num-workers 8 \
  --seed <0..5> --deterministic --output <registered-artifact>
```

The controller and nohup logs have SHA-256
`0f6e8aa10ff133fcfa8df155541b30f6e37252ae3f9c0d6deaa32ab5b6d5079e`
and `81b177d06ac1267b35ac90647395d025d41f60cca8ae164992ad9a9c7149b7a0`.
Their byte-identical local copies, together with all twelve result artifacts,
are under
`reports/generated/cars-pa-distill-confirmation-2026-08-24/`.
The exact artifact paths are
`reports/generated/cars-pa-distill-confirmation-2026-08-24/{proxy_anchor.d55241a64a5a,pa_distill.080a45b8c144}.deterministic.seed{0..5}.json`.
Candidate seed-0--5 artifact hashes are, in order:

```text
dd7a3d593c1bebcb2607b4ed91f468b1e9961a706811ed80922f1d6b7f51283d
cdd4c85ca55c00aefb240614dcc69b55592e6a4e454143c0de4230ef416af07a
572f367c6740b179a5cc4b30951a122fbaf1f17cd34876fba31b42a4e13385f7
8074ef2f8ee396c35c60ecb6d682bee6bf565f2eb53c1a6dd07d6ee496c0118b
30e41256fff2821f0a1be7707e5dac5aff248f4dfebdf715e7009e1bcd3f3450
e34027de77dd53b140e715620a47cbd26b4ed7dcfd43d02f956c63795576e186
```

Baseline seed-0--5 artifact hashes are, in order:

```text
8937d91322153fd44645f141afdb43607449bd4f616b7f0bf12a9fff64c089eb
50eca4b8bd203aaf1efa54095ce0d90545eafe96a8b83bc4a164b07001f7d2ea
e8a83ae6d08885ffdbebfa259a94f04346236c04cf126c0a03ee2dfcb0f8bb77
e468d5e85468abbcf7f199fbdcde99782776e046c3da9a3652997b6cab1172da
811b6fb8402cabaca1cbe60b7259f8927fba72ed37498ba5b929ba90b4978d14
ed4821369eee583a010ac73f9723dfc2c28684a7f18788089ffba9364b0a9162
```

PyTorch warned that `adaptive_max_pool2d_backward_cuda` has no deterministic
CUDA implementation.  `deterministic=true` fixed supported algorithms and the
cuBLAS workspace but did not make that operation bitwise deterministic.  The
Cars result is therefore a paired six-seed estimate, not a bitwise-replay claim.
The artifacts bind the exact recipe digests, seeds, deterministic flag, and
completed objective, but they do not embed the repository source commit; the
launcher is preserved at `b5e949f`, while the mutable DGX worktree is not an
independent source-identity authority.  This is a reproducibility limitation of
the Cars companion result and another reason it is not the primary publication
claim.

## CAP F0 publication-failure closure

The UniCOM covariance-adjusted-prototype (CAP) screen completed its scientific
computation but did not publish a scientific result.  The receipt's
`stage=publication` is set only after `run()` returns the assembled result;
the observed `ValueError` therefore arose during strict result validation or
non-finite-value serialization.  No candidate value was persisted, recovered,
or inferred.  The receipt contents are outcome-blind, but the occurrence of a
publication failure after computation is not a scientific null.

Two scientific processes were run.  Attempt 1 (source `77a092f2`, handoff
`bd954fbce3bb675c8f0840c1d8a75b8c170ae0e4`) exited 2 after candidate
computation began because the original validator required two FP64 reduction
orders to be bit-identical.  Attempt 2 (source `515e8cd`, handoff `33827f9`)
first ran two candidate-free parent replays.  Both passed, are retained under
`reports/generated/unicom-cap-f0-515e8cd-evidence/`, and are byte-identical
(SHA-256
`d476e59b8506bdb68384e6673531cf958a4d50a7cd264a646f699437906baef7`).
They reproduce the frozen tensor hashes:

```text
d183c0d26d451cc5184f4da0a2112766fb5b32d206ea711011f573b3b4aa9613
bfabb3159677577cf8e6489a40b4765c4510c07a0c18e9094443a01de4cf244b
a56392a806fcf028876a0d1933c0095a7e20aad46cbb8f84f8c8d96d8468e8cd
c1fe4cb49668e9b02796ca2fe48432518174cb3495cb1970d7e26ee3a187fd8f
```

and the frozen metric hashes:

```text
5de610b1d6038a18b51221fd88280c00cbd5d11701ac31830877f9b3284e8be0
9505bf5ba965b04d6bad39896e8c4a442a46791b9b53c6ab426bd83e83532a9b
889bb182ae2f2ceb14f6e35122079f141df1af87354ac8fbf7c5d6927ecb1e4f
196f82dea9e9699df8e5efd08ab3ab0fa3923bd36ea793d46bd2cc66c5740025
```

Attempt 2 then exited 2 and published only the outcome-blind failure receipt
`reports/generated/unicom-cap-f0-515e8cd-attempt2-failure.json`.

The attempt-2 command in the detached handoff checkout was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/home/riomus/group-learning/.venv/bin/python -B \
  scripts/screen_unicom_cap_f0.py \
  --config docs/unicom_cap_f0_run_config.json \
  --unicom-checkout /home/riomus/UniCOM \
  --checkpoint /home/riomus/checkpoints/FP16-ViT-L-14-336px.pt \
  --dataset-root "/home/riomus/datasets/In-shop Clothes Retrieval Benchmark" \
  --parent-result reports/generated/unicom-spherical-probe-ed2e789.json \
  --output reports/generated/unicom-cap-f0-515e8cd.json
```

Each candidate-free replay used the same command plus
`--parent-replay-only`; the two processes were sequential and fresh.

The receipt has SHA-256
`8b636848cd81c8e8cb3dc6f84aff06dd3758437e7c887acc082f5707127660b1`
and production strict validation passes.  It binds source
`515e8cdfde3a6b8c0449554878ed391e6a1edf3d`, handoff
`33827f95b5076acb8139dbea97f5d65bb7adac73`, attempt 2, stage
`publication`, error code `unexpected_exception`, built-in exception type
`ValueError`, and `result_published=false`.

The two replay payloads and attempt-2 stdout/stderr are preserved byte-for-byte
under `reports/generated/unicom-cap-f0-515e8cd-evidence/`.  Attempt stderr has
SHA-256 `68fdd73bb4c104a38a3eb57bf9a44aae29a4aee409ab6160d45f0eb9ab623def`;
stdout is empty with the standard empty-file SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The temporary DGX checkout was cleaned before the receipt was transferred.
The local reconstruction is byte-canonical under the frozen writer and is the
exact 420-byte payload that writer emits for the observed fields; production
`strict_json_object` and `validate_failure_receipt` both pass.  The original
DGX file was removed before transfer, and no independent durable hash record of
that original was retained.  The recovery authority forbids a third attempt,
so CAP F0 is permanently closed as a publication failure.  This failure does
not support or refute CAP scientifically and does not change the imprinting or
Cars results.

## Reproducibility and claim boundary

Supported:

- proxy imprinting improves the fixed official UniCOM/In-Shop matched baseline
  with statistically supported quality and faster time-to-baseline-quality;
- deployment inference and storage are unchanged;
- PA EMA self-distillation improves final Cars196 quality over its exact matched
  Proxy Anchor baseline across six paired seeds.

Not supported:

- global SOTA;
- a cross-dataset proxy-imprinting claim (Cars tested a different method);
- bitwise deterministic Cars training;
- any scientific CAP conclusion;
- a useful custom-kernel opportunity in the measured UniCOM path.

The appropriate publication claim is therefore: **a prospectively evaluated,
training-only class-proxy initialization improves a strong matched UniCOM
baseline's quality and time-to-quality without deployment cost across paired
seeds**.  A global
SOTA headline requires a new, same-backbone comparison against the current
published frontier and a prospective proxy-imprinting replication on another
dataset.
