# UniCOM classifier-imprinting replication result

## Outcome

The preregistered six-seed replication supports the selected `imprinted_raw`
candidate on the fixed In-Shop train/holdout protocol. The strict summary reports
`quality_claim_supported=true`, all registered trajectory/resource predicates true,
and `claim_supported=true`.

This is evidence for a better quality/convergence operating point than the matched
current-lineage random-initialization baseline. It is **not** an official In-Shop test
result and does not establish global state of the art.

## Method

UniCOM is the pretrained embedding backbone, not the new contribution. The intervention
changes only the initial class-proxy tensor. Instead of random proxies, it averages the
backbone's normalized embeddings for each optimization identity, normalizes each class
mean, and norm-matches the result to the random initializer. The implementation consumes
the same random-initialization stream and restores all RNG domains before training. Both
arms then use the same data partition, batches, loss, optimizer, schedule, epochs, and
model. Deployment remains byte-size equivalent and uses the same inference path.

## Registered quality result

| Seed | Random mAP@R | Imprinted mAP@R | Delta | Random R@1 | Imprinted R@1 | Delta | Epoch speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.902848 | 0.917281 | +0.014433 | 0.983689 | 0.992472 | +0.008783 | 1.5x |
| 2 | 0.899012 | 0.918503 | +0.019491 | 0.983689 | 0.992472 | +0.008783 | 4.0x |
| 3 | 0.898221 | 0.915735 | +0.017514 | 0.979925 | 0.989962 | +0.010038 | 1.5x |
| 4 | 0.902250 | 0.918802 | +0.016551 | 0.986198 | 0.992472 | +0.006274 | 2.0x |
| 5 | 0.897696 | 0.919407 | +0.021711 | 0.982434 | 0.991217 | +0.008783 | 2.0x |
| 6 | 0.902526 | 0.918297 | +0.015771 | 0.984944 | 0.991217 | +0.006274 | 1.5x |

- Mean mAP@R delta: **+0.017578** (+1.76 points).
- Paired Student-t 95% interval: **[+0.014804, +0.020353]**.
- Sample standard deviation: **0.002644**.
- All six deltas are positive; exact two-sided sign-test p-value: **0.03125**.
- Mean Recall@1 delta: **+0.008156** (+0.82 points); all six deltas exceed the
  preregistered -0.00125 guard.
- The imprinted arm reaches the matched quality target in 1.5x to 4.0x fewer epochs on
  every seed.

## Cost and performance result

- For prospective seeds 2--6, compute to matched quality is reduced by **30.8% to
  73.1%**.
- Full 16-epoch profiled compute is **1.7% to 2.2% higher** because the candidate adds
  a one-time class-imprinting pass. Raw wall time is descriptive and inconsistent across
  the nondeterministic GPU environment; it is not used as a gate.
- Peak GPU allocation is 87,167 MiB for imprinted versus 87,187 MiB for random on every
  seed (20 MiB lower).
- Checkpoint storage is equal (58,283,916,296 bytes per arm), deployment storage is equal
  (3,632,816,144 bytes), and inference architecture/path is unchanged. Measured inference
  latency across seeds is approximately 11.96--12.11 ms/image under the registered
  batch-128 protocol.
- The fusible non-backbone fraction is about 0.046%, far below the 10% kernel threshold;
  `kernel_eligible=false`. A custom kernel is therefore not justified for this candidate.

## Evidence

- Reviewed source commit: `3a6495258531037684487db9287c116f870b2486`.
- Seed-2 pair SHA-256: `d64f6a96427fb9bd7c486a46fc1155e8c5b65330011371895628012cd25f078b`.
- Seed-3 pair SHA-256: `e5dee52cafe2f97a41f4e44df0b4a836a1be0bcdf2e7b43e9d12e7df928390f3`.
- Seed-4 pair SHA-256: `000a8e5b24cec1954efc14a7a86331d0a8ceccef2fa82eef6ecbf2a13ce7268d`.
- Seed-5 pair SHA-256: `98d65d8ac39f9f9c47e59cb2e443cfbeb01cc2f3a64364aa05d7b15a9c1c4196`.
- Seed-6 pair SHA-256: `1a24a0cc24f591d9213bc4328309674d2b92cab8b26f7f3d34c3c16f38c88c4b`.
- Final summary SHA-256: `cb15ffd66a45b26f65cdae99c411767f773abecdf625b6703c34fb6b7399bf23`.
- DGX queue exit: 0; finish: 2026-08-22T12:59:34Z.
- All five new pairs and the summary pass their production strict validators after
  byte-identical transfer to the repository worktree.

## Claim boundary and next work

The supported claim is: on the fixed train/holdout protocol, class-proxy imprinting gives
a statistically supported quality improvement and a non-inferior resource/trajectory
frontier relative to matched random initialization, with unchanged deployment.

An official test-set comparison, external-baseline table, or global SOTA claim requires a
separate prospective protocol. The next research step is that official evaluation and
cross-dataset replication, not a custom kernel for this already profile-ineligible path.
