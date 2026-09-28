# Actual SOP/In-Shop joint cached gate: KILL

The [frozen gate](sop_inshop_joint_cached_gate_2026-09-28.md) fails both matched
comparisons. Stop this fixed cached configuration before encoder training,
additional seeds, official evaluation or serving certification. Production
training and serving defaults remain native. The joint SOTA-quality and
matched full-pipeline speed goal is still unmet.

## Verified measurements

Dataset: original **In-Shop official TRAIN fit identities**, internally split
into 6,757 training images / 995 products and 6,514 validation images /
997 products (**3,440 queries / 3,074 gallery images**). Joint/sham arms add
5,190 **SOP official TRAIN fit** images / 995 identities. No outer TRAIN-held
or official query/gallery/TEST read. Frozen pretrained SigLIP2 Large/256
1024-D caches, common target-only PCA128 initialization and native packed
130-byte scorer. CPU-only, eight threads on DGX Spark; no encoder updates.

| Arm, seed179033, 100 updates ×64 cached presentations | Packed R@1 | mAP@R | Cached train + init, s | Bank rows |
| --- | ---: | ---: | ---: | ---: |
| Native In-Shop-only control | 87.093023% | 0.6274668562 | 2.156810 | 6,757 |
| Actual joint SOP/In-Shop | 86.162791% | 0.6146671932 | 2.495538 | 11,947 |
| Same joint pixels/counts, shuffled SOP labels | 86.162791% | 0.6154450940 | 2.924605 | 11,947 |

All300 updates have finite recorded losses and preclip gradients. Sham
changes **99.845857%** of external labels while preserving target labels and
the global label histogram. Joint/sham get32 target+32 external presentations
per update; control gets64 target presentations. Joint changes target
exposure, class count, bank size and proxy initialization. The sham comparison
controls images/counts/exposure but its proxies necessarily reflect shuffled
semantics. These are system comparisons, not isolated identity-count effects.

| Joint minus baseline | R@1 delta, percentage points (95% product CI) | mAP@R delta (95% product CI) | Decision |
| --- | ---: | ---: | --- |
| Native control | **-0.930233 [-1.314174, -0.551028]** | **-0.012799663 [-0.015104224, -0.010653858]** | KILL |
| Shuffled-label joint | **0.000000 [-0.294490, +0.302506]** | **-0.000777901 [-0.002098087, +0.000597436]** | KILL |

Both predeclared R@1 floors/positive lower bounds and both mAP nonregression
conditions fail. Independent local replay reconstructed roles, quality means,
all5,000 product-bootstrap interval endpoints, update counts, finite histories,
bank/parameter counts and decisions from immutable receipts. These intervals
condition on one training seed and a reused internal panel; they are not seed
population inference or untouched generalization evidence. No raw-embedding
scorer rerun was performed; the actual scorer source was independently matched
between local and remote trees.

Full main wall: **9.042441885 s**. Original service
`sfora-sop-inshop-joint-cached-v1`, invocation
`7b3d46aed3e9478f965b1dfa805bfc31`, terminal **inactive / MainPID0 /
Resultsuccess / ExecMainStatus0**. No duplicate job or consultation.
Raw receipt SHA256:
`dcacf0738bb28b38c126d7d7e0c20b2cdf6c255fa77080a66daad9ed62f593f8`.

Raw receipt, journal and independent verification are in the
[evidence directory](evidence/compact_metric/sop-siglip2-substrate-v1/sop-inshop-joint-cached-v1/verification.json).
Qualified runtime source is preserved at commit `8463d9ea`, runner SHA256
`6f73f64c18e3994b65452ff464c72723c00505aa2658bba44d5670809ab8a2e1`.
Remote source/result directories remain intact.

## Explicit instrumentation corrections and scope

- The external service watchdog was150s; the preregistration text said140s.
  The main120s alarm and100-update budgets were unchanged. The run finished
  far below either cap. This discrepancy is recorded, not hidden or used to
  increase the training budget.
- One helper-manifest entry hashed Torch's `no_grad` decorator wrapper rather
  than the scoring function's defining file. Independent remote/local SHA256
  checks authenticated the actual scorer:
  `8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250`.
  Future manifests now use `inspect.unwrap`; a runnable self-check catches
  the wrapper regression. No original receipt was edited or job repeated.

**Unmeasured:** encoder training images/s, VRAM, serving memory and full
image-to-top-k p50/p95/p99/QPS at either batch size. Cached train/init seconds
are not encoder throughput. Cross-dataset pixel duplicates remain unaudited.

The frozen-head failure does not universally disprove joint encoder learning.
It supplies no positive reason to fund this configuration's next stage. Do
not reopen it with new steps, seeds, label permutations or a GPU run. Keep
the usable native library path and seek a causally distinct representation or
supervision intervention supported by an actual cheap diagnostic.
