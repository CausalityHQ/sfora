# In-Shop frozen-layer cache cost screen, 27 September 2026

The current In-Shop `freeze_emb` trainer freezes embeddings and vision blocks
0–11 but executes their forward pass on each newly augmented image. Before
building a feature-cache training path, measure its maximum plausible compute
saving on the DGX Spark GB10 using the seed-179026 true-freeze checkpoint.

The [probe](../scripts/probe_inshop_frozen_cache_cost.py) SHA-256
`493a10c02b99f6a25f3dd421773ff27709593c3776ce58d7cea475896dbdd992`
split the installed SigLIP vision forward after block 11. The split output
matched the complete model's pooled output **exactly** on the same synthetic
batch of 64 images under BF16 autocast. Ten timed calls per path followed
three warmups; each call synchronized CUDA. This is a model-compute probe,
not an image-loader, optimizer, member-bank or full-training measurement.

| Batch-64 path | Median wall |
| --- | ---: |
| Full frozen-lower forward and backward through trainable upper layers | 0.65023 s |
| Forward and backward from cached block-11 activations | 0.48848 s |
| Generate frozen block-11 activations once | 0.16120 s |

The cacheable activation is FP32, shape 256 tokens × 1,024 dimensions,
**1 MiB per image**. Four fixed views of the 13,283 fit images require
**51.89 GiB**. At 1,000 batch-64 updates, this synthetic compute accounting
gives `4 × 13,283 / 64 × 0.16120 + 1,000 × 0.48848 = 622.3 s`, versus
`1,000 × 0.65023 = 650.2 s` without caching: only **4.3%** saving before
the cache's I/O and augmentation-diversity cost. At 2,000 updates the same
formula suggests a **14.6%** compute saving, but total training compute still
rises from 650.2 to 1,110.8 s. These are estimates from measured segments,
not measured end-to-end training, quality or public latency.

**Decision:** do not implement the four-view cache as the next production
change. It provides no compelling same-budget speed gain, consumes a large
fraction of shared memory, and fixes augmentation views. The 192/224-pixel
screens also failed their quality rule. The next treatment must target
representation learning or a different encoder path and pass paired
seed-level quality, full training cost and public image-to-top-k gates.

The sole service `sfora-inshop-frozen-cache-cost-v1.service`, invocation
`0ea516408d5c4f709a6a475c961c7c25`, exited successfully. The raw
[timings](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-frozen-cache-cost-v1.json)
SHA-256 are `bbb33b78d7ea2b2b188f6934cde9d30bcbdc9e1f9a7094ebc12db9c8133e117b`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-frozen-cache-cost-v1.journal.log)
SHA-256 is `c97769da63711ece6f3f02b13cada6eca6dfd150dbf52740531fc3eeeff66e34`.
