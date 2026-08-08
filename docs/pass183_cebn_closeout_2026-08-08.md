# Pass183 CE-BN closeout (In-Shop seed 0)

The preregistered hard class-excluded batch-normalisation run is a clear
negative.  It completed the matched 60-epoch In-Shop protocol on seed 0.

| arm | final R@1 | raw best-over-training | local-trend selection diagnostic |
|---|---:|---:|---:|
| Proxy Anchor control | 0.913701 | 0.916303 | 0.913877 |
| CE-BN (hard) | 0.684344 | 0.684344 | 0.671402 |

Against the paired control, the raw final delta is **-22.936 pt** and the raw
best-over-training delta is **-23.196 pt**.  Using the repository's
leave-one-out local-neighbour diagnostic (a descriptive selection diagnostic,
not an unbiased correction), the delta is **-24.247 pt**.  CE-BN therefore
fails the preregistered `>= +0.30 pt` criterion by a wide margin and is closed;
there is no second seed or replication.

The CPU probe (+1.357 pt after applying CE-BN to already-trained descriptors)
was not predictive of end-to-end training.  The likely mechanism is a
train/evaluation mismatch: CE-BN changes the representation presented to the
loss using batch statistics, while deployment/evaluation uses the ordinary
encoder path.  This closeout does not use concurrent CPU wall-clock timings as
cost evidence; those timings were invalid under parallel workload.

