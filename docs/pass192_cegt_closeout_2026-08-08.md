# Pass192 CEGT closeout (In-Shop seed 0)

CEGT completed the preregistered matched 60-epoch run. It keeps the ordinary
descriptor for deployment and adds only the fixed `0.05` stop-gradient
class-excluded target to the training loss.

| arm | final R@1 | raw best-over-training | local-trend diagnostic |
|---|---:|---:|---:|
| Proxy Anchor control | 0.913701 | 0.916303 | 0.913877 |
| CEGT | 0.913701 | 0.916374 | 0.914563 |

The raw final delta is **0.000 pt**. The raw best-over-training delta is
**+0.007 pt**, and the leave-one-out local-trend diagnostic delta is
**+0.068 pt**. The latter is descriptive rather than an unbiased correction.
All are below the preregistered +0.30-point pass threshold and the +0.15-point
falsification boundary; CEGT is therefore a negative/near-null, not a result.
No second seed or replication is justified.

This is still informative: using the CE-BN signal only as a gradient target
preserved the ordinary control path and removed the catastrophic collapse, but
it did not transfer the descriptor-only +1.357 pt CPU signal into benchmark
performance. The CPU wall-clock issue is irrelevant here; this verdict uses
paired benchmark histories and reports both raw and selection diagnostics.

