# Pass190 soft CE-BN closeout (In-Shop seed 0)

The preregistered soft blend (`lambda=0.70`) also fails to replicate the
descriptor-only CPU signal.  It completed the matched 60-epoch protocol.

| arm | final R@1 | raw best-over-training | local-trend diagnostic |
|---|---:|---:|---:|
| Proxy Anchor control | 0.913701 | 0.916303 | 0.913877 |
| soft CE-BN | 0.889014 | 0.890350 | 0.887185 |

Raw final delta is **-2.469 pt**; raw best-over-training delta is
**-2.595 pt**.  The repository's leave-one-out local-neighbour selection
diagnostic (descriptive, not an unbiased correction) gives **-2.669 pt**.
The preregistered pass condition was a corrected delta of at least +0.30 pt;
the falsification condition was below +0.15 pt or any non-positive raw delta.
This is a clear failure.  No second seed, ablation, or replication is
justified.

The soft path learned more slowly and avoided the catastrophic hard collapse,
but it converged well below the ordinary deployment control.  Together with
the hard result, this closes the CE-BN line: applying label-excluded moments
directly to the training embedding is not a valid way to exploit the CPU
descriptor probe.  Concurrent CPU wall-clock timings remain excluded from the
evidence; these values come from the paired benchmark histories.

