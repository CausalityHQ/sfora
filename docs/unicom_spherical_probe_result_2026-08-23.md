# UniCOM spherical-probe causal screen result

The preregistered screen closed the conservative optimized-proxy-direction
candidate. The exact reviewed source was
`ed2e7893b05d3b5105ff992691efccc5b13ad5a0`; the immutable result is
`reports/generated/unicom-spherical-probe-ed2e789.json`, SHA-256
`d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf`.
The production validator and an independent local reload both passed.

The one scientific process ran on `spark-2751` with the pinned UniCOM
`d71992ed969e6c271436ac0a0ee1f3ca61474ac0` checkpoint and official In-Shop
train partition. It completed in 239.53 seconds with 5,679 MiB peak allocated
GPU memory. An earlier harness command used an invalid local remote-shell
wrapper and exited 127 before Python started; it created no process, GPU work,
temporary file, or result.

## Decision

The registered decision is `CLOSE_DIRECTION`. Every seed passed every loss,
uncertainty, accuracy, stratum, and gradient-direction predicate. The sole
failed predicate for all three seeds was the preregistered mean head-cosine
floor of 0.95:

| seed | mean head cosine | paired loss delta | mask 95% lower | identity 95% lower | unrepresented 95% lower | accuracy delta | gradient median cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.947578 | 0.200482 | 0.199375 | 0.158880 | 0.240851 | 0.026388 | 0.878770 |
| 1 | 0.947491 | 0.208586 | 0.207449 | 0.167924 | 0.256053 | 0.025859 | 0.878464 |
| 2 | 0.947437 | 0.204887 | 0.203790 | 0.163411 | 0.262949 | 0.025521 | 0.872869 |

All 64 masks and all 64 unrepresented-stratum masks improved for every seed.
The large cached-feature gains show that optimizing the classifier direction
changes the objective materially, but the fitted heads moved outside the frozen
conservative neighborhood. The threshold is not widened or reinterpreted after
seeing the result, and the planned 2x2 fine-tuning continuation is not launched.

## Routing

Close this exact conservative-direction candidate. Preserve the positive signal
as hypothesis-generation evidence only: a future candidate must be independently
motivated and preregistered, not a smaller-step or relaxed-cosine retune chosen to
clear this observed boundary. The next research cycle should target a distinct
training-time mechanism that can improve time-to-quality without changing
inference or storage cost.
