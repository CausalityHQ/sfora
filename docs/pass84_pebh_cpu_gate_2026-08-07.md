# Pass 84 PEBH CPU gate (2026-08-07)

The diagnostic used only corrected In-Shop train-final packs, splitting
identities by `SHA256(label) mod 5`. It compared a parameter-matched self-only
bilinear head with Positive-Exchange Bilinear Head (PEBH).

| seed | Δ positive | Δ foreign | Δ LOO R@1 |
|---:|---:|---:|---:|
| 0 | +0.003157 | +0.016617 | +0.001000 |
| 1 | +0.005361 | +0.019104 | -0.001333 |
| 2 | +0.007038 | +0.013075 | +0.001333 |
| 3 | +0.003434 | -0.013378 | +0.000667 |

Mean positive gain is `+0.004747`, below the preregistered `+0.0050` gate.
The mean foreign change is `+0.008855`, well above the allowed `+0.0020`,
and seed 1 violates the LOO degradation limit. PEBH therefore fails the CPU
authorization gate and no GPU implementation or run is authorized.

The result is consistent with exchange improving same-class similarity by
pulling the entire representation toward broad hubs, rather than repairing
the positive-side transfer deficit.
