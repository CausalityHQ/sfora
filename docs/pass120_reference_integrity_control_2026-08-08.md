# Pass 120 current-code Proxy Anchor integrity control

## Why this control is required

Two corrected-corpus In-Shop artifacts labelled as seed-0 Proxy Anchor and carrying
the same recipe digest
`97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361`
do not agree:

| artifact | raw best R@1 | best epoch | final R@1 | local-neighbour trend |
|---|---:|---:|---:|---:|
| `inshop_corrected_pa_seed0.json` | 0.916303 | 41 | 0.913701 | 0.913877 |
| `pass119_inshop.auto.seed0.json` | 0.918695 | 41 | 0.918624 | 0.916057 |

The raw-best difference is `+0.239` point and the final difference is `+0.492`
point, both larger than the registered CIS effect.  Their effective recipe fields
differ only by subsequently added zero-weight defaults and the presence of a final
checkpoint path, but neither report binds the executing trainer source.  The
difference therefore cannot be identified as stochastic variance versus source drift.
It is not valid to choose the lower historical reference after seeing a candidate.

## Frozen control

After all four Pass-120 CIS/control artifacts exist, run one unchanged `auto` Proxy
Anchor arm at seed 0 in the same deployed checkout, process environment, corrected
In-Shop corpus, and 8,580-step protocol.  Save both its report and final checkpoint.
The run is an integrity control, not a new candidate and not a hyperparameter search.
It may refine the comparison reference but cannot repair a CIS failure against its
three same-path controls.

Theoretical cost is one ordinary Proxy Anchor forward/backward training run with no
extra deployed state.  No conclusion uses wall-clock time measured on the shared host.
