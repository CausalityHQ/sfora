# ECT zero-training probe — FAIL, no GPU candidate run

I ran the mandatory feasibility probe on the existing In-Shop corrected
Proxy-Anchor checkpoint (`inshop_corrected_pa_seed0.pt`) using 64 training
images and the real BN-Inception feature maps. This was inference only; no
weights or data splits were changed.

| beta | plateau hinge active | must-switch proxy hinge active | mean replaced area | mean anchor cosine |
|---:|---:|---:|---:|---:|
| .15 | 23.4% | 100% | .021 | .930 |
| .25 | 37.5% | 100% | .031 | .905 |
| .40 | 76.6% | 100% | .066 | .819 |
| .60 | 98.4% | 100% | .129 | .700 |
| .85 | 100% | 100% | .312 | .522 |

The must-switch proxy hinge is active for every sampled beta. Under the
pre-registered review rule (“abort if either hinge is approximately 0% or 100%
active across all beta”), ECT is unsatisfiable at the operating point. The
replaced-area fraction also correlates strongly with beta (r=.897), so the
target regime is nearly an area/mass schedule rather than a demonstrated
evidence-consensus signal. This is a Gate-3 stop; no ECT training GPU hours
were spent.
