# Corrected In-Shop Proxy Anchor reference preregistration

Date: 2026-08-03. Registered before training on the corrected pixels.

Run seed 0 with the pinned official Proxy Anchor recipe
`proxy_anchor.inshop.official-51db570`, BN-Inception/512 dimensions, and the newly
validated 256-pixel corpus. Report raw best-test R@1 and independently recomputed
frozen-final R@1. The published checkpoint gives 0.91764 and the authors report about
0.919, so predict raw best in **[0.907, 0.929]**. A value outside that interval blocks
candidate screening and triggers a recipe/artifact audit rather than tuning.

This run establishes a reference, not a method result. The old 0.9035 baseline,
0.12-point sigma, and one-seed decisiveness rule are withdrawn because they were
measured on `img_highres`. No candidate may inherit them.
