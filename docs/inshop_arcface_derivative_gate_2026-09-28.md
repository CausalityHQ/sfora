# ArcFace derivative: frozen fit-cache contract

Teacher transfer is closed. Test one causally distinct solver hypothesis:
the faithful UNICOM straight-through target derivative may differ materially
from the analytic derivative of its angular-margin objective. Preserve all
forward logits and original coefficient/margin/scale; this is not a bug in
the reference replica or a novel loss. Existing NumPy training and shard-audit
tests already distinguish the mathematical and reference gradients.

First CPU only, original official TRAIN fit13283/2004 products, seed179024,
pinned pretrained cache/PCA and first17 original1000-schedule batches. No
optimizer, encoder job, new source, held/official read or quality inference.
Reuse native logits twice (reference margin0.3/scale64 and raw cosine); replace
target backward with analytic cos(acos(c)+0.3) while attaching reference
forward via exact zero-offset. Differentiate only c within[-1+1e-6,1-1e-6];
zero target cotangent outside this interior. This finite pole policy is an
explicit regularization: no globally exact derivative claim at singularities.
Original ArcFace+8bank objective and singleton rank inactivity stay fixed.

Before evidence, GO only if:

- All17 forward-logit matrices bit-identical to native, all1088 source rows
  have finite/nonzero gradients, double-precision interior gradcheck passes,
  exact parallel/antiparallel endpoint gradients finite with forward parity.
- Full-objective source-gradient median angle≥7.5degrees, median relative
  gradient difference≥25%, median norm ratio in[0.5,3], max radial dot<1e-5.
  These materiality floors screen a direction-changing solver; pure gradient
  scaling can disappear under clipping/Adam and does not justify GPU work.
- All hash/inventory/PCA/schedule guards pass and external total≤120s.

Any failure KILLs this fixed derivative configuration without easing floors,
changing epsilon or coefficient. Passing only motivates one separately frozen
<=120s paired17-update trainable-encoder smoke, reviewed for gradient route,
endpoints/clipping/provenance before launch. It proves no held improvement,
SOTA, speed gain, or novelty. Production UNICOM/reference API stays unchanged.
