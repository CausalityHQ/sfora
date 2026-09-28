# Actual SOP/In-Shop joint supervision: data readiness gate

One hypothesis: concurrent real SOP identities may regularize In-Shop metric
learning better than sequential SOP warm start or pooling In-Shop categories.
This is known joint supervision, not an invention. The earlier category-only
cached failure does not measure this intervention. No production default or
checkpoint changes are authorized by a metadata pass.

First run the read-only
[inventory audit](../scripts/audit_sop_inshop_joint_inventory.py) on DGX CPU,
under the existing exclusive experiment lock and a 90-second process timeout.
It reads official TRAIN metadata, file hashes and NPY headers, not image pixels,
feature values, gradients, held quality or official TEST data. No CUDA work.

Frozen checks before outcomes:

- Both pretrained feature files must match their existing authority hashes,
  the same SigLIP2 Large/256 weights and the SOP ordered TRAIN row digest.
- Reuse the original In-Shop fit partition, then its existing deterministic
  half-product split. Remove singleton products: expect 6,757 training rows,
  995 products and 6,514 internal validation rows. Neither may overlap outer
  held products; training and internal validation products must be disjoint.
- Select exactly 995 SOP products solely by the fixed SHA256 identity order
  in code, from the existing SOP fit partition (seed179019/fraction0.9).
  Include all their rows. Exclude SOP development products.
- Explicit `sop:`/`inshop:` namespaces must remain disjoint. This prevents
  numeric identity collisions; it does not prove cross-dataset pixel uniqueness.
- Combined bank at most 20,000 rows, at most 32 other positives per identity,
  audit main wall at most 60 seconds. Stop on any failed guard or timeout.

A pass permits designing one fixed-budget, cached head comparison: native
In-Shop-only versus actual joint supervision versus the same joint images
with count-preserving shuffled SOP product labels. All use common In-Shop-only
PCA/head initialization, native objectives and packed scoring. Equal total
image presentations reduce target-domain exposure in the mixed arms; report
that tradeoff explicitly. A positive result must exceed both matched controls,
then qualify an encoder smoke before any long training. Freeze this later
quality gate before scoring; no automatic GPU promotion from this audit.

The internal panel has been used for earlier diagnostics. Any subsequent
result is exploratory and conditional, not untouched confirmation or SOTA.
Serving cost requires measurements even if model geometry is unchanged.

## Terminal audit and explicit engineering correction

Original native execution session `94547` exited zero. The audit main took
**0.922717 s** with no CUDA/model/quality work. Source and all split/namespace
checks passed. The original receipt says **KILL_DATA_READINESS**, because the
frozen maximum-32-positive guard failed. That receipt is immutable in the
[evidence archive](evidence/compact_metric/sop-siglip2-substrate-v1/sop-inshop-joint-inventory-v1/receipt.json).

Independent Counter replay traced this failure to the **unchanged native
In-Shop inventory**, not SOP: native fit has 6,757 images / 995 products,
including one 111-image identity (**110 other positives**). Selected SOP has
5,190 images / 995 products, at most 12 images per identity (**11 positives**).
The combined bank is **11,947 rows**; internal validation remains 6,514 images
/ 997 disjoint products. The 32 limit would reject the already used control.

This is an engineering guard error, not a negative joint-training quality
result. Before reading any new feature values or quality, replace that guard
with **bank <=20,000 rows and external positive width no larger than native**.
The implementation has a runnable self-check for native-width acceptance,
external-width rejection and total-row rejection. Re-evaluating the original
inventories independently passes the corrected guard; the
[correction receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-inshop-joint-inventory-v1/verification.json)
records original decision, root cause, source hashes and corrected decision.
No second remote audit, quality training or GPU job was launched to do this.

Decision is **GO_CACHED_JOINT_DESIGN_ONLY**, not a quality GO or production
change. The subsequent fixed-budget comparison still needs a frozen runtime,
authority, stability, quality and sham-control gate. Resource feasibility must
be measured there; row/positive counts alone do not establish wall time or RAM.
