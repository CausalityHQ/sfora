# In-Shop positive-count diagnostic, 27 September 2026

The failed 22-block and first-16-block freeze arms both reduced training cost
but lost mAP@R on the same 12,599 official **TRAIN** held-only queries/gallery.
This **post hoc** read groups their already archived per-query AP differences
by held-product image count. It does not revise either frozen gate or select a
new method.

| Held product images | Queries | 22-block minus 24-block AP, 100 updates | Freeze-16 minus freeze-12 AP, 1,000 updates |
| --- | ---: | ---: | ---: |
| 2–3 | 650 | +0.00154 | −0.00423 |
| 4–5 | 4,846 | +0.00751 | −0.00605 |
| 6–8 | 2,421 | −0.02017 | −0.01097 |
| 9+ | 4,682 | −0.02205 | −0.00802 |
| **All** | **12,599** | **−0.00910** | **−0.00763** |

The depth arm lost AP mainly on products with at least six held images; the
freeze arm lost AP across all product-size groups. This pattern motivates a
specific check of multi-positive ranking in any subsequent learning method.
It does not prove which encoder operation or training gradient caused the
loss, and product size is confounded with other product properties.

The [source-bound CPU diagnostic](../scripts/diagnose_inshop_positive_count_loss.py)
SHA-256 `c8f42ad702073dc990cf26f850a5ad11bbc055eeca9be86b8644856039b50c44`
verified the four receipt hashes and official partition, then wrote the
[raw report](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-positive-count-posthoc-v1.json)
SHA-256 `b1b87fed9b22592de1b5db79ef440d9f925cb6736a59a3404431ffd489941224`.
