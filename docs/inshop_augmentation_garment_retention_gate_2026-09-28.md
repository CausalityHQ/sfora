# TRAIN-fit identity-support crop audit

Source MAIN closed after paired100 mAP confidence included zero. New mechanism
is different from the closed wider-crop/view-coverage arm: native augmentation
may remove the garment that supplies the product label. This tests label-support
damage, not more framing diversity, background removal or a serving crop.

One CPU-only <=120s read-only audit, before any new encoder run. Use official
TRAIN fit13283/2004 only, original pinned partition/fit and garment-box file.
Select512 fit rows by SHA256(image_name) order, independent of scores/products.
For each image, read/hash the original file, validate inclusive1-based garment
box, and sample8 native torchvision RandomResizedCrop get_params with area
(.8,1) and aspect(.75,4/3), seed179024, same API as ImageRows. The subsequent
horizontal flip preserves support area and is irrelevant to this audit. Draws
are a deterministic exposure sample, not a replay of all training augmentation.

Measure intersection area / released garment-box area for each crop, using
half-open pixel rectangles after converting the annotation exactly as the
existing boxed_image helper. Record crop rectangles and per-image
fractions; no pixels need decoding, no model/features/held/official reads.

GO for a separately frozen paired crop-support representation diagnostic ONLY
if >=10% of4096 draws retain<90% garment area AND >=5% retain<75%. These are
predeclared resource floors for substantial support loss, not quality/statistical
proof. Failure closes this fixed support-preservation hypothesis before training.
PASS does not authorize a crop-scale search, train run, packed-quality claim,
official selection or SOTA. Next positive diagnostic would compare native crop
with annotation-constrained crop at identical fit image IDs/labels/flip draws,
under matched actual encoder outputs and native packed scoring; only then
consider a bounded paired training smoke. Native augmentation/default unchanged.

## Terminal KILL

Original CPU-only unit `sfora-inshop-crop-support-f0-v1`, invocation
`f76f131e7c944efabda7ff98c153db04`, exit0. Audit wall1.095380011s,512 fit
images/4096 sampled crops, no encoder/GPU/optimizer/held-image access.
ReceiptSHA `365a1d631f9f3f502d7b49a095126d7341a20777df29dc9b2fee5b4c7dde0a1c`.
Stored under `evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-support-f0-v1/`.

| Official In-Shop TRAIN fit; native crop | Measured | Frozen floor | Decision |
| --- | ---: | ---: | --- |
| Crop retains <90% garment area | 319/4096 = 7.788086% | >=10% | KILL |
| Crop retains <75% garment area | 9/4096 = 0.219727% | >=5% | KILL |
| Training/R@1/mAP/serving p50/p95/p99/QPS | Not measured | None authorized | No follow-up |

Independent local integer pixel-set intersections reproduce all4096 fractions
exactly; source hash matches committed probe, annotation/partition/fit hashes
verified. All512 boxes valid. No threshold relaxation, box-constrained crop
encoder diagnostic or training. Retain native augmentation. This does not prove
crop invariance optimal or rule out smaller identity-detail loss; it rejects
the specified substantial garment-support damage premise cheaply.
