# PE-Core prototype screen passes; training and production unqualified

The single frozen PE-Core-B16-224 source pilot passed all gross-deficit and
encoder-cost gates against pinned pretrained SigLIP2 Large/256. This licenses a
separately frozen TRAIN full-cache/paired training-feasibility proposal. It does
not establish unseen-product, official benchmark or production speed superiority.
No fine-tuning or official query/gallery/TEST read occurred.

## Verified pilot evidence

DeepFashion In-Shop official TRAIN **fit-only**512 products, one query and one
gallery image/product (1024 images); all belong to the13283-image fit partition.
Both independently fit centered PCA128 on their512 gallery images only, then
use existing signed-int8/f16 inverse-norm cosine arithmetic and lower-ordinal
ties. These are within-fit prototype queries, not an unseen-product split.

| DGX GB10 fixed pilot | SigLIP2 Large/256 control | PE-Core-B16-224 |
|---|---:|---:|
| Packed prototype R@1 (%) |74.609375|76.3671875|
| Raw-source prototype R@1 (%) |74.21875|77.1484375|
| Batch32 encoder-only p50,10 alternating calls (ms) |262.627224|70.453487|

Packed treatment-minus-control gain is+1.7578125 percentage points; paired
product-bootstrap95% **lower bound −1.953125 points**,5000 PCG64 draws,
seed179031. This passes the frozen point≥−3/lower≥−5 gross-deficit guards;
the negative lower bound does not establish a positive quality effect.
Encoder median ratio0.268264 passes≤0.8. No decode/preprocess/packing/search,
image-to-top-k, p95/p99 or product QPS was measured by these encoder calls.
Each model used its own native PIL RGB squash/normalization and input224/256;
this is a source/system comparison, not an isolated architecture/training effect.
Both used authentic original FP32 parameters with FP16 CUDA autocast.

PE first-four-fit-image FP16-versus-FP32 feature cosines were
0.99998063,0.99998969,0.99998599,0.99998605; all exceeded frozen0.999.
Every encoded feature was finite/nonzero; own native pixel checks passed.
One positive/image means mAP@R of this prototype equals R@1; this is not a
multirelevant-image mAP@R evaluation or evidence of final retrieval quality.

## Verified authority and resources

Original exclusive-lock systemd invocation`336ca2d13ca74be09f8cf5c7b629506e`
completed exit0; frozen whole-process120s/8GiB host limits. GNU wall32.41s,
maximum process RSS2954288KiB. Script post-import wall28.999370s; combined
source export15.039857s. CUDA peak allocated2030565888 bytes includes model
initialization. These are pilot costs, not training throughput or full-pipeline
latency. GPU compute apps and running Sfora units were empty afterward.

CPU acquisition separately completed33.96s/1868972KiB RSS, exit0;1.790786632GB
file verified against official LFS SHA,163 visual state entries loaded strictly
on CPU, all original FP32,93672960 vision parameters. Native CPU preflight
v2 separately cost3.38s and froze source/import/image/version authorities.
[Reviewed source gate and corrective prerequisites](inshop_pe_core_source_gate_2026-09-28.md).
The prospective executable/import manifest was frozen and pushed in`0699cd3b`
before outputs; original unpromoted Rust candidate remains untouched.

Independent CPU replay from saved1024x1024 source matrices reproduced every
packed/raw hit, the scalar-stream paired bootstrap, all timing medians and
all frozen criteria. Raw matrices, their hashes, source authority, journals,
resource receipts and replay are retained:
[evidence directory](evidence/compact_metric/sop-siglip2-substrate-v1/pe-core-source-v1/).
The replay runs with CUDA hidden, Torch on DGX and repository source on
PYTHONPATH: `python verify.py /home/riomus/runs/sfora-pe-core-source-v1`.

Next: design one matched TRAIN full-cache and actual image-training feasibility
comparison, preserving original checkpoint values, native processors, fit-only
initialization, identical image/update budgets and exact packed scorer. Verify
frozen/trained parameter routes and resource cost before quality, predeclare
paired quality/uncertainty stop rules before additional reads, then obtain
consequential critique. A later positive result still needs paired seeds,
protocol-matched SOP/In-Shop official qualification, CUB/Cars transfer and
matched full-pipeline serving gates. Closed Base and other failed routes remain
closed. The full production joint objective is active and unmet.
