# PE-Core-B16-224: bounded source feasibility gate

This changes pretrained representation, not the closed SigLIP2 Base teacher,
head, data-scaling, nonlinear-residual, crop or loss routes. Native PE is untested
in the repository. No training, model download or GPU work has run for PE.
The full SOP/In-Shop quality-and-speed goal remains unmet.

The [official model card](https://huggingface.co/facebook/PE-Core-B16-224)
describes a12-layer768-wide vision tower,224px patch16 input, learned attention
pool and1024-coordinate output projection, distilled from PE-Core-G. This is
published architecture, not a measured advantage on In-Shop or Spark. Its native
processor squashes bilinearly to224x224 and normalizes RGB with mean/std0.5.
[Official implementation](https://github.com/facebookresearch/perception_models).

## Source authority and prerequisites

HF API confirms ungated Apache2.0 checkpoint revision
`a16450b46fef32363459920c2685a1b4ef13dcd9`;
`PE-Core-B16-224.pt`1790786632 bytes, official LFS SHA256
`0a5c220aa083488e0fc9221f766dced8a576b7074662d9b9da923da1e8844fce`.
Pin native source to GitHub commit
`3e352cca660658d4b5c90f42a7808b11469e4c66` and hash imported source.
Metadata receipts are archived in `pe-core-source-v1/`.
Native `VisionTransformer` exposes image-only loading; the loader uses
`torch.load(weights_only=True)` but `strict=False`. The probe must explicitly
filter visual state, require exact key/shape coverage and use strict loading;
never silently accept initialized missing weights or download unpinned main.
Native config file on HF is empty, so pinned GitHub config is authoritative.
DGX has Torch/torchvision/timm/regex but lacks einops/ftfy. Use isolated pinned
pure-Python dependency wheels if needed, without changing the deployed venv;
record hashes and versions. No FlashAttention or full PE stack install.
DINOv3 owner grant remains unknown: normal HF credential discovery is absent.
PE is independently public; no mirror or gated-model bypass is involved.

Before any GPU: independently inspect native model/processor loading, strict
state authority and dependencies; freeze executable plus fit-only image manifest;
pass one native CPU import/config/processor and packed tie/PCA leakage check.
Acquire only this1.79GB checkpoint under600s/8GiB host bounds and verify SHA.
Failure stops before encoder work; no alternative PE size is searched.

## One prospective fit-only source pilot

Reuse `probe_inshop_siglip2_base_pilot.py` arithmetic, PCA and paired bootstrap.
Select512 non-singleton products from existing13283 official TRAIN fit rows,
ordered by SHA256(`inshop-pe-core-pilot-v1\0`+UTF8 product), then two images each
by relative-path SHA ascending. Freeze all1024 image hashes before outputs.
No outer TRAIN holdout or official query/gallery is read.

Compare pinned pretrained SigLIP2 Large/256 native pooler against native
PE-Core-B16-224 final projected output. Both vision parameters FP32 with native
FP16 CUDA autocast, same1024 images/batch32. Processors differ by architecture:
check each wrapper pixel-for-pixel against its own unmodified native processor;
do not claim cross-model pixel equality or an isolated learning-method effect.
Normalize sources, fit each centered PCA128 using only its512 gallery images,
pack with existing int8/f16-norm format and score the512 paired queries against
that gallery. This is a fit-only prototype screen, not unseen-product quality.
Retain raw/packed hit arrays and paired product-bootstrap95 lower, seed179031,
5000 draws. Use lower-gallery-ordinal score tie ordering. Native metadata and
numerical failures stop immediately.

Time10 alternating forward calls/arm after3 warmups at batch32 on each arm's
fixed native pixels. Require PE median time<=0.8x Large. Do not call this decode,
preprocess, packing, search, image-to-top-k, p99 or product QPS measurement.
Require packed PE-minus-Large R@1>=−3 percentage points AND paired95 lower>=−5
percentage points. These are the existing smaller-source gross-deficit rules,
frozen unchanged before PE outcomes. Require whole GPU pilot<=120s and peak
allocated CUDA<10GB, systemd MemoryMax8GiB and exclusive shared DGX GPU lock.
Gross quality, cost, numerical, authority or resource failure closes this fixed
checkpoint route without resolution/depth/projection/threshold rescue.

PASS only licenses proposing one TRAIN-fit full-cache/paired training-feasibility
comparison with separately frozen gates and consequential critique. It is not
permission for official reads, production promotion, longer training or a claim
that final quality can recover. Source acquisition/training cost and actual
end-to-end serving cost remain required. Base's prior failed confidence bound
and DINOv2/MODA/FashionSigLIP/TIPS failures stay closed.

## Terminal dual review and corrected acquisition prerequisites

Group`c3033edfc7374e78` completed normally: Opus`296df55242b94989`112s and
Astra`12b8fd27c1f24844`70s, both conditionally approve this distinct source gate.
Full labelled answers are in `pe-core-source-v1/dual-result.json`; no fallback.
Their corrections are accepted before any PE weight download or GPU work:

- Both arms must load true original checkpoint values directly into FP32;
  do not inherit the prior Base script's FP16-load-then-FP32-roundtrip.
- PE acquisition validates on CPU with `weights_only=True`, `mmap=True`,
  `map_location="cpu"`, anchored one-prefix stripping with collision rejection,
  complete visual-key/shape coverage, finite tensors and strict state loading.
  Bypass the native unpinned Hub downloader entirely.
- Hash all native package initializers, tokenizer, vocabulary and model files.
  Dependencies include einops0.8.1, ftfy6.3.1 and already installed wcwidth.
  Two hash-verified pure-Python wheels were extracted into the isolated probe
  directory; deployed venv is untouched. Its Python has no pip, so no installer
  or setup code ran. Reject absolute/traversal/.pth wheel paths before extraction.
- Native PIL input must already be RGB; record Pillow/torchvision versions,
  native own-processor pixels and separate fixed224/256 timing tensors.
  Fork a new pilot script; do not edit the previous failed Base experiment.
- Before timing, PE FP16 autocast versus native FP32 on four fixed fit images
  must have cosine≥0.999, with finite nonzero features; otherwise stop.
- Bootstrap must use declared PCG64 seed179031 explicitly; the old helper's
  hardcoded179019 is not reused unchanged. Save paired hits for replay.
- CUDA peak accounting begins before initialization, with whole-process native
 120s deadline/8GiB memory guard and GPU lock covering all CUDA use. Timeout/OOM
  gets a terminal failure receipt and cannot be reported as a quality negative.

CPU checks passed native import/config, finite random1024-output geometry,
black/white processor arithmetic, deterministic image inventory, packed128
geometry/tie ordering and anchored visual-state/missing-key/collision rejection.
These are source/fixture checks, not loaded-checkpoint numerical qualification.
Acquisition may now run once under600s/8GiB CPU limits, then strict validation;
GPU remains gated on actual acquired-state authority and a frozen pilot.

## Acquisition complete; executable GPU gate ready

Original CPU unit`2df2eaa9752a4f6f98aaae324e5ef005` finished exit0 in33.96s,
1868972KiB maximum process RSS. The pinned1.790GB file matched its SHA and
all163 visual entries loaded strictly, finite, original FP32; the vision model
has93672960 counted parameters. No GPU/quality was used by acquisition.
Native CPU preflight v2 finished3.38s; versions and1024 image hashes are frozen.

`scripts/probe_inshop_pe_core_pilot.py` implements the corrected prospective
pilot. Its executable/import manifest and CPU log are archived. One CPU check
passed PCG64 bootstrap endpoints and exact packed lower-ordinal ties. The first
manifest capture caught Torch's virtual `_classes.py` pseudo-module; metadata
hashing now skips paths that are not real files. This was resolved before any
GPU/quality, without changing the numerical or decision protocol. Ruff passed.

The CPU check can be repeated with CUDA hidden and the isolated native/source/
dependency paths on PYTHONPATH by importing the pilot, asserting
`lower_bound(np.zeros(512))==0`, `lower_bound(np.ones(512))==1`, and
`packed_hits(x,x).tolist()==[1,0]` for two unit128 rows both equal to coordinate0.
Calling `code_authority()` checks every real imported file in the probe root.
The saved manifest is required unchanged before GPU initialization.

The prospective launcher will enforce120s whole process/8GiB host, exclusive
shared GPU lock; initial allocation peak is retained. Timing keeps separate
native224/256 tensors. Raw source matrices, packed hits, calls and bootstrap
inputs are retained for independent replay. No outcome or serving claim exists
yet, and no closed checkpoint route is reopened.
