# Fixed224 L14 FP16 source and full-fit initialization prerequisite

The pinned native CPU source PASS permits ONE native FP16 acquisition on DGX.
This supplies a useful trained-model path: own complete fit features → own
PCA/head/proxies/bank → discard17-step training mechanics → fresh100-update
TRAIN quality pilot. No B16 features/initializers/freezer are reused for L14.
No optimizer/input/prefix/precision sweep, no official/held decoding here.
Full production SOP+In-Shop joint quality/speed remains mandatory.

Reuse externally authenticated full-fit manifest SHA
4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff:
13283 fit images/2004products/12singletons, excluded12599 TRAIN-held images.
CPU preflight rechecks official partition, exact split/ordinal/path/product
mapping, class disjointness and every image hash. Pin L14 source PASS receipt
SHA6038ad876a209b3ae29925427ef6d54c7f23ff084be3863bb8fb74158bb052f7,
public checkpoint revision/bytes/hash, native manifest/implementation and
executing helper origins/hashes. Freeze imported code and package versions
before GPU. Altered authority must fail before CUDA; malformed feature
shape/dtype/nonfinite/nonunit states are rejected by a runnable CPU fixture.
CPU cap120s whole/8GiB/no swap/CUDA hidden.

Actual GPU uses original FP32 weights/native336 config, fixed native224 RGB,
FP16 autocast, TF32 disabled, eval/inference only, batch32. Require exact307
visual keys/shapes/source-state and strict loading. First4fit FP16 versus
FP32 cosine>=.999, save individual values before assert; first2processor
pixels must reproduce the authenticated CPU source hash. Preserve first4
reference vectors for saved-cache audit. Atomic existing export helper writes
13283×1024 finite unit FP32 descriptors in frozen order, rehashing files as
decoded. Saved first4 versus fresh FP16 and FP32 both require cosine>=.999.
Verify registered source and foreign rotary/grid authority unchanged.

ONE GPU service300s whole (299+1 shutdown),8GiB host/no swap,
peak allocatedCUDA<10,000,000,000bytes, both lifetime GPU locks acquired
inside the service. No overlapping jobs. Record original exit/time/invocation,
feature SHA, first4precision/cache cosines, export time/images/s/VRAM and
source/code/environment authority. Any failure closes this acquisition without
retry/dtype/input/threshold/budget rescue. Preserve completed evidence.
This measures acquisition cost only, not optimizer cost or public latency.

On PASS, independently audit saved matrix/reference hashes, unit norm,
first4cosines and authority, then freeze real native half-prefix12 L14
training using unchanged augmented control exposure/loss/AdamW prescription.
Same17mechanics120s and fresh100pilot300s limits, update median<=.71769696s,
TRAIN-held R1>=95.1720176%,MAP>=77.6237120%, positive product95% lower bounds
against archived densePE for both. Those quality/training gates are still
unexecuted, and their initialization/updated-native parity must be qualified.
A survivor must advance to updated serving and fresh confirmation.

## CPU prerequisite PASS

Actual CPU preflight invocationa605820c62194ad585aaff51e938390c exited0 in
5.04s whole/976,236KiB processRSS/no swap. Frozen preflightSHA
5653617efcdc9bcc09d8efaff9c0a8f3302775060fa8a00aefab923df46fbeb1.
Runnable fixture RED missing exporter then GREEN changed-manifest-before-CUDA
plus bad shape/dtype/nonfinite/unit-norm rejection; Ruff PASS. Original
frozen authority separately reached hidden-CUDA guard before any attempt
file, feature allocation or optimizer. No full repository suite/research.
