# In-Shop description supervision: cheapest TRAIN falsifier

Packing and preservation of the SOP source failed their gates. Test one
distinct mechanism: identity-only training lacks explicit garment-detail
supervision; existing product descriptions could supply it through a
training-only image–text term. This is established multimodal supervision,
not a novelty or SOTA claim; it adds no serving operation if adopted.

Before an encoder run, require usable color plus first-description text
for at least **95%** of the 2,004 fit products, and at least **90% unique**
captions. Use only the pinned official-TRAIN 13,283 fit images, never held
or official outcomes. Remove identity shortcuts by rejecting item-ID text;
do not feed product IDs, care instructions or size tables to a text model.
Any failure KILLs this metadata route before training.

If metadata passes, the sole next gate is a <=2-minute frozen SigLIP2 text
alignment diagnostic, using cached pretrained image features. Choose 512
distinct fit products by fixed hash, one image per product by fixed hash,
and each image's nearest different fit product as its hard impostor.
True-description versus impostor-description cosine must win on **65%**
of queries, its product-bootstrap 95% lower bound must exceed **50%**,
and it must exceed a caption permutation within clothing category by
**10 percentage points** with a positive paired lower bound. Bind every
image/cache/tokenizer/model/caption/selection hash. A metadata or alignment
pass is only a falsifier screen; it does not prove learnability or justify
full-budget training. A passing alignment requires a fixed <=2-minute
gradient/cost smoke before any paired quality training gate.
