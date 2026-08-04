# Repository-blind Fable outcome brief

Date: 2026-08-04. Status: scheduled to launch after the Fable session quota
resets. The process has filesystem tools disabled and only primary-source web
search/fetch enabled.

This round deliberately supplies no candidate mechanism and no catalogue of
failed ideas. It asks an independent principal scientist to solve supervised
open-set image similarity learning: train on standard training classes and
retrieve unseen classes on CUB-200-2011, Cars196, DeepFashion In-Shop or SOP.
The required deployment is one single-model, single-view cosine descriptor.
Training may use benchmark training pixels and class labels plus standard
ImageNet initialization, but no external teacher/foundation model after
initialization, generated data, text, extra annotation, test adaptation,
reranking or ensemble. Preferred cost is at most 1.5 times ordinary training.

The only performance context supplied is:

- broadly comparable ResNet/cosine R@1 horizons: 0.766 CUB, 0.949 Cars196 and
  0.939 In-Shop;
- higher-capacity single-model observations: Potential Field 0.878 CUB and
  0.882 SOP with ViT, and 0.947 Cars196 with DINO;
- corrected local BN-Inception Proxy Anchor frozen-final In-Shop R@1 0.9137,
  explicitly identified as a weak reference rather than the target.

The model must return at most one genuinely novel training relation, loss,
architecture or optimization object with a quantitative chance to cross a
matched-capacity horizon and replicate on a second dataset, or `NONE`. Before a
proposal it must search adversarially across primary literature and reject a
renamed method or unoriginal component conjunction. A proposal must define its
mathematical object, closest primary sources and distinction, minimum
training-only provenance measurement, decisive controls, frozen final forecasts,
falsifier and quantitative horizon-crossing argument. If it cannot do so, it must
return `NONE` and the single measurement most likely to unlock a defensible
method.

This corrects the earlier retry brief, which gave only the narrower ResNet/cosine
horizons and named occupied mechanisms. The current brief exposes the stronger
capacity track and does not steer invention toward or away from any mechanism.
