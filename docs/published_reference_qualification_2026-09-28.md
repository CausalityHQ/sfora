# Published reference qualification, 28 September 2026

**Decision: retain UNICOM's dated 91.2% SOP and 96.7% In-Shop R@1 gates;
do not claim that this bounded screen certifies the current global frontier.**
These are verified published numbers, not new local measurements. No weights,
images or embeddings were downloaded, no inference/training ran, and no closed
training recipe is reopened by this screen.

Additional accessible primary-source checks: [STIR arXiv2304.13393v1](https://arxiv.org/html/2304.13393v1)
Table1 reports embedding-only ViT-Triplet SOP/In-Shop R@1 of86.5%/92.1%;
STIR-Symmetric with5 reranked image candidates reports88.3%/95.0%.
Neither exceeds the existing dated UNICOM gates. The second row includes
pixel-level pairwise reranking and must be identified as a different serving
pipeline. Table2 uses mAP@5/@10, not our mAP@R. These are published values,
not reproduced measurements or a hardware-matched latency comparison.

[AE-SVC, ICLR2025/arXiv2410.07022v2](https://arxiv.org/html/2410.07022v2)
states that its projection/distillation is fitted to the reference corpus,
with queries unavailable, and reports main results as mAP@k curves and search
operation counts. This does not supply a qualified stronger supervised
TRAIN-only official R@1 target or measured image-to-result latency. No AE-SVC
training/head/whitening intervention is selected from this check. The first
ICLR proceedings PDF fetch timed out; the accessible arXivv2 text supplied
the protocol check, without retrying the failed endpoint. These two checks
still do not certify the complete current published frontier.

| Primary source and version | SOP TEST R@1 (%) | In-Shop official query/gallery R@1 (%) | Qualification |
| --- | ---: | ---: | --- |
| [UNICOM, arXiv2304.05884v1](https://arxiv.org/html/2304.05884v1), Table4 | 91.2 | 96.7 | Supervised ViT-L/14-336; strongest verified values among these screened tables |
| [Three Things to Know about Deep Metric Learning, arXiv2412.12432v1](https://arxiv.org/html/2412.12432v1), Table5 | 90.8 | — | DiHT-initialized ViT-L/14, 512-D, RS@k; paper evaluates no In-Shop dataset |
| [CHEST, arXiv2510.05643v1](https://arxiv.org/html/2510.05643v1), Tables4–5 | 88.2 | 94.5 | Best proposed SOP row is Euclidean ViT-B, 1024-D; its 512-D hyperbolic SOP row is88.0%, not88.2%. In-Shop both rows94.5% |
| [Rethinking Metric Learning, IEEE Access2025, DOI10.1109/ACCESS.2025.3637551](https://cau.scholarworks.kr/item/fd5d2186-cb02-4e28-b085-8fb566315ba1) | Unverified | Unverified | Institutional abstract confirms four datasets; numerical comparison table unavailable through current access |

UNICOM Table4 is supervised dataset fine-tuning, whereas Table2 is unsupervised
retrieval. Do not substitute Table2's SOP74.5% or In-Shop86.7% for the target.
Its Appendix Table11 states SOP59551/11318 training images/classes and
60502/11316 test images/classes; In-Shop25882/3997 train and26830/3985
combined test images/classes. The deployed official In-Shop protocol uses
14218 queries and12612 gallery images. SOP's test images query the test gallery
with their own image excluded. Neither protocol equals our TRAIN identity
holdout. UNICOM's LAION400M pretraining and336-pixel encoder also require
explicit data, representation and preprocessing qualification before a matched
system comparison. A pretrained public encoder is not automatically the
dataset-finetuned model that produced Table4.

CHEST's best-architecture statements compare the backbones in its own tables,
not UNICOM L/14-336. Its published In-Shop mAP@R63.6% and SOP67.3% are from
the Euclidean1024-D rows; these are not Sfora's TRAIN-holdout mAP values.
Three Things' best SOP90.8% is R@1; its97.7% is R@10. Neither source supplies
a matched full decode-to-top-k p99 comparator on DGX GB10.

The IEEE Access institutional PDF was text-readable through the browser but
its result tables were not. Two browser page-render attempts failed. Direct
fetches from both devbox and DGX returned HTTP404 at the recorded institutional
bitstream URL. The alternative institutional landing page exposes the DOI and
abstract, not the missing table. This is an access/verification gap, not evidence
of a lower result. An unidentified OpenReview2026 PDF returned a browser
challenge and remains unqualified. Active-learning results with partial labels,
LookBench/Moda results, and architecture-specific claims do not establish an
absolute standard-protocol joint target. Do not treat this screen as exhaustive.

## Consequences and next validation gates

The older exploratory local official results remain SOP91.7419% and
In-Shop95.4823% R@1. The latter is1.2177 percentage points below the dated
96.7% reference. The newer freeze has no official result, and no matched
end-to-end speed win has been demonstrated. No interpolation or TRAIN-to-TEST
projection closes either requirement.

1. Resolve or explicitly carry the unverified newer-paper tables before a
   strongest-published claim. Keep source version, metric column and official
   protocol attached to every comparator.
2. Before another GPU quality arm, freeze one causally distinct intervention,
   its TRAIN-only decision rule and the action enabled by its outcome. A paper
   title or a missing artifact alone does not authorize another loss sweep.
3. Any surviving method needs packed-score R@1/mAP@R, uncertainty and matched
   controls before official confirmation and CUB/Cars transfer qualification.
4. A product speed claim still requires matched full image-to-top-k batch1/32
   timing; p99 requires10000 interleaved paired calls and a confidence interval.

DGX compute-process query was empty at this checkpoint. All prior Rust tests,
thread/cache gates, artifact preflight and reviews are terminal. The protected
runtime top-k source remains unchanged and unpromoted. Full joint goal remains
active and unmet; this turn provides reference verification, not quality gain.

## Fresh author/publisher route, 28 September evening

The authors' [CAU lab publication page](https://sites.google.com/view/cau-cvml/cvmlcau/journal)
links ESA directly to [IEEE document11269835](https://ieeexplore.ieee.org/document/11269835).
That publisher endpoint returned a JavaScript/robot-verification page; its normal
public stamp endpoint was inaccessible through the browser. No new numerical
table was recovered. The old failed institutional bitstream was not retried.

The same author page identified a further paper,
[Confidence Controls Deep Metric Learning](https://link.springer.com/article/10.1007/s10994-026-07032-y),
published31 March2026, Machine Learning115 article77. The publisher exposes an
abstract and subscription preview, not protocol-qualified result tables. Public
table1/2 endpoints were inaccessible. No accessible preprint emerged from the
bounded exact-title search. SOP/In-Shop values, model/pretraining, dimensions,
official protocol and full-system speed remain unverified for this paper.
The abstract's relative improvement claim is not an absolute benchmark value
and cannot define or clear either production quality gate.

Neither route certifies the global frontier or changes the dated numerical
references. No purchase, credentials, training, model selection or production
change followed. These exact failed publisher routes are recorded to avoid
repeating them as fresh evidence in later turns.
