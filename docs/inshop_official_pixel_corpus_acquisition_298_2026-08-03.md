# Official In-Shop pixel-corpus acquisition and prospective profile

Date: 2026-08-03. This record was committed before evaluating any checkpoint on
the replacement pixels.

## Acquisition

The previously used `img_highres` corpus is invalid for the retrieval benchmark.
The replacement was downloaded from the public Kaggle mirror
`sartma/deepfashion-inshop`, whose archive contains the standard 256-by-256 JPEG
tree and official annotation files. The downloaded outer archive is
`deepfashion-inshop.zip`, 831,457,419 compressed bytes and 858,303,188
uncompressed bytes over 52,716 members. `unzip -t` passes every member. Its SHA-256 is
`07d7f4addaa504eb13b720ec02ee5046de022f2dc7373904e4c8f4137e9b5313`.
Its partition file has SHA-256
`cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`.

The mirror is not treated as authoritative merely because its dimensions look
right. It must reproduce the Proxy Anchor authors' published checkpoint inside
the already registered `[0.917, 0.921]` R@1 interval before it can support a
training result.

## Prospective structural lock

Before checkpoint inference, an independent byte-level pass observed the exact
official partition counts:

| split | rows | identities | within-split duplicate SHA-256 groups |
| --- | ---: | ---: | ---: |
| train | 25,882 | 3,997 | 0 |
| query | 14,218 | 3,985 | 0 |
| gallery | 12,612 | 3,985 | 0 |

There are zero byte-identical groups across train/query, train/gallery, or
query/gallery. These values replace the profiles measured on `img_highres`; the
old duplicate-content observations were properties of the wrong pixel corpus.
The exporter constants are locked to these values before the deciding inference.

The extracted root is new and preserves the invalid corpus for forensic
comparison. No candidate training is authorized until the published-checkpoint
diagnostic passes.
