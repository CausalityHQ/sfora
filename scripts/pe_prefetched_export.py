"""One pending CPU preparation, unchanged B32 authority and atomic writer."""

from contextlib import closing
from pe_l14_prefetch import prefetch
from export_sop_siglip2_train import export_features


def export_prefetched(rows, prepare, encode, output, *, width):
    def prepare_batch(index):
        batch = rows[index * 32 : (index + 1) * 32]
        return tuple(batch), prepare(batch)

    with closing(prefetch(prepare_batch, (len(rows) + 31) // 32)) as inputs:
        def consume(batch):
            expected, prepared = next(inputs)
            assert tuple(batch) == expected, "prefetched export row order differs"
            return encode(batch, prepared)

        export_features(rows, consume, output, width=width, batch_size=32)
