#!/usr/bin/env python3
"""Exhaustive byte-domain and public preprocessing parity check."""

from PIL import Image
import torch
from torchvision.transforms.v2 import functional as tvf

from probe_sop_normalization_lookup import initialize_lookup, lookup, process_lookup
from probe_sop_preprocessing_worker import check_pixels, process


def main():
    initialize_lookup()
    values = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    for count in (1, 32):
        for channels_last in (False, True):
            pixels = values.expand(count, 3, 16, 16).contiguous(
                memory_format=torch.channels_last
                if channels_last
                else torch.contiguous_format
            )
            expected = tvf.normalize(pixels.float(), [127.5] * 3, [127.5] * 3)
            check_pixels(expected, lookup(pixels))
    try:
        lookup(values.float())
    except AssertionError:
        pass
    else:
        raise AssertionError("unsupported float pixels accepted")
    images = []
    for i, (mode, channels) in enumerate((("RGB", 3), ("L", 1), ("P", 1), ("RGBA", 4))):
        size = (37 + i, 61 - i)
        image = Image.frombytes(
            mode,
            size,
            bytes((j * 17 + i * 31) % 256 for j in range(size[0] * size[1] * channels)),
        )
        if mode == "P":
            image.putpalette([v for k in range(256) for v in (k, 255 - k, k * 3 % 256)])
        images.append(image)
    for batch in ([images[0]], images * 8):
        check_pixels(process(batch), process_lookup(batch))
    print("PASS all256 bytes, contiguous/channels-last, B1/B32 and RGB/L/P/RGBA")


if __name__ == "__main__":
    main()
