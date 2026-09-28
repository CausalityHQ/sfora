#!/usr/bin/env python3
"""Pixel-parity fixtures across a real spawned preprocessing worker."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from PIL import Image
from probe_sop_preprocessing_worker import check_pixels, initialize, process


def main():
    snapshot = Path(
        "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
    )
    initialize(snapshot, 20)
    images = []
    for i, (mode, channels) in enumerate((("RGB", 3), ("L", 1), ("P", 1), ("RGBA", 4))):
        size = (37 + i, 61 - i)
        values = bytes(
            (i * 31 + j * 17) % 256 for j in range(size[0] * size[1] * channels)
        )
        image = Image.frombytes(mode, size, values)
        if mode == "P":
            image.putpalette([v for k in range(256) for v in (k, 255 - k, k * 3 % 256)])
        images.append(image)
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(snapshot, 1),
    ) as worker:
        for batch in ([images[0]], images * 8):
            reference = process(batch)
            actual = worker.submit(process, batch).result(timeout=30)
            check_pixels(reference, actual)
            if not reference.is_contiguous():
                try:
                    check_pixels(reference, actual.contiguous())
                except AssertionError:
                    pass
                else:
                    raise AssertionError("changed pixel strides accepted")
    print("PASS real spawn/IPC bitwise pixels, RGB/L/P/RGBA and batch1/32")


if __name__ == "__main__":
    main()
