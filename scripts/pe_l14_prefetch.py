"""One pending CPU input; the caller must not use CPU RNG while it runs."""

from concurrent.futures import ThreadPoolExecutor


def prefetch(prepare, count):
    if not count:
        return
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(prepare, 0)
        for index in range(count):
            current = pending.result()
            if index + 1 < count:
                pending = worker.submit(prepare, index + 1)
            yield current
