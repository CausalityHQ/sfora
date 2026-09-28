#!/usr/bin/env python3
"""One-worker ordering, one pending item, exception propagation."""

from threading import Event, get_ident

from pe_l14_prefetch import prefetch


def main():
    owner = get_ident()
    started, release = Event(), Event()
    calls = []

    def prepare(index):
        assert get_ident() != owner
        calls.append(index)
        if index == 1:
            started.set()
            assert release.wait(2)
        return index

    iterator = prefetch(prepare, 3)
    assert next(iterator) == 0
    assert started.wait(2) and calls == [0, 1]
    release.set()
    assert list(iterator) == [1, 2] and calls == [0, 1, 2]

    def broken(index):
        if index == 1:
            raise ValueError("worker failure")
        return index

    iterator = prefetch(broken, 2)
    assert next(iterator) == 0
    try:
        next(iterator)
    except ValueError as error:
        assert str(error) == "worker failure"
    else:
        raise AssertionError("worker exception was swallowed")
    assert list(prefetch(prepare, 0)) == []
    print("PASS bounded prefetch ordering, one pending item, worker exceptions")


if __name__ == "__main__":
    main()
