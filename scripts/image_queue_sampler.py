"""Deterministic image queues for an externally supplied 100 x 64 schedule."""

from hashlib import sha256
from numbers import Integral


def queue_schedule(target, original_batches, seed):
    """Return row-index lists with the original class slots and queued images.

    ``target[row]`` must label every row with a class in contiguous ``0..C-1``.
    The original schedule must have 100 batches of 64 distinct classes and
    cover all C classes. Member ordinals are global target row indices; cycles
    start at zero and rank by SHA256 of ASCII
    ``image-queue-v1:seed:class:cycle:ordinal``, then by ordinal on a hash tie.
    Each class exhausts its current cycle before beginning its next one.
    Inputs are not changed, and no RNG is used. Malformed inputs raise ValueError.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    try:
        target = list(target)
        batches = [list(batch) for batch in original_batches]
    except TypeError as exc:
        raise ValueError("target and batches must be iterable row sequences") from exc
    if not target:
        raise ValueError("target must not be empty")

    members = {}
    for row, label in enumerate(target):
        if isinstance(label, bool) or not isinstance(label, Integral) or label < 0:
            raise ValueError("target classes must be nonnegative integers")
        label = int(label)
        target[row] = label
        members.setdefault(label, []).append(row)
    class_count = len(members)
    if set(members) != set(range(class_count)):
        raise ValueError("target classes must be contiguous from zero")
    if class_count < 64:
        raise ValueError("target must contain at least 64 classes")
    if len(batches) != 100:
        raise ValueError("original schedule must contain exactly 100 batches")

    class_batches = []
    seen = set()
    for batch in batches:
        if len(batch) != 64:
            raise ValueError("each original batch must contain exactly 64 rows")
        classes = []
        for row in batch:
            if (isinstance(row, bool) or not isinstance(row, Integral)
                    or not 0 <= row < len(target)):
                raise ValueError("original row indices must be integers in target range")
            classes.append(target[int(row)])
        if len(set(classes)) != 64:
            raise ValueError("each original batch must contain 64 distinct classes")
        seen.update(classes)
        class_batches.append(classes)
    if seen != set(members):
        raise ValueError("original schedule must cover all target classes")

    uses = [0] * class_count
    queues = [[] for _ in range(class_count)]
    result = []
    for classes in class_batches:
        batch = []
        for label in classes:
            cycle, offset = divmod(uses[label], len(members[label]))
            if offset == 0:
                queues[label] = sorted(
                    members[label],
                    key=lambda row: (sha256(
                        f"image-queue-v1:{seed}:{label}:{cycle}:{row}".encode("ascii")
                    ).digest(), row),
                )
            batch.append(queues[label][offset])
            uses[label] += 1
        result.append(batch)
    return result
