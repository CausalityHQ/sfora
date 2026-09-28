#!/usr/bin/env python3
"""Cheap counterexamples to raw angular displacement and a strict Adam LR bound."""

import math

# A signed coordinate permutation moves every unit vector by 90 degrees while
# preserving cosine scores and signed-int8 code dot products/norms exactly.
x = ((1.0, 0.0), (0.0, 1.0), (math.sqrt(0.5), math.sqrt(0.5)))
y = tuple((-b, a) for a, b in x)


def dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def pack(row):
    return tuple(round(v / max(map(abs, row)) * 127) for v in row)


assert all(
    math.isclose(dot(a, b), dot(c, d), abs_tol=1e-15)
    and dot(pack(a), pack(b)) == dot(pack(c), pack(d))
    for a, c in zip(x, y)
    for b, d in zip(x, y)
)
assert all(
    math.isclose(math.acos(dot(a, b)), math.pi / 2, abs_tol=1e-15) for a, b in zip(x, y)
)

# Both scalar gradients obey clip1. Bias-corrected Adam's second update still
# exceeds lr; lr*updates*sqrt(n) is not a universal update bound.
b1, b2, g1, g2 = 0.9, 0.999, 0.9, 1.0
m = ((1 - b1) * b1 * g1 + (1 - b1) * g2) / (1 - b1**2)
v = ((1 - b2) * b2 * g1**2 + (1 - b2) * g2**2) / (1 - b2**2)
factor = m / (math.sqrt(v) + 1e-8)
assert factor > 1
m = v = total = 0.0
for step in range(1, 101):
    gradient = g1 if step == 1 else g2
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * gradient**2
    total += (m / (1 - b1**step)) / (math.sqrt(v / (1 - b2**step)) + 1e-8)
assert total > 100
print(
    "PASS unchanged float/packed geometry at 90-degree displacement; Adam step2/lr",
    factor,
    "100-step displacement/lr",
    total,
)
