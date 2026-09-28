#!/usr/bin/env python3
"""Closed-form counterexample, not an image-model training experiment."""


def main():
    # Two-output linear toy: fit target diag(1, .5), held target diag(1, 0).
    # Same starting zero matrix and exposure; projected update keeps only (0,0).
    # Closed-form steps at eta=.005. No optimizer or data fitting is executed.
    dense, rank1 = [], []
    for step in range(101):
        q = 1 - 0.995**step
        dense.append((1 - q) ** 2 + (0.5 * q) ** 2)
        rank1.append((1 - q) ** 2)
    assert all(b < a for a, b in zip(dense, dense[1:]))
    assert all(b < a for a, b in zip(rank1, rank1[1:]))
    assert all(b < a for a, b in zip(dense[1:], rank1[1:]))
    print(
        f"PASS both held errors improve throughout; rank1 better: dense100={dense[-1]:.9f}, rank1={rank1[-1]:.9f}"
    )
    print(
        "This disproves a universal inference from monotone absolute held improvement; it does not predict PE LoRA quality or isolate factor-Adam behavior."
    )


if __name__ == "__main__":
    main()
