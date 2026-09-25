"""In-Shop fit/holdout keeps all products with no second image in training."""

from preflight_inshop_siglip2_coverage import fit_and_holdout


def test_singleton_products_never_enter_retrieval_holdout() -> None:
    labels = tuple(label for index in range(10) for label in (str(index), str(index))) + (
        "singleton",
    )
    fit, held = fit_and_holdout(labels)
    assert 20 in fit and 20 not in held
    assert {labels[index] for index in fit}.isdisjoint({labels[index] for index in held})
    assert fit_and_holdout(labels) == (fit, held)
