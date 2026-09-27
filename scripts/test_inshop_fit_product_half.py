"""Small invariance check for the nested product subset."""

from train_inshop_siglip2_unseen_gallery import half_fit_products


def test_half_fit_keeps_complete_products() -> None:
    labels = ("a", "a", "b", "b", "c", "c", "d", "d", "held")
    fit = tuple(range(8))
    rows = half_fit_products(labels, fit)
    assert len({labels[row] for row in rows}) == 2
    assert len(rows) == 4
    assert set(rows).issubset(fit)
    assert rows == tuple(row for row in fit if labels[row] in {labels[i] for i in rows})
    assert rows == half_fit_products(labels, fit)
