import math

from diag_sop_ocr_text_overlap_v1 import case_flags, image_rows, summarise, tokens, wilson


def test_tokens_normalise_and_drop_stopwords():
    assert tokens("NONE") == set()
    assert tokens(" None. ") == set()
    assert tokens("Made in CHINA for the USA with Sony") == {"sony"}
    # length >= 3 runs, digit-containing runs of length >= 2, punctuation splits
    assert tokens("Logitech M-510 x2 a 7 OK B4") == {"logitech", "510", "x2", "b4"}
    assert tokens("") == set()


def test_overlap_and_discriminative():
    q = {"sony", "x100"}
    assert case_flags(q, set(), [set(), {"x100"}]) == {
        "has_text": True, "overlap_pos": True, "overlap_wrong": False, "discriminative": True}
    assert case_flags(q, {"sony"}, [{"sony"}]) == {
        "has_text": True, "overlap_pos": True, "overlap_wrong": True, "discriminative": False}
    assert case_flags(set(), {"sony"}, [{"sony"}]) == {
        "has_text": False, "overlap_pos": False, "overlap_wrong": False, "discriminative": False}
    assert case_flags(q, set(), [])["overlap_pos"] is False


def test_wilson_values():
    lo, hi = wilson(30, 300)
    assert math.isclose(lo, 0.0710, abs_tol=5e-4) and math.isclose(hi, 0.1393, abs_tol=5e-4)
    lo, hi = wilson(0, 10)
    assert abs(lo) < 1e-12 and math.isclose(hi, 0.2775, abs_tol=5e-4)
    assert all(math.isnan(x) for x in wilson(0, 0))
    s = summarise([{"has_text": True, "overlap_pos": True, "overlap_wrong": False, "discriminative": True}] * 3)
    assert s["n"] == 3 and s["discriminative"]["count"] == 3 and s["overlap_wrong"]["fraction"] == 0


def test_image_rows_deduplicate():
    cases = [{"query": 5, "top1": 1, "positives": [2, 3]}, {"query": 1, "top1": 5, "positives": [3, 9]}]
    assert image_rows(cases) == [1, 2, 3, 5, 9]
