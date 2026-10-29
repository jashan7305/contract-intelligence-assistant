from src.metrics import char_precision_recall, merge_intervals

def test_exact_span():
    assert char_precision_recall([(10, 20)], [(10, 20)]) == (1.0, 1.0)

def test_partial_overlap():
    p, r = char_precision_recall([(0, 5)], [(2, 8)])
    assert round(p, 3) == round(3/5, 3)
    assert round(r, 3) == round(3/6, 3)

def test_merge_overlaps():
    assert merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
