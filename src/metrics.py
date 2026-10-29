def merge_intervals(intervals):
    intervals = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]

def covered_chars(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))

def char_precision_recall_by_doc(predicted_by_doc, gold_by_doc):
    """Character-level precision/recall; offsets are compared only within the same document."""
    overlap_total = 0
    predicted_total = 0
    gold_total = 0
    for doc, spans in predicted_by_doc.items():
        predicted_total += covered_chars(spans)
    for doc, spans in gold_by_doc.items():
        gold_total += covered_chars(spans)
    for doc in set(predicted_by_doc) | set(gold_by_doc):
        pred = merge_intervals(predicted_by_doc.get(doc, []))
        gold = merge_intervals(gold_by_doc.get(doc, []))
        i = j = 0
        while i < len(pred) and j < len(gold):
            overlap_total += max(0, min(pred[i][1], gold[j][1]) - max(pred[i][0], gold[j][0]))
            if pred[i][1] < gold[j][1]:
                i += 1
            else:
                j += 1
    precision = overlap_total / predicted_total if predicted_total else 0.0
    recall = overlap_total / gold_total if gold_total else 0.0
    return precision, recall

def char_precision_recall(predicted, gold):
    """Single-document convenience wrapper retained for tests."""
    return char_precision_recall_by_doc({"_": predicted}, {"_": gold})
