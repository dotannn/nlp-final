

def jaccard_index(predicted, gt):
    return len(set(predicted) & set(gt)) / float(len(set(predicted + gt)))


def total_jaccard(preds, gts):
    if len(preds) != len(gts):
        raise RuntimeError("predicted and gt lists must have same size! predicted = %d, gt = %d" % (len(preds), len(gts)))
    return [jaccard_index(p, g) for p,g in zip(preds, gts)]
