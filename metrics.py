import numpy as np
from torch import sigmoid

def recall(predicted, gt):
    return len(set(predicted) & set(gt)) / float(len(gt))


def precision(predicted, gt):
    return len(set(predicted) & set(gt)) / float(len(predicted))


def f1(predicted, gt):
    recall_val = recall(predicted, gt)
    precision_val = precision(predicted, gt)
    if {recall_val, precision_val} == {0}:
        return 0.
    return 2 * recall_val * precision_val / (precision_val + recall_val)


def jaccard_index(predicted, gt):
    return len(set(predicted) & set(gt)) / float(len(set(predicted + gt)))


def eval_result(preds, gts, eval_fnuc=jaccard_index):
    if len(preds) != len(gts):
        raise RuntimeError("predicted and gt lists must have same size! predicted = %d, gt = %d" % (len(preds), len(gts)))
    return [eval_fnuc(p, g) for p,g in zip(preds, gts)]



METRIC_NAME_TO_FUNC = {
    "recall": recall,
    "precision": precision,
    "f1": f1,
    # "jaccard_index": jaccard_index,
}

