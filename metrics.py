import numpy as np
from torch import sigmoid


def jaccard_index(predicted, gt, thresh=0.5):
    predicted = sigmoid(predicted)
    gt = np.where(gt>thresh)[0].tolist()
    predicted = np.where(predicted>thresh)[0].tolist()

    return len(set(predicted) & set(gt)) / float(len(set(predicted + gt)))


def total_jaccard(preds, gts, thresh=0.5):
    if len(preds) != len(gts):
        raise RuntimeError("predicted and gt lists must have same size! predicted = %d, gt = %d" % (len(preds), len(gts)))
    return np.array([jaccard_index(p, g, thresh) for p,g in zip(preds, gts)]).mean()
