from abc import abstractmethod
from metrics import METRIC_NAME_TO_FUNC, eval_result
import numpy as np

class GenreClassifier(object):
    def __init__(self, n_classes):
        self._n_classes = n_classes

    @abstractmethod
    def train(self, summaries, labels):
        return

    @abstractmethod
    def predict(self, summaries):
        return

    def eval(self, predicted, gt):
        metric_name_to_score = {}
        for metric_name, metric_func in METRIC_NAME_TO_FUNC.items():
            result = eval_result(predicted, gt, eval_fnuc=metric_func)
            metric_name_to_score[metric_name] = np.mean(result)
        return metric_name_to_score


