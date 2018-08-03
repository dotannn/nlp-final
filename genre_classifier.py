from abc import abstractmethod
from metrics import METRIC_NAME_TO_FUNC, eval_result
import numpy as np


class GenreClassifier(object):
    """ abstract genre classifier, all classifier will derive from this class"""

    def __init__(self, n_classes):
        self._n_classes = n_classes

    @abstractmethod
    def train(self, train_data, train_labels, val_data=None, val_labels=None):
        """ function that runs the train process of the model
        Args
            train_data - the train input X
            train_labels - train data labels y
            val_data - data for validation purposes val_X
            val_labels - labels of the validation data val_y

        """
        return

    @abstractmethod
    def predict(self, summaries):
        """ get list of plot summaries as inputs and returns the predicted genres for each plot summary

        Args:
            summaries - list of plot summaries.
        """
        return

    def eval(self, predicted, gt):
        """ evaluate predicted results compared to ground-truth

        Args:
            predicted - the predicted genres.
            gt - ground-truth labels.

        """
        metric_name_to_score = {}
        for metric_name, metric_func in METRIC_NAME_TO_FUNC.items():
            result = eval_result(predicted, gt, eval_fnuc=metric_func)
            metric_name_to_score[metric_name] = np.mean(result)
        return metric_name_to_score


