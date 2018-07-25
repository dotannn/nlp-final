from abc import abstractmethod


class GenreClassifier(object):
    def __init__(self, n_classes):
        self._n_classes = n_classes

    @abstractmethod
    def train(self, corpus):
        return

    @abstractmethod
    def eval(self, summaries, labels):
        return

    @abstractmethod
    def predict(self, summaries):
        return

