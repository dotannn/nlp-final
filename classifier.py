from abc import abstractmethod


class GenreClassifier(object):
    def __init__(self, classes):
        self._classes = classes


    @abstractmethod
    def train(self, corpus):
        return

    @abstractmethod
    def eval(self, summaries, labels):
        return

    @abstractmethod
    def predict(self, summaries):
        return

