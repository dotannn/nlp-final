from .classifier import GenreClassifier


class NaiveBayesGenreClassifier(GenreClassifier):
    def __init__(self, classes):
        super(NaiveBayesGenreClassifier).__init__(classes)

    def train(self, corpus):
        pass
