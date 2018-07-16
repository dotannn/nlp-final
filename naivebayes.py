from .classifier import GenreClassifier

# steps:
## 1. tokenization.
## 2. extract features - word frequencies
## 3.


class NaiveBayesGenreClassifier(GenreClassifier):
    def __init__(self, classes):
        super(NaiveBayesGenreClassifier).__init__(classes)

    def train(self, corpus):
        pass
