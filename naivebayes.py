from classifier import GenreClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random

class NaiveBayesGenreClassifier(GenreClassifier):
    def __init__(self,):
        self.text_clf = None

    def train(self, summaries, labels):

        norm_labels = [random.choice(label) if type(label) == list else label for label in labels]

        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        self.text_clf = text_clf.fit(summaries, norm_labels)

    def predict(self, summaries):
        predicted = self.text_clf.predict(summaries)
        return [[predict] for predict in predicted]
