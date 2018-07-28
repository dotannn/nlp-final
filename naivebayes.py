from genre_classifier import GenreClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random

class NaiveBayesGenreClassifier(GenreClassifier):
    def __init__(self,):
        self.text_clf = None

    def train(self, train_data, train_labels, val_data=None, val_labels=None):

        if isinstance(train_data[0], list):
            train_data = [" ".join(words) for words in train_data]

        norm_labels = [random.choice(label) if type(label) == list else label for label in train_labels]

        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        self.text_clf = text_clf.fit( train_data, norm_labels )

    def predict(self, summaries):
        if isinstance(summaries[0], list):
            summaries = [" ".join(words) for words in summaries]
        predicted = self.text_clf.predict(summaries)
        return [[predict] for predict in predicted]
