import spacy
import nltk
import gensim
import numpy as np

# spacy.load('en')
from spacy.lang.en import English
from nltk.stem.wordnet import WordNetLemmatizer

from classifier import GenreClassifier


nltk.download('stopwords')



class LLDA:
    # from https://github.com/shuyo/iir/blob/master/lda/llda.py
    def __init__(self, alpha=0.001, beta=0.001):
        self.alpha = alpha
        self.beta = beta

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label):
        if not label: return np.ones(len(self.labelmap))
        vec = np.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        labelset.insert(0, "common")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.labels = np.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)
        V = len(self.vocas)

        self.z_m_n = []
        self.n_m_z = np.zeros((M, self.K), dtype=int)
        self.n_z_t = np.zeros((self.K, V), dtype=int)
        self.n_z = np.zeros(self.K, dtype=int)

        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            #z_n = [label[x] for x in numpy.random.randint(len(label), size=N_m)]
            z_n = [np.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            for n in range(len(doc)):
                t = doc[n]
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = label * (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, np.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

class LDATopicModeling(GenreClassifier):
    def __init__(self, topics):
        super(LDATopicModeling).__init__(classes=topics)
        self._lemmatizer = WordNetLemmatizer()
        self._parser = English()
        self._en_stop = set(nltk.corpus.stopwords.words('english'))

    def _preprocess(self, plot, min_len=4):
        tokens = self._parser(plot)
        # filter out too short words:
        tokens = [token for token in tokens if len( token ) > min_len]

        # filter out stop words
        tokens = [token for token in tokens if token not in self._en_stop]

        # lemmatize tokens
        tokens = [self._lemmatizer.lemmatize( token ) for token in tokens]
        return tokens

    def train(self, corpus):
        dictionary = gensim.corpora.Dictionary.load( 'dictionary.gensim' )

        self._ldamodel = gensim.models.ldamodel.LdaModel( corpus=corpus,
                                                          num_topics=1,
                                                          id2word=dictionary)

    def predict(self, summaries):
        self._ldamodel.p





