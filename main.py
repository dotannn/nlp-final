import functools
import random
import numpy as np
import string

from nltk.stem import *
from nltk.corpus import reuters

from lda import LLDA



#TODO we'll do:
## 1. naivebayes with BoW features
## 2. some topic modeling with NLTK and L-LDA
## 3. our super-cool RNN
## 4. data handling
## 5. metrics handling & reports




stemmer = SnowballStemmer("english")
print(stemmer.stem("morphy"))

idlist = random.sample(reuters.fileids(), 100)

labels = []
corpus = []
for id in idlist:
    labels.append(reuters.categories(id))
    corpus.append([x.lower() for x in reuters.words(id) if x[0] in string.ascii_letters])
    reuters.words(id).close()
labelset = list(set(functools.reduce(list.__add__, labels)))


llda = LLDA()


llda.set_corpus(labelset, corpus, labels)
print ("M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), 20))

for i in range(15):
    print("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
print ("perplexity : %.4f" % llda.perplexity())

phi = llda.phi()
for k, label in enumerate(labelset):
    print ("\n-- label %d : %s" % (k, label))
    for w in np.argsort(-phi[k])[:20]:
        print ("%s: %.4f" % (llda.vocas[w], phi[k,w]))


llda.