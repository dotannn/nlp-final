import functools
import random
import numpy as np
import string
from collections import defaultdict, Counter
from nltk.stem import *
from nltk.corpus import reuters
import logging
import pandas as pd
from functools import partial
import torch.optim as optim
import pickle
# from torch.utils.data import DataLoader

# from lda import LLDA
from fastai.text import LanguageModelLoader, LanguageModelData, accuracy, TextDataset, SortSampler, SortishSampler, \
    ModelData, TextModel, RNN_Learner, to_gpu, DataLoader
from fastai.lm_rnn import *
from abc import *

from utils import *

import spacy
from spacy.symbols import ORTH




# TRAIN_MOVIE_GENRES_PLOT_CSV = "data/trn_movie_genres_plot.csv"
# VAL_MOVIE_GENRES_PLOT_CSV = "data/val_movie_genres_plot.csv"
#
#
# chunksize=24000
#
#
# trn_data = pd.read_csv(TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=chunksize)
# val_data = pd.read_csv(TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=chunksize)
#
#
# tok_trn, trn_labels = get_all(trn_data, 1)
# tok_val, val_labels = get_all(val_data, 1)


# freq = Counter(p for o in tok_trn for p in o)
# freq.most_common(25)
#
#
# max_vocab = 60000
#
# itos = [o for o,c in freq.most_common(max_vocab)]
# itos.insert(0, '_pad_')
# itos.insert(0, '_unk_')
#
# stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
# vs=len(itos)
#
# trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
# val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
#
# np.save('tmp/trn_ids.npy', trn_lm)
# np.save('tmp/val_ids.npy', val_lm)
#
# pickle.dump(itos, open('tmp/itos.pkl', 'wb'))

trn_lm = np.load('tmp/trn_ids.npy')
val_lm = np.load('tmp/val_ids.npy')

trn_labels = np.load("tmp/trn_labels.npy")
val_labels = np.load("tmp/val_labels.npy")

itos = pickle.load( Path('tmp/itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
vs=len(itos)

em_sz,nh,nl = 400,1150,3

wd=1e-7
bptt=70
bs=24
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData("data", 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


learner= md.get_model(opt_fn, em_sz, nh, nl,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])


from metrics import total_jaccard
learner.metrics = [accuracy] #,total_jaccard]
# learner.freeze_to(-1)
learner.unfreeze()

lr=1e-3
lrs = lr


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

# learner.sched.plot()


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)

learner.save_encoder('lm1_enc')

# learner.sched.plot_loss()

##### TRAIN CLASSIFIER:
#
# trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
# val_clas = np.array([[stoi[o] for o in p] for p in tok_val])
#
#
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

from genre import Genre
c = len(Genre.ALL_GENRE)

trn_clas = trn_lm
val_clas = val_lm

reduced_train_labels = np.array([ l[0] for l in trn_labels ])
reduced_val_labels = np.array([ l[0] for l in val_labels ])

c = int(reduced_train_labels.max())

trn_ds = TextDataset(trn_clas, reduced_train_labels)
val_ds = TextDataset(val_clas, reduced_val_labels)

trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, num_workers=1, transpose=True, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, num_workers=1, transpose=True, pad_idx=1, sampler=val_samp)

md = ModelData("data", trn_dl, val_dl)

dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5

m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 128, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = [accuracy] #total_jaccard]

lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])

lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

wd = 1e-7
wd = 0
learn.load_encoder('lm1_enc')

learn.freeze_to(-1)

# learn.lr_find(lrs/1000)
# learn.sched.plot()

learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))
learn.save('clas_0')


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
learn.save('clas_1')
learn.unfreeze()
learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))

learn.save('clas_2')

## Our supercool RNN:
# 0. prepare the data in csv: movie-name |  genres | plot
# 1. tokenize:   (download: python -m spacy download en)
#   a.fixup
#   b. add BOS
#   c. add t_up - all uppercase
# 2. replace word to number:
#   a. only 60,002 words:
#   b. all low frequency words would go to _unk_  = 0 [use default dict]
#   c. add _pad_ word = 1
#   d. save the vocabulary!
# 2. [opt] pre-train on wikipedia [or use pretrained network]
# 3. learn language model from plots
#   a. first single epoch on last layer (last layer is the embedding weights)
#   b. about 15 epochs on full model unfreezed  - weight decay
#   c. how language model is compared to word2vec -   word2vec is a single embedding matrix (each word have a vector) but
#      language model is dependent on history and context.
#   d. AWD-LSTM,  70 at a time after each epoch would be same 70 (because this is stateful we cant shuffle)
#      so we randomly change the number 70 to other random number)
# 4. learn classifier


#

#
#
#
#
# trn_dl = LanguageModelLoader(np.concatenate(trn_data), bs, bptt)
# val_dl = LanguageModelLoader(np.concatenate(val_data), bs, bptt)
#
#
# md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)