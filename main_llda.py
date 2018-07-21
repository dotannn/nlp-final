import pandas as pd
from utils import *
from pathlib import Path
import sys

from llda import LLDA



def simple_parse_data(path, chunksizen, n_samples):
    data = pd.read_csv( path, header=None, chunksize=chunksizen, nrows=n_samples )



TRAIN_MOVIE_GENRES_PLOT_CSV = Path("data/trn_movie_genres_plot.csv")
VAL_MOVIE_GENRES_PLOT_CSV = Path("data/val_movie_genres_plot.csv")
GENRES_TYPES_FILE = "data/genres.txt"

ALPHA = 0.001
BETHA = 0.001
NUM_OF_ITERATIONS = 2

MAX_SAMPLES = 300
CHUNKSIZE = 24000
print("###################################################")
print("################ LLDA BASELINE ####################")
print("###################################################")


print("reading data set ... start")

print("")
if MAX_SAMPLES > 0:
    train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES )
    val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES // 10 )
else:
    train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )
    val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )

txt_train, train_labels = get_all_tokenized( train_data, 1, do_tokenize=False)
txt_val, val_labels = get_all_tokenized( val_data, 1, do_tokenize=False)


txt_train = [t.split() for t in txt_train]
txt_val = [t.split() for t in txt_val]

label_set = {l for labels in train_labels for l in labels}
label_set.update({l for labels in val_labels for l in labels})
label_set = list(label_set)


t_set = {l for labels in txt_train for l in labels}
t_set = list(t_set)

print("num of sampels: ",  len(txt_train))

print("init LLDA ... start")



print("reading data set ... done")
llda = LLDA(len(label_set), ALPHA, BETHA)
llda.set_corpus(label_set, txt_train, train_labels)
print("init LLDA ... done")

print("init learn ... start")
for i in range(NUM_OF_ITERATIONS):
    print("iter %s/%s -- %s %%" % (i , NUM_OF_ITERATIONS, (float(i) * 100) / NUM_OF_ITERATIONS) )
    llda.inference()
print("iter %s/%s -- %s %%" % (NUM_OF_ITERATIONS, NUM_OF_ITERATIONS, 100))

print("init learn ... done")
phi = llda.phi()
for v, voca in enumerate(llda.vocas):
    # import ipdb; ipdb.set_trace() # NO_COMMIT
    print(','.join([voca]+[str(x) for x in llda.n_z_t[:,v]]))
    # import ipdb; ipdb.set_trace() # NO_COMMIT
    print(','.join([voca]+[str(x) for x in phi[:,v]]))
import ipdb; ipdb.set_trace() # NO_COMMIT)
