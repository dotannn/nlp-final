from collections import defaultdict
from utils import *
from prepare_data import generate_input_csv
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

from rnn import RNNGenreClassifier

TRAIN_MOVIE_GENRES_PLOT_CSV = Path("data/trn_movie_genres_plot.csv")
VAL_MOVIE_GENRES_PLOT_CSV = Path("data/val_movie_genres_plot.csv")
GENRES_TYPES_FILE = "data/genres.txt"

TRAIN_IDS = Path("tmp/train_ids.npy")
VAL_IDS = Path("tmp/val_ids.npy")
TRAIN_LABELS = Path("tmp/train_labels.npy")
VAL_LABELS = Path("tmp/val_labels.npy")
IDX_TO_TOKEN = Path("tmp/idx_to_token.pkl")

USE_CACHE = True
MAX_SAMPLES = 3000

# Load texts and labels:
if TRAIN_IDS.exists() and VAL_IDS.exists() and IDX_TO_TOKEN.exists() and USE_CACHE:
    train_ids = np.load(TRAIN_IDS)
    val_ids = np.load(VAL_IDS)
    train_labels = np.load(TRAIN_LABELS)
    val_labels = np.load(VAL_LABELS)
    idx_to_token = pickle.load(Path(IDX_TO_TOKEN).open('rb'))
    vocab = Vocabulary(idx_to_token)
else:
    if not TRAIN_MOVIE_GENRES_PLOT_CSV.exists() and USE_CACHE:
        generate_input_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, VAL_MOVIE_GENRES_PLOT_CSV )

    CHUNKSIZE = 24000
    if MAX_SAMPLES > 0:
        train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES )
        val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES // 10 )
    else:
        train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )
        val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )

    token_train, train_labels = get_all_tokenized( train_data, 1 )
    token_val, val_labels = get_all_tokenized( val_data, 1)

    train_ids, val_ids, vocab = numericalize_texts(token_train, token_val)

    np.save(str(TRAIN_IDS), train_ids)
    np.save(str(VAL_IDS), val_ids)
    np.save(str(TRAIN_LABELS), train_labels)
    np.save(str(VAL_LABELS), val_labels)
    pickle.dump(vocab._idx_to_token, open(IDX_TO_TOKEN, 'wb'))

GENRES = Path( GENRES_TYPES_FILE ).open( "r" ).readlines()
GENRES = list( map( lambda x: x.strip(), GENRES ) )
n_genres = len(GENRES)

genre_classifier = RNNGenreClassifier(n_classes=n_genres, vocab=vocab)

genre_classifier.fit(train_ids, train_labels, batch_size=16, val_ids=val_ids, val_labels=val_labels)

# TODO , for multi-label replace crit to binary_cross_entropy_with_logits

