from utils import *
from prepare_data import generate_input_csv
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from argparse import ArgumentParser

from vocabulary import Vocabulary
from rnn import RNNGenreClassifier
from naivebayes import NaiveBayesGenreClassifier

# input data:
TRAIN_MOVIE_GENRES_PLOT_CSV = Path("data/trn_movie_genres_plot.csv")
VAL_MOVIE_GENRES_PLOT_CSV = Path("data/val_movie_genres_plot.csv")
GENRES_TYPES_FILE = "data/genres.txt"

# cache paths:
TRAIN_TOKENS = Path("tmp/train_tokens.npy")
TRAIN_TEXTS = Path("tmp/train_texts.npy")
VAL_TOKENS = Path("tmp/val_tokens.npy")
VAL_TEXTS = Path("tmp/val_texts.npy")
TRAIN_LABELS = Path("tmp/train_labels.npy")
VAL_LABELS = Path("tmp/val_labels.npy")
IDX_TO_TOKEN = Path("tmp/idx_to_token.pkl")

# defaults:
USE_CACHE = False
MAX_SAMPLES = -1
BATCH_SIZE = 8

if __name__ == '__main__':
    parser = ArgumentParser(description="NLP@OpenU final project")
    parser.add_argument("-bs", "--batch_size", help="training batch size", type=int, default=BATCH_SIZE)
    parser.add_argument("-m", "--max_samples", help="maximum samples to use in training", type=int, default=MAX_SAMPLES)
    parser.add_argument("-nc", "--no_cache", help="disable caching of input data", action="store_true")

    args = parser.parse_args()

    print( "Args:", args)


    USE_CACHE = not args.no_cache
    MAX_SAMPLES = args.max_samples
    BATCH_SIZE = args.batch_size

    # Load texts and labels:
    if not TRAIN_MOVIE_GENRES_PLOT_CSV.exists() or not USE_CACHE:
        generate_input_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, VAL_MOVIE_GENRES_PLOT_CSV )

    CHUNKSIZE = 24000
    if MAX_SAMPLES > 0:
        train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES )
        val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES // 10 )
    else:
        train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )
        val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )

    if TRAIN_TOKENS.exists() and VAL_TOKENS.exists() and IDX_TO_TOKEN.exists() and USE_CACHE:
        token_train = np.load( TRAIN_TOKENS )
        texts_train = np.load( TRAIN_TEXTS )
        token_val = np.load( VAL_TOKENS )
        texts_val = np.load( VAL_TEXTS )
        train_labels = np.load( TRAIN_LABELS )
        val_labels = np.load( VAL_LABELS )
        idx_to_token = pickle.load( Path( IDX_TO_TOKEN ).open( 'rb' ) )
        vocab = Vocabulary( idx_to_token )

    else:
        texts_train, token_train, train_labels = get_all_tokenized( train_data, 1 )
        texts_val, token_val, val_labels = get_all_tokenized( val_data, 1 )
        vocab = Vocabulary.from_text( token_train )

    if USE_CACHE:
        np.save( str( TRAIN_TOKENS ), token_train )
        np.save( str( TRAIN_TEXTS ), texts_train )
        np.save( str( VAL_TOKENS ), token_val )
        np.save( str( VAL_TEXTS ), texts_val )
        np.save( str( TRAIN_LABELS ), train_labels )
        np.save( str( VAL_LABELS ), val_labels )
        pickle.dump( vocab._idx_to_token, open( IDX_TO_TOKEN, 'wb' ) )

    # load genres:
    GENRES = Path( GENRES_TYPES_FILE ).open( "r" ).readlines()
    GENRES = list( map( lambda x: x.strip(), GENRES ) )
    n_genres = len( GENRES )

    print( "Train Ours embedding_size=250, n_hidden_activations=720" )

    genre_classifier = RNNGenreClassifier( n_classes=n_genres, vocab=vocab, batch_size=BATCH_SIZE,
                                           n_hidden_activations=720)

    genre_classifier.train( train_data=token_train, train_labels=train_labels, val_data=token_val,
                            val_labels=val_labels )

    val_predict = genre_classifier.predict( token_val )
    res = genre_classifier.eval( val_predict, val_labels )

    print( "Finished" )
    print( "Train baseline" )

    baseline = NaiveBayesGenreClassifier()
    baseline.train( texts_train, train_labels )

    val_predict_baseline = baseline.predict( texts_val )

    res_baseline = baseline.eval( val_predict_baseline, val_labels )

    from tabulate import tabulate

    print( tabulate( [['Naive-bayes(baseline)', res_baseline['precision'], res_baseline['recall'], res_baseline['f1'],
                       res_baseline['jaccard_index']],
                      ['Ours[250, 720]', res['precision'], res['recall'], res['f1'], res['jaccard_index']]],
                     headers=['Name', 'Precision', 'Recall', 'F-score', 'Jaccard'] ) )



