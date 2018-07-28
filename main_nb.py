from utils import *
from pathlib import Path
import pandas as pd
from naivebayes import NaiveBayesGenreClassifier

RESULT_DIR = "bn_tmp"

TRAIN_MOVIE_GENRES_PLOT_CSV = Path("data/trn_movie_genres_plot.csv")
VAL_MOVIE_GENRES_PLOT_CSV = Path("data/val_movie_genres_plot.csv")
GENRES_TYPES_FILE = "data/genres.txt"

CHUNKSIZE = 24000
MAX_SAMPLES = 5000

if MAX_SAMPLES > 0:
    train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES )
    val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE, nrows=MAX_SAMPLES // 10 )
else:
    train_data = pd.read_csv( TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )
    val_data = pd.read_csv( VAL_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=CHUNKSIZE )

train_data, train_labels = get_all_tokenized( train_data, 1, do_tokenize=False)
val_data, val_labels = get_all_tokenized( val_data, 1, do_tokenize=False)

naive_bayes_genre_classifier = NaiveBayesGenreClassifier()
naive_bayes_genre_classifier.train(train_data, train_labels)
val_predict = naive_bayes_genre_classifier.predict(val_data)
print(naive_bayes_genre_classifier.eval(val_predict, val_labels))
