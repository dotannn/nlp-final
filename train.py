from nltk.corpus import stopwords
import pandas as pd
from utils import get_texts, get_all_tokenized
print(stopwords.words('english'))




TRAIN_MOVIE_GENRES_PLOT_CSV = "data/trn_movie_genres_plot.csv"
VAL_MOVIE_GENRES_PLOT_CSV = "data/val_movie_genres_plot.csv"


chunksize = 24000


trn_data = pd.read_csv(TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=chunksize)
val_data = pd.read_csv(TRAIN_MOVIE_GENRES_PLOT_CSV, header=None, chunksize=chunksize)


for df in trn_data:
    labels = df[0].values
    text = df[1].values

    pass