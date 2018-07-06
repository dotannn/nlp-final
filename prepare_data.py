## prepare CSV File with:   movie-name |  genres | plot
import pandas as pd
import numpy as np
from pathlib import Path
import json

import sklearn


IMDB_PLOTS_FILE = "data/plot.list"
IMDB_GENRES_FILE = "data/genres.list.gz"
GENRES_TYPES_FILE = "data/genres.json"


DATA_DIR = "data"
GENRES_PKL = "genres.pkl"
MOVIE_GENRES_PLOT_CSV = "movie_genres_plot.csv"

GENRES = []

# prepare genres dataframe

genres_pkl = Path("%s/%s" % (DATA_DIR,GENRES_PKL))
if not genres_pkl.exists():
    cols_genres = ['movie', 'genres']
    skip_genres = 383
    df_genres = pd.read_csv(IMDB_GENRES_FILE, header=None, names=cols_genres, skiprows=skip_genres, encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python')
    GENRES = df_genres['genres'].unique().tolist()
    df_genres = df_genres.groupby('movie')['genres'].apply(list)
    df_genres.to_pickle(genres_pkl)
    Path( GENRES_TYPES_FILE ).open( 'w' ).writelines( f'{o}\n' for o in GENRES )

GENRES = Path( GENRES_TYPES_FILE ).open("r").readlines()
GENRES = list(map( lambda x: x.strip(), GENRES))

genres_dict = {}
for idx, label in enumerate(GENRES):
    genres_dict[label] = idx

df_genres = pd.read_pickle(genres_pkl)

movies = []
plots = []
genres = []


trn_movie_genres_plot_csv = Path("%s/trn_%s" % (DATA_DIR, MOVIE_GENRES_PLOT_CSV))
val_movie_genres_plot_csv = Path("%s/val_%s" % (DATA_DIR, MOVIE_GENRES_PLOT_CSV))

if not trn_movie_genres_plot_csv.exists():
    # iterate plots:
    with open( IMDB_PLOTS_FILE, 'r', encoding='ISO-8859-1' ) as f:
        movie_name = ""
        plot = ""
        for line in f:
            if line.startswith("MV: "):
                if len(movie_name) > 0 and len(plot) > 0 and "{" not in movie_name:
                    genre = df_genres.get(movie_name)
                    if genre is not None and len(genre) > 0:
                        movies.append(movie_name)
                        plots.append(plot.replace("\n", " "))
                        genres.append(list(map( lambda x: genres_dict.get(x), genre)))

                plot = ""
                movie_name = line[4:].strip()
            if line.startswith("PL: "):
                plot += line[4:]

    # shuffle the data
    idx = np.random.permutation( len( movies ) )

    trn_val_split = int(len( movies ) * 0.9)
    # movies = np.array(movies)
    # genres = np.array( genres )
    # plots = np.array( plots )

    movies = [movies[i - 1] for i in idx]
    genres = [genres[i - 1] for i in idx]
    plots = [plots[i - 1] for i in idx]

    # movies = movies[idx]
    # genres = genres[idx]
    # plots = plots[idx]

    train_movies = movies[:trn_val_split]
    val_movies = movies[trn_val_split:]

    train_genres = genres[:trn_val_split]
    val_genres = genres[trn_val_split:]

    train_plots = plots[:trn_val_split]
    val_plots = plots[trn_val_split:]

    trn_movie_genres_plot = pd.DataFrame(
        # {'movie': train_movies,
        {'genres': train_genres,
         'plot': train_plots
        }, columns=["genres", "plot"]) #"movie",

    val_movie_genres_plot = pd.DataFrame(
        # {'movie': val_movies,
        {'genres': val_genres,
         'plot': val_plots
         }, columns=["genres", "plot"] ) #"movie"

    trn_movie_genres_plot.to_csv(trn_movie_genres_plot_csv, header=False, index=False)
    val_movie_genres_plot.to_csv( val_movie_genres_plot_csv, header=False, index=False )



print("data preparation finished!")



