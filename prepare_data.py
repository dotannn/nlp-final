## prepare CSV File with:   movie-name |  genres | plot
from collections import Counter

import pandas as pd
import numpy as np
from pathlib import Path

IMDB_PLOTS_FILE = "data/plot.list"
IMDB_GENRES_FILE = "data/genres.list.gz"
GENRES_TYPES_FILE = "data/genres.txt"
DATA_DIR = "data"
GENRES_PKL = "genres.pkl"
MOVIE_GENRES_PLOT_CSV = "movie_genres_plot.csv"


FILTERED_GENRES = ['short', 'lifestyle', 'hardcore', 'sex', 'commercial', 'erotica', 'experimental']

def load_genres():
    genres_pkl = Path( "%s/%s" % (DATA_DIR, GENRES_PKL) )
    if not genres_pkl.exists():
        cols_genres = ['movie', 'genres']
        skip_genres = 383
        df_genres = pd.read_csv( IMDB_GENRES_FILE, header=None, names=cols_genres, skiprows=skip_genres,
                                 encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python' )
        GENRES = df_genres['genres'].unique().tolist()
        GENRES = [x.lower() for x in GENRES]
        GENRES = list( set( GENRES ) - set( FILTERED_GENRES ) )
        df_genres = df_genres.groupby( 'movie' )['genres'].apply( list )
        df_genres.to_pickle( genres_pkl )
        Path( GENRES_TYPES_FILE ).open( 'w' ).writelines( f'{o}\n' for o in GENRES )

    GENRES = Path( GENRES_TYPES_FILE ).open( "r" ).readlines()
    GENRES = list( map( lambda x: x.strip(), GENRES ) )
    df_genres = pd.read_pickle(str(genres_pkl))
    return df_genres, GENRES


def load_plots( df_genres, genres_dict, index_genres=True):
    movies = []
    plots = []
    genres = []

    # iterate plots:
    with open( IMDB_PLOTS_FILE, 'r', encoding='ISO-8859-1' ) as f:
        movie_name = ""
        plot = ""
        for line in f:
            if line.startswith( "MV: " ):
                if len( movie_name ) > 0 and len( plot ) > 0 and "{" not in movie_name:
                    genre = df_genres.get( movie_name )
                    if genre is not None and len( genre ) > 0:
                        genre = [x.lower() for x in genre]
                        if len(set(genre) & set(FILTERED_GENRES)) == 0:
                            movies.append( movie_name )
                            plots.append( plot.replace( "\n", " " ) )
                            if index_genres:
                                genres.append( list( map( lambda x: genres_dict.get( x.lower() ), genre ) ) )
                            else:
                                genres.append(genre)
                plot = ""
                movie_name = line[4:].strip()
            if line.startswith( "PL: " ):
                plot += line[4:]
    return movies, plots, genres


def disp_data_stats(genres, genres_dict):
    idx2genre = {}
    for genre, idx in genres_dict.items():
        idx2genre[idx] = genre

    n_movies = len(genres)
    print("Num of movies: %d" % len(genres))

    n_genres_list = [len( l ) for l in genres]
    n_genres_freq = Counter( n_genres_list )

    for amount, freq in n_genres_freq.items():
        print("Num of movies w/ %d genres: %d(%.2f%%)" % (amount, freq, (freq * 100.0) /n_movies ))
    flat_genres = [genre for sublist in genres for genre in sublist]
    genres_stats = Counter(flat_genres)

    print('Genres distributions:')
    for gidx, val in genres_stats.items():
        print("\t%s: %d(%.2f%%)" % (idx2genre[gidx], val, (val * 100.0) / float(n_movies)))


def generate_input_csv(train_csv_out, val_csv_out, split_ratio=0.9, index_genres=True):
    df_genres, genres_list = load_genres()
    genres_dict = {}
    for idx, label in enumerate(genres_list):
        genres_dict[label] = idx

    movies, plots, genres = load_plots(df_genres, genres_dict, index_genres=index_genres)

    disp_data_stats(genres, genres_dict)
    # shuffle the data
    idx = np.random.permutation( len( movies ) )
    genres = [genres[i - 1] for i in idx]
    plots = [plots[i - 1] for i in idx]

    # split to train and validation
    train_val_split = int(len( movies ) * split_ratio)
    train_genres = genres[:train_val_split]
    val_genres = genres[train_val_split:]
    train_plots = plots[:train_val_split]
    val_plots = plots[train_val_split:]

    # pack in DataFrame
    trn_movie_genres_plot = pd.DataFrame(
        {'genres': train_genres,
         'plot': train_plots
        }, columns=["genres", "plot"])

    val_movie_genres_plot = pd.DataFrame(
        {'genres': val_genres,
         'plot': val_plots
         }, columns=["genres", "plot"] )


    # save to csv
    trn_movie_genres_plot.to_csv(train_csv_out, header=False, index=False)
    val_movie_genres_plot.to_csv( val_csv_out, header=False, index=False )
    print("data preparation finished!")



