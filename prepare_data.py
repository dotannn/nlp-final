## prepare CSV File with:   movie-name |  genres | plot
import pandas as pd
import numpy as np
from pathlib import Path

IMDB_PLOTS_FILE = "data/plot.list"
IMDB_GENRES_FILE = "data/genres.list.gz"
GENRES_TYPES_FILE = "data/genres.txt"
DATA_DIR = "data"
GENRES_PKL = "genres.pkl"
MOVIE_GENRES_PLOT_CSV = "movie_genres_plot.csv"



def load_genres():
    genres_pkl = Path( "%s/%s" % (DATA_DIR, GENRES_PKL) )
    if not genres_pkl.exists():
        cols_genres = ['movie', 'genres']
        skip_genres = 383
        df_genres = pd.read_csv( IMDB_GENRES_FILE, header=None, names=cols_genres, skiprows=skip_genres,
                                 encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python' )
        GENRES = df_genres['genres'].unique().tolist()
        df_genres = df_genres.groupby( 'movie' )['genres'].apply( list )
        df_genres.to_pickle( genres_pkl )
        Path( GENRES_TYPES_FILE ).open( 'w' ).writelines( f'{o}\n' for o in GENRES )

    GENRES = Path( GENRES_TYPES_FILE ).open( "r" ).readlines()
    GENRES = list( map( lambda x: x.strip(), GENRES ) )
    df_genres = pd.read_pickle(str(genres_pkl))
    return df_genres, GENRES


def load_plots( df_genres, genres_dict):
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
                        movies.append( movie_name )
                        plots.append( plot.replace( "\n", " " ) )
                        genres.append( list( map( lambda x: genres_dict.get( x ), genre ) ) )

                plot = ""
                movie_name = line[4:].strip()
            if line.startswith( "PL: " ):
                plot += line[4:]
    return movies, plots, genres


def generate_input_csv(train_csv_out, val_csv_out, split_ratio=0.9):
    df_genres, genres_list = load_genres()
    genres_dict = {}
    for idx, label in enumerate(genres_list):
        genres_dict[label] = idx

    movies, plots, genres = load_plots(df_genres, genres_dict)

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



