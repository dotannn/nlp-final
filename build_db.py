import csv
import json
from imdb import IMDb

from genre import Genre




def get_movie_data(movie_id):
    if movie_id is None:
        return None

    # create an instance of the IMDb class
    ia = IMDb()

    print("***********   ", movie_id, "   ***********")

    # get a movie
    movie = ia.get_movie(movie_id)
    genres = [Genre.norm_genre(genre) for genre in movie['genres']]
    plot = norm_plot(movie.get('plot'))

    return {
        "movie_id": movie_id,
        "title": movie.get("title")
        "genres": genres,
        "plot": plot,
        "year": movie.get("year"),
    }


def norm_plot(plots):
    return plots[0].split("::")[0] if plots else ""


def extract_movie_id(tconst):
    return tconst[2:]

def get_movie_ids(movie_db_path, filters=None):
    movie_ids = []
    with open(movie_db_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movie_id = extract_movie_id(row['tconst'])

            # check if we wnat to filter out movies 
            if filters:
                pass

            movie_ids.append(movie_id)
    return movie_ids


def build_movie_dataset(input_db_path, output_dataset_path):
    movie_ids = get_movie_ids(input_db_path)

    movies_data = []

    # getting movie data
    for movie_id in movie_ids[:50]:
        movie_data = get_movie_data(movie_id)
        movies_data.append(movie_data)

    # saving to data set
    with open(output_dataset_path, 'w') as outfile:
        for movie_data in movies_data:
            json.dump(movie_data, outfile)
            outfile.write('\n')



#build_movie_dataset('data_set/title.basic.filter.movie.genre.csv', 'data_set/title.basic.movie.plot.json')
