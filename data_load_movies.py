import pandas as pd

def load_movies(word_embeddings=True):
    embeddings_df = pd.read_csv('data/ml-latest-small/movie-embeddings.csv', index_col='movieId').drop(columns='new_index')
    movie_attributes_df = pd.read_csv('data/ml-latest-small/movies-with-numerized-categories.csv', index_col='movieId').drop(columns='new_index')
    print(embeddings_df.shape)
    print(movie_attributes_df.shape)

    if word_embeddings:
        movies_df = movie_attributes_df.merge(embeddings_df, on='movieId', how='inner')
    else:
        movies_df = movie_attributes_df

    mapping = { key: i for i, key in enumerate(movies_df.index.unique()) }

    return movies_df, mapping