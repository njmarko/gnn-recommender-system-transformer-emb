import pandas as pd

movies_df = pd.read_csv('data/ml-latest-small/movies-with-descriptions.csv')
# old_index,index,movieId,title,genres,imdbId,tmdbId,imdb_title_id,year,duration,language,country,description
movies_df = movies_df.drop(columns=['old_index', 'new_index', 'imdbId', 'tmdbId', 'imdb_title_id', 'country', 'title', 'description'])

movie_genre_columns = movies_df['genres'].str.get_dummies(sep='|')
language_columns = movies_df['language'].str.get_dummies(sep=', ')


movies_df = pd.concat([movies_df.drop(columns=['genres', 'language']), movie_genre_columns, language_columns], axis=1)


movies_df.to_csv('data/ml-latest-small/movies-with-numerized-categories.csv')
