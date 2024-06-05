import pandas as pd 

movies_df = pd.read_csv('data/ml-latest-small/movies.csv')


movies_with_year = movies_df['title'].str.rstrip().str[-6:].str.match('\([0-9]{4}\)')

movies_without_year_count = len(movies_df[~movies_with_year])

print(f'Movies without a year in the title: {movies_without_year_count}')

print('The movies are:')

print(movies_df[~movies_with_year])


described_movies = pd.read_csv('data/imbd_reviewed_movies.csv', )
movie_links = pd.read_csv('data/ml-latest-small/links.csv', dtype={'imdbId': str, 'tmdbId': str})

movie_links['imdbId'] = 'tt' + movie_links['imdbId']

print(len(movie_links[movie_links['imdbId'].str.len() != 9]))

print(movie_links.head(20))

print(described_movies.columns)

movie_description_missing = described_movies['description'].isna().sum()
movie_duration_missing = described_movies['duration'].isna().sum()
movie_year_missing = described_movies['year'].isna().sum()
movie_country_missing = described_movies['country'].isna().sum()
movie_language_missing = described_movies['language'].isna().sum()

print(f'Movies without description: {movie_description_missing / len(described_movies) * 100:.2f}%')
print(f'Movies without duration: {movie_duration_missing / len(described_movies) * 100:.2f}%')
print(f'Movies without year: {movie_year_missing / len(described_movies) * 100:.2f}%')
print(f'Movies without country: {movie_country_missing / len(described_movies) * 100:.2f}%')
print(f'Movies without language: {movie_language_missing / len(described_movies) * 100:.2f}%')

described_movies = described_movies[['imdb_title_id', 'year', 'duration', 'language', 'country', 'description']]

described_movies_cleaned = described_movies.dropna()

print(f'Kept rows: {len(described_movies_cleaned) / len(described_movies) * 100: .2f}%')


movie_imdb_ids = set(movie_links['imdbId'])
described_movie_ids = set(described_movies_cleaned['imdb_title_id'])

id_diff = movie_imdb_ids.difference(described_movie_ids)

print(f'Non matching IDs: {len(id_diff)}')
print(f'Percentage of non-described films: {int(len(id_diff) / len(movie_links) * 100)}%')

# print(described_movies.head(50))


described_movies_cleaned = described_movies[described_movies['description'].notna()]

movies_df = movies_df.merge(movie_links, on='movieId', how='left')
print(movies_df.columns)

print(movies_df.imdbId.isna().sum())
print(movies_df.tmdbId.isna().sum())

movies_described_df = movies_df.merge(described_movies_cleaned, left_on='imdbId', right_on='imdb_title_id', how='left')

print(movies_described_df.head())

movies_described_df.to_csv('data/ml-latest-small/processed-movies.csv')
