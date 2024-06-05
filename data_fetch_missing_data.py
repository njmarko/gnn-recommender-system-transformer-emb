import pandas as pd
import requests
import json 
import time

movies_df = pd.read_csv('data/ml-latest-small/processed-movies.csv', dtype={'tmdbId': str})

# movie_genre_columns = movies_df['genres'].str.get_dummies(sep='|')
# language_columns = movies_df['language'].str.get_dummies(sep=', ')

# print(language_columns.columns)

# movies_df = pd.concat([movies_df.drop(columns=['genres', 'language', 'country']), movie_genre_columns, language_columns], axis=1)

# print(movies_df.head())


def fetch_api_data(row):
    # for each movie fetch this data 
    # year,duration,language,country,description
    if pd.notna(row['imdb_title_id']):
        # print('Movie already has a description: %s' % row['title'])
        return row

    url = f"https://api.themoviedb.org/3/movie/{row['tmdbId']}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkMjk5NmNmZDNkODI5NTE5NjNlNDk2MDNkNzI3ZGVjMiIsInN1YiI6IjY2MjIyMWIyN2EzYzUyMDE3ZDRkOTVjOSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.TNbTd42REiaJouoiGAxAwhD0bDJiZgXOFxJuf4f9qDw"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        print('No data for movie %s. Imputing default data.' % row['title'])
        row['year'] = '2000'
        row['duration'] = 90
        row['language'] = 'English'
        row['description'] = 'N/A'
    elif response.status_code // 100 != 2:
        print(f'Movie detail fetch failed with status code {response.status_code}. Reason: {response.text}')
        raise ValueError('Something went wrong')

    response = json.loads(response.text)

    print('Fetching data for: %s' % row['title'])
    print('Movie is titled as %s on TMDB' % response.get('original_title', row['title']))
    print('\n')

    row['year'] = response.get('release_date', '2000')[:5]
    row['duration'] = response.get('runtime', 90)
    row['language'] = ', '.join(map(lambda language: language['english_name'], response.get('spoken_languages', [{'english_name': 'English'}])))
    # row['country'] = ', '.join(map(lambda country: country['name'], response['production_countries']))
    row['description'] = response.get('overview', 'N/A')
    time.sleep(0.05)  # timeout, to not exceed API quota
    return row

movies_df = movies_df.apply(fetch_api_data, axis=1)

movies_df.to_csv('data/ml-latest-small/movies-with-descriptions.csv')
