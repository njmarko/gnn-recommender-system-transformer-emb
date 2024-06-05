import pandas as pd

from sentence_transformers import SentenceTransformer


def progress_coroutine(print_on = 1, total_iterations='unk'):
    print("Starting progress monitor")

    iterations = 0
    while True:
        yield
        iterations += 1
        if (iterations % print_on == 0):
            print("{} iterations done of {}".format(iterations, total_iterations))

def trace_progress(func, progress = None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

movie_descriptions = pd.read_csv('data/ml-latest-small/movies-with-descriptions.csv')

def embed_text(sentence):
    if not pd.notna(sentence):
        sentence = 'Description not provided'
    return model.encode(sentence)

co1 = progress_coroutine(print_on=10, total_iterations=len(movie_descriptions))
next(co1)

movie_descriptions['embeddings'] = movie_descriptions['description'].apply(trace_progress(embed_text, progress=co1))

movie_embeddings_df = movie_descriptions[['movieId', 'embeddings']]

movie_embeddings_df = movie_embeddings_df.join(pd.DataFrame(movie_embeddings_df['embeddings'].to_list()).add_prefix('embedding_')).drop('embeddings', axis=1)

movie_embeddings_df.to_csv('data/ml-latest-small/movie-embeddings.csv')
