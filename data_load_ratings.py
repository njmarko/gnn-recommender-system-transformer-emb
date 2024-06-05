import pandas as pd

# _, user_mapping = load_node_csv(rating_path, index_col='userId')

def load_ratings():
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv', index_col='userId')
    mapping = {key: i for i, key in enumerate(ratings_df.index.unique())}

    return ratings_df, mapping