import torch
import pandas as pd

from data_load_movies import load_movies
from data_load_ratings import load_ratings

def create_edge_index():
    # Load movies and their mapping
    _, movie_mapping = load_movies()
    
    # Load ratings and their mapping
    ratings_df, user_mapping = load_ratings()
    
    # Create edges using the indices from the mappings
    edge_index = []
    for index, row in ratings_df.iterrows():
        user_idx = user_mapping[row['userId']]
        movie_idx = movie_mapping[row['movieId']]
        edge_index.append((user_idx, movie_idx))
    
    return edge_index

# Let's test this function to see if it builds the edge index correctly.

if __name__ == "__main__":
    create_edge_index()
